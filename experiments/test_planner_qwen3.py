import argparse
import asyncio
import time
from typing import Literal

from pydantic import BaseModel, Field

from llm_compiler.compiler import (
    CompiledDAG,
    PlanCompilationError,
    PlanCompiler,
    ToolRegistry,
    ToolSpec,
)
from llm_compiler.executor import ExecutionResult, Executor
from llm_compiler.joiner import Joiner, JoinerOutputError, JoinerResult
from llm_compiler.llm import create_ollama_chat_model
from llm_compiler.planner import ExecutionPlan, Planner, PlannerOutputError

# =============================================================================
# Configurazione esperimento
# =============================================================================

MODEL_NAME = "qwen3:8b"
REQUEST = (
    "Calcola la differenza tra la popolazione dell'Italia "
    "e quella della Francia (Italia meno Francia)."
)

# Latenza simulata per rendere osservabili timeline e speedup reale
# (impostare a 0.0 per test puramente in-memory).
SIMULATED_LOOKUP_DELAY_S = 0.15
SIMULATED_CALC_DELAY_S = 0.05


# =============================================================================
# Contratti I/O e implementazioni dei tool
# =============================================================================


class PopulationInput(BaseModel):
    country: str = Field(min_length=1)


class PopulationOutput(BaseModel):
    country: str
    population: int


class CalculatorInput(BaseModel):
    operation: Literal["add", "subtract", "multiply", "divide"]
    a: float
    b: float


class CalculatorOutput(BaseModel):
    result: float


POPULATIONS: dict[str, int] = {
    "italy": 58_990_000,
    "france": 68_400_000,
    "germany": 83_500_000,
    "spain": 48_800_000,
}


async def population_lookup(country: str) -> dict[str, str | int]:
    if SIMULATED_LOOKUP_DELAY_S > 0:
        await asyncio.sleep(SIMULATED_LOOKUP_DELAY_S)
    key = country.strip().lower()
    if key not in POPULATIONS:
        raise ValueError(f"Unknown country: {country}")
    return {"country": country, "population": POPULATIONS[key]}


def calculator(
    operation: Literal["add", "subtract", "multiply", "divide"],
    a: float,
    b: float,
) -> dict[str, float]:
    if SIMULATED_CALC_DELAY_S > 0:
        time.sleep(SIMULATED_CALC_DELAY_S)
    if operation == "add":
        return {"result": a + b}
    if operation == "subtract":
        return {"result": a - b}
    if operation == "multiply":
        return {"result": a * b}
    if operation == "divide":
        if b == 0:
            raise ZeroDivisionError("Division by zero.")
        return {"result": a / b}
    raise ValueError(f"Unsupported operation: {operation}")


def build_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ToolSpec(
                name="population_lookup",
                description="Get country population",
                func=population_lookup,
                input_schema=PopulationInput,
                output_schema=PopulationOutput,
            ),
            ToolSpec(
                name="calculator",
                description="Perform arithmetic (a <op> b)",
                func=calculator,
                input_schema=CalculatorInput,
                output_schema=CalculatorOutput,
            ),
        ]
    )


# =============================================================================
# Helpers di stampa per sezione di responsabilità
# =============================================================================


def print_header(step: str, owner: str, subtitle: str) -> None:
    line = "=" * 78
    print(f"\n{line}")
    print(f"[{step}] {owner} — {subtitle}")
    print(line)


def print_registry_section(registry: ToolRegistry, catalog: dict[str, str]) -> None:
    print_header("1/6", "TOOL REGISTRY", "Contratti e catalogo generato")
    print(f"Tool registrati: {len(registry)} ({', '.join(registry.names())})")
    for name, desc in catalog.items():
        print(f"  - {name}: {desc}")


def print_planner_section(
    request: str,
    model: str,
    plan: ExecutionPlan,
    planning_ms: float,
) -> None:
    print_header("2/6", "PLANNER", f"Generazione ExecutionPlan ({model})")
    print(f"Richiesta utente: {request}")
    print(f"Latenza planning: {planning_ms:.2f} ms ({planning_ms / 1000:.2f} s)")
    print(f"Task generati:    {len(plan.tasks)}")
    print("\nPiano JSON:")
    print(plan.model_dump_json(indent=2))


def print_compiler_section(dag: CompiledDAG, compile_ms: float) -> None:
    print_header("3/6", "PLAN COMPILER", "Validazione semantica e CompiledDAG")
    edge_count = sum(len(deps) for deps in dag.dependencies.values())
    max_width = max((len(level) for level in dag.levels), default=0)

    print("Stato compilazione:    VALID")
    print(f"Latenza compilazione:  {compile_ms:.2f} ms")
    print(f"Nodi / Archi DAG:      {len(dag.tasks)} nodi, {edge_count} archi")
    print(f"Profondità (livelli):  {len(dag.levels)}")
    print(f"Larghezza max teorica: {max_width} task paralleli")
    print(f"Ordine topologico:     {list(dag.topological_order)}")
    print(f"Livelli paralleli:     {[list(level) for level in dag.levels]}")
    print(f"Root tasks:            {list(dag.root_tasks)}")
    print(f"Leaf tasks:            {list(dag.leaf_tasks)}")
    print("\nDipendenze:")
    for task_id in dag.topological_order:
        deps = sorted(dag.dependencies[task_id])
        print(f"  - {task_id} <- {deps}")


def _render_timeline_bar(
    started_at_ms: float | None,
    finished_at_ms: float | None,
    total_ms: float,
    width: int = 30,
) -> str:
    if started_at_ms is None or finished_at_ms is None or total_ms <= 0:
        return "." * width
    start_idx = min(width - 1, max(0, int((started_at_ms / total_ms) * width)))
    end_idx = min(width, max(start_idx + 1, int((finished_at_ms / total_ms) * width)))
    return "." * start_idx + "#" * (end_idx - start_idx) + "." * (width - end_idx)


def print_executor_section(
    parallel_res: ExecutionResult,
    sequential_res: ExecutionResult,
) -> None:
    print_header(
        "4/6",
        "EXECUTOR",
        "Esecuzione event-driven parallela vs baseline sequenziale",
    )
    total_tasks = len(parallel_res.results)
    succeeded = len(parallel_res.succeeded_task_ids)
    failed = len(parallel_res.failed_task_ids)
    skipped = len(parallel_res.skipped_task_ids)
    success_rate = (succeeded / total_tasks * 100.0) if total_tasks else 0.0

    print(
        f"Run parallelo (max_concurrency={parallel_res.max_concurrency}): "
        f"{parallel_res.total_duration_ms:.2f} ms"
    )
    print(
        f"Run sequenziale (max_concurrency={sequential_res.max_concurrency}): "
        f"{sequential_res.total_duration_ms:.2f} ms"
    )
    print(
        f"Esiti task: {succeeded} success / {failed} error / {skipped} skipped "
        f"(success rate: {success_rate:.1f}%)"
    )

    print("\nTimeline esecuzione parallela:")
    for task_id, res in parallel_res.results.items():
        bar = _render_timeline_bar(
            res.started_at_ms,
            res.finished_at_ms,
            parallel_res.total_duration_ms,
        )
        if res.status == "success":
            window = f"{res.started_at_ms:6.1f} -> {res.finished_at_ms:6.1f} ms"
            detail = f"output={res.output}"
        elif res.status == "error":
            window = f"{res.started_at_ms or 0.0:6.1f} -> {res.finished_at_ms or 0.0:6.1f} ms"
            detail = f"error[{res.error_stage}/{res.error_type}]={res.error}"
        else:
            window = "   skipped         "
            detail = f"reason={res.skip_reason}"

        print(
            f"  {task_id:<8} | {res.tool_name:<18} | {res.status:<7} | "
            f"[{bar}] | {window} | {res.duration_ms:6.2f} ms | {detail}"
        )


def print_joiner_section(
    model: str,
    joiner_res: JoinerResult,
    joining_ms: float,
) -> None:
    print_header("5/6", "JOINER", f"Sintesi risposta finale ({model})")
    print(f"Status deterministico: {joiner_res.status}")
    print(f"Latenza joining:       {joining_ms:.2f} ms ({joining_ms / 1000:.2f} s)")
    if joiner_res.missing_information:
        print("Informazioni mancanti:")
        for item in joiner_res.missing_information:
            print(f"  - {item}")
    else:
        print("Informazioni mancanti: nessuna")
    print("\nRisposta finale:")
    print(f"  {joiner_res.answer}")


def print_metrics_summary(
    planning_ms: float,
    compile_ms: float,
    parallel_res: ExecutionResult,
    sequential_res: ExecutionResult,
    joining_ms: float,
    llm_calls: int,
) -> None:
    print_header("6/6", "METRICHE", "Riepilogo sperimentale end-to-end")

    total_tasks = len(parallel_res.results)
    success_rate = (
        len(parallel_res.succeeded_task_ids) / total_tasks * 100.0
        if total_tasks
        else 0.0
    )
    speedup = (
        sequential_res.total_duration_ms / parallel_res.total_duration_ms
        if parallel_res.total_duration_ms > 0
        else 0.0
    )
    e2e_parallel_ms = (
        planning_ms + compile_ms + parallel_res.total_duration_ms + joining_ms
    )

    rows = [
        ("Planning latency", f"{planning_ms:.2f} ms ({planning_ms / 1000:.2f} s)"),
        ("Compilation latency", f"{compile_ms:.2f} ms"),
        ("Execution latency (parallel)", f"{parallel_res.total_duration_ms:.2f} ms"),
        ("Execution latency (sequential)", f"{sequential_res.total_duration_ms:.2f} ms"),
        ("Joining latency", f"{joining_ms:.2f} ms ({joining_ms / 1000:.2f} s)"),
        ("Cumulative task time (sum Ti)", f"{parallel_res.sum_task_duration_ms:.2f} ms"),
        ("Effective parallelism (sum Ti / T_par)", f"{parallel_res.effective_parallelism:.2f}x"),
        ("Measured speedup (T_seq / T_par)", f"{speedup:.2f}x"),
        ("Task success rate", f"{success_rate:.1f}% ({len(parallel_res.succeeded_task_ids)}/{total_tasks})"),
        ("Plan validity", "100.0% (1/1)"),
        ("LLM calls", str(llm_calls)),
        (
            "End-to-end latency (plan+compile+exec+join)",
            f"{e2e_parallel_ms:.2f} ms ({e2e_parallel_ms / 1000:.2f} s)",
        ),
    ]

    label_width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"  {label:<{label_width}} : {value}")


# =============================================================================
# Pipeline principale
# =============================================================================


async def main(*, reasoning: bool = True) -> None:
    llm_calls = 0

    # 1. Tool Registry
    registry = build_registry()
    catalog = registry.to_planner_catalog()
    print_registry_section(registry, catalog)
    print(
        f"Configurazione modello: {MODEL_NAME} "
        f"(planner reasoning={reasoning}, joiner reasoning=False, temperature=0.0)"
    )

    # 2. Planner
    planner_llm = create_ollama_chat_model(
        model=MODEL_NAME,
        temperature=0.0,
        reasoning=reasoning,
    )
    joiner_llm = create_ollama_chat_model(
        model=MODEL_NAME,
        temperature=0.0,
        reasoning=False,
    )
    planner = Planner(llm=planner_llm)

    t0 = time.perf_counter()
    llm_calls += 1
    try:
        plan = await planner.aplan(request=REQUEST, available_tools=catalog)
        planning_ms = (time.perf_counter() - t0) * 1000.0
    except PlannerOutputError as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print_header("2/6", "PLANNER", f"Errore di parsing ({elapsed_ms:.2f} ms)")
        print(exc)
        print("RAW OUTPUT:", exc.raw_output)
        return

    print_planner_section(REQUEST, MODEL_NAME, plan, planning_ms)

    # 3. Plan Compiler
    compiler = PlanCompiler(registry=registry)

    t1 = time.perf_counter()
    try:
        dag = compiler.compile(plan)
        compile_ms = (time.perf_counter() - t1) * 1000.0
    except PlanCompilationError as exc:
        print_header("3/6", "PLAN COMPILER", "Errore di compilazione")
        print(exc)
        return

    print_compiler_section(dag, compile_ms)

    # 4. Executor (parallelo vs baseline sequenziale sullo stesso CompiledDAG)
    parallel_executor = Executor(registry=registry, max_concurrency=8)
    sequential_executor = Executor(registry=registry, max_concurrency=1)

    parallel_result = await parallel_executor.execute(dag)
    sequential_result = await sequential_executor.execute(dag)
    print_executor_section(parallel_result, sequential_result)

    # 5. Joiner
    joiner = Joiner(llm=joiner_llm)

    t2 = time.perf_counter()
    llm_calls += 1
    try:
        joiner_result = await joiner.join(
            request=REQUEST,
            execution_result=parallel_result,
            plan=plan,
        )
        joining_ms = (time.perf_counter() - t2) * 1000.0
    except JoinerOutputError as exc:
        elapsed_ms = (time.perf_counter() - t2) * 1000.0
        print_header("5/6", "JOINER", f"Errore di parsing ({elapsed_ms:.2f} ms)")
        print(exc)
        print("RAW OUTPUT:", exc.raw_output)
        return

    print_joiner_section(MODEL_NAME, joiner_result, joining_ms)

    # 6. Riepilogo metriche
    print_metrics_summary(
        planning_ms=planning_ms,
        compile_ms=compile_ms,
        parallel_res=parallel_result,
        sequential_res=sequential_result,
        joining_ms=joining_ms,
        llm_calls=llm_calls,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline diretta Python: Registry -> Planner -> Compiler -> Executor -> Joiner."
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Disabilita la modalità reasoning sul Planner (attiva di default).",
    )
    args = parser.parse_args()
    asyncio.run(main(reasoning=not args.no_reasoning))
