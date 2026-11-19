import argparse
import asyncio
import time
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from langgraph_llm_compiler.compiler import (
    CompiledDAG,
    PlanCompilationError,
    PlanCompiler,
    ToolRegistry,
    ToolSpec,
)
from langgraph_llm_compiler.executor import ExecutionResult, Executor
from langgraph_llm_compiler.joiner import Joiner, JoinerOutputError, JoinerResult
from langgraph_llm_compiler.llm import create_ollama_chat_model
from langgraph_llm_compiler.planner import ExecutionPlan, Planner, PlannerOutputError

# =============================================================================
# Experiment configuration
# =============================================================================

MODEL_NAME = "qwen3:8b"
SIMULATED_EXCHANGE_DELAY_S = 0.12
SIMULATED_SYNC_TOOL_DELAY_S = 0.04


# =============================================================================
# I/O contracts (including nested schemas) and tool implementations
# =============================================================================


class CountryStatsInput(BaseModel):
    country: str = Field(min_length=1)


class EconomyMetrics(BaseModel):
    gdp_billion_usd: float
    unemployment_pct: float


class CountryStatsOutput(BaseModel):
    country: str
    population: int
    area_km2: float
    economy: EconomyMetrics


class ExchangeRateInput(BaseModel):
    from_currency: Literal["USD", "EUR", "JPY"]
    to_currency: Literal["USD", "EUR", "JPY"]


class ExchangeRateOutput(BaseModel):
    pair: str
    rate: float


class CalculatorInput(BaseModel):
    operation: Literal["add", "subtract", "multiply", "divide"]
    a: float
    b: float


class CalculatorOutput(BaseModel):
    result: float


class PercentageDiffInput(BaseModel):
    base: float
    target: float


class PercentageDiffOutput(BaseModel):
    delta: float
    percent_change: float


# Asymmetric latencies to showcase event-driven scheduling without wave barriers.
COUNTRY_DATA: dict[str, tuple[float, dict[str, object]]] = {
    "italy": (
        0.10,
        {
            "country": "Italy",
            "population": 58_990_000,
            "area_km2": 301_340.0,
            "economy": {"gdp_billion_usd": 2_300.0, "unemployment_pct": 6.8},
        },
    ),
    "germany": (
        0.14,
        {
            "country": "Germany",
            "population": 83_500_000,
            "area_km2": 357_588.0,
            "economy": {"gdp_billion_usd": 4_525.0, "unemployment_pct": 3.4},
        },
    ),
    "spain": (
        0.18,
        {
            "country": "Spain",
            "population": 48_800_000,
            "area_km2": 505_990.0,
            "economy": {"gdp_billion_usd": 1_650.0, "unemployment_pct": 11.3},
        },
    ),
    "france": (
        0.38,
        {
            "country": "France",
            "population": 68_400_000,
            "area_km2": 551_695.0,
            "economy": {"gdp_billion_usd": 3_130.0, "unemployment_pct": 7.4},
        },
    ),
    "japan": (
        0.24,
        {
            "country": "Japan",
            "population": 124_000_000,
            "area_km2": 377_975.0,
            "economy": {"gdp_billion_usd": 4_110.0, "unemployment_pct": 2.5},
        },
    ),
}

EXCHANGE_RATES: dict[tuple[str, str], float] = {
    ("USD", "EUR"): 0.92,
    ("EUR", "USD"): 1.087,
    ("USD", "JPY"): 152.0,
    ("JPY", "USD"): 0.00658,
    ("EUR", "JPY"): 165.2,
    ("JPY", "EUR"): 0.00605,
}


async def country_stats(country: str) -> dict[str, object]:
    key = country.strip().lower()
    if key not in COUNTRY_DATA:
        await asyncio.sleep(0.05)
        raise ValueError(f"Country '{country}' not found in dataset.")
    delay_s, payload = COUNTRY_DATA[key]
    await asyncio.sleep(delay_s)
    return payload


async def exchange_rate(
    from_currency: Literal["USD", "EUR", "JPY"],
    to_currency: Literal["USD", "EUR", "JPY"],
) -> dict[str, str | float]:
    await asyncio.sleep(SIMULATED_EXCHANGE_DELAY_S)
    if from_currency == to_currency:
        return {"pair": f"{from_currency}/{to_currency}", "rate": 1.0}
    rate = EXCHANGE_RATES[(from_currency, to_currency)]
    return {"pair": f"{from_currency}/{to_currency}", "rate": rate}


def calculator(
    operation: Literal["add", "subtract", "multiply", "divide"],
    a: float,
    b: float,
) -> dict[str, float]:
    time.sleep(SIMULATED_SYNC_TOOL_DELAY_S)
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


def percentage_diff(base: float, target: float) -> dict[str, float]:
    time.sleep(SIMULATED_SYNC_TOOL_DELAY_S)
    if base == 0:
        raise ZeroDivisionError("Base value cannot be zero.")
    delta = target - base
    return {"delta": delta, "percent_change": (delta / base) * 100.0}


def build_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ToolSpec(
                name="country_stats",
                description=(
                    "Get demographic and economic stats for a country in English "
                    "(population, area_km2, nested economy.gdp_billion_usd, "
                    "economy.unemployment_pct)"
                ),
                func=country_stats,
                input_schema=CountryStatsInput,
                output_schema=CountryStatsOutput,
            ),
            ToolSpec(
                name="exchange_rate",
                description="Get currency exchange rate from one currency to another",
                func=exchange_rate,
                input_schema=ExchangeRateInput,
                output_schema=ExchangeRateOutput,
            ),
            ToolSpec(
                name="calculator",
                description="Perform binary arithmetic (a <op> b)",
                func=calculator,
                input_schema=CalculatorInput,
                output_schema=CalculatorOutput,
            ),
            ToolSpec(
                name="percentage_diff",
                description=(
                    "Compute delta (target - base) and percentage change "
                    "from base to target"
                ),
                func=percentage_diff,
                input_schema=PercentageDiffInput,
                output_schema=PercentageDiffOutput,
            ),
        ]
    )


# =============================================================================
# Complex experimental scenarios
# =============================================================================


@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    focus: str
    request: str


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="multi_branch_diamond",
        title="Asymmetric multi-branch DAG + nested outputs",
        focus=(
            "4 parallel roots with asymmetric latencies (France 380ms vs Italy 100ms), "
            "nested output_path ['economy', 'gdp_billion_usd'], "
            "event-driven execution without wave barriers."
        ),
        request=(
            "Fetch statistics for Italy, France, and Germany, and get the USD to EUR "
            "exchange rate. Then: "
            "(1) compute the population density (population divided by area_km2) for Italy and Germany; "
            "(2) compute the percentage change of Germany's density (target) relative to Italy's density (base); "
            "(3) convert France's GDP in billion USD into billion EUR by multiplying it by the USD->EUR rate."
        ),
    ),
    Scenario(
        name="wide_fanout_tree",
        title="Wide fan-out (4 countries) + 4-level reduction tree",
        focus=(
            "4 parallel lookups, 2 parallel sums, total sum, and final average "
            "using nested output_path on economy.gdp_billion_usd."
        ),
        request=(
            "Fetch statistics for Italy, France, Germany, and Spain. "
            "Compute the sum of GDP (gdp_billion_usd) of Italy and France, "
            "the sum of GDP of Germany and Spain, "
            "then add the two partial sums to get the combined GDP of all 4 countries, "
            "and finally divide the total by the static number 4 to compute the average GDP."
        ),
    ),
    Scenario(
        name="partial_failure_recovery",
        title="Partial failure + transitive skip + transparent Joiner",
        focus=(
            "country_stats('Atlantis') fails, the dependent subtraction is skipped, "
            "Italy's independent density branch succeeds, and the Joiner returns status='partial' "
            "without hallucinating missing data."
        ),
        request=(
            "Fetch statistics for Italy and Atlantis. "
            "Compute Italy's population density (population divided by area_km2) "
            "and compute the difference between Italy's population and Atlantis's population "
            "(Italy minus Atlantis)."
        ),
    ),
)


@dataclass(frozen=True)
class ScenarioSummary:
    name: str
    plan_valid: bool
    nodes: int
    levels: int
    planning_ms: float
    compile_ms: float
    exec_par_ms: float
    exec_seq_ms: float
    joining_ms: float
    effective_parallelism: float
    speedup: float
    success_rate: float
    joiner_status: str
    llm_calls: int


# =============================================================================
# Printing helpers grouped by component responsibility
# =============================================================================


def print_banner(title: str) -> None:
    line = "#" * 78
    print(f"\n{line}")
    print(f"# {title}")
    print(line)


def print_header(step: str, owner: str, subtitle: str) -> None:
    line = "=" * 78
    print(f"\n{line}")
    print(f"[{step}] {owner} — {subtitle}")
    print(line)


def print_registry_section(registry: ToolRegistry, catalog: dict[str, str]) -> None:
    print_header("1/6", "TOOL REGISTRY", "I/O contracts and generated catalog")
    print(f"Registered tools: {len(registry)} ({', '.join(registry.names())})")
    for name, desc in catalog.items():
        print(f"  - {name}: {desc}")


def print_planner_section(
    scenario: Scenario,
    model: str,
    plan: ExecutionPlan,
    planning_ms: float,
) -> None:
    print_header("2/6", "PLANNER", f"ExecutionPlan generation ({model})")
    print(f"Scenario:         {scenario.name} — {scenario.title}")
    print(f"Focus:            {scenario.focus}")
    print(f"User request:     {scenario.request}")
    print(f"Planning latency: {planning_ms:.2f} ms ({planning_ms / 1000:.2f} s)")
    print(f"Generated tasks:  {len(plan.tasks)}")
    print("\nCompact plan:")
    for task in plan.tasks:
        bindings = {
            k: f"{b.task_id}.{'.'.join(b.output_path)}" if b.output_path else b.task_id
            for k, b in task.bindings.items()
        }
        print(
            f"  - {task.task_id:<8} | {task.tool_name:<16} | "
            f"static={task.static_inputs} | bindings={bindings}"
        )


def print_compiler_section(dag: CompiledDAG, compile_ms: float) -> None:
    print_header("3/6", "PLAN COMPILER", "Semantic validation and CompiledDAG")
    edge_count = sum(len(deps) for deps in dag.dependencies.values())
    max_width = max((len(level) for level in dag.levels), default=0)

    print("Compilation status:    VALID")
    print(f"Compilation latency:   {compile_ms:.2f} ms")
    print(f"DAG nodes / edges:     {len(dag.tasks)} nodes, {edge_count} edges")
    print(f"Depth (levels):        {len(dag.levels)}")
    print(f"Max theoretical width: {max_width} parallel tasks")
    print(f"Topological order:     {list(dag.topological_order)}")
    print(f"Parallel levels:       {[list(level) for level in dag.levels]}")
    print(f"Root tasks:            {list(dag.root_tasks)}")
    print(f"Leaf tasks:            {list(dag.leaf_tasks)}")


def _render_timeline_bar(
    started_at_ms: float | None,
    finished_at_ms: float | None,
    total_ms: float,
    width: int = 32,
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
        "Event-driven parallel execution vs sequential baseline",
    )
    total_tasks = len(parallel_res.results)
    succeeded = len(parallel_res.succeeded_task_ids)
    failed = len(parallel_res.failed_task_ids)
    skipped = len(parallel_res.skipped_task_ids)
    success_rate = (succeeded / total_tasks * 100.0) if total_tasks else 0.0

    print(
        f"Parallel run   (max_concurrency={parallel_res.max_concurrency}): "
        f"{parallel_res.total_duration_ms:.2f} ms"
    )
    print(
        f"Sequential run (max_concurrency={sequential_res.max_concurrency}): "
        f"{sequential_res.total_duration_ms:.2f} ms"
    )
    print(
        f"Task outcomes: {succeeded} success / {failed} error / {skipped} skipped "
        f"(success rate: {success_rate:.1f}%)"
    )

    print("\nParallel execution timeline:")
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
            f"  {task_id:<8} | {res.tool_name:<15} | {res.status:<7} | "
            f"[{bar}] | {window} | {res.duration_ms:6.1f} ms | {detail}"
        )


def print_joiner_section(
    model: str,
    joiner_res: JoinerResult,
    joining_ms: float,
) -> None:
    print_header("5/6", "JOINER", f"Final answer synthesis ({model})")
    print(f"Deterministic status: {joiner_res.status}")
    print(f"Joining latency:      {joining_ms:.2f} ms ({joining_ms / 1000:.2f} s)")
    if joiner_res.missing_information:
        print("Missing information:")
        for item in joiner_res.missing_information:
            print(f"  - {item}")
    else:
        print("Missing information: none")
    print("\nFinal answer:")
    print(f"  {joiner_res.answer}")


def print_metrics_summary(summary: ScenarioSummary) -> None:
    print_header("6/6", "METRICS", f"Summary for scenario '{summary.name}'")
    e2e_ms = (
        summary.planning_ms
        + summary.compile_ms
        + summary.exec_par_ms
        + summary.joining_ms
    )
    rows = [
        ("Planning latency", f"{summary.planning_ms:.2f} ms ({summary.planning_ms / 1000:.2f} s)"),
        ("Compilation latency", f"{summary.compile_ms:.2f} ms"),
        ("Execution latency (parallel)", f"{summary.exec_par_ms:.2f} ms"),
        ("Execution latency (sequential)", f"{summary.exec_seq_ms:.2f} ms"),
        ("Joining latency", f"{summary.joining_ms:.2f} ms ({summary.joining_ms / 1000:.2f} s)"),
        ("Effective parallelism", f"{summary.effective_parallelism:.2f}x"),
        ("Measured speedup (T_seq / T_par)", f"{summary.speedup:.2f}x"),
        ("Task success rate", f"{summary.success_rate:.1f}%"),
        ("Joiner status", summary.joiner_status),
        ("Plan validity", "VALID" if summary.plan_valid else "INVALID"),
        ("LLM calls", str(summary.llm_calls)),
        ("End-to-end latency", f"{e2e_ms:.2f} ms ({e2e_ms / 1000:.2f} s)"),
    ]
    label_width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"  {label:<{label_width}} : {value}")


def print_comparative_table(summaries: list[ScenarioSummary]) -> None:
    print_banner("CROSS-SCENARIO COMPARISON")
    header = (
        f"{'Scenario':<25} | {'Valid':<5} | {'Nodes/Lvl':<9} | "
        f"{'Plan(s)':>7} | {'Par(ms)':>7} | {'Seq(ms)':>7} | "
        f"{'Speedup':>7} | {'Join(s)':>7} | {'Status':<8} | {'LLM':>3}"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s.name:<25} | {('OK' if s.plan_valid else 'FAIL'):<5} | "
            f"{f'{s.nodes}/{s.levels}':<9} | "
            f"{s.planning_ms / 1000:7.2f} | {s.exec_par_ms:7.1f} | "
            f"{s.exec_seq_ms:7.1f} | {s.speedup:6.2f}x | "
            f"{s.joining_ms / 1000:7.2f} | {s.joiner_status:<8} | {s.llm_calls:>3}"
        )


# =============================================================================
# Single scenario runner
# =============================================================================


async def run_scenario(
    scenario: Scenario,
    catalog: dict[str, str],
    planner: Planner,
    compiler: PlanCompiler,
    parallel_executor: Executor,
    sequential_executor: Executor,
    joiner: Joiner,
) -> ScenarioSummary:
    print_banner(f"SCENARIO: {scenario.name} — {scenario.title}")
    llm_calls = 0

    # 2. Planner
    t0 = time.perf_counter()
    llm_calls += 1
    try:
        plan = await planner.aplan(request=scenario.request, available_tools=catalog)
        planning_ms = (time.perf_counter() - t0) * 1000.0
    except PlannerOutputError as exc:
        planning_ms = (time.perf_counter() - t0) * 1000.0
        print_header("2/6", "PLANNER", f"Parsing error ({planning_ms:.2f} ms)")
        print(exc)
        print("RAW OUTPUT:", exc.raw_output)
        return ScenarioSummary(
            name=scenario.name,
            plan_valid=False,
            nodes=0,
            levels=0,
            planning_ms=planning_ms,
            compile_ms=0.0,
            exec_par_ms=0.0,
            exec_seq_ms=0.0,
            joining_ms=0.0,
            effective_parallelism=0.0,
            speedup=0.0,
            success_rate=0.0,
            joiner_status="parse_err",
            llm_calls=llm_calls,
        )

    print_planner_section(scenario, MODEL_NAME, plan, planning_ms)

    # 3. Compiler
    t1 = time.perf_counter()
    try:
        dag = compiler.compile(plan)
        compile_ms = (time.perf_counter() - t1) * 1000.0
    except PlanCompilationError as exc:
        compile_ms = (time.perf_counter() - t1) * 1000.0
        print_header("3/6", "PLAN COMPILER", f"Compilation error ({compile_ms:.2f} ms)")
        print(exc)
        return ScenarioSummary(
            name=scenario.name,
            plan_valid=False,
            nodes=len(plan.tasks),
            levels=0,
            planning_ms=planning_ms,
            compile_ms=compile_ms,
            exec_par_ms=0.0,
            exec_seq_ms=0.0,
            joining_ms=0.0,
            effective_parallelism=0.0,
            speedup=0.0,
            success_rate=0.0,
            joiner_status="comp_err",
            llm_calls=llm_calls,
        )

    print_compiler_section(dag, compile_ms)

    # 4. Executor (parallel vs sequential baseline on the same CompiledDAG)
    parallel_res = await parallel_executor.execute(dag)
    sequential_res = await sequential_executor.execute(dag)
    print_executor_section(parallel_res, sequential_res)

    # 5. Joiner
    t2 = time.perf_counter()
    llm_calls += 1
    try:
        joiner_res = await joiner.join(
            request=scenario.request,
            execution_result=parallel_res,
            plan=plan,
        )
        joining_ms = (time.perf_counter() - t2) * 1000.0
    except JoinerOutputError as exc:
        joining_ms = (time.perf_counter() - t2) * 1000.0
        print_header("5/6", "JOINER", f"Parsing error ({joining_ms:.2f} ms)")
        print(exc)
        print("RAW OUTPUT:", exc.raw_output)
        joiner_res = None

    if joiner_res is not None:
        print_joiner_section(MODEL_NAME, joiner_res, joining_ms)

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

    summary = ScenarioSummary(
        name=scenario.name,
        plan_valid=True,
        nodes=len(dag.tasks),
        levels=len(dag.levels),
        planning_ms=planning_ms,
        compile_ms=compile_ms,
        exec_par_ms=parallel_res.total_duration_ms,
        exec_seq_ms=sequential_res.total_duration_ms,
        joining_ms=joining_ms,
        effective_parallelism=parallel_res.effective_parallelism,
        speedup=speedup,
        success_rate=success_rate,
        joiner_status=joiner_res.status if joiner_res is not None else "join_err",
        llm_calls=llm_calls,
    )
    print_metrics_summary(summary)
    return summary


# =============================================================================
# Entrypoint
# =============================================================================


async def main(selected: str, reasoning: bool) -> None:
    registry = build_registry()
    catalog = registry.to_planner_catalog()
    print_registry_section(registry, catalog)
    print(
        f"Model config: {MODEL_NAME} "
        f"(planner reasoning={reasoning}, joiner reasoning=False, temperature=0.0)"
    )

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
    compiler = PlanCompiler(registry=registry)
    parallel_executor = Executor(registry=registry, max_concurrency=8)
    sequential_executor = Executor(registry=registry, max_concurrency=1)
    joiner = Joiner(llm=joiner_llm)

    chosen_scenarios = (
        list(SCENARIOS)
        if selected == "all"
        else [s for s in SCENARIOS if s.name == selected]
    )

    summaries: list[ScenarioSummary] = []
    for scenario in chosen_scenarios:
        summary = await run_scenario(
            scenario=scenario,
            catalog=catalog,
            planner=planner,
            compiler=compiler,
            parallel_executor=parallel_executor,
            sequential_executor=sequential_executor,
            joiner=joiner,
        )
        summaries.append(summary)

    if len(summaries) > 1:
        print_comparative_table(summaries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-scenario stress test for Planner -> Compiler -> Executor -> Joiner."
    )
    parser.add_argument(
        "--scenario",
        choices=["all", *(s.name for s in SCENARIOS)],
        default="all",
        help="Scenario to run (default: all).",
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Disable Qwen3 reasoning mode (enabled by default).",
    )
    args = parser.parse_args()
    asyncio.run(main(args.scenario, reasoning=not args.no_reasoning))
