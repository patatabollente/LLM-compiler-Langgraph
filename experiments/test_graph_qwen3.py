import argparse
import asyncio
import statistics
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from langgraph_llm_compiler.compiler import PlanCompiler, ToolRegistry, ToolSpec
from langgraph_llm_compiler.executor import Executor
from langgraph_llm_compiler.graph import build_compiler_graph
from langgraph_llm_compiler.joiner import Joiner, JoinerOutput
from langgraph_llm_compiler.llm import create_ollama_chat_model
from langgraph_llm_compiler.planner import (
    ExecutionPlan,
    InputBinding,
    Planner,
    TaskSpec,
)

# =============================================================================
# Experiment configuration (aligned with test_planner_qwen3.py)
# =============================================================================

MODEL_NAME = "qwen3:8b"
REQUEST = (
    "Calcola la differenza tra la popolazione dell'Italia "
    "e quella della Francia (Italia meno Francia)."
)
SIMULATED_LOOKUP_DELAY_S = 0.15
SIMULATED_CALC_DELAY_S = 0.05


# =============================================================================
# Tool contracts & implementations
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


def build_registry(
    *,
    lookup_delay_s: float = SIMULATED_LOOKUP_DELAY_S,
    calc_delay_s: float = SIMULATED_CALC_DELAY_S,
) -> ToolRegistry:
    async def population_lookup(country: str) -> dict[str, str | int]:
        if lookup_delay_s > 0:
            await asyncio.sleep(lookup_delay_s)
        key = country.strip().lower()
        if key not in POPULATIONS:
            raise ValueError(f"Unknown country: {country}")
        return {"country": country, "population": POPULATIONS[key]}

    def calculator(
        operation: Literal["add", "subtract", "multiply", "divide"],
        a: float,
        b: float,
    ) -> dict[str, float]:
        if calc_delay_s > 0:
            time.sleep(calc_delay_s)
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
# Deterministic fake LLM for Experiment B (orchestration overhead isolation)
# =============================================================================


class _FakeStructuredRunnable:
    def __init__(self, payload: Any) -> None:
        self._response = {"parsed": payload, "parsing_error": None, "raw": "{}"}

    async def ainvoke(self, messages: list[Any]) -> dict[str, Any]:
        return self._response


class _FakeChatModel:
    def __init__(self, payload: Any) -> None:
        self._runnable = _FakeStructuredRunnable(payload)

    def with_structured_output(self, **kwargs: Any) -> _FakeStructuredRunnable:
        return self._runnable


def _deterministic_plan() -> ExecutionPlan:
    return ExecutionPlan(
        objective="Calculate Italy minus France population.",
        tasks=[
            TaskSpec(
                task_id="task_1",
                tool_name="population_lookup",
                description="Get Italy population.",
                inputs={"country": "Italy"},
            ),
            TaskSpec(
                task_id="task_2",
                tool_name="population_lookup",
                description="Get France population.",
                inputs={"country": "France"},
            ),
            TaskSpec(
                task_id="task_3",
                tool_name="calculator",
                description="Subtract France population from Italy.",
                inputs={
                    "operation": "subtract",
                    "a": InputBinding(task_id="task_1", output_path=["population"]),
                    "b": InputBinding(task_id="task_2", output_path=["population"]),
                },
            ),
        ],
    )


# =============================================================================
# Direct Python baseline runner (same components, no LangGraph)
# =============================================================================


async def run_direct_pipeline(
    *,
    request: str,
    catalog: dict[str, str],
    planner: Planner,
    compiler: PlanCompiler,
    executor: Executor,
    joiner: Joiner,
) -> dict[str, Any]:
    wall_t0 = time.perf_counter()

    t0 = time.perf_counter()
    plan = await planner.aplan(request=request, available_tools=catalog)
    planning_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    dag = compiler.compile(plan)
    compilation_ms = (time.perf_counter() - t1) * 1000.0

    t2 = time.perf_counter()
    exec_result = await executor.execute(dag)
    execution_ms = (time.perf_counter() - t2) * 1000.0

    t3 = time.perf_counter()
    joiner_result = await joiner.join(
        request=request,
        execution_result=exec_result,
        plan=plan,
    )
    joining_ms = (time.perf_counter() - t3) * 1000.0

    wall_ms = (time.perf_counter() - wall_t0) * 1000.0
    component_ms = planning_ms + compilation_ms + execution_ms + joining_ms

    return {
        "status": joiner_result.status,
        "final_answer": joiner_result.answer,
        "planning_ms": planning_ms,
        "compilation_ms": compilation_ms,
        "execution_ms": execution_ms,
        "joining_ms": joining_ms,
        "component_ms": component_ms,
        "wall_ms": wall_ms,
        "llm_calls": 2,
    }


# =============================================================================
# Experiment A: Live end-to-end with Qwen3 (LangGraph vs direct pipeline)
# =============================================================================


async def run_experiment_a(*, reasoning: bool, runs: int) -> None:
    line = "=" * 78
    print(f"\n{line}")
    print(
        f"[EXPERIMENT A] Live LangGraph vs Direct Pipeline "
        f"({MODEL_NAME}, planner reasoning={reasoning}, joiner reasoning=False, runs={runs})"
    )
    print(line)

    registry = build_registry()
    catalog = registry.to_planner_catalog()

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
    executor = Executor(registry=registry, max_concurrency=8)
    joiner = Joiner(llm=joiner_llm)

    graph = build_compiler_graph(
        registry=registry,
        planner=planner,
        compiler=compiler,
        executor=executor,
        joiner=joiner,
    )

    print("\nLangGraph Mermaid topology:")
    print(graph.get_graph().draw_mermaid().strip())

    graph_wall_samples: list[float] = []
    graph_comp_samples: list[float] = []
    direct_wall_samples: list[float] = []

    for run_idx in range(1, runs + 1):
        print(f"\n--- Run {run_idx}/{runs} ---")

        # 1) LangGraph invocation
        t0 = time.perf_counter()
        state = await graph.ainvoke({"request": REQUEST})
        graph_wall_ms = (time.perf_counter() - t0) * 1000.0
        metrics = state["metrics"]
        graph_comp_ms = metrics.component_latency_ms
        overhead_ms = graph_wall_ms - graph_comp_ms

        graph_wall_samples.append(graph_wall_ms)
        graph_comp_samples.append(graph_comp_ms)

        print(
            f"  [LangGraph] status={state.get('status')} | "
            f"plan_valid={metrics.plan_valid} | llm_calls={metrics.llm_calls} | "
            f"plan={metrics.planning_latency_ms or 0:.1f}ms | "
            f"comp={metrics.compilation_latency_ms or 0:.2f}ms | "
            f"exec={metrics.execution_latency_ms or 0:.1f}ms | "
            f"join={metrics.joining_latency_ms or 0:.1f}ms"
        )
        print(
            f"              component_latency_ms={graph_comp_ms:.2f} ms | "
            f"graph_wall_time_ms={graph_wall_ms:.2f} ms | "
            f"overhead={overhead_ms:.2f} ms"
        )
        if state.get("error") is not None:
            print(f"              ERROR: {state['error']}")
        else:
            print(f"              answer: {state.get('final_answer')}")

        # 2) Direct Python invocation
        direct = await run_direct_pipeline(
            request=REQUEST,
            catalog=catalog,
            planner=planner,
            compiler=compiler,
            executor=executor,
            joiner=joiner,
        )
        direct_wall_samples.append(direct["wall_ms"])
        print(
            f"  [Direct]    status={direct['status']} | llm_calls={direct['llm_calls']} | "
            f"plan={direct['planning_ms']:.1f}ms | comp={direct['compilation_ms']:.2f}ms | "
            f"exec={direct['execution_ms']:.1f}ms | join={direct['joining_ms']:.1f}ms | "
            f"wall={direct['wall_ms']:.2f} ms"
        )

    print("\nSummary (medians across live runs):")
    print(f"  LangGraph graph_wall_time_ms (median) : {statistics.median(graph_wall_samples):.2f} ms")
    print(f"  LangGraph component_latency  (median) : {statistics.median(graph_comp_samples):.2f} ms")
    print(f"  Direct Python wall time      (median) : {statistics.median(direct_wall_samples):.2f} ms")


# =============================================================================
# Experiment B: Deterministic orchestration overhead isolation (no Ollama)
# =============================================================================


async def run_experiment_b(*, iterations: int) -> None:
    line = "=" * 78
    print(f"\n{line}")
    print(
        f"[EXPERIMENT B] Orchestration Overhead Isolation "
        f"(deterministic fake LLMs, 0ms tools, iterations={iterations})"
    )
    print(line)

    registry = build_registry(lookup_delay_s=0.0, calc_delay_s=0.0)
    catalog = registry.to_planner_catalog()

    planner = Planner(llm=_FakeChatModel(_deterministic_plan()))  # type: ignore[arg-type]
    compiler = PlanCompiler(registry=registry)
    executor = Executor(registry=registry, max_concurrency=8)
    joiner = Joiner(
        llm=_FakeChatModel(
            JoinerOutput(answer="La differenza tra Italia e Francia è -9.410.000.")
        )  # type: ignore[arg-type]
    )

    graph = build_compiler_graph(
        registry=registry,
        planner=planner,
        compiler=compiler,
        executor=executor,
        joiner=joiner,
    )

    # Warm-up run
    await graph.ainvoke({"request": REQUEST})
    await run_direct_pipeline(
        request=REQUEST,
        catalog=catalog,
        planner=planner,
        compiler=compiler,
        executor=executor,
        joiner=joiner,
    )

    direct_wall: list[float] = []
    graph_wall: list[float] = []
    graph_components: list[float] = []
    graph_overhead: list[float] = []

    for _ in range(iterations):
        direct = await run_direct_pipeline(
            request=REQUEST,
            catalog=catalog,
            planner=planner,
            compiler=compiler,
            executor=executor,
            joiner=joiner,
        )
        direct_wall.append(direct["wall_ms"])

        t0 = time.perf_counter()
        state = await graph.ainvoke({"request": REQUEST})
        wall_ms = (time.perf_counter() - t0) * 1000.0
        comp_ms = state["metrics"].component_latency_ms
        graph_wall.append(wall_ms)
        graph_components.append(comp_ms)
        graph_overhead.append(wall_ms - comp_ms)

    med_direct = statistics.median(direct_wall)
    med_graph = statistics.median(graph_wall)
    med_comp = statistics.median(graph_components)
    med_overhead = statistics.median(graph_overhead)

    print(f"  Direct Python wall time     : median={med_direct:.3f} ms | mean={statistics.mean(direct_wall):.3f} ms")
    print(f"  LangGraph graph_wall_time   : median={med_graph:.3f} ms | mean={statistics.mean(graph_wall):.3f} ms")
    print(f"  LangGraph component_latency : median={med_comp:.3f} ms | mean={statistics.mean(graph_components):.3f} ms")
    print(f"  LangGraph internal overhead : median={med_overhead:.3f} ms (graph_wall - component_latency)")
    print(f"  Delta vs Direct Python      : median={med_graph - med_direct:+.3f} ms")


async def main(
    *,
    mode: Literal["live", "overhead", "both"],
    reasoning: bool,
    runs: int,
    overhead_iters: int,
) -> None:
    if mode in ("overhead", "both"):
        await run_experiment_b(iterations=overhead_iters)
    if mode in ("live", "both"):
        await run_experiment_a(reasoning=reasoning, runs=runs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LangGraph v1 live experiment (A) and orchestration overhead benchmark (B)."
    )
    parser.add_argument(
        "--mode",
        choices=["live", "overhead", "both"],
        default="both",
        help="Experiment mode: 'live' (Qwen3), 'overhead' (deterministic fake), or 'both' (default).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of repeated live runs for Experiment A (default: 1).",
    )
    parser.add_argument(
        "--overhead-iters",
        type=int,
        default=50,
        help="Number of deterministic iterations for Experiment B (default: 50).",
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Disable reasoning mode on Planner (enabled by default).",
    )
    args = parser.parse_args()
    asyncio.run(
        main(
            mode=args.mode,
            reasoning=not args.no_reasoning,
            runs=args.runs,
            overhead_iters=args.overhead_iters,
        )
    )
