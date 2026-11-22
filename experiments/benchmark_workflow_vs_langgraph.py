"""Deterministic (and optional live) benchmark: Pure-Python `workflow.py` vs LangGraph `StateGraph`.

Runs the EXACT same `llm_compiler` core components and step functions across
4 execution paths of increasing transition depth (2 -> 4 -> 8 hops):
  1. empty_plan_shortcut  (2 hops: planner -> joiner)
  2. v1_linear_happy_path (4 hops: planner -> compiler -> executor -> joiner)
  3. v2_compile_retry_x2  (8 hops: 2 compile retries -> executor -> joiner)
  4. v3_runtime_replan_x1 (8 hops: Round 1 partial -> replan -> Round 2 complete)

Measures:
  - Behavioral equivalence (status, final_answer, llm_calls, compile_retries, replans)
  - Pure orchestration wall time (median & p95 over N deterministic iterations)
  - Exact LangGraph overhead (ms total & ms per transition hop)

Usage:
    uv run python experiments/benchmark_workflow_vs_langgraph.py --iters 100
"""

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any, Literal

from pydantic import BaseModel

from langgraph_adapters.compiler_graph import build_compiler_graph
from llm_compiler import (
    ExecutionPlan,
    Executor,
    InputBinding,
    Joiner,
    JoinerDecisionOutput,
    JoinerOutput,
    PlanCompiler,
    Planner,
    TaskSpec,
    ToolRegistry,
    ToolSpec,
    run_compiler_workflow,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ---------------------------------------------------------------------------
# Zero-latency deterministic fakes (isolate pure orchestration cost)
# ---------------------------------------------------------------------------
class _SeqRunnable:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self._idx = 0

    def reset(self) -> None:
        self._idx = 0

    async def ainvoke(self, messages: list[Any]) -> dict[str, Any]:
        pos = min(self._idx, len(self._responses) - 1)
        self._idx += 1
        return self._responses[pos]


class _FakeLLM:
    def __init__(self, by_schema: dict[type[Any], list[dict[str, Any]]]) -> None:
        self._runnables = {s: _SeqRunnable(r) for s, r in by_schema.items()}

    def reset(self) -> None:
        for r in self._runnables.values():
            r.reset()

    def with_structured_output(self, **kwargs: Any) -> _SeqRunnable:
        return self._runnables[kwargs["schema"]]


class _LookupIn(BaseModel):
    item: str


class _LookupOut(BaseModel):
    value: float


class _MathIn(BaseModel):
    op: Literal["add", "subtract"]
    a: float
    b: float


class _MathOut(BaseModel):
    result: float


def _build_zero_ms_registry() -> ToolRegistry:
    async def primary(item: str) -> dict[str, float]:
        if item == "FAIL":
            raise RuntimeError("primary unavailable")
        return {"value": 100.0}

    async def backup(item: str) -> dict[str, float]:
        return {"value": 40.0}

    def calc(op: Literal["add", "subtract"], a: float, b: float) -> dict[str, float]:
        return {"result": a + b if op == "add" else a - b}

    return ToolRegistry(
        [
            ToolSpec("primary", "Primary lookup", primary, _LookupIn, _LookupOut),
            ToolSpec("backup", "Backup lookup", backup, _LookupIn, _LookupOut),
            ToolSpec("calc", "Arithmetic", calc, _MathIn, _MathOut),
        ]
    )


def _valid_plan() -> ExecutionPlan:
    return ExecutionPlan(
        objective="Add OK and OK2",
        tasks=[
            TaskSpec(task_id="t1", tool_name="primary", description="get A", inputs={"item": "A"}),
            TaskSpec(task_id="t2", tool_name="primary", description="get B", inputs={"item": "B"}),
            TaskSpec(
                task_id="t3",
                tool_name="calc",
                description="sum",
                inputs={
                    "op": "add",
                    "a": InputBinding(task_id="t1", output_path=["value"]),
                    "b": InputBinding(task_id="t2", output_path=["value"]),
                },
            ),
        ],
    )


def _bad_plan() -> ExecutionPlan:
    return ExecutionPlan(
        objective="Bad plan",
        tasks=[TaskSpec(task_id="t1", tool_name="unknown_tool", description="bad", inputs={})],
    )


def _replan_round1_plan() -> ExecutionPlan:
    return ExecutionPlan(
        objective="Fetch A and FAIL",
        tasks=[
            TaskSpec(task_id="t1", tool_name="primary", description="get A", inputs={"item": "A"}),
            TaskSpec(task_id="t2", tool_name="primary", description="get FAIL", inputs={"item": "FAIL"}),
            TaskSpec(
                task_id="t3",
                tool_name="calc",
                description="sum",
                inputs={
                    "op": "add",
                    "a": InputBinding(task_id="t1", output_path=["value"]),
                    "b": InputBinding(task_id="t2", output_path=["value"]),
                },
            ),
        ],
    )


def _replan_round2_plan() -> ExecutionPlan:
    return ExecutionPlan(
        objective="Recover FAIL with backup",
        tasks=[
            TaskSpec(task_id="t1", tool_name="backup", description="backup", inputs={"item": "FAIL"}),
            TaskSpec(
                task_id="t2",
                tool_name="calc",
                description="sum",
                inputs={
                    "op": "add",
                    "a": 100.0,
                    "b": InputBinding(task_id="t1", output_path=["value"]),
                },
            ),
        ],
    )


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    hops: int
    max_compile_retries: int
    max_replans: int
    planner_responses: list[dict[str, Any]]
    joiner_responses: dict[type[Any], list[dict[str, Any]]]


SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        scenario_id="empty_plan_shortcut",
        hops=2,
        max_compile_retries=0,
        max_replans=0,
        planner_responses=[
            {"parsed": ExecutionPlan(objective="Greeting", tasks=[]), "parsing_error": None, "raw": "{}"}
        ],
        joiner_responses={
            JoinerOutput: [{"parsed": JoinerOutput(answer="Ciao!"), "parsing_error": None, "raw": "{}"}]
        },
    ),
    ScenarioSpec(
        scenario_id="v1_linear_happy_path",
        hops=4,
        max_compile_retries=0,
        max_replans=0,
        planner_responses=[{"parsed": _valid_plan(), "parsing_error": None, "raw": "{}"}],
        joiner_responses={
            JoinerOutput: [{"parsed": JoinerOutput(answer="200.0"), "parsing_error": None, "raw": "{}"}]
        },
    ),
    ScenarioSpec(
        scenario_id="v2_compile_retry_x2",
        hops=8,
        max_compile_retries=2,
        max_replans=0,
        planner_responses=[
            {"parsed": _bad_plan(), "parsing_error": None, "raw": "{}"},
            {"parsed": _bad_plan(), "parsing_error": None, "raw": "{}"},
            {"parsed": _valid_plan(), "parsing_error": None, "raw": "{}"},
        ],
        joiner_responses={
            JoinerOutput: [{"parsed": JoinerOutput(answer="200.0"), "parsing_error": None, "raw": "{}"}]
        },
    ),
    ScenarioSpec(
        scenario_id="v3_runtime_replan_x1",
        hops=8,
        max_compile_retries=2,
        max_replans=1,
        planner_responses=[
            {"parsed": _replan_round1_plan(), "parsing_error": None, "raw": "{}"},
            {"parsed": _replan_round2_plan(), "parsing_error": None, "raw": "{}"},
        ],
        joiner_responses={
            JoinerDecisionOutput: [
                {
                    "parsed": JoinerDecisionOutput(
                        action="replan",
                        replan_reason="Use backup for FAIL and add 100.0.",
                    ),
                    "parsing_error": None,
                    "raw": "{}",
                }
            ],
            JoinerOutput: [{"parsed": JoinerOutput(answer="140.0"), "parsing_error": None, "raw": "{}"}],
        },
    ),
)


@dataclass
class ScenarioResult:
    scenario_id: str
    hops: int
    iters: int
    equivalent: bool
    py_median_ms: float
    py_p95_ms: float
    lg_median_ms: float
    lg_p95_ms: float
    overhead_median_ms: float
    overhead_per_hop_ms: float
    lg_compile_ms: float


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return ordered[idx]


def _state_signature(state: dict[str, Any]) -> tuple[Any, ...]:
    m = state["metrics"]
    return (
        state.get("status"),
        state.get("final_answer"),
        state.get("error"),
        m.llm_calls,
        m.compile_attempts,
        m.compile_retries,
        m.replan_attempts,
        m.plan_valid,
    )


async def benchmark_scenario(spec: ScenarioSpec, iters: int) -> ScenarioResult:
    registry = _build_zero_ms_registry()
    compiler = PlanCompiler(registry=registry)
    executor = Executor(registry=registry, max_concurrency=8)

    planner_llm = _FakeLLM({ExecutionPlan: spec.planner_responses})
    joiner_llm = _FakeLLM(spec.joiner_responses)
    planner = Planner(llm=planner_llm)  # type: ignore[arg-type]
    joiner = Joiner(llm=joiner_llm)  # type: ignore[arg-type]

    # Time LangGraph one-time graph compilation
    tc0 = time.perf_counter()
    graph = build_compiler_graph(
        registry=registry,
        planner=planner,
        compiler=compiler,
        executor=executor,
        joiner=joiner,
        max_compile_retries=spec.max_compile_retries,
        max_replans=spec.max_replans,
    )
    lg_compile_ms = (time.perf_counter() - tc0) * 1000.0

    # Warm-up 1 iteration each + verify strict behavioral equivalence
    planner_llm.reset()
    joiner_llm.reset()
    py_ref = await run_compiler_workflow(
        "Test request",
        registry=registry,
        planner=planner,
        compiler=compiler,
        executor=executor,
        joiner=joiner,
        max_compile_retries=spec.max_compile_retries,
        max_replans=spec.max_replans,
    )

    planner_llm.reset()
    joiner_llm.reset()
    lg_ref = await graph.ainvoke({"request": "Test request"})

    equivalent = _state_signature(py_ref) == _state_signature(lg_ref)

    py_times: list[float] = []
    lg_times: list[float] = []

    for _ in range(iters):
        planner_llm.reset()
        joiner_llm.reset()
        t0 = time.perf_counter()
        await run_compiler_workflow(
            "Test request",
            registry=registry,
            planner=planner,
            compiler=compiler,
            executor=executor,
            joiner=joiner,
            max_compile_retries=spec.max_compile_retries,
            max_replans=spec.max_replans,
        )
        py_times.append((time.perf_counter() - t0) * 1000.0)

        planner_llm.reset()
        joiner_llm.reset()
        t1 = time.perf_counter()
        await graph.ainvoke({"request": "Test request"})
        lg_times.append((time.perf_counter() - t1) * 1000.0)

    py_med = median(py_times)
    lg_med = median(lg_times)
    ovh = lg_med - py_med

    return ScenarioResult(
        scenario_id=spec.scenario_id,
        hops=spec.hops,
        iters=iters,
        equivalent=equivalent,
        py_median_ms=round(py_med, 3),
        py_p95_ms=round(_p95(py_times), 3),
        lg_median_ms=round(lg_med, 3),
        lg_p95_ms=round(_p95(lg_times), 3),
        overhead_median_ms=round(ovh, 3),
        overhead_per_hop_ms=round(ovh / spec.hops, 3),
        lg_compile_ms=round(lg_compile_ms, 3),
    )


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Pure-Python workflow.py vs LangGraph StateGraph."
    )
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    print(
        f"Benchmark: Pure-Python `run_compiler_workflow` vs LangGraph `StateGraph` "
        f"({args.iters} deterministic iterations per scenario)\n"
    )
    print(
        f"{'Scenario':<22} | {'Hops':>4} | {'Match':>5} | {'Python med':>10} | "
        f"{'LG med':>10} | {'LG Overhead':>11} | {'Ovh/hop':>8} | {'LG build':>8}"
    )
    print("-" * 100)

    results: list[ScenarioResult] = []
    for spec in SCENARIOS:
        res = await benchmark_scenario(spec, args.iters)
        results.append(res)
        print(
            f"{res.scenario_id:<22} | {res.hops:>4} | "
            f"{'YES' if res.equivalent else 'NO':>5} | "
            f"{res.py_median_ms:>7.3f} ms | {res.lg_median_ms:>7.3f} ms | "
            f"{res.overhead_median_ms:>+8.3f} ms | "
            f"{res.overhead_per_hop_ms:>5.3f} ms | {res.lg_compile_ms:>5.2f} ms"
        )
    print("-" * 100)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "benchmark_workflow_vs_langgraph.json"
    payload = {
        "benchmark": "pure_python_workflow_vs_langgraph",
        "timestamp_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "iterations_per_scenario": args.iters,
        "results": [asdict(r) for r in results],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved JSON -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
