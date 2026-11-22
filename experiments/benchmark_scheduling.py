import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from llm_compiler.compiler import (
    CompiledDAG,
    PlanCompiler,
    ToolRegistry,
    ToolSpec,
)
from llm_compiler.executor import (
    ExecutionResult,
    Executor,
    TaskExecutionResult,
    resolve_task_inputs,
)
from llm_compiler.planner import ExecutionPlan, InputBinding, TaskSpec

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MAX_COL = 86


# =============================================================================
# Minimal ANSI styling helper
# =============================================================================


class Style:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def cyan(self, text: str) -> str:
        return self._wrap("1;36", text)

    def green(self, text: str) -> str:
        return self._wrap("1;32", text)

    def yellow(self, text: str) -> str:
        return self._wrap("1;33", text)

    def magenta(self, text: str) -> str:
        return self._wrap("1;35", text)


# =============================================================================
# Deterministic synthetic tool for exact latency control
# =============================================================================


def build_synthetic_registry() -> ToolRegistry:
    async def timed_step(
        label: str,
        delay_ms: float,
        in_a: Any = None,
        in_b: Any = None,
        in_c: Any = None,
        in_d: Any = None,
    ) -> dict[str, Any]:
        await asyncio.sleep(delay_ms / 1000.0)
        return {
            "label": label,
            "inputs_received": sum(x is not None for x in (in_a, in_b, in_c, in_d)),
        }

    return ToolRegistry(
        [
            ToolSpec(
                name="timed_step",
                description="Deterministic async step with controlled latency",
                func=timed_step,
            )
        ]
    )


# =============================================================================
# Alternative engine: dynamic LangGraph StateGraph (Pregel BSP supersteps)
# =============================================================================


def _merge_results(
    left: dict[str, TaskExecutionResult],
    right: dict[str, TaskExecutionResult],
) -> dict[str, TaskExecutionResult]:
    merged = dict(left)
    merged.update(right)
    return merged


class _BSPState(TypedDict):
    results: Annotated[dict[str, TaskExecutionResult], _merge_results]


async def execute_via_langgraph_bsp(
    dag: CompiledDAG,
    registry: ToolRegistry,
) -> tuple[ExecutionResult, float]:
    """Compile a CompiledDAG directly into a dynamic LangGraph StateGraph (Pregel BSP).

    Returns (ExecutionResult, graph_compile_ms).
    """
    compile_t0 = time.perf_counter()
    builder: StateGraph[_BSPState, Any, _BSPState, _BSPState] = StateGraph(_BSPState)

    t0_holder: list[float] = [0.0]

    def make_node(task: TaskSpec) -> Any:
        tool_spec = registry[task.tool_name]

        async def _node(state: _BSPState) -> dict[str, dict[str, TaskExecutionResult]]:
            start_t = time.perf_counter()
            resolved = resolve_task_inputs(task, state["results"])
            output = await tool_spec.func(**resolved)
            end_t = time.perf_counter()
            return {
                "results": {
                    task.task_id: TaskExecutionResult(
                        task_id=task.task_id,
                        tool_name=task.tool_name,
                        status="success",
                        output=output,
                        duration_ms=(end_t - start_t) * 1000.0,
                        started_at_ms=(start_t - t0_holder[0]) * 1000.0,
                        finished_at_ms=(end_t - t0_holder[0]) * 1000.0,
                    )
                }
            }

        return _node

    for tid in dag.topological_order:
        builder.add_node(tid, make_node(dag.get_task(tid)))

    for root_id in dag.root_tasks:
        builder.add_edge(START, root_id)

    for tid in dag.topological_order:
        deps = sorted(dag.dependencies[tid])
        if len(deps) == 1:
            builder.add_edge(deps[0], tid)
        elif len(deps) > 1:
            builder.add_edge(deps, tid)

    for leaf_id in dag.leaf_tasks:
        builder.add_edge(leaf_id, END)

    subgraph = builder.compile()
    graph_compile_ms = (time.perf_counter() - compile_t0) * 1000.0

    t0_holder[0] = time.perf_counter()
    final_state = await subgraph.ainvoke({"results": {}})
    total_ms = (time.perf_counter() - t0_holder[0]) * 1000.0

    ordered = {tid: final_state["results"][tid] for tid in dag.topological_order}
    return (
        ExecutionResult(
            objective=dag.objective,
            results=ordered,
            total_duration_ms=total_ms,
            max_concurrency=None,
        ),
        graph_compile_ms,
    )


# =============================================================================
# 4 benchmark DAG topologies
# =============================================================================


@dataclass(frozen=True)
class TopologySpec:
    name: str
    title: str
    description: str
    plan: ExecutionPlan


def _task(
    tid: str,
    delay_ms: float,
    deps: list[str] | None = None,
) -> TaskSpec:
    inputs: dict[str, Any] = {"label": tid, "delay_ms": delay_ms}
    slot_names = ("in_a", "in_b", "in_c", "in_d")
    for idx, dep in enumerate(deps or ()):
        inputs[slot_names[idx]] = InputBinding(task_id=dep, output_path=["label"])
    return TaskSpec(
        task_id=tid,
        tool_name="timed_step",
        description=f"Step {tid} ({delay_ms:.0f}ms)",
        inputs=inputs,
    )


TOPOLOGIES: tuple[TopologySpec, ...] = (
    TopologySpec(
        name="symmetric_fanout",
        title="Symmetric Fan-Out (fairness control)",
        description="4 equal roots (120ms each) -> 1 join (40ms). BSP and Event-driven should tie.",
        plan=ExecutionPlan(
            objective="Symmetric 4-way fan-out",
            tasks=[
                _task("T1", 120),
                _task("T2", 120),
                _task("T3", 120),
                _task("T4", 120),
                _task("T5", 40, ["T1", "T2", "T3", "T4"]),
            ],
        ),
    ),
    TopologySpec(
        name="asymmetric_branches",
        title="Asymmetric 2-Branch Diamond",
        description="Branch A: T1(50)->T3(250) | Branch B: T2(300)->T4(50) | Join: T5(50).",
        plan=ExecutionPlan(
            objective="Classic asymmetric BSP barrier test",
            tasks=[
                _task("T1", 50),
                _task("T2", 300),
                _task("T3", 250, ["T1"]),
                _task("T4", 50, ["T2"]),
                _task("T5", 50, ["T3", "T4"]),
            ],
        ),
    ),
    TopologySpec(
        name="staggered_multi_branch",
        title="Multi-Branch Diamond (real test2 profile)",
        description="Italy(100)->T5(40), Germany(140)->T6(40) -> T7(40) | France(380)+FX(120)->T8(40).",
        plan=ExecutionPlan(
            objective="Realistic multi-branch latency profile from test2",
            tasks=[
                _task("T1", 100),
                _task("T2", 380),
                _task("T3", 140),
                _task("T4", 120),
                _task("T5", 40, ["T1"]),
                _task("T6", 40, ["T3"]),
                _task("T7", 40, ["T5", "T6"]),
                _task("T8", 40, ["T2", "T4"]),
            ],
        ),
    ),
    TopologySpec(
        name="deep_asymmetric_pipeline",
        title="Deep 4-Level Asymmetric Pipeline",
        description="A1(40)->A2(120)->A3(120) vs B1(260)->B2(40)->B3(40) -> J(40).",
        plan=ExecutionPlan(
            objective="Deep asymmetric 4-level pipeline",
            tasks=[
                _task("A1", 40),
                _task("B1", 260),
                _task("A2", 120, ["A1"]),
                _task("B2", 40, ["B1"]),
                _task("A3", 120, ["A2"]),
                _task("B3", 40, ["B2"]),
                _task("J", 40, ["A3", "B3"]),
            ],
        ),
    ),
)


@dataclass(frozen=True)
class TopologyBenchmarkResult:
    name: str
    title: str
    nodes: int
    levels: int
    repeats: int
    seq_median_ms: float
    bsp_exec_median_ms: float
    bsp_compile_median_ms: float
    bsp_total_median_ms: float
    event_median_ms: float
    speedup_bsp: float
    speedup_event: float
    event_vs_bsp_gain_pct: float
    event_vs_bsp_saved_ms: float


# =============================================================================
# Human-readable visual display helpers
# =============================================================================


def _gantt_bar(
    started_ms: float | None,
    finished_ms: float | None,
    scale_ms: float,
    width: int = 34,
) -> str:
    if started_ms is None or finished_ms is None or scale_ms <= 0:
        return "·" * width
    s_idx = min(width - 1, max(0, int((started_ms / scale_ms) * width)))
    e_idx = min(width, max(s_idx + 1, int((finished_ms / scale_ms) * width)))
    return "·" * s_idx + "█" * (e_idx - s_idx) + "·" * (width - e_idx)


def _metric_bar(value_ms: float, max_ms: float, width: int = 28) -> str:
    filled = max(1, int((value_ms / max_ms) * width)) if max_ms > 0 else 1
    return "█" * min(width, filled) + "░" * max(0, width - filled)


def print_topology_result(
    style: Style,
    spec: TopologySpec,
    res: TopologyBenchmarkResult,
    *,
    bsp_sample: ExecutionResult,
    event_sample: ExecutionResult,
    show_gantt: bool,
) -> None:
    print(f"\n{style.cyan(f' {spec.name} ')} {style.bold(spec.title)}")
    print(f"  {style.dim(spec.description)}")
    print(f"  {style.dim(f'DAG: {res.nodes} nodes, {res.levels} levels (median of {res.repeats} runs)')}")

    max_ms = max(res.seq_median_ms, res.bsp_exec_median_ms, res.event_median_ms)
    print()
    print(
        f"  Sequential (max=1)     │{style.dim(_metric_bar(res.seq_median_ms, max_ms))}│ "
        f"{res.seq_median_ms:6.1f} ms  (1.00x)"
    )
    print(
        f"  LangGraph Pregel BSP   │{style.yellow(_metric_bar(res.bsp_exec_median_ms, max_ms))}│ "
        f"{res.bsp_exec_median_ms:6.1f} ms  ({res.speedup_bsp:.2f}x) "
        f"{style.dim(f'[+{res.bsp_compile_median_ms:.1f}ms compile]')}"
    )
    gain_str = (
        style.green(f"  → {res.event_vs_bsp_gain_pct:+.1f}% vs BSP ({res.event_vs_bsp_saved_ms:+.0f}ms)")
        if res.event_vs_bsp_gain_pct >= 2.0
        else style.dim(f"  → {res.event_vs_bsp_gain_pct:+.1f}% vs BSP (tie)")
    )
    print(
        f"  Executor Event-driven  │{style.green(_metric_bar(res.event_median_ms, max_ms))}│ "
        f"{res.event_median_ms:6.1f} ms  ({style.green(f'{res.speedup_event:.2f}x')}){gain_str}"
    )

    if show_gantt:
        scale_ms = max(bsp_sample.total_duration_ms, event_sample.total_duration_ms)
        print(style.dim(f"\n  Side-by-side Gantt (same time scale: 0 → {scale_ms:.0f} ms):"))
        print(f"  {style.yellow('LangGraph Pregel BSP')} ({bsp_sample.total_duration_ms:.0f}ms):")
        for tid, tres in bsp_sample.results.items():
            bar = _gantt_bar(tres.started_at_ms, tres.finished_at_ms, scale_ms)
            print(
                f"    {tid:<4} │{style.yellow(bar)}│ "
                f"{tres.started_at_ms:5.0f}→{tres.finished_at_ms:5.0f}ms"
            )
        print(f"  {style.green('Executor Event-driven')} ({event_sample.total_duration_ms:.0f}ms):")
        for tid, tres in event_sample.results.items():
            bar = _gantt_bar(tres.started_at_ms, tres.finished_at_ms, scale_ms)
            print(
                f"    {tid:<4} │{style.green(bar)}│ "
                f"{tres.started_at_ms:5.0f}→{tres.finished_at_ms:5.0f}ms"
            )


def print_summary_table(style: Style, results: list[TopologyBenchmarkResult]) -> None:
    print(f"\n{style.bold('═' * MAX_COL)}")
    print(style.bold("  SCHEDULING BENCHMARK SUMMARY — EVENT-DRIVEN vs LANGGRAPH PREGEL BSP"))
    print(style.bold("═" * MAX_COL))
    header = (
        f"  {'Topology':<25} {'Seq':>7} {'BSP':>7} {'Event':>7} "
        f"{'Spd BSP':>8} {'Spd Evt':>8} {'Evt vs BSP':>12}"
    )
    print(style.bold(header))
    print(style.dim("  " + "─" * (MAX_COL - 2)))
    for r in results:
        gain = f"-{r.event_vs_bsp_gain_pct:.1f}%" if r.event_vs_bsp_gain_pct > 0 else f"+{abs(r.event_vs_bsp_gain_pct):.1f}%"
        gain_colored = style.green(f"{gain:>12}") if r.event_vs_bsp_gain_pct >= 2.0 else style.dim(f"{gain:>12}")
        print(
            f"  {r.name:<25} {r.seq_median_ms:5.0f}ms {r.bsp_exec_median_ms:5.0f}ms "
            f"{r.event_median_ms:5.0f}ms {r.speedup_bsp:7.2f}x {style.green(f'{r.speedup_event:7.2f}x')} "
            f"{gain_colored}"
        )


# =============================================================================
# Runner
# =============================================================================


async def main(*, repeats: int, all_gantt: bool, use_color: bool, save: bool) -> None:
    style = Style(enabled=use_color)
    registry = build_synthetic_registry()
    compiler = PlanCompiler(registry)
    seq_executor = Executor(registry, max_concurrency=1)
    event_executor = Executor(registry, max_concurrency=8)

    print(f"\n{style.bold('═' * MAX_COL)}")
    print(style.bold("  DETERMINISTIC SCHEDULING BENCHMARK (ZERO LLM VARIANCE)"))
    print(style.bold("═" * MAX_COL))
    print(
        "  Engines : Sequential (max=1) vs LangGraph Dynamic StateGraph (Pregel BSP) "
        "vs Event-driven Executor (max=8)"
    )
    print(f"  Repeats : {repeats} runs per engine per topology (reporting medians)")

    all_results: list[TopologyBenchmarkResult] = []

    for spec in TOPOLOGIES:
        dag = compiler.compile(spec.plan)

        seq_times: list[float] = []
        bsp_exec_times: list[float] = []
        bsp_compile_times: list[float] = []
        event_times: list[float] = []

        last_bsp: ExecutionResult | None = None
        last_event: ExecutionResult | None = None

        for _ in range(repeats):
            seq_res = await seq_executor.execute(dag)
            assert seq_res.all_succeeded
            seq_times.append(seq_res.total_duration_ms)

            bsp_res, bsp_comp_ms = await execute_via_langgraph_bsp(dag, registry)
            assert bsp_res.all_succeeded
            assert len(bsp_res.results) == len(dag.tasks)
            bsp_exec_times.append(bsp_res.total_duration_ms)
            bsp_compile_times.append(bsp_comp_ms)
            last_bsp = bsp_res

            event_res = await event_executor.execute(dag)
            assert event_res.all_succeeded
            event_times.append(event_res.total_duration_ms)
            last_event = event_res

        med_seq = statistics.median(seq_times)
        med_bsp_exec = statistics.median(bsp_exec_times)
        med_bsp_comp = statistics.median(bsp_compile_times)
        med_bsp_total = med_bsp_exec + med_bsp_comp
        med_event = statistics.median(event_times)

        saved_ms = med_bsp_exec - med_event
        gain_pct = (saved_ms / med_bsp_exec * 100.0) if med_bsp_exec > 0 else 0.0

        bench_res = TopologyBenchmarkResult(
            name=spec.name,
            title=spec.title,
            nodes=len(dag.tasks),
            levels=len(dag.levels),
            repeats=repeats,
            seq_median_ms=round(med_seq, 2),
            bsp_exec_median_ms=round(med_bsp_exec, 2),
            bsp_compile_median_ms=round(med_bsp_comp, 2),
            bsp_total_median_ms=round(med_bsp_total, 2),
            event_median_ms=round(med_event, 2),
            speedup_bsp=round(med_seq / med_bsp_exec, 2),
            speedup_event=round(med_seq / med_event, 2),
            event_vs_bsp_gain_pct=round(gain_pct, 1),
            event_vs_bsp_saved_ms=round(saved_ms, 2),
        )
        all_results.append(bench_res)

        assert last_bsp is not None and last_event is not None
        print_topology_result(
            style,
            spec,
            bench_res,
            bsp_sample=last_bsp,
            event_sample=last_event,
            show_gantt=all_gantt or spec.name == "asymmetric_branches",
        )

    print_summary_table(style, all_results)

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / "benchmark_scheduling.json"
        payload = {
            "benchmark": "deterministic_scheduling",
            "timestamp_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
            "repeats": repeats,
            "results": [asdict(r) for r in all_results],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n  {style.green('Saved JSON')} : {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deterministic scheduling benchmark: Event-driven vs LangGraph Pregel BSP."
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeated runs per topology (default: 3).",
    )
    parser.add_argument(
        "--all-gantt",
        action="store_true",
        help="Show side-by-side Gantt timeline for every topology.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI terminal colors.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving JSON results to results/benchmark_scheduling.json.",
    )
    args = parser.parse_args()
    asyncio.run(
        main(
            repeats=args.repeats,
            all_gantt=args.all_gantt,
            use_color=(not args.no_color) and sys.stdout.isatty(),
            save=not args.no_save,
        )
    )
