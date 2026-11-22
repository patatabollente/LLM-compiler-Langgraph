import argparse
import asyncio
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from langgraph_adapters.compiler_graph import CompilerState, build_compiler_graph
from llm_compiler.compiler import (
    CompiledDAG,
    PlanCompiler,
    ToolRegistry,
    ToolSpec,
)
from llm_compiler.executor import ExecutionResult, Executor
from llm_compiler.joiner import Joiner, JoinerResult
from llm_compiler.llm import create_ollama_chat_model
from llm_compiler.planner import ExecutionPlan, Planner

# =============================================================================
# Experiment configuration
# =============================================================================

MODEL_NAME = "qwen3:8b"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SIMULATED_EXCHANGE_DELAY_S = 0.12
SIMULATED_SYNC_TOOL_DELAY_S = 0.04


# =============================================================================
# Minimal ANSI styling helper (cleanly disabled with --no-color or non-TTY)
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

    def blue(self, text: str) -> str:
        return self._wrap("1;34", text)

    def green(self, text: str) -> str:
        return self._wrap("1;32", text)

    def yellow(self, text: str) -> str:
        return self._wrap("1;33", text)

    def red(self, text: str) -> str:
        return self._wrap("1;31", text)

    def magenta(self, text: str) -> str:
        return self._wrap("1;35", text)


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
        focus="4 parallel roots (France 380ms vs Italy 100ms) + nested economy.gdp_billion_usd",
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
        focus="4 parallel lookups -> 2 parallel sums -> total -> average (/ 4)",
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
        focus="Atlantis fails -> subtraction skipped -> Italy density completes -> status='partial'",
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
    path: str
    plan_valid: bool
    nodes: int
    levels: int
    planning_ms: float
    compile_ms: float
    exec_par_ms: float
    exec_seq_ms: float
    joining_ms: float
    graph_wall_ms: float
    overhead_ms: float
    effective_parallelism: float
    speedup: float
    success_rate: float
    joiner_status: str
    llm_calls: int


# =============================================================================
# Clean, human-readable, zero-wrap display helpers (<= 88 columns)
# =============================================================================

MAX_COL = 84


def _format_output_human(tool_name: str, output: Any) -> str:
    """Render tool outputs concisely so terminal rows never wrap awkwardly."""
    if not isinstance(output, dict):
        return str(output)

    if tool_name == "country_stats" and "population" in output:
        econ = output.get("economy") or {}
        gdp = econ.get("gdp_billion_usd")
        unemp = econ.get("unemployment_pct")
        return (
            f"{output['country']}: pop={output['population']:,} | "
            f"area={output['area_km2']:,.0f} km² | GDP=${gdp:,.0f}B | unemp={unemp}%"
        )
    if tool_name == "exchange_rate" and "rate" in output:
        return f"{output['pair']} = {output['rate']}"
    if tool_name == "calculator" and "result" in output:
        return f"result = {output['result']:,.2f}"
    if tool_name == "percentage_diff" and "percent_change" in output:
        return f"delta = {output['delta']:+,.2f} ({output['percent_change']:+.2f}%)"
    return str(output)


def _format_task_call(task: Any) -> str:
    """Format a TaskSpec as a clean call expression: foo(a='x', b <- task_1.val)."""
    parts: list[str] = []
    for k, v in task.static_inputs.items():
        parts.append(f"{k}={v!r}")
    for k, b in task.bindings.items():
        ref = f"{b.task_id}.{'.'.join(b.output_path)}" if b.output_path else b.task_id
        parts.append(f"{k} ← {ref}")
    return f"{task.tool_name}({', '.join(parts)})"


def _render_timeline_bar(
    started_at_ms: float | None,
    finished_at_ms: float | None,
    total_ms: float,
    width: int = 24,
) -> str:
    if started_at_ms is None or finished_at_ms is None or total_ms <= 0:
        return "·" * width
    start_idx = min(width - 1, max(0, int((started_at_ms / total_ms) * width)))
    end_idx = min(width, max(start_idx + 1, int((finished_at_ms / total_ms) * width)))
    return "·" * start_idx + "█" * (end_idx - start_idx) + "·" * (width - end_idx)


def _print_phase_bar(style: Style, step: str, title: str, right_tag: str = "") -> None:
    label = f" {step} · {title} "
    tag = f" {right_tag} " if right_tag else ""
    fill = max(2, MAX_COL - len(label) - len(tag))
    print(f"\n{style.cyan(label)}{style.dim('─' * fill)}{style.dim(tag)}")


def print_setup(
    style: Style,
    registry: ToolRegistry,
    graph: Any,
    planner_reasoning: bool,
) -> None:
    print(f"\n{style.bold('═' * MAX_COL)}")
    print(style.bold("  LANGGRAPH LLM COMPILER — STEP-BY-STEP VISUAL TRACE"))
    print(style.bold("═" * MAX_COL))
    print(
        f"  Model    : {style.bold(MODEL_NAME)} "
        f"(planner reasoning={planner_reasoning}, joiner reasoning=False)"
    )
    print(f"  Tools    : {', '.join(registry.names())}")
    try:
        ascii_tree = graph.get_graph().draw_ascii()
        print(style.dim("\n  LangGraph StateGraph (draw_ascii):"))
        for line in ascii_tree.splitlines():
            print(f"    {line}")
    except Exception:  # noqa: BLE001
        print(
            "  Workflow : "
            + style.magenta("START")
            + " → "
            + style.cyan("planner")
            + " → "
            + style.cyan("compiler")
            + " → "
            + style.cyan("executor")
            + " → "
            + style.cyan("joiner")
            + " → "
            + style.magenta("END")
            + style.dim("  (error → END)")
        )


def print_question(style: Style, scenario: Scenario) -> None:
    print(f"\n{style.blue('━' * MAX_COL)}")
    print(f"  {style.bold('SCENARIO:')} {style.blue(scenario.name)} — {scenario.title}")
    print(f"  {style.dim('Focus: ' + scenario.focus)}")
    print(style.blue("━" * MAX_COL))

    _print_phase_bar(style, "0", "USER REQUEST", "START → planner")
    for line in textwrap.wrap(scenario.request, width=MAX_COL - 4):
        print(f"  {line}")


def print_planner_phase(
    style: Style,
    update: dict[str, Any],
    *,
    show_json: bool,
) -> None:
    metrics = update["metrics"]
    err = update.get("error")
    plan: ExecutionPlan | None = update.get("plan")
    ms = metrics.planning_latency_ms or 0.0

    _print_phase_bar(style, "1", "PLANNER (LangGraph node)", f"{ms / 1000:.2f}s · LLM #{metrics.llm_calls}")

    if err is not None or plan is None:
        print(f"  {style.red('FAILED')}: {err.error_type if err else 'Unknown'} — {err.message if err else ''}")
        return

    print(f"  {style.green('OK')} · Generated {style.bold(str(len(plan.tasks)))} tasks:")
    for task in plan.tasks:
        call_str = _format_task_call(task)
        wrapped = textwrap.wrap(call_str, width=MAX_COL - 13)
        print(f"    {style.bold(f'{task.task_id:<6}')}  {wrapped[0]}")
        for cont in wrapped[1:]:
            print(f"            {cont}")

    if show_json:
        print(style.dim("\n  Raw ExecutionPlan JSON:"))
        for line in plan.model_dump_json(indent=2).splitlines():
            print(style.dim(f"    {line}"))


def print_compiler_phase(style: Style, update: dict[str, Any]) -> None:
    metrics = update["metrics"]
    err = update.get("error")
    dag: CompiledDAG | None = update.get("compiled_dag")
    ms = metrics.compilation_latency_ms or 0.0

    _print_phase_bar(style, "2", "COMPILER (LangGraph node)", f"{ms:.2f}ms")

    if err is not None or dag is None:
        print(f"  {style.red('INVALID PLAN')} ({err.error_type if err else 'Unknown'}):")
        if err is not None and err.issues:
            for issue in err.issues:
                print(f"    {style.red('✗')} {issue}")
        elif err is not None:
            print(f"    {err.message}")
        return

    edge_count = sum(len(deps) for deps in dag.dependencies.values())
    max_width = max((len(level) for level in dag.levels), default=0)
    print(
        f"  {style.green('VALID DAG')} · {len(dag.tasks)} nodes, {edge_count} edges, "
        f"{len(dag.levels)} levels (max parallel width: {max_width})"
    )

    print(style.dim("\n  Parallelism layers:"))
    for idx, level in enumerate(dag.levels):
        boxes = "  ".join(f"[{tid}:{dag.get_task(tid).tool_name}]" for tid in level)
        print(f"    L{idx} │ {boxes}")

    # Group dependencies cleanly by downstream consumer
    consumers = [tid for tid in dag.topological_order if dag.get_task(tid).bindings]
    if consumers:
        print(style.dim("\n  Dataflow (who reads from whom):"))
        for tid in consumers:
            bindings = dag.get_task(tid).bindings
            sources = ", ".join(
                f"{b.task_id}.{'.'.join(b.output_path) if b.output_path else '*'}→{p}"
                for p, b in bindings.items()
            )
            print(f"    {style.bold(f'{tid:<6}')} ◄── {sources}")


def print_executor_phase(
    style: Style,
    update: dict[str, Any],
    sequential_res: ExecutionResult | None,
) -> None:
    metrics = update["metrics"]
    err = update.get("error")
    exec_res: ExecutionResult | None = update.get("execution_result")
    ms = metrics.execution_latency_ms or 0.0

    speedup_tag = ""
    if exec_res is not None and sequential_res is not None and exec_res.total_duration_ms > 0:
        sp = sequential_res.total_duration_ms / exec_res.total_duration_ms
        speedup_tag = f" · speedup {sp:.2f}x"

    _print_phase_bar(style, "3", "EXECUTOR (Event-driven async)", f"{ms:.1f}ms{speedup_tag}")

    if err is not None or exec_res is None:
        print(f"  {style.red('EXECUTOR ERROR')}: {err.message if err else 'Unknown'}")
        return

    ok = len(exec_res.succeeded_task_ids)
    fail = len(exec_res.failed_task_ids)
    skip = len(exec_res.skipped_task_ids)

    summary_line = (
        f"  Tasks: {style.green(f'{ok} ok')} / "
        f"{style.red(f'{fail} err') if fail else '0 err'} / "
        f"{style.yellow(f'{skip} skip') if skip else '0 skip'}"
    )
    if sequential_res is not None and exec_res.total_duration_ms > 0:
        sp = sequential_res.total_duration_ms / exec_res.total_duration_ms
        summary_line += (
            f"   │   Parallel: {style.bold(f'{exec_res.total_duration_ms:.1f}ms')} "
            f"vs Seq: {sequential_res.total_duration_ms:.1f}ms "
            f"({style.green(f'{sp:.2f}x')})"
        )
    print(summary_line)

    print(style.dim("\n  Timeline (0ms ──────────────► end):"))
    for tid, res in exec_res.results.items():
        bar = _render_timeline_bar(
            res.started_at_ms,
            res.finished_at_ms,
            exec_res.total_duration_ms,
        )
        if res.status == "success":
            badge = style.green("OK  ")
            timing = f"{res.started_at_ms:5.0f}→{res.finished_at_ms:5.0f}ms ({res.duration_ms:4.0f}ms)"
            detail = _format_output_human(res.tool_name, res.output)
        elif res.status == "error":
            badge = style.red("ERR ")
            timing = f"{res.started_at_ms or 0:5.0f}→{res.finished_at_ms or 0:5.0f}ms ({res.duration_ms:4.0f}ms)"
            detail = style.red(f"{res.error_type}: {res.error}")
        else:
            badge = style.yellow("SKIP")
            timing = "     skipped      "
            detail = style.yellow(res.skip_reason or "upstream failed")

        print(f"    {style.bold(f'{tid:<6}')} {res.tool_name:<15} {badge} │{bar}│ {timing}")
        print(f"           {style.dim('└─')} {detail}")


def print_joiner_phase(style: Style, update: dict[str, Any]) -> None:
    metrics = update["metrics"]
    err = update.get("error")
    joiner_res: JoinerResult | None = update.get("joiner_result")
    ms = metrics.joining_latency_ms or 0.0

    _print_phase_bar(style, "4", "JOINER (LangGraph node)", f"{ms / 1000:.2f}s · LLM #{metrics.llm_calls}")

    if err is not None or joiner_res is None:
        print(f"  {style.red('FAILED')}: {err.message if err else 'Unknown'}")
        return

    status_badge = (
        style.green("COMPLETE")
        if joiner_res.status == "complete"
        else style.yellow("PARTIAL")
        if joiner_res.status == "partial"
        else style.red("FAILED")
    )
    print(f"  Status : {status_badge}")

    if joiner_res.missing_information:
        print(f"  {style.yellow('Missing')}:")
        for item in joiner_res.missing_information:
            for idx, line in enumerate(textwrap.wrap(item, width=MAX_COL - 8)):
                prefix = "    • " if idx == 0 else "      "
                print(f"{prefix}{line}")

    print(f"\n  {style.bold('Answer:')}")
    for para in joiner_res.answer.splitlines():
        if not para.strip():
            print()
            continue
        for line in textwrap.wrap(para, width=MAX_COL - 6):
            print(f"    {line}")


def print_scenario_footer(style: Style, summary: ScenarioSummary) -> None:
    _print_phase_bar(style, "✓", "LANGGRAPH SUMMARY", summary.name)
    print(f"  Path     : {style.magenta(summary.path)}")
    print(
        f"  Timings  : Plan {summary.planning_ms / 1000:.2f}s │ "
        f"Compile {summary.compile_ms:.2f}ms │ "
        f"Exec {summary.exec_par_ms:.1f}ms │ "
        f"Join {summary.joining_ms / 1000:.2f}s"
    )
    print(
        f"  Totals   : Wall {style.bold(f'{summary.graph_wall_ms / 1000:.2f}s')} │ "
        f"LangGraph overhead {summary.overhead_ms:.2f}ms │ "
        f"Speedup {style.green(f'{summary.speedup:.2f}x')} │ "
        f"LLM calls: {summary.llm_calls}"
    )


def print_comparative_table(style: Style, summaries: list[ScenarioSummary]) -> None:
    print(f"\n{style.bold('═' * MAX_COL)}")
    print(style.bold("  CROSS-SCENARIO SUMMARY"))
    print(style.bold("═" * MAX_COL))
    header = (
        f"  {'Scenario':<24} {'Valid':<5} {'DAG':<5} "
        f"{'Plan':>6} {'Exec':>6} {'Spd':>6} {'Join':>6} {'Ovh':>6} {'Status':<8}"
    )
    print(style.bold(header))
    print(style.dim("  " + "─" * (MAX_COL - 2)))
    for s in summaries:
        valid = style.green("OK   ") if s.plan_valid else style.red("FAIL ")
        status_color = (
            style.green(f"{s.joiner_status:<8}")
            if s.joiner_status == "complete"
            else style.yellow(f"{s.joiner_status:<8}")
            if s.joiner_status == "partial"
            else style.red(f"{s.joiner_status:<8}")
        )
        print(
            f"  {s.name:<24} {valid} {f'{s.nodes}n/{s.levels}l':<5} "
            f"{s.planning_ms / 1000:5.2f}s {s.exec_par_ms:4.0f}ms "
            f"{s.speedup:5.2f}x {s.joining_ms / 1000:5.2f}s "
            f"{s.overhead_ms:4.1f}ms {status_color}"
        )


# =============================================================================
# Visual artifact export (LangGraph PNG + Markdown report with Mermaid DAGs)
# =============================================================================


@dataclass(frozen=True)
class ScenarioArtifact:
    scenario: Scenario
    summary: ScenarioSummary
    state: CompilerState


def render_dag_mermaid(
    dag: CompiledDAG,
    execution_result: ExecutionResult | None = None,
) -> str:
    """Render a CompiledDAG as a Mermaid flowchart colored by execution outcome."""
    lines = [
        "flowchart TD",
        "    classDef ok fill:#dcfce7,stroke:#16a34a,color:#14532d",
        "    classDef err fill:#fee2e2,stroke:#dc2626,color:#7f1d1d",
        "    classDef skip fill:#fef3c7,stroke:#d97706,color:#78350f",
    ]

    for tid in dag.topological_order:
        task = dag.get_task(tid)
        status_cls = ""
        meta = ""
        if execution_result is not None and tid in execution_result.results:
            res = execution_result.results[tid]
            if res.status == "success":
                status_cls = ":::ok"
                meta = f" | OK {res.duration_ms:.0f}ms"
            elif res.status == "error":
                status_cls = ":::err"
                meta = f" | ERR {res.error_type or ''}".rstrip()
            else:
                status_cls = ":::skip"
                meta = " | SKIPPED"

        label = f"{tid}: {task.tool_name}{meta}".replace('"', "'")
        lines.append(f'    {tid}["{label}"]{status_cls}')

    for tid in dag.topological_order:
        task = dag.get_task(tid)
        for param, binding in task.bindings.items():
            path = ".".join(binding.output_path) if binding.output_path else "*"
            edge_label = f"{path} -> {param}".replace('"', "'")
            lines.append(f'    {binding.task_id} -->|"{edge_label}"| {tid}')

    return "\n".join(lines)


def export_visual_artifacts(
    *,
    style: Style,
    graph: Any,
    artifacts: list[ScenarioArtifact],
    selected: str,
    planner_reasoning: bool,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    png_path = RESULTS_DIR / "langgraph_workflow.png"
    md_path = RESULTS_DIR / f"trace_{selected}.md"

    png_saved = False
    try:
        graph.get_graph().draw_mermaid_png(output_file_path=str(png_path))
        png_saved = True
    except Exception as exc:  # noqa: BLE001
        print(style.dim(f"\n  [export] Skipped PNG export ({type(exc).__name__}: {exc})"))

    md_lines: list[str] = [
        f"# LangGraph LLM Compiler Trace (`{selected}`)",
        "",
        f"- **Model**: `{MODEL_NAME}` (`planner reasoning={planner_reasoning}`, `joiner reasoning=False`)",
        f"- **Scenarios**: {len(artifacts)}",
        "",
        "## LangGraph StateGraph",
        "",
    ]
    if png_saved:
        md_lines.extend([f"![LangGraph Workflow]({png_path.name})", ""])
    md_lines.extend(
        [
            "```mermaid",
            graph.get_graph().draw_mermaid().strip(),
            "```",
            "",
        ]
    )

    for art in artifacts:
        s = art.summary
        dag = art.state.get("compiled_dag")
        exec_res = art.state.get("execution_result")
        answer = art.state.get("final_answer")

        md_lines.extend(
            [
                f"## Scenario: `{art.scenario.name}` — {art.scenario.title}",
                "",
                f"> **User Request:** {art.scenario.request}",
                "",
                f"- **LangGraph path**: `{s.path}`",
                (
                    f"- **Timings**: Plan `{s.planning_ms / 1000:.2f}s` | "
                    f"Compile `{s.compile_ms:.2f}ms` | "
                    f"Exec `{s.exec_par_ms:.1f}ms` (seq `{s.exec_seq_ms:.1f}ms`, "
                    f"speedup **`{s.speedup:.2f}x`**) | "
                    f"Join `{s.joining_ms / 1000:.2f}s` | Overhead `{s.overhead_ms:.2f}ms`"
                ),
                f"- **Status**: `{s.joiner_status}` | **LLM calls**: `{s.llm_calls}`",
                "",
            ]
        )

        if dag is not None:
            md_lines.extend(
                [
                    "### Compiled DAG",
                    "",
                    "```mermaid",
                    render_dag_mermaid(dag, exec_res),
                    "```",
                    "",
                ]
            )

        if exec_res is not None:
            md_lines.extend(
                [
                    "### Task Execution Details",
                    "",
                    "| Task | Tool | Status | Window (ms) | Duration | Result / Error |",
                    "| :--- | :--- | :---: | :---: | ---: | :--- |",
                ]
            )
            for tid, res in exec_res.results.items():
                if res.status == "success":
                    win = f"{res.started_at_ms:.0f} -> {res.finished_at_ms:.0f}"
                    detail = _format_output_human(res.tool_name, res.output)
                elif res.status == "error":
                    win = f"{res.started_at_ms or 0:.0f} -> {res.finished_at_ms or 0:.0f}"
                    detail = f"{res.error_type}: {res.error}"
                else:
                    win = "skipped"
                    detail = res.skip_reason or "upstream failed"
                detail_safe = detail.replace("|", "\\|")
                md_lines.append(
                    f"| `{tid}` | `{res.tool_name}` | `{res.status}` | {win} | "
                    f"{res.duration_ms:.1f} ms | {detail_safe} |"
                )
            md_lines.append("")

        if answer:
            md_lines.extend(["### Final Answer", "", answer, ""])

    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"\n{style.dim('─' * MAX_COL)}")
    if png_saved:
        print(f"  {style.green('Saved PNG')}    : {png_path}")
    print(f"  {style.green('Saved Report')} : {md_path}")


# =============================================================================
# Single scenario runner via LangGraph astream
# =============================================================================


async def run_scenario(
    *,
    style: Style,
    scenario: Scenario,
    graph: Any,
    sequential_executor: Executor,
    show_json: bool,
) -> ScenarioArtifact:
    print_question(style, scenario)

    visited_nodes: list[str] = ["START"]
    final_state: CompilerState = {"request": scenario.request}
    seq_res: ExecutionResult | None = None

    wall_t0 = time.perf_counter()
    async for chunk in graph.astream({"request": scenario.request}):
        for node_name, update in chunk.items():
            visited_nodes.append(node_name)
            final_state.update(update)

            if node_name == "planner":
                print_planner_phase(style, update, show_json=show_json)
            elif node_name == "compiler":
                print_compiler_phase(style, update)
            elif node_name == "executor":
                dag = final_state.get("compiled_dag")
                exec_res = update.get("execution_result")
                if dag is not None and exec_res is not None:
                    seq_t0 = time.perf_counter()
                    seq_res = await sequential_executor.execute(dag)
                    wall_t0 += time.perf_counter() - seq_t0
                print_executor_phase(style, update, seq_res)
            elif node_name == "joiner":
                print_joiner_phase(style, update)

    graph_wall_ms = (time.perf_counter() - wall_t0) * 1000.0
    visited_nodes.append("END")

    metrics = final_state["metrics"]
    dag = final_state.get("compiled_dag")
    exec_res = final_state.get("execution_result")

    total_tasks = len(exec_res.results) if exec_res is not None else 0
    success_rate = (
        len(exec_res.succeeded_task_ids) / total_tasks * 100.0 if total_tasks else 0.0
    )
    exec_par_ms = exec_res.total_duration_ms if exec_res is not None else 0.0
    exec_seq_ms = seq_res.total_duration_ms if seq_res is not None else 0.0
    speedup = (exec_seq_ms / exec_par_ms) if exec_par_ms > 0 and exec_seq_ms > 0 else 0.0

    summary = ScenarioSummary(
        name=scenario.name,
        path=" → ".join(visited_nodes),
        plan_valid=bool(metrics.plan_valid),
        nodes=len(dag.tasks) if dag is not None else 0,
        levels=len(dag.levels) if dag is not None else 0,
        planning_ms=metrics.planning_latency_ms or 0.0,
        compile_ms=metrics.compilation_latency_ms or 0.0,
        exec_par_ms=exec_par_ms,
        exec_seq_ms=exec_seq_ms,
        joining_ms=metrics.joining_latency_ms or 0.0,
        graph_wall_ms=graph_wall_ms,
        overhead_ms=max(0.0, graph_wall_ms - metrics.component_latency_ms),
        effective_parallelism=exec_res.effective_parallelism if exec_res else 0.0,
        speedup=speedup,
        success_rate=success_rate,
        joiner_status=final_state.get("status") or "failed",
        llm_calls=metrics.llm_calls,
    )
    print_scenario_footer(style, summary)
    return ScenarioArtifact(scenario=scenario, summary=summary, state=final_state)


# =============================================================================
# Entrypoint
# =============================================================================


async def main(
    *,
    selected: str,
    reasoning: bool,
    show_json: bool,
    use_color: bool,
    export: bool,
    max_compile_retries: int,
) -> None:
    style = Style(enabled=use_color)
    registry = build_registry()

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

    graph = build_compiler_graph(
        registry=registry,
        planner=planner,
        compiler=compiler,
        executor=parallel_executor,
        joiner=joiner,
        max_compile_retries=max_compile_retries,
    )

    print_setup(style, registry, graph, planner_reasoning=reasoning)

    chosen_scenarios = (
        list(SCENARIOS)
        if selected == "all"
        else [s for s in SCENARIOS if s.name == selected]
    )

    artifacts: list[ScenarioArtifact] = []
    for scenario in chosen_scenarios:
        artifact = await run_scenario(
            style=style,
            scenario=scenario,
            graph=graph,
            sequential_executor=sequential_executor,
            show_json=show_json,
        )
        artifacts.append(artifact)

    if len(artifacts) > 1:
        print_comparative_table(style, [a.summary for a in artifacts])

    if export:
        export_visual_artifacts(
            style=style,
            graph=graph,
            artifacts=artifacts,
            selected=selected,
            planner_reasoning=reasoning,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean visual step-by-step LangGraph LLM Compiler trace."
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
        help="Disable Qwen3 reasoning mode on Planner (enabled by default).",
    )
    parser.add_argument(
        "--show-json",
        action="store_true",
        help="Also print the raw ExecutionPlan JSON in Phase 1.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI terminal colors.",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip exporting PNG and Markdown Mermaid report to results/.",
    )
    parser.add_argument(
        "--max-compile-retries",
        type=int,
        default=2,
        help="Max compile-time self-correction retries (default: 2; use 0 for v1 linear).",
    )
    args = parser.parse_args()
    asyncio.run(
        main(
            selected=args.scenario,
            reasoning=not args.no_reasoning,
            show_json=args.show_json,
            use_color=(not args.no_color) and sys.stdout.isatty(),
            export=not args.no_export,
            max_compile_retries=args.max_compile_retries,
        )
    )
