import argparse
import asyncio
import json
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from langgraph_llm_compiler.compiler import PlanCompiler, ToolRegistry, ToolSpec
from langgraph_llm_compiler.executor import Executor
from langgraph_llm_compiler.graph import build_compiler_graph
from langgraph_llm_compiler.joiner import Joiner
from langgraph_llm_compiler.llm import create_ollama_chat_model
from langgraph_llm_compiler.planner import Planner

# =============================================================================
# Configuration & tool contracts (identical across runs for reproducibility)
# =============================================================================

MODEL_NAME = "qwen3:8b"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SIMULATED_EXCHANGE_DELAY_S = 0.12
SIMULATED_SYNC_TOOL_DELAY_S = 0.04


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
# Fixed v1 benchmark dataset (8 queries across difficulty & resilience levels)
# =============================================================================


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    difficulty: Literal["easy", "medium", "hard", "resilience"]
    expected_status: Literal["complete", "partial"]
    request: str


BENCHMARK_CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        case_id="q01_single_lookup",
        difficulty="easy",
        expected_status="complete",
        request=(
            "Fetch statistics for Italy and report its population, "
            "GDP in billion USD, and unemployment rate."
        ),
    ),
    BenchmarkCase(
        case_id="q02_binary_diff",
        difficulty="easy",
        expected_status="complete",
        request=(
            "Fetch statistics for Italy and France and compute the population "
            "difference (Italy minus France)."
        ),
    ),
    BenchmarkCase(
        case_id="q03_nested_fx_conversion",
        difficulty="medium",
        expected_status="complete",
        request=(
            "Fetch Germany's statistics and the USD to EUR exchange rate, "
            "then convert Germany's GDP in billion USD into billion EUR "
            "by multiplying the GDP by the exchange rate."
        ),
    ),
    BenchmarkCase(
        case_id="q04_static_constant_avg",
        difficulty="medium",
        expected_status="complete",
        request=(
            "Fetch statistics for Italy and Spain, compute the sum of their GDPs "
            "(gdp_billion_usd), and divide the sum by the static number 2 "
            "to compute their average GDP."
        ),
    ),
    BenchmarkCase(
        case_id="q05_density_comparison",
        difficulty="medium",
        expected_status="complete",
        request=(
            "Fetch statistics for Italy and Japan, compute each country's "
            "population density (population divided by area_km2), and compute "
            "the percentage change from Italy's density (base) to Japan's density (target)."
        ),
    ),
    BenchmarkCase(
        case_id="q06_wide_fanout_tree",
        difficulty="hard",
        expected_status="complete",
        request=(
            "Fetch statistics for Italy, France, Germany, and Spain. "
            "Compute the sum of GDP (gdp_billion_usd) of Italy and France, "
            "the sum of GDP of Germany and Spain, then add the two partial sums "
            "to get the combined GDP of all 4 countries, and finally divide the "
            "total by the static number 4 to compute the average GDP."
        ),
    ),
    BenchmarkCase(
        case_id="q07_multi_branch_diamond",
        difficulty="hard",
        expected_status="complete",
        request=(
            "Fetch statistics for Italy, France, and Germany, and get the USD to EUR "
            "exchange rate. Then: "
            "(1) compute the population density (population divided by area_km2) for Italy and Germany; "
            "(2) compute the percentage change of Germany's density (target) relative to Italy's density (base); "
            "(3) convert France's GDP in billion USD into billion EUR by multiplying it by the USD->EUR rate."
        ),
    ),
    BenchmarkCase(
        case_id="q08_partial_failure",
        difficulty="resilience",
        expected_status="partial",
        request=(
            "Fetch statistics for Italy and Atlantis. "
            "Compute Italy's population density (population divided by area_km2) "
            "and compute the difference between Italy's population and Atlantis's population "
            "(Italy minus Atlantis)."
        ),
    ),
)


# =============================================================================
# Recorded per-case and aggregate telemetry
# =============================================================================


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    difficulty: str
    expected_status: str
    planner_reasoning: bool
    status: str
    status_matches_expected: bool
    plan_valid: bool | None
    error_stage: str | None
    error_type: str | None
    compilation_issues: list[str]
    compilation_issue_codes: list[str]
    dag_nodes: int
    dag_levels: int
    dag_max_width: int
    planning_latency_ms: float | None
    compilation_latency_ms: float | None
    execution_latency_ms: float | None
    sequential_execution_ms: float | None
    joining_latency_ms: float | None
    component_latency_ms: float
    graph_wall_time_ms: float
    orchestration_overhead_ms: float
    effective_parallelism: float | None
    measured_speedup: float | None
    task_total: int
    task_succeeded: int
    task_failed: int
    task_skipped: int
    task_success_rate: float | None
    llm_calls: int
    compile_attempts: int
    final_answer: str | None
    missing_information: list[str]


def _extract_issue_code(formatted_issue: str) -> str:
    if formatted_issue.startswith("[") and "]" in formatted_issue:
        return formatted_issue[1 : formatted_issue.index("]")]
    return "unknown"


async def evaluate_mode(
    *,
    cases: list[BenchmarkCase],
    planner_reasoning: bool,
) -> tuple[list[CaseRecord], dict[str, object]]:
    mode_label = f"planner_reasoning={planner_reasoning}"
    line = "=" * 96
    print(f"\n{line}")
    print(
        f"BENCHMARK v1 BASELINE — {MODEL_NAME} "
        f"(planner reasoning={planner_reasoning}, joiner reasoning=False, cases={len(cases)})"
    )
    print(line)

    registry = build_registry()
    planner_llm = create_ollama_chat_model(
        model=MODEL_NAME,
        temperature=0.0,
        reasoning=planner_reasoning,
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
    )

    records: list[CaseRecord] = []

    for idx, case in enumerate(cases, start=1):
        t0 = time.perf_counter()
        state = await graph.ainvoke({"request": case.request})
        graph_wall_ms = (time.perf_counter() - t0) * 1000.0

        metrics = state["metrics"]
        err = state.get("error")
        dag = state.get("compiled_dag")
        exec_res = state.get("execution_result")
        joiner_res = state.get("joiner_result")

        seq_ms: float | None = None
        speedup: float | None = None
        eff_par: float | None = None
        if dag is not None and exec_res is not None:
            seq_res = await sequential_executor.execute(dag)
            seq_ms = seq_res.total_duration_ms
            eff_par = exec_res.effective_parallelism
            if exec_res.total_duration_ms > 0:
                speedup = seq_ms / exec_res.total_duration_ms

        issues = list(err.issues) if err is not None else []
        issue_codes = [_extract_issue_code(item) for item in issues]

        task_total = len(exec_res.results) if exec_res is not None else 0
        task_succeeded = (
            len(exec_res.succeeded_task_ids) if exec_res is not None else 0
        )
        task_failed = len(exec_res.failed_task_ids) if exec_res is not None else 0
        task_skipped = len(exec_res.skipped_task_ids) if exec_res is not None else 0
        task_success_rate = (
            (task_succeeded / task_total * 100.0) if task_total > 0 else None
        )

        status = state.get("status") or "failed"
        rec = CaseRecord(
            case_id=case.case_id,
            difficulty=case.difficulty,
            expected_status=case.expected_status,
            planner_reasoning=planner_reasoning,
            status=status,
            status_matches_expected=(status == case.expected_status),
            plan_valid=metrics.plan_valid,
            error_stage=err.stage if err is not None else None,
            error_type=err.error_type if err is not None else None,
            compilation_issues=issues,
            compilation_issue_codes=issue_codes,
            dag_nodes=len(dag.tasks) if dag is not None else 0,
            dag_levels=len(dag.levels) if dag is not None else 0,
            dag_max_width=(
                max((len(level) for level in dag.levels), default=0)
                if dag is not None
                else 0
            ),
            planning_latency_ms=metrics.planning_latency_ms,
            compilation_latency_ms=metrics.compilation_latency_ms,
            execution_latency_ms=metrics.execution_latency_ms,
            sequential_execution_ms=seq_ms,
            joining_latency_ms=metrics.joining_latency_ms,
            component_latency_ms=metrics.component_latency_ms,
            graph_wall_time_ms=graph_wall_ms,
            orchestration_overhead_ms=graph_wall_ms - metrics.component_latency_ms,
            effective_parallelism=eff_par,
            measured_speedup=speedup,
            task_total=task_total,
            task_succeeded=task_succeeded,
            task_failed=task_failed,
            task_skipped=task_skipped,
            task_success_rate=task_success_rate,
            llm_calls=metrics.llm_calls,
            compile_attempts=metrics.compile_attempts,
            final_answer=state.get("final_answer"),
            missing_information=(
                list(joiner_res.missing_information) if joiner_res is not None else []
            ),
        )
        records.append(rec)

        valid_str = "VALID" if rec.plan_valid else "FAIL "
        speedup_str = f"{rec.measured_speedup:.2f}x" if rec.measured_speedup else "  -  "
        plan_s = (rec.planning_latency_ms or 0.0) / 1000.0
        join_s = (rec.joining_latency_ms or 0.0) / 1000.0
        match_mark = "OK" if rec.status_matches_expected else "MISMATCH"
        extra = f"issues={rec.compilation_issue_codes}" if rec.compilation_issue_codes else ""
        print(
            f"  [{idx}/{len(cases)}] {rec.case_id:<25} ({rec.difficulty:<10}) | "
            f"{valid_str} | {rec.dag_nodes:>2}n/{rec.dag_levels}l | "
            f"plan={plan_s:5.2f}s | exec={rec.execution_latency_ms or 0:5.1f}ms | "
            f"spd={speedup_str} | join={join_s:4.2f}s | "
            f"{rec.status:<8} ({match_mark:<8}) {extra}"
        )

    valid_count = sum(1 for r in records if r.plan_valid is True)
    matched_count = sum(1 for r in records if r.status_matches_expected)
    issue_counter = Counter(
        code for r in records for code in r.compilation_issue_codes
    )

    plan_latencies = [
        r.planning_latency_ms for r in records if r.planning_latency_ms is not None
    ]
    comp_latencies = [
        r.compilation_latency_ms
        for r in records
        if r.compilation_latency_ms is not None
    ]
    exec_latencies = [
        r.execution_latency_ms for r in records if r.execution_latency_ms is not None
    ]
    join_latencies = [
        r.joining_latency_ms for r in records if r.joining_latency_ms is not None
    ]
    speedups = [r.measured_speedup for r in records if r.measured_speedup is not None]
    overheads = [r.orchestration_overhead_ms for r in records]
    walls = [r.graph_wall_time_ms for r in records]

    summary: dict[str, object] = {
        "mode": mode_label,
        "planner_reasoning": planner_reasoning,
        "cases_total": len(records),
        "plan_valid_count": valid_count,
        "plan_validity_pct": round(valid_count / len(records) * 100.0, 2)
        if records
        else 0.0,
        "status_match_count": matched_count,
        "status_match_pct": round(matched_count / len(records) * 100.0, 2)
        if records
        else 0.0,
        "compilation_issue_breakdown": dict(issue_counter),
        "total_llm_calls": sum(r.llm_calls for r in records),
        "median_planning_ms": round(statistics.median(plan_latencies), 2)
        if plan_latencies
        else None,
        "median_compilation_ms": round(statistics.median(comp_latencies), 3)
        if comp_latencies
        else None,
        "median_execution_ms": round(statistics.median(exec_latencies), 2)
        if exec_latencies
        else None,
        "median_joining_ms": round(statistics.median(join_latencies), 2)
        if join_latencies
        else None,
        "median_graph_wall_ms": round(statistics.median(walls), 2) if walls else None,
        "median_overhead_ms": round(statistics.median(overheads), 2)
        if overheads
        else None,
        "median_speedup": round(statistics.median(speedups), 2) if speedups else None,
        "max_speedup": round(max(speedups), 2) if speedups else None,
    }
    return records, summary


def print_mode_comparison(summaries: list[dict[str, object]]) -> None:
    line = "#" * 96
    print(f"\n{line}")
    print("# AGGREGATE COMPARISON — v1 BASELINE (LangGraph StateGraph, max_compile_retries=0)")
    print(line)
    header = (
        f"{'Mode':<24} | {'Plan Valid':<12} | {'Status OK':<12} | "
        f"{'Med Plan(s)':>11} | {'Med Join(s)':>11} | {'Med Spd':>7} | "
        f"{'Med Ovh(ms)':>11} | {'Issues'}"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        valid_str = f"{s['plan_valid_count']}/{s['cases_total']} ({s['plan_validity_pct']:.0f}%)"
        match_str = f"{s['status_match_count']}/{s['cases_total']} ({s['status_match_pct']:.0f}%)"
        med_plan = (
            f"{float(s['median_planning_ms']) / 1000:.2f}"
            if s["median_planning_ms"] is not None
            else "-"
        )
        med_join = (
            f"{float(s['median_joining_ms']) / 1000:.2f}"
            if s["median_joining_ms"] is not None
            else "-"
        )
        med_spd = (
            f"{float(s['median_speedup']):.2f}x"
            if s["median_speedup"] is not None
            else "-"
        )
        med_ovh = (
            f"{float(s['median_overhead_ms']):.2f}"
            if s["median_overhead_ms"] is not None
            else "-"
        )
        print(
            f"{s['mode']!s:<24} | {valid_str:<12} | {match_str:<12} | "
            f"{med_plan:>11} | {med_join:>11} | {med_spd:>7} | "
            f"{med_ovh:>11} | {s['compilation_issue_breakdown']}"
        )


async def main(
    *,
    reasoning_modes: Literal["both", "on", "off"],
    case_filter: str,
    output_path: Path | None,
    save: bool,
) -> None:
    if case_filter == "all":
        cases = list(BENCHMARK_CASES)
    else:
        wanted = {cid.strip() for cid in case_filter.split(",") if cid.strip()}
        cases = [c for c in BENCHMARK_CASES if c.case_id in wanted]
        if not cases:
            raise ValueError(f"No benchmark cases matched filter: {case_filter}")

    if reasoning_modes == "both":
        modes = [False, True]
    elif reasoning_modes == "on":
        modes = [True]
    else:
        modes = [False]

    all_records: list[CaseRecord] = []
    summaries: list[dict[str, object]] = []

    for reasoning in modes:
        records, summary = await evaluate_mode(
            cases=cases,
            planner_reasoning=reasoning,
        )
        all_records.extend(records)
        summaries.append(summary)

    print_mode_comparison(summaries)

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        target = output_path or (RESULTS_DIR / f"v1_baseline_{timestamp}.json")
        payload = {
            "benchmark": "v1_baseline",
            "timestamp_utc": timestamp,
            "model": MODEL_NAME,
            "max_compile_retries": 0,
            "joiner_reasoning": False,
            "summaries": summaries,
            "records": [asdict(r) for r in all_records],
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved structured benchmark report to: {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="v1 Baseline benchmark suite on LangGraph StateGraph (reasoning=False vs True)."
    )
    parser.add_argument(
        "--reasoning-modes",
        choices=["both", "on", "off"],
        default="both",
        help="Planner reasoning mode(s) to evaluate (default: both).",
    )
    parser.add_argument(
        "--cases",
        default="all",
        help="Comma-separated case_ids or 'all' (default: all).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Custom output JSON path (default: results/v1_baseline_<timestamp>.json).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save JSON report to results/.",
    )
    args = parser.parse_args()
    asyncio.run(
        main(
            reasoning_modes=args.reasoning_modes,
            case_filter=args.cases,
            output_path=args.output,
            save=not args.no_save,
        )
    )
