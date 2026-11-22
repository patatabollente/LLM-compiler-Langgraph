"""Live benchmark comparing v2 (max_replans=0) vs v3 (max_replans=2) on qwen3:8b.

Tests four runtime scenarios that static compile-time validation (v2) alone cannot solve:
  1. r01_fx_primary_to_backup  : Primary FX API fails at runtime (503) -> fallback to fx_backup
  2. r02_alias_recovery        : Country alias 'UK' fails with hint -> retry with 'United Kingdom'
  3. r03_two_stage_discovery   : List of group members discovered in Round 1 -> fetch GDPs in Round 2
  4. r04_unrecoverable_control : Non-existent 'Atlantis' -> v3 should NOT waste a replan

Usage:
    uv run python experiments/benchmark_v3_replanning.py
    uv run python experiments/benchmark_v3_replanning.py --reasoning
    uv run python experiments/benchmark_v3_replanning.py --joiner-reasoning
"""

import argparse
import asyncio
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any, Literal

from pydantic import BaseModel, Field

from langgraph_adapters.compiler_graph import build_compiler_graph
from llm_compiler.compiler import PlanCompiler, ToolRegistry, ToolSpec
from llm_compiler.executor import Executor
from llm_compiler.joiner import Joiner
from llm_compiler.llm import create_ollama_chat_model
from llm_compiler.planner import Planner

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ---------------------------------------------------------------------------
# Tool schemas & handlers
# ---------------------------------------------------------------------------
class CountryStatsInput(BaseModel):
    country: str = Field(description="Full country name, e.g. 'Italy', 'Germany', 'Spain'.")


class CountryStatsOutput(BaseModel):
    country: str
    population: int
    area_km2: float
    gdp_billion_usd: float


class FxConvertInput(BaseModel):
    amount: float
    from_currency: Literal["USD", "EUR"]
    to_currency: Literal["USD", "EUR"]


class FxConvertOutput(BaseModel):
    converted_amount: float
    rate: float
    source: str


class GroupMembersInput(BaseModel):
    group_name: str = Field(description="Regional group code, e.g. 'Atlas-7'.")


class GroupMembersOutput(BaseModel):
    group: str
    members: list[str]


class CalculatorInput(BaseModel):
    operation: Literal["add", "subtract", "multiply", "divide"]
    a: float
    b: float


class CalculatorOutput(BaseModel):
    result: float


_COUNTRY_DATA: dict[str, dict[str, Any]] = {
    "italy": {
        "country": "Italy",
        "population": 58_990_000,
        "area_km2": 301_340.0,
        "gdp_billion_usd": 2300.0,
    },
    "germany": {
        "country": "Germany",
        "population": 84_480_000,
        "area_km2": 357_592.0,
        "gdp_billion_usd": 4525.0,
    },
    "spain": {
        "country": "Spain",
        "population": 48_350_000,
        "area_km2": 505_990.0,
        "gdp_billion_usd": 1580.0,
    },
    "portugal": {
        "country": "Portugal",
        "population": 10_470_000,
        "area_km2": 92_212.0,
        "gdp_billion_usd": 287.0,
    },
    "united kingdom": {
        "country": "United Kingdom",
        "population": 67_800_000,
        "area_km2": 243_610.0,
        "gdp_billion_usd": 3340.0,
    },
}

_FX_RATES: dict[tuple[str, str], float] = {
    ("USD", "EUR"): 0.92,
    ("EUR", "USD"): 1.09,
    ("USD", "USD"): 1.0,
    ("EUR", "EUR"): 1.0,
}

_GROUPS: dict[str, list[str]] = {
    "atlas-7": ["Spain", "Portugal"],
}


async def country_stats(country: str) -> dict[str, Any]:
    await asyncio.sleep(0.06)
    key = country.strip().lower()
    if key in ("uk", "u.k.", "britain"):
        raise ValueError(
            f"Country '{country}' not found. Did you mean 'United Kingdom'?"
        )
    if key not in _COUNTRY_DATA:
        raise ValueError(
            f"Country '{country}' does not exist in dataset. "
            "No alternative tool contains data for it."
        )
    return dict(_COUNTRY_DATA[key])


async def fx_primary(
    amount: float,
    from_currency: Literal["USD", "EUR"],
    to_currency: Literal["USD", "EUR"],
) -> dict[str, Any]:
    await asyncio.sleep(0.05)
    raise RuntimeError(
        "503 Service Unavailable: fx_primary is temporarily down. "
        "Please use an alternative FX conversion tool."
    )


async def fx_backup(
    amount: float,
    from_currency: Literal["USD", "EUR"],
    to_currency: Literal["USD", "EUR"],
) -> dict[str, Any]:
    await asyncio.sleep(0.08)
    rate = _FX_RATES[(from_currency, to_currency)]
    return {
        "converted_amount": round(amount * rate, 4),
        "rate": rate,
        "source": "fx_backup",
    }


async def regional_group_members(group_name: str) -> dict[str, Any]:
    await asyncio.sleep(0.05)
    key = group_name.strip().lower()
    if key not in _GROUPS:
        raise ValueError(f"Regional group '{group_name}' not found.")
    return {"group": group_name, "members": list(_GROUPS[key])}


async def calculator(
    operation: Literal["add", "subtract", "multiply", "divide"],
    a: float,
    b: float,
) -> dict[str, float]:
    await asyncio.sleep(0.03)
    if operation == "add":
        res = a + b
    elif operation == "subtract":
        res = a - b
    elif operation == "multiply":
        res = a * b
    else:
        if b == 0:
            raise ZeroDivisionError("Division by zero.")
        res = a / b
    return {"result": round(res, 4)}


def build_v3_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ToolSpec(
                name="country_stats",
                description=(
                    "Fetch population, area_km2, and gdp_billion_usd for a country."
                ),
                func=country_stats,
                input_schema=CountryStatsInput,
                output_schema=CountryStatsOutput,
            ),
            ToolSpec(
                name="fx_primary",
                description=(
                    "Primary real-time FX conversion service (preferred default for "
                    "converting currencies)."
                ),
                func=fx_primary,
                input_schema=FxConvertInput,
                output_schema=FxConvertOutput,
            ),
            ToolSpec(
                name="fx_backup",
                description=(
                    "Fallback FX conversion service. Use ONLY if fx_primary fails at "
                    "runtime."
                ),
                func=fx_backup,
                input_schema=FxConvertInput,
                output_schema=FxConvertOutput,
            ),
            ToolSpec(
                name="regional_group_members",
                description=(
                    "Return the list of member country names belonging to an internal "
                    "regional cohort code (e.g. 'Atlas-7')."
                ),
                func=regional_group_members,
                input_schema=GroupMembersInput,
                output_schema=GroupMembersOutput,
            ),
            ToolSpec(
                name="calculator",
                description="Perform binary arithmetic (add, subtract, multiply, divide).",
                func=calculator,
                input_schema=CalculatorInput,
                output_schema=CalculatorOutput,
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Benchmark cases & numerical grader
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReplanCase:
    case_id: str
    category: str
    request: str
    expected_status: str
    expected_numbers: tuple[float, ...]
    expect_replan_in_v3: bool
    abs_tol: float = 0.25
    rel_tol: float = 0.005


CASES: tuple[ReplanCase, ...] = (
    ReplanCase(
        case_id="r01_fx_primary_to_backup",
        category="fallback tool recovery",
        request=(
            "Fetch Italy's and Germany's GDP in billion USD using country_stats, "
            "convert both GDPs from USD to EUR using fx_primary, and compute the "
            "difference (Germany minus Italy) in billion EUR. Report both converted "
            "EUR GDPs and their difference."
        ),
        expected_status="complete",
        # Italy=2300*0.92=2116.0, Germany=4525*0.92=4163.0, diff=2047.0
        expected_numbers=(2116.0, 4163.0, 2047.0),
        expect_replan_in_v3=True,
    ),
    ReplanCase(
        case_id="r02_alias_recovery",
        category="runtime alias hint",
        request=(
            "Fetch statistics for Italy and UK using country_stats, and compute "
            "their combined total GDP in billion USD (Italy plus UK). Report both "
            "individual GDPs and their combined total."
        ),
        expected_status="complete",
        # Italy=2300.0, UK=3340.0, sum=5640.0
        expected_numbers=(2300.0, 3340.0, 5640.0),
        expect_replan_in_v3=True,
    ),
    ReplanCase(
        case_id="r03_two_stage_discovery",
        category="2-stage dynamic discovery",
        request=(
            "First look up which countries belong to the internal cohort 'Atlas-7' "
            "using regional_group_members. Then fetch each member country's GDP "
            "in billion USD using country_stats and compute their combined total GDP. "
            "Report each member's GDP and their combined total."
        ),
        expected_status="complete",
        # Spain=1580.0, Portugal=287.0, sum=1867.0
        expected_numbers=(1580.0, 287.0, 1867.0),
        expect_replan_in_v3=True,
    ),
    ReplanCase(
        case_id="r04_unrecoverable_control",
        category="unrecoverable (no waste)",
        request=(
            "Fetch statistics for Italy and Atlantis using country_stats. "
            "Compute Italy's population density (population divided by area_km2) "
            "and the population difference (Italy minus Atlantis)."
        ),
        expected_status="partial",
        # Italy density = 58_990_000 / 301_340 = 195.7589
        expected_numbers=(195.76,),
        expect_replan_in_v3=False,
    ),
)

_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def extract_numbers(text: str | None) -> list[float]:
    if not text:
        return []
    nums: list[float] = []
    for raw in _NUMBER_RE.findall(text):
        try:
            nums.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return nums


def grade_numbers(
    answer: str | None,
    expected: tuple[float, ...],
    *,
    abs_tol: float,
    rel_tol: float,
) -> tuple[bool, list[float]]:
    found = extract_numbers(answer)
    missing: list[float] = []
    for target in expected:
        tol = max(abs_tol, abs(target) * rel_tol)
        if not any(abs(val - target) <= tol for val in found):
            missing.append(target)
    return len(missing) == 0, missing


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
@dataclass
class RunRecord:
    case_id: str
    category: str
    version: str
    max_replans: int
    status: str | None
    status_ok: bool
    numeric_ok: bool
    missing_numbers: list[float]
    replan_attempts: int
    replan_policy_ok: bool
    compile_attempts: int
    compile_retries: int
    rounds_executed: int
    llm_calls: int
    graph_wall_ms: float
    replan_reasons: list[str]
    error_stage: str | None
    error_type: str | None
    replan_error_type: str | None
    final_answer: str | None


async def run_case(
    case: ReplanCase,
    *,
    version_label: str,
    max_compile_retries: int,
    max_replans: int,
    model_name: str,
    planner_reasoning: bool,
    joiner_reasoning: bool,
) -> RunRecord:
    registry = build_v3_registry()
    planner_llm = create_ollama_chat_model(
        model=model_name,
        temperature=0.0,
        reasoning=planner_reasoning,
    )
    joiner_llm = create_ollama_chat_model(
        model=model_name,
        temperature=0.0,
        reasoning=joiner_reasoning,
    )
    graph = build_compiler_graph(
        registry=registry,
        planner=Planner(llm=planner_llm),
        compiler=PlanCompiler(registry=registry),
        executor=Executor(registry=registry, max_concurrency=8),
        joiner=Joiner(llm=joiner_llm),
        max_compile_retries=max_compile_retries,
        max_replans=max_replans,
    )

    t0 = time.perf_counter()
    state = await graph.ainvoke({"request": case.request})
    wall_ms = (time.perf_counter() - t0) * 1000.0

    metrics = state["metrics"]
    err = state.get("error")
    replan_err = state.get("replan_error")
    history = state.get("execution_history", [])
    replan_reasons = [r.replan_reason for r in history if r.replan_reason]

    answer = state.get("final_answer")
    status = state.get("status")
    status_ok = status == case.expected_status
    numeric_ok, missing_nums = grade_numbers(
        answer,
        case.expected_numbers,
        abs_tol=case.abs_tol,
        rel_tol=case.rel_tol,
    )

    if max_replans == 0:
        replan_policy_ok = metrics.replan_attempts == 0
    elif case.expect_replan_in_v3:
        replan_policy_ok = metrics.replan_attempts >= 1
    else:
        replan_policy_ok = metrics.replan_attempts == 0

    rounds_executed = len(history) + (1 if state.get("execution_result") else 0)

    return RunRecord(
        case_id=case.case_id,
        category=case.category,
        version=version_label,
        max_replans=max_replans,
        status=status,
        status_ok=status_ok,
        numeric_ok=numeric_ok,
        missing_numbers=missing_nums,
        replan_attempts=metrics.replan_attempts,
        replan_policy_ok=replan_policy_ok,
        compile_attempts=metrics.compile_attempts,
        compile_retries=metrics.compile_retries,
        rounds_executed=rounds_executed,
        llm_calls=metrics.llm_calls,
        graph_wall_ms=wall_ms,
        replan_reasons=replan_reasons,
        error_stage=err.stage if err else None,
        error_type=err.error_type if err else None,
        replan_error_type=replan_err.error_type if replan_err else None,
        final_answer=answer,
    )


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live benchmark comparing v2 (0 replans) vs v3 (2 replans)."
    )
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--reasoning", action="store_true", help="Enable Planner reasoning.")
    parser.add_argument(
        "--joiner-reasoning",
        action="store_true",
        help="Enable Joiner reasoning for replan decisions.",
    )
    parser.add_argument("--max-compile-retries", type=int, default=2)
    parser.add_argument("--max-replans", type=int, default=2)
    args = parser.parse_args()

    configs = [
        ("v2 (replans=0)", 0),
        (f"v3 (replans={args.max_replans})", args.max_replans),
    ]

    print(
        f"Running v3 Runtime Replanning Benchmark on {args.model} "
        f"(planner_reasoning={args.reasoning}, joiner_reasoning={args.joiner_reasoning}, "
        f"max_compile_retries={args.max_compile_retries})\n"
    )

    all_records: list[RunRecord] = []
    for version_label, max_replans in configs:
        print(f"=== {version_label} ===")
        for case in CASES:
            rec = await run_case(
                case,
                version_label=version_label,
                max_compile_retries=args.max_compile_retries,
                max_replans=max_replans,
                model_name=args.model,
                planner_reasoning=args.reasoning,
                joiner_reasoning=args.joiner_reasoning,
            )
            all_records.append(rec)
            num_badge = "EXACT" if rec.numeric_ok else f"MISS({rec.missing_numbers})"
            print(
                f"  {rec.case_id:<26} | status={rec.status!s:<8} | "
                f"{num_badge:<22} | replans={rec.replan_attempts} | "
                f"c_ret={rec.compile_retries} | llm={rec.llm_calls} | "
                f"{rec.graph_wall_ms / 1000:.2f}s"
            )
            if rec.replan_reasons:
                for idx, reason in enumerate(rec.replan_reasons, start=1):
                    print(f"      └─ replan #{idx}: {reason}")
        print()

    summaries: list[dict[str, Any]] = []
    print("=" * 94)
    print(
        f"{'Version':<18} | {'Status OK':<10} | {'Num Exact':<10} | "
        f"{'Replans':<8} | {'NoWaste':<8} | {'LLM':<5} | {'Med Wall':<9}"
    )
    print("-" * 94)
    for version_label, max_replans in configs:
        recs = [r for r in all_records if r.version == version_label]
        status_ok = sum(r.status_ok for r in recs)
        num_ok = sum(r.numeric_ok for r in recs)
        total_replans = sum(r.replan_attempts for r in recs)
        control = next(r for r in recs if r.case_id == "r04_unrecoverable_control")
        no_waste = control.replan_attempts == 0
        total_llm = sum(r.llm_calls for r in recs)
        med_wall = median(r.graph_wall_ms for r in recs)
        summary = {
            "version": version_label,
            "max_replans": max_replans,
            "cases_total": len(recs),
            "status_ok_count": status_ok,
            "numeric_exact_count": num_ok,
            "numeric_exact_pct": round(100.0 * num_ok / len(recs), 1),
            "total_replans": total_replans,
            "unrecoverable_no_waste": no_waste,
            "total_llm_calls": total_llm,
            "median_wall_ms": round(med_wall, 2),
        }
        summaries.append(summary)
        print(
            f"{version_label:<18} | {status_ok}/{len(recs):<8} | "
            f"{num_ok}/{len(recs):<8} | {total_replans:<8} | "
            f"{no_waste!s:<8} | {total_llm:<5} | {med_wall / 1000:>6.2f} s"
        )
    print("=" * 94)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"v3_replanning_v2_vs_v3_{ts}.json"
    payload = {
        "benchmark": "v3_replanning_v2_vs_v3",
        "timestamp_utc": ts,
        "model": args.model,
        "planner_reasoning": args.reasoning,
        "joiner_reasoning": args.joiner_reasoning,
        "max_compile_retries": args.max_compile_retries,
        "max_replans": args.max_replans,
        "summaries": summaries,
        "records": [asdict(r) for r in all_records],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved JSON report -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
