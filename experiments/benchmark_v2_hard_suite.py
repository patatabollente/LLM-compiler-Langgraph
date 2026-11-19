import argparse
import asyncio
import json
import re
import statistics
import sys
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

MODEL_NAME = "qwen3:8b"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MAX_COL = 96


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

    def red(self, text: str) -> str:
        return self._wrap("1;31", text)


# =============================================================================
# 10-tool realistic catalog (6 active domain tools + 4 distractors)
# Deeply nested schemas & non-trivial parameter names to stress-test the Planner
# =============================================================================


class CountryInput(BaseModel):
    country: str = Field(min_length=1)


class Demographics(BaseModel):
    population: int
    area_km2: float


class GdpMetrics(BaseModel):
    nominal_billion_usd: float


class LaborMetrics(BaseModel):
    unemployment_pct: float


class MacroEconomy(BaseModel):
    gdp: GdpMetrics
    labor: LaborMetrics
    inflation_pct: float


class CountryMacroOutput(BaseModel):
    country: str
    demographics: Demographics
    economy: MacroEconomy


class GenerationProfile(BaseModel):
    total_twh: float
    renewable_pct: float


class EmissionsProfile(BaseModel):
    co2_mt: float


class EnergyProfileOutput(BaseModel):
    country: str
    generation: GenerationProfile
    emissions: EmissionsProfile


class FxQuoteInput(BaseModel):
    base_currency: Literal["USD", "EUR", "JPY", "GBP"]
    quote_currency: Literal["USD", "EUR", "JPY", "GBP"]


class FxQuoteOutput(BaseModel):
    pair: str
    rate: float


class MathOpInput(BaseModel):
    op: Literal["add", "sub", "mul", "div"]
    left: float
    right: float


class MathOpOutput(BaseModel):
    value: float


class PctChangeInput(BaseModel):
    baseline: float
    current: float


class PctChangeOutput(BaseModel):
    absolute_delta: float
    pct_delta: float


class TaxEstimateInput(BaseModel):
    gross_amount: float
    country_code: Literal["IT", "DE", "FR", "ES", "JP"]


class TaxEstimateOutput(BaseModel):
    country_code: str
    effective_rate_pct: float
    net_amount: float


# Distractor tool schemas
class WeatherInput(BaseModel):
    city: str
    days: int = 3


class WeatherOutput(BaseModel):
    city: str
    temp_c: float


class StockInput(BaseModel):
    ticker: str


class StockOutput(BaseModel):
    ticker: str
    price_usd: float


class TimezoneInput(BaseModel):
    city: str


class TimezoneOutput(BaseModel):
    city: str
    utc_offset_hours: int


class UnitConvertInput(BaseModel):
    value: float
    from_unit: Literal["km", "mi", "kg", "lb"]
    to_unit: Literal["km", "mi", "kg", "lb"]


class UnitConvertOutput(BaseModel):
    converted_value: float


MACRO_DATA: dict[str, tuple[float, dict[str, object]]] = {
    "italy": (
        0.08,
        {
            "country": "Italy",
            "demographics": {"population": 58_990_000, "area_km2": 301_340.0},
            "economy": {
                "gdp": {"nominal_billion_usd": 2_300.0},
                "labor": {"unemployment_pct": 6.8},
                "inflation_pct": 1.8,
            },
        },
    ),
    "germany": (
        0.12,
        {
            "country": "Germany",
            "demographics": {"population": 83_500_000, "area_km2": 357_588.0},
            "economy": {
                "gdp": {"nominal_billion_usd": 4_525.0},
                "labor": {"unemployment_pct": 3.4},
                "inflation_pct": 2.4,
            },
        },
    ),
    "spain": (
        0.15,
        {
            "country": "Spain",
            "demographics": {"population": 48_800_000, "area_km2": 505_990.0},
            "economy": {
                "gdp": {"nominal_billion_usd": 1_650.0},
                "labor": {"unemployment_pct": 11.3},
                "inflation_pct": 2.9,
            },
        },
    ),
    "france": (
        0.26,
        {
            "country": "France",
            "demographics": {"population": 68_400_000, "area_km2": 551_695.0},
            "economy": {
                "gdp": {"nominal_billion_usd": 3_130.0},
                "labor": {"unemployment_pct": 7.4},
                "inflation_pct": 2.1,
            },
        },
    ),
    "japan": (
        0.18,
        {
            "country": "Japan",
            "demographics": {"population": 124_000_000, "area_km2": 377_975.0},
            "economy": {
                "gdp": {"nominal_billion_usd": 4_110.0},
                "labor": {"unemployment_pct": 2.5},
                "inflation_pct": 2.7,
            },
        },
    ),
}

ENERGY_DATA: dict[str, tuple[float, dict[str, object]]] = {
    "italy": (
        0.09,
        {
            "country": "Italy",
            "generation": {"total_twh": 275.0, "renewable_pct": 41.0},
            "emissions": {"co2_mt": 315.0},
        },
    ),
    "germany": (
        0.14,
        {
            "country": "Germany",
            "generation": {"total_twh": 510.0, "renewable_pct": 52.0},
            "emissions": {"co2_mt": 674.0},
        },
    ),
    "spain": (
        0.11,
        {
            "country": "Spain",
            "generation": {"total_twh": 280.0, "renewable_pct": 50.0},
            "emissions": {"co2_mt": 225.0},
        },
    ),
    "france": (
        0.22,
        {
            "country": "France",
            "generation": {"total_twh": 495.0, "renewable_pct": 26.0},
            "emissions": {"co2_mt": 290.0},
        },
    ),
    "japan": (
        0.16,
        {
            "country": "Japan",
            "generation": {"total_twh": 980.0, "renewable_pct": 22.0},
            "emissions": {"co2_mt": 1_040.0},
        },
    ),
}

FX_RATES: dict[tuple[str, str], float] = {
    ("USD", "EUR"): 0.92,
    ("EUR", "USD"): 1.087,
    ("USD", "JPY"): 152.0,
    ("USD", "GBP"): 0.79,
    ("EUR", "GBP"): 0.86,
}

TAX_RATES: dict[str, float] = {
    "IT": 43.0,
    "DE": 38.0,
    "FR": 41.0,
    "ES": 39.0,
    "JP": 35.0,
}


async def country_macro(country: str) -> dict[str, object]:
    key = country.strip().lower()
    if key not in MACRO_DATA:
        await asyncio.sleep(0.04)
        raise ValueError(f"Country '{country}' not found in macro dataset.")
    delay_s, payload = MACRO_DATA[key]
    await asyncio.sleep(delay_s)
    return payload


async def energy_profile(country: str) -> dict[str, object]:
    key = country.strip().lower()
    if key not in ENERGY_DATA:
        await asyncio.sleep(0.04)
        raise ValueError(f"Country '{country}' not found in energy dataset.")
    delay_s, payload = ENERGY_DATA[key]
    await asyncio.sleep(delay_s)
    return payload


async def fx_quote(
    base_currency: Literal["USD", "EUR", "JPY", "GBP"],
    quote_currency: Literal["USD", "EUR", "JPY", "GBP"],
) -> dict[str, str | float]:
    await asyncio.sleep(0.08)
    if base_currency == quote_currency:
        return {"pair": f"{base_currency}/{quote_currency}", "rate": 1.0}
    if (base_currency, quote_currency) not in FX_RATES:
        raise ValueError(f"Unsupported FX pair {base_currency}/{quote_currency}")
    return {
        "pair": f"{base_currency}/{quote_currency}",
        "rate": FX_RATES[(base_currency, quote_currency)],
    }


def math_op(
    op: Literal["add", "sub", "mul", "div"],
    left: float,
    right: float,
) -> dict[str, float]:
    time.sleep(0.03)
    if op == "add":
        return {"value": left + right}
    if op == "sub":
        return {"value": left - right}
    if op == "mul":
        return {"value": left * right}
    if op == "div":
        if right == 0:
            raise ZeroDivisionError("Division by zero.")
        return {"value": left / right}
    raise ValueError(f"Unsupported op: {op}")


def pct_change(baseline: float, current: float) -> dict[str, float]:
    time.sleep(0.03)
    if baseline == 0:
        raise ZeroDivisionError("Baseline cannot be zero.")
    delta = current - baseline
    return {"absolute_delta": delta, "pct_delta": (delta / baseline) * 100.0}


def tax_estimate(
    gross_amount: float,
    country_code: Literal["IT", "DE", "FR", "ES", "JP"],
) -> dict[str, str | float]:
    time.sleep(0.03)
    rate = TAX_RATES[country_code]
    return {
        "country_code": country_code,
        "effective_rate_pct": rate,
        "net_amount": gross_amount * (1.0 - rate / 100.0),
    }


def build_hard_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ToolSpec(
                name="country_macro",
                description=(
                    "Get macroeconomic and demographic indicators for a country "
                    "(nested demographics, economy.gdp.nominal_billion_usd, "
                    "economy.labor.unemployment_pct, economy.inflation_pct)"
                ),
                func=country_macro,
                input_schema=CountryInput,
                output_schema=CountryMacroOutput,
            ),
            ToolSpec(
                name="energy_profile",
                description=(
                    "Get electricity generation (total_twh, renewable_pct) "
                    "and CO2 emissions (co2_mt) for a country"
                ),
                func=energy_profile,
                input_schema=CountryInput,
                output_schema=EnergyProfileOutput,
            ),
            ToolSpec(
                name="fx_quote",
                description="Get foreign exchange conversion rate between two currencies",
                func=fx_quote,
                input_schema=FxQuoteInput,
                output_schema=FxQuoteOutput,
            ),
            ToolSpec(
                name="math_op",
                description="Execute binary arithmetic (left <op> right)",
                func=math_op,
                input_schema=MathOpInput,
                output_schema=MathOpOutput,
            ),
            ToolSpec(
                name="pct_change",
                description="Compute absolute delta and percentage change from baseline to current",
                func=pct_change,
                input_schema=PctChangeInput,
                output_schema=PctChangeOutput,
            ),
            ToolSpec(
                name="tax_estimate",
                description="Estimate effective tax rate and net amount for a gross income in a country",
                func=tax_estimate,
                input_schema=TaxEstimateInput,
                output_schema=TaxEstimateOutput,
            ),
            # Distractor tools
            ToolSpec(
                name="weather_forecast",
                description="Get short-term temperature forecast for a city",
                func=lambda city, days=3: {"city": city, "temp_c": 20.0},
                input_schema=WeatherInput,
                output_schema=WeatherOutput,
            ),
            ToolSpec(
                name="stock_quote",
                description="Get latest equity share price in USD for a stock ticker",
                func=lambda ticker: {"ticker": ticker, "price_usd": 100.0},
                input_schema=StockInput,
                output_schema=StockOutput,
            ),
            ToolSpec(
                name="timezone_offset",
                description="Get UTC offset in hours for a city",
                func=lambda city: {"city": city, "utc_offset_hours": 1},
                input_schema=TimezoneInput,
                output_schema=TimezoneOutput,
            ),
            ToolSpec(
                name="unit_convert",
                description="Convert physical measurements between km/mi or kg/lb",
                func=lambda value, from_unit, to_unit: {"converted_value": value},
                input_schema=UnitConvertInput,
                output_schema=UnitConvertOutput,
            ),
        ]
    )


# =============================================================================
# Hard benchmark cases with automatic ground-truth numerical grading
# =============================================================================


@dataclass(frozen=True)
class HardCase:
    case_id: str
    category: str
    expected_status: Literal["complete", "partial"]
    expected_numbers: tuple[float, ...]
    request: str


HARD_CASES: tuple[HardCase, ...] = (
    HardCase(
        case_id="h01_deep_nested_fx",
        category="3-level nested path + FX",
        expected_status="complete",
        expected_numbers=(4163.0, 3781.2, 381.8),
        request=(
            "Convert Germany's nominal GDP from billion USD to billion EUR, "
            "and also convert Japan's nominal GDP from billion USD to billion EUR. "
            "Then calculate how many billion EUR larger Germany's GDP is than Japan's."
        ),
    ),
    HardCase(
        case_id="h02_co2_per_million",
        category="cross-tool join + pct",
        expected_status="complete",
        expected_numbers=(5.34, 8.07, 51.16),
        request=(
            "Compare Italy and Germany on CO2 emissions per million inhabitants "
            "(CO2 emissions in megatons divided by population in millions, where "
            "population in millions is population divided by 1000000). Report both "
            "countries' values and the percentage change from Italy (baseline) to Germany (current)."
        ),
    ),
    HardCase(
        case_id="h03_renewable_twh_groups",
        category="4-country multi-stage",
        expected_status="complete",
        expected_numbers=(252.75, 393.9),
        request=(
            "For Italy, Spain, Germany, and France, compute each country's renewable "
            "electricity output in TWh (total_twh multiplied by renewable_pct, then divided by 100). "
            "Then compute the combined renewable TWh of Italy and Spain together, "
            "and the combined renewable TWh of Germany and France together."
        ),
    ),
    HardCase(
        case_id="h04_tax_fx_chain",
        category="FX + country_code enum",
        expected_status="complete",
        expected_numbers=(184000.0, 114080.0, 104880.0, 9200.0),
        request=(
            "Take a gross consulting contract of 200000 USD, convert it to EUR using "
            "the USD to EUR exchange rate, estimate the net amount after taxes in Germany "
            "and in Italy, and compute the net income difference (Germany net minus Italy net)."
        ),
    ),
    HardCase(
        case_id="h05_misery_index",
        category="multi-branch + pct",
        expected_status="complete",
        expected_numbers=(8.6, 14.2, 5.8, 144.83),
        request=(
            "Compute the Misery Index (inflation percentage plus unemployment percentage) "
            "for Italy, Spain, and Germany, and calculate the percentage change in the "
            "Misery Index from Germany (baseline) to Spain (current)."
        ),
    ),
    HardCase(
        case_id="h06_partial_multi_tool",
        category="resilience (partial)",
        expected_status="partial",
        expected_numbers=(128.7,),
        request=(
            "Fetch macroeconomic and energy data for France and for Wakanda. "
            "Compute France's renewable electricity output in TWh (total_twh multiplied by "
            "renewable_pct divided by 100), and compute the difference in nominal GDP "
            "between France and Wakanda (France minus Wakanda)."
        ),
    ),
)


_NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def extract_numbers(text: str) -> list[float]:
    numbers: list[float] = []
    for match in _NUMBER_RE.findall(text):
        cleaned = match.replace(",", "")
        try:
            numbers.append(float(cleaned))
        except ValueError:
            continue
    return numbers


def grade_answer_numbers(
    answer: str | None,
    expected: tuple[float, ...],
) -> tuple[bool, list[float]]:
    """Return (all_matched, missing_expected_numbers) using tolerant numeric matching."""
    if not answer:
        return False, list(expected)

    found = extract_numbers(answer)
    missing: list[float] = []
    for target in expected:
        tol = max(0.15, abs(target) * 0.01)
        if not any(abs(val - target) <= tol or abs(abs(val) - abs(target)) <= tol for val in found):
            missing.append(target)
    return len(missing) == 0, missing


# =============================================================================
# Evaluation runner
# =============================================================================


@dataclass(frozen=True)
class HardCaseResult:
    case_id: str
    category: str
    version: str
    max_compile_retries: int
    plan_valid: bool | None
    status: str
    status_ok: bool
    numeric_ok: bool
    missing_numbers: list[float]
    compile_attempts: int
    compile_retries: int
    compile_error_history: list[list[str]]
    error_stage: str | None
    error_type: str | None
    dag_nodes: int
    dag_levels: int
    planning_ms: float | None
    compilation_ms: float | None
    execution_ms: float | None
    sequential_ms: float | None
    joining_ms: float | None
    graph_wall_ms: float
    speedup: float | None
    llm_calls: int
    final_answer: str | None


async def run_suite_for_version(
    *,
    style: Style,
    cases: list[HardCase],
    max_compile_retries: int,
    planner_reasoning: bool,
) -> tuple[list[HardCaseResult], dict[str, object]]:
    version = f"v2 (retries={max_compile_retries})" if max_compile_retries > 0 else "v1 (retries=0)"
    print(f"\n{style.bold('═' * MAX_COL)}")
    print(
        style.bold(
            f"  HARD SUITE — {version} | {MODEL_NAME} "
            f"(planner reasoning={planner_reasoning}, joiner reasoning=False, 10 tools)"
        )
    )
    print(style.bold("═" * MAX_COL))

    registry = build_hard_registry()
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

    graph = build_compiler_graph(
        registry=registry,
        planner=Planner(llm=planner_llm),
        compiler=PlanCompiler(registry=registry),
        executor=Executor(registry=registry, max_concurrency=8),
        joiner=Joiner(llm=joiner_llm),
        max_compile_retries=max_compile_retries,
    )
    seq_executor = Executor(registry=registry, max_concurrency=1)

    results: list[HardCaseResult] = []
    for idx, case in enumerate(cases, start=1):
        t0 = time.perf_counter()
        state = await graph.ainvoke({"request": case.request})
        wall_ms = (time.perf_counter() - t0) * 1000.0

        metrics = state["metrics"]
        err = state.get("error")
        dag = state.get("compiled_dag")
        exec_res = state.get("execution_result")
        answer = state.get("final_answer")
        status = state.get("status") or "failed"
        history = list(state.get("compile_error_history", []))

        seq_ms: float | None = None
        speedup: float | None = None
        if dag is not None and exec_res is not None and exec_res.total_duration_ms > 0:
            seq_res = await seq_executor.execute(dag)
            seq_ms = seq_res.total_duration_ms
            speedup = seq_ms / exec_res.total_duration_ms

        status_ok = status == case.expected_status
        numeric_ok, missing_nums = grade_answer_numbers(answer, case.expected_numbers)

        rec = HardCaseResult(
            case_id=case.case_id,
            category=case.category,
            version=version,
            max_compile_retries=max_compile_retries,
            plan_valid=metrics.plan_valid,
            status=status,
            status_ok=status_ok,
            numeric_ok=numeric_ok,
            missing_numbers=missing_nums,
            compile_attempts=metrics.compile_attempts,
            compile_retries=metrics.compile_retries,
            compile_error_history=history,
            error_stage=err.stage if err is not None else None,
            error_type=err.error_type if err is not None else None,
            dag_nodes=len(dag.tasks) if dag is not None else 0,
            dag_levels=len(dag.levels) if dag is not None else 0,
            planning_ms=metrics.planning_latency_ms,
            compilation_ms=metrics.compilation_latency_ms,
            execution_ms=metrics.execution_latency_ms,
            sequential_ms=seq_ms,
            joining_ms=metrics.joining_latency_ms,
            graph_wall_ms=wall_ms,
            speedup=speedup,
            llm_calls=metrics.llm_calls,
            final_answer=answer,
        )
        results.append(rec)

        valid_badge = style.green("VALID") if rec.plan_valid else style.red("FAIL ")
        num_badge = style.green("NUM:OK  ") if rec.numeric_ok else style.red("NUM:MISS")
        retry_tag = (
            style.yellow(f"retry={rec.compile_retries}")
            if rec.compile_retries > 0
            else style.dim("retry=0")
        )
        spd_str = f"{rec.speedup:.2f}x" if rec.speedup else "  -  "
        print(
            f"  [{idx}/{len(cases)}] {rec.case_id:<24} │ {valid_badge} │ {retry_tag} │ "
            f"{rec.dag_nodes:>2}n/{rec.dag_levels}l │ wall={rec.graph_wall_ms / 1000:5.2f}s │ "
            f"spd={spd_str} │ {rec.status:<8} │ {num_badge} │ LLM={rec.llm_calls}"
        )
        if rec.compile_error_history:
            for att_idx, errs in enumerate(rec.compile_error_history, start=1):
                short_codes = [
                    e[1 : e.index("]")] if e.startswith("[") and "]" in e else "err"
                    for e in errs
                ]
                print(style.dim(f"        └─ compile attempt #{att_idx} issues: {short_codes}"))
        if not rec.numeric_ok and rec.plan_valid:
            print(style.yellow(f"        └─ missing expected numbers: {rec.missing_numbers}"))

    valid_n = sum(1 for r in results if r.plan_valid is True)
    status_n = sum(1 for r in results if r.status_ok)
    num_n = sum(1 for r in results if r.status_ok and r.numeric_ok)
    recovered_n = sum(
        1 for r in results if r.plan_valid is True and r.compile_retries > 0
    )
    total_retries = sum(r.compile_retries for r in results)
    issue_codes = Counter(
        e[1 : e.index("]")] if e.startswith("[") and "]" in e else "err"
        for r in results
        for att in r.compile_error_history
        for e in att
    )
    walls = [r.graph_wall_ms for r in results]
    speedups = [r.speedup for r in results if r.speedup is not None]

    summary: dict[str, object] = {
        "version": version,
        "max_compile_retries": max_compile_retries,
        "cases_total": len(results),
        "plan_valid_count": valid_n,
        "plan_validity_pct": round(valid_n / len(results) * 100.0, 1) if results else 0.0,
        "status_ok_count": status_n,
        "numeric_exact_count": num_n,
        "numeric_exact_pct": round(num_n / len(results) * 100.0, 1) if results else 0.0,
        "recovered_via_retry_count": recovered_n,
        "total_compile_retries": total_retries,
        "total_llm_calls": sum(r.llm_calls for r in results),
        "median_wall_ms": round(statistics.median(walls), 2) if walls else None,
        "median_speedup": round(statistics.median(speedups), 2) if speedups else None,
        "compile_issue_breakdown": dict(issue_codes),
    }
    return results, summary


def print_comparison(style: Style, summaries: list[dict[str, object]]) -> None:
    print(f"\n{style.bold('═' * MAX_COL)}")
    print(style.bold("  HARD SUITE COMPARISON — v1 (0 retries) vs v2 (compile-time self-correction)"))
    print(style.bold("═" * MAX_COL))
    header = (
        f"  {'Version':<17} │ {'Plan Valid':<12} │ {'Num Exact':<12} │ "
        f"{'Recovered':>9} │ {'Retries':>7} │ {'LLM':>4} │ {'Med Wall':>8} │ {'Med Spd':>7}"
    )
    print(style.bold(header))
    print(style.dim("  " + "─" * (MAX_COL - 2)))
    for s in summaries:
        valid_s = f"{s['plan_valid_count']}/{s['cases_total']} ({s['plan_validity_pct']:.0f}%)"
        exact_s = f"{s['numeric_exact_count']}/{s['cases_total']} ({s['numeric_exact_pct']:.0f}%)"
        med_wall = f"{float(s['median_wall_ms']) / 1000:.2f}s" if s["median_wall_ms"] else "-"
        med_spd = f"{float(s['median_speedup']):.2f}x" if s["median_speedup"] else "-"
        print(
            f"  {s['version']!s:<17} │ {valid_s:<12} │ {exact_s:<12} │ "
            f"{s['recovered_via_retry_count']!s:>9} │ {s['total_compile_retries']!s:>7} │ "
            f"{s['total_llm_calls']!s:>4} │ {med_wall:>8} │ {med_spd:>7}"
        )
        if s["compile_issue_breakdown"]:
            print(style.dim(f"    issues encountered: {s['compile_issue_breakdown']}"))


async def main(
    *,
    planner_reasoning: bool,
    v2_retries: int,
    case_filter: str,
    use_color: bool,
    save: bool,
) -> None:
    style = Style(enabled=use_color)
    if case_filter == "all":
        cases = list(HARD_CASES)
    else:
        wanted = {c.strip() for c in case_filter.split(",") if c.strip()}
        cases = [c for c in HARD_CASES if c.case_id in wanted]
        if not cases:
            raise ValueError(f"No hard cases matched filter: {case_filter}")

    v1_results, v1_summary = await run_suite_for_version(
        style=style,
        cases=cases,
        max_compile_retries=0,
        planner_reasoning=planner_reasoning,
    )
    v2_results, v2_summary = await run_suite_for_version(
        style=style,
        cases=cases,
        max_compile_retries=v2_retries,
        planner_reasoning=planner_reasoning,
    )

    print_comparison(style, [v1_summary, v2_summary])

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out_path = RESULTS_DIR / f"hard_suite_v1_vs_v2_{timestamp}.json"
        payload = {
            "benchmark": "hard_suite_v1_vs_v2",
            "timestamp_utc": timestamp,
            "model": MODEL_NAME,
            "planner_reasoning": planner_reasoning,
            "joiner_reasoning": False,
            "catalog_size": 10,
            "summaries": [v1_summary, v2_summary],
            "records": [asdict(r) for r in (*v1_results, *v2_results)],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n  {style.green('Saved JSON')} : {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hard suite benchmark (10 tools, nested paths, auto numeric grading): v1 vs v2."
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Enable Qwen3 reasoning mode on Planner (default: False for fast stress test).",
    )
    parser.add_argument(
        "--v2-retries",
        type=int,
        default=2,
        help="max_compile_retries for v2 run (default: 2).",
    )
    parser.add_argument(
        "--cases",
        default="all",
        help="Comma-separated case IDs or 'all' (default: all).",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving JSON report to results/.",
    )
    args = parser.parse_args()
    asyncio.run(
        main(
            planner_reasoning=args.reasoning,
            v2_retries=args.v2_retries,
            case_filter=args.cases,
            use_color=(not args.no_color) and sys.stdout.isatty(),
            save=not args.no_save,
        )
    )
