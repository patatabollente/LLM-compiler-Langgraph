"""Apples-to-apples live benchmark: LLM Compiler (v3) vs LangGraph ReAct agent.

Compares BOTH architectures on the exact same model (qwen3:8b, temperature=0.0,
identical num_predict cap) and the exact same tool handlers across two distinct
regimes:

  Suite 1 — `standard` (6 cases, fast tools 40-180ms + arithmetic + runtime recovery):
    - Wide parallel DAGs (c01, c02, c05)
    - Runtime 503 fallback (c03: fx_primary -> fx_backup)
    - Runtime alias recovery (c04: UK -> United Kingdom)
    - Unrecoverable partial failure (c06: Atlantis)

  Suite 2 — `io_heavy` (3 cases, realistic 250-750ms external APIs, non-mental-math):
    - Multi-hop opaque service chains where every step requires a real external
      tool call (region lead -> tax policy -> server quote / asymmetric audit /
      signed multi-service compliance report) so neither architecture can skip
      tool calls or compute answers in-head.

Usage:
    uv run python experiments/benchmark_compiler_vs_react.py --suite both
    uv run python experiments/benchmark_compiler_vs_react.py --suite standard
    uv run python experiments/benchmark_compiler_vs_react.py --suite io_heavy
"""

import argparse
import asyncio
import json
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.outputs import LLMResult
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from langgraph_llm_compiler.compiler import PlanCompiler, ToolRegistry, ToolSpec
from langgraph_llm_compiler.executor import Executor
from langgraph_llm_compiler.graph import build_compiler_graph
from langgraph_llm_compiler.joiner import Joiner
from langgraph_llm_compiler.llm import create_ollama_chat_model
from langgraph_llm_compiler.planner import Planner

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ---------------------------------------------------------------------------
# Token & call accounting callback (shared identically by Compiler and ReAct)
# ---------------------------------------------------------------------------
class TokenUsageCallback(BaseCallbackHandler):
    """Collects LLM call count and input/output token counts across runs."""

    def __init__(self) -> None:
        self.llm_calls: int = 0
        self.input_tokens: int = 0
        self.output_tokens: int = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.llm_calls += 1
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg is None:
                    continue
                usage = getattr(msg, "usage_metadata", None)
                if isinstance(usage, dict):
                    self.input_tokens += int(usage.get("input_tokens", 0) or 0)
                    self.output_tokens += int(usage.get("output_tokens", 0) or 0)
                    return
                meta = getattr(msg, "response_metadata", None)
                if isinstance(meta, dict):
                    self.input_tokens += int(meta.get("prompt_eval_count", 0) or 0)
                    self.output_tokens += int(meta.get("eval_count", 0) or 0)
                    return

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ToolCallCounter:
    def __init__(self) -> None:
        self.count: int = 0


# ---------------------------------------------------------------------------
# Suite 1: Standard tools (40-180ms + arithmetic + runtime recovery)
# ---------------------------------------------------------------------------
class CountryStatsInput(BaseModel):
    country: str = Field(
        description="Full country name, e.g. 'Italy', 'France', 'Germany', 'Spain'."
    )


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


class PercentChangeInput(BaseModel):
    baseline: float = Field(description="Original/baseline value (denominator).")
    current: float = Field(description="New/comparison value.")


class PercentChangeOutput(BaseModel):
    percent_change: float


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
    "france": {
        "country": "France",
        "population": 68_400_000,
        "area_km2": 551_695.0,
        "gdp_billion_usd": 3130.0,
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


def build_standard_registry(counter: ToolCallCounter) -> ToolRegistry:
    async def country_stats(country: str) -> dict[str, Any]:
        counter.count += 1
        await asyncio.sleep(0.18)
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
        counter.count += 1
        await asyncio.sleep(0.12)
        raise RuntimeError(
            "503 Service Unavailable: fx_primary is temporarily down. "
            "Please use fx_backup instead."
        )

    async def fx_backup(
        amount: float,
        from_currency: Literal["USD", "EUR"],
        to_currency: Literal["USD", "EUR"],
    ) -> dict[str, Any]:
        counter.count += 1
        await asyncio.sleep(0.15)
        rate = _FX_RATES[(from_currency, to_currency)]
        return {
            "converted_amount": round(amount * rate, 4),
            "rate": rate,
            "source": "fx_backup",
        }

    async def percent_change(baseline: float, current: float) -> dict[str, float]:
        counter.count += 1
        await asyncio.sleep(0.04)
        if baseline == 0:
            raise ZeroDivisionError("Baseline cannot be zero.")
        pct = ((current - baseline) / baseline) * 100.0
        return {"percent_change": round(pct, 4)}

    async def calculator(
        operation: Literal["add", "subtract", "multiply", "divide"],
        a: float,
        b: float,
    ) -> dict[str, float]:
        counter.count += 1
        await asyncio.sleep(0.04)
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
                    "Fallback FX conversion service. Use ONLY if fx_primary fails."
                ),
                func=fx_backup,
                input_schema=FxConvertInput,
                output_schema=FxConvertOutput,
            ),
            ToolSpec(
                name="percent_change",
                description=(
                    "Compute percentage change ((current - baseline) / baseline) * 100."
                ),
                func=percent_change,
                input_schema=PercentChangeInput,
                output_schema=PercentChangeOutput,
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
# Suite 2: I/O-heavy opaque multi-hop API tools (cannot be skipped mentally!)
# ---------------------------------------------------------------------------
class RegionInput(BaseModel):
    region: Literal["South-EU", "Central-EU", "North-EU", "West-EU"]


class RegionLeadOutput(BaseModel):
    region: str
    lead_hub: str


class PolicyInput(BaseModel):
    hub_code: str = Field(description="Hub identifier returned by get_region_lead.")


class PolicyOutput(BaseModel):
    hub_code: str
    policy_id: str
    tier: str


class QuoteInput(BaseModel):
    policy_id: str = Field(description="Policy ID returned by fetch_tax_policy.")
    contract_usd: float


class QuoteOutput(BaseModel):
    policy_id: str
    quote_code: str
    net_usd: float


class AuditBundleInput(BaseModel):
    quote_code_a: str
    quote_code_b: str


class AuditBundleOutput(BaseModel):
    bundle_id: str
    combined_net_usd: float
    delta_usd: float


_REGION_HUBS: dict[str, tuple[str, float]] = {
    "South-EU": ("HUB-MIL-01", 0.25),   # fast first hop
    "Central-EU": ("HUB-FRA-02", 0.75), # slow first hop (asymmetric!)
    "North-EU": ("HUB-STO-03", 0.30),
    "West-EU": ("HUB-PAR-04", 0.70),
}

_HUB_POLICIES: dict[str, tuple[str, str, float, float]] = {
    # hub_code -> (policy_id, tier, effective_net_ratio, latency_s)
    "HUB-MIL-01": ("POL-IT-884", "T2", 0.57, 0.75),  # slow second hop!
    "HUB-FRA-02": ("POL-DE-319", "T1", 0.62, 0.25),  # fast second hop!
    "HUB-STO-03": ("POL-SE-512", "T1", 0.64, 0.65),
    "HUB-PAR-04": ("POL-FR-701", "T2", 0.59, 0.30),
}

_QUOTES_STORE: dict[str, float] = {}


def build_io_heavy_registry(counter: ToolCallCounter) -> ToolRegistry:
    """Realistic 250-750ms asymmetric multi-hop external APIs with opaque IDs."""

    async def get_region_lead(
        region: Literal["South-EU", "Central-EU", "North-EU", "West-EU"],
    ) -> dict[str, str]:
        counter.count += 1
        hub, delay_s = _REGION_HUBS[region]
        await asyncio.sleep(delay_s)
        return {"region": region, "lead_hub": hub}

    async def fetch_tax_policy(hub_code: str) -> dict[str, str]:
        counter.count += 1
        if hub_code not in _HUB_POLICIES:
            raise ValueError(f"Unknown hub_code '{hub_code}'.")
        policy_id, tier, _, delay_s = _HUB_POLICIES[hub_code]
        await asyncio.sleep(delay_s)
        return {"hub_code": hub_code, "policy_id": policy_id, "tier": tier}

    async def compute_server_quote(
        policy_id: str,
        contract_usd: float,
    ) -> dict[str, Any]:
        counter.count += 1
        await asyncio.sleep(0.45)
        match = next(
            (ratio for pid, _, ratio, _ in _HUB_POLICIES.values() if pid == policy_id),
            None,
        )
        if match is None:
            raise ValueError(f"Unknown policy_id '{policy_id}'.")
        net = round(contract_usd * match, 2)
        quote_code = f"Q-{policy_id.split('-')[-1]}-{int(net)}"
        _QUOTES_STORE[quote_code] = net
        return {"policy_id": policy_id, "quote_code": quote_code, "net_usd": net}

    async def issue_audit_bundle(
        quote_code_a: str,
        quote_code_b: str,
    ) -> dict[str, Any]:
        counter.count += 1
        await asyncio.sleep(0.40)
        if quote_code_a not in _QUOTES_STORE or quote_code_b not in _QUOTES_STORE:
            raise ValueError("One or both quote codes not found in server store.")
        net_a = _QUOTES_STORE[quote_code_a]
        net_b = _QUOTES_STORE[quote_code_b]
        combined = round(net_a + net_b, 2)
        delta = round(abs(net_a - net_b), 2)
        bundle_id = f"BND-{quote_code_a.split('-')[1]}-{quote_code_b.split('-')[1]}"
        return {
            "bundle_id": bundle_id,
            "combined_net_usd": combined,
            "delta_usd": delta,
        }

    return ToolRegistry(
        [
            ToolSpec(
                name="get_region_lead",
                description=(
                    "Look up the primary hub_code for a region "
                    "('South-EU', 'Central-EU', 'North-EU', 'West-EU')."
                ),
                func=get_region_lead,
                input_schema=RegionInput,
                output_schema=RegionLeadOutput,
            ),
            ToolSpec(
                name="fetch_tax_policy",
                description="Fetch the active policy_id and tier for a hub_code.",
                func=fetch_tax_policy,
                input_schema=PolicyInput,
                output_schema=PolicyOutput,
            ),
            ToolSpec(
                name="compute_server_quote",
                description=(
                    "Call the pricing service with policy_id and contract_usd to "
                    "receive an official quote_code and net_usd."
                ),
                func=compute_server_quote,
                input_schema=QuoteInput,
                output_schema=QuoteOutput,
            ),
            ToolSpec(
                name="issue_audit_bundle",
                description=(
                    "Combine two official quote_codes on the audit server to receive "
                    "a signed bundle_id, combined_net_usd, and delta_usd."
                ),
                func=issue_audit_bundle,
                input_schema=AuditBundleInput,
                output_schema=AuditBundleOutput,
            ),
        ]
    )


def registry_to_langchain_tools(registry: ToolRegistry) -> list[StructuredTool]:
    """Wrap the exact ToolRegistry specs into LangChain StructuredTools for ReAct."""
    lc_tools: list[StructuredTool] = []
    for name in registry:
        spec = registry[name]

        async def _runner(_spec: ToolSpec = spec, **kwargs: Any) -> str:
            out = await _spec.func(**kwargs)
            return json.dumps(out, ensure_ascii=False, sort_keys=True)

        lc_tools.append(
            StructuredTool.from_function(
                coroutine=_runner,
                name=spec.name,
                description=spec.description,
                args_schema=spec.input_schema,
            )
        )
    return lc_tools


# ---------------------------------------------------------------------------
# Fair LangGraph ReAct Agent (same model, same tools, concurrent ToolNode)
# ---------------------------------------------------------------------------
REACT_SYSTEM_PROMPT = """
You are a helpful, accurate assistant with access to tools.

Follow these rules strictly:
1. Always use the provided tools to retrieve data and perform ALL required calculations or server operations (do NOT compute results in your head when a tool is provided).
2. When multiple tool calls are independent, call them in parallel in the same turn.
3. Never invent or guess data from memory. If a tool fails with an unrecoverable error (such as a non-existent country), report what succeeded and transparently explain what could not be retrieved.
4. If a tool fails with a recoverable error (such as a service outage with a backup tool available, or a country-name spelling suggestion), retry using the fallback tool or corrected argument.
5. In your final answer, clearly report every number and identifier requested by the user.
""".strip()


class ReActState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def build_react_graph(llm: Any, registry: ToolRegistry) -> Any:
    lc_tools = registry_to_langchain_tools(registry)
    llm_with_tools = llm.bind_tools(lc_tools)
    tool_node = ToolNode(lc_tools, handle_tool_errors=True)

    async def agent_node(state: ReActState) -> dict[str, list[AnyMessage]]:
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: ReActState) -> Literal["tools", "__end__"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    builder: StateGraph[ReActState, Any, ReActState, ReActState] = StateGraph(ReActState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )
    builder.add_edge("tools", "agent")
    return builder.compile()


# ---------------------------------------------------------------------------
# Benchmark cases & automatic grader (numbers + optional opaque tokens/IDs)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CompareCase:
    suite: Literal["standard", "io_heavy"]
    case_id: str
    category: str
    request: str
    expected_numbers: tuple[float, ...] = ()
    expected_tokens: tuple[str, ...] = ()
    abs_tol: float = 0.35
    rel_tol: float = 0.005


STANDARD_CASES: tuple[CompareCase, ...] = (
    CompareCase(
        suite="standard",
        case_id="c01_multi_branch_diamond",
        category="8-tool diamond DAG",
        request=(
            "Fetch statistics for Italy, France, and Germany using country_stats. "
            "Compute each country's population density (population divided by area_km2). "
            "Then compute the percentage change in population density from France "
            "(baseline) to Germany (current), and compute Germany's density minus "
            "Italy's density. Report all 3 densities, the percentage change, and the "
            "density difference."
        ),
        expected_numbers=(195.76, 123.98, 236.25, 90.55, 40.49),
    ),
    CompareCase(
        suite="standard",
        case_id="c02_wide_fanout_tree",
        category="8-tool 4-level tree",
        request=(
            "Fetch statistics for Italy, France, Germany, and Spain using country_stats. "
            "Compute the combined GDP of Italy plus France, and the combined GDP of "
            "Germany plus Spain. Then compute both the ratio ((Germany + Spain) divided "
            "by (Italy + France)) and the difference ((Germany + Spain) minus "
            "(Italy + France)). Report both combined GDPs, the ratio, and the difference."
        ),
        expected_numbers=(5430.0, 6105.0, 1.1243, 675.0),
        abs_tol=0.02,
    ),
    CompareCase(
        suite="standard",
        case_id="c03_fx_runtime_fallback",
        category="runtime 503 fallback",
        request=(
            "Fetch Italy's and Germany's GDP in billion USD using country_stats, "
            "convert both GDPs from USD to EUR using fx_primary, and compute the "
            "difference (Germany minus Italy) in billion EUR. Report both converted "
            "EUR GDPs and their difference."
        ),
        expected_numbers=(2116.0, 4163.0, 2047.0),
    ),
    CompareCase(
        suite="standard",
        case_id="c04_alias_runtime_recovery",
        category="runtime alias hint",
        request=(
            "Fetch statistics for Italy and UK using country_stats. Compute their "
            "combined total GDP in billion USD (Italy plus UK) and the percentage "
            "change in GDP from Italy (baseline) to UK (current). Report both "
            "individual GDPs, their combined total, and the percentage change."
        ),
        expected_numbers=(2300.0, 3340.0, 5640.0, 45.22),
    ),
    CompareCase(
        suite="standard",
        case_id="c05_four_country_avg_density",
        category="12-tool deep fanout",
        request=(
            "Fetch statistics for Italy, France, Germany, and Spain using country_stats. "
            "Compute each country's population density (population divided by area_km2), "
            "sum all four densities, and divide the sum by 4 to get the average density. "
            "Report all four densities and the overall average density."
        ),
        expected_numbers=(195.76, 123.98, 236.25, 95.56, 162.89),
    ),
    CompareCase(
        suite="standard",
        case_id="c06_partial_unrecoverable",
        category="unrecoverable partial",
        request=(
            "Fetch statistics for Italy and Atlantis using country_stats. Compute "
            "Italy's population density (population divided by area_km2) and the "
            "population difference (Italy minus Atlantis). Report Italy's density "
            "and clearly explain what could not be calculated."
        ),
        expected_numbers=(195.76,),
    ),
)

IO_HEAVY_CASES: tuple[CompareCase, ...] = (
    CompareCase(
        suite="io_heavy",
        case_id="io01_asymmetric_audit_bundle",
        category="7-API 4-level asymmetric",
        request=(
            "For both 'South-EU' and 'Central-EU' regions: look up each region's "
            "lead_hub using get_region_lead, fetch its tax policy_id using "
            "fetch_tax_policy, and request an official quote for a 100000 USD contract "
            "using compute_server_quote. Finally, combine both quote_codes using "
            "issue_audit_bundle (South-EU as quote_code_a, Central-EU as quote_code_b). "
            "Report both net_usd values, the bundle_id, combined_net_usd, and delta_usd."
        ),
        expected_numbers=(57000.0, 62000.0, 119000.0, 5000.0),
        expected_tokens=("BND-884-319",),
    ),
    CompareCase(
        suite="io_heavy",
        case_id="io02_four_region_two_bundles",
        category="14-API 4-level wide DAG",
        request=(
            "Run the full 4-step pipeline for all four regions ('South-EU', "
            "'Central-EU', 'North-EU', 'West-EU'): get each region's lead_hub, fetch "
            "each hub's tax policy_id, compute a server quote for a 100000 USD contract "
            "for each policy, then issue TWO audit bundles using issue_audit_bundle: "
            "Bundle 1 combining South-EU (quote_code_a) and Central-EU (quote_code_b), "
            "and Bundle 2 combining North-EU (quote_code_a) and West-EU (quote_code_b). "
            "Report both bundle_ids and both combined_net_usd totals."
        ),
        # South+Central = 57000+62000 = 119000 (BND-884-319)
        # North+West    = 64000+59000 = 123000 (BND-512-701)
        expected_numbers=(119000.0, 123000.0),
        expected_tokens=("BND-884-319", "BND-512-701"),
    ),
    CompareCase(
        suite="io_heavy",
        case_id="io03_three_region_quotes",
        category="9-API 3-level parallel",
        request=(
            "For 'South-EU', 'North-EU', and 'West-EU', retrieve each region's "
            "lead_hub via get_region_lead, fetch its policy_id via fetch_tax_policy, "
            "and compute a server quote for a 200000 USD contract via "
            "compute_server_quote. Report all three quote_codes and all three net_usd "
            "values."
        ),
        # South: 114000 (Q-884-114000), North: 128000 (Q-512-128000), West: 118000 (Q-701-118000)
        expected_numbers=(114000.0, 128000.0, 118000.0),
        expected_tokens=("Q-884-114000", "Q-512-128000", "Q-701-118000"),
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


def grade_answer(
    answer: str | None,
    case: CompareCase,
) -> tuple[bool, list[str]]:
    missing: list[str] = []
    found = extract_numbers(answer)
    for target in case.expected_numbers:
        tol = max(case.abs_tol, abs(target) * case.rel_tol)
        if not any(abs(val - target) <= tol for val in found):
            missing.append(str(target))
    haystack = answer or ""
    for token in case.expected_tokens:
        if token not in haystack:
            missing.append(token)
    return len(missing) == 0, missing


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
@dataclass
class ArchRecord:
    suite: str
    case_id: str
    category: str
    architecture: str
    exact_ok: bool
    missing_items: list[str]
    wall_ms: float
    llm_calls: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    compile_retries: int
    replan_attempts: int
    error: str | None
    final_answer: str | None


async def run_llm_compiler(
    case: CompareCase,
    registry_factory: Callable[[ToolCallCounter], ToolRegistry],
    *,
    model_name: str,
    reasoning: bool,
    num_predict: int,
    max_compile_retries: int,
    max_replans: int,
) -> ArchRecord:
    counter = ToolCallCounter()
    registry = registry_factory(counter)
    cb = TokenUsageCallback()

    planner_llm = create_ollama_chat_model(
        model=model_name,
        temperature=0.0,
        reasoning=reasoning,
        num_predict=num_predict,
    )
    joiner_llm = create_ollama_chat_model(
        model=model_name,
        temperature=0.0,
        reasoning=False,
        num_predict=num_predict,
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
    state = await graph.ainvoke(
        {"request": case.request},
        config={"callbacks": [cb]},
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0

    metrics = state["metrics"]
    answer = state.get("final_answer")
    err = state.get("error")
    ok, missing = grade_answer(answer, case)

    return ArchRecord(
        suite=case.suite,
        case_id=case.case_id,
        category=case.category,
        architecture="LLM Compiler v3",
        exact_ok=ok,
        missing_items=missing,
        wall_ms=wall_ms,
        llm_calls=metrics.llm_calls,
        tool_calls=counter.count,
        input_tokens=cb.input_tokens,
        output_tokens=cb.output_tokens,
        total_tokens=cb.total_tokens,
        compile_retries=metrics.compile_retries,
        replan_attempts=metrics.replan_attempts,
        error=f"{err.stage}:{err.error_type}" if err else None,
        final_answer=answer,
    )


async def run_react_agent(
    case: CompareCase,
    registry_factory: Callable[[ToolCallCounter], ToolRegistry],
    *,
    model_name: str,
    reasoning: bool,
    num_predict: int,
    recursion_limit: int = 25,
) -> ArchRecord:
    counter = ToolCallCounter()
    registry = registry_factory(counter)
    cb = TokenUsageCallback()

    llm = create_ollama_chat_model(
        model=model_name,
        temperature=0.0,
        reasoning=reasoning,
        num_predict=num_predict,
    )
    react_graph = build_react_graph(llm, registry)

    t0 = time.perf_counter()
    err_str: str | None = None
    answer: str | None = None
    try:
        state = await react_graph.ainvoke(
            {
                "messages": [
                    SystemMessage(content=REACT_SYSTEM_PROMPT),
                    HumanMessage(content=case.request),
                ]
            },
            config={"callbacks": [cb], "recursion_limit": recursion_limit},
        )
        final_msg = state["messages"][-1]
        content = getattr(final_msg, "content", None)
        answer = content if isinstance(content, str) else str(content)
    except Exception as exc:  # noqa: BLE001
        err_str = f"{type(exc).__name__}: {exc}"

    wall_ms = (time.perf_counter() - t0) * 1000.0
    ok, missing = grade_answer(answer, case)

    return ArchRecord(
        suite=case.suite,
        case_id=case.case_id,
        category=case.category,
        architecture="LangGraph ReAct",
        exact_ok=ok,
        missing_items=missing,
        wall_ms=wall_ms,
        llm_calls=cb.llm_calls,
        tool_calls=counter.count,
        input_tokens=cb.input_tokens,
        output_tokens=cb.output_tokens,
        total_tokens=cb.total_tokens,
        compile_retries=0,
        replan_attempts=0,
        error=err_str,
        final_answer=answer,
    )


def summarize_suite(suite_name: str, records: list[ArchRecord]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    print("=" * 108)
    print(f"SUITE SUMMARY: {suite_name.upper()}")
    print(
        f"{'Architecture':<18} | {'Exact':<9} | {'Med Wall':<9} | {'Mean Wall':<9} | "
        f"{'Tot LLM':<7} | {'Med LLM':<7} | {'Tools':<5} | {'Tot Tok':<9} | {'Med Tok':<7}"
    )
    print("-" * 108)
    for arch in ("LLM Compiler v3", "LangGraph ReAct"):
        recs = [r for r in records if r.suite == suite_name and r.architecture == arch]
        if not recs:
            continue
        exact = sum(r.exact_ok for r in recs)
        med_wall = median(r.wall_ms for r in recs)
        mean_wall = mean(r.wall_ms for r in recs)
        tot_llm = sum(r.llm_calls for r in recs)
        med_llm = median(r.llm_calls for r in recs)
        tot_tools = sum(r.tool_calls for r in recs)
        tot_tok = sum(r.total_tokens for r in recs)
        med_tok = median(r.total_tokens for r in recs)
        summary = {
            "suite": suite_name,
            "architecture": arch,
            "cases_total": len(recs),
            "exact_count": exact,
            "exact_pct": round(100.0 * exact / len(recs), 1),
            "median_wall_ms": round(med_wall, 2),
            "mean_wall_ms": round(mean_wall, 2),
            "total_llm_calls": tot_llm,
            "median_llm_calls": med_llm,
            "total_tool_calls": tot_tools,
            "total_input_tokens": sum(r.input_tokens for r in recs),
            "total_output_tokens": sum(r.output_tokens for r in recs),
            "total_tokens": tot_tok,
            "median_tokens": med_tok,
        }
        summaries.append(summary)
        print(
            f"{arch:<18} | {exact}/{len(recs):<7} | {med_wall / 1000:>6.2f} s | "
            f"{mean_wall / 1000:>6.2f} s | {tot_llm:<7} | {med_llm:<7.1f} | "
            f"{tot_tools:<5} | {tot_tok:<9} | {med_tok:<7.0f}"
        )
    print("=" * 108)
    print()
    return summaries


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apples-to-apples live benchmark: LLM Compiler v3 vs LangGraph ReAct."
    )
    parser.add_argument(
        "--suite",
        choices=("standard", "io_heavy", "both"),
        default="both",
    )
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--reasoning", action="store_true")
    parser.add_argument("--num-predict", type=int, default=2048)
    parser.add_argument("--max-compile-retries", type=int, default=2)
    parser.add_argument("--max-replans", type=int, default=2)
    args = parser.parse_args()

    suites_to_run: list[
        tuple[str, tuple[CompareCase, ...], Callable[[ToolCallCounter], ToolRegistry]]
    ] = []
    if args.suite in ("standard", "both"):
        suites_to_run.append(("standard", STANDARD_CASES, build_standard_registry))
    if args.suite in ("io_heavy", "both"):
        suites_to_run.append(("io_heavy", IO_HEAVY_CASES, build_io_heavy_registry))

    print(
        f"Benchmark: LLM Compiler v3 vs LangGraph ReAct on {args.model} "
        f"(suite={args.suite}, reasoning={args.reasoning}, "
        f"num_predict={args.num_predict}, retries={args.max_compile_retries}, "
        f"replans={args.max_replans})\n"
    )

    all_records: list[ArchRecord] = []
    all_summaries: list[dict[str, Any]] = []

    for suite_name, cases, factory in suites_to_run:
        print(f"================ SUITE: {suite_name.upper()} ================\n")
        for case in cases:
            print(f"--- {case.case_id} ({case.category}) ---")
            comp_rec = await run_llm_compiler(
                case,
                factory,
                model_name=args.model,
                reasoning=args.reasoning,
                num_predict=args.num_predict,
                max_compile_retries=args.max_compile_retries,
                max_replans=args.max_replans,
            )
            react_rec = await run_react_agent(
                case,
                factory,
                model_name=args.model,
                reasoning=args.reasoning,
                num_predict=args.num_predict,
            )
            all_records.extend([comp_rec, react_rec])

            for rec in (comp_rec, react_rec):
                badge = "EXACT" if rec.exact_ok else f"MISS({rec.missing_items})"
                extra = (
                    f" (c_ret={rec.compile_retries}, rep={rec.replan_attempts})"
                    if rec.architecture == "LLM Compiler v3"
                    else ""
                )
                print(
                    f"  {rec.architecture:<16} | {badge:<26} | "
                    f"{rec.wall_ms / 1000:>6.2f}s | llm={rec.llm_calls:<2} | "
                    f"tools={rec.tool_calls:<2} | tok={rec.total_tokens:<6}{extra}"
                )
            print()

        all_summaries.extend(summarize_suite(suite_name, all_records))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"compiler_vs_react_{args.suite}_{ts}.json"
    payload = {
        "benchmark": "compiler_v3_vs_langgraph_react",
        "suite": args.suite,
        "timestamp_utc": ts,
        "model": args.model,
        "reasoning": args.reasoning,
        "num_predict": args.num_predict,
        "max_compile_retries": args.max_compile_retries,
        "max_replans": args.max_replans,
        "summaries": all_summaries,
        "records": [asdict(r) for r in all_records],
    }
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved JSON report -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
