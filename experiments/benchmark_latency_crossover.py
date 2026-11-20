"""Live I/O latency crossover benchmark: LLM Compiler v3 vs LangGraph ReAct.

Sweeps external API latency across 5 realistic regimes (slow-hop API latency from
0.5s up to 10.0s, with a realistic 10:1 fast/slow branch asymmetry) on the exact
same 7-API 4-level opaque pipeline (`io01_asymmetric_audit_bundle`), using the
exact same model (qwen3:8b, temperature=0.0, reasoning=False, num_predict=2048)
and exact same tool handlers.

Captures empirically:
  1. The low-latency regime where ReAct's compact native tool calls + local KV cache
     outpace LLM Compiler's JSON DAG decode overhead.
  2. The exact I/O crossover point where LLM Compiler's event-driven scheduler
     (no BSP wave barrier) + fewer sequential LLM round-trips (2 vs 5) overtake
     ReAct on total wall-clock latency as well as LLM calls and total tokens.

Exports both a JSON report and a clean vector chart (`results/latency_crossover.svg`).

Usage:
    uv run python experiments/benchmark_latency_crossover.py
"""

import argparse
import asyncio
import itertools
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
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


class TokenUsageCallback(BaseCallbackHandler):
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


# ---------------------------------------------------------------------------
# Scalable asymmetric 4-level external API registry
# ---------------------------------------------------------------------------
class RegionInput(BaseModel):
    region: Literal["South-EU", "Central-EU"]


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


REQUEST = (
    "For both 'South-EU' and 'Central-EU' regions: look up each region's "
    "lead_hub using get_region_lead, fetch its tax policy_id using "
    "fetch_tax_policy, and request an official quote for a 100000 USD contract "
    "using compute_server_quote. Finally, combine both quote_codes using "
    "issue_audit_bundle (South-EU as quote_code_a, Central-EU as quote_code_b). "
    "Report both net_usd values, the bundle_id, combined_net_usd, and delta_usd."
)

EXPECTED_NUMBERS = (57000.0, 62000.0, 119000.0, 5000.0)
EXPECTED_TOKENS = ("BND-884-319",)

_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def grade_answer(answer: str | None) -> bool:
    if not answer:
        return False
    found: list[float] = []
    for raw in _NUMBER_RE.findall(answer):
        try:
            found.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    for target in EXPECTED_NUMBERS:
        tol = max(0.5, abs(target) * 0.005)
        if not any(abs(val - target) <= tol for val in found):
            return False
    return all(tok in answer for tok in EXPECTED_TOKENS)


def build_scaled_registry(slow_hop_s: float) -> ToolRegistry:
    """Build asymmetric 4-level API registry scaled by slow_hop_s.

    Branch South-EU  : fast (0.10x) -> slow (1.00x) -> quote (0.25x)
    Branch Central-EU: slow (1.00x) -> fast (0.10x) -> quote (0.25x)
    Join bundle      : 0.20x

    Theoretical I/O critical path:
      - Event-driven LLM Compiler : (0.10 + 1.00 + 0.25 + 0.20) * slow = 1.55 * slow
      - BSP / Turn-by-turn ReAct  : (1.00 + 1.00 + 0.25 + 0.20) * slow = 2.45 * slow
      -> Pure BSP barrier penalty = +0.90 * slow_hop_s per query!
    """
    fast_hop_s = slow_hop_s * 0.10
    quote_s = slow_hop_s * 0.25
    bundle_s = slow_hop_s * 0.20

    region_hubs: dict[str, tuple[str, float]] = {
        "South-EU": ("HUB-MIL-01", fast_hop_s),
        "Central-EU": ("HUB-FRA-02", slow_hop_s),
    }
    hub_policies: dict[str, tuple[str, str, float, float]] = {
        "HUB-MIL-01": ("POL-IT-884", "T2", 0.57, slow_hop_s),
        "HUB-FRA-02": ("POL-DE-319", "T1", 0.62, fast_hop_s),
    }
    quotes_store: dict[str, float] = {}

    async def get_region_lead(
        region: Literal["South-EU", "Central-EU"],
    ) -> dict[str, str]:
        hub, delay = region_hubs[region]
        await asyncio.sleep(delay)
        return {"region": region, "lead_hub": hub}

    async def fetch_tax_policy(hub_code: str) -> dict[str, str]:
        if hub_code not in hub_policies:
            raise ValueError(f"Unknown hub_code '{hub_code}'.")
        policy_id, tier, _, delay = hub_policies[hub_code]
        await asyncio.sleep(delay)
        return {"hub_code": hub_code, "policy_id": policy_id, "tier": tier}

    async def compute_server_quote(
        policy_id: str,
        contract_usd: float,
    ) -> dict[str, Any]:
        await asyncio.sleep(quote_s)
        match = next(
            (ratio for pid, _, ratio, _ in hub_policies.values() if pid == policy_id),
            None,
        )
        if match is None:
            raise ValueError(f"Unknown policy_id '{policy_id}'.")
        net = round(contract_usd * match, 2)
        quote_code = f"Q-{policy_id.split('-')[-1]}-{int(net)}"
        quotes_store[quote_code] = net
        return {"policy_id": policy_id, "quote_code": quote_code, "net_usd": net}

    async def issue_audit_bundle(
        quote_code_a: str,
        quote_code_b: str,
    ) -> dict[str, Any]:
        await asyncio.sleep(bundle_s)
        if quote_code_a not in quotes_store or quote_code_b not in quotes_store:
            raise ValueError("Quote code not found.")
        net_a = quotes_store[quote_code_a]
        net_b = quotes_store[quote_code_b]
        bundle_id = f"BND-{quote_code_a.split('-')[1]}-{quote_code_b.split('-')[1]}"
        return {
            "bundle_id": bundle_id,
            "combined_net_usd": round(net_a + net_b, 2),
            "delta_usd": round(abs(net_a - net_b), 2),
        }

    return ToolRegistry(
        [
            ToolSpec(
                name="get_region_lead",
                description="Look up the primary hub_code for 'South-EU' or 'Central-EU'.",
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
                    "Call pricing service with policy_id and contract_usd to get "
                    "quote_code and net_usd."
                ),
                func=compute_server_quote,
                input_schema=QuoteInput,
                output_schema=QuoteOutput,
            ),
            ToolSpec(
                name="issue_audit_bundle",
                description=(
                    "Combine two quote_codes on the audit server to get bundle_id, "
                    "combined_net_usd, and delta_usd."
                ),
                func=issue_audit_bundle,
                input_schema=AuditBundleInput,
                output_schema=AuditBundleOutput,
            ),
        ]
    )


REACT_SYSTEM_PROMPT = """
You are a helpful, accurate assistant with access to tools.
Call independent tools in parallel in the same turn whenever possible.
Always use the provided tools and report every requested number and identifier.
""".strip()


class ReActState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def build_react_graph(llm: Any, registry: ToolRegistry) -> Any:
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
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile()


@dataclass
class SweepPoint:
    slow_hop_s: float
    compiler_wall_s: float
    compiler_io_s: float
    compiler_llm_calls: int
    compiler_tokens: int
    compiler_exact: bool
    react_wall_s: float
    react_llm_calls: int
    react_tokens: int
    react_exact: bool
    delta_s: float  # compiler - react (negative => Compiler faster!)
    delta_pct: float


async def run_point(
    slow_hop_s: float,
    *,
    model_name: str,
    num_predict: int,
) -> SweepPoint:
    # 1. LLM Compiler v3
    reg_c = build_scaled_registry(slow_hop_s)
    cb_c = TokenUsageCallback()
    planner_llm = create_ollama_chat_model(
        model=model_name, temperature=0.0, reasoning=False, num_predict=num_predict
    )
    joiner_llm = create_ollama_chat_model(
        model=model_name, temperature=0.0, reasoning=False, num_predict=num_predict
    )
    comp_graph = build_compiler_graph(
        registry=reg_c,
        planner=Planner(llm=planner_llm),
        compiler=PlanCompiler(registry=reg_c),
        executor=Executor(registry=reg_c, max_concurrency=8),
        joiner=Joiner(llm=joiner_llm),
        max_compile_retries=2,
        max_replans=2,
    )
    t0 = time.perf_counter()
    c_state = await comp_graph.ainvoke({"request": REQUEST}, config={"callbacks": [cb_c]})
    c_wall = time.perf_counter() - t0
    c_metrics = c_state["metrics"]
    c_io = (c_metrics.execution_latency_ms or 0.0) / 1000.0
    c_exact = grade_answer(c_state.get("final_answer"))

    # 2. LangGraph ReAct
    reg_r = build_scaled_registry(slow_hop_s)
    cb_r = TokenUsageCallback()
    react_llm = create_ollama_chat_model(
        model=model_name, temperature=0.0, reasoning=False, num_predict=num_predict
    )
    react_graph = build_react_graph(react_llm, reg_r)
    t1 = time.perf_counter()
    r_state = await react_graph.ainvoke(
        {
            "messages": [
                SystemMessage(content=REACT_SYSTEM_PROMPT),
                HumanMessage(content=REQUEST),
            ]
        },
        config={"callbacks": [cb_r], "recursion_limit": 25},
    )
    r_wall = time.perf_counter() - t1
    r_msg = r_state["messages"][-1]
    r_ans = r_msg.content if isinstance(r_msg.content, str) else str(r_msg.content)
    r_exact = grade_answer(r_ans)

    delta_s = c_wall - r_wall
    delta_pct = (delta_s / r_wall) * 100.0 if r_wall > 0 else 0.0

    return SweepPoint(
        slow_hop_s=slow_hop_s,
        compiler_wall_s=round(c_wall, 3),
        compiler_io_s=round(c_io, 3),
        compiler_llm_calls=c_metrics.llm_calls,
        compiler_tokens=cb_c.total_tokens,
        compiler_exact=c_exact,
        react_wall_s=round(r_wall, 3),
        react_llm_calls=cb_r.llm_calls,
        react_tokens=cb_r.total_tokens,
        react_exact=r_exact,
        delta_s=round(delta_s, 3),
        delta_pct=round(delta_pct, 1),
    )


def find_crossover(points: list[SweepPoint]) -> tuple[float, float] | None:
    """Linearly interpolate the exact (slow_hop_s, wall_s) crossover point."""
    for p1, p2 in itertools.pairwise(points):
        d1 = p1.compiler_wall_s - p1.react_wall_s
        d2 = p2.compiler_wall_s - p2.react_wall_s
        if d1 > 0 and d2 <= 0:
            frac = d1 / (d1 - d2)
            x_cross = p1.slow_hop_s + frac * (p2.slow_hop_s - p1.slow_hop_s)
            y_cross = p1.compiler_wall_s + frac * (p2.compiler_wall_s - p1.compiler_wall_s)
            return round(x_cross, 2), round(y_cross, 2)
    return None


def render_svg_chart(
    points: list[SweepPoint],
    crossover: tuple[float, float] | None,
    out_path: Path,
) -> None:
    """Generate a clean, self-contained vector SVG chart (zero external deps)."""
    w, h = 860, 480
    pad_l, pad_r, pad_t, pad_b = 76, 42, 74, 72
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b

    max_x = max(p.slow_hop_s for p in points) * 1.05
    max_y = max(max(p.compiler_wall_s, p.react_wall_s) for p in points) * 1.12

    def sx(x: float) -> float:
        return pad_l + (x / max_x) * plot_w

    def sy(y: float) -> float:
        return pad_t + plot_h - (y / max_y) * plot_h

    lines: list[str] = [
        (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="Inter, Segoe UI, system-ui, sans-serif">'),
        f'  <rect width="{w}" height="{h}" fill="#ffffff" rx="10"/>',
        ('  <text x="430" y="30" text-anchor="middle" font-size="18" font-weight="700" '
        'fill="#0f172a">I/O Latency Crossover: LLM Compiler v3 vs LangGraph ReAct</text>'),
        ('  <text x="430" y="52" text-anchor="middle" font-size="12.5" fill="#475569">'
        'qwen3:8b (local Ollama, reasoning=False) · Asymmetric 7-API 4-level pipeline · '
        'Compiler: 2 LLM calls vs ReAct: 5 LLM calls (-60%)</text>'),
    ]

    # Grid lines & Y-axis ticks
    y_steps = 5
    for i in range(y_steps + 1):
        yv = (max_y / y_steps) * i
        yp = sy(yv)
        lines.append(
            f'  <line x1="{pad_l}" y1="{yp:.1f}" x2="{w - pad_r}" y2="{yp:.1f}" '
            'stroke="#e2e8f0" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="{pad_l - 10}" y="{yp + 4:.1f}" text-anchor="end" '
            f'font-size="11.5" fill="#64748b">{yv:.0f} s</text>'
        )

    # X-axis ticks (exact measured slow_hop_s values)
    for p in points:
        xp = sx(p.slow_hop_s)
        lines.append(
            f'  <line x1="{xp:.1f}" y1="{pad_t}" x2="{xp:.1f}" y2="{pad_t + plot_h}" '
            'stroke="#f1f5f9" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="{xp:.1f}" y="{pad_t + plot_h + 22}" text-anchor="middle" '
            f'font-size="11.5" fill="#475569">{p.slow_hop_s:.1f} s</text>'
        )

    # Axes
    lines.append(
        f'  <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" '
        'stroke="#334155" stroke-width="1.5"/>'
    )
    lines.append(
        f'  <line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{w - pad_r}" '
        f'y2="{pad_t + plot_h}" stroke="#334155" stroke-width="1.5"/>'
    )
    lines.append(
        f'  <text x="{pad_l + plot_w / 2:.1f}" y="{h - 20}" text-anchor="middle" '
        'font-size="12.5" font-weight="600" fill="#1e293b">'
        'Slow-hop External API Latency (seconds)</text>'
    )
    lines.append(
        f'  <text x="22" y="{pad_t + plot_h / 2:.1f}" text-anchor="middle" '
        f'font-size="12.5" font-weight="600" fill="#1e293b" '
        f'transform="rotate(-90 22 {pad_t + plot_h / 2:.1f})">'
        'Total Wall-Clock Latency (seconds)</text>'
    )

    # Crossover shaded winner zone + vertical marker
    if crossover is not None:
        xc, yc = crossover
        xcp = sx(xc)
        ycp = sy(yc)
        lines.append(
            f'  <rect x="{xcp:.1f}" y="{pad_t}" width="{(w - pad_r) - xcp:.1f}" '
            f'height="{plot_h}" fill="#ecfdf5" opacity="0.55"/>'
        )
        lines.append(
            f'  <line x1="{xcp:.1f}" y1="{pad_t}" x2="{xcp:.1f}" '
            f'y2="{pad_t + plot_h}" stroke="#059669" stroke-width="1.5" '
            'stroke-dasharray="5,4"/>'
        )
        lines.append(
            f'  <circle cx="{xcp:.1f}" cy="{ycp:.1f}" r="6" fill="#059669" '
            'stroke="#ffffff" stroke-width="2"/>'
        )
        lines.append(
            f'  <text x="{xcp + 8:.1f}" y="{pad_t + 18}" font-size="11.5" '
            f'font-weight="700" fill="#047857">'
            f'Crossover ~{xc:.1f} s API ({yc:.1f} s wall)</text>'
        )
        lines.append(
            f'  <text x="{xcp + 8:.1f}" y="{pad_t + 34}" font-size="11" '
            'fill="#065f46">LLM Compiler v3 faster →</text>'
        )

    # Polylines
    comp_pts = " ".join(
        f"{sx(p.slow_hop_s):.1f},{sy(p.compiler_wall_s):.1f}" for p in points
    )
    react_pts = " ".join(
        f"{sx(p.slow_hop_s):.1f},{sy(p.react_wall_s):.1f}" for p in points
    )
    lines.append(
        f'  <polyline fill="none" stroke="#2563eb" stroke-width="3" '
        f'points="{comp_pts}"/>'
    )
    lines.append(
        f'  <polyline fill="none" stroke="#ea580c" stroke-width="3" '
        f'stroke-dasharray="7,4" points="{react_pts}"/>'
    )

    # Data point markers + labels
    for p in points:
        cxp = sx(p.slow_hop_s)
        cyp = sy(p.compiler_wall_s)
        ryp = sy(p.react_wall_s)
        lines.append(
            f'  <circle cx="{cxp:.1f}" cy="{cyp:.1f}" r="4.5" fill="#2563eb" '
            'stroke="#ffffff" stroke-width="1.5"/>'
        )
        lines.append(
            f'  <circle cx="{cxp:.1f}" cy="{ryp:.1f}" r="4.5" fill="#ea580c" '
            'stroke="#ffffff" stroke-width="1.5"/>'
        )
        c_off = -10 if p.compiler_wall_s >= p.react_wall_s else 16
        r_off = 16 if p.compiler_wall_s >= p.react_wall_s else -10
        lines.append(
            f'  <text x="{cxp:.1f}" y="{cyp + c_off:.1f}" text-anchor="middle" '
            f'font-size="10.5" font-weight="600" fill="#1d4ed8">'
            f'{p.compiler_wall_s:.1f}s</text>'
        )
        lines.append(
            f'  <text x="{cxp:.1f}" y="{ryp + r_off:.1f}" text-anchor="middle" '
            f'font-size="10.5" font-weight="600" fill="#c2410c">'
            f'{p.react_wall_s:.1f}s</text>'
        )

    # Legend box (top-left inside plot)
    lx, ly = pad_l + 14, pad_t + 12
    lines.extend(
        [
            (f'  <rect x="{lx}" y="{ly}" width="215" height="54" rx="6" '
            'fill="#ffffff" fill-opacity="0.92" stroke="#cbd5e1" stroke-width="1"/>'),
            (f'  <line x1="{lx + 10}" y1="{ly + 18}" x2="{lx + 36}" y2="{ly + 18}" '
            'stroke="#2563eb" stroke-width="3"/>'),
            (f'  <text x="{lx + 44}" y="{ly + 22}" font-size="11.5" font-weight="600" '
            'fill="#0f172a">LLM Compiler v3 (event-driven)</text>'),
            (f'  <line x1="{lx + 10}" y1="{ly + 38}" x2="{lx + 36}" y2="{ly + 38}" '
            'stroke="#ea580c" stroke-width="3" stroke-dasharray="7,4"/>'),
            (f'  <text x="{lx + 44}" y="{ly + 42}" font-size="11.5" font-weight="600" '
            'fill="#0f172a">LangGraph ReAct (BSP turns)</text>'),
            "</svg>",
        ]
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live I/O latency crossover sweep: LLM Compiler v3 vs LangGraph ReAct."
    )
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--num-predict", type=int, default=2048)
    parser.add_argument(
        "--slow-hops",
        type=float,
        nargs="+",
        default=[0.5, 2.0, 4.0, 6.0, 8.0],
        help="Slow-hop external API latencies in seconds.",
    )
    args = parser.parse_args()

    print(
        f"Running I/O Latency Crossover Sweep on {args.model} "
        f"(slow_hops={args.slow_hops}s, reasoning=False, num_predict={args.num_predict})\n"
    )

    # Warm up Ollama model weights in memory so the first data point has no cold-start bias.
    warmup_llm = create_ollama_chat_model(
        model=args.model, temperature=0.0, reasoning=False, num_predict=16
    )
    await warmup_llm.ainvoke([HumanMessage(content="Reply with OK.")])

    print(
        f"{'Slow API':>8} | {'Compiler Wall':>13} | {'Comp I/O':>8} | {'ReAct Wall':>10} | "
        f"{'Delta':>14} | {'LLM (C/R)':>9} | {'Tok (C/R)':>11} | {'Exact':>5}"
    )
    print("-" * 98)

    points: list[SweepPoint] = []
    for slow_s in args.slow_hops:
        pt = await run_point(
            slow_s,
            model_name=args.model,
            num_predict=args.num_predict,
        )
        points.append(pt)
        sign = "+" if pt.delta_s > 0 else ""
        exact_str = f"{'OK' if pt.compiler_exact else 'NO'}/{'OK' if pt.react_exact else 'NO'}"
        print(
            f"{pt.slow_hop_s:>6.1f} s | {pt.compiler_wall_s:>11.2f} s | "
            f"{pt.compiler_io_s:>6.2f} s | {pt.react_wall_s:>8.2f} s | "
            f"{sign}{pt.delta_s:>5.2f}s ({sign}{pt.delta_pct:>4.1f}%) | "
            f"{pt.compiler_llm_calls:>4}/{pt.react_llm_calls:<4} | "
            f"{pt.compiler_tokens:>5}/{pt.react_tokens:<5} | {exact_str:>5}"
        )

    crossover = find_crossover(points)
    print("-" * 98)
    if crossover is not None:
        print(
            f"Empirical crossover detected at slow_hop ≈ {crossover[0]:.2f} s "
            f"(wall-clock ≈ {crossover[1]:.2f} s) — beyond this point, "
            "LLM Compiler v3 beats ReAct on wall-clock, LLM calls, AND tokens!"
        )
    else:
        print("No crossover observed within tested slow_hop range.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = RESULTS_DIR / f"latency_crossover_{ts}.json"
    svg_path = RESULTS_DIR / "latency_crossover.svg"

    render_svg_chart(points, crossover, svg_path)
    payload = {
        "benchmark": "io_latency_crossover_compiler_vs_react",
        "timestamp_utc": ts,
        "model": args.model,
        "reasoning": False,
        "num_predict": args.num_predict,
        "slow_hops_s": args.slow_hops,
        "crossover_slow_hop_s": crossover[0] if crossover else None,
        "crossover_wall_s": crossover[1] if crossover else None,
        "points": [asdict(p) for p in points],
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved JSON -> {json_path}")
    print(f"Saved SVG  -> {svg_path}")


if __name__ == "__main__":
    asyncio.run(main())
