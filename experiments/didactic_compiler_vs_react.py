"""Didactic side-by-side lab: LLM Compiler v3 vs LangGraph ReAct.

Runs the exact same requests, tools, latencies, and LLM (`qwen3:8b` via Ollama)
through both architectures while narrating every single phase/turn live and
printing a real millisecond-accurate ASCII Gantt timeline + takeaway analysis.

Scenarios:
  1. `asymmetric_dag` — Multi-hop asymmetric 5-tool pipeline (fast->slow vs slow->fast -> merge).
     Highlights: 1-shot DAG planning, static pre-run validation, event-driven parallelism
     without turn barriers, and 2 LLM calls vs 4 LLM calls in ReAct.
  2. `runtime_recovery` — Primary shipping API fails with 503 on one code while the other succeeds.
     Highlights: best-effort isolation + automatic `skipped` propagation + v3 replanning
     with `replan_context` vs ReAct's natural turn-by-turn adaptation.

Usage:
    uv run python experiments/didactic_compiler_vs_react.py
    uv run python experiments/didactic_compiler_vs_react.py --scenario 1
    uv run python experiments/didactic_compiler_vs_react.py --scenario 2
    uv run python experiments/didactic_compiler_vs_react.py --no-color
"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import LLMResult
from pydantic import BaseModel, ConfigDict, Field

from langgraph_adapters.compiler_graph import build_compiler_graph
from langgraph_adapters.react_graph import build_react_graph
from llm_compiler import (
    Executor,
    Joiner,
    PlanCompiler,
    Planner,
    ToolRegistry,
    ToolSpec,
    create_ollama_chat_model,
)
from llm_compiler.planner.schemas import ExecutionPlan, InputBinding

RESULTS_MD_PATH = Path(__file__).resolve().parents[1] / "results" / "didactic_compiler_vs_react.md"


# ---------------------------------------------------------------------------
# Terminal styling + dual logger (stdout + Markdown report)
# ---------------------------------------------------------------------------


class Style:
    def __init__(self, *, color: bool = True) -> None:
        self.color = color

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def cyan(self, text: str) -> str:
        return self._wrap("36", text)

    def blue(self, text: str) -> str:
        return self._wrap("34", text)

    def magenta(self, text: str) -> str:
        return self._wrap("35", text)

    def green(self, text: str) -> str:
        return self._wrap("32", text)

    def yellow(self, text: str) -> str:
        return self._wrap("33", text)

    def red(self, text: str) -> str:
        return self._wrap("31", text)


class DualWriter:
    """Prints styled output to stdout and records clean plain-text lines for Markdown."""

    def __init__(self, style: Style) -> None:
        self.style = style
        self._md_lines: list[str] = []

    def out(self, styled: str, md: str | None = None) -> None:
        print(styled)
        self._md_lines.append(styled if md is None else md)

    def blank(self) -> None:
        print()
        self._md_lines.append("")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self._md_lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Token & Timeline instrumentation (shared identically by Compiler and ReAct)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimelineSpan:
    kind: str  # "llm" | "tool" | "compiler"
    label: str
    start_ms: float
    end_ms: float
    status: str = "ok"  # "ok" | "error" | "skipped"


class RunTracker:
    """Records millisecond-accurate spans for LLM calls and tool invocations."""

    def __init__(self) -> None:
        self.t0: float = time.perf_counter()
        self.spans: list[TimelineSpan] = []

    def reset(self) -> None:
        self.t0 = time.perf_counter()
        self.spans.clear()

    def now_ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000.0

    def record(
        self,
        *,
        kind: str,
        label: str,
        start_ms: float,
        end_ms: float,
        status: str = "ok",
    ) -> None:
        self.spans.append(
            TimelineSpan(
                kind=kind,
                label=label,
                start_ms=start_ms,
                end_ms=end_ms,
                status=status,
            )
        )


class TokenAndSpanCallback(BaseCallbackHandler):
    """Captures input/output tokens and wall spans for every LLM call."""

    def __init__(self, tracker: RunTracker, prefix: str = "LLM") -> None:
        self.tracker = tracker
        self.prefix = prefix
        self.calls: int = 0
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self._active_starts: dict[str, tuple[int, float]] = {}

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        self.calls += 1
        self._active_starts[str(run_id)] = (self.calls, self.tracker.now_ms())

    def on_llm_end(self, response: LLMResult, *, run_id: Any, **kwargs: Any) -> None:
        end_ms = self.tracker.now_ms()
        call_idx, start_ms = self._active_starts.pop(str(run_id), (self.calls, end_ms))
        self.tracker.record(
            kind="llm",
            label=f"{self.prefix} #{call_idx}",
            start_ms=start_ms,
            end_ms=end_ms,
            status="ok",
        )
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg is None:
                    continue
                usage = getattr(msg, "usage_metadata", None)
                if isinstance(usage, dict):
                    self.input_tokens += int(usage.get("input_tokens", 0) or 0)
                    self.output_tokens += int(usage.get("output_tokens", 0) or 0)
                    continue
                meta = getattr(msg, "response_metadata", None)
                if isinstance(meta, dict):
                    self.input_tokens += int(meta.get("prompt_eval_count", 0) or 0)
                    self.output_tokens += int(meta.get("eval_count", 0) or 0)


# ---------------------------------------------------------------------------
# Scenario 1: Asymmetric 5-tool DAG (fast->slow vs slow->fast -> merge)
# ---------------------------------------------------------------------------


class OrderInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str = Field(description="Order identifier, e.g. 'ORD-42'.")


class OrderOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str
    order_value_eur: float
    customer_tier_code: str


class RiskAuditInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_value_eur: float
    customer_tier_code: str


class RiskAuditOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    risk_score: float


class StockInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sku: str = Field(description="Product SKU code, e.g. 'SKU-X9'.")


class StockOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sku: str
    missing_units: int
    unit_cost_eur: float


class RestockCostInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    missing_units: int
    unit_cost_eur: float


class RestockCostOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    restock_cost_eur: float


class ExecutiveScoreInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    risk_score: float
    restock_cost_eur: float


class ExecutiveScoreOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    priority_index: float
    recommendation: str


def build_scenario_1_registry(tracker: RunTracker) -> ToolRegistry:
    """Asymmetric 5-tool pipeline with opaque server formulas (no mental math shortcuts)."""

    async def fetch_order_summary(order_id: str) -> dict[str, Any]:
        t_start = tracker.now_ms()
        await asyncio.sleep(0.30)  # Fast root (Branch A)
        if order_id != "ORD-42":
            tracker.record(
                kind="tool",
                label=f"fetch_order({order_id})",
                start_ms=t_start,
                end_ms=tracker.now_ms(),
                status="error",
            )
            raise ValueError(f"Unknown order_id '{order_id}'.")
        tracker.record(
            kind="tool",
            label=f"fetch_order({order_id})",
            start_ms=t_start,
            end_ms=tracker.now_ms(),
        )
        return {
            "order_id": "ORD-42",
            "order_value_eur": 12400.0,
            "customer_tier_code": "TIER-B",
        }

    async def run_risk_audit(order_value_eur: float, customer_tier_code: str) -> dict[str, Any]:
        t_start = tracker.now_ms()
        await asyncio.sleep(1.80)  # Slow dependent (Branch A)
        tier_mult = {"TIER-A": 0.4, "TIER-B": 1.35, "TIER-C": 2.1}.get(customer_tier_code, 1.0)
        score = round((order_value_eur / 500.0) * tier_mult + 7.25, 2)
        tracker.record(
            kind="tool",
            label="run_risk_audit",
            start_ms=t_start,
            end_ms=tracker.now_ms(),
        )
        return {"risk_score": score}

    async def fetch_warehouse_stock(sku: str) -> dict[str, Any]:
        t_start = tracker.now_ms()
        await asyncio.sleep(1.80)  # Slow root (Branch B)
        if sku != "SKU-X9":
            tracker.record(
                kind="tool",
                label=f"fetch_stock({sku})",
                start_ms=t_start,
                end_ms=tracker.now_ms(),
                status="error",
            )
            raise ValueError(f"Unknown SKU '{sku}'.")
        tracker.record(
            kind="tool",
            label=f"fetch_stock({sku})",
            start_ms=t_start,
            end_ms=tracker.now_ms(),
        )
        return {"sku": "SKU-X9", "missing_units": 35, "unit_cost_eur": 48.0}

    async def compute_restock_cost(missing_units: int, unit_cost_eur: float) -> dict[str, Any]:
        t_start = tracker.now_ms()
        await asyncio.sleep(0.30)  # Fast dependent (Branch B)
        # Opaque supplier surcharge (+12% express handling + 95 EUR fixed pallet fee)
        cost = round(missing_units * unit_cost_eur * 1.12 + 95.0, 2)
        tracker.record(
            kind="tool",
            label="compute_restock_cost",
            start_ms=t_start,
            end_ms=tracker.now_ms(),
        )
        return {"restock_cost_eur": cost}

    async def build_executive_score(risk_score: float, restock_cost_eur: float) -> dict[str, Any]:
        t_start = tracker.now_ms()
        await asyncio.sleep(0.30)  # Final merge node
        priority = round(risk_score * 1.5 + (restock_cost_eur / 100.0) * 0.8, 2)
        rec = "ESCALATE_IMMEDIATELY" if priority >= 70.0 else "STANDARD_REVIEW"
        tracker.record(
            kind="tool",
            label="build_executive_score",
            start_ms=t_start,
            end_ms=tracker.now_ms(),
        )
        return {"priority_index": priority, "recommendation": rec}

    return ToolRegistry(
        [
            ToolSpec(
                name="fetch_order_summary",
                description=(
                    "Fetch order value (EUR) and customer tier code for an order ID "
                    "(e.g. 'ORD-42')."
                ),
                func=fetch_order_summary,
                input_schema=OrderInput,
                output_schema=OrderOutput,
            ),
            ToolSpec(
                name="run_risk_audit",
                description=(
                    "Run the proprietary risk audit model given order_value_eur and "
                    "customer_tier_code, returning risk_score."
                ),
                func=run_risk_audit,
                input_schema=RiskAuditInput,
                output_schema=RiskAuditOutput,
            ),
            ToolSpec(
                name="fetch_warehouse_stock",
                description=(
                    "Fetch missing_units and unit_cost_eur from the warehouse ERP for a SKU "
                    "(e.g. 'SKU-X9')."
                ),
                func=fetch_warehouse_stock,
                input_schema=StockInput,
                output_schema=StockOutput,
            ),
            ToolSpec(
                name="compute_restock_cost",
                description=(
                    "Compute supplier restock_cost_eur including express handling and pallet fees "
                    "from missing_units and unit_cost_eur."
                ),
                func=compute_restock_cost,
                input_schema=RestockCostInput,
                output_schema=RestockCostOutput,
            ),
            ToolSpec(
                name="build_executive_score",
                description=(
                    "Combine risk_score and restock_cost_eur using the executive scoring formula "
                    "to return priority_index and recommendation."
                ),
                func=build_executive_score,
                input_schema=ExecutiveScoreInput,
                output_schema=ExecutiveScoreOutput,
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Scenario 2: Runtime failure (503) + Backup tool recovery
# ---------------------------------------------------------------------------


class QuoteInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    zone_code: str = Field(description="Shipping zone code, e.g. 'EU-NORTH' or 'LEGACY-99'.")


class QuoteOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    zone_code: str
    shipping_eur: float
    carrier: str


class InvoiceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cost_a_eur: float
    cost_b_eur: float


class InvoiceOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subtotal_eur: float
    insurance_fee_eur: float
    grand_total_eur: float


def build_scenario_2_registry(tracker: RunTracker) -> ToolRegistry:
    """Shipping quotes where primary fails with 503 on LEGACY-99 and backup recovers."""

    async def shipping_quote_primary(zone_code: str) -> dict[str, Any]:
        t_start = tracker.now_ms()
        await asyncio.sleep(0.35)
        if zone_code == "EU-NORTH":
            tracker.record(
                kind="tool",
                label="primary(EU-NORTH)",
                start_ms=t_start,
                end_ms=tracker.now_ms(),
            )
            return {"zone_code": "EU-NORTH", "shipping_eur": 42.50, "carrier": "FastEuro"}
        if zone_code == "LEGACY-99":
            tracker.record(
                kind="tool",
                label="primary(LEGACY-99)",
                start_ms=t_start,
                end_ms=tracker.now_ms(),
                status="error",
            )
            raise RuntimeError(
                "503 Service Unavailable for zone 'LEGACY-99' on primary carrier. "
                "Use 'shipping_quote_backup' instead."
            )
        raise ValueError(f"Unknown zone '{zone_code}'.")

    async def shipping_quote_backup(zone_code: str) -> dict[str, Any]:
        t_start = tracker.now_ms()
        await asyncio.sleep(0.45)
        rates = {"EU-NORTH": 46.00, "LEGACY-99": 87.50}
        if zone_code not in rates:
            tracker.record(
                kind="tool",
                label=f"backup({zone_code})",
                start_ms=t_start,
                end_ms=tracker.now_ms(),
                status="error",
            )
            raise ValueError(f"Unknown zone '{zone_code}'.")
        tracker.record(
            kind="tool",
            label=f"backup({zone_code})",
            start_ms=t_start,
            end_ms=tracker.now_ms(),
        )
        return {
            "zone_code": zone_code,
            "shipping_eur": rates[zone_code],
            "carrier": "SafeBackup",
        }

    async def calculate_invoice_total(cost_a_eur: float, cost_b_eur: float) -> dict[str, Any]:
        t_start = tracker.now_ms()
        await asyncio.sleep(0.25)
        subtotal = round(cost_a_eur + cost_b_eur, 2)
        insurance = round(subtotal * 0.05 + 3.0, 2)  # Opaque billing fee (5% + 3 EUR)
        total = round(subtotal + insurance, 2)
        tracker.record(
            kind="tool",
            label="calculate_invoice_total",
            start_ms=t_start,
            end_ms=tracker.now_ms(),
        )
        return {
            "subtotal_eur": subtotal,
            "insurance_fee_eur": insurance,
            "grand_total_eur": total,
        }

    return ToolRegistry(
        [
            ToolSpec(
                name="shipping_quote_primary",
                description=(
                    "Primary shipping carrier API. Prefer this first to fetch shipping_eur "
                    "for a zone_code ('EU-NORTH', 'LEGACY-99')."
                ),
                func=shipping_quote_primary,
                input_schema=QuoteInput,
                output_schema=QuoteOutput,
            ),
            ToolSpec(
                name="shipping_quote_backup",
                description=(
                    "Backup shipping carrier API. Use as fallback if shipping_quote_primary "
                    "returns a 503 outage for a zone_code."
                ),
                func=shipping_quote_backup,
                input_schema=QuoteInput,
                output_schema=QuoteOutput,
            ),
            ToolSpec(
                name="calculate_invoice_total",
                description=(
                    "Compute the official billing subtotal_eur, insurance_fee_eur, and "
                    "grand_total_eur from two shipping costs (cost_a_eur, cost_b_eur)."
                ),
                func=calculate_invoice_total,
                input_schema=InvoiceInput,
                output_schema=InvoiceOutput,
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Scenario specs & run summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DidacticScenario:
    id: int
    slug: str
    title: str
    goal: str
    topology_note: str
    request: str
    expected_numbers: tuple[str, ...]
    build_registry: Any


SCENARIOS: list[DidacticScenario] = [
    DidacticScenario(
        id=1,
        slug="asymmetric_dag",
        title="Scenario 1 — Il Grafo Asimmetrico (Parallelismo, Dataflow e Chiamate LLM)",
        goal=(
            "Capire perché LLM Compiler fa 2 sole chiamate LLM ed esegue i rami senza barriere "
            "mentre ReAct deve fermarsi a ogni livello per ripassare dall'LLM."
        ),
        topology_note=(
            "  Ramo A: fetch_order_summary (0.3s) ──▶ run_risk_audit (1.8s) ──┐\n"
            "                                                                  ├──▶ build_executive_score (0.3s)\n"
            "  Ramo B: fetch_warehouse_stock (1.8s) ─▶ compute_restock_cost (0.3s) ┘\n"
            "  (Lavoro sequenziale totale = 4.5s | Ottimo event-driven = 2.4s | Ottimo a turni = 3.9s)"
        ),
        request=(
            "Valuta l'ordine 'ORD-42' e lo stock del prodotto 'SKU-X9'. "
            "Recupera i dati dell'ordine e calcola il suo risk_score con run_risk_audit; "
            "in parallelo recupera lo stock di 'SKU-X9' e calcola il restock_cost_eur con "
            "compute_restock_cost. Infine combina risk_score e restock_cost_eur usando "
            "build_executive_score e riporta tutti i valori (risk_score, restock_cost_eur, "
            "priority_index e recommendation) in italiano."
        ),
        # Expected:
        #   risk_score = (12400/500)*1.35 + 7.25 = 40.73
        #   restock_cost_eur = 35*48*1.12 + 95 = 1976.6
        #   priority_index = 40.73*1.5 + (1976.6/100)*0.8 = 61.095 + 15.8128 = 76.91
        expected_numbers=("40.73", "1976.6", "76.91"),
        build_registry=build_scenario_1_registry,
    ),
    DidacticScenario(
        id=2,
        slug="runtime_recovery",
        title="Scenario 2 — L'Imprevisto a Runtime (Errore 503 + Tool di Backup)",
        goal=(
            "Confrontare il recupero strutturato di LLM Compiler v3 (best-effort + skipped + "
            "replan_context) con l'adattamento reattivo turno-per-turno di ReAct."
        ),
        topology_note=(
            "  shipping_quote_primary('EU-NORTH')  ──▶ OK (42.50 EUR)\n"
            "  shipping_quote_primary('LEGACY-99') ──▶ ERRORE 503! (backup: 87.50 EUR)\n"
            "  calculate_invoice_total(42.50, 87.50) ──▶ subtotal 130.00 + fee 9.50 = 139.50 EUR"
        ),
        request=(
            "Recupera le tariffe di spedizione per le zone 'EU-NORTH' e 'LEGACY-99' "
            "usando prima shipping_quote_primary (se fallisce, usa il backup disponibile), "
            "poi calcola la fattura ufficiale con calculate_invoice_total e riporta "
            "subtotal_eur, insurance_fee_eur e grand_total_eur in italiano."
        ),
        # Expected: subtotal=130.0, insurance=130*0.05+3=9.5, grand_total=139.5
        expected_numbers=("130", "9.5", "139.5"),
        build_registry=build_scenario_2_registry,
    ),
]


@dataclass
class RunReport:
    arch: str
    status: str
    wall_ms: float
    llm_ms: float
    tool_wall_ms: float
    llm_calls: int
    tool_calls: int
    tool_errors: int
    compile_retries: int
    replans: int
    input_tokens: int
    output_tokens: int
    exact_match: bool
    final_answer: str
    spans: list[TimelineSpan] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def _check_expected(answer: str, expected: tuple[str, ...]) -> bool:
    normalized = answer.replace(",", ".")
    return all(val in normalized for val in expected)


def _format_plan_inputs(inputs: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, val in inputs.items():
        if isinstance(val, InputBinding):
            path_str = "." + ".".join(val.output_path) if val.output_path else ""
            parts.append(f"{key} ◀── {val.task_id}{path_str}")
        else:
            parts.append(f"{key}={val!r}")
    return ", ".join(parts)


def _render_gantt(
    title: str,
    spans: list[TimelineSpan],
    total_ms: float,
    *,
    width: int = 48,
    style: Style,
) -> list[tuple[str, str]]:
    """Return pairs of (styled_line, plain_md_line) for an ASCII Gantt chart."""
    scale_ms = max(total_ms, 1.0)
    lines: list[tuple[str, str]] = []
    header_plain = f"  {title} (0 → {total_ms / 1000:.2f} s)"
    lines.append((style.bold(header_plain), header_plain))

    for sp in spans:
        s_idx = min(width - 1, max(0, int((sp.start_ms / scale_ms) * width)))
        e_idx = min(width, max(s_idx + 1, int((sp.end_ms / scale_ms) * width)))
        bar_char = "▓" if sp.kind == "llm" else ("✖" if sp.status == "error" else "█")
        raw_bar = ("·" * s_idx) + (bar_char * (e_idx - s_idx)) + ("·" * (width - e_idx))

        if sp.kind == "llm":
            colored_bar = style.magenta(raw_bar)
            tag = "[LLM ]"
        elif sp.status == "error":
            colored_bar = style.red(raw_bar)
            tag = "[ERR ]"
        else:
            colored_bar = style.green(raw_bar)
            tag = "[TOOL]"

        timing = f"{sp.start_ms / 1000:5.2f}s → {sp.end_ms / 1000:5.2f}s ({(sp.end_ms - sp.start_ms) / 1000:4.2f}s)"
        label = f"{sp.label:<24}"
        styled = f"    {tag} {label} │{colored_bar}│ {timing}"
        plain = f"    {tag} {label} │{raw_bar}│ {timing}"
        lines.append((styled, plain))

    return lines


def _compute_tool_union_ms(spans: list[TimelineSpan]) -> float:
    """Compute real wall-clock union of all tool execution intervals (handles parallelism)."""
    intervals = sorted((sp.start_ms, sp.end_ms) for sp in spans if sp.kind == "tool")
    if not intervals:
        return 0.0
    merged: list[tuple[float, float]] = [intervals[0]]
    for start, end in intervals[1:]:
        prev_s, prev_e = merged[-1]
        if start <= prev_e:
            merged[-1] = (prev_s, max(prev_e, end))
        else:
            merged.append((start, end))
    return sum(e - s for s, e in merged)


# ---------------------------------------------------------------------------
# Runner A: LLM Compiler v3 (narrated step-by-step via astream)
# ---------------------------------------------------------------------------


async def run_compiler_narrated(
    scenario: DidacticScenario,
    model: str,
    reasoning: bool,
    writer: DualWriter,
) -> RunReport:
    st = writer.style
    tracker = RunTracker()
    cb = TokenAndSpanCallback(tracker, prefix="Compiler LLM")
    registry = scenario.build_registry(tracker)

    planner_llm = create_ollama_chat_model(
        model=model, temperature=0.0, reasoning=reasoning, num_predict=2048
    )
    joiner_llm = create_ollama_chat_model(
        model=model, temperature=0.0, reasoning=False, num_predict=1024
    )
    graph = build_compiler_graph(
        registry=registry,
        planner=Planner(planner_llm),
        compiler=PlanCompiler(registry),
        executor=Executor(registry, max_concurrency=8, task_timeout_s=10.0),
        joiner=Joiner(joiner_llm),
        max_compile_retries=2,
        max_replans=2,
    )

    writer.out(
        st.cyan("┌──────────────────────────────────────────────────────────────────────────┐"),
        "### Parte A — LLM Compiler v3 (passo dopo passo)\n",
    )
    writer.out(
        st.cyan("│  ARCHITETTURA A: LLM COMPILER v3 (Plan -> Compile -> Execute -> Join)    │"),
        "",
    )
    writer.out(
        st.cyan("└──────────────────────────────────────────────────────────────────────────┘"),
        "",
    )

    tracker.reset()
    t0 = time.perf_counter()
    final_state: dict[str, Any] = {}
    round_num = 1
    step_num = 0
    tool_calls_total = 0
    tool_errors_total = 0

    async for chunk in graph.astream(
        {"request": scenario.request},
        config={"callbacks": [cb]},
    ):
        for node_name, update in chunk.items():
            final_state.update(update)
            step_num += 1

            if node_name == "planner":
                plan: ExecutionPlan | None = update.get("plan")
                err = update.get("error")
                writer.out(
                    st.bold(st.blue(f"\n  [Step {step_num} · Round {round_num}] PLANNER (chiamata LLM)")),
                    f"\n**[Step {step_num} · Round {round_num}] PLANNER (chiamata LLM)**",
                )
                if err is not None:
                    writer.out(st.red(f"    ERRORE: {err.error_type}: {err.message}"))
                elif plan is not None:
                    writer.out(
                        f"    Obiettivo: {plan.objective}",
                        f"- Obiettivo: `{plan.objective}`",
                    )
                    writer.out(
                        f"    Ha pianificato {len(plan.tasks)} task in UN SOLO COLPO:",
                        f"- Ha pianificato **{len(plan.tasks)} task** in una sola chiamata LLM:",
                    )
                    for task in plan.tasks:
                        line = (
                            f"      • {task.task_id:<7} │ {task.tool_name:<24} │ "
                            f"{_format_plan_inputs(task.inputs)}"
                        )
                        writer.out(line, f"  - `{task.task_id}` → `{task.tool_name}` ({_format_plan_inputs(task.inputs)})")
                    writer.out(
                        st.dim(
                            "    ↳ Nota didattica: i simboli '◀──' sono InputBinding: dichiarano "
                            "il dataflow prima che parta un solo tool!"
                        ),
                        "  - *Nota didattica: i simboli `◀──` sono `InputBinding`: dichiarano il dataflow prima che parta un solo tool.*",
                    )

            elif node_name == "compiler":
                dag = update.get("compiled_dag")
                err = update.get("error")
                m = update.get("metrics")
                comp_ms = (m.compilation_latency_ms or 0.0) if m else 0.0
                writer.out(
                    st.bold(
                        st.blue(
                            f"\n  [Step {step_num} · Round {round_num}] PLAN COMPILER "
                            f"(0 LLM, {comp_ms:.2f} ms)"
                        )
                    ),
                    f"\n**[Step {step_num} · Round {round_num}] PLAN COMPILER (0 LLM, {comp_ms:.2f} ms)**",
                )
                if err is not None:
                    writer.out(st.red(f"    BOCCIATO ({err.error_type}): {err.message}"))
                    for issue in err.issues:
                        writer.out(st.red(f"      - {issue}"))
                    writer.out(
                        st.yellow("    ↳ Ritorna al Planner con feedback strutturato (v2 self-correction).")
                    )
                elif dag is not None:
                    writer.out(
                        st.green(
                            f"    APPROVATO: CompiledDAG valido ({len(dag.tasks)} nodi, "
                            f"{len(dag.levels)} livelli, 0 errori statici)"
                        ),
                        f"- **APPROVATO**: `CompiledDAG` valido ({len(dag.tasks)} nodi, {len(dag.levels)} livelli, 0 errori statici)",
                    )
                    for lvl_idx, lvl in enumerate(dag.levels):
                        names = ", ".join(f"{tid}:{dag.tasks[tid].tool_name}" for tid in lvl)
                        writer.out(f"      Livello L{lvl_idx}: [{names}]", f"  - Livello `L{lvl_idx}`: `{names}`")

            elif node_name == "executor":
                res = update.get("execution_result")
                writer.out(
                    st.bold(
                        st.blue(
                            f"\n  [Step {step_num} · Round {round_num}] EXECUTOR "
                            f"(0 LLM, scheduler asyncio event-driven)"
                        )
                    ),
                    f"\n**[Step {step_num} · Round {round_num}] EXECUTOR (0 LLM, scheduler asyncio event-driven)**",
                )
                if res is not None:
                    writer.out(
                        f"    Wall-clock DAG: {res.total_duration_ms / 1000:.2f}s | "
                        f"Somma durate task: {res.sum_task_duration_ms / 1000:.2f}s | "
                        f"Parallelismo effettivo: {res.effective_parallelism:.2f}x",
                        f"- Wall-clock DAG: `{res.total_duration_ms / 1000:.2f}s` | "
                        f"Somma durate task: `{res.sum_task_duration_ms / 1000:.2f}s` | "
                        f"Parallelismo effettivo: `{res.effective_parallelism:.2f}x`",
                    )
                    ordered_items = sorted(
                        res.results.items(),
                        key=lambda kv: (kv[1].started_at_ms is None, kv[1].started_at_ms or 0.0, kv[0]),
                    )
                    for tid, tr in ordered_items:
                        if tr.status == "success":
                            tool_calls_total += 1
                            t_win = f"{(tr.started_at_ms or 0) / 1000:4.2f}s → {(tr.finished_at_ms or 0) / 1000:4.2f}s"
                            writer.out(
                                st.green(f"      [OK  ] {tid:<7} ({tr.tool_name:<23}) {t_win} │ {tr.output}"),
                                f"  - `[OK]` `{tid}` (`{tr.tool_name}`) `{t_win}` → `{tr.output}`",
                            )
                        elif tr.status == "error":
                            tool_calls_total += 1
                            tool_errors_total += 1
                            t_win = f"{(tr.started_at_ms or 0) / 1000:4.2f}s → {(tr.finished_at_ms or 0) / 1000:4.2f}s"
                            writer.out(
                                st.red(f"      [ERR ] {tid:<7} ({tr.tool_name:<23}) {t_win} │ {tr.error}"),
                                f"  - `[ERR]` `{tid}` (`{tr.tool_name}`) `{t_win}` → `{tr.error}`",
                            )
                        else:
                            writer.out(
                                st.yellow(f"      [SKIP] {tid:<7} ({tr.tool_name:<23}) saltato │ {tr.skip_reason}"),
                                f"  - `[SKIP]` `{tid}` (`{tr.tool_name}`) → *{tr.skip_reason}*",
                            )

            elif node_name == "joiner":
                jr = update.get("joiner_result")
                writer.out(
                    st.bold(st.blue(f"\n  [Step {step_num} · Round {round_num}] JOINER (chiamata LLM + stato deterministico)")),
                    f"\n**[Step {step_num} · Round {round_num}] JOINER (chiamata LLM + stato deterministico)**",
                )
                if jr is not None and jr.action == "replan":
                    writer.out(
                        st.yellow(f"    DECISIONE: action='replan' → Avvio Round {round_num + 1}!"),
                        f"- **DECISIONE**: `action='replan'` → Avvio **Round {round_num + 1}**",
                    )
                    writer.out(
                        st.yellow(f"    Motivo: {jr.replan_reason}"),
                        f"- Motivo: *{jr.replan_reason}*",
                    )
                    writer.out(
                        st.dim(
                            "    ↳ Nota didattica: i risultati riusciti del Round 1 vengono salvati "
                            "in `replan_context`, così il Round 2 NON ripeterà le chiamate già riuscite!"
                        ),
                        "- *Nota didattica: i risultati già riusciti vengono salvati in `replan_context`, così il Round 2 non ripeterà le chiamate già riuscite.*",
                    )
                    round_num += 1
                elif jr is not None:
                    writer.out(
                        st.green(f"    DECISIONE: action='finish' | status='{jr.status}'"),
                        f"- **DECISIONE**: `action='finish'` | `status='{jr.status}'`",
                    )

    wall_ms = (time.perf_counter() - t0) * 1000.0
    answer = str(final_state.get("final_answer") or "")
    metrics = final_state["metrics"]
    exact = _check_expected(answer, scenario.expected_numbers)
    llm_ms = sum(sp.end_ms - sp.start_ms for sp in tracker.spans if sp.kind == "llm")
    tool_wall_ms = _compute_tool_union_ms(tracker.spans)

    writer.out(st.bold("\n  Risposta finale LLM Compiler v3:"), "\n**Risposta finale LLM Compiler v3:**")
    for line in answer.splitlines():
        writer.out(f"    {line}", f"> {line}")
    writer.blank()

    return RunReport(
        arch="LLM Compiler v3",
        status=str(final_state.get("status") or "failed"),
        wall_ms=wall_ms,
        llm_ms=llm_ms,
        tool_wall_ms=tool_wall_ms,
        llm_calls=metrics.llm_calls,
        tool_calls=tool_calls_total,
        tool_errors=tool_errors_total,
        compile_retries=metrics.compile_retries,
        replans=metrics.replan_attempts,
        input_tokens=cb.input_tokens,
        output_tokens=cb.output_tokens,
        exact_match=exact,
        final_answer=answer,
        spans=list(tracker.spans),
    )


# ---------------------------------------------------------------------------
# Runner B: LangGraph ReAct (narrated turn-by-turn via astream)
# ---------------------------------------------------------------------------


async def run_react_narrated(
    scenario: DidacticScenario,
    model: str,
    reasoning: bool,
    writer: DualWriter,
) -> RunReport:
    st = writer.style
    tracker = RunTracker()
    cb = TokenAndSpanCallback(tracker, prefix="ReAct LLM")
    registry = scenario.build_registry(tracker)

    llm = create_ollama_chat_model(
        model=model, temperature=0.0, reasoning=reasoning, num_predict=2048
    )
    graph = build_react_graph(
        registry=registry,
        llm=llm,
        max_iterations=10,
        task_timeout_s=10.0,
    )

    writer.out(
        st.magenta("┌──────────────────────────────────────────────────────────────────────────┐"),
        "\n### Parte B — LangGraph ReAct (turno per turno)\n",
    )
    writer.out(
        st.magenta("│  ARCHITETTURA B: LANGGRAPH ReAct (Agent -> Tools -> Agent -> Tools ...)  │"),
        "",
    )
    writer.out(
        st.magenta("└──────────────────────────────────────────────────────────────────────────┘"),
        "",
    )

    tracker.reset()
    t0 = time.perf_counter()
    final_state: dict[str, Any] = {}
    turn = 0

    async for chunk in graph.astream(
        {"request": scenario.request},
        config={"callbacks": [cb]},
    ):
        for node_name, update in chunk.items():
            final_state.update(update)
            msgs = update.get("messages") or []

            if node_name == "agent":
                turn += 1
                ai_msgs = [m for m in msgs if isinstance(m, AIMessage)]
                ai_msg = ai_msgs[-1] if ai_msgs else None
                writer.out(
                    st.bold(st.magenta(f"\n  [Turno {turn} · AGENT] (chiamata LLM #{turn})")),
                    f"\n**[Turno {turn} · AGENT] (chiamata LLM #{turn})**",
                )
                if ai_msg is not None and ai_msg.tool_calls:
                    writer.out(
                        f"    Decide sul momento di chiamare {len(ai_msg.tool_calls)} tool:",
                        f"- Decide sul momento di chiamare **{len(ai_msg.tool_calls)} tool**:",
                    )
                    for tc in ai_msg.tool_calls:
                        writer.out(
                            f"      • {tc['name']}({tc['args']})",
                            f"  - `{tc['name']}({tc['args']})`",
                        )
                    if turn == 1 and scenario.id == 1:
                        writer.out(
                            st.dim(
                                "    ↳ Nota didattica: non può ancora pianificare i passi successivi "
                                "perché non esiste un meccanismo di InputBinding tra turni!"
                            ),
                            "  - *Nota didattica: non può ancora pianificare i passi successivi perché non ha un `InputBinding` esplicito tra tool.*",
                        )
                else:
                    writer.out(
                        st.green("    Nessuna altra tool call → scrive la risposta finale!"),
                        "- Nessuna altra tool call → scrive la risposta finale.",
                    )

            elif node_name == "tools":
                tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
                writer.out(
                    st.bold(st.magenta(f"\n  [Turno {turn} · TOOLS] (esegue il batch e ATTENDE tutto il batch)")),
                    f"\n**[Turno {turn} · TOOLS] (esegue il batch e attende tutto il batch)**",
                )
                for tm in tool_msgs:
                    is_err = getattr(tm, "status", None) == "error"
                    if is_err:
                        writer.out(
                            st.red(f"      [ERR ] {tm.name}: {tm.content}"),
                            f"  - `[ERR]` `{tm.name}` → `{tm.content}`",
                        )
                    else:
                        writer.out(
                            st.green(f"      [OK  ] {tm.name}: {tm.content}"),
                            f"  - `[OK]` `{tm.name}` → `{tm.content}`",
                        )
                if turn == 1 and scenario.id == 1:
                    writer.out(
                        st.dim(
                            "    ↳ Nota didattica: anche se fetch_order_summary finisce in 0.30s, "
                            "ReAct resta fermo finché non termina anche fetch_warehouse_stock (1.80s)!"
                        ),
                        "  - *Nota didattica: anche se `fetch_order_summary` finisce in 0.30s, il turno resta bloccato finché non termina anche `fetch_warehouse_stock` (1.80s).*",
                    )

    wall_ms = (time.perf_counter() - t0) * 1000.0
    answer = str(final_state.get("final_answer") or "")
    m = final_state["metrics"]
    exact = _check_expected(answer, scenario.expected_numbers)
    llm_ms = sum(sp.end_ms - sp.start_ms for sp in tracker.spans if sp.kind == "llm")
    tool_wall_ms = _compute_tool_union_ms(tracker.spans)

    writer.out(st.bold("\n  Risposta finale LangGraph ReAct:"), "\n**Risposta finale LangGraph ReAct:**")
    for line in answer.splitlines():
        writer.out(f"    {line}", f"> {line}")
    writer.blank()

    return RunReport(
        arch="LangGraph ReAct",
        status=str(final_state.get("status") or "failed"),
        wall_ms=wall_ms,
        llm_ms=llm_ms,
        tool_wall_ms=tool_wall_ms,
        llm_calls=m.llm_calls,
        tool_calls=m.tool_calls,
        tool_errors=m.tool_errors,
        compile_retries=0,
        replans=0,
        input_tokens=m.input_tokens or cb.input_tokens,
        output_tokens=m.output_tokens or cb.output_tokens,
        exact_match=exact,
        final_answer=answer,
        spans=list(tracker.spans),
    )


# ---------------------------------------------------------------------------
# Side-by-side didactic summary (Gantt + Table + Key Takeaways)
# ---------------------------------------------------------------------------


def print_comparison(
    scenario: DidacticScenario,
    comp: RunReport,
    react: RunReport,
    writer: DualWriter,
) -> None:
    st = writer.style
    shared_scale = max(comp.wall_ms, react.wall_ms)

    writer.out(
        st.bold("════════════════════════════════════════════════════════════════════════════"),
        "\n### Confronto visivo e lezione didattica\n",
    )
    writer.out(
        st.bold(f"  TIMELINE REALE A CONFRONTO (stessa scala: 0 → {shared_scale / 1000:.2f} s)"),
        f"```text\nTIMELINE REALE A CONFRONTO (stessa scala: 0 -> {shared_scale / 1000:.2f} s)",
    )
    writer.out(
        st.dim("  Legenda: ▓ = chiamata LLM | █ = esecuzione Tool OK | ✖ = Tool in errore"),
        "Legenda: ▓ = chiamata LLM | █ = esecuzione Tool OK | ✖ = Tool in errore",
    )
    writer.blank()

    for styled, plain in _render_gantt("LLM Compiler v3", comp.spans, shared_scale, style=st):
        writer.out(styled, plain)
    writer.blank()
    for styled, plain in _render_gantt("LangGraph ReAct", react.spans, shared_scale, style=st):
        writer.out(styled, plain)

    writer.out("", "```\n")

    # Summary comparison table
    writer.out(
        st.bold("  TABELLA DI CONFRONTO:"),
        "| Metrica | LLM Compiler v3 | LangGraph ReAct | Differenza |",
    )
    writer.out(
        "  ──────────────────────────────────────────────────────────────────────────",
        "| :--- | ---: | ---: | :--- |",
    )

    rows = [
        (
            "Risultato esatto",
            "EXACT" if comp.exact_match else "MISS",
            "EXACT" if react.exact_match else "MISS",
            "Parità" if comp.exact_match == react.exact_match else "Vince chi è EXACT",
        ),
        (
            "Chiamate LLM",
            f"{comp.llm_calls}",
            f"{react.llm_calls}",
            f"{comp.llm_calls - react.llm_calls:+d} chiamate",
        ),
        (
            "Tempo I/O Tool (unione reale)",
            f"{comp.tool_wall_ms / 1000:.2f} s",
            f"{react.tool_wall_ms / 1000:.2f} s",
            f"{(comp.tool_wall_ms - react.tool_wall_ms) / 1000:+.2f} s",
        ),
        (
            "Tempo totale LLM",
            f"{comp.llm_ms / 1000:.2f} s",
            f"{react.llm_ms / 1000:.2f} s",
            f"{(comp.llm_ms - react.llm_ms) / 1000:+.2f} s",
        ),
        (
            "Wall-clock totale",
            f"{comp.wall_ms / 1000:.2f} s",
            f"{react.wall_ms / 1000:.2f} s",
            f"{(comp.wall_ms - react.wall_ms) / 1000:+.2f} s",
        ),
        (
            "Token (In / Out / Tot)",
            f"{comp.input_tokens}/{comp.output_tokens}/{comp.total_tokens}",
            f"{react.input_tokens}/{react.output_tokens}/{react.total_tokens}",
            f"{comp.total_tokens - react.total_tokens:+d} tot",
        ),
        (
            "Validazione statica pre-run",
            "Sì (PlanCompiler)",
            "No (scopre a runtime)",
            "DAG certificato prima di eseguire",
        ),
        (
            "Retry / Replan",
            f"{comp.compile_retries} compile / {comp.replans} replan",
            "Loop nativo a turni",
            "—",
        ),
    ]
    for metric, v_c, v_r, delta in rows:
        writer.out(
            f"  {metric:<30} │ {v_c:>18} │ {v_r:>18} │ {delta}",
            f"| {metric} | `{v_c}` | `{v_r}` | {delta} |",
        )

    writer.blank()
    writer.out(st.bold("  COSA CI INSEGNA QUESTO SCENARIO:"), "\n**Cosa ci insegna questo scenario:**")
    if scenario.id == 1:
        io_saved = (react.tool_wall_ms - comp.tool_wall_ms) / 1000.0
        writer.out(
            f"  1. Parallelismo event-driven reale: LLM Compiler ha completato tutti i tool in "
            f"{comp.tool_wall_ms / 1000:.2f}s contro {react.tool_wall_ms / 1000:.2f}s di ReAct "
            f"(risparmio I/O: {io_saved:+.2f}s) perché `run_risk_audit` è partito subito senza aspettare `fetch_warehouse_stock`.",
            f"1. **Parallelismo event-driven reale**: LLM Compiler ha eseguito l'intero DAG di tool in `{comp.tool_wall_ms / 1000:.2f}s` contro `{react.tool_wall_ms / 1000:.2f}s` di ReAct (`{io_saved:+.2f}s` risparmiati sull'I/O), perché `run_risk_audit` è partito appena finito `fetch_order_summary` senza aspettare `fetch_warehouse_stock`.",
        )
        writer.out(
            f"  2. Round-trip LLM ridotti: {comp.llm_calls} chiamate LLM (Planner + Joiner) contro "
            f"{react.llm_calls} in ReAct — su API cloud (dove ogni chiamata paga TTFT di rete) questo dimezza i tempi morti.",
            f"2. **Meno chiamate LLM**: `{comp.llm_calls}` chiamate LLM (`Planner` + `Joiner`) contro `{react.llm_calls}` in ReAct.",
        )
        writer.out(
            f"  3. Trade-off token in locale: il Planner scrive un piano JSON più lungo ({comp.output_tokens} token di output vs {react.output_tokens} di ReAct), "
            f"ma consuma meno input ripetuti ({comp.input_tokens} vs {react.input_tokens}) perché non rispedisce l'intera chat history a ogni turno.",
            f"3. **Trade-off token (Output JSON vs Input ripetuti)**: il Planner genera più token di output (`{comp.output_tokens}` vs `{react.output_tokens}`), ma evita di rispedire tutta la cronologia a ogni turno (`{comp.input_tokens}` input vs `{react.input_tokens}`).",
        )
    else:
        writer.out(
            "  1. Gestione sicura del guasto in LLM Compiler v3: quando `primary(LEGACY-99)` va in 503, "
            "`calculate_invoice_total` viene marcato `SKIPPED` (non gira con dati sporchi!), `EU-NORTH` viene salvato "
            "in `replan_context`, e il Round 2 chiama SOLO il backup + il totale finale.",
            "1. **Gestione sicura del guasto in LLM Compiler v3**: `calculate_invoice_total` viene marcato `SKIPPED`, il risultato già riuscito di `EU-NORTH` viene preservato in `replan_context`, e il Round 2 esegue solo il backup + il totale.",
        )
        writer.out(
            f"  2. Punto di forza naturale di ReAct: su imprevisti esplorativi, ReAct legge il 503 nel ToolMessage "
            f"e si adatta al turno subito dopo con soli {react.output_tokens} token di output totali (contro {comp.output_tokens} di Compiler v3).",
            f"2. **Punto di forza naturale di ReAct**: su imprevisti runtime, ReAct si adatta al turno successivo con soli `{react.output_tokens}` token di output totali (contro `{comp.output_tokens}` di Compiler v3).",
        )
        writer.out(
            "  3. Sintesi onesta: LLM Compiler v3 brilla quando c'è un grafo multi-step parallelo pianificabile in anticipo "
            "(Scenario 1); ReAct brilla quando il percorso è incerto/esplorativo e i tool sono veloci (Scenario 2).",
            "3. **Sintesi onesta**: LLM Compiler v3 brilla sui flussi multi-step paralleli e asimmetrici (Scenario 1); ReAct brilla nei flussi brevi, veloci ed esplorativi (Scenario 2).",
        )
    writer.blank()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Didactic LLM Compiler v3 vs LangGraph ReAct lab.")
    parser.add_argument("--model", default="qwen3:8b", help="Ollama model name.")
    parser.add_argument("--reasoning", action="store_true", help="Enable thinking mode on Planner/ReAct.")
    parser.add_argument(
        "--scenario",
        choices=["all", "1", "2"],
        default="all",
        help="Which didactic scenario to run (default: all).",
    )
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI terminal colors.")
    args = parser.parse_args()

    style = Style(color=not args.no_color)
    writer = DualWriter(style)

    selected = [
        sc
        for sc in SCENARIOS
        if args.scenario == "all" or str(sc.id) == args.scenario
    ]

    # Warm up Ollama once so Scenario 1 doesn't pay cold-model-load jitter
    warmup_llm = create_ollama_chat_model(
        model=args.model, temperature=0.0, reasoning=False, num_predict=16
    )
    await warmup_llm.ainvoke("Say OK.")

    writer.out(
        style.bold(
            f"=== LABORATORIO DIDATTICO: LLM Compiler v3 vs LangGraph ReAct ({args.model}, reasoning={args.reasoning}) ==="
        ),
        f"# Laboratorio Didattico: LLM Compiler v3 vs LangGraph ReAct\n\n"
        f"- **Modello:** `{args.model}` (`temperature=0.0`, `reasoning={args.reasoning}`)\n"
        f"- **Obiettivo:** osservare passo dopo passo come ragionano ed eseguono gli stessi identici tool.",
    )
    writer.blank()

    for sc in selected:
        writer.out(
            style.bold(style.yellow("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")),
            f"\n---\n\n## {sc.title}\n",
        )
        writer.out(style.bold(style.yellow(f"  {sc.title}")), "")
        writer.out(f"  Obiettivo: {sc.goal}", f"**Obiettivo:** {sc.goal}\n")
        writer.out(style.dim(sc.topology_note), f"```text\n{sc.topology_note}\n```\n")
        writer.out(f"\n  Richiesta identica per entrambi:\n  \"{sc.request}\"\n", f"**Richiesta identica per entrambi:**\n> {sc.request}\n")

        comp_report = await run_compiler_narrated(sc, args.model, args.reasoning, writer)
        react_report = await run_react_narrated(sc, args.model, args.reasoning, writer)
        print_comparison(sc, comp_report, react_report, writer)

    writer.save(RESULTS_MD_PATH)
    print(style.green(f"Report didattico salvato in: {RESULTS_MD_PATH}"))


if __name__ == "__main__":
    asyncio.run(main())
