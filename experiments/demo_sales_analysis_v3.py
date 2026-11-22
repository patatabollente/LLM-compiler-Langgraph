"""Standalone v3 simulation with ONLY the `analyze_sales_data` tool.

Demonstrates how easily a new domain tool plugs into LLM Compiler v3 without
modifying a single line of `src/langgraph_llm_compiler/`.

Runs 3 queries on a deterministic synthetic e-commerce sales table:
  1. Single targeted aggregation (revenue by category in NORTH)
  2. Parallel multi-query fan-out (compare NORTH vs SOUTH vs ISLAND + overall units)
  3. Resilience / self-correction (invalid zone 'NORD' caught by v2/v3 + nonexistent
     filter handling)

Usage:
    uv run python experiments/demo_sales_analysis_v3.py
    uv run python experiments/demo_sales_analysis_v3.py --reasoning
"""

import argparse
import asyncio
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from langgraph_adapters.compiler_graph import build_compiler_graph
from llm_compiler.compiler import PlanCompiler, ToolRegistry, ToolSpec
from llm_compiler.executor import Executor
from llm_compiler.joiner import Joiner
from llm_compiler.llm import create_ollama_chat_model
from llm_compiler.planner import Planner

# ---------------------------------------------------------------------------
# Deterministic synthetic e-commerce sales dataset (zero world-knowledge leak)
# ---------------------------------------------------------------------------
SALES_TABLE: tuple[dict[str, Any], ...] = (
    {"order_id": "ORD-001", "month": "2025-01", "zone": "NORTH", "category": "Electronics", "sku": "SKU-101", "units": 12, "revenue_eur": 5400.0, "margin_eur": 1620.0},
    {"order_id": "ORD-002", "month": "2025-01", "zone": "NORTH", "category": "Home",        "sku": "SKU-201", "units": 20, "revenue_eur": 2400.0, "margin_eur": 840.0},
    {"order_id": "ORD-003", "month": "2025-01", "zone": "NORTH", "category": "Accessories", "sku": "SKU-301", "units": 35, "revenue_eur": 1050.0, "margin_eur": 525.0},
    {"order_id": "ORD-004", "month": "2025-02", "zone": "NORTH", "category": "Electronics", "sku": "SKU-102", "units": 15, "revenue_eur": 7500.0, "margin_eur": 2250.0},
    {"order_id": "ORD-005", "month": "2025-02", "zone": "NORTH", "category": "Home",        "sku": "SKU-202", "units": 18, "revenue_eur": 2700.0, "margin_eur": 945.0},
    {"order_id": "ORD-006", "month": "2025-03", "zone": "NORTH", "category": "Electronics", "sku": "SKU-101", "units": 10, "revenue_eur": 4500.0, "margin_eur": 1350.0},
    {"order_id": "ORD-007", "month": "2025-01", "zone": "SOUTH", "category": "Electronics", "sku": "SKU-101", "units": 8,  "revenue_eur": 3600.0, "margin_eur": 1080.0},
    {"order_id": "ORD-008", "month": "2025-01", "zone": "SOUTH", "category": "Home",        "sku": "SKU-201", "units": 25, "revenue_eur": 3000.0, "margin_eur": 1050.0},
    {"order_id": "ORD-009", "month": "2025-02", "zone": "SOUTH", "category": "Accessories", "sku": "SKU-302", "units": 40, "revenue_eur": 1600.0, "margin_eur": 800.0},
    {"order_id": "ORD-010", "month": "2025-02", "zone": "SOUTH", "category": "Electronics", "sku": "SKU-102", "units": 6,  "revenue_eur": 3000.0, "margin_eur": 900.0},
    {"order_id": "ORD-011", "month": "2025-03", "zone": "SOUTH", "category": "Home",        "sku": "SKU-202", "units": 14, "revenue_eur": 2100.0, "margin_eur": 735.0},
    {"order_id": "ORD-012", "month": "2025-03", "zone": "SOUTH", "category": "Accessories", "sku": "SKU-301", "units": 30, "revenue_eur": 900.0,  "margin_eur": 450.0},
    {"order_id": "ORD-013", "month": "2025-01", "zone": "CENTRAL", "category": "Electronics", "sku": "SKU-102", "units": 11, "revenue_eur": 5500.0, "margin_eur": 1650.0},
    {"order_id": "ORD-014", "month": "2025-01", "zone": "CENTRAL", "category": "Home",        "sku": "SKU-201", "units": 16, "revenue_eur": 1920.0, "margin_eur": 672.0},
    {"order_id": "ORD-015", "month": "2025-02", "zone": "CENTRAL", "category": "Accessories", "sku": "SKU-302", "units": 25, "revenue_eur": 1000.0, "margin_eur": 500.0},
    {"order_id": "ORD-016", "month": "2025-03", "zone": "CENTRAL", "category": "Electronics", "sku": "SKU-101", "units": 9,  "revenue_eur": 4050.0, "margin_eur": 1215.0},
    {"order_id": "ORD-017", "month": "2025-01", "zone": "ISLAND",  "category": "Home",        "sku": "SKU-202", "units": 10, "revenue_eur": 1500.0, "margin_eur": 525.0},
    {"order_id": "ORD-018", "month": "2025-02", "zone": "ISLAND",  "category": "Electronics", "sku": "SKU-101", "units": 5,  "revenue_eur": 2250.0, "margin_eur": 675.0},
    {"order_id": "ORD-019", "month": "2025-02", "zone": "ISLAND",  "category": "Accessories", "sku": "SKU-301", "units": 20, "revenue_eur": 600.0,  "margin_eur": 300.0},
    {"order_id": "ORD-020", "month": "2025-03", "zone": "ISLAND",  "category": "Home",        "sku": "SKU-201", "units": 12, "revenue_eur": 1440.0, "margin_eur": 504.0},
)


# ---------------------------------------------------------------------------
# Single tool schema & implementation: analyze_sales_data
# ---------------------------------------------------------------------------
class SalesAnalysisInput(BaseModel):
    metric: Literal["revenue_eur", "margin_eur", "units"] = Field(
        description="Numeric column to aggregate."
    )
    aggregation: Literal["sum", "avg", "count", "min", "max"] = Field(
        description="Aggregation operation to apply."
    )
    group_by: Literal["category", "zone", "month", "all"] = Field(
        default="all",
        description="Dimension to group results by, or 'all' for a single overall value.",
    )
    filter_zone: Literal["NORTH", "SOUTH", "CENTRAL", "ISLAND"] | None = Field(
        default=None,
        description="Optional filter restricting rows to a single zone.",
    )
    filter_category: Literal["Electronics", "Home", "Accessories"] | None = Field(
        default=None,
        description="Optional filter restricting rows to a single product category.",
    )
    filter_month: str | None = Field(
        default=None,
        description="Optional filter restricting rows to a single month (YYYY-MM, e.g. '2025-01').",
    )


class SalesAnalysisOutput(BaseModel):
    metric: str
    aggregation: str
    group_by: str
    filters: dict[str, str]
    row_count: int
    results: dict[str, float]


async def analyze_sales_data(
    metric: Literal["revenue_eur", "margin_eur", "units"],
    aggregation: Literal["sum", "avg", "count", "min", "max"],
    group_by: Literal["category", "zone", "month", "all"] = "all",
    filter_zone: Literal["NORTH", "SOUTH", "CENTRAL", "ISLAND"] | None = None,
    filter_category: Literal["Electronics", "Home", "Accessories"] | None = None,
    filter_month: str | None = None,
) -> dict[str, Any]:
    await asyncio.sleep(0.15)  # simulate realistic DB/analytics query latency

    rows = list(SALES_TABLE)
    filters: dict[str, str] = {}
    if filter_zone is not None:
        filters["zone"] = filter_zone
        rows = [r for r in rows if r["zone"] == filter_zone]
    if filter_category is not None:
        filters["category"] = filter_category
        rows = [r for r in rows if r["category"] == filter_category]
    if filter_month is not None:
        filters["month"] = filter_month
        rows = [r for r in rows if r["month"] == filter_month]

    if not rows:
        valid_months = sorted({r["month"] for r in SALES_TABLE})
        raise ValueError(
            f"No sales rows matched filters {filters}. "
            f"Available months are: {valid_months}."
        )

    buckets: dict[str, list[float]] = {}
    for row in rows:
        key = "all" if group_by == "all" else str(row[group_by])
        buckets.setdefault(key, []).append(float(row[metric]))

    results: dict[str, float] = {}
    for key in sorted(buckets):
        vals = buckets[key]
        if aggregation == "sum":
            agg_val = sum(vals)
        elif aggregation == "avg":
            agg_val = sum(vals) / len(vals)
        elif aggregation == "count":
            agg_val = float(len(vals))
        elif aggregation == "min":
            agg_val = min(vals)
        else:
            agg_val = max(vals)
        results[key] = round(agg_val, 2)

    return {
        "metric": metric,
        "aggregation": aggregation,
        "group_by": group_by,
        "filters": filters,
        "row_count": len(rows),
        "results": results,
    }


def build_sales_only_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ToolSpec(
                name="analyze_sales_data",
                description=(
                    "Query and aggregate the e-commerce sales table by metric "
                    "(revenue_eur, margin_eur, units), aggregation (sum, avg, count, "
                    "min, max), optional grouping (category, zone, month, all), and "
                    "optional filters (filter_zone, filter_category, filter_month)."
                ),
                func=analyze_sales_data,
                input_schema=SalesAnalysisInput,
                output_schema=SalesAnalysisOutput,
            )
        ]
    )


QUERIES: tuple[tuple[str, str], ...] = (
    (
        "s01_single_zone_breakdown",
        "Qual è il ricavo totale (revenue_eur) per ogni categoria nella zona NORTH?",
    ),
    (
        "s02_parallel_multi_angle_analysis",
        (
            "Fammi un'analisi completa in parallelo: "
            "1) ricavo totale per zona (NORTH, SOUTH, CENTRAL, ISLAND), "
            "2) margine medio (margin_eur) per categoria su tutto il dataset, "
            "3) andamento mensile delle unità vendute (units) solo per Electronics."
        ),
    ),
    (
        "s03_partial_invalid_month",
        (
            "Confronta i ricavi totali per categoria del mese 2025-01 "
            "con quelli del mese 2025-04."
        ),
    ),
)


async def run_demo(model: str, reasoning: bool) -> None:
    registry = build_sales_only_registry()
    planner_llm = create_ollama_chat_model(
        model=model, temperature=0.0, reasoning=reasoning, num_predict=2048
    )
    joiner_llm = create_ollama_chat_model(
        model=model, temperature=0.0, reasoning=False, num_predict=2048
    )
    graph = build_compiler_graph(
        registry=registry,
        planner=Planner(llm=planner_llm),
        compiler=PlanCompiler(registry=registry),
        executor=Executor(registry=registry, max_concurrency=8),
        joiner=Joiner(llm=joiner_llm),
        max_compile_retries=2,
        max_replans=2,
    )

    print(f"=== LLM Compiler v3 Simulation (ONLY tool: analyze_sales_data) | {model} ===\n")
    for qid, request in QUERIES:
        print(f"--- {qid} ---")
        print(f"Richiesta: {request}")
        t0 = time.perf_counter()
        state = await graph.ainvoke({"request": request})
        wall_s = time.perf_counter() - t0

        metrics = state["metrics"]
        plan = state.get("plan")
        dag = state.get("compiled_dag")
        exec_res = state.get("execution_result")

        if plan is not None:
            print(f"Piano generato ({len(plan.tasks)} task):")
            for t in plan.tasks:
                print(f"  - {t.task_id}: {t.tool_name}({t.static_inputs})")
        if dag is not None:
            print(f"DAG compilato: {len(dag.tasks)} nodi, {len(dag.levels)} livelli")
        if exec_res is not None:
            print(
                f"Executor: {exec_res.total_duration_ms:.1f} ms "
                f"(effective_parallelism={exec_res.effective_parallelism:.2f}x)"
            )
            for tid, res in exec_res.results.items():
                if res.status == "success":
                    print(f"  [OK]  {tid}: {res.output['results']} (rows={res.output['row_count']})")
                else:
                    print(f"  [ERR] {tid}: {res.error}")

        print(
            f"Stato: {state.get('status')} | Wall: {wall_s:.2f}s | "
            f"LLM calls: {metrics.llm_calls} | compile_retries: {metrics.compile_retries} | "
            f"replans: {metrics.replan_attempts}"
        )
        print(f"Risposta Joiner:\n{state.get('final_answer')}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--reasoning", action="store_true")
    args = parser.parse_args()
    asyncio.run(run_demo(model=args.model, reasoning=args.reasoning))
