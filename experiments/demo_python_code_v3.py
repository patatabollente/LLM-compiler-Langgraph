"""Standalone v3 simulation featuring sandboxed `run_python_code` + `analyze_sales_data`.

Demonstrates:
  1. Pure sandboxed Python code execution (progressive tiered discount simulation).
  2. Chaining `analyze_sales_data` -> `run_python_code` via `InputBinding`
     (mean, median, stdev across zones — stats not built into the fixed analytics tool).
  3. Parallel fan-in: 2 parallel `analyze_sales_data` calls (revenues + margins)
     feeding into `run_python_code` (`data` + `extra_data`) to compute margin % per category.

Security model for `run_python_code`:
  - AST validation blocks all `import`/`from ... import`, dunder attribute access,
    and dangerous builtins (`open`, `eval`, `exec`, `getattr`, etc.).
  - Runs with an explicit safe-builtins whitelist + pre-loaded `math` and `statistics`.
  - Bounded by Executor `task_timeout_s`.

Usage:
    uv run python experiments/demo_python_code_v3.py
    uv run python experiments/demo_python_code_v3.py --reasoning
"""

import argparse
import asyncio
import time

from demo_sales_analysis_v3 import (
    SalesAnalysisInput,
    SalesAnalysisOutput,
    analyze_sales_data,
)

from langgraph_llm_compiler.compiler import PlanCompiler, ToolRegistry, ToolSpec
from langgraph_llm_compiler.executor import Executor
from langgraph_llm_compiler.graph import build_compiler_graph
from langgraph_llm_compiler.joiner import Joiner
from langgraph_llm_compiler.llm import create_ollama_chat_model
from langgraph_llm_compiler.planner import Planner
from langgraph_llm_compiler.tools import build_python_sandbox_tool_spec


def build_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ToolSpec(
                name="analyze_sales_data",
                description=(
                    "Query and aggregate the e-commerce sales table by metric "
                    "(revenue_eur, margin_eur, units), aggregation (sum, avg, count, "
                    "min, max), optional grouping (category, zone, month, all), and "
                    "optional filters (filter_zone, filter_category, filter_month). "
                    "Leave filter_zone/filter_category/filter_month omitted when "
                    "grouping across all values."
                ),
                func=analyze_sales_data,
                input_schema=SalesAnalysisInput,
                output_schema=SalesAnalysisOutput,
            ),
            build_python_sandbox_tool_spec(),
        ]
    )


QUERIES: tuple[tuple[str, str], ...] = (
    (
        "p01_progressive_discount_simulation",
        (
            "Usa run_python_code per simulare uno sconto progressivo a scaglioni "
            "su un ordine da 3200 EUR: 0% sui primi 1000 EUR, 10% sulla fascia tra "
            "1000 e 2500 EUR, e 20% sulla parte oltre 2500 EUR. Restituisci lo "
            "sconto totale in euro e il prezzo finale netto."
        ),
    ),
    (
        "p02_sales_plus_python_stats",
        (
            "Recupera con analyze_sales_data il ricavo totale (revenue_eur) per "
            "ciascuna zona (tutte le zone, senza filtrare una singola zona), poi "
            "passa quei risultati a run_python_code (nel parametro data) per "
            "calcolare media, mediana e deviazione standard campionaria "
            "(statistics.stdev) dei ricavi tra le zone."
        ),
    ),
    (
        "p03_parallel_fanin_margin_pct",
        (
            "Recupera in parallelo con analyze_sales_data sia il ricavo totale "
            "(revenue_eur) per categoria sia il margine totale (margin_eur) per "
            "categoria su tutto il dataset. Poi passa i ricavi in `data` e i margini "
            "in `extra_data` a run_python_code per calcolare la percentuale di "
            "margine (margin / revenue * 100, arrotondata a 2 decimali) per ogni "
            "categoria e individuare la categoria più profittevole."
        ),
    ),
    (
        "p04_parallel_fanin_margin_pct",
        (
            "Restituiscimi un grafico contenente tutti i ricavi di questo mese per ciascuna zona, con un grafico a barre colorato per zona. "
        ),
    ),
)


async def run_demo(model: str, reasoning: bool) -> None:
    registry = build_registry()
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
        executor=Executor(registry=registry, max_concurrency=8, task_timeout_s=3.0),
        joiner=Joiner(llm=joiner_llm),
        max_compile_retries=2,
        max_replans=2,
    )

    print(
        f"=== LLM Compiler v3 Simulation: analyze_sales_data + run_python_code "
        f"| {model} (reasoning={reasoning}) ===\n"
    )
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
            print(f"Piano ({len(plan.tasks)} task):")
            for t in plan.tasks:
                if t.tool_name == "run_python_code":
                    bindings_str = {
                        k: f"{v.task_id}.{'.'.join(v.output_path)}"
                        for k, v in t.bindings.items()
                    }
                    print(f"  - {t.task_id}: run_python_code(bindings={bindings_str})")
                    code_preview = str(t.static_inputs.get("code", "")).strip()
                    for line in code_preview.splitlines():
                        print(f"      | {line}")
                else:
                    print(f"  - {t.task_id}: {t.tool_name}({t.static_inputs})")
        if dag is not None:
            print(f"DAG: {len(dag.tasks)} nodi, {len(dag.levels)} livelli")
        if exec_res is not None:
            print(
                f"Executor: {exec_res.total_duration_ms:.1f} ms "
                f"(effective_parallelism={exec_res.effective_parallelism:.2f}x)"
            )
            for tid, res in exec_res.results.items():
                if res.status == "success":
                    print(f"  [OK]  {tid} ({res.tool_name}): {res.output}")
                else:
                    print(f"  [ERR] {tid} ({res.tool_name}): [{res.error_stage}] {res.error}")

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
