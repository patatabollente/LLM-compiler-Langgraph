"""Live turn-by-turn demo of the modular LangGraph ReAct package (`langgraph_llm_compiler.react`).

Streams every `agent` and `tools` step via `graph.astream(...)` so you can inspect
exactly what tool calls the LLM emitted, what code it generated, and what outputs
or errors came back on every single iteration.

Usage:
    uv run python experiments/demo_react_sales_python.py
    uv run python experiments/demo_react_sales_python.py --only p03
    uv run python experiments/demo_react_sales_python.py --reasoning
"""

import argparse
import asyncio
import time
from typing import Any

from demo_python_code_v3 import QUERIES, build_registry
from langchain_core.messages import AIMessage, ToolMessage

from langgraph_adapters.react_graph import build_react_graph
from llm_compiler.llm import create_ollama_chat_model


def _print_tool_call(tc: dict[str, Any]) -> None:
    name = tc.get("name", "?")
    args = tc.get("args", {})
    if name == "run_python_code" and isinstance(args, dict):
        other_args = {k: v for k, v in args.items() if k != "code"}
        print(f"    -> call {name}({other_args})")
        code = str(args.get("code", "")).strip()
        for line in code.splitlines():
            print(f"         | {line}")
    else:
        print(f"    -> call {name}({args})")


async def run_demo(model: str, reasoning: bool, only: str | None) -> None:
    registry = build_registry()
    llm = create_ollama_chat_model(
        model=model, temperature=0.0, reasoning=reasoning, num_predict=2048
    )
    graph = build_react_graph(
        registry=registry,
        llm=llm,
        max_iterations=8,
        task_timeout_s=3.0,
    )

    selected = [
        (qid, req)
        for qid, req in QUERIES
        if only is None or only.lower() in qid.lower()
    ]
    if not selected:
        print(f"No query matched --only '{only}'. Available: {[q for q, _ in QUERIES]}")
        return

    print(
        f"=== LangGraph ReAct Turn-by-Turn Trace | {model} (reasoning={reasoning}) ===\n"
    )
    for qid, request in selected:
        print(f"================ {qid} ================")
        print(f"Richiesta: {request}\n")

        t0 = time.perf_counter()
        turn = 0
        final_state: dict[str, Any] = {}

        async for chunk in graph.astream({"request": request}):
            for node_name, update in chunk.items():
                final_state.update(update)
                msgs = update.get("messages") or []

                if node_name == "agent":
                    turn += 1
                    ai_msgs = [m for m in msgs if isinstance(m, AIMessage)]
                    ai_msg = ai_msgs[-1] if ai_msgs else None
                    err = update.get("error")
                    if ai_msg is not None and ai_msg.tool_calls:
                        print(f"  [Iter {turn} · AGENT] emette {len(ai_msg.tool_calls)} tool call:")
                        for tc in ai_msg.tool_calls:
                            _print_tool_call(tc)
                        if err is not None:
                            print(f"    !! STOP: {err.error_type}: {err.message}")
                    elif err is not None:
                        print(f"  [Iter {turn} · AGENT] ERRORE: {err.error_type}: {err.message}")
                    else:
                        print(f"  [Iter {turn} · AGENT] risposta finale pronta.")

                elif node_name == "tools":
                    tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
                    print(f"  [Iter {turn} · TOOLS] {len(tool_msgs)} risultati:")
                    for tm in tool_msgs:
                        is_err = getattr(tm, "status", None) == "error"
                        badge = "ERR" if is_err else "OK "
                        print(f"    [{badge}] {tm.name}: {tm.content}")

        wall_s = time.perf_counter() - t0
        m = final_state["metrics"]
        print(
            f"\nRiepilogo: status={final_state.get('status')} | Wall={wall_s:.2f}s | "
            f"LLM={m.llm_calls} | Tools={m.tool_calls} (errors={m.tool_errors}) | "
            f"Tokens={m.total_tokens} (in={m.input_tokens}, out={m.output_tokens})"
        )
        if final_state.get("error"):
            print(f"Errore: {final_state['error']}")
        print(f"Risposta ReAct:\n{final_state.get('final_answer')}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--reasoning", action="store_true")
    parser.add_argument("--only", default=None, help="Filter query ID substring (e.g. p03).")
    args = parser.parse_args()
    asyncio.run(run_demo(model=args.model, reasoning=args.reasoning, only=args.only))
