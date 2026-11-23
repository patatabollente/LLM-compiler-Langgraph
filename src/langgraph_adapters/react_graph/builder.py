"""LangGraph StateGraph builder for Parallel Tool-Calling ReAct."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from llm_compiler.compiler.registry import ToolRegistry
from llm_compiler.state import WorkflowStatus

from .instrumentation import extract_message_token_usage
from .state import ReActError, ReActMetrics, ReActState, update_react_metrics
from .telemetry import TokenUsage
from .tools import registry_to_langchain_tools

if TYPE_CHECKING:
    from .instrumentation import ReActRunInstrumentation

DEFAULT_REACT_SYSTEM_PROMPT = """
You are a helpful, accurate assistant with access to tools.

Follow these rules strictly:
1. Language: Always answer in the exact same language as the user's request.
2. Tool use: Always use the provided tools to retrieve data, run analyses, or execute calculations. Never invent data or guess missing values from memory.
3. Parallelism: When multiple tool calls are independent, emit them together in the same turn.
4. Recovery: If a tool returns a recoverable error (such as a temporary outage with a backup tool available, a spelling hint, or a fixable code error), retry with the fallback tool or corrected arguments.
5. Transparency: If an error is unrecoverable (e.g., missing entity with no alternative tool), stop retrying, report what succeeded, and clearly explain what could not be retrieved.
""".strip()


def _classify_react_status(metrics: ReActMetrics) -> WorkflowStatus:
    """Deterministically classify completion from observed tool call outcomes.

    If the latest tool batch succeeded with zero errors (even after recovering
    from an earlier turn's error), status is 'complete'. Otherwise, status is
    'partial' if at least one tool call succeeded overall, else 'failed'.
    """
    if metrics.tool_calls == 0 or metrics.last_tool_turn_errors == 0:
        return "complete"
    if metrics.tool_errors < metrics.tool_calls:
        return "partial"
    return "failed"


def _extract_answer_text(msg: AIMessage) -> str:
    content = msg.content
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def _chunk_has_token(chunk: Any) -> bool:
    content = getattr(chunk, "content", None)
    if isinstance(content, str) and content:
        return True
    if isinstance(content, list) and content:
        return True
    return bool(
        getattr(chunk, "tool_calls", None) or getattr(chunk, "tool_call_chunks", None)
    )


def _merge_stream_chunks(chunks: list[Any]) -> Any:
    if not chunks:
        return AIMessage(content="")
    merged = chunks[0]
    for chunk in chunks[1:]:
        merged = merged + chunk
    return merged


def _make_agent_router(
    max_iterations: int,
) -> Callable[[ReActState], Literal["tools", "__end__"]]:
    def _route_after_agent(state: ReActState) -> Literal["tools", "__end__"]:
        if state.get("error") is not None:
            return END
        messages = state.get("messages") or []
        if not messages:
            return END
        last = messages[-1]
        metrics = state.get("metrics") or ReActMetrics()
        if isinstance(last, AIMessage) and last.tool_calls:
            if metrics.turns >= max_iterations:
                return END
            return "tools"
        return END

    return _route_after_agent


def build_react_graph(
    *,
    registry: ToolRegistry,
    llm: BaseChatModel,
    system_prompt: str | None = DEFAULT_REACT_SYSTEM_PROMPT,
    max_iterations: int = 10,
    task_timeout_s: float | None = None,
    allow_parallel_tool_calls: bool = True,
    stream_llm: bool = False,
    instrumentation: ReActRunInstrumentation | None = None,
) -> CompiledStateGraph[ReActState, Any, ReActState, ReActState]:
    """Build and compile a modular LangGraph ReAct agent backed by ToolRegistry.

    Uses the same `{"request": "..."}` input schema as `build_compiler_graph(...)`,
    shares the same `ToolRegistry` (preserving runtime Pydantic I/O validation and
    optional per-tool timeouts), and tracks deterministic `ReActMetrics`.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1.")

    lc_tools = registry_to_langchain_tools(
        registry,
        task_timeout_s=task_timeout_s,
        instrumentation=instrumentation,
    )
    llm_with_tools = llm.bind_tools(lc_tools)
    tool_node = ToolNode(lc_tools, handle_tool_errors=True)

    async def agent_node(state: ReActState) -> dict[str, object]:
        current_metrics = state.get("metrics") or ReActMetrics()
        request = state["request"].strip()
        if not request:
            return {
                "final_answer": None,
                "status": "failed",
                "error": ReActError(
                    stage="agent",
                    error_type="ValueError",
                    message="The user request cannot be empty.",
                ),
                "metrics": current_metrics,
            }

        existing_messages = list(state.get("messages") or [])
        seed_messages: list[AnyMessage] = []
        if not existing_messages:
            if system_prompt:
                seed_messages.append(SystemMessage(content=system_prompt))
            seed_messages.append(HumanMessage(content=request))
        prompt_messages = [*existing_messages, *seed_messages]

        turn_index = instrumentation.start_turn() if instrumentation is not None else 0
        t0 = time.perf_counter()
        try:
            if stream_llm:
                chunks: list[Any] = []
                async for chunk in llm_with_tools.astream(prompt_messages):
                    if instrumentation is not None and _chunk_has_token(chunk):
                        instrumentation.record_first_token()
                    chunks.append(chunk)
                response = _merge_stream_chunks(chunks)
            else:
                response = await llm_with_tools.ainvoke(prompt_messages)
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if instrumentation is not None:
                instrumentation.finish_turn(
                    turn_index=turn_index,
                    token_usage=TokenUsage(source="unavailable"),
                    error=f"{type(exc).__name__}: {exc}",
                )
            return {
                "messages": seed_messages,
                "final_answer": None,
                "status": "failed",
                "error": ReActError(
                    stage="agent",
                    error_type=type(exc).__name__,
                    message=str(exc) or type(exc).__name__,
                ),
                "metrics": update_react_metrics(
                    state,
                    llm_latency_ms=current_metrics.llm_latency_ms + elapsed_ms,
                    llm_calls=current_metrics.llm_calls + 1,
                ),
            }

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        token_usage = extract_message_token_usage(response)
        in_tok = token_usage.input_tokens or 0
        out_tok = token_usage.output_tokens or 0
        next_turns = current_metrics.turns + 1
        updated_metrics = update_react_metrics(
            state,
            llm_latency_ms=current_metrics.llm_latency_ms + elapsed_ms,
            llm_calls=current_metrics.llm_calls + 1,
            turns=next_turns,
            input_tokens=current_metrics.input_tokens + in_tok,
            output_tokens=current_metrics.output_tokens + out_tok,
        )

        emitted_messages: list[AnyMessage] = [*seed_messages, response]
        raw_tool_calls = (
            list(response.tool_calls)
            if isinstance(response, AIMessage) and response.tool_calls
            else []
        )
        answer_text = (
            _extract_answer_text(response)
            if isinstance(response, AIMessage)
            else str(response).strip()
        )

        if instrumentation is not None:
            instrumentation.finish_turn(
                turn_index=turn_index,
                token_usage=token_usage,
                tool_calls=raw_tool_calls,
                text=answer_text or None,
            )

        if raw_tool_calls:
            if next_turns >= max_iterations:
                return {
                    "messages": emitted_messages,
                    "final_answer": None,
                    "status": "failed",
                    "error": ReActError(
                        stage="agent",
                        error_type="MaxIterationsError",
                        message=(
                            f"ReAct agent reached max_iterations={max_iterations} "
                            "while still requesting tool calls."
                        ),
                    ),
                    "metrics": updated_metrics,
                }
            return {
                "messages": emitted_messages,
                "error": None,
                "metrics": updated_metrics,
            }

        return {
            "messages": emitted_messages,
            "final_answer": answer_text or None,
            "status": _classify_react_status(updated_metrics),
            "error": None,
            "metrics": updated_metrics,
        }

    async def tools_node(state: ReActState) -> dict[str, object]:
        current_metrics = state.get("metrics") or ReActMetrics()
        t0 = time.perf_counter()
        try:
            if allow_parallel_tool_calls:
                result = await tool_node.ainvoke(state)
                tool_messages = list(result.get("messages", []))
            else:
                # Enforce strict sequential execution of tool calls within a turn.
                messages = state.get("messages") or []
                last_msg = messages[-1] if messages else None
                tool_messages = []
                if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        single_state = {
                            "messages": [AIMessage(content="", tool_calls=[tc])]
                        }
                        step_res = await tool_node.ainvoke(single_state)
                        tool_messages.extend(step_res.get("messages", []))
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "final_answer": None,
                "status": "failed",
                "error": ReActError(
                    stage="tools",
                    error_type=type(exc).__name__,
                    message=str(exc) or type(exc).__name__,
                ),
                "metrics": update_react_metrics(
                    state,
                    tool_latency_ms=current_metrics.tool_latency_ms + elapsed_ms,
                ),
            }

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        err_count = 0
        for msg in tool_messages:
            if isinstance(msg, ToolMessage):
                is_err = getattr(msg, "status", None) == "error"
                if is_err:
                    err_count += 1
                if instrumentation is not None:
                    content_str = (
                        msg.content if isinstance(msg.content, str) else str(msg.content)
                    )
                    instrumentation.record_tool_message(
                        msg.tool_call_id,
                        content=content_str,
                        is_error=is_err,
                    )

        return {
            "messages": tool_messages,
            "error": None,
            "metrics": update_react_metrics(
                state,
                tool_latency_ms=current_metrics.tool_latency_ms + elapsed_ms,
                tool_calls=current_metrics.tool_calls + len(tool_messages),
                tool_errors=current_metrics.tool_errors + err_count,
                last_tool_turn_errors=err_count,
            ),
        }

    def _route_after_tools(state: ReActState) -> Literal["agent", "__end__"]:
        return END if state.get("error") is not None else "agent"

    builder: StateGraph[ReActState, Any, ReActState, ReActState] = StateGraph(ReActState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        _make_agent_router(max_iterations),
        {"tools": "tools", END: END},
    )
    builder.add_conditional_edges(
        "tools",
        _route_after_tools,
        {"agent": "agent", END: END},
    )

    return builder.compile()
