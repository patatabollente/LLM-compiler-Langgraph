import time
from collections.abc import Callable
from typing import Any, Literal

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

from langgraph_llm_compiler.compiler.registry import ToolRegistry
from langgraph_llm_compiler.graph.state import WorkflowStatus

from .state import ReActError, ReActMetrics, ReActState, update_react_metrics
from .tools import registry_to_langchain_tools

DEFAULT_REACT_SYSTEM_PROMPT = """
You are a helpful, accurate assistant with access to tools.

Follow these rules strictly:
1. Language: Always answer in the exact same language as the user's request.
2. Tool use: Always use the provided tools to retrieve data, run analyses, or execute calculations. Never invent data or guess missing values from memory.
3. Parallelism: When multiple tool calls are independent, emit them together in the same turn.
4. Recovery: If a tool returns a recoverable error (such as a temporary outage with a backup tool available, a spelling hint, or a fixable code error), retry with the fallback tool or corrected arguments.
5. Transparency: If an error is unrecoverable (e.g., missing entity with no alternative tool), stop retrying, report what succeeded, and clearly explain what could not be retrieved.
""".strip()


def _extract_token_counts(msg: Any) -> tuple[int, int]:
    usage = getattr(msg, "usage_metadata", None)
    if isinstance(usage, dict):
        return (
            int(usage.get("input_tokens", 0) or 0),
            int(usage.get("output_tokens", 0) or 0),
        )
    meta = getattr(msg, "response_metadata", None)
    if isinstance(meta, dict):
        return (
            int(meta.get("prompt_eval_count", 0) or 0),
            int(meta.get("eval_count", 0) or 0),
        )
    return 0, 0


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
    system_prompt: str = DEFAULT_REACT_SYSTEM_PROMPT,
    max_iterations: int = 10,
    task_timeout_s: float | None = None,
) -> CompiledStateGraph[ReActState, Any, ReActState, ReActState]:
    """Build and compile a modular LangGraph ReAct agent backed by ToolRegistry.

    Uses the same `{"request": "..."}` input schema as `build_compiler_graph(...)`,
    shares the same `ToolRegistry` (preserving runtime Pydantic I/O validation and
    optional per-tool timeouts), and tracks deterministic `ReActMetrics`.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1.")

    lc_tools = registry_to_langchain_tools(registry, task_timeout_s=task_timeout_s)
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
            seed_messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=request),
            ]
        prompt_messages = [*existing_messages, *seed_messages]

        t0 = time.perf_counter()
        try:
            response = await llm_with_tools.ainvoke(prompt_messages)
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
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
        in_tok, out_tok = _extract_token_counts(response)
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

        if isinstance(response, AIMessage) and response.tool_calls:
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

        answer_text = (
            _extract_answer_text(response)
            if isinstance(response, AIMessage)
            else str(response).strip()
        )
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
            result = await tool_node.ainvoke(state)
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
        tool_messages = list(result.get("messages", []))
        err_count = sum(
            1
            for msg in tool_messages
            if isinstance(msg, ToolMessage) and getattr(msg, "status", None) == "error"
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
