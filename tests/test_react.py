import asyncio
import json
from typing import Any, Literal

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel

from langgraph_adapters.react_graph import (
    ReActError,
    ReActMetrics,
    build_react_graph,
    registry_to_langchain_tools,
)
from llm_compiler.compiler import ToolRegistry, ToolSpec


class FakeToolBoundRunnable:
    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = list(responses)
        self.received_messages: list[list[Any]] = []

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.received_messages.append(list(messages))
        idx = min(len(self.received_messages) - 1, len(self._responses) - 1)
        return self._responses[idx]


class FakeReActChatModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self.runnable = FakeToolBoundRunnable(responses)
        self.bound_tools: list[Any] = []

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> FakeToolBoundRunnable:
        self.bound_tools = list(tools)
        return self.runnable


class LookupInput(BaseModel):
    item: str


class LookupOutput(BaseModel):
    item: str
    value: float


class MathInput(BaseModel):
    op: Literal["add", "subtract"]
    a: float
    b: float


class MathOutput(BaseModel):
    result: float


def _build_registry(*, slow_item: str | None = None, fail_item: str | None = None) -> ToolRegistry:
    async def fetch_item(item: str) -> dict[str, Any]:
        if slow_item and item == slow_item:
            await asyncio.sleep(0.5)
        if fail_item and item == fail_item:
            raise ValueError(f"Item '{item}' not found.")
        prices = {"A": 100.0, "B": 40.0}
        return {"item": item, "value": prices[item]}

    def compute(op: Literal["add", "subtract"], a: float, b: float) -> dict[str, float]:
        return {"result": a + b if op == "add" else a - b}

    return ToolRegistry(
        [
            ToolSpec(
                name="fetch_item",
                description="Fetch numeric value for an item.",
                func=fetch_item,
                input_schema=LookupInput,
                output_schema=LookupOutput,
            ),
            ToolSpec(
                name="compute",
                description="Add or subtract two numbers.",
                func=compute,
                input_schema=MathInput,
                output_schema=MathOutput,
            ),
        ]
    )


def test_registry_to_langchain_tools_supports_sync_async_and_output_validation():
    def bad_output_tool(item: str) -> dict[str, Any]:
        return {"wrong_key": item}

    reg = ToolRegistry(
        [
            ToolSpec(
                name="bad_tool",
                description="Returns invalid schema",
                func=bad_output_tool,
                input_schema=LookupInput,
                output_schema=LookupOutput,
            )
        ]
    )
    lc_tools = registry_to_langchain_tools(reg)
    assert len(lc_tools) == 1

    with pytest.raises(Exception):  # noqa: B017
        asyncio.run(lc_tools[0].ainvoke({"item": "X"}))

    with pytest.raises(ValueError, match="task_timeout_s"):
        registry_to_langchain_tools(reg, task_timeout_s=0)


def test_react_graph_parallel_tool_calls_and_multi_turn_complete():
    registry = _build_registry()
    fake_llm = FakeReActChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "c1", "name": "fetch_item", "args": {"item": "A"}},
                    {"id": "c2", "name": "fetch_item", "args": {"item": "B"}},
                ],
                usage_metadata={"input_tokens": 120, "output_tokens": 45, "total_tokens": 165},
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "c3",
                        "name": "compute",
                        "args": {"op": "subtract", "a": 100.0, "b": 40.0},
                    }
                ],
                usage_metadata={"input_tokens": 200, "output_tokens": 30, "total_tokens": 230},
            ),
            AIMessage(
                content="La differenza tra A e B è 60.0.",
                usage_metadata={"input_tokens": 250, "output_tokens": 20, "total_tokens": 270},
            ),
        ]
    )

    graph = build_react_graph(
        registry=registry,
        llm=fake_llm,  # type: ignore[arg-type]
        max_iterations=6,
    )

    state = asyncio.run(graph.ainvoke({"request": "Calcola A meno B"}))

    assert state["status"] == "complete"
    assert state["error"] is None
    assert state["final_answer"] == "La differenza tra A e B è 60.0."

    metrics = state["metrics"]
    assert isinstance(metrics, ReActMetrics)
    assert metrics.llm_calls == 3
    assert metrics.turns == 3
    assert metrics.tool_calls == 3
    assert metrics.tool_errors == 0
    assert metrics.input_tokens == 570
    assert metrics.output_tokens == 95
    assert metrics.total_tokens == 665
    assert metrics.component_latency_ms > 0

    tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 3
    assert json.loads(str(tool_msgs[-1].content)) == {"result": 60.0}


def test_react_graph_handles_tool_error_and_marks_partial():
    registry = _build_registry(fail_item="Z")
    fake_llm = FakeReActChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "c1", "name": "fetch_item", "args": {"item": "A"}},
                    {"id": "c2", "name": "fetch_item", "args": {"item": "Z"}},
                ],
            ),
            AIMessage(content="A vale 100.0, mentre Z non esiste."),
        ]
    )

    graph = build_react_graph(registry=registry, llm=fake_llm)  # type: ignore[arg-type]
    state = asyncio.run(graph.ainvoke({"request": "Dammi A e Z"}))

    assert state["status"] == "partial"
    assert state["error"] is None
    assert state["final_answer"] == "A vale 100.0, mentre Z non esiste."
    assert state["metrics"].tool_calls == 2
    assert state["metrics"].tool_errors == 1
    assert state["metrics"].last_tool_turn_errors == 1


def test_react_graph_marks_complete_after_recovering_from_earlier_tool_error():
    registry = _build_registry(fail_item="Z")
    fake_llm = FakeReActChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"id": "c1", "name": "fetch_item", "args": {"item": "Z"}}],
            ),
            AIMessage(
                content="",
                tool_calls=[{"id": "c2", "name": "fetch_item", "args": {"item": "A"}}],
            ),
            AIMessage(content="Recuperato con successo: A vale 100.0."),
        ]
    )

    graph = build_react_graph(registry=registry, llm=fake_llm)  # type: ignore[arg-type]
    state = asyncio.run(graph.ainvoke({"request": "Dammi il valore valido"}))

    assert state["status"] == "complete"
    assert state["error"] is None
    assert state["final_answer"] == "Recuperato con successo: A vale 100.0."
    assert state["metrics"].tool_calls == 2
    assert state["metrics"].tool_errors == 1
    assert state["metrics"].last_tool_turn_errors == 0


def test_react_graph_enforces_per_tool_timeout():
    registry = _build_registry(slow_item="A")
    fake_llm = FakeReActChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"id": "c1", "name": "fetch_item", "args": {"item": "A"}}],
            ),
            AIMessage(content="Il tool è andato in timeout."),
        ]
    )

    graph = build_react_graph(
        registry=registry,
        llm=fake_llm,  # type: ignore[arg-type]
        task_timeout_s=0.05,
    )
    state = asyncio.run(graph.ainvoke({"request": "Dammi A"}))

    assert state["status"] == "failed"
    assert state["error"] is None
    assert state["metrics"].tool_calls == 1
    assert state["metrics"].tool_errors == 1
    tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert "timed out" in str(tool_msgs[0].content)


def test_react_graph_stops_cleanly_at_max_iterations():
    registry = _build_registry()
    looping_msg = AIMessage(
        content="",
        tool_calls=[{"id": "c_loop", "name": "fetch_item", "args": {"item": "A"}}],
    )
    fake_llm = FakeReActChatModel([looping_msg])

    graph = build_react_graph(
        registry=registry,
        llm=fake_llm,  # type: ignore[arg-type]
        max_iterations=2,
    )
    state = asyncio.run(graph.ainvoke({"request": "Loop"}))

    assert state["status"] == "failed"
    assert state["final_answer"] is None
    err = state["error"]
    assert isinstance(err, ReActError)
    assert err.stage == "agent"
    assert err.error_type == "MaxIterationsError"
    assert state["metrics"].turns == 2
    assert state["metrics"].tool_calls == 1


def test_react_graph_rejects_empty_request_and_invalid_config():
    registry = _build_registry()
    fake_llm = FakeReActChatModel([AIMessage(content="unused")])

    with pytest.raises(ValueError, match="max_iterations"):
        build_react_graph(registry=registry, llm=fake_llm, max_iterations=0)  # type: ignore[arg-type]

    graph = build_react_graph(registry=registry, llm=fake_llm)  # type: ignore[arg-type]
    state = asyncio.run(graph.ainvoke({"request": "   "}))
    assert state["status"] == "failed"
    assert state["error"] is not None
    assert state["error"].error_type == "ValueError"
    assert state["metrics"].llm_calls == 0


def test_react_graph_streaming_order_and_mermaid():
    registry = _build_registry()
    fake_llm = FakeReActChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"id": "c1", "name": "fetch_item", "args": {"item": "A"}}],
            ),
            AIMessage(content="Fatto: 100.0"),
        ]
    )
    graph = build_react_graph(registry=registry, llm=fake_llm)  # type: ignore[arg-type]

    async def collect() -> list[str]:
        visited: list[str] = []
        async for chunk in graph.astream({"request": "Dammi A"}):
            visited.extend(chunk.keys())
        return visited

    assert asyncio.run(collect()) == ["agent", "tools", "agent"]
    mermaid = graph.get_graph().draw_mermaid()
    assert "agent -.-> tools" in mermaid
    assert "tools -.-> agent" in mermaid
