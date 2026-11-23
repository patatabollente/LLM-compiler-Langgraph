import asyncio
import json
from typing import Any, Literal

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from pydantic import BaseModel

from langgraph_adapters.react_graph import (
    InferenceParams,
    OpenAICompatibleConfig,
    ParallelReactLangGraphRunner,
    ParallelRunResult,
    ReActError,
    ReActMetrics,
    ToolBinding,
    build_react_graph,
    normalize_to_registry,
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

    async def astream(self, messages: list[Any]):
        self.received_messages.append(list(messages))
        idx = min(len(self.received_messages) - 1, len(self._responses) - 1)
        resp = self._responses[idx]
        if resp.tool_calls:
            yield resp
            return
        text = str(resp.content)
        mid = max(1, len(text) // 2)
        yield AIMessageChunk(content=text[:mid])
        await asyncio.sleep(0.01)
        yield AIMessageChunk(
            content=text[mid:],
            usage_metadata=resp.usage_metadata,
            response_metadata=resp.response_metadata,
        )


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


def test_parallel_react_runner_captures_waves_speedup_ttft_and_cached_tokens():
    async def slow_lookup(key: str) -> dict[str, Any]:
        await asyncio.sleep(0.06)
        return {"key": key, "val": 10 if key == "x" else 20}

    fake_llm = FakeReActChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "t1", "name": "slow_lookup", "args": {"key": "x"}},
                    {"id": "t2", "name": "slow_lookup", "args": {"key": "y"}},
                ],
                usage_metadata={
                    "input_tokens": 100,
                    "output_tokens": 40,
                    "total_tokens": 140,
                    "input_token_details": {"cache_read": 30},
                },
            ),
            AIMessage(
                content="Risultato combinato: 30.",
                usage_metadata={
                    "input_tokens": 180,
                    "output_tokens": 25,
                    "total_tokens": 205,
                    "input_token_details": {"cache_read": 90},
                },
            ),
        ]
    )

    runner = ParallelReactLangGraphRunner(
        tools=[slow_lookup],  # plain Python callable!
        llm=fake_llm,  # type: ignore[arg-type]
        max_turns=5,
        allow_parallel_tool_calls=True,
    )
    res = runner.run_sync("Somma x e y")

    assert isinstance(res, ParallelRunResult)
    assert res.status == "completed"
    assert res.workflow_status == "complete"
    assert res.termination_reason == "completed"
    assert res.timed_out is False
    assert res.answer == "Risultato combinato: 30."
    assert res.llm_round_trips == 2
    assert res.tool_call_count == 2
    assert res.time_to_first_token_ms is not None and res.time_to_first_token_ms >= 0.0

    # Token usage + cached prompt tokens breakdown
    assert res.tokens.source == "provider"
    assert res.tokens.input_tokens == 280
    assert res.tokens.output_tokens == 65
    assert res.tokens.total_tokens == 345
    assert res.tokens.cached_input_tokens == 120

    # Per-turn telemetry
    assert len(res.turns) == 2
    assert res.turns[0].tool_call_ids == ["t1", "t2"]
    assert res.turns[0].cached_input_tokens == 30
    assert res.turns[1].cached_input_tokens == 90

    # Per-wave parallel speedup telemetry (2 x 60ms in parallel -> wall ~60ms, speedup > 1.4)
    assert len(res.waves) == 1
    wave = res.waves[0]
    assert wave.tool_call_count == 2
    assert wave.tool_names == ["slow_lookup", "slow_lookup"]
    assert wave.speedup is not None and wave.speedup > 1.4
    assert wave.parallel_efficiency is not None and 0.7 < wave.parallel_efficiency <= 1.0
    assert res.overhead_ms >= 0.0


def test_parallel_react_runner_sequential_mode_disables_parallel_wave_overlap():
    async def delayed(tag: str) -> str:
        await asyncio.sleep(0.05)
        return f"done:{tag}"

    fake_llm = FakeReActChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "s1", "name": "delayed", "args": {"tag": "a"}},
                    {"id": "s2", "name": "delayed", "args": {"tag": "b"}},
                ],
            ),
            AIMessage(content="Fatto in sequenza."),
        ]
    )

    runner = ParallelReactLangGraphRunner(
        tools=[delayed],
        llm=fake_llm,  # type: ignore[arg-type]
        allow_parallel_tool_calls=False,
    )
    res = runner.run_sync("Esegui a e b")

    assert res.status == "completed"
    assert len(res.waves) == 1
    # Executed sequentially -> wall_ms >= ~100ms, speedup close to 1.0
    assert res.waves[0].wall_ms >= 90.0
    assert res.waves[0].speedup is not None and res.waves[0].speedup <= 1.15


def test_parallel_react_runner_supports_tool_binding_json_schema_and_dict_tools():
    def custom_add(a: int, b: int = 1) -> int:
        return a + b

    binding = ToolBinding(
        function=custom_add,
        name="add_nums",
        description="Add two integers.",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First integer"},
                "b": {"type": "integer", "default": 1},
            },
            "required": ["a"],
            "additionalProperties": False,
        },
    )
    dict_tool = {
        "name": "echo_msg",
        "description": "Echo text back.",
        "func": lambda text: f"echo:{text}",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    }

    reg = normalize_to_registry([binding, dict_tool])
    assert set(reg) == {"add_nums", "echo_msg"}

    fake_llm = FakeReActChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "b1", "name": "add_nums", "args": {"a": 5, "b": 7}},
                    {"id": "b2", "name": "echo_msg", "args": {"text": "ciao"}},
                ],
            ),
            AIMessage(content="12 e echo:ciao"),
        ]
    )
    runner = ParallelReactLangGraphRunner(
        tools=[binding, dict_tool],
        llm=fake_llm,  # type: ignore[arg-type]
    )
    res = runner.run_sync("Somma 5+7 ed echo ciao")
    assert res.status == "completed"
    assert [tc.result for tc in res.tool_calls] == ["12", "echo:ciao"]


def test_parallel_react_runner_handles_run_timeout_and_max_turns_exceeded():
    async def hang(x: str) -> str:
        await asyncio.sleep(1.0)
        return x

    fake_timeout_llm = FakeReActChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"id": "h1", "name": "hang", "args": {"x": "go"}}],
            ),
            AIMessage(content="never"),
        ]
    )
    timeout_runner = ParallelReactLangGraphRunner(
        tools=[hang],
        llm=fake_timeout_llm,  # type: ignore[arg-type]
        timeout_seconds=0.05,
    )
    res_timeout = timeout_runner.run_sync("Aspetta")
    assert res_timeout.status == "error"
    assert res_timeout.timed_out is True
    assert res_timeout.termination_reason == "timeout"

    fake_loop_llm = FakeReActChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"id": "l1", "name": "fetch_item", "args": {"item": "A"}}],
            )
        ]
    )
    loop_runner = ParallelReactLangGraphRunner(
        tools=_build_registry(),
        llm=fake_loop_llm,  # type: ignore[arg-type]
        max_turns=2,
    )
    res_loop = loop_runner.run_sync("Loop")
    assert res_loop.status == "max_turns_exceeded"
    assert res_loop.termination_reason == "request_limit"
    assert res_loop.workflow_status == "failed"


def test_openai_compatible_config_and_runner_validation():
    cfg = OpenAICompatibleConfig(
        model="qwen3:8b",
        base_url="http://localhost:11434/v1",
        provider="ollama",
    )
    params = InferenceParams(temperature=0.0, max_tokens=512, reasoning=False)
    runner = ParallelReactLangGraphRunner(
        tools=[],
        model_config=cfg,
        inference_params=params,
    )
    assert runner.model_config is not None
    assert runner.model_config.model == "qwen3:8b"
    assert runner.inference_params.max_tokens == 512

    with pytest.raises(ValueError, match="llm=.*model_config="):
        ParallelReactLangGraphRunner(tools=[])

    with pytest.raises(ValueError, match="max_turns"):
        ParallelReactLangGraphRunner(
            tools=[],
            model_config=cfg,
            max_turns=0,
        )

    with pytest.raises(ValueError, match="timeout_seconds"):
        ParallelReactLangGraphRunner(
            tools=[],
            model_config=cfg,
            timeout_seconds=-1.0,
        )
