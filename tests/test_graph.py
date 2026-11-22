import asyncio
from typing import Any, Literal

import pytest
from pydantic import BaseModel

from langgraph_adapters.compiler_graph import (
    WorkflowError,
    WorkflowMetrics,
    build_compiler_graph,
)
from llm_compiler.compiler import PlanCompiler, ToolRegistry, ToolSpec
from llm_compiler.executor import Executor
from llm_compiler.joiner import Joiner, JoinerDecisionOutput, JoinerOutput
from llm_compiler.planner import (
    ExecutionPlan,
    InputBinding,
    Planner,
    TaskSpec,
)


class FakeStructuredRunnable:
    def __init__(self, response: dict[str, Any]) -> None:
        self._response = response
        self.call_count: int = 0

    def invoke(self, messages: list[Any]) -> dict[str, Any]:
        self.call_count += 1
        return self._response

    async def ainvoke(self, messages: list[Any]) -> dict[str, Any]:
        self.call_count += 1
        return self._response


class FakeChatModel:
    def __init__(self, response: dict[str, Any]) -> None:
        self.runnable = FakeStructuredRunnable(response)

    def with_structured_output(self, **kwargs: Any) -> FakeStructuredRunnable:
        return self.runnable


class PopulationInput(BaseModel):
    country: str


class PopulationOutput(BaseModel):
    country: str
    population: int


class CalculatorInput(BaseModel):
    operation: Literal["add", "subtract", "multiply", "divide"]
    a: float
    b: float


class CalculatorOutput(BaseModel):
    result: float


def _build_registry(*, fail_country: str | None = None) -> ToolRegistry:
    async def lookup(country: str) -> dict[str, str | int]:
        if fail_country and country.lower() == fail_country.lower():
            raise RuntimeError(f"Service unavailable for {country}")
        pops = {"italy": 58_990_000, "france": 68_400_000}
        return {"country": country, "population": pops[country.lower()]}

    def calc(
        operation: Literal["add", "subtract", "multiply", "divide"],
        a: float,
        b: float,
    ) -> dict[str, float]:
        if operation == "subtract":
            return {"result": a - b}
        return {"result": a + b}

    return ToolRegistry(
        [
            ToolSpec(
                name="population_lookup",
                description="Get country population",
                func=lookup,
                input_schema=PopulationInput,
                output_schema=PopulationOutput,
            ),
            ToolSpec(
                name="calculator",
                description="Binary arithmetic",
                func=calc,
                input_schema=CalculatorInput,
                output_schema=CalculatorOutput,
            ),
        ]
    )


def _valid_plan() -> ExecutionPlan:
    return ExecutionPlan(
        objective="Compute Italy minus France population.",
        tasks=[
            TaskSpec(
                task_id="task_1",
                tool_name="population_lookup",
                description="Get Italy population.",
                inputs={"country": "Italy"},
            ),
            TaskSpec(
                task_id="task_2",
                tool_name="population_lookup",
                description="Get France population.",
                inputs={"country": "France"},
            ),
            TaskSpec(
                task_id="task_3",
                tool_name="calculator",
                description="Subtract France from Italy.",
                inputs={
                    "operation": "subtract",
                    "a": InputBinding(task_id="task_1", output_path=["population"]),
                    "b": InputBinding(task_id="task_2", output_path=["population"]),
                },
            ),
        ],
    )


def test_graph_complete_end_to_end():
    registry = _build_registry()
    planner_llm = FakeChatModel(
        {"parsed": _valid_plan(), "parsing_error": None, "raw": "{}"}
    )
    joiner_llm = FakeChatModel(
        {
            "parsed": JoinerOutput(
                answer="La differenza tra Italia e Francia è -9.410.000."
            ),
            "parsing_error": None,
            "raw": "{}",
        }
    )

    graph = build_compiler_graph(
        registry=registry,
        planner=Planner(llm=planner_llm),  # type: ignore[arg-type]
        compiler=PlanCompiler(registry=registry),
        executor=Executor(registry=registry),
        joiner=Joiner(llm=joiner_llm),  # type: ignore[arg-type]
    )

    final_state = asyncio.run(
        graph.ainvoke({"request": "Differenza popolazione Italia meno Francia"})
    )

    assert final_state["status"] == "complete"
    assert final_state["error"] is None
    assert final_state["final_answer"] == "La differenza tra Italia e Francia è -9.410.000."
    assert final_state["plan"] == _valid_plan()
    assert final_state["compiled_dag"] is not None
    assert final_state["execution_result"] is not None
    assert final_state["execution_result"].results["task_3"].output == {
        "result": -9_410_000.0
    }
    assert final_state["joiner_result"] is not None
    assert final_state["joiner_result"].missing_information == []

    metrics = final_state["metrics"]
    assert isinstance(metrics, WorkflowMetrics)
    assert metrics.llm_calls == 2
    assert metrics.compile_attempts == 1
    assert metrics.replan_attempts == 0
    assert metrics.plan_valid is True
    assert metrics.planning_latency_ms is not None
    assert metrics.compilation_latency_ms is not None
    assert metrics.execution_latency_ms is not None
    assert metrics.joining_latency_ms is not None
    assert metrics.component_latency_ms == pytest.approx(
        metrics.planning_latency_ms
        + metrics.compilation_latency_ms
        + metrics.execution_latency_ms
        + metrics.joining_latency_ms
    )


def test_graph_stops_on_planner_error_and_leaves_plan_valid_none():
    registry = _build_registry()
    planner_llm = FakeChatModel(
        {
            "parsed": None,
            "parsing_error": ValueError("malformed json"),
            "raw": "{bad",
        }
    )
    joiner_llm = FakeChatModel(
        {"parsed": JoinerOutput(answer="should not run"), "parsing_error": None, "raw": "{}"}
    )

    graph = build_compiler_graph(
        registry=registry,
        planner=Planner(llm=planner_llm),  # type: ignore[arg-type]
        compiler=PlanCompiler(registry=registry),
        executor=Executor(registry=registry),
        joiner=Joiner(llm=joiner_llm),  # type: ignore[arg-type]
    )

    final_state = asyncio.run(graph.ainvoke({"request": "Compute difference"}))

    assert final_state["status"] == "failed"
    assert final_state["plan"] is None
    assert "compiled_dag" not in final_state
    assert "execution_result" not in final_state
    assert joiner_llm.runnable.call_count == 0

    err = final_state["error"]
    assert isinstance(err, WorkflowError)
    assert err.stage == "planner"
    assert err.error_type == "PlannerOutputError"

    metrics = final_state["metrics"]
    assert metrics.plan_valid is None
    assert metrics.llm_calls == 1
    assert metrics.compile_attempts == 0
    assert metrics.compilation_latency_ms is None


def test_graph_stops_on_empty_request_without_counting_llm_call():
    registry = _build_registry()
    planner_llm = FakeChatModel(
        {"parsed": _valid_plan(), "parsing_error": None, "raw": "{}"}
    )
    joiner_llm = FakeChatModel(
        {"parsed": JoinerOutput(answer="nope"), "parsing_error": None, "raw": "{}"}
    )

    graph = build_compiler_graph(
        registry=registry,
        planner=Planner(llm=planner_llm),  # type: ignore[arg-type]
        compiler=PlanCompiler(registry=registry),
        executor=Executor(registry=registry),
        joiner=Joiner(llm=joiner_llm),  # type: ignore[arg-type]
    )

    final_state = asyncio.run(graph.ainvoke({"request": "   "}))

    assert final_state["status"] == "failed"
    assert final_state["error"] is not None
    assert final_state["error"].stage == "planner"
    assert final_state["error"].error_type == "ValueError"
    assert final_state["metrics"].llm_calls == 0
    assert final_state["metrics"].plan_valid is None
    assert planner_llm.runnable.call_count == 0


def test_graph_stops_on_compiler_error_and_marks_plan_valid_false():
    registry = _build_registry()
    invalid_plan = ExecutionPlan(
        objective="Plan with unknown tool and bad dependency",
        tasks=[
            TaskSpec(
                task_id="task_1",
                tool_name="unknown_tool",
                description="Invalid tool",
                inputs={"x": InputBinding(task_id="task_99")},
            )
        ],
    )
    planner_llm = FakeChatModel(
        {"parsed": invalid_plan, "parsing_error": None, "raw": "{}"}
    )
    joiner_llm = FakeChatModel(
        {"parsed": JoinerOutput(answer="nope"), "parsing_error": None, "raw": "{}"}
    )

    graph = build_compiler_graph(
        registry=registry,
        planner=Planner(llm=planner_llm),  # type: ignore[arg-type]
        compiler=PlanCompiler(registry=registry),
        executor=Executor(registry=registry),
        joiner=Joiner(llm=joiner_llm),  # type: ignore[arg-type]
    )

    final_state = asyncio.run(graph.ainvoke({"request": "Do something"}))

    assert final_state["status"] == "failed"
    assert final_state["plan"] == invalid_plan
    assert final_state["compiled_dag"] is None
    assert "execution_result" not in final_state
    assert joiner_llm.runnable.call_count == 0

    err = final_state["error"]
    assert err is not None
    assert err.stage == "compiler"
    assert err.error_type == "PlanCompilationError"
    assert len(err.issues) == 2

    metrics = final_state["metrics"]
    assert metrics.plan_valid is False
    assert metrics.compile_attempts == 1
    assert metrics.llm_calls == 1
    assert metrics.planning_latency_ms is not None
    assert metrics.compilation_latency_ms is not None
    assert metrics.execution_latency_ms is None


def test_graph_continues_to_joiner_on_partial_task_failure():
    registry = _build_registry(fail_country="Italy")
    planner_llm = FakeChatModel(
        {"parsed": _valid_plan(), "parsing_error": None, "raw": "{}"}
    )
    joiner_llm = FakeChatModel(
        {
            "parsed": JoinerOutput(
                answer="Ho recuperato solo la Francia (68.400.000); Italia fallita."
            ),
            "parsing_error": None,
            "raw": "{}",
        }
    )

    graph = build_compiler_graph(
        registry=registry,
        planner=Planner(llm=planner_llm),  # type: ignore[arg-type]
        compiler=PlanCompiler(registry=registry),
        executor=Executor(registry=registry),
        joiner=Joiner(llm=joiner_llm),  # type: ignore[arg-type]
    )

    final_state = asyncio.run(graph.ainvoke({"request": "Italia meno Francia"}))

    assert final_state["error"] is None
    assert final_state["status"] == "partial"
    assert final_state["final_answer"] == (
        "Ho recuperato solo la Francia (68.400.000); Italia fallita."
    )
    assert final_state["execution_result"] is not None
    assert final_state["execution_result"].succeeded_task_ids == ("task_2",)
    assert final_state["execution_result"].failed_task_ids == ("task_1",)
    assert final_state["execution_result"].skipped_task_ids == ("task_3",)
    assert final_state["joiner_result"] is not None
    assert len(final_state["joiner_result"].missing_information) == 2
    assert final_state["metrics"].llm_calls == 2


def test_graph_stops_on_unexpected_executor_exception():
    class ExplodingExecutor(Executor):
        async def execute(self, dag: Any) -> Any:
            raise RuntimeError("Executor infrastructure failure")

    registry = _build_registry()
    planner_llm = FakeChatModel(
        {"parsed": _valid_plan(), "parsing_error": None, "raw": "{}"}
    )
    joiner_llm = FakeChatModel(
        {"parsed": JoinerOutput(answer="should not run"), "parsing_error": None, "raw": "{}"}
    )

    graph = build_compiler_graph(
        registry=registry,
        planner=Planner(llm=planner_llm),  # type: ignore[arg-type]
        compiler=PlanCompiler(registry=registry),
        executor=ExplodingExecutor(registry=registry),
        joiner=Joiner(llm=joiner_llm),  # type: ignore[arg-type]
    )

    final_state = asyncio.run(graph.ainvoke({"request": "Italia meno Francia"}))

    assert final_state["status"] == "failed"
    assert final_state["execution_result"] is None
    assert "joiner_result" not in final_state
    assert joiner_llm.runnable.call_count == 0

    err = final_state["error"]
    assert err is not None
    assert err.stage == "executor"
    assert err.error_type == "RuntimeError"
    assert "infrastructure failure" in err.message
    assert final_state["metrics"].llm_calls == 1
    assert final_state["metrics"].execution_latency_ms is not None


def test_graph_propagates_cancellation_instead_of_swallowing_it():
    started = asyncio.Event()

    class WaitingExecutor(Executor):
        async def execute(self, dag: Any) -> Any:
            started.set()
            await asyncio.sleep(10.0)
            raise AssertionError("unreachable")

    registry = _build_registry()
    planner_llm = FakeChatModel(
        {"parsed": _valid_plan(), "parsing_error": None, "raw": "{}"}
    )
    joiner_llm = FakeChatModel(
        {"parsed": JoinerOutput(answer="nope"), "parsing_error": None, "raw": "{}"}
    )

    graph = build_compiler_graph(
        registry=registry,
        planner=Planner(llm=planner_llm),  # type: ignore[arg-type]
        compiler=PlanCompiler(registry=registry),
        executor=WaitingExecutor(registry=registry),
        joiner=Joiner(llm=joiner_llm),  # type: ignore[arg-type]
    )

    async def cancel_while_executing() -> None:
        run_task = asyncio.create_task(graph.ainvoke({"request": "Italia meno Francia"}))
        await started.wait()
        run_task.cancel()
        await run_task

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(cancel_while_executing())

    assert joiner_llm.runnable.call_count == 0


def test_graph_handles_joiner_error_preserving_upstream_state():
    registry = _build_registry()
    planner_llm = FakeChatModel(
        {"parsed": _valid_plan(), "parsing_error": None, "raw": "{}"}
    )
    joiner_llm = FakeChatModel(
        {
            "parsed": None,
            "parsing_error": ValueError("empty answer"),
            "raw": '{"answer": ""}',
        }
    )

    graph = build_compiler_graph(
        registry=registry,
        planner=Planner(llm=planner_llm),  # type: ignore[arg-type]
        compiler=PlanCompiler(registry=registry),
        executor=Executor(registry=registry),
        joiner=Joiner(llm=joiner_llm),  # type: ignore[arg-type]
    )

    final_state = asyncio.run(graph.ainvoke({"request": "Italia meno Francia"}))

    assert final_state["status"] == "failed"
    assert final_state["final_answer"] is None
    assert final_state["error"] is not None
    assert final_state["error"].stage == "joiner"
    assert final_state["error"].error_type == "JoinerOutputError"
    assert final_state["plan"] == _valid_plan()
    assert final_state["compiled_dag"] is not None
    assert final_state["execution_result"] is not None
    assert final_state["metrics"].llm_calls == 2
    assert final_state["metrics"].plan_valid is True


def test_graph_streaming_order_and_mermaid_export():
    registry = _build_registry()
    planner_llm = FakeChatModel(
        {"parsed": _valid_plan(), "parsing_error": None, "raw": "{}"}
    )
    joiner_llm = FakeChatModel(
        {"parsed": JoinerOutput(answer="Risposta ok."), "parsing_error": None, "raw": "{}"}
    )

    graph = build_compiler_graph(
        registry=registry,
        planner=Planner(llm=planner_llm),  # type: ignore[arg-type]
        compiler=PlanCompiler(registry=registry),
        executor=Executor(registry=registry),
        joiner=Joiner(llm=joiner_llm),  # type: ignore[arg-type]
    )

    async def collect_nodes() -> list[str]:
        visited: list[str] = []
        async for chunk in graph.astream({"request": "Italia meno Francia"}):
            visited.extend(chunk.keys())
        return visited

    visited = asyncio.run(collect_nodes())
    assert visited == ["planner", "compiler", "executor", "joiner"]

    mermaid = graph.get_graph().draw_mermaid()
    for node_name in ("planner", "compiler", "executor", "joiner"):
        assert node_name in mermaid
    # v1 (max_compile_retries=0) must not expose a compiler -> planner cycle.
    assert "compiler -.-> planner" not in mermaid


class SequenceStructuredRunnable:
    """Fake runnable returning a sequence of responses across retries."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.received_messages: list[list[Any]] = []

    async def ainvoke(self, messages: list[Any]) -> dict[str, Any]:
        self.received_messages.append(messages)
        idx = min(len(self.received_messages) - 1, len(self._responses) - 1)
        return self._responses[idx]


class SequenceChatModel:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.runnable = SequenceStructuredRunnable(responses)

    def with_structured_output(self, **kwargs: Any) -> SequenceStructuredRunnable:
        return self.runnable


def _malformed_binding_plan() -> ExecutionPlan:
    return ExecutionPlan.model_validate(
        {
            "objective": "Compute Italy minus France population.",
            "tasks": [
                {
                    "task_id": "task_1",
                    "tool_name": "population_lookup",
                    "description": "Get Italy population.",
                    "inputs": {"country": "Italy"},
                },
                {
                    "task_id": "task_2",
                    "tool_name": "population_lookup",
                    "description": "Get France population.",
                    "inputs": {"country": "France"},
                },
                {
                    "task_id": "task_3",
                    "tool_name": "calculator",
                    "description": "Subtract France from Italy.",
                    "inputs": {
                        "operation": "subtract",
                        "a": {"task_id": "task_1", "output_path": ["population"]},
                        "b": {"task_id": "task_2", "output, path": ["population"]},
                    },
                },
            ],
        }
    )


def test_graph_rejects_negative_max_compile_retries():
    registry = _build_registry()
    planner_llm = FakeChatModel(
        {"parsed": _valid_plan(), "parsing_error": None, "raw": "{}"}
    )
    joiner_llm = FakeChatModel(
        {"parsed": JoinerOutput(answer="ok"), "parsing_error": None, "raw": "{}"}
    )

    with pytest.raises(ValueError, match="max_compile_retries"):
        build_compiler_graph(
            registry=registry,
            planner=Planner(llm=planner_llm),  # type: ignore[arg-type]
            compiler=PlanCompiler(registry=registry),
            executor=Executor(registry=registry),
            joiner=Joiner(llm=joiner_llm),  # type: ignore[arg-type]
            max_compile_retries=-1,
        )


def test_graph_v2_self_correction_recovers_on_first_retry():
    registry = _build_registry()
    bad_plan = _malformed_binding_plan()
    good_plan = _valid_plan()

    planner_llm = SequenceChatModel(
        [
            {"parsed": bad_plan, "parsing_error": None, "raw": "{bad}"},
            {"parsed": good_plan, "parsing_error": None, "raw": "{good}"},
        ]
    )
    joiner_llm = FakeChatModel(
        {
            "parsed": JoinerOutput(answer="Differenza: -9.410.000."),
            "parsing_error": None,
            "raw": "{}",
        }
    )

    graph = build_compiler_graph(
        registry=registry,
        planner=Planner(llm=planner_llm),  # type: ignore[arg-type]
        compiler=PlanCompiler(registry=registry),
        executor=Executor(registry=registry),
        joiner=Joiner(llm=joiner_llm),  # type: ignore[arg-type]
        max_compile_retries=2,
    )

    # v2 topology exposes the compiler -> planner feedback edge.
    assert "compiler -.-> planner" in graph.get_graph().draw_mermaid()

    final_state = asyncio.run(graph.ainvoke({"request": "Italia meno Francia"}))

    assert final_state["status"] == "complete"
    assert final_state["error"] is None
    assert final_state["compile_feedback"] is None
    assert final_state["final_answer"] == "Differenza: -9.410.000."
    assert final_state["plan"] == good_plan
    assert final_state["execution_result"] is not None
    assert final_state["execution_result"].results["task_3"].output == {
        "result": -9_410_000.0
    }

    # Verify the retry prompt actually received previous_plan + malformed_binding issue.
    assert len(planner_llm.runnable.received_messages) == 2
    retry_human_msg = planner_llm.runnable.received_messages[1][1].content
    assert "PREVIOUS EXECUTION PLAN (FAILED COMPILATION):" in retry_human_msg
    assert "malformed_binding" in retry_human_msg
    assert "output, path" in retry_human_msg

    # Verify error history preserved even after successful recovery.
    history = final_state["compile_error_history"]
    assert len(history) == 1
    assert any("malformed_binding" in item for item in history[0])

    metrics = final_state["metrics"]
    assert metrics.plan_valid is True
    assert metrics.compile_attempts == 2
    assert metrics.compile_retries == 1
    assert metrics.llm_calls == 3  # 2 planner + 1 joiner
    assert metrics.planning_latency_ms is not None
    assert metrics.compilation_latency_ms is not None


def test_graph_v2_stops_when_compile_retry_budget_exhausted():
    registry = _build_registry()
    bad_plan = _malformed_binding_plan()

    planner_llm = SequenceChatModel(
        [
            {"parsed": bad_plan, "parsing_error": None, "raw": "{bad1}"},
            {"parsed": bad_plan, "parsing_error": None, "raw": "{bad2}"},
            {"parsed": _valid_plan(), "parsing_error": None, "raw": "{unused}"},
        ]
    )
    joiner_llm = FakeChatModel(
        {"parsed": JoinerOutput(answer="should not run"), "parsing_error": None, "raw": "{}"}
    )

    graph = build_compiler_graph(
        registry=registry,
        planner=Planner(llm=planner_llm),  # type: ignore[arg-type]
        compiler=PlanCompiler(registry=registry),
        executor=Executor(registry=registry),
        joiner=Joiner(llm=joiner_llm),  # type: ignore[arg-type]
        max_compile_retries=1,
    )

    final_state = asyncio.run(graph.ainvoke({"request": "Italia meno Francia"}))

    assert final_state["status"] == "failed"
    assert final_state["error"] is not None
    assert final_state["error"].stage == "compiler"
    assert final_state["error"].error_type == "PlanCompilationError"
    assert "execution_result" not in final_state
    assert joiner_llm.runnable.call_count == 0
    assert len(final_state["compile_error_history"]) == 2

    metrics = final_state["metrics"]
    assert metrics.plan_valid is False
    assert metrics.compile_attempts == 2
    assert metrics.compile_retries == 1
    assert metrics.llm_calls == 2


def test_graph_v2_does_not_retry_planner_output_error():
    registry = _build_registry()
    planner_llm = SequenceChatModel(
        [
            {"parsed": None, "parsing_error": ValueError("bad json"), "raw": "{"},
            {"parsed": _valid_plan(), "parsing_error": None, "raw": "{}"},
        ]
    )
    joiner_llm = FakeChatModel(
        {"parsed": JoinerOutput(answer="nope"), "parsing_error": None, "raw": "{}"}
    )

    graph = build_compiler_graph(
        registry=registry,
        planner=Planner(llm=planner_llm),  # type: ignore[arg-type]
        compiler=PlanCompiler(registry=registry),
        executor=Executor(registry=registry),
        joiner=Joiner(llm=joiner_llm),  # type: ignore[arg-type]
        max_compile_retries=2,
    )

    final_state = asyncio.run(graph.ainvoke({"request": "Italia meno Francia"}))

    assert final_state["status"] == "failed"
    assert final_state["error"] is not None
    assert final_state["error"].stage == "planner"
    assert final_state["metrics"].llm_calls == 1
    assert final_state["metrics"].compile_attempts == 0
    assert final_state["metrics"].compile_retries == 0


def test_graph_v2_streaming_order_with_compile_retry():
    registry = _build_registry()
    planner_llm = SequenceChatModel(
        [
            {"parsed": _malformed_binding_plan(), "parsing_error": None, "raw": "{}"},
            {"parsed": _valid_plan(), "parsing_error": None, "raw": "{}"},
        ]
    )
    joiner_llm = FakeChatModel(
        {"parsed": JoinerOutput(answer="ok"), "parsing_error": None, "raw": "{}"}
    )

    graph = build_compiler_graph(
        registry=registry,
        planner=Planner(llm=planner_llm),  # type: ignore[arg-type]
        compiler=PlanCompiler(registry=registry),
        executor=Executor(registry=registry),
        joiner=Joiner(llm=joiner_llm),  # type: ignore[arg-type]
        max_compile_retries=2,
    )

    async def collect() -> list[str]:
        visited: list[str] = []
        async for chunk in graph.astream({"request": "Italia meno Francia"}):
            visited.extend(chunk.keys())
        return visited

    assert asyncio.run(collect()) == [
        "planner",
        "compiler",
        "planner",
        "compiler",
        "executor",
        "joiner",
    ]


class SchemaDispatchChatModel:
    """Fake chat model returning a dedicated sequence runnable per requested schema."""

    def __init__(self, responses_by_schema: dict[type[Any], list[dict[str, Any]]]) -> None:
        self.runnables: dict[type[Any], SequenceStructuredRunnable] = {
            schema: SequenceStructuredRunnable(resps)
            for schema, resps in responses_by_schema.items()
        }
        self.requested_schemas: list[type[Any]] = []

    def with_structured_output(self, **kwargs: Any) -> SequenceStructuredRunnable:
        schema = kwargs["schema"]
        self.requested_schemas.append(schema)
        return self.runnables[schema]


def _build_registry_with_backup() -> ToolRegistry:
    """Registry where primary population_lookup fails for Italy, backup_lookup succeeds."""

    async def primary_lookup(country: str) -> dict[str, str | int]:
        if country.lower() == "italy":
            raise RuntimeError("Primary lookup service unavailable for Italy")
        pops = {"france": 68_400_000}
        return {"country": country, "population": pops[country.lower()]}

    async def backup_lookup(country: str) -> dict[str, str | int]:
        pops = {"italy": 58_990_000, "france": 68_400_000}
        return {"country": country, "population": pops[country.lower()]}

    def calc(
        operation: Literal["add", "subtract", "multiply", "divide"],
        a: float,
        b: float,
    ) -> dict[str, float]:
        if operation == "subtract":
            return {"result": a - b}
        return {"result": a + b}

    return ToolRegistry(
        [
            ToolSpec(
                name="population_lookup",
                description="Primary country population lookup",
                func=primary_lookup,
                input_schema=PopulationInput,
                output_schema=PopulationOutput,
            ),
            ToolSpec(
                name="backup_lookup",
                description="Backup country population lookup",
                func=backup_lookup,
                input_schema=PopulationInput,
                output_schema=PopulationOutput,
            ),
            ToolSpec(
                name="calculator",
                description="Binary arithmetic",
                func=calc,
                input_schema=CalculatorInput,
                output_schema=CalculatorOutput,
            ),
        ]
    )


def _round2_backup_plan() -> ExecutionPlan:
    return ExecutionPlan(
        objective="Recover Italy via backup_lookup and subtract France (68400000).",
        tasks=[
            TaskSpec(
                task_id="task_1",
                tool_name="backup_lookup",
                description="Get Italy population from backup.",
                inputs={"country": "Italy"},
            ),
            TaskSpec(
                task_id="task_2",
                tool_name="calculator",
                description="Subtract France population from Italy.",
                inputs={
                    "operation": "subtract",
                    "a": InputBinding(task_id="task_1", output_path=["population"]),
                    "b": 68_400_000,
                },
            ),
        ],
    )


def test_graph_rejects_negative_max_replans():
    registry = _build_registry()
    planner_llm = FakeChatModel(
        {"parsed": _valid_plan(), "parsing_error": None, "raw": "{}"}
    )
    joiner_llm = FakeChatModel(
        {"parsed": JoinerOutput(answer="ok"), "parsing_error": None, "raw": "{}"}
    )

    with pytest.raises(ValueError, match="max_replans"):
        build_compiler_graph(
            registry=registry,
            planner=Planner(llm=planner_llm),  # type: ignore[arg-type]
            compiler=PlanCompiler(registry=registry),
            executor=Executor(registry=registry),
            joiner=Joiner(llm=joiner_llm),  # type: ignore[arg-type]
            max_replans=-1,
        )


def test_graph_v3_runtime_replan_recovers_via_backup_tool():
    registry = _build_registry_with_backup()
    round1_plan = _valid_plan()
    round2_plan = _round2_backup_plan()

    planner_llm = SequenceChatModel(
        [
            {"parsed": round1_plan, "parsing_error": None, "raw": "{r1}"},
            {"parsed": round2_plan, "parsing_error": None, "raw": "{r2}"},
        ]
    )
    joiner_llm = SchemaDispatchChatModel(
        {
            JoinerDecisionOutput: [
                {
                    "parsed": JoinerDecisionOutput(
                        action="replan",
                        replan_reason=(
                            "population_lookup failed for Italy; use backup_lookup for "
                            "Italy and compute Italy minus France (68400000)."
                        ),
                    ),
                    "parsing_error": None,
                    "raw": "{replan}",
                }
            ],
            JoinerOutput: [
                {
                    "parsed": JoinerOutput(
                        answer="Differenza Italia meno Francia: -9.410.000."
                    ),
                    "parsing_error": None,
                    "raw": "{finish}",
                }
            ],
        }
    )

    graph = build_compiler_graph(
        registry=registry,
        planner=Planner(llm=planner_llm),  # type: ignore[arg-type]
        compiler=PlanCompiler(registry=registry),
        executor=Executor(registry=registry),
        joiner=Joiner(llm=joiner_llm),  # type: ignore[arg-type]
        max_compile_retries=1,
        max_replans=1,
    )

    mermaid = graph.get_graph().draw_mermaid()
    assert "joiner -.-> planner" in mermaid
    assert "compiler -.-> planner" in mermaid

    final_state = asyncio.run(graph.ainvoke({"request": "Italia meno Francia"}))

    assert final_state["status"] == "complete"
    assert final_state["error"] is None
    assert final_state.get("replan_error") is None
    assert final_state["final_answer"] == "Differenza Italia meno Francia: -9.410.000."
    assert final_state["execution_result"] is not None
    assert final_state["execution_result"].results["task_2"].output == {
        "result": -9_410_000.0
    }

    # Execution history preserves Round 1 snapshot.
    history = final_state["execution_history"]
    assert len(history) == 1
    assert history[0].round_index == 1
    assert history[0].execution_result.failed_task_ids == ("task_1",)

    # Round 2 Planner prompt received the replan context with Round 1 France output.
    round2_prompt = planner_llm.runnable.received_messages[1][1].content
    assert "RUNTIME REPLANNING CONTEXT (PREVIOUS ROUNDS):" in round2_prompt
    assert "backup_lookup" in round2_prompt
    assert "68400000" in round2_prompt

    # Round 2 Joiner hit max_replans=1 cap -> forced JoinerOutput with Round 1 history.
    final_joiner_prompt = (
        joiner_llm.runnables[JoinerOutput].received_messages[0][1].content
    )
    assert "PREVIOUS EXECUTION ROUNDS:" in final_joiner_prompt
    assert "Round 1" in final_joiner_prompt

    metrics = final_state["metrics"]
    assert metrics.replan_attempts == 1
    assert metrics.compile_attempts == 2
    assert metrics.compile_retries == 0  # 0 compile retries despite 2 rounds!
    assert metrics.llm_calls == 4  # 2 planner + 2 joiner
    assert metrics.plan_valid is True


def test_graph_v3_and_v2_fresh_compile_retry_budget_on_round_2():
    """Round 1 uses 1 compile retry, then replans; Round 2 gets a fresh retry budget."""
    registry = _build_registry_with_backup()
    bad_round2_plan = ExecutionPlan(
        objective="Bad round 2 plan with unknown tool",
        tasks=[
            TaskSpec(
                task_id="task_1",
                tool_name="nonexistent_backup",
                description="Wrong backup tool name",
                inputs={"country": "Italy"},
            )
        ],
    )

    planner_llm = SequenceChatModel(
        [
            {"parsed": _malformed_binding_plan(), "parsing_error": None, "raw": "{r1b}"},
            {"parsed": _valid_plan(), "parsing_error": None, "raw": "{r1g}"},
            {"parsed": bad_round2_plan, "parsing_error": None, "raw": "{r2b}"},
            {"parsed": _round2_backup_plan(), "parsing_error": None, "raw": "{r2g}"},
        ]
    )
    joiner_llm = SchemaDispatchChatModel(
        {
            JoinerDecisionOutput: [
                {
                    "parsed": JoinerDecisionOutput(
                        action="replan",
                        replan_reason="Use backup_lookup for Italy.",
                    ),
                    "parsing_error": None,
                    "raw": "{replan}",
                }
            ],
            JoinerOutput: [
                {
                    "parsed": JoinerOutput(answer="Recuperato: -9.410.000."),
                    "parsing_error": None,
                    "raw": "{finish}",
                }
            ],
        }
    )

    graph = build_compiler_graph(
        registry=registry,
        planner=Planner(llm=planner_llm),  # type: ignore[arg-type]
        compiler=PlanCompiler(registry=registry),
        executor=Executor(registry=registry),
        joiner=Joiner(llm=joiner_llm),  # type: ignore[arg-type]
        max_compile_retries=1,
        max_replans=1,
    )

    final_state = asyncio.run(graph.ainvoke({"request": "Italia meno Francia"}))

    assert final_state["status"] == "complete"
    assert final_state["error"] is None
    assert final_state["final_answer"] == "Recuperato: -9.410.000."

    # On Round 2 compile retry (call index 3), Planner must receive BOTH replan_context
    # AND compile feedback!
    round2_retry_prompt = planner_llm.runnable.received_messages[3][1].content
    assert "RUNTIME REPLANNING CONTEXT (PREVIOUS ROUNDS):" in round2_retry_prompt
    assert "Use backup_lookup for Italy." in round2_retry_prompt
    assert "COMPILATION ERRORS TO FIX:" in round2_retry_prompt
    assert "nonexistent_backup" in round2_retry_prompt

    metrics = final_state["metrics"]
    assert metrics.replan_attempts == 1
    assert metrics.compile_attempts == 4
    assert metrics.compile_retries == 2  # 1 in Round 1 + 1 in Round 2
    assert metrics.llm_calls == 6  # 4 planner + 2 joiner
    assert metrics.plan_valid is True


def test_graph_v3_graceful_degradation_when_replan_planner_fails():
    """If Round 2 fails at Planner, v3 must fall back to Joiner with Round 1 partial results."""
    registry = _build_registry_with_backup()

    planner_llm = SequenceChatModel(
        [
            {"parsed": _valid_plan(), "parsing_error": None, "raw": "{r1}"},
            {
                "parsed": None,
                "parsing_error": ValueError("truncated JSON on round 2"),
                "raw": "{truncated",
            },
        ]
    )
    joiner_llm = SchemaDispatchChatModel(
        {
            JoinerDecisionOutput: [
                {
                    "parsed": JoinerDecisionOutput(
                        action="replan",
                        replan_reason="Try backup_lookup for Italy.",
                    ),
                    "parsing_error": None,
                    "raw": "{replan}",
                }
            ],
            JoinerOutput: [
                {
                    "parsed": JoinerOutput(
                        answer="Risposta parziale dal Round 1: Francia = 68.400.000."
                    ),
                    "parsing_error": None,
                    "raw": "{fallback_finish}",
                }
            ],
        }
    )

    graph = build_compiler_graph(
        registry=registry,
        planner=Planner(llm=planner_llm),  # type: ignore[arg-type]
        compiler=PlanCompiler(registry=registry),
        executor=Executor(registry=registry),
        joiner=Joiner(llm=joiner_llm),  # type: ignore[arg-type]
        max_compile_retries=1,
        max_replans=2,
    )

    final_state = asyncio.run(graph.ainvoke({"request": "Italia meno Francia"}))

    # Preserves Round 1 partial answer instead of crashing to final_answer=None!
    assert final_state["status"] == "partial"
    assert final_state["error"] is None
    assert final_state["final_answer"] == (
        "Risposta parziale dal Round 1: Francia = 68.400.000."
    )
    replan_err = final_state.get("replan_error")
    assert replan_err is not None
    assert replan_err.stage == "planner"
    assert replan_err.error_type == "PlannerOutputError"
    assert final_state["metrics"].replan_attempts == 1
    assert final_state["metrics"].llm_calls == 4


def test_graph_v3_finishes_on_round_1_without_replan_overhead():
    """When Joiner chooses finish on Round 1, v3 terminates cleanly with 0 replans."""
    registry = _build_registry()
    planner_llm = FakeChatModel(
        {"parsed": _valid_plan(), "parsing_error": None, "raw": "{}"}
    )
    joiner_llm = SchemaDispatchChatModel(
        {
            JoinerDecisionOutput: [
                {
                    "parsed": JoinerDecisionOutput(
                        action="finish",
                        answer="Differenza: -9.410.000.",
                    ),
                    "parsing_error": None,
                    "raw": "{finish}",
                }
            ],
            JoinerOutput: [],
        }
    )

    graph = build_compiler_graph(
        registry=registry,
        planner=Planner(llm=planner_llm),  # type: ignore[arg-type]
        compiler=PlanCompiler(registry=registry),
        executor=Executor(registry=registry),
        joiner=Joiner(llm=joiner_llm),  # type: ignore[arg-type]
        max_compile_retries=1,
        max_replans=2,
    )

    final_state = asyncio.run(graph.ainvoke({"request": "Italia meno Francia"}))

    assert final_state["status"] == "complete"
    assert final_state["error"] is None
    assert final_state["final_answer"] == "Differenza: -9.410.000."
    assert final_state["metrics"].replan_attempts == 0
    assert final_state["metrics"].llm_calls == 2
    assert len(joiner_llm.runnables[JoinerOutput].received_messages) == 0


def test_graph_direct_response_bypasses_compiler_and_executor_for_empty_plan():
    """When Planner returns tasks=[], the graph routes directly to Joiner.join_direct()."""
    registry = _build_registry()
    empty_plan = ExecutionPlan(
        objective="Conversational greeting — no tool calls required.",
        tasks=[],
    )
    planner_llm = FakeChatModel(
        {"parsed": empty_plan, "parsing_error": None, "raw": "{}"}
    )
    joiner_llm = SchemaDispatchChatModel(
        {
            JoinerDecisionOutput: [],
            JoinerOutput: [
                {
                    "parsed": JoinerOutput(answer="Ciao! Tutto bene, grazie!"),
                    "parsing_error": None,
                    "raw": "{direct}",
                }
            ],
        }
    )

    graph = build_compiler_graph(
        registry=registry,
        planner=Planner(llm=planner_llm),  # type: ignore[arg-type]
        compiler=PlanCompiler(registry=registry),
        executor=Executor(registry=registry),
        joiner=Joiner(llm=joiner_llm),  # type: ignore[arg-type]
        max_compile_retries=2,
        max_replans=2,
    )

    final_state = asyncio.run(graph.ainvoke({"request": "come va?"}))

    assert final_state["status"] == "complete"
    assert final_state["error"] is None
    assert final_state["final_answer"] == "Ciao! Tutto bene, grazie!"
    assert final_state.get("compiled_dag") is None
    assert final_state.get("execution_result") is None
    metrics = final_state["metrics"]
    assert metrics.llm_calls == 2  # 1 planner + 1 joiner
    assert metrics.compile_attempts == 0
    assert metrics.replan_attempts == 0
    assert metrics.compilation_latency_ms is None
    assert metrics.execution_latency_ms is None


def test_graph_v3_empty_replan_finalizes_from_prior_rounds():
    """If Planner returns tasks=[] on a replan round, Joiner synthesizes from prior history."""
    registry = _build_registry_with_backup()
    planner_llm = SequenceChatModel(
        [
            {"parsed": _valid_plan(), "parsing_error": None, "raw": "{r1}"},
            {
                "parsed": ExecutionPlan(objective="No further tools needed.", tasks=[]),
                "parsing_error": None,
                "raw": "{r2_empty}",
            },
        ]
    )
    joiner_llm = SchemaDispatchChatModel(
        {
            JoinerDecisionOutput: [
                {
                    "parsed": JoinerDecisionOutput(
                        action="replan",
                        replan_reason="Check if any backup is needed.",
                    ),
                    "parsing_error": None,
                    "raw": "{replan}",
                }
            ],
            JoinerOutput: [
                {
                    "parsed": JoinerOutput(
                        answer="Dati parziali dal Round 1: Francia = 68.400.000."
                    ),
                    "parsing_error": None,
                    "raw": "{finish}",
                }
            ],
        }
    )

    graph = build_compiler_graph(
        registry=registry,
        planner=Planner(llm=planner_llm),  # type: ignore[arg-type]
        compiler=PlanCompiler(registry=registry),
        executor=Executor(registry=registry),
        joiner=Joiner(llm=joiner_llm),  # type: ignore[arg-type]
        max_compile_retries=1,
        max_replans=2,
    )

    final_state = asyncio.run(graph.ainvoke({"request": "Italia meno Francia"}))

    assert final_state["status"] == "partial"
    assert final_state["error"] is None
    assert final_state["final_answer"] == "Dati parziali dal Round 1: Francia = 68.400.000."
    assert final_state["metrics"].replan_attempts == 1
    assert final_state["metrics"].llm_calls == 4



