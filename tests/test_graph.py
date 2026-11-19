import asyncio
from typing import Any, Literal

import pytest
from pydantic import BaseModel

from langgraph_llm_compiler.compiler import PlanCompiler, ToolRegistry, ToolSpec
from langgraph_llm_compiler.executor import Executor
from langgraph_llm_compiler.graph import (
    WorkflowError,
    WorkflowMetrics,
    build_compiler_graph,
)
from langgraph_llm_compiler.joiner import Joiner, JoinerOutput
from langgraph_llm_compiler.planner import (
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
