import asyncio
import time
from typing import Any, Literal

import pytest
from pydantic import BaseModel

from langgraph_llm_compiler.compiler import PlanCompiler, ToolRegistry, ToolSpec
from langgraph_llm_compiler.executor import (
    BindingResolutionError,
    ExecutionResult,
    Executor,
    TaskExecutionResult,
    resolve_output_path,
    resolve_task_inputs,
)
from langgraph_llm_compiler.planner import ExecutionPlan, InputBinding, TaskSpec


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


def test_resolve_output_path_and_task_inputs():
    nested = {"statistics": {"population": 59_000_000}}
    assert resolve_output_path(nested, []) == nested
    assert resolve_output_path(nested, ["statistics", "population"]) == 59_000_000

    model_output = PopulationOutput(country="Italy", population=58_990_000)
    assert resolve_output_path(model_output, ["population"]) == 58_990_000

    with pytest.raises(BindingResolutionError, match="missing"):
        resolve_output_path(nested, ["statistics", "missing"])

    with pytest.raises(BindingResolutionError, match="Cannot traverse"):
        resolve_output_path(nested, ["statistics", "population", "sub"])

    task = TaskSpec(
        task_id="task_3",
        tool_name="calculator",
        description="Subtract",
        inputs={
            "operation": "subtract",
            "a": InputBinding(task_id="task_1", output_path=["population"]),
            "b": InputBinding(task_id="task_2", output_path=["population"]),
        },
    )
    results = {
        "task_1": TaskExecutionResult(
            task_id="task_1",
            tool_name="population_lookup",
            status="success",
            output={"country": "Italy", "population": 58_990_000},
        ),
        "task_2": TaskExecutionResult(
            task_id="task_2",
            tool_name="population_lookup",
            status="success",
            output={"country": "France", "population": 68_400_000},
        ),
    }

    resolved = resolve_task_inputs(task, results)
    assert resolved == {
        "operation": "subtract",
        "a": 58_990_000,
        "b": 68_400_000,
    }
    assert task.static_inputs == {"operation": "subtract"}

    # Upstream failure must raise BindingResolutionError.
    failed_results = {
        **results,
        "task_1": TaskExecutionResult(
            task_id="task_1",
            tool_name="population_lookup",
            status="error",
            error="boom",
            error_type="RuntimeError",
            error_stage="tool_execution",
        ),
    }
    with pytest.raises(BindingResolutionError, match="status 'error'"):
        resolve_task_inputs(task, failed_results)


def test_executor_init_validation():
    registry = ToolRegistry()
    with pytest.raises(ValueError, match="max_concurrency"):
        Executor(registry, max_concurrency=0)
    with pytest.raises(ValueError, match="task_timeout_s"):
        Executor(registry, task_timeout_s=0.0)
    with pytest.raises(ValueError, match="task_timeout_s"):
        Executor(registry, task_timeout_s=-1.0)


def test_event_driven_five_task_dag_has_no_wave_barrier():
    """Verify T3 starts and finishes while slower sibling T2 is still running."""

    async def delayed(
        value: str, delay_s: float, upstream: str | None = None
    ) -> dict[str, str]:
        await asyncio.sleep(delay_s)
        combined = f"{upstream}->{value}" if upstream else value
        return {"value": combined}

    async def join(left: str, right: str, delay_s: float = 0.01) -> dict[str, str]:
        await asyncio.sleep(delay_s)
        return {"value": f"{left}+{right}"}

    registry = ToolRegistry(
        [
            ToolSpec(name="delayed", description="Delayed step", func=delayed),
            ToolSpec(name="join", description="Join two branches", func=join),
        ]
    )
    compiler = PlanCompiler(registry)

    plan = ExecutionPlan(
        objective="Five-task event-driven test",
        tasks=[
            TaskSpec(
                task_id="T1",
                tool_name="delayed",
                description="Fast root",
                inputs={"value": "A", "delay_s": 0.02},
            ),
            TaskSpec(
                task_id="T2",
                tool_name="delayed",
                description="Slow root",
                inputs={"value": "B", "delay_s": 0.25},
            ),
            TaskSpec(
                task_id="T3",
                tool_name="delayed",
                description="Depends on fast T1",
                inputs={
                    "value": "C",
                    "delay_s": 0.02,
                    "upstream": InputBinding(task_id="T1", output_path=["value"]),
                },
            ),
            TaskSpec(
                task_id="T4",
                tool_name="delayed",
                description="Depends on slow T2",
                inputs={
                    "value": "D",
                    "delay_s": 0.02,
                    "upstream": InputBinding(task_id="T2", output_path=["value"]),
                },
            ),
            TaskSpec(
                task_id="T5",
                tool_name="join",
                description="Depends on T3 and T4",
                inputs={
                    "delay_s": 0.01,
                    "left": InputBinding(task_id="T3", output_path=["value"]),
                    "right": InputBinding(task_id="T4", output_path=["value"]),
                },
            ),
        ],
    )

    dag = compiler.compile(plan)
    result = asyncio.run(Executor(registry, max_concurrency=8).execute(dag))

    assert result.all_succeeded is True
    assert result.results["T5"].output == {"value": "A->C+B->D"}

    t2 = result.results["T2"]
    t3 = result.results["T3"]
    t4 = result.results["T4"]
    t5 = result.results["T5"]

    assert t2.finished_at_ms is not None
    assert t3.started_at_ms is not None and t3.finished_at_ms is not None
    assert t4.finished_at_ms is not None
    assert t5.started_at_ms is not None

    # Core event-driven proof: T3 started and finished before T2 completed.
    assert t3.started_at_ms < t2.finished_at_ms
    assert t3.finished_at_ms < t2.finished_at_ms
    # T5 waited for both predecessors.
    assert t5.started_at_ms >= t3.finished_at_ms - 0.5
    assert t5.started_at_ms >= t4.finished_at_ms - 0.5


def test_max_concurrency_one_runs_sequentially():
    active = 0
    peak_active = 0

    async def work(item: int) -> dict[str, int]:
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        await asyncio.sleep(0.03)
        active -= 1
        return {"item": item}

    registry = ToolRegistry([ToolSpec(name="work", description="Work", func=work)])
    plan = ExecutionPlan(
        objective="Sequential baseline",
        tasks=[
            TaskSpec(
                task_id=f"t{i}",
                tool_name="work",
                description=f"Task {i}",
                inputs={"item": i},
            )
            for i in range(1, 4)
        ],
    )
    dag = PlanCompiler(registry).compile(plan)

    result = asyncio.run(Executor(registry, max_concurrency=1).execute(dag))

    assert result.all_succeeded is True
    assert result.max_concurrency == 1
    assert peak_active == 1

    t1, t2, t3 = result.results["t1"], result.results["t2"], result.results["t3"]
    assert t1.finished_at_ms is not None and t2.started_at_ms is not None
    assert t2.finished_at_ms is not None and t3.started_at_ms is not None
    assert t2.started_at_ms >= t1.finished_at_ms - 0.5
    assert t3.started_at_ms >= t2.finished_at_ms - 0.5
    assert result.effective_parallelism <= 1.05


def test_unlimited_and_bounded_concurrency():
    active = 0
    peak_active = 0

    async def work(item: int) -> dict[str, int]:
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        await asyncio.sleep(0.05)
        active -= 1
        return {"item": item}

    registry = ToolRegistry([ToolSpec(name="work", description="Work", func=work)])
    plan = ExecutionPlan(
        objective="Parallel tasks",
        tasks=[
            TaskSpec(
                task_id=f"t{i}",
                tool_name="work",
                description=f"Task {i}",
                inputs={"item": i},
            )
            for i in range(1, 5)
        ],
    )
    dag = PlanCompiler(registry).compile(plan)

    # Unlimited concurrency
    res_unlimited = asyncio.run(Executor(registry, max_concurrency=None).execute(dag))
    assert res_unlimited.all_succeeded is True
    assert res_unlimited.max_concurrency is None
    assert peak_active == 4
    assert res_unlimited.effective_parallelism > 2.0

    # Bounded concurrency = 2
    active = 0
    peak_active = 0
    res_bounded = asyncio.run(Executor(registry, max_concurrency=2).execute(dag))
    assert res_bounded.all_succeeded is True
    assert res_bounded.max_concurrency == 2
    assert peak_active == 2


def test_sync_and_async_tools_run_concurrently():
    def sync_lookup(country: str) -> dict[str, str | int]:
        time.sleep(0.08)
        pops = {"Italy": 58_990_000, "France": 68_400_000}
        return {"country": country, "population": pops[country]}

    async def async_lookup(country: str) -> PopulationOutput:
        await asyncio.sleep(0.08)
        pops = {"Italy": 58_990_000, "France": 68_400_000}
        return PopulationOutput(country=country, population=pops[country])

    def calc(
        operation: Literal["add", "subtract", "multiply", "divide"],
        a: float,
        b: float,
    ) -> dict[str, float]:
        if operation == "subtract":
            return {"result": a - b}
        return {"result": a + b}

    registry = ToolRegistry(
        [
            ToolSpec(
                name="sync_lookup",
                description="Sync lookup",
                func=sync_lookup,
                input_schema=PopulationInput,
                output_schema=PopulationOutput,
            ),
            ToolSpec(
                name="async_lookup",
                description="Async lookup",
                func=async_lookup,
                input_schema=PopulationInput,
                output_schema=PopulationOutput,
            ),
            ToolSpec(
                name="calculator",
                description="Calculator",
                func=calc,
                input_schema=CalculatorInput,
                output_schema=CalculatorOutput,
            ),
        ]
    )

    plan = ExecutionPlan(
        objective="Mix sync and async tools",
        tasks=[
            TaskSpec(
                task_id="task_1",
                tool_name="sync_lookup",
                description="Italy (sync)",
                inputs={"country": "Italy"},
            ),
            TaskSpec(
                task_id="task_2",
                tool_name="async_lookup",
                description="France (async)",
                inputs={"country": "France"},
            ),
            TaskSpec(
                task_id="task_3",
                tool_name="calculator",
                description="Difference",
                inputs={
                    "operation": "subtract",
                    "a": InputBinding(task_id="task_1", output_path=["population"]),
                    "b": InputBinding(task_id="task_2", output_path=["population"]),
                },
            ),
        ],
    )

    dag = PlanCompiler(registry).compile(plan)
    result = asyncio.run(Executor(registry).execute(dag))

    assert result.all_succeeded is True
    assert result.results["task_2"].output == {
        "country": "France",
        "population": 68_400_000,
    }
    assert result.results["task_3"].output == {"result": -9_410_000.0}
    assert result.effective_parallelism > 1.3


def test_best_effort_failure_and_transitive_skip_propagation():
    async def fetch(country: str) -> dict[str, Any]:
        if country == "Italy":
            raise RuntimeError("Upstream service unavailable")
        await asyncio.sleep(0.02)
        return {"country": country, "population": 68_400_000}

    async def combine(a: Any, b: Any = None) -> dict[str, Any]:
        return {"combined": (a, b)}

    registry = ToolRegistry(
        [
            ToolSpec(name="fetch", description="Fetch", func=fetch),
            ToolSpec(name="combine", description="Combine", func=combine),
        ]
    )

    plan = ExecutionPlan(
        objective="Transitive skip propagation",
        tasks=[
            TaskSpec(
                task_id="task_1",
                tool_name="fetch",
                description="Fails",
                inputs={"country": "Italy"},
            ),
            TaskSpec(
                task_id="task_2",
                tool_name="fetch",
                description="Independent branch succeeds",
                inputs={"country": "France"},
            ),
            TaskSpec(
                task_id="task_3",
                tool_name="combine",
                description="Direct dependent of failed task_1",
                inputs={
                    "a": InputBinding(task_id="task_1", output_path=["population"])
                },
            ),
            TaskSpec(
                task_id="task_5",
                tool_name="combine",
                description="Transitive dependent via task_3",
                inputs={
                    "a": InputBinding(task_id="task_3", output_path=["combined"]),
                    "b": InputBinding(task_id="task_2", output_path=["population"]),
                },
            ),
        ],
    )

    dag = PlanCompiler(registry).compile(plan)
    result = asyncio.run(Executor(registry).execute(dag))

    assert result.all_succeeded is False
    assert result.succeeded_task_ids == ("task_2",)
    assert result.failed_task_ids == ("task_1",)
    assert result.skipped_task_ids == ("task_3", "task_5")

    r1 = result.results["task_1"]
    assert r1.status == "error"
    assert r1.error_type == "RuntimeError"
    assert r1.error_stage == "tool_execution"
    assert "unavailable" in (r1.error or "")

    for skipped_id in ("task_3", "task_5"):
        r_skip = result.results[skipped_id]
        assert r_skip.status == "skipped"
        assert r_skip.started_at_ms is None
        assert r_skip.finished_at_ms is None
        assert r_skip.duration_ms == 0.0
        assert r_skip.skip_reason is not None

    # ExecutionResult must be cleanly JSON-serializable for LangGraph state / benchmarks.
    reloaded = ExecutionResult.model_validate_json(result.model_dump_json())
    assert reloaded == result


def test_task_timeout_marks_error_and_skips_dependents():
    async def slow_tool() -> dict[str, int]:
        await asyncio.sleep(0.5)
        return {"value": 1}

    async def fast_tool() -> dict[str, int]:
        await asyncio.sleep(0.01)
        return {"value": 2}

    async def consumer(x: int) -> dict[str, int]:
        return {"value": x + 1}

    registry = ToolRegistry(
        [
            ToolSpec(name="slow", description="Slow", func=slow_tool),
            ToolSpec(name="fast", description="Fast", func=fast_tool),
            ToolSpec(name="consumer", description="Consumer", func=consumer),
        ]
    )

    plan = ExecutionPlan(
        objective="Timeout test",
        tasks=[
            TaskSpec(task_id="t_slow", tool_name="slow", description="Times out"),
            TaskSpec(task_id="t_fast", tool_name="fast", description="Succeeds"),
            TaskSpec(
                task_id="t_after",
                tool_name="consumer",
                description="Skipped after timeout",
                inputs={"x": InputBinding(task_id="t_slow", output_path=["value"])},
            ),
        ],
    )

    dag = PlanCompiler(registry).compile(plan)
    result = asyncio.run(Executor(registry, task_timeout_s=0.05).execute(dag))

    assert result.succeeded_task_ids == ("t_fast",)
    assert result.failed_task_ids == ("t_slow",)
    assert result.skipped_task_ids == ("t_after",)

    r_slow = result.results["t_slow"]
    assert r_slow.error_stage == "timeout"
    assert r_slow.error_type == "TimeoutError"


def test_runtime_validation_distinguishes_binding_input_and_output_errors():
    registry = ToolRegistry(
        [
            ToolSpec(
                name="untyped_source",
                description="Returns dynamic dict",
                func=lambda payload: payload,
            ),
            ToolSpec(
                name="strict_calc",
                description="Strict calculator",
                func=lambda operation, a, b: {"result": a + b},
                input_schema=CalculatorInput,
                output_schema=CalculatorOutput,
            ),
            ToolSpec(
                name="broken_output_tool",
                description="Returns invalid output schema",
                func=lambda country: {"country": country},
                input_schema=PopulationInput,
                output_schema=PopulationOutput,
            ),
        ]
    )

    plan = ExecutionPlan(
        objective="Runtime validation stages",
        tasks=[
            TaskSpec(
                task_id="src_missing_key",
                tool_name="untyped_source",
                description="Missing expected key",
                inputs={"payload": {"other": 123}},
            ),
            TaskSpec(
                task_id="fails_binding",
                tool_name="strict_calc",
                description="Fails binding resolution",
                inputs={
                    "operation": "add",
                    "b": 1.0,
                    "a": InputBinding(task_id="src_missing_key", output_path=["value"]),
                },
            ),
            TaskSpec(
                task_id="src_wrong_type",
                tool_name="untyped_source",
                description="Returns non-float string",
                inputs={"payload": {"value": "not-a-number"}},
            ),
            TaskSpec(
                task_id="fails_input_validation",
                tool_name="strict_calc",
                description="Fails input schema validation",
                inputs={
                    "operation": "add",
                    "b": 1.0,
                    "a": InputBinding(task_id="src_wrong_type", output_path=["value"]),
                },
            ),
            TaskSpec(
                task_id="fails_output_validation",
                tool_name="broken_output_tool",
                description="Fails output schema validation",
                inputs={"country": "Italy"},
            ),
        ],
    )

    dag = PlanCompiler(registry).compile(plan)
    result = asyncio.run(Executor(registry).execute(dag))

    assert result.results["fails_binding"].status == "error"
    assert result.results["fails_binding"].error_stage == "binding"
    assert result.results["fails_binding"].error_type == "BindingResolutionError"

    assert result.results["fails_input_validation"].status == "error"
    assert result.results["fails_input_validation"].error_stage == "input_validation"
    assert result.results["fails_input_validation"].error_type == "ValidationError"

    assert result.results["fails_output_validation"].status == "error"
    assert result.results["fails_output_validation"].error_stage == "output_validation"
    assert result.results["fails_output_validation"].error_type == "ValidationError"
