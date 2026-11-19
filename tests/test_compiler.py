from typing import Any, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field

from langgraph_llm_compiler.compiler import (
    CompiledDAG,
    PlanCompilationError,
    PlanCompiler,
    ToolRegistry,
    ToolSpec,
)
from langgraph_llm_compiler.planner import ExecutionPlan, InputBinding, TaskSpec


class PopulationInput(BaseModel):
    country: str = Field(min_length=1)
    year: int | None = None


class PopulationOutput(BaseModel):
    country: str
    population: int


class CalculatorInput(BaseModel):
    operation: Literal["add", "subtract", "multiply", "divide"]
    a: float
    b: float


class CalculatorOutput(BaseModel):
    result: float


class NestedStats(BaseModel):
    median: float


class ReportOutput(BaseModel):
    stats: NestedStats
    metadata: dict[str, Any]


class FlexibleInput(BaseModel):
    model_config = ConfigDict(extra="allow")
    required_key: str


def _build_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ToolSpec(
                name="population_lookup",
                description="Get country population",
                func=lambda country, year=None: {
                    "country": country,
                    "population": 60_000_000,
                },
                input_schema=PopulationInput,
                output_schema=PopulationOutput,
            ),
            ToolSpec(
                name="calculator",
                description="Perform arithmetic (a <op> b)",
                func=lambda operation, a, b: {"result": a - b},
                input_schema=CalculatorInput,
                output_schema=CalculatorOutput,
            ),
        ]
    )


def test_tool_registry_and_planner_catalog_generation():
    registry = _build_registry()

    assert len(registry) == 2
    assert "population_lookup" in registry
    assert registry.names() == ("population_lookup", "calculator")
    assert registry["calculator"].name == "calculator"
    assert registry.get("missing") is None

    catalog = registry.to_planner_catalog()
    assert set(catalog) == {"population_lookup", "calculator"}
    assert "country (str)" in catalog["population_lookup"]
    assert "year (int | None, optional)" in catalog["population_lookup"]
    assert '"population": int' in catalog["population_lookup"]
    assert "'add' | 'subtract' | 'multiply' | 'divide'" in catalog["calculator"]
    assert '"result": float' in catalog["calculator"]


def test_tool_spec_and_registry_validation():
    with pytest.raises(ValueError, match="name"):
        ToolSpec(name="   ", description="ok", func=lambda: None)

    with pytest.raises(ValueError, match="description"):
        ToolSpec(name="ok", description="   ", func=lambda: None)

    with pytest.raises(TypeError, match="callable"):
        ToolSpec(name="ok", description="ok", func="not-callable")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="input_schema"):
        ToolSpec(name="ok", description="ok", func=lambda: None, input_schema=dict)  # type: ignore[arg-type]

    registry = ToolRegistry()
    spec = ToolSpec(name="ping", description="Ping", func=lambda: "pong")
    registry.register(spec)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(spec)

    updated = ToolSpec(name="ping", description="Ping v2", func=lambda: "pong2")
    registry.register(updated, overwrite=True)
    assert registry["ping"].description == "Ping v2"


def test_compile_valid_parallel_plan():
    registry = _build_registry()
    compiler = PlanCompiler(registry)

    plan = ExecutionPlan(
        objective="Compute population difference between Italy and France",
        tasks=[
            TaskSpec(
                task_id="task_1",
                tool_name="population_lookup",
                description="Get Italy population",
                inputs={"country": "Italy"},
            ),
            TaskSpec(
                task_id="task_2",
                tool_name="population_lookup",
                description="Get France population",
                inputs={"country": "France"},
            ),
            TaskSpec(
                task_id="task_3",
                tool_name="calculator",
                description="Subtract France from Italy",
                inputs={
                    "operation": "subtract",
                    "a": InputBinding(task_id="task_1", output_path=["population"]),
                    "b": InputBinding(task_id="task_2", output_path=["population"]),
                },
            ),
        ],
    )

    dag = compiler.compile(plan)

    assert isinstance(dag, CompiledDAG)
    assert dag.objective == plan.objective
    assert dag.topological_order == ("task_1", "task_2", "task_3")
    assert dag.levels == (("task_1", "task_2"), ("task_3",))
    assert dag.root_tasks == ("task_1", "task_2")
    assert dag.leaf_tasks == ("task_3",)
    assert dag.dependencies["task_1"] == frozenset()
    assert dag.dependencies["task_3"] == frozenset({"task_1", "task_2"})
    assert dag.dependents["task_1"] == frozenset({"task_3"})
    assert dag.dependents["task_3"] == frozenset()
    assert dag.get_task("task_2").static_inputs == {"country": "France"}
    assert dag.is_ready("task_1", set()) is True
    assert dag.is_ready("task_3", {"task_1"}) is False
    assert dag.is_ready("task_3", {"task_1", "task_2"}) is True

    # CompiledDAG mappings must be read-only.
    with pytest.raises(TypeError):
        dag.tasks["task_4"] = plan.tasks[0]  # type: ignore[index]
    with pytest.raises(TypeError):
        dag.dependencies["task_1"] = frozenset()  # type: ignore[index]


def test_compile_multi_level_diamond_dag():
    registry = ToolRegistry(
        [ToolSpec(name="noop", description="No-op tool", func=lambda **kw: kw)]
    )
    compiler = PlanCompiler(registry)

    plan = ExecutionPlan(
        objective="Multi-wave DAG",
        tasks=[
            TaskSpec(task_id="t1", tool_name="noop", description="root 1"),
            TaskSpec(task_id="t2", tool_name="noop", description="root 2"),
            TaskSpec(
                task_id="t3",
                tool_name="noop",
                description="depends on t1",
                inputs={"x": InputBinding(task_id="t1")},
            ),
            TaskSpec(
                task_id="t4",
                tool_name="noop",
                description="depends on t1 and t3 (must wait for level 1)",
                inputs={
                    "x": InputBinding(task_id="t1"),
                    "y": InputBinding(task_id="t3"),
                },
            ),
            TaskSpec(
                task_id="t5",
                tool_name="noop",
                description="depends on t2 and t4",
                inputs={
                    "x": InputBinding(task_id="t2"),
                    "y": InputBinding(task_id="t4"),
                },
            ),
        ],
    )

    dag = compiler.compile(plan)

    assert dag.levels == (("t1", "t2"), ("t3",), ("t4",), ("t5",))
    assert dag.topological_order == ("t1", "t2", "t3", "t4", "t5")
    assert dag.leaf_tasks == ("t5",)


def test_compile_mixed_static_constant_and_dynamic_binding():
    """Verify static numeric constants (e.g. b=4) and bindings coexist cleanly in inputs."""
    compiler = PlanCompiler(_build_registry())

    plan = ExecutionPlan(
        objective="Divide population by static constant 4",
        tasks=[
            TaskSpec(
                task_id="task_1",
                tool_name="population_lookup",
                description="Italy",
                inputs={"country": "Italy"},
            ),
            TaskSpec(
                task_id="task_2",
                tool_name="calculator",
                description="Divide by 4",
                inputs={
                    "operation": "divide",
                    "a": InputBinding(task_id="task_1", output_path=["population"]),
                    "b": 4,
                },
            ),
        ],
    )

    dag = compiler.compile(plan)
    assert dag.levels == (("task_1",), ("task_2",))
    assert dag.get_task("task_2").static_inputs == {"operation": "divide", "b": 4}
    assert dag.get_task("task_2").bindings == {
        "a": InputBinding(task_id="task_1", output_path=["population"])
    }


def test_detects_duplicate_task_ids():
    compiler = PlanCompiler(_build_registry())
    plan = ExecutionPlan(
        objective="Duplicate IDs",
        tasks=[
            TaskSpec(
                task_id="task_1",
                tool_name="population_lookup",
                description="Italy",
                inputs={"country": "Italy"},
            ),
            TaskSpec(
                task_id="task_1",
                tool_name="population_lookup",
                description="France",
                inputs={"country": "France"},
            ),
        ],
    )

    with pytest.raises(PlanCompilationError) as exc_info:
        compiler.compile(plan)

    assert [i.code for i in exc_info.value.issues] == ["duplicate_task_id"]


def test_detects_unknown_tool():
    compiler = PlanCompiler(_build_registry())
    plan = ExecutionPlan(
        objective="Unknown tool",
        tasks=[
            TaskSpec(
                task_id="task_1",
                tool_name="web_search",
                description="Search",
                inputs={"query": "Italy"},
            )
        ],
    )

    with pytest.raises(PlanCompilationError) as exc_info:
        compiler.compile(plan)

    assert [i.code for i in exc_info.value.issues] == ["unknown_tool"]


def test_detects_missing_and_self_dependencies():
    compiler = PlanCompiler(_build_registry())
    plan = ExecutionPlan(
        objective="Broken references",
        tasks=[
            TaskSpec(
                task_id="task_1",
                tool_name="calculator",
                description="Broken",
                inputs={
                    "operation": "add",
                    "a": InputBinding(task_id="task_1", output_path=["result"]),
                    "b": InputBinding(task_id="task_99", output_path=["result"]),
                },
            )
        ],
    )

    with pytest.raises(PlanCompilationError) as exc_info:
        compiler.compile(plan)

    codes = {(i.code, i.field) for i in exc_info.value.issues}
    assert codes == {
        ("self_dependency", "a"),
        ("missing_dependency", "b"),
    }


def test_detects_cycles_with_path():
    registry = ToolRegistry(
        [ToolSpec(name="step", description="Step", func=lambda **kw: kw)]
    )
    compiler = PlanCompiler(registry)

    plan = ExecutionPlan(
        objective="Cyclic plan",
        tasks=[
            TaskSpec(
                task_id="t1",
                tool_name="step",
                description="t1",
                inputs={"x": InputBinding(task_id="t3")},
            ),
            TaskSpec(
                task_id="t2",
                tool_name="step",
                description="t2",
                inputs={"x": InputBinding(task_id="t1")},
            ),
            TaskSpec(
                task_id="t3",
                tool_name="step",
                description="t3",
                inputs={"x": InputBinding(task_id="t2")},
            ),
            TaskSpec(
                task_id="t4",
                tool_name="step",
                description="downstream of cycle",
                inputs={"x": InputBinding(task_id="t3")},
            ),
        ],
    )

    with pytest.raises(PlanCompilationError) as exc_info:
        compiler.compile(plan)

    assert len(exc_info.value.issues) == 1
    issue = exc_info.value.issues[0]
    assert issue.code == "cycle_detected"
    assert "t1" in issue.message and "t2" in issue.message and "t3" in issue.message


def test_validates_input_schemas_and_static_values():
    registry = _build_registry()
    registry.register(
        ToolSpec(
            name="flexible",
            description="Allows extra kwargs",
            func=lambda **kw: kw,
            input_schema=FlexibleInput,
        )
    )
    compiler = PlanCompiler(registry)

    # Flexible tool with extra="allow" accepts extra inputs.
    ok_dag = compiler.compile(
        ExecutionPlan(
            objective="Extra allowed",
            tasks=[
                TaskSpec(
                    task_id="t1",
                    tool_name="flexible",
                    description="Flexible call",
                    inputs={"required_key": "ok", "extra_key": 42},
                )
            ],
        )
    )
    assert ok_dag.topological_order == ("t1",)

    # Missing required param, unknown param, and invalid Literal static value.
    bad_plan = ExecutionPlan(
        objective="Invalid schema usage",
        tasks=[
            TaskSpec(
                task_id="t1",
                tool_name="population_lookup",
                description="Valid lookup",
                inputs={"country": "Italy"},
            ),
            TaskSpec(
                task_id="t2",
                tool_name="calculator",
                description="Bad calculator call",
                inputs={
                    "operation": "modulo",
                    "unused": 123,
                    "a": InputBinding(task_id="t1", output_path=["population"]),
                },
            ),
        ],
    )

    with pytest.raises(PlanCompilationError) as exc_info:
        compiler.compile(bad_plan)

    issues = {(i.code, i.task_id, i.field) for i in exc_info.value.issues}
    assert issues == {
        ("missing_required_argument", "t2", "b"),
        ("unknown_argument", "t2", "unused"),
        ("invalid_static_argument", "t2", "operation"),
    }


def test_validates_nested_and_invalid_output_paths():
    registry = ToolRegistry(
        [
            ToolSpec(
                name="report",
                description="Generate report",
                func=lambda: None,
                output_schema=ReportOutput,
            ),
            ToolSpec(
                name="consumer",
                description="Consume value",
                func=lambda x: x,
            ),
        ]
    )
    compiler = PlanCompiler(registry)

    valid_plan = ExecutionPlan(
        objective="Valid nested paths",
        tasks=[
            TaskSpec(task_id="t1", tool_name="report", description="Report"),
            TaskSpec(
                task_id="t2",
                tool_name="consumer",
                description="Read nested model and dict",
                inputs={
                    "m": InputBinding(task_id="t1", output_path=["stats", "median"]),
                    "k": InputBinding(
                        task_id="t1", output_path=["metadata", "any_key", "deep"]
                    ),
                },
            ),
        ],
    )
    dag = compiler.compile(valid_plan)
    assert dag.levels == (("t1",), ("t2",))

    invalid_plan = ExecutionPlan(
        objective="Invalid nested paths",
        tasks=[
            TaskSpec(task_id="t1", tool_name="report", description="Report"),
            TaskSpec(
                task_id="t2",
                tool_name="consumer",
                description="Invalid paths",
                inputs={
                    "bad_field": InputBinding(
                        task_id="t1", output_path=["stats", "mean"]
                    ),
                    "into_scalar": InputBinding(
                        task_id="t1", output_path=["stats", "median", "sub"]
                    ),
                },
            ),
        ],
    )

    with pytest.raises(PlanCompilationError) as exc_info:
        compiler.compile(invalid_plan)

    issues = {(i.code, i.task_id, i.field) for i in exc_info.value.issues}
    assert issues == {
        ("invalid_output_path", "t2", "bad_field"),
        ("invalid_output_path", "t2", "into_scalar"),
    }


def test_collects_all_errors_in_single_pass():
    compiler = PlanCompiler(_build_registry())

    broken_plan = ExecutionPlan(
        objective="Everything broken at once",
        tasks=[
            TaskSpec(
                task_id="t1",
                tool_name="population_lookup",
                description="Italy",
                inputs={
                    "country": "Italy",
                    "year": InputBinding(task_id="t2"),
                },
            ),
            TaskSpec(
                task_id="t2",
                tool_name="calculator",
                description="Invalid op, bad path, missing dep, cyclic",
                inputs={
                    "operation": "modulo",
                    "a": InputBinding(task_id="t1", output_path=["wrong_field"]),
                    "b": InputBinding(task_id="missing_task"),
                },
            ),
            TaskSpec(
                task_id="t2",
                tool_name="nonexistent_tool",
                description="Duplicate ID and unknown tool",
            ),
        ],
    )

    with pytest.raises(PlanCompilationError) as exc_info:
        compiler.compile(broken_plan)

    codes = {i.code for i in exc_info.value.issues}
    assert codes == {
        "duplicate_task_id",
        "unknown_tool",
        "invalid_static_argument",
        "missing_dependency",
        "invalid_output_path",
        "cycle_detected",
    }


def test_detects_malformed_binding_with_clear_diagnostic():
    """Regression test for typos like 'output, path' in a binding dict."""
    registry = _build_registry()
    registry.register(
        ToolSpec(name="untyped_consumer", description="No schema", func=lambda x: x)
    )
    compiler = PlanCompiler(registry)

    plan = ExecutionPlan.model_validate(
        {
            "objective": "Malformed binding regression",
            "tasks": [
                {
                    "task_id": "task_1",
                    "tool_name": "population_lookup",
                    "description": "Italy",
                    "inputs": {"country": "Italy"},
                },
                {
                    "task_id": "task_2",
                    "tool_name": "calculator",
                    "description": "Typo in output_path key",
                    "inputs": {
                        "operation": "subtract",
                        "a": {"task_id": "task_1", "output_path": ["population"]},
                        "b": {"task_id": "task_1", "output, path": ["population"]},
                    },
                },
                {
                    "task_id": "task_3",
                    "tool_name": "untyped_consumer",
                    "description": "Malformed binding on unschema'd tool",
                    "inputs": {
                        "x": {"task_id": "   ", "output_path": ["population"]},
                    },
                },
            ],
        }
    )

    with pytest.raises(PlanCompilationError) as exc_info:
        compiler.compile(plan)

    issues = [(i.code, i.task_id, i.field) for i in exc_info.value.issues]
    assert issues == [
        ("malformed_binding", "task_2", "b"),
        ("malformed_binding", "task_3", "x"),
    ]
    assert "output, path" in exc_info.value.issues[0].message

