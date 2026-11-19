import pytest
from pydantic import ValidationError

from langgraph_llm_compiler.planner.schemas import (
    ExecutionPlan,
    InputBinding,
    TaskSpec,
)


def test_valid_execution_plan_with_unified_inputs():
    payload = {
        "objective": "Compute population difference between Italy and France",
        "tasks": [
            {
                "task_id": "task_1",
                "tool_name": "population_lookup",
                "description": "Fetch Italy population",
                "inputs": {"country": "Italy"},
            },
            {
                "task_id": "task_2",
                "tool_name": "population_lookup",
                "description": "Fetch France population",
                "inputs": {"country": "France"},
            },
            {
                "task_id": "task_3",
                "tool_name": "calculator",
                "description": "Subtract France population from Italy",
                "inputs": {
                    "operation": "subtract",
                    "a": {"task_id": "task_1", "output_path": ["population"]},
                    "b": {"task_id": "task_2", "output_path": ["population"]},
                },
            },
        ],
    }

    plan = ExecutionPlan.model_validate(payload)

    assert len(plan.tasks) == 3
    assert plan.tasks[0].static_inputs == {"country": "Italy"}
    assert plan.tasks[0].bindings == {}
    assert plan.tasks[2].static_inputs == {"operation": "subtract"}
    assert plan.tasks[2].bindings == {
        "a": InputBinding(task_id="task_1", output_path=["population"]),
        "b": InputBinding(task_id="task_2", output_path=["population"]),
    }


def test_input_binding_defaults_to_empty_output_path():
    binding = InputBinding(task_id="task_1")
    assert binding.output_path == []

    task = TaskSpec.model_validate(
        {
            "task_id": "task_2",
            "tool_name": "consumer",
            "description": "Consume whole output",
            "inputs": {"x": {"task_id": "task_1"}},
        }
    )
    assert task.bindings["x"] == InputBinding(task_id="task_1", output_path=[])


def test_supports_nested_and_heterogeneous_static_inputs():
    task = TaskSpec(
        task_id="t1",
        tool_name="matrix_ops",
        description="Invert matrix",
        inputs={
            "matrix": [[1, 2], [3, 4]],
            "operation": "inverse",
            "precision": 4,
            "meta": {"source": "manual"},
        },
    )
    assert task.static_inputs["matrix"] == [[1, 2], [3, 4]]
    assert task.static_inputs["meta"] == {"source": "manual"}
    assert task.bindings == {}


def test_strips_whitespace_and_rejects_blank_strings():
    task = TaskSpec(
        task_id="  task_1  ",
        tool_name="  lookup  ",
        description="  Valid description  ",
        inputs={
            "x": InputBinding(task_id="  task_0  ", output_path=["  population  "])
        },
    )
    assert task.task_id == "task_1"
    assert task.tool_name == "lookup"
    assert task.description == "Valid description"
    assert task.bindings["x"].task_id == "task_0"
    assert task.bindings["x"].output_path == ["population"]

    with pytest.raises(ValidationError):
        TaskSpec(task_id="   ", tool_name="lookup", description="ok")

    with pytest.raises(ValidationError):
        InputBinding(task_id="task_1", output_path=["   "])


def test_forbids_extra_and_legacy_fields():
    # Legacy separate fields must be rejected.
    with pytest.raises(ValidationError):
        TaskSpec(
            task_id="task_1",
            tool_name="lookup",
            description="desc",
            dependencies=[],  # type: ignore[call-arg]
        )

    with pytest.raises(ValidationError):
        TaskSpec(
            task_id="task_1",
            tool_name="lookup",
            description="desc",
            arguments={"country": "Italy"},  # type: ignore[call-arg]
        )

    with pytest.raises(ValidationError):
        TaskSpec(
            task_id="task_1",
            tool_name="lookup",
            description="desc",
            input_bindings={},  # type: ignore[call-arg]
        )

    with pytest.raises(ValidationError):
        InputBinding(task_id="task_1", output_path=[], extra_key="nope")  # type: ignore[call-arg]

    with pytest.raises(ValidationError):
        ExecutionPlan(
            objective="obj",
            tasks=[TaskSpec(task_id="t1", tool_name="x", description="d")],
            unknown_field=True,  # type: ignore[call-arg]
        )


def test_requires_at_least_one_task():
    with pytest.raises(ValidationError):
        ExecutionPlan(objective="Empty plan", tasks=[])


def test_models_are_frozen():
    plan = ExecutionPlan(
        objective="Immutability test",
        tasks=[TaskSpec(task_id="t1", tool_name="x", description="d")],
    )

    with pytest.raises(ValidationError):
        plan.objective = "Modified"  # type: ignore[misc]

    with pytest.raises(ValidationError):
        plan.tasks[0].task_id = "t2"  # type: ignore[misc]


def test_json_roundtrip():
    raw_json = """
    {
      "objective": "Comparison",
      "tasks": [
        {
          "task_id": "t1",
          "tool_name": "calc",
          "description": "Divide by 4",
          "inputs": {
            "operation": "divide",
            "a": {"task_id": "t0", "output_path": ["val"]},
            "b": 4
          }
        }
      ]
    }
    """
    plan = ExecutionPlan.model_validate_json(raw_json)
    assert plan.tasks[0].static_inputs == {"operation": "divide", "b": 4}
    assert plan.tasks[0].bindings == {
        "a": InputBinding(task_id="t0", output_path=["val"])
    }

    reloaded = ExecutionPlan.model_validate_json(plan.model_dump_json())
    assert reloaded == plan
