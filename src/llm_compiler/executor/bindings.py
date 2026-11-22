import copy
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from llm_compiler.planner.schemas import InputBinding, TaskSpec

from .results import TaskExecutionResult


class BindingResolutionError(RuntimeError):
    """Raised when a task input binding or output_path cannot be resolved at runtime."""


def resolve_output_path(output: Any, output_path: list[str]) -> Any:
    """Traverse a sequence of string keys into an upstream task output."""
    current = output

    for idx, key in enumerate(output_path):
        traversed = ".".join(output_path[: idx + 1])

        if isinstance(current, Mapping):
            if key not in current:
                raise BindingResolutionError(
                    f"Key '{key}' not found while traversing output_path '{traversed}'."
                )
            current = current[key]
            continue

        if isinstance(current, BaseModel):
            if key not in type(current).model_fields and not hasattr(current, key):
                raise BindingResolutionError(
                    f"Field '{key}' not found on {type(current).__name__} "
                    f"while traversing output_path '{traversed}'."
                )
            current = getattr(current, key)
            continue

        raise BindingResolutionError(
            f"Cannot traverse key '{key}' at output_path '{traversed}' "
            f"on value of type '{type(current).__name__}'."
        )

    return current


def resolve_task_inputs(
    task: TaskSpec,
    task_results: Mapping[str, TaskExecutionResult],
) -> dict[str, Any]:
    """Resolve static values and dynamic InputBindings from task.inputs into tool kwargs."""
    resolved: dict[str, Any] = {}

    for param_name, value in task.inputs.items():
        if not isinstance(value, InputBinding):
            resolved[param_name] = copy.deepcopy(value)
            continue

        upstream = task_results.get(value.task_id)
        if upstream is None:
            raise BindingResolutionError(
                f"Task '{task.task_id}' parameter '{param_name}' references upstream "
                f"task '{value.task_id}', which has no recorded result."
            )

        if upstream.status != "success":
            raise BindingResolutionError(
                f"Task '{task.task_id}' parameter '{param_name}' cannot read from "
                f"upstream task '{value.task_id}' with status '{upstream.status}'."
            )

        try:
            resolved[param_name] = resolve_output_path(
                upstream.output, value.output_path
            )
        except BindingResolutionError as exc:
            raise BindingResolutionError(
                f"Task '{task.task_id}' parameter '{param_name}' failed to resolve "
                f"binding from '{value.task_id}': {exc}"
            ) from exc

    return resolved
