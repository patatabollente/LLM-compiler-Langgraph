import json
from collections.abc import Mapping
from types import UnionType
from typing import Annotated, Any, Union, get_args, get_origin

from pydantic import BaseModel, TypeAdapter, ValidationError

from llm_compiler.planner.schemas import ExecutionPlan, InputBinding, TaskSpec

from .dag import CompilationIssue, CompiledDAG, PlanCompilationError
from .registry import ToolRegistry, ToolSpec


def _unwrap_optional_models(annotation: Any) -> tuple[type[BaseModel], ...] | None:
    """Return BaseModel subclasses if annotation is a BaseModel or Union of BaseModels/None."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return (annotation,)

    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        non_none = [arg for arg in get_args(annotation) if arg is not type(None)]
        if non_none and all(
            isinstance(arg, type) and issubclass(arg, BaseModel) for arg in non_none
        ):
            return tuple(non_none)

    return None


def _unwrap_mapping_value_type(annotation: Any) -> tuple[bool, Any]:
    """Check if annotation represents a mapping type and return (is_mapping, value_type)."""
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        non_none = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(non_none) == 1:
            return _unwrap_mapping_value_type(non_none[0])
        return False, None

    if annotation is dict or annotation is Mapping:
        return True, Any

    if origin is dict or (isinstance(origin, type) and issubclass(origin, Mapping)):
        args = get_args(annotation)
        value_type = args[1] if len(args) >= 2 else Any
        return True, value_type

    return False, None


class PlanCompiler:
    """Validates an ExecutionPlan against a ToolRegistry and compiles it into a CompiledDAG."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def compile(self, plan: ExecutionPlan) -> CompiledDAG:
        issues: list[CompilationIssue] = []

        # 1. Check task_id uniqueness while building task lookup.
        tasks_by_id: dict[str, TaskSpec] = {}
        for task in plan.tasks:
            if task.task_id in tasks_by_id:
                issues.append(
                    CompilationIssue(
                        code="duplicate_task_id",
                        task_id=task.task_id,
                        field="task_id",
                        message=f"Duplicate task_id '{task.task_id}' in execution plan.",
                    )
                )
            else:
                tasks_by_id[task.task_id] = task

        # 2-4 & 6. Validate each task in a single pass.
        for task in plan.tasks:
            tool_spec = self._registry.get(task.tool_name)
            if tool_spec is None:
                issues.append(
                    CompilationIssue(
                        code="unknown_tool",
                        task_id=task.task_id,
                        field="tool_name",
                        message=f"Tool '{task.tool_name}' is not registered in ToolRegistry.",
                    )
                )

            malformed_params = self._validate_malformed_bindings(
                task, tasks_by_id, issues
            )
            self._validate_bindings(task, tasks_by_id, issues)

            if tool_spec is not None:
                self._validate_input_schema(
                    task, tool_spec, issues, skip_static_params=malformed_params
                )

        # 5. Build dependency graph on valid edges and detect cycles / compute levels.
        dependencies: dict[str, set[str]] = {tid: set() for tid in tasks_by_id}
        dependents: dict[str, set[str]] = {tid: set() for tid in tasks_by_id}

        for task_id, task in tasks_by_id.items():
            for binding in task.bindings.values():
                upstream_id = binding.task_id
                if upstream_id in tasks_by_id and upstream_id != task_id:
                    dependencies[task_id].add(upstream_id)
                    dependents[upstream_id].add(task_id)

        topological_order, levels, cycles = self._compute_schedule(
            tasks_by_id, dependencies, dependents
        )
        for cycle in cycles:
            issues.append(
                CompilationIssue(
                    code="cycle_detected",
                    task_id=cycle[0],
                    message=f"Circular dependency detected: {' -> '.join(cycle)}.",
                )
            )

        if issues:
            raise PlanCompilationError(issues)

        return CompiledDAG(
            objective=plan.objective,
            tasks=tasks_by_id,
            dependencies={tid: frozenset(deps) for tid, deps in dependencies.items()},
            dependents={tid: frozenset(deps) for tid, deps in dependents.items()},
            topological_order=topological_order,
            levels=levels,
        )

    @staticmethod
    def _looks_like_nested_binding(
        value: Any,
        tasks_by_id: Mapping[str, TaskSpec],
    ) -> bool:
        if isinstance(value, InputBinding):
            return True
        if not isinstance(value, Mapping) or "task_id" not in value:
            return False
        if "output_path" in value:
            return True
        raw_task_id = value.get("task_id")
        return isinstance(raw_task_id, str) and raw_task_id.strip() in tasks_by_id

    @classmethod
    def _find_nested_bindings(
        cls,
        value: Any,
        tasks_by_id: Mapping[str, TaskSpec],
        *,
        prefix: str = "",
    ) -> list[tuple[str, Any]]:
        """Recursively locate InputBinding-like objects nested inside static containers."""
        found: list[tuple[str, Any]] = []
        if isinstance(value, Mapping):
            for key, item in value.items():
                child_path = f"{prefix}.{key}" if prefix else str(key)
                if cls._looks_like_nested_binding(item, tasks_by_id):
                    found.append((child_path, item))
                else:
                    found.extend(
                        cls._find_nested_bindings(
                            item, tasks_by_id, prefix=child_path
                        )
                    )
        elif isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                child_path = f"{prefix}[{idx}]"
                if cls._looks_like_nested_binding(item, tasks_by_id):
                    found.append((child_path, item))
                else:
                    found.extend(
                        cls._find_nested_bindings(
                            item, tasks_by_id, prefix=child_path
                        )
                    )
        return found

    @classmethod
    def _validate_malformed_bindings(
        cls,
        task: TaskSpec,
        tasks_by_id: Mapping[str, TaskSpec],
        issues: list[CompilationIssue],
    ) -> set[str]:
        malformed: set[str] = set()
        for param_name, value in task.static_inputs.items():
            if isinstance(value, Mapping) and "task_id" in value:
                try:
                    InputBinding.model_validate(value)
                except ValidationError as exc:
                    malformed.add(param_name)
                    details = "; ".join(
                        f"{'.'.join(str(p) for p in err['loc']) or 'binding'}: {err['msg']}"
                        for err in exc.errors()
                    )
                    issues.append(
                        CompilationIssue(
                            code="malformed_binding",
                            task_id=task.task_id,
                            field=param_name,
                            message=(
                                f"Parameter '{param_name}' looks like a malformed "
                                "InputBinding (allowed keys: 'task_id', 'output_path'): "
                                f"{details}."
                            ),
                        )
                    )
                continue

            nested_bindings = cls._find_nested_bindings(value, tasks_by_id)
            if nested_bindings:
                malformed.add(param_name)
                if len(nested_bindings) == 1:
                    path, nested_obj = nested_bindings[0]
                    raw_obj = (
                        nested_obj.model_dump()
                        if isinstance(nested_obj, InputBinding)
                        else dict(nested_obj)
                    )
                    try:
                        concrete_binding = json.dumps(raw_obj, ensure_ascii=False)
                    except TypeError:
                        concrete_binding = repr(raw_obj)
                    message = (
                        f"Parameter '{param_name}' wraps an InputBinding inside "
                        f"'{path}'. InputBinding objects cannot be nested inside "
                        f"dicts or lists; set '{param_name}' directly to "
                        f"{concrete_binding} instead of wrapping it under '{path}'."
                    )
                else:
                    locations = ", ".join(f"'{p}'" for p, _ in nested_bindings)
                    message = (
                        f"Parameter '{param_name}' contains nested InputBinding "
                        f"objects at {locations}. InputBinding objects cannot be "
                        "nested inside dicts or lists; bind each upstream output "
                        "directly to a top-level tool parameter."
                    )
                issues.append(
                    CompilationIssue(
                        code="nested_binding",
                        task_id=task.task_id,
                        field=param_name,
                        message=message,
                    )
                )
        return malformed

    def _validate_bindings(
        self,
        task: TaskSpec,
        tasks_by_id: Mapping[str, TaskSpec],
        issues: list[CompilationIssue],
    ) -> None:
        for param_name, binding in task.bindings.items():
            if binding.task_id == task.task_id:
                issues.append(
                    CompilationIssue(
                        code="self_dependency",
                        task_id=task.task_id,
                        field=param_name,
                        message=(
                            f"Parameter '{param_name}' cannot reference its own task "
                            f"'{task.task_id}'."
                        ),
                    )
                )
                continue

            upstream_task = tasks_by_id.get(binding.task_id)
            if upstream_task is None:
                issues.append(
                    CompilationIssue(
                        code="missing_dependency",
                        task_id=task.task_id,
                        field=param_name,
                        message=(
                            f"Parameter '{param_name}' references unknown task "
                            f"'{binding.task_id}'."
                        ),
                    )
                )
                continue

            upstream_tool = self._registry.get(upstream_task.tool_name)
            if (
                upstream_tool is not None
                and upstream_tool.output_schema is not None
                and binding.output_path
            ):
                self._validate_output_path(
                    task=task,
                    param_name=param_name,
                    upstream_task_id=upstream_task.task_id,
                    upstream_schema=upstream_tool.output_schema,
                    output_path=binding.output_path,
                    issues=issues,
                )

    @staticmethod
    def _validate_input_schema(
        task: TaskSpec,
        tool_spec: ToolSpec,
        issues: list[CompilationIssue],
        *,
        skip_static_params: set[str] | None = None,
    ) -> None:
        schema = tool_spec.input_schema
        if schema is None:
            return

        skipped = skip_static_params or set()
        model_fields = schema.model_fields
        static_inputs = task.static_inputs
        bindings = task.bindings
        provided_params = set(task.inputs)

        missing_required = [
            fname
            for fname, finfo in model_fields.items()
            if finfo.is_required() and fname not in provided_params
        ]
        for fname in missing_required:
            issues.append(
                CompilationIssue(
                    code="missing_required_argument",
                    task_id=task.task_id,
                    field=fname,
                    message=(
                        f"Missing required parameter '{fname}' for tool "
                        f"'{tool_spec.name}'."
                    ),
                )
            )

        allow_extra = schema.model_config.get("extra") == "allow"
        unknown_params: list[str] = []
        if not allow_extra:
            unknown_params = sorted(provided_params - set(model_fields))
            for pname in unknown_params:
                issues.append(
                    CompilationIssue(
                        code="unknown_argument",
                        task_id=task.task_id,
                        field=pname,
                        message=(
                            f"Unknown parameter '{pname}' for tool '{tool_spec.name}'."
                        ),
                    )
                )

        # If all inputs are static and parameter names are valid, validate the whole schema
        # (including any model-level validators).
        if not bindings and not skipped and not missing_required and not unknown_params:
            try:
                schema.model_validate(static_inputs)
            except ValidationError as exc:
                for err in exc.errors():
                    loc = ".".join(str(part) for part in err.get("loc", ())) or None
                    issues.append(
                        CompilationIssue(
                            code="invalid_static_argument",
                            task_id=task.task_id,
                            field=loc,
                            message=(
                                f"Invalid static value for tool '{tool_spec.name}': "
                                f"{err['msg']}."
                            ),
                        )
                    )
            return

        # Otherwise validate each known static argument individually against its field type.
        for pname, value in static_inputs.items():
            if pname in skipped:
                continue
            finfo = model_fields.get(pname)
            if finfo is None or finfo.annotation is None:
                continue

            field_type: Any = (
                Annotated[(finfo.annotation, *finfo.metadata)]
                if finfo.metadata
                else finfo.annotation
            )
            try:
                TypeAdapter(field_type).validate_python(value)
            except ValidationError as exc:
                first_msg = exc.errors()[0]["msg"] if exc.errors() else str(exc)
                issues.append(
                    CompilationIssue(
                        code="invalid_static_argument",
                        task_id=task.task_id,
                        field=pname,
                        message=(
                            f"Invalid static value for parameter '{pname}' of tool "
                            f"'{tool_spec.name}': {first_msg}."
                        ),
                    )
                )

    @staticmethod
    def _validate_output_path(
        task: TaskSpec,
        param_name: str,
        upstream_task_id: str,
        upstream_schema: type[BaseModel],
        output_path: list[str],
        issues: list[CompilationIssue],
    ) -> None:
        current_type: Any = upstream_schema

        for idx, segment in enumerate(output_path):
            if current_type is Any:
                return

            models = _unwrap_optional_models(current_type)
            if models is not None:
                matching_annotations = [
                    model.model_fields[segment].annotation
                    for model in models
                    if segment in model.model_fields
                ]
                if not matching_annotations:
                    traversed = ".".join(output_path[: idx + 1])
                    issues.append(
                        CompilationIssue(
                            code="invalid_output_path",
                            task_id=task.task_id,
                            field=param_name,
                            message=(
                                f"Invalid output_path '{traversed}' on upstream task "
                                f"'{upstream_task_id}': field '{segment}' does not exist."
                            ),
                        )
                    )
                    return

                current_type = (
                    matching_annotations[0]
                    if len(matching_annotations) == 1
                    else Union[tuple(matching_annotations)]  # noqa: UP007
                )
                continue

            is_mapping, value_type = _unwrap_mapping_value_type(current_type)
            if is_mapping:
                current_type = value_type
                continue

            traversed = ".".join(output_path[: idx + 1])
            issues.append(
                CompilationIssue(
                    code="invalid_output_path",
                    task_id=task.task_id,
                    field=param_name,
                    message=(
                        f"Invalid output_path '{traversed}' on upstream task "
                        f"'{upstream_task_id}': cannot traverse key '{segment}' into "
                        f"non-mapping type '{current_type}'."
                    ),
                )
            )
            return

    @staticmethod
    def _compute_schedule(
        tasks_by_id: Mapping[str, TaskSpec],
        dependencies: Mapping[str, set[str]],
        dependents: Mapping[str, set[str]],
    ) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...], list[tuple[str, ...]]]:
        order_index = {tid: idx for idx, tid in enumerate(tasks_by_id)}
        in_degree = {tid: len(dependencies[tid]) for tid in tasks_by_id}

        current_level = [tid for tid in tasks_by_id if in_degree[tid] == 0]
        topological_order: list[str] = []
        levels: list[tuple[str, ...]] = []

        while current_level:
            levels.append(tuple(current_level))
            topological_order.extend(current_level)

            next_level: list[str] = []
            for tid in current_level:
                for dep_tid in sorted(dependents[tid], key=order_index.__getitem__):
                    in_degree[dep_tid] -= 1
                    if in_degree[dep_tid] == 0:
                        next_level.append(dep_tid)

            next_level.sort(key=order_index.__getitem__)
            current_level = next_level

        if len(topological_order) == len(tasks_by_id):
            return tuple(topological_order), tuple(levels), []

        visited_in_topo = set(topological_order)
        remaining = [tid for tid in tasks_by_id if tid not in visited_in_topo]
        cycles = PlanCompiler._find_cycles(remaining, dependencies, order_index)
        return tuple(topological_order), tuple(levels), cycles

    @staticmethod
    def _find_cycles(
        nodes: list[str],
        dependencies: Mapping[str, set[str]],
        order_index: Mapping[str, int],
    ) -> list[tuple[str, ...]]:
        node_set = set(nodes)
        state: dict[str, int] = {node: 0 for node in nodes}  # 0=unvisited, 1=visiting, 2=done
        stack: list[str] = []
        stack_pos: dict[str, int] = {}
        cycles: list[tuple[str, ...]] = []
        seen_cycle_keys: set[tuple[str, ...]] = set()

        def dfs(node: str) -> None:
            state[node] = 1
            stack_pos[node] = len(stack)
            stack.append(node)

            for dep in sorted(
                (d for d in dependencies[node] if d in node_set),
                key=order_index.__getitem__,
            ):
                if state[dep] == 0:
                    dfs(dep)
                elif state[dep] == 1:
                    cycle = tuple(stack[stack_pos[dep] :] + [dep])
                    # Canonicalize cycle rotation for deduplication.
                    core = cycle[:-1]
                    min_idx = min(range(len(core)), key=lambda i: core[i])
                    canonical = core[min_idx:] + core[:min_idx]
                    if canonical not in seen_cycle_keys:
                        seen_cycle_keys.add(canonical)
                        cycles.append(cycle)

            stack.pop()
            del stack_pos[node]
            state[node] = 2

        for node in nodes:
            if state[node] == 0:
                dfs(node)

        return cycles
