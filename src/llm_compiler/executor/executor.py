import asyncio
import inspect
import time
from collections import deque
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, ValidationError

from llm_compiler.compiler.dag import CompiledDAG
from llm_compiler.compiler.registry import ToolRegistry
from llm_compiler.planner.schemas import TaskSpec

from .bindings import BindingResolutionError, resolve_task_inputs
from .results import ErrorStage, ExecutionResult, TaskExecutionResult


def _is_async_callable(func: Callable[..., Any]) -> bool:
    return inspect.iscoroutinefunction(func)


def _extract_validated_kwargs(model: BaseModel) -> dict[str, Any]:
    kwargs = {name: getattr(model, name) for name in type(model).model_fields}
    if model.model_extra:
        kwargs.update(model.model_extra)
    return kwargs


class Executor:
    """Event-driven asynchronous executor for CompiledDAG artifacts.

    Notes on timeouts:
        For async tools (`async def`), `task_timeout_s` cancels the awaiting coroutine
        via `asyncio.wait_for`. For synchronous tools (`def`) executed via
        `asyncio.to_thread`, timing out stops awaiting the worker thread and records
        a timeout error, but Python cannot forcefully terminate an already-running
        OS thread; the thread may continue running in the background until it returns.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        max_concurrency: int | None = 8,
        task_timeout_s: float | None = None,
    ) -> None:
        if max_concurrency is not None and max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1 or None.")
        if task_timeout_s is not None and task_timeout_s <= 0:
            raise ValueError("task_timeout_s must be > 0 or None.")

        self._registry = registry
        self._max_concurrency = max_concurrency
        self._task_timeout_s = task_timeout_s

    async def execute(self, dag: CompiledDAG) -> ExecutionResult:
        """Execute a CompiledDAG using an event-driven ready-queue scheduler."""
        t0 = time.perf_counter()
        order_index = {tid: idx for idx, tid in enumerate(dag.topological_order)}

        results: dict[str, TaskExecutionResult] = {}
        succeeded: set[str] = set()
        scheduled_or_done: set[str] = set(dag.root_tasks)
        ready_queue: deque[str] = deque(dag.root_tasks)
        running: dict[asyncio.Task[TaskExecutionResult], str] = {}

        def mark_descendants_skipped(failed_task_id: str) -> None:
            queue: deque[tuple[str, str, str]] = deque(
                (dep_id, failed_task_id, "failed")
                for dep_id in sorted(
                    dag.dependents[failed_task_id], key=order_index.__getitem__
                )
            )
            while queue:
                dep_id, cause_id, cause_verb = queue.popleft()
                if dep_id in results:
                    continue

                dep_task = dag.get_task(dep_id)
                results[dep_id] = TaskExecutionResult(
                    task_id=dep_id,
                    tool_name=dep_task.tool_name,
                    status="skipped",
                    skip_reason=(
                        f"Skipped because upstream task '{cause_id}' {cause_verb}."
                    ),
                    duration_ms=0.0,
                    started_at_ms=None,
                    finished_at_ms=None,
                )
                scheduled_or_done.add(dep_id)

                for next_dep in sorted(
                    dag.dependents[dep_id], key=order_index.__getitem__
                ):
                    if next_dep not in results:
                        queue.append((next_dep, dep_id, "was skipped"))

        def launch_ready() -> None:
            while ready_queue and (
                self._max_concurrency is None or len(running) < self._max_concurrency
            ):
                task_id = ready_queue.popleft()
                task_spec = dag.get_task(task_id)
                async_task = asyncio.create_task(
                    self._run_single_task(task_spec, results, t0)
                )
                running[async_task] = task_id

        launch_ready()

        while running:
            done, _ = await asyncio.wait(
                running.keys(),
                return_when=asyncio.FIRST_COMPLETED,
            )
            done_sorted = sorted(done, key=lambda t: order_index[running[t]])

            for finished_task in done_sorted:
                task_id = running.pop(finished_task)
                task_result = finished_task.result()
                results[task_id] = task_result

                if task_result.status == "success":
                    succeeded.add(task_id)
                    for dep_id in sorted(
                        dag.dependents[task_id], key=order_index.__getitem__
                    ):
                        if dep_id not in scheduled_or_done and dag.is_ready(
                            dep_id, succeeded
                        ):
                            scheduled_or_done.add(dep_id)
                            ready_queue.append(dep_id)
                else:
                    mark_descendants_skipped(task_id)

            launch_ready()

        total_duration_ms = (time.perf_counter() - t0) * 1000.0
        ordered_results = {tid: results[tid] for tid in dag.topological_order}

        return ExecutionResult(
            objective=dag.objective,
            results=ordered_results,
            total_duration_ms=total_duration_ms,
            max_concurrency=self._max_concurrency,
        )

    async def _run_single_task(
        self,
        task: TaskSpec,
        task_results: Mapping[str, TaskExecutionResult],
        t0: float,
    ) -> TaskExecutionResult:
        start_t = time.perf_counter()
        started_at_ms = (start_t - t0) * 1000.0

        def build_error(
            *,
            stage: ErrorStage,
            error_type: str,
            message: str,
        ) -> TaskExecutionResult:
            end_t = time.perf_counter()
            return TaskExecutionResult(
                task_id=task.task_id,
                tool_name=task.tool_name,
                status="error",
                error=message,
                error_type=error_type,
                error_stage=stage,
                duration_ms=(end_t - start_t) * 1000.0,
                started_at_ms=started_at_ms,
                finished_at_ms=(end_t - t0) * 1000.0,
            )

        tool_spec = self._registry.get(task.tool_name)
        if tool_spec is None:
            return build_error(
                stage="tool_execution",
                error_type="KeyError",
                message=f"Tool '{task.tool_name}' is not registered in ToolRegistry.",
            )

        # 1-3. Combine static arguments and resolved input_bindings.
        try:
            resolved_inputs = resolve_task_inputs(task, task_results)
        except BindingResolutionError as exc:
            return build_error(
                stage="binding",
                error_type=type(exc).__name__,
                message=str(exc),
            )

        # 4. Validate complete inputs against input_schema (when declared).
        if tool_spec.input_schema is not None:
            try:
                validated_input = tool_spec.input_schema.model_validate(resolved_inputs)
            except ValidationError as exc:
                return build_error(
                    stage="input_validation",
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            call_kwargs = _extract_validated_kwargs(validated_input)
        else:
            call_kwargs = resolved_inputs

        # 5. Invoke the tool (await async def / to_thread for sync def) with optional timeout.
        try:
            if _is_async_callable(tool_spec.func):
                coro = tool_spec.func(**call_kwargs)
            else:
                coro = asyncio.to_thread(tool_spec.func, **call_kwargs)

            if self._task_timeout_s is not None:
                raw_output = await asyncio.wait_for(coro, timeout=self._task_timeout_s)
            else:
                raw_output = await coro
        except TimeoutError:
            return build_error(
                stage="timeout",
                error_type="TimeoutError",
                message=(
                    f"Task '{task.task_id}' (tool '{task.tool_name}') timed out "
                    f"after {self._task_timeout_s}s."
                ),
            )
        except Exception as exc:  # noqa: BLE001
            return build_error(
                stage="tool_execution",
                error_type=type(exc).__name__,
                message=str(exc) or type(exc).__name__,
            )

        # 6. Validate and normalize output against output_schema (when declared).
        if tool_spec.output_schema is not None:
            try:
                validated_output = tool_spec.output_schema.model_validate(raw_output)
            except ValidationError as exc:
                return build_error(
                    stage="output_validation",
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            normalized_output: Any = validated_output.model_dump(mode="json")
        elif isinstance(raw_output, BaseModel):
            normalized_output = raw_output.model_dump(mode="json")
        else:
            normalized_output = raw_output

        # 7. Record successful result and timing.
        end_t = time.perf_counter()
        return TaskExecutionResult(
            task_id=task.task_id,
            tool_name=task.tool_name,
            status="success",
            output=normalized_output,
            duration_ms=(end_t - start_t) * 1000.0,
            started_at_ms=started_at_ms,
            finished_at_ms=(end_t - t0) * 1000.0,
        )
