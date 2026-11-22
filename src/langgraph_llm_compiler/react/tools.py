import asyncio
import inspect
import json
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from langgraph_llm_compiler.compiler.registry import ToolRegistry, ToolSpec


def _serialize_tool_output(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump()
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return repr(value)


def _wrap_tool_spec(
    spec: ToolSpec,
    *,
    task_timeout_s: float | None = None,
) -> StructuredTool:
    """Wrap a ToolSpec into a LangChain StructuredTool with I/O validation & timeout."""

    async def _invoke(**kwargs: Any) -> str:
        call_kwargs = dict(kwargs)
        if spec.input_schema is not None:
            validated_in = spec.input_schema.model_validate(call_kwargs)
            call_kwargs = validated_in.model_dump()

        async def _run_handler() -> Any:
            if inspect.iscoroutinefunction(spec.func):
                return await spec.func(**call_kwargs)
            return await asyncio.to_thread(spec.func, **call_kwargs)

        if task_timeout_s is not None:
            try:
                raw_output = await asyncio.wait_for(_run_handler(), timeout=task_timeout_s)
            except TimeoutError as exc:
                raise TimeoutError(
                    f"Tool '{spec.name}' timed out after {task_timeout_s:.2f}s."
                ) from exc
        else:
            raw_output = await _run_handler()

        if spec.output_schema is not None:
            validated_out = spec.output_schema.model_validate(raw_output)
            raw_output = validated_out.model_dump()

        return _serialize_tool_output(raw_output)

    return StructuredTool.from_function(
        coroutine=_invoke,
        name=spec.name,
        description=spec.description,
        args_schema=spec.input_schema,
    )


def registry_to_langchain_tools(
    registry: ToolRegistry,
    *,
    task_timeout_s: float | None = None,
) -> list[StructuredTool]:
    """Convert all registered ToolSpecs into LangChain StructuredTools."""
    if task_timeout_s is not None and task_timeout_s <= 0:
        raise ValueError("task_timeout_s must be > 0 when provided.")

    return [
        _wrap_tool_spec(registry[name], task_timeout_s=task_timeout_s)
        for name in registry
    ]
