"""Tool adapters and flexible tool normalization for LangGraph ReAct."""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, get_type_hints

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field, create_model

from llm_compiler.compiler.registry import ToolRegistry, ToolSpec

if TYPE_CHECKING:
    from .instrumentation import ReActRunInstrumentation

_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MAX_SERIALIZED_RESULT_CHARS = 100_000


@dataclass(frozen=True)
class ToolBinding:
    """A Python callable with optional tool metadata and a JSON Schema.

    Mirrors `parallel_react_llamaindex.tools.ToolBinding` so the exact same
    tool definitions can be passed to both LangGraph and LlamaIndex runners.
    """

    function: Callable[..., Any]
    name: str | None = None
    description: str | None = None
    parameters: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise TypeError("ToolBinding.function must be callable.")
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError("ToolBinding.name must be a string or None.")
        if self.description is not None and not isinstance(self.description, str):
            raise TypeError("ToolBinding.description must be a string or None.")
        if self.parameters is not None and not isinstance(self.parameters, Mapping):
            raise TypeError("ToolBinding.parameters must be a JSON Schema mapping or None.")

    @classmethod
    def from_spec(cls, spec: Any, function: Callable[..., Any]) -> ToolBinding:
        if isinstance(spec, Mapping):
            name = spec.get("name")
            description = spec.get("description")
            parameters = spec.get("parameters")
        else:
            name = getattr(spec, "name", None)
            description = getattr(spec, "description", None)
            parameters = getattr(spec, "parameters", None)
        if not isinstance(name, str) or not name:
            raise ValueError("Tool spec must provide a non-empty 'name'.")
        if parameters is not None and not isinstance(parameters, Mapping):
            raise ValueError("Tool spec 'parameters' must be a JSON Schema mapping.")
        return cls(
            function=function,
            name=name,
            description=description,
            parameters=parameters,
        )


ToolLike = ToolSpec | ToolBinding | Callable[..., Any] | Mapping[str, Any]


def _default_tool_name(function: Callable[..., Any]) -> str | None:
    target: Any = function
    while isinstance(target, functools.partial):
        target = target.func
    name = getattr(target, "__name__", None)
    if isinstance(name, str) and name and name != "<lambda>":
        return name
    name = type(target).__name__
    return name if name and name != "function" else None


def _is_async_callable(function: Callable[..., Any]) -> bool:
    target: Any = function
    while isinstance(target, functools.partial):
        target = target.func
    return inspect.iscoroutinefunction(target) or inspect.iscoroutinefunction(
        type(target).__call__
    )


def _json_schema_prop_to_type(prop: Mapping[str, Any]) -> Any:
    if "enum" in prop and isinstance(prop["enum"], Sequence) and prop["enum"]:
        return Literal[tuple(prop["enum"])]  # type: ignore[valid-type]
    t = prop.get("type")
    if isinstance(t, list):
        non_null = [item for item in t if item != "null"]
        base = _json_schema_prop_to_type({**prop, "type": non_null[0]}) if non_null else Any
        return base | None if "null" in t else base
    if t == "string":
        return str
    if t == "integer":
        return int
    if t == "number":
        return float
    if t == "boolean":
        return bool
    if t == "array":
        items = prop.get("items")
        if isinstance(items, Mapping):
            return list[_json_schema_prop_to_type(items)]  # type: ignore[misc]
        return list[Any]
    if t == "object":
        return dict[str, Any]
    return Any


def _args_model_from_json_schema(name: str, schema: Mapping[str, Any]) -> type[BaseModel]:
    props = schema.get("properties")
    if props is None:
        props = {}
    if not isinstance(props, Mapping):
        raise TypeError(f"JSON Schema 'properties' for tool {name!r} must be a mapping.")
    required = set(schema.get("required") or ())
    extra_policy = "forbid" if schema.get("additionalProperties") is False else "ignore"

    fields: dict[str, Any] = {}
    for field_name, prop in props.items():
        if not isinstance(prop, Mapping):
            continue
        py_type = _json_schema_prop_to_type(prop)
        desc = prop.get("description") if isinstance(prop.get("description"), str) else None
        if field_name in required:
            fields[field_name] = (py_type, Field(description=desc))
        else:
            default = prop.get("default")
            fields[field_name] = (py_type | None, Field(default=default, description=desc))

    safe_name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return create_model(
        f"{safe_name}_Args",
        __config__=ConfigDict(extra=extra_policy),
        **fields,
    )


def _args_model_from_callable(name: str, function: Callable[..., Any]) -> type[BaseModel]:
    sig = inspect.signature(function)
    try:
        hints = get_type_hints(function)
    except Exception:  # noqa: BLE001
        hints = {}

    fields: dict[str, Any] = {}
    for param_name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        annotation = hints.get(param_name, Any)
        if param.default is inspect.Parameter.empty:
            fields[param_name] = (annotation, ...)
        else:
            fields[param_name] = (annotation, param.default)

    safe_name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return create_model(
        f"{safe_name}_Args",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


def _to_tool_spec(item: ToolLike) -> ToolSpec:
    if isinstance(item, ToolSpec):
        return item

    if isinstance(item, ToolBinding):
        binding = item
    elif isinstance(item, Mapping):
        function = item.get("function") or item.get("callable") or item.get("func")
        if not callable(function):
            raise TypeError(
                "Tool mappings must include a callable under 'function' (or 'callable'/'func')."
            )
        parameters = item.get("parameters")
        if parameters is not None and not isinstance(parameters, Mapping):
            raise ValueError("Tool mapping 'parameters' must be a JSON Schema mapping.")
        binding = ToolBinding(
            function=function,
            name=item.get("name"),
            description=item.get("description"),
            parameters=parameters,
        )
    elif callable(item):
        binding = ToolBinding(function=item)
    else:
        raise TypeError(
            f"Unsupported tool object {item!r}; pass a ToolSpec, ToolBinding, callable, or mapping."
        )

    name = binding.name if binding.name is not None else _default_tool_name(binding.function)
    if not isinstance(name, str) or not name:
        raise ValueError("Tool callables must have a name or an explicit ToolBinding.name.")
    if not _TOOL_NAME_RE.fullmatch(name):
        raise ValueError(
            f"Invalid tool name {name!r}; expected 1-64 letters, digits, underscores or hyphens."
        )

    if binding.parameters is not None:
        input_schema = _args_model_from_json_schema(name, binding.parameters)
    else:
        input_schema = _args_model_from_callable(name, binding.function)

    description = (
        binding.description
        or inspect.getdoc(binding.function)
        or f"Call the {name} tool."
    ).strip()

    return ToolSpec(
        name=name,
        description=description,
        func=binding.function,
        input_schema=input_schema,
        output_schema=None,
    )


def normalize_to_registry(tools: ToolRegistry | Sequence[ToolLike]) -> ToolRegistry:
    """Normalize a `ToolRegistry` or sequence of `ToolLike` items into a `ToolRegistry`."""
    if isinstance(tools, ToolRegistry):
        return tools
    specs = [_to_tool_spec(item) for item in tools]
    return ToolRegistry(specs)


def _serialize_tool_output(value: Any) -> str:
    if isinstance(value, str):
        rendered = value
    else:
        if isinstance(value, BaseModel):
            value = value.model_dump()
        try:
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except TypeError:
            rendered = repr(value)
    if len(rendered) > _MAX_SERIALIZED_RESULT_CHARS:
        return rendered[:_MAX_SERIALIZED_RESULT_CHARS] + "... [truncated]"
    return rendered


def _wrap_tool_spec(
    spec: ToolSpec,
    *,
    task_timeout_s: float | None = None,
    instrumentation: ReActRunInstrumentation | None = None,
) -> StructuredTool:
    """Wrap a ToolSpec into a LangChain StructuredTool with I/O validation, timeout & tracing."""

    async def _invoke(**kwargs: Any) -> str:
        raw_kwargs = dict(kwargs)
        call_kwargs = dict(kwargs)
        if spec.input_schema is not None:
            validated_in = spec.input_schema.model_validate(call_kwargs)
            call_kwargs = validated_in.model_dump()

        rec = (
            instrumentation.start_tool(spec.name, raw_kwargs, call_kwargs)
            if instrumentation is not None
            else None
        )

        async def _run_handler() -> Any:
            if _is_async_callable(spec.func):
                val = spec.func(**call_kwargs)
                return await val if inspect.isawaitable(val) else val
            val = await asyncio.to_thread(spec.func, **call_kwargs)
            return await val if inspect.isawaitable(val) else val

        try:
            if task_timeout_s is not None:
                try:
                    raw_output = await asyncio.wait_for(
                        _run_handler(), timeout=task_timeout_s
                    )
                except TimeoutError as exc:
                    raise TimeoutError(
                        f"Tool '{spec.name}' timed out after {task_timeout_s:.2f}s."
                    ) from exc
            else:
                raw_output = await _run_handler()

            if spec.output_schema is not None:
                validated_out = spec.output_schema.model_validate(raw_output)
                raw_output = validated_out.model_dump()

            serialized = _serialize_tool_output(raw_output)
            if instrumentation is not None and rec is not None:
                instrumentation.finish_tool(rec, result=serialized)
            return serialized
        except Exception as exc:
            if instrumentation is not None and rec is not None:
                instrumentation.finish_tool(rec, error=exc)
            raise

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
    instrumentation: ReActRunInstrumentation | None = None,
) -> list[StructuredTool]:
    """Convert all registered ToolSpecs into LangChain StructuredTools."""
    if task_timeout_s is not None and task_timeout_s <= 0:
        raise ValueError("task_timeout_s must be > 0 when provided.")

    return [
        _wrap_tool_spec(
            registry[name],
            task_timeout_s=task_timeout_s,
            instrumentation=instrumentation,
        )
        for name in registry
    ]
