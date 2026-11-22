from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel


def _format_type(annotation: Any) -> str:
    """Format a Python type annotation into a concise schema string for prompts."""
    if annotation is None or annotation is type(None):
        return "None"

    if annotation is Any:
        return "Any"

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        fields = ", ".join(
            f'"{fname}": {_format_type(finfo.annotation)}'
            for fname, finfo in annotation.model_fields.items()
        )
        return f"{{{fields}}}"

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Literal:
        return " | ".join(repr(arg) for arg in args)

    if origin in (Union, UnionType):
        return " | ".join(_format_type(arg) for arg in args)

    if origin in (list, tuple, set, frozenset):
        inner = ", ".join(_format_type(arg) for arg in args) if args else "Any"
        return f"{origin.__name__}[{inner}]"

    if origin is dict:
        if len(args) == 2:
            return f"dict[{_format_type(args[0])}, {_format_type(args[1])}]"
        return "dict[str, Any]"

    if isinstance(annotation, type):
        return annotation.__name__

    return str(annotation)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Metadata, optional schemas, and callable handler for an executable tool."""

    name: str
    description: str
    func: Callable[..., Any]
    input_schema: type[BaseModel] | None = None
    output_schema: type[BaseModel] | None = None

    def __post_init__(self) -> None:
        normalized_name = self.name.strip()
        normalized_description = self.description.strip()

        if not normalized_name:
            raise ValueError("Tool name cannot be empty.")
        if not normalized_description:
            raise ValueError("Tool description cannot be empty.")
        if not callable(self.func):
            raise TypeError(f"Tool '{normalized_name}' handler must be callable.")
        if self.input_schema is not None and not (
            isinstance(self.input_schema, type)
            and issubclass(self.input_schema, BaseModel)
        ):
            raise TypeError("input_schema must be a Pydantic BaseModel subclass.")
        if self.output_schema is not None and not (
            isinstance(self.output_schema, type)
            and issubclass(self.output_schema, BaseModel)
        ):
            raise TypeError("output_schema must be a Pydantic BaseModel subclass.")

        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "description", normalized_description)

    def to_planner_description(self) -> str:
        """Render a prompt-ready description including input/output contracts."""
        parts = [self.description.rstrip(".") + "."]

        if self.input_schema is not None:
            arg_parts: list[str] = []
            for fname, finfo in self.input_schema.model_fields.items():
                type_str = _format_type(finfo.annotation)
                opt = "" if finfo.is_required() else ", optional"
                arg_parts.append(f"{fname} ({type_str}{opt})")
            if arg_parts:
                parts.append(f"Args: {', '.join(arg_parts)}.")
            else:
                parts.append("Args: none.")

        if self.output_schema is not None:
            parts.append(f"Output: {_format_type(self.output_schema)}.")

        return " ".join(parts)


class ToolRegistry:
    """Single source of truth for registered tools across Planner, Compiler, and Executor."""

    def __init__(self, tools: Iterable[ToolSpec] = ()) -> None:
        self._tools: dict[str, ToolSpec] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: ToolSpec, *, overwrite: bool = False) -> None:
        if not overwrite and tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def __getitem__(self, name: str) -> ToolSpec:
        return self._tools[name]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[str]:
        return iter(self._tools)

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools.keys())

    def to_planner_catalog(self) -> dict[str, str]:
        """Generate the available_tools mapping expected by Planner."""
        return {
            name: spec.to_planner_description()
            for name, spec in self._tools.items()
        }
