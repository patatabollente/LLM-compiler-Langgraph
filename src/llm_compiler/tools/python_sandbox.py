import ast
import contextlib
import io
import math
import statistics
from typing import Any

from pydantic import BaseModel, Field

from llm_compiler.compiler.registry import ToolSpec

_FORBIDDEN_CALLS = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "__import__",
        "input",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "breakpoint",
        "dir",
    }
)

_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "pow": pow,
    "print": print,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


class SandboxSecurityError(ValueError):
    """Raised when Python code violates AST sandbox restrictions."""


def _validate_safe_ast(code: str) -> ast.Module:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"Python syntax error: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise SandboxSecurityError(
                "Import statements are forbidden in the sandbox. "
                "Use the pre-loaded `math` and `statistics` modules directly."
            )
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise SandboxSecurityError(
                f"Access to dunder attribute '{node.attr}' is forbidden in the sandbox."
            )
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise SandboxSecurityError(
                f"Access to dunder name '{node.id}' is forbidden in the sandbox."
            )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FORBIDDEN_CALLS
        ):
            raise SandboxSecurityError(
                f"Call to '{node.func.id}()' is forbidden in the sandbox."
            )
    return tree


class PythonCodeInput(BaseModel):
    """Architecture-neutral input contract for sandboxed Python execution."""

    code: str = Field(
        description=(
            "Pure Python code to execute in a restricted sandbox. Do NOT use `import` "
            "statements: `math` and `statistics` are already pre-loaded. If your code "
            "reads `data` or `extra_data`, you MUST pass those values explicitly as "
            "arguments (or define literal values directly inside `code`). Assign the "
            "final JSON-serializable answer to a variable named `result`."
        )
    )
    data: Any = Field(
        default=None,
        description=(
            "Optional primary input payload passed into `code` as variable `data`. "
            "Pass the actual JSON object/list/number when `code` reads `data`."
        ),
    )
    extra_data: Any = Field(
        default=None,
        description=(
            "Optional secondary input payload passed into `code` as variable "
            "`extra_data`. Pass the actual JSON object/list/number when `code` "
            "reads `extra_data`."
        ),
    )


class PythonCodeOutput(BaseModel):
    """Structured output returned by sandboxed Python execution."""

    result: Any
    stdout: str


def run_python_code(
    code: str,
    data: Any = None,
    extra_data: Any = None,
) -> dict[str, Any]:
    """Execute validated Python code in a restricted, side-effect-free sandbox."""
    tree = _validate_safe_ast(code)
    compiled = compile(tree, filename="<sandbox>", mode="exec")

    env: dict[str, Any] = {
        "__builtins__": dict(_SAFE_BUILTINS),
        "math": math,
        "statistics": statistics,
        "data": data,
        "extra_data": extra_data,
    }

    stdout_buf = io.StringIO()
    with contextlib.redirect_stdout(stdout_buf):
        exec(compiled, env, env)  # noqa: S102

    stdout_text = stdout_buf.getvalue().strip()
    if "result" not in env:
        if stdout_text:
            return {"result": stdout_text, "stdout": stdout_text}
        raise ValueError(
            "Code executed without assigning `result` or printing output. "
            "Please assign the computed answer to `result`."
        )

    return {"result": env["result"], "stdout": stdout_text}


def build_python_sandbox_tool_spec() -> ToolSpec:
    """Create the architecture-neutral ToolSpec for `run_python_code`."""
    return ToolSpec(
        name="run_python_code",
        description=(
            "Execute safe Python code for custom math, statistics, simulations, "
            "or multi-dict calculations. Pre-loaded variables: `math`, `statistics`, "
            "`data`, `extra_data`. Do NOT use import statements; always pass any "
            "required input dicts/lists in `data`/`extra_data` (or define them "
            "directly in `code`) and assign the final answer to `result`."
        ),
        func=run_python_code,
        input_schema=PythonCodeInput,
        output_schema=PythonCodeOutput,
    )
