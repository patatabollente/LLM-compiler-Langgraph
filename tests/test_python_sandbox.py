import pytest

from langgraph_llm_compiler.tools import (
    PythonCodeInput,
    PythonCodeOutput,
    SandboxSecurityError,
    build_python_sandbox_tool_spec,
    run_python_code,
)


def test_run_python_code_supports_math_statistics_and_data_inputs():
    code = """
vals = list(data.values())
margins = {k: round(extra_data[k] / data[k] * 100, 2) for k in data}
result = {
    "mean": statistics.mean(vals),
    "sqrt_max": round(math.sqrt(max(vals)), 2),
    "margins": margins,
}
"""
    out = run_python_code(
        code=code,
        data={"A": 100.0, "B": 400.0},
        extra_data={"A": 25.0, "B": 120.0},
    )
    assert out["result"] == {
        "mean": 250.0,
        "sqrt_max": 20.0,
        "margins": {"A": 25.0, "B": 30.0},
    }
    assert out["stdout"] == ""


def test_run_python_code_falls_back_to_stdout_when_result_omitted():
    out = run_python_code(code="print('hello', 42)")
    assert out == {"result": "hello 42", "stdout": "hello 42"}


def test_run_python_code_rejects_empty_output_and_syntax_errors():
    with pytest.raises(ValueError, match="without assigning"):
        run_python_code(code="x = 1 + 1")

    with pytest.raises(ValueError, match="syntax error"):
        run_python_code(code="def broken(:\n    pass")


@pytest.mark.parametrize(
    "bad_code",
    [
        "import os\nresult = os.getcwd()",
        "from pathlib import Path\nresult = str(Path('.'))",
        "result = open('/etc/passwd').read()",
        "result = eval('1 + 1')",
        "result = ().__class__.__base__.__subclasses__()",
        "result = __builtins__",
    ],
)
def test_run_python_code_blocks_unsafe_ast_constructs(bad_code: str):
    with pytest.raises(SandboxSecurityError):
        run_python_code(code=bad_code)


def test_build_python_sandbox_tool_spec_is_architecture_neutral():
    spec = build_python_sandbox_tool_spec()
    assert spec.name == "run_python_code"
    assert spec.input_schema is PythonCodeInput
    assert spec.output_schema is PythonCodeOutput
    # Must not leak architecture-specific terminology into tool descriptions.
    desc = spec.to_planner_description()
    assert "InputBinding" not in desc
    assert "CompiledDAG" not in desc
    assert "ReAct" not in desc
