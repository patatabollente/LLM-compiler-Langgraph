import ast
import asyncio
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from llm_compiler import (
    ExecutionPlan,
    Executor,
    InputBinding,
    Joiner,
    JoinerDecisionOutput,
    JoinerOutput,
    PlanCompiler,
    Planner,
    TaskSpec,
    ToolRegistry,
    ToolSpec,
    run_compiler_workflow,
)

SRC_CORE_DIR = Path(__file__).resolve().parent.parent / "src" / "llm_compiler"


def test_llm_compiler_core_has_zero_langgraph_imports():
    """Formal architectural boundary: `src/llm_compiler` must NEVER import LangGraph."""
    violations: list[str] = []
    py_files = sorted(SRC_CORE_DIR.rglob("*.py"))
    assert len(py_files) > 0

    for py_file in py_files:
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(("langgraph", "langgraph_adapters")):
                        violations.append(f"{py_file.relative_to(SRC_CORE_DIR)}: import {alias.name}")
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith(("langgraph", "langgraph_adapters"))
            ):
                violations.append(
                    f"{py_file.relative_to(SRC_CORE_DIR)}: from {node.module} import ..."
                )

    assert violations == [], f"Forbidden LangGraph imports in core: {violations}"


class _SeqRunnable:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def ainvoke(self, messages: list[Any]) -> dict[str, Any]:
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[idx]


class _SchemaLLM:
    def __init__(self, by_schema: dict[type[Any], list[dict[str, Any]]]) -> None:
        self._runnables = {s: _SeqRunnable(r) for s, r in by_schema.items()}

    def with_structured_output(self, **kwargs: Any) -> _SeqRunnable:
        return self._runnables[kwargs["schema"]]


class _LookupIn(BaseModel):
    item: str


class _LookupOut(BaseModel):
    value: float


class _MathIn(BaseModel):
    op: Literal["add", "subtract"]
    a: float
    b: float


class _MathOut(BaseModel):
    result: float


def _make_registry() -> ToolRegistry:
    async def primary(item: str) -> dict[str, float]:
        if item == "FAIL":
            raise RuntimeError("primary down")
        return {"value": 100.0}

    async def backup(item: str) -> dict[str, float]:
        return {"value": 40.0}

    def calc(op: Literal["add", "subtract"], a: float, b: float) -> dict[str, float]:
        return {"result": a + b if op == "add" else a - b}

    return ToolRegistry(
        [
            ToolSpec("primary", "Primary lookup", primary, _LookupIn, _LookupOut),
            ToolSpec("backup", "Backup lookup", backup, _LookupIn, _LookupOut),
            ToolSpec("calc", "Arithmetic", calc, _MathIn, _MathOut),
        ]
    )


def test_pure_python_workflow_runs_v2_compile_retry_and_v3_replan_without_langgraph():
    registry = _make_registry()
    bad_plan = ExecutionPlan(
        objective="Bad plan with unknown tool",
        tasks=[TaskSpec(task_id="t1", tool_name="nope", description="bad", inputs={})],
    )
    round1_plan = ExecutionPlan(
        objective="Fetch OK and FAIL",
        tasks=[
            TaskSpec(task_id="t1", tool_name="primary", description="get OK", inputs={"item": "OK"}),
            TaskSpec(task_id="t2", tool_name="primary", description="get FAIL", inputs={"item": "FAIL"}),
            TaskSpec(
                task_id="t3",
                tool_name="calc",
                description="sum",
                inputs={
                    "op": "add",
                    "a": InputBinding(task_id="t1", output_path=["value"]),
                    "b": InputBinding(task_id="t2", output_path=["value"]),
                },
            ),
        ],
    )
    round2_plan = ExecutionPlan(
        objective="Recover via backup and add 100",
        tasks=[
            TaskSpec(task_id="t1", tool_name="backup", description="backup", inputs={"item": "FAIL"}),
            TaskSpec(
                task_id="t2",
                tool_name="calc",
                description="sum",
                inputs={
                    "op": "add",
                    "a": 100.0,
                    "b": InputBinding(task_id="t1", output_path=["value"]),
                },
            ),
        ],
    )

    planner_llm = _SchemaLLM(
        {
            ExecutionPlan: [
                {"parsed": bad_plan, "parsing_error": None, "raw": "{}"},
                {"parsed": round1_plan, "parsing_error": None, "raw": "{}"},
                {"parsed": round2_plan, "parsing_error": None, "raw": "{}"},
            ]
        }
    )
    joiner_llm = _SchemaLLM(
        {
            JoinerDecisionOutput: [
                {
                    "parsed": JoinerDecisionOutput(
                        action="replan",
                        replan_reason="Use backup tool for FAIL and add 100.0.",
                    ),
                    "parsing_error": None,
                    "raw": "{}",
                }
            ],
            JoinerOutput: [
                {"parsed": JoinerOutput(answer="Totale: 140.0."), "parsing_error": None, "raw": "{}"}
            ],
        }
    )

    state = asyncio.run(
        run_compiler_workflow(
            "Somma OK e FAIL",
            registry=registry,
            planner=Planner(llm=planner_llm),  # type: ignore[arg-type]
            compiler=PlanCompiler(registry=registry),
            executor=Executor(registry=registry),
            joiner=Joiner(llm=joiner_llm),  # type: ignore[arg-type]
            max_compile_retries=1,
            max_replans=1,
        )
    )

    assert state["status"] == "complete"
    assert state["error"] is None
    assert state["final_answer"] == "Totale: 140.0."
    assert state["metrics"].compile_retries == 1
    assert state["metrics"].replan_attempts == 1
    assert state["metrics"].llm_calls == 5  # 3 planner + 2 joiner


def test_pure_python_workflow_handles_empty_plan_direct_shortcut():
    registry = _make_registry()
    empty_plan = ExecutionPlan(objective="Conversational greeting", tasks=[])

    planner_llm = _SchemaLLM(
        {ExecutionPlan: [{"parsed": empty_plan, "parsing_error": None, "raw": "{}"}]}
    )
    joiner_llm = _SchemaLLM(
        {JoinerOutput: [{"parsed": JoinerOutput(answer="Ciao!"), "parsing_error": None, "raw": "{}"}]}
    )

    state = asyncio.run(
        run_compiler_workflow(
            "Ciao!",
            registry=registry,
            planner=Planner(llm=planner_llm),  # type: ignore[arg-type]
            compiler=PlanCompiler(registry=registry),
            executor=Executor(registry=registry),
            joiner=Joiner(llm=joiner_llm),  # type: ignore[arg-type]
        )
    )

    assert state["status"] == "complete"
    assert state["final_answer"] == "Ciao!"
    assert state["metrics"].compile_attempts == 0
    assert state["metrics"].llm_calls == 2
