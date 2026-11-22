"""Pure-Python (framework-agnostic) async runner for LLM Compiler v1/v2/v3.

Executes the exact same step functions and state transitions as the LangGraph
adapter (`langgraph_adapters.compiler_graph`), without importing `langgraph`.
"""

from typing import cast

from .compiler.compiler import PlanCompiler
from .compiler.registry import ToolRegistry
from .executor.executor import Executor
from .joiner.joiner import Joiner
from .planner.planner import Planner
from .state import CompilerState
from .steps import (
    make_compiler_step,
    make_executor_step,
    make_joiner_step,
    make_planner_step,
)


async def run_compiler_workflow(
    request: str,
    *,
    registry: ToolRegistry,
    planner: Planner,
    compiler: PlanCompiler,
    executor: Executor,
    joiner: Joiner,
    max_compile_retries: int = 0,
    max_replans: int = 0,
) -> CompilerState:
    """Run the full LLM Compiler workflow (v1/v2/v3) in pure Python without LangGraph."""
    if max_compile_retries < 0:
        raise ValueError("max_compile_retries must be >= 0.")
    if max_replans < 0:
        raise ValueError("max_replans must be >= 0.")

    catalog = registry.to_planner_catalog()
    planner_step = make_planner_step(planner, catalog)
    compiler_step = make_compiler_step(compiler)
    executor_step = make_executor_step(executor)
    joiner_step = make_joiner_step(joiner, catalog=catalog, max_replans=max_replans)

    state: CompilerState = {"request": request}
    current: str = "planner"

    while current != "__end__":
        if current == "planner":
            update = await planner_step(state)
            state = cast(CompilerState, {**state, **update})
            if state.get("error") is None:
                plan = state.get("plan")
                if plan is not None and len(plan.tasks) == 0:
                    current = "joiner"
                else:
                    current = "compiler"
            elif max_replans > 0 and state.get("execution_history"):
                current = "joiner"
            else:
                current = "__end__"

        elif current == "compiler":
            update = await compiler_step(state)
            state = cast(CompilerState, {**state, **update})
            err = state.get("error")
            if err is None:
                current = "executor"
            elif (
                err.error_type == "PlanCompilationError"
                and state.get("round_compile_attempts", 0) <= max_compile_retries
            ):
                current = "planner"
            elif max_replans > 0 and state.get("execution_history"):
                current = "joiner"
            else:
                current = "__end__"

        elif current == "executor":
            update = await executor_step(state)
            state = cast(CompilerState, {**state, **update})
            if state.get("error") is None or (
                max_replans > 0 and state.get("execution_history")
            ):
                current = "joiner"
            else:
                current = "__end__"

        elif current == "joiner":
            update = await joiner_step(state)
            state = cast(CompilerState, {**state, **update})
            joiner_result = state.get("joiner_result")
            if (
                state.get("error") is None
                and max_replans > 0
                and joiner_result is not None
                and joiner_result.action == "replan"
            ):
                current = "planner"
            else:
                current = "__end__"

    return state
