from collections.abc import Callable
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from langgraph_llm_compiler.compiler.compiler import PlanCompiler
from langgraph_llm_compiler.compiler.registry import ToolRegistry
from langgraph_llm_compiler.executor.executor import Executor
from langgraph_llm_compiler.joiner.joiner import Joiner
from langgraph_llm_compiler.planner.planner import Planner

from .nodes import (
    make_compiler_node,
    make_executor_node,
    make_joiner_node,
    make_planner_node,
)
from .state import CompilerState, WorkflowMetrics


def _route_after_planner(state: CompilerState) -> Literal["compiler", "__end__"]:
    return END if state.get("error") is not None else "compiler"


def _make_compiler_router(
    max_compile_retries: int,
) -> Callable[[CompilerState], Literal["executor", "planner", "__end__"]]:
    def _route_after_compiler(
        state: CompilerState,
    ) -> Literal["executor", "planner", "__end__"]:
        err = state.get("error")
        if err is None:
            return "executor"

        metrics = state.get("metrics") or WorkflowMetrics()
        if (
            err.error_type == "PlanCompilationError"
            and metrics.compile_attempts <= max_compile_retries
        ):
            return "planner"
        return END

    return _route_after_compiler


def _route_after_executor(state: CompilerState) -> Literal["joiner", "__end__"]:
    return END if state.get("error") is not None else "joiner"


def build_compiler_graph(
    *,
    registry: ToolRegistry,
    planner: Planner,
    compiler: PlanCompiler,
    executor: Executor,
    joiner: Joiner,
    max_compile_retries: int = 0,
) -> CompiledStateGraph[CompilerState, Any, CompilerState, CompilerState]:
    """Build and compile the LangGraph LLM Compiler workflow.

    Args:
        max_compile_retries: Maximum number of compile-time self-correction retries
            (`compiler -> planner`) on `PlanCompilationError`. Defaults to `0`
            (linear v1 behavior with no retries).
    """
    if max_compile_retries < 0:
        raise ValueError("max_compile_retries must be >= 0.")

    catalog = registry.to_planner_catalog()

    builder: StateGraph[CompilerState, Any, CompilerState, CompilerState] = StateGraph(
        CompilerState
    )
    builder.add_node("planner", make_planner_node(planner, catalog))
    builder.add_node("compiler", make_compiler_node(compiler))
    builder.add_node("executor", make_executor_node(executor))
    builder.add_node("joiner", make_joiner_node(joiner))

    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner",
        _route_after_planner,
        {"compiler": "compiler", END: END},
    )

    compiler_targets: dict[str, str] = {"executor": "executor", END: END}
    if max_compile_retries > 0:
        compiler_targets["planner"] = "planner"

    builder.add_conditional_edges(
        "compiler",
        _make_compiler_router(max_compile_retries),
        compiler_targets,
    )
    builder.add_conditional_edges(
        "executor",
        _route_after_executor,
        {"joiner": "joiner", END: END},
    )
    builder.add_edge("joiner", END)

    return builder.compile()
