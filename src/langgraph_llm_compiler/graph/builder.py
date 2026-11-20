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
from .state import CompilerState


def _make_planner_router(
    max_replans: int,
) -> Callable[[CompilerState], Literal["compiler", "joiner", "__end__"]]:
    def _route_after_planner(
        state: CompilerState,
    ) -> Literal["compiler", "joiner", "__end__"]:
        if state.get("error") is None:
            return "compiler"
        if max_replans > 0 and state.get("execution_history"):
            return "joiner"
        return END

    return _route_after_planner


def _make_compiler_router(
    max_compile_retries: int,
    max_replans: int,
) -> Callable[[CompilerState], Literal["executor", "planner", "joiner", "__end__"]]:
    def _route_after_compiler(
        state: CompilerState,
    ) -> Literal["executor", "planner", "joiner", "__end__"]:
        err = state.get("error")
        if err is None:
            return "executor"

        round_attempts = state.get("round_compile_attempts", 0)
        if (
            err.error_type == "PlanCompilationError"
            and round_attempts <= max_compile_retries
        ):
            return "planner"
        if max_replans > 0 and state.get("execution_history"):
            return "joiner"
        return END

    return _route_after_compiler


def _make_executor_router(
    max_replans: int,
) -> Callable[[CompilerState], Literal["joiner", "__end__"]]:
    def _route_after_executor(
        state: CompilerState,
    ) -> Literal["joiner", "__end__"]:
        if state.get("error") is None:
            return "joiner"
        if max_replans > 0 and state.get("execution_history"):
            return "joiner"
        return END

    return _route_after_executor


def _make_joiner_router(
    max_replans: int,
) -> Callable[[CompilerState], Literal["planner", "__end__"]]:
    def _route_after_joiner(
        state: CompilerState,
    ) -> Literal["planner", "__end__"]:
        if state.get("error") is not None:
            return END
        joiner_result = state.get("joiner_result")
        if (
            max_replans > 0
            and joiner_result is not None
            and joiner_result.action == "replan"
        ):
            return "planner"
        return END

    return _route_after_joiner


def build_compiler_graph(
    *,
    registry: ToolRegistry,
    planner: Planner,
    compiler: PlanCompiler,
    executor: Executor,
    joiner: Joiner,
    max_compile_retries: int = 0,
    max_replans: int = 0,
) -> CompiledStateGraph[CompilerState, Any, CompilerState, CompilerState]:
    """Build and compile the LangGraph LLM Compiler workflow.

    Args:
        max_compile_retries: Maximum number of compile-time self-correction retries
            (`compiler -> planner`) per execution round on `PlanCompilationError`.
            Defaults to `0` (linear v1 behavior with no compile retries).
        max_replans: Maximum number of runtime replans (`joiner -> planner`)
            allowed when the Joiner requests follow-up execution. Defaults to `0`
            (v1/v2 behavior with no runtime replanning).
    """
    if max_compile_retries < 0:
        raise ValueError("max_compile_retries must be >= 0.")
    if max_replans < 0:
        raise ValueError("max_replans must be >= 0.")

    catalog = registry.to_planner_catalog()

    builder: StateGraph[CompilerState, Any, CompilerState, CompilerState] = StateGraph(
        CompilerState
    )
    builder.add_node("planner", make_planner_node(planner, catalog))
    builder.add_node("compiler", make_compiler_node(compiler))
    builder.add_node("executor", make_executor_node(executor))
    builder.add_node(
        "joiner",
        make_joiner_node(joiner, catalog=catalog, max_replans=max_replans),
    )

    builder.add_edge(START, "planner")

    planner_targets: dict[str, str] = {"compiler": "compiler", END: END}
    if max_replans > 0:
        planner_targets["joiner"] = "joiner"
    builder.add_conditional_edges(
        "planner",
        _make_planner_router(max_replans),
        planner_targets,
    )

    compiler_targets: dict[str, str] = {"executor": "executor", END: END}
    if max_compile_retries > 0:
        compiler_targets["planner"] = "planner"
    if max_replans > 0:
        compiler_targets["joiner"] = "joiner"
    builder.add_conditional_edges(
        "compiler",
        _make_compiler_router(max_compile_retries, max_replans),
        compiler_targets,
    )

    builder.add_conditional_edges(
        "executor",
        _make_executor_router(max_replans),
        {"joiner": "joiner", END: END},
    )

    if max_replans > 0:
        builder.add_conditional_edges(
            "joiner",
            _make_joiner_router(max_replans),
            {"planner": "planner", END: END},
        )
    else:
        builder.add_edge("joiner", END)

    return builder.compile()
