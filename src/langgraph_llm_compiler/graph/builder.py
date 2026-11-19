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


def _route_after_planner(state: CompilerState) -> Literal["compiler", "__end__"]:
    return END if state.get("error") is not None else "compiler"


def _route_after_compiler(state: CompilerState) -> Literal["executor", "__end__"]:
    return END if state.get("error") is not None else "executor"


def _route_after_executor(state: CompilerState) -> Literal["joiner", "__end__"]:
    return END if state.get("error") is not None else "joiner"


def build_compiler_graph(
    *,
    registry: ToolRegistry,
    planner: Planner,
    compiler: PlanCompiler,
    executor: Executor,
    joiner: Joiner,
) -> CompiledStateGraph[CompilerState, Any, CompilerState, CompilerState]:
    """Build and compile the LangGraph v1 linear workflow with error exits."""
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
    builder.add_conditional_edges(
        "compiler",
        _route_after_compiler,
        {"executor": "executor", END: END},
    )
    builder.add_conditional_edges(
        "executor",
        _route_after_executor,
        {"joiner": "joiner", END: END},
    )
    builder.add_edge("joiner", END)

    return builder.compile()
