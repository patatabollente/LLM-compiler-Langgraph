import time
from collections.abc import Awaitable, Callable, Mapping

from langgraph_llm_compiler.compiler.compiler import PlanCompiler
from langgraph_llm_compiler.compiler.dag import PlanCompilationError
from langgraph_llm_compiler.executor.executor import Executor
from langgraph_llm_compiler.joiner.joiner import Joiner, JoinerOutputError
from langgraph_llm_compiler.planner.planner import Planner, PlannerOutputError

from .state import CompilerState, WorkflowError, WorkflowMetrics, update_metrics

GraphNode = Callable[[CompilerState], Awaitable[dict[str, object]]]


def make_planner_node(
    planner: Planner,
    catalog: Mapping[str, str],
) -> GraphNode:
    """Create the async LangGraph node wrapping Planner.aplan()."""

    async def planner_node(state: CompilerState) -> dict[str, object]:
        current_metrics = state.get("metrics") or WorkflowMetrics()
        request = state["request"]

        t0 = time.perf_counter()
        try:
            plan = await planner.aplan(request=request, available_tools=catalog)
        except ValueError as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "plan": None,
                "status": "failed",
                "error": WorkflowError(
                    stage="planner",
                    error_type=type(exc).__name__,
                    message=str(exc) or type(exc).__name__,
                ),
                "metrics": update_metrics(
                    state,
                    planning_latency_ms=elapsed_ms,
                    plan_valid=None,
                ),
            }
        except PlannerOutputError as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "plan": None,
                "status": "failed",
                "error": WorkflowError(
                    stage="planner",
                    error_type=type(exc).__name__,
                    message=str(exc) or type(exc).__name__,
                ),
                "metrics": update_metrics(
                    state,
                    planning_latency_ms=elapsed_ms,
                    llm_calls=current_metrics.llm_calls + 1,
                    plan_valid=None,
                ),
            }
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "plan": None,
                "status": "failed",
                "error": WorkflowError(
                    stage="planner",
                    error_type=type(exc).__name__,
                    message=str(exc) or type(exc).__name__,
                ),
                "metrics": update_metrics(
                    state,
                    planning_latency_ms=elapsed_ms,
                    llm_calls=current_metrics.llm_calls + 1,
                    plan_valid=None,
                ),
            }

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "plan": plan,
            "error": None,
            "metrics": update_metrics(
                state,
                planning_latency_ms=elapsed_ms,
                llm_calls=current_metrics.llm_calls + 1,
            ),
        }

    return planner_node


def make_compiler_node(compiler: PlanCompiler) -> GraphNode:
    """Create the async LangGraph node wrapping PlanCompiler.compile()."""

    async def compiler_node(state: CompilerState) -> dict[str, object]:
        current_metrics = state.get("metrics") or WorkflowMetrics()
        plan = state.get("plan")
        if plan is None:
            return {
                "compiled_dag": None,
                "status": "failed",
                "error": WorkflowError(
                    stage="compiler",
                    error_type="ValueError",
                    message="Cannot compile: 'plan' is missing from workflow state.",
                ),
                "metrics": update_metrics(state, plan_valid=False),
            }

        t0 = time.perf_counter()
        attempts = current_metrics.compile_attempts + 1
        try:
            dag = compiler.compile(plan)
        except PlanCompilationError as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "compiled_dag": None,
                "status": "failed",
                "error": WorkflowError(
                    stage="compiler",
                    error_type=type(exc).__name__,
                    message=str(exc),
                    issues=[issue.format() for issue in exc.issues],
                ),
                "metrics": update_metrics(
                    state,
                    compilation_latency_ms=elapsed_ms,
                    compile_attempts=attempts,
                    plan_valid=False,
                ),
            }
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "compiled_dag": None,
                "status": "failed",
                "error": WorkflowError(
                    stage="compiler",
                    error_type=type(exc).__name__,
                    message=str(exc) or type(exc).__name__,
                ),
                "metrics": update_metrics(
                    state,
                    compilation_latency_ms=elapsed_ms,
                    compile_attempts=attempts,
                    plan_valid=False,
                ),
            }

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "compiled_dag": dag,
            "error": None,
            "metrics": update_metrics(
                state,
                compilation_latency_ms=elapsed_ms,
                compile_attempts=attempts,
                plan_valid=True,
            ),
        }

    return compiler_node


def make_executor_node(executor: Executor) -> GraphNode:
    """Create the async LangGraph node wrapping Executor.execute()."""

    async def executor_node(state: CompilerState) -> dict[str, object]:
        dag = state.get("compiled_dag")
        if dag is None:
            return {
                "execution_result": None,
                "status": "failed",
                "error": WorkflowError(
                    stage="executor",
                    error_type="ValueError",
                    message="Cannot execute: 'compiled_dag' is missing from workflow state.",
                ),
            }

        t0 = time.perf_counter()
        try:
            execution_result = await executor.execute(dag)
        except Exception as exc:  # noqa: BLE001
            # asyncio.CancelledError inherits from BaseException, so it propagates intact.
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "execution_result": None,
                "status": "failed",
                "error": WorkflowError(
                    stage="executor",
                    error_type=type(exc).__name__,
                    message=str(exc) or type(exc).__name__,
                ),
                "metrics": update_metrics(state, execution_latency_ms=elapsed_ms),
            }

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "execution_result": execution_result,
            "error": None,
            "metrics": update_metrics(state, execution_latency_ms=elapsed_ms),
        }

    return executor_node


def make_joiner_node(joiner: Joiner) -> GraphNode:
    """Create the async LangGraph node wrapping Joiner.join()."""

    async def joiner_node(state: CompilerState) -> dict[str, object]:
        current_metrics = state.get("metrics") or WorkflowMetrics()
        execution_result = state.get("execution_result")
        if execution_result is None:
            return {
                "joiner_result": None,
                "final_answer": None,
                "status": "failed",
                "error": WorkflowError(
                    stage="joiner",
                    error_type="ValueError",
                    message="Cannot join: 'execution_result' is missing from workflow state.",
                ),
            }

        t0 = time.perf_counter()
        try:
            joiner_result = await joiner.join(
                request=state["request"],
                execution_result=execution_result,
                plan=state.get("plan"),
            )
        except ValueError as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "joiner_result": None,
                "final_answer": None,
                "status": "failed",
                "error": WorkflowError(
                    stage="joiner",
                    error_type=type(exc).__name__,
                    message=str(exc) or type(exc).__name__,
                ),
                "metrics": update_metrics(state, joining_latency_ms=elapsed_ms),
            }
        except JoinerOutputError as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "joiner_result": None,
                "final_answer": None,
                "status": "failed",
                "error": WorkflowError(
                    stage="joiner",
                    error_type=type(exc).__name__,
                    message=str(exc) or type(exc).__name__,
                ),
                "metrics": update_metrics(
                    state,
                    joining_latency_ms=elapsed_ms,
                    llm_calls=current_metrics.llm_calls + 1,
                ),
            }
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "joiner_result": None,
                "final_answer": None,
                "status": "failed",
                "error": WorkflowError(
                    stage="joiner",
                    error_type=type(exc).__name__,
                    message=str(exc) or type(exc).__name__,
                ),
                "metrics": update_metrics(
                    state,
                    joining_latency_ms=elapsed_ms,
                    llm_calls=current_metrics.llm_calls + 1,
                ),
            }

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "joiner_result": joiner_result,
            "final_answer": joiner_result.answer,
            "status": joiner_result.status,
            "error": None,
            "metrics": update_metrics(
                state,
                joining_latency_ms=elapsed_ms,
                llm_calls=current_metrics.llm_calls + 1,
            ),
        }

    return joiner_node
