import time
from collections.abc import Awaitable, Callable, Mapping

from llm_compiler.compiler.compiler import PlanCompiler
from llm_compiler.compiler.dag import PlanCompilationError
from llm_compiler.executor.executor import Executor
from llm_compiler.joiner.joiner import (
    Joiner,
    JoinerOutputError,
    format_replan_context,
)
from llm_compiler.joiner.schemas import ExecutionRound
from llm_compiler.planner.planner import Planner, PlannerOutputError

from .state import CompilerState, WorkflowError, WorkflowMetrics, update_metrics

WorkflowStep = Callable[[CompilerState], Awaitable[dict[str, object]]]


def make_planner_step(
    planner: Planner,
    catalog: Mapping[str, str],
) -> WorkflowStep:
    """Create the async workflow step wrapping Planner.aplan()."""

    async def planner_step(state: CompilerState) -> dict[str, object]:
        current_metrics = state.get("metrics") or WorkflowMetrics()
        request = state["request"]
        feedback = state.get("compile_feedback")
        previous_plan = state.get("plan") if feedback else None
        replan_context = state.get("replan_context")

        t0 = time.perf_counter()
        try:
            plan = await planner.aplan(
                request=request,
                available_tools=catalog,
                previous_plan=previous_plan,
                feedback=feedback,
                replan_context=replan_context,
            )
        except ValueError as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            total_planning_ms = (current_metrics.planning_latency_ms or 0.0) + elapsed_ms
            return {
                "plan": None,
                "compile_feedback": None,
                "status": "failed",
                "error": WorkflowError(
                    stage="planner",
                    error_type=type(exc).__name__,
                    message=str(exc) or type(exc).__name__,
                ),
                "metrics": update_metrics(
                    state,
                    planning_latency_ms=total_planning_ms,
                    plan_valid=None,
                ),
            }
        except PlannerOutputError as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            total_planning_ms = (current_metrics.planning_latency_ms or 0.0) + elapsed_ms
            return {
                "plan": None,
                "compile_feedback": None,
                "status": "failed",
                "error": WorkflowError(
                    stage="planner",
                    error_type=type(exc).__name__,
                    message=str(exc) or type(exc).__name__,
                ),
                "metrics": update_metrics(
                    state,
                    planning_latency_ms=total_planning_ms,
                    llm_calls=current_metrics.llm_calls + 1,
                    plan_valid=None,
                ),
            }
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            total_planning_ms = (current_metrics.planning_latency_ms or 0.0) + elapsed_ms
            return {
                "plan": None,
                "compile_feedback": None,
                "status": "failed",
                "error": WorkflowError(
                    stage="planner",
                    error_type=type(exc).__name__,
                    message=str(exc) or type(exc).__name__,
                ),
                "metrics": update_metrics(
                    state,
                    planning_latency_ms=total_planning_ms,
                    llm_calls=current_metrics.llm_calls + 1,
                    plan_valid=None,
                ),
            }

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_planning_ms = (current_metrics.planning_latency_ms or 0.0) + elapsed_ms
        return {
            "plan": plan,
            "compile_feedback": None,
            "status": None,
            "error": None,
            "metrics": update_metrics(
                state,
                planning_latency_ms=total_planning_ms,
                llm_calls=current_metrics.llm_calls + 1,
            ),
        }

    return planner_step


def make_compiler_step(compiler: PlanCompiler) -> WorkflowStep:
    """Create the async workflow step wrapping PlanCompiler.compile()."""

    async def compiler_step(state: CompilerState) -> dict[str, object]:
        current_metrics = state.get("metrics") or WorkflowMetrics()
        plan = state.get("plan")
        if plan is None:
            return {
                "compiled_dag": None,
                "compile_feedback": None,
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
        round_attempts = state.get("round_compile_attempts", 0) + 1
        total_retries = current_metrics.compile_retries + (1 if round_attempts > 1 else 0)
        try:
            dag = compiler.compile(plan)
        except PlanCompilationError as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            total_comp_ms = (current_metrics.compilation_latency_ms or 0.0) + elapsed_ms
            formatted_issues = [issue.format() for issue in exc.issues]
            feedback_text = "\n".join(f"- {item}" for item in formatted_issues)
            history = [*state.get("compile_error_history", []), formatted_issues]
            return {
                "compiled_dag": None,
                "compile_feedback": feedback_text,
                "compile_error_history": history,
                "round_compile_attempts": round_attempts,
                "status": "failed",
                "error": WorkflowError(
                    stage="compiler",
                    error_type=type(exc).__name__,
                    message=str(exc),
                    issues=formatted_issues,
                ),
                "metrics": update_metrics(
                    state,
                    compilation_latency_ms=total_comp_ms,
                    compile_attempts=attempts,
                    compile_retries=total_retries,
                    plan_valid=False,
                ),
            }
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            total_comp_ms = (current_metrics.compilation_latency_ms or 0.0) + elapsed_ms
            return {
                "compiled_dag": None,
                "compile_feedback": None,
                "round_compile_attempts": round_attempts,
                "status": "failed",
                "error": WorkflowError(
                    stage="compiler",
                    error_type=type(exc).__name__,
                    message=str(exc) or type(exc).__name__,
                ),
                "metrics": update_metrics(
                    state,
                    compilation_latency_ms=total_comp_ms,
                    compile_attempts=attempts,
                    compile_retries=total_retries,
                    plan_valid=False,
                ),
            }

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_comp_ms = (current_metrics.compilation_latency_ms or 0.0) + elapsed_ms
        return {
            "compiled_dag": dag,
            "compile_feedback": None,
            "round_compile_attempts": round_attempts,
            "status": None,
            "error": None,
            "metrics": update_metrics(
                state,
                compilation_latency_ms=total_comp_ms,
                compile_attempts=attempts,
                compile_retries=total_retries,
                plan_valid=True,
            ),
        }

    return compiler_step


def make_executor_step(executor: Executor) -> WorkflowStep:
    """Create the async workflow step wrapping Executor.execute()."""

    async def executor_step(state: CompilerState) -> dict[str, object]:
        current_metrics = state.get("metrics") or WorkflowMetrics()
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
            total_exec_ms = (current_metrics.execution_latency_ms or 0.0) + elapsed_ms
            return {
                "execution_result": None,
                "status": "failed",
                "error": WorkflowError(
                    stage="executor",
                    error_type=type(exc).__name__,
                    message=str(exc) or type(exc).__name__,
                ),
                "metrics": update_metrics(state, execution_latency_ms=total_exec_ms),
            }

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_exec_ms = (current_metrics.execution_latency_ms or 0.0) + elapsed_ms
        return {
            "execution_result": execution_result,
            "error": None,
            "metrics": update_metrics(state, execution_latency_ms=total_exec_ms),
        }

    return executor_step


def make_joiner_step(
    joiner: Joiner,
    *,
    catalog: Mapping[str, str] | None = None,
    max_replans: int = 0,
) -> WorkflowStep:
    """Create the async workflow step wrapping Joiner.join()."""

    async def joiner_step(state: CompilerState) -> dict[str, object]:
        current_metrics = state.get("metrics") or WorkflowMetrics()
        history = list(state.get("execution_history", []))
        upstream_error = state.get("error")

        # Graceful degradation: if a replan round failed before execution completed,
        # synthesize the best available answer from prior completed rounds.
        degraded_from_replan = upstream_error is not None and len(history) > 0
        raw_plan = state.get("plan")
        direct_no_tools = False
        if degraded_from_replan:
            last_round = history[-1]
            prior_rounds = history[:-1]
            execution_result = last_round.execution_result
            plan = last_round.plan
            allow_replan = False
            recorded_replan_error: WorkflowError | None = upstream_error
        elif raw_plan is not None and len(raw_plan.tasks) == 0:
            recorded_replan_error = state.get("replan_error")
            allow_replan = False
            if history:
                last_round = history[-1]
                prior_rounds = history[:-1]
                execution_result = last_round.execution_result
                plan = last_round.plan
            else:
                direct_no_tools = True
                prior_rounds = []
                execution_result = None
                plan = raw_plan
        else:
            execution_result = state.get("execution_result")
            plan = raw_plan
            prior_rounds = history
            allow_replan = (
                max_replans > 0 and current_metrics.replan_attempts < max_replans
            )
            recorded_replan_error = state.get("replan_error")

        if not direct_no_tools and execution_result is None:
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
            if direct_no_tools:
                joiner_result = await joiner.join_direct(
                    request=state["request"],
                    plan=plan,
                    tool_catalog=catalog,
                )
            else:
                assert execution_result is not None
                joiner_result = await joiner.join(
                    request=state["request"],
                    execution_result=execution_result,
                    plan=plan,
                    prior_rounds=prior_rounds,
                    tool_catalog=catalog if allow_replan else None,
                    allow_replan=allow_replan,
                )
        except ValueError as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            total_join_ms = (current_metrics.joining_latency_ms or 0.0) + elapsed_ms
            return {
                "joiner_result": None,
                "final_answer": None,
                "status": "failed",
                "error": WorkflowError(
                    stage="joiner",
                    error_type=type(exc).__name__,
                    message=str(exc) or type(exc).__name__,
                ),
                "metrics": update_metrics(state, joining_latency_ms=total_join_ms),
            }
        except JoinerOutputError as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            total_join_ms = (current_metrics.joining_latency_ms or 0.0) + elapsed_ms
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
                    joining_latency_ms=total_join_ms,
                    llm_calls=current_metrics.llm_calls + 1,
                ),
            }
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            total_join_ms = (current_metrics.joining_latency_ms or 0.0) + elapsed_ms
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
                    joining_latency_ms=total_join_ms,
                    llm_calls=current_metrics.llm_calls + 1,
                ),
            }

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_join_ms = (current_metrics.joining_latency_ms or 0.0) + elapsed_ms

        if joiner_result.action == "replan":
            reason = joiner_result.replan_reason or "Follow-up replan requested."
            archived_round = ExecutionRound(
                round_index=len(history) + 1,
                plan=plan,
                execution_result=execution_result,
                replan_reason=reason,
            )
            new_history = [*history, archived_round]
            replan_ctx = format_replan_context(new_history, reason)
            return {
                "joiner_result": joiner_result,
                "final_answer": None,
                "status": None,
                "error": None,
                "compile_feedback": None,
                "round_compile_attempts": 0,
                "execution_history": new_history,
                "replan_reason": reason,
                "replan_context": replan_ctx,
                "metrics": update_metrics(
                    state,
                    joining_latency_ms=total_join_ms,
                    llm_calls=current_metrics.llm_calls + 1,
                    replan_attempts=current_metrics.replan_attempts + 1,
                ),
            }

        result_update: dict[str, object] = {
            "joiner_result": joiner_result,
            "final_answer": joiner_result.answer,
            "status": joiner_result.status,
            "error": None,
            "replan_reason": None,
            "replan_context": None,
            "metrics": update_metrics(
                state,
                joining_latency_ms=total_join_ms,
                llm_calls=current_metrics.llm_calls + 1,
            ),
        }
        if recorded_replan_error is not None:
            result_update["replan_error"] = recorded_replan_error
        return result_update

    return joiner_step
