from typing import Any, Literal, NotRequired, Required, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from llm_compiler.compiler.dag import CompiledDAG
from llm_compiler.executor.results import ExecutionResult
from llm_compiler.joiner.schemas import ExecutionRound, JoinerResult
from llm_compiler.planner.schemas import ExecutionPlan

WorkflowStage = Literal["planner", "compiler", "executor", "joiner"]
WorkflowStatus = Literal["complete", "partial", "failed"]


class WorkflowError(BaseModel):
    """Serializable error recorded when a workflow node fails."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    stage: WorkflowStage
    error_type: str
    message: str
    issues: list[str] = Field(default_factory=list)


class WorkflowMetrics(BaseModel):
    """Per-component telemetry collected across the LangGraph workflow."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    planning_latency_ms: float | None = None
    compilation_latency_ms: float | None = None
    execution_latency_ms: float | None = None
    joining_latency_ms: float | None = None
    llm_calls: int = 0
    compile_attempts: int = 0
    compile_retries: int = 0
    replan_attempts: int = 0
    plan_valid: bool | None = None

    @property
    def component_latency_ms(self) -> float:
        """Sum of recorded latencies for executed components (excludes graph overhead)."""
        latencies = (
            self.planning_latency_ms,
            self.compilation_latency_ms,
            self.execution_latency_ms,
            self.joining_latency_ms,
        )
        return sum(val for val in latencies if val is not None)


class CompilerState(TypedDict, total=False):
    """Shared state passed through the LangGraph LLM Compiler workflow."""

    request: Required[str]
    plan: NotRequired[ExecutionPlan | None]
    compiled_dag: NotRequired[CompiledDAG | None]
    execution_result: NotRequired[ExecutionResult | None]
    joiner_result: NotRequired[JoinerResult | None]
    final_answer: NotRequired[str | None]
    status: NotRequired[WorkflowStatus | None]
    error: NotRequired[WorkflowError | None]
    replan_error: NotRequired[WorkflowError | None]
    compile_feedback: NotRequired[str | None]
    compile_error_history: NotRequired[list[list[str]]]
    round_compile_attempts: NotRequired[int]
    execution_history: NotRequired[list[ExecutionRound]]
    replan_reason: NotRequired[str | None]
    replan_context: NotRequired[str | None]
    metrics: NotRequired[WorkflowMetrics]


def update_metrics(state: CompilerState, **updates: Any) -> WorkflowMetrics:
    """Return a new WorkflowMetrics instance preserving prior fields."""
    current = state.get("metrics") or WorkflowMetrics()
    return current.model_copy(update=updates)
