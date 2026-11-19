from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

TaskStatus = Literal["success", "error", "skipped"]
ErrorStage = Literal[
    "binding",
    "input_validation",
    "tool_execution",
    "output_validation",
    "timeout",
]


class TaskExecutionResult(BaseModel):
    """Result of executing a single task within a compiled DAG."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    tool_name: str
    status: TaskStatus
    output: Any = None
    error: str | None = None
    error_type: str | None = None
    error_stage: ErrorStage | None = None
    skip_reason: str | None = None
    duration_ms: float = 0.0
    # Milliseconds relative to the beginning of DAG execution.
    # None means the task was never started (e.g. skipped).
    started_at_ms: float | None = None
    finished_at_ms: float | None = None


class ExecutionResult(BaseModel):
    """Result of executing an entire compiled DAG."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    objective: str
    results: dict[str, TaskExecutionResult]
    total_duration_ms: float
    max_concurrency: int | None = None

    @property
    def all_succeeded(self) -> bool:
        return all(res.status == "success" for res in self.results.values())

    @property
    def succeeded_task_ids(self) -> tuple[str, ...]:
        return tuple(
            tid for tid, res in self.results.items() if res.status == "success"
        )

    @property
    def failed_task_ids(self) -> tuple[str, ...]:
        return tuple(tid for tid, res in self.results.items() if res.status == "error")

    @property
    def skipped_task_ids(self) -> tuple[str, ...]:
        return tuple(
            tid for tid, res in self.results.items() if res.status == "skipped"
        )

    @property
    def sum_task_duration_ms(self) -> float:
        return sum(res.duration_ms for res in self.results.values())

    @property
    def effective_parallelism(self) -> float:
        if self.total_duration_ms <= 0:
            return 0.0
        return self.sum_task_duration_ms / self.total_duration_ms
