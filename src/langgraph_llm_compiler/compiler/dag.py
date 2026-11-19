from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from types import MappingProxyType

from langgraph_llm_compiler.planner.schemas import TaskSpec


@dataclass(frozen=True, slots=True)
class CompilationIssue:
    """Single semantic validation error found while compiling an ExecutionPlan."""

    code: str
    message: str
    task_id: str | None = None
    field: str | None = None

    def format(self) -> str:
        location_parts: list[str] = []
        if self.task_id is not None:
            location_parts.append(f"task '{self.task_id}'")
        if self.field is not None:
            location_parts.append(f"field '{self.field}'")
        prefix = f"[{self.code}]"
        if location_parts:
            prefix = f"{prefix} ({', '.join(location_parts)})"
        return f"{prefix}: {self.message}"


class PlanCompilationError(ValueError):
    """Raised when an ExecutionPlan fails semantic validation or DAG compilation."""

    def __init__(self, issues: tuple[CompilationIssue, ...] | list[CompilationIssue]) -> None:
        self.issues: tuple[CompilationIssue, ...] = tuple(issues)
        details = "\n".join(f"- {issue.format()}" for issue in self.issues)
        super().__init__(
            f"ExecutionPlan compilation failed with {len(self.issues)} issue(s):\n{details}"
        )


@dataclass(frozen=True, slots=True)
class CompiledDAG:
    """Read-only compiled DAG artifact ready for parallel execution."""

    objective: str
    tasks: Mapping[str, TaskSpec]
    dependencies: Mapping[str, frozenset[str]]
    dependents: Mapping[str, frozenset[str]]
    topological_order: tuple[str, ...]
    levels: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.tasks, MappingProxyType):
            object.__setattr__(self, "tasks", MappingProxyType(dict(self.tasks)))
        if not isinstance(self.dependencies, MappingProxyType):
            object.__setattr__(
                self,
                "dependencies",
                MappingProxyType({k: frozenset(v) for k, v in self.dependencies.items()}),
            )
        if not isinstance(self.dependents, MappingProxyType):
            object.__setattr__(
                self,
                "dependents",
                MappingProxyType({k: frozenset(v) for k, v in self.dependents.items()}),
            )

    @property
    def root_tasks(self) -> tuple[str, ...]:
        """Tasks with no upstream dependencies."""
        return self.levels[0] if self.levels else ()

    @property
    def leaf_tasks(self) -> tuple[str, ...]:
        """Tasks that no other task depends on."""
        return tuple(
            task_id for task_id in self.topological_order if not self.dependents[task_id]
        )

    def get_task(self, task_id: str) -> TaskSpec:
        return self.tasks[task_id]

    def is_ready(self, task_id: str, completed: AbstractSet[str]) -> bool:
        """Return True if all upstream dependencies of task_id are in completed."""
        return self.dependencies[task_id].issubset(completed)
