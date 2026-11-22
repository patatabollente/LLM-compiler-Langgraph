from typing import Annotated, Any, Literal, NotRequired, Required, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict

from llm_compiler.state import WorkflowStatus

ReActStage = Literal["agent", "tools"]


class ReActError(BaseModel):
    """Serializable error recorded when the ReAct workflow fails fatally."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    stage: ReActStage
    error_type: str
    message: str


class ReActMetrics(BaseModel):
    """Telemetry collected across a LangGraph ReAct agent run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    llm_latency_ms: float = 0.0
    tool_latency_ms: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    last_tool_turn_errors: int = 0
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def component_latency_ms(self) -> float:
        return self.llm_latency_ms + self.tool_latency_ms


class ReActState(TypedDict, total=False):
    """Shared state passed through the LangGraph ReAct workflow.

    Uses the same `{"request": "..."}` entry contract as `CompilerState`
    so ReAct and LLM Compiler graphs can be benchmarked side by side.
    """

    request: Required[str]
    messages: NotRequired[Annotated[list[AnyMessage], add_messages]]
    final_answer: NotRequired[str | None]
    status: NotRequired[WorkflowStatus | None]
    error: NotRequired[ReActError | None]
    metrics: NotRequired[ReActMetrics]


def update_react_metrics(state: ReActState, **updates: Any) -> ReActMetrics:
    """Return a new ReActMetrics instance preserving prior fields."""
    current = state.get("metrics") or ReActMetrics()
    return current.model_copy(update=updates)
