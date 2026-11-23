"""Serializable result and telemetry models for a LangGraph Parallel ReAct run.

Field names and formulas intentionally mirror `parallel_react_llamaindex.telemetry`
so LangGraph and LlamaIndex Parallel ReAct runs can be compared apples-to-apples.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from llm_compiler.state import WorkflowStatus

RunStatus = Literal["completed", "max_turns_exceeded", "error"]
TokenUsageSource = Literal["provider", "estimated", "mixed", "unavailable"]


class TokenUsage(BaseModel):
    """Provider token counts and optional cached-prompt breakdown."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = Field(
        default=None,
        description="Provider-reported total when available, otherwise input_tokens + output_tokens.",
    )
    cached_input_tokens: int | None = None
    source: TokenUsageSource = "unavailable"


class TurnTrace(BaseModel):
    """One LLM request/response cycle."""

    turn_index: int
    start_ms: float
    end_ms: float
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None
    tool_call_count: int = 0
    tool_call_ids: list[str] = Field(default_factory=list)
    text: str | None = None
    error: str | None = None


class ToolCallTrace(BaseModel):
    """One dispatched tool call inside a wave."""

    wave_index: int = -1
    turn_index: int = -1
    tool_call_id: str
    tool_name: str
    raw_arguments: dict[str, Any] = Field(default_factory=dict)
    arguments: dict[str, Any] | None = None
    result: str | None = None
    error: str | None = None
    error_type: str | None = None
    ok: bool = True
    start_ms: float = 0.0
    end_ms: float = 0.0
    duration_ms: float = 0.0


class WaveTrace(BaseModel):
    """All tool calls emitted by a single model response."""

    wave_index: int
    turn_index: int
    tool_call_count: int
    tool_call_ids: list[str] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)
    start_ms: float
    end_ms: float
    wall_ms: float
    sum_tool_ms: float
    speedup: float | None = None
    parallel_efficiency: float | None = None


class ParallelRunResult(BaseModel):
    """Serializable outcome and per-turn/per-wave telemetry for one run."""

    status: RunStatus
    workflow_status: WorkflowStatus | None = Field(
        default=None,
        description="Recovery-aware deterministic completion status ('complete' | 'partial' | 'failed').",
    )
    answer: str | None = None
    error: str | None = None
    termination_reason: str | None = None
    timed_out: bool = False

    total_wall_ms: float = 0.0
    time_to_first_token_ms: float | None = None
    sum_llm_ms: float = 0.0
    sum_wave_ms: float = 0.0
    overhead_ms: float = 0.0
    llm_round_trips: int = 0
    tool_call_count: int = 0

    tokens: TokenUsage = Field(default_factory=TokenUsage)
    turns: list[TurnTrace] = Field(default_factory=list)
    waves: list[WaveTrace] = Field(default_factory=list)
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)

    applied_settings: dict[str, Any] = Field(default_factory=dict)
    skipped_settings: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
