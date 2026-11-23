"""Per-run telemetry instrumentation for LangGraph Parallel ReAct."""

from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from llm_compiler.state import WorkflowStatus

from .telemetry import (
    ParallelRunResult,
    RunStatus,
    TokenUsage,
    TokenUsageSource,
    ToolCallTrace,
    TurnTrace,
    WaveTrace,
)


def _valid_count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _valid_positive_count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    return None


def _sum_optional(values: Iterable[int | None]) -> int | None:
    nums = [v for v in values if v is not None]
    return sum(nums) if nums else None


def _json_safe_value(value: Any, depth: int = 0) -> Any:
    if depth > 12:
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, BaseModel):
        return _json_safe_value(value.model_dump(mode="python"), depth + 1)
    if isinstance(value, Mapping):
        return {str(k): _json_safe_value(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe_value(item, depth + 1) for item in value]
    return repr(value)


def _json_safe_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {str(k): _json_safe_value(v) for k, v in arguments.items()}


def extract_message_token_usage(msg: Any) -> TokenUsage:
    """Extract provider token usage (including cached input tokens) from a LangChain message."""
    usage = getattr(msg, "usage_metadata", None)
    meta = getattr(msg, "response_metadata", None)

    in_tok: int | None = None
    out_tok: int | None = None
    total_tok: int | None = None
    cached_tok: int | None = None

    if isinstance(usage, Mapping):
        in_tok = _valid_positive_count(usage.get("input_tokens"))
        out_tok = _valid_positive_count(usage.get("output_tokens"))
        total_tok = _valid_positive_count(usage.get("total_tokens"))
        in_details = usage.get("input_token_details")
        if isinstance(in_details, Mapping):
            cached_tok = _valid_count(in_details.get("cache_read"))

    if isinstance(meta, Mapping):
        if in_tok is None:
            in_tok = _valid_positive_count(
                meta.get("prompt_eval_count") or meta.get("prompt_tokens")
            )
        if out_tok is None:
            out_tok = _valid_positive_count(
                meta.get("eval_count") or meta.get("completion_tokens")
            )
        if cached_tok is None:
            cached_tok = _valid_count(meta.get("prompt_eval_cached_count"))
        token_usage = meta.get("token_usage")
        if isinstance(token_usage, Mapping):
            if in_tok is None:
                in_tok = _valid_positive_count(token_usage.get("prompt_tokens"))
            if out_tok is None:
                out_tok = _valid_positive_count(token_usage.get("completion_tokens"))
            if total_tok is None:
                total_tok = _valid_positive_count(token_usage.get("total_tokens"))
            if cached_tok is None:
                details = token_usage.get("prompt_tokens_details")
                if isinstance(details, Mapping):
                    cached_tok = _valid_count(details.get("cached_tokens"))

    if in_tok is not None and out_tok is not None:
        source: TokenUsageSource = "provider"
        if total_tok is None:
            total_tok = in_tok + out_tok
    elif in_tok is not None or out_tok is not None or total_tok is not None:
        source = "mixed"
        if total_tok is None:
            total_tok = (in_tok or 0) + (out_tok or 0)
    else:
        source = "unavailable"

    return TokenUsage(
        input_tokens=in_tok,
        output_tokens=out_tok,
        total_tokens=total_tok,
        cached_input_tokens=cached_tok,
        source=source,
    )


@dataclass
class _ToolExecutionRecord:
    tool_name: str
    raw_arguments: dict[str, Any]
    arguments: dict[str, Any]
    start_ms: float
    end_ms: float | None = None
    result: str | None = None
    error: str | None = None
    error_type: str | None = None


class ReActRunInstrumentation:
    """Per-run accumulator shared by agent_node and wrapped tools."""

    def __init__(self, started_at: float | None = None) -> None:
        self.started_at = time.perf_counter() if started_at is None else started_at
        self.turns: list[TurnTrace] = []
        self._active_turn_start_ms: float | None = None
        self.time_to_first_token_ms: float | None = None
        self._planned_calls: list[dict[str, Any]] = []
        self._executions: dict[tuple[str, str], deque[_ToolExecutionRecord]] = defaultdict(deque)
        self._tool_results_by_id: dict[str, tuple[str | None, bool]] = {}
        self.status: RunStatus = "completed"
        self.termination_reason: str = "completed"
        self.timed_out: bool = False
        self.error: str | None = None

    def elapsed_ms(self) -> float:
        return max(0.0, (time.perf_counter() - self.started_at) * 1000.0)

    def start_turn(self) -> int:
        self._active_turn_start_ms = self.elapsed_ms()
        return len(self.turns)

    def record_first_token(self) -> None:
        if self.time_to_first_token_ms is None and self._active_turn_start_ms is not None:
            self.time_to_first_token_ms = max(0.0, self.elapsed_ms() - self._active_turn_start_ms)

    def finish_turn(
        self,
        *,
        turn_index: int,
        token_usage: TokenUsage,
        tool_calls: Sequence[Mapping[str, Any]] = (),
        text: str | None = None,
        error: str | None = None,
    ) -> None:
        start_ms = (
            self._active_turn_start_ms
            if self._active_turn_start_ms is not None
            else self.elapsed_ms()
        )
        end_ms = max(start_ms, self.elapsed_ms())
        self._active_turn_start_ms = None

        call_ids: list[str] = []
        for idx, tc in enumerate(tool_calls):
            tc_id = str(tc.get("id") or f"turn_{turn_index}_call_{idx}")
            call_ids.append(tc_id)
            raw_args = tc.get("args") if isinstance(tc.get("args"), Mapping) else {}
            self._planned_calls.append(
                {
                    "id": tc_id,
                    "name": str(tc.get("name") or ""),
                    "raw_arguments": dict(raw_args),
                    "turn": turn_index,
                }
            )

        self.turns.append(
            TurnTrace(
                turn_index=turn_index,
                start_ms=start_ms,
                end_ms=end_ms,
                latency_ms=max(0.0, end_ms - start_ms),
                input_tokens=token_usage.input_tokens,
                output_tokens=token_usage.output_tokens,
                total_tokens=token_usage.total_tokens,
                cached_input_tokens=token_usage.cached_input_tokens,
                tool_call_count=len(call_ids),
                tool_call_ids=call_ids,
                text=text,
                error=error,
            )
        )

    def start_tool(
        self,
        name: str,
        raw_arguments: Mapping[str, Any],
        validated_arguments: Mapping[str, Any],
    ) -> _ToolExecutionRecord:
        rec = _ToolExecutionRecord(
            tool_name=name,
            raw_arguments=dict(raw_arguments),
            arguments=dict(validated_arguments),
            start_ms=self.elapsed_ms(),
        )
        self._executions[self._call_key(name, raw_arguments)].append(rec)
        return rec

    def finish_tool(
        self,
        rec: _ToolExecutionRecord,
        *,
        result: str | None = None,
        error: Exception | None = None,
    ) -> None:
        rec.end_ms = self.elapsed_ms()
        if error is not None:
            rec.error_type = type(error).__name__
            rec.error = f"{type(error).__name__}: {error}"
        else:
            rec.result = result

    def record_tool_message(
        self,
        tool_call_id: str,
        *,
        content: str | None,
        is_error: bool,
    ) -> None:
        self._tool_results_by_id[str(tool_call_id)] = (content, is_error)

    def build_result(
        self,
        *,
        answer: str | None,
        workflow_status: WorkflowStatus | None,
        applied_settings: Mapping[str, Any] = (),
        skipped_settings: Sequence[str] = (),
    ) -> ParallelRunResult:
        traces = self._build_tool_traces()
        waves = self._build_waves(traces)
        total_wall_ms = self.elapsed_ms()
        sum_llm_ms = sum(t.latency_ms for t in self.turns)
        sum_wave_ms = sum(w.wall_ms for w in waves)
        tokens = self._aggregate_tokens()

        warnings: list[str] = []
        if tokens.source == "unavailable" and self.turns:
            warnings.append(
                "Token usage is unavailable because no countable provider usage was reported."
            )

        return ParallelRunResult(
            status=self.status,
            workflow_status=workflow_status,
            answer=answer,
            error=self.error,
            termination_reason=self.termination_reason,
            timed_out=self.timed_out,
            total_wall_ms=total_wall_ms,
            time_to_first_token_ms=self.time_to_first_token_ms,
            sum_llm_ms=sum_llm_ms,
            sum_wave_ms=sum_wave_ms,
            overhead_ms=max(0.0, total_wall_ms - sum_llm_ms - sum_wave_ms),
            llm_round_trips=len(self.turns),
            tool_call_count=len(traces),
            tokens=tokens,
            turns=list(self.turns),
            waves=waves,
            tool_calls=traces,
            applied_settings=dict(applied_settings),
            skipped_settings=list(skipped_settings),
            warnings=warnings,
        )

    def _aggregate_tokens(self) -> TokenUsage:
        if not self.turns:
            return TokenUsage(source="unavailable")
        in_tok = _sum_optional(t.input_tokens for t in self.turns)
        out_tok = _sum_optional(t.output_tokens for t in self.turns)
        tot_tok = _sum_optional(t.total_tokens for t in self.turns)
        cached_tok = _sum_optional(t.cached_input_tokens for t in self.turns)

        if in_tok is not None and out_tok is not None:
            source: TokenUsageSource = "provider"
            if tot_tok is None:
                tot_tok = in_tok + out_tok
        elif in_tok is not None or out_tok is not None or tot_tok is not None:
            source = "mixed"
            if tot_tok is None:
                tot_tok = (in_tok or 0) + (out_tok or 0)
        else:
            source = "unavailable"

        return TokenUsage(
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=tot_tok,
            cached_input_tokens=cached_tok,
            source=source,
        )

    def _build_tool_traces(self) -> list[ToolCallTrace]:
        traces: list[ToolCallTrace] = []
        for call in self._planned_calls:
            key = self._call_key(call["name"], call["raw_arguments"])
            queue = self._executions.get(key)
            rec = queue.popleft() if queue else None
            msg_content, msg_is_err = self._tool_results_by_id.get(call["id"], (None, False))

            call_error = rec.error if rec else None
            error_type = rec.error_type if rec else None
            if call_error is None and msg_is_err:
                call_error = msg_content or "Tool returned an error."
                error_type = "ToolError"

            not_executed = rec is None and call["id"] not in self._tool_results_by_id
            if call_error is None and not_executed:
                call_error = "Tool did not complete before the workflow ended."
                error_type = "NotExecuted"

            start_ms = rec.start_ms if rec is not None else 0.0
            end_ms = rec.end_ms if (rec is not None and rec.end_ms is not None) else start_ms
            result = rec.result if rec is not None else (None if msg_is_err else msg_content)

            traces.append(
                ToolCallTrace(
                    wave_index=call["turn"],
                    turn_index=call["turn"],
                    tool_call_id=call["id"],
                    tool_name=call["name"],
                    raw_arguments=_json_safe_arguments(call["raw_arguments"]),
                    arguments=_json_safe_arguments(
                        rec.arguments if rec is not None else call["raw_arguments"]
                    ),
                    result=result,
                    error=call_error,
                    error_type=error_type,
                    ok=call_error is None,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    duration_ms=max(0.0, end_ms - start_ms),
                )
            )
        return traces

    @staticmethod
    def _build_waves(traces: Sequence[ToolCallTrace]) -> list[WaveTrace]:
        groups: dict[int, list[ToolCallTrace]] = defaultdict(list)
        for trace in traces:
            groups[trace.wave_index].append(trace)

        waves: list[WaveTrace] = []
        for wave_index, group in sorted(groups.items()):
            executed = [t for t in group if t.end_ms > 0.0 or t.start_ms > 0.0]
            if executed:
                start_ms = min(t.start_ms for t in executed)
                end_ms = max(t.end_ms for t in executed)
            else:
                start_ms = 0.0
                end_ms = 0.0
            wall_ms = max(0.0, end_ms - start_ms)
            sum_tool_ms = sum(t.duration_ms for t in group)
            speedup = (
                min(float(len(group)), sum_tool_ms / wall_ms) if wall_ms > 0.0 else None
            )
            waves.append(
                WaveTrace(
                    wave_index=wave_index,
                    turn_index=wave_index,
                    tool_call_count=len(group),
                    tool_call_ids=[t.tool_call_id for t in group],
                    tool_names=[t.tool_name for t in group],
                    start_ms=start_ms,
                    end_ms=end_ms,
                    wall_ms=wall_ms,
                    sum_tool_ms=sum_tool_ms,
                    speedup=speedup,
                    parallel_efficiency=(
                        min(1.0, speedup / len(group)) if speedup is not None else None
                    ),
                )
            )
        return waves

    @staticmethod
    def _call_key(name: str, arguments: Mapping[str, Any]) -> tuple[str, str]:
        return name, json.dumps(
            _json_safe_arguments(arguments),
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
