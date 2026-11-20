import asyncio
from typing import Any

import pytest
from pydantic import ValidationError

from langgraph_llm_compiler.executor import ExecutionResult, TaskExecutionResult
from langgraph_llm_compiler.joiner import (
    REPLAN_SYSTEM_PROMPT,
    ExecutionRound,
    Joiner,
    JoinerDecisionOutput,
    JoinerOutput,
    JoinerOutputError,
    JoinerResult,
    classify_execution_status,
    extract_missing_information,
    format_replan_context,
)
from langgraph_llm_compiler.planner import ExecutionPlan, InputBinding, TaskSpec


class FakeStructuredRunnable:
    def __init__(self, response: dict[str, Any]) -> None:
        self._response = response
        self.received_messages: list[Any] = []
        self.call_count: int = 0

    async def ainvoke(self, messages: list[Any]) -> dict[str, Any]:
        self.call_count += 1
        self.received_messages = messages
        return self._response


class FakeChatModel:
    def __init__(self, response: dict[str, Any]) -> None:
        self.runnable = FakeStructuredRunnable(response)
        self.structured_output_kwargs: dict[str, Any] = {}

    def with_structured_output(self, **kwargs: Any) -> FakeStructuredRunnable:
        self.structured_output_kwargs = kwargs
        return self.runnable


def _sample_plan() -> ExecutionPlan:
    return ExecutionPlan(
        objective="Calculate the difference between the population of Italy and France.",
        tasks=[
            TaskSpec(
                task_id="task_1",
                tool_name="population_lookup",
                description="Get the population of Italy.",
                inputs={"country": "Italy"},
            ),
            TaskSpec(
                task_id="task_2",
                tool_name="population_lookup",
                description="Get the population of France.",
                inputs={"country": "France"},
            ),
            TaskSpec(
                task_id="task_3",
                tool_name="calculator",
                description="Subtract the population of France from Italy.",
                inputs={
                    "operation": "subtract",
                    "a": InputBinding(task_id="task_1", output_path=["population"]),
                    "b": InputBinding(task_id="task_2", output_path=["population"]),
                },
            ),
        ],
    )


def _complete_execution_result() -> ExecutionResult:
    return ExecutionResult(
        objective="Calculate the difference between the population of Italy and France.",
        total_duration_ms=200.0,
        max_concurrency=8,
        results={
            "task_1": TaskExecutionResult(
                task_id="task_1",
                tool_name="population_lookup",
                status="success",
                output={"country": "Italy", "population": 58_990_000},
                duration_ms=150.0,
                started_at_ms=0.0,
                finished_at_ms=150.0,
            ),
            "task_2": TaskExecutionResult(
                task_id="task_2",
                tool_name="population_lookup",
                status="success",
                output={"country": "France", "population": 68_400_000},
                duration_ms=150.0,
                started_at_ms=0.0,
                finished_at_ms=150.0,
            ),
            "task_3": TaskExecutionResult(
                task_id="task_3",
                tool_name="calculator",
                status="success",
                output={"result": -9_410_000.0},
                duration_ms=50.0,
                started_at_ms=150.0,
                finished_at_ms=200.0,
            ),
        },
    )


def _partial_execution_result() -> ExecutionResult:
    return ExecutionResult(
        objective="Calculate the difference between the population of Italy and France.",
        total_duration_ms=150.0,
        max_concurrency=8,
        results={
            "task_1": TaskExecutionResult(
                task_id="task_1",
                tool_name="population_lookup",
                status="error",
                error="Upstream service unavailable",
                error_type="RuntimeError",
                error_stage="tool_execution",
                duration_ms=10.0,
                started_at_ms=0.0,
                finished_at_ms=10.0,
            ),
            "task_2": TaskExecutionResult(
                task_id="task_2",
                tool_name="population_lookup",
                status="success",
                output={"country": "France", "population": 68_400_000},
                duration_ms=150.0,
                started_at_ms=0.0,
                finished_at_ms=150.0,
            ),
            "task_3": TaskExecutionResult(
                task_id="task_3",
                tool_name="calculator",
                status="skipped",
                skip_reason="Skipped because upstream task 'task_1' failed.",
                duration_ms=0.0,
                started_at_ms=None,
                finished_at_ms=None,
            ),
        },
    )


def _failed_execution_result() -> ExecutionResult:
    return ExecutionResult(
        objective="Calculate the difference between the population of Italy and France.",
        total_duration_ms=20.0,
        max_concurrency=8,
        results={
            "task_1": TaskExecutionResult(
                task_id="task_1",
                tool_name="population_lookup",
                status="error",
                error="Timeout after 1.0s",
                error_type="TimeoutError",
                error_stage="timeout",
                duration_ms=10.0,
                started_at_ms=0.0,
                finished_at_ms=10.0,
            ),
            "task_2": TaskExecutionResult(
                task_id="task_2",
                tool_name="population_lookup",
                status="error",
                error="Service offline",
                error_type="ConnectionError",
                error_stage="tool_execution",
                duration_ms=10.0,
                started_at_ms=0.0,
                finished_at_ms=10.0,
            ),
            "task_3": TaskExecutionResult(
                task_id="task_3",
                tool_name="calculator",
                status="skipped",
                skip_reason="Skipped because upstream task 'task_1' failed.",
                duration_ms=0.0,
                started_at_ms=None,
                finished_at_ms=None,
            ),
        },
    )


def test_joiner_complete_execution_with_plan():
    fake_llm = FakeChatModel(
        {
            "parsed": JoinerOutput(
                answer="La differenza tra Italia e Francia è di -9.410.000 abitanti."
            ),
            "parsing_error": None,
            "raw": '{"answer": "..."}',
        }
    )
    joiner = Joiner(llm=fake_llm)  # type: ignore[arg-type]

    request = "Calcola la differenza tra la popolazione dell'Italia e della Francia."
    result = asyncio.run(
        joiner.join(
            request=request,
            execution_result=_complete_execution_result(),
            plan=_sample_plan(),
        )
    )

    assert isinstance(result, JoinerResult)
    assert result.status == "complete"
    assert result.missing_information == []
    assert "-9.410.000" in result.answer

    assert fake_llm.structured_output_kwargs["schema"] is JoinerOutput
    assert fake_llm.structured_output_kwargs["method"] == "json_schema"
    assert fake_llm.structured_output_kwargs["include_raw"] is True
    assert fake_llm.runnable.call_count == 1

    messages = fake_llm.runnable.received_messages
    assert len(messages) == 2
    human_content = messages[1].content
    assert request in human_content
    assert "Get the population of Italy." in human_content
    assert "Subtract the population of France from Italy." in human_content
    assert "-9410000.0" in human_content
    assert "MISSING INFORMATION:" not in human_content


def test_joiner_partial_execution_builds_deterministic_missing_info():
    fake_llm = FakeChatModel(
        {
            "parsed": JoinerOutput(
                answer=(
                    "Ho recuperato solo la popolazione della Francia (68.400.000), "
                    "ma non quella dell'Italia."
                )
            ),
            "parsing_error": None,
            "raw": "{}",
        }
    )
    joiner = Joiner(llm=fake_llm)  # type: ignore[arg-type]

    result = asyncio.run(
        joiner.join(
            request="Differenza popolazione Italia e Francia",
            execution_result=_partial_execution_result(),
            plan=_sample_plan(),
        )
    )

    assert result.status == "partial"
    assert len(result.missing_information) == 2
    # Must follow topological order (task_1 before task_3) and use plan descriptions.
    assert result.missing_information[0].startswith("Get the population of Italy.")
    assert "Upstream service unavailable" in result.missing_information[0]
    assert result.missing_information[1].startswith(
        "Subtract the population of France from Italy."
    )
    assert "Skipped because upstream task 'task_1' failed." in result.missing_information[1]

    human_content = fake_llm.runnable.received_messages[1].content
    assert "EXECUTION STATUS:\npartial" in human_content
    assert "Upstream service unavailable" in human_content
    assert "Skipped because upstream task 'task_1' failed." in human_content
    assert "MISSING INFORMATION:" in human_content


def test_joiner_all_failed_execution_still_calls_llm():
    fake_llm = FakeChatModel(
        {
            "parsed": JoinerOutput(
                answer="Non è stato possibile recuperare i dati richiesti a causa di errori dei servizi."
            ),
            "parsing_error": None,
            "raw": "{}",
        }
    )
    joiner = Joiner(llm=fake_llm)  # type: ignore[arg-type]

    result = asyncio.run(
        joiner.join(
            request="Differenza popolazione Italia e Francia",
            execution_result=_failed_execution_result(),
            plan=_sample_plan(),
        )
    )

    assert fake_llm.runnable.call_count == 1
    assert result.status == "failed"
    assert len(result.missing_information) == 3
    assert "TimeoutError" in result.missing_information[0]
    assert "ConnectionError" in result.missing_information[1]
    assert "skipped" in result.missing_information[2]


def test_joiner_without_plan_falls_back_to_tool_name():
    fake_llm = FakeChatModel(
        {
            "parsed": JoinerOutput(answer="Dati parziali disponibili solo per la Francia."),
            "parsing_error": None,
            "raw": "{}",
        }
    )
    joiner = Joiner(llm=fake_llm)  # type: ignore[arg-type]

    partial_res = _partial_execution_result()
    result = asyncio.run(
        joiner.join(
            request="Differenza popolazione Italia e Francia",
            execution_result=partial_res,
            plan=None,
        )
    )

    assert result.status == "partial"
    assert len(result.missing_information) == 2
    assert result.missing_information[0].startswith("population_lookup")
    assert result.missing_information[1].startswith("calculator")
    assert all(item.strip() for item in result.missing_information)

    # Direct check of pure helper functions too.
    assert classify_execution_status(partial_res) == "partial"
    assert extract_missing_information(partial_res, plan=None) == result.missing_information


def test_joiner_rejects_empty_request_and_empty_results():
    fake_llm = FakeChatModel(
        {
            "parsed": JoinerOutput(answer="ok"),
            "parsing_error": None,
            "raw": "{}",
        }
    )
    joiner = Joiner(llm=fake_llm)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="request"):
        asyncio.run(joiner.join("   ", _complete_execution_result()))

    empty_exec = ExecutionResult(
        objective="Empty",
        results={},
        total_duration_ms=0.0,
    )
    with pytest.raises(ValueError, match="at least one task"):
        asyncio.run(joiner.join("Domanda valida", empty_exec))

    assert fake_llm.runnable.call_count == 0


def test_joiner_raises_output_error_preserving_raw_output():
    raw_payload = '{"answer": ""}'
    fake_llm = FakeChatModel(
        {
            "parsed": None,
            "parsing_error": ValueError("answer must have at least 1 character"),
            "raw": raw_payload,
        }
    )
    joiner = Joiner(llm=fake_llm)  # type: ignore[arg-type]

    with pytest.raises(JoinerOutputError) as exc_info:
        asyncio.run(joiner.join("Richiesta test", _complete_execution_result()))

    assert exc_info.value.raw_output == raw_payload


def test_joiner_prompt_preserves_original_language_instruction():
    fake_llm = FakeChatModel(
        {
            "parsed": JoinerOutput(answer="Risposta in italiano."),
            "parsing_error": None,
            "raw": "{}",
        }
    )
    joiner = Joiner(llm=fake_llm)  # type: ignore[arg-type]

    italian_request = (
        "Calcola la differenza tra la popolazione dell'Italia e quella della Francia."
    )
    asyncio.run(
        joiner.join(
            request=italian_request,
            execution_result=_complete_execution_result(),
            plan=_sample_plan(),
        )
    )

    system_msg, human_msg = fake_llm.runnable.received_messages
    assert "same language as the ORIGINAL USER REQUEST" in system_msg.content
    assert f"ORIGINAL USER REQUEST:\n{italian_request}" in human_msg.content
    assert "EXECUTION OBJECTIVE:\nCalculate the difference" in human_msg.content


def test_joiner_models_are_frozen_and_forbid_extra_fields():
    out = JoinerOutput(answer="  Risposta valida  ")
    assert out.answer == "Risposta valida"

    with pytest.raises(ValidationError):
        out.answer = "Modificata"  # type: ignore[misc]

    with pytest.raises(ValidationError):
        JoinerOutput(answer="   ")

    with pytest.raises(ValidationError):
        JoinerOutput(answer="ok", extra_key=1)  # type: ignore[call-arg]

    res = JoinerResult(answer="ok", status="complete")
    assert res.missing_information == []

    with pytest.raises(ValidationError):
        res.status = "failed"  # type: ignore[misc]

    with pytest.raises(ValidationError):
        JoinerResult(answer="ok", status="partial", missing_information=["   "])

    with pytest.raises(ValidationError):
        JoinerResult(answer="ok", status="complete", unknown=True)  # type: ignore[call-arg]


def test_joiner_decision_output_validation_and_empty_normalization():
    fin = JoinerDecisionOutput(action="finish", answer="  Fatto! ", replan_reason="   ")
    assert fin.action == "finish"
    assert fin.answer == "Fatto!"
    assert fin.replan_reason is None

    rep = JoinerDecisionOutput(
        action="replan", answer="", replan_reason="Use backup_fx_api instead."
    )
    assert rep.action == "replan"
    assert rep.answer is None
    assert rep.replan_reason == "Use backup_fx_api instead."

    with pytest.raises(ValidationError):
        JoinerDecisionOutput(action="finish", answer=None)

    with pytest.raises(ValidationError):
        JoinerDecisionOutput(action="replan", replan_reason="   ")


def test_joiner_v3_replan_action_and_catalog_prompt():
    fake_llm = FakeChatModel(
        {
            "parsed": JoinerDecisionOutput(
                action="replan",
                answer="ignored filler",
                replan_reason="primary lookup failed; call backup_lookup for Italy.",
            ),
            "parsing_error": None,
            "raw": "{}",
        }
    )
    joiner = Joiner(llm=fake_llm)  # type: ignore[arg-type]

    result = asyncio.run(
        joiner.join(
            request="Differenza popolazione Italia meno Francia",
            execution_result=_partial_execution_result(),
            plan=_sample_plan(),
            tool_catalog={
                "population_lookup": "Primary lookup.",
                "backup_lookup": "Backup lookup.",
            },
            allow_replan=True,
        )
    )

    assert fake_llm.structured_output_kwargs["schema"] is JoinerDecisionOutput
    assert result.action == "replan"
    assert result.answer is None
    assert result.replan_reason == "primary lookup failed; call backup_lookup for Italy."
    assert result.status == "partial"

    system_msg, human_msg = fake_llm.runnable.received_messages
    assert system_msg.content == REPLAN_SYSTEM_PROMPT
    assert "AVAILABLE TOOLS:" in human_msg.content
    assert "backup_lookup: Backup lookup." in human_msg.content


def test_classify_execution_status_across_prior_rounds():
    prior_partial = ExecutionRound(
        round_index=1,
        plan=_sample_plan(),
        execution_result=_partial_execution_result(),
        replan_reason="Retry missing Italy with backup.",
    )
    all_failed_round2 = ExecutionResult(
        objective="Backup attempt",
        total_duration_ms=20.0,
        max_concurrency=8,
        results={
            "task_1": TaskExecutionResult(
                task_id="task_1",
                tool_name="backup_lookup",
                status="error",
                error="Also down",
                error_type="RuntimeError",
                error_stage="tool_execution",
                duration_ms=20.0,
                started_at_ms=0.0,
                finished_at_ms=20.0,
            )
        },
    )
    all_succeeded_round2 = ExecutionResult(
        objective="Backup attempt",
        total_duration_ms=20.0,
        max_concurrency=8,
        results={
            "task_1": TaskExecutionResult(
                task_id="task_1",
                tool_name="backup_lookup",
                status="success",
                output={"country": "Italy", "population": 58_990_000},
                duration_ms=20.0,
                started_at_ms=0.0,
                finished_at_ms=20.0,
            )
        },
    )

    assert classify_execution_status(all_succeeded_round2, [prior_partial]) == "complete"
    assert classify_execution_status(all_failed_round2, [prior_partial]) == "partial"
    assert classify_execution_status(all_failed_round2, ()) == "failed"


def test_format_replan_context_includes_succeeded_and_failed_details():
    prior_round = ExecutionRound(
        round_index=1,
        plan=_sample_plan(),
        execution_result=_partial_execution_result(),
        replan_reason="Use backup tool for Italy.",
    )
    ctx = format_replan_context([prior_round], "Use backup tool for Italy.")
    assert "JOINER REPLAN GUIDANCE:" in ctx
    assert "Use backup tool for Italy." in ctx
    assert "ROUND 1:" in ctx
    assert "68400000" in ctx
    assert "Upstream service unavailable" in ctx

