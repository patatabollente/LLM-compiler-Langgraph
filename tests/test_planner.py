import asyncio
from typing import Any

import pytest

from langgraph_llm_compiler.planner import (
    ExecutionPlan,
    InputBinding,
    Planner,
    PlannerOutputError,
    TaskSpec,
)


class FakeStructuredRunnable:
    def __init__(self, response: dict[str, Any]) -> None:
        self._response = response
        self.received_messages: list[Any] = []

    def invoke(self, messages: list[Any]) -> dict[str, Any]:
        self.received_messages = messages
        return self._response

    async def ainvoke(self, messages: list[Any]) -> dict[str, Any]:
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
        objective="Differenza popolazioni Italia e Francia",
        tasks=[
            TaskSpec(
                task_id="task_1",
                tool_name="population_lookup",
                description="Popolazione Italia",
                inputs={"country": "Italy"},
            ),
            TaskSpec(
                task_id="task_2",
                tool_name="population_lookup",
                description="Popolazione Francia",
                inputs={"country": "France"},
            ),
            TaskSpec(
                task_id="task_3",
                tool_name="calculator",
                description="Sottrazione",
                inputs={
                    "operation": "subtract",
                    "a": InputBinding(task_id="task_1", output_path=["population"]),
                    "b": InputBinding(task_id="task_2", output_path=["population"]),
                },
            ),
        ],
    )


def test_planner_configures_structured_output_and_returns_plan():
    expected = _sample_plan()
    fake_llm = FakeChatModel({"parsed": expected, "parsing_error": None, "raw": "{}"})
    planner = Planner(llm=fake_llm)  # type: ignore[arg-type]

    tools = {
        "population_lookup": "Args: country (str). Output: {population: int}",
        "calculator": "Args: operation, a, b. Output: {result: float}",
    }

    plan = planner.plan("  Calcola la differenza tra Italia e Francia  ", tools)

    assert plan == expected
    assert fake_llm.structured_output_kwargs["schema"] is ExecutionPlan
    assert fake_llm.structured_output_kwargs["method"] == "json_schema"
    assert fake_llm.structured_output_kwargs["include_raw"] is True

    messages = fake_llm.runnable.received_messages
    assert len(messages) == 2
    assert "population_lookup" in messages[1].content
    assert "Calcola la differenza tra Italia e Francia" in messages[1].content


def test_planner_aplan_returns_plan():
    expected = _sample_plan()
    fake_llm = FakeChatModel({"parsed": expected, "parsing_error": None, "raw": "{}"})
    planner = Planner(llm=fake_llm)  # type: ignore[arg-type]

    plan = asyncio.run(
        planner.aplan("Calcola la differenza", {"calculator": "Aritmetica"})
    )
    assert plan == expected


def test_planner_rejects_invalid_inputs():
    fake_llm = FakeChatModel({"parsed": _sample_plan(), "parsing_error": None, "raw": "{}"})
    planner = Planner(llm=fake_llm)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="request"):
        planner.plan("   ", {"calculator": "desc"})

    with pytest.raises(ValueError, match="available tool"):
        planner.plan("Fai qualcosa", {})

    with pytest.raises(ValueError, match="Tool names and descriptions"):
        planner.plan("Fai qualcosa", {"   ": "desc"})

    with pytest.raises(ValueError, match="Tool names and descriptions"):
        planner.plan("Fai qualcosa", {"calculator": "   "})


def test_planner_raises_output_error_preserving_raw_output():
    raw_payload = '{"objective": "bad", "tasks": []}'
    fake_llm = FakeChatModel(
        {
            "parsed": None,
            "parsing_error": ValueError("tasks must have at least 1 item"),
            "raw": raw_payload,
        }
    )
    planner = Planner(llm=fake_llm)  # type: ignore[arg-type]

    with pytest.raises(PlannerOutputError) as exc_info:
        planner.plan("Richiesta test", {"calculator": "desc"})

    assert exc_info.value.raw_output == raw_payload


def test_planner_includes_previous_plan_and_feedback_on_retry():
    expected = _sample_plan()
    fake_llm = FakeChatModel({"parsed": expected, "parsing_error": None, "raw": "{}"})
    planner = Planner(llm=fake_llm)  # type: ignore[arg-type]

    # Blank feedback must not inject correction sections.
    planner.plan("Calcola", {"calculator": "desc"}, previous_plan=expected, feedback="   ")
    first_prompt = fake_llm.runnable.received_messages[1].content
    assert "COMPILATION ERRORS TO FIX:" not in first_prompt
    assert "PREVIOUS EXECUTION PLAN" not in first_prompt

    # Real feedback + previous_plan must inject both sections cleanly.
    feedback = "- [malformed_binding] (task 'task_3', field 'b'): extra key 'output, path'"
    asyncio.run(
        planner.aplan(
            "Calcola",
            {"calculator": "desc"},
            previous_plan=expected,
            feedback=feedback,
        )
    )
    retry_prompt = fake_llm.runnable.received_messages[1].content
    assert "PREVIOUS EXECUTION PLAN (FAILED COMPILATION):" in retry_prompt
    assert '"task_id": "task_3"' in retry_prompt
    assert "COMPILATION ERRORS TO FIX:" in retry_prompt
    assert "malformed_binding" in retry_prompt
