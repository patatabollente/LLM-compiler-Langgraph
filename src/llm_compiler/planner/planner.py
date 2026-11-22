from collections.abc import Mapping
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .schemas import ExecutionPlan

SYSTEM_PROMPT = """
You are a planning component in an LLM Compiler architecture.

Your task is to transform the user's request into a structured execution plan.

Follow these rules:

1. Use only tools present in the available tools catalog.
2. Give every task a unique identifier (task_1, task_2, etc.).
3. Break the request into clear, atomic tasks.
4. Independent tasks should not depend on each other.
5. Put every tool parameter inside the single "inputs" dictionary.
6. Never use separate "arguments", "input_bindings", or "dependencies" fields.
7. For a statically known parameter value (such as a string, boolean, or constant number like 4), put the literal value directly in "inputs".
8. For a dynamic parameter value obtained from a previous task's output, put an InputBinding object {"task_id": "<upstream_id>", "output_path": ["key1", "key2"]} directly as that parameter's value in "inputs". Never wrap an InputBinding inside a nested dict or list (e.g. use "value": {"task_id": "task_1", "output_path": ["items"]}, NOT "value": {"items": {"task_id": "task_1", "output_path": ["items"]}}).
9. Use output_path to identify the required field in a task output (for nested objects, list every key in order, e.g. ["economy", "gdp"]). Use [] to refer to the entire output.
10. Make sure all referenced tasks exist in the execution plan and never reference a task's own task_id.
11. Avoid circular dependencies.
12. Respect the input parameter names and output structure described in the tool catalog.
13. If a follow-up step requires calling a tool separately for each element of a list whose items are only known at runtime (and cannot be bound via a single scalar output_path), plan ONLY the discovery task(s) in this plan — do NOT invent list-index paths or pass a list into a scalar parameter.
14. If the user's request does not require any tool from AVAILABLE TOOLS (for example a greeting, casual conversation, or a request completely outside the available tools), return an empty "tasks": [] list — NEVER invent a fake tool name such as "none" or "noop".

Examples of valid tasks (using abstract placeholder tool names — always use only tools from AVAILABLE TOOLS):

1) Static inputs only:
{
  "task_id": "task_1",
  "tool_name": "fetch_record",
  "description": "Fetch metrics for item A.",
  "inputs": {"item": "A"}
}

2) Dynamic bindings only:
{
  "task_id": "task_3",
  "tool_name": "compare_values",
  "description": "Compare task_1 score against task_2 score.",
  "inputs": {
    "first": {"task_id": "task_1", "output_path": ["metrics", "score"]},
    "second": {"task_id": "task_2", "output_path": ["metrics", "score"]}
  }
}

3) Mix of static constants and dynamic bindings in the same "inputs" dictionary:
{
  "task_id": "task_4",
  "tool_name": "scale_value",
  "description": "Divide task_3 delta by the static constant 4.",
  "inputs": {
    "mode": "divide",
    "value": {"task_id": "task_3", "output_path": ["delta"]},
    "factor": 4
  }
}

Return a structured execution plan.
Do not execute any tool.
""".strip()


class PlannerOutputError(RuntimeError):
    """Raised when the LLM fails to produce a valid ExecutionPlan."""

    def __init__(self, message: str, raw_output: Any = None) -> None:
        super().__init__(message)
        self.raw_output = raw_output


class Planner:
    """Generates structured execution plans using an injected chat model."""

    def __init__(
        self,
        llm: BaseChatModel,
        structured_output_method: str = "json_schema",
    ) -> None:
        self._llm = llm
        self._structured_llm = self._llm.with_structured_output(
            schema=ExecutionPlan,
            method=structured_output_method,
            include_raw=True,
        )

    def _build_messages(
        self,
        request: str,
        available_tools: Mapping[str, str],
        *,
        previous_plan: ExecutionPlan | None = None,
        feedback: str | None = None,
        replan_context: str | None = None,
    ) -> list[SystemMessage | HumanMessage]:
        if not request.strip():
            raise ValueError("The user request cannot be empty.")

        if not available_tools:
            raise ValueError("At least one available tool is required.")

        for tool_name, description in available_tools.items():
            if not tool_name.strip() or not description.strip():
                raise ValueError("Tool names and descriptions cannot be empty.")

        tool_catalog = "\n".join(
            f"- {name.strip()}: {description.strip()}"
            for name, description in available_tools.items()
        )

        sections = [
            f"AVAILABLE TOOLS:\n{tool_catalog}",
            f"USER REQUEST:\n{request.strip()}",
        ]

        normalized_replan = replan_context.strip() if replan_context else ""
        if normalized_replan:
            sections.append(
                f"RUNTIME REPLANNING CONTEXT (PREVIOUS ROUNDS):\n{normalized_replan}\n\n"
                "Generate a fresh, self-contained follow-up ExecutionPlan covering ONLY "
                "the remaining or fallback steps needed to fulfill the user request.\n"
                "IMPORTANT RULES FOR REPLANNING:\n"
                "- Do NOT re-run calls that already succeeded.\n"
                "- Do NOT repeat failed tool calls with identical arguments.\n"
                "- Every InputBinding may ONLY reference task_ids defined inside THIS "
                "new plan. NEVER reference a task_id from a previous round; copy any "
                "already-retrieved literal value (number or string) directly into inputs."
            )

        normalized_feedback = feedback.strip() if feedback else ""
        if normalized_feedback:
            if previous_plan is not None:
                sections.append(
                    "PREVIOUS EXECUTION PLAN (FAILED COMPILATION):\n"
                    f"{previous_plan.model_dump_json(indent=2)}"
                )
            sections.append(
                f"COMPILATION ERRORS TO FIX:\n{normalized_feedback}\n\n"
                "Fix ONLY the listed compilation errors while preserving the valid "
                "parts of the plan, and return a corrected ExecutionPlan."
            )

        return [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content="\n\n".join(sections)),
        ]

    @staticmethod
    def _extract_plan(response: dict[str, Any]) -> ExecutionPlan:
        parsed = response.get("parsed")
        error = response.get("parsing_error")
        raw = response.get("raw")

        if error is not None or not isinstance(parsed, ExecutionPlan):
            raise PlannerOutputError(
                f"Failed to parse ExecutionPlan: {error}",
                raw_output=raw,
            ) from error

        return parsed

    def plan(
        self,
        request: str,
        available_tools: Mapping[str, str],
        *,
        previous_plan: ExecutionPlan | None = None,
        feedback: str | None = None,
        replan_context: str | None = None,
    ) -> ExecutionPlan:
        """Generate a validated ExecutionPlan synchronously."""
        messages = self._build_messages(
            request,
            available_tools,
            previous_plan=previous_plan,
            feedback=feedback,
            replan_context=replan_context,
        )
        response = self._structured_llm.invoke(messages)
        return self._extract_plan(response)

    async def aplan(
        self,
        request: str,
        available_tools: Mapping[str, str],
        *,
        previous_plan: ExecutionPlan | None = None,
        feedback: str | None = None,
        replan_context: str | None = None,
    ) -> ExecutionPlan:
        """Generate a validated ExecutionPlan asynchronously."""
        messages = self._build_messages(
            request,
            available_tools,
            previous_plan=previous_plan,
            feedback=feedback,
            replan_context=replan_context,
        )
        response = await self._structured_llm.ainvoke(messages)
        return self._extract_plan(response)