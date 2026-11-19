import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph_llm_compiler.executor.results import ExecutionResult
from langgraph_llm_compiler.planner.schemas import ExecutionPlan, TaskSpec

from .schemas import JoinerOutput, JoinerResult, JoinerStatus

SYSTEM_PROMPT = """
You are the Joiner component in an LLM Compiler architecture.

Your task is to synthesize a clear, accurate final answer for the user based on the executed task results.

Follow these rules strictly:

1. Language: Always respond in the exact same language as the ORIGINAL USER REQUEST, regardless of the language used in the plan objective or task descriptions.
2. Grounding: Use ONLY the outputs of tasks with status "success". Never invent, estimate, or fill in missing data from memory.
3. Transparency: If the execution status is "partial" or "failed", clearly explain what information was retrieved (if any) and what could not be obtained due to errors or skipped tasks. When all tasks failed, briefly explain what went wrong without fabricating an answer.
4. User-friendly: Write a natural response for the end user. Do not expose internal identifiers or architectural details such as task_1, task_2, input_bindings, output_path, or CompiledDAG.

Return only the structured answer.
""".strip()


class JoinerOutputError(RuntimeError):
    """Raised when the LLM fails to produce a valid JoinerOutput."""

    def __init__(self, message: str, raw_output: Any = None) -> None:
        super().__init__(message)
        self.raw_output = raw_output


def classify_execution_status(execution_result: ExecutionResult) -> JoinerStatus:
    """Deterministically classify execution completeness from task statuses."""
    total = len(execution_result.results)
    if total == 0:
        raise ValueError("ExecutionResult must contain at least one task result.")

    succeeded = len(execution_result.succeeded_task_ids)
    if succeeded == total:
        return "complete"
    if succeeded == 0:
        return "failed"
    return "partial"


def extract_missing_information(
    execution_result: ExecutionResult,
    plan: ExecutionPlan | None = None,
) -> list[str]:
    """Deterministically list missing information from failed/skipped tasks in topological order."""
    task_map: dict[str, TaskSpec] = (
        {task.task_id: task for task in plan.tasks} if plan is not None else {}
    )

    missing: list[str] = []
    for task_id, res in execution_result.results.items():
        if res.status == "success":
            continue

        task_spec = task_map.get(task_id)
        if task_spec is not None:
            label = task_spec.description.strip()
        else:
            label = res.tool_name.strip() or task_id.strip()

        if res.status == "error":
            err_msg = (res.error or "").strip()
            err_type = (res.error_type or "").strip()
            if err_msg and err_type and err_type not in err_msg:
                detail = f"error ({err_type}): {err_msg}"
            elif err_msg:
                detail = f"error: {err_msg}"
            elif err_type:
                detail = f"error: {err_type}"
            else:
                detail = "error: execution failed"
        else:
            reason = (res.skip_reason or "").strip()
            detail = (
                f"skipped: {reason}" if reason else "skipped: upstream dependency failed"
            )

        missing.append(f"{label} — {detail}")

    return missing


def _format_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return repr(value)


class Joiner:
    """Synthesizes a final user-facing answer from an ExecutionResult and optional ExecutionPlan."""

    def __init__(
        self,
        llm: BaseChatModel,
        structured_output_method: str = "json_schema",
    ) -> None:
        self._llm = llm
        self._structured_llm = self._llm.with_structured_output(
            schema=JoinerOutput,
            method=structured_output_method,
            include_raw=True,
        )

    @staticmethod
    def _build_messages(
        request: str,
        execution_result: ExecutionResult,
        plan: ExecutionPlan | None,
        status: JoinerStatus,
        missing_information: list[str],
    ) -> list[SystemMessage | HumanMessage]:
        normalized_request = request.strip()
        if not normalized_request:
            raise ValueError("The original user request cannot be empty.")

        task_map: dict[str, TaskSpec] = (
            {task.task_id: task for task in plan.tasks} if plan is not None else {}
        )

        task_lines: list[str] = []
        for task_id, res in execution_result.results.items():
            task_spec = task_map.get(task_id)
            header = f"- {task_id} (tool={res.tool_name}, status={res.status})"
            details: list[str] = []

            if task_spec is not None:
                details.append(f"  description: {task_spec.description}")
                if task_spec.static_inputs:
                    details.append(
                        f"  arguments: {_format_json(task_spec.static_inputs)}"
                    )

            if res.status == "success":
                details.append(f"  output: {_format_json(res.output)}")
            elif res.status == "error":
                stage = res.error_stage or "unknown"
                err_type = res.error_type or "Error"
                err_msg = res.error or "Unknown error"
                details.append(f"  error [{stage}/{err_type}]: {err_msg}")
            else:
                reason = res.skip_reason or "Upstream dependency did not succeed."
                details.append(f"  skip_reason: {reason}")

            task_lines.append("\n".join([header, *details]))

        sections = [
            f"ORIGINAL USER REQUEST:\n{normalized_request}",
            f"EXECUTION OBJECTIVE:\n{execution_result.objective}",
            f"EXECUTION STATUS:\n{status}",
            "TASK RESULTS:\n" + "\n".join(task_lines),
        ]

        if missing_information:
            missing_block = "\n".join(f"- {item}" for item in missing_information)
            sections.append(f"MISSING INFORMATION:\n{missing_block}")

        return [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content="\n\n".join(sections)),
        ]

    @staticmethod
    def _extract_output(response: dict[str, Any]) -> JoinerOutput:
        parsed = response.get("parsed")
        error = response.get("parsing_error")
        raw = response.get("raw")

        if error is not None or not isinstance(parsed, JoinerOutput):
            raise JoinerOutputError(
                f"Failed to parse JoinerOutput: {error}",
                raw_output=raw,
            ) from error

        return parsed

    async def join(
        self,
        request: str,
        execution_result: ExecutionResult,
        plan: ExecutionPlan | None = None,
    ) -> JoinerResult:
        """Synthesize the final JoinerResult asynchronously."""
        if not request.strip():
            raise ValueError("The original user request cannot be empty.")

        status = classify_execution_status(execution_result)
        missing_information = extract_missing_information(execution_result, plan=plan)

        messages = self._build_messages(
            request=request,
            execution_result=execution_result,
            plan=plan,
            status=status,
            missing_information=missing_information,
        )
        response = await self._structured_llm.ainvoke(messages)
        output = self._extract_output(response)

        return JoinerResult(
            answer=output.answer,
            status=status,
            missing_information=missing_information,
        )
