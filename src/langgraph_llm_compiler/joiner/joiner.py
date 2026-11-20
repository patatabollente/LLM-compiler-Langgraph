import json
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph_llm_compiler.executor.results import ExecutionResult
from langgraph_llm_compiler.planner.schemas import ExecutionPlan, TaskSpec

from .schemas import (
    ExecutionRound,
    JoinerDecisionOutput,
    JoinerOutput,
    JoinerResult,
    JoinerStatus,
)

SYSTEM_PROMPT = """
You are the Joiner component in an LLM Compiler architecture.

Your task is to synthesize a clear, accurate final answer for the user based on the executed task results.

Follow these rules strictly:

1. Language: Always respond in the exact same language as the ORIGINAL USER REQUEST, regardless of the language used in the plan objective or task descriptions.
2. Grounding: Use ONLY the outputs of tasks with status "success" (across all execution rounds, if multiple rounds were run). Never invent, estimate, or fill in missing data from memory.
3. Transparency: If the execution status is "partial" or "failed", clearly explain what information was retrieved (if any) and what could not be obtained due to errors or skipped tasks. When all tasks failed, briefly explain what went wrong without fabricating an answer.
4. User-friendly: Write a natural response for the end user. Do not expose internal identifiers or architectural details such as task_1, task_2, input_bindings, output_path, or CompiledDAG.

Return only the structured answer.
""".strip()

REPLAN_SYSTEM_PROMPT = """
You are the Joiner component in an LLM Compiler architecture with runtime replanning enabled.

Inspect the executed task results against the ORIGINAL USER REQUEST and decide between two actions:
- "finish": deliver the final answer to the user now (provide "answer", set "replan_reason" to null).
- "replan": request a targeted follow-up plan from the Planner (provide "replan_reason", set "answer" to null).

Follow these decision rules strictly:

1. Choose "replan" ONLY when one of these two conditions holds:
   (a) A runtime failure occurred AND another tool in AVAILABLE TOOLS (for example a backup/fallback tool) or corrected input arguments can realistically recover the missing result.
   (b) Earlier tasks succeeded in discovering intermediate data (such as an entity name, identifier, or target value) that is required to run follow-up tool calls still needed to fulfill the ORIGINAL USER REQUEST.
2. Choose "finish" immediately when:
   (a) All parts of the ORIGINAL USER REQUEST are already satisfied by the successful task outputs across the executed rounds, OR
   (b) A runtime error is unrecoverable (for example, the requested entity does not exist and no alternative tool in AVAILABLE TOOLS can supply it). Never waste a replan on unrecoverable errors.
3. When choosing "replan", write a concise, concrete "replan_reason" stating:
   - what is still needed to fulfill the request,
   - which fallback tool or follow-up tool call(s) to use (and which failing call NOT to repeat),
   - the exact literal values already retrieved in prior rounds that should be reused directly (NEVER refer to old task IDs like task_1 or task_2, because previous-round task IDs do not exist in the new plan).
4. When choosing "finish", follow all synthesis rules:
   - Respond in the exact same language as the ORIGINAL USER REQUEST.
   - Use ONLY outputs of tasks with status "success" across all rounds; never invent or estimate missing values.
   - If the status is "partial" or "failed", transparently state what was obtained and what could not be retrieved.
   - Write a natural, user-friendly response without exposing internal identifiers (task_1, output_path, CompiledDAG, etc.).

Return only the structured decision object.
""".strip()


class JoinerOutputError(RuntimeError):
    """Raised when the LLM fails to produce a valid JoinerOutput or JoinerDecisionOutput."""

    def __init__(self, message: str, raw_output: Any = None) -> None:
        super().__init__(message)
        self.raw_output = raw_output


def classify_execution_status(
    execution_result: ExecutionResult,
    prior_rounds: Sequence[ExecutionRound] = (),
) -> JoinerStatus:
    """Deterministically classify execution completeness from task statuses."""
    total = len(execution_result.results)
    if total == 0:
        raise ValueError("ExecutionResult must contain at least one task result.")

    succeeded = len(execution_result.succeeded_task_ids)
    if succeeded == total:
        return "complete"

    any_prior_succeeded = any(
        len(round_item.execution_result.succeeded_task_ids) > 0
        for round_item in prior_rounds
    )
    if succeeded > 0 or any_prior_succeeded:
        return "partial"
    return "failed"


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


def _format_task_results(
    execution_result: ExecutionResult,
    plan: ExecutionPlan | None,
) -> str:
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
                details.append(f"  arguments: {_format_json(task_spec.static_inputs)}")

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

    return "\n".join(task_lines)


def format_replan_context(
    prior_rounds: Sequence[ExecutionRound],
    replan_reason: str,
) -> str:
    """Format completed rounds and Joiner guidance into a Planner replan context."""
    normalized_reason = replan_reason.strip()
    if not normalized_reason:
        raise ValueError("replan_reason cannot be empty.")

    sections = [f"JOINER REPLAN GUIDANCE:\n{normalized_reason}"]

    for round_item in prior_rounds:
        task_map: dict[str, TaskSpec] = (
            {task.task_id: task for task in round_item.plan.tasks}
            if round_item.plan is not None
            else {}
        )
        succeeded_lines: list[str] = []
        failed_lines: list[str] = []

        for task_id, res in round_item.execution_result.results.items():
            task_spec = task_map.get(task_id)
            desc = f" — {task_spec.description}" if task_spec is not None else ""
            args_str = (
                f", args={_format_json(task_spec.static_inputs)}"
                if task_spec is not None and task_spec.static_inputs
                else ""
            )

            if res.status == "success":
                succeeded_lines.append(
                    f"- {res.tool_name}({args_str.lstrip(', ')}){desc}\n"
                    f"  output: {_format_json(res.output)}"
                )
            elif res.status == "error":
                stage = res.error_stage or "unknown"
                err_type = res.error_type or "Error"
                err_msg = res.error or "Unknown error"
                failed_lines.append(
                    f"- {res.tool_name}({args_str.lstrip(', ')}){desc}\n"
                    f"  error [{stage}/{err_type}]: {err_msg}"
                )
            else:
                reason = res.skip_reason or "Upstream dependency failed."
                failed_lines.append(
                    f"- {res.tool_name}({args_str.lstrip(', ')}){desc}\n"
                    f"  skipped: {reason}"
                )

        round_parts = [f"ROUND {round_item.round_index}:"]
        if succeeded_lines:
            round_parts.append(
                "Already-retrieved results (do NOT re-run; copy any needed literal "
                "values directly into static inputs — NEVER create an InputBinding "
                "pointing to a previous-round task):\n"
                + "\n".join(succeeded_lines)
            )
        if failed_lines:
            round_parts.append(
                "Failed or skipped tasks (do NOT repeat the exact same failing call):\n"
                + "\n".join(failed_lines)
            )
        sections.append("\n".join(round_parts))

    return "\n\n".join(sections)


class Joiner:
    """Synthesizes a final user-facing answer or requests a runtime replan."""

    def __init__(
        self,
        llm: BaseChatModel,
        structured_output_method: str = "json_schema",
    ) -> None:
        self._llm = llm
        self._structured_output_method = structured_output_method
        self._structured_llm = self._llm.with_structured_output(
            schema=JoinerOutput,
            method=structured_output_method,
            include_raw=True,
        )
        self._decision_llm: Any | None = None

    def _get_decision_llm(self) -> Any:
        if self._decision_llm is None:
            self._decision_llm = self._llm.with_structured_output(
                schema=JoinerDecisionOutput,
                method=self._structured_output_method,
                include_raw=True,
            )
        return self._decision_llm

    @staticmethod
    def _build_messages(
        request: str,
        execution_result: ExecutionResult,
        plan: ExecutionPlan | None,
        status: JoinerStatus,
        missing_information: list[str],
        *,
        prior_rounds: Sequence[ExecutionRound] = (),
        tool_catalog: Mapping[str, str] | None = None,
        allow_replan: bool = False,
    ) -> list[SystemMessage | HumanMessage]:
        normalized_request = request.strip()
        if not normalized_request:
            raise ValueError("The original user request cannot be empty.")

        sections = [f"ORIGINAL USER REQUEST:\n{normalized_request}"]

        if allow_replan and tool_catalog:
            catalog_lines = "\n".join(
                f"- {name.strip()}: {desc.strip()}"
                for name, desc in tool_catalog.items()
            )
            sections.append(f"AVAILABLE TOOLS:\n{catalog_lines}")

        if prior_rounds:
            prior_blocks: list[str] = []
            for round_item in prior_rounds:
                block = (
                    f"--- Round {round_item.round_index} "
                    f"(objective: {round_item.execution_result.objective}) ---\n"
                    + _format_task_results(round_item.execution_result, round_item.plan)
                )
                if round_item.replan_reason:
                    block += f"\nReplan triggered because: {round_item.replan_reason}"
                prior_blocks.append(block)
            sections.append("PREVIOUS EXECUTION ROUNDS:\n" + "\n\n".join(prior_blocks))

        round_label = (
            f"CURRENT EXECUTION ROUND (Round {len(prior_rounds) + 1})"
            if prior_rounds
            else "TASK RESULTS"
        )
        sections.extend(
            [
                f"EXECUTION OBJECTIVE:\n{execution_result.objective}",
                f"EXECUTION STATUS:\n{status}",
                f"{round_label}:\n{_format_task_results(execution_result, plan)}",
            ]
        )

        if missing_information:
            missing_block = "\n".join(f"- {item}" for item in missing_information)
            sections.append(f"MISSING INFORMATION:\n{missing_block}")

        system_prompt = REPLAN_SYSTEM_PROMPT if allow_replan else SYSTEM_PROMPT
        return [
            SystemMessage(content=system_prompt),
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

    @staticmethod
    def _extract_decision(response: dict[str, Any]) -> JoinerDecisionOutput:
        parsed = response.get("parsed")
        error = response.get("parsing_error")
        raw = response.get("raw")

        if error is not None or not isinstance(parsed, JoinerDecisionOutput):
            raise JoinerOutputError(
                f"Failed to parse JoinerDecisionOutput: {error}",
                raw_output=raw,
            ) from error

        return parsed

    async def join(
        self,
        request: str,
        execution_result: ExecutionResult,
        plan: ExecutionPlan | None = None,
        *,
        prior_rounds: Sequence[ExecutionRound] = (),
        tool_catalog: Mapping[str, str] | None = None,
        allow_replan: bool = False,
    ) -> JoinerResult:
        """Synthesize a final JoinerResult or request a runtime replan asynchronously."""
        if not request.strip():
            raise ValueError("The original user request cannot be empty.")

        status = classify_execution_status(execution_result, prior_rounds=prior_rounds)
        missing_information = extract_missing_information(execution_result, plan=plan)

        messages = self._build_messages(
            request=request,
            execution_result=execution_result,
            plan=plan,
            status=status,
            missing_information=missing_information,
            prior_rounds=prior_rounds,
            tool_catalog=tool_catalog,
            allow_replan=allow_replan,
        )

        if not allow_replan:
            response = await self._structured_llm.ainvoke(messages)
            output = self._extract_output(response)
            return JoinerResult(
                action="finish",
                answer=output.answer,
                replan_reason=None,
                status=status,
                missing_information=missing_information,
            )

        decision_llm = self._get_decision_llm()
        response = await decision_llm.ainvoke(messages)
        decision = self._extract_decision(response)

        if decision.action == "replan":
            return JoinerResult(
                action="replan",
                answer=None,
                replan_reason=decision.replan_reason,
                status=status,
                missing_information=missing_information,
            )

        return JoinerResult(
            action="finish",
            answer=decision.answer,
            replan_reason=None,
            status=status,
            missing_information=missing_information,
        )
