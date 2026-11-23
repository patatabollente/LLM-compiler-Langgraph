"""High-level Parallel ReAct runner for LangGraph with rich per-wave telemetry.

API and telemetry schema mirror `parallel_react_llamaindex.ParallelReactLlamaIndexRunner`
so LangGraph ReAct and LlamaIndex ReAct can be benchmarked side by side on the
exact same tools, prompts, and result models.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from llm_compiler.compiler.registry import ToolRegistry

from .builder import DEFAULT_REACT_SYSTEM_PROMPT, build_react_graph
from .config import InferenceParams, OpenAICompatibleConfig, build_react_llm
from .instrumentation import ReActRunInstrumentation
from .telemetry import ParallelRunResult
from .tools import ToolLike, normalize_to_registry


class ParallelReactLangGraphRunner:
    """Run an isolated Parallel ReAct task on LangGraph with per-turn and per-wave telemetry.

    Accepts either our `ToolRegistry` (preserving both `input_schema` and `output_schema`
    runtime validation) or any sequence of `ToolLike` items (`ToolSpec`, `ToolBinding`,
    plain Python callables, or tool dicts).
    """

    def __init__(
        self,
        *,
        tools: ToolRegistry | Sequence[ToolLike] = (),
        system_prompt: str | None = DEFAULT_REACT_SYSTEM_PROMPT,
        model_config: OpenAICompatibleConfig | Mapping[str, Any] | None = None,
        inference_params: InferenceParams | Mapping[str, Any] | None = None,
        llm: BaseChatModel | None = None,
        max_turns: int = 10,
        timeout_seconds: float | None = None,
        task_timeout_s: float | None = None,
        allow_parallel_tool_calls: bool = True,
        stream_llm: bool = True,
    ) -> None:
        if llm is None and model_config is None:
            raise ValueError(
                "Provide either llm= (a LangChain BaseChatModel) or model_config=."
            )
        if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
            raise ValueError("max_turns must be an integer >= 1.")
        if timeout_seconds is not None and (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be None or a positive finite number.")
        if task_timeout_s is not None and (
            isinstance(task_timeout_s, bool)
            or not isinstance(task_timeout_s, (int, float))
            or not math.isfinite(task_timeout_s)
            or task_timeout_s <= 0
        ):
            raise ValueError("task_timeout_s must be None or a positive finite number.")
        if not isinstance(allow_parallel_tool_calls, bool):
            raise TypeError("allow_parallel_tool_calls must be a bool.")
        if not isinstance(stream_llm, bool):
            raise TypeError("stream_llm must be a bool.")
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be a string or None.")

        self.system_prompt = system_prompt
        self.model_config = self._normalize_model_config(model_config)
        self.inference_params = self._normalize_inference_params(inference_params)
        self.llm = llm
        self.max_turns = max_turns
        self.timeout_seconds = (
            float(timeout_seconds) if timeout_seconds is not None else None
        )
        self.task_timeout_s = float(task_timeout_s) if task_timeout_s is not None else None
        self.allow_parallel_tool_calls = allow_parallel_tool_calls
        self.stream_llm = stream_llm
        self._registry = normalize_to_registry(tools)

    async def run(self, prompt: str) -> ParallelRunResult:
        """Execute one task and return a `ParallelRunResult` with turn/wave/tool traces."""
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string.")

        instrumentation = ReActRunInstrumentation(started_at=time.perf_counter())
        answer: str | None = None
        workflow_status = None

        try:
            if self.llm is not None:
                model = self.llm
            else:
                assert self.model_config is not None
                model = build_react_llm(self.model_config, self.inference_params)

            graph = build_react_graph(
                registry=self._registry,
                llm=model,
                system_prompt=self.system_prompt,
                max_iterations=self.max_turns,
                task_timeout_s=self.task_timeout_s,
                allow_parallel_tool_calls=self.allow_parallel_tool_calls,
                stream_llm=self.stream_llm,
                instrumentation=instrumentation,
            )

            coro = graph.ainvoke(
                {"request": prompt},
                config={"recursion_limit": max(25, self.max_turns * 3 + 5)},
            )
            if self.timeout_seconds is not None:
                final_state = await asyncio.wait_for(coro, timeout=self.timeout_seconds)
            else:
                final_state = await coro

            answer = final_state.get("final_answer")
            workflow_status = final_state.get("status")
            err = final_state.get("error")
            if err is not None:
                instrumentation.error = f"{err.error_type}: {err.message}"
                if err.error_type == "MaxIterationsError":
                    instrumentation.status = "max_turns_exceeded"
                    instrumentation.termination_reason = "request_limit"
                else:
                    instrumentation.status = "error"
                    instrumentation.termination_reason = err.error_type
            else:
                instrumentation.status = "completed"
                instrumentation.termination_reason = "completed"
                if answer is None:
                    answer = ""
        except TimeoutError as exc:
            instrumentation.status = "error"
            instrumentation.termination_reason = "timeout"
            instrumentation.timed_out = True
            instrumentation.error = (
                f"TimeoutError: Run timed out after {self.timeout_seconds:.2f}s."
                if self.timeout_seconds is not None
                else f"TimeoutError: {exc}"
            )
            workflow_status = "failed"
        except Exception as exc:  # noqa: BLE001
            instrumentation.status = "error"
            instrumentation.termination_reason = type(exc).__name__
            instrumentation.error = f"{type(exc).__name__}: {exc}"
            workflow_status = "failed"

        dumped_params = self.inference_params.model_dump(exclude_none=True)
        return instrumentation.build_result(
            answer=answer,
            workflow_status=workflow_status,
            applied_settings=(dumped_params if self.llm is None else {}),
            skipped_settings=([] if self.llm is None else list(dumped_params)),
        )

    def run_sync(self, prompt: str) -> ParallelRunResult:
        """Synchronous convenience wrapper; use `await run()` inside async code."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(prompt))
        raise RuntimeError(
            "run_sync() cannot be called from a running event loop; await run() instead."
        )

    @staticmethod
    def _normalize_model_config(
        model_config: OpenAICompatibleConfig | Mapping[str, Any] | None,
    ) -> OpenAICompatibleConfig | None:
        if model_config is None:
            return None
        if isinstance(model_config, OpenAICompatibleConfig):
            return model_config
        return OpenAICompatibleConfig.model_validate(model_config)

    @staticmethod
    def _normalize_inference_params(
        inference_params: InferenceParams | Mapping[str, Any] | None,
    ) -> InferenceParams:
        if inference_params is None:
            return InferenceParams()
        if isinstance(inference_params, InferenceParams):
            return inference_params
        return InferenceParams.model_validate(inference_params)


ParallelReactRunner = ParallelReactLangGraphRunner
