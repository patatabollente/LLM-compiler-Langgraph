"""Configuration models for OpenAI-compatible and Ollama endpoints."""

from __future__ import annotations

import os
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field

from llm_compiler.llm import create_ollama_chat_model


class OpenAICompatibleConfig(BaseModel):
    """Connection details for an Ollama or OpenAI-compatible inference server.

    Aligned with `parallel_react_llamaindex.config.OpenAICompatibleConfig` so
    LangGraph and LlamaIndex ReAct runners share the same configuration schema.
    Instantiating this model never contacts the endpoint.
    """

    model_config = ConfigDict(extra="forbid")

    model: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Model identifier understood by the endpoint. Required only when the "
            "runner constructs a chat model."
        ),
    )
    base_url: str = Field(
        default="http://localhost:11434",
        min_length=1,
        description="Server base URL (e.g. http://localhost:11434 for Ollama or http://localhost:8000/v1).",
    )
    api_key: str | None = Field(
        default=None,
        description="Explicit API key; takes precedence over api_key_env.",
    )
    api_key_env: str | None = Field(
        default=None,
        description=(
            "Environment variable read when api_key is omitted. Defaults to OPENAI_API_KEY "
            "for OpenAI and MODEL_API_KEY for OpenAI-compatible endpoints."
        ),
    )
    provider: Literal["ollama", "openai-compatible", "openai"] = Field(
        default="ollama",
        description=(
            "'ollama' uses local ChatOllama; 'openai-compatible' and 'openai' use "
            "langchain-openai ChatOpenAI (optional dependency)."
        ),
    )
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Optional extra HTTP headers forwarded to the provider client.",
    )


class InferenceParams(BaseModel):
    """Sampling settings. `None` means omit the field from the request."""

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, gt=0)
    seed: int | None = Field(default=None, ge=0)
    reasoning: bool = Field(
        default=False,
        description="Enable thinking/reasoning mode on Ollama models that support it.",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Escape hatch for extra provider-specific request parameters.",
    )


def build_react_llm(
    config: OpenAICompatibleConfig,
    params: InferenceParams,
) -> BaseChatModel:
    """Build a LangChain chat model from `OpenAICompatibleConfig` and `InferenceParams`."""
    if not config.model:
        raise ValueError("OpenAICompatibleConfig.model is required to build an LLM.")

    if config.provider == "ollama":
        base_url = config.base_url.rstrip("/").removesuffix("/v1")
        return create_ollama_chat_model(
            model=config.model,
            temperature=params.temperature if params.temperature is not None else 0.0,
            reasoning=params.reasoning,
            base_url=base_url,
            num_predict=params.max_tokens,
        )

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "provider='openai' and provider='openai-compatible' require 'langchain-openai'. "
            "Install it (`uv add langchain-openai`) or use provider='ollama' / inject llm=."
        ) from exc

    api_key_env = config.api_key_env or (
        "OPENAI_API_KEY" if config.provider == "openai" else "MODEL_API_KEY"
    )
    api_key = config.api_key or os.environ.get(api_key_env) or "not-needed"
    model_kwargs: dict[str, Any] = {
        key: value
        for key, value in {"top_p": params.top_p, "seed": params.seed, **params.extra}.items()
        if value is not None
    }
    return ChatOpenAI(
        model=config.model,
        base_url=config.base_url,
        api_key=api_key,
        temperature=params.temperature if params.temperature is not None else 0.0,
        max_tokens=params.max_tokens,
        model_kwargs=model_kwargs,
        default_headers=config.headers or None,
        stream_usage=True,
    )
