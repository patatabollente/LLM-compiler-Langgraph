from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from langgraph_llm_compiler.planner.schemas import NonEmptyStr

JoinerStatus = Literal["complete", "partial", "failed"]

_BASE_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    str_strip_whitespace=True,
)


class JoinerOutput(BaseModel):
    """Structured output generated directly by the LLM in Joiner v1."""

    model_config = _BASE_CONFIG

    answer: NonEmptyStr = Field(
        description="Final user-facing answer written in the user's original language."
    )


class JoinerResult(BaseModel):
    """Application-level result returned by the Joiner component."""

    model_config = _BASE_CONFIG

    answer: NonEmptyStr
    status: JoinerStatus
    missing_information: list[NonEmptyStr] = Field(default_factory=list)
