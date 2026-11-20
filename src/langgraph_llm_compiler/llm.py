from langchain_ollama import ChatOllama


def create_ollama_chat_model(
    model: str = "qwen3:8b",
    temperature: float = 0.0,
    reasoning: bool = True,
    base_url: str | None = None,
    num_predict: int | None = None,
) -> ChatOllama:
    """Create a configured ChatOllama instance."""
    kwargs: dict[str, object] = {
        "model": model,
        "temperature": temperature,
        "reasoning": reasoning,
        "base_url": base_url,
    }
    if num_predict is not None:
        kwargs["num_predict"] = num_predict
    return ChatOllama(**kwargs)  # type: ignore[arg-type]