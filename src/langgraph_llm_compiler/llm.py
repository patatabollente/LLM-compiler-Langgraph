from langchain_ollama import ChatOllama


def create_ollama_chat_model(
    model: str = "qwen3:8b",
    temperature: float = 0.0,
    reasoning: bool = True,
    base_url: str | None = None,
) -> ChatOllama:
    """Create a configured ChatOllama instance."""
    return ChatOllama(
        model=model,
        temperature=temperature,
        reasoning=reasoning,
        base_url=base_url,
    )