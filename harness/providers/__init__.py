"""Provider adapters for the benchmark runner.

Each adapter translates between the canonical Message / ToolCall /
AssistantTurn types defined in `base.py` and a specific LLM backend
(Anthropic, OpenAI, or any OpenAI-compatible server).

Adapters MUST NOT auto-repair malformed tool calls or invent tool_calls
from text content. See `doc/design.md` section "NEW Harness — Runner
and Provider Adapters" (D2, D3) for the rationale.
"""

from .base import (
    AssistantTurn,
    Message,
    ModelConfig,
    Provider,
    ToolCall,
    TokenUsage,
    assistant_message,
    tool_result_message,
    user_message,
)

# Concrete adapters are imported lazily via `get_provider` to avoid
# pulling in the anthropic / openai SDKs when only the types are needed
# (e.g. the runner smoke test using a FakeProvider).


def get_provider(kind: str, **kwargs):
    """Factory for concrete provider adapters.

    `kind`:
      - "anthropic"           → AnthropicProvider(**kwargs)
      - "openai"              → OpenAIProvider(**kwargs)
      - "gemini"              → GeminiProvider(**kwargs)
      - "openai_compatible"   → OpenAICompatibleProvider(**kwargs)
    """
    if kind == "anthropic":
        from .anthropic import AnthropicProvider
        return AnthropicProvider(**kwargs)
    if kind == "openai":
        from .openai import OpenAIProvider
        return OpenAIProvider(**kwargs)
    if kind == "gemini":
        from .gemini import GeminiProvider
        return GeminiProvider(**kwargs)
    if kind == "openai_compatible":
        from .openai_compatible import OpenAICompatibleProvider
        return OpenAICompatibleProvider(**kwargs)
    raise ValueError(f"Unknown provider kind: {kind}")


__all__ = [
    "AssistantTurn",
    "Message",
    "ModelConfig",
    "Provider",
    "ToolCall",
    "TokenUsage",
    "assistant_message",
    "tool_result_message",
    "user_message",
    "get_provider",
]
