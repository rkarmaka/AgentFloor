"""OpenAI-compatible adapter.

Thin subclass of `OpenAIProvider` that takes a `base_url` pointing at
any OpenAI-compatible server: vLLM, Ollama, Together, Groq, Fireworks,
DeepSeek, Mistral's API, etc. The translation logic is inherited
unchanged — the only difference is the HTTP endpoint and the
`name` identifier we stamp on results.

Usage:

    from harness.providers.openai_compatible import OpenAICompatibleProvider

    # vLLM on localhost
    p = OpenAICompatibleProvider(
        backend="vllm",
        base_url="http://localhost:8000/v1",
        api_key="dummy",   # vLLM ignores but SDK requires a value
    )

    # Ollama on localhost
    p = OpenAICompatibleProvider(
        backend="ollama",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )

    # Together
    p = OpenAICompatibleProvider(
        backend="together",
        base_url="https://api.together.xyz/v1",
        api_key=os.environ["TOGETHER_API_KEY"],
    )

The `backend` string is purely a label used in `self.name` so result
files are bucketed correctly (`results/openai_compatible_vllm/...`).

WARNING: some OpenAI-compatible servers implement tool calling only
partially or via different templates. If a model's native tool-calling
support is broken on a given backend, the adapter will honestly report
zero tool calls — this will surface in the benchmark as an apparent
capability gap. That is intended (D3: native-only, no content parsing).
Verify tool calling works on each backend+model combo before running
a full sweep.
"""

from __future__ import annotations

from .openai import OpenAIProvider


class OpenAICompatibleProvider(OpenAIProvider):
    """An OpenAIProvider pointed at a non-OpenAI base_url.

    Inherits all translation and retry logic from OpenAIProvider.
    The only differences are:
    - `name` is `openai_compatible:{backend}` so result files are
      bucketed per backend.
    - `__init__` requires `base_url` and `backend`; `api_key` is
      optional but most SDKs insist on at least a dummy value.
    """

    def __init__(
        self,
        *,
        backend: str,
        base_url: str,
        api_key: str | None = None,
    ):
        if not backend:
            raise ValueError("`backend` is required (e.g. 'vllm', 'ollama', 'together')")
        if not base_url:
            raise ValueError("`base_url` is required for OpenAICompatibleProvider")
        # OpenAI's SDK requires *some* api_key even for servers that ignore it.
        super().__init__(api_key=api_key or "dummy", base_url=base_url)
        self.name = f"openai_compatible:{backend}"
        self.backend = backend
        self.base_url = base_url
