"""Thin wrapper around the Tinker LLM API (OpenAI-compatible)."""
from __future__ import annotations

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

TINKER_API_KEY = os.getenv("TINKER_API_KEY", "")
TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3.1"

def get_client() -> OpenAI:
    return OpenAI(api_key=TINKER_API_KEY, base_url=TINKER_BASE_URL)

class LLMResponse:
    """Wrapper around the OpenAI response, like requests.Response."""

    def __init__(self, raw):
        self._raw = raw
        self.text = raw.choices[0].message.content or ""
        self.role = raw.choices[0].message.role
        self.finish_reason = raw.choices[0].finish_reason
        self.usage = raw.usage
        self.model = raw.model

    def json(self) -> dict:
        """Full response as a dict."""
        return self._raw.model_dump()

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"LLMResponse(text={self.text!r:.80}, finish={self.finish_reason})"


def chat(
    messages: list[dict[str, str]],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.6,
    max_tokens: int = 1024,
    stop: list[str] | None = None,
    top_p: float = 1.0,
) -> LLMResponse:
    """Send a chat completion request. Returns LLMResponse with .text, .json(), .usage, etc."""
    client = get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop,
        top_p=top_p,
    )
    return LLMResponse(resp)
