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

def chat(
    messages: list[dict[str, str]],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.6,
    max_tokens: int = 1024,
    stop: list[str] | None = None,
    top_p: float = 1.0,
) -> str:
    """Send a chat completion request and return the response text."""
    client = get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop,
        top_p=top_p,
    )
    return resp.choices[0].message.content or ""
