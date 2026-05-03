"""Pick which provider (DeepSeek by default, Anthropic on opt-in) backs the agents.

DeepSeek exposes an Anthropic-compatible endpoint, so the same `anthropic.Anthropic`
client works for both — only the api_key, base_url, and model name differ.
"""
from __future__ import annotations

import os

# https://api-docs.deepseek.com/guides/anthropic_api
_DEEPSEEK_BASE_URL = "https://api.deepseek.com/anthropic"
_DEEPSEEK_MODEL = "deepseek-chat"
_ANTHROPIC_MODEL = "claude-sonnet-4-6"


def _use_anthropic() -> bool:
    return os.getenv("LLM_PROVIDER", "").strip().lower() == "anthropic"


def client_kwargs() -> dict:
    """kwargs for `anthropic.Anthropic(...)` — picks DeepSeek by default."""
    if _use_anthropic():
        return {"api_key": os.getenv("ANTHROPIC_API_KEY") or ""}
    return {
        "api_key": os.getenv("DEEPSEEK_API_KEY") or "",
        "base_url": os.getenv("DEEPSEEK_BASE_URL", _DEEPSEEK_BASE_URL),
    }


def default_model() -> str:
    return _ANTHROPIC_MODEL if _use_anthropic() else _DEEPSEEK_MODEL
