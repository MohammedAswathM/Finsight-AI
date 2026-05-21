"""Shared LLM factory + tiny utilities used by every agent/node."""
from __future__ import annotations

import time
from functools import lru_cache
from typing import Any, Dict, List

from langchain_groq import ChatGroq

from config import GROQ_MODEL, get_groq_keys, require_groq


@lru_cache(maxsize=16)
def _get_llm_cached(temperature: float, api_key: str) -> ChatGroq:
    return ChatGroq(
        model=GROQ_MODEL,
        groq_api_key=api_key,
        temperature=temperature,
        timeout=60,
        max_retries=1,
    )


def get_llm(temperature: float = 0.0, api_key: str | None = None) -> ChatGroq:
    """Single LLM factory. All nodes must use this — no direct ChatGroq() elsewhere."""
    return _get_llm_cached(temperature, api_key or require_groq())


def is_rate_limit_error(exc: Exception) -> bool:
    """Best-effort detector for Groq quota/rate-limit failures."""
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "rate limit",
            "ratelimit",
            "rate_limit",
            "429",
            "quota",
            "too many requests",
        )
    )


def invoke_prompt_with_fallback(prompt: Any, payload: Dict[str, Any], temperature: float = 0.0) -> Any:
    """Invoke a prompt through Groq, rotating keys only on quota/rate-limit errors."""
    keys = get_groq_keys()
    if not keys:
        require_groq()

    last_exc: Exception | None = None
    for index, key in enumerate(keys, start=1):
        llm = get_llm(temperature=temperature, api_key=key)
        try:
            return (prompt | llm).invoke(payload)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if not is_rate_limit_error(exc) or index == len(keys):
                raise
            time.sleep(1.0)

    if last_exc:
        raise last_exc
    raise RuntimeError("No Groq API keys available.")


def invoke_text_with_fallback(prompt: str, temperature: float = 0.0) -> Any:
    """Invoke a raw string prompt through Groq with the same key rotation policy."""
    keys = get_groq_keys()
    if not keys:
        require_groq()

    last_exc: Exception | None = None
    for index, key in enumerate(keys, start=1):
        try:
            return get_llm(temperature=temperature, api_key=key).invoke(prompt)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if not is_rate_limit_error(exc) or index == len(keys):
                raise
            time.sleep(1.0)

    if last_exc:
        raise last_exc
    raise RuntimeError("No Groq API keys available.")


def append_trace(message: str) -> List[str]:
    """Return a one-element list so LangGraph's `operator.add` reducer appends it."""
    return [message]


def strip_code_fence(content: str) -> str:
    """LLMs often wrap JSON in ```json ... ``` — strip it safely."""
    content = content.strip()
    if content.startswith("```"):
        parts = content.split("```")
        if len(parts) >= 2:
            content = parts[1]
            if content.lstrip().lower().startswith("json"):
                content = content.split("\n", 1)[1] if "\n" in content else content[4:]
    return content.strip()


def safe_get(state: Dict[str, Any], key: str, default: str = "NOT AVAILABLE") -> str:
    val = state.get(key)
    if val is None or val == "":
        return default
    return str(val)
