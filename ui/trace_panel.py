"""Trace formatting helpers for the Chainlit UI."""
from __future__ import annotations

from typing import Iterable


def format_trace(trace_log: Iterable[str] | None) -> str:
    entries = [str(item) for item in (trace_log or []) if item]
    if not entries:
        return "No agent trace available."
    return "\n".join(f"{idx}. {entry}" for idx, entry in enumerate(entries, start=1))
