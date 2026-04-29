"""Chainlit UI for FinSight AI.

Run:
    chainlit run ui/app.py
"""
from __future__ import annotations

import asyncio
import base64
import time
from pathlib import Path
from typing import Any

import chainlit as cl

from orchestrator.graph import run_graph
from ui.trace_panel import format_trace


async def _read_uploaded_image(message: cl.Message) -> str | None:
    """Read the first uploaded image as base64 for the graph state."""
    for element in message.elements or []:
        path = getattr(element, "path", None)
        mime = getattr(element, "mime", "") or ""
        if path and (mime.startswith("image/") or Path(path).suffix.lower() in {".png", ".jpg", ".jpeg"}):
            return base64.b64encode(Path(path).read_bytes()).decode("utf-8")
    return None


def _append_badges(report: str, result: dict[str, Any]) -> str:
    fraud = result.get("fraud_score")
    if fraud:
        probability = float(fraud.get("fraud_probability", 0.0))
        report += f"\n\n## Fraud Risk\n{fraud.get('risk_level', 'UNKNOWN')} ({probability:.2%})"

    forecast = result.get("forecast")
    if forecast:
        confidence = float(forecast.get("confidence", 0.0))
        report += f"\n\n## Forecast\n{forecast.get('direction', 'UNAVAILABLE')} ({confidence:.2%} confidence)"
    return report


@cl.on_chat_start
async def on_chat_start() -> None:
    await cl.Message(
        content=(
            "# FinSight AI\n"
            "Ask a financial research question. You can also attach a chart image for the orchestrator state."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    query = (message.content or "").strip()
    if not query:
        await cl.Message(content="Please enter a financial question.").send()
        return

    started = time.perf_counter()
    status = cl.Message(content="Running FinSight agents...")
    await status.send()

    try:
        graph_input = {
            "query": query,
            "image_data": await _read_uploaded_image(message),
            "retry_count": 0,
            "trace_log": [],
        }
        # run_graph is sync and can block for several seconds — offload to a thread
        # so the Chainlit event loop stays responsive.
        result = await asyncio.to_thread(run_graph, graph_input)
    except Exception as exc:  # noqa: BLE001
        await cl.Message(content=f"FinSight run failed: `{exc}`").send()
        return

    elapsed_ms = (time.perf_counter() - started) * 1000
    report = result.get("final_report") or "No report generated."
    await cl.Message(content=_append_badges(report, result)).send()

    chart_path = result.get("chart_path")
    if chart_path and Path(chart_path).exists():
        await cl.Message(
            content="Price chart",
            elements=[cl.Image(name="chart", path=chart_path, display="inline")],
        ).send()

    trace = format_trace(result.get("trace_log"))
    await cl.Message(content=f"## Agent Trace\n```text\n{trace}\n\nTotal runtime: {elapsed_ms:.0f} ms\n```").send()
