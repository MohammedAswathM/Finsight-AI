"""Chainlit UI for FinSight AI.

Run:
    chainlit run ui/app.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

# Chainlit loads this file as a script; ensure the project root is importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import chainlit as cl

from orchestrator.graph import run_graph
from ui.trace_panel import format_trace


def _append_badges(report: str, result: dict[str, Any]) -> str:
    fraud = result.get("fraud_score")
    if fraud and fraud.get("risk_level") not in {"NOT_ASSESSED", "UNKNOWN"}:
        probability = float(fraud.get("fraud_probability", 0.0))
        report += f"\n\n## Fraud Risk\n{fraud.get('risk_level', 'UNKNOWN')} ({probability:.2%})"

    forecast = result.get("forecast")
    if forecast and forecast.get("direction") not in {"UNAVAILABLE", None}:
        confidence = float(forecast.get("confidence", 0.0))
        report += f"\n\n## Forecast\n{forecast.get('direction', 'UNAVAILABLE')} ({confidence:.2%} confidence)"
    return report


@cl.on_chat_start
async def on_chat_start() -> None:
    await cl.Message(
        content=(
            "# FinSight AI\n"
            "Ask a financial research question about the indexed companies and trained model outputs."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    query = (message.content or "").strip()
    if not query:
        await cl.Message(content="Please enter a financial question.").send()
        return
    if message.elements:
        await cl.Message(
            content=(
                "Image upload is disabled for this demo because the current pipeline does "
                "not include OCR or a vision model. Please ask a text question about one "
                "of the supported tickers: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM, V, JNJ."
            )
        ).send()
        return

    started = time.perf_counter()
    status = cl.Message(content="Running FinSight agents...")
    await status.send()

    try:
        graph_input = {
            "query": query,
            "image_data": None,
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
