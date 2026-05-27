"""Chainlit UI for FinSight AI.

Run:
    chainlit run ui/app.py
"""
from __future__ import annotations

import asyncio
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Chainlit loads this file as a script; ensure the project root is importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import chainlit as cl

from orchestrator.graph import run_graph
from ui.trace_panel import format_trace

REPORT_DIR = _PROJECT_ROOT / "outputs" / "reports"
GUARDRAIL_TRACE_PREFIX = "Input guard:"


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


def _report_filename(query: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", query.lower()).strip("_")[:60]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"finsight_report_{slug or 'query'}_{timestamp}.md"


def _write_report_file(query: str, report: str, result: dict[str, Any], elapsed_ms: float) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    trace = format_trace(result.get("trace_log"))
    chart_path = result.get("chart_path") or "N/A"
    report_path = REPORT_DIR / _report_filename(query)
    report_path.write_text(
        (
            "# FinSight AI Report\n\n"
            f"**Query:** {query}\n\n"
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"**Runtime:** {elapsed_ms:.0f} ms\n\n"
            f"**Chart Path:** {chart_path}\n\n"
            "---\n\n"
            f"{report}\n\n"
            "---\n\n"
            "## Agent Trace\n\n"
            "```text\n"
            f"{trace}\n\n"
            f"Total runtime: {elapsed_ms:.0f} ms\n"
            "```\n"
        ),
        encoding="utf-8",
    )
    return report_path


def _should_offer_download(result: dict[str, Any]) -> bool:
    trace = result.get("trace_log") or []
    if any(str(item).startswith(GUARDRAIL_TRACE_PREFIX) for item in trace):
        return False
    return bool(result.get("final_report"))


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
    display_report = _append_badges(report, result)
    await cl.Message(content=display_report).send()

    if _should_offer_download(result):
        try:
            report_path = _write_report_file(query, display_report, result, elapsed_ms)
            await cl.Message(
                content="Download report",
                elements=[
                    cl.File(
                        name="FinSight_Report.md",
                        content=report_path.read_bytes(),
                        mime="text/markdown",
                        display="inline",
                    )
                ],
            ).send()
        except Exception as exc:  # noqa: BLE001
            await cl.Message(content=f"Report download could not be prepared: `{exc}`").send()

    chart_path = result.get("chart_path")
    if chart_path:
        resolved_chart_path = (_PROJECT_ROOT / chart_path).resolve()
        if resolved_chart_path.exists():
            try:
                await cl.Message(
                    content="Price chart",
                    elements=[cl.Image(name="chart", path=str(resolved_chart_path), display="inline")],
                ).send()
            except Exception as exc:  # noqa: BLE001
                await cl.Message(content=f"Price chart could not be displayed: `{exc}`").send()
        else:
            await cl.Message(content=f"Price chart file was not found: `{chart_path}`").send()

    trace = format_trace(result.get("trace_log"))
    await cl.Message(content=f"## Agent Trace\n```text\n{trace}\n\nTotal runtime: {elapsed_ms:.0f} ms\n```").send()
