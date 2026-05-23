"""LangGraph StateGraph — the FinSight AI brain.

Topology:
    planner -> {rag, sql, sentiment, forecast} (parallel fan-out)
    sql     -> {chart, fraud}
    {rag, chart, fraud, sentiment, forecast} -> evaluator
    evaluator --conditional--> retry_bump -> planner   (loop)
                           \\-> synthesizer -> END

Defensive imports: teammates' agents may not be merged yet. We provide stub
fallbacks so the graph is always runnable and the import never breaks CI.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from langgraph.graph import END, StateGraph

from agents.base_agent import append_trace
from orchestrator.evaluator import evaluator_node, increment_retry, route_after_eval
from orchestrator.planner import planner_node
from orchestrator.synthesizer import synthesizer_node
from state import AgentState

# ---------------------------------------------------------------------------
# Defensive agent imports — stubs keep the graph runnable pre-integration.
# ---------------------------------------------------------------------------

try:
    from agents.rag_agent import run as rag_run  # type: ignore
except Exception:  # noqa: BLE001
    def rag_run(state: AgentState) -> Dict[str, Any]:
        return {
            "rag_result": "[stub] RAG agent not yet merged.",
            "sources": [],
            "trace_log": append_trace("RAG agent: STUB (awaiting Member 1 PR)"),
        }

try:
    from agents.sql_agent import run_sql as sql_run  # type: ignore
except Exception:  # noqa: BLE001
    def sql_run(state: AgentState) -> Dict[str, Any]:
        return {
            "sql_result": "[stub] SQL agent not yet merged.",
            "trace_log": append_trace("SQL agent: STUB (awaiting Member 2 PR)"),
        }

try:
    from agents.chart_agent import run_chart as chart_run  # type: ignore
except Exception:  # noqa: BLE001
    def chart_run(state: AgentState) -> Dict[str, Any]:
        return {
            "chart_path": None,
            "trace_log": append_trace("Chart agent: STUB (awaiting Member 2 PR)"),
        }

try:
    from agents.sentiment_agent import run as sentiment_run  # type: ignore
except Exception:  # noqa: BLE001
    def sentiment_run(state: AgentState) -> Dict[str, Any]:
        return {
            "sentiment_result": "[stub] Sentiment agent not yet merged.",
            "trace_log": append_trace("Sentiment agent: STUB (awaiting Member 4 PR)"),
        }

try:
    from agents.fraud_agent import run as fraud_run  # type: ignore
except Exception:  # noqa: BLE001
    def fraud_run(state: AgentState) -> Dict[str, Any]:
        return {
            "fraud_score": {
                "fraud_probability": 0.0,
                "is_fraud": False,
                "risk_level": "LOW",
                "confidence": 0.0,
                "message": "Fraud agent: STUB (awaiting Member 1 PR)",
            },
            "trace_log": append_trace("Fraud agent: STUB (awaiting Member 1 PR)"),
        }


# ---------------------------------------------------------------------------
# Forecast node — owned by Member 3. Uses predict_trend() wrapper.
# ---------------------------------------------------------------------------

_TICKERS = {
    "AAPL": "AAPL",
    "APPLE": "AAPL",
    "MSFT": "MSFT",
    "MICROSOFT": "MSFT",
    "NVDA": "NVDA",
    "NVIDIA": "NVDA",
    "TSLA": "TSLA",
    "TESLA": "TSLA",
    "AMZN": "AMZN",
    "AMAZON": "AMZN",
    "GOOGL": "GOOGL",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "META": "META",
    "AMD": "AMD",
    "NFLX": "NFLX",
    "NETFLIX": "NFLX",
}
_STOP_SYMBOLS = {"THE", "AND", "FOR", "WITH", "FROM", "SHOW", "WHAT", "WAS", "ARE", "HOW", "Q"}
_SUPPORTED_DEMO_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "JNJ"]
_FINANCIAL_TERMS = {
    "10-k",
    "filing",
    "filings",
    "revenue",
    "sales",
    "risk",
    "risks",
    "stock",
    "price",
    "trend",
    "market",
    "sentiment",
    "news",
    "outlook",
    "forecast",
    "volatility",
    "fraud",
    "transaction",
    "earnings",
    "income",
    "cash",
    "dividend",
    "valuation",
    "shares",
}
_SUPPORTED_COMPANY_NAMES = {
    "apple",
    "microsoft",
    "google",
    "alphabet",
    "amazon",
    "nvidia",
    "meta",
    "tesla",
    "jpmorgan",
    "visa",
    "johnson",
}


def _extract_ticker(query: str) -> Optional[str]:
    q = query.upper()
    for alias, ticker in _TICKERS.items():
        if alias in q:
            return ticker
    return None


def _unsupported_symbol(query: str) -> Optional[str]:
    for token in re.findall(r"\b[A-Z]{2,5}\b", query or ""):
        if token not in _STOP_SYMBOLS and token not in set(_TICKERS.values()) and token not in _SUPPORTED_DEMO_TICKERS:
            return token
    return None


def _is_in_scope_query(query: str) -> bool:
    lowered = (query or "").lower()
    if _extract_ticker(query):
        return True
    if any(name in lowered for name in _SUPPORTED_COMPANY_NAMES):
        return True
    return any(term in lowered for term in _FINANCIAL_TERMS)


def _guardrail_report(reason: str) -> str:
    supported = ", ".join(_SUPPORTED_DEMO_TICKERS)
    return (
        "## Out of Scope\n"
        f"{reason}\n\n"
        "## Supported Questions\n"
        "Ask about SEC filing facts, revenue, risks, stock prices, charts, sentiment, "
        "20-day outlook, volatility, or credit-card transaction fraud.\n\n"
        "## Supported Tickers\n"
        f"{supported}"
    )


def forecast_node(state: AgentState) -> Dict[str, Any]:
    """Run BOTH the direction forecaster and the volatility-regime predictor.

    Both are trained ML models (AIML Infra course). Their outputs are merged
    into a single `forecast` dict so the state contract stays unchanged.
    """
    ticker = _extract_ticker(state.get("query", ""))
    if not ticker:
        return {
            "forecast": {
                "direction": "UNAVAILABLE",
                "confidence": 0.0,
                "message": "No supported ticker was found for forecasting.",
            },
            "trace_log": append_trace("Forecast agent: skipped unsupported or missing ticker"),
        }

    merged: Dict[str, Any] = {"ticker": ticker, "days_ahead": 20}
    trace_parts = []

    try:
        from models.forecaster import predict_trend

        dir_result = predict_trend(ticker)
        merged["direction"] = dir_result.get("direction", "UNAVAILABLE")
        merged["direction_confidence"] = dir_result.get("confidence", 0.0)
        merged["up_probability"] = dir_result.get("up_probability")
        if "error" in dir_result:
            merged["direction_error"] = dir_result["error"]
        trace_parts.append(
            f"direction={merged['direction']} ({merged['direction_confidence']:.2f})"
        )
    except Exception as exc:  # noqa: BLE001
        merged["direction"] = "UNAVAILABLE"
        merged["direction_confidence"] = 0.0
        merged["direction_error"] = str(exc)
        trace_parts.append(f"direction=ERROR({exc})")

    try:
        from models.volatility_predictor import predict_volatility

        vol_result = predict_volatility(ticker)
        merged["volatility_regime"] = vol_result.get("regime", "UNAVAILABLE")
        merged["volatility_confidence"] = vol_result.get("confidence", 0.0)
        merged["high_vol_probability"] = vol_result.get("high_probability")
        if "error" in vol_result:
            merged["volatility_error"] = vol_result["error"]
        trace_parts.append(
            f"vol={merged['volatility_regime']} ({merged['volatility_confidence']:.2f})"
        )
    except Exception as exc:  # noqa: BLE001
        merged["volatility_regime"] = "UNAVAILABLE"
        merged["volatility_confidence"] = 0.0
        merged["volatility_error"] = str(exc)
        trace_parts.append(f"vol=ERROR({exc})")

    # Keep `confidence` as the legacy direction-confidence for any downstream
    # consumers expecting the old schema.
    merged["confidence"] = merged.get("direction_confidence", 0.0)

    return {
        "forecast": merged,
        "trace_log": append_trace(f"Forecast agent: {ticker} -> " + ", ".join(trace_parts)),
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def _agent_selected(state: AgentState, agent_name: str) -> bool:
    agents = state.get("agents_to_call") or []
    if agent_name == "fraud" and state.get("transaction_features"):
        return True
    return agent_name in agents


def _run_if_selected(agent_name: str, fn):
    def _wrapped(state: AgentState) -> Dict[str, Any]:
        if not _agent_selected(state, agent_name):
            return {"trace_log": append_trace(f"{agent_name.title()} agent: skipped by planner")}
        return fn(state)

    return _wrapped


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("rag", _run_if_selected("rag", rag_run))
    graph.add_node("sql", _run_if_selected("sql", sql_run))
    graph.add_node("chart", _run_if_selected("chart", chart_run))
    graph.add_node("sentiment", _run_if_selected("sentiment", sentiment_run))
    graph.add_node("fraud", _run_if_selected("fraud", fraud_run))
    graph.add_node("forecaster", _run_if_selected("forecast", forecast_node))
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("retry_bump", increment_retry)
    graph.add_node("synthesizer", synthesizer_node)

    graph.set_entry_point("planner")

    # Fan-out from planner. LangGraph runs these in parallel; convergence
    # at evaluator is handled by the reducer on trace_log.
    graph.add_edge("planner", "rag")
    graph.add_edge("planner", "sql")
    graph.add_edge("planner", "sentiment")
    graph.add_edge("planner", "forecaster")

    # SQL -> chart and fraud. Fraud depends on SQL-populated transaction_features.
    graph.add_edge("sql", "chart")
    graph.add_edge("sql", "fraud")

    # Converge at evaluator.
    graph.add_edge("rag", "evaluator")
    graph.add_edge("chart", "evaluator")
    graph.add_edge("sentiment", "evaluator")
    graph.add_edge("fraud", "evaluator")
    graph.add_edge("forecaster", "evaluator")

    # Reflection loop.
    graph.add_conditional_edges(
        "evaluator",
        route_after_eval,
        {"retry": "retry_bump", "proceed": "synthesizer"},
    )
    graph.add_edge("retry_bump", "planner")
    graph.add_edge("synthesizer", END)

    return graph


_compiled = None


def _compile():
    global _compiled
    if _compiled is None:
        _compiled = build_graph().compile()
    return _compiled


def run_graph(inputs: Dict[str, Any]) -> AgentState:
    """Public entry point — called by the Gradio UI (Member 4)."""
    defaults: Dict[str, Any] = {
        "query": "",
        "image_data": None,
        "plan": None,
        "agents_to_call": None,
        "rag_result": None,
        "sources": None,
        "sql_result": None,
        "chart_path": None,
        "sentiment_result": None,
        "transaction_features": None,
        "fraud_score": None,
        "forecast": None,
        "eval_score": None,
        "eval_feedback": None,
        "retry_count": 0,
        "final_report": None,
        "trace_log": [],
    }
    defaults.update(inputs)
    query = defaults.get("query", "")
    unsupported = _unsupported_symbol(query)
    if unsupported:
        defaults.update(
            {
                "final_report": _guardrail_report(
                    f"`{unsupported}` is not part of the indexed filing corpus or trained model universe."
                ),
                "trace_log": append_trace(
                    f"Input guard: unsupported ticker/symbol '{unsupported}' skipped agent execution"
                ),
            }
        )
        return defaults

    if not _is_in_scope_query(query):
        defaults.update(
            {
                "final_report": _guardrail_report(
                    "FinSight AI is a financial research demo, so this question was not sent to the agents."
                ),
                "trace_log": append_trace("Input guard: out-of-scope query skipped agent execution"),
            }
        )
        return defaults
    return _compile().invoke(defaults)


if __name__ == "__main__":
    import json as _json

    demo = run_graph({"query": "Analyse Apple's Q4 2024 performance and show price trend"})
    print("\n=== FINAL REPORT ===\n")
    print(demo.get("final_report"))
    print("\n=== TRACE ===\n")
    print("\n".join(demo.get("trace_log", [])))
    print("\n=== FULL STATE (truncated) ===\n")
    print(_json.dumps({k: str(v)[:200] for k, v in demo.items()}, indent=2))
