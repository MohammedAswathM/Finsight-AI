"""
FinSight AI — Shared State Contract.

This is the single source of truth passed between every node in the LangGraph.
DO NOT modify without notifying all team members (see FINSIGHT_AI_BRAIN.md §7).

Key best-practice note: `trace_log` uses `operator.add` as a reducer so parallel
agent branches (rag / sql / sentiment / forecast) can all append to it without
clobbering each other's updates. Without this, LangGraph raises
InvalidUpdateError when multiple parallel nodes write to the same key.
"""
from __future__ import annotations

import operator
from typing import Annotated, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    # --- INPUT ---
    query: str
    image_data: Optional[str]

    # --- PLANNING (Member 3) ---
    plan: Optional[List[str]]
    agents_to_call: Optional[List[str]]

    # --- RAG (Member 1) ---
    rag_result: Optional[str]
    sources: Optional[List[str]]

    # --- SQL + CHART (Member 2) ---
    sql_result: Optional[str]
    chart_path: Optional[str]

    # --- SENTIMENT (Member 4) ---
    sentiment_result: Optional[str]

    # --- FRAUD DETECTION (Member 1) ---
    transaction_features: Optional[Dict[str, float]]  # Kaggle credit card fraud features

    # --- ML MODEL OUTPUTS (AIML Infra) ---
    fraud_score: Optional[Dict]      # {"fraud_probability", "is_fraud", "risk_level"}
    forecast: Optional[Dict]         # {"direction", "confidence", "days_ahead"}

    # --- EVALUATION + CONTROL (Member 3) ---
    eval_score: Optional[float]
    eval_feedback: Optional[str]
    retry_count: int

    # --- FINAL OUTPUT (Member 3) ---
    final_report: Optional[str]

    # --- TRACE LOG — reducer allows parallel appends ---
    trace_log: Annotated[List[str], operator.add]
