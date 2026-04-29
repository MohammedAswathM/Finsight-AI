"""Synthesizer node — combines all agent outputs into the final markdown report."""
from __future__ import annotations

from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate

from agents.base_agent import append_trace, get_llm, safe_get
from state import AgentState

SYNTHESIZER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a financial-research report writer. Combine the provided research \
data into a clear, structured markdown report using EXACTLY these sections (omit a \
section only if its data is truly missing):

## Filing Analysis
[Summarise key facts from RAG. Cite sources inline as [source].]

## Price & Market Data
[Key price statistics and trends from SQL output.]

## Fraud Risk Assessment
[Include ONLY if fraud_score is present and not None.]

## News Sentiment
[Overall sentiment + 1–2 key themes.]

## 20-Day Outlook
[Two ML-model outputs: direction (UP/DOWN/FLAT with confidence) and volatility regime (HIGH/LOW with confidence). Present both clearly.]

## Summary
[2–3 sentence synthesis tying everything together.]

Rules:
- Never invent numbers. If a section has no data, write "Data not available."
- Keep each section tight (<= 5 lines).
- Respond with the markdown report only — no preamble, no code fence.""",
        ),
        (
            "human",
            """Query: {query}

RAG: {rag_result}
Sources: {sources}
SQL: {sql_result}
Fraud: {fraud_score}
Sentiment: {sentiment_result}
Forecast: {forecast}""",
        ),
    ]
)


def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    llm = get_llm(temperature=0.2)
    chain = SYNTHESIZER_PROMPT | llm

    try:
        response = chain.invoke(
            {
                "query": state["query"],
                "rag_result": safe_get(state, "rag_result", "No filing data retrieved."),
                "sources": ", ".join(state.get("sources") or []) or "none",
                "sql_result": safe_get(state, "sql_result", "No price data retrieved."),
                "fraud_score": safe_get(state, "fraud_score", "Not assessed"),
                "sentiment_result": safe_get(state, "sentiment_result", "No sentiment data."),
                "forecast": safe_get(state, "forecast", "No forecast available."),
            }
        )
        report = response.content
    except Exception as exc:
        report = (
            f"# FinSight AI Report\n\n"
            f"Report synthesis failed: {exc}\n\n"
            f"Raw RAG: {safe_get(state, 'rag_result')}\n"
            f"Raw SQL: {safe_get(state, 'sql_result')}\n"
            f"Raw Sentiment: {safe_get(state, 'sentiment_result')}\n"
        )

    return {
        "final_report": report,
        "trace_log": append_trace("Synthesizer: final report generated"),
    }
