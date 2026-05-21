"""Critic / evaluator node + conditional router for the reflection loop."""
from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate

from agents.base_agent import append_trace, invoke_prompt_with_fallback, safe_get, strip_code_fence
from state import AgentState

PASS_THRESHOLD = 0.7
MAX_RETRIES = 2

EVALUATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a financial-research quality evaluator. Review the combined agent \
output and score it from 0.0 to 1.0.

Scoring rubric:
- 1.0       : All requested data present, internally consistent, well-sourced.
- 0.7–0.9   : Most data present; minor gaps.
- 0.4–0.6   : Significant data missing or inconsistent.
- 0.0–0.3   : Major failure; most data missing or hallucinated.

Be strict about: missing citations on financial claims, sentiment/filing mismatch, \
empty agent outputs, and generic placeholders.

Respond with JSON ONLY:
{{
  "score": <float 0.0-1.0>,
  "feedback": "<one-sentence diagnosis of what is weak or missing>"
}}""",
        ),
        (
            "human",
            """Query: {query}

--- RAG ---
{rag_result}

Sources: {sources}

--- SQL ---
{sql_result}

--- SENTIMENT ---
{sentiment_result}

--- FORECAST ---
{forecast}

--- FRAUD ---
{fraud_score}

Evaluate the combined output.""",
        ),
    ]
)


def evaluator_node(state: AgentState) -> Dict[str, Any]:
    try:
        response = invoke_prompt_with_fallback(
            EVALUATOR_PROMPT,
            {
                "query": state["query"],
                "rag_result": safe_get(state, "rag_result", "NOT RETRIEVED"),
                "sql_result": safe_get(state, "sql_result", "NOT RETRIEVED"),
                "sentiment_result": safe_get(state, "sentiment_result", "NOT RETRIEVED"),
                "forecast": safe_get(state, "forecast", "NOT COMPUTED"),
                "fraud_score": safe_get(state, "fraud_score", "NOT COMPUTED"),
                "sources": safe_get(state, "sources", "[]"),
            },
        )
        parsed = json.loads(strip_code_fence(response.content))
        score = float(parsed.get("score", 0.5))
        feedback = str(parsed.get("feedback", ""))
    except Exception as exc:
        score = 0.6  # lean "acceptable" on parse failure so we don't thrash
        feedback = f"Evaluator parse error: {exc}"

    score = max(0.0, min(1.0, score))

    return {
        "eval_score": score,
        "eval_feedback": feedback,
        "trace_log": append_trace(f"Evaluator: score={score:.2f} — {feedback[:120]}"),
    }


def route_after_eval(state: AgentState) -> str:
    """Conditional-edge router called AFTER evaluator_node."""
    score = state.get("eval_score") or 0.0
    retry_count = state.get("retry_count", 0) or 0

    if score >= PASS_THRESHOLD:
        return "proceed"
    if retry_count >= MAX_RETRIES:
        return "proceed"  # give up gracefully, synthesize best-effort
    return "retry"


def increment_retry(state: AgentState) -> Dict[str, Any]:
    """Tiny node used on the retry branch to bump retry_count."""
    current = state.get("retry_count", 0) or 0
    return {
        "retry_count": current + 1,
        "trace_log": append_trace(f"Retry loop: attempt {current + 1}/{MAX_RETRIES}"),
    }
