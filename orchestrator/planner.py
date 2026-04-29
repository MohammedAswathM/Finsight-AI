"""Planner node — first node in the graph. Decides which agents to call."""
from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate

from agents.base_agent import append_trace, get_llm, strip_code_fence
from state import AgentState

VALID_AGENTS = {"rag", "sql", "chart", "sentiment", "fraud", "forecast"}

PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a financial-research planner. Given a user query, decide which \
research agents are needed. Respond with a JSON object ONLY — no prose, no code fence.

Schema:
{{
  "plan": ["short step 1", "short step 2", ...],
  "agents_to_call": ["rag", "sql", "chart", "sentiment", "fraud", "forecast"]
}}

Rules for agents_to_call:
- "rag"       -> company filings, 10-K/10-Q, earnings, revenue, guidance, risks
- "sql"       -> stock prices, volume, historical OHLCV, trends over a period
- "chart"     -> any visual/trend request (always pair with "sql")
- "sentiment" -> news, market mood, analyst opinion, headlines
- "fraud"     -> transaction fraud, payment risk, suspicious card activity (pair with "sql")
- "forecast"  -> outlook, prediction, future price direction
- For broad/comprehensive questions, include ALL six agents.
- Always include at least one agent.""",
        ),
        ("human", "User query: {query}"),
    ]
)


def planner_node(state: AgentState) -> Dict[str, Any]:
    llm = get_llm()
    chain = PLANNER_PROMPT | llm

    try:
        response = chain.invoke({"query": state["query"]})
        parsed = json.loads(strip_code_fence(response.content))
        plan = parsed.get("plan") or ["Research the query."]
        agents = [a for a in parsed.get("agents_to_call", []) if a in VALID_AGENTS]
        if not agents:
            agents = ["rag", "sql", "chart", "sentiment", "forecast"]
    except Exception as exc:  # LLM/JSON failure — safe fallback: call everyone
        plan = ["Fallback: run all agents"]
        agents = ["rag", "sql", "chart", "sentiment", "fraud", "forecast"]
        return {
            "plan": plan,
            "agents_to_call": agents,
            "trace_log": append_trace(f"Planner: fallback (parse error: {exc}) -> {agents}"),
        }

    return {
        "plan": plan,
        "agents_to_call": agents,
        "trace_log": append_trace(f"Planner: routing to {agents}"),
    }


if __name__ == "__main__":
    out = planner_node({"query": "Analyse Apple's Q4 2024 and show price trend"})
    print(json.dumps(out, indent=2, default=str))
