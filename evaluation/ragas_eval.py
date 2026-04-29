"""RAGAS evaluation for five FinSight financial queries."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd
from datasets import Dataset

from orchestrator.graph import run_graph

SAMPLE_QUERIES = [
    "What does Apple's latest filing say about revenue growth?",
    "Summarize Microsoft's key risk factors from filings.",
    "Analyze Nvidia financial performance and market sentiment.",
    "Show AAPL closing price trend and explain the outlook.",
    "Compare Amazon filing highlights with recent news sentiment.",
]


def _contexts(result: Dict[str, Any]) -> List[str]:
    values = [
        result.get("rag_result"),
        result.get("sql_result"),
        result.get("sentiment_result"),
    ]
    return [str(value) for value in values if value] or ["No retrieved context available."]


def build_dataset() -> Dataset:
    rows = []
    for query in SAMPLE_QUERIES:
        result = run_graph({"query": query, "retry_count": 0, "trace_log": []})
        rows.append(
            {
                "question": query,
                "answer": result.get("final_report") or "",
                "contexts": _contexts(result),
            }
        )
    return Dataset.from_list(rows)


def run_ragas() -> pd.DataFrame:
    dataset = build_dataset()
    try:
        from langchain_groq import ChatGroq
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import answer_relevancy, context_precision, faithfulness

        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise RuntimeError("GROQ_API_KEY is required for Groq judge evaluation.")

        judge = LangchainLLMWrapper(
            ChatGroq(
                model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                groq_api_key=groq_key,
                temperature=0,
            )
        )
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=judge,
        )
        return result.to_pandas()
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame(
            {
                "question": SAMPLE_QUERIES,
                "faithfulness": [None] * len(SAMPLE_QUERIES),
                "answer_relevancy": [None] * len(SAMPLE_QUERIES),
                "context_precision": [None] * len(SAMPLE_QUERIES),
                "note": [f"RAGAS unavailable: {exc}"] * len(SAMPLE_QUERIES),
            }
        )


if __name__ == "__main__":
    print(run_ragas().to_string(index=False))
