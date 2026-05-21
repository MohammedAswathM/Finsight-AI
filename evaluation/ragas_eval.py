"""RAGAS evaluation for FinSight financial queries.

The script is intentionally resumable because free Groq quotas can be exhausted
mid-evaluation. It writes:
    outputs/evaluation/ragas_dataset.csv
    outputs/evaluation/ragas_results.csv

Configure one or more judge keys with either GROQ_API_KEY or GROQ_API_KEYS.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from datasets import Dataset

os.environ.setdefault("RAG_USE_COMPRESSION", "0")

from agents.base_agent import is_rate_limit_error
from config import GROQ_MODEL, get_groq_keys
from orchestrator.graph import run_graph

SAMPLE_QUERIES = [
    "What does Microsoft's latest filing say about revenue growth?",
    "Summarize Microsoft's key risk factors from filings.",
    "Analyze Meta filing highlights and recent market sentiment.",
    "Show AAPL closing price trend and explain the outlook.",
    "Compare Amazon filing highlights with recent news sentiment.",
]

OUTPUT_DIR = Path("outputs") / "evaluation"
DATASET_PATH = OUTPUT_DIR / "ragas_dataset.csv"
RESULTS_PATH = OUTPUT_DIR / "ragas_results.csv"


def _contexts(result: Dict[str, Any]) -> List[str]:
    values = [
        result.get("rag_result"),
        result.get("sql_result"),
        result.get("sentiment_result"),
    ]
    return [str(value) for value in values if value] or ["No retrieved context available."]


def build_dataset(queries: List[str] | None = None, force: bool = False) -> Dataset:
    selected_queries = queries or SAMPLE_QUERIES
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if DATASET_PATH.exists() and not force:
        df = pd.read_csv(DATASET_PATH)
        df = df[df["question"].isin(selected_queries)].copy()
        df["contexts"] = df["contexts"].apply(lambda value: str(value).split("\n---CONTEXT---\n"))
        return Dataset.from_pandas(df, preserve_index=False)

    rows = []
    for query in selected_queries:
        result = run_graph({"query": query, "retry_count": 0, "trace_log": []})
        rows.append(
            {
                "question": query,
                "answer": result.get("final_report") or "",
                "contexts": _contexts(result),
            }
        )
        checkpoint = pd.DataFrame(
            {
                "question": [row["question"] for row in rows],
                "answer": [row["answer"] for row in rows],
                "contexts": ["\n---CONTEXT---\n".join(row["contexts"]) for row in rows],
            }
        )
        checkpoint.to_csv(DATASET_PATH, index=False)
    return Dataset.from_list(rows)


def _evaluate_one(row: Dict[str, Any], groq_key: str) -> pd.DataFrame:
    single = Dataset.from_list([row])
    from langchain_groq import ChatGroq
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import LLMContextPrecisionWithoutReference, answer_relevancy, faithfulness
    from ragas.run_config import RunConfig
    from retrieval.vectorstore import get_embeddings

    judge = LangchainLLMWrapper(
        ChatGroq(
            model=GROQ_MODEL,
            groq_api_key=groq_key,
            temperature=0,
            timeout=45,
            max_retries=0,
        )
    )
    result = evaluate(
        single,
        metrics=[faithfulness, answer_relevancy, LLMContextPrecisionWithoutReference()],
        llm=judge,
        embeddings=get_embeddings(),
        run_config=RunConfig(timeout=60, max_retries=0, max_wait=5, max_workers=1),
        raise_exceptions=True,
        show_progress=False,
    )
    return result.to_pandas()


def _select_keys(key_indexes: str | None = None) -> list[str]:
    keys = get_groq_keys()
    configured = key_indexes or os.getenv("RAGAS_GROQ_KEY_INDEXES")
    if not configured:
        return keys

    selected: list[str] = []
    for raw_index in configured.split(","):
        raw_index = raw_index.strip()
        if not raw_index:
            continue
        index = int(raw_index)
        if index < 1 or index > len(keys):
            raise ValueError(
                f"Requested Groq key index {index}, but only {len(keys)} key(s) are configured."
            )
        selected.append(keys[index - 1])
    return selected


def run_ragas(
    force_dataset: bool = False,
    limit: int | None = None,
    key_indexes: str | None = None,
    reset_results: bool = False,
    sleep_between: float = 20.0,
) -> pd.DataFrame:
    queries = SAMPLE_QUERIES[:limit] if limit else SAMPLE_QUERIES
    dataset = build_dataset(queries=queries, force=force_dataset)
    if reset_results and RESULTS_PATH.exists():
        RESULTS_PATH.unlink()

    keys = _select_keys(key_indexes)
    if not keys:
        return pd.DataFrame(
            {
                "question": queries,
                "faithfulness": [None] * len(queries),
                "answer_relevancy": [None] * len(queries),
                "context_precision": [None] * len(queries),
                "note": ["RAGAS unavailable: no GROQ_API_KEY or GROQ_API_KEYS configured"] * len(queries),
            }
        )

    existing = pd.read_csv(RESULTS_PATH) if RESULTS_PATH.exists() else pd.DataFrame()
    if not existing.empty and "question" not in existing and "user_input" in existing:
        existing["question"] = existing["user_input"]
    metric_cols = [
        "faithfulness",
        "answer_relevancy",
        "llm_context_precision_without_reference",
        "context_precision",
    ]
    completed = set()
    if not existing.empty and "question" in existing:
        for _, existing_row in existing.iterrows():
            available_metrics = [col for col in metric_cols if col in existing_row.index]
            if available_metrics and existing_row[available_metrics].notna().all():
                completed.add(existing_row["question"])
    rows = existing.to_dict("records") if not existing.empty else []

    try:
        for row in dataset:
            if row["question"] in completed:
                continue

            last_exc: Exception | None = None
            for key in keys:
                try:
                    scored = _evaluate_one(dict(row), key)
                    scored["question"] = row["question"]
                    scored_metric_cols = [
                        col
                        for col in metric_cols
                        if col in scored.columns
                    ]
                    if not scored_metric_cols or not scored[scored_metric_cols].notna().all(axis=None):
                        raise RuntimeError("RAGAS returned incomplete metrics for this key.")
                    scored["note"] = (
                        "ok"
                    )
                    rows.extend(scored.to_dict("records"))
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if not is_rate_limit_error(exc) and "incomplete metrics" not in str(exc):
                        break
            else:
                last_exc = last_exc or RuntimeError("RAGAS judge failed.")

            if last_exc and (not rows or rows[-1].get("question") != row["question"]):
                rows.append(
                    {
                        "question": row["question"],
                        "faithfulness": None,
                        "answer_relevancy": None,
                        "context_precision": None,
                        "note": f"RAGAS unavailable: {last_exc}",
                    }
                )

            pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)
            if sleep_between > 0:
                import time

                time.sleep(sleep_between)

        return pd.DataFrame(rows)
    except Exception as exc:  # noqa: BLE001
        fallback = pd.DataFrame(rows) if rows else pd.DataFrame({"question": queries})
        fallback["note"] = fallback.get("note", f"RAGAS unavailable: {exc}")
        return fallback


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run resumable RAGAS evaluation.")
    parser.add_argument("--force-dataset", action="store_true", help="Rebuild graph outputs before judging.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N sample queries.")
    parser.add_argument(
        "--key-indexes",
        default=None,
        help="Comma-separated 1-based Groq key indexes to use, e.g. '2,3' skips GROQ_API_KEY.",
    )
    parser.add_argument("--reset-results", action="store_true", help="Clear previous RAGAS result checkpoint.")
    parser.add_argument(
        "--sleep-between",
        type=float,
        default=20.0,
        help="Seconds to pause between evaluated queries to respect Groq TPM limits.",
    )
    args = parser.parse_args()
    print(
        run_ragas(
            force_dataset=args.force_dataset,
            limit=args.limit,
            key_indexes=args.key_indexes,
            reset_results=args.reset_results,
            sleep_between=args.sleep_between,
        ).to_string(index=False)
    )
