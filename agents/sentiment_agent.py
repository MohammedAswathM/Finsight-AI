"""News sentiment agent for FinSight AI.

Member 4 deliverable: fetch free Yahoo Finance RSS headlines and score them
with Member 2's local FinBERT wrapper. No OpenAI or GPT-4o dependency.
"""
from __future__ import annotations

import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import feedparser

if __package__ is None and str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.base_agent import append_trace
from models.sentiment_model import predict_sentiment
from state import AgentState

_TICKER_ALIASES = {
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
    "JPM": "JPM",
    "JPMORGAN": "JPM",
}

_STOPWORDS = {"THE", "AND", "FOR", "WITH", "FROM", "SHOW", "WHAT", "NEWS", "PRICE"}


def extract_ticker(query: str) -> str:
    """Infer a ticker from a user query."""
    text = (query or "").upper()
    for alias, ticker in _TICKER_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", text):
            return ticker

    for token in re.findall(r"\b[A-Z]{2,5}\b", text):
        if token not in _STOPWORDS:
            return token
    return "AAPL"


def fetch_headlines(query: str, limit: int = 10) -> List[str]:
    """Fetch recent finance headlines using Yahoo Finance RSS."""
    ticker = extract_ticker(query)
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    feed = feedparser.parse(url)

    headlines: List[str] = []
    for entry in getattr(feed, "entries", [])[:limit]:
        title = getattr(entry, "title", "").strip()
        if title:
            headlines.append(title)
    return headlines


def summarize_scores(scores: Iterable[Dict[str, Any]], headlines: List[str]) -> str:
    """Create the compact sentiment summary expected by the orchestrator."""
    score_list = list(scores)
    if not score_list:
        return "NEUTRAL (0.00): No recent finance headlines were available."

    labels = [str(item.get("label", "neutral")).lower() for item in score_list]
    dominant = Counter(labels).most_common(1)[0][0]
    avg_score = sum(float(item.get("score", 0.0)) for item in score_list) / len(score_list)
    key_themes = "; ".join(headlines[:3]) if headlines else "No headline themes available"

    return (
        f"{dominant.upper()} ({avg_score:.2f}): "
        f"Based on {len(score_list)} recent headlines. Key themes: {key_themes}"
    )


def _fallback_sentiment(text: str) -> Dict[str, Any]:
    """Small offline fallback used only when the local FinBERT artifact is absent."""
    lowered = text.lower()
    positive_terms = {"beat", "beats", "growth", "rally", "gain", "surge", "strong", "record", "bullish"}
    negative_terms = {"miss", "loss", "falls", "drop", "weak", "risk", "lawsuit", "bearish", "cut"}
    pos_hits = sum(term in lowered for term in positive_terms)
    neg_hits = sum(term in lowered for term in negative_terms)
    if pos_hits > neg_hits:
        return {"label": "positive", "score": 0.60, "summary": "POSITIVE (0.60 fallback)"}
    if neg_hits > pos_hits:
        return {"label": "negative", "score": 0.60, "summary": "NEGATIVE (0.60 fallback)"}
    return {"label": "neutral", "score": 0.50, "summary": "NEUTRAL (0.50 fallback)"}


def score_headlines(headlines: List[str]) -> tuple[List[Dict[str, Any]], str | None]:
    """Score headlines with local FinBERT, falling back if its artifact is unavailable."""
    finbert_dir = Path(__file__).resolve().parents[1] / "models" / "finbert-finetuned"
    if not finbert_dir.exists():
        return [_fallback_sentiment(headline) for headline in headlines], (
            "models/finbert-finetuned not found"
        )

    try:
        return [predict_sentiment(headline) for headline in headlines], None
    except Exception as exc:  # noqa: BLE001
        return [_fallback_sentiment(headline) for headline in headlines], str(exc)


def run(state: AgentState) -> Dict[str, Any]:
    """Fetch headlines, classify them with FinBERT, and update AgentState."""
    started = time.perf_counter()
    try:
        query = state.get("query", "")
        ticker = extract_ticker(query)
        headlines = fetch_headlines(query)
        if not headlines:
            headlines = [f"No recent Yahoo Finance headlines found for {ticker}."]

        scores, model_error = score_headlines(headlines)
        elapsed_ms = (time.perf_counter() - started) * 1000
        sentiment = summarize_scores(scores, headlines)
        if model_error:
            sentiment += " FinBERT artifact unavailable; lexical fallback used."
        return {
            "sentiment_result": sentiment,
            "trace_log": append_trace(
                f"Sentiment agent: analyzed {len(headlines)} headlines for {ticker} in {elapsed_ms:.0f} ms"
            ),
        }
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "sentiment_result": f"Sentiment analysis failed: {exc}",
            "trace_log": append_trace(f"Sentiment agent: failed in {elapsed_ms:.0f} ms ({exc})"),
        }


if __name__ == "__main__":
    print(run({"query": "What is the current news sentiment around Nvidia?"}))
