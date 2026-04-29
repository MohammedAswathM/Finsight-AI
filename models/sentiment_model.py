"""FinBERT sentiment inference wrapper.

Provides a single public function:
    predict_sentiment(text: str) -> dict
It prefers a locally fine-tuned model at `models/finbert-finetuned/` if present,
otherwise it falls back to the base `ProsusAI/finbert` checkpoint.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_BASE_MODEL = "ProsusAI/finbert"
_FINETUNED_DIR = Path(__file__).parent / "finbert-finetuned"


@lru_cache(maxsize=2)
def _load_tokenizer_and_model():
    model_source = str(_FINETUNED_DIR) if _FINETUNED_DIR.exists() else _BASE_MODEL
    tok = AutoTokenizer.from_pretrained(model_source, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_source)
    model.eval()
    # Read label mapping from model config to handle both base and fine-tuned models
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    return tok, model, id2label


def predict_sentiment(text: str) -> Dict[str, object]:
    """Predict sentiment for a single sentence.

    Returns:
        {"label": "positive"|"neutral"|"negative", "score": float, "summary": str}
    """
    text = (text or "").strip()
    if not text:
        return {"label": "neutral", "score": 0.0, "summary": "NEUTRAL (0.00)"}

    tok, model, id2label = _load_tokenizer_and_model()
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
        score, idx = torch.max(probs, dim=-1)

    label = id2label.get(int(idx), "neutral")
    score_f = float(score.item())
    return {"label": label, "score": score_f, "summary": f"{label.upper()} ({score_f:.2f})"}


if __name__ == "__main__":
    demo = "Apple beats Q4 earnings estimates"
    print(predict_sentiment(demo))
