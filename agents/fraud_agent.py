"""
agents/fraud_agent.py
---------------------
Real fraud detection agent for FinSight AI.

Uses the trained fraud detector model saved by `models/train_fraud.py`.
The agent expects transaction features in `state["transaction_features"]`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from state import AgentState  # noqa: E402
from models.fraud_detector import predict_fraud  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"


def _load_sample_transaction_features() -> Dict[str, float]:
    """Load one deterministic sample from the held-out fraud test split."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Fraud dataset not found at {DATA_PATH}. "
            "Provide state['transaction_features'] or add data/creditcard.csv."
        )

    df = pd.read_csv(DATA_PATH)
    if "Class" not in df.columns:
        raise ValueError("Expected target column 'Class' in data/creditcard.csv.")

    X = df.drop(columns=["Class"])
    y = df["Class"]
    _, X_test, _, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return {key: float(value) for key, value in X_test.iloc[0].to_dict().items()}


def run(state: AgentState) -> Dict[str, Any]:
    """
    Fraud agent entry point.

    Reads:
      - state["transaction_features"] : Dict[str, float]
    Returns:
      - Delta dict with fraud_score and trace_log (does not mutate input state)
    """
    trace_prefix = "state transaction_features"
    transaction_features = state.get("transaction_features")
    if not transaction_features:
        transaction_features = _load_sample_transaction_features()
        trace_prefix = "held-out creditcard.csv sample"

    try:
        fraud_result = predict_fraud(transaction_features)
        fraud_label = "FRAUD" if fraud_result["is_fraud"] else "LEGIT"
        trace_msg = (
            f"Fraud agent: fraud_probability={fraud_result['fraud_probability']:.4f} "
            f"risk_level={fraud_result['risk_level']} source={trace_prefix}"
        )
        logger.info("Fraud agent produced risk=%s prob=%.4f", fraud_result["risk_level"], fraud_result["fraud_probability"])
        return {
            "fraud_score": fraud_result,
            "fraud_label": fraud_label,
            "trace_log": [trace_msg],
        }
    except Exception as exc:
        error_msg = f"Fraud agent: ERROR — {exc}"
        logger.error("Fraud agent error: %s", exc)
        return {
            "fraud_score": {
                "fraud_probability": 0.0,
                "is_fraud": False,
                "risk_level": "LOW",
                "confidence": 0.0,
                "message": f"Fraud prediction failed: {exc}",
            },
            "fraud_label": "UNKNOWN",
            "trace_log": [error_msg],
        }


if __name__ == "__main__":
    sample_features = {
        "Time": 10000,
        **{f"V{i}": 0.0 for i in range(1, 29)},
        "Amount": 50.0,
    }

    state: AgentState = {
        "query": "Test fraud detection",
        "plan": None,
        "agents_to_call": None,
        "rag_result": None,
        "sources": None,
        "sql_result": None,
        "chart_path": None,
        "sentiment_result": None,
        "fraud_score": None,
        "forecast": None,
        "eval_score": None,
        "eval_feedback": None,
        "retry_count": 0,
        "final_report": None,
        "trace_log": [],
        "transaction_features": sample_features,
    }

    result = run(state)
    print(result["fraud_score"])
    print(result["trace_log"])
