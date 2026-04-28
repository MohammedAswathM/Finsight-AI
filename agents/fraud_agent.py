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


def run(state: AgentState) -> AgentState:
    """
    Fraud agent entry point.

    Reads:
      - state["transaction_features"] : Dict[str, float]
    Writes:
      - state["fraud_score"] : Dict
      - state["trace_log"] : List[str]
    """
    transaction_features = state.get("transaction_features")
    if not transaction_features:
        message = (
            "Fraud agent skipped: no transaction_features provided. "
            "Provide Kaggle credit card fraud features in state['transaction_features']."
        )
        state["fraud_score"] = {
            "fraud_probability": 0.0,
            "is_fraud": False,
            "risk_level": "LOW",
            "confidence": 0.0,
            "message": message,
        }
        state["trace_log"] = state.get("trace_log", []) + [
            "Fraud agent: no transaction_features provided"
        ]
        return state

    try:
        fraud_result = predict_fraud(transaction_features)
        state["fraud_score"] = fraud_result
        state["trace_log"] = state.get("trace_log", []) + [
            f"Fraud agent: fraud_probability={fraud_result['fraud_probability']:.4f} "
            f"risk_level={fraud_result['risk_level']}"
        ]
        logger.info("Fraud agent produced risk=%s prob=%.4f", fraud_result["risk_level"], fraud_result["fraud_probability"])
    except Exception as exc:
        state["fraud_score"] = {
            "fraud_probability": 0.0,
            "is_fraud": False,
            "risk_level": "LOW",
            "confidence": 0.0,
            "message": f"Fraud prediction failed: {exc}",
        }
        state["trace_log"] = state.get("trace_log", []) + [
            f"Fraud agent: ERROR — {exc}"
        ]
        logger.error("Fraud agent error: %s", exc)

    return state


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
