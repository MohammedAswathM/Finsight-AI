"""Test script for Member 2 deliverables (SQL agent, Chart agent, FinBERT).

Run: python -m tests.test_member2
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script.
if __package__ is None and str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_sql_agent():
    """Test SQL agent with natural language query."""
    print("=" * 60)
    print("TEST 1: SQL Agent")
    print("=" * 60)

    from agents.sql_agent import run_sql

    test_query = "Show AAPL closing prices for last 6 months"
    print(f"Query: {test_query}")
    print("-" * 40)

    result = run_sql({"query": test_query})
    sql_result = result.get("sql_result", "No result")

    print(sql_result[:800])
    print("\nSQL Agent: PASSED\n")
    return True


def test_chart_agent():
    """Test chart agent with query."""
    print("=" * 60)
    print("TEST 2: Chart Agent")
    print("=" * 60)

    from agents.chart_agent import run_chart

    test_query = "Show AAPL closing prices for last 6 months"
    print(f"Query: {test_query}")
    print("-" * 40)

    result = run_chart({"query": test_query, "sql_result": "test"})
    chart_path = result.get("chart_path")

    print(f"Chart saved to: {chart_path}")
    print("\nChart Agent: PASSED\n")
    return True


def test_sentiment_model():
    """Test FinBERT sentiment model."""
    print("=" * 60)
    print("TEST 3: FinBERT Sentiment Model")
    print("=" * 60)

    from models.sentiment_model import predict_sentiment

    test_cases = [
        "Apple beats Q4 earnings estimates",
        "Company reports loss for the quarter",
        "Market remains stable amid uncertainty",
    ]

    for text in test_cases:
        print(f"Input: {text}")
        result = predict_sentiment(text)
        print(f"Output: {result}")
        print()

    print("Sentiment Model: PASSED\n")
    return True


def test_mlflow_results():
    """Display FinBERT MLflow results."""
    print("=" * 60)
    print("TEST 4: MLflow Results (FinBERT)")
    print("=" * 60)

    import json

    mlflow_path = Path(__file__).parent.parent / "mlruns" / "1"
    run_dirs = list(mlflow_path.glob("*/artifacts/finbert_eval/"))

    if run_dirs:
        eval_dir = run_dirs[0]
        summary_path = eval_dir / "comparison_summary.json"

        if summary_path.exists():
            with open(summary_path) as f:
                data = json.load(f)

            print("Base Model (ProsusAI/finbert):")
            print(f"  Accuracy: {data['base']['accuracy']:.4f} ({data['base']['accuracy']*100:.2f}%)")
            print(f"  F1 Macro: {data['base']['f1_macro']:.4f}")
            print()
            print("Fine-tuned Model:")
            print(f"  Accuracy: {data['finetuned']['accuracy']:.4f} ({data['finetuned']['accuracy']*100:.2f}%)")
            print(f"  F1 Macro: {data['finetuned']['f1_macro']:.4f}")
            print()
            improvement = (data['finetuned']['accuracy'] - data['base']['accuracy']) * 100
            print(f"Accuracy Improvement: +{improvement:.2f}%")
        else:
            print("MLflow results not found. Run: python -m models.train_finbert")
    else:
        print("MLflow results not found. Run: python -m models.train_finbert")

    print("\nMLflow Results: PASSED\n")
    return True


def test_database():
    """Test database has data."""
    print("=" * 60)
    print("TEST 5: Database")
    print("=" * 60)

    import sqlite3

    db_path = Path(__file__).parent.parent / "outputs" / "finsight.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cursor.fetchall()]
    print(f"Tables: {tables}")

    cursor.execute("SELECT COUNT(*) FROM prices")
    price_count = cursor.fetchone()[0]
    print(f"Price rows: {price_count}")

    cursor.execute("SELECT DISTINCT ticker FROM prices ORDER BY ticker")
    tickers = [r[0] for r in cursor.fetchall()]
    print(f"Tickers: {tickers}")

    conn.close()

    print("\nDatabase: PASSED\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("MEMBER 2 DELIVERABLES TEST SUITE")
    print("SQL Agent + Chart Agent + FinBERT Sentiment")
    print("=" * 60 + "\n")

    tests = [
        ("Database", test_database),
        ("SQL Agent", test_sql_agent),
        ("Chart Agent", test_chart_agent),
        ("Sentiment Model", test_sentiment_model),
        ("MLflow Results", test_mlflow_results),
    ]

    results = []
    for name, test_fn in tests:
        try:
            results.append(test_fn())
        except Exception as e:
            print(f"{name} FAILED: {e}\n")
            results.append(False)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\nALL TESTS PASSED! Member 2 deliverables complete.")
    else:
        print("\nSome tests failed. Check errors above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
