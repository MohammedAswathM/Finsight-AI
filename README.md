# FinSight AI

Multi-agent financial research system. See `FINSIGHT_AI_BRAIN.md` for the full spec.

## Quickstart

```bash
# 1. Python 3.11+
python --version

# 2. Install deps
pip install -r requirements.txt

# 3. Configure secrets
cp .env.example .env     # then edit GROQ_API_KEY

# 4. Train the Member-3 forecaster (optional — graph runs without it)
python -m models.train_forecaster

# 4.5 Train the fraud detection model (optional until dataset is available)
# Download the dataset first:
curl -L -o data/creditcard.csv "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
# Then train the model:
python -m models.train_fraud

# 5. Smoke-test the orchestrator end-to-end (uses stubs for unmerged agents)
python -m orchestrator.graph

# 6. Launch UI (after Member 4 merges)
python -m ui.app
```

## Team branches

| Member | Branch             | Owns                                              |
| ------ | ------------------ | ------------------------------------------------- |
| 1      | `feature/fraud-model` | RAG agent, retrieval/, real fraud model training  |
| 2      | `feature/sql-chart`   | SQL/chart agents, data/, FinBERT fine-tune        |
| 3      | `feature/orchestrator` (this branch → `main`) | orchestrator/, forecaster, state contract |
| 4      | `feature/ui-eval`  | Gradio UI, sentiment agent, RAGAS eval, MLOps report |

Only Member 3 merges to `main`. Everyone else opens PRs against `main`; Member 3
reviews using the checklist in `FINSIGHT_AI_BRAIN.md` §18.

## MLflow

On Windows, we store runs at `C:\ProgramData\finsight-ai\mlruns` (space-free
path — avoids two separate MLflow-on-Windows bugs with our spacey project path).

View the UI:

```bash
mlflow ui --backend-store-uri "file:///C:/ProgramData/finsight-ai/mlruns"
```

Then open http://127.0.0.1:5000. Three experiments: `fraud-detection`,
`finbert-sentiment`, `stock-forecaster`.
