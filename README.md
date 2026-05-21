# FinSight AI

Multi-agent financial research assistant. A LangGraph orchestrator coordinates six specialist agents (RAG over SEC 10-K filings, SQL over price data, sentiment over news headlines, fraud detection, price-trend forecasting, chart generation) and synthesizes their outputs into a single cited report.

Built as a college team project across two courses — **AIML Agentic** (the agent stack) and **AIML Infrastructure Engineering** (the ML models + MLOps).

## Stack

100% free / open-source. No paid services, no cloud costs.

- **LLM inference:** Groq free tier (`llama-3.3-70b-versatile`)
- **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (CPU)
- **Vector store:** ChromaDB (local persistent)
- **Relational store:** SQLite
- **Agent framework:** LangGraph + LangChain 0.2
- **MLOps:** MLflow file store
- **UI:** Chainlit
- **Evaluation:** RAGAS (LLM-as-judge)
- **ML models:** scikit-learn, LightGBM, XGBoost, FinBERT (transformers)

## Architecture

```
                   ┌─────────────┐
                   │   planner   │  Groq LLM decides which agents to involve
                   └──────┬──────┘
        ┌────────┬────────┼────────┬────────┐
        ▼        ▼        ▼        ▼        ▼
      ┌───┐   ┌───┐   ┌────────┐ ┌─────┐ ┌──────────┐
      │RAG│   │SQL│   │sentiment│ │fraud│ │forecaster│
      └─┬─┘   └─┬─┘   └────┬───┘ └──┬──┘ └────┬─────┘
        │       │ ┌──chart─┘        │         │
        │       │ │                 │         │
        ▼       ▼ ▼                 ▼         ▼
                ┌───────────┐
                │ evaluator │  scores answer; routes back to planner if too weak
                └─────┬─────┘
                ┌─────▼─────┐
                │synthesizer│  composes the final markdown report
                └───────────┘
```

State flows through a typed `AgentState` (see `state.py`); `trace_log` uses an additive reducer so parallel branches don't clobber each other.

## Repository layout

```
finsight-ai/
├── orchestrator/          ← graph, planner, evaluator, synthesizer (Member 3)
├── agents/                ← one file per specialist agent
├── retrieval/             ← ChromaDB vectorstore + ParentDocumentRetriever (Member 1)
├── data/                  ← SQLite setup + price fetchers (Member 2)
├── models/                ← trained ML wrappers + training scripts
│   ├── train_fraud.py        XGBoost + LightGBM (Member 1)
│   ├── train_finbert.py      FinBERT fine-tune (Member 2)
│   ├── train_forecaster.py   Price-direction LightGBM (Member 3)
│   └── train_volatility.py   Volatility-regime classifier (Member 3)
├── ui/                    ← Chainlit app (Member 4)
├── evaluation/            ← RAGAS, MLflow comparison, latency benchmark
├── tests/                 ← deliverable smoke tests
├── state.py               ← AgentState TypedDict (shared contract)
├── config.py              ← env loader (single source of truth for keys)
└── FINSIGHT_AI_BRAIN.md   ← internal design document
```

## Running the system

```bash
# Chainlit web UI — the demo surface
chainlit run ui/app.py
```

Opens at http://localhost:8000. Type a financial question; the orchestrator runs, you see a structured report, an inline chart, and a per-node trace.

```bash
# Headless graph run (prints the same content to stdout)
python -m orchestrator.graph

# MLflow tracking UI (training metrics + artifacts)
mlflow ui --backend-store-uri file:./mlruns
```

## Sample queries

Best-tested queries for the demo:

- `Analyze Apple 2024 10-K revenue, price trend, sentiment, and outlook` — full agent fan-out
- `Analyze Microsoft 10-K cloud revenue and recent sentiment` — filing + sentiment query
- `Show AAPL closing price trend and explain the outlook` — SQL + chart + forecaster
- `Summarize Amazon's risk factors from filings` — RAG-heavy, returns real legal-risk content
- `Compare Microsoft and Nvidia Q4 results` — multi-ticker, often triggers the reflection retry loop

Indexed filing coverage: Apple 2024 plus Microsoft, Amazon, Alphabet, and Meta 2023 10-K filings.

## Team

| Member | Branch | Owns |
|---|---|---|
| 1 | `feature/rag` | RAG agent, retrieval/, fraud detection model |
| 2 | `feature/sql-chart` | SQL + chart agents, data/, FinBERT fine-tune |
| 3 | `feature/orchestrator` → main | orchestrator/, forecaster, state.py, config.py |
| 4 | `feature/ui-eval` | Chainlit UI, sentiment agent, RAGAS eval |

The orchestrator is the only branch that merges to `main`. See `FINSIGHT_AI_BRAIN.md` for the full integration contract.

## Known limitations

- **Forecaster default ticker**: when the planner can't extract a ticker from the query, the forecaster defaults to AAPL. Reports for non-AAPL questions will note "data not available for X, but here's AAPL's forecast."
- **Free Groq daily quota**: a heavy session of UI queries + RAGAS can exhaust this. Configure `GROQ_API_KEYS` or `GROQ_API_KEY_2`/`GROQ_API_KEY_3` for rate-limit failover.
- **Trace duplicates**: the LangGraph join node fires once per super-step that has new inputs, so the trace can show two `Synthesizer: done` lines. The final state is correct.

## License

Course project — not for production use.
