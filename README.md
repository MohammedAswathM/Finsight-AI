# FinSight AI — Part A: RAG Agent

Retrieval-Augmented Generation pipeline for SEC 10-K financial filings.

## Project Structure

```
finsight_rag/
├── state.py                   ← Shared AgentState TypedDict
├── requirements.txt
├── agents/
│   └── rag_agent.py           ← run(state) entry point  ← YOUR DELIVERABLE
├── retrieval/
│   ├── vectorstore.py         ← ChromaDB + HuggingFace embeddings
│   ├── ingest.py              ← Download 10-Ks → parse → embed
│   └── retriever.py           ← ParentDocumentRetriever + ContextualCompression
├── data/pdfs/                 ← Downloaded SEC HTM filings (auto-created)
└── chroma_db/                 ← Persistent vector store (auto-created)
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Pre-ingest filings — happens automatically on first run() call
python -m retrieval.ingest

# 3. Test the agent standalone
python agents/rag_agent.py

# 4. (Optional) Train the fraud detection model
# Download the dataset first:
curl -L -o data/creditcard.csv "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
# Then train:
python -m models.train_fraud
```

## Key Design Decisions

### Embeddings — 100% Free
```python
HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```
~90 MB download, runs on CPU, no API key required.

### Two-Layer Retrieval
1. **ParentDocumentRetriever** — embeds small 200-token child chunks for
   precise semantic matching, returns full 1 000-token parent chunks to
   preserve financial context.

2. **ContextualCompressionRetriever + LLMChainExtractor** — for each
   candidate chunk, an LLM extracts only the sentences directly relevant
   to the query, cutting noise by ~60-70%.

### Compression LLM
- Uses **flan-t5-base** locally (free, ~900 MB) when no `OPENAI_API_KEY`
  is set.
- Automatically upgrades to **GPT-3.5-turbo** if `OPENAI_API_KEY` is
  present in the environment.

## Required Signature (Satisfied)

```python
from state import AgentState

def run(state: AgentState) -> AgentState:
    query = state["query"]
    result = retriever.get_relevant_documents(query)
    state["rag_result"] = format_result(result)
    state["sources"] = [doc.metadata["source"] for doc in result]
    return state
```

## SEC Filings Ingested

| Company   | Filing | Period  |
|-----------|--------|---------|
| Apple     | 10-K   | FY 2023 |
| Microsoft | 10-K   | FY 2023 |
| Amazon    | 10-K   | FY 2023 |
| Alphabet  | 10-K   | FY 2023 |
| Meta      | 10-K   | FY 2023 |

## Connecting to the Orchestrator

```python
from agents.rag_agent import run
from state import AgentState

state: AgentState = {"query": "What is Apple's revenue for 2023?"}
state = run(state)

print(state["rag_result"])   # cited financial passages
print(state["sources"])      # ['Apple_2023_10K.htm', ...]
```
