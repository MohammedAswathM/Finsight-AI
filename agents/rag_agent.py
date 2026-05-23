"""
agents/rag_agent.py
-------------------
FinSight AI — RAG Agent

Responsible for:
  • Loading the ChromaDB vector store (or triggering ingestion on first run)
  • Running the two-layer retrieval pipeline
  • Formatting cited financial excerpts for the orchestrator

Required signature (non-negotiable per project spec):
    from state import AgentState
    def run(state: AgentState) -> AgentState

The agent is designed to be called by the LangGraph orchestrator but can
also be tested standalone (see __main__ block at the bottom).
"""

from __future__ import annotations

import logging
import sys
import os
import re
from pathlib import Path
from typing import List, Optional

from langchain.storage import LocalFileStore, create_kv_docstore
from langchain.schema import Document

# ── make sure project root is on the path when running standalone ──────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR = PROJECT_ROOT

from state import AgentState  # noqa: E402
from retrieval.vectorstore import get_vectorstore  # noqa: E402
from retrieval.retriever import build_retriever, get_relevant_documents  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ── module-level singletons (initialised lazily on first call) ─────────────
_retriever = None
_docstore  = None
_vectorstore = None

_COMPANY_ALIASES = {
    "apple": "Apple_2024",
    "aapl": "Apple_2024",
    "microsoft": "Microsoft_2023",
    "msft": "Microsoft_2023",
    "amazon": "Amazon_2023",
    "amzn": "Amazon_2023",
    "alphabet": "Alphabet_2023",
    "google": "Alphabet_2023",
    "googl": "Alphabet_2023",
    "meta": "Meta_2023",
    "facebook": "Meta_2023",
}


def _company_filter_from_query(query: str) -> Optional[str]:
    lowered = query.lower()
    for alias, company in _COMPANY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            return company
    return None


def _ensure_retriever():
    """
    Initialise the retriever singleton.
    If no documents are in the vector store, trigger ingestion automatically.
    """
    global _retriever, _docstore, _vectorstore

    if _retriever is not None:
        return _retriever

    vectorstore = get_vectorstore()
    _vectorstore = vectorstore

    # Check whether anything has been ingested yet
    try:
        count = vectorstore._collection.count()
    except Exception:
        count = 0

    if count == 0:
        logger.info(
            "Vector store is empty — running ingestion pipeline automatically…"
        )
        from retrieval.ingest import ingest
        retriever_obj, docstore_obj = ingest(reset=False)
        if retriever_obj is None:
            raise RuntimeError(
                "Ingestion failed — no documents were loaded. "
                "Check your network connection to SEC EDGAR."
            )
        _retriever = retriever_obj
        _docstore  = docstore_obj
    else:
        logger.info("Vector store contains %d chunks — skipping ingestion.", count)
        _docstore  = create_kv_docstore(LocalFileStore(str(BASE_DIR / "docstore")))
        _retriever = build_retriever(
            vectorstore=vectorstore,
            docstore=_docstore,
        )

    return _retriever


def _filter_company_docs(docs, company: Optional[str]):
    if not company:
        return docs
    return [doc for doc in docs if doc.metadata.get("company") == company]


def _fallback_company_search(query: str, company: Optional[str], k: int = 5):
    if not company or _vectorstore is None:
        return []


def _lexical_company_search(query: str, company: Optional[str]):
    """Small deterministic supplement for exact financial table labels."""
    if not company:
        return []
    lowered = query.lower()
    if not any(term in lowered for term in ("total revenue", "total revenues", "net sales")):
        return []

    filing_path = BASE_DIR / "data" / "pdfs" / f"{company}_10K.htm"
    if not filing_path.exists():
        return []

    try:
        from bs4 import BeautifulSoup

        text = BeautifulSoup(filing_path.read_text(encoding="utf-8", errors="ignore"), "lxml").get_text("\n")
        normalized = re.sub(r"\n{2,}", "\n", text)
        patterns = ["Total revenue", "Total net sales", "Net sales"]
        for pattern in patterns:
            if pattern == "Total revenue":
                index = normalized.lower().rfind(pattern.lower())
            else:
                index = normalized.lower().find(pattern.lower())
            if index < 0:
                continue
            start = max(0, index - 450)
            end = min(len(normalized), index + 750)
            return [
                Document(
                    page_content=normalized[start:end].strip(),
                    metadata={
                        "source": filing_path.name,
                        "company": company,
                        "filing": "10-K",
                    },
                )
            ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Lexical filing search failed: %s", exc)
    return []
    try:
        return _vectorstore.similarity_search(query, k=k, filter={"company": company})
    except Exception as exc:  # noqa: BLE001
        logger.warning("Company-filtered vector search failed: %s", exc)
        return []


# ── formatting helper ──────────────────────────────────────────────────────

def format_result(docs) -> str:
    """
    Convert a list of retrieved LangChain Documents into a readable,
    cited string suitable for passing to an LLM orchestrator.

    Format per chunk:
        [SOURCE: <filename> | COMPANY: <name>]
        <document text>
        ---
    """
    if not docs:
        return "No relevant financial information found for the given query."

    parts: List[str] = []
    for i, doc in enumerate(docs, start=1):
        source  = doc.metadata.get("source",  "unknown")
        company = doc.metadata.get("company", "unknown")
        page    = doc.metadata.get("page",    "")
        page_str = f" | PAGE: {page}" if page != "" else ""

        header = f"[{i}] SOURCE: {source} | COMPANY: {company}{page_str}"
        parts.append(f"{header}\n{doc.page_content.strip()}")

    return "\n\n---\n\n".join(parts)


# ── required agent entry-point ─────────────────────────────────────────────

def run(state: AgentState) -> AgentState:
    """
    RAG agent entry point — called by the LangGraph orchestrator.

    Reads
    -----
    state["query"] : str   Natural-language financial question

    Writes
    ------
    state["rag_result"] : str         Formatted cited passages
    state["sources"]    : List[str]   Unique source filenames
    state["trace_log"]  : List[str]   Appends execution trace
    """
    query = state.get("query", "").strip()
    if not query:
        return {
            "rag_result": "No query provided to RAG agent.",
            "sources": [],
            "trace_log": ["RAG agent: no query provided"],
        }

    logger.info("RAG agent — query: '%s'", query[:120])

    try:
        retriever = _ensure_retriever()
        company = _company_filter_from_query(query)
        docs = get_relevant_documents(query, retriever=retriever)
        filtered_docs = _filter_company_docs(docs, company)
        if company and len(filtered_docs) < min(2, len(docs)):
            filtered_docs = _fallback_company_search(query, company, k=5)
        docs = filtered_docs if company else docs
        lexical_docs = _lexical_company_search(query, company)
        if lexical_docs:
            seen_sources = {doc.page_content for doc in lexical_docs}
            docs = lexical_docs + [doc for doc in docs if doc.page_content not in seen_sources]
            docs = docs[:5]
        return {
            "rag_result": format_result(docs),
            "sources": [doc.metadata.get("source", "unknown") for doc in docs],
            "trace_log": [f"RAG agent: retrieved {len(docs)} chunks from vector store"],
        }
    except Exception as exc:
        return {
            "rag_result": f"RAG retrieval failed: {str(exc)}",
            "sources": [],
            "trace_log": [f"RAG agent: ERROR — {str(exc)}"],
        }


# ── standalone test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    test_queries = [
        "What are Apple's total revenues for fiscal year 2023?",
        "Describe Microsoft's cloud segment operating income.",
        "What risk factors does Amazon mention related to competition?",
        "How does Alphabet report its advertising revenue?",
        "What are Meta's capital expenditure plans?",
    ]

    print("=" * 70)
    print("FinSight AI — RAG Agent Standalone Test")
    print("=" * 70)

    for q in test_queries[:2]:   # run 2 queries to keep test fast
        print(f"\nQUERY: {q}")
        print("-" * 60)
        test_state: AgentState = {"query": q}
        result_state = run(test_state)

        print("RAG RESULT (first 500 chars):")
        print(result_state["rag_result"][:500])
        print("\nSOURCES:")
        for src in result_state.get("sources", []):
            print(f"  • {src}")
        print("=" * 70)
