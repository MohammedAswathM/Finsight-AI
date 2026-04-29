"""
retrieval/retriever.py
----------------------
Builds the two-layer retrieval stack used by the RAG agent:

  Layer 1 — ParentDocumentRetriever
      Matches small child chunks (200 tokens) embedded in Chroma,
      then surfaces the larger parent chunk (1 000 tokens) so the
      LLM gets enough surrounding context.

  Layer 2 — ContextualCompressionRetriever + LLMChainExtractor
      Wraps Layer 1. For each candidate parent chunk the extractor
      prompts an LLM to pull out only the sentences directly relevant
      to the query, compressing noise and improving answer quality.

      The extractor uses the shared Groq stack locally. If Groq cannot
      be loaded, compression is disabled and the parent retriever is returned.

Public API
----------
    build_retriever(vectorstore, docstore) -> BaseRetriever
    get_relevant_documents(query, retriever) -> List[Document]
"""

import logging
from typing import List, Optional

from langchain.schema import Document, BaseRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.storage import InMemoryStore

from retrieval.vectorstore import get_vectorstore
from retrieval.ingest import build_parent_document_retriever

logger = logging.getLogger(__name__)


# ── LLM for compression (free, local) ─────────────────────────────────────

def _get_compression_llm():
    """
    Returns a LangChain-compatible LLM for the extractor.

    Uses the shared Groq stack only. If Groq cannot be loaded, the
    compression layer is disabled and the parent retriever is returned.
    """
    try:
        from langchain_groq import ChatGroq

        logger.info("Using Groq ChatGroq for compression.")
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    except Exception as exc:
        logger.warning("Could not load Groq compression model: %s. Compression disabled.", exc)
        return None


# ── build full retriever ───────────────────────────────────────────────────

def build_retriever(
    vectorstore=None,
    docstore: Optional[InMemoryStore] = None,
    k: int = 5,
    use_compression: bool = True,
) -> BaseRetriever:
    """
    Construct and return the full retrieval pipeline.

    Parameters
    ----------
    vectorstore  : Chroma instance (loaded from disk if None)
    docstore     : InMemoryStore holding parent chunks (new store if None)
    k            : number of documents to retrieve
    use_compression : if True, wrap with ContextualCompressionRetriever

    Returns
    -------
    BaseRetriever (ContextualCompressionRetriever or ParentDocumentRetriever)
    """
    if vectorstore is None:
        vectorstore = get_vectorstore()

    if docstore is None:
        docstore = InMemoryStore()

    # Layer 1: ParentDocumentRetriever
    parent_retriever, _ = build_parent_document_retriever(vectorstore, docstore)
    parent_retriever.search_kwargs = {"k": k}
    logger.info("ParentDocumentRetriever built (k=%d).", k)

    if not use_compression:
        return parent_retriever

    # Layer 2: ContextualCompressionRetriever
    llm = _get_compression_llm()
    if llm is None:
        logger.warning("Skipping compression layer — returning base retriever.")
        return parent_retriever

    compressor  = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=parent_retriever,
    )
    logger.info("ContextualCompressionRetriever built.")
    return compression_retriever


# ── convenience wrapper ────────────────────────────────────────────────────

def get_relevant_documents(
    query: str,
    retriever: Optional[BaseRetriever] = None,
    **kwargs,
) -> List[Document]:
    """
    High-level helper.  Builds a retriever on-the-fly if none is supplied.

    Parameters
    ----------
    query     : natural-language query string
    retriever : pre-built retriever (optional, useful in tests)

    Returns
    -------
    List[Document] with metadata["source"] and metadata["company"] set.
    """
    if retriever is None:
        retriever = build_retriever(**kwargs)

    logger.info("Retrieving documents for query: '%s'", query[:80])
    docs = retriever.get_relevant_documents(query)
    logger.info("Retrieved %d document(s).", len(docs))
    return docs
