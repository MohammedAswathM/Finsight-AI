"""
retrieval/vectorstore.py
------------------------
Initialises (or loads) a persistent ChromaDB vector store using
sentence-transformers embeddings (100 % free, no API key required).

Usage
-----
    from retrieval.vectorstore import get_vectorstore
    vs = get_vectorstore()          # load existing
    vs = get_vectorstore(reset=True) # wipe & rebuild
"""

import os
import logging
from pathlib import Path

import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
CHROMA_DIR   = BASE_DIR / "chroma_db"
COLLECTION   = "finsight_10k"

# ── embedding model (runs locally, ~90 MB download on first use) ───────────
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFace embedding instance."""
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore(reset: bool = False) -> Chroma:
    """
    Return the persistent Chroma vector store.

    Parameters
    ----------
    reset : bool
        If True, delete the existing collection before returning so that
        fresh documents can be ingested from scratch.
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = _get_embeddings()

    if reset:
        logger.info("Resetting ChromaDB collection '%s'…", COLLECTION)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            client.delete_collection(COLLECTION)
            logger.info("Collection deleted.")
        except Exception:
            pass  # collection may not exist yet

    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    logger.info(
        "ChromaDB vectorstore ready (collection='%s', path='%s').",
        COLLECTION,
        CHROMA_DIR,
    )
    return vectorstore


def get_embeddings() -> HuggingFaceEmbeddings:
    """Public accessor for the embedding model (used by ingest & retriever)."""
    return _get_embeddings()
