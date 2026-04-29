"""
retrieval/ingest.py
-------------------
Download SEC 10-K filings and ingest them into ChromaDB using a
ParentDocumentRetriever strategy:

  • "parent" chunks  → 1 000-token windows stored in an InMemoryStore
  • "child"  chunks  → 200-token windows embedded & stored in Chroma

This lets the retriever surface small, precise matches while returning
the broader parent context to the LLM — a key technique for financial
documents where a single sentence often needs its surrounding paragraph.

Usage
-----
    python -m retrieval.ingest              # download + ingest all 10-Ks
    python -m retrieval.ingest --reset      # wipe DB first, then ingest
"""

import os
import sys
import logging
import argparse
import requests
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.retrievers import ParentDocumentRetriever

from retrieval.vectorstore import get_vectorstore, get_embeddings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
PDF_DIR   = BASE_DIR / "data" / "pdfs"

# ── SEC EDGAR direct links for 5 recent 10-K filings ──────────────────────
# Apple, Microsoft, Amazon, Alphabet, Meta — all filed 2024
SEC_10K_URLS: List[dict] = [
    {
        "company": "Apple_Inc",
        "url": (
            "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/"
            "aapl20230930.htm"
        ),
        # Fallback: use a freely available EDGAR full-text search result
        "pdf_fallback": None,
    },
]

# Better approach: use EDGAR full-text search API to get actual PDF accessions
EDGAR_SEARCH_URL = (
    "https://efts.sec.gov/LATEST/search-index"
    "?q=%2210-K%22&dateRange=custom&startdt=2024-01-01"
    "&forms=10-K&_source=filing_index"
)

# Curated direct PDF URLs from SEC EDGAR (verified accessible, no auth needed)
FILINGS: List[dict] = [
    {
        "company": "Apple_2023",
        "url": (
            "https://www.sec.gov/Archives/edgar/data/320193/"
            "000032019323000106/aapl20230930.htm"
        ),
    },
    {
        "company": "Microsoft_2023",
        "url": (
            "https://www.sec.gov/Archives/edgar/data/789019/"
            "000095017023035122/msft-20230630.htm"
        ),
    },
    {
        "company": "Amazon_2023",
        "url": (
            "https://www.sec.gov/Archives/edgar/data/1018724/"
            "000101872424000008/amzn-20231231.htm"
        ),
    },
    {
        "company": "Alphabet_2023",
        "url": (
            "https://www.sec.gov/Archives/edgar/data/1652044/"
            "000165204424000022/goog-20231231.htm"
        ),
    },
    {
        "company": "Meta_2023",
        "url": (
            "https://www.sec.gov/Archives/edgar/data/1326801/"
            "000132680124000012/meta-20231231.htm"
        ),
    },
]

# ── splitter settings ──────────────────────────────────────────────────────
PARENT_CHUNK_SIZE  = 1_000   # tokens approximated by chars/4
PARENT_CHUNK_OVERLAP = 100
CHILD_CHUNK_SIZE   = 200
CHILD_CHUNK_OVERLAP = 20


def _download_htm_as_text(url: str, dest_path: Path) -> bool:
    """Download an SEC HTM filing and save as plain-text (.txt)."""
    headers = {
        "User-Agent": "FinSight-AI research@finsight.ai",   # EDGAR requires this
        "Accept-Encoding": "gzip, deflate",
    }
    try:
        r = requests.get(url, headers=headers, timeout=60)
        r.raise_for_status()
        dest_path.write_bytes(r.content)
        logger.info("Downloaded %s → %s", url, dest_path.name)
        return True
    except Exception as exc:
        logger.warning("Could not download %s: %s", url, exc)
        return False


def download_filings(pdf_dir: Path = PDF_DIR) -> List[Path]:
    """
    Download SEC 10-K HTML filings to *pdf_dir*.
    Returns list of successfully downloaded file paths.
    """
    pdf_dir.mkdir(parents=True, exist_ok=True)
    downloaded: List[Path] = []

    for filing in FILINGS:
        dest = pdf_dir / f"{filing['company']}_10K.htm"
        if dest.exists():
            logger.info("Already exists, skipping: %s", dest.name)
            downloaded.append(dest)
            continue
        if _download_htm_as_text(filing["url"], dest):
            downloaded.append(dest)

    logger.info("Downloaded %d / %d filings.", len(downloaded), len(FILINGS))
    return downloaded


def load_documents(file_paths: List[Path]):
    """
    Load SEC filings using the appropriate LangChain loader.
    HTM/HTML files are loaded with BSHTMLLoader; PDFs with PyPDFLoader.
    """
    from langchain_community.document_loaders import BSHTMLLoader, TextLoader

    all_docs = []
    for fp in file_paths:
        try:
            suffix = fp.suffix.lower()
            if suffix in {".htm", ".html"}:
                loader = BSHTMLLoader(str(fp), open_encoding="utf-8")
            elif suffix == ".pdf":
                loader = PyPDFLoader(str(fp))
            else:
                loader = TextLoader(str(fp), encoding="utf-8")

            docs = loader.load()
            # Tag every chunk with the source company name
            company = fp.stem.replace("_10K", "")
            for doc in docs:
                doc.metadata["source"]  = fp.name
                doc.metadata["company"] = company
                doc.metadata["filing"]  = "10-K"
            all_docs.extend(docs)
            logger.info("Loaded %d pages from %s", len(docs), fp.name)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", fp, exc)

    logger.info("Total document pages loaded: %d", len(all_docs))
    return all_docs


def build_parent_document_retriever(vectorstore, docstore=None):
    """
    Construct a ParentDocumentRetriever.

    Child splitter  → small chunks embedded in Chroma
    Parent splitter → larger chunks stored in docstore for full context
    """
    if docstore is None:
        docstore = LocalFileStore(BASE_DIR / "docstore")

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    return retriever, docstore


def ingest(reset: bool = False) -> tuple:
    """
    Full ingestion pipeline:
      1. Download 10-K filings
      2. Load & parse documents
      3. Split into parent/child chunks
      4. Embed child chunks → Chroma
      5. Return (retriever, docstore)
    """
    vectorstore = get_vectorstore(reset=reset)
    file_paths  = download_filings()

    if not file_paths:
        logger.error("No filings downloaded — cannot ingest.")
        return None, None

    documents = load_documents(file_paths)
    if not documents:
        logger.error("No documents parsed — check file format.")
        return None, None

    retriever, docstore = build_parent_document_retriever(vectorstore)

    logger.info("Adding %d document pages to ParentDocumentRetriever…", len(documents))
    # Add documents one by one to avoid ChromaDB batch size limits
    for i, doc in enumerate(documents):
        retriever.add_documents([doc], ids=None)
        if (i + 1) % 10 == 0:
            logger.info("Added %d/%d documents", i + 1, len(documents))
    logger.info("Ingestion complete.")

    return retriever, docstore


# ── CLI entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest SEC 10-K filings into ChromaDB")
    parser.add_argument("--reset", action="store_true", help="Wipe the DB before ingesting")
    args = parser.parse_args()
    ingest(reset=args.reset)
