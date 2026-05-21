"""Environment loader. Every module imports keys from here, never os.getenv directly."""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
GROQ_API_KEYS: str | None = os.getenv("GROQ_API_KEYS")
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY") or None
NEWSAPI_KEY: str | None = os.getenv("NEWSAPI_KEY") or None

# Leave blank to use MLflow's local file store (./mlruns) — recommended default.
# Set to http://127.0.0.1:5000 only if you're running `mlflow server` separately.
MLFLOW_TRACKING_URI: str | None = os.getenv("MLFLOW_TRACKING_URI") or None
CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")

GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def get_groq_keys() -> list[str]:
    """Return configured Groq keys in failover order.

    Supports either:
    - GROQ_API_KEYS=key1,key2,key3
    - GROQ_API_KEY=key1 plus optional GROQ_API_KEY_2, GROQ_API_KEY_3, ...
    """
    keys: list[str] = []
    if GROQ_API_KEYS:
        keys.extend(key.strip() for key in GROQ_API_KEYS.split(",") if key.strip())
    if GROQ_API_KEY:
        keys.append(GROQ_API_KEY.strip())

    index = 2
    while True:
        key = os.getenv(f"GROQ_API_KEY_{index}")
        if not key:
            break
        keys.append(key.strip())
        index += 1

    deduped: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key and key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped


def require_groq() -> str:
    keys = get_groq_keys()
    if not keys:
        raise RuntimeError(
            "No Groq API key set. Copy .env.example to .env and add GROQ_API_KEY "
            "or GROQ_API_KEYS from https://console.groq.com/keys"
        )
    return keys[0]
