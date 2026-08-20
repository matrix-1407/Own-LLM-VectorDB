"""
vectordb/persistence.py
=======================
Disk persistence for VectorDB and DocumentDB.

Saves and loads data as JSON snapshots so the database survives server
restarts.  Index structures (HNSW / KD-Tree) are NOT serialized — they
are cheaply rebuilt from the stored vectors during load, which keeps
the snapshot format simple and version-agnostic.

Usage (called automatically by app.py on startup and shutdown)::

    from vectordb.persistence import save_vector_db, load_vector_db
    from vectordb.persistence import save_document_db, load_document_db

    save_vector_db(db, Path("data/vectordb_index.json"))
    load_vector_db(db, Path("data/vectordb_index.json"))

    save_document_db(doc_db, Path("data/document_index.json"))
    load_document_db(doc_db, Path("data/document_index.json"))
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectordb.vector_db import VectorDB
    from vectordb.document_db import DocumentDB

logger = logging.getLogger(__name__)

# ── Internal helpers ──────────────────────────────────────────────────────────

def _atomic_write(path: Path, data: str) -> None:
    """Write *data* to *path* atomically via a temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


# ══════════════════════════════════════════════════════════════════════════════
# VectorDB (16-D demo index)
# ══════════════════════════════════════════════════════════════════════════════

def save_vector_db(db: "VectorDB", path: Path) -> int:
    """
    Serialise all demo vectors to *path* as a JSON file.

    Returns the number of items saved.
    """
    items = db.all_items()
    payload = {
        "version": 1,
        "dims": db.dims,
        "items": [
            {
                "id": item.id,
                "metadata": item.metadata,
                "category": item.category,
                "emb": item.emb,
            }
            for item in items
        ],
    }
    _atomic_write(path, json.dumps(payload, ensure_ascii=False))
    logger.info("[Persistence] VectorDB saved: %d items → %s", len(items), path)
    return len(items)


def load_vector_db(db: "VectorDB", path: Path) -> int:
    """
    Restore demo vectors from *path*.  The HNSW / KD-Tree / BruteForce indexes
    are fully rebuilt from the stored embeddings.

    Returns the number of items loaded, or 0 if the file does not exist.
    """
    if not path.exists():
        logger.info("[Persistence] No VectorDB snapshot found at %s — starting fresh.", path)
        return 0

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("[Persistence] Could not read VectorDB snapshot (%s) — skipping.", exc)
        return 0

    from vectordb.metrics import cosine

    count = 0
    for item in payload.get("items", []):
        try:
            db.insert(
                metadata=item["metadata"],
                category=item["category"],
                emb=item["emb"],
                dist_fn=cosine,
            )
            count += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Persistence] Skipping corrupt VectorDB item: %s", exc)

    logger.info("[Persistence] VectorDB loaded: %d items from %s", count, path)
    return count


# ══════════════════════════════════════════════════════════════════════════════
# DocumentDB (768-D RAG index)
# ══════════════════════════════════════════════════════════════════════════════

def save_document_db(doc_db: "DocumentDB", path: Path) -> int:
    """
    Serialise all document chunks to *path* as a JSON file.

    Returns the number of chunks saved.
    """
    items = doc_db.all_items()
    payload = {
        "version": 1,
        "dims": doc_db.get_dims(),
        "items": [
            {
                "id": item.id,
                "title": item.title,
                "text": item.text,
                "emb": item.emb,
            }
            for item in items
        ],
    }
    _atomic_write(path, json.dumps(payload, ensure_ascii=False))
    logger.info("[Persistence] DocumentDB saved: %d chunks → %s", len(items), path)
    return len(items)


def load_document_db(doc_db: "DocumentDB", path: Path) -> int:
    """
    Restore document chunks from *path*.  HNSW and BM25 indexes are fully
    rebuilt from the stored text and embeddings.

    Returns the number of chunks loaded, or 0 if the file does not exist.
    """
    if not path.exists():
        logger.info("[Persistence] No DocumentDB snapshot found at %s — starting fresh.", path)
        return 0

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("[Persistence] Could not read DocumentDB snapshot (%s) — skipping.", exc)
        return 0

    count = 0
    for item in payload.get("items", []):
        try:
            doc_db.insert(
                title=item["title"],
                text=item["text"],
                emb=item["emb"],
            )
            count += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Persistence] Skipping corrupt DocumentDB chunk: %s", exc)

    logger.info("[Persistence] DocumentDB loaded: %d chunks from %s", count, path)
    return count


# ══════════════════════════════════════════════════════════════════════════════
# Snapshot metadata helper (for /persist/status API)
# ══════════════════════════════════════════════════════════════════════════════

def snapshot_info(path: Path) -> dict:
    """Return file-level metadata dict for the given snapshot path."""
    exists = path.exists()
    return {
        "path": str(path),
        "exists": exists,
        "sizeBytes": path.stat().st_size if exists else 0,
    }
