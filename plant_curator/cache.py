import hashlib
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

CACHE_DIR = Path.home() / ".plant_curator"
CACHE_DB = CACHE_DIR / "cache.db"


def file_hash(path: Path, n_bytes: int = 256 * 1024) -> str:
    """Fast content-aware hash: file size + first N bytes. Robust to renames/moves."""
    size = path.stat().st_size
    h = hashlib.md5(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()


@dataclass
class CacheRow:
    sharpness: Optional[float]
    exposure: Optional[float]
    colorfulness: Optional[float]
    captured_at: Optional[datetime]
    embedding: Optional[np.ndarray]


@contextmanager
def _conn():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(CACHE_DB)
    c.execute("""
        CREATE TABLE IF NOT EXISTS photos (
            hash TEXT PRIMARY KEY,
            sharpness REAL,
            exposure REAL,
            colorfulness REAL,
            captured_at TEXT,
            embedding BLOB
        )
    """)
    try:
        yield c
        c.commit()
    finally:
        c.close()


def get(hashes: list[str]) -> dict[str, CacheRow]:
    if not hashes:
        return {}
    out: dict[str, CacheRow] = {}
    with _conn() as c:
        placeholders = ",".join("?" * len(hashes))
        for row in c.execute(
            f"SELECT hash, sharpness, exposure, colorfulness, captured_at, embedding "
            f"FROM photos WHERE hash IN ({placeholders})",
            hashes,
        ):
            h, sh, ex, co, ts, emb = row
            out[h] = CacheRow(
                sharpness=sh,
                exposure=ex,
                colorfulness=co,
                captured_at=datetime.fromisoformat(ts) if ts else None,
                embedding=np.frombuffer(emb, dtype=np.float32) if emb else None,
            )
    return out


def put(
    h: str,
    *,
    sharpness: Optional[float] = None,
    exposure: Optional[float] = None,
    colorfulness: Optional[float] = None,
    captured_at: Optional[datetime] = None,
    embedding: Optional[np.ndarray] = None,
) -> None:
    with _conn() as c:
        existing = c.execute(
            "SELECT sharpness, exposure, colorfulness, captured_at, embedding "
            "FROM photos WHERE hash = ?", (h,)
        ).fetchone()
        if existing:
            sh, ex, co, ts, emb = existing
            sharpness = sharpness if sharpness is not None else sh
            exposure = exposure if exposure is not None else ex
            colorfulness = colorfulness if colorfulness is not None else co
            captured_at = captured_at if captured_at is not None else (
                datetime.fromisoformat(ts) if ts else None
            )
            embedding = embedding if embedding is not None else (
                np.frombuffer(emb, dtype=np.float32) if emb else None
            )
        c.execute(
            "INSERT OR REPLACE INTO photos "
            "(hash, sharpness, exposure, colorfulness, captured_at, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                h,
                sharpness,
                exposure,
                colorfulness,
                captured_at.isoformat() if captured_at else None,
                embedding.astype(np.float32).tobytes() if embedding is not None else None,
            ),
        )
