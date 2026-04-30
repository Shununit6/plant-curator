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
            embedding BLOB,
            liked INTEGER NOT NULL DEFAULT 0
        )
    """)
    try:
        c.execute("ALTER TABLE photos ADD COLUMN liked INTEGER NOT NULL DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # column already exists
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


def set_liked(h: str, liked: bool = True) -> None:
    with _conn() as c:
        existing = c.execute("SELECT 1 FROM photos WHERE hash = ?", (h,)).fetchone()
        if existing:
            c.execute("UPDATE photos SET liked = ? WHERE hash = ?", (1 if liked else 0, h))
        else:
            c.execute("INSERT INTO photos (hash, liked) VALUES (?, ?)",
                      (h, 1 if liked else 0))


def get_liked_embeddings() -> np.ndarray:
    """Return all liked embeddings as an (N, D) array. Empty if none."""
    with _conn() as c:
        rows = c.execute(
            "SELECT embedding FROM photos WHERE liked = 1 AND embedding IS NOT NULL"
        ).fetchall()
    if not rows:
        return np.empty((0, 512), dtype=np.float32)
    return np.stack([np.frombuffer(r[0], dtype=np.float32) for r in rows])


def count_liked() -> int:
    with _conn() as c:
        return c.execute("SELECT COUNT(*) FROM photos WHERE liked = 1").fetchone()[0]


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
