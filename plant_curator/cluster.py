"""Clustering and selection on L2-normalized CLIP embeddings."""
from datetime import datetime
from typing import List, Sequence, Tuple

import numpy as np


def collapse_bursts(
    indexed: Sequence[Tuple[int, datetime, np.ndarray]],
    gap_seconds: float = 90,
    sim_threshold: float = 0.75,
) -> List[List[int]]:
    """Group photos that are close in time AND visually similar.

    indexed: sequence of (original_index, captured_at, embedding).
    Returns a list of groups, each a list of original indices.
    Photos enter the active group if they are within gap_seconds of the
    last photo and have cosine similarity >= sim_threshold to it.
    """
    if not indexed:
        return []
    items = sorted(indexed, key=lambda r: r[1])
    groups: List[List[int]] = [[items[0][0]]]
    last_t = items[0][1]
    last_emb = items[0][2]
    for idx, ts, emb in items[1:]:
        gap = (ts - last_t).total_seconds()
        sim = float(emb @ last_emb)
        if gap <= gap_seconds and sim >= sim_threshold:
            groups[-1].append(idx)
        else:
            groups.append([idx])
        last_t, last_emb = ts, emb
    return groups


def select_mmr(
    embeds: np.ndarray,
    scores: np.ndarray,
    n: int,
    lam: float = 0.4,
) -> List[int]:
    """Greedy MMR (Maximal Marginal Relevance) selection.

    Returns indices of n picks that balance high scores (lam) with being
    different from already-picked items (1 - lam). Embeddings must be L2-normalized.
    """
    m = len(scores)
    if n >= m:
        return list(range(m))

    # Min-max normalize scores into [0, 1] so they're comparable to (1 - cos_sim).
    s_min, s_max = float(scores.min()), float(scores.max())
    s_norm = (scores - s_min) / (s_max - s_min) if s_max > s_min else np.ones_like(scores)

    selected: List[int] = [int(np.argmax(s_norm))]
    while len(selected) < n:
        sel_arr = embeds[np.array(selected)]
        sims = embeds @ sel_arr.T
        max_sims = sims.max(axis=1)
        novelty = 1.0 - max_sims
        mmr = lam * s_norm + (1 - lam) * novelty
        for i in selected:
            mmr[i] = -np.inf
        selected.append(int(np.argmax(mmr)))
    return selected


def kmeans(X: np.ndarray, k: int, n_iter: int = 30, seed: int = 42) -> np.ndarray:
    """Cluster N x D unit-norm vectors into k groups by cosine similarity.
    Returns an array of length N with cluster index per row."""
    n = len(X)
    if k >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False)
    centroids = X[idx].copy()
    labels = np.zeros(n, dtype=np.int32)

    for _ in range(n_iter):
        sims = X @ centroids.T
        new_labels = sims.argmax(axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        new_centroids = centroids.copy()
        for j in range(k):
            mask = labels == j
            if not mask.any():
                continue
            m = X[mask].mean(axis=0)
            norm = np.linalg.norm(m)
            if norm > 0:
                new_centroids[j] = m / norm
        centroids = new_centroids

    return labels
