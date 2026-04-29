"""Spherical k-means on L2-normalized CLIP embeddings."""
import numpy as np


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
