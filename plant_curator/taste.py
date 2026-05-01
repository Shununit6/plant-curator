"""Compute a 'taste vector' from liked photo embeddings.

The taste vector is the L2-normalized centroid of all liked CLIP
embeddings. New photos can be scored by cosine similarity to it,
giving a soft 'how aligned with the user's taste' score.
"""
from typing import Optional

import numpy as np

from . import cache as cache_mod

MIN_EXAMPLES = 5  # below this, the centroid is too noisy to trust


def compute_taste() -> Optional[np.ndarray]:
    """Discriminative taste direction.

    With both liked and disliked examples, returns the L2-normalized
    direction (mean_liked - mean_disliked) — the axis that maximally
    separates the two classes under a Gaussian-shared-covariance assumption.
    With only liked examples, falls back to the centroid of likes.
    """
    liked = cache_mod.get_liked_embeddings()
    if len(liked) < MIN_EXAMPLES:
        return None
    mu_liked = liked.mean(axis=0)

    disliked = cache_mod.get_disliked_embeddings()
    if len(disliked) >= MIN_EXAMPLES:
        mu_disliked = disliked.mean(axis=0)
        v = mu_liked - mu_disliked
    else:
        v = mu_liked

    norm = float(np.linalg.norm(v))
    if norm == 0:
        return None
    return (v / norm).astype(np.float32)


def aesthetic_scores(embeddings: np.ndarray, taste: np.ndarray) -> np.ndarray:
    """Cosine similarity of each embedding against the taste vector.
    Returns an (N,) array shifted to roughly [0, 1] by mapping sim from [-1,1]."""
    sims = embeddings @ taste
    return (sims + 1.0) / 2.0
