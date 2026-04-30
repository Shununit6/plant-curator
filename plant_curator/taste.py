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
    """Returns L2-normalized centroid of liked embeddings, or None if too few."""
    embeds = cache_mod.get_liked_embeddings()
    if len(embeds) < MIN_EXAMPLES:
        return None
    centroid = embeds.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm == 0:
        return None
    return (centroid / norm).astype(np.float32)


def aesthetic_scores(embeddings: np.ndarray, taste: np.ndarray) -> np.ndarray:
    """Cosine similarity of each embedding against the taste vector.
    Returns an (N,) array shifted to roughly [0, 1] by mapping sim from [-1,1]."""
    sims = embeddings @ taste
    return (sims + 1.0) / 2.0
