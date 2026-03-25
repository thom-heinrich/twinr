"""Shared centroid helpers for routing bundle builders."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .contracts import ROUTE_LABEL_VALUES

def compute_label_centroids(
    embeddings: np.ndarray,
    samples: Sequence[object],
    *,
    labels: Sequence[str] = ROUTE_LABEL_VALUES,
) -> np.ndarray:
    """Return one centroid per label in the requested order."""

    if embeddings.shape[0] != len(samples):
        raise ValueError("Embedding rows must match the number of labeled samples.")
    rows: list[np.ndarray] = []
    for label in labels:
        matching_rows = [
            embeddings[index]
            for index, sample in enumerate(samples)
            if str(getattr(sample, "label", "")) == label
        ]
        if not matching_rows:
            raise ValueError(f"Cannot build router bundle without training samples for {label!r}.")
        rows.append(np.mean(np.vstack(matching_rows), axis=0))
    return np.vstack(rows).astype(np.float32)
