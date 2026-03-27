"""Shared centroid helpers for routing bundle builders."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .contracts import ROUTE_LABEL_VALUES

# CHANGELOG: 2026-03-27
# BUG-1: Reject malformed, non-2D, zero-width, and non-finite embeddings instead of silently
#        emitting corrupted centroids or cryptic downstream failures.
# BUG-2: Compute centroids with float64 accumulation to avoid avoidable float32 drift, then
#        cast back to float32 for compact bundle storage.
# SEC-1: Harden router-bundle construction against malformed / poisoned numeric inputs via
#        strict numeric, label, and weight validation.
# IMP-1: Replace repeated Python label scans with vectorized one-pass accumulation for lower CPU
#        overhead on Raspberry Pi 4.
# IMP-2: # BREAKING: Default to cosine-space prototype building (L2-normalize inputs and
#        centroids). Added optional sample weights and a robust median aggregation mode.

_EPSILON = 1e-12


def _normalize_label_names(labels: Sequence[object]) -> list[str]:
    normalized = [str(label) for label in labels]
    if not normalized:
        raise ValueError("At least one route label is required to build centroids.")

    seen: set[str] = set()
    duplicates: set[str] = set()
    for label in normalized:
        if label in seen:
            duplicates.add(label)
        else:
            seen.add(label)

    if duplicates:
        duplicate_list = ", ".join(repr(label) for label in sorted(duplicates))
        raise ValueError(f"Route labels must be unique. Duplicates: {duplicate_list}.")

    return normalized


def _as_real_2d_array(embeddings: np.ndarray) -> np.ndarray:
    array = np.asarray(embeddings)

    if array.ndim != 2:
        raise ValueError(
            "Embeddings must be a 2D real-valued array of shape [n_samples, embedding_dim]."
        )
    if array.shape[1] == 0:
        raise ValueError("Embeddings must have a non-zero embedding dimension.")
    if np.iscomplexobj(array):
        raise ValueError("Embeddings must be real-valued.")
    if array.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("Embeddings must be a real-valued numeric array.")
    if not np.isfinite(array).all():
        raise ValueError("Embeddings must contain only finite numeric values.")

    return np.ascontiguousarray(array, dtype=np.float64)


def _validate_sample_weights(
    sample_weights: Sequence[float] | np.ndarray | None,
    *,
    n_samples: int,
) -> np.ndarray:
    if sample_weights is None:
        return np.ones(n_samples, dtype=np.float64)

    weights = np.asarray(sample_weights)
    if weights.ndim != 1 or weights.shape[0] != n_samples:
        raise ValueError("Sample weights must be a 1D array with one value per sample.")
    if np.iscomplexobj(weights):
        raise ValueError("Sample weights must be real-valued.")
    if weights.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("Sample weights must be numeric.")
    if not np.isfinite(weights).all():
        raise ValueError("Sample weights must contain only finite numeric values.")

    weights = weights.astype(np.float64, copy=False)
    if np.any(weights < 0.0):
        raise ValueError("Sample weights must be non-negative.")

    return weights


def _l2_normalize_rows(matrix: np.ndarray, *, name: str) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    zero_mask = norms[:, 0] <= _EPSILON
    if np.any(zero_mask):
        first_bad = ", ".join(str(index) for index in np.flatnonzero(zero_mask)[:5])
        suffix = "..." if int(zero_mask.sum()) > 5 else ""
        raise ValueError(f"{name} contains zero-norm row(s) at index {first_bad}{suffix}.")
    return matrix / norms


def compute_label_centroids(
    embeddings: np.ndarray,
    samples: Sequence[object],
    *,
    labels: Sequence[str] = ROUTE_LABEL_VALUES,
    aggregation: str = "mean",
    # BREAKING: Default switched to True so router prototypes are built in cosine space.
    # This matches the dominant 2026 embedding-router setup (normalized embeddings +
    # cosine/dot-product scoring). Use normalize=False to recover the legacy raw mean.
    normalize: bool = True,
    sample_weights: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Return one centroid per label in the requested order.

    Parameters
    ----------
    embeddings:
        2D array with shape ``[n_samples, embedding_dim]``.
    samples:
        Sequence of training samples. Each sample must expose a ``label`` attribute.
    labels:
        Route labels to emit, in output order. Labels must be unique.
    aggregation:
        ``"mean"`` for the standard centroid, or ``"median"`` for a more robust
        feature-wise prototype when labels are noisy. ``sample_weights`` are only
        supported with ``aggregation="mean"``.
    normalize:
        When ``True``, L2-normalize input embeddings before aggregation and
        L2-normalize the resulting centroids.
    sample_weights:
        Optional non-negative per-sample weights for confidence-weighted centroids.

    Returns
    -------
    np.ndarray
        Float32 centroid matrix with shape ``[len(labels), embedding_dim]``.
    """

    label_names = _normalize_label_names(labels)
    embeddings_2d = _as_real_2d_array(embeddings)

    n_samples = len(samples)
    if embeddings_2d.shape[0] != n_samples:
        raise ValueError("Embedding rows must match the number of labeled samples.")
    if n_samples == 0:
        raise ValueError("Cannot build router bundle without labeled samples.")

    aggregation = aggregation.lower().strip()
    if aggregation not in {"mean", "median"}:
        raise ValueError("aggregation must be either 'mean' or 'median'.")
    if aggregation == "median" and sample_weights is not None:
        raise ValueError("sample_weights are only supported with aggregation='mean'.")

    weights = _validate_sample_weights(sample_weights, n_samples=n_samples)
    sample_labels = np.fromiter(
        (str(getattr(sample, "label", "")) for sample in samples),
        dtype=object,
        count=n_samples,
    )

    working = embeddings_2d
    if normalize:
        working = _l2_normalize_rows(working, name="Embeddings")

    label_to_index = {label: index for index, label in enumerate(label_names)}

    if aggregation == "mean":
        matched_mask = np.fromiter(
            (label in label_to_index for label in sample_labels),
            dtype=bool,
            count=n_samples,
        )
        matched_count = int(matched_mask.sum())

        sums = np.zeros((len(label_names), working.shape[1]), dtype=np.float64)
        weight_sums = np.zeros(len(label_names), dtype=np.float64)

        if matched_count:
            matched_labels = sample_labels[matched_mask]
            target_indices = np.fromiter(
                (label_to_index[label] for label in matched_labels),
                dtype=np.int64,
                count=matched_count,
            )
            matched_weights = weights[matched_mask]
            np.add.at(sums, target_indices, working[matched_mask] * matched_weights[:, None])
            np.add.at(weight_sums, target_indices, matched_weights)

        missing_labels = [
            label_names[index] for index, total_weight in enumerate(weight_sums) if total_weight <= 0.0
        ]
        if missing_labels:
            if len(missing_labels) == 1:
                raise ValueError(
                    f"Cannot build router bundle without training samples for {missing_labels[0]!r}."
                )
            missing = ", ".join(repr(label) for label in missing_labels)
            raise ValueError(
                "Cannot build router bundle without training samples for all requested labels. "
                f"Missing: {missing}."
            )

        centroids = sums / weight_sums[:, None]
    else:
        rows: list[np.ndarray] = []
        for label in label_names:
            label_rows = working[sample_labels == label]
            if label_rows.shape[0] == 0:
                raise ValueError(f"Cannot build router bundle without training samples for {label!r}.")
            rows.append(np.median(label_rows, axis=0))
        centroids = np.vstack(rows)

    if not np.isfinite(centroids).all():
        raise ValueError("Centroid computation produced non-finite values.")
    if normalize:
        centroids = _l2_normalize_rows(centroids, name="Centroids")

    return np.ascontiguousarray(centroids.astype(np.float32, copy=False))