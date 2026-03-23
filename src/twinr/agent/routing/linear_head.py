"""Train compact linear classifier heads over frozen transcript embeddings.

This module keeps the discriminative classifier-head training logic out of the
runtime router services. The runtime still uses the same ONNX encoder on the
Pi; only the lightweight label head changes from centroid similarity to a
trained linear softmax head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class LinearHeadTrainingResult:
    """Store one trained multiclass linear head plus selection metadata."""

    weights: np.ndarray
    bias: np.ndarray
    inverse_regularization: float
    score: float


def fit_multiclass_linear_head(
    embeddings: np.ndarray,
    labels: Sequence[str],
    *,
    label_order: Sequence[str],
    dev_embeddings: np.ndarray | None = None,
    dev_labels: Sequence[str] | None = None,
    inverse_regularization_values: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
    max_iter: int = 4000,
    random_state: int = 20260322,
) -> LinearHeadTrainingResult:
    """Fit and select one multinomial linear head over frozen embeddings.

    Args:
        embeddings: Rank-2 matrix with one row per training sample.
        labels: Training labels aligned to ``embeddings``.
        label_order: Canonical label order for the exported head.
        dev_embeddings: Optional held-out embeddings for model selection.
        dev_labels: Optional held-out labels aligned to ``dev_embeddings``.
        inverse_regularization_values: Candidate ``C`` values passed to
            ``sklearn.linear_model.LogisticRegression``.
        max_iter: Maximum optimizer iterations per candidate.
        random_state: Stable seed for deterministic solver behavior.

    Returns:
        The best-scoring linear head in the requested label order.

    Raises:
        RuntimeError: If ``scikit-learn`` is unavailable in the current Python
            environment.
        ValueError: If shapes or label sets are invalid.
    """

    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover - depends on training env.
        raise RuntimeError(
            "Linear semantic-router training requires the `scikit-learn` package."
        ) from exc
    train_matrix = np.asarray(embeddings, dtype=np.float32)
    if train_matrix.ndim != 2 or train_matrix.shape[0] <= 0:
        raise ValueError("Linear-head training requires a rank-2 non-empty embedding matrix.")
    if train_matrix.shape[0] != len(labels):
        raise ValueError("Training embeddings and labels must have matching lengths.")
    canonical_labels = tuple(str(label).strip() for label in label_order if str(label).strip())
    if not canonical_labels:
        raise ValueError("Linear-head training requires a non-empty canonical label order.")
    label_to_index = {label: index for index, label in enumerate(canonical_labels)}
    missing_labels = sorted({str(label).strip() for label in labels} - set(label_to_index))
    if missing_labels:
        raise ValueError(f"Training labels are not present in the canonical order: {missing_labels}")
    train_targets = np.asarray([label_to_index[str(label).strip()] for label in labels], dtype=np.int64)
    selection_matrix = train_matrix
    selection_targets = train_targets
    if dev_embeddings is not None or dev_labels is not None:
        if dev_embeddings is None or dev_labels is None:
            raise ValueError("dev_embeddings and dev_labels must either both be provided or both be omitted.")
        selection_matrix = np.asarray(dev_embeddings, dtype=np.float32)
        if selection_matrix.ndim != 2 or selection_matrix.shape[0] != len(dev_labels):
            raise ValueError("Dev embeddings and dev labels must have matching rank-2 shapes.")
        dev_missing_labels = sorted({str(label).strip() for label in dev_labels} - set(label_to_index))
        if dev_missing_labels:
            raise ValueError(f"Dev labels are not present in the canonical order: {dev_missing_labels}")
        selection_targets = np.asarray([label_to_index[str(label).strip()] for label in dev_labels], dtype=np.int64)
    best_result: LinearHeadTrainingResult | None = None
    for inverse_regularization in tuple(float(value) for value in inverse_regularization_values):
        candidate = LogisticRegression(
            C=max(1e-6, inverse_regularization),
            class_weight=None,
            fit_intercept=True,
            max_iter=max(200, int(max_iter)),
            random_state=int(random_state),
            solver="lbfgs",
        )
        candidate.fit(train_matrix, train_targets)
        score = float(candidate.score(selection_matrix, selection_targets))
        result = LinearHeadTrainingResult(
            weights=np.asarray(candidate.coef_, dtype=np.float32),
            bias=np.asarray(candidate.intercept_, dtype=np.float32),
            inverse_regularization=float(inverse_regularization),
            score=score,
        )
        if best_result is None or result.score > best_result.score or (
            np.isclose(result.score, best_result.score) and result.inverse_regularization < best_result.inverse_regularization
        ):
            best_result = result
    if best_result is None:
        raise RuntimeError("Linear-head training did not evaluate any candidate models.")
    return best_result
