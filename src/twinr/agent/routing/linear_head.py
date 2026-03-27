"""Train compact linear classifier heads over frozen transcript embeddings.

This module keeps the discriminative classifier-head training logic out of the
runtime router services. The runtime still uses the same ONNX encoder on the
Pi; only the lightweight label head changes from centroid similarity to a
trained linear softmax head.
"""

# CHANGELOG: 2026-03-27
# BUG-1: Fixed incorrect weight/bias export for binary classification and
#        partially observed canonical label sets; the module now always exports
#        one logit row per canonical label in canonical order.
# BUG-2: Fixed hyper-parameter leakage when no dev split is provided; model
#        selection now uses stratified cross-validation instead of train-set
#        accuracy, avoiding systematically over-optimistic C selection.
# BUG-3: Fixed silent coercion of invalid C values (e.g. <= 0 or NaN) to 1e-6;
#        invalid regularization grids now fail fast.
# SEC-1: Added practical resource guards for matrix size, candidate-grid size
#        and iteration count to reduce CPU/RAM exhaustion risk on constrained
#        devices if this helper is ever wrapped by an API.
# IMP-1: Upgraded the trainer to 2026 practice for frozen-embedding probes:
#        class-imbalance-aware model selection, optional class balancing, and
#        optional feature standardization with coefficients folded back into the
#        exported raw-space linear head.
# IMP-2: Added exportable multiclass temperature scaling. The learned scalar is
#        folded into the returned weights/bias, so runtime remains a single
#        matrix multiply plus softmax.
# BREAKING: The default model-selection metric is now macro-F1 instead of raw
#           accuracy because accuracy hides failures on minority intents.
# BREAKING: When some canonical labels are absent from training data, their
#           exported rows are now explicitly suppressed instead of silently
#           returning a shape-misaligned head.

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence
import warnings

import numpy as np

_UNTRAINED_CLASS_BIAS = -80.0


@dataclass(frozen=True, slots=True)
class LinearHeadTrainingResult:
    """Store one trained multiclass linear head plus selection metadata."""

    weights: np.ndarray
    bias: np.ndarray
    inverse_regularization: float
    score: float
    class_weight_mode: str = "none"
    selection_metric: str = "macro_f1"
    calibration_beta: float = 1.0
    calibrated_log_loss: float | None = None
    uses_standardization: bool = False
    trained_labels: tuple[str, ...] = ()
    suppressed_labels: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _CandidateConfig:
    inverse_regularization: float
    class_weight_mode: str
    uses_standardization: bool


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
    selection_metric: Literal["macro_f1", "balanced_accuracy", "accuracy"] = "macro_f1",
    class_weight_modes: Sequence[str | None] = (None, "balanced"),
    use_standardization_values: Sequence[bool] = (False, True),
    max_matrix_elements: int = 16_000_000,
    max_candidate_models: int = 64,
    max_allowed_iter: int = 20_000,
) -> LinearHeadTrainingResult:
    """Fit and select one multinomial linear head over frozen embeddings.

    Args:
        embeddings: Rank-2 matrix with one row per training sample.
        labels: Training labels aligned to ``embeddings``.
        label_order: Canonical label order for the exported head.
        dev_embeddings: Optional held-out embeddings for calibration/final scoring.
        dev_labels: Optional held-out labels aligned to ``dev_embeddings``.
        inverse_regularization_values: Candidate ``C`` values passed to
            ``sklearn.linear_model.LogisticRegression``.
        max_iter: Maximum optimizer iterations per candidate.
        random_state: Stable seed for deterministic solver behavior.
        selection_metric: Higher-is-better metric used for candidate selection.
            The 2026 default is ``"macro_f1"`` to avoid majority-class bias.
        class_weight_modes: Candidate class-weighting modes. Supported values are
            ``None`` and ``"balanced"``.
        use_standardization_values: Whether to evaluate a z-scored feature space.
            The selected scaler is folded back into the exported raw-space head,
            so runtime remains drop-in compatible.
        max_matrix_elements: Guardrail against oversized dense matrices.
        max_candidate_models: Guardrail against oversized hyperparameter grids.
        max_allowed_iter: Guardrail against excessive optimizer iterations.

    Returns:
        The best-scoring linear head in the requested label order. Returned
        ``weights`` and ``bias`` already include any selected standardization and
        calibrated temperature scaling, so the runtime can stay unchanged.

    Raises:
        RuntimeError: If ``scikit-learn`` or ``SciPy`` are unavailable in the
            current Python environment.
        ValueError: If shapes, labels, or hyperparameters are invalid.
    """

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
        from sklearn.model_selection import StratifiedKFold
    except ImportError as exc:  # pragma: no cover - depends on training env.
        raise RuntimeError(
            "Linear semantic-router training requires the `scikit-learn` package."
        ) from exc
    try:
        from scipy.optimize import minimize_scalar
        from scipy.special import logsumexp
    except ImportError as exc:  # pragma: no cover - depends on training env.
        raise RuntimeError(
            "Linear semantic-router temperature calibration requires the `scipy` package."
        ) from exc

    def _to_dense_matrix(name: str, value: np.ndarray, *, expected_rows: int | None = None) -> np.ndarray:
        matrix = np.asarray(value, dtype=np.float64, order="C")
        if matrix.ndim != 2 or matrix.shape[0] <= 0:
            raise ValueError(f"{name} requires a rank-2 non-empty matrix.")
        if expected_rows is not None and matrix.shape[0] != expected_rows:
            raise ValueError(f"{name} rows must match the aligned label count.")
        if matrix.size > int(max_matrix_elements):
            raise ValueError(
                f"{name} exceeds the configured safety limit of {int(max_matrix_elements):,} elements."
            )
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} contains NaN or infinite values.")
        return matrix

    def _canonicalize_order(raw_labels: Sequence[str]) -> tuple[str, ...]:
        canonical = tuple(str(label).strip() for label in raw_labels if str(label).strip())
        if not canonical:
            raise ValueError("Linear-head training requires a non-empty canonical label order.")
        if len(set(canonical)) != len(canonical):
            raise ValueError("Canonical label order must not contain duplicates.")
        return canonical

    def _encode_targets(raw_labels: Sequence[str], *, label_to_index: dict[str, int], field_name: str) -> np.ndarray:
        normalized = [str(label).strip() for label in raw_labels]
        missing = sorted({label for label in normalized if label not in label_to_index})
        if missing:
            raise ValueError(f"{field_name} labels are not present in the canonical order: {missing}")
        return np.asarray([label_to_index[label] for label in normalized], dtype=np.int64)

    def _validate_inverse_regularization_values(values: Sequence[float]) -> tuple[float, ...]:
        cleaned: list[float] = []
        for raw in values:
            candidate = float(raw)
            if not math.isfinite(candidate) or candidate <= 0.0:
                raise ValueError(
                    "inverse_regularization_values must contain only finite values > 0."
                )
            cleaned.append(candidate)
        deduplicated = tuple(dict.fromkeys(cleaned))
        if not deduplicated:
            raise ValueError("At least one inverse_regularization value is required.")
        return deduplicated

    def _validate_class_weight_modes(values: Sequence[str | None]) -> tuple[str, ...]:
        normalized: list[str] = []
        for raw in values:
            if raw is None:
                normalized.append("none")
                continue
            value = str(raw).strip().lower()
            if value not in {"none", "balanced"}:
                raise ValueError(
                    "class_weight_modes supports only None, 'none', or 'balanced'."
                )
            normalized.append(value)
        deduplicated = tuple(dict.fromkeys(normalized))
        if not deduplicated:
            raise ValueError("At least one class_weight mode is required.")
        return deduplicated

    def _validate_standardization_values(values: Sequence[bool]) -> tuple[bool, ...]:
        normalized = tuple(dict.fromkeys(bool(value) for value in values))
        if not normalized:
            raise ValueError("At least one standardization mode is required.")
        return normalized

    def _build_candidate_configs() -> tuple[_CandidateConfig, ...]:
        configs = tuple(
            _CandidateConfig(
                inverse_regularization=inverse_regularization,
                class_weight_mode=class_weight_mode,
                uses_standardization=uses_standardization,
            )
            for inverse_regularization in regularization_grid
            for class_weight_mode in class_weight_grid
            for uses_standardization in standardization_grid
        )
        if len(configs) > int(max_candidate_models):
            raise ValueError(
                f"Candidate grid has {len(configs)} models, above the safety limit of "
                f"{int(max_candidate_models)}."
            )
        return configs

    def _compute_standardization_stats(matrix: np.ndarray, enabled: bool) -> tuple[np.ndarray, np.ndarray]:
        if not enabled:
            return np.zeros(matrix.shape[1], dtype=np.float64), np.ones(matrix.shape[1], dtype=np.float64)
        mean = np.mean(matrix, axis=0, dtype=np.float64)
        scale = np.std(matrix, axis=0, dtype=np.float64)
        scale = np.where(scale > 1e-12, scale, 1.0)
        return mean.astype(np.float64, copy=False), scale.astype(np.float64, copy=False)

    def _apply_standardization(matrix: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
        transformed = (matrix - mean) / scale
        return np.asarray(transformed, dtype=np.float64, order="C")

    def _fit_estimator(matrix: np.ndarray, targets: np.ndarray, config: _CandidateConfig):
        if len(np.unique(targets)) < 2:
            raise ValueError("Linear-head training requires at least two observed training labels.")
        estimator = LogisticRegression(
            C=config.inverse_regularization,
            class_weight=None if config.class_weight_mode == "none" else config.class_weight_mode,
            fit_intercept=True,
            max_iter=max(200, int(max_iter)),
            random_state=int(random_state),
            solver="lbfgs",
        )
        estimator.fit(matrix, targets)
        return estimator

    def _export_aligned_head(
        estimator,
        *,
        canonical_size: int,
        mean: np.ndarray,
        scale: np.ndarray,
        beta: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
        coef = np.asarray(estimator.coef_, dtype=np.float64, order="C")
        intercept = np.asarray(estimator.intercept_, dtype=np.float64).reshape(-1)
        observed_classes = tuple(int(value) for value in np.asarray(estimator.classes_, dtype=np.int64).tolist())
        weights = np.zeros((canonical_size, coef.shape[1]), dtype=np.float64)
        bias = np.full(canonical_size, _UNTRAINED_CLASS_BIAS, dtype=np.float64)

        if len(observed_classes) == 2 and coef.shape[0] == 1:
            negative_class, positive_class = observed_classes
            weights[negative_class, :] = 0.0
            bias[negative_class] = 0.0
            weights[positive_class, :] = coef[0]
            bias[positive_class] = float(intercept[0])
        elif coef.shape[0] == len(observed_classes):
            for row_index, class_index in enumerate(observed_classes):
                weights[class_index, :] = coef[row_index]
                bias[class_index] = float(intercept[row_index])
        else:
            raise RuntimeError(
                "Unexpected LogisticRegression coefficient layout; unable to export aligned head."
            )

        raw_weights = weights / scale[None, :]
        raw_bias = bias - np.sum(raw_weights * mean[None, :], axis=1)
        if beta != 1.0:
            raw_weights = raw_weights * beta
            raw_bias = raw_bias * beta
        return (
            np.asarray(raw_weights, dtype=np.float32),
            np.asarray(raw_bias, dtype=np.float32),
            observed_classes,
        )

    def _predict_logits(matrix: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
        return np.asarray(matrix @ weights.T + bias[None, :], dtype=np.float64, order="C")

    def _argmax_predictions(logits: np.ndarray) -> np.ndarray:
        return np.asarray(np.argmax(logits, axis=1), dtype=np.int64)

    def _score_from_logits(targets: np.ndarray, logits: np.ndarray) -> float:
        predictions = _argmax_predictions(logits)
        if selection_metric == "accuracy":
            return float(accuracy_score(targets, predictions))
        if selection_metric == "balanced_accuracy":
            return float(balanced_accuracy_score(targets, predictions))
        if selection_metric == "macro_f1":
            return float(f1_score(targets, predictions, average="macro", zero_division=0))
        raise ValueError(f"Unsupported selection_metric: {selection_metric}")

    def _log_loss_from_logits(targets: np.ndarray, logits: np.ndarray, beta: float = 1.0) -> float:
        scaled_logits = np.asarray(beta * logits, dtype=np.float64, order="C")
        losses = logsumexp(scaled_logits, axis=1) - scaled_logits[np.arange(targets.shape[0]), targets]
        return float(np.mean(losses))

    def _fit_temperature_beta(logits: np.ndarray, targets: np.ndarray) -> float:
        if logits.shape[0] < 2 or len(np.unique(targets)) < 2:
            return 1.0

        def objective(log_beta: float) -> float:
            return _log_loss_from_logits(targets, logits, beta=math.exp(float(log_beta)))

        result = minimize_scalar(
            objective,
            bounds=(-10.0, 10.0),
            method="bounded",
            options={"xatol": 1e-6},
        )
        if not result.success:  # pragma: no cover - scipy failures are uncommon.
            warnings.warn(
                f"Temperature scaling failed to converge ({result.message}); using beta=1.0.",
                RuntimeWarning,
                stacklevel=2,
            )
            return 1.0
        return float(math.exp(float(result.x)))

    def _make_stratified_cv_splits(targets: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        class_counts = np.bincount(targets, minlength=len(canonical_labels))
        positive_counts = class_counts[class_counts > 0]
        if positive_counts.size == 0:
            raise ValueError("Linear-head training requires at least one observed class.")
        min_count = int(np.min(positive_counts))
        if min_count < 2 or targets.shape[0] < 4:
            return ()
        n_splits = min(5, min_count)
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=int(random_state),
        )
        return tuple(splitter.split(train_matrix, targets))

    def _evaluate_candidate_cv(config: _CandidateConfig, splits: tuple[tuple[np.ndarray, np.ndarray], ...]) -> float:
        oof_logits = np.empty((train_matrix.shape[0], len(canonical_labels)), dtype=np.float64)
        oof_targets = np.empty(train_matrix.shape[0], dtype=np.int64)
        for train_indices, val_indices in splits:
            fold_train = train_matrix[train_indices]
            fold_targets = train_targets[train_indices]
            mean, scale = _compute_standardization_stats(fold_train, config.uses_standardization)
            fold_train_scaled = _apply_standardization(fold_train, mean, scale)
            estimator = _fit_estimator(fold_train_scaled, fold_targets, config)
            weights, bias, _ = _export_aligned_head(
                estimator,
                canonical_size=len(canonical_labels),
                mean=mean,
                scale=scale,
                beta=1.0,
            )
            oof_logits[val_indices, :] = _predict_logits(train_matrix[val_indices], weights, bias)
            oof_targets[val_indices] = train_targets[val_indices]
        return _score_from_logits(oof_targets, oof_logits)

    def _collect_oof_logits_for_candidate(
        config: _CandidateConfig,
        splits: tuple[tuple[np.ndarray, np.ndarray], ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        oof_logits = np.empty((train_matrix.shape[0], len(canonical_labels)), dtype=np.float64)
        oof_targets = np.empty(train_matrix.shape[0], dtype=np.int64)
        for train_indices, val_indices in splits:
            fold_train = train_matrix[train_indices]
            fold_targets = train_targets[train_indices]
            mean, scale = _compute_standardization_stats(fold_train, config.uses_standardization)
            fold_train_scaled = _apply_standardization(fold_train, mean, scale)
            estimator = _fit_estimator(fold_train_scaled, fold_targets, config)
            weights, bias, _ = _export_aligned_head(
                estimator,
                canonical_size=len(canonical_labels),
                mean=mean,
                scale=scale,
                beta=1.0,
            )
            oof_logits[val_indices, :] = _predict_logits(train_matrix[val_indices], weights, bias)
            oof_targets[val_indices] = train_targets[val_indices]
        return oof_logits, oof_targets

    def _fit_final_candidate(config: _CandidateConfig) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
        mean, scale = _compute_standardization_stats(train_matrix, config.uses_standardization)
        train_scaled = _apply_standardization(train_matrix, mean, scale)
        estimator = _fit_estimator(train_scaled, train_targets, config)
        return _export_aligned_head(
            estimator,
            canonical_size=len(canonical_labels),
            mean=mean,
            scale=scale,
            beta=1.0,
        )

    if int(max_iter) > int(max_allowed_iter):
        raise ValueError(
            f"max_iter={int(max_iter)} exceeds the configured safety limit of {int(max_allowed_iter)}."
        )

    train_matrix = _to_dense_matrix("embeddings", embeddings, expected_rows=len(labels))
    canonical_labels = _canonicalize_order(label_order)
    label_to_index = {label: index for index, label in enumerate(canonical_labels)}
    train_targets = _encode_targets(labels, label_to_index=label_to_index, field_name="Training")

    if len(np.unique(train_targets)) < 2:
        raise ValueError("Linear-head training requires at least two distinct training labels.")

    dev_matrix: np.ndarray | None = None
    dev_targets: np.ndarray | None = None
    if dev_embeddings is not None or dev_labels is not None:
        if dev_embeddings is None or dev_labels is None:
            raise ValueError(
                "dev_embeddings and dev_labels must either both be provided or both be omitted."
            )
        dev_matrix = _to_dense_matrix("dev_embeddings", dev_embeddings, expected_rows=len(dev_labels))
        if dev_matrix.shape[1] != train_matrix.shape[1]:
            raise ValueError("Train and dev embeddings must have the same feature dimension.")
        dev_targets = _encode_targets(dev_labels, label_to_index=label_to_index, field_name="Dev")

    regularization_grid = _validate_inverse_regularization_values(inverse_regularization_values)
    class_weight_grid = _validate_class_weight_modes(class_weight_modes)
    standardization_grid = _validate_standardization_values(use_standardization_values)
    candidate_configs = _build_candidate_configs()

    observed_train_classes = set(int(value) for value in np.unique(train_targets).tolist())
    suppressed_labels = tuple(
        canonical_labels[index]
        for index in range(len(canonical_labels))
        if index not in observed_train_classes
    )
    if suppressed_labels:
        warnings.warn(
            "Some canonical labels are absent from the training set and will be exported "
            f"as suppressed rows: {list(suppressed_labels)}",
            RuntimeWarning,
            stacklevel=2,
        )

    cv_splits = _make_stratified_cv_splits(train_targets)
    best_config: _CandidateConfig | None = None
    best_score: float | None = None

    def _is_better(candidate_score: float, candidate_config: _CandidateConfig) -> bool:
        nonlocal best_score, best_config
        if best_score is None or best_config is None:
            return True
        if candidate_score > best_score:
            return True
        if not np.isclose(candidate_score, best_score):
            return False
        if candidate_config.inverse_regularization < best_config.inverse_regularization:
            return True
        if not np.isclose(candidate_config.inverse_regularization, best_config.inverse_regularization):
            return False
        if int(candidate_config.uses_standardization) < int(best_config.uses_standardization):
            return True
        if candidate_config.uses_standardization != best_config.uses_standardization:
            return False
        return candidate_config.class_weight_mode == "none" and best_config.class_weight_mode != "none"

    if cv_splits:
        for config in candidate_configs:
            score = _evaluate_candidate_cv(config, cv_splits)
            if _is_better(score, config):
                best_score = float(score)
                best_config = config
    else:
        warnings.warn(
            "Not enough per-class support for stratified cross-validation; falling back to "
            "single-fit selection on the training set. Selection/calibration quality will be optimistic.",
            RuntimeWarning,
            stacklevel=2,
        )
        for config in candidate_configs:
            weights, bias, _ = _fit_final_candidate(config)
            logits = _predict_logits(train_matrix, weights, bias)
            score = _score_from_logits(train_targets, logits)
            if _is_better(score, config):
                best_score = float(score)
                best_config = config

    if best_config is None or best_score is None:
        raise RuntimeError("Linear-head training did not evaluate any candidate models.")

    final_weights, final_bias, observed_final_classes = _fit_final_candidate(best_config)

    calibration_beta = 1.0
    calibrated_log_loss: float | None = None
    reported_score = float(best_score)

    if dev_matrix is not None and dev_targets is not None:
        dev_logits = _predict_logits(dev_matrix, final_weights, final_bias)
        calibration_beta = _fit_temperature_beta(dev_logits, dev_targets)
        calibrated_log_loss = _log_loss_from_logits(dev_targets, dev_logits, beta=calibration_beta)
        reported_score = _score_from_logits(dev_targets, dev_logits)
    elif cv_splits:
        oof_logits, oof_targets = _collect_oof_logits_for_candidate(best_config, cv_splits)
        calibration_beta = _fit_temperature_beta(oof_logits, oof_targets)
        calibrated_log_loss = _log_loss_from_logits(oof_targets, oof_logits, beta=calibration_beta)
        reported_score = _score_from_logits(oof_targets, oof_logits)
    else:
        train_logits = _predict_logits(train_matrix, final_weights, final_bias)
        calibrated_log_loss = _log_loss_from_logits(train_targets, train_logits, beta=1.0)
        reported_score = _score_from_logits(train_targets, train_logits)

    final_weights = np.asarray(final_weights * calibration_beta, dtype=np.float32)
    final_bias = np.asarray(final_bias * calibration_beta, dtype=np.float32)

    return LinearHeadTrainingResult(
        weights=final_weights,
        bias=final_bias,
        inverse_regularization=float(best_config.inverse_regularization),
        score=float(reported_score),
        class_weight_mode=best_config.class_weight_mode,
        selection_metric=selection_metric,
        calibration_beta=float(calibration_beta),
        calibrated_log_loss=None if calibrated_log_loss is None else float(calibrated_log_loss),
        uses_standardization=bool(best_config.uses_standardization),
        trained_labels=tuple(canonical_labels[index] for index in observed_final_classes),
        suppressed_labels=suppressed_labels,
    )