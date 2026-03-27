# CHANGELOG: 2026-03-27
# BUG-1: Fixed incorrect shared-encoder reuse across bundle pairs that only matched shape settings
#        (`max_length/pooling/output_name/normalize_embeddings`) but actually pointed at different
#        encoder artifacts. This previously produced silent misrouting whenever user-intent and route
#        bundles were trained in different embedding spaces.
# BUG-2: Added bundle-compatibility validation so user-intent -> allowed-route mappings are checked
#        against the installed route bundle labels at startup instead of failing later after an OTA or
#        partial bundle rollout.
# SEC-1: Hardened local artifact loading for classifier weights/centroids with explicit
#        `allow_pickle=False`, regular-file checks, optional trusted-root enforcement, and optional
#        artifact-size limits to reduce practical model-swap / local-DoS risk on appliance-style
#        Raspberry Pi deployments.
# IMP-1: Added exact-text LRU embedding caches so repeated transcripts can reuse embeddings without
#        re-running ONNX inference. This is a lightweight, Pi-friendly approximation of the
#        memory-augmented routing trend.
# IMP-2: Added calibrated-routing hooks (`min_confidence_for_constrained_routing` +
#        `low_confidence_route_labels` metadata) so deployments can plug in offline-calibrated
#        act/hold-style safety thresholds without another API break.
# IMP-3: Added stricter metadata/artifact validation, clearer dimension mismatch errors, and more
#        accurate latency accounting for the local user-intent stage.

"""Run the user-centered stage and backend route stage on one shared embedding."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
import math
import os
import threading
import time
from typing import Any, Iterable, Sequence

import numpy as np

from .bundle import SemanticRouterBundle
from .inference import OnnxSentenceEncoder, _coerce_embedding_row, _l2_normalize, _softmax
from .policy import SemanticRouterPolicy
from .service import LocalSemanticRouter
from .user_intent import (
    TwoStageSemanticRouteDecision,
    UserIntentDecision,
    allowed_route_labels_for_user_intent,
)
from .user_intent_bundle import UserIntentBundle, UserIntentBundleMetadata


class _EmbeddingLRUCache:
    """Tiny thread-safe LRU cache for exact-text embedding reuse."""

    __slots__ = ("_max_entries", "_items", "_lock")

    def __init__(self, max_entries: int) -> None:
        self._max_entries = max(0, int(max_entries))
        self._items: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.RLock()

    @property
    def enabled(self) -> bool:
        return self._max_entries > 0

    def get(self, key: str) -> np.ndarray | None:
        if not self.enabled:
            return None
        with self._lock:
            value = self._items.get(key)
            if value is None:
                return None
            self._items.move_to_end(key)
            return value.copy()

    def put(self, key: str, value: np.ndarray) -> None:
        if not self.enabled:
            return
        cached_value = _coerce_embedding_row(value).astype(np.float32, copy=False)
        with self._lock:
            self._items[key] = cached_value.copy()
            self._items.move_to_end(key)
            while len(self._items) > self._max_entries:
                self._items.popitem(last=False)


def _normalize_label_sequence(value: Iterable[str] | str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _dedupe_preserve_order(labels: Iterable[str] | str) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw_label in _normalize_label_sequence(labels):
        label = str(raw_label)
        if label in seen:
            continue
        seen.add(label)
        ordered.append(label)
    return tuple(ordered)


def _resolved_path_identity(value: Any) -> str:
    path_str = os.fspath(value)
    candidate = Path(path_str).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return str(candidate)


def _get_metadata_value(metadata: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if hasattr(metadata, name):
            value = getattr(metadata, name)
            if value is not None:
                return value
    return default


def _validate_positive_finite(value: float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return numeric


def _validate_artifact_path(
    path_like: os.PathLike[str] | str,
    *,
    label: str,
    allow_directory: bool = False,
) -> Path:
    path = Path(os.fspath(path_like)).expanduser()
    resolved = path.resolve(strict=True)
    if allow_directory:
        if not (resolved.is_dir() or resolved.is_file()):
            raise ValueError(f"{label} must point to an existing file or directory: {resolved}")
    else:
        if not resolved.is_file():
            raise ValueError(f"{label} must point to an existing regular file: {resolved}")

    trusted_root = os.getenv("TWINR_TRUSTED_BUNDLE_ROOT")
    if trusted_root:
        root = Path(trusted_root).expanduser().resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"{label} must live under trusted bundle root {root}: {resolved}"
            ) from exc

    if resolved.is_file():
        max_mb = os.getenv("TWINR_MAX_NUMPY_ARTIFACT_MB")
        if max_mb:
            max_bytes = int(float(max_mb) * 1024 * 1024)
            if resolved.stat().st_size > max_bytes:
                raise ValueError(
                    f"{label} exceeds configured artifact limit ({max_mb} MiB): {resolved}"
                )
    return resolved


def _load_npy_float32(path_like: os.PathLike[str] | str, *, label: str) -> np.ndarray:
    path = _validate_artifact_path(path_like, label=label, allow_directory=False)
    array = np.load(path, allow_pickle=False, mmap_mode="r")
    matrix = np.asarray(array, dtype=np.float32)
    if matrix.size == 0:
        raise ValueError(f"{label} must not be empty: {path}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must contain only finite values: {path}")
    return matrix


def _encoder_compatibility_key(bundle: UserIntentBundle | SemanticRouterBundle) -> tuple[Any, ...]:
    metadata = bundle.metadata
    explicit_fingerprint = _get_metadata_value(
        metadata,
        "encoder_fingerprint",
        "shared_encoder_fingerprint",
        "embedding_space_id",
        default=None,
    )
    if explicit_fingerprint is not None:
        encoder_identity = ("fingerprint", str(explicit_fingerprint))
    else:
        encoder_identity = (
            "artifacts",
            _resolved_path_identity(bundle.model_path),
            _resolved_path_identity(bundle.tokenizer_path),
        )
    return (
        encoder_identity,
        getattr(metadata, "max_length", None),
        getattr(metadata, "pooling", None),
        getattr(metadata, "output_name", None),
        getattr(metadata, "normalize_embeddings", None),
        _get_metadata_value(
            metadata,
            "truncate_dim",
            "embedding_dim",
            "matryoshka_dim",
            "active_layer",
            "early_exit_layer",
            default=None,
        ),
    )


def _bundles_can_share_encoder(
    user_intent_bundle: UserIntentBundle,
    route_bundle: SemanticRouterBundle,
) -> bool:
    return _encoder_compatibility_key(user_intent_bundle) == _encoder_compatibility_key(route_bundle)


def _maybe_set_latency_ms(decision: Any, latency_ms: float) -> Any:
    if hasattr(decision, "latency_ms"):
        try:
            setattr(decision, "latency_ms", float(latency_ms))
        except Exception:
            pass
    return decision


@dataclass(slots=True)
class LocalUserIntentRouter:
    """Classify transcripts into user-centered routing classes locally."""

    bundle: UserIntentBundle
    encoder: OnnxSentenceEncoder | None = None
    centroids: np.ndarray | None = None
    metadata: UserIntentBundleMetadata = field(init=False, repr=False)
    _centroids: np.ndarray | None = field(init=False, default=None, repr=False)
    _weights: np.ndarray | None = field(init=False, default=None, repr=False)
    _bias: np.ndarray | None = field(init=False, default=None, repr=False)
    _expected_embedding_dim: int = field(init=False, default=0, repr=False)
    _truncate_to_expected_dim: bool = field(init=False, default=False, repr=False)
    _temperature: float = field(init=False, default=1.0, repr=False)

    def __post_init__(self) -> None:
        self.metadata = self.bundle.metadata
        if not self.metadata.labels:
            raise ValueError("User intent bundles must define at least one label.")
        if len(set(self.metadata.labels)) != len(self.metadata.labels):
            raise ValueError("User intent bundle labels must be unique.")
        self._temperature = _validate_positive_finite(self.metadata.temperature, "metadata.temperature")

        truncate_dim = _get_metadata_value(
            self.metadata,
            "truncate_dim",
            "embedding_dim",
            "matryoshka_dim",
            default=None,
        )
        if truncate_dim is not None:
            truncate_dim = int(truncate_dim)
            if truncate_dim <= 0:
                raise ValueError("Configured embedding truncate dimension must be positive.")
        self._truncate_to_expected_dim = bool(
            _get_metadata_value(
                self.metadata,
                "allow_embedding_truncation",
                "allow_matryoshka_truncation",
                default=False,
            )
        ) or truncate_dim is not None

        self.encoder = self.encoder or OnnxSentenceEncoder(
            model_path=self.bundle.model_path,
            tokenizer_path=self.bundle.tokenizer_path,
            max_length=self.metadata.max_length,
            pooling=self.metadata.pooling,
            output_name=self.metadata.output_name,
            normalize_embeddings=self.metadata.normalize_embeddings,
        )
        self._centroids = None
        self._weights = None
        self._bias = None

        if self.metadata.classifier_type == "embedding_centroid_v1":
            centroids_path = self.bundle.centroids_path
            if centroids_path is None and self.centroids is None:
                raise ValueError("Centroid user-intent bundles must provide centroids.npy.")
            if self.centroids is not None:
                centroid_matrix = np.asarray(self.centroids, dtype=np.float32)
            else:
                assert centroids_path is not None
                centroid_matrix = _load_npy_float32(
                    centroids_path,
                    label="User intent centroids artifact",
                )
            if centroid_matrix.ndim != 2:
                raise ValueError("User intent centroids must be a rank-2 matrix.")
            if centroid_matrix.shape[0] != len(self.metadata.labels):
                raise ValueError(
                    "User intent centroid rows must match the configured label order."
                )
            if self.metadata.normalize_centroids:
                centroid_matrix = _l2_normalize(centroid_matrix)
            self._centroids = centroid_matrix
            self._expected_embedding_dim = int(centroid_matrix.shape[1])
        else:
            if self.bundle.weights_path is None or self.bundle.bias_path is None:
                raise ValueError("Linear user-intent bundles must provide weights.npy and bias.npy.")
            weight_matrix = _load_npy_float32(
                self.bundle.weights_path,
                label="User intent linear weights artifact",
            )
            bias_vector = _load_npy_float32(
                self.bundle.bias_path,
                label="User intent linear bias artifact",
            )
            if weight_matrix.ndim != 2:
                raise ValueError("User intent linear weights must be a rank-2 matrix.")
            if weight_matrix.shape[0] != len(self.metadata.labels):
                raise ValueError(
                    "User intent linear weights must match the configured label order."
                )
            if bias_vector.shape not in {(len(self.metadata.labels),), (len(self.metadata.labels), 1)}:
                raise ValueError("User intent linear bias must have one entry per label.")
            self._weights = weight_matrix
            self._bias = bias_vector.reshape(-1)
            self._expected_embedding_dim = int(weight_matrix.shape[1])

        if self._expected_embedding_dim <= 0:
            raise ValueError("User intent classifier must expect at least one embedding dimension.")
        if truncate_dim is not None and int(truncate_dim) != self._expected_embedding_dim:
            raise ValueError(
                "Configured embedding truncate dimension does not match the classifier artifact width."
            )

    def classify(self, text: str) -> UserIntentDecision:
        """Return one scored local user-intent decision."""

        cleaned = str(text or "").strip()
        if not cleaned:
            raise ValueError("LocalUserIntentRouter.classify requires non-empty text.")
        encoder = self.encoder
        if encoder is None:
            raise RuntimeError("LocalUserIntentRouter encoder was not initialized.")
        started = time.perf_counter()
        embedding = _coerce_embedding_row(encoder.encode([cleaned])[0:1])
        latency_ms = (time.perf_counter() - started) * 1000.0
        return self.classify_embedding(embedding, latency_ms=latency_ms)

    def classify_embedding(
        self,
        embedding: np.ndarray,
        *,
        latency_ms: float = 0.0,
    ) -> UserIntentDecision:
        """Return one user-intent decision for an already encoded embedding."""

        started = time.perf_counter()
        normalized_embedding = self._prepare_embedding_row(embedding)
        logits = self._classifier_logits(normalized_embedding[0])
        probabilities = _softmax(logits / self._temperature)
        if probabilities.ndim != 1 or probabilities.shape[0] != len(self.metadata.labels):
            raise RuntimeError("User intent classifier returned an invalid probability vector.")

        if probabilities.shape[0] == 1:
            top_index = 0
            second_index = 0
        else:
            top2 = np.argpartition(probabilities, -2)[-2:]
            top2 = top2[np.argsort(probabilities[top2])[::-1]]
            top_index = int(top2[0])
            second_index = int(top2[1])

        total_latency_ms = float(latency_ms + (time.perf_counter() - started) * 1000.0)
        return UserIntentDecision(
            label=self.metadata.labels[top_index],
            confidence=float(probabilities[top_index]),
            margin=float(probabilities[top_index] - probabilities[second_index]),
            scores={
                label: float(probabilities[index])
                for index, label in enumerate(self.metadata.labels)
            },
            model_id=self.metadata.model_id,
            latency_ms=total_latency_ms,
        )

    def _prepare_embedding_row(self, embedding: np.ndarray) -> np.ndarray:
        embedding_row = _coerce_embedding_row(embedding).astype(np.float32, copy=False)
        current_dim = int(embedding_row.shape[1])
        expected_dim = self._expected_embedding_dim

        if current_dim != expected_dim:
            if current_dim > expected_dim and self._truncate_to_expected_dim:
                embedding_row = embedding_row[:, :expected_dim]
            else:
                raise ValueError(
                    f"User intent embedding dimension mismatch: expected {expected_dim}, got {current_dim}."
                )

        if self.metadata.normalize_embeddings:
            embedding_row = _l2_normalize(embedding_row)
        return embedding_row

    def _classifier_logits(self, embedding_row: np.ndarray) -> np.ndarray:
        """Return one user-intent logit vector for a single normalized embedding."""

        if self.metadata.classifier_type == "embedding_centroid_v1":
            if self._centroids is None:
                raise RuntimeError("Centroid user-intent state was not initialized.")
            return np.matmul(self._centroids, embedding_row)
        if self._weights is None or self._bias is None:
            raise RuntimeError("Linear user-intent state was not initialized.")
        return np.matmul(self._weights, embedding_row) + self._bias


class TwoStageLocalSemanticRouter:
    """Classify transcripts via a user-intent stage plus constrained backend stage."""

    def __init__(
        self,
        user_intent_bundle: UserIntentBundle,
        route_bundle: SemanticRouterBundle,
        *,
        user_intent_router: LocalUserIntentRouter | None = None,
        route_router: LocalSemanticRouter | None = None,
        embedding_cache_size: int = 128,
        strict_bundle_validation: bool = True,
    ) -> None:
        shared_encoder = _resolve_shared_encoder(
            user_intent_bundle,
            route_bundle,
            user_intent_router=user_intent_router,
            route_router=route_router,
        )
        self.user_intent_router = user_intent_router or LocalUserIntentRouter(
            user_intent_bundle,
            encoder=shared_encoder,
        )
        self.route_router = route_router or LocalSemanticRouter(
            route_bundle,
            encoder=shared_encoder,
        )
        self.user_intent_bundle = user_intent_bundle
        self.route_bundle = route_bundle
        self.shared_encoder = shared_encoder

        cache_entries = max(0, int(embedding_cache_size))
        self._shared_embedding_cache = _EmbeddingLRUCache(cache_entries)
        self._user_intent_embedding_cache = _EmbeddingLRUCache(cache_entries if shared_encoder is None else 0)
        self._route_embedding_cache = _EmbeddingLRUCache(cache_entries if shared_encoder is None else 0)

        self._route_labels = _dedupe_preserve_order(
            getattr(self.route_bundle.metadata, "labels", ()) or ()
        )
        self._route_label_set = frozenset(self._route_labels)

        self._allowed_route_labels_by_user_intent = self._compile_allowed_route_labels(
            strict_bundle_validation=strict_bundle_validation
        )
        self._min_user_intent_confidence_for_constrained_routing = self._load_optional_confidence_threshold()
        self._low_confidence_route_labels = self._compile_low_confidence_route_labels(
            strict_bundle_validation=strict_bundle_validation
        )

    def classify(
        self,
        text: str,
        *,
        policy: SemanticRouterPolicy | None = None,
    ) -> TwoStageSemanticRouteDecision:
        """Return one user-centered decision plus one constrained backend route."""

        cleaned = str(text or "").strip()
        if not cleaned:
            raise ValueError("TwoStageLocalSemanticRouter.classify requires non-empty text.")

        if self.shared_encoder is None:
            user_intent = self._classify_user_intent_with_router_encoder(cleaned)
            allowed_route_labels = self._select_allowed_route_labels(user_intent)
            route_decision = self._classify_route_with_router_encoder(
                cleaned,
                policy=policy,
                allowed_labels=allowed_route_labels,
            )
            return TwoStageSemanticRouteDecision(
                user_intent=user_intent,
                route_decision=route_decision,
                allowed_route_labels=allowed_route_labels,
            )

        embedding, encode_latency_ms = self._encode_with_cache(
            cleaned,
            self.shared_encoder,
            self._shared_embedding_cache,
        )
        user_intent = self.user_intent_router.classify_embedding(
            embedding,
            latency_ms=encode_latency_ms,
        )
        allowed_route_labels = self._select_allowed_route_labels(user_intent)

        route_started = time.perf_counter()
        route_decision = self.route_router.classify_embedding(
            embedding,
            policy=policy,
            latency_ms=encode_latency_ms,
            allowed_labels=allowed_route_labels,
        )
        route_latency_ms = encode_latency_ms + (time.perf_counter() - route_started) * 1000.0
        route_decision = _maybe_set_latency_ms(route_decision, route_latency_ms)

        return TwoStageSemanticRouteDecision(
            user_intent=user_intent,
            route_decision=route_decision,
            allowed_route_labels=allowed_route_labels,
        )

    def _classify_user_intent_with_router_encoder(self, cleaned: str) -> UserIntentDecision:
        encoder = getattr(self.user_intent_router, "encoder", None)
        if encoder is None:
            started = time.perf_counter()
            decision = self.user_intent_router.classify(cleaned)
            return _maybe_set_latency_ms(decision, (time.perf_counter() - started) * 1000.0)

        embedding, encode_latency_ms = self._encode_with_cache(
            cleaned,
            encoder,
            self._user_intent_embedding_cache,
        )
        return self.user_intent_router.classify_embedding(
            embedding,
            latency_ms=encode_latency_ms,
        )

    def _classify_route_with_router_encoder(
        self,
        cleaned: str,
        *,
        policy: SemanticRouterPolicy | None,
        allowed_labels: Sequence[str],
    ):
        encoder = getattr(self.route_router, "encoder", None)
        if encoder is None:
            started = time.perf_counter()
            decision = self.route_router.classify(
                cleaned,
                policy=policy,
                allowed_labels=allowed_labels,
            )
            return _maybe_set_latency_ms(decision, (time.perf_counter() - started) * 1000.0)

        embedding, encode_latency_ms = self._encode_with_cache(
            cleaned,
            encoder,
            self._route_embedding_cache,
        )
        started = time.perf_counter()
        decision = self.route_router.classify_embedding(
            embedding,
            policy=policy,
            latency_ms=encode_latency_ms,
            allowed_labels=allowed_labels,
        )
        total_latency_ms = encode_latency_ms + (time.perf_counter() - started) * 1000.0
        return _maybe_set_latency_ms(decision, total_latency_ms)

    def _encode_with_cache(
        self,
        text: str,
        encoder: OnnxSentenceEncoder,
        cache: _EmbeddingLRUCache,
    ) -> tuple[np.ndarray, float]:
        cached = cache.get(text)
        if cached is not None:
            return cached, 0.0

        started = time.perf_counter()
        embedding = _coerce_embedding_row(encoder.encode([text])[0:1]).astype(np.float32, copy=False)
        latency_ms = (time.perf_counter() - started) * 1000.0
        cache.put(text, embedding)
        return embedding, latency_ms

    def _compile_allowed_route_labels(
        self,
        *,
        strict_bundle_validation: bool,
    ) -> dict[str, tuple[str, ...]]:
        compiled: dict[str, tuple[str, ...]] = {}
        for intent_label in self.user_intent_bundle.metadata.labels:
            raw_allowed = _dedupe_preserve_order(allowed_route_labels_for_user_intent(intent_label))
            if self._route_label_set:
                allowed = tuple(label for label in raw_allowed if label in self._route_label_set)
            else:
                allowed = raw_allowed

            if strict_bundle_validation and not allowed:
                raise ValueError(
                    "No valid backend routes remain after applying user-intent constraints for "
                    f"intent label {intent_label!r}. Check bundle/version compatibility."
                )
            compiled[str(intent_label)] = allowed
        return compiled

    def _compile_low_confidence_route_labels(
        self,
        *,
        strict_bundle_validation: bool,
    ) -> tuple[str, ...] | None:
        raw_labels = _get_metadata_value(
            self.user_intent_bundle.metadata,
            "low_confidence_route_labels",
            "fallback_route_labels",
            default=None,
        )
        if raw_labels is None:
            return None

        labels = _dedupe_preserve_order(raw_labels)
        if self._route_label_set:
            labels = tuple(label for label in labels if label in self._route_label_set)

        if strict_bundle_validation and not labels:
            raise ValueError(
                "Configured low-confidence route labels do not match the installed route bundle."
            )
        return labels

    def _load_optional_confidence_threshold(self) -> float | None:
        threshold = _get_metadata_value(
            self.user_intent_bundle.metadata,
            "min_confidence_for_constrained_routing",
            "low_confidence_threshold",
            "abstain_below_confidence",
            default=None,
        )
        if threshold is None:
            return None
        numeric = float(threshold)
        if not math.isfinite(numeric) or not (0.0 <= numeric <= 1.0):
            raise ValueError("Configured user-intent confidence threshold must be between 0 and 1.")
        return numeric

    def _select_allowed_route_labels(self, user_intent: UserIntentDecision) -> tuple[str, ...]:
        if (
            self._min_user_intent_confidence_for_constrained_routing is not None
            and self._low_confidence_route_labels is not None
            and user_intent.confidence < self._min_user_intent_confidence_for_constrained_routing
        ):
            return self._low_confidence_route_labels
        return self._allowed_route_labels_by_user_intent[user_intent.label]


def _build_shared_encoder(
    user_intent_bundle: UserIntentBundle,
    route_bundle: SemanticRouterBundle,
) -> OnnxSentenceEncoder | None:
    """Return one shared encoder when both stages use the same embedding space."""

    if not _bundles_can_share_encoder(user_intent_bundle, route_bundle):
        return None

    user_metadata = user_intent_bundle.metadata
    return OnnxSentenceEncoder(
        model_path=user_intent_bundle.model_path,
        tokenizer_path=user_intent_bundle.tokenizer_path,
        max_length=user_metadata.max_length,
        pooling=user_metadata.pooling,
        output_name=user_metadata.output_name,
        normalize_embeddings=user_metadata.normalize_embeddings,
    )


def _resolve_shared_encoder(
    user_intent_bundle: UserIntentBundle,
    route_bundle: SemanticRouterBundle,
    *,
    user_intent_router: LocalUserIntentRouter | None,
    route_router: LocalSemanticRouter | None,
) -> OnnxSentenceEncoder | None:
    """Prefer an injected shared encoder before building one from bundle files."""

    injected_encoder = getattr(user_intent_router, "encoder", None)
    if injected_encoder is not None and injected_encoder is getattr(route_router, "encoder", None):
        return injected_encoder
    return _build_shared_encoder(user_intent_bundle, route_bundle)