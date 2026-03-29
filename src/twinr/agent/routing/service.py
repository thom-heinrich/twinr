"""Run local semantic-route inference from an ONNX sentence encoder bundle."""

from __future__ import annotations

# CHANGELOG: 2026-03-27
# BUG-1: Fixed constrained-routing scoring. `allowed_labels` are now applied before
#        winner selection, confidence, margin, and score computation, so route filters
#        no longer produce negative margins or falsely low-confidence decisions.
# BUG-2: Added fail-fast validation for temperature, non-finite classifier parameters,
#        label uniqueness, and embedding/classifier dimensionality to prevent late
#        runtime crashes and silent NaN outputs from corrupt or mismatched bundles.
# SEC-1: Added bounded pre-tokenization input truncation to reduce practical CPU/RAM
#        denial-of-service risk from extremely large transcripts on Raspberry Pi.
# IMP-1: Added vectorized batch APIs (`classify_texts`, `classify_embeddings_batch`)
#        so ONNX encoding and classifier scoring can be amortized across multiple texts.
# IMP-2: Added metadata-driven encoder-kwargs passthrough and optional prefix
#        truncation hooks for ORT-tuned / Matryoshka-style bundles without breaking
#        older encoder implementations.

import inspect
import os
import time
from collections.abc import Mapping, Sequence

import numpy as np

from .bundle import SemanticRouterBundle
from .contracts import SemanticRouteDecision
from .inference import OnnxSentenceEncoder, _coerce_embedding_row, _l2_normalize
from .policy import SemanticRouterPolicy

# BREAKING: inputs longer than this limit are truncated before tokenization by default
# to bound worst-case CPU/RAM usage on edge devices. Set
# TWINR_LOCAL_SEMANTIC_ROUTER_MAX_INPUT_CHARS=0 to disable.
DEFAULT_MAX_INPUT_CHARS = 4096


class LocalSemanticRouter:
    """Classify transcripts into `parametric/web/memory/tool` locally."""

    def __init__(
        self,
        bundle: SemanticRouterBundle,
        *,
        encoder: OnnxSentenceEncoder | None = None,
        policy: SemanticRouterPolicy | None = None,
        centroids: np.ndarray | None = None,
    ) -> None:
        self.bundle = bundle
        self.metadata = bundle.metadata
        self.labels = tuple(str(label) for label in self.metadata.labels)
        if not self.labels:
            raise ValueError("Semantic router bundles must define at least one label.")
        self._label_to_index = self._build_label_index(self.labels)
        self.policy = policy or self.metadata.policy()
        self.max_input_chars = self._resolve_max_input_chars()
        self._configured_truncate_dim = self._resolve_configured_truncate_dim()
        self._allow_prefix_truncation = self._resolve_allow_prefix_truncation()
        self.encoder = encoder or self._build_encoder(bundle)

        temperature = float(self.metadata.temperature)
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("Semantic router temperature must be finite and > 0.")
        self._temperature = temperature

        self._centroids: np.ndarray | None = None
        self._weights: np.ndarray | None = None
        self._bias: np.ndarray | None = None

        if str(self.metadata.classifier_type or "").strip() == "embedding_centroid_v1":
            centroids_path = bundle.centroids_path
            if centroids_path is None and centroids is None:
                raise ValueError("Centroid router bundles must provide centroids.npy.")
            if centroids is not None:
                centroid_matrix = np.asarray(centroids, dtype=np.float32)
            else:
                assert centroids_path is not None
                centroid_matrix = np.asarray(
                    np.load(centroids_path, allow_pickle=False),
                    dtype=np.float32,
                )
            if centroid_matrix.ndim != 2:
                raise ValueError("Semantic router centroids must be a rank-2 matrix.")
            if centroid_matrix.shape[0] != len(self.labels):
                raise ValueError(
                    "Semantic router centroid rows must match the configured label order."
                )
            if not np.isfinite(centroid_matrix).all():
                raise ValueError("Semantic router centroids must contain only finite values.")
            if self.metadata.normalize_centroids:
                centroid_matrix = _l2_normalize(centroid_matrix)
            self._centroids = np.ascontiguousarray(centroid_matrix, dtype=np.float32)
            self._feature_dim = int(self._centroids.shape[1])
        else:
            if bundle.weights_path is None or bundle.bias_path is None:
                raise ValueError(
                    "Linear router bundles must provide weights.npy and bias.npy."
                )
            weight_matrix = np.asarray(
                np.load(bundle.weights_path, allow_pickle=False),
                dtype=np.float32,
            )
            bias_vector = np.asarray(
                np.load(bundle.bias_path, allow_pickle=False),
                dtype=np.float32,
            )
            if weight_matrix.ndim != 2:
                raise ValueError("Semantic router linear weights must be a rank-2 matrix.")
            if weight_matrix.shape[0] != len(self.labels):
                raise ValueError(
                    "Semantic router linear weights must match the configured label order."
                )
            if bias_vector.shape not in {
                (len(self.labels),),
                (len(self.labels), 1),
            }:
                raise ValueError(
                    "Semantic router linear bias must have one entry per label."
                )
            if not np.isfinite(weight_matrix).all() or not np.isfinite(bias_vector).all():
                raise ValueError(
                    "Semantic router linear weights and bias must contain only finite values."
                )
            self._weights = np.ascontiguousarray(weight_matrix, dtype=np.float32)
            self._bias = np.ascontiguousarray(
                bias_vector.reshape(-1),
                dtype=np.float32,
            )
            self._feature_dim = int(self._weights.shape[1])

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Expose the underlying embedding path for bundle calibration tools."""

        return self.encoder.encode(texts)

    def warmup(self, probe_text: str = "warmup") -> None:
        """Prime the local route encoder before the first live turn."""

        warmup = getattr(self.encoder, "warmup", None)
        if callable(warmup):
            try:
                warmup(probe_text)
            except TypeError:
                warmup()
            return
        self.classify(probe_text)

    def classify(
        self,
        text: str,
        *,
        policy: SemanticRouterPolicy | None = None,
        allowed_labels: Sequence[str] | None = None,
    ) -> SemanticRouteDecision:
        """Return one scored local semantic route decision."""

        cleaned = self._prepare_text(text)
        if not cleaned:
            raise ValueError("LocalSemanticRouter.classify requires non-empty text.")
        started = time.perf_counter()
        embedding = self.encoder.encode([cleaned])
        latency_ms = (time.perf_counter() - started) * 1000.0
        return self.classify_embedding(
            embedding,
            policy=policy,
            latency_ms=latency_ms,
            allowed_labels=allowed_labels,
        )

    def classify_texts(
        self,
        texts: Sequence[str],
        *,
        policy: SemanticRouterPolicy | None = None,
        allowed_labels: Sequence[str] | None = None,
    ) -> list[SemanticRouteDecision]:
        """Batch-route multiple texts through one encoder call."""

        cleaned_texts = [self._prepare_text(text) for text in texts]
        if not cleaned_texts:
            return []
        if any(not text for text in cleaned_texts):
            raise ValueError("LocalSemanticRouter.classify_texts requires non-empty texts.")
        started = time.perf_counter()
        embeddings = self.encoder.encode(cleaned_texts)
        total_latency_ms = (time.perf_counter() - started) * 1000.0
        return self.classify_embeddings_batch(
            embeddings,
            policy=policy,
            total_latency_ms=total_latency_ms,
            allowed_labels=allowed_labels,
        )

    def classify_embedding(
        self,
        embedding: np.ndarray,
        *,
        policy: SemanticRouterPolicy | None = None,
        latency_ms: float,
        allowed_labels: Sequence[str] | None = None,
    ) -> SemanticRouteDecision:
        """Return one route decision for an already encoded embedding."""

        normalized_embedding = self._prepare_embedding_row(embedding)
        probabilities = self._probabilities_for_row(
            normalized_embedding[0],
            allowed_labels=allowed_labels,
        )
        top_index = int(np.argmax(probabilities))
        second_probability = self._second_best_probability(
            probabilities,
            top_index=top_index,
        )
        decided_policy = policy or self.policy
        decision = SemanticRouteDecision(
            label=self.labels[top_index],
            confidence=float(probabilities[top_index]),
            margin=float(probabilities[top_index] - second_probability),
            scores=self._scores_dict(
                probabilities,
                allowed_labels=allowed_labels,
            ),
            model_id=self.metadata.model_id,
            latency_ms=max(0.0, float(latency_ms)),
        )
        return decision.with_policy(decided_policy)

    def classify_embeddings_batch(
        self,
        embeddings: np.ndarray,
        *,
        policy: SemanticRouterPolicy | None = None,
        total_latency_ms: float,
        allowed_labels: Sequence[str] | None = None,
    ) -> list[SemanticRouteDecision]:
        """Vectorized batch classification for precomputed embeddings."""

        prepared = self._prepare_embedding_batch(embeddings)
        probabilities = self._probabilities_for_batch(
            prepared,
            allowed_labels=allowed_labels,
        )
        row_latency_ms = max(0.0, float(total_latency_ms)) / float(len(prepared))
        decided_policy = policy or self.policy
        decisions: list[SemanticRouteDecision] = []
        for row in probabilities:
            top_index = int(np.argmax(row))
            second_probability = self._second_best_probability(
                row,
                top_index=top_index,
            )
            decisions.append(
                SemanticRouteDecision(
                    label=self.labels[top_index],
                    confidence=float(row[top_index]),
                    margin=float(row[top_index] - second_probability),
                    scores=self._scores_dict(
                        row,
                        allowed_labels=allowed_labels,
                    ),
                    model_id=self.metadata.model_id,
                    latency_ms=row_latency_ms,
                ).with_policy(decided_policy)
            )
        return decisions

    def _classifier_logits(self, embedding_row: np.ndarray) -> np.ndarray:
        """Return one per-label logit vector for a single normalized embedding."""

        if self._centroids is not None:
            return np.matmul(self._centroids, embedding_row)
        if self._weights is None or self._bias is None:
            raise RuntimeError("Linear router state was not initialized.")
        return np.matmul(self._weights, embedding_row) + self._bias

    def _classifier_logits_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """Return one per-label logit vector for each normalized embedding row."""

        if self._centroids is not None:
            return np.matmul(embeddings, self._centroids.T)
        if self._weights is None or self._bias is None:
            raise RuntimeError("Linear router state was not initialized.")
        return np.matmul(embeddings, self._weights.T) + self._bias

    def _prepare_text(self, text: str) -> str:
        cleaned = str(text or "").strip()
        if self.max_input_chars is not None and len(cleaned) > self.max_input_chars:
            cleaned = cleaned[: self.max_input_chars]
        return cleaned

    def _prepare_embedding_row(self, embedding: np.ndarray) -> np.ndarray:
        prepared = _coerce_embedding_row(embedding)
        prepared = self._align_embedding_dimensions(prepared)
        if self.metadata.normalize_embeddings:
            prepared = _l2_normalize(prepared)
        prepared = np.asarray(prepared, dtype=np.float32)
        if not np.isfinite(prepared).all():
            raise ValueError("Semantic router embeddings must contain only finite values.")
        return prepared

    def _prepare_embedding_batch(self, embeddings: np.ndarray) -> np.ndarray:
        prepared = np.asarray(embeddings, dtype=np.float32)
        if prepared.ndim == 1:
            prepared = prepared.reshape(1, -1)
        if prepared.ndim != 2 or prepared.shape[0] == 0:
            raise ValueError(
                "Semantic router batch embeddings must be a non-empty rank-2 matrix."
            )
        prepared = self._align_embedding_dimensions(prepared)
        if self.metadata.normalize_embeddings:
            prepared = _l2_normalize(prepared)
        prepared = np.ascontiguousarray(prepared, dtype=np.float32)
        if not np.isfinite(prepared).all():
            raise ValueError("Semantic router embeddings must contain only finite values.")
        return prepared

    def _align_embedding_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        current_dim = int(embeddings.shape[1])
        if current_dim == self._feature_dim:
            return embeddings
        if current_dim > self._feature_dim and self._can_prefix_truncate_embeddings():
            return embeddings[:, : self._feature_dim]
        raise ValueError(
            "Semantic router embedding dimension does not match classifier input "
            f"dimension: got {current_dim}, expected {self._feature_dim}."
        )

    def _can_prefix_truncate_embeddings(self) -> bool:
        return self._allow_prefix_truncation or (
            self._configured_truncate_dim is not None
            and self._configured_truncate_dim == self._feature_dim
        )

    def _probabilities_for_row(
        self,
        embedding_row: np.ndarray,
        *,
        allowed_labels: Sequence[str] | None,
    ) -> np.ndarray:
        logits = self._classifier_logits(embedding_row)
        allowed_mask = self._allowed_mask(allowed_labels)
        if allowed_mask is not None:
            logits = np.where(allowed_mask, logits, -np.inf)
        return self._softmax_vector(logits / self._temperature)

    def _probabilities_for_batch(
        self,
        embeddings: np.ndarray,
        *,
        allowed_labels: Sequence[str] | None,
    ) -> np.ndarray:
        logits = self._classifier_logits_batch(embeddings)
        allowed_mask = self._allowed_mask(allowed_labels)
        if allowed_mask is not None:
            logits = np.where(allowed_mask.reshape(1, -1), logits, -np.inf)
        return self._softmax_rows(logits / self._temperature)

    def _allowed_mask(self, allowed_labels: Sequence[str] | None) -> np.ndarray | None:
        if not allowed_labels:
            return None
        normalized_allowed = {
            str(value or "").strip().casefold()
            for value in allowed_labels
            if str(value or "").strip()
        }
        if not normalized_allowed:
            raise ValueError(
                "LocalSemanticRouter.classify requires at least one valid allowed label."
            )
        indices = [
            self._label_to_index[label]
            for label in normalized_allowed
            if label in self._label_to_index
        ]
        if not indices:
            raise ValueError(
                "LocalSemanticRouter.classify requires at least one valid allowed label."
            )
        mask = np.zeros(len(self.labels), dtype=bool)
        mask[indices] = True
        return mask

    def _scores_dict(
        self,
        probabilities: np.ndarray,
        *,
        allowed_labels: Sequence[str] | None,
    ) -> dict[str, float]:
        allowed_mask = self._allowed_mask(allowed_labels)
        if allowed_mask is None:
            return {
                label: float(probabilities[index])
                for index, label in enumerate(self.labels)
            }
        return {
            label: float(probabilities[index]) if allowed_mask[index] else 0.0
            for index, label in enumerate(self.labels)
        }

    @staticmethod
    def _softmax_vector(logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float32)
        finite = np.isfinite(logits)
        if not finite.any():
            raise ValueError("Semantic router logits contained no finite values after masking.")
        max_logit = float(np.max(logits[finite]))
        stabilized = np.where(finite, logits - max_logit, -np.inf)
        exp_logits = np.where(finite, np.exp(stabilized), 0.0)
        denominator = float(np.sum(exp_logits))
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError(
                "Semantic router softmax produced an invalid normalization constant."
            )
        return (exp_logits / denominator).astype(np.float32, copy=False)

    @classmethod
    def _softmax_rows(cls, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float32)
        if logits.ndim != 2:
            raise ValueError("Semantic router batch logits must be a rank-2 matrix.")
        probabilities = np.empty_like(logits, dtype=np.float32)
        for row_index in range(logits.shape[0]):
            probabilities[row_index] = cls._softmax_vector(logits[row_index])
        return probabilities

    @staticmethod
    def _second_best_probability(probabilities: np.ndarray, *, top_index: int) -> float:
        if probabilities.ndim != 1:
            raise ValueError("Semantic router probabilities must be rank-1.")
        if probabilities.shape[0] <= 1:
            return 0.0
        mask = np.ones(probabilities.shape[0], dtype=bool)
        mask[top_index] = False
        remaining = probabilities[mask]
        if remaining.size == 0:
            return 0.0
        return float(np.max(remaining))

    @staticmethod
    def _build_label_index(labels: Sequence[str]) -> dict[str, int]:
        label_to_index: dict[str, int] = {}
        for index, label in enumerate(labels):
            normalized = str(label).strip().casefold()
            if not normalized:
                raise ValueError("Semantic router labels must not be empty.")
            if normalized in label_to_index:
                raise ValueError("Semantic router labels must be unique ignoring case.")
            label_to_index[normalized] = index
        return label_to_index

    def _build_encoder(self, bundle: SemanticRouterBundle) -> OnnxSentenceEncoder:
        base_kwargs: dict[str, object] = {
            "model_path": bundle.model_path,
            "tokenizer_path": bundle.tokenizer_path,
            "max_length": self.metadata.max_length,
            "pooling": self.metadata.pooling,
            "output_name": self.metadata.output_name,
            "normalize_embeddings": self.metadata.normalize_embeddings,
        }
        optional_kwargs: dict[str, object] = {}
        encoder_kwargs = getattr(self.metadata, "encoder_kwargs", None)
        if isinstance(encoder_kwargs, Mapping):
            optional_kwargs.update(encoder_kwargs)

        for name in (
            "providers",
            "provider_options",
            "session_options",
            "intra_op_num_threads",
            "inter_op_num_threads",
            "graph_optimization_level",
            "execution_mode",
            "backend",
            "truncate_dim",
            "num_hidden_layers",
        ):
            value = getattr(self.metadata, name, None)
            if value is not None:
                optional_kwargs[name] = value

        kwargs = {**base_kwargs, **optional_kwargs}
        signature = inspect.signature(OnnxSentenceEncoder)
        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if accepts_var_kwargs:
            return OnnxSentenceEncoder(**kwargs)

        supported_kwargs = {
            name: value
            for name, value in kwargs.items()
            if name in signature.parameters
        }
        return OnnxSentenceEncoder(**supported_kwargs)

    def _resolve_max_input_chars(self) -> int | None:
        raw_value = getattr(self.metadata, "max_input_chars", None)
        if raw_value is None:
            raw_value = os.getenv(
                "TWINR_LOCAL_SEMANTIC_ROUTER_MAX_INPUT_CHARS",
                str(DEFAULT_MAX_INPUT_CHARS),
            )
        if raw_value in (None, "", False):
            return None
        value = int(raw_value)
        if value <= 0:
            return None
        return value

    def _resolve_configured_truncate_dim(self) -> int | None:
        for attr_name in ("truncate_dim", "embedding_dim", "classifier_input_dim"):
            value = getattr(self.metadata, attr_name, None)
            if value is None:
                continue
            parsed = int(value)
            if parsed > 0:
                return parsed

        encoder_kwargs = getattr(self.metadata, "encoder_kwargs", None)
        if isinstance(encoder_kwargs, Mapping):
            for key in ("truncate_dim", "embedding_dim", "classifier_input_dim"):
                value = encoder_kwargs.get(key)
                if value is None:
                    continue
                parsed = int(value)
                if parsed > 0:
                    return parsed
        return None

    def _resolve_allow_prefix_truncation(self) -> bool:
        for attr_name in (
            "allow_prefix_truncation",
            "allow_embedding_truncation",
            "matryoshka",
            "adaptive_layers",
        ):
            value = getattr(self.metadata, attr_name, None)
            if value is not None:
                return bool(value)

        env_value = os.getenv(
            "TWINR_LOCAL_SEMANTIC_ROUTER_ALLOW_PREFIX_TRUNCATION",
            "",
        )
        return env_value.strip().lower() in {"1", "true", "yes", "on"}
