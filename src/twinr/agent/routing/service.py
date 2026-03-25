"""Run local semantic-route inference from an ONNX sentence encoder bundle."""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from .bundle import SemanticRouterBundle
from .contracts import SemanticRouteDecision
from .inference import (
    OnnxSentenceEncoder,
    _coerce_embedding_row,
    _l2_normalize,
    _mask_probabilities,
    _second_index_from_probabilities,
    _softmax,
)
from .policy import SemanticRouterPolicy


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
        self.policy = policy or self.metadata.policy()
        self.encoder = encoder or OnnxSentenceEncoder(
            model_path=bundle.model_path,
            tokenizer_path=bundle.tokenizer_path,
            max_length=self.metadata.max_length,
            pooling=self.metadata.pooling,
            output_name=self.metadata.output_name,
            normalize_embeddings=self.metadata.normalize_embeddings,
        )
        self._centroids: np.ndarray | None = None
        self._weights: np.ndarray | None = None
        self._bias: np.ndarray | None = None
        if self.metadata.classifier_type == "embedding_centroid_v1":
            centroids_path = bundle.centroids_path
            if centroids_path is None and centroids is None:
                raise ValueError("Centroid router bundles must provide centroids.npy.")
            if centroids is not None:
                centroid_matrix = np.asarray(centroids, dtype=np.float32)
            else:
                assert centroids_path is not None
                centroid_matrix = np.load(centroids_path).astype(np.float32)
            if centroid_matrix.ndim != 2:
                raise ValueError("Semantic router centroids must be a rank-2 matrix.")
            if centroid_matrix.shape[0] != len(self.metadata.labels):
                raise ValueError(
                    "Semantic router centroid rows must match the configured label order."
                )
            if self.metadata.normalize_centroids:
                centroid_matrix = _l2_normalize(centroid_matrix)
            self._centroids = centroid_matrix
        else:
            if bundle.weights_path is None or bundle.bias_path is None:
                raise ValueError("Linear router bundles must provide weights.npy and bias.npy.")
            weight_matrix = np.load(bundle.weights_path).astype(np.float32)
            bias_vector = np.load(bundle.bias_path).astype(np.float32)
            if weight_matrix.ndim != 2:
                raise ValueError("Semantic router linear weights must be a rank-2 matrix.")
            if weight_matrix.shape[0] != len(self.metadata.labels):
                raise ValueError(
                    "Semantic router linear weights must match the configured label order."
                )
            if bias_vector.shape not in {(len(self.metadata.labels),), (len(self.metadata.labels), 1)}:
                raise ValueError("Semantic router linear bias must have one entry per label.")
            self._weights = weight_matrix
            self._bias = bias_vector.reshape(-1)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Expose the underlying embedding path for bundle calibration tools."""

        return self.encoder.encode(texts)

    def classify(
        self,
        text: str,
        *,
        policy: SemanticRouterPolicy | None = None,
        allowed_labels: Sequence[str] | None = None,
    ) -> SemanticRouteDecision:
        """Return one scored local semantic route decision."""

        cleaned = str(text or "").strip()
        if not cleaned:
            raise ValueError("LocalSemanticRouter.classify requires non-empty text.")
        started = time.perf_counter()
        embedding = self.encoder.encode([cleaned])[0:1]
        return self.classify_embedding(
            embedding,
            policy=policy,
            latency_ms=(time.perf_counter() - started) * 1000.0,
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

        normalized_embedding = _coerce_embedding_row(embedding)
        if self.metadata.normalize_embeddings:
            normalized_embedding = _l2_normalize(normalized_embedding)
        logits = self._classifier_logits(normalized_embedding[0])
        probabilities = _softmax(logits / self.metadata.temperature)
        top_index = self._select_top_index(probabilities, allowed_labels=allowed_labels)
        second_index = _second_index_from_probabilities(probabilities, top_index)
        decided_policy = policy or self.policy
        decision = SemanticRouteDecision(
            label=self.metadata.labels[top_index],
            confidence=float(probabilities[top_index]),
            margin=float(probabilities[top_index] - probabilities[second_index]),
            scores=_mask_probabilities(
                probabilities,
                labels=self.metadata.labels,
                allowed_labels=allowed_labels,
            ),
            model_id=self.metadata.model_id,
            latency_ms=float(latency_ms),
        )
        return decision.with_policy(decided_policy)

    def _classifier_logits(self, embedding_row: np.ndarray) -> np.ndarray:
        """Return one per-label logit vector for a single normalized embedding."""

        if self.metadata.classifier_type == "embedding_centroid_v1":
            if self._centroids is None:
                raise RuntimeError("Centroid router state was not initialized.")
            return np.matmul(self._centroids, embedding_row)
        if self._weights is None or self._bias is None:
            raise RuntimeError("Linear router state was not initialized.")
        return np.matmul(self._weights, embedding_row) + self._bias

    def _select_top_index(
        self,
        probabilities: np.ndarray,
        *,
        allowed_labels: Sequence[str] | None,
    ) -> int:
        """Return the winning centroid index under optional label constraints."""

        if not allowed_labels:
            return int(np.argsort(probabilities)[::-1][0])
        normalized_allowed = {
            str(value or "").strip().lower()
            for value in allowed_labels
            if str(value or "").strip()
        }
        indices = [
            index
            for index, label in enumerate(self.metadata.labels)
            if label in normalized_allowed
        ]
        if not indices:
            raise ValueError(
                "LocalSemanticRouter.classify_embedding requires at least one valid allowed label."
            )
        return max(indices, key=lambda index: (float(probabilities[index]), -index))
