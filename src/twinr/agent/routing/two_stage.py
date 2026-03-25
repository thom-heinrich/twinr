"""Run the user-centered stage and backend route stage on one shared embedding."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

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

    def __post_init__(self) -> None:
        self.metadata = self.bundle.metadata
        self.encoder = self.encoder or OnnxSentenceEncoder(
            model_path=self.bundle.model_path,
            tokenizer_path=self.bundle.tokenizer_path,
            max_length=self.metadata.max_length,
            pooling=self.metadata.pooling,
            output_name=self.metadata.output_name,
            normalize_embeddings=self.metadata.normalize_embeddings,
        )
        self._centroids: np.ndarray | None = None
        self._weights: np.ndarray | None = None
        self._bias: np.ndarray | None = None
        if self.metadata.classifier_type == "embedding_centroid_v1":
            centroids_path = self.bundle.centroids_path
            if centroids_path is None and self.centroids is None:
                raise ValueError("Centroid user-intent bundles must provide centroids.npy.")
            if self.centroids is not None:
                centroid_matrix = np.asarray(self.centroids, dtype=np.float32)
            else:
                assert centroids_path is not None
                centroid_matrix = np.load(centroids_path).astype(np.float32)
            if centroid_matrix.ndim != 2:
                raise ValueError("User intent centroids must be a rank-2 matrix.")
            if centroid_matrix.shape[0] != len(self.metadata.labels):
                raise ValueError(
                    "User intent centroid rows must match the configured label order."
                )
            if self.metadata.normalize_centroids:
                centroid_matrix = _l2_normalize(centroid_matrix)
            self._centroids = centroid_matrix
        else:
            if self.bundle.weights_path is None or self.bundle.bias_path is None:
                raise ValueError("Linear user-intent bundles must provide weights.npy and bias.npy.")
            weight_matrix = np.load(self.bundle.weights_path).astype(np.float32)
            bias_vector = np.load(self.bundle.bias_path).astype(np.float32)
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

    def classify(self, text: str) -> UserIntentDecision:
        """Return one scored local user-intent decision."""

        cleaned = str(text or "").strip()
        if not cleaned:
            raise ValueError("LocalUserIntentRouter.classify requires non-empty text.")
        started = time.perf_counter()
        encoder = self.encoder
        if encoder is None:
            raise RuntimeError("LocalUserIntentRouter encoder was not initialized.")
        embedding = encoder.encode([cleaned])[0:1]
        return self.classify_embedding(
            embedding,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def classify_embedding(
        self,
        embedding: np.ndarray,
        *,
        latency_ms: float,
    ) -> UserIntentDecision:
        """Return one user-intent decision for an already encoded embedding."""

        normalized_embedding = _coerce_embedding_row(embedding)
        if self.metadata.normalize_embeddings:
            normalized_embedding = _l2_normalize(normalized_embedding)
        logits = self._classifier_logits(normalized_embedding[0])
        probabilities = _softmax(logits / self.metadata.temperature)
        top_indices = np.argsort(probabilities)[::-1]
        top_index = int(top_indices[0])
        second_index = int(top_indices[1]) if probabilities.shape[0] > 1 else top_index
        return UserIntentDecision(
            label=self.metadata.labels[top_index],
            confidence=float(probabilities[top_index]),
            margin=float(probabilities[top_index] - probabilities[second_index]),
            scores={
                label: float(probabilities[index])
                for index, label in enumerate(self.metadata.labels)
            },
            model_id=self.metadata.model_id,
            latency_ms=float(latency_ms),
        )

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
            user_intent = self.user_intent_router.classify(cleaned)
            allowed_route_labels = allowed_route_labels_for_user_intent(user_intent.label)
            route_decision = self.route_router.classify(
                cleaned,
                policy=policy,
                allowed_labels=allowed_route_labels,
            )
            return TwoStageSemanticRouteDecision(
                user_intent=user_intent,
                route_decision=route_decision,
                allowed_route_labels=allowed_route_labels,
            )
        started = time.perf_counter()
        embedding = self.shared_encoder.encode([cleaned])[0:1]
        user_intent = self.user_intent_router.classify_embedding(
            embedding,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
        allowed_route_labels = allowed_route_labels_for_user_intent(user_intent.label)
        route_decision = self.route_router.classify_embedding(
            embedding,
            policy=policy,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            allowed_labels=allowed_route_labels,
        )
        return TwoStageSemanticRouteDecision(
            user_intent=user_intent,
            route_decision=route_decision,
            allowed_route_labels=allowed_route_labels,
        )


def _build_shared_encoder(
    user_intent_bundle: UserIntentBundle,
    route_bundle: SemanticRouterBundle,
) -> OnnxSentenceEncoder | None:
    """Return one shared encoder when both stages use compatible model settings."""

    user_metadata = user_intent_bundle.metadata
    route_metadata = route_bundle.metadata
    shared_shape = (
        user_metadata.max_length == route_metadata.max_length
        and user_metadata.pooling == route_metadata.pooling
        and user_metadata.output_name == route_metadata.output_name
        and user_metadata.normalize_embeddings == route_metadata.normalize_embeddings
    )
    if not shared_shape:
        return None
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
):
    """Prefer an injected shared encoder before building one from bundle files."""

    injected_encoder = getattr(user_intent_router, "encoder", None)
    if injected_encoder is not None and injected_encoder is getattr(route_router, "encoder", None):
        return injected_encoder
    return _build_shared_encoder(user_intent_bundle, route_bundle)
