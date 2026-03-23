"""Run local semantic-route inference from an ONNX sentence encoder bundle."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Sequence

import numpy as np

from .bundle import SemanticRouterBundle
from .contracts import SemanticRouteDecision
from .policy import SemanticRouterPolicy


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Return a row-wise L2-normalized copy of one embedding matrix."""

    array = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    safe_norms = np.where(norms <= 1e-12, 1.0, norms)
    return array / safe_norms


@dataclass(slots=True)
class OnnxSentenceEncoder:
    """Encode text with a local ONNX sentence model plus `tokenizer.json`.

    The encoder intentionally loads heavyweight tokenizer/runtime dependencies
    lazily so tests and non-router deployments do not need them until the local
    router is actually enabled.
    """

    model_path: Path
    tokenizer_path: Path
    max_length: int
    pooling: str = "mean"
    output_name: str | None = None
    normalize_embeddings: bool = True
    _tokenizer: object | None = field(init=False, default=None, repr=False)
    _session: object | None = field(init=False, default=None, repr=False)
    _session_input_names: tuple[str, ...] | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path)
        self.tokenizer_path = Path(self.tokenizer_path)
        self.max_length = max(1, int(self.max_length))
        self.pooling = str(self.pooling or "mean").strip().lower()
        if self.pooling not in {"mean", "cls", "prepooled"}:
            raise ValueError(f"Unsupported ONNX sentence pooling mode: {self.pooling}")
        self.output_name = str(self.output_name or "").strip() or None
        self.normalize_embeddings = bool(self.normalize_embeddings)

    def _load_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local semantic routing requires the `tokenizers` package."
            ) from exc
        self._tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        self._tokenizer.enable_truncation(max_length=self.max_length)
        return self._tokenizer

    def _load_session(self):
        if self._session is not None:
            return self._session
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "Local semantic routing requires the `onnxruntime` package."
            ) from exc
        session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        self._session = session
        self._session_input_names = tuple(item.name for item in session.get_inputs())
        return session

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return one embedding per input text."""

        normalized_texts = [str(text or "").strip() for text in texts]
        if not normalized_texts:
            raise ValueError("OnnxSentenceEncoder.encode requires at least one text.")
        tokenizer = self._load_tokenizer()
        session = self._load_session()
        encodings = tokenizer.encode_batch(normalized_texts)
        seq_len = max(1, max(len(encoding.ids) for encoding in encodings))
        input_ids = np.zeros((len(encodings), seq_len), dtype=np.int64)
        attention_mask = np.zeros((len(encodings), seq_len), dtype=np.int64)
        token_type_ids = np.zeros((len(encodings), seq_len), dtype=np.int64)
        for row_index, encoding in enumerate(encodings):
            ids = np.asarray(encoding.ids[:seq_len], dtype=np.int64)
            mask = np.asarray(encoding.attention_mask[:seq_len], dtype=np.int64)
            type_ids = np.asarray(encoding.type_ids[:seq_len], dtype=np.int64)
            width = int(ids.shape[0])
            input_ids[row_index, :width] = ids
            attention_mask[row_index, :width] = mask
            if type_ids.size:
                token_type_ids[row_index, :width] = type_ids
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        input_names = self._session_input_names or tuple(item.name for item in session.get_inputs())
        if "token_type_ids" in input_names:
            inputs["token_type_ids"] = token_type_ids
        output_names = [self.output_name] if self.output_name else None
        outputs = session.run(output_names, inputs)
        if not outputs:
            raise RuntimeError("Semantic router ONNX session returned no outputs.")
        raw_embeddings = np.asarray(outputs[0], dtype=np.float32)
        embeddings = self._pool_embeddings(raw_embeddings, attention_mask)
        if self.normalize_embeddings:
            embeddings = _l2_normalize(embeddings)
        return embeddings.astype(np.float32, copy=False)

    def _pool_embeddings(
        self,
        raw_embeddings: np.ndarray,
        attention_mask: np.ndarray,
    ) -> np.ndarray:
        """Return fixed-size sentence embeddings from one ONNX output tensor."""

        if raw_embeddings.ndim == 2:
            return raw_embeddings
        if raw_embeddings.ndim != 3:
            raise ValueError(
                f"Unsupported ONNX semantic-router output rank: {raw_embeddings.ndim}"
            )
        if self.pooling == "cls":
            return raw_embeddings[:, 0, :]
        if self.pooling == "prepooled":
            raise ValueError("Prepooled routing bundles must emit rank-2 embeddings.")
        mask = attention_mask.astype(np.float32)[..., None]
        summed = np.sum(raw_embeddings * mask, axis=1)
        counts = np.clip(np.sum(mask, axis=1), 1.0, None)
        return summed / counts


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
            if bundle.centroids_path is None and centroids is None:
                raise ValueError("Centroid router bundles must provide centroids.npy.")
            centroid_matrix = (
                np.asarray(centroids, dtype=np.float32)
                if centroids is not None
                else np.load(bundle.centroids_path).astype(np.float32)
            )
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


def _softmax(values: np.ndarray) -> np.ndarray:
    """Return a numerically-stable softmax vector."""

    shifted = np.asarray(values, dtype=np.float32) - float(np.max(values))
    exp_values = np.exp(shifted)
    denominator = float(np.sum(exp_values))
    if denominator <= 1e-12:
        return np.full_like(exp_values, 1.0 / float(exp_values.shape[0]))
    return exp_values / denominator


def _coerce_embedding_row(embedding: np.ndarray) -> np.ndarray:
    """Return one rank-2 single-row embedding matrix."""

    array = np.asarray(embedding, dtype=np.float32)
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim != 2 or array.shape[0] != 1:
        raise ValueError("Local semantic routing expects exactly one embedding row.")
    return array


def _second_index_from_probabilities(probabilities: np.ndarray, top_index: int) -> int:
    """Return the strongest competing index in the full probability vector."""

    if probabilities.shape[0] <= 1:
        return int(top_index)
    ordered = np.argsort(probabilities)[::-1]
    for candidate in ordered:
        if int(candidate) != int(top_index):
            return int(candidate)
    return int(top_index)


def _mask_probabilities(
    probabilities: np.ndarray,
    *,
    labels: Sequence[str],
    allowed_labels: Sequence[str] | None,
) -> dict[str, float]:
    """Return raw scores with disallowed labels zeroed out when constrained."""

    if not allowed_labels:
        return {
            label: float(probabilities[index])
            for index, label in enumerate(labels)
        }
    normalized_allowed = {
        str(value or "").strip().lower()
        for value in allowed_labels
        if str(value or "").strip()
    }
    return {
        label: (float(probabilities[index]) if label in normalized_allowed else 0.0)
        for index, label in enumerate(labels)
    }
