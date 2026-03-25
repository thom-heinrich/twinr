"""Shared ONNX sentence-encoder and numeric helpers for local routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Return a row-wise L2-normalized copy of one embedding matrix."""

    array = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    safe_norms = np.where(norms <= 1e-12, 1.0, norms)
    return array / safe_norms


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


@dataclass(slots=True)
class OnnxSentenceEncoder:
    """Encode text with a local ONNX sentence model plus `tokenizer.json`.

    The encoder intentionally loads heavyweight tokenizer/runtime dependencies
    lazily so tests and non-router deployments do not need them until local
    routing is actually enabled.
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
