"""Shared ONNX sentence-encoder and numeric helpers for local routing."""

from __future__ import annotations

# CHANGELOG: 2026-03-27
# BUG-1: Fixed constrained-label masking so case/whitespace differences no longer silently zero out valid labels.
# BUG-2: Fixed lazy tokenizer/session initialization race under concurrent access, which could duplicate heavyweight loads on multi-threaded routers.
# BUG-3: Fixed ONNX input incompatibilities by casting tensors to the dtypes declared by the model (e.g. int32 vs int64) and by supporting common input aliases.
# SEC-1: Added local artifact validation plus optional SHA-256 integrity checks for model/tokenizer files before loading.
# SEC-2: Added bounded chunked encoding to prevent memory spikes / practical DoS from oversized batches on Raspberry Pi deployments.
# IMP-1: Upgraded tokenization to use tokenizer-native padding plus encode_batch_fast() when available, reducing Python overhead.
# IMP-2: Added edge-oriented ONNX Runtime session tuning knobs, automatic ORT sidecar preference, automatic output selection, and support for more exported sentence-model variants.

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
import threading
from typing import Any, Iterator, Sequence

import numpy as np


_ARTIFACT_HASH_CHUNK_BYTES = 1024 * 1024
_DEFAULT_MAX_MODEL_BYTES = 768 * 1024 * 1024
_DEFAULT_MAX_TOKENIZER_BYTES = 64 * 1024 * 1024
_PRELOADED_ORT_SESSION_LOCK = threading.RLock()
_PRELOADED_ORT_SESSIONS: dict[str, object] = {}


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Return an L2-normalized copy of one embedding vector or matrix."""

    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim != 2:
        raise ValueError(f"_l2_normalize expects rank-1 or rank-2 input, got rank {array.ndim}.")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    safe_norms = np.where(norms <= 1e-12, 1.0, norms)
    return np.divide(array, safe_norms, out=np.empty_like(array), where=safe_norms > 0)


def _softmax(values: np.ndarray) -> np.ndarray:
    """Return a numerically-stable softmax vector."""

    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if vector.size == 0:
        return vector
    shifted = vector - np.max(vector)
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

    vector = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    if vector.shape[0] <= 1:
        return int(top_index)
    ordered = np.argsort(vector)[::-1]
    for candidate in ordered:
        if int(candidate) != int(top_index):
            return int(candidate)
    return int(top_index)


def _normalize_label(value: str) -> str:
    return str(value or "").strip().lower()


def _mask_probabilities(
    probabilities: np.ndarray,
    *,
    labels: Sequence[str],
    allowed_labels: Sequence[str] | None,
) -> dict[str, float]:
    """Return raw scores with disallowed labels zeroed out when constrained."""

    vector = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    if vector.shape[0] != len(labels):
        raise ValueError("Probability vector length must match label count.")
    if not allowed_labels:
        return {label: float(vector[index]) for index, label in enumerate(labels)}
    normalized_allowed = {
        _normalize_label(value)
        for value in allowed_labels
        if _normalize_label(value)
    }
    return {
        label: (
            float(vector[index])
            if _normalize_label(label) in normalized_allowed
            else 0.0
        )
        for index, label in enumerate(labels)
    }


def _iter_chunks(items: Sequence[str], chunk_size: int | None) -> Iterator[Sequence[str]]:
    if not items:
        return
    if chunk_size is None or chunk_size <= 0 or len(items) <= chunk_size:
        yield items
        return
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def _ensure_local_regular_file(
    path: Path,
    *,
    label: str,
    max_bytes: int | None,
) -> Path:
    resolved = path.expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"{label} must point to a regular file: {resolved}")
    size_bytes = resolved.stat().st_size
    if size_bytes <= 0:
        raise ValueError(f"{label} is empty: {resolved}")
    if max_bytes is not None and size_bytes > int(max_bytes):
        raise ValueError(
            f"{label} is too large for this deployment ({size_bytes} bytes > {int(max_bytes)} bytes): {resolved}"
        )
    return resolved


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_ARTIFACT_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file_hash(path: Path, *, label: str, expected_sha256: str | None) -> None:
    if expected_sha256 is None:
        return
    normalized = str(expected_sha256).strip().lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{label} SHA-256 must be a 64-character lowercase hex digest.")
    actual = _sha256_file(path)
    if actual != normalized:
        raise ValueError(
            f"{label} SHA-256 mismatch for {path}. Expected {normalized}, got {actual}."
        )


def _onnx_type_to_numpy_dtype(type_name: str | None) -> np.dtype:
    normalized = str(type_name or "").strip().lower()
    mapping = {
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(int16)": np.int16,
        "tensor(int8)": np.int8,
        "tensor(uint64)": np.uint64,
        "tensor(uint32)": np.uint32,
        "tensor(uint16)": np.uint16,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
    }
    return np.dtype(mapping.get(normalized, np.int64))


def _cache_preloaded_ort_session(model_path: Path, session: object) -> None:
    key = str(model_path.expanduser().resolve(strict=False))
    with _PRELOADED_ORT_SESSION_LOCK:
        _PRELOADED_ORT_SESSIONS[key] = session


def _pop_preloaded_ort_session(model_path: Path) -> object | None:
    key = str(model_path.expanduser().resolve(strict=False))
    with _PRELOADED_ORT_SESSION_LOCK:
        return _PRELOADED_ORT_SESSIONS.pop(key, None)


@dataclass(frozen=True, slots=True)
class _OrtIoSpec:
    name: str
    type_name: str
    shape: tuple[Any, ...]


@dataclass(slots=True)
class OnnxSentenceEncoder:
    """Encode text with a local ONNX/ORT sentence model plus `tokenizer.json`.

    The encoder intentionally loads heavyweight tokenizer/runtime dependencies
    lazily so tests and non-router deployments do not need them until local
    routing is actually enabled.

    Notes
    -----
    * The API stays drop-in compatible with the original class.
    * Extra knobs are optional and target real Raspberry Pi edge deployments.
    """

    model_path: Path
    tokenizer_path: Path
    max_length: int
    pooling: str = "mean"
    output_name: str | None = None
    normalize_embeddings: bool = True

    # 2026 edge/runtime additions (all optional / backwards-compatible)
    batch_size: int | None = 32
    # BREAKING: when both `model.onnx` and `model.ort` exist, the `.ort` sidecar is preferred by default.
    prefer_ort_model: bool = True
    model_sha256: str | None = None
    tokenizer_sha256: str | None = None
    # BREAKING: oversized artifacts are rejected by default to avoid accidental / malicious Pi-4 memory exhaustion.
    max_model_bytes: int | None = _DEFAULT_MAX_MODEL_BYTES
    max_tokenizer_bytes: int | None = _DEFAULT_MAX_TOKENIZER_BYTES
    intra_op_num_threads: int | None = None
    inter_op_num_threads: int | None = None
    # BREAKING: default is now edge-friendly (`False`) instead of ONNX Runtime's CPU-hungry spinning default.
    allow_spinning: bool | None = False
    graph_optimization_level: str = "all"
    execution_mode: str = "sequential"
    pad_to_multiple_of: int | None = None

    _tokenizer: object | None = field(init=False, default=None, repr=False)
    _session: object | None = field(init=False, default=None, repr=False)
    _tokenizer_lock: threading.RLock = field(init=False, default_factory=threading.RLock, repr=False)
    _session_lock: threading.RLock = field(init=False, default_factory=threading.RLock, repr=False)
    _session_input_specs: tuple[_OrtIoSpec, ...] = field(init=False, default=(), repr=False)
    _session_output_specs: tuple[_OrtIoSpec, ...] = field(init=False, default=(), repr=False)
    _model_artifact_path: Path | None = field(init=False, default=None, repr=False)
    _selected_output_name: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path)
        self.tokenizer_path = Path(self.tokenizer_path)
        self.max_length = max(1, int(self.max_length))
        self.pooling = str(self.pooling or "mean").strip().lower()
        if self.pooling not in {"mean", "cls", "prepooled"}:
            raise ValueError(f"Unsupported ONNX sentence pooling mode: {self.pooling}")
        self.output_name = str(self.output_name or "").strip() or None
        self.normalize_embeddings = bool(self.normalize_embeddings)

        if self.batch_size is not None:
            self.batch_size = max(1, int(self.batch_size))
        if self.max_model_bytes is not None:
            self.max_model_bytes = max(1, int(self.max_model_bytes))
        if self.max_tokenizer_bytes is not None:
            self.max_tokenizer_bytes = max(1, int(self.max_tokenizer_bytes))
        if self.intra_op_num_threads is not None:
            self.intra_op_num_threads = max(0, int(self.intra_op_num_threads))
        if self.inter_op_num_threads is not None:
            self.inter_op_num_threads = max(0, int(self.inter_op_num_threads))
        self.graph_optimization_level = str(self.graph_optimization_level or "all").strip().lower()
        if self.graph_optimization_level not in {"disable", "basic", "extended", "all"}:
            raise ValueError(
                "graph_optimization_level must be one of: disable, basic, extended, all."
            )
        self.execution_mode = str(self.execution_mode or "sequential").strip().lower()
        if self.execution_mode not in {"sequential", "parallel"}:
            raise ValueError("execution_mode must be either 'sequential' or 'parallel'.")
        if self.pad_to_multiple_of is not None:
            self.pad_to_multiple_of = max(1, int(self.pad_to_multiple_of))

    def _resolve_model_artifact_path(self) -> Path:
        candidate = self.model_path.expanduser()
        if self.prefer_ort_model and candidate.suffix.lower() == ".onnx":
            ort_candidate = candidate.with_suffix(".ort")
            if ort_candidate.is_file():
                candidate = ort_candidate
        verified = _ensure_local_regular_file(
            candidate,
            label="ONNX sentence model",
            max_bytes=self.max_model_bytes,
        )
        _verify_file_hash(
            verified,
            label="ONNX sentence model",
            expected_sha256=self.model_sha256,
        )
        return verified

    def _resolve_tokenizer_artifact_path(self) -> Path:
        verified = _ensure_local_regular_file(
            self.tokenizer_path.expanduser(),
            label="Tokenizer JSON",
            max_bytes=self.max_tokenizer_bytes,
        )
        _verify_file_hash(
            verified,
            label="Tokenizer JSON",
            expected_sha256=self.tokenizer_sha256,
        )
        return verified

    def _tokenizer_padding_kwargs(self, tokenizer) -> dict[str, Any]:
        padding_kwargs: dict[str, Any] = {"length": None}
        existing_padding = getattr(tokenizer, "padding", None)
        if isinstance(existing_padding, dict):
            for key in ("direction", "pad_id", "pad_type_id", "pad_token"):
                if key in existing_padding and existing_padding[key] is not None:
                    padding_kwargs[key] = existing_padding[key]
        else:
            token_to_id = getattr(tokenizer, "token_to_id", None)
            if callable(token_to_id):
                for candidate in ("[PAD]", "<pad>", "<PAD>", "<|pad|>"):
                    token_id = token_to_id(candidate)
                    if token_id is not None:
                        padding_kwargs["pad_id"] = int(token_id)
                        padding_kwargs["pad_token"] = candidate
                        break
        if self.pad_to_multiple_of is not None:
            padding_kwargs["pad_to_multiple_of"] = self.pad_to_multiple_of
        return padding_kwargs

    def _load_tokenizer(self):
        tokenizer = self._tokenizer
        if tokenizer is not None:
            return tokenizer
        with self._tokenizer_lock:
            tokenizer = self._tokenizer
            if tokenizer is not None:
                return tokenizer
            try:
                from tokenizers import Tokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "Local semantic routing requires the `tokenizers` package."
                ) from exc
            tokenizer_path = self._resolve_tokenizer_artifact_path()
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            tokenizer.enable_truncation(max_length=self.max_length)
            tokenizer.enable_padding(**self._tokenizer_padding_kwargs(tokenizer))
            self._tokenizer = tokenizer
            return tokenizer

    def _build_session_options(self, ort) -> object:
        session_options = ort.SessionOptions()
        if self.intra_op_num_threads is not None:
            session_options.intra_op_num_threads = self.intra_op_num_threads
        if self.execution_mode == "parallel":
            session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            if self.inter_op_num_threads is not None:
                session_options.inter_op_num_threads = self.inter_op_num_threads
        else:
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        graph_level = {
            "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }[self.graph_optimization_level]
        session_options.graph_optimization_level = graph_level
        if self.allow_spinning is not None:
            spin_value = "1" if self.allow_spinning else "0"
            session_options.add_session_config_entry("session.intra_op.allow_spinning", spin_value)
            session_options.add_session_config_entry("session.inter_op.allow_spinning", spin_value)
        return session_options

    def _load_session(self):
        session = self._session
        if session is not None:
            return session
        with self._session_lock:
            session = self._session
            if session is not None:
                return session
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError(
                    "Local semantic routing requires the `onnxruntime` package."
                ) from exc
            model_artifact_path = self._resolve_model_artifact_path()
            session = _pop_preloaded_ort_session(model_artifact_path)
            if session is None:
                session = ort.InferenceSession(
                    str(model_artifact_path),
                    sess_options=self._build_session_options(ort),
                    providers=["CPUExecutionProvider"],
                )
            self._session = session
            self._model_artifact_path = model_artifact_path
            self._session_input_specs = tuple(
                _OrtIoSpec(
                    name=item.name,
                    type_name=str(getattr(item, "type", "") or ""),
                    shape=tuple(getattr(item, "shape", ()) or ()),
                )
                for item in session.get_inputs()
            )
            self._session_output_specs = tuple(
                _OrtIoSpec(
                    name=item.name,
                    type_name=str(getattr(item, "type", "") or ""),
                    shape=tuple(getattr(item, "shape", ()) or ()),
                )
                for item in session.get_outputs()
            )
            self._selected_output_name = self._select_output_name()
            return session

    def _select_output_name(self) -> str | None:
        if self.output_name:
            available = {spec.name for spec in self._session_output_specs}
            if self.output_name not in available:
                raise ValueError(
                    f"Configured ONNX output '{self.output_name}' was not found. "
                    f"Available outputs: {sorted(available)}"
                )
            return self.output_name
        if not self._session_output_specs:
            return None
        strong_embedding_names = (
            "sentence_embedding",
            "sentence_embeddings",
            "embedding",
            "embeddings",
            "text_embeds",
        )
        weak_prepooled_names = (
            "pooled_output",
            "pooler_output",
            "output",
        )
        by_name = {spec.name.lower(): spec.name for spec in self._session_output_specs}
        for preferred in strong_embedding_names:
            if preferred in by_name:
                return by_name[preferred]
        if self.pooling == "prepooled":
            for preferred in weak_prepooled_names:
                if preferred in by_name:
                    return by_name[preferred]
            for spec in self._session_output_specs:
                if len(spec.shape) == 2:
                    return spec.name
        return self._session_output_specs[0].name

    def _get_input_spec(self, *aliases: str) -> _OrtIoSpec | None:
        wanted = {alias.strip().lower() for alias in aliases if alias.strip()}
        for spec in self._session_input_specs:
            if spec.name.strip().lower() in wanted:
                return spec
        return None

    def _cast_for_input(self, array: np.ndarray, spec: _OrtIoSpec) -> np.ndarray:
        target_dtype = _onnx_type_to_numpy_dtype(spec.type_name)
        if array.dtype == target_dtype:
            return array
        return array.astype(target_dtype, copy=False)

    def _model_supports_attention_mask(self) -> bool:
        return self._get_input_spec("attention_mask", "input_mask", "mask") is not None

    def _encode_batch(self, tokenizer, texts: Sequence[str]):
        with self._tokenizer_lock:
            encode_batch_fast = getattr(tokenizer, "encode_batch_fast", None)
            if callable(encode_batch_fast):
                return encode_batch_fast(texts)
            return tokenizer.encode_batch(texts)

    def _build_model_inputs(self, encodings: Sequence[object]) -> dict[str, np.ndarray]:
        if not encodings:
            raise ValueError("OnnxSentenceEncoder.encode requires at least one encoding.")
        input_ids = np.asarray([encoding.ids for encoding in encodings], dtype=np.int64)
        attention_mask = np.asarray([encoding.attention_mask for encoding in encodings], dtype=np.int64)
        inputs: dict[str, np.ndarray] = {}

        input_ids_spec = self._get_input_spec("input_ids", "ids")
        if input_ids_spec is None:
            if not self._session_input_specs:
                raise RuntimeError("Semantic router ONNX session exposes no inputs.")
            input_ids_spec = self._session_input_specs[0]
        inputs[input_ids_spec.name] = self._cast_for_input(input_ids, input_ids_spec)

        attention_spec = self._get_input_spec("attention_mask", "input_mask", "mask")
        if attention_spec is not None:
            inputs[attention_spec.name] = self._cast_for_input(attention_mask, attention_spec)

        type_spec = self._get_input_spec("token_type_ids", "segment_ids", "type_ids")
        if type_spec is not None:
            type_rows = []
            for encoding in encodings:
                row = getattr(encoding, "type_ids", None) or [0] * len(encoding.ids)
                if len(row) != len(encoding.ids):
                    row = list(row)[: len(encoding.ids)]
                    if len(row) < len(encoding.ids):
                        row.extend([0] * (len(encoding.ids) - len(row)))
                type_rows.append(row)
            token_type_ids = np.asarray(type_rows, dtype=np.int64)
            inputs[type_spec.name] = self._cast_for_input(token_type_ids, type_spec)

        position_spec = self._get_input_spec("position_ids")
        if position_spec is not None:
            seq_len = int(input_ids.shape[1])
            position_ids = np.broadcast_to(
                np.arange(seq_len, dtype=np.int64),
                input_ids.shape,
            ).copy()
            inputs[position_spec.name] = self._cast_for_input(position_ids, position_spec)

        unsupported_inputs = [
            spec.name
            for spec in self._session_input_specs
            if spec.name not in inputs
        ]
        if unsupported_inputs:
            raise ValueError(
                "Unsupported ONNX sentence-model inputs. "
                f"Add handling for: {unsupported_inputs}"
            )
        return inputs

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return one embedding per input text."""

        normalized_texts = [str(text or "").strip() for text in texts]
        if not normalized_texts:
            raise ValueError("OnnxSentenceEncoder.encode requires at least one text.")
        tokenizer = self._load_tokenizer()
        session = self._load_session()
        batches = []
        effective_batch_size = self.batch_size if self._model_supports_attention_mask() else 1
        for chunk in _iter_chunks(normalized_texts, effective_batch_size):
            encodings = self._encode_batch(tokenizer, chunk)
            if not encodings:
                continue
            model_inputs = self._build_model_inputs(encodings)
            if self._selected_output_name is None:
                outputs = session.run(None, model_inputs)
            else:
                outputs = session.run([self._selected_output_name], model_inputs)
            if not outputs:
                raise RuntimeError("Semantic router ONNX session returned no outputs.")
            raw_embeddings = np.asarray(outputs[0], dtype=np.float32)
            mask = np.asarray([encoding.attention_mask for encoding in encodings], dtype=np.int64)
            batch_embeddings = self._pool_embeddings(raw_embeddings, mask)
            if self.normalize_embeddings:
                batch_embeddings = _l2_normalize(batch_embeddings)
            batches.append(batch_embeddings.astype(np.float32, copy=False))
        if not batches:
            raise RuntimeError("Semantic router ONNX session produced no embeddings.")
        if len(batches) == 1:
            return batches[0]
        return np.concatenate(batches, axis=0).astype(np.float32, copy=False)

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
            raise ValueError(
                "Prepooled routing bundles must emit rank-2 embeddings or expose a rank-2 embedding output."
            )
        mask = attention_mask.astype(np.float32, copy=False)[..., None]
        summed = np.sum(raw_embeddings * mask, axis=1)
        counts = np.clip(np.sum(mask, axis=1), 1.0, None)
        return summed / counts
