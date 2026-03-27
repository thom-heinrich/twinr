# CHANGELOG: 2026-03-27
# BUG-1: Fixed silent boolean coercion from JSON strings such as "false" -> True.
# BUG-2: Fixed false-positive "validation" by validating tokenizer/model/classifier contents, not just filenames.
# BUG-3: Fixed acceptance of directories, symlinks, hardlinks, empty files, and mismatched classifier tensors.
# SEC-1: Added canonical path containment and ONNX external-data checks to block bundle escapes via traversal/symlink/hardlink abuse.
# SEC-2: Added optional SHA-256 manifest verification and detached Ed25519 signature verification for bundle integrity.
# IMP-1: Prefer edge-optimized model.ort bundles over model.onnx when both exist.
# IMP-2: Added safetensors support for classifier tensors and safe NumPy loading with allow_pickle=False.
# IMP-3: Expose validated model IO metadata, embedding_dim, artifact digests, and integrity status to downstream code.

"""Load and validate versioned user-intent bundles for the first router stage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
import base64
import binascii
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Iterable

import numpy as np

from .inference import _cache_preloaded_ort_session
from .user_intent import USER_INTENT_LABEL_VALUES, normalize_user_intent_label

_USER_INTENT_BUNDLE_SCHEMA_VERSION = 1
_SUPPORTED_USER_INTENT_CLASSIFIER_TYPES = {
    "embedding_centroid_v1",
    "embedding_linear_softmax_v1",
}
_ALLOWED_POOLING = {"mean", "cls", "prepooled"}
_DEFAULT_REFERENCE_DATE = "2026-03-22"
_DEFAULT_MODEL_FILENAMES = ("model.ort", "model.onnx")
_DEFAULT_TOKENIZER_FILENAME = "tokenizer.json"
_DEFAULT_CENTROIDS_FILENAMES = ("centroids.safetensors", "centroids.npy")
_DEFAULT_WEIGHTS_FILENAMES = ("weights.safetensors", "weights.npy")
_DEFAULT_BIAS_FILENAMES = ("bias.safetensors", "bias.npy")
_DEFAULT_INTEGRITY_MANIFEST_FILENAME = "bundle_integrity.json"
_DEFAULT_INTEGRITY_SIGNATURE_FILENAME = "bundle_integrity.sig"
_DEFAULT_MAX_ARTIFACT_BYTES = 512 * 1024 * 1024


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _parse_bool(field_name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean, got {value!r}")


def _parse_int(field_name: str, value: Any, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer, got {value!r}")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{field_name} must not be empty")
        try:
            parsed = int(text)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an integer, got {value!r}") from exc
    else:
        raise ValueError(f"{field_name} must be an integer, got {value!r}")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}, got {parsed}")
    return parsed


def _parse_optional_int(field_name: str, value: Any, *, minimum: int | None = None) -> int | None:
    if value is None or value == "":
        return None
    return _parse_int(field_name, value, minimum=minimum)


def _parse_float(field_name: str, value: Any, *, minimum: float | None = None, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a float, got {value!r}")
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{field_name} must not be empty")
        try:
            parsed = float(text)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a float, got {value!r}") from exc
    else:
        raise ValueError(f"{field_name} must be a float, got {value!r}")
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    if positive and parsed <= 0.0:
        raise ValueError(f"{field_name} must be > 0, got {parsed}")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}, got {parsed}")
    return parsed


def _parse_str(field_name: str, value: Any, *, allow_empty: bool = False, lowercase: bool = False) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value.strip()
    else:
        text = str(value).strip()
    if lowercase:
        text = text.lower()
    if not allow_empty and not text:
        raise ValueError(f"{field_name} must not be empty")
    return text


def _normalize_bundle_relative_path(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    text = _parse_str(field_name, value, allow_empty=False)
    pure = PurePosixPath(text.replace("\\", "/"))
    if pure.is_absolute():
        raise ValueError(f"{field_name} must be relative to the bundle root, got {value!r}")
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{field_name} must not contain '.', '..', or empty segments, got {value!r}")
    normalized = pure.as_posix()
    if normalized.startswith("../") or normalized == "..":
        raise ValueError(f"{field_name} escapes the bundle root: {value!r}")
    return normalized


def _parse_optional_date(field_name: str, value: Any) -> str | None:
    text = _parse_str(field_name, value, allow_empty=True)
    if not text:
        return None
    try:
        date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be ISO-8601 YYYY-MM-DD, got {value!r}") from exc
    return text


def _resolve_bundle_member(root_dir: Path, relative_name: str) -> Path:
    candidate = (root_dir / relative_name).resolve(strict=False)
    if not _is_relative_to(candidate, root_dir):
        raise ValueError(f"Bundle member escapes root directory: {relative_name!r}")
    return candidate


def _validate_regular_file(path: Path, *, root_dir: Path, description: str, max_bytes: int | None = None) -> os.stat_result:
    # BREAKING: bundle artifacts must now be regular files; symlinks and hardlinks are rejected.
    resolved = path.resolve(strict=False)
    if not _is_relative_to(resolved, root_dir):
        raise ValueError(f"{description} escapes the bundle root: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path.name}")
    if path.is_symlink():
        raise ValueError(f"{description} must not be a symlink: {path.name}")
    stat_result = path.stat()
    if not path.is_file():
        raise ValueError(f"{description} must be a regular file: {path.name}")
    if stat_result.st_nlink != 1:
        raise ValueError(f"{description} must not be a hardlink: {path.name}")
    if stat_result.st_size <= 0:
        raise ValueError(f"{description} must not be empty: {path.name}")
    if max_bytes is not None and stat_result.st_size > max_bytes:
        raise ValueError(
            f"{description} exceeds the size limit of {max_bytes} bytes: {path.name} ({stat_result.st_size} bytes)"
        )
    return stat_result


def _select_required_file(
    root_dir: Path,
    *,
    description: str,
    explicit_relative_name: str | None,
    default_candidates: tuple[str, ...],
) -> Path:
    if explicit_relative_name is not None:
        path = _resolve_bundle_member(root_dir, explicit_relative_name)
        if not path.exists():
            raise FileNotFoundError(
                f"Bundle metadata points to missing {description}: {explicit_relative_name}"
            )
        return path
    for relative_name in default_candidates:
        path = _resolve_bundle_member(root_dir, relative_name)
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Missing {description}. Looked for: {', '.join(default_candidates)}"
    )


def _file_storage_kind(path: Path | None) -> str | None:
    if path is None:
        return None
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return "npy"
    if suffix == ".safetensors":
        return "safetensors"
    return suffix.lstrip(".") or "unknown"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_mapping(path: Path, *, description: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{description} is not valid JSON: {path.name}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{description} must contain a top-level JSON object: {path.name}")
    return payload


@lru_cache(maxsize=1)
def _get_onnxruntime():
    import onnxruntime as ort
    return ort


@lru_cache(maxsize=1)
def _get_onnx():
    import onnx
    return onnx


@lru_cache(maxsize=1)
def _get_tokenizer_cls():
    from tokenizers import Tokenizer
    return Tokenizer


@lru_cache(maxsize=1)
def _get_safetensors_numpy_load_file():
    from safetensors.numpy import load_file
    return load_file


@lru_cache(maxsize=1)
def _get_cryptography_ed25519():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from cryptography.hazmat.primitives.serialization import load_pem_public_key
    return Ed25519PublicKey, load_pem_public_key


@dataclass(frozen=True, slots=True)
class UserIntentBundleMetadata:
    schema_version: int
    classifier_type: str
    labels: tuple[str, ...]
    model_id: str
    max_length: int
    pooling: str = "mean"
    output_name: str | None = None
    temperature: float = 1.0
    normalize_embeddings: bool = True
    normalize_centroids: bool = True
    reference_date: str | None = _DEFAULT_REFERENCE_DATE
    model_filename: str | None = None
    tokenizer_filename: str = _DEFAULT_TOKENIZER_FILENAME
    centroids_filename: str | None = None
    weights_filename: str | None = None
    bias_filename: str | None = None
    integrity_manifest_filename: str | None = None
    integrity_signature_filename: str | None = None
    embedding_dim: int | None = None

    def __post_init__(self) -> None:
        try:
            labels_iterable = tuple(self.labels)
        except TypeError as exc:
            raise ValueError("labels must be an iterable of strings") from exc
        normalized_labels = tuple(
            normalize_user_intent_label(_parse_str("labels[]", label, allow_empty=False, lowercase=False))
            for label in labels_iterable
        )
        if len(normalized_labels) != len(USER_INTENT_LABEL_VALUES):
            raise ValueError(
                "User intent bundles must declare exactly the "
                "wissen/nachschauen/persoenlich/machen_oder_pruefen labels."
            )
        if len(set(normalized_labels)) != len(USER_INTENT_LABEL_VALUES):
            raise ValueError("User intent labels must be unique after normalization.")
        if set(normalized_labels) != set(USER_INTENT_LABEL_VALUES):
            raise ValueError(
                "User intent bundles must declare exactly the "
                "wissen/nachschauen/persoenlich/machen_oder_pruefen labels."
            )
        schema_version = _parse_int("schema_version", self.schema_version, minimum=1)
        classifier_type = _parse_str("classifier_type", self.classifier_type, allow_empty=False, lowercase=True)
        model_id = _parse_str("model_id", self.model_id, allow_empty=True)
        max_length = _parse_int("max_length", self.max_length, minimum=1)
        pooling = _parse_str("pooling", self.pooling or "mean", allow_empty=False, lowercase=True)
        output_name = _parse_str("output_name", self.output_name, allow_empty=True) or None
        temperature = _parse_float("temperature", self.temperature, positive=True)
        normalize_embeddings = _parse_bool("normalize_embeddings", self.normalize_embeddings)
        normalize_centroids = _parse_bool("normalize_centroids", self.normalize_centroids)
        reference_date = _parse_optional_date("reference_date", self.reference_date)
        model_filename = _normalize_bundle_relative_path(self.model_filename, field_name="model_filename")
        tokenizer_filename = (
            _normalize_bundle_relative_path(self.tokenizer_filename, field_name="tokenizer_filename")
            or _DEFAULT_TOKENIZER_FILENAME
        )
        centroids_filename = _normalize_bundle_relative_path(self.centroids_filename, field_name="centroids_filename")
        weights_filename = _normalize_bundle_relative_path(self.weights_filename, field_name="weights_filename")
        bias_filename = _normalize_bundle_relative_path(self.bias_filename, field_name="bias_filename")
        integrity_manifest_filename = _normalize_bundle_relative_path(
            self.integrity_manifest_filename,
            field_name="integrity_manifest_filename",
        )
        integrity_signature_filename = _normalize_bundle_relative_path(
            self.integrity_signature_filename,
            field_name="integrity_signature_filename",
        )
        embedding_dim = _parse_optional_int("embedding_dim", self.embedding_dim, minimum=1)

        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "classifier_type", classifier_type)
        object.__setattr__(self, "labels", normalized_labels)
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "max_length", max_length)
        object.__setattr__(self, "pooling", pooling)
        object.__setattr__(self, "output_name", output_name)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "normalize_embeddings", normalize_embeddings)
        object.__setattr__(self, "normalize_centroids", normalize_centroids)
        object.__setattr__(self, "reference_date", reference_date)
        object.__setattr__(self, "model_filename", model_filename)
        object.__setattr__(self, "tokenizer_filename", tokenizer_filename)
        object.__setattr__(self, "centroids_filename", centroids_filename)
        object.__setattr__(self, "weights_filename", weights_filename)
        object.__setattr__(self, "bias_filename", bias_filename)
        object.__setattr__(self, "integrity_manifest_filename", integrity_manifest_filename)
        object.__setattr__(self, "integrity_signature_filename", integrity_signature_filename)
        object.__setattr__(self, "embedding_dim", embedding_dim)

        if schema_version != _USER_INTENT_BUNDLE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported user intent bundle schema: {schema_version}")
        if classifier_type not in _SUPPORTED_USER_INTENT_CLASSIFIER_TYPES:
            raise ValueError(f"Unsupported user intent classifier type: {classifier_type}")
        if pooling not in _ALLOWED_POOLING:
            raise ValueError(f"Unsupported user intent pooling mode: {pooling}")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UserIntentBundleMetadata":
        return cls(
            schema_version=payload.get("schema_version", _USER_INTENT_BUNDLE_SCHEMA_VERSION),
            classifier_type=payload.get("classifier_type", "embedding_centroid_v1"),
            labels=tuple(payload.get("labels", ()) or ()),
            model_id=payload.get("model_id", ""),
            max_length=payload.get("max_length", 128),
            pooling=payload.get("pooling", "mean"),
            output_name=payload.get("output_name"),
            temperature=payload.get("temperature", 1.0),
            normalize_embeddings=payload.get("normalize_embeddings", True),
            normalize_centroids=payload.get("normalize_centroids", True),
            reference_date=payload.get("reference_date", _DEFAULT_REFERENCE_DATE),
            model_filename=payload.get("model_filename"),
            tokenizer_filename=payload.get("tokenizer_filename", _DEFAULT_TOKENIZER_FILENAME),
            centroids_filename=payload.get("centroids_filename"),
            weights_filename=payload.get("weights_filename"),
            bias_filename=payload.get("bias_filename"),
            integrity_manifest_filename=payload.get("integrity_manifest_filename"),
            integrity_signature_filename=payload.get("integrity_signature_filename"),
            embedding_dim=payload.get("embedding_dim"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "classifier_type": self.classifier_type,
            "labels": list(self.labels),
            "model_id": self.model_id,
            "max_length": self.max_length,
            "pooling": self.pooling,
            "output_name": self.output_name,
            "temperature": self.temperature,
            "normalize_embeddings": self.normalize_embeddings,
            "normalize_centroids": self.normalize_centroids,
            "reference_date": self.reference_date,
            "model_filename": self.model_filename,
            "tokenizer_filename": self.tokenizer_filename,
            "centroids_filename": self.centroids_filename,
            "weights_filename": self.weights_filename,
            "bias_filename": self.bias_filename,
            "integrity_manifest_filename": self.integrity_manifest_filename,
            "integrity_signature_filename": self.integrity_signature_filename,
            "embedding_dim": self.embedding_dim,
        }


@dataclass(frozen=True, slots=True)
class UserIntentBundleArtifact:
    relative_path: str
    path: Path
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class UserIntentBundle:
    root_dir: Path
    metadata: UserIntentBundleMetadata
    model_path: Path
    tokenizer_path: Path
    centroids_path: Path | None
    weights_path: Path | None
    bias_path: Path | None
    metadata_path: Path
    model_format: str = "onnx"
    classifier_artifact_format: str = "npy"
    resolved_output_name: str | None = None
    model_input_names: tuple[str, ...] = ()
    model_output_names: tuple[str, ...] = ()
    embedding_dim: int | None = None
    artifacts: tuple[UserIntentBundleArtifact, ...] = ()
    integrity_manifest_path: Path | None = None
    integrity_manifest_verified: bool = False
    signature_verified: bool = False


def _load_numpy_array(path: Path) -> np.ndarray:
    try:
        array = np.load(path, allow_pickle=False, mmap_mode="r")
    except Exception as exc:
        raise ValueError(f"Failed to load NumPy artifact {path.name}: {exc}") from exc
    if not isinstance(array, np.ndarray):
        raise ValueError(f"NumPy artifact {path.name} did not contain a single ndarray.")
    return array


def _load_safetensors_array(path: Path, *, expected_keys: tuple[str, ...]) -> np.ndarray:
    try:
        load_file = _get_safetensors_numpy_load_file()
    except ImportError as exc:
        raise RuntimeError(
            f"{path.name} uses the safetensors format but safetensors is not installed."
        ) from exc
    tensors = load_file(path)
    if not tensors:
        raise ValueError(f"Safetensors artifact {path.name} contains no tensors.")
    for key in expected_keys:
        if key in tensors:
            return tensors[key]
    if len(tensors) == 1:
        return next(iter(tensors.values()))
    raise ValueError(
        f"Safetensors artifact {path.name} must contain one tensor or one of the keys "
        f"{', '.join(expected_keys)}."
    )


def _load_classifier_array(path: Path, *, expected_keys: tuple[str, ...]) -> np.ndarray:
    kind = _file_storage_kind(path)
    if kind == "npy":
        return _load_numpy_array(path)
    if kind == "safetensors":
        return _load_safetensors_array(path, expected_keys=expected_keys)
    raise ValueError(f"Unsupported classifier artifact format for {path.name}: {path.suffix}")


def _require_floating_real_array(array: np.ndarray, *, description: str) -> None:
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{description} must use a floating dtype, got {array.dtype}.")
    if not np.isfinite(np.asarray(array)).all():
        raise ValueError(f"{description} contains NaN or infinite values.")


def _validate_classifier_artifacts(
    *,
    classifier_type: str,
    labels_count: int,
    centroids_path: Path | None,
    weights_path: Path | None,
    bias_path: Path | None,
    embedding_dim: int | None,
) -> int:
    if classifier_type == "embedding_centroid_v1":
        if centroids_path is None:
            raise ValueError("Centroid classifier requires a centroids artifact.")
        centroids = _load_classifier_array(centroids_path, expected_keys=("centroids", "embeddings"))
        _require_floating_real_array(centroids, description="centroids")
        if centroids.ndim != 2:
            raise ValueError(f"centroids must be a rank-2 matrix, got shape {centroids.shape}.")
        if centroids.shape[0] != labels_count:
            raise ValueError(
                f"centroids first dimension must equal the label count {labels_count}, "
                f"got shape {centroids.shape}."
            )
        if centroids.shape[1] <= 0:
            raise ValueError(f"centroids embedding dimension must be > 0, got {centroids.shape}.")
        if embedding_dim is not None and centroids.shape[1] != embedding_dim:
            raise ValueError(
                f"centroids embedding dimension {centroids.shape[1]} does not match the model "
                f"embedding dimension {embedding_dim}."
            )
        return int(centroids.shape[1])

    if weights_path is None or bias_path is None:
        raise ValueError("Linear classifier requires both weights and bias artifacts.")
    weights = _load_classifier_array(weights_path, expected_keys=("weights", "weight"))
    bias = _load_classifier_array(bias_path, expected_keys=("bias",))
    _require_floating_real_array(weights, description="weights")
    _require_floating_real_array(bias, description="bias")
    if weights.ndim != 2:
        raise ValueError(f"weights must be a rank-2 matrix, got shape {weights.shape}.")
    if bias.ndim != 1:
        raise ValueError(f"bias must be a rank-1 vector, got shape {bias.shape}.")
    if bias.shape[0] != labels_count:
        raise ValueError(
            f"bias length must equal the label count {labels_count}, got shape {bias.shape}."
        )
    if weights.shape[0] != labels_count:
        if weights.shape[1] == labels_count:
            raise ValueError(
                f"weights has shape {weights.shape}. This looks transposed. Expected "
                f"(num_labels, hidden_dim) = ({labels_count}, hidden_dim)."
            )
        raise ValueError(
            f"weights first dimension must equal the label count {labels_count}, got shape {weights.shape}."
        )
    if weights.shape[1] <= 0:
        raise ValueError(f"weights hidden dimension must be > 0, got shape {weights.shape}.")
    if embedding_dim is not None and weights.shape[1] != embedding_dim:
        raise ValueError(
            f"weights hidden dimension {weights.shape[1]} does not match the model "
            f"embedding dimension {embedding_dim}."
        )
    return int(weights.shape[1])


def _iter_onnx_graph_tensors(graph: Any, onnx_module: Any) -> Iterable[Any]:
    yield from graph.initializer
    yield from getattr(graph, "sparse_initializer", ())
    attribute_type = onnx_module.AttributeProto
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == attribute_type.TENSOR:
                yield attribute.t
            elif attribute.type == attribute_type.TENSORS:
                yield from attribute.tensors
            elif attribute.type == attribute_type.SPARSE_TENSOR:
                yield attribute.sparse_tensor
            elif attribute.type == attribute_type.SPARSE_TENSORS:
                yield from attribute.sparse_tensors
            elif attribute.type == attribute_type.GRAPH:
                yield from _iter_onnx_graph_tensors(attribute.g, onnx_module)
            elif attribute.type == attribute_type.GRAPHS:
                for subgraph in attribute.graphs:
                    yield from _iter_onnx_graph_tensors(subgraph, onnx_module)


def _validate_onnx_external_data(model_path: Path, *, root_dir: Path) -> tuple[Path, ...]:
    # BREAKING: .onnx bundles now require the optional 'onnx' package for deep validation.
    try:
        onnx_module = _get_onnx()
    except ImportError as exc:
        raise RuntimeError(
            "Deep validation of .onnx bundles requires the 'onnx' package. "
            "Use model.ort for minimal deployments or install onnx."
        ) from exc
    try:
        model = onnx_module.load(str(model_path), load_external_data=False)
    except Exception as exc:
        raise ValueError(f"Failed to parse ONNX model {model_path.name}: {exc}") from exc
    model_dir = model_path.parent.resolve(strict=True)
    external_paths: dict[str, Path] = {}
    for tensor in _iter_onnx_graph_tensors(model.graph, onnx_module):
        if getattr(tensor, "data_location", None) != onnx_module.TensorProto.EXTERNAL:
            continue
        entries = {entry.key: entry.value for entry in tensor.external_data}
        location = entries.get("location")
        if not location:
            raise ValueError(f"External ONNX tensor is missing a location in {model_path.name}.")
        normalized = _normalize_bundle_relative_path(location, field_name="external_data.location")
        assert normalized is not None
        candidate = (model_dir / normalized).resolve(strict=False)
        if not _is_relative_to(candidate, model_dir):
            raise ValueError(
                f"External ONNX data escapes the model directory: {location!r} in {model_path.name}"
            )
        if not _is_relative_to(candidate, root_dir):
            raise ValueError(
                f"External ONNX data escapes the bundle root: {location!r} in {model_path.name}"
            )
        _validate_regular_file(
            candidate,
            root_dir=root_dir,
            description=f"external ONNX data referenced by {model_path.name}",
            max_bytes=None,
        )
        external_paths[normalized] = candidate
    try:
        onnx_module.checker.check_model(str(model_path))
    except Exception as exc:
        if not _onnxruntime_can_load_model(model_path):
            raise ValueError(f"ONNX model validation failed for {model_path.name}: {exc}") from exc
    return tuple(path for _, path in sorted(external_paths.items()))


def _onnxruntime_can_load_model(model_path: Path) -> bool:
    try:
        ort = _get_onnxruntime()
    except ImportError:
        return False
    try:
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
    except Exception:
        return False
    return True


def _extract_static_sequence_limit(inputs: Iterable[Any]) -> int | None:
    limits: list[int] = []
    for node_arg in inputs:
        shape = getattr(node_arg, "shape", None)
        if not isinstance(shape, (list, tuple)) or len(shape) < 2:
            continue
        dim = shape[1]
        if isinstance(dim, int) and dim > 0:
            limits.append(dim)
    if not limits:
        return None
    return min(limits)


def _extract_last_static_dim(node_arg: Any) -> int | None:
    shape = getattr(node_arg, "shape", None)
    if not isinstance(shape, (list, tuple)) or not shape:
        return None
    dim = shape[-1]
    if isinstance(dim, int) and dim > 0:
        return dim
    return None


def _validate_model(
    model_path: Path,
    *,
    root_dir: Path,
    metadata: UserIntentBundleMetadata,
) -> tuple[str, tuple[str, ...], tuple[str, ...], str | None, int | None, tuple[Path, ...]]:
    model_format = "ort" if model_path.suffix.lower() == ".ort" else "onnx"
    external_files: tuple[Path, ...] = ()
    if model_format == "onnx":
        external_files = _validate_onnx_external_data(model_path, root_dir=root_dir)

    try:
        ort = _get_onnxruntime()
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required to validate user intent bundles.") from exc
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=session_options, providers=["CPUExecutionProvider"])
    if model_format == "ort":
        # Reuse the validation session for the first encoder load so Pi startup
        # does not instantiate the same heavyweight ORT graph twice in-process.
        _cache_preloaded_ort_session(model_path, session)
    inputs = tuple(session.get_inputs())
    outputs = tuple(session.get_outputs())
    if not outputs:
        raise ValueError(f"Model {model_path.name} exposes no outputs.")
    input_names = tuple(node_arg.name for node_arg in inputs)
    output_names = tuple(node_arg.name for node_arg in outputs)
    resolved_output_name = metadata.output_name
    selected_output = None
    if resolved_output_name is not None:
        for node_arg in outputs:
            if node_arg.name == resolved_output_name:
                selected_output = node_arg
                break
        if selected_output is None:
            raise ValueError(
                f"metadata.output_name={resolved_output_name!r} does not exist in model outputs "
                f"{output_names}."
            )
    elif len(outputs) == 1:
        selected_output = outputs[0]
        resolved_output_name = outputs[0].name

    sequence_limit = _extract_static_sequence_limit(inputs)
    if sequence_limit is not None and metadata.max_length > sequence_limit:
        raise ValueError(
            f"metadata.max_length={metadata.max_length} exceeds the model's static input limit "
            f"of {sequence_limit}."
        )

    embedding_dim = metadata.embedding_dim
    if selected_output is not None:
        inferred_dim = _extract_last_static_dim(selected_output)
        if embedding_dim is None:
            embedding_dim = inferred_dim
        elif inferred_dim is not None and inferred_dim != embedding_dim:
            raise ValueError(
                f"metadata.embedding_dim={embedding_dim} does not match the selected model "
                f"output dimension {inferred_dim} for output {selected_output.name!r}."
            )
        output_shape = getattr(selected_output, "shape", None)
        if metadata.pooling == "prepooled" and isinstance(output_shape, (list, tuple)):
            static_rank = len(output_shape)
            if static_rank != 2:
                raise ValueError(
                    f"pooling='prepooled' expects a rank-2 output, got shape {output_shape} "
                    f"for output {selected_output.name!r}."
                )

    return model_format, input_names, output_names, resolved_output_name, embedding_dim, external_files


def _validate_tokenizer(path: Path, *, metadata: UserIntentBundleMetadata) -> None:
    payload = _load_json_mapping(path, description="tokenizer file")
    if "model" not in payload:
        raise ValueError(f"tokenizer file {path.name} is missing the required 'model' section.")
    truncation = payload.get("truncation")
    if isinstance(truncation, Mapping):
        max_length = truncation.get("max_length")
        if isinstance(max_length, int) and max_length > 0 and metadata.max_length > max_length:
            raise ValueError(
                f"metadata.max_length={metadata.max_length} exceeds tokenizer truncation "
                f"max_length={max_length}."
            )
    try:
        tokenizer_cls = _get_tokenizer_cls()
    except ImportError:
        return
    try:
        tokenizer = tokenizer_cls.from_file(str(path))
    except Exception as exc:
        raise ValueError(f"Tokenizer {path.name} failed to load with tokenizers: {exc}") from exc
    if tokenizer.get_vocab_size() <= 0:
        raise ValueError(f"Tokenizer {path.name} has an empty vocabulary.")


def _load_integrity_manifest(
    root_dir: Path,
    *,
    metadata: UserIntentBundleMetadata,
    require_integrity_manifest: bool,
) -> tuple[Path | None, Mapping[str, Any] | None, Path | None]:
    if metadata.integrity_manifest_filename is not None:
        manifest_path = _resolve_bundle_member(root_dir, metadata.integrity_manifest_filename)
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Bundle metadata points to missing integrity manifest: {metadata.integrity_manifest_filename}"
            )
    else:
        manifest_path = _resolve_bundle_member(root_dir, _DEFAULT_INTEGRITY_MANIFEST_FILENAME)
        if not manifest_path.exists():
            manifest_path = None
    if manifest_path is None:
        if require_integrity_manifest:
            raise FileNotFoundError("Integrity manifest required but bundle_integrity.json is missing.")
        return None, None, None

    _validate_regular_file(
        manifest_path,
        root_dir=root_dir,
        description="integrity manifest",
        max_bytes=2 * 1024 * 1024,
    )
    manifest = _load_json_mapping(manifest_path, description="integrity manifest")
    version = _parse_int("integrity_manifest.version", manifest.get("version", 1), minimum=1)
    if version != 1:
        raise ValueError(f"Unsupported integrity manifest version: {version}")
    algorithm = _parse_str(
        "integrity_manifest.algorithm",
        manifest.get("algorithm", "sha256"),
        allow_empty=False,
        lowercase=True,
    )
    if algorithm != "sha256":
        raise ValueError(f"Unsupported integrity manifest algorithm: {algorithm}")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("Integrity manifest must contain a non-empty 'files' mapping.")

    if metadata.integrity_signature_filename is not None:
        signature_path = _resolve_bundle_member(root_dir, metadata.integrity_signature_filename)
        if not signature_path.exists():
            raise FileNotFoundError(
                f"Bundle metadata points to missing integrity signature: {metadata.integrity_signature_filename}"
            )
    else:
        signature_path = _resolve_bundle_member(root_dir, _DEFAULT_INTEGRITY_SIGNATURE_FILENAME)
        if not signature_path.exists():
            signature_path = None
    if signature_path is not None:
        _validate_regular_file(
            signature_path,
            root_dir=root_dir,
            description="integrity signature",
            max_bytes=64 * 1024,
        )
    return manifest_path, manifest, signature_path


def _normalize_hex_digest(field_name: str, value: Any) -> str:
    digest = _parse_str(field_name, value, allow_empty=False, lowercase=True)
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest, got {value!r}")
    return digest


def _collect_artifact_paths(
    *,
    root_dir: Path,
    core_paths: Iterable[Path],
    manifest: Mapping[str, Any] | None,
    manifest_path: Path | None,
    signature_path: Path | None,
    expected_sha256: Mapping[str, str] | None,
    max_artifact_bytes: int,
) -> tuple[Path, ...]:
    unique: dict[Path, None] = {}
    for path in core_paths:
        unique[path.resolve(strict=True)] = None
    if manifest is not None:
        files = manifest["files"]
        for relative_name in files:
            normalized = _normalize_bundle_relative_path(relative_name, field_name="integrity_manifest.files[]")
            assert normalized is not None
            candidate = _resolve_bundle_member(root_dir, normalized)
            _validate_regular_file(
                candidate,
                root_dir=root_dir,
                description=f"integrity-managed artifact {normalized}",
                max_bytes=max_artifact_bytes,
            )
            unique[candidate.resolve(strict=True)] = None
    if expected_sha256:
        for relative_name in expected_sha256:
            normalized = _normalize_bundle_relative_path(relative_name, field_name="expected_sha256[]")
            assert normalized is not None
            candidate = _resolve_bundle_member(root_dir, normalized)
            _validate_regular_file(
                candidate,
                root_dir=root_dir,
                description=f"expected-sha256 artifact {normalized}",
                max_bytes=max_artifact_bytes,
            )
            unique[candidate.resolve(strict=True)] = None
    if manifest_path is not None:
        unique[manifest_path.resolve(strict=True)] = None
    if signature_path is not None:
        unique[signature_path.resolve(strict=True)] = None
    return tuple(sorted(unique, key=lambda item: str(item)))


def _build_artifact_records(root_dir: Path, paths: Iterable[Path]) -> tuple[UserIntentBundleArtifact, ...]:
    records: list[UserIntentBundleArtifact] = []
    for path in paths:
        stat_result = path.stat()
        relative_path = path.relative_to(root_dir).as_posix()
        records.append(
            UserIntentBundleArtifact(
                relative_path=relative_path,
                path=path,
                size_bytes=int(stat_result.st_size),
                sha256=_sha256_file(path),
            )
        )
    records.sort(key=lambda record: record.relative_path)
    return tuple(records)


def _verify_integrity_manifest(
    *,
    manifest: Mapping[str, Any] | None,
    artifact_records: Mapping[str, UserIntentBundleArtifact],
    required_relative_paths: Iterable[str],
    manifest_relative_path: str | None,
) -> bool:
    if manifest is None:
        return False
    file_digests = manifest["files"]
    required = set(required_relative_paths)
    missing = sorted(required - set(file_digests))
    if missing:
        raise ValueError(
            "Integrity manifest is missing required artifacts: " + ", ".join(missing)
        )
    for relative_name, expected_digest in file_digests.items():
        normalized = _normalize_bundle_relative_path(relative_name, field_name="integrity_manifest.files[]")
        assert normalized is not None
        if manifest_relative_path is not None and normalized == manifest_relative_path:
            raise ValueError("Integrity manifest must not include itself in its file digest list.")
        expected = _normalize_hex_digest(f"integrity_manifest.files[{normalized!r}]", expected_digest)
        actual_record = artifact_records.get(normalized)
        if actual_record is None:
            raise ValueError(f"Integrity manifest references an unhashed artifact: {normalized}")
        if actual_record.sha256 != expected:
            raise ValueError(
                f"SHA-256 mismatch for {normalized}: expected {expected}, got {actual_record.sha256}"
            )
    return True


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _load_ed25519_public_key(public_key: str | bytes) -> Any:
    ed25519_cls, load_pem_public_key = _get_cryptography_ed25519()
    if isinstance(public_key, bytes):
        key_bytes = public_key.strip()
        key_text = None
    else:
        key_text = public_key.strip()
        key_bytes = key_text.encode("utf-8")
    if key_bytes.startswith(b"-----BEGIN"):
        key = load_pem_public_key(key_bytes)
        if not isinstance(key, ed25519_cls):
            raise ValueError("trusted_ed25519_public_key must be an Ed25519 public key.")
        return key
    text = key_text if key_text is not None else key_bytes.decode("utf-8", errors="ignore").strip()
    if text and len(text) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in text):
        return ed25519_cls.from_public_bytes(bytes.fromhex(text))
    padded = text + "=" * (-len(text) % 4)
    try:
        raw_bytes = base64.b64decode(padded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(
            "trusted_ed25519_public_key must be PEM, raw 32-byte base64, or 32-byte hex."
        ) from exc
    return ed25519_cls.from_public_bytes(raw_bytes)


def _verify_signature(
    *,
    manifest: Mapping[str, Any] | None,
    signature_path: Path | None,
    trusted_ed25519_public_key: str | bytes | None,
) -> bool:
    if trusted_ed25519_public_key is None:
        return False
    if manifest is None:
        raise ValueError("trusted_ed25519_public_key was provided but no integrity manifest is present.")
    if signature_path is None:
        raise ValueError("trusted_ed25519_public_key was provided but no integrity signature is present.")
    public_key = _load_ed25519_public_key(trusted_ed25519_public_key)
    signature_bytes = signature_path.read_bytes().strip()
    if not signature_bytes:
        raise ValueError(f"Integrity signature file {signature_path.name} is empty.")
    text = signature_bytes.decode("utf-8", errors="ignore").strip()
    decoded_signature = None
    if text:
        if len(text) == 128 and all(ch in "0123456789abcdefABCDEF" for ch in text):
            decoded_signature = bytes.fromhex(text)
        else:
            padded = text + "=" * (-len(text) % 4)
            try:
                decoded_signature = base64.b64decode(padded, validate=True)
            except (binascii.Error, ValueError):
                decoded_signature = None
    if decoded_signature is not None:
        signature = decoded_signature
    else:
        signature = signature_bytes
    payload = _canonical_json_bytes(manifest)
    try:
        public_key.verify(signature, payload)
    except Exception as exc:
        raise ValueError(f"Integrity signature verification failed: {exc}") from exc
    return True


def _verify_expected_sha256(
    *,
    root_dir: Path,
    artifact_records: Mapping[str, UserIntentBundleArtifact],
    expected_sha256: Mapping[str, str] | None,
) -> None:
    if not expected_sha256:
        return
    for relative_name, expected_digest in expected_sha256.items():
        normalized = _normalize_bundle_relative_path(relative_name, field_name="expected_sha256[]")
        assert normalized is not None
        expected = _normalize_hex_digest(f"expected_sha256[{normalized!r}]", expected_digest)
        actual_record = artifact_records.get(normalized)
        if actual_record is None:
            raise ValueError(f"expected_sha256 references an unhashed artifact: {normalized}")
        if actual_record.sha256 != expected:
            raise ValueError(
                f"SHA-256 mismatch for {normalized}: expected {expected}, got {actual_record.sha256}"
            )


def load_user_intent_bundle(
    path: str | Path,
    *,
    expected_sha256: Mapping[str, str] | None = None,
    require_integrity_manifest: bool = False,
    trusted_ed25519_public_key: str | bytes | None = None,
    max_artifact_bytes: int = _DEFAULT_MAX_ARTIFACT_BYTES,
) -> UserIntentBundle:
    root_dir = Path(path).expanduser().resolve(strict=False)
    if not root_dir.exists():
        raise FileNotFoundError(f"User intent bundle directory does not exist: {root_dir}")
    if not root_dir.is_dir():
        raise ValueError(f"User intent bundle path must be a directory: {root_dir}")

    metadata_path = _resolve_bundle_member(root_dir, "user_intent_metadata.json")
    _validate_regular_file(
        metadata_path,
        root_dir=root_dir,
        description="user intent metadata",
        max_bytes=1024 * 1024,
    )
    payload = _load_json_mapping(metadata_path, description="user intent metadata")
    metadata = UserIntentBundleMetadata.from_dict(payload)

    # BREAKING: if both model.ort and model.onnx exist, model.ort is preferred.
    model_path = _select_required_file(
        root_dir,
        description="model artifact",
        explicit_relative_name=metadata.model_filename,
        default_candidates=_DEFAULT_MODEL_FILENAMES,
    )
    _validate_regular_file(
        model_path,
        root_dir=root_dir,
        description="model artifact",
        max_bytes=max_artifact_bytes,
    )

    tokenizer_path = _select_required_file(
        root_dir,
        description="tokenizer artifact",
        explicit_relative_name=metadata.tokenizer_filename,
        default_candidates=(metadata.tokenizer_filename,),
    )
    _validate_regular_file(
        tokenizer_path,
        root_dir=root_dir,
        description="tokenizer artifact",
        max_bytes=64 * 1024 * 1024,
    )
    _validate_tokenizer(tokenizer_path, metadata=metadata)

    centroids_path: Path | None = None
    weights_path: Path | None = None
    bias_path: Path | None = None
    if metadata.classifier_type == "embedding_centroid_v1":
        centroids_path = _select_required_file(
            root_dir,
            description="centroids artifact",
            explicit_relative_name=metadata.centroids_filename,
            default_candidates=_DEFAULT_CENTROIDS_FILENAMES,
        )
        _validate_regular_file(
            centroids_path,
            root_dir=root_dir,
            description="centroids artifact",
            max_bytes=max_artifact_bytes,
        )
    else:
        preferred_weights = metadata.weights_filename
        preferred_bias = metadata.bias_filename
        if preferred_weights is not None or preferred_bias is not None:
            preferred_kind: str | None = None
            if preferred_weights is not None:
                weights_path = _select_required_file(
                    root_dir,
                    description="weights artifact",
                    explicit_relative_name=preferred_weights,
                    default_candidates=_DEFAULT_WEIGHTS_FILENAMES,
                )
                preferred_kind = _file_storage_kind(weights_path)
            if preferred_bias is not None:
                bias_path = _select_required_file(
                    root_dir,
                    description="bias artifact",
                    explicit_relative_name=preferred_bias,
                    default_candidates=_DEFAULT_BIAS_FILENAMES,
                )
                if preferred_kind is None:
                    preferred_kind = _file_storage_kind(bias_path)
            if weights_path is None:
                weight_candidates = _DEFAULT_WEIGHTS_FILENAMES
                if preferred_kind == "safetensors":
                    weight_candidates = ("weights.safetensors",)
                elif preferred_kind == "npy":
                    weight_candidates = ("weights.npy",)
                weights_path = _select_required_file(
                    root_dir,
                    description="weights artifact",
                    explicit_relative_name=None,
                    default_candidates=weight_candidates,
                )
            if bias_path is None:
                bias_candidates = _DEFAULT_BIAS_FILENAMES
                if preferred_kind == "safetensors":
                    bias_candidates = ("bias.safetensors",)
                elif preferred_kind == "npy":
                    bias_candidates = ("bias.npy",)
                bias_path = _select_required_file(
                    root_dir,
                    description="bias artifact",
                    explicit_relative_name=None,
                    default_candidates=bias_candidates,
                )
        else:
            if _resolve_bundle_member(root_dir, "weights.safetensors").exists() and _resolve_bundle_member(root_dir, "bias.safetensors").exists():
                weights_path = _resolve_bundle_member(root_dir, "weights.safetensors")
                bias_path = _resolve_bundle_member(root_dir, "bias.safetensors")
            elif _resolve_bundle_member(root_dir, "weights.npy").exists() and _resolve_bundle_member(root_dir, "bias.npy").exists():
                weights_path = _resolve_bundle_member(root_dir, "weights.npy")
                bias_path = _resolve_bundle_member(root_dir, "bias.npy")
            else:
                raise FileNotFoundError(
                    "Missing linear classifier artifacts. Looked for weights.safetensors+bias.safetensors "
                    "or weights.npy+bias.npy."
                )
        assert weights_path is not None and bias_path is not None
        _validate_regular_file(
            weights_path,
            root_dir=root_dir,
            description="weights artifact",
            max_bytes=max_artifact_bytes,
        )
        _validate_regular_file(
            bias_path,
            root_dir=root_dir,
            description="bias artifact",
            max_bytes=max_artifact_bytes,
        )
        if _file_storage_kind(weights_path) != _file_storage_kind(bias_path):
            raise ValueError(
                f"Mixed linear classifier artifact formats are not supported: "
                f"{weights_path.name}, {bias_path.name}"
            )

    model_format, model_input_names, model_output_names, resolved_output_name, embedding_dim, external_files = _validate_model(
        model_path,
        root_dir=root_dir,
        metadata=metadata,
    )
    embedding_dim = _validate_classifier_artifacts(
        classifier_type=metadata.classifier_type,
        labels_count=len(metadata.labels),
        centroids_path=centroids_path,
        weights_path=weights_path,
        bias_path=bias_path,
        embedding_dim=embedding_dim,
    )

    manifest_path, manifest, signature_path = _load_integrity_manifest(
        root_dir,
        metadata=metadata,
        require_integrity_manifest=require_integrity_manifest or trusted_ed25519_public_key is not None,
    )

    core_paths = [metadata_path, model_path, tokenizer_path]
    if centroids_path is not None:
        core_paths.append(centroids_path)
    if weights_path is not None:
        core_paths.append(weights_path)
    if bias_path is not None:
        core_paths.append(bias_path)
    core_paths.extend(external_files)
    artifact_paths = _collect_artifact_paths(
        root_dir=root_dir,
        core_paths=core_paths,
        manifest=manifest,
        manifest_path=manifest_path,
        signature_path=signature_path,
        expected_sha256=expected_sha256,
        max_artifact_bytes=max_artifact_bytes,
    )
    artifact_records = _build_artifact_records(root_dir, artifact_paths)
    artifact_record_map = {record.relative_path: record for record in artifact_records}

    required_relative_paths = [
        metadata_path.relative_to(root_dir).as_posix(),
        model_path.relative_to(root_dir).as_posix(),
        tokenizer_path.relative_to(root_dir).as_posix(),
    ]
    if centroids_path is not None:
        required_relative_paths.append(centroids_path.relative_to(root_dir).as_posix())
    if weights_path is not None:
        required_relative_paths.append(weights_path.relative_to(root_dir).as_posix())
    if bias_path is not None:
        required_relative_paths.append(bias_path.relative_to(root_dir).as_posix())
    required_relative_paths.extend(path.relative_to(root_dir).as_posix() for path in external_files)

    integrity_manifest_verified = _verify_integrity_manifest(
        manifest=manifest,
        artifact_records=artifact_record_map,
        required_relative_paths=required_relative_paths,
        manifest_relative_path=manifest_path.relative_to(root_dir).as_posix() if manifest_path is not None else None,
    )
    _verify_expected_sha256(
        root_dir=root_dir,
        artifact_records=artifact_record_map,
        expected_sha256=expected_sha256,
    )
    signature_verified = _verify_signature(
        manifest=manifest,
        signature_path=signature_path,
        trusted_ed25519_public_key=trusted_ed25519_public_key,
    )

    # BREAKING: centroids_path / weights_path / bias_path may now point to .safetensors artifacts.
    classifier_artifact_format = _file_storage_kind(centroids_path or weights_path or bias_path) or "npy"
    return UserIntentBundle(
        root_dir=root_dir,
        metadata=metadata,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        centroids_path=centroids_path,
        weights_path=weights_path,
        bias_path=bias_path,
        metadata_path=metadata_path,
        model_format=model_format,
        classifier_artifact_format=classifier_artifact_format,
        resolved_output_name=resolved_output_name,
        model_input_names=model_input_names,
        model_output_names=model_output_names,
        embedding_dim=embedding_dim,
        artifacts=artifact_records,
        integrity_manifest_path=manifest_path,
        integrity_manifest_verified=integrity_manifest_verified,
        signature_verified=signature_verified,
    )
