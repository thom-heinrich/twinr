"""Load and validate versioned local semantic-router bundles."""

from __future__ import annotations

# CHANGELOG: 2026-03-27
# BUG-1: Reject directories, special files, unreadable files, and zero-byte artifacts instead of
#        accepting them via Path.exists() and failing later during inference.
# BUG-2: Parse metadata/manifests in strict JSON mode and reject NaN/Infinity/duplicate keys that
#        silently corrupt thresholds, temperature, and routing policy.
# BUG-3: Stop fabricating a reference_date when metadata omits it; preserve None so downstream
#        recency logic cannot read a false training/evaluation date.
# SEC-1: Constrain all resolved bundle artifacts to the bundle root and reject path/symlink
#        traversal, including ONNX external-data references that escape the bundle directory.
# SEC-2: Add optional per-file SHA-256/size manifest verification so SD-card corruption or
#        tampered OTA bundles are detected before the router loads them.
# IMP-1: Support edge-friendly ORT-format models (model.ort) in addition to ONNX and prefer ORT
#        when both exist and runtime_format="auto".
# IMP-2: Add deeper bundle validation/provenance hooks (ONNX checker, external-data allowlist,
#        refinement metadata, digest snapshots, safetensors classifier artifacts).

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from datetime import date, datetime
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat as statmod
from typing import Any

from .contracts import ROUTE_LABEL_VALUES, normalize_route_label
from .policy import SemanticRouterPolicy


_LATEST_BUNDLE_SCHEMA_VERSION = 2
_SUPPORTED_BUNDLE_SCHEMA_VERSIONS = frozenset({1, 2})
_SUPPORTED_CLASSIFIER_TYPES = frozenset(
    {
        "embedding_centroid_v1",
        "embedding_linear_softmax_v1",
    }
)
_SUPPORTED_POOLING = frozenset({"mean", "cls", "prepooled"})
_SUPPORTED_RUNTIME_FORMATS = frozenset({"auto", "onnx", "ort"})
_SUPPORTED_CLASSIFIER_FORMATS = frozenset({"auto", "npy", "safetensors"})
_SUPPORTED_REFINEMENT_STRATEGIES = frozenset({"none", "oats_v1"})
_METADATA_MAX_BYTES = 1 * 1024 * 1024
_MANIFEST_MAX_BYTES = 4 * 1024 * 1024
_HASH_BLOCK_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class BundleFileExpectation:
    """Optional integrity constraints for one bundle file."""

    relative_path: str
    size_bytes: int | None = None
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class BundleArtifact:
    """Describe one resolved artifact inside the bundle."""

    role: str
    relative_path: str
    path: Path
    size_bytes: int
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class SemanticRouterBundleMetadata:
    """Describe one versioned local semantic-router bundle."""

    schema_version: int
    classifier_type: str
    labels: tuple[str, ...]
    model_id: str
    max_length: int
    pooling: str = "mean"
    output_name: str | None = None
    temperature: float = 1.0
    thresholds: Mapping[str, float] | None = None
    authoritative_labels: tuple[str, ...] = ("web", "memory", "tool")
    min_margin: float = 0.0
    normalize_embeddings: bool = True
    normalize_centroids: bool = True
    reference_date: str | None = None
    runtime_format: str = "auto"
    model_filename: str | None = None
    tokenizer_filename: str = "tokenizer.json"
    classifier_format: str = "auto"
    centroids_filename: str | None = None
    weights_filename: str | None = None
    bias_filename: str | None = None
    classifier_filename: str | None = None
    bundle_id: str | None = None
    created_at: str | None = None
    refinement_strategy: str = "none"
    files: Mapping[str, Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        normalized_labels = tuple(normalize_route_label(label) for label in self.labels)
        if tuple(sorted(normalized_labels)) != tuple(sorted(ROUTE_LABEL_VALUES)):
            raise ValueError(
                "Semantic router bundles must declare exactly the parametric/web/memory/tool labels."
            )
        normalized_thresholds = _normalize_threshold_mapping(self.thresholds)
        normalized_authoritative_labels = _normalize_authoritative_labels(
            self.authoritative_labels,
            valid_labels=normalized_labels,
        )
        normalized_files = _normalize_manifest_entries(self.files)

        object.__setattr__(
            self, "schema_version", _coerce_int("schema_version", self.schema_version, minimum=1)
        )
        object.__setattr__(self, "labels", normalized_labels)
        object.__setattr__(self, "model_id", _coerce_required_str("model_id", self.model_id))
        object.__setattr__(
            self,
            "classifier_type",
            _coerce_required_str("classifier_type", self.classifier_type).lower(),
        )
        object.__setattr__(
            self, "max_length", _coerce_int("max_length", self.max_length, minimum=1)
        )
        object.__setattr__(self, "pooling", _coerce_required_str("pooling", self.pooling).lower())
        object.__setattr__(self, "output_name", _coerce_optional_str("output_name", self.output_name))
        object.__setattr__(
            self,
            "temperature",
            _coerce_finite_float("temperature", self.temperature, minimum=1e-6),
        )
        object.__setattr__(self, "thresholds", normalized_thresholds)
        object.__setattr__(self, "authoritative_labels", normalized_authoritative_labels)
        object.__setattr__(
            self,
            "min_margin",
            _coerce_finite_float("min_margin", self.min_margin, minimum=0.0),
        )
        object.__setattr__(
            self,
            "normalize_embeddings",
            _coerce_bool("normalize_embeddings", self.normalize_embeddings),
        )
        object.__setattr__(
            self,
            "normalize_centroids",
            _coerce_bool("normalize_centroids", self.normalize_centroids),
        )
        object.__setattr__(
            self,
            "reference_date",
            _normalize_optional_iso_date("reference_date", self.reference_date),
        )
        object.__setattr__(
            self,
            "runtime_format",
            _coerce_required_str("runtime_format", self.runtime_format).lower(),
        )
        object.__setattr__(
            self,
            "model_filename",
            _normalize_relative_path("model_filename", self.model_filename),
        )
        object.__setattr__(
            self,
            "tokenizer_filename",
            _normalize_relative_path("tokenizer_filename", self.tokenizer_filename)
            or "tokenizer.json",
        )
        object.__setattr__(
            self,
            "classifier_format",
            _coerce_required_str("classifier_format", self.classifier_format).lower(),
        )
        object.__setattr__(
            self,
            "centroids_filename",
            _normalize_relative_path("centroids_filename", self.centroids_filename),
        )
        object.__setattr__(
            self,
            "weights_filename",
            _normalize_relative_path("weights_filename", self.weights_filename),
        )
        object.__setattr__(
            self,
            "bias_filename",
            _normalize_relative_path("bias_filename", self.bias_filename),
        )
        object.__setattr__(
            self,
            "classifier_filename",
            _normalize_relative_path("classifier_filename", self.classifier_filename),
        )
        object.__setattr__(self, "bundle_id", _coerce_optional_str("bundle_id", self.bundle_id))
        object.__setattr__(
            self,
            "created_at",
            _normalize_optional_iso_datetime("created_at", self.created_at),
        )
        object.__setattr__(
            self,
            "refinement_strategy",
            _coerce_required_str("refinement_strategy", self.refinement_strategy).lower(),
        )
        object.__setattr__(self, "files", normalized_files)

        if self.schema_version not in _SUPPORTED_BUNDLE_SCHEMA_VERSIONS:
            raise ValueError(f"Unsupported semantic router bundle schema: {self.schema_version}")
        if self.classifier_type not in _SUPPORTED_CLASSIFIER_TYPES:
            raise ValueError(f"Unsupported semantic router classifier type: {self.classifier_type}")
        if self.pooling not in _SUPPORTED_POOLING:
            raise ValueError(f"Unsupported semantic router pooling mode: {self.pooling}")
        if self.runtime_format not in _SUPPORTED_RUNTIME_FORMATS:
            raise ValueError(f"Unsupported runtime_format: {self.runtime_format}")
        if self.classifier_format not in _SUPPORTED_CLASSIFIER_FORMATS:
            raise ValueError(f"Unsupported classifier_format: {self.classifier_format}")
        if self.refinement_strategy not in _SUPPORTED_REFINEMENT_STRATEGIES:
            raise ValueError(f"Unsupported refinement_strategy: {self.refinement_strategy}")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SemanticRouterBundleMetadata":
        """Build metadata from one JSON-compatible mapping."""

        if not isinstance(payload, Mapping):
            raise TypeError("Semantic router metadata payload must be a mapping.")
        return cls(
            schema_version=_payload_value(payload, "schema_version", 1),
            classifier_type=_payload_value(payload, "classifier_type", "embedding_centroid_v1"),
            labels=tuple(_payload_value(payload, "labels", ()) or ()),
            model_id=_payload_value(payload, "model_id", ""),
            max_length=_payload_value(payload, "max_length", 128),
            pooling=_payload_value(payload, "pooling", "mean"),
            output_name=_payload_value(payload, "output_name", None),
            temperature=_payload_value(payload, "temperature", 1.0),
            thresholds=_payload_value(payload, "thresholds", None),
            authoritative_labels=tuple(
                _payload_value(payload, "authoritative_labels", ("web", "memory", "tool"))
                or ()
            ),
            min_margin=_payload_value(payload, "min_margin", 0.0),
            normalize_embeddings=_payload_value(payload, "normalize_embeddings", True),
            normalize_centroids=_payload_value(payload, "normalize_centroids", True),
            reference_date=_payload_value(payload, "reference_date", None),
            runtime_format=_payload_value(payload, "runtime_format", "auto"),
            model_filename=_payload_value(payload, "model_filename", None),
            tokenizer_filename=_payload_value(payload, "tokenizer_filename", "tokenizer.json"),
            classifier_format=_payload_value(payload, "classifier_format", "auto"),
            centroids_filename=_payload_value(payload, "centroids_filename", None),
            weights_filename=_payload_value(payload, "weights_filename", None),
            bias_filename=_payload_value(payload, "bias_filename", None),
            classifier_filename=_payload_value(payload, "classifier_filename", None),
            bundle_id=_payload_value(payload, "bundle_id", None),
            created_at=_payload_value(payload, "created_at", None),
            refinement_strategy=_payload_value(payload, "refinement_strategy", "none"),
            files=_payload_value(payload, "files", None),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable metadata payload."""

        return {
            "schema_version": self.schema_version,
            "classifier_type": self.classifier_type,
            "labels": list(self.labels),
            "model_id": self.model_id,
            "max_length": self.max_length,
            "pooling": self.pooling,
            "output_name": self.output_name,
            "temperature": self.temperature,
            "thresholds": dict(self.thresholds or {}),
            "authoritative_labels": list(self.authoritative_labels),
            "min_margin": self.min_margin,
            "normalize_embeddings": self.normalize_embeddings,
            "normalize_centroids": self.normalize_centroids,
            "reference_date": self.reference_date,
            "runtime_format": self.runtime_format,
            "model_filename": self.model_filename,
            "tokenizer_filename": self.tokenizer_filename,
            "classifier_format": self.classifier_format,
            "centroids_filename": self.centroids_filename,
            "weights_filename": self.weights_filename,
            "bias_filename": self.bias_filename,
            "classifier_filename": self.classifier_filename,
            "bundle_id": self.bundle_id,
            "created_at": self.created_at,
            "refinement_strategy": self.refinement_strategy,
            "files": {path: dict(spec) for path, spec in dict(self.files or {}).items()},
        }

    def policy(self) -> SemanticRouterPolicy:
        """Return the authority policy embedded in the bundle metadata."""

        return SemanticRouterPolicy(
            thresholds=dict(self.thresholds or {}),
            authoritative_labels=self.authoritative_labels,
            min_margin=self.min_margin,
        )


@dataclass(frozen=True, slots=True)
class SemanticRouterBundle:
    """Store resolved bundle paths plus validated metadata."""

    root_dir: Path
    metadata: SemanticRouterBundleMetadata
    model_path: Path
    tokenizer_path: Path
    centroids_path: Path | None
    weights_path: Path | None
    bias_path: Path | None
    metadata_path: Path
    runtime_format: str
    classifier_format: str
    classifier_artifacts: Mapping[str, BundleArtifact] = field(default_factory=dict)
    artifacts: Mapping[str, BundleArtifact] = field(default_factory=dict)
    manifest_path: Path | None = None
    integrity_verified: bool = False


def load_semantic_router_bundle(path: str | Path) -> SemanticRouterBundle:
    """Load one semantic-router bundle from disk and validate required files."""

    root_dir = _resolve_bundle_root(path)
    metadata_relpath = "router_metadata.json"
    metadata_path = _resolve_bundle_file(root_dir, metadata_relpath, role="router metadata")
    payload = _read_strict_json_file(metadata_path, max_bytes=_METADATA_MAX_BYTES)
    metadata = SemanticRouterBundleMetadata.from_dict(payload)

    manifest_path, manifest_expectations = _load_optional_manifest(root_dir)
    metadata_expectations = _normalize_manifest_entries(metadata.files)
    merged_expectations = _merge_manifest_expectations(
        metadata_expectations,
        manifest_expectations,
    )

    tokenizer_relpath = metadata.tokenizer_filename or "tokenizer.json"
    tokenizer_path = _resolve_bundle_file(root_dir, tokenizer_relpath, role="tokenizer")

    model_relpath, model_path, resolved_runtime_format = _resolve_model_artifact(root_dir, metadata)
    classifier_format, classifier_relpaths, classifier_paths = _resolve_classifier_artifacts(
        root_dir,
        metadata,
    )

    external_data_relpaths: tuple[str, ...] = ()
    if resolved_runtime_format == "onnx":
        external_data_relpaths = _validate_onnx_model_artifact(model_path, root_dir)

    metadata = replace(
        metadata,
        runtime_format=resolved_runtime_format,
        model_filename=model_relpath,
        classifier_format=classifier_format,
        centroids_filename=classifier_relpaths.get("centroids"),
        weights_filename=classifier_relpaths.get("weights"),
        bias_filename=classifier_relpaths.get("bias"),
        classifier_filename=classifier_relpaths.get("classifier"),
    )

    artifacts: dict[str, BundleArtifact] = {}
    artifacts["metadata"] = _make_artifact(
        role="metadata",
        relative_path=metadata_relpath,
        path=metadata_path,
        expected=merged_expectations.get(metadata_relpath),
    )
    artifacts["tokenizer"] = _make_artifact(
        role="tokenizer",
        relative_path=tokenizer_relpath,
        path=tokenizer_path,
        expected=merged_expectations.get(tokenizer_relpath),
    )
    artifacts["model"] = _make_artifact(
        role="model",
        relative_path=model_relpath,
        path=model_path,
        expected=merged_expectations.get(model_relpath),
    )

    classifier_artifacts: dict[str, BundleArtifact] = {}
    for role, relpath in classifier_relpaths.items():
        artifact = _make_artifact(
            role=role,
            relative_path=relpath,
            path=classifier_paths[role],
            expected=merged_expectations.get(relpath),
        )
        classifier_artifacts[role] = artifact
        artifacts[role] = artifact

    for external_relpath in external_data_relpaths:
        role = f"external_data:{external_relpath}"
        if role not in artifacts:
            path_obj = _resolve_bundle_file(root_dir, external_relpath, role=role)
            artifacts[role] = _make_artifact(
                role=role,
                relative_path=external_relpath,
                path=path_obj,
                expected=merged_expectations.get(external_relpath),
            )

    integrity_verified = _verify_manifest_coverage_and_integrity(
        manifest_path=manifest_path,
        expectations=merged_expectations,
        required_relative_paths=(
            metadata_relpath,
            tokenizer_relpath,
            model_relpath,
            *classifier_relpaths.values(),
            *external_data_relpaths,
        ),
        artifacts=artifacts,
    )

    centroids_path = classifier_paths.get("centroids")
    weights_path = classifier_paths.get("weights")
    bias_path = classifier_paths.get("bias")

    # BREAKING: when both model.ort and model.onnx exist and runtime_format="auto",
    # we now prefer model.ort because ORT format is the current edge-first deployment target.
    # BREAKING: when classifier_format != "npy", downstream code must read
    # bundle.classifier_artifacts instead of assuming *_path fields are populated.
    return SemanticRouterBundle(
        root_dir=root_dir,
        metadata=metadata,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        centroids_path=centroids_path,
        weights_path=weights_path,
        bias_path=bias_path,
        metadata_path=metadata_path,
        runtime_format=resolved_runtime_format,
        classifier_format=classifier_format,
        classifier_artifacts=classifier_artifacts,
        artifacts=artifacts,
        manifest_path=manifest_path,
        integrity_verified=integrity_verified,
    )


def _payload_value(payload: Mapping[str, Any], key: str, default: Any) -> Any:
    sentinel = object()
    value = payload.get(key, sentinel)
    return default if value is sentinel else value


def _coerce_required_str(name: str, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} must be a non-empty string.")
    return text


def _coerce_optional_str(name: str, value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_bool(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean.")


def _coerce_int(name: str, value: Any, *, minimum: int | None = None) -> int:
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return result


def _coerce_finite_float(name: str, value: Any, *, minimum: float | None = None) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return result


def _normalize_optional_iso_date(name: str, value: Any) -> str | None:
    text = _coerce_optional_str(name, value)
    if text is None:
        return None
    date.fromisoformat(text)
    return text


def _normalize_optional_iso_datetime(name: str, value: Any) -> str | None:
    text = _coerce_optional_str(name, value)
    if text is None:
        return None
    text_value = str(text)
    if text_value.endswith("Z"):
        normalized = text_value.removesuffix("Z") + "+00:00"
    else:
        normalized = text_value
    datetime.fromisoformat(normalized)
    return text


def _normalize_relative_path(name: str, value: Any) -> str | None:
    text = _coerce_optional_str(name, value)
    if text is None:
        return None
    candidate = PurePosixPath(text.replace("\\", "/"))
    if candidate.is_absolute():
        raise ValueError(f"{name} must be a relative path, got absolute path {text!r}.")
    if not candidate.parts:
        raise ValueError(f"{name} must not be empty.")
    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"{name} must not contain '.', '..', or empty segments: {text!r}.")
    return candidate.as_posix()


def _normalize_threshold_mapping(payload: Mapping[str, Any] | None) -> dict[str, float]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise TypeError("thresholds must be a mapping.")
    normalized: dict[str, float] = {}
    for label, value in payload.items():
        normalized_label = normalize_route_label(label)
        normalized[normalized_label] = _coerce_finite_float(
            f"thresholds[{normalized_label!r}]",
            value,
        )
    return normalized


def _normalize_authoritative_labels(
    values: tuple[str, ...] | list[str] | Any,
    *,
    valid_labels: tuple[str, ...],
) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("authoritative_labels must be a sequence.")
    seen: set[str] = set()
    normalized: list[str] = []
    for raw in values:
        label = normalize_route_label(raw)
        if label not in valid_labels:
            raise ValueError(
                f"authoritative_labels contains {label!r} which is not present in labels."
            )
        if label in seen:
            raise ValueError(f"Duplicate authoritative label: {label!r}")
        seen.add(label)
        normalized.append(label)
    return tuple(normalized)


def _normalize_manifest_entries(
    payload: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise TypeError("files must be a mapping of relative path -> metadata.")
    normalized: dict[str, dict[str, Any]] = {}
    for relative_path, raw_spec in payload.items():
        normalized_path = _normalize_relative_path("files key", relative_path)
        if normalized_path is None:
            raise ValueError("files key must not be empty.")
        if isinstance(raw_spec, str):
            normalized_spec = {"sha256": _normalize_hex_digest(raw_spec, algorithm="sha256")}
        elif isinstance(raw_spec, Mapping):
            normalized_spec: dict[str, Any] = {}
            if "sha256" in raw_spec and raw_spec["sha256"] is not None:
                normalized_spec["sha256"] = _normalize_hex_digest(
                    raw_spec["sha256"],
                    algorithm="sha256",
                )
            size_value = raw_spec.get("size_bytes", raw_spec.get("size"))
            if size_value is not None:
                size_bytes = _coerce_int(
                    f"files[{normalized_path!r}].size_bytes",
                    size_value,
                    minimum=0,
                )
                normalized_spec["size_bytes"] = size_bytes
        else:
            raise TypeError(
                f"files[{normalized_path!r}] must be a mapping or a SHA-256 hex digest string."
            )
        normalized[normalized_path] = normalized_spec
    return normalized


def _normalize_hex_digest(value: Any, *, algorithm: str) -> str:
    text = _coerce_required_str(f"{algorithm} digest", value).lower()
    digest_lengths = {"sha256": 64, "sha1": 40}
    expected_length = digest_lengths[algorithm]
    allowed = set("0123456789abcdef")
    if len(text) != expected_length or any(ch not in allowed for ch in text):
        raise ValueError(f"Invalid {algorithm} digest: {value!r}")
    return text


def _resolve_bundle_root(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    try:
        root_dir = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Semantic router bundle root does not exist: {candidate}") from exc
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Semantic router bundle root is not a directory: {root_dir}")
    return root_dir


def _is_within_root(root_dir: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root_dir)
        return True
    except ValueError:
        return False


def _resolve_bundle_file(root_dir: Path, relative_path: str, *, role: str) -> Path:
    normalized_relpath = _normalize_relative_path(role, relative_path)
    if normalized_relpath is None:
        raise ValueError(f"{role} relative path is required.")
    unresolved = root_dir / Path(normalized_relpath)
    try:
        resolved = unresolved.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Semantic router bundle at {root_dir} is missing required {role}: {normalized_relpath}"
        ) from exc
    if not _is_within_root(root_dir, resolved):
        raise PermissionError(
            f"Resolved {role} path escapes the bundle root: {normalized_relpath} -> {resolved}"
        )
    _require_regular_file(resolved, role=role)
    return resolved


def _try_resolve_bundle_file(root_dir: Path, relative_path: str, *, role: str) -> Path | None:
    normalized_relpath = _normalize_relative_path(role, relative_path)
    if normalized_relpath is None:
        return None
    unresolved = root_dir / Path(normalized_relpath)
    try:
        resolved = unresolved.resolve(strict=True)
    except FileNotFoundError:
        return None
    if not _is_within_root(root_dir, resolved):
        raise PermissionError(
            f"Resolved {role} path escapes the bundle root: {normalized_relpath} -> {resolved}"
        )
    _require_regular_file(resolved, role=role)
    return resolved


def _require_regular_file(path: Path, *, role: str) -> None:
    try:
        file_stat = path.stat()
    except OSError as exc:
        raise OSError(f"Unable to stat {role} at {path}: {exc}") from exc
    if not statmod.S_ISREG(file_stat.st_mode):
        raise ValueError(f"{role} must be a regular file: {path}")
    if file_stat.st_size <= 0:
        raise ValueError(f"{role} must not be empty: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"{role} is not readable: {path}")


def _read_strict_json_file(path: Path, *, max_bytes: int) -> Mapping[str, Any]:
    _require_regular_file(path, role=str(path.name))
    file_size = path.stat().st_size
    if file_size > max_bytes:
        raise ValueError(f"JSON file {path} exceeds maximum allowed size of {max_bytes} bytes.")
    text = path.read_text(encoding="utf-8")
    payload = json.loads(
        text,
        object_pairs_hook=_reject_duplicate_object_keys,
        parse_constant=_reject_nonfinite_json_constant,
    )
    if not isinstance(payload, Mapping):
        raise TypeError(f"JSON file {path} must contain a top-level object.")
    return payload


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(token: str) -> None:
    raise ValueError(f"Non-finite JSON constant is not allowed in bundle metadata: {token}")


def _load_optional_manifest(root_dir: Path) -> tuple[Path | None, dict[str, dict[str, Any]]]:
    manifest_relpath = "router_manifest.json"
    manifest_path = _try_resolve_bundle_file(root_dir, manifest_relpath, role="router manifest")
    if manifest_path is None:
        return None, {}
    payload = _read_strict_json_file(manifest_path, max_bytes=_MANIFEST_MAX_BYTES)
    files_payload = payload.get("files", payload)
    if not isinstance(files_payload, Mapping):
        raise TypeError(
            "router_manifest.json must either be a files mapping or contain a top-level 'files' mapping."
        )
    return manifest_path, _normalize_manifest_entries(files_payload)  # type: ignore[arg-type]


def _merge_manifest_expectations(
    first: Mapping[str, dict[str, Any]],
    second: Mapping[str, dict[str, Any]],
) -> dict[str, BundleFileExpectation]:
    merged: dict[str, BundleFileExpectation] = {}
    for source in (first, second):
        for relative_path, spec in source.items():
            current = BundleFileExpectation(
                relative_path=relative_path,
                size_bytes=spec.get("size_bytes"),
                sha256=spec.get("sha256"),
            )
            existing = merged.get(relative_path)
            if existing is None:
                merged[relative_path] = current
                continue
            if existing != current:
                raise ValueError(
                    f"Conflicting integrity metadata for {relative_path!r}: {existing} vs {current}"
                )
    return merged


def _resolve_model_artifact(
    root_dir: Path,
    metadata: SemanticRouterBundleMetadata,
) -> tuple[str, Path, str]:
    if metadata.model_filename:
        resolved = _resolve_bundle_file(root_dir, metadata.model_filename, role="model")
        suffix = resolved.suffix.lower()
        if suffix not in {".onnx", ".ort"}:
            raise ValueError(
                f"Unsupported model artifact extension for {metadata.model_filename!r}: expected .onnx or .ort"
            )
        runtime_format = "ort" if suffix == ".ort" else "onnx"
        if metadata.runtime_format != "auto" and runtime_format != metadata.runtime_format:
            raise ValueError(
                f"model_filename={metadata.model_filename!r} does not match runtime_format={metadata.runtime_format!r}"
            )
        return metadata.model_filename, resolved, runtime_format

    ordered_candidates = {
        "ort": ("model.ort",),
        "onnx": ("model.onnx",),
        "auto": ("model.ort", "model.onnx"),
    }[metadata.runtime_format]
    for candidate_relpath in ordered_candidates:
        resolved = _try_resolve_bundle_file(root_dir, candidate_relpath, role="model")
        if resolved is None:
            continue
        return candidate_relpath, resolved, ("ort" if candidate_relpath.endswith(".ort") else "onnx")

    raise FileNotFoundError(
        f"Semantic router bundle at {root_dir} is missing a supported model artifact: "
        + ", ".join(ordered_candidates)
    )


def _resolve_classifier_artifacts(
    root_dir: Path,
    metadata: SemanticRouterBundleMetadata,
) -> tuple[str, dict[str, str], dict[str, Path]]:
    if metadata.classifier_type == "embedding_centroid_v1":
        return _resolve_centroid_artifacts(root_dir, metadata)
    return _resolve_linear_artifacts(root_dir, metadata)


def _resolve_centroid_artifacts(
    root_dir: Path,
    metadata: SemanticRouterBundleMetadata,
) -> tuple[str, dict[str, str], dict[str, Path]]:
    npy_candidates = tuple(
        candidate
        for candidate in (
            metadata.centroids_filename,
            "centroids.npy",
        )
        if candidate
    )
    safetensors_candidates = tuple(
        candidate
        for candidate in (
            metadata.classifier_filename,
            metadata.centroids_filename,
            "centroids.safetensors",
            "classifier.safetensors",
        )
        if candidate
    )

    if metadata.classifier_format in {"auto", "npy"}:
        for relpath in dict.fromkeys(npy_candidates):
            resolved = _try_resolve_bundle_file(root_dir, relpath, role="centroids")
            if resolved is not None:
                return "npy", {"centroids": relpath}, {"centroids": resolved}
    if metadata.classifier_format in {"auto", "safetensors"}:
        for relpath in dict.fromkeys(safetensors_candidates):
            resolved = _try_resolve_bundle_file(root_dir, relpath, role="centroid classifier")
            if resolved is not None:
                return "safetensors", {"classifier": relpath}, {"classifier": resolved}

    expected = list(dict.fromkeys((*npy_candidates, *safetensors_candidates)))
    raise FileNotFoundError(
        f"Semantic router bundle at {root_dir} is missing required centroid classifier artifacts: "
        + ", ".join(expected or ("centroids.npy", "centroids.safetensors"))
    )


def _resolve_linear_artifacts(
    root_dir: Path,
    metadata: SemanticRouterBundleMetadata,
) -> tuple[str, dict[str, str], dict[str, Path]]:
    weights_relpath = metadata.weights_filename or "weights.npy"
    bias_relpath = metadata.bias_filename or "bias.npy"

    if metadata.classifier_format in {"auto", "npy"}:
        weights_path = _try_resolve_bundle_file(root_dir, weights_relpath, role="weights")
        bias_path = _try_resolve_bundle_file(root_dir, bias_relpath, role="bias")
        if weights_path is not None and bias_path is not None:
            return (
                "npy",
                {"weights": weights_relpath, "bias": bias_relpath},
                {"weights": weights_path, "bias": bias_path},
            )
        if metadata.classifier_format == "npy" and (weights_path is not None or bias_path is not None):
            missing = [
                name
                for name, obj in (("weights", weights_path), ("bias", bias_path))
                if obj is None
            ]
            raise FileNotFoundError(
                f"Semantic router bundle at {root_dir} is missing required linear classifier files: "
                + ", ".join(missing)
            )

    safetensors_candidates = tuple(
        candidate
        for candidate in (
            metadata.classifier_filename,
            "classifier.safetensors",
            "weights.safetensors",
        )
        if candidate
    )
    if metadata.classifier_format in {"auto", "safetensors"}:
        for relpath in dict.fromkeys(safetensors_candidates):
            resolved = _try_resolve_bundle_file(root_dir, relpath, role="linear classifier")
            if resolved is not None:
                return "safetensors", {"classifier": relpath}, {"classifier": resolved}

    raise FileNotFoundError(
        f"Semantic router bundle at {root_dir} is missing required linear classifier artifacts: "
        + ", ".join(dict.fromkeys((weights_relpath, bias_relpath, *safetensors_candidates)))
    )


def _make_artifact(
    *,
    role: str,
    relative_path: str,
    path: Path,
    expected: BundleFileExpectation | None,
) -> BundleArtifact:
    _require_regular_file(path, role=role)
    file_stat = path.stat()
    actual_sha256: str | None = None
    if expected is not None:
        if expected.size_bytes is not None and file_stat.st_size != expected.size_bytes:
            raise ValueError(
                f"{role} size mismatch for {relative_path!r}: expected {expected.size_bytes}, got {file_stat.st_size}"
            )
        if expected.sha256 is not None:
            actual_sha256 = _hash_file(path, algorithm="sha256")
            if actual_sha256 != expected.sha256:
                raise ValueError(
                    f"{role} SHA-256 mismatch for {relative_path!r}: expected {expected.sha256}, got {actual_sha256}"
                )
    return BundleArtifact(
        role=role,
        relative_path=relative_path,
        path=path,
        size_bytes=file_stat.st_size,
        sha256=actual_sha256,
    )


def _verify_manifest_coverage_and_integrity(
    *,
    manifest_path: Path | None,
    expectations: Mapping[str, BundleFileExpectation],
    required_relative_paths: tuple[str, ...],
    artifacts: Mapping[str, BundleArtifact],
) -> bool:
    if not expectations:
        return False

    required_set = set(required_relative_paths)
    if manifest_path is not None and not required_set.issubset(expectations.keys()):
        missing = sorted(required_set.difference(expectations.keys()))
        raise ValueError(
            f"router_manifest.json exists but does not cover all required bundle artifacts: {', '.join(missing)}"
        )
    hashed_required = {artifact.relative_path for artifact in artifacts.values() if artifact.sha256 is not None}
    if manifest_path is not None and not required_set.issubset(hashed_required):
        missing = sorted(required_set.difference(hashed_required))
        raise ValueError(
            f"router_manifest.json exists but SHA-256 verification did not cover: {', '.join(missing)}"
        )
    return required_set.issubset(hashed_required)


def _hash_file(path: Path, *, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK_BYTES), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _validate_onnx_model_artifact(model_path: Path, root_dir: Path) -> tuple[str, ...]:
    try:
        import onnx
        from onnx.external_data_helper import uses_external_data
    except Exception:
        return ()

    try:
        onnx.checker.check_model(str(model_path))
    except Exception as exc:  # pragma: no cover - optional dependency path
        if not _onnxruntime_can_load_model(model_path):
            raise ValueError(f"ONNX model validation failed for {model_path}: {exc}") from exc

    try:
        model = onnx.load(str(model_path), load_external_data=False)
    except TypeError:
        return ()

    external_paths: list[str] = []
    for tensor in _iter_onnx_tensors(model):
        if not uses_external_data(tensor):
            continue
        info = {entry.key: entry.value for entry in tensor.external_data}
        location = info.get("location")
        if not location:
            raise ValueError(f"External ONNX tensor {tensor.name!r} is missing its location field.")
        relpath = _normalize_relative_path(
            f"external data for tensor {tensor.name!r}",
            location,
        )
        if relpath is None:
            raise ValueError(f"External ONNX tensor {tensor.name!r} has an empty location.")
        path_obj = _resolve_bundle_file(
            root_dir,
            relpath,
            role=f"external data for tensor {tensor.name!r}",
        )
        checksum = info.get("checksum")
        if checksum:
            normalized_checksum = _normalize_hex_digest(checksum, algorithm="sha1")
            actual_checksum = _hash_file(path_obj, algorithm="sha1")
            if actual_checksum != normalized_checksum:
                raise ValueError(
                    f"External data checksum mismatch for tensor {tensor.name!r}: "
                    f"expected {normalized_checksum}, got {actual_checksum}"
                )
        external_paths.append(relpath)
    return tuple(dict.fromkeys(external_paths))


def _onnxruntime_can_load_model(model_path: Path) -> bool:
    try:
        import onnxruntime as ort
    except Exception:
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


def _iter_onnx_tensors(model: Any) -> Iterator[Any]:
    yield from _iter_onnx_tensors_from_graph(model.graph)
    for function in getattr(model, "functions", ()):
        yield from _iter_onnx_attribute_tensors(function)


def _iter_onnx_tensors_from_graph(graph: Any) -> Iterator[Any]:
    yield from graph.initializer
    yield from _iter_onnx_attribute_tensors(graph)


def _iter_onnx_attribute_tensors(graph_or_function: Any) -> Iterator[Any]:
    for node in graph_or_function.node:
        for attribute in node.attribute:
            if attribute.HasField("t"):
                yield attribute.t
            yield from attribute.tensors
            if getattr(attribute, "g", None):
                yield from _iter_onnx_tensors_from_graph(attribute.g)
            for nested_graph in getattr(attribute, "graphs", ()):
                yield from _iter_onnx_tensors_from_graph(nested_graph)
