"""Load and validate versioned local semantic-router bundles."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import ROUTE_LABEL_VALUES, normalize_route_label
from .policy import SemanticRouterPolicy


_BUNDLE_SCHEMA_VERSION = 1
_SUPPORTED_CLASSIFIER_TYPES = {
    "embedding_centroid_v1",
    "embedding_linear_softmax_v1",
}


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
    reference_date: str | None = "2026-03-22"

    def __post_init__(self) -> None:
        normalized_labels = tuple(normalize_route_label(label) for label in self.labels)
        if tuple(sorted(normalized_labels)) != tuple(sorted(ROUTE_LABEL_VALUES)):
            raise ValueError(
                "Semantic router bundles must declare exactly the parametric/web/memory/tool labels."
            )
        normalized_thresholds = {
            normalize_route_label(label): float(value)
            for label, value in dict(self.thresholds or {}).items()
        }
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "labels", normalized_labels)
        object.__setattr__(self, "model_id", str(self.model_id or "").strip())
        object.__setattr__(self, "classifier_type", str(self.classifier_type or "").strip().lower())
        object.__setattr__(self, "max_length", max(1, int(self.max_length)))
        object.__setattr__(self, "pooling", str(self.pooling or "mean").strip().lower())
        object.__setattr__(self, "output_name", str(self.output_name or "").strip() or None)
        object.__setattr__(self, "temperature", max(1e-6, float(self.temperature)))
        object.__setattr__(self, "thresholds", normalized_thresholds)
        object.__setattr__(
            self,
            "authoritative_labels",
            tuple(normalize_route_label(label) for label in self.authoritative_labels),
        )
        object.__setattr__(self, "min_margin", max(0.0, float(self.min_margin)))
        object.__setattr__(self, "normalize_embeddings", bool(self.normalize_embeddings))
        object.__setattr__(self, "normalize_centroids", bool(self.normalize_centroids))
        object.__setattr__(self, "reference_date", str(self.reference_date or "").strip() or None)
        if self.schema_version != _BUNDLE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported semantic router bundle schema: {self.schema_version}"
            )
        if self.classifier_type not in _SUPPORTED_CLASSIFIER_TYPES:
            raise ValueError(
                f"Unsupported semantic router classifier type: {self.classifier_type}"
            )
        if self.pooling not in {"mean", "cls", "prepooled"}:
            raise ValueError(f"Unsupported semantic router pooling mode: {self.pooling}")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SemanticRouterBundleMetadata":
        """Build metadata from one JSON-compatible mapping."""

        return cls(
            schema_version=payload.get("schema_version", _BUNDLE_SCHEMA_VERSION),
            classifier_type=payload.get("classifier_type", "embedding_centroid_v1"),
            labels=tuple(payload.get("labels", ()) or ()),
            model_id=str(payload.get("model_id", "") or ""),
            max_length=int(payload.get("max_length", 128) or 128),
            pooling=str(payload.get("pooling", "mean") or "mean"),
            output_name=payload.get("output_name"),
            temperature=float(payload.get("temperature", 1.0) or 1.0),
            thresholds=payload.get("thresholds"),
            authoritative_labels=tuple(payload.get("authoritative_labels", ("web", "memory", "tool")) or ()),
            min_margin=float(payload.get("min_margin", 0.0) or 0.0),
            normalize_embeddings=bool(payload.get("normalize_embeddings", True)),
            normalize_centroids=bool(payload.get("normalize_centroids", True)),
            reference_date=payload.get("reference_date", "2026-03-22"),
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


def load_semantic_router_bundle(path: str | Path) -> SemanticRouterBundle:
    """Load one semantic-router bundle from disk and validate required files."""

    root_dir = Path(path).expanduser().resolve(strict=False)
    metadata_path = root_dir / "router_metadata.json"
    model_path = root_dir / "model.onnx"
    tokenizer_path = root_dir / "tokenizer.json"
    missing = tuple(
        str(candidate.name)
        for candidate in (metadata_path, model_path, tokenizer_path)
        if not candidate.exists()
    )
    if missing:
        raise FileNotFoundError(
            f"Semantic router bundle at {root_dir} is missing required files: {', '.join(missing)}"
        )
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = SemanticRouterBundleMetadata.from_dict(payload)
    centroids_path = root_dir / "centroids.npy"
    weights_path = root_dir / "weights.npy"
    bias_path = root_dir / "bias.npy"
    if metadata.classifier_type == "embedding_centroid_v1":
        required_files = (centroids_path,)
        resolved_centroids_path = centroids_path
        resolved_weights_path = None
        resolved_bias_path = None
    else:
        required_files = (weights_path, bias_path)
        resolved_centroids_path = None
        resolved_weights_path = weights_path
        resolved_bias_path = bias_path
    classifier_missing = tuple(
        str(candidate.name)
        for candidate in required_files
        if not candidate.exists()
    )
    if classifier_missing:
        raise FileNotFoundError(
            f"Semantic router bundle at {root_dir} is missing required classifier files: "
            f"{', '.join(classifier_missing)}"
        )
    return SemanticRouterBundle(
        root_dir=root_dir,
        metadata=metadata,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        centroids_path=resolved_centroids_path,
        weights_path=resolved_weights_path,
        bias_path=resolved_bias_path,
        metadata_path=metadata_path,
    )
