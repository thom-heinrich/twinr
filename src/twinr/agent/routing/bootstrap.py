"""Build local semantic-router bundles from labeled transcript datasets.

Twinr router bundles reuse one ONNX sentence encoder and attach either
per-label embedding centroids or one trained linear softmax head. This keeps
the on-device runtime tiny while still giving the project a reproducible path
for calibration on real transcript datasets.

Example:

```bash
PYTHONPATH=src python3 -m twinr.agent.routing.bootstrap \
  --source-dir /models/paraphrase-multilingual-MiniLM-L12-v2-onnx \
  --dataset test/fixtures/router/router_samples.jsonl \
  --output-dir artifacts/router/multilingual-minilm-router
```
"""

from __future__ import annotations

from dataclasses import replace
import argparse
import json
from pathlib import Path
import shutil
from typing import Sequence

import numpy as np

from .bundle import SemanticRouterBundleMetadata, load_semantic_router_bundle
from .contracts import ROUTE_LABEL_VALUES
from .evaluation import LabeledRouteSample, load_labeled_route_samples, score_semantic_router, tune_policy_thresholds
from .linear_head import fit_multiclass_linear_head
from .service import LocalSemanticRouter, OnnxSentenceEncoder


def compute_label_centroids(
    embeddings: np.ndarray,
    samples: Sequence[LabeledRouteSample],
    *,
    labels: Sequence[str] = ROUTE_LABEL_VALUES,
) -> np.ndarray:
    """Return one centroid per label in the requested order."""

    if embeddings.shape[0] != len(samples):
        raise ValueError("Embedding rows must match the number of labeled samples.")
    rows: list[np.ndarray] = []
    for label in labels:
        matching_rows = [
            embeddings[index]
            for index, sample in enumerate(samples)
            if sample.label == label
        ]
        if not matching_rows:
            raise ValueError(f"Cannot build router bundle without training samples for {label!r}.")
        rows.append(np.mean(np.vstack(matching_rows), axis=0))
    return np.vstack(rows).astype(np.float32)


def build_centroid_router_bundle(
    *,
    source_dir: str | Path,
    output_dir: str | Path,
    train_samples: Sequence[LabeledRouteSample],
    dev_samples: Sequence[LabeledRouteSample] = (),
    model_id: str | None = None,
    max_length: int = 128,
    pooling: str = "mean",
    output_name: str | None = None,
    reference_date: str = "2026-03-22",
) -> Path:
    """Build one centroid-based router bundle from local source model files."""

    source_path = Path(source_dir).expanduser().resolve(strict=False)
    output_path = Path(output_dir).expanduser().resolve(strict=False)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = source_path / "model.onnx"
    tokenizer_path = source_path / "tokenizer.json"
    if not model_path.exists() or not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Semantic router source dir {source_path} must contain model.onnx and tokenizer.json"
        )
    encoder = OnnxSentenceEncoder(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        max_length=max_length,
        pooling=pooling,
        output_name=output_name,
        normalize_embeddings=True,
    )
    train_embeddings = encoder.encode([sample.text for sample in train_samples])
    centroids = compute_label_centroids(train_embeddings, train_samples)
    metadata = SemanticRouterBundleMetadata(
        schema_version=1,
        classifier_type="embedding_centroid_v1",
        labels=ROUTE_LABEL_VALUES,
        model_id=model_id or source_path.name,
        max_length=max_length,
        pooling=pooling,
        output_name=output_name,
        thresholds={"web": 0.6, "memory": 0.6, "tool": 0.6, "parametric": 1.0},
        authoritative_labels=("web", "memory", "tool"),
        min_margin=0.08,
        normalize_embeddings=True,
        normalize_centroids=True,
        reference_date=reference_date,
    )
    shutil.copy2(model_path, output_path / "model.onnx")
    shutil.copy2(tokenizer_path, output_path / "tokenizer.json")
    np.save(output_path / "centroids.npy", centroids)
    (output_path / "router_metadata.json").write_text(
        json.dumps(metadata.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if dev_samples:
        bundle = load_semantic_router_bundle(output_path)
        router = LocalSemanticRouter(
            bundle,
            encoder=encoder,
            centroids=centroids,
        )
        tuned_policy = tune_policy_thresholds(score_semantic_router(router, dev_samples))
        metadata = replace(
            metadata,
            thresholds=dict(tuned_policy.thresholds),
            authoritative_labels=tuned_policy.authoritative_labels,
            min_margin=tuned_policy.min_margin,
        )
        (output_path / "router_metadata.json").write_text(
            json.dumps(metadata.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return output_path


def build_linear_router_bundle(
    *,
    source_dir: str | Path,
    output_dir: str | Path,
    train_samples: Sequence[LabeledRouteSample],
    dev_samples: Sequence[LabeledRouteSample] = (),
    model_id: str | None = None,
    max_length: int = 128,
    pooling: str = "mean",
    output_name: str | None = None,
    reference_date: str = "2026-03-22",
) -> Path:
    """Build one linear-head router bundle from local source model files."""

    source_path = Path(source_dir).expanduser().resolve(strict=False)
    output_path = Path(output_dir).expanduser().resolve(strict=False)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = source_path / "model.onnx"
    tokenizer_path = source_path / "tokenizer.json"
    if not model_path.exists() or not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Semantic router source dir {source_path} must contain model.onnx and tokenizer.json"
        )
    encoder = OnnxSentenceEncoder(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        max_length=max_length,
        pooling=pooling,
        output_name=output_name,
        normalize_embeddings=True,
    )
    train_embeddings = encoder.encode([sample.text for sample in train_samples])
    dev_embeddings = (
        encoder.encode([sample.text for sample in dev_samples])
        if dev_samples
        else None
    )
    trained_head = fit_multiclass_linear_head(
        train_embeddings,
        [sample.label for sample in train_samples],
        label_order=ROUTE_LABEL_VALUES,
        dev_embeddings=dev_embeddings,
        dev_labels=[sample.label for sample in dev_samples] if dev_samples else None,
    )
    metadata = SemanticRouterBundleMetadata(
        schema_version=1,
        classifier_type="embedding_linear_softmax_v1",
        labels=ROUTE_LABEL_VALUES,
        model_id=model_id or source_path.name,
        max_length=max_length,
        pooling=pooling,
        output_name=output_name,
        thresholds={"web": 0.6, "memory": 0.6, "tool": 0.6, "parametric": 1.0},
        authoritative_labels=("web", "memory", "tool"),
        min_margin=0.08,
        normalize_embeddings=True,
        normalize_centroids=False,
        reference_date=reference_date,
    )
    shutil.copy2(model_path, output_path / "model.onnx")
    shutil.copy2(tokenizer_path, output_path / "tokenizer.json")
    np.save(output_path / "weights.npy", trained_head.weights)
    np.save(output_path / "bias.npy", trained_head.bias)
    (output_path / "router_metadata.json").write_text(
        json.dumps(metadata.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if dev_samples:
        bundle = load_semantic_router_bundle(output_path)
        router = LocalSemanticRouter(
            bundle,
            encoder=encoder,
        )
        tuned_policy = tune_policy_thresholds(score_semantic_router(router, dev_samples))
        metadata = replace(
            metadata,
            thresholds=dict(tuned_policy.thresholds),
            authoritative_labels=tuned_policy.authoritative_labels,
            min_margin=tuned_policy.min_margin,
        )
        (output_path / "router_metadata.json").write_text(
            json.dumps(metadata.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return output_path


def build_centroid_router_bundle_from_jsonl(
    *,
    source_dir: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    model_id: str | None = None,
    max_length: int = 128,
    pooling: str = "mean",
    output_name: str | None = None,
    reference_date: str = "2026-03-22",
) -> Path:
    """Build one bundle from a JSONL dataset with optional `train`/`dev` splits."""

    samples = load_labeled_route_samples(dataset_path)
    train_samples = tuple(sample for sample in samples if (sample.split or "train") == "train")
    dev_samples = tuple(sample for sample in samples if sample.split == "dev")
    if not train_samples:
        train_samples = samples
    return build_centroid_router_bundle(
        source_dir=source_dir,
        output_dir=output_dir,
        train_samples=train_samples,
        dev_samples=dev_samples,
        model_id=model_id,
        max_length=max_length,
        pooling=pooling,
        output_name=output_name,
        reference_date=reference_date,
    )


def build_linear_router_bundle_from_jsonl(
    *,
    source_dir: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    model_id: str | None = None,
    max_length: int = 128,
    pooling: str = "mean",
    output_name: str | None = None,
    reference_date: str = "2026-03-22",
) -> Path:
    """Build one linear-head router bundle from a JSONL dataset."""

    samples = load_labeled_route_samples(dataset_path)
    train_samples = tuple(sample for sample in samples if (sample.split or "train") == "train")
    dev_samples = tuple(sample for sample in samples if sample.split == "dev")
    if not train_samples:
        train_samples = samples
    return build_linear_router_bundle(
        source_dir=source_dir,
        output_dir=output_dir,
        train_samples=train_samples,
        dev_samples=dev_samples,
        model_id=model_id,
        max_length=max_length,
        pooling=pooling,
        output_name=output_name,
        reference_date=reference_date,
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Twinr semantic-router bundle.")
    parser.add_argument("--source-dir", required=True, help="Directory containing model.onnx and tokenizer.json")
    parser.add_argument("--dataset", required=True, help="JSONL file with text/label and optional split fields")
    parser.add_argument("--output-dir", required=True, help="Destination bundle directory")
    parser.add_argument("--model-id", default=None, help="Optional bundle/model identifier override")
    parser.add_argument("--max-length", type=int, default=128, help="Tokenizer truncation length")
    parser.add_argument("--pooling", default="mean", choices=("mean", "cls", "prepooled"))
    parser.add_argument("--output-name", default=None, help="Optional explicit ONNX output name")
    parser.add_argument("--classifier", default="centroid", choices=("centroid", "linear"))
    parser.add_argument(
        "--reference-date",
        default="2026-03-22",
        help="Frozen reference date for parametric-vs-web labeling semantics",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for building local router bundles."""

    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    builder = build_linear_router_bundle_from_jsonl if args.classifier == "linear" else build_centroid_router_bundle_from_jsonl
    builder(
        source_dir=args.source_dir,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        model_id=args.model_id,
        max_length=args.max_length,
        pooling=args.pooling,
        output_name=args.output_name,
        reference_date=args.reference_date,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via module CLI.
    raise SystemExit(main())
