"""Build local user-intent bundles from labeled transcript datasets.

The first router stage can use either centroid similarity or a trained linear
softmax head over the frozen ONNX sentence embeddings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Sequence

import numpy as np

from .bootstrap import compute_label_centroids
from .linear_head import fit_multiclass_linear_head
from .service import OnnxSentenceEncoder
from .user_intent import USER_INTENT_LABEL_VALUES
from .user_intent_bundle import UserIntentBundleMetadata
from .user_intent_evaluation import (
    LabeledUserIntentSample,
    load_labeled_user_intent_samples,
)


def build_centroid_user_intent_bundle(
    *,
    source_dir: str | Path,
    output_dir: str | Path,
    train_samples: Sequence[LabeledUserIntentSample],
    model_id: str | None = None,
    max_length: int = 128,
    pooling: str = "mean",
    output_name: str | None = None,
    reference_date: str = "2026-03-22",
) -> Path:
    """Build one centroid-based user-intent bundle from local source model files."""

    source_path = Path(source_dir).expanduser().resolve(strict=False)
    output_path = Path(output_dir).expanduser().resolve(strict=False)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = source_path / "model.onnx"
    tokenizer_path = source_path / "tokenizer.json"
    if not model_path.exists() or not tokenizer_path.exists():
        raise FileNotFoundError(
            f"User intent source dir {source_path} must contain model.onnx and tokenizer.json"
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
    centroids = compute_label_centroids(
        train_embeddings,
        train_samples,
        labels=USER_INTENT_LABEL_VALUES,
    )
    metadata = UserIntentBundleMetadata(
        schema_version=1,
        classifier_type="embedding_centroid_v1",
        labels=USER_INTENT_LABEL_VALUES,
        model_id=model_id or source_path.name,
        max_length=max_length,
        pooling=pooling,
        output_name=output_name,
        normalize_embeddings=True,
        normalize_centroids=True,
        reference_date=reference_date,
    )
    shutil.copy2(model_path, output_path / "model.onnx")
    shutil.copy2(tokenizer_path, output_path / "tokenizer.json")
    np.save(output_path / "centroids.npy", centroids)
    (output_path / "user_intent_metadata.json").write_text(
        json.dumps(metadata.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def build_linear_user_intent_bundle(
    *,
    source_dir: str | Path,
    output_dir: str | Path,
    train_samples: Sequence[LabeledUserIntentSample],
    dev_samples: Sequence[LabeledUserIntentSample] = (),
    model_id: str | None = None,
    max_length: int = 128,
    pooling: str = "mean",
    output_name: str | None = None,
    reference_date: str = "2026-03-22",
) -> Path:
    """Build one linear-head user-intent bundle from local source model files."""

    source_path = Path(source_dir).expanduser().resolve(strict=False)
    output_path = Path(output_dir).expanduser().resolve(strict=False)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = source_path / "model.onnx"
    tokenizer_path = source_path / "tokenizer.json"
    if not model_path.exists() or not tokenizer_path.exists():
        raise FileNotFoundError(
            f"User intent source dir {source_path} must contain model.onnx and tokenizer.json"
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
        [str(sample.label) for sample in train_samples],
        label_order=USER_INTENT_LABEL_VALUES,
        dev_embeddings=dev_embeddings,
        dev_labels=[str(sample.label) for sample in dev_samples] if dev_samples else None,
    )
    metadata = UserIntentBundleMetadata(
        schema_version=1,
        classifier_type="embedding_linear_softmax_v1",
        labels=USER_INTENT_LABEL_VALUES,
        model_id=model_id or source_path.name,
        max_length=max_length,
        pooling=pooling,
        output_name=output_name,
        normalize_embeddings=True,
        normalize_centroids=False,
        reference_date=reference_date,
    )
    shutil.copy2(model_path, output_path / "model.onnx")
    shutil.copy2(tokenizer_path, output_path / "tokenizer.json")
    np.save(output_path / "weights.npy", trained_head.weights)
    np.save(output_path / "bias.npy", trained_head.bias)
    (output_path / "user_intent_metadata.json").write_text(
        json.dumps(metadata.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def build_centroid_user_intent_bundle_from_jsonl(
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
    """Build one user-intent bundle from a JSONL dataset with train/dev/test splits."""

    samples = load_labeled_user_intent_samples(dataset_path)
    train_samples = tuple(sample for sample in samples if (sample.split or "train") == "train")
    if not train_samples:
        train_samples = samples
    return build_centroid_user_intent_bundle(
        source_dir=source_dir,
        output_dir=output_dir,
        train_samples=train_samples,
        model_id=model_id,
        max_length=max_length,
        pooling=pooling,
        output_name=output_name,
        reference_date=reference_date,
    )


def build_linear_user_intent_bundle_from_jsonl(
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
    """Build one linear-head user-intent bundle from a JSONL dataset."""

    samples = load_labeled_user_intent_samples(dataset_path)
    train_samples = tuple(sample for sample in samples if (sample.split or "train") == "train")
    dev_samples = tuple(sample for sample in samples if sample.split == "dev")
    if not train_samples:
        train_samples = samples
    return build_linear_user_intent_bundle(
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
    parser = argparse.ArgumentParser(description="Build a Twinr user-intent bundle.")
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
        help="Frozen reference date for user-intent bootstrap provenance",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for building user-intent bundles."""

    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    builder = (
        build_linear_user_intent_bundle_from_jsonl
        if args.classifier == "linear"
        else build_centroid_user_intent_bundle_from_jsonl
    )
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
