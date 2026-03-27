"""Load labeled user-intent datasets and evaluate the first router stage."""

from __future__ import annotations

# CHANGELOG: 2026-03-27
# BUG-1: Fixed incorrect macro-F1 on partial evaluation slices by evaluating labels
#        observed in the slice (gold and predicted), instead of always averaging
#        across the global label universe.
# BUG-2: Fixed evaluation crashes when the router emits an unknown / malformed
#        label by normalizing predictions and tracking invalid outputs explicitly.
# BUG-3: Fixed JSONL parsing based on read_text().splitlines(); that approach can
#        split on non-JSONL Unicode line separators and loads the whole file into
#        memory at once.
# SEC-1: Hardened dataset loading against symlink / FIFO / device-file abuse and
#        oversized edge-device inputs by requiring regular files and adding
#        file-size / decompressed-size caps.
# SEC-2: Added bounded gzip support plus per-line limits so .jsonl.gz datasets
#        do not become an easy decompression-bomb / RAM-exhaustion path on
#        Raspberry Pi deployments.
# IMP-1: Added a streaming iterator API and an optional msgspec fast path for
#        typed, high-throughput JSON decoding on ARM devices.
# IMP-2: Added batch router scoring, richer evaluation metrics (weighted F1,
#        balanced accuracy, coverage, optional Brier / log-loss), and latency /
#        throughput reporting for real edge benchmarking.

from contextlib import contextmanager
from dataclasses import dataclass
import gzip
import json
import math
import os
from pathlib import Path
import stat
import time
from typing import Any, BinaryIO, Callable, Iterable, Iterator, Mapping, Sequence

from .user_intent import USER_INTENT_LABEL_VALUES, UserIntentDecision, normalize_user_intent_label

try:  # Optional frontier fast path.
    import msgspec  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    msgspec = None  # type: ignore[assignment]


# BREAKING: edge-safe defaults now reject oversized datasets unless the caller
# explicitly disables the limits with `max_file_size_bytes=None` and/or
# `max_decompressed_bytes=None`.
_DEFAULT_MAX_FILE_SIZE_BYTES = 128 * 1024 * 1024
_DEFAULT_MAX_DECOMPRESSED_BYTES = 128 * 1024 * 1024
_DEFAULT_MAX_LINE_BYTES = 4 * 1024 * 1024

# BREAKING: router batch APIs are now used by default when available. Pass
# `batch_size=1` to force single-sample scoring.
_DEFAULT_BATCH_SIZE = 32

_INVALID_PREDICTED_LABEL = "__invalid_prediction__"
_UTF8_BOM = b"\xef\xbb\xbf"

_JSON_DECODER = msgspec.json.Decoder() if msgspec is not None else None


@dataclass(frozen=True, slots=True)
class LabeledUserIntentSample:
    """Store one labeled transcript for user-intent training or evaluation."""

    text: str
    label: str
    sample_id: str | None = None
    split: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", str(self.text or "").strip())
        object.__setattr__(self, "label", normalize_user_intent_label(self.label))
        object.__setattr__(self, "sample_id", str(self.sample_id or "").strip() or None)
        object.__setattr__(self, "split", str(self.split or "").strip().lower() or None)
        object.__setattr__(self, "source", str(self.source or "").strip() or None)
        if not self.text:
            raise ValueError("LabeledUserIntentSample.text must not be empty.")


@dataclass(frozen=True, slots=True)
class ScoredUserIntentRecord:
    """Store one gold user intent together with one scored router decision."""

    sample: LabeledUserIntentSample
    decision: UserIntentDecision
    predicted_label: str | None = None
    raw_predicted_label: str | None = None
    latency_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class UserIntentRouterEvaluation:
    """Summarize user-intent quality over one labeled evaluation slice."""

    total: int
    accuracy: float
    macro_f1: float
    confusion_matrix: dict[str, dict[str, int]]
    per_label: dict[str, dict[str, float]]
    weighted_f1: float = 0.0
    balanced_accuracy: float = 0.0
    coverage: float = 0.0
    invalid_prediction_total: int = 0
    labels_evaluated: tuple[str, ...] = ()
    latency_avg_ms: float | None = None
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    throughput_samples_per_second: float | None = None
    probability_coverage: float = 0.0
    brier_score: float | None = None
    log_loss: float | None = None


def iter_labeled_user_intent_samples(
    path: str | Path,
    *,
    max_samples: int | None = None,
    max_file_size_bytes: int | None = _DEFAULT_MAX_FILE_SIZE_BYTES,
    max_decompressed_bytes: int | None = _DEFAULT_MAX_DECOMPRESSED_BYTES,
    max_line_bytes: int | None = _DEFAULT_MAX_LINE_BYTES,
    reject_special_files: bool = True,
) -> Iterator[LabeledUserIntentSample]:
    """Stream labeled user-intent samples from a JSONL or JSONL.GZ file."""

    file_path = Path(path)
    emitted = 0
    decompressed_bytes = 0

    with _open_jsonl_binary_stream(
        file_path,
        max_file_size_bytes=max_file_size_bytes,
        reject_special_files=reject_special_files,
    ) as stream:
        line_number = 0
        while True:
            raw_line = (
                stream.readline(max_line_bytes + 1)
                if max_line_bytes is not None
                else stream.readline()
            )
            if not raw_line:
                return
            line_number += 1

            if max_line_bytes is not None and len(raw_line) > max_line_bytes:
                raise ValueError(
                    f"Refusing to read JSONL line longer than {max_line_bytes} bytes at {file_path}:{line_number}"
                )

            if line_number == 1 and raw_line.startswith(_UTF8_BOM):
                raw_line = raw_line[len(_UTF8_BOM) :]
            if not raw_line.strip():
                continue

            decompressed_bytes += len(raw_line)
            if max_decompressed_bytes is not None and decompressed_bytes > max_decompressed_bytes:
                raise ValueError(
                    f"Refusing to read dataset larger than {max_decompressed_bytes} decompressed bytes: {file_path}"
                )

            payload = _decode_jsonl_payload(raw_line, file_path=file_path, line_number=line_number)
            yield _payload_to_sample(payload, file_path=file_path, line_number=line_number)

            emitted += 1
            if max_samples is not None and emitted >= max_samples:
                return


def load_labeled_user_intent_samples(
    path: str | Path,
    *,
    max_samples: int | None = None,
    max_file_size_bytes: int | None = _DEFAULT_MAX_FILE_SIZE_BYTES,
    max_decompressed_bytes: int | None = _DEFAULT_MAX_DECOMPRESSED_BYTES,
    max_line_bytes: int | None = _DEFAULT_MAX_LINE_BYTES,
    reject_special_files: bool = True,
) -> tuple[LabeledUserIntentSample, ...]:
    """Load labeled user-intent samples from one JSONL or JSONL.GZ file."""

    return tuple(
        iter_labeled_user_intent_samples(
            path,
            max_samples=max_samples,
            max_file_size_bytes=max_file_size_bytes,
            max_decompressed_bytes=max_decompressed_bytes,
            max_line_bytes=max_line_bytes,
            reject_special_files=reject_special_files,
        )
    )


def split_labeled_user_intent_samples(
    samples: Sequence[LabeledUserIntentSample],
) -> dict[str, tuple[LabeledUserIntentSample, ...]]:
    """Group user-intent samples by their explicit `split` field."""

    grouped: dict[str, list[LabeledUserIntentSample]] = {}
    for sample in samples:
        key = sample.split or "unspecified"
        grouped.setdefault(key, []).append(sample)
    return {key: tuple(values) for key, values in grouped.items()}


def score_user_intent_router(
    router: Any,
    samples: Sequence[LabeledUserIntentSample],
    *,
    batch_size: int | None = _DEFAULT_BATCH_SIZE,
) -> tuple[ScoredUserIntentRecord, ...]:
    """Run the user-intent router over labeled samples and return scored records."""

    if not samples:
        return ()

    records: list[ScoredUserIntentRecord] = []
    effective_batch_size = max(1, int(batch_size or 1))
    batch_classifier = _resolve_batch_classifier(router) if effective_batch_size > 1 else None

    if batch_classifier is None:
        classify = getattr(router, "classify", None)
        if classify is None or not callable(classify):
            raise TypeError("router must expose a callable `classify(text)` method.")
        for sample in samples:
            start = time.perf_counter()
            decision = classify(sample.text)
            latency_ms = (time.perf_counter() - start) * 1000.0
            raw_label, predicted_label = _extract_predicted_label(decision)
            records.append(
                ScoredUserIntentRecord(
                    sample=sample,
                    decision=decision,
                    predicted_label=predicted_label,
                    raw_predicted_label=raw_label,
                    latency_ms=latency_ms,
                )
            )
        return tuple(records)

    for batch in _batched(samples, effective_batch_size):
        texts = [sample.text for sample in batch]
        start = time.perf_counter()
        decisions = tuple(batch_classifier(texts))
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if len(decisions) != len(batch):
            raise ValueError(
                "router batch classifier returned a different number of decisions than inputs: "
                f"{len(decisions)} != {len(batch)}"
            )

        per_item_latency_ms = elapsed_ms / max(len(batch), 1)
        for sample, decision in zip(batch, decisions):
            raw_label, predicted_label = _extract_predicted_label(decision)
            records.append(
                ScoredUserIntentRecord(
                    sample=sample,
                    decision=decision,
                    predicted_label=predicted_label,
                    raw_predicted_label=raw_label,
                    latency_ms=per_item_latency_ms,
                )
            )
    return tuple(records)


def evaluate_user_intent_records(
    records: Sequence[ScoredUserIntentRecord],
    *,
    labels: Sequence[str] | None = None,
) -> UserIntentRouterEvaluation:
    """Compute classification metrics for scored user-intent records."""

    total = len(records)
    if labels is None:
        gold_labels_observed = {record.sample.label for record in records}
        predicted_labels_observed = {
            predicted_label
            for predicted_label in (_record_predicted_label(record) for record in records)
            if predicted_label != _INVALID_PREDICTED_LABEL
        }
        labels_evaluated = tuple(
            label
            for label in USER_INTENT_LABEL_VALUES
            if label in gold_labels_observed or label in predicted_labels_observed
        )
        if not labels_evaluated and total > 0:
            labels_evaluated = tuple(
                label for label in USER_INTENT_LABEL_VALUES if label in gold_labels_observed
            )
    else:
        labels_evaluated = tuple(dict.fromkeys(normalize_user_intent_label(label) for label in labels))
        gold_outside = sorted({record.sample.label for record in records if record.sample.label not in labels_evaluated})
        if gold_outside:
            raise ValueError(
                "Provided `labels` do not cover all gold labels in `records`: "
                + ", ".join(gold_outside)
            )

    predicted_columns = list(labels_evaluated)
    invalid_prediction_total = 0
    for record in records:
        if _evaluation_predicted_label(record, labels_evaluated) == _INVALID_PREDICTED_LABEL:
            invalid_prediction_total += 1
    if invalid_prediction_total:
        predicted_columns.append(_INVALID_PREDICTED_LABEL)

    confusion_matrix = {
        gold: {predicted: 0 for predicted in predicted_columns}
        for gold in labels_evaluated
    }

    correct = 0
    probability_vectors: list[tuple[str, dict[str, float]]] = []
    latency_values_ms: list[float] = []
    for record in records:
        gold = record.sample.label
        predicted = _evaluation_predicted_label(record, labels_evaluated)
        if gold in confusion_matrix:
            confusion_matrix[gold][predicted] = confusion_matrix[gold].get(predicted, 0) + 1
        if predicted == gold:
            correct += 1
        if record.latency_ms > 0:
            latency_values_ms.append(float(record.latency_ms))
        probability_vector = _extract_probability_vector(record.decision)
        if probability_vector is not None:
            probability_vectors.append((gold, probability_vector))

    per_label: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    weighted_f1_numerator = 0.0
    recall_values_for_balanced_accuracy: list[float] = []

    for label in labels_evaluated:
        row = confusion_matrix[label]
        support = float(sum(row.values()))
        true_positive = float(row.get(label, 0))
        false_positive = float(
            sum(confusion_matrix[gold].get(label, 0) for gold in labels_evaluated if gold != label)
        )
        false_negative = float(
            sum(count for predicted_label, count in row.items() if predicted_label != label)
        )
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1_values.append(f1)
        weighted_f1_numerator += f1 * support
        if support > 0:
            recall_values_for_balanced_accuracy.append(recall)

    total_support = float(sum(metrics["support"] for metrics in per_label.values()))
    latency_avg_ms = _safe_ratio(sum(latency_values_ms), len(latency_values_ms)) if latency_values_ms else None
    latency_p50_ms = _percentile(latency_values_ms, 50.0) if latency_values_ms else None
    latency_p95_ms = _percentile(latency_values_ms, 95.0) if latency_values_ms else None
    throughput_samples_per_second = (
        _safe_ratio(len(latency_values_ms), sum(latency_values_ms) / 1000.0) if latency_values_ms else None
    )

    probability_coverage = _safe_ratio(len(probability_vectors), total)
    brier_score = _multiclass_brier_score(probability_vectors) if probability_vectors else None
    log_loss = _multiclass_log_loss(probability_vectors) if probability_vectors else None

    return UserIntentRouterEvaluation(
        total=total,
        accuracy=_safe_ratio(correct, total),
        macro_f1=_safe_ratio(sum(f1_values), len(f1_values)),
        confusion_matrix=confusion_matrix,
        per_label=per_label,
        weighted_f1=_safe_ratio(weighted_f1_numerator, total_support),
        balanced_accuracy=_safe_ratio(
            sum(recall_values_for_balanced_accuracy), len(recall_values_for_balanced_accuracy)
        ),
        coverage=_safe_ratio(total - invalid_prediction_total, total),
        invalid_prediction_total=invalid_prediction_total,
        labels_evaluated=labels_evaluated,
        latency_avg_ms=latency_avg_ms,
        latency_p50_ms=latency_p50_ms,
        latency_p95_ms=latency_p95_ms,
        throughput_samples_per_second=throughput_samples_per_second,
        probability_coverage=probability_coverage,
        brier_score=brier_score,
        log_loss=log_loss,
    )


def _decode_jsonl_payload(raw_line: bytes, *, file_path: Path, line_number: int) -> Mapping[str, object]:
    try:
        if _JSON_DECODER is not None:
            payload = _JSON_DECODER.decode(raw_line)
        else:
            payload = json.loads(raw_line)
    except Exception as exc:
        raise ValueError(f"Invalid JSONL at {file_path}:{line_number}: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise ValueError(
            f"Invalid JSONL object at {file_path}:{line_number}: expected a JSON object, got {type(payload).__name__}."
        )
    return payload


def _payload_to_sample(
    payload: Mapping[str, object],
    *,
    file_path: Path,
    line_number: int,
) -> LabeledUserIntentSample:
    try:
        return LabeledUserIntentSample(
            text=payload["text"],
            label=payload["label"],
            sample_id=payload.get("id"),
            split=payload.get("split"),
            source=payload.get("source"),
        )
    except KeyError as exc:
        missing_key = str(exc).strip("'")
        raise ValueError(f"Missing required key `{missing_key}` at {file_path}:{line_number}.") from exc
    except Exception as exc:
        raise ValueError(f"Invalid labeled sample at {file_path}:{line_number}: {exc}") from exc


@contextmanager
def _open_jsonl_binary_stream(
    file_path: Path,
    *,
    max_file_size_bytes: int | None,
    reject_special_files: bool,
) -> Iterator[BinaryIO]:
    raw_stream = _open_safe_binary_file(
        file_path,
        max_file_size_bytes=max_file_size_bytes,
        reject_special_files=reject_special_files,
    )
    try:
        if file_path.suffix.lower() == ".gz":
            gzip_stream = gzip.GzipFile(fileobj=raw_stream, mode="rb")
            try:
                yield gzip_stream
            finally:
                gzip_stream.close()
        else:
            yield raw_stream
    finally:
        raw_stream.close()


def _open_safe_binary_file(
    file_path: Path,
    *,
    max_file_size_bytes: int | None,
    reject_special_files: bool,
) -> BinaryIO:
    path_str = os.fspath(file_path)
    try:
        initial_stat = os.stat(path_str, follow_symlinks=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file does not exist: {file_path}") from None

    if reject_special_files and not stat.S_ISREG(initial_stat.st_mode):
        raise ValueError(f"Dataset path must be a regular file: {file_path}")

    if max_file_size_bytes is not None and initial_stat.st_size > max_file_size_bytes:
        raise ValueError(
            f"Refusing to read dataset larger than {max_file_size_bytes} bytes: {file_path}"
        )

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if reject_special_files and hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY

    try:
        fd = os.open(path_str, flags)
    except OSError as exc:
        raise OSError(f"Unable to open dataset file {file_path}: {exc}") from exc

    stream = os.fdopen(fd, "rb", closefd=True)
    if not reject_special_files:
        return stream

    opened_stat = os.fstat(stream.fileno())
    if not stat.S_ISREG(opened_stat.st_mode):
        stream.close()
        raise ValueError(f"Dataset path must resolve to a regular file: {file_path}")
    if (opened_stat.st_dev, opened_stat.st_ino) != (initial_stat.st_dev, initial_stat.st_ino):
        stream.close()
        raise OSError(f"Dataset file changed while opening: {file_path}")
    return stream


def _resolve_batch_classifier(router: Any) -> Callable[[Sequence[str]], Iterable[UserIntentDecision]] | None:
    for method_name in ("classify_many", "classify_batch", "batch_classify"):
        method = getattr(router, method_name, None)
        if callable(method):
            return method
    return None


def _batched(values: Sequence[LabeledUserIntentSample], batch_size: int) -> Iterator[Sequence[LabeledUserIntentSample]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _extract_predicted_label(decision: Any) -> tuple[str | None, str | None]:
    raw_label = getattr(decision, "label", None)
    if raw_label is None:
        return None, None
    try:
        normalized_label = normalize_user_intent_label(raw_label)
    except Exception:
        return str(raw_label), None
    return str(raw_label), normalized_label


def _evaluation_predicted_label(record: ScoredUserIntentRecord, labels_evaluated: Sequence[str]) -> str:
    predicted_label = _record_predicted_label(record)
    if predicted_label == _INVALID_PREDICTED_LABEL:
        return predicted_label
    if labels_evaluated and predicted_label not in labels_evaluated:
        return _INVALID_PREDICTED_LABEL
    return predicted_label


def _record_predicted_label(record: ScoredUserIntentRecord) -> str:
    predicted_label = record.predicted_label
    if predicted_label:
        return predicted_label
    _, normalized_label = _extract_predicted_label(record.decision)
    if normalized_label:
        return normalized_label
    return _INVALID_PREDICTED_LABEL


def _extract_probability_vector(decision: Any) -> dict[str, float] | None:
    for attribute_name in (
        "probabilities",
        "label_probabilities",
        "probability_by_label",
        "scores",
        "label_scores",
        "scores_by_label",
    ):
        candidate = getattr(decision, attribute_name, None)
        if not isinstance(candidate, Mapping):
            continue

        probabilities: dict[str, float] = {}
        for raw_label, raw_value in candidate.items():
            try:
                normalized_label = normalize_user_intent_label(raw_label)
            except Exception:
                continue
            if not isinstance(raw_value, (int, float)) or not math.isfinite(float(raw_value)):
                return None
            value = float(raw_value)
            if value < 0.0 or value > 1.0:
                return None
            probabilities[normalized_label] = value

        if not probabilities:
            continue

        total_probability = sum(probabilities.values())
        if not math.isclose(total_probability, 1.0, rel_tol=1e-3, abs_tol=1e-3):
            return None

        return {label: probabilities.get(label, 0.0) for label in USER_INTENT_LABEL_VALUES}
    return None


def _multiclass_brier_score(probability_vectors: Sequence[tuple[str, Mapping[str, float]]]) -> float:
    if not probability_vectors:
        return 0.0
    total_loss = 0.0
    for gold_label, probabilities in probability_vectors:
        row_loss = 0.0
        for label in USER_INTENT_LABEL_VALUES:
            target = 1.0 if label == gold_label else 0.0
            predicted_probability = float(probabilities.get(label, 0.0))
            row_loss += (target - predicted_probability) ** 2
        total_loss += row_loss
    return total_loss / len(probability_vectors)


def _multiclass_log_loss(probability_vectors: Sequence[tuple[str, Mapping[str, float]]]) -> float:
    if not probability_vectors:
        return 0.0
    epsilon = 1e-15
    total_loss = 0.0
    for gold_label, probabilities in probability_vectors:
        gold_probability = float(probabilities.get(gold_label, 0.0))
        bounded_probability = min(max(gold_probability, epsilon), 1.0 - epsilon)
        total_loss += -math.log(bounded_probability)
    return total_loss / len(probability_vectors)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("_percentile requires at least one value.")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    clamped = min(max(float(percentile), 0.0), 100.0)
    rank = (len(ordered) - 1) * (clamped / 100.0)
    lower_index = int(math.floor(rank))
    upper_index = int(math.ceil(rank))
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = rank - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)