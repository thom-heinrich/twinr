"""Evaluate, label, and train wakeword assets from current policy captures.

This module turns manifest or ops-labeled recordings into deterministic
evaluation metrics, writes the latest eval report, and searches for candidate
calibration overrides that improve wakeword precision and recall. It also
trains optional openWakeWord custom verifier assets from labeled room-capture
manifests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import wave

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.audio import AmbientAudioCaptureWindow, AmbientAudioLevelSample
from twinr.ops import TwinrOpsEventStore, resolve_ops_paths_for_config

from .calibration import WakewordCalibrationProfile, WakewordCalibrationStore, apply_wakeword_calibration
from .cascade import WakewordSequenceCaptureVerifier
from .kws import WakewordSherpaOnnxSpotter
from .policy import SttWakewordVerifier, WakewordDecisionPolicy, normalize_wakeword_backend
from .spotter import WakewordOpenWakeWordSpotter

_POSITIVE_LABELS = {"correct", "false_negative", "positive"}
_NEGATIVE_LABELS = {
    "false_positive",
    "background_tv",
    "cross_talk",
    "far_field",
    "noise",
    "negative",
}
_IGNORED_LABELS = {"unclear"}
_MANIFEST_AUDIO_PATH_KEYS = ("captured_audio_path", "audio_path")


def _normalize_label(value: object | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace(" ", "_")
    return normalized or None


def _expected_detected(label: str | None) -> bool | None:
    normalized = _normalize_label(label)
    if normalized in _POSITIVE_LABELS:
        return True
    if normalized in _NEGATIVE_LABELS:
        return False
    return None


def _pcm16_rms(samples: bytes) -> int:
    if not samples:
        return 0
    usable = len(samples) - (len(samples) % 2)
    if usable <= 0:
        return 0
    from array import array

    pcm = array("h")
    pcm.frombytes(samples[:usable])
    mean_square = sum(sample * sample for sample in pcm) / len(pcm)
    return int(math.sqrt(mean_square))


def _capture_from_wav(path: Path) -> AmbientAudioCaptureWindow:
    with wave.open(str(path), "rb") as wav_file:
        channels = int(wav_file.getnchannels())
        sample_rate = int(wav_file.getframerate())
        sample_width = int(wav_file.getsampwidth())
        if sample_width != 2:
            raise ValueError(f"{path} must be 16-bit PCM WAV.")
        frame_count = int(wav_file.getnframes())
        pcm_bytes = wav_file.readframes(frame_count)
    duration_ms = int((frame_count / max(1, sample_rate)) * 1000)
    rms = _pcm16_rms(pcm_bytes)
    sample = AmbientAudioLevelSample(
        duration_ms=duration_ms,
        chunk_count=1,
        active_chunk_count=1 if rms > 0 else 0,
        average_rms=rms,
        peak_rms=rms,
        active_ratio=1.0 if rms > 0 else 0.0,
    )
    return AmbientAudioCaptureWindow(
        sample=sample,
        pcm_bytes=pcm_bytes,
        sample_rate=sample_rate,
        channels=channels,
    )


@dataclass(frozen=True, slots=True)
class WakewordEvalEntry:
    """Describe one labeled audio clip used for wakeword evaluation."""

    audio_path: Path
    label: str
    source: str = "manifest"
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class WakewordEvalMetrics:
    """Report aggregate wakeword evaluation counts and ratios."""

    total: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    precision: float
    recall: float
    false_positive_rate: float
    false_negative_rate: float


@dataclass(frozen=True, slots=True)
class WakewordEvalReport:
    """Describe one evaluation run and its persisted report path."""

    metrics: WakewordEvalMetrics
    evaluated_entries: int
    report_path: Path | None = None


@dataclass(frozen=True, slots=True)
class WakewordAutotuneRecommendation:
    """Describe the best calibration profile found during autotune."""

    profile: WakewordCalibrationProfile
    metrics: WakewordEvalMetrics
    score: float
    profile_path: Path | None = None


@dataclass(frozen=True, slots=True)
class WakewordVerifierTrainingReport:
    """Describe one custom-verifier training run from labeled captures."""

    manifest_path: Path
    output_path: Path
    model_name: str
    positive_clips: int
    negative_clips: int
    negative_seconds: float


def _load_manifest_payloads(manifest_path: str | Path) -> tuple[Path, list[dict[str, object]]]:
    """Load one wakeword manifest as either JSONL or one JSON array."""

    path = Path(manifest_path).expanduser().resolve(strict=False)
    if not path.exists():
        raise FileNotFoundError(path)
    raw_text = path.read_text(encoding="utf-8")
    stripped = raw_text.strip()
    if not stripped:
        return path, []
    if stripped.startswith("["):
        payload = json.loads(stripped)
        if not isinstance(payload, list):
            raise ValueError("Wakeword eval manifest must be a JSON array or JSONL file.")
        raw_entries = payload
    else:
        raw_entries = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
    entries: list[dict[str, object]] = []
    for item in raw_entries:
        if not isinstance(item, dict):
            raise ValueError("Wakeword eval manifest items must be JSON objects.")
        entries.append(item)
    return path, entries


def _resolve_manifest_audio_path(payload: dict[str, object], *, manifest_path: Path) -> Path:
    """Resolve the best available audio path for one manifest item."""

    for key in _MANIFEST_AUDIO_PATH_KEYS:
        raw_value = str(payload.get(key) or "").strip()
        if not raw_value:
            continue
        audio_path = Path(raw_value).expanduser()
        if not audio_path.is_absolute():
            audio_path = (manifest_path.parent / audio_path).resolve(strict=False)
        return audio_path
    raise ValueError("Wakeword eval manifest items require audio_path or captured_audio_path.")


def _wav_duration_seconds(path: Path) -> float:
    """Return the duration of one WAV file in seconds."""

    with wave.open(str(path), "rb") as wav_file:
        frame_count = int(wav_file.getnframes())
        sample_rate = int(wav_file.getframerate())
    return frame_count / max(1, sample_rate)


def _unique_reference_clip_paths(paths: list[Path]) -> list[str]:
    """Return stable deduplicated clip paths for verifier training."""

    unique_paths = dict.fromkeys(path.expanduser().resolve(strict=False) for path in paths)
    return [str(path) for path in unique_paths]


def _primary_backend_for_config(config: TwinrConfig) -> str:
    return normalize_wakeword_backend(
        getattr(config, "wakeword_primary_backend", config.wakeword_backend),
        default="openwakeword",
    )


def _primary_threshold_for_config(config: TwinrConfig) -> float | None:
    primary_backend = _primary_backend_for_config(config)
    if primary_backend == "kws":
        return float(config.wakeword_kws_keywords_threshold)
    if primary_backend == "openwakeword":
        return float(config.wakeword_openwakeword_threshold)
    return None


def _build_local_verifier(config: TwinrConfig):
    if _primary_backend_for_config(config) != "openwakeword":
        return None
    if not config.wakeword_openwakeword_sequence_verifier_models:
        return None
    return WakewordSequenceCaptureVerifier(
        verifier_models=dict(config.wakeword_openwakeword_sequence_verifier_models),
        threshold=config.wakeword_openwakeword_sequence_verifier_threshold,
    )


def _build_eval_spotter(
    *,
    config: TwinrConfig,
    model_factory=None,
):
    primary_backend = _primary_backend_for_config(config)
    if primary_backend not in {"openwakeword", "kws"}:
        raise ValueError(
            "Wakeword eval currently supports only local detector backends: openwakeword or kws."
        )
    if primary_backend == "kws":
        return WakewordSherpaOnnxSpotter(
            tokens_path=config.wakeword_kws_tokens_path or "",
            encoder_path=config.wakeword_kws_encoder_path or "",
            decoder_path=config.wakeword_kws_decoder_path or "",
            joiner_path=config.wakeword_kws_joiner_path or "",
            keywords_file_path=config.wakeword_kws_keywords_file_path or "",
            phrases=config.wakeword_phrases,
            project_root=config.project_root,
            sample_rate=config.wakeword_kws_sample_rate,
            feature_dim=config.wakeword_kws_feature_dim,
            max_active_paths=config.wakeword_kws_max_active_paths,
            keywords_score=config.wakeword_kws_keywords_score,
            keywords_threshold=config.wakeword_kws_keywords_threshold,
            num_trailing_blanks=config.wakeword_kws_num_trailing_blanks,
            num_threads=config.wakeword_kws_num_threads,
            provider=config.wakeword_kws_provider,
            keyword_spotter_factory=model_factory,
        )
    return WakewordOpenWakeWordSpotter(
        wakeword_models=config.wakeword_openwakeword_models,
        phrases=config.wakeword_phrases,
        threshold=config.wakeword_openwakeword_threshold,
        vad_threshold=config.wakeword_openwakeword_vad_threshold,
        patience_frames=config.wakeword_openwakeword_patience_frames,
        activation_samples=config.wakeword_openwakeword_activation_samples,
        deactivation_threshold=config.wakeword_openwakeword_deactivation_threshold,
        enable_speex_noise_suppression=config.wakeword_openwakeword_enable_speex,
        inference_framework=config.wakeword_openwakeword_inference_framework,
        backend=None,
        transcribe_on_detect=False,
        model_factory=model_factory,
    )


def load_eval_manifest(manifest_path: str | Path) -> list[WakewordEvalEntry]:
    """Load labeled wakeword clips from a JSONL manifest."""

    path, raw_entries = _load_manifest_payloads(manifest_path)
    entries: list[WakewordEvalEntry] = []
    for payload in raw_entries:
        audio_path = _resolve_manifest_audio_path(payload, manifest_path=path)
        label = _normalize_label(payload.get("label"))
        expected = _expected_detected(label)
        if expected is None:
            continue
        entries.append(
            WakewordEvalEntry(
                audio_path=audio_path,
                label=label or "unclear",
                source=str(payload.get("source") or "manifest"),
                notes=str(payload.get("notes")).strip() if payload.get("notes") is not None else None,
            )
        )
    return entries


def train_wakeword_custom_verifier_from_manifest(
    *,
    manifest_path: str | Path,
    output_path: str | Path,
    model_name: str,
    inference_framework: str = "onnx",
) -> WakewordVerifierTrainingReport:
    """Train one local openWakeWord verifier from labeled room captures.

    This follows the official openWakeWord verifier guidance: at least three
    positive wakeword clips plus roughly ten seconds of negative speech or
    prior false activations collected close to deployment.
    """

    manifest, raw_entries = _load_manifest_payloads(manifest_path)
    positive_paths: list[Path] = []
    negative_paths: list[Path] = []
    negative_seconds = 0.0
    for payload in raw_entries:
        label = _normalize_label(payload.get("label"))
        expected = _expected_detected(label)
        if expected is None:
            continue
        audio_path = _resolve_manifest_audio_path(payload, manifest_path=manifest)
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        if expected:
            positive_paths.append(audio_path)
            continue
        negative_paths.append(audio_path)
        negative_seconds += _wav_duration_seconds(audio_path)
    positive_clips = _unique_reference_clip_paths(positive_paths)
    negative_clips = _unique_reference_clip_paths(negative_paths)
    if len(positive_clips) < 3:
        raise ValueError(
            "Custom wakeword verifier training requires at least 3 positive clips from the target deployment."
        )
    if not negative_clips or negative_seconds < 10.0:
        raise ValueError(
            "Custom wakeword verifier training requires at least about 10 seconds of negative speech/background audio."
        )
    resolved_output = Path(output_path).expanduser().resolve(strict=False)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_model_name = str(model_name or "").strip()
    candidate_model_path = Path(resolved_model_name).expanduser()
    if resolved_model_name and (candidate_model_path.is_absolute() or candidate_model_path.exists()):
        resolved_model_name = str(candidate_model_path.resolve(strict=False))

    import openwakeword

    openwakeword.train_custom_verifier(
        positive_reference_clips=positive_clips,
        negative_reference_clips=negative_clips,
        output_path=str(resolved_output),
        model_name=resolved_model_name,
        inference_framework=str(inference_framework or "onnx").strip().lower() or "onnx",
    )
    return WakewordVerifierTrainingReport(
        manifest_path=manifest,
        output_path=resolved_output,
        model_name=resolved_model_name,
        positive_clips=len(positive_clips),
        negative_clips=len(negative_clips),
        negative_seconds=negative_seconds,
    )


def append_wakeword_capture_label(
    config: TwinrConfig,
    *,
    capture_path: str | Path,
    label: str,
    notes: str | None = None,
) -> dict[str, object]:
    """Append or update an operator label for one stored wakeword capture."""

    normalized_label = _normalize_label(label)
    if _expected_detected(normalized_label) is None:
        raise ValueError(f"Unsupported wakeword label: {label}")
    store = TwinrOpsEventStore.from_config(config)
    return store.append(
        event="wakeword_capture_labeled",
        message="Updated the operator label for a stored wakeword capture.",
        data={
            "capture_path": str(Path(capture_path).expanduser()),
            "label": normalized_label,
            "notes": notes or "",
        },
    )


def load_labeled_ops_captures(config: TwinrConfig) -> list[WakewordEvalEntry]:
    """Load labeled wakeword captures from Twinr ops events."""

    events_path = resolve_ops_paths_for_config(config).events_path
    if not events_path.exists():
        return []
    labels_by_path: dict[str, tuple[str, str | None]] = {}
    decisions_by_path: dict[str, dict[str, object]] = {}
    for raw_line in events_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            continue
        event = str(payload.get("event") or "")
        data = payload.get("data")
        if not isinstance(data, dict):
            continue
        capture_path = str(data.get("capture_path") or "").strip()
        if not capture_path:
            continue
        if event == "wakeword_capture_labeled":
            label = _normalize_label(data.get("label"))
            if label is not None:
                labels_by_path[capture_path] = (label, str(data.get("notes") or "").strip() or None)
        elif event == "wakeword_decision":
            decisions_by_path[capture_path] = data
    entries: list[WakewordEvalEntry] = []
    for capture_path, (label, notes) in labels_by_path.items():
        if _expected_detected(label) is None:
            continue
        entries.append(
            WakewordEvalEntry(
                audio_path=Path(capture_path).expanduser(),
                label=label,
                source=str(decisions_by_path.get(capture_path, {}).get("source") or "ops"),
                notes=notes,
            )
        )
    return entries


def _metrics_from_counts(
    *,
    true_positive: int,
    false_positive: int,
    true_negative: int,
    false_negative: int,
) -> WakewordEvalMetrics:
    total = true_positive + false_positive + true_negative + false_negative
    precision = true_positive / max(1, (true_positive + false_positive))
    recall = true_positive / max(1, (true_positive + false_negative))
    false_positive_rate = false_positive / max(1, (false_positive + true_negative))
    false_negative_rate = false_negative / max(1, (false_negative + true_positive))
    return WakewordEvalMetrics(
        total=total,
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        false_positive_rate=false_positive_rate,
        false_negative_rate=false_negative_rate,
    )


def evaluate_wakeword_entries(
    *,
    config: TwinrConfig,
    entries: list[WakewordEvalEntry],
    backend=None,
    model_factory=None,
) -> WakewordEvalReport:
    """Evaluate labeled wakeword clips against the configured policy."""

    verifier = (
        None
        if backend is None or config.wakeword_verifier_mode == "disabled"
        else SttWakewordVerifier(
            backend=backend,
            phrases=config.wakeword_phrases,
            language=config.openai_realtime_language,
        )
    )
    local_verifier = _build_local_verifier(config)
    policy = WakewordDecisionPolicy(
        primary_backend=config.wakeword_primary_backend,
        fallback_backend=config.wakeword_fallback_backend,
        verifier_mode=config.wakeword_verifier_mode,
        verifier_margin=config.wakeword_verifier_margin,
        primary_threshold=_primary_threshold_for_config(config),
        verifier=verifier,
        local_verifier=local_verifier,
    )
    spotter = _build_eval_spotter(
        config=config,
        model_factory=model_factory,
    )
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    evaluated_entries = 0
    for entry in entries:
        expected = _expected_detected(entry.label)
        if expected is None:
            continue
        capture = _capture_from_wav(entry.audio_path)
        match = spotter.detect(capture)
        decision = policy.decide(match=match, capture=capture, source=entry.source)
        evaluated_entries += 1
        if expected and decision.detected:
            counts["tp"] += 1
        elif expected:
            counts["fn"] += 1
        elif decision.detected:
            counts["fp"] += 1
        else:
            counts["tn"] += 1
    return WakewordEvalReport(
        metrics=_metrics_from_counts(
            true_positive=counts["tp"],
            false_positive=counts["fp"],
            true_negative=counts["tn"],
            false_negative=counts["fn"],
        ),
        evaluated_entries=evaluated_entries,
    )


def run_wakeword_eval(
    *,
    config: TwinrConfig,
    manifest_path: str | Path | None = None,
    backend=None,
    model_factory=None,
) -> WakewordEvalReport:
    """Run wakeword evaluation and persist the latest JSON report."""

    entries = (
        load_eval_manifest(manifest_path)
        if manifest_path is not None
        else load_labeled_ops_captures(config)
    )
    report = evaluate_wakeword_entries(
        config=config,
        entries=entries,
        backend=backend,
        model_factory=model_factory,
    )
    reports_root = resolve_ops_paths_for_config(config).artifacts_root / "ops" / "wakeword_eval"
    reports_root.mkdir(parents=True, exist_ok=True)
    report_path = reports_root / "latest_eval.json"
    report_path.write_text(
        json.dumps(
            {
                "evaluated_entries": report.evaluated_entries,
                "metrics": {
                    "total": report.metrics.total,
                    "true_positive": report.metrics.true_positive,
                    "false_positive": report.metrics.false_positive,
                    "true_negative": report.metrics.true_negative,
                    "false_negative": report.metrics.false_negative,
                    "precision": round(report.metrics.precision, 4),
                    "recall": round(report.metrics.recall, 4),
                    "false_positive_rate": round(report.metrics.false_positive_rate, 4),
                    "false_negative_rate": round(report.metrics.false_negative_rate, 4),
                },
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return WakewordEvalReport(
        metrics=report.metrics,
        evaluated_entries=report.evaluated_entries,
        report_path=report_path,
    )


def autotune_wakeword_profile(
    *,
    config: TwinrConfig,
    manifest_path: str | Path | None = None,
    backend=None,
    model_factory=None,
) -> WakewordAutotuneRecommendation:
    """Search calibration candidates and persist the best recommendation."""

    entries = (
        load_eval_manifest(manifest_path)
        if manifest_path is not None
        else load_labeled_ops_captures(config)
    )
    threshold_candidates = (0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.5)
    patience_candidates = tuple(sorted({1, config.wakeword_openwakeword_patience_frames, config.wakeword_openwakeword_patience_frames + 1}))
    activation_candidates = tuple(sorted({1, config.wakeword_openwakeword_activation_samples, config.wakeword_openwakeword_activation_samples + 1}))
    best_profile = WakewordCalibrationProfile(
        primary_backend=config.wakeword_primary_backend,
        fallback_backend=config.wakeword_fallback_backend,
        verifier_mode=config.wakeword_verifier_mode,
        verifier_margin=config.wakeword_verifier_margin,
        threshold=config.wakeword_openwakeword_threshold,
        vad_threshold=config.wakeword_openwakeword_vad_threshold,
        patience_frames=config.wakeword_openwakeword_patience_frames,
        activation_samples=config.wakeword_openwakeword_activation_samples,
        deactivation_threshold=config.wakeword_openwakeword_deactivation_threshold,
        notes="autotune baseline",
    )
    best_report = evaluate_wakeword_entries(config=config, entries=entries, backend=backend, model_factory=model_factory)
    best_score = (best_report.metrics.precision * 0.6) + (best_report.metrics.recall * 0.4)
    best_score -= best_report.metrics.false_positive_rate
    for threshold in threshold_candidates:
        for patience in patience_candidates:
            for activation in activation_candidates:
                candidate_profile = WakewordCalibrationProfile(
                    primary_backend=config.wakeword_primary_backend,
                    fallback_backend=config.wakeword_fallback_backend,
                    verifier_mode=config.wakeword_verifier_mode,
                    verifier_margin=config.wakeword_verifier_margin,
                    threshold=threshold,
                    vad_threshold=config.wakeword_openwakeword_vad_threshold,
                    patience_frames=patience,
                    activation_samples=activation,
                    deactivation_threshold=config.wakeword_openwakeword_deactivation_threshold,
                    notes="autotune recommendation",
                )
                candidate_config = apply_wakeword_calibration(config, candidate_profile)
                report = evaluate_wakeword_entries(
                    config=candidate_config,
                    entries=entries,
                    backend=backend,
                    model_factory=model_factory,
                )
                score = (report.metrics.precision * 0.6) + (report.metrics.recall * 0.4)
                score -= report.metrics.false_positive_rate
                if score > best_score or (
                    score == best_score
                    and report.metrics.false_positive_rate < best_report.metrics.false_positive_rate
                ):
                    best_score = score
                    best_profile = candidate_profile
                    best_report = report
    recommended_store = WakewordCalibrationStore.recommended_from_config(config)
    saved_profile = recommended_store.save(best_profile)
    return WakewordAutotuneRecommendation(
        profile=saved_profile,
        metrics=best_report.metrics,
        score=best_score,
        profile_path=recommended_store.path,
    )


__all__ = [
    "WakewordAutotuneRecommendation",
    "WakewordEvalEntry",
    "WakewordEvalMetrics",
    "WakewordEvalReport",
    "WakewordVerifierTrainingReport",
    "append_wakeword_capture_label",
    "autotune_wakeword_profile",
    "load_eval_manifest",
    "load_labeled_ops_captures",
    "run_wakeword_eval",
    "train_wakeword_custom_verifier_from_manifest",
]
