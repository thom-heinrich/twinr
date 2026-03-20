"""Replay wakeword manifests through Twinr's runtime-faithful stream path.

This module exists to keep wakeword promotion honest. Earlier clip-level
evaluation overstated candidate quality versus the real Pi runtime because it
did not replay audio through the streaming frame detector plus the normal
decision policy. The helpers here use the same streaming spotter shape, build
the same localized capture windows, and aggregate suite plus ambient
false-accept guards into one promotion report.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import wave

import numpy as np
from scipy.signal import resample_poly

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.audio import AmbientAudioCaptureWindow, AmbientAudioLevelSample
from twinr.ops import resolve_ops_paths_for_config

from .cascade import WakewordSequenceCaptureVerifier
from .evaluation import (
    WakewordEvalEntry,
    WakewordEvalMetrics,
    _expected_detected,
    _metrics_from_counts,
    load_eval_manifest,
)
from .policy import SttWakewordVerifier, WakewordDecisionPolicy
from .spotter import WakewordOpenWakeWordFrameSpotter

_STREAM_CAPTURE_WINDOW_MS = 2500
_REPORT_DIRNAME = "wakeword_eval"


@dataclass(frozen=True, slots=True)
class WakewordStreamEvalReport:
    """Describe one runtime-faithful wakeword stream replay result.

    Attributes:
        metrics: Aggregate clip-level counts derived from replayed detections.
        evaluated_entries: Number of labeled clips evaluated from the manifest.
        accepted_detection_count: Number of accepted wakeword activations seen
            across the replay, including repeated activations on long clips.
        total_audio_seconds: Total replayed audio duration in seconds.
        report_path: Optional persisted JSON report path.
    """

    metrics: WakewordEvalMetrics
    evaluated_entries: int
    accepted_detection_count: int
    total_audio_seconds: float
    report_path: Path | None = None


@dataclass(frozen=True, slots=True)
class WakewordPromotionSuiteSpec:
    """Describe one labeled replay suite used for promotion blocking.

    Attributes:
        name: Human-readable suite name written into reports.
        manifest_path: JSONL or JSON-array manifest passed to replay.
        max_false_negatives: Optional maximum allowed false negatives.
        max_false_positives: Optional maximum allowed false positives.
        min_precision: Optional minimum required precision.
        min_recall: Optional minimum required recall.
    """

    name: str
    manifest_path: Path
    max_false_negatives: int | None = None
    max_false_positives: int | None = None
    min_precision: float | None = None
    min_recall: float | None = None


@dataclass(frozen=True, slots=True)
class WakewordAmbientGuardSpec:
    """Describe one long-form ambient negative guard for promotion.

    Attributes:
        name: Human-readable guard name written into reports.
        manifest_path: Manifest containing only negative ambient clips.
        max_false_accepts_per_hour: Maximum allowed accepted activations per
            audio hour across the manifest.
    """

    name: str
    manifest_path: Path
    max_false_accepts_per_hour: float


@dataclass(frozen=True, slots=True)
class WakewordPromotionSpec:
    """Describe the authoritative promotion suites for one candidate."""

    suites: tuple[WakewordPromotionSuiteSpec, ...]
    ambient_guards: tuple[WakewordAmbientGuardSpec, ...] = ()


@dataclass(frozen=True, slots=True)
class WakewordPromotionSuiteResult:
    """Describe the replay result for one labeled promotion suite."""

    spec: WakewordPromotionSuiteSpec
    report: WakewordStreamEvalReport


@dataclass(frozen=True, slots=True)
class WakewordAmbientGuardResult:
    """Describe the replay result for one ambient false-accept guard."""

    spec: WakewordAmbientGuardSpec
    report: WakewordStreamEvalReport
    false_accepts_per_hour: float


@dataclass(frozen=True, slots=True)
class WakewordPromotionReport:
    """Describe one end-to-end wakeword promotion replay."""

    spec_path: Path
    suite_results: tuple[WakewordPromotionSuiteResult, ...]
    ambient_results: tuple[WakewordAmbientGuardResult, ...]
    blockers: tuple[str, ...]
    passed: bool
    report_path: Path | None = None


def _load_pcm16_wav_for_stream_replay(
    path: Path,
    *,
    target_sample_rate: int,
    target_channels: int,
) -> tuple[bytes, float]:
    """Load one WAV file and normalize it for runtime-faithful stream replay."""

    with wave.open(str(path), "rb") as wav_file:
        source_channels = int(wav_file.getnchannels())
        source_rate = int(wav_file.getframerate())
        sample_width = int(wav_file.getsampwidth())
        frame_count = int(wav_file.getnframes())
        pcm_bytes = wav_file.readframes(frame_count)
    if sample_width != 2:
        raise ValueError(f"{path} must be 16-bit PCM WAV.")
    if target_channels != 1:
        raise ValueError("Wakeword stream replay currently requires mono target audio.")
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    if source_channels > 1:
        samples = (
            samples.reshape(-1, source_channels)
            .astype(np.int32)
            .mean(axis=1)
            .round()
            .clip(-32768, 32767)
            .astype(np.int16)
        )
    if source_rate != target_sample_rate:
        samples = resample_poly(samples.astype(np.float32), target_sample_rate, source_rate)
        samples = np.clip(samples, -32768.0, 32767.0).astype(np.int16)
    duration_seconds = float(samples.shape[0]) / max(1, int(target_sample_rate))
    return samples.tobytes(), duration_seconds


def _pcm16_rms(samples: bytes) -> int:
    """Return the integer RMS of one PCM16 fragment."""

    if not samples:
        return 0
    usable = len(samples) - (len(samples) % 2)
    if usable <= 0:
        return 0
    pcm = np.frombuffer(samples[:usable], dtype=np.int16).astype(np.int32)
    mean_square = float(np.mean(np.square(pcm, dtype=np.int64)))
    return int(math.sqrt(max(0.0, mean_square)))


def _build_stream_capture_window(
    *,
    frames: deque[tuple[bytes, int]],
    sample_rate: int,
    channels: int,
    chunk_ms: int,
    speech_threshold: int,
    duration_ms: int = _STREAM_CAPTURE_WINDOW_MS,
) -> AmbientAudioCaptureWindow:
    """Build one localized capture window from recent replayed frames."""

    target_frames = max(1, math.ceil(max(chunk_ms, duration_ms) / max(1, chunk_ms)))
    selected_frames = list(frames)[-target_frames:]
    if not selected_frames:
        selected_frames = [(b"", 0)]
    rms_values = [rms for _chunk, rms in selected_frames]
    active_chunk_count = sum(1 for rms in rms_values if rms >= speech_threshold)
    sample = AmbientAudioLevelSample(
        duration_ms=len(selected_frames) * chunk_ms,
        chunk_count=len(selected_frames),
        active_chunk_count=active_chunk_count,
        average_rms=int(sum(rms_values) / max(1, len(rms_values))),
        peak_rms=max(rms_values),
        active_ratio=active_chunk_count / max(1, len(selected_frames)),
    )
    return AmbientAudioCaptureWindow(
        sample=sample,
        pcm_bytes=b"".join(chunk for chunk, _rms in selected_frames),
        sample_rate=sample_rate,
        channels=channels,
    )


def _build_stream_policy(
    *,
    config: TwinrConfig,
    backend,
) -> WakewordDecisionPolicy:
    """Build the runtime-faithful wakeword decision policy for replay."""

    verifier = (
        None
        if backend is None or config.wakeword_verifier_mode == "disabled"
        else SttWakewordVerifier(
            backend=backend,
            phrases=config.wakeword_phrases,
            language=config.openai_realtime_language,
        )
    )
    local_verifier = (
        None
        if not config.wakeword_openwakeword_sequence_verifier_models
        else WakewordSequenceCaptureVerifier(
            verifier_models=dict(config.wakeword_openwakeword_sequence_verifier_models),
            threshold=config.wakeword_openwakeword_sequence_verifier_threshold,
        )
    )
    return WakewordDecisionPolicy(
        primary_backend=config.wakeword_primary_backend,
        fallback_backend=config.wakeword_fallback_backend,
        verifier_mode=config.wakeword_verifier_mode,
        verifier_margin=config.wakeword_verifier_margin,
        primary_threshold=config.wakeword_openwakeword_threshold,
        verifier=verifier,
        local_verifier=local_verifier,
    )


def _build_stream_spotter(
    *,
    config: TwinrConfig,
    model_factory,
) -> WakewordOpenWakeWordFrameSpotter:
    """Build the runtime-faithful frame spotter for manifest replay."""

    if int(config.audio_sample_rate) != 16000 or int(config.audio_channels) != 1:
        raise ValueError(
            "Runtime-faithful wakeword stream replay requires TWINR_AUDIO_SAMPLE_RATE=16000 and TWINR_AUDIO_CHANNELS=1."
        )
    if not config.wakeword_openwakeword_models:
        raise ValueError("Wakeword stream replay requires at least one configured openWakeWord model.")
    return WakewordOpenWakeWordFrameSpotter(
        wakeword_models=config.wakeword_openwakeword_models,
        phrases=config.wakeword_phrases,
        threshold=config.wakeword_openwakeword_threshold,
        vad_threshold=config.wakeword_openwakeword_vad_threshold,
        patience_frames=config.wakeword_openwakeword_patience_frames,
        activation_samples=config.wakeword_openwakeword_activation_samples,
        deactivation_threshold=config.wakeword_openwakeword_deactivation_threshold,
        enable_speex_noise_suppression=config.wakeword_openwakeword_enable_speex,
        inference_framework=config.wakeword_openwakeword_inference_framework,
        custom_verifier_models=dict(config.wakeword_openwakeword_custom_verifier_models),
        custom_verifier_threshold=config.wakeword_openwakeword_custom_verifier_threshold,
        model_factory=model_factory,
    )


def _replay_stream_entry(
    *,
    config: TwinrConfig,
    entry: WakewordEvalEntry,
    policy: WakewordDecisionPolicy,
    spotter: WakewordOpenWakeWordFrameSpotter,
) -> tuple[bool, int, float]:
    """Replay one labeled entry through the stream detector and policy."""

    replay_pcm, duration_seconds = _load_pcm16_wav_for_stream_replay(
        entry.audio_path,
        target_sample_rate=int(config.audio_sample_rate),
        target_channels=int(config.audio_channels),
    )
    frame_bytes = spotter.frame_bytes_for_channels(int(config.audio_channels))
    chunk_ms = max(
        20,
        int((frame_bytes * 1000) / (int(config.audio_sample_rate) * int(config.audio_channels) * 2)),
    )
    history_frames = max(1, math.ceil(max(chunk_ms, _STREAM_CAPTURE_WINDOW_MS) / max(1, chunk_ms)))
    recent_frames: deque[tuple[bytes, int]] = deque(maxlen=history_frames)
    accepted_detection_count = 0
    last_detection_at_s: float | None = None
    max_offset = len(replay_pcm) - (len(replay_pcm) % frame_bytes)
    spotter.reset()
    for offset in range(0, max_offset, frame_bytes):
        frame = replay_pcm[offset:offset + frame_bytes]
        recent_frames.append((frame, _pcm16_rms(frame)))
        match = spotter.process_pcm_bytes(frame, channels=int(config.audio_channels))
        if match is None:
            continue
        elapsed_s = float(offset + frame_bytes) / float(int(config.audio_sample_rate) * int(config.audio_channels) * 2)
        if (
            last_detection_at_s is not None
            and (elapsed_s - last_detection_at_s) < float(config.wakeword_attempt_cooldown_s)
        ):
            continue
        last_detection_at_s = elapsed_s
        capture_window = _build_stream_capture_window(
            frames=recent_frames,
            sample_rate=int(config.audio_sample_rate),
            channels=int(config.audio_channels),
            chunk_ms=chunk_ms,
            speech_threshold=int(config.audio_speech_threshold),
        )
        decision = policy.decide(match=match, capture=capture_window, source="streaming_spotter")
        if decision.detected:
            accepted_detection_count += 1
    return accepted_detection_count > 0, accepted_detection_count, duration_seconds


def evaluate_wakeword_stream_entries(
    *,
    config: TwinrConfig,
    entries: list[WakewordEvalEntry],
    backend=None,
    model_factory=None,
) -> WakewordStreamEvalReport:
    """Replay labeled clips through the runtime-faithful streaming path."""

    policy = _build_stream_policy(config=config, backend=backend)
    spotter = _build_stream_spotter(config=config, model_factory=model_factory)
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    evaluated_entries = 0
    accepted_detection_count = 0
    total_audio_seconds = 0.0
    for entry in entries:
        expected = _expected_detected(entry.label)
        if expected is None:
            continue
        detected, clip_detection_count, clip_duration_seconds = _replay_stream_entry(
            config=config,
            entry=entry,
            policy=policy,
            spotter=spotter,
        )
        accepted_detection_count += clip_detection_count
        total_audio_seconds += clip_duration_seconds
        evaluated_entries += 1
        if expected and detected:
            counts["tp"] += 1
        elif expected:
            counts["fn"] += 1
        elif detected:
            counts["fp"] += 1
        else:
            counts["tn"] += 1
    return WakewordStreamEvalReport(
        metrics=_metrics_from_counts(
            true_positive=counts["tp"],
            false_positive=counts["fp"],
            true_negative=counts["tn"],
            false_negative=counts["fn"],
        ),
        evaluated_entries=evaluated_entries,
        accepted_detection_count=accepted_detection_count,
        total_audio_seconds=total_audio_seconds,
    )


def run_wakeword_stream_eval(
    *,
    config: TwinrConfig,
    manifest_path: str | Path,
    backend=None,
    model_factory=None,
) -> WakewordStreamEvalReport:
    """Run runtime-faithful stream replay and persist the latest JSON report."""

    entries = load_eval_manifest(manifest_path)
    report = evaluate_wakeword_stream_entries(
        config=config,
        entries=entries,
        backend=backend,
        model_factory=model_factory,
    )
    reports_root = resolve_ops_paths_for_config(config).artifacts_root / "ops" / _REPORT_DIRNAME
    reports_root.mkdir(parents=True, exist_ok=True)
    report_path = reports_root / "latest_stream_eval.json"
    payload = {
        "eval_mode": "runtime_stream_replay",
        "evaluated_entries": report.evaluated_entries,
        "accepted_detection_count": report.accepted_detection_count,
        "total_audio_seconds": round(report.total_audio_seconds, 6),
        "accepted_detections_per_hour": round(
            report.accepted_detection_count / max(report.total_audio_seconds / 3600.0, 1e-9),
            6,
        ),
        "metrics": {
            "total": report.metrics.total,
            "true_positive": report.metrics.true_positive,
            "false_positive": report.metrics.false_positive,
            "true_negative": report.metrics.true_negative,
            "false_negative": report.metrics.false_negative,
            "precision": round(report.metrics.precision, 6),
            "recall": round(report.metrics.recall, 6),
            "false_positive_rate": round(report.metrics.false_positive_rate, 6),
            "false_negative_rate": round(report.metrics.false_negative_rate, 6),
        },
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return WakewordStreamEvalReport(
        metrics=report.metrics,
        evaluated_entries=report.evaluated_entries,
        accepted_detection_count=report.accepted_detection_count,
        total_audio_seconds=report.total_audio_seconds,
        report_path=report_path,
    )


def load_wakeword_promotion_spec(spec_path: str | Path) -> WakewordPromotionSpec:
    """Load one JSON promotion spec with labeled suites and ambient guards."""

    path = Path(spec_path).expanduser().resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Wakeword promotion spec must be one JSON object.")

    def _resolve_manifest(item: dict[str, object]) -> Path:
        raw_path = str(item.get("manifest_path") or item.get("manifest") or item.get("path") or "").strip()
        if not raw_path:
            raise ValueError("Wakeword promotion spec items require manifest_path.")
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = (path.parent / candidate).resolve(strict=False)
        return candidate

    def _optional_nonnegative_int(value: object | None) -> int | None:
        if value is None:
            return None
        normalized = int(value)
        if normalized < 0:
            raise ValueError("Promotion suite integer limits must be non-negative.")
        return normalized

    def _optional_probability(value: object | None) -> float | None:
        if value is None:
            return None
        normalized = float(value)
        if not 0.0 <= normalized <= 1.0:
            raise ValueError("Promotion suite probability limits must be within [0.0, 1.0].")
        return normalized

    suites_payload = payload.get("suites") or []
    if not isinstance(suites_payload, list):
        raise ValueError("Wakeword promotion spec field suites must be a JSON array.")
    suites: list[WakewordPromotionSuiteSpec] = []
    for raw_suite in suites_payload:
        if not isinstance(raw_suite, dict):
            raise ValueError("Wakeword promotion suites must be JSON objects.")
        name = str(raw_suite.get("name") or "").strip()
        if not name:
            raise ValueError("Wakeword promotion suites require a non-empty name.")
        suites.append(
            WakewordPromotionSuiteSpec(
                name=name,
                manifest_path=_resolve_manifest(raw_suite),
                max_false_negatives=_optional_nonnegative_int(raw_suite.get("max_false_negatives")),
                max_false_positives=_optional_nonnegative_int(raw_suite.get("max_false_positives")),
                min_precision=_optional_probability(raw_suite.get("min_precision")),
                min_recall=_optional_probability(raw_suite.get("min_recall")),
            )
        )
    if not suites:
        raise ValueError("Wakeword promotion spec requires at least one suite.")

    ambient_payload = payload.get("ambient_guards")
    if ambient_payload is None:
        ambient_payload = payload.get("ambient") or []
    if not isinstance(ambient_payload, list):
        raise ValueError("Wakeword promotion spec field ambient_guards must be a JSON array.")
    ambient_guards: list[WakewordAmbientGuardSpec] = []
    for raw_guard in ambient_payload:
        if not isinstance(raw_guard, dict):
            raise ValueError("Wakeword ambient guards must be JSON objects.")
        name = str(raw_guard.get("name") or "").strip()
        if not name:
            raise ValueError("Wakeword ambient guards require a non-empty name.")
        max_false_accepts_per_hour = float(raw_guard.get("max_false_accepts_per_hour"))
        if max_false_accepts_per_hour < 0.0:
            raise ValueError("max_false_accepts_per_hour must be non-negative.")
        ambient_guards.append(
            WakewordAmbientGuardSpec(
                name=name,
                manifest_path=_resolve_manifest(raw_guard),
                max_false_accepts_per_hour=max_false_accepts_per_hour,
            )
        )

    return WakewordPromotionSpec(
        suites=tuple(suites),
        ambient_guards=tuple(ambient_guards),
    )


def run_wakeword_promotion_eval(
    *,
    config: TwinrConfig,
    spec_path: str | Path,
    backend=None,
    model_factory=None,
) -> WakewordPromotionReport:
    """Run suite plus ambient promotion guards and persist a JSON report."""

    spec_file = Path(spec_path).expanduser().resolve(strict=True)
    spec = load_wakeword_promotion_spec(spec_file)
    suite_results: list[WakewordPromotionSuiteResult] = []
    ambient_results: list[WakewordAmbientGuardResult] = []
    blockers: list[str] = []

    for suite in spec.suites:
        report = evaluate_wakeword_stream_entries(
            config=config,
            entries=load_eval_manifest(suite.manifest_path),
            backend=backend,
            model_factory=model_factory,
        )
        suite_results.append(WakewordPromotionSuiteResult(spec=suite, report=report))
        if suite.max_false_negatives is not None and report.metrics.false_negative > suite.max_false_negatives:
            blockers.append(
                f"{suite.name}: false_negative={report.metrics.false_negative} exceeds max_false_negatives={suite.max_false_negatives}"
            )
        if suite.max_false_positives is not None and report.metrics.false_positive > suite.max_false_positives:
            blockers.append(
                f"{suite.name}: false_positive={report.metrics.false_positive} exceeds max_false_positives={suite.max_false_positives}"
            )
        if suite.min_precision is not None and report.metrics.precision < suite.min_precision:
            blockers.append(
                f"{suite.name}: precision={report.metrics.precision:.6f} is below min_precision={suite.min_precision:.6f}"
            )
        if suite.min_recall is not None and report.metrics.recall < suite.min_recall:
            blockers.append(
                f"{suite.name}: recall={report.metrics.recall:.6f} is below min_recall={suite.min_recall:.6f}"
            )

    for ambient_guard in spec.ambient_guards:
        entries = load_eval_manifest(ambient_guard.manifest_path)
        if any(_expected_detected(entry.label) is not False for entry in entries):
            raise ValueError(
                f"Ambient guard {ambient_guard.name} must contain only negative labels."
            )
        report = evaluate_wakeword_stream_entries(
            config=config,
            entries=entries,
            backend=backend,
            model_factory=model_factory,
        )
        false_accepts_per_hour = report.accepted_detection_count / max(report.total_audio_seconds / 3600.0, 1e-9)
        ambient_results.append(
            WakewordAmbientGuardResult(
                spec=ambient_guard,
                report=report,
                false_accepts_per_hour=false_accepts_per_hour,
            )
        )
        if false_accepts_per_hour > ambient_guard.max_false_accepts_per_hour:
            blockers.append(
                f"{ambient_guard.name}: false_accepts_per_hour={false_accepts_per_hour:.6f} exceeds max_false_accepts_per_hour={ambient_guard.max_false_accepts_per_hour:.6f}"
            )

    reports_root = resolve_ops_paths_for_config(config).artifacts_root / "ops" / _REPORT_DIRNAME
    reports_root.mkdir(parents=True, exist_ok=True)
    report_path = reports_root / "latest_promotion.json"
    payload = {
        "passed": not blockers,
        "blockers": blockers,
        "spec_path": str(spec_file),
        "suites": [
            {
                "name": result.spec.name,
                "manifest_path": str(result.spec.manifest_path),
                "constraints": {
                    "max_false_negatives": result.spec.max_false_negatives,
                    "max_false_positives": result.spec.max_false_positives,
                    "min_precision": result.spec.min_precision,
                    "min_recall": result.spec.min_recall,
                },
                "accepted_detection_count": result.report.accepted_detection_count,
                "total_audio_seconds": round(result.report.total_audio_seconds, 6),
                "metrics": {
                    "total": result.report.metrics.total,
                    "true_positive": result.report.metrics.true_positive,
                    "false_positive": result.report.metrics.false_positive,
                    "true_negative": result.report.metrics.true_negative,
                    "false_negative": result.report.metrics.false_negative,
                    "precision": round(result.report.metrics.precision, 6),
                    "recall": round(result.report.metrics.recall, 6),
                    "false_positive_rate": round(result.report.metrics.false_positive_rate, 6),
                    "false_negative_rate": round(result.report.metrics.false_negative_rate, 6),
                },
            }
            for result in suite_results
        ],
        "ambient_guards": [
            {
                "name": result.spec.name,
                "manifest_path": str(result.spec.manifest_path),
                "max_false_accepts_per_hour": result.spec.max_false_accepts_per_hour,
                "accepted_detection_count": result.report.accepted_detection_count,
                "total_audio_seconds": round(result.report.total_audio_seconds, 6),
                "false_accepts_per_hour": round(result.false_accepts_per_hour, 6),
            }
            for result in ambient_results
        ],
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return WakewordPromotionReport(
        spec_path=spec_file,
        suite_results=tuple(suite_results),
        ambient_results=tuple(ambient_results),
        blockers=tuple(blockers),
        passed=not blockers,
        report_path=report_path,
    )


__all__ = [
    "WakewordAmbientGuardResult",
    "WakewordAmbientGuardSpec",
    "WakewordPromotionReport",
    "WakewordPromotionSpec",
    "WakewordPromotionSuiteResult",
    "WakewordPromotionSuiteSpec",
    "WakewordStreamEvalReport",
    "evaluate_wakeword_stream_entries",
    "load_wakeword_promotion_spec",
    "run_wakeword_promotion_eval",
    "run_wakeword_stream_eval",
]
