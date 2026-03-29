# CHANGELOG: 2026-03-29
# BUG-1: Fixed a deterministic crash when proactive PCM capture conflicts exist but the proactive target is not ReSpeaker; the old code could call observe() on None.
# BUG-2: Fixed unbounded failure propagation and partial cleanup during sampler/provider setup and observe(); failures now degrade into bounded snapshots and all created closeables are closed best-effort.
# BUG-3: Fixed temporal cold-starting of ReSpeaker audio policy tracking by reusing a per-config tracker and serializing tracker.observe() calls.
# SEC-1: Hardened operator-facing CLI rendering against terminal-control injection and oversized untrusted device/runtime text.
# SEC-2: Added bounded capture deadlines, in-process single-flight capture locking, and timeout backoff to reduce trivial local DoS from wedged or hostile audio hardware.
# IMP-1: Resolved real assistant-output activity via optional config hooks instead of hardcoding assistant_output_active_predicate=lambda: False whenever runtime state is available.
# IMP-2: Added richer machine-readable diagnostic metadata plus an optional bounded auxiliary-evidence hook for Pi-4-friendly neural VAD / wake-word / DDSD enrichers.

"""Build bounded ReSpeaker audio-perception snapshots for diagnostics.

This module reuses the same normalized audio observation and ReSpeaker policy
surfaces that the proactive monitor already trusts. It exists so operator-facing
CLI checks and post-recovery hardware smokes can validate real runtime facts
such as speech, non-speech activity, background-media suspicion, and
device-directed speech candidacy without duplicating monitor orchestration.

2026 upgrade notes
------------------
- Snapshot capture is now genuinely bounded from the caller's perspective.
- Runtime failures degrade into explicit diagnostic facts instead of crashing.
- Policy tracking is reused across calls so temporal policy signals stay
  runtime-faithful.
- Operator-facing text rendering is sanitized to stay single-line and
  terminal-safe.
- Deployments can optionally attach modern neural auxiliary evidence
  (e.g. Silero / Cobra / openWakeWord / distilled DDSD) without coupling this
  module to a specific vendor or model family.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
import logging
from queue import SimpleQueue
import threading
import time
from typing import Any, Callable, Mapping, cast
import weakref

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.audio import AmbientAudioSampler, AudioCaptureReadinessProbe
from twinr.hardware.respeaker import ReSpeakerSignalProvider, config_targets_respeaker
from twinr.proactive.social import SocialAudioObservation
from twinr.proactive.social.observers import (
    AmbientAudioObservationProvider,
    ProactiveAudioSnapshot,
    ReSpeakerAudioObservationProvider,
)

from .audio_policy import ReSpeakerAudioPolicySnapshot, ReSpeakerAudioPolicyTracker
from .service_impl.compat import _proactive_pcm_capture_conflicts_with_voice_orchestrator


logger = logging.getLogger(__name__)

_SNAPSHOT_SCHEMA_VERSION = "2026-03-29"
_MIN_CAPTURE_TIMEOUT_S = 0.5
_MAX_CAPTURE_TIMEOUT_S = 6.0
_DEFAULT_CAPTURE_LOCK_TIMEOUT_S = 0.2
_DEFAULT_TIMEOUT_COOLDOWN_S = 2.0
_DEFAULT_AUXILIARY_EVIDENCE_TIMEOUT_S = 0.35
_MIN_SAMPLE_MS = 10
_DEFAULT_SAMPLE_MS = 1000
_MAX_TEXT_FIELD_CHARS = 240
_MAX_AUXILIARY_EVIDENCE_ITEMS = 16
_MAX_CAPTURE_LOCK_CACHE_SIZE = 32
_MAX_TRACKER_CACHE_SIZE = 16

_CAPTURE_LOCKS: "OrderedDict[str, threading.Lock]" = OrderedDict()
_CAPTURE_COOLDOWN_UNTIL: dict[str, float] = {}
_POLICY_TRACKERS_WEAK: "weakref.WeakKeyDictionary[object, _CachedPolicyTracker]" = weakref.WeakKeyDictionary()
_POLICY_TRACKERS_NON_WEAK: "OrderedDict[int, _CachedPolicyTracker]" = OrderedDict()
_TRACKER_LOCKS: "OrderedDict[int, threading.Lock]" = OrderedDict()
_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class ReSpeakerAudioPerceptionGuardSnapshot:
    """Summarize conservative room-context and device-directedness hints."""

    room_context: str
    device_directed_speech_candidate: bool
    guard_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ReSpeakerAudioPerceptionSnapshot:
    """Bundle one runtime-faithful audio observation with guard metadata."""

    audio_snapshot: ProactiveAudioSnapshot
    audio_policy_snapshot: ReSpeakerAudioPolicySnapshot
    perception_guard: ReSpeakerAudioPerceptionGuardSnapshot
    respeaker_targeted: bool
    capture_probe: AudioCaptureReadinessProbe | None = None
    capture_deadline_s: float | None = None
    capture_elapsed_ms: int | None = None
    capture_error_code: str | None = None
    capture_error_message: str | None = None
    collection_timed_out: bool = False
    auxiliary_evidence: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class _SyntheticAudioCaptureReadinessProbe:
    ready: bool | None = None
    captured_chunk_count: int = 0
    target_chunk_count: int = 0
    captured_bytes: int = 0


@dataclass(frozen=True, slots=True)
class _SyntheticSocialAudioObservation:
    speech_detected: bool | None = None
    distress_detected: bool | None = None
    room_quiet: bool | None = None
    recent_speech_age_s: float | None = None
    assistant_output_active: bool | None = None
    azimuth_deg: int | None = None
    direction_confidence: float | None = None
    device_runtime_mode: str | None = None
    host_control_ready: bool | None = None
    transport_reason: str | None = None
    non_speech_audio_likely: bool | None = None
    background_media_likely: bool | None = None
    speech_overlap_likely: bool | None = None
    barge_in_detected: bool | None = None
    mute_active: bool | None = None


@dataclass(frozen=True, slots=True)
class _SyntheticProactiveAudioSnapshot:
    observation: SocialAudioObservation
    signal_snapshot: object | None = None
    sample: object | None = None


@dataclass(frozen=True, slots=True)
class _SyntheticReSpeakerAudioPolicySnapshot:
    presence_audio_active: bool | None = None
    recent_follow_up_speech: bool | None = None
    room_busy_or_overlapping: bool | None = None
    quiet_window_open: bool | None = None
    non_speech_audio_likely: bool | None = None
    background_media_likely: bool | None = None
    barge_in_recent: bool | None = None
    speaker_direction_stable: bool | None = None
    mute_blocks_voice_capture: bool | None = None
    resume_window_open: bool | None = None
    initiative_block_reason: str | None = None
    speech_delivery_defer_reason: str | None = None
    runtime_alert_code: str | None = None
    runtime_alert_message: str | None = None


@dataclass(slots=True)
class _CachedPolicyTracker:
    tracker: ReSpeakerAudioPolicyTracker
    observe_lock: threading.Lock


class _CaptureUnavailable(RuntimeError):
    """Raised when no safe capture path is currently available."""

    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = _normalize_metric_key(code, fallback="capture_unavailable")
        self.message = _normalize_optional_text(message) if message is not None else None
        super().__init__(self.message or self.code)


def observe_audio_perception_once(config: TwinrConfig) -> ReSpeakerAudioPerceptionSnapshot:
    """Capture one bounded audio-perception snapshot from current config."""

    sample_ms = _resolve_sample_ms(config)
    capture_timeout_s = _resolve_capture_timeout_s(config, sample_ms=sample_ms)
    capture_started = time.monotonic()
    respeaker_targeted = config_targets_respeaker(
        getattr(config, "audio_input_device", None),
        getattr(config, "proactive_audio_input_device", None),
    )
    capture_key = _capture_device_key(config=config, respeaker_targeted=respeaker_targeted)
    cooldown_until = _capture_cooldown_until(capture_key)
    now = time.monotonic()
    if now < cooldown_until:
        return _build_degraded_perception_snapshot(
            respeaker_targeted=respeaker_targeted,
            capture_deadline_s=capture_timeout_s,
            capture_elapsed_ms=int((now - capture_started) * 1000),
            error_code="capture_backoff_active",
            error_message=(
                f"audio capture is in cooldown for {max(cooldown_until - now, 0.0):.3f}s after a previous timeout"
            ),
            transport_reason="capture_backoff_active",
            runtime_mode="cooldown",
            capture_probe_ready=False,
        )

    capture_lock = _get_capture_lock(capture_key)
    lock_timeout_s = _resolve_capture_lock_timeout_s(config, capture_timeout_s)
    if not capture_lock.acquire(timeout=lock_timeout_s):
        return _build_degraded_perception_snapshot(
            respeaker_targeted=respeaker_targeted,
            capture_deadline_s=capture_timeout_s,
            capture_elapsed_ms=int((time.monotonic() - capture_started) * 1000),
            error_code="capture_busy",
            error_message="another in-process audio-perception snapshot is already probing the same device",
            transport_reason="capture_busy",
            runtime_mode="busy",
            capture_probe_ready=False,
        )

    timed_out = False
    try:
        try:
            audio_snapshot, capture_probe = _run_with_timeout(
                description="audio_perception_capture",
                timeout_s=capture_timeout_s,
                fn=lambda: _capture_audio_snapshot(
                    config=config,
                    sample_ms=sample_ms,
                    respeaker_targeted=respeaker_targeted,
                ),
            )
            capture_error_code = None
            capture_error_message = None
        except TimeoutError as exc:
            timed_out = True
            _set_capture_cooldown(
                capture_key,
                duration_s=_resolve_timeout_cooldown_s(config, capture_timeout_s),
            )
            snapshot = _build_degraded_perception_snapshot(
                respeaker_targeted=respeaker_targeted,
                capture_deadline_s=capture_timeout_s,
                capture_elapsed_ms=int((time.monotonic() - capture_started) * 1000),
                error_code="capture_timeout",
                error_message=str(exc),
                transport_reason="capture_timeout",
                runtime_mode="timeout",
                capture_probe_ready=False,
                collection_timed_out=True,
            )
            return _attach_auxiliary_evidence(config, snapshot)
        except _CaptureUnavailable as exc:
            snapshot = _build_degraded_perception_snapshot(
                respeaker_targeted=respeaker_targeted,
                capture_deadline_s=capture_timeout_s,
                capture_elapsed_ms=int((time.monotonic() - capture_started) * 1000),
                error_code=exc.code,
                error_message=exc.message,
                transport_reason=exc.code,
                runtime_mode="blocked",
                capture_probe_ready=False,
            )
            return _attach_auxiliary_evidence(config, snapshot)
        except Exception as exc:
            logger.exception("observe_audio_perception_once capture failed")
            snapshot = _build_degraded_perception_snapshot(
                respeaker_targeted=respeaker_targeted,
                capture_deadline_s=capture_timeout_s,
                capture_elapsed_ms=int((time.monotonic() - capture_started) * 1000),
                error_code="capture_failed",
                error_message=str(exc),
                transport_reason="capture_failed",
                runtime_mode="error",
                capture_probe_ready=False,
            )
            return _attach_auxiliary_evidence(config, snapshot)

        capture_error_code = None
        capture_error_message = None
        audio_policy_snapshot = _observe_audio_policy_snapshot(
            config=config,
            audio=audio_snapshot.observation,
        )
        snapshot = ReSpeakerAudioPerceptionSnapshot(
            audio_snapshot=audio_snapshot,
            audio_policy_snapshot=audio_policy_snapshot,
            perception_guard=derive_respeaker_audio_perception_guard(
                audio=audio_snapshot.observation,
                policy_snapshot=audio_policy_snapshot,
            ),
            respeaker_targeted=respeaker_targeted,
            capture_probe=capture_probe,
            capture_deadline_s=capture_timeout_s,
            capture_elapsed_ms=int((time.monotonic() - capture_started) * 1000),
            capture_error_code=capture_error_code,
            capture_error_message=capture_error_message,
            collection_timed_out=timed_out,
        )
        return _attach_auxiliary_evidence(config, snapshot)
    finally:
        capture_lock.release()


def _capture_audio_snapshot(
    *,
    config: TwinrConfig,
    sample_ms: int,
    respeaker_targeted: bool,
) -> tuple[ProactiveAudioSnapshot, AudioCaptureReadinessProbe | None]:
    """Build one audio snapshot and best-effort probe info."""

    distress_enabled = bool(
        getattr(config, "proactive_enabled", False)
        and getattr(config, "proactive_audio_distress_enabled", False)
    )
    shared_voice_capture_conflict = _proactive_pcm_capture_conflicts_with_voice_orchestrator(
        config,
        require_active_owner=True,
    )
    sampler: AmbientAudioSampler | None = None
    fallback_observer: object | None = None
    observer: object | None = None
    capture_probe: AudioCaptureReadinessProbe | None = None
    try:
        if not shared_voice_capture_conflict:
            sampler = AmbientAudioSampler.from_config(config)
            fallback_observer = AmbientAudioObservationProvider(
                sampler=sampler,
                sample_ms=sample_ms,
                distress_enabled=distress_enabled,
            )
        elif respeaker_targeted:
            logger.info(
                "observe_audio_perception_once skipped PCM probe because the active voice orchestrator owns the same capture device"
            )
        else:
            raise _CaptureUnavailable(
                "shared_voice_capture_conflict",
                "the active voice orchestrator owns the proactive capture device",
            )

        observer = fallback_observer
        if respeaker_targeted:
            if sampler is not None:
                capture_probe = sampler.require_readable_frames(
                    duration_ms=_resolve_capture_probe_duration_ms(
                        sample_ms=sample_ms,
                        chunk_ms=getattr(sampler, "chunk_ms", 0),
                    ),
                )
            observer = ReSpeakerAudioObservationProvider(
                signal_provider=ReSpeakerSignalProvider(
                    sensor_window_ms=sample_ms,
                    assistant_output_active_predicate=_resolve_assistant_output_active_predicate(config),
                ),
                fallback_observer=fallback_observer,
            )

        if observer is None:
            raise _CaptureUnavailable(
                "observer_unavailable",
                "no audio observer could be constructed for the current configuration",
            )

        audio_snapshot = cast(ProactiveAudioSnapshot, observer.observe())
        return audio_snapshot, capture_probe
    finally:
        _close_quietly(observer)
        if fallback_observer is not observer:
            _close_quietly(fallback_observer)
        if sampler is not fallback_observer and sampler is not observer:
            _close_quietly(sampler)


def derive_respeaker_audio_perception_guard(
    *,
    audio: SocialAudioObservation,
    policy_snapshot: ReSpeakerAudioPolicySnapshot,
) -> ReSpeakerAudioPerceptionGuardSnapshot:
    """Derive one conservative room-context guard from normalized audio facts."""

    runtime_alert_code = _normalize_optional_text(policy_snapshot.runtime_alert_code)
    if runtime_alert_code not in {None, "ready"}:
        return ReSpeakerAudioPerceptionGuardSnapshot(
            room_context="capture_blocked",
            device_directed_speech_candidate=False,
            guard_reason=runtime_alert_code,
        )
    if policy_snapshot.mute_blocks_voice_capture is True:
        return ReSpeakerAudioPerceptionGuardSnapshot(
            room_context="muted",
            device_directed_speech_candidate=False,
            guard_reason="mute_blocks_voice_capture",
        )
    if audio.assistant_output_active is True:
        return ReSpeakerAudioPerceptionGuardSnapshot(
            room_context="assistant_output",
            device_directed_speech_candidate=False,
            guard_reason="assistant_output_active",
        )
    if policy_snapshot.background_media_likely is True or audio.background_media_likely is True:
        return ReSpeakerAudioPerceptionGuardSnapshot(
            room_context="background_media",
            device_directed_speech_candidate=False,
            guard_reason="background_media_active",
        )
    if policy_snapshot.non_speech_audio_likely is True or audio.non_speech_audio_likely is True:
        return ReSpeakerAudioPerceptionGuardSnapshot(
            room_context="non_speech_activity",
            device_directed_speech_candidate=False,
            guard_reason="non_speech_audio_active",
        )
    if policy_snapshot.room_busy_or_overlapping is True or audio.speech_overlap_likely is True:
        return ReSpeakerAudioPerceptionGuardSnapshot(
            room_context="overlapping_speech",
            device_directed_speech_candidate=False,
            guard_reason="room_busy_or_overlapping",
        )
    if audio.speech_detected is True:
        if policy_snapshot.presence_audio_active is not True:
            return ReSpeakerAudioPerceptionGuardSnapshot(
                room_context="speech",
                device_directed_speech_candidate=False,
                guard_reason="presence_audio_inactive",
            )
        if policy_snapshot.speaker_direction_stable is not True:
            return ReSpeakerAudioPerceptionGuardSnapshot(
                room_context="speech",
                device_directed_speech_candidate=False,
                guard_reason="speaker_direction_unstable",
            )
        return ReSpeakerAudioPerceptionGuardSnapshot(
            room_context="speech",
            device_directed_speech_candidate=True,
            guard_reason=None,
        )
    if policy_snapshot.quiet_window_open is True or audio.room_quiet is True:
        return ReSpeakerAudioPerceptionGuardSnapshot(
            room_context="quiet",
            device_directed_speech_candidate=False,
            guard_reason="quiet_room",
        )
    return ReSpeakerAudioPerceptionGuardSnapshot(
        room_context="unknown",
        device_directed_speech_candidate=False,
        guard_reason="insufficient_audio_evidence",
    )


def render_audio_perception_snapshot_lines(
    snapshot: ReSpeakerAudioPerceptionSnapshot,
) -> tuple[str, ...]:
    """Render one operator-readable line set for CLI and shell diagnostics."""

    audio_snapshot = snapshot.audio_snapshot
    observation = audio_snapshot.observation
    policy_snapshot = snapshot.audio_policy_snapshot
    signal_snapshot = audio_snapshot.signal_snapshot
    sample = audio_snapshot.sample
    capture_probe = snapshot.capture_probe
    lines = [
        f"proactive_audio_snapshot_schema={_SNAPSHOT_SCHEMA_VERSION}",
        f"proactive_audio_target={'respeaker' if snapshot.respeaker_targeted else 'ambient'}",
        f"proactive_audio_capture_deadline_s={_format_optional_float(snapshot.capture_deadline_s)}",
        f"proactive_audio_capture_elapsed_ms={_format_optional_int(snapshot.capture_elapsed_ms)}",
        f"proactive_audio_capture_timed_out={_format_optional_bool(snapshot.collection_timed_out)}",
        f"proactive_audio_capture_error_code={_format_optional_text(snapshot.capture_error_code)}",
        f"proactive_audio_capture_error_message={_format_optional_text(snapshot.capture_error_message)}",
        f"proactive_audio_capture_probe_ready={_format_optional_bool(None if capture_probe is None else capture_probe.ready)}",
        "proactive_audio_capture_probe_chunks="
        f"{'unknown' if capture_probe is None else f'{capture_probe.captured_chunk_count}/{capture_probe.target_chunk_count}'}",
        f"proactive_audio_capture_probe_bytes={_format_optional_int(None if capture_probe is None else capture_probe.captured_bytes)}",
        f"proactive_speech_detected={_format_optional_bool(observation.speech_detected)}",
        f"proactive_distress_detected={_format_optional_bool(observation.distress_detected)}",
        f"proactive_room_quiet={_format_optional_bool(observation.room_quiet)}",
        f"proactive_recent_speech_age_s={_format_optional_float(observation.recent_speech_age_s)}",
        f"proactive_assistant_output_active={_format_optional_bool(observation.assistant_output_active)}",
        f"proactive_audio_azimuth_deg={_format_optional_int(observation.azimuth_deg)}",
        f"proactive_audio_direction_confidence={_format_optional_float(observation.direction_confidence)}",
        f"proactive_audio_device_runtime_mode={_format_optional_text(observation.device_runtime_mode)}",
        f"proactive_audio_host_control_ready={_format_optional_bool(observation.host_control_ready)}",
        f"proactive_audio_transport_reason={_format_optional_text(observation.transport_reason)}",
        f"proactive_non_speech_audio_likely={_format_optional_bool(observation.non_speech_audio_likely)}",
        f"proactive_background_media_likely={_format_optional_bool(observation.background_media_likely)}",
        f"proactive_speech_overlap_likely={_format_optional_bool(observation.speech_overlap_likely)}",
        f"proactive_barge_in_detected={_format_optional_bool(observation.barge_in_detected)}",
        f"proactive_mute_active={_format_optional_bool(observation.mute_active)}",
        f"proactive_audio_peak_rms={_format_optional_int(None if sample is None else getattr(sample, 'peak_rms', None))}",
        f"proactive_audio_average_rms={_format_optional_int(None if sample is None else getattr(sample, 'average_rms', None))}",
        f"proactive_audio_active_ratio={_format_optional_float(None if sample is None else getattr(sample, 'active_ratio', None))}",
        f"proactive_audio_policy_presence_audio_active={_format_optional_bool(policy_snapshot.presence_audio_active)}",
        f"proactive_audio_policy_recent_follow_up_speech={_format_optional_bool(policy_snapshot.recent_follow_up_speech)}",
        f"proactive_audio_policy_room_busy_or_overlapping={_format_optional_bool(policy_snapshot.room_busy_or_overlapping)}",
        f"proactive_audio_policy_quiet_window_open={_format_optional_bool(policy_snapshot.quiet_window_open)}",
        f"proactive_audio_policy_non_speech_audio_likely={_format_optional_bool(policy_snapshot.non_speech_audio_likely)}",
        f"proactive_audio_policy_background_media_likely={_format_optional_bool(policy_snapshot.background_media_likely)}",
        f"proactive_audio_policy_barge_in_recent={_format_optional_bool(policy_snapshot.barge_in_recent)}",
        f"proactive_audio_policy_speaker_direction_stable={_format_optional_bool(policy_snapshot.speaker_direction_stable)}",
        f"proactive_audio_policy_mute_blocks_voice_capture={_format_optional_bool(policy_snapshot.mute_blocks_voice_capture)}",
        f"proactive_audio_policy_resume_window_open={_format_optional_bool(policy_snapshot.resume_window_open)}",
        f"proactive_audio_policy_initiative_block_reason={_format_optional_text(policy_snapshot.initiative_block_reason)}",
        f"proactive_audio_policy_speech_delivery_defer_reason={_format_optional_text(policy_snapshot.speech_delivery_defer_reason)}",
        f"proactive_audio_policy_runtime_alert_code={_format_optional_text(policy_snapshot.runtime_alert_code)}",
        f"proactive_audio_policy_runtime_alert_message={_format_optional_text(policy_snapshot.runtime_alert_message)}",
        f"proactive_audio_room_context={_normalize_metric_key(snapshot.perception_guard.room_context, fallback='unknown')}",
        "proactive_device_directed_speech_candidate="
        f"{_format_optional_bool(snapshot.perception_guard.device_directed_speech_candidate)}",
        f"proactive_audio_guard_reason={_format_optional_text(snapshot.perception_guard.guard_reason)}",
    ]
    if signal_snapshot is not None:
        lines.extend(
            (
                f"proactive_audio_signal_source={_format_optional_text(getattr(signal_snapshot, 'source', None))}",
                "proactive_audio_signal_requires_elevated_permissions="
                f"{_format_optional_bool(getattr(signal_snapshot, 'requires_elevated_permissions', None))}",
            )
        )
    if snapshot.auxiliary_evidence:
        lines.append(f"proactive_audio_auxiliary_evidence_count={len(snapshot.auxiliary_evidence)}")
        for key, value in snapshot.auxiliary_evidence:
            lines.append(f"proactive_audio_aux_{key}={value}")
    return tuple(lines)


def _observe_audio_policy_snapshot(
    *,
    config: TwinrConfig,
    audio: SocialAudioObservation,
) -> ReSpeakerAudioPolicySnapshot:
    """Observe one policy snapshot while preserving tracker continuity."""

    cached_tracker = _get_cached_policy_tracker(config)
    try:
        with cached_tracker.observe_lock:
            return cast(
                ReSpeakerAudioPolicySnapshot,
                cached_tracker.tracker.observe(now=time.monotonic(), audio=audio),
            )
    except Exception as exc:
        logger.exception("observe_audio_perception_once policy observation failed")
        return cast(
            ReSpeakerAudioPolicySnapshot,
            _SyntheticReSpeakerAudioPolicySnapshot(
                runtime_alert_code="policy_observe_failed",
                runtime_alert_message=str(exc),
            ),
        )


def _get_cached_policy_tracker(config: TwinrConfig) -> _CachedPolicyTracker:
    """Return a reusable tracker for one config object."""

    for attr_name in (
        "proactive_audio_policy_tracker",
        "respeaker_audio_policy_tracker",
        "audio_policy_tracker",
    ):
        provided = getattr(config, attr_name, None)
        if provided is not None and hasattr(provided, "observe"):
            return _CachedPolicyTracker(
                tracker=cast(ReSpeakerAudioPolicyTracker, provided),
                observe_lock=_get_or_create_tracker_lock(provided),
            )

    with _CACHE_LOCK:
        try:
            cached = _POLICY_TRACKERS_WEAK.get(config)
        except TypeError:
            cached = None
        if cached is not None:
            return cached

        non_weak_key = id(config)
        cached = _POLICY_TRACKERS_NON_WEAK.get(non_weak_key)
        if cached is not None:
            _POLICY_TRACKERS_NON_WEAK.move_to_end(non_weak_key)
            return cached

        cached = _CachedPolicyTracker(
            tracker=ReSpeakerAudioPolicyTracker.from_config(config),
            observe_lock=threading.Lock(),
        )
        try:
            _POLICY_TRACKERS_WEAK[config] = cached
        except TypeError:
            _POLICY_TRACKERS_NON_WEAK[non_weak_key] = cached
            _POLICY_TRACKERS_NON_WEAK.move_to_end(non_weak_key)
            while len(_POLICY_TRACKERS_NON_WEAK) > _MAX_TRACKER_CACHE_SIZE:
                _POLICY_TRACKERS_NON_WEAK.popitem(last=False)
        return cached


def _get_or_create_tracker_lock(owner: object) -> threading.Lock:
    owner_key = id(owner)
    with _CACHE_LOCK:
        tracker_lock = _TRACKER_LOCKS.get(owner_key)
        if tracker_lock is None:
            tracker_lock = threading.Lock()
            _TRACKER_LOCKS[owner_key] = tracker_lock
        _TRACKER_LOCKS.move_to_end(owner_key)
        while len(_TRACKER_LOCKS) > _MAX_TRACKER_CACHE_SIZE:
            _TRACKER_LOCKS.popitem(last=False)
        return tracker_lock


def _resolve_capture_probe_duration_ms(*, sample_ms: int, chunk_ms: object) -> int:
    """Keep the readiness probe short, deterministic, and at least one chunk."""

    chunk_ms_int = _coerce_int(chunk_ms, default=0, minimum=0)
    bounded_sample_ms = max(sample_ms, chunk_ms_int)
    bounded_sample_ms = min(bounded_sample_ms, 250)
    return max(chunk_ms_int, bounded_sample_ms, 1)


def _resolve_sample_ms(config: TwinrConfig) -> int:
    return _coerce_int(
        getattr(config, "proactive_audio_sample_ms", _DEFAULT_SAMPLE_MS),
        default=_DEFAULT_SAMPLE_MS,
        minimum=_MIN_SAMPLE_MS,
    )


def _resolve_capture_timeout_s(config: TwinrConfig, *, sample_ms: int) -> float:
    default_timeout = min(max((sample_ms / 1000.0) + 0.75, _MIN_CAPTURE_TIMEOUT_S), _MAX_CAPTURE_TIMEOUT_S)
    return _coerce_float(
        _first_config_value(
            config,
            "proactive_audio_snapshot_timeout_s",
            "proactive_audio_capture_timeout_s",
            "proactive_audio_observe_timeout_s",
            default=default_timeout,
        ),
        default=default_timeout,
        minimum=_MIN_CAPTURE_TIMEOUT_S,
        maximum=_MAX_CAPTURE_TIMEOUT_S,
    )


def _resolve_capture_lock_timeout_s(config: TwinrConfig, capture_timeout_s: float) -> float:
    return _coerce_float(
        _first_config_value(
            config,
            "proactive_audio_capture_lock_timeout_s",
            "proactive_audio_snapshot_lock_timeout_s",
            default=min(_DEFAULT_CAPTURE_LOCK_TIMEOUT_S, capture_timeout_s),
        ),
        default=min(_DEFAULT_CAPTURE_LOCK_TIMEOUT_S, capture_timeout_s),
        minimum=0.0,
        maximum=max(capture_timeout_s, 0.0),
    )


def _resolve_timeout_cooldown_s(config: TwinrConfig, capture_timeout_s: float) -> float:
    default_cooldown = max(_DEFAULT_TIMEOUT_COOLDOWN_S, capture_timeout_s)
    return _coerce_float(
        _first_config_value(
            config,
            "proactive_audio_timeout_cooldown_s",
            "proactive_audio_capture_timeout_cooldown_s",
            default=default_cooldown,
        ),
        default=default_cooldown,
        minimum=0.0,
    )


def _resolve_auxiliary_evidence_timeout_s(config: TwinrConfig) -> float:
    return _coerce_float(
        _first_config_value(
            config,
            "proactive_audio_auxiliary_evidence_timeout_s",
            "proactive_audio_snapshot_aux_timeout_s",
            default=_DEFAULT_AUXILIARY_EVIDENCE_TIMEOUT_S,
        ),
        default=_DEFAULT_AUXILIARY_EVIDENCE_TIMEOUT_S,
        minimum=0.0,
        maximum=2.0,
    )


def _resolve_assistant_output_active_predicate(config: TwinrConfig) -> Callable[[], bool]:
    """Best-effort hook into real assistant-output activity."""

    for attr_name in (
        "assistant_output_active_predicate",
        "proactive_assistant_output_active_predicate",
        "voice_output_active_predicate",
        "speaker_output_active_predicate",
    ):
        candidate = getattr(config, attr_name, None)
        if callable(candidate):
            return lambda candidate=candidate, attr_name=attr_name: _call_bool_predicate(candidate, attr_name)
        if isinstance(candidate, bool):
            return lambda candidate=candidate: candidate

    for attr_name in (
        "assistant_output_active",
        "proactive_assistant_output_active",
        "voice_output_active",
        "speaker_output_active",
    ):
        candidate = getattr(config, attr_name, None)
        if callable(candidate):
            return lambda candidate=candidate, attr_name=attr_name: _call_bool_predicate(candidate, attr_name)
        if isinstance(candidate, bool):
            return lambda candidate=candidate: candidate

    return lambda: False


def _call_bool_predicate(predicate: Callable[[], Any], attr_name: str) -> bool:
    try:
        return bool(predicate())
    except Exception:
        logger.exception("assistant-output predicate %s failed", attr_name)
        return False


def _attach_auxiliary_evidence(
    config: TwinrConfig,
    snapshot: ReSpeakerAudioPerceptionSnapshot,
) -> ReSpeakerAudioPerceptionSnapshot:
    """Optionally enrich the snapshot with bounded auxiliary model evidence.

    Expected hook contract:
        config.proactive_audio_auxiliary_snapshot_provider(snapshot=..., config=...)
        -> Mapping[str, scalar]
    """

    provider = _resolve_auxiliary_snapshot_provider(config)
    if provider is None:
        return snapshot

    timeout_s = _resolve_auxiliary_evidence_timeout_s(config)
    try:
        raw_evidence = _run_with_timeout(
            description="audio_perception_auxiliary_evidence",
            timeout_s=timeout_s,
            fn=lambda: provider(snapshot=snapshot, config=config),
        )
    except TimeoutError as exc:
        evidence = (("provider_timeout", "true"), ("provider_error", _format_optional_text(str(exc))),)
    except Exception as exc:
        logger.exception("observe_audio_perception_once auxiliary evidence failed")
        evidence = (("provider_failed", "true"), ("provider_error", _format_optional_text(str(exc))),)
    else:
        evidence = _normalize_auxiliary_evidence(raw_evidence)

    if not evidence:
        return snapshot
    return replace(snapshot, auxiliary_evidence=evidence)


def _resolve_auxiliary_snapshot_provider(
    config: TwinrConfig,
) -> Callable[..., Mapping[str, object] | object] | None:
    for attr_name in (
        "proactive_audio_auxiliary_snapshot_provider",
        "audio_perception_auxiliary_snapshot_provider",
        "respeaker_audio_auxiliary_snapshot_provider",
    ):
        candidate = getattr(config, attr_name, None)
        if callable(candidate):
            return cast(Callable[..., Mapping[str, object] | object], candidate)
    return None


def _normalize_auxiliary_evidence(raw_evidence: object) -> tuple[tuple[str, str], ...]:
    if raw_evidence is None:
        return ()
    if isinstance(raw_evidence, Mapping):
        items = raw_evidence.items()
    else:
        items = (("value", raw_evidence),)

    normalized: list[tuple[str, str]] = []
    for raw_key, raw_value in items:
        key = _normalize_metric_key(raw_key, fallback="value")
        value = _format_scalar(raw_value)
        normalized.append((key, value))
        if len(normalized) >= _MAX_AUXILIARY_EVIDENCE_ITEMS:
            break
    normalized.sort(key=lambda item: item[0])
    return tuple(normalized)


def _build_degraded_perception_snapshot(
    *,
    respeaker_targeted: bool,
    capture_deadline_s: float,
    capture_elapsed_ms: int,
    error_code: str,
    error_message: str | None,
    transport_reason: str,
    runtime_mode: str,
    capture_probe_ready: bool | None,
    collection_timed_out: bool = False,
) -> ReSpeakerAudioPerceptionSnapshot:
    audio_snapshot = _build_degraded_audio_snapshot(
        transport_reason=transport_reason,
        runtime_mode=runtime_mode,
    )
    audio_policy_snapshot = cast(
        ReSpeakerAudioPolicySnapshot,
        _SyntheticReSpeakerAudioPolicySnapshot(
            runtime_alert_code=error_code,
            runtime_alert_message=error_message,
        ),
    )
    perception_guard = derive_respeaker_audio_perception_guard(
        audio=audio_snapshot.observation,
        policy_snapshot=audio_policy_snapshot,
    )
    return ReSpeakerAudioPerceptionSnapshot(
        audio_snapshot=audio_snapshot,
        audio_policy_snapshot=audio_policy_snapshot,
        perception_guard=perception_guard,
        respeaker_targeted=respeaker_targeted,
        capture_probe=cast(
            AudioCaptureReadinessProbe | None,
            _SyntheticAudioCaptureReadinessProbe(ready=capture_probe_ready),
        ),
        capture_deadline_s=capture_deadline_s,
        capture_elapsed_ms=capture_elapsed_ms,
        capture_error_code=error_code,
        capture_error_message=error_message,
        collection_timed_out=collection_timed_out,
    )


def _build_degraded_audio_snapshot(
    *,
    transport_reason: str,
    runtime_mode: str,
) -> ProactiveAudioSnapshot:
    observation = cast(
        SocialAudioObservation,
        _SyntheticSocialAudioObservation(
            device_runtime_mode=runtime_mode,
            host_control_ready=False,
            transport_reason=transport_reason,
        ),
    )
    return cast(
        ProactiveAudioSnapshot,
        _SyntheticProactiveAudioSnapshot(
            observation=observation,
            signal_snapshot=None,
            sample=None,
        ),
    )


def _capture_device_key(*, config: TwinrConfig, respeaker_targeted: bool) -> str:
    return "|".join(
        (
            "respeaker" if respeaker_targeted else "ambient",
            _normalize_optional_text(getattr(config, "audio_input_device", None)) or "default",
            _normalize_optional_text(getattr(config, "proactive_audio_input_device", None)) or "default",
        )
    )


def _capture_cooldown_until(capture_key: str) -> float:
    with _CACHE_LOCK:
        cooldown_until = _CAPTURE_COOLDOWN_UNTIL.get(capture_key, 0.0)
        if cooldown_until <= time.monotonic():
            _CAPTURE_COOLDOWN_UNTIL.pop(capture_key, None)
            return 0.0
        return cooldown_until


def _set_capture_cooldown(capture_key: str, duration_s: float) -> None:
    if duration_s <= 0.0:
        return
    with _CACHE_LOCK:
        _CAPTURE_COOLDOWN_UNTIL[capture_key] = time.monotonic() + duration_s


def _get_capture_lock(capture_key: str) -> threading.Lock:
    with _CACHE_LOCK:
        capture_lock = _CAPTURE_LOCKS.get(capture_key)
        if capture_lock is None:
            capture_lock = threading.Lock()
            _CAPTURE_LOCKS[capture_key] = capture_lock
        _CAPTURE_LOCKS.move_to_end(capture_key)
        while len(_CAPTURE_LOCKS) > _MAX_CAPTURE_LOCK_CACHE_SIZE:
            _CAPTURE_LOCKS.popitem(last=False)
        return capture_lock


def _run_with_timeout(*, description: str, timeout_s: float, fn: Callable[[], Any]) -> Any:
    if timeout_s <= 0.0:
        return fn()

    queue: SimpleQueue[tuple[bool, Any]] = SimpleQueue()

    def _runner() -> None:
        try:
            queue.put((True, fn()))
        except BaseException as exc:
            queue.put((False, exc))

    worker = threading.Thread(
        target=_runner,
        name=f"{__name__}:{description}",
        daemon=True,
    )
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        raise TimeoutError(f"{description} timed out after {timeout_s:.3f}s")
    success, payload = queue.get()
    if success:
        return payload
    raise cast(BaseException, payload)


def _close_quietly(resource: object | None) -> None:
    close_method = cast(Callable[[], None] | None, getattr(resource, "close", None))
    if close_method is None:
        return
    try:
        close_method()
    except Exception:
        logger.debug("suppressed close() failure for %r", resource, exc_info=True)


def _format_optional_bool(value: bool | None) -> str:
    """Render one optional boolean as CLI-safe text."""

    if value is None:
        return "unknown"
    return "true" if value else "false"


def _format_optional_float(value: float | None) -> str:
    """Render one optional float as CLI-safe text."""

    if value is None:
        return "unknown"
    return f"{float(value):.3f}"


def _format_optional_int(value: int | None) -> str:
    """Render one optional integer as CLI-safe text."""

    if value is None:
        return "unknown"
    return str(int(value))


def _format_optional_text(value: object) -> str:
    """Render one optional scalar as one trimmed single-line string."""

    normalized = _normalize_optional_text(value)
    return normalized or "unknown"


def _format_scalar(value: object) -> str:
    """Render one scalar for auxiliary evidence lines."""

    if isinstance(value, bool):
        return _format_optional_bool(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return _format_optional_int(value)
    if isinstance(value, float):
        return _format_optional_float(value)
    return _format_optional_text(value)


def _normalize_optional_text(value: object) -> str | None:
    """Normalize one optional text field to a trimmed control-safe single line."""

    if value is None:
        return None
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    elif isinstance(value, str):
        text = value
    else:
        text = str(value)

    # Remove non-printable control characters (including ANSI escapes) and
    # collapse whitespace so shell/CLI diagnostics stay stable and parseable.
    sanitized = "".join(ch if ch.isprintable() else " " for ch in text)
    normalized = " ".join(sanitized.split()).strip()
    if not normalized:
        return None
    if len(normalized) > _MAX_TEXT_FIELD_CHARS:
        return normalized[: _MAX_TEXT_FIELD_CHARS - 1].rstrip() + "…"
    return normalized


def _normalize_metric_key(value: object, *, fallback: str) -> str:
    normalized = _normalize_optional_text(value)
    if normalized is None:
        return fallback
    pieces: list[str] = []
    previous_was_sep = False
    for char in normalized.lower():
        if char.isascii() and char.isalnum():
            pieces.append(char)
            previous_was_sep = False
            continue
        if previous_was_sep:
            continue
        pieces.append("_")
        previous_was_sep = True
    metric_key = "".join(pieces).strip("_")
    return metric_key or fallback


def _coerce_int(value: object, *, default: int, minimum: int | None = None) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        coerced = default
    if minimum is not None:
        coerced = max(coerced, minimum)
    return coerced


def _coerce_float(
    value: object,
    *,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        coerced = default
    if minimum is not None:
        coerced = max(coerced, minimum)
    if maximum is not None:
        coerced = min(coerced, maximum)
    return coerced


def _first_config_value(config: TwinrConfig, *names: str, default: object) -> object:
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    return default


__all__ = [
    "ReSpeakerAudioPerceptionGuardSnapshot",
    "ReSpeakerAudioPerceptionSnapshot",
    "derive_respeaker_audio_perception_guard",
    "observe_audio_perception_once",
    "render_audio_perception_snapshot_lines",
]