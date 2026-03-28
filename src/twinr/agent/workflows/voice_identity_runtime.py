# CHANGELOG: 2026-03-28
# BUG-1: Do not advance orchestrator sync revision before notify_identity_profiles(...) succeeds;
#        failed notifications now retry instead of silently leaving the live gateway stale.
# BUG-2: Parse env-backed config values safely; string values such as "false", "16000", or
#        malformed numerics no longer enable policies accidentally or crash passive-update logic.
# BUG-3: Deduplicate retried passive updates for the same PCM payload and continue adaptive
#        enrollment even when runtime persistence emits an error.
# SEC-1: Reject malformed or oversized PCM payloads before assessment/enrollment to reduce
#        practical CPU/RAM DoS risk on Raspberry Pi 4 deployments.
# SEC-2: Block passive adaptation on positive spoof/replay/synthetic signals and redact emitted
#        user identifiers by default to reduce biometric privacy leakage.
# IMP-1: Add concurrency-safe lazy manager initialization and resilient sync keys using
#        revision plus profile fingerprint fallback.
# IMP-2: Upgrade passive-update policy with cooldowns, quality gates, confidence-margin support,
#        optional exclusive-speaker/liveness signals, and richer trace metadata.

"""Keep household voice identity sync and passive updates out of runtime loops."""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Mapping
from typing import Any

from twinr.hardware.household_identity import HouseholdIdentityManager
from twinr.hardware.household_voice_identity import HouseholdVoiceAssessment
from twinr.orchestrator.voice_contracts import (
    OrchestratorVoiceIdentityProfile,
    OrchestratorVoiceIdentityProfilesEvent,
)

_PCM_SAMPLE_WIDTH_BYTES = 2
_MAX_DEFAULT_PCM_DURATION_MS = 30_000
_MAX_DEFAULT_PCM_BYTES = 4 * 1024 * 1024
_PASSIVE_UPDATE_SPEAKER_ASSOCIATION_MIN_CONFIDENCE = 0.82
_PASSIVE_UPDATE_ALLOWED_STATUSES = frozenset({"likely_user", "known_other_user"})
_PASSIVE_UPDATE_DEFAULT_COOLDOWN_SEC = 300.0
_PASSIVE_UPDATE_DEFAULT_DEDUPE_WINDOW_SEC = 900.0
_PASSIVE_UPDATE_DEFAULT_MIN_MARGIN = 0.08
_PASSIVE_UPDATE_DEFAULT_MIN_SNR_DB = 8.0
_PASSIVE_UPDATE_DEFAULT_MIN_SPEECH_RATIO = 0.55
_PASSIVE_UPDATE_DEFAULT_MAX_CLIPPING_RATIO = 0.10
_PASSIVE_UPDATE_DEFAULT_MAX_SILENCE_RATIO = 0.45
_RECENT_AUDIO_CACHE_SIZE = 64

_STATE_CREATE_LOCK = threading.Lock()


def _coerce_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    return {}


def _coerce_optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    return None


def _coerce_float(
    value: object,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str, bytes, bytearray)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if minimum is not None and parsed < minimum:
        return None
    if maximum is not None and parsed > maximum:
        return None
    return parsed


def _coerce_optional_float(value: object) -> float | None:
    return _coerce_float(value, minimum=0.0, maximum=1.0)


def _coerce_int(
    value: object,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str, bytes, bytearray)):
        return None
    try:
        if isinstance(value, int):
            parsed = value
        elif isinstance(value, float):
            if not value.is_integer():
                return None
            parsed = int(value)
        else:
            text = str(value).strip()
            if not text:
                return None
            parsed_float = float(text)
            if not parsed_float.is_integer():
                return None
            parsed = int(parsed_float)
    except (TypeError, ValueError):
        return None
    if minimum is not None and parsed < minimum:
        return None
    if maximum is not None and parsed > maximum:
        return None
    return parsed


def _coerce_optional_int(value: object) -> int | None:
    return _coerce_int(value)


def _config_bool(config: Any, name: str, default: bool) -> bool:
    parsed = _coerce_optional_bool(getattr(config, name, default))
    if parsed is None:
        return default
    return parsed


def _config_float(
    config: Any,
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    parsed = _coerce_float(getattr(config, name, default), minimum=minimum, maximum=maximum)
    if parsed is None:
        return default
    return parsed


def _config_int(
    config: Any,
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    parsed = _coerce_int(getattr(config, name, default), minimum=minimum, maximum=maximum)
    if parsed is None:
        return default
    return parsed


def _safe_emit(loop: Any, text: str) -> None:
    emitter = getattr(loop, "_try_emit", None)
    if callable(emitter):
        emitter(text)
        return
    emitter = getattr(loop, "emit", None)
    if callable(emitter):
        emitter(text)


def _emit_kv(loop: Any, key: str, value: object) -> None:
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    if not text:
        return
    _safe_emit(loop, f"{key}={text}")


def _trace_event(loop: Any, name: str, *, kind: str, details: dict[str, object]) -> None:
    tracer = getattr(loop, "_trace_event", None)
    if callable(tracer):
        tracer(name, kind=kind, details=details)


def _safe_error_text(loop: Any, exc: Exception) -> str:
    formatter = getattr(loop, "_safe_error_text", None)
    if callable(formatter):
        return str(formatter(exc))
    return type(exc).__name__


def _loop_attr(loop: Any, name: str, factory: Any) -> Any:
    value = getattr(loop, name, None)
    if value is not None:
        return value
    with _STATE_CREATE_LOCK:
        value = getattr(loop, name, None)
        if value is None:
            value = factory()
            setattr(loop, name, value)
    return value


def _manager_lock(loop: Any) -> threading.RLock:
    return _loop_attr(loop, "_household_voice_identity_manager_lock", threading.RLock)


def _sync_lock(loop: Any) -> threading.RLock:
    return _loop_attr(loop, "_household_voice_identity_sync_lock", threading.RLock)


def _passive_update_lock(loop: Any) -> threading.RLock:
    return _loop_attr(loop, "_household_voice_identity_passive_update_lock", threading.RLock)


def _passive_update_timestamps(loop: Any) -> dict[str, float]:
    return _loop_attr(loop, "_voice_profile_passive_update_timestamps", dict)


def _passive_update_recent_audio(loop: Any) -> dict[str, float]:
    return _loop_attr(loop, "_voice_profile_passive_update_recent_audio", dict)


def _redact_identifier(value: object) -> str:
    text = str(value).strip()
    if not text:
        return ""
    digest = hashlib.blake2s(text.encode("utf-8"), digest_size=6).hexdigest()
    return f"anon_{digest}"


# BREAKING: plaintext user identifiers are no longer emitted by default.
# Set config.voice_profile_emit_plain_user_id=True to restore legacy plaintext emission.
def _display_user_id(loop: Any, user_id: object) -> str:
    if user_id is None:
        return ""
    config = getattr(loop, "config", None)
    if config is not None and _config_bool(config, "voice_profile_emit_plain_user_id", False):
        return str(user_id)
    return _redact_identifier(user_id)


def _emit_user_id(loop: Any, key: str, user_id: object) -> None:
    rendered = _display_user_id(loop, user_id)
    if rendered:
        _emit_kv(loop, key, rendered)


def _assessment_attr(assessment: object, *names: str) -> object:
    mapping = _coerce_mapping(assessment)
    for name in names:
        if name in mapping:
            return mapping[name]
    for name in names:
        if hasattr(assessment, name):
            return getattr(assessment, name)
    return None


def _assessment_optional_float(assessment: object, *names: str) -> float | None:
    return _coerce_float(_assessment_attr(assessment, *names))


def _assessment_optional_bool(assessment: object, *names: str) -> bool | None:
    return _coerce_optional_bool(_assessment_attr(assessment, *names))


def _normalize_percent_like(value: float) -> float:
    if value > 1.0 and value <= 100.0:
        return value / 100.0
    return value


def _assessment_confidence_margin(assessment: object) -> float | None:
    direct = _assessment_optional_float(
        assessment,
        "confidence_margin",
        "margin",
        "score_margin",
        "best_margin",
        "top1_top2_margin",
    )
    if direct is not None:
        return _normalize_percent_like(direct)

    for field_name in ("voiceprint_confidences", "confidence_by_user", "candidate_confidences"):
        values = _coerce_mapping(_assessment_attr(assessment, field_name))
        numeric_scores: list[float] = []
        for raw in values.values():
            score = _coerce_float(raw)
            if score is None:
                continue
            numeric_scores.append(_normalize_percent_like(score))
        if len(numeric_scores) >= 2:
            numeric_scores.sort(reverse=True)
            margin = numeric_scores[0] - numeric_scores[1]
            if margin >= 0.0:
                return margin
    return None


def _household_identity_manager(loop: Any) -> HouseholdIdentityManager | None:
    manager = getattr(loop, "household_identity_manager", None)
    if manager is not None:
        return manager
    camera = getattr(loop, "camera", None)
    if camera is None:
        return None
    with _manager_lock(loop):
        manager = getattr(loop, "household_identity_manager", None)
        if manager is not None:
            return manager
        try:
            manager = HouseholdIdentityManager.from_config(
                loop.config,
                camera=camera,
                camera_lock=getattr(loop, "_camera_lock", None),
            )
        except Exception as exc:
            _emit_kv(loop, "household_voice_profile_error", _safe_error_text(loop, exc))
            return None
        setattr(loop, "household_identity_manager", manager)
        return manager


def _stable_profile_fingerprint(
    event: OrchestratorVoiceIdentityProfilesEvent,
) -> str:
    digest = hashlib.blake2s(digest_size=16)
    digest.update(str(getattr(event, "revision", None)).encode("utf-8", errors="ignore"))
    for profile in getattr(event, "profiles", ()) or ():
        for value in (
            getattr(profile, "user_id", None),
            getattr(profile, "display_name", None),
            getattr(profile, "primary_user", None),
            getattr(profile, "sample_count", None),
            getattr(profile, "average_duration_ms", None),
            getattr(profile, "updated_at", None),
        ):
            digest.update(repr(value).encode("utf-8", errors="ignore"))
            digest.update(b"\x1f")
        digest.update(b"\x1e")
    return digest.hexdigest()


def _pcm_bytes_per_frame(channels: int) -> int:
    return max(1, channels) * _PCM_SAMPLE_WIDTH_BYTES


def _pcm_duration_ms(audio_pcm: bytes, *, sample_rate: int, channels: int) -> int | None:
    if sample_rate <= 0 or channels <= 0:
        return None
    frame_bytes = _pcm_bytes_per_frame(channels)
    if frame_bytes <= 0 or len(audio_pcm) % frame_bytes != 0:
        return None
    frame_count = len(audio_pcm) // frame_bytes
    return int(round((frame_count / sample_rate) * 1000.0))


def _normalize_audio_pcm(loop: Any, audio_pcm: bytes) -> tuple[bytes, int, int, int] | None:
    if isinstance(audio_pcm, memoryview):
        audio_bytes = audio_pcm.tobytes()
    elif isinstance(audio_pcm, bytearray):
        audio_bytes = bytes(audio_pcm)
    elif isinstance(audio_pcm, bytes):
        audio_bytes = audio_pcm
    else:
        _emit_kv(loop, "household_voice_profile_error", "invalid_audio_type")
        return None

    config = getattr(loop, "config", None)
    if config is None:
        _emit_kv(loop, "household_voice_profile_error", "missing_config")
        return None

    sample_rate = _config_int(config, "openai_realtime_input_sample_rate", 16_000, minimum=1)
    channels = _config_int(config, "audio_channels", 1, minimum=1, maximum=8)
    duration_ms = _pcm_duration_ms(audio_bytes, sample_rate=sample_rate, channels=channels)
    if duration_ms is None:
        _emit_kv(loop, "household_voice_profile_error", "invalid_pcm_alignment")
        return None
    if duration_ms <= 0:
        _emit_kv(loop, "household_voice_profile_error", "empty_pcm")
        return None

    # BREAKING: oversized PCM payloads are rejected before assessment/enrollment.
    # Increase config.voice_profile_max_assessment_bytes only if you explicitly want larger turns.
    max_bytes = _config_int(
        config,
        "voice_profile_max_assessment_bytes",
        _MAX_DEFAULT_PCM_BYTES,
        minimum=1024,
    )
    if len(audio_bytes) > max_bytes:
        _emit_kv(loop, "household_voice_profile_error", "pcm_too_many_bytes")
        _trace_event(
            loop,
            "voice_profile_pcm_rejected",
            kind="security",
            details={"bytes": len(audio_bytes), "byte_limit": max_bytes},
        )
        return None

    max_duration_ms = _config_int(
        config,
        "voice_profile_max_assessment_duration_ms",
        _MAX_DEFAULT_PCM_DURATION_MS,
        minimum=500,
    )
    if duration_ms > max_duration_ms:
        _emit_kv(loop, "household_voice_profile_error", "pcm_too_large")
        _trace_event(
            loop,
            "voice_profile_pcm_rejected",
            kind="security",
            details={"duration_ms": duration_ms, "limit_ms": max_duration_ms},
        )
        return None

    return audio_bytes, sample_rate, channels, duration_ms


def build_voice_identity_profiles_event(
    loop: Any,
) -> OrchestratorVoiceIdentityProfilesEvent | None:
    """Return the current read-only household voice profile snapshot."""

    manager = _household_identity_manager(loop)
    if manager is None:
        return None

    try:
        voice_monitor = manager.voice_monitor
        profiles = voice_monitor.voice_profiles()
        revision = voice_monitor.profile_revision()
    except Exception as exc:
        _emit_kv(loop, "household_voice_profile_error", _safe_error_text(loop, exc))
        return None

    return OrchestratorVoiceIdentityProfilesEvent(
        revision=revision,
        profiles=tuple(
            OrchestratorVoiceIdentityProfile(
                user_id=profile.user_id,
                display_name=profile.display_name,
                primary_user=profile.primary_user,
                embedding=profile.embedding,
                sample_count=profile.sample_count,
                average_duration_ms=profile.average_duration_ms,
                updated_at=profile.updated_at,
            )
            for profile in profiles
        ),
    )


def sync_voice_orchestrator_identity_profiles(
    loop: Any,
    *,
    force: bool = False,
) -> None:
    """Push changed household voice profiles into the live voice gateway."""

    voice_orchestrator = getattr(loop, "voice_orchestrator", None)
    if voice_orchestrator is None:
        return

    notify = getattr(voice_orchestrator, "notify_identity_profiles", None)
    if not callable(notify):
        return

    with _sync_lock(loop):
        event = build_voice_identity_profiles_event(loop)
        if event is None:
            return

        sync_key = (getattr(event, "revision", None), _stable_profile_fingerprint(event))
        last_sync_key = getattr(loop, "_last_voice_identity_profile_sync_key", None)
        if not force and sync_key == last_sync_key:
            return

        try:
            notify(event)
        except Exception as exc:
            _emit_kv(loop, "household_voice_profile_sync_error", _safe_error_text(loop, exc))
            _trace_event(
                loop,
                "voice_identity_profiles_sync_failed",
                kind="error",
                details={
                    "revision": getattr(event, "revision", None),
                    "profile_count": len(getattr(event, "profiles", ()) or ()),
                    "error": _safe_error_text(loop, exc),
                },
            )
            return

        setattr(loop, "_last_voice_identity_profile_sync_key", sync_key)
        setattr(loop, "_last_voice_identity_profile_revision", getattr(event, "revision", None))
        setattr(loop, "_last_voice_identity_profile_count", len(getattr(event, "profiles", ()) or ()))
        _trace_event(
            loop,
            "voice_identity_profiles_synced",
            kind="mutation",
            details={
                "revision": getattr(event, "revision", None),
                "profile_count": len(getattr(event, "profiles", ()) or ()),
                "fingerprint": sync_key[1],
            },
        )


def update_household_voice_assessment_from_pcm(
    loop: Any,
    audio_pcm: bytes,
    *,
    source: str,
) -> HouseholdVoiceAssessment | None:
    """Assess one PCM sample against household profiles and maybe update them."""

    manager = _household_identity_manager(loop)
    if manager is None:
        return None

    normalized = _normalize_audio_pcm(loop, audio_pcm)
    if normalized is None:
        return None
    audio_bytes, sample_rate, channels, duration_ms = normalized

    try:
        assessment = manager.assess_voice(
            audio_bytes,
            sample_rate=sample_rate,
            channels=channels,
        )
    except Exception as exc:
        _emit_kv(loop, "household_voice_profile_error", _safe_error_text(loop, exc))
        return None

    if getattr(assessment, "status", None) == "not_enrolled":
        return assessment

    if bool(getattr(assessment, "should_persist", False)):
        try:
            loop.runtime.update_user_voice_assessment(
                status=assessment.status,
                confidence=assessment.confidence,
                checked_at=assessment.checked_at,
                user_id=assessment.matched_user_id,
                user_display_name=assessment.matched_user_display_name,
                match_source="household_voice_identity",
            )
        except Exception as exc:
            _emit_kv(loop, "voice_profile_persist_error", _safe_error_text(loop, exc))

        _emit_kv(loop, "voice_profile_status", getattr(assessment, "status", "unknown"))
        confidence = _coerce_optional_float(getattr(assessment, "confidence", None))
        if confidence is not None:
            _emit_kv(loop, "voice_profile_confidence", f"{confidence:.2f}")
        _emit_user_id(loop, "voice_profile_user_id", getattr(assessment, "matched_user_id", None))

    if _maybe_apply_passive_voice_update(
        loop,
        manager=manager,
        audio_pcm=audio_bytes,
        assessment=assessment,
        source=source,
        sample_rate=sample_rate,
        channels=channels,
        duration_ms=duration_ms,
    ):
        sync_voice_orchestrator_identity_profiles(loop, force=True)
    return assessment


def _quality_fact_sources(loop: Any) -> tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
    facts = _coerce_mapping(getattr(loop, "_latest_sensor_observation_facts", None))
    speaker_association = _coerce_mapping(facts.get("speaker_association"))
    camera = _coerce_mapping(facts.get("camera"))
    audio_policy = _coerce_mapping(facts.get("audio_policy"))
    audio_quality = _coerce_mapping(facts.get("audio_quality"))
    return speaker_association, camera, audio_policy, audio_quality


def _assessment_or_quality_float(
    assessment: object,
    audio_quality: Mapping[str, object],
    *names: str,
) -> float | None:
    value = _assessment_optional_float(assessment, *names)
    if value is not None:
        return value
    for name in names:
        value = _coerce_float(audio_quality.get(name))
        if value is not None:
            return value
    return None


def _assessment_or_quality_bool(
    assessment: object,
    audio_quality: Mapping[str, object],
    *names: str,
) -> bool | None:
    value = _assessment_optional_bool(assessment, *names)
    if value is not None:
        return value
    for name in names:
        value = _coerce_optional_bool(audio_quality.get(name))
        if value is not None:
            return value
    return None


def _spoof_risk_reason(
    assessment: object,
    *,
    facts: Mapping[str, object],
) -> str | None:
    positive_assessment_flags = (
        ("spoof_likely", "assessment_spoof_likely"),
        ("replay_likely", "assessment_replay_likely"),
        ("synthetic_likely", "assessment_synthetic_likely"),
        ("deepfake_likely", "assessment_deepfake_likely"),
        ("fake_audio_likely", "assessment_fake_audio_likely"),
    )
    for field_name, reason in positive_assessment_flags:
        if _assessment_optional_bool(assessment, field_name) is True:
            return reason

    negative_assessment_flags = (
        ("liveness_passed", "assessment_liveness_failed"),
        ("is_live", "assessment_not_live"),
        ("anti_spoof_passed", "assessment_anti_spoof_failed"),
    )
    for field_name, reason in negative_assessment_flags:
        value = _assessment_optional_bool(assessment, field_name)
        if value is False:
            return reason

    audio_policy = _coerce_mapping(facts.get("audio_policy"))
    anti_spoof = _coerce_mapping(facts.get("anti_spoof"))
    liveness = _coerce_mapping(facts.get("liveness"))

    if _coerce_optional_bool(audio_policy.get("replay_attack_likely")) is True:
        return "audio_policy_replay_attack_likely"
    if _coerce_optional_bool(audio_policy.get("synthetic_speech_likely")) is True:
        return "audio_policy_synthetic_speech_likely"
    if _coerce_optional_bool(audio_policy.get("spoof_likely")) is True:
        return "audio_policy_spoof_likely"
    if _coerce_optional_bool(anti_spoof.get("spoof_likely")) is True:
        return "anti_spoof_spoof_likely"
    if _coerce_optional_bool(anti_spoof.get("passed")) is False:
        return "anti_spoof_failed"
    if _coerce_optional_bool(liveness.get("passed")) is False:
        return "liveness_failed"
    if _coerce_optional_bool(liveness.get("is_live")) is False:
        return "liveness_not_live"
    return None


def _trace_passive_update_skip(
    loop: Any,
    *,
    source: str,
    assessment: HouseholdVoiceAssessment,
    reason: str,
    confidence: float | None = None,
    duration_ms: int | None = None,
) -> None:
    _trace_event(
        loop,
        "voice_profile_passive_update_skipped",
        kind="decision",
        details={
            "source": source,
            "reason": reason,
            "user_id": _display_user_id(loop, getattr(assessment, "matched_user_id", None)),
            "status": getattr(assessment, "status", None),
            "confidence": confidence,
            "duration_ms": duration_ms,
        },
    )


def _maybe_apply_passive_voice_update(
    loop: Any,
    *,
    manager: HouseholdIdentityManager,
    audio_pcm: bytes,
    assessment: HouseholdVoiceAssessment,
    source: str,
    sample_rate: int,
    channels: int,
    duration_ms: int,
) -> bool:
    """Merge a clean accepted turn back into the enrolled profile when safe."""

    config = getattr(loop, "config", None)
    if config is None:
        return False
    if not _config_bool(config, "voice_profile_passive_update_enabled", True):
        return False

    if getattr(assessment, "status", None) not in _PASSIVE_UPDATE_ALLOWED_STATUSES:
        return False
    if getattr(assessment, "matched_user_id", None) is None:
        return False

    confidence = _coerce_optional_float(getattr(assessment, "confidence", None))
    min_confidence = _config_float(
        config,
        "voice_profile_passive_update_min_confidence",
        0.86,
        minimum=0.0,
        maximum=1.0,
    )
    if confidence is None or confidence < min_confidence:
        _trace_passive_update_skip(
            loop,
            source=source,
            assessment=assessment,
            reason="low_confidence",
            confidence=confidence,
            duration_ms=duration_ms,
        )
        return False

    min_duration_ms = _config_int(
        config,
        "voice_profile_passive_update_min_duration_ms",
        2500,
        minimum=200,
    )
    max_duration_ms = _config_int(
        config,
        "voice_profile_passive_update_max_duration_ms",
        _MAX_DEFAULT_PCM_DURATION_MS,
        minimum=min_duration_ms,
    )
    if duration_ms < min_duration_ms:
        _trace_passive_update_skip(
            loop,
            source=source,
            assessment=assessment,
            reason="duration_too_short",
            confidence=confidence,
            duration_ms=duration_ms,
        )
        return False
    if duration_ms > max_duration_ms:
        _trace_passive_update_skip(
            loop,
            source=source,
            assessment=assessment,
            reason="duration_too_long",
            confidence=confidence,
            duration_ms=duration_ms,
        )
        return False

    try:
        summary = manager.voice_monitor.summary(assessment.matched_user_id)
    except Exception as exc:
        _emit_kv(loop, "voice_profile_passive_update_error", _safe_error_text(loop, exc))
        return False
    if summary is None or not bool(getattr(summary, "enrolled", False)):
        return False

    speaker_association, camera, audio_policy, audio_quality = _quality_fact_sources(loop)

    if _coerce_optional_bool(speaker_association.get("associated")) is not True:
        _trace_passive_update_skip(
            loop,
            source=source,
            assessment=assessment,
            reason="speaker_not_associated",
            confidence=confidence,
            duration_ms=duration_ms,
        )
        return False

    min_assoc_confidence = _config_float(
        config,
        "voice_profile_passive_update_speaker_association_min_confidence",
        _PASSIVE_UPDATE_SPEAKER_ASSOCIATION_MIN_CONFIDENCE,
        minimum=0.0,
        maximum=1.0,
    )
    association_confidence = _coerce_optional_float(speaker_association.get("confidence")) or 0.0
    if association_confidence < min_assoc_confidence:
        _trace_passive_update_skip(
            loop,
            source=source,
            assessment=assessment,
            reason="speaker_association_low_confidence",
            confidence=confidence,
            duration_ms=duration_ms,
        )
        return False

    if _config_bool(config, "voice_profile_passive_update_require_visible_person", True):
        if _coerce_optional_bool(camera.get("person_visible")) is not True:
            _trace_passive_update_skip(
                loop,
                source=source,
                assessment=assessment,
                reason="no_visible_person",
                confidence=confidence,
                duration_ms=duration_ms,
            )
            return False

    if _config_bool(config, "voice_profile_passive_update_require_single_visible_person", True):
        if _coerce_optional_int(camera.get("person_count")) != 1:
            _trace_passive_update_skip(
                loop,
                source=source,
                assessment=assessment,
                reason="visible_person_count_not_one",
                confidence=confidence,
                duration_ms=duration_ms,
            )
            return False

    if _coerce_optional_bool(audio_policy.get("background_media_likely")) is True:
        _trace_passive_update_skip(
            loop,
            source=source,
            assessment=assessment,
            reason="background_media_likely",
            confidence=confidence,
            duration_ms=duration_ms,
        )
        return False

    if _coerce_optional_bool(audio_policy.get("speech_overlap_likely")) is True:
        _trace_passive_update_skip(
            loop,
            source=source,
            assessment=assessment,
            reason="speech_overlap_likely",
            confidence=confidence,
            duration_ms=duration_ms,
        )
        return False

    if _config_bool(config, "voice_profile_passive_update_require_exclusive_speaker", False):
        exclusive_flag = _coerce_optional_bool(
            speaker_association.get("exclusive")
        )
        if exclusive_flag is None:
            exclusive_flag = _coerce_optional_bool(audio_policy.get("exclusive_speaker_segment"))
        if exclusive_flag is not True:
            _trace_passive_update_skip(
                loop,
                source=source,
                assessment=assessment,
                reason="exclusive_speaker_required",
                confidence=confidence,
                duration_ms=duration_ms,
            )
            return False

    spoof_reason = _spoof_risk_reason(
        assessment,
        facts=_coerce_mapping(getattr(loop, "_latest_sensor_observation_facts", None)),
    )
    if spoof_reason is not None:
        _trace_passive_update_skip(
            loop,
            source=source,
            assessment=assessment,
            reason=spoof_reason,
            confidence=confidence,
            duration_ms=duration_ms,
        )
        return False

    min_margin = _config_float(
        config,
        "voice_profile_passive_update_min_confidence_margin",
        _PASSIVE_UPDATE_DEFAULT_MIN_MARGIN,
        minimum=0.0,
        maximum=1.0,
    )
    margin = _assessment_confidence_margin(assessment)
    if margin is not None and margin < min_margin:
        _trace_passive_update_skip(
            loop,
            source=source,
            assessment=assessment,
            reason="confidence_margin_too_small",
            confidence=confidence,
            duration_ms=duration_ms,
        )
        return False

    min_snr_db = _config_float(
        config,
        "voice_profile_passive_update_min_snr_db",
        _PASSIVE_UPDATE_DEFAULT_MIN_SNR_DB,
    )
    snr_db = _assessment_or_quality_float(assessment, audio_quality, "snr_db")
    if snr_db is not None and snr_db < min_snr_db:
        _trace_passive_update_skip(
            loop,
            source=source,
            assessment=assessment,
            reason="snr_too_low",
            confidence=confidence,
            duration_ms=duration_ms,
        )
        return False

    min_speech_ratio = _config_float(
        config,
        "voice_profile_passive_update_min_speech_ratio",
        _PASSIVE_UPDATE_DEFAULT_MIN_SPEECH_RATIO,
        minimum=0.0,
        maximum=1.0,
    )
    speech_ratio = _assessment_or_quality_float(
        assessment,
        audio_quality,
        "speech_ratio",
        "voiced_ratio",
        "vad_ratio",
    )
    if speech_ratio is not None and speech_ratio < min_speech_ratio:
        _trace_passive_update_skip(
            loop,
            source=source,
            assessment=assessment,
            reason="speech_ratio_too_low",
            confidence=confidence,
            duration_ms=duration_ms,
        )
        return False

    max_clipping_ratio = _config_float(
        config,
        "voice_profile_passive_update_max_clipping_ratio",
        _PASSIVE_UPDATE_DEFAULT_MAX_CLIPPING_RATIO,
        minimum=0.0,
        maximum=1.0,
    )
    clipping_ratio = _assessment_or_quality_float(
        assessment,
        audio_quality,
        "clipping_ratio",
        "clip_ratio",
    )
    if clipping_ratio is not None and clipping_ratio > max_clipping_ratio:
        _trace_passive_update_skip(
            loop,
            source=source,
            assessment=assessment,
            reason="clipping_ratio_too_high",
            confidence=confidence,
            duration_ms=duration_ms,
        )
        return False

    max_silence_ratio = _config_float(
        config,
        "voice_profile_passive_update_max_silence_ratio",
        _PASSIVE_UPDATE_DEFAULT_MAX_SILENCE_RATIO,
        minimum=0.0,
        maximum=1.0,
    )
    silence_ratio = _assessment_or_quality_float(
        assessment,
        audio_quality,
        "silence_ratio",
        "nonspeech_ratio",
    )
    if silence_ratio is not None and silence_ratio > max_silence_ratio:
        _trace_passive_update_skip(
            loop,
            source=source,
            assessment=assessment,
            reason="silence_ratio_too_high",
            confidence=confidence,
            duration_ms=duration_ms,
        )
        return False

    with _passive_update_lock(loop):
        now = time.monotonic()
        # BREAKING: passive updates are cooldown-gated and duplicate-audio deduped by default.
        # Set cooldown/dedupe windows to 0.0 to restore legacy "update every eligible turn" behavior.
        cooldown_sec = _config_float(
            config,
            "voice_profile_passive_update_cooldown_sec",
            _PASSIVE_UPDATE_DEFAULT_COOLDOWN_SEC,
            minimum=0.0,
        )
        dedupe_window_sec = _config_float(
            config,
            "voice_profile_passive_update_dedupe_window_sec",
            _PASSIVE_UPDATE_DEFAULT_DEDUPE_WINDOW_SEC,
            minimum=0.0,
        )

        timestamps = _passive_update_timestamps(loop)
        user_key = str(assessment.matched_user_id)
        last_update_at = timestamps.get(user_key)
        if (
            cooldown_sec > 0.0
            and last_update_at is not None
            and (now - last_update_at) < cooldown_sec
        ):
            _trace_passive_update_skip(
                loop,
                source=source,
                assessment=assessment,
                reason="cooldown_active",
                confidence=confidence,
                duration_ms=duration_ms,
            )
            return False

        audio_digest = hashlib.blake2s(audio_pcm, digest_size=16).hexdigest()
        recent_audio = _passive_update_recent_audio(loop)
        expired_keys = [key for key, expiry in recent_audio.items() if expiry <= now]
        for key in expired_keys:
            recent_audio.pop(key, None)
        if len(recent_audio) > _RECENT_AUDIO_CACHE_SIZE:
            oldest_keys = sorted(recent_audio, key=recent_audio.get)[: len(recent_audio) - _RECENT_AUDIO_CACHE_SIZE]
            for key in oldest_keys:
                recent_audio.pop(key, None)

        dedupe_key = f"{user_key}:{audio_digest}"
        if dedupe_window_sec > 0.0 and dedupe_key in recent_audio:
            _trace_passive_update_skip(
                loop,
                source=source,
                assessment=assessment,
                reason="duplicate_audio_within_window",
                confidence=confidence,
                duration_ms=duration_ms,
            )
            return False

        try:
            updated_summary, _member = manager.enroll_voice(
                audio_pcm,
                sample_rate=sample_rate,
                channels=channels,
                user_id=assessment.matched_user_id,
                display_name=assessment.matched_user_display_name,
            )
        except Exception as exc:
            _emit_kv(loop, "voice_profile_passive_update_error", _safe_error_text(loop, exc))
            return False

        timestamps[user_key] = now
        if dedupe_window_sec > 0.0:
            recent_audio[dedupe_key] = now + dedupe_window_sec

    _emit_kv(loop, "voice_profile_passive_update", "applied")
    _emit_user_id(loop, "voice_profile_passive_update_user_id", assessment.matched_user_id)
    _trace_event(
        loop,
        "voice_profile_passive_update_applied",
        kind="mutation",
        details={
            "source": source,
            "user_id": _display_user_id(loop, assessment.matched_user_id),
            "confidence": confidence,
            "confidence_margin": margin,
            "duration_ms": duration_ms,
            "sample_rate": sample_rate,
            "channels": channels,
            "sample_count": getattr(updated_summary, "sample_count", None),
            "snr_db": snr_db,
            "speech_ratio": speech_ratio,
        },
    )
    return True


__all__ = [
    "build_voice_identity_profiles_event",
    "sync_voice_orchestrator_identity_profiles",
    "update_household_voice_assessment_from_pcm",
]