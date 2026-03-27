"""Keep household voice identity sync and passive updates out of runtime loops."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from twinr.hardware.household_identity import HouseholdIdentityManager
from twinr.hardware.household_voice_identity import HouseholdVoiceAssessment
from twinr.orchestrator.voice_contracts import (
    OrchestratorVoiceIdentityProfile,
    OrchestratorVoiceIdentityProfilesEvent,
)


_PASSIVE_UPDATE_SPEAKER_ASSOCIATION_MIN_CONFIDENCE = 0.82
_PASSIVE_UPDATE_ALLOWED_STATUSES = frozenset({"likely_user", "known_other_user"})


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


def _coerce_optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str, bytes, bytearray)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0.0 or parsed > 1.0:
        return None
    return parsed


def _coerce_optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str, bytes, bytearray)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_emit(loop: Any, text: str) -> None:
    emitter = getattr(loop, "_try_emit", None)
    if callable(emitter):
        emitter(text)
        return
    emitter = getattr(loop, "emit", None)
    if callable(emitter):
        emitter(text)


def _trace_event(loop: Any, name: str, *, kind: str, details: dict[str, object]) -> None:
    tracer = getattr(loop, "_trace_event", None)
    if callable(tracer):
        tracer(name, kind=kind, details=details)


def _safe_error_text(loop: Any, exc: Exception) -> str:
    formatter = getattr(loop, "_safe_error_text", None)
    if callable(formatter):
        return str(formatter(exc))
    return type(exc).__name__


def _household_identity_manager(loop: Any) -> HouseholdIdentityManager | None:
    manager = getattr(loop, "household_identity_manager", None)
    if manager is not None:
        return manager
    camera = getattr(loop, "camera", None)
    if camera is None:
        return None
    try:
        manager = HouseholdIdentityManager.from_config(
            loop.config,
            camera=camera,
            camera_lock=getattr(loop, "_camera_lock", None),
        )
    except Exception as exc:
        _safe_emit(loop, f"household_voice_profile_error={_safe_error_text(loop, exc)}")
        return None
    setattr(loop, "household_identity_manager", manager)
    return manager


def build_voice_identity_profiles_event(
    loop: Any,
) -> OrchestratorVoiceIdentityProfilesEvent | None:
    """Return the current read-only household voice profile snapshot."""

    manager = _household_identity_manager(loop)
    if manager is None:
        return None
    profiles = manager.voice_monitor.voice_profiles()
    return OrchestratorVoiceIdentityProfilesEvent(
        revision=manager.voice_monitor.profile_revision(),
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
    event = build_voice_identity_profiles_event(loop)
    if event is None:
        return
    last_revision = getattr(loop, "_last_voice_identity_profile_revision", None)
    if not force and event.revision == last_revision:
        return
    setattr(loop, "_last_voice_identity_profile_revision", event.revision)
    setattr(loop, "_last_voice_identity_profile_count", len(event.profiles))
    notify = getattr(voice_orchestrator, "notify_identity_profiles", None)
    if callable(notify):
        notify(event)
        _trace_event(
            loop,
            "voice_identity_profiles_synced",
            kind="mutation",
            details={
                "revision": event.revision,
                "profile_count": len(event.profiles),
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
    config = loop.config
    try:
        assessment = manager.assess_voice(
            audio_pcm,
            sample_rate=config.openai_realtime_input_sample_rate,
            channels=config.audio_channels,
        )
    except Exception as exc:
        _safe_emit(loop, f"household_voice_profile_error={_safe_error_text(loop, exc)}")
        return None
    if assessment.status == "not_enrolled":
        return assessment
    if assessment.should_persist:
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
            _safe_emit(loop, f"voice_profile_persist_error={_safe_error_text(loop, exc)}")
            return assessment
        _safe_emit(loop, f"voice_profile_status={assessment.status}")
        if assessment.confidence is not None:
            _safe_emit(loop, f"voice_profile_confidence={assessment.confidence:.2f}")
        if assessment.matched_user_id:
            _safe_emit(loop, f"voice_profile_user_id={assessment.matched_user_id}")
    if _maybe_apply_passive_voice_update(
        loop,
        manager=manager,
        audio_pcm=audio_pcm,
        assessment=assessment,
        source=source,
    ):
        sync_voice_orchestrator_identity_profiles(loop, force=True)
    return assessment


def _maybe_apply_passive_voice_update(
    loop: Any,
    *,
    manager: HouseholdIdentityManager,
    audio_pcm: bytes,
    assessment: HouseholdVoiceAssessment,
    source: str,
) -> bool:
    """Merge a clean accepted turn back into the enrolled profile when safe."""

    config = loop.config
    if not bool(getattr(config, "voice_profile_passive_update_enabled", True)):
        return False
    if assessment.status not in _PASSIVE_UPDATE_ALLOWED_STATUSES:
        return False
    if assessment.matched_user_id is None:
        return False
    confidence = float(assessment.confidence or 0.0)
    if confidence < float(getattr(config, "voice_profile_passive_update_min_confidence", 0.86)):
        return False
    duration_ms = int(
        round(
            (len(audio_pcm) / max(1, int(config.audio_channels) * 2))
            / max(1, int(config.openai_realtime_input_sample_rate))
            * 1000.0
        )
    )
    if duration_ms < int(getattr(config, "voice_profile_passive_update_min_duration_ms", 2500)):
        return False
    summary = manager.voice_monitor.summary(assessment.matched_user_id)
    if not summary.enrolled:
        return False

    facts = _coerce_mapping(getattr(loop, "_latest_sensor_observation_facts", None))
    speaker_association = _coerce_mapping(facts.get("speaker_association"))
    camera = _coerce_mapping(facts.get("camera"))
    audio_policy = _coerce_mapping(facts.get("audio_policy"))
    if _coerce_optional_bool(speaker_association.get("associated")) is not True:
        return False
    if (
        (_coerce_optional_float(speaker_association.get("confidence")) or 0.0)
        < _PASSIVE_UPDATE_SPEAKER_ASSOCIATION_MIN_CONFIDENCE
    ):
        return False
    if _coerce_optional_bool(camera.get("person_visible")) is not True:
        return False
    if _coerce_optional_int(camera.get("person_count")) != 1:
        return False
    if _coerce_optional_bool(audio_policy.get("background_media_likely")) is True:
        return False
    if _coerce_optional_bool(audio_policy.get("speech_overlap_likely")) is True:
        return False

    try:
        updated_summary, _member = manager.enroll_voice(
            audio_pcm,
            sample_rate=config.openai_realtime_input_sample_rate,
            channels=config.audio_channels,
            user_id=assessment.matched_user_id,
            display_name=assessment.matched_user_display_name,
        )
    except Exception as exc:
        _safe_emit(loop, f"voice_profile_passive_update_error={_safe_error_text(loop, exc)}")
        return False

    _safe_emit(loop, "voice_profile_passive_update=applied")
    _safe_emit(loop, f"voice_profile_passive_update_user_id={assessment.matched_user_id}")
    _trace_event(
        loop,
        "voice_profile_passive_update_applied",
        kind="mutation",
            details={
                "source": source,
                "user_id": assessment.matched_user_id,
                "confidence": confidence,
                "duration_ms": duration_ms,
                "sample_count": updated_summary.sample_count,
            },
        )
    return True


__all__ = [
    "build_voice_identity_profiles_event",
    "sync_voice_orchestrator_identity_profiles",
    "update_household_voice_assessment_from_pcm",
]
