# CHANGELOG: 2026-03-29
# BUG-1: Block false-clear when camera.person_visible=True but camera.person_count is missing or unknown.
# BUG-2: Block contradictory camera facts (e.g. person_visible=True with person_count<=0) and snapshot nested maps to avoid torn reads.
# SEC-1: Add freshness / replay guards so stale camera or audio facts cannot silently keep the room marked clear.
# IMP-1: Add versioned, inspectable payload fields for evidence completeness, contradictions, modality timestamps, and ages.
# IMP-2: Add configurable policy thresholds plus optional overlap / multi-speaker audio hooks and minimum-clear-confidence abstention.

"""Derive a fail-closed room-ambiguity guard for targeted inferences.

This guard answers one narrow question: is the current room context clear
enough for person-targeted inferences such as identity hints or affect
proxies? It does not decide whether Twinr should speak; it only exposes a
small inspectable contract that later runtime policy can consume.

2026 upgrade notes:
- clear decisions now require explicit single-person evidence;
- contradictory or stale sensor evidence blocks by default;
- the guard can consume richer upstream audio/vision signals without
  breaking older callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from twinr.proactive.runtime.claim_metadata import (
    RuntimeClaimMetadata,
    coerce_mapping,
    coerce_optional_bool,
    coerce_optional_float,
    coerce_optional_int,
    coerce_optional_ratio,
    mean_confidence,
    normalize_text,
)

_SCHEMA_VERSION = "2026-03-29"
_MIN_DIRECTION_CONFIDENCE = 0.75
_MAX_CAMERA_AGE_SECONDS = 2.5
_MAX_AUDIO_AGE_SECONDS = 1.5
_MAX_OVERLAP_PROBABILITY = 0.20
_MIN_PERSON_COUNT_STABLE_FRAMES = 2
_MIN_DIRECTION_STABLE_MS = 250
_MINIMUM_CLEAR_CONFIDENCE = 0.82
_TIMESTAMP_KEYS = ("observed_at", "updated_at", "timestamp", "ts")

_REASON_NO_VISIBLE_PERSON = "no_visible_person"
_REASON_MISSING_SINGLE_PERSON_EVIDENCE = "missing_single_person_evidence"
_REASON_CAMERA_PERSON_COUNT_UNKNOWN = "camera_person_count_unknown"
_REASON_MULTI_PERSON_CONTEXT = "multi_person_context"
_REASON_ROOM_BUSY = "room_busy_or_overlapping"
_REASON_BACKGROUND_MEDIA = "background_media_active"
_REASON_LOW_CONFIDENCE_AUDIO_DIRECTION = "low_confidence_audio_direction"
_REASON_CONTRADICTORY_CAMERA_SIGNALS = "contradictory_camera_signals"
_REASON_CAMERA_STALE = "camera_stale_or_missing_timestamp"
_REASON_AUDIO_STALE = "audio_stale_or_missing_timestamp"
_REASON_AUDIO_MULTI_SPEAKER = "multi_speaker_audio_context"
_REASON_AUDIO_OVERLAP = "audio_overlap_probability_high"
_REASON_CAMERA_NOT_STABLE = "camera_person_count_not_stable_yet"
_REASON_DIRECTION_NOT_STABLE_LONG_ENOUGH = "speaker_direction_not_stable_long_enough"
_REASON_INSUFFICIENT_CLEAR_CONFIDENCE = "insufficient_clear_confidence"
_SOFT_GENERIC_PROMPT_REASONS = frozenset(
    {
        _REASON_MISSING_SINGLE_PERSON_EVIDENCE,
        _REASON_CAMERA_PERSON_COUNT_UNKNOWN,
        _REASON_CAMERA_NOT_STABLE,
        _REASON_INSUFFICIENT_CLEAR_CONFIDENCE,
    }
)


def ambiguous_room_guard_requires_hard_block(reason: object | None) -> bool:
    """Return whether a guard reason should still hard-block generic prompts.

    The guard is a conservative claim surface for target-safe room context.
    Some reasons mean "we cannot yet prove a single clear target", but they do
    not by themselves prove a dangerous room state for generic low-sensitivity
    prompts. Explicit multi-person, busy, overlap, stale, or no-person states
    remain hard blocks.
    """

    normalized = normalize_text(reason).lower()
    if not normalized:
        return False
    return normalized not in _SOFT_GENERIC_PROMPT_REASONS


def _default_claim() -> RuntimeClaimMetadata:
    return RuntimeClaimMetadata(confidence=0.0, source="camera_plus_audio_policy", requires_confirmation=False)


@dataclass(frozen=True, slots=True)
class AmbiguousRoomGuardSnapshot:
    """Describe whether person-targeted inference is safe in the current room."""

    observed_at: float | None = None
    schema_version: str = _SCHEMA_VERSION
    clear: bool = False
    guard_active: bool = True
    reason: str | None = None
    policy_recommendation: str = "block_targeted_inference"
    claim: RuntimeClaimMetadata = field(default_factory=_default_claim)
    person_visible: bool = False
    camera_person_count: int | None = None
    camera_person_count_unknown: bool = False
    room_busy_or_overlapping: bool = False
    background_media_likely: bool = False
    speaker_direction_stable: bool | None = None
    direction_confidence: float | None = None
    audio_context_active: bool = False
    contradictory_inputs: bool = False
    evidence_complete: bool = False
    camera_observed_at: float | None = None
    audio_observed_at: float | None = None
    camera_age_seconds: float | None = None
    audio_age_seconds: float | None = None
    overlap_probability: float | None = None
    tracked_audio_source_count: int | None = None
    minimum_direction_confidence: float = _MIN_DIRECTION_CONFIDENCE
    minimum_clear_confidence: float = _MINIMUM_CLEAR_CONFIDENCE

    def to_automation_facts(self) -> dict[str, object]:
        """Serialize the guard into automation-friendly facts."""

        payload = {
            "observed_at": self.observed_at,
            "schema_version": self.schema_version,
            "clear": self.clear,
            "guard_active": self.guard_active,
            "reason": self.reason,
            "policy_recommendation": self.policy_recommendation,
            "person_visible": self.person_visible,
            "camera_person_count": self.camera_person_count,
            "camera_person_count_unknown": self.camera_person_count_unknown,
            "room_busy_or_overlapping": self.room_busy_or_overlapping,
            "background_media_likely": self.background_media_likely,
            "speaker_direction_stable": self.speaker_direction_stable,
            "direction_confidence": self.direction_confidence,
            "audio_context_active": self.audio_context_active,
            "contradictory_inputs": self.contradictory_inputs,
            "evidence_complete": self.evidence_complete,
            "camera_observed_at": self.camera_observed_at,
            "audio_observed_at": self.audio_observed_at,
            "camera_age_seconds": self.camera_age_seconds,
            "audio_age_seconds": self.audio_age_seconds,
            "overlap_probability": self.overlap_probability,
            "tracked_audio_source_count": self.tracked_audio_source_count,
            "minimum_direction_confidence": self.minimum_direction_confidence,
            "minimum_clear_confidence": self.minimum_clear_confidence,
        }
        payload.update(self.claim.to_payload())
        return payload

    def event_data(self) -> dict[str, object]:
        """Serialize the guard into compact flat event fields."""

        return {
            "ambiguous_room_guard_active": self.guard_active,
            "ambiguous_room_guard_reason": self.reason,
            "ambiguous_room_guard_confidence": self.claim.confidence,
            "ambiguous_room_guard_policy": self.policy_recommendation,
            "ambiguous_room_guard_schema_version": self.schema_version,
        }

    @classmethod
    def from_fact_map(
        cls,
        value: object | None,
    ) -> "AmbiguousRoomGuardSnapshot | None":
        """Parse one serialized ambiguity-guard payload."""

        payload = coerce_mapping(value)
        if not payload:
            return None
        return cls(
            observed_at=coerce_optional_float(payload.get("observed_at")),
            schema_version=normalize_text(payload.get("schema_version")) or _SCHEMA_VERSION,
            clear=coerce_optional_bool(payload.get("clear")) is True,
            guard_active=coerce_optional_bool(payload.get("guard_active")) is True,
            reason=normalize_text(payload.get("reason")) or None,
            policy_recommendation=normalize_text(payload.get("policy_recommendation")) or "block_targeted_inference",
            claim=RuntimeClaimMetadata.from_payload(
                payload,
                default_source="camera_plus_audio_policy",
            ),
            person_visible=coerce_optional_bool(payload.get("person_visible")) is True,
            camera_person_count=coerce_optional_int(payload.get("camera_person_count")),
            camera_person_count_unknown=coerce_optional_bool(payload.get("camera_person_count_unknown")) is True,
            room_busy_or_overlapping=coerce_optional_bool(payload.get("room_busy_or_overlapping")) is True,
            background_media_likely=coerce_optional_bool(payload.get("background_media_likely")) is True,
            speaker_direction_stable=coerce_optional_bool(payload.get("speaker_direction_stable")),
            direction_confidence=coerce_optional_ratio(payload.get("direction_confidence")),
            audio_context_active=coerce_optional_bool(payload.get("audio_context_active")) is True,
            contradictory_inputs=coerce_optional_bool(payload.get("contradictory_inputs")) is True,
            evidence_complete=coerce_optional_bool(payload.get("evidence_complete")) is True,
            camera_observed_at=coerce_optional_float(payload.get("camera_observed_at")),
            audio_observed_at=coerce_optional_float(payload.get("audio_observed_at")),
            camera_age_seconds=_coerce_non_negative_float(payload.get("camera_age_seconds")),
            audio_age_seconds=_coerce_non_negative_float(payload.get("audio_age_seconds")),
            overlap_probability=coerce_optional_ratio(payload.get("overlap_probability")),
            tracked_audio_source_count=coerce_optional_int(payload.get("tracked_audio_source_count")),
            minimum_direction_confidence=coerce_optional_ratio(payload.get("minimum_direction_confidence")) or _MIN_DIRECTION_CONFIDENCE,
            minimum_clear_confidence=coerce_optional_ratio(payload.get("minimum_clear_confidence")) or _MINIMUM_CLEAR_CONFIDENCE,
        )


def derive_ambiguous_room_guard(
    *,
    observed_at: float | None,
    live_facts: dict[str, object] | object,
) -> AmbiguousRoomGuardSnapshot:
    """Return one conservative room-ambiguity guard snapshot."""

    facts = _mapping_snapshot(live_facts)
    camera = _mapping_snapshot(facts.get("camera"))
    respeaker = _mapping_snapshot(facts.get("respeaker"))
    audio_policy = _mapping_snapshot(facts.get("audio_policy"))
    vad = _mapping_snapshot(facts.get("vad"))
    guard_policy = _mapping_snapshot(facts.get("guard_policy"))

    reference_observed_at = observed_at
    if reference_observed_at is None:
        reference_observed_at = _extract_timestamp(facts)

    min_direction_confidence = _policy_ratio(
        guard_policy,
        "min_direction_confidence",
        default=_MIN_DIRECTION_CONFIDENCE,
    )
    max_camera_age_seconds = _policy_non_negative_float(
        guard_policy,
        "max_camera_age_seconds",
        default=_MAX_CAMERA_AGE_SECONDS,
    )
    max_audio_age_seconds = _policy_non_negative_float(
        guard_policy,
        "max_audio_age_seconds",
        default=_MAX_AUDIO_AGE_SECONDS,
    )
    max_overlap_probability = _policy_ratio(
        guard_policy,
        "max_overlap_probability",
        default=_MAX_OVERLAP_PROBABILITY,
    )
    min_person_count_stable_frames = _policy_int(
        guard_policy,
        "min_person_count_stable_frames",
        default=_MIN_PERSON_COUNT_STABLE_FRAMES,
    )
    min_direction_stable_ms = _policy_int(
        guard_policy,
        "min_direction_stable_ms",
        default=_MIN_DIRECTION_STABLE_MS,
    )
    minimum_clear_confidence = _policy_ratio(
        guard_policy,
        "minimum_clear_confidence",
        default=_MINIMUM_CLEAR_CONFIDENCE,
    )
    # BREAKING: clear decisions now require a camera timestamp by default.
    # Set guard_policy['require_camera_timestamp_for_clear']=False to recover the legacy behavior.
    require_camera_timestamp_for_clear = _policy_bool(
        guard_policy,
        "require_camera_timestamp_for_clear",
        default=True,
    )
    # BREAKING: if audio context is active, clear decisions now require fresh audio timestamps by default.
    require_audio_timestamp_when_audio_active = _policy_bool(
        guard_policy,
        "require_audio_timestamp_when_audio_active",
        default=True,
    )

    person_count = coerce_optional_int(camera.get("person_count"))
    explicit_person_visible = coerce_optional_bool(camera.get("person_visible"))
    person_count_unknown_flag = coerce_optional_bool(camera.get("person_count_unknown")) is True
    person_count_confidence = _first_ratio(camera, "single_person_confidence", "person_count_confidence")
    person_count_stable_frames = _first_int(camera, "person_count_stable_frames", "stable_frames")
    camera_observed_at = _extract_timestamp(camera)
    camera_age_seconds = _age_seconds(reference_observed_at, camera_observed_at)
    camera_stale = _is_stale(camera_age_seconds, max_camera_age_seconds)

    if explicit_person_visible is None:
        person_visible = person_count is not None and person_count > 0
    else:
        person_visible = explicit_person_visible

    contradictory_camera_signals = (
        explicit_person_visible is True and person_count is not None and person_count <= 0
    ) or (
        explicit_person_visible is False and person_count is not None and person_count > 0
    )

    missing_single_person_evidence = person_visible and person_count is None
    person_count_unknown = person_count_unknown_flag or missing_single_person_evidence
    single_person_stable = (
        person_count_stable_frames is None or person_count_stable_frames >= min_person_count_stable_frames
    )

    room_busy = coerce_optional_bool(audio_policy.get("room_busy_or_overlapping")) is True
    background_media = (
        coerce_optional_bool(audio_policy.get("background_media_likely")) is True
        or normalize_text(audio_policy.get("speech_delivery_defer_reason")) == _REASON_BACKGROUND_MEDIA
    )
    speaker_direction_stable = _first_bool(audio_policy, "speaker_direction_stable")
    if speaker_direction_stable is None:
        speaker_direction_stable = _first_bool(respeaker, "speaker_direction_stable")
    direction_confidence = _first_ratio(respeaker, "direction_confidence", "speaker_direction_confidence")
    direction_stable_for_ms = _first_int(
        audio_policy,
        "speaker_direction_stable_for_ms",
        "direction_stable_for_ms",
    )
    if direction_stable_for_ms is None:
        direction_stable_for_ms = _first_int(respeaker, "direction_stable_for_ms")

    overlap_probability = _first_ratio(
        audio_policy,
        "overlap_probability",
        "room_overlap_probability",
    )
    if overlap_probability is None:
        overlap_probability = _first_ratio(respeaker, "overlap_probability", "speaker_overlap_probability")
    tracked_audio_source_count = _first_int(
        audio_policy,
        "active_speaker_count",
        "tracked_audio_source_count",
        "speaker_count_estimate",
    )
    if tracked_audio_source_count is None:
        tracked_audio_source_count = _first_int(
            respeaker,
            "tracked_source_count",
            "speaker_count_estimate",
        )

    audio_context_active = any(
        (
            coerce_optional_bool(audio_policy.get("presence_audio_active")) is True,
            coerce_optional_bool(audio_policy.get("recent_follow_up_speech")) is True,
            coerce_optional_bool(audio_policy.get("resume_window_open")) is True,
            coerce_optional_bool(vad.get("speech_detected")) is True,
        )
    )
    audio_observed_at = _latest_timestamp(
        _extract_timestamp(audio_policy),
        _extract_timestamp(respeaker),
        _extract_timestamp(vad),
    )
    audio_age_seconds = _age_seconds(reference_observed_at, audio_observed_at)
    audio_stale = _is_stale(audio_age_seconds, max_audio_age_seconds)

    has_camera_clear_evidence = (
        person_visible
        and person_count == 1
        and not person_count_unknown
        and not contradictory_camera_signals
        and single_person_stable
        and (not require_camera_timestamp_for_clear or (camera_observed_at is not None and not camera_stale))
    )
    has_audio_clear_evidence = (
        not audio_context_active
        or (
            speaker_direction_stable is True
            and direction_confidence is not None
            and direction_confidence >= min_direction_confidence
            and (direction_stable_for_ms is None or direction_stable_for_ms >= min_direction_stable_ms)
            and (tracked_audio_source_count is None or tracked_audio_source_count <= 1)
            and (overlap_probability is None or overlap_probability < max_overlap_probability)
            and (
                not require_audio_timestamp_when_audio_active
                or (audio_observed_at is not None and not audio_stale)
            )
        )
    )
    evidence_complete = (
        has_camera_clear_evidence
        and has_audio_clear_evidence
        and not room_busy
        and not background_media
    )

    common_kwargs = {
        "observed_at": reference_observed_at,
        "person_visible": person_visible,
        "camera_person_count": person_count,
        "camera_person_count_unknown": person_count_unknown,
        "room_busy_or_overlapping": room_busy,
        "background_media_likely": background_media,
        "speaker_direction_stable": speaker_direction_stable,
        "direction_confidence": direction_confidence,
        "audio_context_active": audio_context_active,
        "contradictory_inputs": contradictory_camera_signals,
        "evidence_complete": evidence_complete,
        "camera_observed_at": camera_observed_at,
        "audio_observed_at": audio_observed_at,
        "camera_age_seconds": camera_age_seconds,
        "audio_age_seconds": audio_age_seconds,
        "overlap_probability": overlap_probability,
        "tracked_audio_source_count": tracked_audio_source_count,
        "minimum_direction_confidence": min_direction_confidence,
        "minimum_clear_confidence": minimum_clear_confidence,
    }

    if contradictory_camera_signals:
        return _blocked_snapshot(
            reason=_REASON_CONTRADICTORY_CAMERA_SIGNALS,
            confidence=0.98,
            **common_kwargs,
        )
    if not person_visible:
        return _blocked_snapshot(
            reason=_REASON_NO_VISIBLE_PERSON,
            confidence=0.78,
            **common_kwargs,
        )
    if person_count_unknown_flag:
        return _blocked_snapshot(
            reason=_REASON_CAMERA_PERSON_COUNT_UNKNOWN,
            confidence=0.86,
            **common_kwargs,
        )
    if missing_single_person_evidence:
        return _blocked_snapshot(
            reason=_REASON_MISSING_SINGLE_PERSON_EVIDENCE,
            confidence=0.93,
            **common_kwargs,
        )
    if person_count is not None and person_count > 1:
        return _blocked_snapshot(
            reason=_REASON_MULTI_PERSON_CONTEXT,
            confidence=0.97,
            **common_kwargs,
        )
    if person_count is not None and person_count <= 0:
        return _blocked_snapshot(
            reason=_REASON_CONTRADICTORY_CAMERA_SIGNALS,
            confidence=0.98,
            **common_kwargs,
        )
    if not single_person_stable:
        return _blocked_snapshot(
            reason=_REASON_CAMERA_NOT_STABLE,
            confidence=0.87,
            **common_kwargs,
        )
    if require_camera_timestamp_for_clear and (camera_observed_at is None or camera_stale):
        return _blocked_snapshot(
            reason=_REASON_CAMERA_STALE,
            confidence=0.95 if camera_observed_at is None else 0.92,
            **common_kwargs,
        )
    if room_busy:
        return _blocked_snapshot(
            reason=_REASON_ROOM_BUSY,
            confidence=0.91,
            **common_kwargs,
        )
    if background_media:
        return _blocked_snapshot(
            reason=_REASON_BACKGROUND_MEDIA,
            confidence=0.84,
            **common_kwargs,
        )
    if overlap_probability is not None and overlap_probability >= max_overlap_probability:
        return _blocked_snapshot(
            reason=_REASON_AUDIO_OVERLAP,
            confidence=max(0.84, overlap_probability),
            **common_kwargs,
        )
    if audio_context_active and tracked_audio_source_count is not None and tracked_audio_source_count > 1:
        return _blocked_snapshot(
            reason=_REASON_AUDIO_MULTI_SPEAKER,
            confidence=0.93,
            **common_kwargs,
        )
    if audio_context_active and require_audio_timestamp_when_audio_active and (audio_observed_at is None or audio_stale):
        return _blocked_snapshot(
            reason=_REASON_AUDIO_STALE,
            confidence=0.94 if audio_observed_at is None else 0.90,
            **common_kwargs,
        )
    if audio_context_active and direction_stable_for_ms is not None and direction_stable_for_ms < min_direction_stable_ms:
        return _blocked_snapshot(
            reason=_REASON_DIRECTION_NOT_STABLE_LONG_ENOUGH,
            confidence=0.83,
            **common_kwargs,
        )
    if (
        audio_context_active
        and (
            speaker_direction_stable is not True
            or direction_confidence is None
            or direction_confidence < min_direction_confidence
        )
    ):
        return _blocked_snapshot(
            reason=_REASON_LOW_CONFIDENCE_AUDIO_DIRECTION,
            confidence=0.74 if direction_confidence is None else max(0.62, 1.0 - direction_confidence / 2.0),
            **common_kwargs,
        )

    clear_confidence = mean_confidence(
        (
            0.96 if person_visible else None,
            person_count_confidence or (0.95 if person_count == 1 else None),
            _freshness_confidence(camera_age_seconds, max_camera_age_seconds),
            1.0 if not contradictory_camera_signals else 0.0,
            1.0 if single_person_stable else 0.0,
            None
            if not audio_context_active
            else (
                direction_confidence if speaker_direction_stable is True else 0.0
            ),
            None
            if not audio_context_active
            else _freshness_confidence(audio_age_seconds, max_audio_age_seconds),
            None if overlap_probability is None else 1.0 - overlap_probability,
            None
            if tracked_audio_source_count is None
            else (1.0 if tracked_audio_source_count <= 1 else 0.0),
        )
    ) or minimum_clear_confidence

    if clear_confidence < minimum_clear_confidence:
        return _blocked_snapshot(
            reason=_REASON_INSUFFICIENT_CLEAR_CONFIDENCE,
            confidence=max(clear_confidence, 0.60),
            **common_kwargs,
        )

    return AmbiguousRoomGuardSnapshot(
        observed_at=reference_observed_at,
        schema_version=_SCHEMA_VERSION,
        clear=True,
        guard_active=False,
        reason=None,
        policy_recommendation="clear",
        claim=RuntimeClaimMetadata(
            confidence=clear_confidence,
            source="camera_plus_audio_policy",
            requires_confirmation=False,
        ),
        person_visible=person_visible,
        camera_person_count=person_count,
        camera_person_count_unknown=False,
        room_busy_or_overlapping=False,
        background_media_likely=False,
        speaker_direction_stable=speaker_direction_stable,
        direction_confidence=direction_confidence,
        audio_context_active=audio_context_active,
        contradictory_inputs=contradictory_camera_signals,
        evidence_complete=evidence_complete,
        camera_observed_at=camera_observed_at,
        audio_observed_at=audio_observed_at,
        camera_age_seconds=camera_age_seconds,
        audio_age_seconds=audio_age_seconds,
        overlap_probability=overlap_probability,
        tracked_audio_source_count=tracked_audio_source_count,
        minimum_direction_confidence=min_direction_confidence,
        minimum_clear_confidence=minimum_clear_confidence,
    )


def _blocked_snapshot(
    *,
    observed_at: float | None,
    reason: str,
    confidence: float,
    person_visible: bool,
    camera_person_count: int | None,
    camera_person_count_unknown: bool,
    room_busy_or_overlapping: bool,
    background_media_likely: bool,
    speaker_direction_stable: bool | None,
    direction_confidence: float | None,
    audio_context_active: bool,
    contradictory_inputs: bool,
    evidence_complete: bool,
    camera_observed_at: float | None,
    audio_observed_at: float | None,
    camera_age_seconds: float | None,
    audio_age_seconds: float | None,
    overlap_probability: float | None,
    tracked_audio_source_count: int | None,
    minimum_direction_confidence: float,
    minimum_clear_confidence: float,
) -> AmbiguousRoomGuardSnapshot:
    """Build one blocked ambiguity-guard snapshot."""

    return AmbiguousRoomGuardSnapshot(
        observed_at=observed_at,
        schema_version=_SCHEMA_VERSION,
        clear=False,
        guard_active=True,
        reason=reason,
        policy_recommendation="block_targeted_inference",
        claim=RuntimeClaimMetadata(
            confidence=confidence,
            source="camera_plus_audio_policy",
            requires_confirmation=False,
        ),
        person_visible=person_visible,
        camera_person_count=camera_person_count,
        camera_person_count_unknown=camera_person_count_unknown,
        room_busy_or_overlapping=room_busy_or_overlapping,
        background_media_likely=background_media_likely,
        speaker_direction_stable=speaker_direction_stable,
        direction_confidence=direction_confidence,
        audio_context_active=audio_context_active,
        contradictory_inputs=contradictory_inputs,
        evidence_complete=evidence_complete,
        camera_observed_at=camera_observed_at,
        audio_observed_at=audio_observed_at,
        camera_age_seconds=camera_age_seconds,
        audio_age_seconds=audio_age_seconds,
        overlap_probability=overlap_probability,
        tracked_audio_source_count=tracked_audio_source_count,
        minimum_direction_confidence=minimum_direction_confidence,
        minimum_clear_confidence=minimum_clear_confidence,
    )


def _mapping_snapshot(value: object | None) -> dict[str, object]:
    """Return one shallow immutable-by-convention snapshot of a mapping-like object."""

    return dict(coerce_mapping(value))


def _extract_timestamp(mapping: dict[str, object]) -> float | None:
    for key in _TIMESTAMP_KEYS:
        timestamp = coerce_optional_float(mapping.get(key))
        if timestamp is not None:
            return timestamp
    return None


def _latest_timestamp(*timestamps: float | None) -> float | None:
    present = [timestamp for timestamp in timestamps if timestamp is not None]
    if not present:
        return None
    return max(present)


def _age_seconds(reference_time: float | None, event_time: float | None) -> float | None:
    if reference_time is None or event_time is None:
        return None
    return max(0.0, reference_time - event_time)


def _is_stale(age_seconds: float | None, max_age_seconds: float) -> bool:
    return age_seconds is not None and age_seconds > max_age_seconds


def _freshness_confidence(age_seconds: float | None, max_age_seconds: float) -> float | None:
    if age_seconds is None:
        return None
    if max_age_seconds <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - age_seconds / max_age_seconds))


def _coerce_non_negative_float(value: object | None) -> float | None:
    parsed = coerce_optional_float(value)
    if parsed is None or parsed < 0.0:
        return None
    return parsed


def _first_ratio(mapping: dict[str, object], *keys: str) -> float | None:
    for key in keys:
        value = coerce_optional_ratio(mapping.get(key))
        if value is not None:
            return value
    return None


def _first_bool(mapping: dict[str, object], *keys: str) -> bool | None:
    for key in keys:
        value = coerce_optional_bool(mapping.get(key))
        if value is not None:
            return value
    return None


def _first_int(mapping: dict[str, object], *keys: str) -> int | None:
    for key in keys:
        value = coerce_optional_int(mapping.get(key))
        if value is not None:
            return value
    return None


def _policy_bool(mapping: dict[str, object], key: str, *, default: bool) -> bool:
    parsed = coerce_optional_bool(mapping.get(key))
    return default if parsed is None else parsed


def _policy_ratio(mapping: dict[str, object], key: str, *, default: float) -> float:
    parsed = coerce_optional_ratio(mapping.get(key))
    return default if parsed is None else parsed


def _policy_non_negative_float(mapping: dict[str, object], key: str, *, default: float) -> float:
    parsed = _coerce_non_negative_float(mapping.get(key))
    return default if parsed is None else parsed


def _policy_int(mapping: dict[str, object], key: str, *, default: int) -> int:
    parsed = coerce_optional_int(mapping.get(key))
    return default if parsed is None else max(0, parsed)


__all__ = [
    "AmbiguousRoomGuardSnapshot",
    "ambiguous_room_guard_requires_hard_block",
    "derive_ambiguous_room_guard",
]
