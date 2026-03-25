"""Derive a fail-closed room-ambiguity guard for targeted inferences.

This guard answers one narrow question: is the current room context clear
enough for person-targeted inferences such as identity hints or affect
proxies? It does not decide whether Twinr should speak; it only exposes a
small inspectable contract that later runtime policy can consume.
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


_MIN_DIRECTION_CONFIDENCE = 0.75


def _default_claim() -> RuntimeClaimMetadata:
    return RuntimeClaimMetadata(confidence=0.0, source="camera_plus_audio_policy", requires_confirmation=False)


@dataclass(frozen=True, slots=True)
class AmbiguousRoomGuardSnapshot:
    """Describe whether person-targeted inference is safe in the current room."""

    observed_at: float | None = None
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

    def to_automation_facts(self) -> dict[str, object]:
        """Serialize the guard into automation-friendly facts."""

        payload = {
            "observed_at": self.observed_at,
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
        )


def derive_ambiguous_room_guard(
    *,
    observed_at: float | None,
    live_facts: dict[str, object] | object,
) -> AmbiguousRoomGuardSnapshot:
    """Return one conservative room-ambiguity guard snapshot."""

    facts = coerce_mapping(live_facts)
    camera = coerce_mapping(facts.get("camera"))
    respeaker = coerce_mapping(facts.get("respeaker"))
    audio_policy = coerce_mapping(facts.get("audio_policy"))
    vad = coerce_mapping(facts.get("vad"))

    person_count = coerce_optional_int(camera.get("person_count"))
    explicit_person_visible = coerce_optional_bool(camera.get("person_visible"))
    person_visible = explicit_person_visible is True or (
        explicit_person_visible is None and person_count is not None and person_count > 0
    )
    person_count_unknown = coerce_optional_bool(camera.get("person_count_unknown")) is True
    room_busy = coerce_optional_bool(audio_policy.get("room_busy_or_overlapping")) is True
    background_media = (
        coerce_optional_bool(audio_policy.get("background_media_likely")) is True
        or normalize_text(audio_policy.get("speech_delivery_defer_reason")) == "background_media_active"
    )
    speaker_direction_stable = coerce_optional_bool(audio_policy.get("speaker_direction_stable"))
    direction_confidence = coerce_optional_ratio(respeaker.get("direction_confidence"))
    audio_context_active = any(
        (
            coerce_optional_bool(audio_policy.get("presence_audio_active")) is True,
            coerce_optional_bool(audio_policy.get("recent_follow_up_speech")) is True,
            coerce_optional_bool(audio_policy.get("resume_window_open")) is True,
            coerce_optional_bool(vad.get("speech_detected")) is True,
        )
    )

    if not person_visible:
        return _blocked_snapshot(
            observed_at=observed_at,
            reason="no_visible_person",
            confidence=0.78,
            person_visible=False,
            camera_person_count=person_count,
            camera_person_count_unknown=person_count_unknown,
            room_busy_or_overlapping=room_busy,
            background_media_likely=background_media,
            speaker_direction_stable=speaker_direction_stable,
            direction_confidence=direction_confidence,
            audio_context_active=audio_context_active,
        )
    if person_count_unknown:
        return _blocked_snapshot(
            observed_at=observed_at,
            reason="camera_person_count_unknown",
            confidence=0.86,
            person_visible=True,
            camera_person_count=person_count,
            camera_person_count_unknown=True,
            room_busy_or_overlapping=room_busy,
            background_media_likely=background_media,
            speaker_direction_stable=speaker_direction_stable,
            direction_confidence=direction_confidence,
            audio_context_active=audio_context_active,
        )
    if person_count is not None and person_count > 1:
        return _blocked_snapshot(
            observed_at=observed_at,
            reason="multi_person_context",
            confidence=0.97,
            person_visible=True,
            camera_person_count=person_count,
            camera_person_count_unknown=False,
            room_busy_or_overlapping=room_busy,
            background_media_likely=background_media,
            speaker_direction_stable=speaker_direction_stable,
            direction_confidence=direction_confidence,
            audio_context_active=audio_context_active,
        )
    if room_busy:
        return _blocked_snapshot(
            observed_at=observed_at,
            reason="room_busy_or_overlapping",
            confidence=0.91,
            person_visible=True,
            camera_person_count=person_count,
            camera_person_count_unknown=False,
            room_busy_or_overlapping=True,
            background_media_likely=background_media,
            speaker_direction_stable=speaker_direction_stable,
            direction_confidence=direction_confidence,
            audio_context_active=audio_context_active,
        )
    if background_media:
        return _blocked_snapshot(
            observed_at=observed_at,
            reason="background_media_active",
            confidence=0.84,
            person_visible=True,
            camera_person_count=person_count,
            camera_person_count_unknown=False,
            room_busy_or_overlapping=False,
            background_media_likely=True,
            speaker_direction_stable=speaker_direction_stable,
            direction_confidence=direction_confidence,
            audio_context_active=audio_context_active,
        )
    if (
        audio_context_active
        and (
            speaker_direction_stable is not True
            or direction_confidence is None
            or direction_confidence < _MIN_DIRECTION_CONFIDENCE
        )
    ):
        return _blocked_snapshot(
            observed_at=observed_at,
            reason="low_confidence_audio_direction",
            confidence=0.74 if direction_confidence is None else max(0.62, 1.0 - direction_confidence / 2.0),
            person_visible=True,
            camera_person_count=person_count,
            camera_person_count_unknown=False,
            room_busy_or_overlapping=False,
            background_media_likely=False,
            speaker_direction_stable=speaker_direction_stable,
            direction_confidence=direction_confidence,
            audio_context_active=True,
        )

    clear_confidence = mean_confidence(
        (
            0.88 if person_visible else None,
            0.92 if person_count == 1 else 0.76,
            0.86,
            None
            if not audio_context_active
            else (
                max(_MIN_DIRECTION_CONFIDENCE, direction_confidence or 0.0)
                if speaker_direction_stable is True
                else None
            ),
        )
    ) or 0.82
    return AmbiguousRoomGuardSnapshot(
        observed_at=observed_at,
        clear=True,
        guard_active=False,
        reason=None,
        policy_recommendation="clear",
        claim=RuntimeClaimMetadata(
            confidence=clear_confidence,
            source="camera_plus_audio_policy",
            requires_confirmation=False,
        ),
        person_visible=True,
        camera_person_count=person_count,
        camera_person_count_unknown=False,
        room_busy_or_overlapping=False,
        background_media_likely=False,
        speaker_direction_stable=speaker_direction_stable,
        direction_confidence=direction_confidence,
        audio_context_active=audio_context_active,
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
) -> AmbiguousRoomGuardSnapshot:
    """Build one blocked ambiguity-guard snapshot."""

    return AmbiguousRoomGuardSnapshot(
        observed_at=observed_at,
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
    )


__all__ = [
    "AmbiguousRoomGuardSnapshot",
    "derive_ambiguous_room_guard",
]
