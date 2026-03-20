"""Block sensitive proactive behavior on weak ReSpeaker privacy context.

This gate is intentionally narrow: it does not decide whether a proactive
candidate is useful, only whether current room context is too ambiguous for
sensitive/private proactive speech.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from twinr.memory.longterm.core.ontology import normalize_memory_sensitivity
from twinr.proactive.runtime.ambiguous_room_guard import (
    AmbiguousRoomGuardSnapshot,
    derive_ambiguous_room_guard,
)
from twinr.proactive.runtime.claim_metadata import coerce_optional_float


_SENSITIVE_MEMORY_LEVELS = frozenset({"private", "sensitive", "critical"})


@dataclass(frozen=True, slots=True)
class ReSpeakerSensitiveBehaviorGateDecision:
    """Describe whether sensitive proactive behavior is allowed right now."""

    allowed: bool
    reason: str | None
    candidate_sensitivity: str
    multi_person_likely: bool = False
    low_confidence_audio: bool = False
    audio_context_active: bool = False
    private_content_routing_allowed: bool = True
    camera_person_count: int | None = None
    direction_confidence: float | None = None
    speaker_direction_stable: bool | None = None
    ambiguous_room_reason: str | None = None
    ambiguous_room_confidence: float | None = None

    def event_data(self) -> dict[str, object]:
        """Render one JSON-safe event payload for runtime diagnostics."""

        return {
            "candidate_sensitivity": self.candidate_sensitivity,
            "sensitive_behavior_allowed": self.allowed,
            "sensitive_behavior_block_reason": self.reason,
            "sensitive_multi_person_likely": self.multi_person_likely,
            "sensitive_low_confidence_audio": self.low_confidence_audio,
            "sensitive_audio_context_active": self.audio_context_active,
            "private_content_routing_allowed": self.private_content_routing_allowed,
            "camera_person_count": self.camera_person_count,
            "audio_direction_confidence": self.direction_confidence,
            "audio_speaker_direction_stable": self.speaker_direction_stable,
            "ambiguous_room_reason": self.ambiguous_room_reason,
            "ambiguous_room_confidence": self.ambiguous_room_confidence,
        }


def evaluate_respeaker_sensitive_behavior_gate(
    *,
    candidate_sensitivity: object,
    live_facts: Mapping[str, object] | None,
    ambiguous_room_guard: AmbiguousRoomGuardSnapshot | None = None,
) -> ReSpeakerSensitiveBehaviorGateDecision:
    """Return whether current room context is safe enough for sensitive speech."""

    sensitivity = _normalize_sensitivity(candidate_sensitivity)
    if sensitivity not in _SENSITIVE_MEMORY_LEVELS:
        return ReSpeakerSensitiveBehaviorGateDecision(
            allowed=True,
            reason=None,
            candidate_sensitivity=sensitivity,
            private_content_routing_allowed=True,
        )

    facts = live_facts if isinstance(live_facts, Mapping) else {}
    guard = ambiguous_room_guard or AmbiguousRoomGuardSnapshot.from_fact_map(
        facts.get("ambiguous_room_guard"),
    ) or derive_ambiguous_room_guard(
        observed_at=coerce_optional_float(
            facts.get("sensor", {}).get("observed_at") if isinstance(facts.get("sensor"), Mapping) else None
        ),
        live_facts=facts,
    )
    multi_person_likely = guard.reason in {
        "camera_person_count_unknown",
        "multi_person_context",
        "room_busy_or_overlapping",
    }
    low_confidence_audio = guard.reason == "low_confidence_audio_direction"

    if guard.guard_active:
        return ReSpeakerSensitiveBehaviorGateDecision(
            allowed=False,
            reason=guard.reason,
            candidate_sensitivity=sensitivity,
            multi_person_likely=multi_person_likely,
            low_confidence_audio=low_confidence_audio,
            audio_context_active=guard.audio_context_active,
            private_content_routing_allowed=False,
            camera_person_count=guard.camera_person_count,
            direction_confidence=guard.direction_confidence,
            speaker_direction_stable=guard.speaker_direction_stable,
            ambiguous_room_reason=guard.reason,
            ambiguous_room_confidence=guard.claim.confidence,
        )
    return ReSpeakerSensitiveBehaviorGateDecision(
        allowed=True,
        reason=None,
        candidate_sensitivity=sensitivity,
        multi_person_likely=False,
        low_confidence_audio=False,
        audio_context_active=guard.audio_context_active,
        private_content_routing_allowed=True,
        camera_person_count=guard.camera_person_count,
        direction_confidence=guard.direction_confidence,
        speaker_direction_stable=guard.speaker_direction_stable,
        ambiguous_room_reason=None,
        ambiguous_room_confidence=guard.claim.confidence,
    )


def _normalize_sensitivity(value: object) -> str:
    try:
        return normalize_memory_sensitivity(str(value or "normal"))
    except Exception:
        return "sensitive"
__all__ = [
    "ReSpeakerSensitiveBehaviorGateDecision",
    "evaluate_respeaker_sensitive_behavior_gate",
]
