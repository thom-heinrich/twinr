"""Derive coarse affect proxies without claiming emotion as fact.

The output of this module is intentionally small and prompt-oriented. It may
support a calm follow-up question, but it must never be treated as a diagnosis,
emotion label, or durable truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from twinr.proactive.runtime.ambiguous_room_guard import (
    AmbiguousRoomGuardSnapshot,
    derive_ambiguous_room_guard,
)
from twinr.proactive.runtime.claim_metadata import (
    RuntimeClaimMetadata,
    coerce_mapping,
    coerce_optional_bool,
    coerce_optional_ratio,
    mean_confidence,
    normalize_text,
)


def _default_claim() -> RuntimeClaimMetadata:
    return RuntimeClaimMetadata(confidence=0.0, source="camera_pose_attention_audio", requires_confirmation=False)


@dataclass(frozen=True, slots=True)
class AffectProxySnapshot:
    """Describe one coarse affect proxy for prompt-level runtime use."""

    observed_at: float | None = None
    state: str = "unknown"
    policy_recommendation: str = "ignore"
    block_reason: str | None = None
    claim: RuntimeClaimMetadata = field(default_factory=_default_claim)
    body_pose: str | None = None
    smiling: bool | None = None
    looking_toward_device: bool | None = None
    engaged_with_device: bool | None = None
    room_quiet: bool | None = None
    low_motion: bool | None = None

    def to_automation_facts(self) -> dict[str, object]:
        """Serialize the affect proxy into automation-friendly facts."""

        payload = {
            "observed_at": self.observed_at,
            "state": self.state,
            "policy_recommendation": self.policy_recommendation,
            "block_reason": self.block_reason,
            "body_pose": self.body_pose,
            "smiling": self.smiling,
            "looking_toward_device": self.looking_toward_device,
            "engaged_with_device": self.engaged_with_device,
            "room_quiet": self.room_quiet,
            "low_motion": self.low_motion,
        }
        payload.update(self.claim.to_payload())
        return payload

    def event_data(self) -> dict[str, object]:
        """Serialize the affect proxy into compact flat event fields."""

        return {
            "affect_proxy_state": self.state,
            "affect_proxy_confidence": self.claim.confidence,
            "affect_proxy_policy": self.policy_recommendation,
        }


def derive_affect_proxy(
    *,
    observed_at: float | None,
    live_facts: dict[str, object] | object,
    ambiguous_room_guard: AmbiguousRoomGuardSnapshot | None = None,
) -> AffectProxySnapshot:
    """Return one conservative affect-proxy snapshot."""

    guard = ambiguous_room_guard or derive_ambiguous_room_guard(
        observed_at=observed_at,
        live_facts=live_facts,
    )
    facts = coerce_mapping(live_facts)
    camera = coerce_mapping(facts.get("camera"))
    vad = coerce_mapping(facts.get("vad"))
    pir = coerce_mapping(facts.get("pir"))

    person_visible = _known_bool(camera, "person_visible")
    body_pose = _known_text(camera, "body_pose")
    smiling = _known_bool(camera, "smiling")
    looking_toward_device = _known_bool(camera, "looking_toward_device")
    engaged_with_device = _known_bool(camera, "engaged_with_device")
    room_quiet = coerce_optional_bool(vad.get("room_quiet"))
    low_motion = coerce_optional_bool(pir.get("low_motion"))
    attention_score = _known_ratio(camera, "visual_attention_score")

    if guard.guard_active:
        return AffectProxySnapshot(
            observed_at=observed_at,
            state="unknown",
            policy_recommendation="ignore",
            block_reason=guard.reason,
            claim=RuntimeClaimMetadata(
                confidence=guard.claim.confidence,
                source="camera_pose_attention_audio_guarded",
                requires_confirmation=False,
            ),
            body_pose=body_pose,
            smiling=smiling,
            looking_toward_device=looking_toward_device,
            engaged_with_device=engaged_with_device,
            room_quiet=room_quiet,
            low_motion=low_motion,
        )
    if person_visible is not True:
        return AffectProxySnapshot(
            observed_at=observed_at,
            state="unknown",
            policy_recommendation="ignore",
            block_reason="no_visible_person",
            claim=RuntimeClaimMetadata(
                confidence=0.0,
                source="camera_pose_attention_audio",
                requires_confirmation=False,
            ),
            room_quiet=room_quiet,
            low_motion=low_motion,
        )
    if body_pose in {"floor", "lying_low"} and room_quiet is True:
        return AffectProxySnapshot(
            observed_at=observed_at,
            state="concern_cue",
            policy_recommendation="prompt_only",
            block_reason=None,
            claim=RuntimeClaimMetadata(
                confidence=0.84,
                source="camera_pose_plus_audio_quiet",
                requires_confirmation=True,
            ),
            body_pose=body_pose,
            smiling=smiling,
            looking_toward_device=looking_toward_device,
            engaged_with_device=engaged_with_device,
            room_quiet=room_quiet,
            low_motion=low_motion,
        )
    if body_pose == "slumped" and low_motion is True and room_quiet is True:
        return AffectProxySnapshot(
            observed_at=observed_at,
            state="concern_cue",
            policy_recommendation="prompt_only",
            block_reason=None,
            claim=RuntimeClaimMetadata(
                confidence=0.72,
                source="camera_pose_plus_pir_plus_audio",
                requires_confirmation=True,
            ),
            body_pose=body_pose,
            smiling=smiling,
            looking_toward_device=looking_toward_device,
            engaged_with_device=engaged_with_device,
            room_quiet=room_quiet,
            low_motion=low_motion,
        )
    if smiling is True and (looking_toward_device is True or engaged_with_device is True):
        confidence = mean_confidence((0.62, attention_score, 0.68 if engaged_with_device is True else None)) or 0.62
        return AffectProxySnapshot(
            observed_at=observed_at,
            state="positive_contact",
            policy_recommendation="prompt_only",
            block_reason=None,
            claim=RuntimeClaimMetadata(
                confidence=confidence,
                source="camera_expression_plus_attention",
                requires_confirmation=True,
            ),
            body_pose=body_pose,
            smiling=smiling,
            looking_toward_device=looking_toward_device,
            engaged_with_device=engaged_with_device,
            room_quiet=room_quiet,
            low_motion=low_motion,
        )
    if room_quiet is True and looking_toward_device is False and engaged_with_device is False:
        return AffectProxySnapshot(
            observed_at=observed_at,
            state="low_engagement",
            policy_recommendation="prompt_only",
            block_reason=None,
            claim=RuntimeClaimMetadata(
                confidence=0.58,
                source="camera_attention_plus_audio_quiet",
                requires_confirmation=True,
            ),
            body_pose=body_pose,
            smiling=smiling,
            looking_toward_device=looking_toward_device,
            engaged_with_device=engaged_with_device,
            room_quiet=room_quiet,
            low_motion=low_motion,
        )
    return AffectProxySnapshot(
        observed_at=observed_at,
        state="none",
        policy_recommendation="ignore",
        block_reason=None,
        claim=RuntimeClaimMetadata(
            confidence=0.64,
            source="camera_pose_attention_audio",
            requires_confirmation=False,
        ),
        body_pose=body_pose,
        smiling=smiling,
        looking_toward_device=looking_toward_device,
        engaged_with_device=engaged_with_device,
        room_quiet=room_quiet,
        low_motion=low_motion,
    )


def _known_bool(payload: dict[str, object], field_name: str) -> bool | None:
    """Return one optional bool only when the matching unknown flag is clear."""

    if coerce_optional_bool(payload.get(f"{field_name}_unknown")) is True:
        return None
    return coerce_optional_bool(payload.get(field_name))


def _known_text(payload: dict[str, object], field_name: str) -> str | None:
    """Return one optional text field only when the matching unknown flag is clear."""

    if coerce_optional_bool(payload.get(f"{field_name}_unknown")) is True:
        return None
    text = normalize_text(payload.get(field_name))
    return text or None


def _known_ratio(payload: dict[str, object], field_name: str) -> float | None:
    """Return one optional ratio only when the matching unknown flag is clear."""

    if coerce_optional_bool(payload.get(f"{field_name}_unknown")) is True:
        return None
    return coerce_optional_ratio(payload.get(field_name))


__all__ = [
    "AffectProxySnapshot",
    "derive_affect_proxy",
]
