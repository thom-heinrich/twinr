"""Aggregate specialized runtime facts into one explicit person-state surface.

This module does not replace the specialized proactive runtime layers such as
presence, audio policy, ambiguity guard, affect proxy, identity fusion, or
smart-home context. It reads those bounded fact surfaces and projects them into
one inspectable ``person_state`` schema with stable axes.

The intent is operational clarity:

- downstream policy gets one consistent per-axis contract
- each axis declares whether it is a derived state, proxy, or risk cue
- confidence and confirmation requirements stay explicit
- smart-home may enrich context, but never manufactures local interaction truth
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from twinr.proactive.runtime.affect_proxy import derive_affect_proxy
from twinr.proactive.runtime.ambiguous_room_guard import derive_ambiguous_room_guard
from twinr.proactive.runtime.claim_metadata import (
    RuntimeClaimMetadata,
    coerce_mapping,
    coerce_optional_bool,
    coerce_optional_ratio,
    mean_confidence,
    normalize_text,
)
from twinr.proactive.runtime.multimodal_initiative import (
    derive_respeaker_multimodal_initiative,
)
from twinr.proactive.runtime.speaker_association import (
    derive_respeaker_speaker_association,
)


_ALLOWED_AXIS_KINDS = frozenset({"observation", "derived_state", "proxy", "risk_cue"})
_NO_GESTURE_TOKENS = frozenset({"", "none", "unknown", "no_gesture", "no_hand_gesture"})
_CHANNELS = frozenset({"speech", "display", "print"})


def _default_axis_claim() -> RuntimeClaimMetadata:
    return RuntimeClaimMetadata(
        confidence=0.0,
        source="person_state_aggregate",
        requires_confirmation=False,
    )


@dataclass(frozen=True, slots=True)
class PersonStateAxisSnapshot:
    """Describe one person-state axis using a shared runtime contract."""

    axis: str
    kind: str
    observed_at: float | None = None
    state: str = "unknown"
    active: bool = False
    policy_recommendation: str = "observe"
    block_reason: str | None = None
    evidence: tuple[str, ...] = ()
    claim: RuntimeClaimMetadata = field(default_factory=_default_axis_claim)

    def __post_init__(self) -> None:
        """Normalize the axis schema into one compact immutable payload."""

        axis = normalize_text(self.axis) or "unknown_axis"
        kind = normalize_text(self.kind) or "derived_state"
        if kind not in _ALLOWED_AXIS_KINDS:
            raise ValueError(f"unsupported person-state axis kind: {kind}")
        state = normalize_text(self.state) or "unknown"
        policy = normalize_text(self.policy_recommendation) or "observe"
        block_reason = normalize_text(self.block_reason) or None
        evidence = tuple(
            dict.fromkeys(
                item
                for item in (normalize_text(value) for value in self.evidence)
                if item
            )
        )
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "active", self.active is True)
        object.__setattr__(self, "policy_recommendation", policy)
        object.__setattr__(self, "block_reason", block_reason)
        object.__setattr__(self, "evidence", evidence)

    def to_automation_facts(self) -> dict[str, object]:
        """Serialize the axis into nested automation-friendly facts."""

        payload = {
            "axis": self.axis,
            "kind": self.kind,
            "observed_at": self.observed_at,
            "state": self.state,
            "active": self.active,
            "policy_recommendation": self.policy_recommendation,
            "block_reason": self.block_reason,
            "evidence": list(self.evidence),
        }
        payload.update(self.claim.to_payload())
        return payload


@dataclass(frozen=True, slots=True)
class PersonStateSnapshot:
    """Bundle the current per-axis person-state view for the proactive runtime."""

    observed_at: float | None
    presence_state: PersonStateAxisSnapshot
    attention_state: PersonStateAxisSnapshot
    interaction_intent_state: PersonStateAxisSnapshot
    conversation_state: PersonStateAxisSnapshot
    safety_state: PersonStateAxisSnapshot
    identity_state: PersonStateAxisSnapshot
    room_clarity_state: PersonStateAxisSnapshot
    home_context_state: PersonStateAxisSnapshot
    schema_version: str = "v1"
    presence_active: bool = False
    interaction_ready: bool = False
    safety_concern_active: bool = False
    calm_personalization_allowed: bool = False
    targeted_inference_blocked: bool = True
    same_room_context_active: bool = False
    home_occupied_likely: bool = False
    recommended_channel: str = "display"

    def __post_init__(self) -> None:
        """Normalize summary booleans and the aggregate channel hint."""

        channel = normalize_text(self.recommended_channel).lower() or "display"
        if channel not in _CHANNELS:
            channel = "display"
        object.__setattr__(self, "presence_active", self.presence_active is True)
        object.__setattr__(self, "interaction_ready", self.interaction_ready is True)
        object.__setattr__(self, "safety_concern_active", self.safety_concern_active is True)
        object.__setattr__(self, "calm_personalization_allowed", self.calm_personalization_allowed is True)
        object.__setattr__(self, "targeted_inference_blocked", self.targeted_inference_blocked is True)
        object.__setattr__(self, "same_room_context_active", self.same_room_context_active is True)
        object.__setattr__(self, "home_occupied_likely", self.home_occupied_likely is True)
        object.__setattr__(self, "recommended_channel", channel)

    def to_automation_facts(self) -> dict[str, object]:
        """Serialize the aggregate person-state snapshot into runtime facts."""

        return {
            "observed_at": self.observed_at,
            "schema_version": self.schema_version,
            "presence_active": self.presence_active,
            "interaction_ready": self.interaction_ready,
            "safety_concern_active": self.safety_concern_active,
            "calm_personalization_allowed": self.calm_personalization_allowed,
            "targeted_inference_blocked": self.targeted_inference_blocked,
            "same_room_context_active": self.same_room_context_active,
            "home_occupied_likely": self.home_occupied_likely,
            "recommended_channel": self.recommended_channel,
            "presence_state": self.presence_state.to_automation_facts(),
            "attention_state": self.attention_state.to_automation_facts(),
            "interaction_intent_state": self.interaction_intent_state.to_automation_facts(),
            "conversation_state": self.conversation_state.to_automation_facts(),
            "safety_state": self.safety_state.to_automation_facts(),
            "identity_state": self.identity_state.to_automation_facts(),
            "room_clarity_state": self.room_clarity_state.to_automation_facts(),
            "home_context_state": self.home_context_state.to_automation_facts(),
        }

    def event_data(self) -> dict[str, object]:
        """Serialize the aggregate into compact flat ops-event fields."""

        return {
            "person_state_presence_state": self.presence_state.state,
            "person_state_attention_state": self.attention_state.state,
            "person_state_interaction_state": self.interaction_intent_state.state,
            "person_state_conversation_state": self.conversation_state.state,
            "person_state_safety_state": self.safety_state.state,
            "person_state_identity_state": self.identity_state.state,
            "person_state_room_clarity_state": self.room_clarity_state.state,
            "person_state_home_context_state": self.home_context_state.state,
            "person_state_presence_active": self.presence_active,
            "person_state_interaction_ready": self.interaction_ready,
            "person_state_safety_concern_active": self.safety_concern_active,
            "person_state_calm_personalization_allowed": self.calm_personalization_allowed,
            "person_state_targeted_inference_blocked": self.targeted_inference_blocked,
            "person_state_recommended_channel": self.recommended_channel,
        }


def derive_person_state(
    *,
    observed_at: float | None,
    live_facts: Mapping[str, object] | object,
) -> PersonStateSnapshot:
    """Return one bounded person-state aggregate from current runtime facts."""

    facts = coerce_mapping(live_facts)
    guard = coerce_mapping(facts.get("ambiguous_room_guard"))
    if not guard:
        try:
            guard = derive_ambiguous_room_guard(
                observed_at=observed_at,
                live_facts=facts,
            ).to_automation_facts()
        except Exception:
            # Background smart-home updates may arrive without any camera/audio
            # payload at all. Keep the aggregate fail-closed in that case
            # instead of requiring every partial producer to satisfy the full
            # ambiguity-guard contract.
            guard = {
                "observed_at": observed_at,
                "clear": False,
                "guard_active": True,
                "reason": "no_visible_person",
                "policy_recommendation": "block_targeted_inference",
                "confidence": 0.78,
                "source": "person_state_fallback_guard",
                "source_type": "derived",
                "requires_confirmation": False,
            }
    speaker_association = coerce_mapping(facts.get("speaker_association"))
    if not speaker_association:
        speaker_association = derive_respeaker_speaker_association(
            observed_at=observed_at,
            live_facts=facts,
        ).to_automation_facts()
    multimodal_initiative = coerce_mapping(facts.get("multimodal_initiative"))
    if not multimodal_initiative:
        multimodal_initiative = derive_respeaker_multimodal_initiative(
            observed_at=observed_at,
            live_facts=facts,
        ).to_automation_facts()
    affect_proxy = coerce_mapping(facts.get("affect_proxy"))
    if not affect_proxy:
        if any(coerce_mapping(facts.get(key)) for key in ("camera", "pir", "vad", "audio_policy")):
            try:
                affect_proxy = derive_affect_proxy(
                    observed_at=observed_at,
                    live_facts=facts,
                ).to_automation_facts()
            except Exception:
                affect_proxy = {}

    presence_state = _derive_presence_state(observed_at=observed_at, facts=facts)
    room_clarity_state = _derive_room_clarity_state(
        observed_at=observed_at,
        guard=guard,
    )
    attention_state = _derive_attention_state(
        observed_at=observed_at,
        facts=facts,
    )
    interaction_intent_state = _derive_interaction_intent_state(
        observed_at=observed_at,
        facts=facts,
        speaker_association=speaker_association,
        multimodal_initiative=multimodal_initiative,
    )
    conversation_state = _derive_conversation_state(
        observed_at=observed_at,
        facts=facts,
        speaker_association=speaker_association,
    )
    safety_state = _derive_safety_state(
        observed_at=observed_at,
        facts=facts,
        affect_proxy=affect_proxy,
    )
    identity_state = _derive_identity_state(
        observed_at=observed_at,
        facts=facts,
        guard=guard,
    )
    home_context_state = _derive_home_context_state(
        observed_at=observed_at,
        facts=facts,
    )

    interaction_ready = (
        presence_state.active
        and attention_state.active
        and interaction_intent_state.state in {"explicit_gesture_request", "showing_intent", "multimodal_turn_ready"}
        and conversation_state.state not in {"room_busy_or_overlapping", "speech_without_clear_target"}
        and room_clarity_state.state == "clear_targetable_room"
    )
    safety_concern_active = safety_state.active and safety_state.state in {
        "concern_cue",
        "floor_pose_quiet",
        "slumped_quiet_cue",
    }
    calm_personalization_allowed = identity_state.policy_recommendation in {
        "calm_personalization_only",
        "personalization_ok",
    }
    targeted_inference_blocked = room_clarity_state.policy_recommendation != "clear"
    same_room_context_active = home_context_state.state == "same_room_supportive_activity"
    home_occupied_likely = home_context_state.state in {
        "same_room_supportive_activity",
        "home_occupied",
    } or coerce_optional_bool(coerce_mapping(facts.get("home_context")).get("home_occupied_likely")) is True
    recommended_channel = _recommended_channel(
        interaction_intent_state=interaction_intent_state,
        conversation_state=conversation_state,
        safety_state=safety_state,
        targeted_inference_blocked=targeted_inference_blocked,
    )
    return PersonStateSnapshot(
        observed_at=observed_at,
        presence_state=presence_state,
        attention_state=attention_state,
        interaction_intent_state=interaction_intent_state,
        conversation_state=conversation_state,
        safety_state=safety_state,
        identity_state=identity_state,
        room_clarity_state=room_clarity_state,
        home_context_state=home_context_state,
        presence_active=presence_state.active,
        interaction_ready=interaction_ready,
        safety_concern_active=safety_concern_active,
        calm_personalization_allowed=calm_personalization_allowed,
        targeted_inference_blocked=targeted_inference_blocked,
        same_room_context_active=same_room_context_active,
        home_occupied_likely=home_occupied_likely,
        recommended_channel=recommended_channel,
    )


def _derive_presence_state(
    *,
    observed_at: float | None,
    facts: Mapping[str, object],
) -> PersonStateAxisSnapshot:
    near = coerce_mapping(facts.get("near_device_presence"))
    camera = coerce_mapping(facts.get("camera"))
    pir = coerce_mapping(facts.get("pir"))
    sensor = coerce_mapping(facts.get("sensor"))
    vad = coerce_mapping(facts.get("vad"))
    audio_policy = coerce_mapping(facts.get("audio_policy"))

    if near:
        occupied_likely = coerce_optional_bool(near.get("occupied_likely")) is True
        person_visible = coerce_optional_bool(near.get("person_visible")) is True
        person_recently_visible = coerce_optional_bool(near.get("person_recently_visible")) is True
        room_motion_recent = coerce_optional_bool(near.get("room_motion_recent")) is True
        speech_recent = coerce_optional_bool(near.get("speech_recent")) is True
        voice_activation_armed = coerce_optional_bool(near.get("voice_activation_armed")) is True
        reason = normalize_text(near.get("reason")) or None
        evidence = _evidence(
            ("near_device_presence.person_visible", person_visible),
            ("near_device_presence.person_recently_visible", person_recently_visible),
            ("near_device_presence.room_motion_recent", room_motion_recent),
            ("near_device_presence.speech_recent", speech_recent),
            ("near_device_presence.voice_activation_armed", voice_activation_armed),
        )
        confidence = _payload_confidence(near) or _fallback_presence_confidence(
            person_visible=person_visible,
            recent_visible=person_recently_visible,
            motion_recent=room_motion_recent,
            speech_recent=speech_recent,
            voice_activation_armed=voice_activation_armed,
        )
        if person_visible:
            state = "occupied_visible"
            active = True
            policy = "local_interaction_ok"
        elif occupied_likely:
            state = "occupied_recent"
            active = True
            policy = "local_interaction_ok"
        else:
            state = "empty"
            active = False
            policy = "observe"
        return _axis(
            axis="presence_state",
            kind="derived_state",
            observed_at=observed_at,
            state=state,
            active=active,
            policy_recommendation=policy,
            block_reason=None if active else (reason or "no_local_presence_signal"),
            confidence=confidence,
            source=normalize_text(near.get("source")) or "near_device_presence",
            source_type=normalize_text(near.get("source_type")) or "observed",
            requires_confirmation=coerce_optional_bool(near.get("requires_confirmation")) is True,
            evidence=evidence,
        )

    person_visible = _known_bool(camera, "person_visible") is True
    person_recently_visible = coerce_optional_bool(camera.get("person_recently_visible")) is True
    motion_recent = coerce_optional_bool(pir.get("motion_detected")) is True
    speech_recent = any(
        (
            coerce_optional_bool(audio_policy.get("presence_audio_active")) is True,
            coerce_optional_bool(audio_policy.get("recent_follow_up_speech")) is True,
            coerce_optional_bool(vad.get("speech_detected")) is True,
        )
    )
    voice_activation_armed = coerce_optional_bool(sensor.get("voice_activation_armed")) is True
    evidence = _evidence(
        ("camera.person_visible", person_visible),
        ("camera.person_recently_visible", person_recently_visible),
        ("pir.motion_detected", motion_recent),
        ("audio_policy.presence_audio_active", coerce_optional_bool(audio_policy.get("presence_audio_active")) is True),
        ("audio_policy.recent_follow_up_speech", coerce_optional_bool(audio_policy.get("recent_follow_up_speech")) is True),
        ("sensor.voice_activation_armed", voice_activation_armed),
    )
    if person_visible:
        return _axis(
            axis="presence_state",
            kind="derived_state",
            observed_at=observed_at,
            state="occupied_visible",
            active=True,
            policy_recommendation="local_interaction_ok",
            confidence=_fallback_presence_confidence(
                person_visible=person_visible,
                recent_visible=person_recently_visible,
                motion_recent=motion_recent,
                speech_recent=speech_recent,
                voice_activation_armed=voice_activation_armed,
            ),
            source="local_runtime_facts",
            evidence=evidence,
        )
    if person_recently_visible or motion_recent or (speech_recent and voice_activation_armed):
        return _axis(
            axis="presence_state",
            kind="derived_state",
            observed_at=observed_at,
            state="occupied_recent",
            active=True,
            policy_recommendation="local_interaction_ok",
            confidence=_fallback_presence_confidence(
                person_visible=False,
                recent_visible=person_recently_visible,
                motion_recent=motion_recent,
                speech_recent=speech_recent,
                voice_activation_armed=voice_activation_armed,
            ),
            source="local_runtime_facts",
            evidence=evidence,
        )
    if speech_recent or voice_activation_armed:
        return _axis(
            axis="presence_state",
            kind="derived_state",
            observed_at=observed_at,
            state="possible_presence",
            active=False,
            policy_recommendation="observe",
            block_reason="uncorroborated_local_presence",
            confidence=0.46,
            source="local_runtime_facts",
            evidence=evidence,
        )
    return _axis(
        axis="presence_state",
        kind="derived_state",
        observed_at=observed_at,
        state="empty",
        active=False,
        policy_recommendation="observe",
        block_reason="no_local_presence_signal",
        confidence=0.74,
        source="local_runtime_facts",
        evidence=evidence,
    )


def _derive_attention_state(
    *,
    observed_at: float | None,
    facts: Mapping[str, object],
) -> PersonStateAxisSnapshot:
    camera = coerce_mapping(facts.get("camera"))
    target = coerce_mapping(facts.get("attention_target"))
    person_visible = _known_bool(camera, "person_visible") is True
    engaged = _known_bool(camera, "engaged_with_device") is True
    looking = _known_bool(camera, "looking_toward_device") is True
    near = _known_bool(camera, "person_near_device") is True
    target_active = coerce_optional_bool(target.get("active")) is True
    confidence = mean_confidence(
        (
            _known_ratio(camera, "visual_attention_score"),
            _payload_confidence(target),
            0.88 if engaged else None,
            0.8 if looking else None,
        )
    ) or (0.0 if not person_visible else 0.62)
    evidence = _evidence(
        ("camera.person_visible", person_visible),
        ("camera.engaged_with_device", engaged),
        ("camera.looking_toward_device", looking),
        ("camera.person_near_device", near),
        ("attention_target.active", target_active),
    )
    if not person_visible:
        return _axis(
            axis="attention_state",
            kind="derived_state",
            observed_at=observed_at,
            state="inactive",
            active=False,
            policy_recommendation="observe",
            block_reason="no_visible_person",
            confidence=0.0,
            source="camera_attention",
            evidence=evidence,
        )
    if engaged:
        return _axis(
            axis="attention_state",
            kind="derived_state",
            observed_at=observed_at,
            state="engaged_with_device",
            active=True,
            policy_recommendation="speech_ok",
            confidence=confidence,
            source="camera_attention_plus_attention_target" if target else "camera_attention",
            evidence=evidence,
        )
    if looking or target_active:
        return _axis(
            axis="attention_state",
            kind="derived_state",
            observed_at=observed_at,
            state="attending_to_device",
            active=True,
            policy_recommendation="display_or_speech",
            confidence=confidence,
            source="camera_attention_plus_attention_target" if target else "camera_attention",
            evidence=evidence,
        )
    return _axis(
        axis="attention_state",
        kind="derived_state",
        observed_at=observed_at,
        state="visible_unfocused" if near else "visible_passive",
        active=False,
        policy_recommendation="display_only",
        confidence=confidence,
        source="camera_attention",
        evidence=evidence,
    )


def _derive_interaction_intent_state(
    *,
    observed_at: float | None,
    facts: Mapping[str, object],
    speaker_association: Mapping[str, object],
    multimodal_initiative: Mapping[str, object],
) -> PersonStateAxisSnapshot:
    camera = coerce_mapping(facts.get("camera"))
    showing_intent = _known_bool(camera, "showing_intent_likely") is True
    hand_near = _known_bool(camera, "hand_or_object_near_camera") is True
    fine_hand_gesture = normalize_text(camera.get("fine_hand_gesture")).lower()
    coarse_gesture = normalize_text(camera.get("coarse_arm_gesture") or camera.get("gesture_event")).lower()
    initiative_ready = coerce_optional_bool(multimodal_initiative.get("ready")) is True
    associated = coerce_optional_bool(speaker_association.get("associated")) is True
    fine_confidence = _known_ratio(camera, "fine_hand_gesture_confidence")
    coarse_confidence = _known_ratio(camera, "coarse_arm_gesture_confidence")
    evidence = _evidence(
        ("camera.showing_intent_likely", showing_intent),
        ("camera.hand_or_object_near_camera", hand_near),
        ("camera.fine_hand_gesture", fine_hand_gesture not in _NO_GESTURE_TOKENS),
        ("camera.coarse_arm_gesture", coarse_gesture not in _NO_GESTURE_TOKENS),
        ("multimodal_initiative.ready", initiative_ready),
        ("speaker_association.associated", associated),
    )
    if fine_hand_gesture not in _NO_GESTURE_TOKENS or coarse_gesture not in _NO_GESTURE_TOKENS:
        return _axis(
            axis="interaction_intent_state",
            kind="proxy",
            observed_at=observed_at,
            state="explicit_gesture_request",
            active=True,
            policy_recommendation="respond_promptly",
            confidence=mean_confidence((fine_confidence, coarse_confidence, 0.84)) or 0.84,
            source="camera_gesture",
            requires_confirmation=True,
            evidence=evidence,
        )
    if showing_intent and hand_near:
        return _axis(
            axis="interaction_intent_state",
            kind="proxy",
            observed_at=observed_at,
            state="showing_intent",
            active=True,
            policy_recommendation="respond_promptly",
            confidence=mean_confidence((_known_ratio(camera, "visual_attention_score"), 0.78)) or 0.78,
            source="camera_showing_intent",
            requires_confirmation=True,
            evidence=evidence,
        )
    if initiative_ready and associated:
        return _axis(
            axis="interaction_intent_state",
            kind="proxy",
            observed_at=observed_at,
            state="multimodal_turn_ready",
            active=True,
            policy_recommendation=_normalize_channel(multimodal_initiative.get("recommended_channel")),
            confidence=_payload_confidence(multimodal_initiative) or 0.74,
            source=normalize_text(multimodal_initiative.get("source")) or "multimodal_initiative",
            evidence=evidence,
        )
    if hand_near:
        return _axis(
            axis="interaction_intent_state",
            kind="proxy",
            observed_at=observed_at,
            state="possible_intent",
            active=False,
            policy_recommendation="display_first",
            confidence=0.56,
            source="camera_showing_intent",
            requires_confirmation=True,
            evidence=evidence,
        )
    return _axis(
        axis="interaction_intent_state",
        kind="proxy",
        observed_at=observed_at,
        state="passive",
        active=False,
        policy_recommendation="observe",
        confidence=0.66,
        source="camera_showing_intent",
        evidence=evidence,
    )


def _derive_conversation_state(
    *,
    observed_at: float | None,
    facts: Mapping[str, object],
    speaker_association: Mapping[str, object],
) -> PersonStateAxisSnapshot:
    vad = coerce_mapping(facts.get("vad"))
    audio_policy = coerce_mapping(facts.get("audio_policy"))
    room_busy = coerce_optional_bool(audio_policy.get("room_busy_or_overlapping")) is True
    presence_audio_active = coerce_optional_bool(audio_policy.get("presence_audio_active")) is True
    recent_follow_up_speech = coerce_optional_bool(audio_policy.get("recent_follow_up_speech")) is True
    quiet_window_open = coerce_optional_bool(audio_policy.get("quiet_window_open")) is True
    speech_detected = coerce_optional_bool(vad.get("speech_detected")) is True
    associated = coerce_optional_bool(speaker_association.get("associated")) is True
    evidence = _evidence(
        ("vad.speech_detected", speech_detected),
        ("audio_policy.presence_audio_active", presence_audio_active),
        ("audio_policy.recent_follow_up_speech", recent_follow_up_speech),
        ("audio_policy.quiet_window_open", quiet_window_open),
        ("audio_policy.room_busy_or_overlapping", room_busy),
        ("speaker_association.associated", associated),
    )
    confidence = mean_confidence(
        (
            _payload_confidence(audio_policy),
            _payload_confidence(speaker_association),
            0.84 if presence_audio_active else None,
            0.76 if recent_follow_up_speech else None,
        )
    ) or 0.62
    if room_busy:
        return _axis(
            axis="conversation_state",
            kind="derived_state",
            observed_at=observed_at,
            state="room_busy_or_overlapping",
            active=True,
            policy_recommendation="defer_speech",
            block_reason="room_busy_or_overlapping",
            confidence=confidence,
            source="audio_policy",
            evidence=evidence,
        )
    if presence_audio_active and associated:
        return _axis(
            axis="conversation_state",
            kind="derived_state",
            observed_at=observed_at,
            state="device_directed_speech",
            active=True,
            policy_recommendation="speech_ok",
            confidence=confidence,
            source="audio_policy_plus_speaker_association",
            evidence=evidence,
        )
    if recent_follow_up_speech and associated:
        return _axis(
            axis="conversation_state",
            kind="derived_state",
            observed_at=observed_at,
            state="follow_up_window",
            active=True,
            policy_recommendation="speech_ok",
            confidence=confidence,
            source="audio_policy_plus_speaker_association",
            evidence=evidence,
        )
    if speech_detected:
        return _axis(
            axis="conversation_state",
            kind="derived_state",
            observed_at=observed_at,
            state="speech_without_clear_target",
            active=True,
            policy_recommendation="display_first",
            block_reason=normalize_text(speaker_association.get("state")) or "speaker_target_unclear",
            confidence=confidence,
            source="audio_policy_plus_speaker_association",
            evidence=evidence,
        )
    if quiet_window_open:
        return _axis(
            axis="conversation_state",
            kind="derived_state",
            observed_at=observed_at,
            state="quiet_window",
            active=False,
            policy_recommendation="observe",
            confidence=max(0.66, confidence),
            source="audio_policy",
            evidence=evidence,
        )
    return _axis(
        axis="conversation_state",
        kind="derived_state",
        observed_at=observed_at,
        state="idle",
        active=False,
        policy_recommendation="observe",
        confidence=0.64,
        source="audio_policy",
        evidence=evidence,
    )


def _derive_safety_state(
    *,
    observed_at: float | None,
    facts: Mapping[str, object],
    affect_proxy: Mapping[str, object],
) -> PersonStateAxisSnapshot:
    camera = coerce_mapping(facts.get("camera"))
    pir = coerce_mapping(facts.get("pir"))
    vad = coerce_mapping(facts.get("vad"))
    affect_state = normalize_text(affect_proxy.get("state")) or "unknown"
    body_pose = normalize_text(affect_proxy.get("body_pose")) or _known_text(camera, "body_pose") or "unknown"
    room_quiet = coerce_optional_bool(affect_proxy.get("room_quiet"))
    if room_quiet is None:
        room_quiet = coerce_optional_bool(vad.get("room_quiet"))
    low_motion = coerce_optional_bool(affect_proxy.get("low_motion"))
    if low_motion is None:
        low_motion = coerce_optional_bool(pir.get("low_motion"))
    evidence = _evidence(
        ("affect_proxy.concern_cue", affect_state == "concern_cue"),
        ("camera.body_pose_floor", body_pose in {"floor", "lying_low"}),
        ("camera.body_pose_slumped", body_pose == "slumped"),
        ("vad.room_quiet", room_quiet is True),
        ("pir.low_motion", low_motion is True),
    )
    if affect_state == "concern_cue":
        return _axis(
            axis="safety_state",
            kind="risk_cue",
            observed_at=observed_at,
            state="concern_cue",
            active=True,
            policy_recommendation=normalize_text(affect_proxy.get("policy_recommendation")) or "prompt_only",
            confidence=_payload_confidence(affect_proxy) or 0.72,
            source=normalize_text(affect_proxy.get("source")) or "affect_proxy",
            source_type=normalize_text(affect_proxy.get("source_type")) or "observed",
            requires_confirmation=coerce_optional_bool(affect_proxy.get("requires_confirmation")) is not False,
            evidence=evidence,
        )
    if body_pose in {"floor", "lying_low"} and room_quiet is True:
        return _axis(
            axis="safety_state",
            kind="risk_cue",
            observed_at=observed_at,
            state="floor_pose_quiet",
            active=True,
            policy_recommendation="prompt_only",
            confidence=0.78,
            source="camera_pose_plus_audio_quiet",
            requires_confirmation=True,
            evidence=evidence,
        )
    if body_pose == "slumped" and room_quiet is True and low_motion is True:
        return _axis(
            axis="safety_state",
            kind="risk_cue",
            observed_at=observed_at,
            state="slumped_quiet_cue",
            active=True,
            policy_recommendation="prompt_only",
            confidence=0.7,
            source="camera_pose_plus_pir_plus_audio",
            requires_confirmation=True,
            evidence=evidence,
        )
    return _axis(
        axis="safety_state",
        kind="risk_cue",
        observed_at=observed_at,
        state="none",
        active=False,
        policy_recommendation="ignore",
        block_reason=normalize_text(affect_proxy.get("block_reason")) or None,
        confidence=max(0.58, _payload_confidence(affect_proxy) or 0.58),
        source=normalize_text(affect_proxy.get("source")) or "affect_proxy",
        source_type=normalize_text(affect_proxy.get("source_type")) or "observed",
        requires_confirmation=coerce_optional_bool(affect_proxy.get("requires_confirmation")) is True,
        evidence=evidence,
    )


def _derive_identity_state(
    *,
    observed_at: float | None,
    facts: Mapping[str, object],
    guard: Mapping[str, object],
) -> PersonStateAxisSnapshot:
    known_user_hint = coerce_mapping(facts.get("known_user_hint"))
    identity_fusion = coerce_mapping(facts.get("identity_fusion"))
    portrait_match = coerce_mapping(facts.get("portrait_match"))
    guard_active = coerce_optional_bool(guard.get("guard_active")) is True
    guard_reason = normalize_text(guard.get("reason")) or None
    evidence = _evidence(
        ("known_user_hint.matches_main_user", coerce_optional_bool(known_user_hint.get("matches_main_user")) is True),
        ("identity_fusion.matches_main_user", coerce_optional_bool(identity_fusion.get("matches_main_user")) is True),
        ("portrait_match.matches_reference_user", coerce_optional_bool(portrait_match.get("matches_reference_user")) is True),
        ("ambiguous_room_guard.guard_active", guard_active),
    )
    if coerce_optional_bool(known_user_hint.get("matches_main_user")) is True:
        return _axis(
            axis="identity_state",
            kind="proxy",
            observed_at=observed_at,
            state="likely_main_user",
            active=True,
            policy_recommendation=normalize_text(known_user_hint.get("policy_recommendation")) or "calm_personalization_only",
            confidence=_payload_confidence(known_user_hint) or 0.72,
            source=normalize_text(known_user_hint.get("source")) or "known_user_hint",
            source_type=normalize_text(known_user_hint.get("source_type")) or "observed",
            requires_confirmation=coerce_optional_bool(known_user_hint.get("requires_confirmation")) is True,
            evidence=evidence,
        )
    if coerce_optional_bool(identity_fusion.get("matches_main_user")) is True:
        return _axis(
            axis="identity_state",
            kind="proxy",
            observed_at=observed_at,
            state="likely_main_user",
            active=True,
            policy_recommendation=normalize_text(identity_fusion.get("policy_recommendation")) or "confirm_first",
            confidence=_payload_confidence(identity_fusion) or 0.74,
            source=normalize_text(identity_fusion.get("source")) or "identity_fusion",
            source_type=normalize_text(identity_fusion.get("source_type")) or "observed",
            requires_confirmation=coerce_optional_bool(identity_fusion.get("requires_confirmation")) is not False,
            evidence=evidence,
        )
    known_user_state = normalize_text(known_user_hint.get("state")) or "unknown"
    portrait_state = normalize_text(portrait_match.get("state")) or "unknown"
    if known_user_state in {"other_enrolled_user_visible", "known_other_user"} or portrait_state == "known_other_user":
        return _axis(
            axis="identity_state",
            kind="proxy",
            observed_at=observed_at,
            state="known_other_user",
            active=True,
            policy_recommendation="blocked",
            block_reason=known_user_state if known_user_state != "unknown" else portrait_state,
            confidence=max(_payload_confidence(known_user_hint) or 0.0, _payload_confidence(portrait_match) or 0.0, 0.68),
            source="known_user_hint" if known_user_state != "unknown" else "portrait_match",
            requires_confirmation=True,
            evidence=evidence,
        )
    if guard_active:
        return _axis(
            axis="identity_state",
            kind="proxy",
            observed_at=observed_at,
            state="blocked_ambiguous_room" if guard_reason else "blocked",
            active=False,
            policy_recommendation="blocked",
            block_reason=guard_reason or "targeted_inference_blocked",
            confidence=_payload_confidence(guard) or 0.78,
            source=normalize_text(guard.get("source")) or "ambiguous_room_guard",
            source_type=normalize_text(guard.get("source_type")) or "observed",
            requires_confirmation=True,
            evidence=evidence,
        )
    if coerce_optional_bool(portrait_match.get("matches_reference_user")) is True:
        return _axis(
            axis="identity_state",
            kind="proxy",
            observed_at=observed_at,
            state="portrait_match_unconfirmed",
            active=True,
            policy_recommendation=normalize_text(portrait_match.get("policy_recommendation")) or "confirm_first",
            confidence=_payload_confidence(portrait_match) or 0.7,
            source=normalize_text(portrait_match.get("source")) or "portrait_match",
            source_type=normalize_text(portrait_match.get("source_type")) or "observed",
            requires_confirmation=coerce_optional_bool(portrait_match.get("requires_confirmation")) is not False,
            evidence=evidence,
        )
    return _axis(
        axis="identity_state",
        kind="proxy",
        observed_at=observed_at,
        state="unknown",
        active=False,
        policy_recommendation="observe",
        block_reason=normalize_text(known_user_hint.get("block_reason")) or None,
        confidence=max(_payload_confidence(known_user_hint) or 0.0, _payload_confidence(identity_fusion) or 0.0, 0.42),
        source="identity_runtime_facts",
        requires_confirmation=True,
        evidence=evidence,
    )


def _derive_room_clarity_state(
    *,
    observed_at: float | None,
    guard: Mapping[str, object],
) -> PersonStateAxisSnapshot:
    guard_active = coerce_optional_bool(guard.get("guard_active")) is True
    clear = coerce_optional_bool(guard.get("clear")) is True
    reason = normalize_text(guard.get("reason")) or None
    evidence = _evidence(
        ("ambiguous_room_guard.guard_active", guard_active),
        ("ambiguous_room_guard.clear", clear),
    )
    if clear and not guard_active:
        return _axis(
            axis="room_clarity_state",
            kind="derived_state",
            observed_at=observed_at,
            state="clear_targetable_room",
            active=True,
            policy_recommendation="clear",
            confidence=_payload_confidence(guard) or 0.82,
            source=normalize_text(guard.get("source")) or "ambiguous_room_guard",
            source_type=normalize_text(guard.get("source_type")) or "observed",
            evidence=evidence,
        )
    return _axis(
        axis="room_clarity_state",
        kind="derived_state",
        observed_at=observed_at,
        state=reason or "targeted_inference_blocked",
        active=False,
        policy_recommendation=normalize_text(guard.get("policy_recommendation")) or "block_targeted_inference",
        block_reason=reason,
        confidence=_payload_confidence(guard) or 0.74,
        source=normalize_text(guard.get("source")) or "ambiguous_room_guard",
        source_type=normalize_text(guard.get("source_type")) or "observed",
        evidence=evidence,
    )


def _derive_home_context_state(
    *,
    observed_at: float | None,
    facts: Mapping[str, object],
) -> PersonStateAxisSnapshot:
    room_context = coerce_mapping(facts.get("room_context"))
    home_context = coerce_mapping(facts.get("home_context"))
    same_room_motion_recent = coerce_optional_bool(room_context.get("same_room_motion_recent")) is True
    same_room_button_recent = coerce_optional_bool(room_context.get("same_room_button_recent")) is True
    home_occupied = coerce_optional_bool(home_context.get("home_occupied_likely")) is True
    alarm_active = coerce_optional_bool(home_context.get("alarm_active")) is True
    device_offline = coerce_optional_bool(home_context.get("device_offline")) is True
    available = room_context or home_context
    evidence = _evidence(
        ("room_context.same_room_motion_recent", same_room_motion_recent),
        ("room_context.same_room_button_recent", same_room_button_recent),
        ("home_context.home_occupied_likely", home_occupied),
        ("home_context.alarm_active", alarm_active),
        ("home_context.device_offline", device_offline),
    )
    confidence = max(_payload_confidence(room_context) or 0.0, _payload_confidence(home_context) or 0.0, 0.56 if available else 0.0)
    if not available:
        return _axis(
            axis="home_context_state",
            kind="derived_state",
            observed_at=observed_at,
            state="unavailable",
            active=False,
            policy_recommendation="context_unavailable",
            block_reason="smart_home_unavailable",
            confidence=0.0,
            source="smart_home_context",
            evidence=evidence,
        )
    if alarm_active:
        return _axis(
            axis="home_context_state",
            kind="derived_state",
            observed_at=observed_at,
            state="alarm_active",
            active=True,
            policy_recommendation="defer_nonessential",
            confidence=confidence,
            source="smart_home_context",
            evidence=evidence,
        )
    if device_offline:
        return _axis(
            axis="home_context_state",
            kind="derived_state",
            observed_at=observed_at,
            state="device_offline",
            active=True,
            policy_recommendation="context_only",
            confidence=confidence,
            source="smart_home_context",
            evidence=evidence,
        )
    if same_room_motion_recent or same_room_button_recent:
        return _axis(
            axis="home_context_state",
            kind="derived_state",
            observed_at=observed_at,
            state="same_room_supportive_activity",
            active=True,
            policy_recommendation="context_support",
            confidence=confidence,
            source="smart_home_context",
            evidence=evidence,
        )
    if home_occupied:
        return _axis(
            axis="home_context_state",
            kind="derived_state",
            observed_at=observed_at,
            state="home_occupied",
            active=True,
            policy_recommendation="context_only",
            confidence=confidence,
            source="smart_home_context",
            evidence=evidence,
        )
    return _axis(
        axis="home_context_state",
        kind="derived_state",
        observed_at=observed_at,
        state="home_idle",
        active=False,
        policy_recommendation="context_only",
        confidence=confidence,
        source="smart_home_context",
        evidence=evidence,
    )


def _recommended_channel(
    *,
    interaction_intent_state: PersonStateAxisSnapshot,
    conversation_state: PersonStateAxisSnapshot,
    safety_state: PersonStateAxisSnapshot,
    targeted_inference_blocked: bool,
) -> str:
    """Return one bounded channel hint from the aggregate axes."""

    if safety_state.active and safety_state.policy_recommendation == "prompt_only" and not targeted_inference_blocked:
        return "speech"
    if conversation_state.state in {"device_directed_speech", "follow_up_window"}:
        return "speech"
    if interaction_intent_state.state in {"explicit_gesture_request", "multimodal_turn_ready"}:
        return "speech" if not targeted_inference_blocked else "display"
    return "display"


def _axis(
    *,
    axis: str,
    kind: str,
    observed_at: float | None,
    state: str,
    active: bool,
    policy_recommendation: str,
    confidence: float,
    source: str,
    source_type: str = "observed",
    requires_confirmation: bool = False,
    block_reason: str | None = None,
    evidence: tuple[str, ...] = (),
) -> PersonStateAxisSnapshot:
    """Build one normalized axis snapshot from plain values."""

    return PersonStateAxisSnapshot(
        axis=axis,
        kind=kind,
        observed_at=observed_at,
        state=state,
        active=active,
        policy_recommendation=policy_recommendation,
        block_reason=block_reason,
        evidence=evidence,
        claim=RuntimeClaimMetadata(
            confidence=confidence,
            source=source,
            source_type=source_type,
            requires_confirmation=requires_confirmation,
        ),
    )


def _payload_confidence(payload: Mapping[str, object]) -> float | None:
    """Return one optional confidence from any serialized runtime claim payload."""

    return coerce_optional_ratio(payload.get("confidence"))


def _fallback_presence_confidence(
    *,
    person_visible: bool,
    recent_visible: bool,
    motion_recent: bool,
    speech_recent: bool,
    voice_activation_armed: bool,
) -> float:
    """Estimate a conservative presence confidence from local corroboration only."""

    return mean_confidence(
        (
            0.92 if person_visible else None,
            0.8 if recent_visible else None,
            0.7 if motion_recent else None,
            0.68 if speech_recent else None,
            0.64 if voice_activation_armed else None,
        )
    ) or 0.0


def _known_bool(payload: Mapping[str, object], field_name: str) -> bool | None:
    """Return one optional bool only when the paired unknown flag is clear."""

    if coerce_optional_bool(payload.get(f"{field_name}_unknown")) is True:
        return None
    return coerce_optional_bool(payload.get(field_name))


def _known_ratio(payload: Mapping[str, object], field_name: str) -> float | None:
    """Return one optional ratio only when the paired unknown flag is clear."""

    if coerce_optional_bool(payload.get(f"{field_name}_unknown")) is True:
        return None
    return coerce_optional_ratio(payload.get(field_name))


def _known_text(payload: Mapping[str, object], field_name: str) -> str | None:
    """Return one optional text field only when the paired unknown flag is clear."""

    if coerce_optional_bool(payload.get(f"{field_name}_unknown")) is True:
        return None
    text = normalize_text(payload.get(field_name))
    return text or None


def _evidence(*items: tuple[str, bool]) -> tuple[str, ...]:
    """Collect the enabled supporting fact labels for one aggregate axis."""

    return tuple(label for label, enabled in items if enabled)


def _normalize_channel(value: object | None) -> str:
    """Normalize one optional channel hint into the small runtime vocabulary."""

    normalized = normalize_text(value).lower()
    if normalized not in _CHANNELS:
        return "display"
    return normalized


__all__ = [
    "PersonStateAxisSnapshot",
    "PersonStateSnapshot",
    "derive_person_state",
]
