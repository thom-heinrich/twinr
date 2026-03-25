"""Normalize compact multimodal voice-intent context for the voice gateway.

The Pi already derives a bounded ``person_state`` aggregate from camera, audio,
and proactive signals. The voice gateway does not need raw sensor payloads; it
only needs a compact summary that can bias audio-owned wake and follow-up
decisions without ever turning vision into a standalone wake trigger.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


def _coerce_mapping(value: object) -> Mapping[str, object]:
    """Return a read-only mapping view for mapping-like payloads."""

    if isinstance(value, Mapping):
        return value
    return {}


def _coerce_optional_bool(value: object) -> bool | None:
    """Normalize optional booleans from runtime facts and websocket payloads."""

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


def _coerce_optional_text(value: object) -> str | None:
    """Return one stripped scalar string or ``None`` when blank."""

    text = str(value or "").strip().lower()
    return text or None


def _axis_state(mapping: Mapping[str, object], key: str) -> str | None:
    """Extract one nested person-state axis label."""

    axis = _coerce_mapping(mapping.get(key))
    return _coerce_optional_text(axis.get("state"))


def _camera_presence_unavailable(camera: Mapping[str, object]) -> bool:
    """Return whether camera-derived visibility is currently unavailable."""

    return any(
        _coerce_optional_bool(camera.get(key)) is False
        for key in ("camera_online", "camera_ready", "camera_ai_ready")
    )


def _camera_unavailable_presence_hold_active(
    *,
    facts: Mapping[str, object],
    camera: Mapping[str, object],
) -> bool:
    """Report whether local near-device presence still attests a wakeable person.

    When the camera stack is temporarily unavailable, the live voice gateway
    still needs one bounded way to keep explicit wake armed for the same nearby
    person instead of collapsing to `person_visible=false` on every IMX500
    dropout. Keep that fallback narrow: require recent/near visibility evidence
    or qualifying speech while the local presence session remains armed.
    """

    near_device_presence = _coerce_mapping(facts.get("near_device_presence"))
    near_person_visible = _coerce_optional_bool(near_device_presence.get("person_visible"))
    near_person_recently_visible = _coerce_optional_bool(
        near_device_presence.get("person_recently_visible")
    )
    near_speech_recent = _coerce_optional_bool(near_device_presence.get("speech_recent"))
    near_voice_activation_armed = _coerce_optional_bool(
        near_device_presence.get("voice_activation_armed")
    )
    camera_person_recently_visible = _coerce_optional_bool(camera.get("person_recently_visible"))
    return any(
        (
            near_person_visible is True,
            near_person_recently_visible is True,
            camera_person_recently_visible is True,
            near_speech_recent is True and near_voice_activation_armed is True,
        )
    )


@dataclass(frozen=True, slots=True)
class VoiceRuntimeIntentContext:
    """Carry the compact multimodal context sent alongside voice runtime state."""

    attention_state: str | None = None
    interaction_intent_state: str | None = None
    person_visible: bool | None = None
    presence_active: bool | None = None
    interaction_ready: bool | None = None
    targeted_inference_blocked: bool | None = None
    recommended_channel: str | None = None

    @classmethod
    def from_sensor_facts(
        cls,
        facts: Mapping[str, object] | None,
    ) -> "VoiceRuntimeIntentContext":
        """Project the live sensor/person-state facts into one compact payload."""

        fact_mapping = _coerce_mapping(facts)
        person_state = _coerce_mapping(fact_mapping.get("person_state"))
        camera = _coerce_mapping(fact_mapping.get("camera"))
        attention_target = _coerce_mapping(fact_mapping.get("attention_target"))
        camera_person_visible = _coerce_optional_bool(camera.get("person_visible"))
        presence_active = _coerce_optional_bool(person_state.get("presence_active"))
        session_focus_active = _coerce_optional_bool(attention_target.get("session_focus_active"))
        person_visible: bool | None
        if camera_person_visible is True:
            person_visible = True
        elif presence_active is True and session_focus_active is True:
            # Keep transcript-first wake armed across short camera dropouts while
            # the Pi still holds focus on the same nearby person.
            person_visible = True
        elif camera_person_visible is False and _camera_presence_unavailable(camera):
            if _camera_unavailable_presence_hold_active(
                facts=fact_mapping,
                camera=camera,
            ):
                # The camera stack may report an explicit false while it is
                # offline or not AI-ready. Keep explicit wake available when
                # other local near-device facts still attest the same nearby
                # person.
                person_visible = True
            else:
                # When the camera is unavailable, explicit audio wake must not
                # be blocked purely because vision cannot currently confirm a
                # person. Mark visibility unknown instead of hard-false so the
                # waiting wake path stays audio-owned.
                person_visible = None
        elif camera_person_visible is None:
            person_visible = presence_active
        else:
            person_visible = camera_person_visible
        return cls(
            attention_state=_axis_state(person_state, "attention_state"),
            interaction_intent_state=_axis_state(person_state, "interaction_intent_state"),
            person_visible=person_visible,
            presence_active=presence_active,
            interaction_ready=_coerce_optional_bool(person_state.get("interaction_ready")),
            targeted_inference_blocked=_coerce_optional_bool(
                person_state.get("targeted_inference_blocked")
            ),
            recommended_channel=_coerce_optional_text(person_state.get("recommended_channel")),
        )

    @classmethod
    def from_runtime_event(cls, event: object) -> "VoiceRuntimeIntentContext":
        """Extract the same compact payload from one runtime-state event object."""

        return cls(
            attention_state=_coerce_optional_text(getattr(event, "attention_state", None)),
            interaction_intent_state=_coerce_optional_text(
                getattr(event, "interaction_intent_state", None)
            ),
            person_visible=_coerce_optional_bool(getattr(event, "person_visible", None)),
            presence_active=_coerce_optional_bool(getattr(event, "presence_active", None)),
            interaction_ready=_coerce_optional_bool(getattr(event, "interaction_ready", None)),
            targeted_inference_blocked=_coerce_optional_bool(
                getattr(event, "targeted_inference_blocked", None)
            ),
            recommended_channel=_coerce_optional_text(getattr(event, "recommended_channel", None)),
        )

    def to_event_fields(self) -> dict[str, object | None]:
        """Serialize the compact context into websocket event fields."""

        return {
            "attention_state": self.attention_state,
            "interaction_intent_state": self.interaction_intent_state,
            "person_visible": self.person_visible,
            "presence_active": self.presence_active,
            "interaction_ready": self.interaction_ready,
            "targeted_inference_blocked": self.targeted_inference_blocked,
            "recommended_channel": self.recommended_channel,
        }

    def trace_details(self) -> dict[str, object]:
        """Return trace-friendly details for structured voice instrumentation."""

        return {
            "intent_attention_state": self.attention_state,
            "intent_interaction_intent_state": self.interaction_intent_state,
            "intent_person_visible": self.person_visible,
            "intent_presence_active": self.presence_active,
            "intent_interaction_ready": self.interaction_ready,
            "intent_targeted_inference_blocked": self.targeted_inference_blocked,
            "intent_recommended_channel": self.recommended_channel,
            "intent_audio_bias_allowed": self.audio_bias_allowed(),
        }

    def audio_bias_allowed(self) -> bool:
        """Allow only an audio-owned wake/follow-up bias, never visual-only wake."""

        if self.targeted_inference_blocked is True:
            return False
        if self.person_visible is False:
            return False
        return self.interaction_ready is True and self.recommended_channel == "speech"

    def waiting_activation_allowed(self) -> bool:
        """Return whether idle transcript-first activation may scan this context.

        Remote-only wake scanning should still accept audio when the camera
        context is temporarily unknown. The explicit wake phrase should only be
        blocked when the Pi explicitly attests there is no local presence; a
        visible-person miss alone must not veto an explicit wake when the
        broader person-state aggregate still says someone is currently at the
        device.
        """

        if self.presence_active is True:
            return True
        if self.presence_active is False:
            return False
        if self.person_visible is False:
            return False
        return True
