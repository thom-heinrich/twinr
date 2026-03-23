"""Normalize compact multimodal voice-intent context for the voice gateway.

The Pi already derives a bounded ``person_state`` aggregate from camera, audio,
and proactive signals. The voice gateway does not need raw sensor payloads; it
only needs a compact summary that can bias audio-owned wake and follow-up
decisions without ever turning vision into a standalone wake trigger.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


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


@dataclass(frozen=True, slots=True)
class VoiceRuntimeIntentContext:
    """Carry the compact multimodal context sent alongside voice runtime state."""

    attention_state: str | None = None
    interaction_intent_state: str | None = None
    person_visible: bool | None = None
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
        person_visible = _coerce_optional_bool(camera.get("person_visible"))
        if person_visible is None:
            person_visible = _coerce_optional_bool(person_state.get("presence_active"))
        return cls(
            attention_state=_axis_state(person_state, "attention_state"),
            interaction_intent_state=_axis_state(person_state, "interaction_intent_state"),
            person_visible=person_visible,
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

