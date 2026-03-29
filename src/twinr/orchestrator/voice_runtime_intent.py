# CHANGELOG: 2026-03-29
# BUG-1: Do not let stale session_focus override an explicit camera person_visible=false when the camera stack is healthy.
# BUG-2: Decode bytes/bytearray/memoryview payloads correctly for bool/text fields; the old code turned b"speech" into "b'speech'".
# BUG-3: Reject NaN/inf speaker confidences to avoid invalid wire/trace output.
# BUG-4: Accept mapping-based runtime events; the old from_runtime_event silently dropped dict payloads.
# SEC-1: Bound and sanitize scalar text inputs to prevent oversized websocket/runtime payloads from causing CPU churn and trace-cardinality explosion on Raspberry Pi.
# IMP-1: Add low-cardinality decision reasons for audio bias / speaker bias / waiting activation instrumentation.
# IMP-2: Add spoof-aware optional speaker gating plus compact event emission (omit_none / include_frontier_fields) for 2026-style edge voice pipelines.

"""Normalize compact multimodal voice-intent context for the voice gateway.

The Pi already derives a bounded ``person_state`` aggregate from camera, audio,
and proactive signals. The voice gateway does not need raw sensor payloads; it
only needs a compact summary that can bias audio-owned wake and follow-up
decisions without ever turning vision into a standalone wake trigger.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final


_STRONG_SPEAKER_BIAS_MIN_CONFIDENCE: Final[float] = 0.82
_MAX_TOKEN_LENGTH: Final[int] = 64
_MAX_BOOL_TEXT_LENGTH: Final[int] = 16
_MAX_RATIO_TEXT_LENGTH: Final[int] = 32
_SAFE_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9_.:/-]{0,63}$")
_EMPTY_MAPPING: Final[Mapping[str, object]] = MappingProxyType({})


def _coerce_mapping(value: object) -> Mapping[str, object]:
    """Return a mapping for mapping-like payloads, else one empty read-only mapping."""

    if isinstance(value, Mapping):
        return value
    return _EMPTY_MAPPING


def _read_field(source: object, key: str) -> object:
    """Read one field from either a mapping payload or an attribute object."""

    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def _normalize_scalar_text(value: object, *, max_length: int) -> str | None:
    """Normalize one scalar value into bounded lowercase text.

    Only simple scalar transport types are accepted here. Avoid calling ``str()``
    on arbitrary objects from runtime payloads so this hot path cannot be forced
    through attacker-controlled ``__str__`` methods, and cap the processed size
    to keep Pi-side CPU and trace payloads bounded.
    """

    if value is None:
        return None
    if isinstance(value, str):
        text = value
    elif isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if not raw:
            return None
        text = raw[: max_length * 4].decode("utf-8", errors="ignore")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        text = str(value)
    else:
        return None
    text = text.strip().lower()
    if not text:
        return None
    if len(text) > max_length:
        text = text[:max_length]
    return text


def _coerce_optional_bool(value: object) -> bool | None:
    """Normalize optional booleans from runtime facts and websocket payloads."""

    if isinstance(value, bool):
        return value
    normalized = _normalize_scalar_text(value, max_length=_MAX_BOOL_TEXT_LENGTH)
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    return None


def _coerce_optional_text(value: object) -> str | None:
    """Return one bounded token string or ``None`` when invalid/blank."""

    # BREAKING: free-form state/channel labels are no longer forwarded verbatim.
    # Only bounded safe tokens survive normalization; everything else becomes None.
    text = _normalize_scalar_text(value, max_length=_MAX_TOKEN_LENGTH)
    if text is None:
        return None
    if not _SAFE_TOKEN_RE.fullmatch(text):
        return None
    return text


def _coerce_optional_ratio(value: object) -> float | None:
    """Normalize one optional ratio into ``[0.0, 1.0]`` or ``None``."""

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        normalized = float(value)
    else:
        text = _normalize_scalar_text(value, max_length=_MAX_RATIO_TEXT_LENGTH)
        if text is None:
            return None
        try:
            normalized = float(text)
        except ValueError:
            return None
    if not math.isfinite(normalized):
        return None
    if normalized < 0.0 or normalized > 1.0:
        return None
    return normalized


def _coerce_first_optional_bool(source: object, *keys: str) -> bool | None:
    """Return the first non-``None`` optional bool found across alias keys."""

    for key in keys:
        value = _coerce_optional_bool(_read_field(source, key))
        if value is not None:
            return value
    return None


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
    person instead of collapsing to ``person_visible=false`` on every IMX500
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
    speaker_associated: bool | None = None
    speaker_association_confidence: float | None = None
    speaker_authentic: bool | None = None
    background_media_likely: bool | None = None
    speech_overlap_likely: bool | None = None

    @classmethod
    def from_sensor_facts(
        cls,
        facts: Mapping[str, object] | None,
    ) -> "VoiceRuntimeIntentContext":
        """Project the live sensor/person-state facts into one compact payload."""

        fact_mapping = _coerce_mapping(facts)
        person_state = _coerce_mapping(fact_mapping.get("person_state"))
        camera = _coerce_mapping(fact_mapping.get("camera"))
        speaker_association = _coerce_mapping(fact_mapping.get("speaker_association"))
        audio_policy = _coerce_mapping(fact_mapping.get("audio_policy"))
        attention_target = _coerce_mapping(fact_mapping.get("attention_target"))

        camera_person_visible = _coerce_optional_bool(camera.get("person_visible"))
        presence_active = _coerce_optional_bool(person_state.get("presence_active"))
        session_focus_active = _coerce_optional_bool(attention_target.get("session_focus_active"))
        camera_unavailable = _camera_presence_unavailable(camera)

        person_visible: bool | None
        if camera_person_visible is True:
            person_visible = True
        elif camera_person_visible is False:
            if camera_unavailable:
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
            else:
                # Respect a healthy explicit negative camera verdict. A stale focus
                # latch must not resurrect visibility after the person already left.
                person_visible = False
        else:
            if camera_unavailable and _camera_unavailable_presence_hold_active(
                facts=fact_mapping,
                camera=camera,
            ):
                person_visible = True
            elif presence_active is True and session_focus_active is True:
                # Keep transcript-first wake armed across short camera dropouts while
                # the Pi still holds focus on the same nearby person.
                person_visible = True
            else:
                person_visible = presence_active

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
            speaker_associated=_coerce_optional_bool(speaker_association.get("associated")),
            speaker_association_confidence=_coerce_optional_ratio(
                speaker_association.get("confidence")
            ),
            speaker_authentic=_coerce_first_optional_bool(
                speaker_association,
                "speaker_authentic",
                "authentic",
                "anti_spoof_passed",
            ),
            background_media_likely=_coerce_optional_bool(
                audio_policy.get("background_media_likely")
            ),
            speech_overlap_likely=_coerce_optional_bool(
                audio_policy.get("speech_overlap_likely")
            ),
        )

    @classmethod
    def from_runtime_event(cls, event: object) -> "VoiceRuntimeIntentContext":
        """Extract the same compact payload from one runtime-state event object."""

        return cls(
            attention_state=_coerce_optional_text(_read_field(event, "attention_state")),
            interaction_intent_state=_coerce_optional_text(
                _read_field(event, "interaction_intent_state")
            ),
            person_visible=_coerce_optional_bool(_read_field(event, "person_visible")),
            presence_active=_coerce_optional_bool(_read_field(event, "presence_active")),
            interaction_ready=_coerce_optional_bool(_read_field(event, "interaction_ready")),
            targeted_inference_blocked=_coerce_optional_bool(
                _read_field(event, "targeted_inference_blocked")
            ),
            recommended_channel=_coerce_optional_text(_read_field(event, "recommended_channel")),
            speaker_associated=_coerce_optional_bool(_read_field(event, "speaker_associated")),
            speaker_association_confidence=_coerce_optional_ratio(
                _read_field(event, "speaker_association_confidence")
            ),
            speaker_authentic=_coerce_first_optional_bool(
                event,
                "speaker_authentic",
                "authentic",
                "anti_spoof_passed",
            ),
            background_media_likely=_coerce_optional_bool(
                _read_field(event, "background_media_likely")
            ),
            speech_overlap_likely=_coerce_optional_bool(
                _read_field(event, "speech_overlap_likely")
            ),
        )

    def to_event_fields(
        self,
        *,
        omit_none: bool = False,
        include_frontier_fields: bool = False,
    ) -> dict[str, object | None]:
        """Serialize the compact context into websocket event fields."""

        fields: dict[str, object | None] = {
            "attention_state": self.attention_state,
            "interaction_intent_state": self.interaction_intent_state,
            "person_visible": self.person_visible,
            "presence_active": self.presence_active,
            "interaction_ready": self.interaction_ready,
            "targeted_inference_blocked": self.targeted_inference_blocked,
            "recommended_channel": self.recommended_channel,
            "speaker_associated": self.speaker_associated,
            "speaker_association_confidence": self.speaker_association_confidence,
            "background_media_likely": self.background_media_likely,
            "speech_overlap_likely": self.speech_overlap_likely,
        }
        if include_frontier_fields:
            # BREAKING: enabling ``include_frontier_fields`` adds new wire keys
            # for spoof-aware downstream policy engines.
            fields["speaker_authentic"] = self.speaker_authentic
            fields["speaker_association_band"] = self.speaker_association_band()
        if not omit_none:
            return fields
        return {key: value for key, value in fields.items() if value is not None}

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
            "intent_speaker_associated": self.speaker_associated,
            "intent_speaker_association_confidence": self.speaker_association_confidence,
            "intent_speaker_authentic": self.speaker_authentic,
            "intent_speaker_association_band": self.speaker_association_band(),
            "intent_background_media_likely": self.background_media_likely,
            "intent_speech_overlap_likely": self.speech_overlap_likely,
            "intent_audio_bias_allowed": self.audio_bias_allowed(),
            "intent_audio_bias_reason": self.audio_bias_reason(),
            "intent_strong_speaker_bias_allowed": self.strong_speaker_bias_allowed(),
            "intent_strong_speaker_bias_reason": self.strong_speaker_bias_reason(),
            "intent_familiar_speaker_bias_allowed": self.familiar_speaker_bias_allowed(),
            "intent_familiar_speaker_bias_reason": self.familiar_speaker_bias_reason(),
            "intent_waiting_activation_allowed": self.waiting_activation_allowed(),
            "intent_waiting_activation_reason": self.waiting_activation_reason(),
        }

    def audio_bias_allowed(self) -> bool:
        """Allow only an audio-owned wake/follow-up bias, never visual-only wake."""

        return self.audio_bias_reason() == "allowed"

    def audio_bias_reason(self) -> str:
        """Explain the audio-bias decision with one bounded reason code."""

        if self.targeted_inference_blocked is True:
            return "targeted_inference_blocked"
        if self.person_visible is False:
            return "person_not_visible"
        if self.interaction_ready is not True:
            return "interaction_not_ready"
        if self.recommended_channel != "speech":
            return "channel_not_speech"
        return "allowed"

    def strong_speaker_bias_allowed(self) -> bool:
        """Return whether waiting wake may use the stronger speaker-associated bias."""

        return self.strong_speaker_bias_reason() == "allowed"

    def strong_speaker_bias_reason(self) -> str:
        """Explain the strong-speaker-bias decision with one bounded reason code."""

        audio_reason = self.audio_bias_reason()
        if audio_reason != "allowed":
            return f"audio_bias_{audio_reason}"
        if self.speaker_authentic is False:
            return "speaker_authenticity_failed"
        if self.speaker_associated is not True:
            return "speaker_not_associated"
        if self.speaker_association_confidence is None:
            return "speaker_confidence_missing"
        if self.speaker_association_confidence < _STRONG_SPEAKER_BIAS_MIN_CONFIDENCE:
            return "speaker_confidence_below_threshold"
        return "allowed"

    def familiar_speaker_bias_allowed(self) -> bool:
        """Return whether known-speaker wake bias may apply in the current context."""

        return self.familiar_speaker_bias_reason() == "allowed"

    def familiar_speaker_bias_reason(self) -> str:
        """Explain the familiar-speaker-bias decision with one bounded reason code."""

        strong_reason = self.strong_speaker_bias_reason()
        if strong_reason != "allowed":
            return f"strong_speaker_bias_{strong_reason}"
        if self.background_media_likely is True:
            return "background_media_likely"
        if self.speech_overlap_likely is True:
            return "speech_overlap_likely"
        return "allowed"

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

    def waiting_activation_reason(self) -> str:
        """Explain the waiting-activation decision with one bounded reason code."""

        if self.presence_active is True:
            return "presence_active"
        if self.presence_active is False:
            return "presence_inactive"
        if self.person_visible is False:
            return "person_not_visible"
        return "allowed_without_presence"

    def speaker_association_band(self) -> str:
        """Return one low-cardinality band for tracing and optional wire export."""

        if self.speaker_authentic is False:
            return "spoof_rejected"
        if self.speaker_associated is not True:
            return "unassociated"
        if self.speaker_association_confidence is None:
            return "missing"
        if self.speaker_association_confidence >= _STRONG_SPEAKER_BIAS_MIN_CONFIDENCE:
            return "strong"
        return "below_strong"