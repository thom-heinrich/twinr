"""Prioritize conservative HDMI attention targets from multimodal runtime signals.

This module keeps short-lived attention-target policy out of the proactive
orchestrator and the display cue renderer. It combines the current camera
anchor with speaker association, showing-intent signals, and bounded
session-focus memory so Twinr can look at the most relevant visible person
without pretending to identify secondary people in the room. Normal
attention-follow stays horizontal-only so stronger up/down body-language poses
remain reserved for explicit semantic face states.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from twinr.agent.base_agent.config import TwinrConfig
from twinr.proactive.runtime.claim_metadata import (
    coerce_mapping,
    coerce_optional_bool,
    coerce_optional_int,
    coerce_optional_ratio,
    mean_confidence,
    normalize_text,
)
from twinr.proactive.runtime.identity_fusion import MultimodalIdentityFusionSnapshot
from twinr.proactive.runtime.speaker_association import (
    ReSpeakerSpeakerAssociationSnapshot,
    derive_respeaker_speaker_association,
)


_DEFAULT_SESSION_FOCUS_HOLD_S = 4.5
_MIN_SESSION_FOCUS_HOLD_S = 0.5
_INTERACTIVE_RUNTIME_STATES = frozenset({"listening", "processing", "answering"})
_ACTIVE_RUNTIME_STATES = _INTERACTIVE_RUNTIME_STATES | {"waiting"}
_VALID_HORIZONTAL = frozenset({"left", "center", "right"})
_VALID_VERTICAL = frozenset({"up", "center", "down"})
_FOLLOW_VERTICAL = "center"
_LEFT_THRESHOLD = 0.36
_RIGHT_THRESHOLD = 0.64


@dataclass(frozen=True, slots=True)
class MultimodalAttentionTargetConfig:
    """Store bounded tuning values for multimodal HDMI attention targeting."""

    session_focus_hold_s: float = _DEFAULT_SESSION_FOCUS_HOLD_S

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "MultimodalAttentionTargetConfig":
        """Build one bounded targeting config from the global Twinr config."""

        try:
            hold_s = float(
                getattr(
                    config,
                    "display_attention_session_focus_hold_s",
                    _DEFAULT_SESSION_FOCUS_HOLD_S,
                )
                or 0.0
            )
        except (TypeError, ValueError):
            hold_s = _DEFAULT_SESSION_FOCUS_HOLD_S
        if hold_s != hold_s or hold_s <= 0.0:
            hold_s = _DEFAULT_SESSION_FOCUS_HOLD_S
        return cls(session_focus_hold_s=max(_MIN_SESSION_FOCUS_HOLD_S, hold_s))


@dataclass(frozen=True, slots=True)
class MultimodalAttentionTargetSnapshot:
    """Describe one conservative HDMI attention target."""

    observed_at: float | None = None
    state: str = "inactive"
    active: bool = False
    target_horizontal: str | None = None
    target_vertical: str | None = None
    target_zone: str | None = None
    focus_source: str = "none"
    runtime_status: str | None = None
    presence_session_id: int | None = None
    speaker_locked: bool = False
    session_focus_active: bool = False
    showing_intent_active: bool = False
    confidence: float = 0.0

    def to_automation_facts(self) -> dict[str, object]:
        """Serialize the snapshot into automation-friendly facts."""

        return {
            "observed_at": self.observed_at,
            "state": self.state,
            "active": self.active,
            "target_horizontal": self.target_horizontal,
            "target_vertical": self.target_vertical,
            "target_zone": self.target_zone,
            "focus_source": self.focus_source,
            "runtime_status": self.runtime_status,
            "presence_session_id": self.presence_session_id,
            "speaker_locked": self.speaker_locked,
            "session_focus_active": self.session_focus_active,
            "showing_intent_active": self.showing_intent_active,
            "confidence": self.confidence,
        }

    def event_data(self) -> dict[str, object]:
        """Serialize the snapshot into flat ops-event fields."""

        return {
            "attention_target_state": self.state,
            "attention_target_active": self.active,
            "attention_target_horizontal": self.target_horizontal,
            "attention_target_vertical": self.target_vertical,
            "attention_target_source": self.focus_source,
            "attention_target_speaker_locked": self.speaker_locked,
            "attention_target_session_focus_active": self.session_focus_active,
        }

    @classmethod
    def from_fact_map(
        cls,
        value: object | None,
    ) -> "MultimodalAttentionTargetSnapshot | None":
        """Parse one serialized attention-target payload conservatively."""

        payload = coerce_mapping(value)
        if not payload:
            return None
        confidence = coerce_optional_ratio(payload.get("confidence"))
        return cls(
            observed_at=_coerce_optional_float(payload.get("observed_at")),
            state=normalize_text(payload.get("state")) or "inactive",
            active=coerce_optional_bool(payload.get("active")) is True,
            target_horizontal=_normalize_direction(payload.get("target_horizontal"), allowed=_VALID_HORIZONTAL),
            target_vertical=_normalize_direction(payload.get("target_vertical"), allowed=_VALID_VERTICAL),
            target_zone=_normalize_direction(payload.get("target_zone"), allowed=_VALID_HORIZONTAL),
            focus_source=normalize_text(payload.get("focus_source")) or "none",
            runtime_status=normalize_text(payload.get("runtime_status")) or None,
            presence_session_id=coerce_optional_int(payload.get("presence_session_id")),
            speaker_locked=coerce_optional_bool(payload.get("speaker_locked")) is True,
            session_focus_active=coerce_optional_bool(payload.get("session_focus_active")) is True,
            showing_intent_active=coerce_optional_bool(payload.get("showing_intent_active")) is True,
            confidence=(0.0 if confidence is None else confidence),
        )


@dataclass(frozen=True, slots=True)
class _AttentionAnchor:
    horizontal: str
    vertical: str
    source: str


@dataclass(slots=True)
class _SessionFocusMemory:
    presence_session_id: int | None
    horizontal: str
    vertical: str
    source: str
    updated_at: float


class MultimodalAttentionTargetTracker:
    """Keep bounded session-focus memory for HDMI attention targeting."""

    def __init__(self, *, config: MultimodalAttentionTargetConfig) -> None:
        self.config = config
        self._focus_memory: _SessionFocusMemory | None = None

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "MultimodalAttentionTargetTracker":
        """Build one tracker from the global Twinr config."""

        return cls(config=MultimodalAttentionTargetConfig.from_config(config))

    def observe(
        self,
        *,
        observed_at: float | None,
        live_facts: Mapping[str, object] | None,
        runtime_status: object | None,
        presence_session_id: int | None,
        speaker_association: ReSpeakerSpeakerAssociationSnapshot | None = None,
        identity_fusion: MultimodalIdentityFusionSnapshot | None = None,
    ) -> MultimodalAttentionTargetSnapshot:
        """Return one prioritized HDMI attention target snapshot."""

        checked_at = 0.0 if observed_at is None else float(observed_at)
        normalized_runtime_status = normalize_text(runtime_status).lower() or None
        camera = coerce_mapping(None if live_facts is None else live_facts.get("camera"))
        vad = coerce_mapping(None if live_facts is None else live_facts.get("vad"))
        current_anchor = _camera_anchor(camera)
        speech_detected = coerce_optional_bool(vad.get("speech_detected")) is True
        current_speaker_association = speaker_association
        if current_speaker_association is None:
            current_speaker_association = derive_respeaker_speaker_association(
                observed_at=observed_at,
                live_facts=(live_facts or {}),
            )
        speaking_to_visible_person = (
            current_anchor is not None
            and speech_detected
            and current_speaker_association.associated
        )
        showing_intent_active = _showing_intent_active(camera)

        if speaking_to_visible_person:
            assert current_anchor is not None
            speaker_anchor = _AttentionAnchor(
                horizontal=current_anchor.horizontal,
                vertical=current_anchor.vertical,
                source="speaker_association",
            )
            self._remember_focus(
                observed_at=checked_at,
                presence_session_id=presence_session_id,
                anchor=speaker_anchor,
            )
            return _snapshot_from_anchor(
                observed_at=observed_at,
                runtime_status=normalized_runtime_status,
                presence_session_id=presence_session_id,
                anchor=speaker_anchor,
                state="active_visible_speaker",
                speaker_locked=True,
                confidence=mean_confidence(
                    (
                        current_speaker_association.confidence,
                        _camera_anchor_confidence(camera),
                    )
                )
                or 0.82,
            )

        if current_anchor is not None and showing_intent_active:
            self._remember_focus(
                observed_at=checked_at,
                presence_session_id=presence_session_id,
                anchor=_AttentionAnchor(
                    horizontal=current_anchor.horizontal,
                    vertical=current_anchor.vertical,
                    source="showing_intent",
                ),
            )
            return _snapshot_from_anchor(
                observed_at=observed_at,
                runtime_status=normalized_runtime_status,
                presence_session_id=presence_session_id,
                anchor=_AttentionAnchor(
                    horizontal=current_anchor.horizontal,
                    vertical=current_anchor.vertical,
                    source="showing_intent",
                ),
                state="showing_intent_visible_person",
                showing_intent_active=True,
                confidence=mean_confidence(
                    (
                        _camera_anchor_confidence(camera),
                        coerce_optional_ratio(camera.get("visual_attention_score")),
                        0.82,
                    )
                )
                or 0.82,
            )

        focus_anchor = self._resolve_focus_anchor(
            observed_at=checked_at,
            runtime_status=normalized_runtime_status,
            presence_session_id=presence_session_id,
            current_anchor=current_anchor,
            camera=camera,
            identity_fusion=identity_fusion,
        )
        if current_anchor is not None and focus_anchor is not None and _prefer_focus_anchor(
            camera=camera,
            runtime_status=normalized_runtime_status,
            focus_anchor=focus_anchor,
            current_anchor=current_anchor,
        ):
            return _snapshot_from_anchor(
                observed_at=observed_at,
                runtime_status=normalized_runtime_status,
                presence_session_id=presence_session_id,
                anchor=focus_anchor,
                state="session_focus_locked",
                session_focus_active=True,
                confidence=mean_confidence(
                    (
                        _focus_anchor_confidence(identity_fusion, camera),
                        _camera_anchor_confidence(camera),
                    )
                )
                or 0.78,
            )

        if current_anchor is not None:
            if focus_anchor is not None and _matches_focus(current_anchor, focus_anchor):
                return _snapshot_from_anchor(
                    observed_at=observed_at,
                    runtime_status=normalized_runtime_status,
                    presence_session_id=presence_session_id,
                    anchor=_AttentionAnchor(
                        horizontal=current_anchor.horizontal,
                        vertical=current_anchor.vertical,
                        source=focus_anchor.source,
                    ),
                    state="session_focus_visible_person",
                    session_focus_active=True,
                    confidence=mean_confidence(
                        (
                            _camera_anchor_confidence(camera),
                            _focus_anchor_confidence(identity_fusion, camera),
                        )
                    )
                    or 0.76,
                )
            return _snapshot_from_anchor(
                observed_at=observed_at,
                runtime_status=normalized_runtime_status,
                presence_session_id=presence_session_id,
                anchor=current_anchor,
                state="visible_primary_person",
                confidence=_camera_anchor_confidence(camera),
            )

        if focus_anchor is not None and _allow_focus_hold(camera=camera, runtime_status=normalized_runtime_status):
            return _snapshot_from_anchor(
                observed_at=observed_at,
                runtime_status=normalized_runtime_status,
                presence_session_id=presence_session_id,
                anchor=focus_anchor,
                state="holding_session_focus",
                session_focus_active=True,
                confidence=_focus_anchor_confidence(identity_fusion, camera),
            )

        return MultimodalAttentionTargetSnapshot(
            observed_at=observed_at,
            state="inactive",
            active=False,
            runtime_status=normalized_runtime_status,
            presence_session_id=presence_session_id,
            confidence=0.0,
        )

    def _remember_focus(
        self,
        *,
        observed_at: float,
        presence_session_id: int | None,
        anchor: _AttentionAnchor,
    ) -> None:
        self._focus_memory = _SessionFocusMemory(
            presence_session_id=presence_session_id,
            horizontal=anchor.horizontal,
            vertical=anchor.vertical,
            source=anchor.source,
            updated_at=observed_at,
        )

    def _resolve_focus_anchor(
        self,
        *,
        observed_at: float,
        runtime_status: str | None,
        presence_session_id: int | None,
        current_anchor: _AttentionAnchor | None,
        camera: Mapping[str, object],
        identity_fusion: MultimodalIdentityFusionSnapshot | None,
    ) -> _AttentionAnchor | None:
        identity_anchor = _identity_focus_anchor(
            identity_fusion=identity_fusion,
            presence_session_id=presence_session_id,
            current_anchor=current_anchor,
        )
        remembered = self._remembered_focus_anchor(
            observed_at=observed_at,
            presence_session_id=presence_session_id,
        )
        if identity_anchor is not None:
            if current_anchor is not None and _matches_focus(current_anchor, identity_anchor):
                self._remember_focus(
                    observed_at=observed_at,
                    presence_session_id=presence_session_id,
                    anchor=_AttentionAnchor(
                        horizontal=current_anchor.horizontal,
                        vertical=current_anchor.vertical,
                        source=identity_anchor.source,
                    ),
                )
                return _AttentionAnchor(
                    horizontal=current_anchor.horizontal,
                    vertical=current_anchor.vertical,
                    source=identity_anchor.source,
                )
            if remembered is None:
                return identity_anchor
        if remembered is None:
            return identity_anchor
        if runtime_status not in _ACTIVE_RUNTIME_STATES and not _allow_focus_hold(
            camera=camera,
            runtime_status=runtime_status,
        ):
            return None
        return remembered

    def _remembered_focus_anchor(
        self,
        *,
        observed_at: float,
        presence_session_id: int | None,
    ) -> _AttentionAnchor | None:
        memory = self._focus_memory
        if memory is None:
            return None
        if memory.presence_session_id != presence_session_id:
            return None
        if (observed_at - memory.updated_at) > self.config.session_focus_hold_s:
            return None
        return _AttentionAnchor(
            horizontal=memory.horizontal,
            vertical=memory.vertical,
            source=memory.source,
        )


def _snapshot_from_anchor(
    *,
    observed_at: float | None,
    runtime_status: str | None,
    presence_session_id: int | None,
    anchor: _AttentionAnchor,
    state: str,
    confidence: float,
    speaker_locked: bool = False,
    session_focus_active: bool = False,
    showing_intent_active: bool = False,
) -> MultimodalAttentionTargetSnapshot:
    return MultimodalAttentionTargetSnapshot(
        observed_at=observed_at,
        state=state,
        active=True,
        target_horizontal=anchor.horizontal,
        target_vertical=anchor.vertical,
        target_zone=anchor.horizontal,
        focus_source=anchor.source,
        runtime_status=runtime_status,
        presence_session_id=presence_session_id,
        speaker_locked=speaker_locked,
        session_focus_active=session_focus_active,
        showing_intent_active=showing_intent_active,
        confidence=round(max(0.0, min(1.0, confidence)), 4),
    )


def _camera_anchor(camera: Mapping[str, object]) -> _AttentionAnchor | None:
    if coerce_optional_bool(camera.get("person_visible")) is not True:
        return None
    horizontal = _camera_horizontal(camera)
    if horizontal is None:
        return None
    return _AttentionAnchor(
        horizontal=horizontal,
        vertical=_FOLLOW_VERTICAL,
        source="camera_primary_person",
    )


def _camera_horizontal(camera: Mapping[str, object]) -> str | None:
    if coerce_optional_bool(camera.get("primary_person_center_x_unknown")) is not True:
        center_x = coerce_optional_ratio(camera.get("primary_person_center_x"))
        if center_x is not None:
            if center_x <= _LEFT_THRESHOLD:
                return "left"
            if center_x >= _RIGHT_THRESHOLD:
                return "right"
            return "center"
    if coerce_optional_bool(camera.get("primary_person_zone_unknown")) is True:
        return None
    return _normalize_direction(camera.get("primary_person_zone"), allowed=_VALID_HORIZONTAL)


def _showing_intent_active(camera: Mapping[str, object]) -> bool:
    if coerce_optional_bool(camera.get("showing_intent_likely")) is True:
        return True
    return (
        coerce_optional_bool(camera.get("person_near_device")) is True
        and coerce_optional_bool(camera.get("looking_toward_device")) is True
    )


def _identity_focus_anchor(
    *,
    identity_fusion: MultimodalIdentityFusionSnapshot | None,
    presence_session_id: int | None,
    current_anchor: _AttentionAnchor | None,
) -> _AttentionAnchor | None:
    if identity_fusion is None:
        return None
    if identity_fusion.presence_session_id != presence_session_id:
        return None
    if identity_fusion.temporal_state != "stable_multimodal_match":
        return None
    if identity_fusion.track_consistency_state not in {"stable_anchor", "speaker_locked"}:
        return None
    horizontal = _normalize_direction(identity_fusion.track_anchor_zone, allowed=_VALID_HORIZONTAL)
    if horizontal is None:
        return None
    vertical = _FOLLOW_VERTICAL if current_anchor is None else current_anchor.vertical
    return _AttentionAnchor(
        horizontal=horizontal,
        vertical=vertical,
        source="identity_fusion_track",
    )


def _prefer_focus_anchor(
    *,
    camera: Mapping[str, object],
    runtime_status: str | None,
    focus_anchor: _AttentionAnchor,
    current_anchor: _AttentionAnchor,
) -> bool:
    if runtime_status not in _INTERACTIVE_RUNTIME_STATES:
        return False
    person_count = coerce_optional_int(camera.get("person_count"))
    if person_count is None or person_count <= 1:
        return False
    return not _matches_focus(focus_anchor, current_anchor)


def _allow_focus_hold(*, camera: Mapping[str, object], runtime_status: str | None) -> bool:
    if runtime_status in _INTERACTIVE_RUNTIME_STATES:
        return True
    return coerce_optional_bool(camera.get("person_recently_visible")) is True


def _matches_focus(left: _AttentionAnchor, right: _AttentionAnchor) -> bool:
    return left.horizontal == right.horizontal


def _camera_anchor_confidence(camera: Mapping[str, object]) -> float:
    return (
        mean_confidence(
            (
                coerce_optional_ratio(camera.get("visual_attention_score")),
                0.84 if coerce_optional_bool(camera.get("engaged_with_device")) is True else None,
                0.8 if coerce_optional_bool(camera.get("looking_toward_device")) is True else None,
                0.76 if coerce_optional_bool(camera.get("person_near_device")) is True else None,
            )
        )
        or 0.7
    )


def _focus_anchor_confidence(
    identity_fusion: MultimodalIdentityFusionSnapshot | None,
    camera: Mapping[str, object],
) -> float:
    if identity_fusion is not None:
        fusion_confidence = coerce_optional_ratio(identity_fusion.claim.confidence)
        if fusion_confidence is not None:
            return fusion_confidence
    return _camera_anchor_confidence(camera)


def _normalize_direction(value: object | None, *, allowed: frozenset[str]) -> str | None:
    normalized = normalize_text(value).lower()
    if normalized in allowed:
        return normalized
    return None


def _coerce_optional_float(value: object | None) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric


__all__ = [
    "MultimodalAttentionTargetConfig",
    "MultimodalAttentionTargetSnapshot",
    "MultimodalAttentionTargetTracker",
]
