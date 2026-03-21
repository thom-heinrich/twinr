"""Track short-lived visible person anchors for continuous HDMI attention.

This module keeps person-anchor matching, motion recency, short hold behavior,
and bounded audio-to-vision handoff out of the higher-level attention policy.
It does not try to identify people. It only keeps a tiny rolling notion of
"which visible anchor is probably the relevant one right now" so Twinr can
follow a person more continuously, prefer the active speaker in simple
multi-person scenes, and otherwise look at the most recently moving person.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from twinr.agent.base_agent.config import TwinrConfig
from twinr.proactive.runtime.claim_metadata import (
    coerce_mapping,
    coerce_optional_bool,
    coerce_optional_ratio,
    normalize_text,
)


_DEFAULT_TRACK_STALE_S = 1.2
_DEFAULT_TARGET_HOLD_S = 0.9
_DEFAULT_MOTION_RECENCY_S = 1.8
_DEFAULT_PREDICTION_HORIZON_S = 0.18
_DEFAULT_MATCH_MAX_DISTANCE = 0.28
_DEFAULT_MOTION_DELTA = 0.03
_DEFAULT_SPEAKER_MAX_DISTANCE = 0.26
_DEFAULT_DIRECTION_MIN_CONFIDENCE = 0.72
_DEFAULT_DIRECTION_MIN_TRACKS = 2
_DEFAULT_AUDIO_MIRROR_MIN_SAMPLES = 3
_DEFAULT_AUDIO_MIRROR_MARGIN = 0.08
_VALID_HORIZONTAL = frozenset({"left", "center", "right"})


@dataclass(frozen=True, slots=True)
class ContinuousAttentionTargetConfig:
    """Store bounded tuning values for continuous visible-person targeting."""

    track_stale_s: float = _DEFAULT_TRACK_STALE_S
    target_hold_s: float = _DEFAULT_TARGET_HOLD_S
    motion_recency_s: float = _DEFAULT_MOTION_RECENCY_S
    prediction_horizon_s: float = _DEFAULT_PREDICTION_HORIZON_S
    match_max_distance: float = _DEFAULT_MATCH_MAX_DISTANCE
    motion_delta: float = _DEFAULT_MOTION_DELTA
    speaker_max_distance: float = _DEFAULT_SPEAKER_MAX_DISTANCE
    direction_min_confidence: float = _DEFAULT_DIRECTION_MIN_CONFIDENCE
    audio_mirror_min_samples: int = _DEFAULT_AUDIO_MIRROR_MIN_SAMPLES
    audio_mirror_margin: float = _DEFAULT_AUDIO_MIRROR_MARGIN

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "ContinuousAttentionTargetConfig":
        """Build one bounded config from the global Twinr config."""

        def _bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
            try:
                value = float(getattr(config, name, default) or default)
            except (TypeError, ValueError):
                value = default
            if value != value:
                value = default
            return max(minimum, min(maximum, value))

        return cls(
            track_stale_s=_bounded_float("display_attention_track_stale_s", _DEFAULT_TRACK_STALE_S, 0.2, 4.0),
            target_hold_s=_bounded_float("display_attention_target_hold_s", _DEFAULT_TARGET_HOLD_S, 0.1, 3.0),
            motion_recency_s=_bounded_float("display_attention_motion_recency_s", _DEFAULT_MOTION_RECENCY_S, 0.2, 5.0),
            prediction_horizon_s=_bounded_float(
                "display_attention_prediction_horizon_s",
                _DEFAULT_PREDICTION_HORIZON_S,
                0.0,
                0.5,
            ),
            match_max_distance=_bounded_float(
                "display_attention_track_match_distance",
                _DEFAULT_MATCH_MAX_DISTANCE,
                0.05,
                0.6,
            ),
            motion_delta=_bounded_float("display_attention_motion_delta", _DEFAULT_MOTION_DELTA, 0.005, 0.2),
            speaker_max_distance=_bounded_float(
                "display_attention_speaker_track_distance",
                _DEFAULT_SPEAKER_MAX_DISTANCE,
                0.08,
                0.5,
            ),
            direction_min_confidence=_bounded_float(
                "display_attention_direction_min_confidence",
                _DEFAULT_DIRECTION_MIN_CONFIDENCE,
                0.4,
                1.0,
            ),
            audio_mirror_min_samples=max(
                1,
                int(getattr(config, "display_attention_audio_mirror_min_samples", _DEFAULT_AUDIO_MIRROR_MIN_SAMPLES) or _DEFAULT_AUDIO_MIRROR_MIN_SAMPLES),
            ),
            audio_mirror_margin=_bounded_float(
                "display_attention_audio_mirror_margin",
                _DEFAULT_AUDIO_MIRROR_MARGIN,
                0.0,
                0.3,
            ),
        )


@dataclass(frozen=True, slots=True)
class ContinuousAttentionTargetSnapshot:
    """Describe one short-lived continuous visible-person target."""

    observed_at: float | None = None
    state: str = "inactive"
    active: bool = False
    focus_source: str = "none"
    target_track_id: str | None = None
    target_center_x: float | None = None
    target_center_y: float | None = None
    target_zone: str | None = None
    target_horizontal: str | None = None
    target_velocity_x: float | None = None
    speaker_locked: bool = False
    motion_locked: bool = False
    visible_track_count: int = 0
    audio_target_x: float | None = None
    audio_mirror_applied: bool = False
    confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class _VisiblePersonAnchor:
    center_x: float
    center_y: float
    zone: str
    confidence: float


@dataclass(slots=True)
class _TrackState:
    track_id: str
    center_x: float
    center_y: float
    zone: str
    confidence: float
    updated_at: float
    last_motion_at: float | None
    velocity_x: float = 0.0

    def predicted_center_x(self, *, horizon_s: float) -> float:
        """Return one bounded short-horizon predicted x coordinate."""

        return _clamp_ratio(self.center_x + (self.velocity_x * max(0.0, horizon_s)))


class ContinuousAttentionTracker:
    """Keep bounded visible-person tracks for local HDMI attention follow."""

    def __init__(self, *, config: ContinuousAttentionTargetConfig) -> None:
        self.config = config
        self._tracks: dict[str, _TrackState] = {}
        self._next_track_index = 1
        self._last_selected_track_id: str | None = None
        self._last_selected_at: float | None = None
        self._audio_mirror_score = 0

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "ContinuousAttentionTracker":
        """Build one tracker from the global Twinr config."""

        return cls(config=ContinuousAttentionTargetConfig.from_config(config))

    def observe(
        self,
        *,
        observed_at: float | None,
        live_facts: Mapping[str, object] | None,
    ) -> ContinuousAttentionTargetSnapshot:
        """Update tracks and return the current visible-person attention target."""

        checked_at = 0.0 if observed_at is None else float(observed_at)
        camera = coerce_mapping(None if live_facts is None else live_facts.get("camera"))
        vad = coerce_mapping(None if live_facts is None else live_facts.get("vad"))
        respeaker = coerce_mapping(None if live_facts is None else live_facts.get("respeaker"))
        audio_policy = coerce_mapping(None if live_facts is None else live_facts.get("audio_policy"))

        anchors = _visible_person_anchors(camera)
        active_tracks = self._update_tracks(observed_at=checked_at, anchors=anchors)
        if not active_tracks:
            self._last_selected_track_id = None
            self._last_selected_at = None
            return ContinuousAttentionTargetSnapshot(
                observed_at=observed_at,
                state="inactive",
                active=False,
                visible_track_count=0,
            )

        speech_detected = coerce_optional_bool(vad.get("speech_detected")) is True
        direction_confidence = coerce_optional_ratio(respeaker.get("direction_confidence"))
        speaker_direction_stable = coerce_optional_bool(audio_policy.get("speaker_direction_stable")) is True
        azimuth_deg = _coerce_optional_int(respeaker.get("azimuth_deg"))

        audio_target_x = None
        if azimuth_deg is not None:
            audio_target_x = _folded_audio_target_x(azimuth_deg)
            if self._audio_mirror_enabled():
                audio_target_x = 1.0 - audio_target_x

        self._update_audio_mirror_calibration(
            speech_detected=speech_detected,
            speaker_direction_stable=speaker_direction_stable,
            direction_confidence=direction_confidence,
            azimuth_deg=azimuth_deg,
            tracks=active_tracks,
        )

        speaker_track = self._speaker_track(
            tracks=active_tracks,
            speech_detected=speech_detected,
            speaker_direction_stable=speaker_direction_stable,
            direction_confidence=direction_confidence,
            audio_target_x=audio_target_x,
        )
        if speaker_track is not None:
            return self._select_snapshot(
                observed_at=observed_at,
                track=speaker_track,
                state="active_visible_speaker_track",
                focus_source="speaker_track",
                speaker_locked=True,
                visible_track_count=len(active_tracks),
                audio_target_x=audio_target_x,
                confidence=_mean_confidence((speaker_track.confidence, direction_confidence)) or 0.82,
            )

        moved_track = self._last_moved_track(observed_at=checked_at, tracks=active_tracks)
        if moved_track is not None and len(active_tracks) >= _DEFAULT_DIRECTION_MIN_TRACKS:
            return self._select_snapshot(
                observed_at=observed_at,
                track=moved_track,
                state="last_moved_visible_person",
                focus_source="last_motion_track",
                motion_locked=True,
                visible_track_count=len(active_tracks),
                audio_target_x=audio_target_x,
                confidence=max(0.64, moved_track.confidence),
            )

        held_track = self._held_track(observed_at=checked_at, tracks=active_tracks)
        if held_track is not None:
            return self._select_snapshot(
                observed_at=observed_at,
                track=held_track,
                state="held_visible_person",
                focus_source="track_hold",
                visible_track_count=len(active_tracks),
                audio_target_x=audio_target_x,
                confidence=max(0.6, held_track.confidence),
            )

        primary_track = active_tracks[0]
        return self._select_snapshot(
            observed_at=observed_at,
            track=primary_track,
            state="active_visible_person",
            focus_source="primary_visible_person",
            visible_track_count=len(active_tracks),
            audio_target_x=audio_target_x,
            confidence=max(0.6, primary_track.confidence),
        )

    def _update_tracks(
        self,
        *,
        observed_at: float,
        anchors: tuple[_VisiblePersonAnchor, ...],
    ) -> list[_TrackState]:
        active_tracks = [
            track
            for track in self._tracks.values()
            if (observed_at - track.updated_at) <= self.config.track_stale_s
        ]
        active_tracks.sort(key=lambda item: item.updated_at, reverse=True)
        next_tracks: dict[str, _TrackState] = {}
        used_track_ids: set[str] = set()
        rendered: list[_TrackState] = []

        for anchor in anchors:
            best_track = None
            best_distance = None
            for track in active_tracks:
                if track.track_id in used_track_ids:
                    continue
                distance = _track_distance(track=track, anchor=anchor)
                if distance > self.config.match_max_distance:
                    continue
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_track = track
            if best_track is None:
                track = _TrackState(
                    track_id=f"visible_track_{self._next_track_index}",
                    center_x=anchor.center_x,
                    center_y=anchor.center_y,
                    zone=anchor.zone,
                    confidence=anchor.confidence,
                    updated_at=observed_at,
                    last_motion_at=None,
                    velocity_x=0.0,
                )
                self._next_track_index += 1
            else:
                dt = max(0.001, observed_at - best_track.updated_at)
                raw_velocity_x = (anchor.center_x - best_track.center_x) / dt
                velocity_x = (best_track.velocity_x * 0.55) + (raw_velocity_x * 0.45)
                moved = (
                    abs(anchor.center_x - best_track.center_x) >= self.config.motion_delta
                    or abs(anchor.center_y - best_track.center_y) >= self.config.motion_delta
                )
                track = _TrackState(
                    track_id=best_track.track_id,
                    center_x=anchor.center_x,
                    center_y=anchor.center_y,
                    zone=anchor.zone,
                    confidence=max(anchor.confidence, best_track.confidence * 0.7),
                    updated_at=observed_at,
                    last_motion_at=(observed_at if moved else best_track.last_motion_at),
                    velocity_x=velocity_x,
                )
                used_track_ids.add(best_track.track_id)
            next_tracks[track.track_id] = track
            rendered.append(track)

        # Keep recently unmatched tracks alive for the bounded stale window so
        # short detector misses do not collapse a real multi-person scene into a
        # false single-person speaker lock on the next speech frame.
        for track in active_tracks:
            if track.track_id in used_track_ids:
                continue
            next_tracks[track.track_id] = track
            rendered.append(track)

        self._tracks = next_tracks
        rendered.sort(key=lambda item: (item.updated_at, item.confidence), reverse=True)
        return rendered

    def _speaker_track(
        self,
        *,
        tracks: list[_TrackState],
        speech_detected: bool,
        speaker_direction_stable: bool,
        direction_confidence: float | None,
        audio_target_x: float | None,
    ) -> _TrackState | None:
        if not speech_detected:
            return None
        if len(tracks) == 1:
            return tracks[0]
        if speaker_direction_stable is not True:
            return None
        if direction_confidence is None or direction_confidence < self.config.direction_min_confidence:
            return None
        if audio_target_x is None:
            return None
        candidate = min(
            tracks,
            key=lambda item: abs(item.predicted_center_x(horizon_s=self.config.prediction_horizon_s) - audio_target_x),
        )
        if abs(candidate.predicted_center_x(horizon_s=self.config.prediction_horizon_s) - audio_target_x) > self.config.speaker_max_distance:
            return None
        return candidate

    def _last_moved_track(
        self,
        *,
        observed_at: float,
        tracks: list[_TrackState],
    ) -> _TrackState | None:
        recent = [
            track
            for track in tracks
            if track.last_motion_at is not None
            and (observed_at - track.last_motion_at) <= self.config.motion_recency_s
        ]
        if not recent:
            return None
        recent.sort(
            key=lambda item: (
                item.last_motion_at,
                item.confidence,
            ),
            reverse=True,
        )
        return recent[0]

    def _held_track(
        self,
        *,
        observed_at: float,
        tracks: list[_TrackState],
    ) -> _TrackState | None:
        if self._last_selected_track_id is None or self._last_selected_at is None:
            return None
        if (observed_at - self._last_selected_at) > self.config.target_hold_s:
            return None
        for track in tracks:
            if track.track_id == self._last_selected_track_id:
                return track
        return None

    def _select_snapshot(
        self,
        *,
        observed_at: float | None,
        track: _TrackState,
        state: str,
        focus_source: str,
        visible_track_count: int,
        audio_target_x: float | None,
        confidence: float,
        speaker_locked: bool = False,
        motion_locked: bool = False,
    ) -> ContinuousAttentionTargetSnapshot:
        self._last_selected_track_id = track.track_id
        self._last_selected_at = 0.0 if observed_at is None else float(observed_at)
        predicted_center_x = track.predicted_center_x(horizon_s=self.config.prediction_horizon_s)
        return ContinuousAttentionTargetSnapshot(
            observed_at=observed_at,
            state=state,
            active=True,
            focus_source=focus_source,
            target_track_id=track.track_id,
            target_center_x=predicted_center_x,
            target_center_y=track.center_y,
            target_zone=track.zone,
            target_horizontal=_horizontal_from_center_x(predicted_center_x),
            target_velocity_x=round(track.velocity_x, 4),
            speaker_locked=speaker_locked,
            motion_locked=motion_locked,
            visible_track_count=visible_track_count,
            audio_target_x=audio_target_x,
            audio_mirror_applied=self._audio_mirror_enabled(),
            confidence=round(max(0.0, min(1.0, confidence)), 4),
        )

    def _update_audio_mirror_calibration(
        self,
        *,
        speech_detected: bool,
        speaker_direction_stable: bool,
        direction_confidence: float | None,
        azimuth_deg: int | None,
        tracks: list[_TrackState],
    ) -> None:
        if not speech_detected or speaker_direction_stable is not True:
            return
        if direction_confidence is None or direction_confidence < self.config.direction_min_confidence:
            return
        if azimuth_deg is None or len(tracks) != 1:
            return
        center_x = tracks[0].center_x
        if abs(center_x - 0.5) < 0.12:
            return
        plain_target_x = _folded_audio_target_x(azimuth_deg)
        direct_error = abs(plain_target_x - center_x)
        mirrored_error = abs((1.0 - plain_target_x) - center_x)
        margin = self.config.audio_mirror_margin
        if mirrored_error + margin < direct_error:
            self._audio_mirror_score = min(self._audio_mirror_score + 1, 8)
        elif direct_error + margin < mirrored_error:
            self._audio_mirror_score = max(self._audio_mirror_score - 1, -8)

    def _audio_mirror_enabled(self) -> bool:
        return self._audio_mirror_score >= self.config.audio_mirror_min_samples


def _visible_person_anchors(camera: Mapping[str, object]) -> tuple[_VisiblePersonAnchor, ...]:
    """Parse bounded visible-person anchors from camera automation facts."""

    raw_people = camera.get("visible_persons")
    anchors: list[_VisiblePersonAnchor] = []
    if isinstance(raw_people, (tuple, list)):
        for item in raw_people:
            payload = coerce_mapping(item)
            center = _coerce_box_center(payload.get("box"))
            if center is None:
                continue
            center_x, center_y = center
            zone = _normalize_horizontal(payload.get("zone")) or _horizontal_from_center_x(center_x)
            anchors.append(
                _VisiblePersonAnchor(
                    center_x=center_x,
                    center_y=center_y,
                    zone=zone or "center",
                    confidence=coerce_optional_ratio(payload.get("confidence")) or 0.0,
                )
            )
    if anchors:
        return tuple(anchors)
    person_visible = coerce_optional_bool(camera.get("person_visible")) is True
    center_x = None if coerce_optional_bool(camera.get("primary_person_center_x_unknown")) is True else coerce_optional_ratio(camera.get("primary_person_center_x"))
    center_y = None if coerce_optional_bool(camera.get("primary_person_center_y_unknown")) is True else coerce_optional_ratio(camera.get("primary_person_center_y"))
    if not person_visible or center_x is None or center_y is None:
        return ()
    zone = _normalize_horizontal(camera.get("primary_person_zone")) or _horizontal_from_center_x(center_x)
    return (
        _VisiblePersonAnchor(
            center_x=center_x,
            center_y=center_y,
            zone=zone or "center",
            confidence=max(0.6, coerce_optional_ratio(camera.get("visual_attention_score")) or 0.0),
        ),
    )


def _track_distance(*, track: _TrackState, anchor: _VisiblePersonAnchor) -> float:
    """Return one small geometric distance between a track and a new anchor."""

    return (
        abs(track.center_x - anchor.center_x)
        + (abs(track.center_y - anchor.center_y) * 0.35)
    )


def _folded_audio_target_x(azimuth_deg: int) -> float:
    """Project one raw azimuth into a front-facing horizontal x coordinate.

    XMOS documents raw azimuth as device-dependent and recommends consuming it
    as system designers require. Twinr keeps that bounded by folding the rear
    hemisphere into the visible front arc and learning whether the current
    installation needs a left/right mirror.
    """

    normalized = azimuth_deg % 360
    front_arc = float(normalized)
    if front_arc > 180.0:
        front_arc = 360.0 - front_arc
    return _clamp_ratio(front_arc / 180.0)


def _horizontal_from_center_x(center_x: float | None) -> str | None:
    """Map one normalized center x value to a coarse horizontal token."""

    if center_x is None:
        return None
    if center_x <= 0.36:
        return "left"
    if center_x >= 0.64:
        return "right"
    return "center"


def _coerce_box_center(value: object) -> tuple[float, float] | None:
    """Return the center of one serialized box when available."""

    payload = coerce_mapping(value)
    try:
        top = float(payload.get("top"))
        left = float(payload.get("left"))
        bottom = float(payload.get("bottom"))
        right = float(payload.get("right"))
    except (TypeError, ValueError):
        return None
    center_x = _clamp_ratio((left + right) / 2.0)
    center_y = _clamp_ratio((top + bottom) / 2.0)
    return (center_x, center_y)


def _normalize_horizontal(value: object | None) -> str | None:
    """Normalize one optional horizontal token."""

    token = normalize_text(value).lower()
    if token in _VALID_HORIZONTAL:
        return token
    return None


def _coerce_optional_int(value: object | None) -> int | None:
    """Parse one optional integer value."""

    if value is None or isinstance(value, bool):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _clamp_ratio(value: float) -> float:
    """Clamp one numeric value into the inclusive unit interval."""

    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _mean_confidence(values: tuple[float | None, ...]) -> float | None:
    """Return the arithmetic mean of available confidence values."""

    present = [value for value in values if value is not None]
    if not present:
        return None
    return round(sum(present) / len(present), 4)


__all__ = [
    "ContinuousAttentionTargetConfig",
    "ContinuousAttentionTargetSnapshot",
    "ContinuousAttentionTracker",
]
