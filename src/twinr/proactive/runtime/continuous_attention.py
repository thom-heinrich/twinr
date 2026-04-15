# CHANGELOG: 2026-03-29
# BUG-1: Fixed non-monotonic/None timestamps keeping stale tracks alive indefinitely, which could preserve ghost people and grow state forever.
# BUG-2: Replaced greedy per-anchor matching with global optimal assignment, reducing real ID swaps in two-person / crossing scenes.
# BUG-3: Hardened numeric parsing so NaN/Inf or malformed boxes cannot poison track state or leak invalid coordinates downstream.
# SEC-1: Bounded visible-anchor scan count and retained-track count to prevent input-driven CPU/memory blowups on Raspberry Pi deployments.
# SEC-2: Made audio-mirror auto-calibration conservative so unrelated speech/noisy single-track scenes cannot easily poison left/right mirroring.
# IMP-1: Added lightweight track lifecycle management (confirmation hits, miss decay, retention ranking) inspired by modern edge MOT patterns.
# IMP-2: Added speaker-lock hysteresis / attentional momentum for short audio dropouts and safer single-track audio plausibility checks.

"""Track short-lived visible person anchors for continuous HDMI attention.

This module keeps person-anchor matching, motion recency, short hold behavior,
and bounded audio-to-vision handoff out of the higher-level attention policy.
It does not try to identify people. It only keeps a tiny rolling notion of
"which visible anchor is probably the relevant one right now" so Twinr can
follow a person more continuously, prefer the active speaker in simple
multi-person scenes, and otherwise look at the most recently moving person.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache

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
_DEFAULT_MAX_VISIBLE_ANCHORS = 6
_DEFAULT_VISIBLE_ANCHOR_SCAN_LIMIT = 12
_DEFAULT_MAX_TRACK_COUNT = 12
_DEFAULT_CONFIRM_HITS = 2
_DEFAULT_SPEAKER_SWITCH_MARGIN = 0.04
_DEFAULT_SINGLE_TRACK_AUDIO_PLAUSIBILITY_DISTANCE = 0.38
_DEFAULT_AUDIO_MIRROR_CALIBRATION_DISTANCE = 0.22
_DEFAULT_UNMATCHED_CONFIDENCE_DECAY = 0.92
_DEFAULT_AUDIO_ALIGNMENT_DECAY = 0.84
_SINGLE_SPEAKER_AUDIO_BLEND = 0.2
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
    direction_min_tracks: int = _DEFAULT_DIRECTION_MIN_TRACKS
    audio_mirror_min_samples: int = _DEFAULT_AUDIO_MIRROR_MIN_SAMPLES
    audio_mirror_margin: float = _DEFAULT_AUDIO_MIRROR_MARGIN
    max_visible_anchors: int = _DEFAULT_MAX_VISIBLE_ANCHORS
    visible_anchor_scan_limit: int = _DEFAULT_VISIBLE_ANCHOR_SCAN_LIMIT
    max_track_count: int = _DEFAULT_MAX_TRACK_COUNT
    confirm_hits: int = _DEFAULT_CONFIRM_HITS
    speaker_switch_margin: float = _DEFAULT_SPEAKER_SWITCH_MARGIN
    single_track_audio_plausibility_distance: float = _DEFAULT_SINGLE_TRACK_AUDIO_PLAUSIBILITY_DISTANCE
    audio_mirror_calibration_distance: float = _DEFAULT_AUDIO_MIRROR_CALIBRATION_DISTANCE

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "ContinuousAttentionTargetConfig":
        """Build one bounded config from the global Twinr config."""

        def _bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
            try:
                raw_value = getattr(config, name, default)
                value = default if raw_value is None else float(raw_value)
            except (TypeError, ValueError):
                value = default
            if not math.isfinite(value):
                value = default
            return max(minimum, min(maximum, value))

        def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
            try:
                raw_value = getattr(config, name, default)
                value = default if raw_value is None else int(float(raw_value))
            except (TypeError, ValueError):
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
            direction_min_tracks=_bounded_int(
                "display_attention_direction_min_tracks",
                _DEFAULT_DIRECTION_MIN_TRACKS,
                1,
                4,
            ),
            audio_mirror_min_samples=_bounded_int(
                "display_attention_audio_mirror_min_samples",
                _DEFAULT_AUDIO_MIRROR_MIN_SAMPLES,
                1,
                8,
            ),
            audio_mirror_margin=_bounded_float(
                "display_attention_audio_mirror_margin",
                _DEFAULT_AUDIO_MIRROR_MARGIN,
                0.0,
                0.3,
            ),
            max_visible_anchors=_bounded_int(
                "display_attention_max_visible_anchors",
                _DEFAULT_MAX_VISIBLE_ANCHORS,
                1,
                12,
            ),
            visible_anchor_scan_limit=_bounded_int(
                "display_attention_visible_anchor_scan_limit",
                _DEFAULT_VISIBLE_ANCHOR_SCAN_LIMIT,
                1,
                32,
            ),
            max_track_count=_bounded_int(
                "display_attention_max_track_count",
                _DEFAULT_MAX_TRACK_COUNT,
                1,
                20,
            ),
            confirm_hits=_bounded_int(
                "display_attention_confirm_hits",
                _DEFAULT_CONFIRM_HITS,
                1,
                4,
            ),
            speaker_switch_margin=_bounded_float(
                "display_attention_speaker_switch_margin",
                _DEFAULT_SPEAKER_SWITCH_MARGIN,
                0.0,
                0.2,
            ),
            single_track_audio_plausibility_distance=_bounded_float(
                "display_attention_single_track_audio_plausibility_distance",
                _DEFAULT_SINGLE_TRACK_AUDIO_PLAUSIBILITY_DISTANCE,
                0.12,
                0.7,
            ),
            audio_mirror_calibration_distance=_bounded_float(
                "display_attention_audio_mirror_calibration_distance",
                _DEFAULT_AUDIO_MIRROR_CALIBRATION_DISTANCE,
                0.05,
                0.4,
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
    velocity_y: float = 0.0
    first_seen_at: float = 0.0
    hit_count: int = 1
    miss_count: int = 0
    seen_count: int = 1
    audio_alignment_score: float = 0.0
    last_audio_alignment_at: float | None = None

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
        self._last_speaker_track_id: str | None = None
        self._last_speaker_at: float | None = None
        self._last_effective_observed_at: float | None = None
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

        checked_at = self._effective_observed_at(observed_at)
        camera = coerce_mapping(None if live_facts is None else live_facts.get("camera"))
        vad = coerce_mapping(None if live_facts is None else live_facts.get("vad"))
        respeaker = coerce_mapping(None if live_facts is None else live_facts.get("respeaker"))
        audio_policy = coerce_mapping(None if live_facts is None else live_facts.get("audio_policy"))

        anchors = _visible_person_anchors(
            camera,
            max_anchors=self.config.max_visible_anchors,
            scan_limit=self.config.visible_anchor_scan_limit,
        )
        active_tracks = self._update_tracks(observed_at=checked_at, anchors=anchors)
        if not active_tracks:
            self._last_selected_track_id = None
            self._last_selected_at = None
            self._last_speaker_track_id = None
            self._last_speaker_at = None
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

        self._update_audio_alignment_scores(
            observed_at=checked_at,
            tracks=active_tracks,
            speech_detected=speech_detected,
            speaker_direction_stable=speaker_direction_stable,
            direction_confidence=direction_confidence,
            audio_target_x=audio_target_x,
        )

        self._update_audio_mirror_calibration(
            speech_detected=speech_detected,
            speaker_direction_stable=speaker_direction_stable,
            direction_confidence=direction_confidence,
            azimuth_deg=azimuth_deg,
            tracks=active_tracks,
        )

        speaker_track = self._speaker_track(
            observed_at=checked_at,
            tracks=active_tracks,
            speech_detected=speech_detected,
            speaker_direction_stable=speaker_direction_stable,
            direction_confidence=direction_confidence,
            audio_target_x=audio_target_x,
        )
        if speaker_track is not None:
            return self._select_snapshot(
                observed_at=observed_at,
                effective_observed_at=checked_at,
                track=speaker_track,
                state="active_visible_speaker_track",
                focus_source="speaker_track",
                speaker_locked=True,
                visible_track_count=len(active_tracks),
                audio_target_x=audio_target_x,
                target_center_x_override=self._speaker_target_center_x(
                    track=speaker_track,
                    visible_track_count=len(active_tracks),
                    audio_target_x=audio_target_x,
                ),
                confidence=max(
                    0.6,
                    _mean_confidence(
                        (
                            speaker_track.confidence,
                            direction_confidence,
                            speaker_track.audio_alignment_score if speech_detected else None,
                        )
                    )
                    or 0.82,
                ),
            )

        held_speaker_track = self._speaker_hold_track(observed_at=checked_at, tracks=active_tracks)
        if speech_detected and held_speaker_track is not None:
            return self._select_snapshot(
                observed_at=observed_at,
                effective_observed_at=checked_at,
                track=held_speaker_track,
                state="active_visible_speaker_track",
                focus_source="speaker_track_hold",
                speaker_locked=True,
                visible_track_count=len(active_tracks),
                audio_target_x=audio_target_x,
                confidence=max(0.68, held_speaker_track.confidence * 0.95),
            )

        moved_track = self._last_moved_track(observed_at=checked_at, tracks=active_tracks)
        if moved_track is not None and len(active_tracks) >= self.config.direction_min_tracks:
            return self._select_snapshot(
                observed_at=observed_at,
                effective_observed_at=checked_at,
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
                effective_observed_at=checked_at,
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
            effective_observed_at=checked_at,
            track=primary_track,
            state="active_visible_person",
            focus_source="primary_visible_person",
            visible_track_count=len(active_tracks),
            audio_target_x=audio_target_x,
            confidence=max(0.6, primary_track.confidence),
        )

    def debug_snapshot(self, *, observed_at: float | None) -> dict[str, object]:
        """Return one bounded debug view of the current visible-track state."""

        checked_at = self._effective_debug_time(observed_at)
        active_tracks = [
            track
            for track in self._tracks.values()
            if (checked_at - track.updated_at) <= self.config.track_stale_s
        ]
        active_tracks.sort(key=lambda item: self._track_sort_key(item, checked_at), reverse=True)
        track_summaries: list[dict[str, object]] = []
        for track in active_tracks[: min(6, self.config.max_visible_anchors)]:
            last_motion_age_s = None
            if track.last_motion_at is not None:
                last_motion_age_s = round(max(0.0, checked_at - track.last_motion_at), 3)
            track_summaries.append(
                {
                    "track_id": track.track_id,
                    "center_x": round(track.center_x, 4),
                    "center_y": round(track.center_y, 4),
                    "zone": track.zone,
                    "confidence": round(track.confidence, 4),
                    "velocity_x": round(track.velocity_x, 4),
                    "velocity_y": round(track.velocity_y, 4),
                    "confirmed": self._track_is_confirmed(track),
                    "hit_count": track.hit_count,
                    "miss_count": track.miss_count,
                    "audio_alignment_score": round(track.audio_alignment_score, 4),
                    "age_s": round(max(0.0, checked_at - track.updated_at), 3),
                    "last_motion_age_s": last_motion_age_s,
                }
            )
        selected_track_age_s = None
        if self._last_selected_at is not None:
            selected_track_age_s = round(max(0.0, checked_at - self._last_selected_at), 3)
        return {
            "selected_track_id": self._last_selected_track_id,
            "selected_track_age_s": selected_track_age_s,
            "speaker_track_id": self._last_speaker_track_id,
            "audio_mirror_score": self._audio_mirror_score,
            "active_track_count": len(active_tracks),
            "tracks": track_summaries,
        }

    def _effective_observed_at(self, observed_at: float | None) -> float:
        raw_value = _coerce_optional_float(observed_at)
        candidate = time.monotonic() if raw_value is None else raw_value
        if self._last_effective_observed_at is None:
            self._last_effective_observed_at = candidate
            return candidate
        if candidate <= self._last_effective_observed_at:
            candidate = self._last_effective_observed_at + 0.001
        self._last_effective_observed_at = candidate
        return candidate

    def _effective_debug_time(self, observed_at: float | None) -> float:
        raw_value = _coerce_optional_float(observed_at)
        if raw_value is not None:
            return raw_value
        if self._last_effective_observed_at is not None:
            return self._last_effective_observed_at
        return time.monotonic()

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
            and _track_is_finite(track)
        ]
        active_tracks.sort(key=lambda item: self._track_sort_key(item, observed_at), reverse=True)

        matched_track_ids: set[str] = set()
        matched_anchor_indexes: set[int] = set()
        next_tracks: dict[str, _TrackState] = {}
        rendered: list[_TrackState] = []

        assignments = _optimal_track_assignment(
            tracks=active_tracks,
            anchors=anchors,
            max_distance=self.config.match_max_distance,
            cost_fn=self._association_distance,
        )

        for anchor_index, track_index in assignments:
            anchor = anchors[anchor_index]
            previous = active_tracks[track_index]
            matched_anchor_indexes.add(anchor_index)
            matched_track_ids.add(previous.track_id)

            dt = max(0.001, observed_at - previous.updated_at)
            raw_velocity_x = (anchor.center_x - previous.center_x) / dt
            raw_velocity_y = (anchor.center_y - previous.center_y) / dt
            velocity_x = (previous.velocity_x * 0.6) + (raw_velocity_x * 0.4)
            velocity_y = (previous.velocity_y * 0.6) + (raw_velocity_y * 0.4)
            moved = (
                abs(anchor.center_x - previous.center_x) >= self.config.motion_delta
                or abs(anchor.center_y - previous.center_y) >= self.config.motion_delta
            )
            track = _TrackState(
                track_id=previous.track_id,
                center_x=anchor.center_x,
                center_y=anchor.center_y,
                zone=anchor.zone,
                confidence=_clamp_confidence(max(anchor.confidence, previous.confidence * 0.75)),
                updated_at=observed_at,
                last_motion_at=(observed_at if moved else previous.last_motion_at),
                velocity_x=_finite_or_zero(velocity_x),
                velocity_y=_finite_or_zero(velocity_y),
                first_seen_at=previous.first_seen_at,
                hit_count=previous.hit_count + 1,
                miss_count=0,
                seen_count=previous.seen_count + 1,
                audio_alignment_score=previous.audio_alignment_score,
                last_audio_alignment_at=previous.last_audio_alignment_at,
            )
            next_tracks[track.track_id] = track
            rendered.append(track)

        for index, anchor in enumerate(anchors):
            if index in matched_anchor_indexes:
                continue
            track = _TrackState(
                track_id=f"visible_track_{self._next_track_index}",
                center_x=anchor.center_x,
                center_y=anchor.center_y,
                zone=anchor.zone,
                confidence=_clamp_confidence(anchor.confidence),
                updated_at=observed_at,
                last_motion_at=None,
                velocity_x=0.0,
                velocity_y=0.0,
                first_seen_at=observed_at,
                hit_count=1,
                miss_count=0,
                seen_count=1,
                audio_alignment_score=0.0,
                last_audio_alignment_at=None,
            )
            self._next_track_index += 1
            next_tracks[track.track_id] = track
            rendered.append(track)

        for track in active_tracks:
            if track.track_id in matched_track_ids:
                continue
            kept = _TrackState(
                track_id=track.track_id,
                center_x=track.center_x,
                center_y=track.center_y,
                zone=track.zone,
                confidence=_clamp_confidence(track.confidence * _DEFAULT_UNMATCHED_CONFIDENCE_DECAY),
                updated_at=track.updated_at,
                last_motion_at=track.last_motion_at,
                velocity_x=track.velocity_x,
                velocity_y=track.velocity_y,
                first_seen_at=track.first_seen_at,
                hit_count=track.hit_count,
                miss_count=track.miss_count + 1,
                seen_count=track.seen_count,
                audio_alignment_score=max(0.0, track.audio_alignment_score * _DEFAULT_AUDIO_ALIGNMENT_DECAY),
                last_audio_alignment_at=track.last_audio_alignment_at,
            )
            next_tracks[kept.track_id] = kept
            rendered.append(kept)

        if len(next_tracks) > self.config.max_track_count:
            ranked = sorted(
                next_tracks.values(),
                key=lambda item: self._track_retention_key(item, observed_at),
                reverse=True,
            )
            keep_ids = {track.track_id for track in ranked[: self.config.max_track_count]}
            next_tracks = {track_id: track for track_id, track in next_tracks.items() if track_id in keep_ids}
            rendered = [track for track in rendered if track.track_id in keep_ids]
            if self._last_selected_track_id not in keep_ids:
                self._last_selected_track_id = None
                self._last_selected_at = None
            if self._last_speaker_track_id not in keep_ids:
                self._last_speaker_track_id = None
                self._last_speaker_at = None

        self._tracks = next_tracks
        rendered = [track for track in rendered if track.track_id in self._tracks]
        rendered.sort(key=lambda item: self._track_sort_key(item, observed_at), reverse=True)
        return rendered

    def _association_distance(self, track: _TrackState, anchor: _VisiblePersonAnchor) -> float:
        predicted_center_x = track.predicted_center_x(horizon_s=min(self.config.prediction_horizon_s, 0.22))
        zone_penalty = 0.03 if track.zone != anchor.zone else 0.0
        return abs(predicted_center_x - anchor.center_x) + (abs(track.center_y - anchor.center_y) * 0.45) + zone_penalty

    def _speaker_track(
        self,
        *,
        observed_at: float,
        tracks: list[_TrackState],
        speech_detected: bool,
        speaker_direction_stable: bool,
        direction_confidence: float | None,
        audio_target_x: float | None,
    ) -> _TrackState | None:
        if not speech_detected:
            return None

        candidate_tracks = [track for track in tracks if self._track_is_confirmed(track)] or tracks
        if len(candidate_tracks) == 1:
            only_track = candidate_tracks[0]
            if (
                audio_target_x is not None
                and speaker_direction_stable is True
                and direction_confidence is not None
                and direction_confidence >= self.config.direction_min_confidence
            ):
                if abs(only_track.predicted_center_x(horizon_s=self.config.prediction_horizon_s) - audio_target_x) > self.config.single_track_audio_plausibility_distance:
                    return None
            return only_track

        if speaker_direction_stable is not True:
            return None
        if direction_confidence is None or direction_confidence < self.config.direction_min_confidence:
            return None
        if audio_target_x is None:
            return None

        candidates: list[tuple[_TrackState, float]] = []
        for track in candidate_tracks:
            distance = abs(track.predicted_center_x(horizon_s=self.config.prediction_horizon_s) - audio_target_x)
            if distance > self.config.speaker_max_distance:
                continue
            candidates.append((track, distance))
        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (
                item[1],
                -item[0].audio_alignment_score,
                -item[0].confidence,
                -(1 if item[0].track_id == self._last_speaker_track_id else 0),
            )
        )
        best_track, best_distance = candidates[0]

        if self._last_speaker_track_id is not None and self._last_speaker_at is not None:
            if (observed_at - self._last_speaker_at) <= self.config.target_hold_s:
                for track, distance in candidates:
                    if track.track_id != self._last_speaker_track_id:
                        continue
                    if (best_distance + self.config.speaker_switch_margin) >= distance:
                        return track
                    break

        return best_track

    def _speaker_hold_track(
        self,
        *,
        observed_at: float,
        tracks: list[_TrackState],
    ) -> _TrackState | None:
        if self._last_speaker_track_id is None or self._last_speaker_at is None:
            return None
        if (observed_at - self._last_speaker_at) > self.config.target_hold_s:
            return None
        for track in tracks:
            if track.track_id == self._last_speaker_track_id:
                return track
        return None

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
        confirmed_recent = [track for track in recent if self._track_is_confirmed(track)]
        candidates = confirmed_recent or recent
        candidates.sort(
            key=lambda item: (
                item.last_motion_at,
                item.hit_count,
                item.confidence,
            ),
            reverse=True,
        )
        return candidates[0]

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
        fresh_tracks = [
            track
            for track in tracks
            if abs(observed_at - track.updated_at) <= 0.001
        ]
        for track in fresh_tracks:
            if track.track_id == self._last_selected_track_id:
                return track
        if fresh_tracks:
            return None
        for track in tracks:
            if track.track_id == self._last_selected_track_id:
                return track
        return None

    def _select_snapshot(
        self,
        *,
        observed_at: float | None,
        effective_observed_at: float,
        track: _TrackState,
        state: str,
        focus_source: str,
        visible_track_count: int,
        audio_target_x: float | None,
        confidence: float,
        speaker_locked: bool = False,
        motion_locked: bool = False,
        target_center_x_override: float | None = None,
    ) -> ContinuousAttentionTargetSnapshot:
        self._last_selected_track_id = track.track_id
        self._last_selected_at = effective_observed_at
        if speaker_locked:
            self._last_speaker_track_id = track.track_id
            self._last_speaker_at = effective_observed_at
        predicted_center_x = (
            track.predicted_center_x(horizon_s=self.config.prediction_horizon_s)
            if target_center_x_override is None
            else _clamp_ratio(target_center_x_override)
        )
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

    def _speaker_target_center_x(
        self,
        *,
        track: _TrackState,
        visible_track_count: int,
        audio_target_x: float | None,
    ) -> float:
        """Return one conservative target x for the current visible speaker.

        In simple one-person scenes, speech should do more than flip an
        internal state label. Blend the visible track with the currently folded
        audio direction so the servo and HDMI follow path can react to the
        speaker's current bearing, but clamp the audio pull to the existing
        speaker-match window so audio noise cannot yank the target far away
        from the confirmed visible person.
        """

        predicted_center_x = track.predicted_center_x(horizon_s=self.config.prediction_horizon_s)
        if visible_track_count != 1 or audio_target_x is None:
            return predicted_center_x
        bounded_audio_target_x = _clamp_ratio(
            max(
                track.center_x - self.config.speaker_max_distance,
                min(track.center_x + self.config.speaker_max_distance, audio_target_x),
            )
        )
        return _clamp_ratio(
            predicted_center_x
            + ((bounded_audio_target_x - predicted_center_x) * _SINGLE_SPEAKER_AUDIO_BLEND)
        )

    def _update_audio_alignment_scores(
        self,
        *,
        observed_at: float,
        tracks: list[_TrackState],
        speech_detected: bool,
        speaker_direction_stable: bool,
        direction_confidence: float | None,
        audio_target_x: float | None,
    ) -> None:
        track_updates: dict[str, _TrackState] = {}
        for track in tracks:
            new_score = max(0.0, min(1.0, track.audio_alignment_score * _DEFAULT_AUDIO_ALIGNMENT_DECAY))
            last_audio_alignment_at = track.last_audio_alignment_at
            if (
                speech_detected
                and speaker_direction_stable is True
                and direction_confidence is not None
                and direction_confidence >= self.config.direction_min_confidence
                and audio_target_x is not None
            ):
                distance = abs(track.predicted_center_x(horizon_s=self.config.prediction_horizon_s) - audio_target_x)
                alignment = max(0.0, 1.0 - (distance / max(self.config.speaker_max_distance, 1e-6)))
                new_score = max(new_score, round(alignment, 4))
                if alignment > 0.0:
                    last_audio_alignment_at = observed_at
            if new_score == track.audio_alignment_score and last_audio_alignment_at == track.last_audio_alignment_at:
                continue
            track_updates[track.track_id] = _TrackState(
                track_id=track.track_id,
                center_x=track.center_x,
                center_y=track.center_y,
                zone=track.zone,
                confidence=track.confidence,
                updated_at=track.updated_at,
                last_motion_at=track.last_motion_at,
                velocity_x=track.velocity_x,
                velocity_y=track.velocity_y,
                first_seen_at=track.first_seen_at,
                hit_count=track.hit_count,
                miss_count=track.miss_count,
                seen_count=track.seen_count,
                audio_alignment_score=new_score,
                last_audio_alignment_at=last_audio_alignment_at,
            )

        if not track_updates:
            return
        self._tracks.update(track_updates)
        for index, track in enumerate(tracks):
            updated = track_updates.get(track.track_id)
            if updated is not None:
                tracks[index] = updated

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
        track = tracks[0]
        if not self._track_is_confirmed(track):
            return
        center_x = track.center_x
        if abs(center_x - 0.5) < 0.12:
            return
        plain_target_x = _folded_audio_target_x(azimuth_deg)
        direct_error = abs(plain_target_x - center_x)
        mirrored_error = abs((1.0 - plain_target_x) - center_x)
        if min(direct_error, mirrored_error) > self.config.audio_mirror_calibration_distance:
            return
        margin = self.config.audio_mirror_margin
        if mirrored_error + margin < direct_error:
            self._audio_mirror_score = min(self._audio_mirror_score + 1, 8)
        elif direct_error + margin < mirrored_error:
            self._audio_mirror_score = max(self._audio_mirror_score - 1, -8)

    def _audio_mirror_enabled(self) -> bool:
        return self._audio_mirror_score >= self.config.audio_mirror_min_samples

    def _track_is_confirmed(self, track: _TrackState) -> bool:
        return track.hit_count >= self.config.confirm_hits or track.confidence >= 0.85

    def _track_sort_key(self, track: _TrackState, observed_at: float) -> tuple[float, int, int, float]:
        freshness = -max(0.0, observed_at - track.updated_at)
        return (
            freshness,
            1 if self._track_is_confirmed(track) else 0,
            track.hit_count,
            track.confidence,
        )

    def _track_retention_key(self, track: _TrackState, observed_at: float) -> tuple[int, int, float, float]:
        selected_bonus = 1 if track.track_id == self._last_selected_track_id else 0
        speaker_bonus = 1 if track.track_id == self._last_speaker_track_id else 0
        freshness = -max(0.0, observed_at - track.updated_at)
        return (
            selected_bonus + speaker_bonus + (2 if self._track_is_confirmed(track) else 0),
            track.hit_count,
            freshness,
            track.confidence,
        )


def _visible_person_anchors(
    camera: Mapping[str, object],
    *,
    max_anchors: int,
    scan_limit: int,
) -> tuple[_VisiblePersonAnchor, ...]:
    """Parse bounded visible-person anchors from camera automation facts."""

    primary_anchor = _primary_visible_person_anchor(camera)
    raw_people = camera.get("visible_persons")
    indexed_anchors: list[tuple[int, _VisiblePersonAnchor]] = []
    if isinstance(raw_people, (tuple, list)):
        for index, item in enumerate(raw_people[: max(1, scan_limit)]):
            payload = coerce_mapping(item)
            center = _coerce_box_center(payload.get("box"))
            if center is None:
                continue
            center_x, center_y = center
            zone = _normalize_horizontal(payload.get("zone")) or _horizontal_from_center_x(center_x)
            indexed_anchors.append(
                (
                    index,
                    _VisiblePersonAnchor(
                        center_x=center_x,
                        center_y=center_y,
                        zone=zone or "center",
                        confidence=_clamp_confidence(coerce_optional_ratio(payload.get("confidence")) or 0.0),
                    ),
                )
            )
    if indexed_anchors:
        if len(indexed_anchors) > max(1, max_anchors):
            indexed_anchors.sort(
                key=lambda item: (item[1].confidence, -abs(item[1].center_x - 0.5)),
                reverse=True,
            )
            indexed_anchors = indexed_anchors[: max(1, max_anchors)]
            indexed_anchors.sort(key=lambda item: item[0])
        anchors = [anchor for _, anchor in indexed_anchors]
        if len(anchors) == 1 and primary_anchor is not None:
            single_anchor = anchors[0]
            return (
                _VisiblePersonAnchor(
                    center_x=primary_anchor.center_x,
                    center_y=single_anchor.center_y,
                    zone=primary_anchor.zone,
                    confidence=max(single_anchor.confidence, primary_anchor.confidence),
                ),
            )
        return tuple(anchors)
    if primary_anchor is None:
        return ()
    return (primary_anchor,)


def _primary_visible_person_anchor(camera: Mapping[str, object]) -> _VisiblePersonAnchor | None:
    """Return the stabilized primary-person anchor when camera facts expose one."""

    person_visible = coerce_optional_bool(camera.get("person_visible")) is True
    center_x = None if coerce_optional_bool(camera.get("primary_person_center_x_unknown")) is True else coerce_optional_ratio(camera.get("primary_person_center_x"))
    center_y = None if coerce_optional_bool(camera.get("primary_person_center_y_unknown")) is True else coerce_optional_ratio(camera.get("primary_person_center_y"))
    if not person_visible or center_x is None or center_y is None:
        return None
    zone = _normalize_horizontal(camera.get("primary_person_zone")) or _horizontal_from_center_x(center_x)
    return _VisiblePersonAnchor(
        center_x=center_x,
        center_y=center_y,
        zone=zone or "center",
        confidence=max(0.6, _clamp_confidence(coerce_optional_ratio(camera.get("visual_attention_score")) or 0.0)),
    )


def _optimal_track_assignment(
    *,
    tracks: list[_TrackState],
    anchors: tuple[_VisiblePersonAnchor, ...],
    max_distance: float,
    cost_fn: Callable[[_TrackState, _VisiblePersonAnchor], float],
) -> tuple[tuple[int, int], ...]:
    """Return a globally optimal small-N assignment.

    This keeps the module dependency-light for Raspberry Pi while still doing
    true global association rather than anchor-order greedy matching.
    """

    if not tracks or not anchors:
        return ()

    costs: list[list[float | None]] = []
    for anchor in anchors:
        row: list[float | None] = []
        for track in tracks:
            cost = cost_fn(track, anchor)
            if not math.isfinite(cost) or cost > max_distance:
                row.append(None)
            else:
                row.append(cost)
        costs.append(row)

    track_count = len(tracks)

    @lru_cache(maxsize=None)
    def _solve(anchor_index: int, used_mask: int) -> tuple[int, float, tuple[tuple[int, int], ...]]:
        if anchor_index >= len(anchors):
            return (0, 0.0, ())

        best_matches, best_cost, best_pairs = _solve(anchor_index + 1, used_mask)

        for track_index in range(track_count):
            if used_mask & (1 << track_index):
                continue
            cost = costs[anchor_index][track_index]
            if cost is None:
                continue
            sub_matches, sub_cost, sub_pairs = _solve(anchor_index + 1, used_mask | (1 << track_index))
            candidate_matches = sub_matches + 1
            candidate_cost = sub_cost + cost
            if candidate_matches > best_matches or (
                candidate_matches == best_matches and candidate_cost < best_cost
            ):
                best_matches = candidate_matches
                best_cost = candidate_cost
                best_pairs = ((anchor_index, track_index),) + sub_pairs

        return (best_matches, best_cost, best_pairs)

    return _solve(0, 0)[2]


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
    top = _coerce_optional_float(payload.get("top"))
    left = _coerce_optional_float(payload.get("left"))
    bottom = _coerce_optional_float(payload.get("bottom"))
    right = _coerce_optional_float(payload.get("right"))
    if top is None or left is None or bottom is None or right is None:
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
    if isinstance(value, int):
        return value
    try:
        converted = int(float(value if isinstance(value, (float, str)) else str(value)))
    except (TypeError, ValueError):
        return None
    return converted


def _coerce_optional_float(value: object | None) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        converted = float(value)
    else:
        try:
            converted = float(str(value))
        except (TypeError, ValueError):
            return None
    if not math.isfinite(converted):
        return None
    return converted


def _clamp_ratio(value: float) -> float:
    """Clamp one numeric value into the inclusive unit interval."""

    if not math.isfinite(value):
        return 0.5
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, 0.0 if not math.isfinite(value) else value))


def _finite_or_zero(value: float) -> float:
    return value if math.isfinite(value) else 0.0


def _track_is_finite(track: _TrackState) -> bool:
    return all(
        math.isfinite(value)
        for value in (
            track.center_x,
            track.center_y,
            track.confidence,
            track.updated_at,
            track.velocity_x,
            track.velocity_y,
            track.audio_alignment_score,
        )
    )


def _mean_confidence(values: tuple[float | None, ...]) -> float | None:
    """Return the arithmetic mean of available confidence values."""

    present = [value for value in values if value is not None and math.isfinite(value)]
    if not present:
        return None
    return round(sum(present) / len(present), 4)


__all__ = [
    "ContinuousAttentionTargetConfig",
    "ContinuousAttentionTargetSnapshot",
    "ContinuousAttentionTracker",
]
