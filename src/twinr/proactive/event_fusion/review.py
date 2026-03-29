"""Select review keyframes from recent fused-event evidence.

This module keeps keyframe-review planning separate from claim scoring. The
planner works on recent visual observations plus already-derived sequence
evidence and returns a compact onset/peak/latest plan that later review or
operator tooling can resolve into actual frames.
"""

# CHANGELOG: 2026-03-29
# BUG-1: Fixed incorrect frame choice when max_frames == 1. The old planner always returned the onset frame first,
# BUG-1: even though a single-frame budget should return the most relevant ("peak") frame.
# BUG-2: Fixed silent under-selection. When onset/peak/latest collided on the same timestamp, the old planner
# BUG-2: returned fewer frames than requested and never backfilled another distinct candidate.
# BUG-3: Removed fragile score lookup keyed by TimedSample objects. Depending on TimedSample/value hashability,
# BUG-3: dict(scored) could raise at runtime and crash review planning.
# BUG-4: Normalized enum-like payload fields to guaranteed strings and hardened against reversed/non-finite windows.
# SEC-1: Bounded candidate/evidence volume and sanitized malformed numeric inputs to reduce cheap CPU-exhaustion
# SEC-1: and NaN/Inf poisoning risks on resource-constrained Raspberry Pi deployments.
# IMP-1: Upgraded from fixed onset/peak/latest picking to training-free relevance+coverage+diversity selection with
# IMP-1: semantic-boundary awareness, adaptive backfilling, and deterministic tie-breaking.
# IMP-2: Replaced linear evidence summation with bounded multimodal fusion so fragmented upstream evidence cannot
# IMP-2: overwhelm frame relevance simply by being split into many overlapping micro-events.

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable

from twinr.proactive.event_fusion.audio_events import AudioMicroEvent
from twinr.proactive.event_fusion.buffers import RollingWindowBuffer, TimedSample
from twinr.proactive.event_fusion.vision_sequences import VisionSequenceEvent
from twinr.proactive.social.engine import SocialBodyPose, SocialMotionState, SocialVisionObservation


@dataclass(frozen=True, slots=True)
class ReviewKeyframeCandidate:
    """Describe one keyframe candidate chosen for downstream review."""

    observed_at: float
    role: str
    relevance_score: float
    person_visible: bool
    person_count: int
    body_pose: str
    motion_state: str
    looking_toward_device: bool
    smiling: bool

    def to_payload(self) -> dict[str, object]:
        """Serialize one candidate into plain review-plan facts."""

        return {
            "observed_at": self.observed_at,
            "role": self.role,
            "relevance_score": self.relevance_score,
            "person_visible": self.person_visible,
            "person_count": self.person_count,
            "body_pose": self.body_pose,
            "motion_state": self.motion_state,
            "looking_toward_device": self.looking_toward_device,
            "smiling": self.smiling,
        }


@dataclass(frozen=True, slots=True)
class KeyframeReviewPlan:
    """Describe a compact review plan for one fused event claim."""

    claim_state: str
    strategy: str = "relevance_plus_coverage"
    window_start_s: float | None = None
    window_end_s: float | None = None
    frames: tuple[ReviewKeyframeCandidate, ...] = field(default_factory=tuple)

    def to_payload(self) -> dict[str, object]:
        """Serialize the review plan into plain automation facts."""

        return {
            "claim_state": self.claim_state,
            "strategy": self.strategy,
            "window_start_s": self.window_start_s,
            "window_end_s": self.window_end_s,
            "frames": [frame.to_payload() for frame in self.frames],
        }


@dataclass(frozen=True, slots=True)
class KeyframeReviewConfig:
    """Store bounded parameters for relevance-plus-coverage frame selection."""

    max_frames: int = 3
    pre_roll_s: float = 0.75
    audio_alignment_half_life_s: float = 1.5
    vision_alignment_half_life_s: float = 1.25
    recency_half_life_s: float = 2.0
    min_frame_separation_s: float = 0.35
    max_candidate_samples: int = 64
    max_vision_evidence: int = 48
    max_audio_evidence: int = 96
    boundary_change_weight: float = 0.22
    diversity_weight: float = 0.30


@dataclass(frozen=True, slots=True)
class _PreparedSequenceEvent:
    window_start_s: float
    window_end_s: float
    confidence: float


@dataclass(frozen=True, slots=True)
class _PreparedAudioEvent:
    observed_at: float
    confidence: float


@dataclass(frozen=True, slots=True)
class _ScoredSample:
    index: int
    sample: TimedSample[SocialVisionObservation]
    base_score: float
    boundary_score: float
    selection_score: float


def build_keyframe_review_plan(
    *,
    claim_state: str,
    observation_buffer: RollingWindowBuffer[SocialVisionObservation],
    vision_evidence: tuple[VisionSequenceEvent, ...],
    audio_evidence: tuple[AudioMicroEvent, ...],
    window_start_s: float | None,
    window_end_s: float | None,
    config: KeyframeReviewConfig | None = None,
) -> KeyframeReviewPlan | None:
    """Return a compact onset/peak/latest plan for one fused claim.

    The planner intentionally does not require actual frame images. It only
    picks the timestamps and local scene facts that a later review stage should
    resolve.
    """

    normalized_window = _normalize_window(window_start_s, window_end_s)
    if normalized_window is None:
        return None
    start_s, end_s = normalized_window

    cfg = config or KeyframeReviewConfig()
    budget = max(1, cfg.max_frames)
    candidate_samples = _prepare_candidate_samples(
        observation_buffer.between(max(0.0, start_s - max(0.0, cfg.pre_roll_s)), end_s),
        config=cfg,
    )
    if not candidate_samples:
        return None

    prepared_vision = _prepare_vision_evidence(
        vision_evidence,
        window_start_s=start_s,
        window_end_s=end_s,
        config=cfg,
    )
    prepared_audio = _prepare_audio_evidence(
        audio_evidence,
        window_start_s=start_s,
        window_end_s=end_s,
        config=cfg,
    )
    scored_samples = _score_samples(
        candidate_samples=candidate_samples,
        now=end_s,
        vision_evidence=prepared_vision,
        audio_evidence=prepared_audio,
        config=cfg,
    )
    if not scored_samples:
        return None

    selected_samples = _select_samples(
        scored_samples=scored_samples,
        window_start_s=start_s,
        window_end_s=end_s,
        budget=budget,
        config=cfg,
    )
    if not selected_samples:
        return None

    return KeyframeReviewPlan(
        claim_state=claim_state,
        strategy="relevance_plus_coverage",
        window_start_s=start_s,
        window_end_s=end_s,
        frames=tuple(
            _to_candidate(
                sample=scored.sample,
                role=role,
                relevance_score=scored.base_score,
            )
            for role, scored in selected_samples
        ),
    )


def _normalize_window(
    window_start_s: float | None,
    window_end_s: float | None,
) -> tuple[float, float] | None:
    """Return a finite, ordered review window or None when the input is unusable."""

    if not _is_finite_number(window_start_s) or not _is_finite_number(window_end_s):
        return None
    start_s = float(window_start_s)
    end_s = float(window_end_s)
    if end_s < start_s:
        start_s, end_s = end_s, start_s
    return start_s, end_s


def _prepare_candidate_samples(
    samples: Iterable[TimedSample[SocialVisionObservation]],
    *,
    config: KeyframeReviewConfig,
) -> tuple[TimedSample[SocialVisionObservation], ...]:
    """Sort, sanitize, deduplicate, and bound observation samples."""

    ordered = sorted(
        (
            sample
            for sample in samples
            if _is_finite_number(getattr(sample, "observed_at", None))
        ),
        key=lambda sample: float(sample.observed_at),
    )
    if not ordered:
        return ()

    deduped: list[TimedSample[SocialVisionObservation]] = []
    for sample in ordered:
        timestamp = float(sample.observed_at)
        if deduped and math.isclose(timestamp, float(deduped[-1].observed_at), abs_tol=1e-6):
            deduped[-1] = sample
        else:
            deduped.append(sample)

    if len(deduped) <= max(1, config.max_candidate_samples):
        return tuple(deduped)

    selected_indices = _thin_candidate_indices(deduped, max_points=max(1, config.max_candidate_samples))
    return tuple(deduped[index] for index in selected_indices)


def _thin_candidate_indices(
    samples: list[TimedSample[SocialVisionObservation]],
    *,
    max_points: int,
) -> tuple[int, ...]:
    """Keep endpoints, change points, and evenly spaced coverage samples."""

    if len(samples) <= max_points:
        return tuple(range(len(samples)))
    if max_points <= 1:
        return (len(samples) - 1,)

    selected: set[int] = {0, len(samples) - 1}
    change_ranked = sorted(
        (
            (_local_change_score(samples, index), index)
            for index in range(1, len(samples) - 1)
        ),
        key=lambda item: (item[0], -abs(item[1] - (len(samples) // 2))),
        reverse=True,
    )
    for _, index in change_ranked:
        if len(selected) >= max_points:
            break
        selected.add(index)

    if len(selected) < max_points:
        stride = (len(samples) - 1) / max(1, max_points - 1)
        for slot in range(max_points):
            selected.add(int(round(slot * stride)))
            if len(selected) >= max_points:
                break

    return tuple(sorted(selected)[:max_points])


def _prepare_vision_evidence(
    vision_evidence: tuple[VisionSequenceEvent, ...],
    *,
    window_start_s: float,
    window_end_s: float,
    config: KeyframeReviewConfig,
) -> tuple[_PreparedSequenceEvent, ...]:
    """Filter and bound vision evidence near the review window."""

    prepared: list[_PreparedSequenceEvent] = []
    horizon_s = max(4.0, config.pre_roll_s + 2.0 * config.vision_alignment_half_life_s)
    for sequence in vision_evidence:
        raw_start = getattr(sequence, "window_start_s", None)
        raw_end = getattr(sequence, "window_end_s", None)
        raw_conf = getattr(sequence, "confidence", None)
        if not _is_finite_number(raw_start) or not _is_finite_number(raw_end):
            continue
        start_s = float(raw_start)
        end_s = float(raw_end)
        if end_s < start_s:
            start_s, end_s = end_s, start_s
        confidence = _sanitize_confidence(raw_conf)
        if confidence <= 0.0:
            continue
        if _interval_distance(window_start_s, window_end_s, start_s, end_s) > horizon_s:
            continue
        prepared.append(
            _PreparedSequenceEvent(
                window_start_s=start_s,
                window_end_s=end_s,
                confidence=confidence,
            )
        )

    prepared.sort(
        key=lambda event: (
            _interval_distance(window_start_s, window_end_s, event.window_start_s, event.window_end_s),
            -event.confidence,
            event.window_start_s,
            event.window_end_s,
        )
    )
    return tuple(prepared[: max(1, config.max_vision_evidence)])


def _prepare_audio_evidence(
    audio_evidence: tuple[AudioMicroEvent, ...],
    *,
    window_start_s: float,
    window_end_s: float,
    config: KeyframeReviewConfig,
) -> tuple[_PreparedAudioEvent, ...]:
    """Filter and bound audio evidence near the review window."""

    prepared: list[_PreparedAudioEvent] = []
    horizon_s = max(4.0, config.pre_roll_s + 3.0 * config.audio_alignment_half_life_s)
    for event in audio_evidence:
        raw_timestamp = getattr(event, "observed_at", None)
        raw_confidence = getattr(event, "confidence", None)
        if not _is_finite_number(raw_timestamp):
            continue
        timestamp = float(raw_timestamp)
        confidence = _sanitize_confidence(raw_confidence)
        if confidence <= 0.0:
            continue
        if timestamp < window_start_s - horizon_s or timestamp > window_end_s + horizon_s:
            continue
        prepared.append(_PreparedAudioEvent(observed_at=timestamp, confidence=confidence))

    prepared.sort(
        key=lambda event: (
            abs(_clamp(event.observed_at, window_start_s, window_end_s) - event.observed_at),
            -event.confidence,
            event.observed_at,
        )
    )
    return tuple(prepared[: max(1, config.max_audio_evidence)])


def _score_samples(
    *,
    candidate_samples: tuple[TimedSample[SocialVisionObservation], ...],
    now: float,
    vision_evidence: tuple[_PreparedSequenceEvent, ...],
    audio_evidence: tuple[_PreparedAudioEvent, ...],
    config: KeyframeReviewConfig,
) -> tuple[_ScoredSample, ...]:
    """Assign base and selection scores to each candidate sample."""

    if not candidate_samples:
        return ()

    base_scores = [
        _candidate_score(
            sample=sample,
            now=now,
            vision_evidence=vision_evidence,
            audio_evidence=audio_evidence,
            config=config,
        )
        for sample in candidate_samples
    ]
    scored: list[_ScoredSample] = []
    for index, sample in enumerate(candidate_samples):
        boundary_score = _boundary_score(candidate_samples, base_scores, index)
        selection_score = base_scores[index] + (config.boundary_change_weight * boundary_score)
        scored.append(
            _ScoredSample(
                index=index,
                sample=sample,
                base_score=round(base_scores[index], 6),
                boundary_score=round(boundary_score, 6),
                selection_score=round(selection_score, 6),
            )
        )
    return tuple(scored)


def _select_samples(
    *,
    scored_samples: tuple[_ScoredSample, ...],
    window_start_s: float,
    window_end_s: float,
    budget: int,
    config: KeyframeReviewConfig,
) -> tuple[tuple[str, _ScoredSample], ...]:
    """Choose anchor and support frames with coverage-aware backfilling."""

    if not scored_samples or budget <= 0:
        return ()

    effective_budget = min(budget, len(scored_samples))
    selected: list[tuple[str, _ScoredSample]] = []
    used_indices: set[int] = set()

    peak = max(
        scored_samples,
        key=lambda item: (item.base_score, item.selection_score, item.sample.observed_at),
    )
    if effective_budget == 1:
        return (("peak", peak),)

    onset = scored_samples[0]
    latest = scored_samples[-1]

    for role, preferred in (("onset", onset), ("peak", peak), ("latest", latest)):
        if len(selected) >= effective_budget:
            break
        distinct = _best_distinct_sample(
            scored_samples,
            preferred=preferred,
            used_indices=used_indices,
            selected=selected,
            window_start_s=window_start_s,
            window_end_s=window_end_s,
            config=config,
            prefer_anchor=(role != "peak"),
        )
        if distinct is None:
            continue
        used_indices.add(distinct.index)
        selected.append((role, distinct))

    support_count = 1
    while len(selected) < effective_budget:
        support = _best_support_sample(
            scored_samples,
            used_indices=used_indices,
            selected=selected,
            window_start_s=window_start_s,
            window_end_s=window_end_s,
            config=config,
        )
        if support is None:
            break
        used_indices.add(support.index)
        selected.append((f"support_{support_count}", support))
        support_count += 1

    return tuple(selected)


def _best_distinct_sample(
    scored_samples: tuple[_ScoredSample, ...],
    *,
    preferred: _ScoredSample,
    used_indices: set[int],
    selected: list[tuple[str, _ScoredSample]],
    window_start_s: float,
    window_end_s: float,
    config: KeyframeReviewConfig,
    prefer_anchor: bool,
) -> _ScoredSample | None:
    """Return the preferred anchor or the best compatible fallback."""

    if preferred.index not in used_indices and _is_temporally_distinct(
        preferred.sample.observed_at,
        selected,
        window_start_s=window_start_s,
        window_end_s=window_end_s,
        config=config,
    ):
        return preferred

    candidates = sorted(
        scored_samples,
        key=lambda item: (
            _anchor_fallback_score(
                item,
                selected=selected,
                window_start_s=window_start_s,
                window_end_s=window_end_s,
                config=config,
                prefer_anchor=prefer_anchor,
            ),
            item.base_score,
            -abs(item.index - preferred.index),
            item.sample.observed_at,
        ),
        reverse=True,
    )
    for candidate in candidates:
        if candidate.index in used_indices:
            continue
        if not _is_temporally_distinct(
            candidate.sample.observed_at,
            selected,
            window_start_s=window_start_s,
            window_end_s=window_end_s,
            config=config,
        ):
            continue
        return candidate
    return None


def _best_support_sample(
    scored_samples: tuple[_ScoredSample, ...],
    *,
    used_indices: set[int],
    selected: list[tuple[str, _ScoredSample]],
    window_start_s: float,
    window_end_s: float,
    config: KeyframeReviewConfig,
) -> _ScoredSample | None:
    """Greedily add a high-value, non-redundant support frame."""

    best_candidate: _ScoredSample | None = None
    best_score = float("-inf")
    for candidate in scored_samples:
        if candidate.index in used_indices:
            continue
        if not _is_temporally_distinct(
            candidate.sample.observed_at,
            selected,
            window_start_s=window_start_s,
            window_end_s=window_end_s,
            config=config,
        ):
            continue
        score = _support_score(
            candidate,
            selected=selected,
            window_start_s=window_start_s,
            window_end_s=window_end_s,
            config=config,
        )
        if score > best_score:
            best_score = score
            best_candidate = candidate
    return best_candidate


def _anchor_fallback_score(
    candidate: _ScoredSample,
    *,
    selected: list[tuple[str, _ScoredSample]],
    window_start_s: float,
    window_end_s: float,
    config: KeyframeReviewConfig,
    prefer_anchor: bool,
) -> float:
    """Score one anchor fallback candidate."""

    coverage = _normalized_distance_to_selected(
        candidate.sample.observed_at,
        selected=selected,
        window_start_s=window_start_s,
        window_end_s=window_end_s,
    )
    boundary_bonus = candidate.boundary_score * 0.15
    anchor_bonus = coverage * 0.20 if prefer_anchor else coverage * 0.10
    return candidate.selection_score + boundary_bonus + anchor_bonus


def _support_score(
    candidate: _ScoredSample,
    *,
    selected: list[tuple[str, _ScoredSample]],
    window_start_s: float,
    window_end_s: float,
    config: KeyframeReviewConfig,
) -> float:
    """Score one support candidate with diversity pressure."""

    coverage = _normalized_distance_to_selected(
        candidate.sample.observed_at,
        selected=selected,
        window_start_s=window_start_s,
        window_end_s=window_end_s,
    )
    return candidate.selection_score + (config.diversity_weight * coverage)


def _normalized_distance_to_selected(
    observed_at: float,
    *,
    selected: list[tuple[str, _ScoredSample]],
    window_start_s: float,
    window_end_s: float,
) -> float:
    """Return distance-to-selected normalized by the review span."""

    if not selected:
        return 1.0
    span = max(1e-6, window_end_s - window_start_s)
    nearest = min(abs(observed_at - item.sample.observed_at) for _, item in selected)
    return min(1.0, nearest / span)


def _is_temporally_distinct(
    observed_at: float,
    selected: list[tuple[str, _ScoredSample]],
    *,
    window_start_s: float,
    window_end_s: float,
    config: KeyframeReviewConfig,
) -> bool:
    """Enforce a small minimum time separation between selected frames."""

    if not selected:
        return True
    span = max(0.0, window_end_s - window_start_s)
    dynamic_floor = span / max(3.0, float(config.max_frames) * 2.0) if span > 0.0 else 0.0
    min_gap = min(max(0.0, config.min_frame_separation_s), dynamic_floor) if dynamic_floor > 0.0 else max(0.0, config.min_frame_separation_s)
    if span > 0.0:
        min_gap = min(min_gap, span / 2.0)
    for _, item in selected:
        if abs(observed_at - item.sample.observed_at) <= max(1e-6, min_gap):
            return False
    return True


def _to_candidate(
    *,
    sample: TimedSample[SocialVisionObservation],
    role: str,
    relevance_score: float,
) -> ReviewKeyframeCandidate:
    """Convert one scored observation sample into a review candidate."""

    observation = sample.value
    return ReviewKeyframeCandidate(
        observed_at=float(sample.observed_at),
        role=role,
        relevance_score=round(relevance_score, 4),
        person_visible=bool(getattr(observation, "person_visible", False)),
        person_count=_safe_nonnegative_int(getattr(observation, "person_count", 0)),
        body_pose=_enum_value(getattr(observation, "body_pose", None)),
        motion_state=_enum_value(getattr(observation, "motion_state", None)),
        looking_toward_device=bool(getattr(observation, "looking_toward_device", False)),
        smiling=bool(getattr(observation, "smiling", False)),
    )


def _candidate_score(
    *,
    sample: TimedSample[SocialVisionObservation],
    now: float,
    vision_evidence: tuple[_PreparedSequenceEvent, ...],
    audio_evidence: tuple[_PreparedAudioEvent, ...],
    config: KeyframeReviewConfig,
) -> float:
    """Score one observation sample using bounded multimodal relevance plus scene salience."""

    observation = sample.value
    observed_at = float(sample.observed_at)

    vision_score = _bounded_union(
        _sanitize_confidence(sequence.confidence)
        * _sequence_alignment_weight(
            observed_at,
            sequence_start_s=sequence.window_start_s,
            sequence_end_s=sequence.window_end_s,
            half_life_s=config.vision_alignment_half_life_s,
        )
        for sequence in vision_evidence
    )
    audio_score = 0.5 * _bounded_union(
        _sanitize_confidence(event.confidence)
        * _exp_half_life(
            abs(observed_at - event.observed_at),
            half_life_s=config.audio_alignment_half_life_s,
        )
        for event in audio_evidence
    )

    pose = _enum_value(getattr(observation, "body_pose", None))
    motion = _enum_value(getattr(observation, "motion_state", None))
    person_visible = bool(getattr(observation, "person_visible", False))
    looking_toward_device = bool(getattr(observation, "looking_toward_device", False))
    engaged_with_device = bool(getattr(observation, "engaged_with_device", False))
    smiling = bool(getattr(observation, "smiling", False))
    hand_or_object_near_camera = bool(getattr(observation, "hand_or_object_near_camera", False))
    showing_intent_likely = bool(getattr(observation, "showing_intent_likely", False))
    person_count = _safe_nonnegative_int(getattr(observation, "person_count", 0))

    scene_score = 0.0
    if person_visible:
        scene_score += 0.12
        scene_score += min(0.08, 0.02 * max(0, person_count - 1))
    if pose in {SocialBodyPose.FLOOR.value, SocialBodyPose.LYING_LOW.value}:
        scene_score += 0.20
    elif pose == SocialBodyPose.SLUMPED.value:
        scene_score += 0.14
    if smiling and (looking_toward_device or engaged_with_device):
        scene_score += 0.14
    if hand_or_object_near_camera or showing_intent_likely:
        scene_score += 0.12
    if motion == SocialMotionState.STILL.value:
        scene_score += 0.05

    recency_score = 0.05 * _exp_half_life(max(0.0, now - observed_at), half_life_s=config.recency_half_life_s)
    return round(vision_score + audio_score + scene_score + recency_score, 6)


def _boundary_score(
    samples: tuple[TimedSample[SocialVisionObservation], ...],
    base_scores: list[float],
    index: int,
) -> float:
    """Estimate how strongly this sample sits on a semantic transition."""

    current = samples[index].value
    previous_change = 0.0
    next_change = 0.0
    previous_score_change = 0.0
    next_score_change = 0.0

    if index > 0:
        previous_change = _observation_change(samples[index - 1].value, current)
        previous_score_change = abs(base_scores[index] - base_scores[index - 1])
    if index + 1 < len(samples):
        next_change = _observation_change(current, samples[index + 1].value)
        next_score_change = abs(base_scores[index] - base_scores[index + 1])

    return max(previous_change, next_change) + 0.5 * max(previous_score_change, next_score_change)


def _local_change_score(
    samples: list[TimedSample[SocialVisionObservation]],
    index: int,
) -> float:
    """Cheap change estimate used for candidate thinning."""

    previous_value = samples[index - 1].value
    current_value = samples[index].value
    next_value = samples[index + 1].value
    return max(
        _observation_change(previous_value, current_value),
        _observation_change(current_value, next_value),
    )


def _observation_change(
    before: SocialVisionObservation,
    after: SocialVisionObservation,
) -> float:
    """Approximate semantic change magnitude between adjacent observations."""

    score = 0.0
    before_visible = bool(getattr(before, "person_visible", False))
    after_visible = bool(getattr(after, "person_visible", False))
    if before_visible != after_visible:
        score += 0.8

    before_count = _safe_nonnegative_int(getattr(before, "person_count", 0))
    after_count = _safe_nonnegative_int(getattr(after, "person_count", 0))
    score += min(0.3, 0.12 * abs(after_count - before_count))

    if _enum_value(getattr(before, "body_pose", None)) != _enum_value(getattr(after, "body_pose", None)):
        score += 0.55
    if _enum_value(getattr(before, "motion_state", None)) != _enum_value(getattr(after, "motion_state", None)):
        score += 0.35

    boolean_fields = (
        "looking_toward_device",
        "smiling",
        "engaged_with_device",
        "hand_or_object_near_camera",
        "showing_intent_likely",
    )
    for field_name in boolean_fields:
        if bool(getattr(before, field_name, False)) != bool(getattr(after, field_name, False)):
            score += 0.18

    return round(score, 6)


def _sequence_alignment_weight(
    observed_at: float,
    *,
    sequence_start_s: float,
    sequence_end_s: float,
    half_life_s: float,
) -> float:
    """Weight one sample by how well it aligns to a sequence window."""

    if sequence_start_s <= observed_at <= sequence_end_s:
        return 1.0
    distance = min(
        abs(observed_at - sequence_start_s),
        abs(observed_at - sequence_end_s),
    )
    return _exp_half_life(distance, half_life_s=half_life_s)


def _interval_distance(
    left_start_s: float,
    left_end_s: float,
    right_start_s: float,
    right_end_s: float,
) -> float:
    """Return the distance between two closed time intervals."""

    if right_end_s < left_start_s:
        return left_start_s - right_end_s
    if right_start_s > left_end_s:
        return right_start_s - left_end_s
    return 0.0


def _bounded_union(weights: Iterable[float]) -> float:
    """Combine bounded evidence weights without letting fragmentation dominate."""

    remaining = 1.0
    saw_any = False
    for raw_weight in weights:
        weight = _clamp(_finite_or_default(raw_weight, 0.0), 0.0, 1.0)
        if weight <= 0.0:
            continue
        saw_any = True
        remaining *= (1.0 - weight)
    return 0.0 if not saw_any else (1.0 - remaining)


def _sanitize_confidence(value: object | None) -> float:
    """Return a bounded, non-negative confidence value."""

    raw = _finite_or_default(value, 0.0)
    if raw <= 0.0:
        return 0.0
    if raw <= 1.0:
        return raw
    return raw / (1.0 + raw)


def _exp_half_life(age_s: float, *, half_life_s: float) -> float:
    """Return an exponential decay weight for one age/half-life pair."""

    if half_life_s <= 0.0:
        return 0.0
    return math.exp(-math.log(2.0) * max(0.0, age_s) / half_life_s)


def _enum_value(value: object | None) -> str:
    """Return the string value for one enum-like field."""

    raw = getattr(value, "value", value)
    return "unknown" if raw is None else str(raw)


def _safe_nonnegative_int(value: object | None) -> int:
    """Return a non-negative integer derived from one numeric-like value."""

    finite_value = _finite_or_default(value, 0.0)
    if finite_value <= 0.0:
        return 0
    return max(0, int(finite_value))


def _is_finite_number(value: object | None) -> bool:
    """Return True when value can be interpreted as a finite float."""

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _finite_or_default(value: object | None, default: float) -> float:
    """Return a finite float or a default fallback."""

    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return default
    return candidate if math.isfinite(candidate) else default


def _clamp(value: float, lower: float, upper: float) -> float:
    """Clamp one float to the inclusive [lower, upper] interval."""

    return max(lower, min(upper, value))


__all__ = [
    "KeyframeReviewConfig",
    "KeyframeReviewPlan",
    "ReviewKeyframeCandidate",
    "build_keyframe_review_plan",
]