# CHANGELOG: 2026-03-29
# BUG-1: Distinguish unknown person visibility from confirmed no-visible-person; abstain instead of emitting incorrect block_reason.
# BUG-2: Prevent false "none" outputs when evidence is sparse, stale, or mixed across async modalities.
# SEC-1: Sanitize upstream categorical strings before re-emitting them into automation/prompt payloads to reduce prompt/data-injection risk from compromised sensors.
# IMP-1: Add consensus-style multimodal scoring with conflict penalties, missing-modality handling, and freshness gating informed by 2025-2026 MER research.
# IMP-2: Add optional baseline/duration hooks and calibrated confidence generation while preserving the public API.

"""Derive coarse affect proxies without claiming emotion as fact.

The output of this module is intentionally small and prompt-oriented. It may
support a calm follow-up question, but it must never be treated as a diagnosis,
emotion label, or durable truth.

Design notes for runtime integration:
- This function is intentionally conservative and now abstains when evidence is
  sparse, stale, or contradictory.
- It accepts optional per-modality timestamps, durations, and baseline deltas
  from `live_facts` when available, but remains drop-in compatible when they
  are absent.
- All upstream free-text categorical values are sanitized before they are
  echoed into prompt- or automation-facing payloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re

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

_FRESHNESS_WINDOW_S = 2.5
_MIN_DECISIVE_COVERAGE = 0.55
_MODALITY_CLOCK_SKEW_S = 0.75
_CONFLICT_MARGIN = 0.18

_BODY_POSE_ALIAS = {
    "laying": "lying_low",
    "lying": "lying_low",
    "prone": "lying_low",
    "supine": "lying_low",
    "on_floor": "floor",
    "ground": "floor",
    "reclined": "lying_low",
}
_ALLOWED_BODY_POSES = frozenset(
    {
        "floor",
        "lying_low",
        "slumped",
        "standing",
        "upright",
        "sitting",
        "seated",
        "walking",
        "kneeling",
        "bending",
    }
)
_SAFE_LABEL_RE = re.compile(r"[^a-z0-9_]+")


def _default_claim() -> RuntimeClaimMetadata:
    return RuntimeClaimMetadata(
        confidence=0.0,
        source="camera_pose_attention_audio",
        requires_confirmation=False,
    )


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
    evidence_tags: tuple[str, ...] = ()
    uncertainty_flags: tuple[str, ...] = ()

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
        if self.evidence_tags:
            payload["evidence_tags"] = list(self.evidence_tags)
        if self.uncertainty_flags:
            payload["uncertainty_flags"] = list(self.uncertainty_flags)
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
    freshness_window_s: float = _FRESHNESS_WINDOW_S,
    min_decisive_coverage: float = _MIN_DECISIVE_COVERAGE,
) -> AffectProxySnapshot:
    """Return one conservative affect-proxy snapshot.

    Optional `live_facts` extensions used when present:
    - `<modality>.observed_at` / `updated_at` / `timestamp` / `ts`
    - `<signal>_duration_s` or `<signal>_streak_s`
    - `camera.face_presence_confidence`, `camera.tracking_confidence`,
      `camera.signal_quality`
    - `camera.smile_delta`, `camera.attention_delta`
    - top-level `affect_baseline` mapping containing the same delta fields
    """

    observed_at = _coerce_optional_timestamp(observed_at)
    guard = (
        ambiguous_room_guard
        if ambiguous_room_guard is not None
        else derive_ambiguous_room_guard(
            observed_at=observed_at,
            live_facts=live_facts,
        )
    )

    facts = coerce_mapping(live_facts)
    camera = coerce_mapping(facts.get("camera"))
    vad = coerce_mapping(facts.get("vad"))
    pir = coerce_mapping(facts.get("pir"))
    baseline = coerce_mapping(facts.get("affect_baseline"))

    camera_observed_at = _extract_timestamp(camera)
    vad_observed_at = _extract_timestamp(vad)
    pir_observed_at = _extract_timestamp(pir)

    camera_fresh = _is_fresh(
        modality_observed_at=camera_observed_at,
        observed_at=observed_at,
        freshness_window_s=freshness_window_s,
    )
    vad_fresh = _is_fresh(
        modality_observed_at=vad_observed_at,
        observed_at=observed_at,
        freshness_window_s=freshness_window_s,
    )
    pir_fresh = _is_fresh(
        modality_observed_at=pir_observed_at,
        observed_at=observed_at,
        freshness_window_s=freshness_window_s,
    )

    uncertainty_flags: list[str] = []
    if observed_at is not None and camera_observed_at is not None and not camera_fresh:
        uncertainty_flags.append("camera_stale")
    if observed_at is not None and vad_observed_at is not None and not vad_fresh:
        uncertainty_flags.append("vad_stale")
    if observed_at is not None and pir_observed_at is not None and not pir_fresh:
        uncertainty_flags.append("pir_stale")

    camera_quality = mean_confidence(
        (
            _known_ratio(camera, "signal_quality"),
            _known_ratio(camera, "face_presence_confidence"),
            _known_ratio(camera, "tracking_confidence"),
        )
    )
    if camera_quality is not None and camera_quality < 0.30:
        camera_fresh = False
        uncertainty_flags.append("camera_low_quality")

    person_visible = _known_bool(camera, "person_visible") if camera_fresh else None
    body_pose = _sanitize_body_pose(_known_text(camera, "body_pose")) if camera_fresh else None
    smiling = _known_bool(camera, "smiling") if camera_fresh else None
    looking_toward_device = _known_bool(camera, "looking_toward_device") if camera_fresh else None
    engaged_with_device = _known_bool(camera, "engaged_with_device") if camera_fresh else None
    room_quiet = coerce_optional_bool(vad.get("room_quiet")) if vad_fresh else None
    low_motion = coerce_optional_bool(pir.get("low_motion")) if pir_fresh else None
    attention_score = _known_ratio(camera, "visual_attention_score") if camera_fresh else None

    smile_delta = _coalesce_float(
        _extract_float(camera, "smile_delta"),
        _extract_float(camera, "smiling_delta"),
        _extract_float(baseline, "smile_delta"),
    )
    attention_delta = _coalesce_float(
        _extract_float(camera, "attention_delta"),
        _extract_float(camera, "visual_attention_delta"),
        _extract_float(baseline, "attention_delta"),
    )

    smiling_duration_s = _extract_duration(camera, "smiling") if camera_fresh else None
    looking_duration_s = _extract_duration(camera, "looking_toward_device") if camera_fresh else None
    engaged_duration_s = _extract_duration(camera, "engaged_with_device") if camera_fresh else None
    quiet_duration_s = _extract_duration(vad, "room_quiet") if vad_fresh else None
    low_motion_duration_s = _extract_duration(pir, "low_motion") if pir_fresh else None
    body_pose_duration_s = _extract_duration(camera, "body_pose") if camera_fresh else None

    if guard.guard_active:
        return AffectProxySnapshot(
            observed_at=observed_at,
            state="unknown",
            policy_recommendation="ignore",
            block_reason=_sanitize_label(guard.reason) or "guard_active",
            claim=RuntimeClaimMetadata(
                confidence=_finite_or_default(getattr(guard.claim, "confidence", None), default=0.0),
                source="camera_pose_attention_audio_guarded",
                requires_confirmation=False,
            ),
            body_pose=body_pose,
            smiling=smiling,
            looking_toward_device=looking_toward_device,
            engaged_with_device=engaged_with_device,
            room_quiet=room_quiet,
            low_motion=low_motion,
            uncertainty_flags=tuple(_dedupe(uncertainty_flags)),
        )

    if person_visible is False:
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
            uncertainty_flags=tuple(_dedupe(uncertainty_flags)),
        )
    if person_visible is None:
        flags = _dedupe([*uncertainty_flags, "person_visibility_unknown"])
        return AffectProxySnapshot(
            observed_at=observed_at,
            state="unknown",
            policy_recommendation="ignore",
            block_reason="person_visibility_unknown",
            claim=RuntimeClaimMetadata(
                confidence=0.0,
                source="camera_pose_attention_audio_abstain",
                requires_confirmation=False,
            ),
            body_pose=body_pose,
            smiling=smiling,
            looking_toward_device=looking_toward_device,
            engaged_with_device=engaged_with_device,
            room_quiet=room_quiet,
            low_motion=low_motion,
            uncertainty_flags=tuple(flags),
        )

    coverage = _coverage_score(
        body_pose=body_pose,
        smiling=smiling,
        looking_toward_device=looking_toward_device,
        engaged_with_device=engaged_with_device,
        room_quiet=room_quiet,
        low_motion=low_motion,
        attention_score=attention_score,
    )

    evidence_tags: list[str] = []

    concern_score = 0.0
    concern_source = "camera_pose_attention_audio_consensus"
    if body_pose in {"floor", "lying_low"}:
        evidence_tags.append(f"body_pose_{body_pose}")
        concern_score += 0.72
        concern_source = "camera_pose_plus_audio_quiet"
        concern_score += _duration_boost(body_pose_duration_s, quick_threshold_s=1.0, full_threshold_s=3.0)
        if room_quiet is True:
            evidence_tags.append("room_quiet")
            concern_score += 0.12
        if low_motion is True:
            evidence_tags.append("low_motion")
            concern_score += 0.08
            concern_source = "camera_pose_plus_pir_plus_audio"
        if looking_toward_device is False:
            evidence_tags.append("not_looking_toward_device")
            concern_score += 0.03
        if engaged_with_device is False:
            evidence_tags.append("not_engaged_with_device")
            concern_score += 0.03
        if attention_delta is not None and attention_delta <= -0.20:
            evidence_tags.append("attention_below_baseline")
            concern_score += 0.04
        if smiling is True:
            concern_score -= 0.12
    elif body_pose == "slumped":
        evidence_tags.append("body_pose_slumped")
        concern_score += 0.40
        concern_source = "camera_pose_plus_pir_plus_audio"
        concern_score += _duration_boost(body_pose_duration_s, quick_threshold_s=1.5, full_threshold_s=4.0)
        if low_motion is True:
            evidence_tags.append("low_motion")
            concern_score += 0.16 + _duration_boost(low_motion_duration_s, quick_threshold_s=2.0, full_threshold_s=8.0)
        if room_quiet is True:
            evidence_tags.append("room_quiet")
            concern_score += 0.12 + _duration_boost(quiet_duration_s, quick_threshold_s=2.0, full_threshold_s=8.0)
        if attention_score is not None and attention_score <= 0.25:
            evidence_tags.append("low_visual_attention")
            concern_score += 0.06
        if attention_delta is not None and attention_delta <= -0.20:
            evidence_tags.append("attention_below_baseline")
            concern_score += 0.04
        if smiling is True:
            concern_score -= 0.10
    concern_score = _clamp01(concern_score)

    positive_score = 0.0
    positive_source = "camera_expression_plus_attention"
    if smiling is True:
        evidence_tags.append("smiling")
        positive_score += 0.34 + _duration_boost(smiling_duration_s, quick_threshold_s=0.4, full_threshold_s=2.5)
        if smile_delta is not None and smile_delta >= 0.20:
            evidence_tags.append("smile_above_baseline")
            positive_score += 0.06
        if looking_toward_device is True:
            evidence_tags.append("looking_toward_device")
            positive_score += 0.18 + _duration_boost(looking_duration_s, quick_threshold_s=0.5, full_threshold_s=2.5)
        if engaged_with_device is True:
            evidence_tags.append("engaged_with_device")
            positive_score += 0.22 + _duration_boost(engaged_duration_s, quick_threshold_s=0.5, full_threshold_s=3.0)
        if attention_score is not None:
            positive_score += max(0.0, min(0.12, (attention_score - 0.60) * 0.30))
    positive_score = _clamp01(positive_score)

    low_engagement_score = 0.0
    low_engagement_source = "camera_attention_plus_audio_quiet"
    if room_quiet is True:
        evidence_tags.append("room_quiet")
        low_engagement_score += 0.22 + _duration_boost(quiet_duration_s, quick_threshold_s=2.0, full_threshold_s=8.0)
    if looking_toward_device is False:
        evidence_tags.append("not_looking_toward_device")
        low_engagement_score += 0.18 + _duration_boost(looking_duration_s, quick_threshold_s=1.0, full_threshold_s=4.0)
    if engaged_with_device is False:
        evidence_tags.append("not_engaged_with_device")
        low_engagement_score += 0.22 + _duration_boost(engaged_duration_s, quick_threshold_s=1.0, full_threshold_s=4.0)
    if low_motion is True:
        evidence_tags.append("low_motion")
        low_engagement_score += 0.08 + _duration_boost(low_motion_duration_s, quick_threshold_s=2.0, full_threshold_s=8.0)
        low_engagement_source = "camera_attention_plus_audio_quiet_plus_pir"
    if attention_score is not None and attention_score <= 0.25:
        evidence_tags.append("low_visual_attention")
        low_engagement_score += 0.12
    if attention_delta is not None and attention_delta <= -0.20:
        evidence_tags.append("attention_below_baseline")
        low_engagement_score += 0.04
    if smiling is True:
        low_engagement_score -= 0.12
    low_engagement_score = _clamp01(low_engagement_score)

    state_scores = {
        "concern_cue": concern_score,
        "positive_contact": positive_score,
        "low_engagement": low_engagement_score,
    }
    thresholds = {
        "concern_cue": 0.62,
        "positive_contact": 0.55,
        "low_engagement": 0.58,
    }

    ranked = sorted(state_scores.items(), key=lambda item: item[1], reverse=True)
    best_state, best_score = ranked[0]
    second_state, second_score = ranked[1]

    if best_score >= thresholds[best_state]:
        second_threshold = thresholds[second_state]
        if second_score >= max(0.50, second_threshold - 0.03) and (best_score - second_score) < _CONFLICT_MARGIN:
            flags = _dedupe([*uncertainty_flags, "conflicting_multimodal_cues"])
            return AffectProxySnapshot(
                observed_at=observed_at,
                state="unknown",
                policy_recommendation="ignore",
                block_reason="conflicting_multimodal_cues",
                claim=RuntimeClaimMetadata(
                    confidence=_clamp01(max(best_score, second_score) * 0.45),
                    source="camera_pose_attention_audio_abstain",
                    requires_confirmation=False,
                ),
                body_pose=body_pose,
                smiling=smiling,
                looking_toward_device=looking_toward_device,
                engaged_with_device=engaged_with_device,
                room_quiet=room_quiet,
                low_motion=low_motion,
                evidence_tags=tuple(_dedupe(evidence_tags)),
                uncertainty_flags=tuple(flags),
            )

        state_source = {
            "concern_cue": concern_source,
            "positive_contact": positive_source,
            "low_engagement": low_engagement_source,
        }[best_state]
        confidence = _calibrated_state_confidence(
            support=best_score,
            coverage=coverage,
            second_best=second_score,
            uncertainty_count=len(uncertainty_flags),
            camera_quality=camera_quality,
        )
        return AffectProxySnapshot(
            observed_at=observed_at,
            state=best_state,
            policy_recommendation="prompt_only",
            block_reason=None,
            claim=RuntimeClaimMetadata(
                confidence=confidence,
                source=state_source,
                requires_confirmation=True,
            ),
            body_pose=body_pose,
            smiling=smiling,
            looking_toward_device=looking_toward_device,
            engaged_with_device=engaged_with_device,
            room_quiet=room_quiet,
            low_motion=low_motion,
            evidence_tags=tuple(_dedupe(evidence_tags)),
            uncertainty_flags=tuple(_dedupe(uncertainty_flags)),
        )

    if coverage < min_decisive_coverage:
        flags = _dedupe([*uncertainty_flags, "insufficient_evidence"])
        return AffectProxySnapshot(
            observed_at=observed_at,
            state="unknown",
            policy_recommendation="ignore",
            block_reason="insufficient_evidence",
            claim=RuntimeClaimMetadata(
                confidence=0.0,
                source="camera_pose_attention_audio_abstain",
                requires_confirmation=False,
            ),
            body_pose=body_pose,
            smiling=smiling,
            looking_toward_device=looking_toward_device,
            engaged_with_device=engaged_with_device,
            room_quiet=room_quiet,
            low_motion=low_motion,
            uncertainty_flags=tuple(flags),
        )

    none_confidence = _calibrated_none_confidence(
        coverage=coverage,
        best_support=best_score,
        uncertainty_count=len(uncertainty_flags),
        camera_quality=camera_quality,
    )
    return AffectProxySnapshot(
        observed_at=observed_at,
        state="none",
        policy_recommendation="ignore",
        block_reason=None,
        claim=RuntimeClaimMetadata(
            confidence=none_confidence,
            source="camera_pose_attention_audio_consensus",
            requires_confirmation=False,
        ),
        body_pose=body_pose,
        smiling=smiling,
        looking_toward_device=looking_toward_device,
        engaged_with_device=engaged_with_device,
        room_quiet=room_quiet,
        low_motion=low_motion,
        evidence_tags=tuple(_dedupe(evidence_tags)),
        uncertainty_flags=tuple(_dedupe(uncertainty_flags)),
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


def _extract_timestamp(payload: dict[str, object]) -> float | None:
    return _coalesce_float(
        _coerce_optional_timestamp(payload.get("observed_at")),
        _coerce_optional_timestamp(payload.get("updated_at")),
        _coerce_optional_timestamp(payload.get("timestamp")),
        _coerce_optional_timestamp(payload.get("ts")),
    )


def _extract_duration(payload: dict[str, object], field_name: str) -> float | None:
    return _coalesce_float(
        _coerce_optional_nonnegative_float(payload.get(f"{field_name}_duration_s")),
        _coerce_optional_nonnegative_float(payload.get(f"{field_name}_streak_s")),
    )


def _extract_float(payload: dict[str, object], field_name: str) -> float | None:
    return _coerce_optional_float(payload.get(field_name))


def _is_fresh(
    *,
    modality_observed_at: float | None,
    observed_at: float | None,
    freshness_window_s: float,
) -> bool:
    if modality_observed_at is None or observed_at is None:
        return True
    age_s = observed_at - modality_observed_at
    if age_s < -_MODALITY_CLOCK_SKEW_S:
        return False
    return age_s <= max(0.0, freshness_window_s)


def _coverage_score(
    *,
    body_pose: str | None,
    smiling: bool | None,
    looking_toward_device: bool | None,
    engaged_with_device: bool | None,
    room_quiet: bool | None,
    low_motion: bool | None,
    attention_score: float | None,
) -> float:
    coverage = 0.0
    if body_pose is not None:
        coverage += 0.18
    if smiling is not None:
        coverage += 0.14
    if looking_toward_device is not None:
        coverage += 0.14
    if engaged_with_device is not None:
        coverage += 0.16
    if room_quiet is not None:
        coverage += 0.16
    if low_motion is not None:
        coverage += 0.12
    if attention_score is not None:
        coverage += 0.10
    return _clamp01(coverage)


def _calibrated_state_confidence(
    *,
    support: float,
    coverage: float,
    second_best: float,
    uncertainty_count: int,
    camera_quality: float | None,
) -> float:
    confidence = support * 0.70 + coverage * 0.22
    confidence -= max(0.0, 0.14 - max(0.0, support - second_best))
    confidence -= min(0.18, uncertainty_count * 0.04)
    if camera_quality is not None:
        confidence = (confidence * 0.80) + (camera_quality * 0.20)
    return _clamp01(max(0.0, confidence))


def _calibrated_none_confidence(
    *,
    coverage: float,
    best_support: float,
    uncertainty_count: int,
    camera_quality: float | None,
) -> float:
    confidence = 0.20 + coverage * 0.48 - best_support * 0.25
    confidence -= min(0.16, uncertainty_count * 0.04)
    if camera_quality is not None:
        confidence = (confidence * 0.85) + (camera_quality * 0.15)
    return _clamp01(max(0.0, confidence))


def _duration_boost(
    duration_s: float | None,
    *,
    quick_threshold_s: float,
    full_threshold_s: float,
) -> float:
    if duration_s is None or duration_s < quick_threshold_s or full_threshold_s <= quick_threshold_s:
        return 0.0
    normalized = (duration_s - quick_threshold_s) / (full_threshold_s - quick_threshold_s)
    return min(0.08, max(0.0, normalized) * 0.08)


def _sanitize_body_pose(value: str | None) -> str | None:
    text = _sanitize_label(value)
    if not text:
        return None
    text = _BODY_POSE_ALIAS.get(text, text)
    if text not in _ALLOWED_BODY_POSES:
        return None
    return text


def _sanitize_label(value: str | None) -> str | None:
    text = normalize_text(value)
    if not text:
        return None
    text = text.lower().replace("-", "_").replace(" ", "_")
    text = _SAFE_LABEL_RE.sub("_", text)
    text = text.strip("_")
    if not text:
        return None
    return text[:32]


def _coerce_optional_timestamp(value: object) -> float | None:
    result = _coerce_optional_float(value)
    if result is None:
        return None
    if result < 0.0:
        return None
    return result


def _coerce_optional_nonnegative_float(value: object) -> float | None:
    result = _coerce_optional_float(value)
    if result is None or result < 0.0:
        return None
    return result


def _coerce_optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _coalesce_float(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def _finite_or_default(value: object, *, default: float) -> float:
    result = _coerce_optional_float(value)
    return default if result is None else result


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 0.98:
        return 0.98
    return value


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        safe_value = _sanitize_label(value)
        if not safe_value or safe_value in seen:
            continue
        seen.add(safe_value)
        result.append(safe_value)
    return result


__all__ = [
    "AffectProxySnapshot",
    "derive_affect_proxy",
]