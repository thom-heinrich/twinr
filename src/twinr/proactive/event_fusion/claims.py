"""Define fused event claims and their fail-closed delivery gates."""

from __future__ import annotations

# CHANGELOG: 2026-03-29
# BUG-1: Replace fragile `is True` sensor-flag checks with bool-like scalar handling so
#        upstream NumPy / int-backed flags cannot silently bypass delivery blocks.
# BUG-2: Reject non-finite / out-of-range confidence values and invalid time windows so
#        malformed claims cannot leak invalid JSON or negative-duration windows downstream.
# BUG-3: Auto-enable review metadata when a claim is REVIEW_ONLY or carries a keyframe
#        plan, preventing silent suppression of intended human review.
# SEC-1: Sanitize and bound externally sourced labels / reason codes to reduce practical
#        log-injection and payload-amplification risk on resource-constrained edge nodes.
# IMP-1: Preserve `preferred_action_level` and emit explicit policy-version / abstention
#        metadata for auditability and post-hoc governance.
# IMP-2: Extend policy context with cross-modal conflict and extra blocked reasons so
#        future uncertainty-aware / inconsistency-aware fusion can fail closed without
#        another schema rewrite.

import math
import operator
import unicodedata
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, Iterable

from twinr.proactive.event_fusion.review import KeyframeReviewPlan
from twinr.proactive.social.engine import SocialObservation


_FUSED_EVENT_POLICY_VERSION: Final[str] = "2026-03-29.fused-claim-v2"
_MAX_LABEL_LEN: Final[int] = 128
_MAX_CODE_LEN: Final[int] = 64
_MAX_SUPPORTING_EVENTS_PER_MODALITY: Final[int] = 16
_MAX_BLOCK_REASONS: Final[int] = 16
_ALLOWED_CODE_CHARS: Final[frozenset[str]] = frozenset(
    "abcdefghijklmnopqrstuvwxyz0123456789_-.:"
)


class FusionActionLevel(StrEnum):
    """Describe how far one fused event may travel in runtime policy."""

    IGNORE = "ignore"
    DIRECT = "direct"
    PROMPT_ONLY = "prompt_only"
    REVIEW_ONLY = "review_only"


def _flag_is_true(value: object) -> bool:
    """Return True for bool-like scalar truth values, else False."""
    if value is None or isinstance(value, str):
        return False
    try:
        raw = operator.index(value)
    except TypeError:
        return value is True
    return raw == 1


def _coerce_boolish(name: str, value: object) -> bool:
    """Coerce a bool-like scalar (bool / NumPy bool / 0 / 1) into bool."""
    if isinstance(value, str):
        raise TypeError(f"{name} must be boolean-like, not str")
    try:
        raw = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a bool-like scalar, got {type(value).__name__}") from exc
    if raw not in (0, 1):
        raise TypeError(f"{name} must be boolean-like 0/1, got {value!r}")
    return bool(raw)


def _coerce_action_level(name: str, value: FusionActionLevel | str) -> FusionActionLevel:
    """Parse an action level from enum or serialized string form."""
    if isinstance(value, FusionActionLevel):
        return value
    if isinstance(value, str):
        try:
            return FusionActionLevel(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a valid FusionActionLevel, got {value!r}") from exc
    raise TypeError(f"{name} must be FusionActionLevel | str, got {type(value).__name__}")


def _sanitize_label(value: object, *, field_name: str, max_len: int = _MAX_LABEL_LEN) -> str:
    """Normalize a human-readable short label into a bounded printable token."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str, got {type(value).__name__}")
    normalized = unicodedata.normalize("NFKC", value)
    normalized = " ".join(normalized.split())
    normalized = "".join(ch for ch in normalized if ch.isprintable())
    normalized = normalized.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if len(normalized) > max_len:
        normalized = normalized[:max_len].rstrip()
    return normalized


def _sanitize_code(value: object, *, field_name: str, max_len: int = _MAX_CODE_LEN) -> str:
    """Normalize a machine-readable reason / code into a bounded safe token."""
    label = _sanitize_label(value, field_name=field_name, max_len=max_len)
    label = label.lower().replace(" ", "_")
    label = "".join(ch for ch in label if ch in _ALLOWED_CODE_CHARS)
    if not label:
        raise ValueError(f"{field_name} must contain at least one safe code character")
    return label[:max_len]


def _normalize_string_sequence(
    values: object,
    *,
    field_name: str,
    max_items: int,
    item_kind: str,
) -> tuple[str, ...]:
    """Normalize one incoming sequence of short strings into a bounded deduped tuple."""
    if values is None:
        return ()
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be an iterable of strings, not a single string")
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be iterable, got {type(values).__name__}") from exc

    items: list[str] = []
    seen: set[str] = set()
    for raw in iterator:
        item = (
            _sanitize_code(raw, field_name=field_name)
            if item_kind == "code"
            else _sanitize_label(raw, field_name=field_name)
        )
        if item in seen:
            continue
        seen.add(item)
        items.append(item)
        if len(items) >= max_items:
            break
    return tuple(items)


def _coerce_finite_float(name: str, value: object) -> float:
    """Convert a numeric scalar into a finite float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be int | float, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _normalize_confidence(value: object) -> float:
    """Validate a closed-world confidence score."""
    confidence = _coerce_finite_float("confidence", value)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be within [0.0, 1.0], got {confidence!r}")
    return confidence


def _normalize_optional_time(name: str, value: object) -> float | None:
    """Validate one optional event-window timestamp."""
    if value is None:
        return None
    return _coerce_finite_float(name, value)


def _normalize_window(
    start: object,
    end: object,
) -> tuple[float | None, float | None]:
    """Validate an optional [start, end] window pair."""
    start_s = _normalize_optional_time("window_start_s", start)
    end_s = _normalize_optional_time("window_end_s", end)
    if start_s is not None and end_s is not None and end_s < start_s:
        raise ValueError(
            f"window_end_s ({end_s!r}) must be >= window_start_s ({start_s!r})"
        )
    return start_s, end_s


@dataclass(frozen=True, slots=True)
class EventFusionPolicyContext:
    """Capture the small V1 gating context used by fused claims."""

    background_media_likely: bool = False
    room_busy_or_overlapping: bool = False
    multi_person_context: bool = False
    cross_modal_conflict: bool = False
    additional_blocked_reasons: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "background_media_likely",
            _coerce_boolish("background_media_likely", self.background_media_likely),
        )
        object.__setattr__(
            self,
            "room_busy_or_overlapping",
            _coerce_boolish("room_busy_or_overlapping", self.room_busy_or_overlapping),
        )
        object.__setattr__(
            self,
            "multi_person_context",
            _coerce_boolish("multi_person_context", self.multi_person_context),
        )
        object.__setattr__(
            self,
            "cross_modal_conflict",
            _coerce_boolish("cross_modal_conflict", self.cross_modal_conflict),
        )
        object.__setattr__(
            self,
            "additional_blocked_reasons",
            _normalize_string_sequence(
                self.additional_blocked_reasons,
                field_name="additional_blocked_reasons",
                max_items=_MAX_BLOCK_REASONS,
                item_kind="code",
            ),
        )

    @classmethod
    def from_observation(
        cls,
        observation: SocialObservation,
        *,
        room_busy_or_overlapping: bool = False,
        cross_modal_conflict: bool = False,
        additional_blocked_reasons: Iterable[str] = (),
    ) -> "EventFusionPolicyContext":
        """Build the V1 gate context from one normalized social observation."""
        multi_person = (
            _flag_is_true(observation.inspected)
            and _flag_is_true(observation.vision.person_visible)
            and int(observation.vision.person_count) > 1
        )
        return cls(
            background_media_likely=_flag_is_true(
                observation.audio.background_media_likely
            ),
            room_busy_or_overlapping=(
                _flag_is_true(room_busy_or_overlapping)
                or _flag_is_true(observation.audio.speech_overlap_likely)
            ),
            multi_person_context=multi_person,
            cross_modal_conflict=_flag_is_true(cross_modal_conflict),
            additional_blocked_reasons=tuple(additional_blocked_reasons),
        )

    @property
    def blocked_reasons(self) -> tuple[str, ...]:
        """Return the active block reasons in stable order."""
        reasons: list[str] = []
        if self.background_media_likely:
            reasons.append("background_media_active")
        if self.room_busy_or_overlapping:
            reasons.append("room_busy_or_overlapping")
        if self.multi_person_context:
            reasons.append("multi_person_context")
        if self.cross_modal_conflict:
            reasons.append("cross_modal_conflict")
        reasons.extend(self.additional_blocked_reasons)
        return tuple(dict.fromkeys(reasons))


@dataclass(frozen=True, slots=True)
class FusedEventClaim:
    """Describe one short-window multimodal event claim."""

    state: str
    active: bool
    confidence: float
    source: str
    source_type: str = "observed"
    requires_confirmation: bool = True
    window_start_s: float | None = None
    window_end_s: float | None = None
    action_level: FusionActionLevel = FusionActionLevel.IGNORE
    delivery_allowed: bool = False
    blocked_by: tuple[str, ...] = field(default_factory=tuple)
    supporting_audio_events: tuple[str, ...] = field(default_factory=tuple)
    supporting_vision_events: tuple[str, ...] = field(default_factory=tuple)
    review_recommended: bool = False
    keyframe_review_plan: KeyframeReviewPlan | None = None
    preferred_action_level: FusionActionLevel | None = None
    policy_version: str = _FUSED_EVENT_POLICY_VERSION

    def __post_init__(self) -> None:
        # BREAKING: malformed claims now fail fast during construction instead of
        # silently propagating invalid confidence, window, or label data.
        requested_action = _coerce_action_level(
            "preferred_action_level",
            self.action_level if self.preferred_action_level is None else self.preferred_action_level,
        )
        resolved_action = _coerce_action_level("action_level", self.action_level)

        active = _coerce_boolish("active", self.active)
        delivery_allowed = _coerce_boolish("delivery_allowed", self.delivery_allowed)
        requires_confirmation = _coerce_boolish(
            "requires_confirmation", self.requires_confirmation
        )
        review_recommended = _coerce_boolish(
            "review_recommended", self.review_recommended
        )

        state = _sanitize_label(self.state, field_name="state")
        source = _sanitize_label(self.source, field_name="source")
        source_type = _sanitize_code(self.source_type, field_name="source_type")
        policy_version = _sanitize_code(
            self.policy_version,
            field_name="policy_version",
            max_len=_MAX_LABEL_LEN,
        )
        confidence = _normalize_confidence(self.confidence)
        window_start_s, window_end_s = _normalize_window(
            self.window_start_s,
            self.window_end_s,
        )
        blocked_by = _normalize_string_sequence(
            self.blocked_by,
            field_name="blocked_by",
            max_items=_MAX_BLOCK_REASONS,
            item_kind="code",
        )
        supporting_audio_events = _normalize_string_sequence(
            self.supporting_audio_events,
            field_name="supporting_audio_events",
            max_items=_MAX_SUPPORTING_EVENTS_PER_MODALITY,
            item_kind="label",
        )
        supporting_vision_events = _normalize_string_sequence(
            self.supporting_vision_events,
            field_name="supporting_vision_events",
            max_items=_MAX_SUPPORTING_EVENTS_PER_MODALITY,
            item_kind="label",
        )

        if blocked_by:
            delivery_allowed = False
        if not delivery_allowed:
            resolved_action = FusionActionLevel.IGNORE

        if requested_action is not FusionActionLevel.DIRECT:
            requires_confirmation = True

        if (
            requested_action is FusionActionLevel.REVIEW_ONLY
            or self.keyframe_review_plan is not None
        ):
            review_recommended = True

        object.__setattr__(self, "state", state)
        object.__setattr__(self, "active", active)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "source_type", source_type)
        object.__setattr__(self, "requires_confirmation", requires_confirmation)
        object.__setattr__(self, "window_start_s", window_start_s)
        object.__setattr__(self, "window_end_s", window_end_s)
        object.__setattr__(self, "action_level", resolved_action)
        object.__setattr__(self, "delivery_allowed", delivery_allowed)
        object.__setattr__(self, "blocked_by", blocked_by)
        object.__setattr__(self, "supporting_audio_events", supporting_audio_events)
        object.__setattr__(self, "supporting_vision_events", supporting_vision_events)
        object.__setattr__(self, "review_recommended", review_recommended)
        object.__setattr__(self, "preferred_action_level", requested_action)
        object.__setattr__(self, "policy_version", policy_version)

    @property
    def supporting_modalities(self) -> tuple[str, ...]:
        """Return the modalities that contributed evidence to this claim."""
        modalities: list[str] = []
        if self.supporting_audio_events:
            modalities.append("audio")
        if self.supporting_vision_events:
            modalities.append("vision")
        return tuple(modalities)

    @property
    def policy_abstained(self) -> bool:
        """Return whether policy suppressed a non-ignore requested action."""
        requested_action = self.preferred_action_level or self.action_level
        return (
            not self.delivery_allowed
            and requested_action is not FusionActionLevel.IGNORE
        )

    def to_payload(self) -> dict[str, object]:
        """Serialize one fused claim into plain automation facts."""
        requested_action = self.preferred_action_level or self.action_level
        # BREAKING: payload now carries audit fields (`preferred_action_level`,
        # `policy_abstained`, `supporting_modalities`, `policy_version`) so
        # downstream consumers can distinguish requested vs. allowed behavior.
        return {
            "state": self.state,
            "active": self.active,
            "confidence": self.confidence,
            "source": self.source,
            "source_type": self.source_type,
            "requires_confirmation": self.requires_confirmation,
            "window_start_s": self.window_start_s,
            "window_end_s": self.window_end_s,
            "action_level": self.action_level.value,
            "preferred_action_level": requested_action.value,
            "delivery_allowed": self.delivery_allowed,
            "policy_abstained": self.policy_abstained,
            "blocked_by": list(self.blocked_by),
            "supporting_audio_events": list(self.supporting_audio_events),
            "supporting_vision_events": list(self.supporting_vision_events),
            "supporting_modalities": list(self.supporting_modalities),
            "review_recommended": self.review_recommended,
            "policy_version": self.policy_version,
            "keyframe_review_plan": (
                None
                if self.keyframe_review_plan is None
                else self.keyframe_review_plan.to_payload()
            ),
        }


def build_fused_claim(
    *,
    state: str,
    confidence: float,
    source: str,
    policy_context: EventFusionPolicyContext,
    window_start_s: float | None,
    window_end_s: float | None,
    preferred_action_level: FusionActionLevel,
    supporting_audio_events: tuple[str, ...] = (),
    supporting_vision_events: tuple[str, ...] = (),
    requires_confirmation: bool = True,
    review_recommended: bool = False,
    keyframe_review_plan: KeyframeReviewPlan | None = None,
) -> FusedEventClaim:
    """Build one fused claim while applying the fail-closed delivery gates."""
    if not isinstance(policy_context, EventFusionPolicyContext):
        raise TypeError(
            "policy_context must be EventFusionPolicyContext, "
            f"got {type(policy_context).__name__}"
        )

    requested_action = _coerce_action_level(
        "preferred_action_level",
        preferred_action_level,
    )
    blocked_by = policy_context.blocked_reasons
    delivery_allowed = not blocked_by

    return FusedEventClaim(
        state=state,
        active=True,
        confidence=confidence,
        source=source,
        source_type="observed_fused",
        requires_confirmation=requires_confirmation,
        window_start_s=window_start_s,
        window_end_s=window_end_s,
        action_level=requested_action if delivery_allowed else FusionActionLevel.IGNORE,
        delivery_allowed=delivery_allowed,
        blocked_by=blocked_by,
        supporting_audio_events=supporting_audio_events,
        supporting_vision_events=supporting_vision_events,
        review_recommended=(
            review_recommended
            or requested_action is FusionActionLevel.REVIEW_ONLY
            or keyframe_review_plan is not None
        ),
        keyframe_review_plan=keyframe_review_plan,
        preferred_action_level=requested_action,
        policy_version=_FUSED_EVENT_POLICY_VERSION,
    )


__all__ = [
    "EventFusionPolicyContext",
    "FusedEventClaim",
    "FusionActionLevel",
    "build_fused_claim",
]