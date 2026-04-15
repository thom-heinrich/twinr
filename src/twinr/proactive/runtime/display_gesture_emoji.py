# CHANGELOG: 2026-03-29
# BUG-1: Replaced wall-clock suppression timing with monotonic timing so RTC/NTP jumps cannot stretch or suppress gesture acknowledgements incorrectly.
# BUG-2: Serialized the publish path and added duplicate-bounce suppression so concurrent or bursty gesture updates no longer race or thrash the HDMI cue store.
# BUG-3: Hardened naive-datetime handling and isolated controller/store failures so auxiliary emoji feedback cannot crash the proactive camera loop.
# BUG-4: Transcript-first voice mode now fail-closes the dedicated HDMI
#        gesture-refresh cadence centrally again so the heavy gesture lane
#        cannot contend with the live voice streaming PID.
# SEC-1: Clamped externally supplied hold times and sanitized source/accent/reason fields to prevent in-process cue-lane lockout and oversized payload abuse.
# IMP-1: Added policy-driven, alias-aware gesture mapping with optional TwinrConfig overrides for fine/coarse gestures, accents, hold times, and classifier-label aliases.
# IMP-2: Activated configurable motion-dominance/custom-only suppression and per-gesture dedupe windows, which better matches 2026 edge-HRI feedback expectations.

"""Mirror recognized camera gestures into bounded HDMI emoji acknowledgements.

This module keeps gesture-to-emoji acknowledgement policy separate from the
main proactive monitor orchestration. It listens only to stabilized rising-edge
camera gesture events and publishes a short-lived right-hand HDMI emoji cue
without touching the face channel.

The dedicated live HDMI path intentionally mirrors only the user-facing
symbols Twinr currently needs to make maximally robust on the Pi, while also
supporting policy/config extension for additional gesture labels:
`thumbs_up`, `thumbs_down`, and `peace_sign`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import logging
import math
import re
import threading
import time
from types import MappingProxyType
from typing import Any, Final, cast

from twinr.agent.base_agent.config import TwinrConfig
from twinr.display.emoji_cues import DisplayEmojiController, DisplayEmojiCueStore, DisplayEmojiSymbol

from ..social.camera_surface import ProactiveCameraSnapshot, ProactiveCameraSurfaceUpdate
from ..social.engine import SocialFineHandGesture, SocialGestureEvent


_LOGGER = logging.getLogger(__name__)

_SOURCE: Final[str] = "proactive_gesture_ack"
_DEFAULT_HOLD_SECONDS: Final[float] = 2.8
_MIN_HOLD_SECONDS: Final[float] = 0.15
_MAX_HOLD_SECONDS: Final[float] = 8.0
_DEFAULT_DEDUPE_WINDOW_S: Final[float] = 0.35
_COARSE_GESTURE_EVENT_NAMES: Final[frozenset[str]] = frozenset(
    {"camera.gesture_detected", "camera.coarse_arm_gesture_detected"}
)
_FINE_GESTURE_EVENT_NAME: Final[str] = "camera.fine_hand_gesture_detected"
_MOTION_COARSE_GESTURES: Final[frozenset[SocialGestureEvent]] = frozenset()
_MOTION_GESTURE_DOMINANCE_WINDOW_S: Final[float] = 1.0
_MIN_REFRESH_INTERVAL_S: Final[float] = 0.1
_CUSTOM_ONLY_FINE_GESTURES: Final[frozenset[SocialFineHandGesture]] = frozenset()

_MAX_SOURCE_LENGTH: Final[int] = 64
_MAX_ACCENT_LENGTH: Final[int] = 32
_MAX_REASON_LENGTH: Final[int] = 128

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")

_FINE_HAND_GESTURE_MAP: Final[dict[SocialFineHandGesture, tuple[DisplayEmojiSymbol, str]]] = {
    SocialFineHandGesture.THUMBS_UP: (DisplayEmojiSymbol.THUMBS_UP, "success"),
    SocialFineHandGesture.THUMBS_DOWN: (DisplayEmojiSymbol.THUMBS_DOWN, "warm"),
    SocialFineHandGesture.PEACE_SIGN: (DisplayEmojiSymbol.VICTORY_HAND, "warm"),
}

_COARSE_GESTURE_MAP: Final[dict[SocialGestureEvent, tuple[DisplayEmojiSymbol, str]]] = {}

_DEFAULT_FINE_GESTURE_ALIASES: Final[dict[str, frozenset[str]]] = {
    "thumbs_up": frozenset({"thumb_up", "thumbsup", "thumbs-up", "like"}),
    "thumbs_down": frozenset({"thumb_down", "thumbsdown", "thumbs-down", "dislike"}),
    "peace_sign": frozenset({"peace", "victory", "victory_sign", "victory_hand", "v_sign", "v-sign"}),
}

_DEFAULT_COARSE_GESTURE_ALIASES: Final[dict[str, frozenset[str]]] = {}


def _normalize_token(raw: object) -> str:
    """Return a compact comparison token for gesture names/aliases."""

    text = str(raw or "").strip().lower()
    if not text:
        return ""
    return _TOKEN_RE.sub("_", text).strip("_")


def _normalize_event_name(raw: object) -> str:
    """Return a normalized camera event name."""

    return str(raw or "").strip().lower()


def _normalize_event_names(event_names: Iterable[str] | None) -> frozenset[str]:
    """Normalize one iterable of event names."""

    normalized = {
        name
        for name in (_normalize_event_name(item) for item in (event_names or ()))
        if name
    }
    return frozenset(normalized)


def _gesture_candidate_tokens(gesture: object) -> tuple[str, ...]:
    """Return stable lookup tokens for one gesture enum/object."""

    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in (
        getattr(gesture, "value", None),
        getattr(gesture, "name", None),
        gesture,
    ):
        token = _normalize_token(candidate)
        if token and token not in seen:
            seen.add(token)
            ordered.append(token)
    return tuple(ordered)


def _gesture_reason_token(gesture: object, *, default: str = "unknown") -> str:
    """Return the primary stable token for one gesture."""

    return next(iter(_gesture_candidate_tokens(gesture)), default)


def _gesture_token_set(gestures: Iterable[object]) -> frozenset[str]:
    """Return all known lookup tokens for a gesture set."""

    tokens = {
        token
        for gesture in gestures
        for token in _gesture_candidate_tokens(gesture)
        if token
    }
    return frozenset(tokens)


def _sanitize_text(raw: object, *, default: str, max_length: int) -> str:
    """Return compact single-line text suitable for store/controller metadata."""

    text = " ".join(str(raw or "").split()).strip()
    if not text:
        return default
    return text[:max_length]


def _sanitize_source(raw: object) -> str:
    """Return a safe cue-source label."""

    return _sanitize_text(raw, default=_SOURCE, max_length=_MAX_SOURCE_LENGTH)


def _sanitize_accent(raw: object) -> str:
    """Return a safe accent label."""

    return _sanitize_text(raw, default="neutral", max_length=_MAX_ACCENT_LENGTH)


def _sanitize_reason(raw: object) -> str:
    """Return a safe decision reason."""

    return _sanitize_text(raw, default="inactive", max_length=_MAX_REASON_LENGTH)


def _coerce_utc_now(now: datetime | None) -> datetime:
    """Return an aware UTC timestamp.

    Naive datetimes are treated as UTC instead of local time so tests and
    callers cannot accidentally shift expiry windows by the host timezone.
    """

    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _bounded_float(
    raw: object,
    *,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Parse one bounded finite float with a fallback default."""

    try:
        value = float(cast(Any, raw))
    except (TypeError, ValueError):
        value = default
    if not math.isfinite(value):
        value = default
    if minimum is not None and value < minimum:
        value = minimum
    if maximum is not None and value > maximum:
        value = maximum
    return value


def _coerce_mapping(raw: object) -> Mapping[str, Any]:
    """Return a mapping payload or an empty mapping."""

    return raw if isinstance(raw, Mapping) else {}


def _coerce_token_set(raw: object) -> frozenset[str]:
    """Return one normalized token set from a config payload."""

    if raw is None:
        return frozenset()
    if isinstance(raw, str):
        items: Iterable[object] = raw.split(",")
    elif isinstance(raw, Iterable):
        items = raw
    else:
        items = (raw,)
    tokens = {_normalize_token(item) for item in items}
    return frozenset(token for token in tokens if token)


def _resolve_display_symbol(raw: object, *, default: DisplayEmojiSymbol) -> DisplayEmojiSymbol:
    """Resolve a display symbol from enum member, enum name, or enum value."""

    if isinstance(raw, DisplayEmojiSymbol):
        return raw
    token = _normalize_token(raw)
    if not token:
        return default
    for member in DisplayEmojiSymbol:
        if _normalize_token(member.name) == token or _normalize_token(getattr(member, "value", None)) == token:
            return member
    return default


@dataclass(frozen=True, slots=True)
class DisplayGestureEmojiDecision:
    """Describe one optional HDMI emoji acknowledgement."""

    active: bool = False
    reason: str = "inactive"
    source: str = _SOURCE
    symbol: DisplayEmojiSymbol = DisplayEmojiSymbol.SPARKLES
    accent: str = "neutral"
    hold_seconds: float = _DEFAULT_HOLD_SECONDS


@dataclass(frozen=True, slots=True)
class DisplayGestureEmojiPublishResult:
    """Summarize one bounded publish attempt."""

    action: str
    decision: DisplayGestureEmojiDecision
    owner: str | None = None


@dataclass(frozen=True, slots=True)
class _DisplayGestureEmojiRule:
    """Internal gesture-to-emoji rule."""

    canonical_key: str
    symbol: DisplayEmojiSymbol
    accent: str = "neutral"
    hold_seconds: float = _DEFAULT_HOLD_SECONDS


@dataclass(frozen=True, slots=True)
class DisplayGestureEmojiPolicy:
    """Runtime policy for mapping recognized gestures to HDMI emoji cues."""

    fine_rules: Mapping[str, _DisplayGestureEmojiRule]
    coarse_rules: Mapping[str, _DisplayGestureEmojiRule]
    motion_coarse_gesture_tokens: frozenset[str]
    custom_only_fine_gesture_tokens: frozenset[str]
    motion_dominance_window_s: float = _MOTION_GESTURE_DOMINANCE_WINDOW_S
    dedupe_window_s: float = _DEFAULT_DEDUPE_WINDOW_S
    min_hold_seconds: float = _MIN_HOLD_SECONDS
    max_hold_seconds: float = _MAX_HOLD_SECONDS


def _build_rule_specs_from_maps(
    *,
    default_map: Mapping[object, tuple[DisplayEmojiSymbol, str]],
    default_aliases: Mapping[str, frozenset[str]],
    default_hold_seconds: float,
) -> dict[str, tuple[_DisplayGestureEmojiRule, frozenset[str]]]:
    """Create canonical gesture rules from enum-keyed defaults."""

    specs: dict[str, tuple[_DisplayGestureEmojiRule, frozenset[str]]] = {}
    for gesture, (symbol, accent) in default_map.items():
        canonical_key = _gesture_reason_token(gesture)
        specs[canonical_key] = (
            _DisplayGestureEmojiRule(
                canonical_key=canonical_key,
                symbol=symbol,
                accent=_sanitize_accent(accent),
                hold_seconds=default_hold_seconds,
            ),
            default_aliases.get(canonical_key, frozenset()),
        )
    return specs


def _resolve_rule_override(
    canonical_key: str,
    payload: object,
    *,
    default_rule: _DisplayGestureEmojiRule | None,
    default_aliases: frozenset[str],
    default_hold_seconds: float,
) -> tuple[_DisplayGestureEmojiRule, frozenset[str]] | None:
    """Resolve one optional config override for a canonical gesture rule."""

    if payload is False:
        return None

    if isinstance(payload, Mapping):
        enabled = payload.get("enabled", True)
        if enabled is False:
            return None
        fallback_symbol = default_rule.symbol if default_rule is not None else DisplayEmojiSymbol.SPARKLES
        fallback_accent = default_rule.accent if default_rule is not None else "neutral"
        fallback_hold = default_rule.hold_seconds if default_rule is not None else default_hold_seconds
        rule = _DisplayGestureEmojiRule(
            canonical_key=canonical_key,
            symbol=_resolve_display_symbol(payload.get("symbol"), default=fallback_symbol),
            accent=_sanitize_accent(payload.get("accent", fallback_accent)),
            hold_seconds=_bounded_float(
                payload.get("hold_seconds", fallback_hold),
                default=fallback_hold,
                minimum=_MIN_HOLD_SECONDS,
                maximum=_MAX_HOLD_SECONDS,
            ),
        )
        aliases = _coerce_token_set(payload.get("aliases"))
        return rule, aliases or default_aliases

    fallback_symbol = default_rule.symbol if default_rule is not None else DisplayEmojiSymbol.SPARKLES
    fallback_accent = default_rule.accent if default_rule is not None else "neutral"
    fallback_hold = default_rule.hold_seconds if default_rule is not None else default_hold_seconds
    return (
        _DisplayGestureEmojiRule(
            canonical_key=canonical_key,
            symbol=_resolve_display_symbol(payload, default=fallback_symbol),
            accent=fallback_accent,
            hold_seconds=fallback_hold,
        ),
        default_aliases,
    )


def _build_rule_index(
    *,
    default_map: Mapping[object, tuple[DisplayEmojiSymbol, str]],
    default_aliases: Mapping[str, frozenset[str]],
    default_hold_seconds: float,
    overrides: Mapping[str, Any] | None = None,
) -> Mapping[str, _DisplayGestureEmojiRule]:
    """Build an alias-aware immutable lookup index for gesture rules."""

    specs = _build_rule_specs_from_maps(
        default_map=default_map,
        default_aliases=default_aliases,
        default_hold_seconds=default_hold_seconds,
    )
    for raw_key, payload in (overrides or {}).items():
        canonical_key = _normalize_token(raw_key)
        if not canonical_key:
            continue
        default_rule, default_rule_aliases = specs.get(canonical_key, (None, frozenset()))
        resolved = _resolve_rule_override(
            canonical_key,
            payload,
            default_rule=default_rule,
            default_aliases=default_rule_aliases,
            default_hold_seconds=default_hold_seconds,
        )
        if resolved is None:
            specs.pop(canonical_key, None)
            continue
        specs[canonical_key] = resolved

    index: dict[str, _DisplayGestureEmojiRule] = {}
    for canonical_key, (rule, aliases) in specs.items():
        for token in {canonical_key, *aliases}:
            normalized = _normalize_token(token)
            if normalized:
                index[normalized] = rule
    return MappingProxyType(index)


def resolve_display_gesture_policy(config: TwinrConfig | None = None) -> DisplayGestureEmojiPolicy:
    """Resolve the runtime gesture-ack policy from config and defaults."""

    global_hold_seconds = _bounded_float(
        getattr(config, "display_gesture_hold_seconds", _DEFAULT_HOLD_SECONDS),
        default=_DEFAULT_HOLD_SECONDS,
        minimum=_MIN_HOLD_SECONDS,
        maximum=_MAX_HOLD_SECONDS,
    )
    dedupe_window_s = _bounded_float(
        getattr(config, "display_gesture_dedupe_window_s", _DEFAULT_DEDUPE_WINDOW_S),
        default=_DEFAULT_DEDUPE_WINDOW_S,
        minimum=0.0,
        maximum=2.0,
    )
    motion_dominance_window_s = _bounded_float(
        getattr(config, "display_gesture_motion_dominance_window_s", _MOTION_GESTURE_DOMINANCE_WINDOW_S),
        default=_MOTION_GESTURE_DOMINANCE_WINDOW_S,
        minimum=0.0,
        maximum=3.0,
    )
    fine_rules = _build_rule_index(
        default_map=cast(Mapping[object, tuple[DisplayEmojiSymbol, str]], _FINE_HAND_GESTURE_MAP),
        default_aliases=_DEFAULT_FINE_GESTURE_ALIASES,
        default_hold_seconds=global_hold_seconds,
        overrides=_coerce_mapping(getattr(config, "display_gesture_fine_map", None)),
    )
    coarse_rules = _build_rule_index(
        default_map=cast(Mapping[object, tuple[DisplayEmojiSymbol, str]], _COARSE_GESTURE_MAP),
        default_aliases=_DEFAULT_COARSE_GESTURE_ALIASES,
        default_hold_seconds=global_hold_seconds,
        overrides=_coerce_mapping(getattr(config, "display_gesture_coarse_map", None)),
    )
    motion_coarse_gesture_tokens = _gesture_token_set(_MOTION_COARSE_GESTURES) | _coerce_token_set(
        getattr(config, "display_gesture_motion_coarse_gestures", None)
    )
    custom_only_fine_gesture_tokens = _gesture_token_set(_CUSTOM_ONLY_FINE_GESTURES) | _coerce_token_set(
        getattr(config, "display_gesture_custom_only_fine_gestures", None)
    )
    return DisplayGestureEmojiPolicy(
        fine_rules=fine_rules,
        coarse_rules=coarse_rules,
        motion_coarse_gesture_tokens=motion_coarse_gesture_tokens,
        custom_only_fine_gesture_tokens=custom_only_fine_gesture_tokens,
        motion_dominance_window_s=motion_dominance_window_s,
        dedupe_window_s=dedupe_window_s,
        min_hold_seconds=_MIN_HOLD_SECONDS,
        max_hold_seconds=_MAX_HOLD_SECONDS,
    )


_DEFAULT_POLICY: Final[DisplayGestureEmojiPolicy] = resolve_display_gesture_policy()


def _match_rule(
    gesture: object,
    *,
    rules: Mapping[str, _DisplayGestureEmojiRule],
) -> _DisplayGestureEmojiRule | None:
    """Resolve one gesture against an alias-aware rule index."""

    for token in _gesture_candidate_tokens(gesture):
        rule = rules.get(token)
        if rule is not None:
            return rule
    return None


def _gesture_matches_tokens(gesture: object, tokens: frozenset[str]) -> bool:
    """Return whether a gesture matches any token in one normalized token set."""

    if not tokens:
        return False
    return any(token in tokens for token in _gesture_candidate_tokens(gesture))


def _resolved_policy(policy: DisplayGestureEmojiPolicy | None) -> DisplayGestureEmojiPolicy:
    """Return the active policy object."""

    return _DEFAULT_POLICY if policy is None else policy


def resolve_display_gesture_refresh_interval(config: TwinrConfig) -> float | None:
    """Return the bounded HDMI gesture-refresh cadence.

    Transcript-first voice mode must fail closed here so every downstream
    support/scheduler gate sees the dedicated gesture lane as unavailable.
    """

    if bool(getattr(config, "voice_orchestrator_enabled", False)):
        return None

    raw_interval = getattr(config, "display_gesture_refresh_interval_s", 0.2)
    try:
        interval_s = float(raw_interval or 0.0)
    except (TypeError, ValueError):
        return 0.2
    if not math.isfinite(interval_s) or interval_s <= 0.0:
        return None
    return max(_MIN_REFRESH_INTERVAL_S, interval_s)


def display_gesture_refresh_supported(
    *,
    config: TwinrConfig,
    vision_observer: object | None,
) -> bool:
    """Return whether a dedicated HDMI gesture-refresh path is safe to run."""

    if resolve_display_gesture_refresh_interval(config) is None:
        return False

    display_driver_token = _normalize_token(getattr(config, "display_driver", ""))
    driver_parts = set(display_driver_token.split("_"))
    if "hdmi" not in driver_parts and not display_driver_token.startswith("hdmi"):
        return False

    supports_gesture_refresh = getattr(vision_observer, "supports_gesture_refresh", None)
    if supports_gesture_refresh is True:
        return True
    if supports_gesture_refresh is False:
        return False
    return callable(getattr(vision_observer, "observe_gesture", None))


def decision_for_fine_hand_gesture(
    gesture: SocialFineHandGesture,
    *,
    policy: DisplayGestureEmojiPolicy | None = None,
) -> DisplayGestureEmojiDecision:
    """Return the user-facing emoji decision for one fine-hand gesture."""

    resolved_policy = _resolved_policy(policy)
    rule = _match_rule(gesture, rules=resolved_policy.fine_rules)
    if rule is None:
        return DisplayGestureEmojiDecision(reason="unsupported_fine_hand_gesture")
    return DisplayGestureEmojiDecision(
        active=True,
        reason=f"fine_hand_gesture:{rule.canonical_key}",
        symbol=rule.symbol,
        accent=rule.accent,
        hold_seconds=rule.hold_seconds,
    )


def decision_for_coarse_gesture(
    gesture: SocialGestureEvent,
    *,
    motion_priority: bool = False,
    policy: DisplayGestureEmojiPolicy | None = None,
) -> DisplayGestureEmojiDecision:
    """Return the user-facing emoji decision for one coarse-arm gesture."""

    resolved_policy = _resolved_policy(policy)
    rule = _match_rule(gesture, rules=resolved_policy.coarse_rules)
    if rule is None:
        return DisplayGestureEmojiDecision(reason="unsupported_coarse_gesture")
    reason_prefix = "motion_coarse_gesture" if motion_priority else "coarse_gesture"
    return DisplayGestureEmojiDecision(
        active=True,
        reason=f"{reason_prefix}:{rule.canonical_key}",
        symbol=rule.symbol,
        accent=rule.accent,
        hold_seconds=rule.hold_seconds,
    )


def derive_display_gesture_emoji(
    *,
    snapshot: ProactiveCameraSnapshot,
    event_names: Iterable[str],
    policy: DisplayGestureEmojiPolicy | None = None,
) -> DisplayGestureEmojiDecision:
    """Translate one stabilized camera gesture update into one emoji ack."""

    resolved_policy = _resolved_policy(policy)
    names = _normalize_event_names(event_names)

    if (
        _gesture_matches_tokens(snapshot.gesture_event, resolved_policy.motion_coarse_gesture_tokens)
        and not snapshot.gesture_event_unknown
        and (
            _COARSE_GESTURE_EVENT_NAMES.intersection(names)
            or (
                _FINE_GESTURE_EVENT_NAME in names
                and snapshot.fine_hand_gesture == SocialFineHandGesture.OPEN_PALM
                and not snapshot.fine_hand_gesture_unknown
            )
        )
    ):
        decision = decision_for_coarse_gesture(
            snapshot.gesture_event,
            motion_priority=True,
            policy=resolved_policy,
        )
        if decision.active:
            return decision

    has_fine_event = _FINE_GESTURE_EVENT_NAME in names
    has_coarse_event = bool(_COARSE_GESTURE_EVENT_NAMES.intersection(names))

    if has_fine_event:
        if not snapshot.fine_hand_gesture_unknown:
            decision = decision_for_fine_hand_gesture(
                snapshot.fine_hand_gesture,
                policy=resolved_policy,
            )
            if decision.active:
                return decision
        if has_coarse_event and not snapshot.gesture_event_unknown:
            coarse_decision = decision_for_coarse_gesture(
                snapshot.gesture_event,
                policy=resolved_policy,
            )
            if coarse_decision.active:
                return coarse_decision
        return DisplayGestureEmojiDecision(reason="unsupported_fine_hand_gesture")

    if has_coarse_event:
        if not snapshot.gesture_event_unknown:
            decision = decision_for_coarse_gesture(
                snapshot.gesture_event,
                policy=resolved_policy,
            )
            if decision.active:
                return decision
        return DisplayGestureEmojiDecision(reason="unsupported_coarse_gesture")

    return DisplayGestureEmojiDecision(reason="no_gesture_event")


@dataclass(slots=True)
class DisplayGestureEmojiPublisher:
    """Persist short-lived HDMI emoji acknowledgements for user gestures."""

    controller: DisplayEmojiController
    source: str = _SOURCE
    policy: DisplayGestureEmojiPolicy = field(default_factory=lambda: _DEFAULT_POLICY)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _recent_motion_priority_until_monotonic_s: float | None = field(default=None, init=False, repr=False)
    _last_published_signature: tuple[DisplayEmojiSymbol, str] | None = field(default=None, init=False, repr=False)
    _last_published_monotonic_s: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Normalize mutable runtime fields."""

        self.source = _sanitize_source(self.source)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayGestureEmojiPublisher":
        """Build one publisher from the configured emoji cue store."""

        return cls(
            controller=DisplayEmojiController.from_config(
                cast(Any, config),
                default_source=_SOURCE,
            ),
            source=_SOURCE,
            policy=resolve_display_gesture_policy(config),
        )

    @property
    def store(self) -> DisplayEmojiCueStore:
        """Expose the underlying emoji cue store for tests."""

        return self.controller.store

    def publish_update(
        self,
        update: ProactiveCameraSurfaceUpdate,
        *,
        now: datetime | None = None,
    ) -> DisplayGestureEmojiPublishResult:
        """Derive and publish one emoji acknowledgement from a camera update."""

        effective_now = _coerce_utc_now(now)
        monotonic_now_s = time.monotonic()

        with self._lock:
            decision = derive_display_gesture_emoji(
                snapshot=update.snapshot,
                event_names=update.event_names,
                policy=self.policy,
            )
            decision = self._apply_recent_motion_priority(
                update=update,
                decision=decision,
                now_monotonic_s=monotonic_now_s,
            )
            return self._publish_locked(
                decision,
                now=effective_now,
                monotonic_now_s=monotonic_now_s,
            )

    def publish(
        self,
        decision: DisplayGestureEmojiDecision,
        *,
        now: datetime | None = None,
    ) -> DisplayGestureEmojiPublishResult:
        """Persist one gesture acknowledgement unless another producer owns the surface."""

        effective_now = _coerce_utc_now(now)
        monotonic_now_s = time.monotonic()

        with self._lock:
            return self._publish_locked(
                decision,
                now=effective_now,
                monotonic_now_s=monotonic_now_s,
            )

    def _publish_locked(
        self,
        decision: DisplayGestureEmojiDecision,
        *,
        now: datetime,
        monotonic_now_s: float,
    ) -> DisplayGestureEmojiPublishResult:
        """Publish one decision while holding the internal publisher lock."""

        prepared_decision = self._prepare_decision_for_publish(decision)
        prepared_decision = self._apply_duplicate_suppression(
            prepared_decision,
            now_monotonic_s=monotonic_now_s,
        )

        if not prepared_decision.active:
            return DisplayGestureEmojiPublishResult(
                action="inactive",
                decision=prepared_decision,
            )

        try:
            active_cue = self.store.load_active(now=now)
        except Exception:
            _LOGGER.exception(
                "Failed to load active HDMI emoji cue",
                extra={
                    "gesture_ack_source": self.source,
                    "gesture_ack_reason": prepared_decision.reason,
                },
            )
            return DisplayGestureEmojiPublishResult(
                action="inactive",
                decision=replace(prepared_decision, active=False, reason="publish_store_error"),
            )

        active_owner = None if active_cue is None else str(active_cue.source or "").strip() or None
        if active_owner is not None and active_owner != self.source:
            return DisplayGestureEmojiPublishResult(
                action="blocked_foreign_cue",
                decision=prepared_decision,
                owner=active_owner,
            )

        try:
            self.controller.show_symbol(
                prepared_decision.symbol,
                accent=prepared_decision.accent,
                source=self.source,
                hold_seconds=prepared_decision.hold_seconds,
                now=now,
            )
        except Exception:
            _LOGGER.exception(
                "Failed to publish HDMI emoji cue",
                extra={
                    "gesture_ack_source": self.source,
                    "gesture_ack_reason": prepared_decision.reason,
                    "gesture_ack_symbol": getattr(prepared_decision.symbol, "name", str(prepared_decision.symbol)),
                },
            )
            return DisplayGestureEmojiPublishResult(
                action="inactive",
                decision=replace(prepared_decision, active=False, reason="publish_show_error"),
                owner=active_owner,
            )

        self._remember_successful_publish(
            prepared_decision,
            monotonic_now_s=monotonic_now_s,
        )
        return DisplayGestureEmojiPublishResult(
            action="updated" if active_owner == self.source else "created",
            decision=prepared_decision,
            owner=active_owner,
        )

    def _prepare_decision_for_publish(
        self,
        decision: DisplayGestureEmojiDecision,
    ) -> DisplayGestureEmojiDecision:
        """Clamp and sanitize one externally supplied decision."""

        return replace(
            decision,
            source=self.source,
            reason=_sanitize_reason(decision.reason),
            accent=_sanitize_accent(decision.accent),
            hold_seconds=_bounded_float(
                decision.hold_seconds,
                default=_DEFAULT_HOLD_SECONDS,
                minimum=self.policy.min_hold_seconds,
                maximum=self.policy.max_hold_seconds,
            ),
        )

    def _decision_signature(
        self,
        decision: DisplayGestureEmojiDecision,
    ) -> tuple[DisplayEmojiSymbol, str]:
        """Return the duplicate-suppression signature for one decision."""

        return decision.symbol, decision.accent

    def _apply_duplicate_suppression(
        self,
        decision: DisplayGestureEmojiDecision,
        *,
        now_monotonic_s: float,
    ) -> DisplayGestureEmojiDecision:
        """Suppress short duplicate bursts from jittery/stuttered gesture updates."""

        if not decision.active:
            return decision

        dedupe_window_s = self.policy.dedupe_window_s
        if dedupe_window_s <= 0.0:
            return decision

        last_signature = self._last_published_signature
        last_monotonic_s = self._last_published_monotonic_s
        if (
            last_signature == self._decision_signature(decision)
            and last_monotonic_s is not None
            and (now_monotonic_s - last_monotonic_s) < dedupe_window_s
        ):
            return replace(decision, active=False, reason="suppressed_duplicate_gesture")
        return decision

    def _remember_successful_publish(
        self,
        decision: DisplayGestureEmojiDecision,
        *,
        monotonic_now_s: float,
    ) -> None:
        """Remember one successful publish for duplicate suppression."""

        self._last_published_signature = self._decision_signature(decision)
        self._last_published_monotonic_s = monotonic_now_s

    def _apply_recent_motion_priority(
        self,
        *,
        update: ProactiveCameraSurfaceUpdate,
        decision: DisplayGestureEmojiDecision,
        now_monotonic_s: float,
    ) -> DisplayGestureEmojiDecision:
        """Prefer recent motion gestures briefly over conflicting custom-only fine-hand blips."""

        if (
            decision.active
            and decision.reason.startswith("motion_coarse_gesture:")
            and _gesture_matches_tokens(
                update.snapshot.gesture_event,
                self.policy.motion_coarse_gesture_tokens,
            )
        ):
            self._recent_motion_priority_until_monotonic_s = now_monotonic_s + min(
                decision.hold_seconds,
                self.policy.motion_dominance_window_s,
            )
            return decision

        if not decision.active or not decision.reason.startswith("fine_hand_gesture:"):
            return decision

        if not _gesture_matches_tokens(
            update.snapshot.fine_hand_gesture,
            self.policy.custom_only_fine_gesture_tokens,
        ):
            return decision

        recent_motion_until = self._recent_motion_priority_until_monotonic_s
        if recent_motion_until is None or now_monotonic_s >= recent_motion_until:
            return decision

        return replace(decision, active=False, reason="suppressed_by_recent_motion_gesture")


__all__ = [
    "DisplayGestureEmojiDecision",
    "DisplayGestureEmojiPolicy",
    "DisplayGestureEmojiPublishResult",
    "DisplayGestureEmojiPublisher",
    "decision_for_coarse_gesture",
    "decision_for_fine_hand_gesture",
    "display_gesture_refresh_supported",
    "derive_display_gesture_emoji",
    "resolve_display_gesture_policy",
    "resolve_display_gesture_refresh_interval",
]
