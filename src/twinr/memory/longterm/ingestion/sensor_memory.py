"""Compile sensor-derived routines and deviations for long-term memory.

This module consumes previously consolidated multimodal pattern objects and
derives bounded routine summaries plus absence deviations for presence and
interaction behavior around the device.
"""

from __future__ import annotations

import logging  # AUDIT-FIX(#5): Add logging so malformed records and unexpected failures degrade gracefully instead of failing silently.
from dataclasses import dataclass
from datetime import date, datetime, timedelta, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError  # AUDIT-FIX(#3): Handle invalid or missing IANA timezone data without crashing the compiler.

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.longterm.core.models import LongTermMemoryObjectV1, LongTermReflectionResultV1, LongTermSourceRefV1

_LOGGER = logging.getLogger(__name__)  # AUDIT-FIX(#5): Keep the compiler best-effort and observable in production.
_DAYPARTS = ("morning", "afternoon", "evening", "night")
_WEEKDAY_CLASSES = ("all_days", "weekday", "weekend")
_WEEKDAY_LABELS = {  # AUDIT-FIX(#8): Use explicit human-readable labels to avoid malformed prose such as "all dayss".
    "all_days": "all days",
    "weekday": "weekdays",
    "weekend": "weekends",
}
_MAX_EVENT_IDS = 32
_DEFAULT_TIMEZONE_NAME = "Europe/Berlin"
_DEFAULT_BASELINE_DAYS = 21
_DEFAULT_MIN_DAYS_OBSERVED = 6
_DEFAULT_MIN_ROUTINE_RATIO = 0.55
_DEFAULT_DEVIATION_MIN_DELTA = 0.45
_DEFAULT_DEVIATION_MIN_DAYPART_PROGRESS = 0.75  # AUDIT-FIX(#2): Wait until most of the current daypart has elapsed before inferring missing presence.


def _normalize_text(value: object | None) -> str:
    """Collapse an arbitrary value into single-line text."""
    return " ".join(str(value or "").split()).strip()


def _coerce_bool(value: object, *, default: bool) -> bool:
    """Parse common boolean-like config and attribute values."""
    if isinstance(value, bool):
        return value
    normalized = _normalize_text(value).lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default if value is None else bool(value)


def _coerce_positive_int(value: object, *, default: int, minimum: int = 1) -> int:
    """Coerce a positive integer with a lower bound."""
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        return default


def _coerce_ratio(value: object, *, default: float) -> float:
    """Coerce a bounded ratio in the inclusive range ``[0, 1]``."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


def _coerce_attributes(value: object) -> dict[str, object]:
    """Coerce arbitrary attribute payloads into a plain mapping."""
    if value is None:
        return {}
    try:
        return dict(value)
    except (TypeError, ValueError):
        return {}


def _daypart_for_datetime(value: datetime) -> str:
    """Map a localized timestamp into the shared daypart buckets."""
    hour = value.hour
    if 5 <= hour < 11:
        return "morning"
    if 11 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "night"


def _routine_day_for_datetime(value: datetime, *, daypart: str) -> date:
    """Map a timestamp into the routine day used for daypart analytics."""
    routine_day = value.date()
    if daypart == "night" and value.hour < 5:
        return routine_day - timedelta(days=1)  # AUDIT-FIX(#1): Treat post-midnight hours as part of the prior night so overnight routines remain consistent.
    return routine_day


def _daypart_bounds(value: datetime, *, daypart: str) -> tuple[datetime, datetime]:
    """Return the localized start and end bounds for one daypart."""
    if daypart == "morning":
        start = value.replace(hour=5, minute=0, second=0, microsecond=0)
        end = value.replace(hour=11, minute=0, second=0, microsecond=0)
        return start, end
    if daypart == "afternoon":
        start = value.replace(hour=11, minute=0, second=0, microsecond=0)
        end = value.replace(hour=17, minute=0, second=0, microsecond=0)
        return start, end
    if daypart == "evening":
        start = value.replace(hour=17, minute=0, second=0, microsecond=0)
        end = value.replace(hour=22, minute=0, second=0, microsecond=0)
        return start, end
    if value.hour >= 22:
        start = value.replace(hour=22, minute=0, second=0, microsecond=0)
        end = (start + timedelta(days=1)).replace(hour=5, minute=0, second=0, microsecond=0)
        return start, end
    end = value.replace(hour=5, minute=0, second=0, microsecond=0)
    start = (end - timedelta(days=1)).replace(hour=22, minute=0, second=0, microsecond=0)
    return start, end


def _daypart_progress(value: datetime, *, daypart: str) -> float:
    """Return how far the given timestamp is through its daypart."""
    start, end = _daypart_bounds(value, daypart=daypart)
    total_seconds = max(1.0, (end - start).total_seconds())
    elapsed_seconds = (value - start).total_seconds()
    if elapsed_seconds <= 0.0:
        return 0.0
    if elapsed_seconds >= total_seconds:
        return 1.0
    return elapsed_seconds / total_seconds


def _weekday_class(value: date) -> str:
    """Classify a date as ``weekday`` or ``weekend``."""
    return "weekend" if value.weekday() >= 5 else "weekday"


def _weekday_label(value: str) -> str:
    """Return the human-readable label for a weekday class."""
    return _WEEKDAY_LABELS.get(value, value.replace("_", " "))


def _eligible_dates(*, reference_day: date, baseline_days: int, weekday_class: str) -> tuple[date, ...]:
    """Return baseline dates eligible for one weekday class."""
    days: list[date] = []
    for offset in range(1, baseline_days + 1):
        candidate = reference_day - timedelta(days=offset)
        if weekday_class != "all_days" and _weekday_class(candidate) != weekday_class:
            continue
        days.append(candidate)
    return tuple(days)


def _parse_event_datetime(event_id: str) -> datetime | None:
    """Extract the timestamp component from a stored event ID."""
    clean = _normalize_text(event_id)
    if not clean:
        return None
    parts = clean.split(":", 2)
    if len(parts) < 2:
        return None
    timestamp = parts[1]
    for pattern in ("%Y%m%dT%H%M%S%f%z", "%Y%m%dT%H%M%S%z"):
        try:
            return datetime.strptime(timestamp, pattern)
        except ValueError:
            continue
    return None


@dataclass(frozen=True, slots=True)
class _RawPatternSignal:
    """Normalized signal used by routine-compilation helpers."""

    routine_type: str
    interaction_type: str | None
    daypart: str
    event_days: frozenset[date]
    event_id_days: tuple[tuple[str, date], ...]
    event_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LongTermSensorMemoryCompiler:
    """Derive routine and deviation summaries from multimodal patterns.

    Attributes:
        timezone_name: IANA timezone used for daypart and date bucketing.
        enabled: Whether sensor-memory compilation should emit summaries.
        baseline_days: Lookback window used to estimate typical behavior.
        min_days_observed: Minimum eligible baseline days required per bucket.
        min_routine_ratio: Minimum support ratio required to emit a routine.
        deviation_min_delta: Minimum expected presence ratio for deviations.
        deviation_min_daypart_progress: Minimum elapsed portion of the current
            daypart before emitting a missing-presence deviation.
    """

    timezone_name: str = _DEFAULT_TIMEZONE_NAME
    enabled: bool = False
    baseline_days: int = _DEFAULT_BASELINE_DAYS
    min_days_observed: int = _DEFAULT_MIN_DAYS_OBSERVED
    min_routine_ratio: float = _DEFAULT_MIN_ROUTINE_RATIO
    deviation_min_delta: float = _DEFAULT_DEVIATION_MIN_DELTA
    deviation_min_daypart_progress: float = _DEFAULT_DEVIATION_MIN_DAYPART_PROGRESS

    def __post_init__(self) -> None:
        """Normalize configuration values into safe runtime defaults."""
        object.__setattr__(
            self,
            "timezone_name",
            _normalize_text(self.timezone_name) or _DEFAULT_TIMEZONE_NAME,
        )  # AUDIT-FIX(#3): Normalize invalid/blank timezone names early and keep a safe default.
        object.__setattr__(
            self,
            "enabled",
            _coerce_bool(self.enabled, default=False),
        )  # AUDIT-FIX(#6): Normalize truthy and falsy config values instead of trusting raw environment coercion.
        object.__setattr__(
            self,
            "baseline_days",
            _coerce_positive_int(self.baseline_days, default=_DEFAULT_BASELINE_DAYS),
        )  # AUDIT-FIX(#6): Clamp numeric config values to safe ranges so bad .env values do not poison the compiler.
        object.__setattr__(
            self,
            "min_days_observed",
            _coerce_positive_int(self.min_days_observed, default=_DEFAULT_MIN_DAYS_OBSERVED),
        )  # AUDIT-FIX(#6): Clamp numeric config values to safe ranges so bad .env values do not poison the compiler.
        object.__setattr__(
            self,
            "min_routine_ratio",
            _coerce_ratio(self.min_routine_ratio, default=_DEFAULT_MIN_ROUTINE_RATIO),
        )  # AUDIT-FIX(#6): Ratios must remain inside [0, 1] to avoid nonsense confidence and threshold math.
        object.__setattr__(
            self,
            "deviation_min_delta",
            _coerce_ratio(self.deviation_min_delta, default=_DEFAULT_DEVIATION_MIN_DELTA),
        )  # AUDIT-FIX(#6): Ratios must remain inside [0, 1] to avoid nonsense confidence and threshold math.
        object.__setattr__(
            self,
            "deviation_min_daypart_progress",
            _coerce_ratio(self.deviation_min_daypart_progress, default=_DEFAULT_DEVIATION_MIN_DAYPART_PROGRESS),
        )  # AUDIT-FIX(#2): Keep the early-alert suppression threshold safe even with bad config values.

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "LongTermSensorMemoryCompiler":
        """Build a compiler from Twinr configuration."""
        return cls(
            timezone_name=getattr(config, "local_timezone_name", _DEFAULT_TIMEZONE_NAME),
            enabled=getattr(config, "long_term_memory_sensor_memory_enabled", False),
            baseline_days=getattr(config, "long_term_memory_sensor_baseline_days", _DEFAULT_BASELINE_DAYS),
            min_days_observed=getattr(config, "long_term_memory_sensor_min_days_observed", _DEFAULT_MIN_DAYS_OBSERVED),
            min_routine_ratio=getattr(config, "long_term_memory_sensor_min_routine_ratio", _DEFAULT_MIN_ROUTINE_RATIO),
            deviation_min_delta=getattr(config, "long_term_memory_sensor_deviation_min_delta", _DEFAULT_DEVIATION_MIN_DELTA),
            deviation_min_daypart_progress=getattr(
                config,
                "long_term_memory_sensor_deviation_min_daypart_progress",
                _DEFAULT_DEVIATION_MIN_DAYPART_PROGRESS,
            ),  # AUDIT-FIX(#2): New config remains backward-compatible by defaulting when absent.
        )

    def compile(
        self,
        *,
        objects: tuple[LongTermMemoryObjectV1, ...],
        now: datetime | None = None,
    ) -> LongTermReflectionResultV1:
        """Compile sensor-derived routines and deviations from raw patterns.

        Args:
            objects: Consolidated long-term memory objects to inspect.
            now: Optional reference time for daypart and baseline calculations.

        Returns:
            A reflection result whose ``created_summaries`` contain the derived
            routines and deviations. Failures degrade to an empty result.
        """
        empty_result = LongTermReflectionResultV1(reflected_objects=(), created_summaries=())
        if not self.enabled:
            return empty_result

        try:
            resolved_timezone = self._resolve_timezone()
            reference = self._normalize_reference_datetime(now=now, timezone=resolved_timezone)
            current_daypart = _daypart_for_datetime(reference)
            reference_day = _routine_day_for_datetime(reference, daypart=current_daypart)  # AUDIT-FIX(#1): Use routine-day semantics so overnight signals map to the correct night and weekday bucket.

            signals: list[_RawPatternSignal] = []
            for item in tuple(objects or ()):
                signal = self._safe_classify_raw_signal(item=item, timezone=resolved_timezone)
                if signal is not None:
                    signals.append(signal)  # AUDIT-FIX(#5): Skip malformed input records instead of aborting the full batch.

            created: list[LongTermMemoryObjectV1] = []
            presence_routines: list[LongTermMemoryObjectV1] = []
            signal_tuple = tuple(signals)
            for weekday_class in _WEEKDAY_CLASSES:
                for daypart in _DAYPARTS:
                    presence = self._build_presence_routine(
                        signals=signal_tuple,
                        weekday_class=weekday_class,
                        daypart=daypart,
                        reference_day=reference_day,
                    )
                    if presence is not None:
                        presence_routines.append(presence)
                        created.append(presence)
                    created.extend(
                        self._build_interaction_routines(
                            signals=signal_tuple,
                            weekday_class=weekday_class,
                            daypart=daypart,
                            reference_day=reference_day,
                        )
                    )
            created.extend(
                self._build_presence_deviations(
                    routines=tuple(presence_routines),
                    signals=signal_tuple,
                    reference=reference,
                    reference_day=reference_day,
                    current_daypart=current_daypart,
                )
            )
            return LongTermReflectionResultV1(reflected_objects=(), created_summaries=tuple(created))
        except Exception:  # AUDIT-FIX(#5): This compiler runs in a long-lived process; unexpected data must not crash the surrounding service.
            _LOGGER.exception("Failed to compile sensor-derived long-term memory objects")
            return empty_result

    def _resolve_timezone(self) -> tzinfo:
        """Resolve the configured timezone or fall back safely."""
        normalized_name = _normalize_text(self.timezone_name) or _DEFAULT_TIMEZONE_NAME
        try:
            return ZoneInfo(normalized_name)
        except (ValueError, ZoneInfoNotFoundError):
            _LOGGER.warning(
                "Falling back from invalid or unavailable timezone %r to %r",
                normalized_name,
                _DEFAULT_TIMEZONE_NAME,
            )  # AUDIT-FIX(#3): Broken timezone config or missing tzdata must not take the compiler down.
            if normalized_name != _DEFAULT_TIMEZONE_NAME:
                try:
                    return ZoneInfo(_DEFAULT_TIMEZONE_NAME)
                except (ValueError, ZoneInfoNotFoundError):
                    _LOGGER.warning(
                        "Fallback timezone %r is also unavailable; using UTC instead",
                        _DEFAULT_TIMEZONE_NAME,
                    )  # AUDIT-FIX(#3): Preserve availability even when system tzdata is missing.
            return datetime.UTC

    def _normalize_reference_datetime(self, *, now: datetime | None, timezone: tzinfo) -> datetime:
        """Normalize the optional reference time into the target timezone."""
        if now is None:
            return datetime.now(timezone)
        if now.tzinfo is None or now.utcoffset() is None:
            return now.replace(tzinfo=timezone)  # AUDIT-FIX(#4): Naive datetimes are ambiguous; bind them to the configured local timezone before deriving day/date buckets.
        return now.astimezone(timezone)  # AUDIT-FIX(#4): Normalize all aware timestamps into the configured timezone before daypart and date calculations.

    def _safe_classify_raw_signal(
        self,
        *,
        item: LongTermMemoryObjectV1,
        timezone: tzinfo,
    ) -> _RawPatternSignal | None:
        """Classify one raw pattern while shielding the compile pass from errors."""
        try:
            return self._classify_raw_signal(item=item, timezone=timezone)
        except Exception:
            _LOGGER.warning(
                "Skipping malformed long-term memory object during sensor routine compilation: memory_id=%r",
                getattr(item, "memory_id", "<missing>"),
                exc_info=True,
            )  # AUDIT-FIX(#5): One corrupt record must not poison the whole compile pass.
            return None

    def _classify_raw_signal(
        self,
        *,
        item: LongTermMemoryObjectV1,
        timezone: tzinfo,
    ) -> _RawPatternSignal | None:
        """Classify one memory object into a normalized routine signal."""
        item = item.canonicalized()
        if item.kind != "pattern" or item.status not in {"active", "candidate", "uncertain"}:
            return None

        attrs = _coerce_attributes(item.attributes)
        if _normalize_text(attrs.get("memory_domain", "")).lower() == "sensor_routine":
            return None

        daypart = _normalize_text(attrs.get("daypart", "")).lower()
        if daypart not in _DAYPARTS:
            return None

        routine_type = ""
        interaction_type: str | None = None
        if item.memory_id.startswith("pattern:presence:"):
            routine_type = "presence"
        elif item.memory_id.startswith("pattern:button:green:start_listening:"):
            routine_type = "interaction"
            interaction_type = "conversation_start"
        elif item.memory_id.startswith("pattern:print:"):
            routine_type = "interaction"
            interaction_type = "print"
        elif item.memory_id.startswith("pattern:camera_use:"):
            routine_type = "interaction"
            interaction_type = "camera_use"
        elif item.memory_id.startswith("pattern:camera_interaction:"):
            routine_type = "interaction"
            interaction_type = "camera_showing"
        if not routine_type:
            return None

        source = getattr(item, "source", None)
        raw_event_values = getattr(source, "event_ids", ()) if source is not None else ()
        event_id_days = self._event_id_days(
            raw_event_values=raw_event_values,
            daypart=daypart,
            timezone=timezone,
        )  # AUDIT-FIX(#1): Convert source timestamps into local routine days so overnight patterns are evaluated correctly.
        if not event_id_days:
            return None  # AUDIT-FIX(#5): Ignore malformed or non-parsable records instead of carrying empty signals deeper into the pipeline.

        return _RawPatternSignal(
            routine_type=routine_type,
            interaction_type=interaction_type,
            daypart=daypart,
            event_days=frozenset(event_day for _, event_day in event_id_days),
            event_id_days=event_id_days,
            event_ids=tuple(event_id for event_id, _ in event_id_days),
        )

    def _event_id_days(
        self,
        *,
        raw_event_values: object,
        daypart: str,
        timezone: tzinfo,
    ) -> tuple[tuple[str, date], ...]:
        """Map source event IDs to routine days in the configured timezone."""
        try:
            iterable = tuple(raw_event_values or ())
        except TypeError:
            return ()

        event_id_days: list[tuple[str, date]] = []
        seen_event_ids: set[str] = set()
        for raw_event_value in iterable:
            event_id = _normalize_text(raw_event_value)
            if not event_id or event_id in seen_event_ids:
                continue
            parsed_datetime = _parse_event_datetime(event_id)
            if parsed_datetime is None:
                continue
            local_datetime = parsed_datetime.astimezone(timezone)  # AUDIT-FIX(#4): Derive local days from the configured timezone, not from whatever offset happened to be embedded in the event ID.
            event_day = _routine_day_for_datetime(local_datetime, daypart=daypart)
            event_id_days.append((event_id, event_day))
            seen_event_ids.add(event_id)
        return tuple(event_id_days)

    def _build_presence_routine(
        self,
        *,
        signals: tuple[_RawPatternSignal, ...],
        weekday_class: str,
        daypart: str,
        reference_day: date,
    ) -> LongTermMemoryObjectV1 | None:
        """Build a presence routine object for one weekday/daypart bucket."""
        relevant = tuple(
            signal
            for signal in signals
            if signal.routine_type == "presence" and signal.daypart == daypart
        )
        if not relevant:
            return None

        observed_days = set(
            _eligible_dates(
                reference_day=reference_day,
                baseline_days=self.baseline_days,
                weekday_class=weekday_class,
            )
        )
        if len(observed_days) < self.min_days_observed:
            return None

        signal_days = {
            event_day
            for signal in relevant
            for event_day in signal.event_days
            if event_day in observed_days
        }
        if not signal_days:
            return None

        ratio = len(signal_days) / len(observed_days)
        if ratio < self.min_routine_ratio:
            return None

        weekday_label = _weekday_label(weekday_class)
        return LongTermMemoryObjectV1(
            memory_id=f"routine:presence:{weekday_class}:{daypart}",
            kind="pattern",
            summary=f"Presence near the device is typical in the {daypart} on {weekday_label}.",  # AUDIT-FIX(#8): Use explicit labels so summaries stay grammatical.
            details="Derived from repeated multimodal presence observations in the bounded sensor-memory window.",
            source=LongTermSourceRefV1(
                source_type="sensor_memory",
                event_ids=self._window_event_ids(signals=relevant, observed_days=observed_days),
                modality="sensor",
            ),
            status="active",
            confidence=min(0.92, 0.45 + (ratio * 0.35) + (min(len(signal_days), 8) * 0.015)),
            sensitivity="low",
            slot_key=f"routine:presence:{weekday_class}:{daypart}",
            value_key="presence_routine",
            valid_from=min(day.isoformat() for day in signal_days),
            valid_to=max(day.isoformat() for day in signal_days),
            attributes={
                "memory_domain": "sensor_routine",
                "routine_type": "presence",
                "weekday_class": weekday_class,
                "daypart": daypart,
                "days_observed": len(observed_days),
                "days_with_presence": len(signal_days),
                "baseline_window_days": self.baseline_days,
                "typical_presence_ratio": round(ratio, 4),
                "recent_support_count": len(signal_days),
                "last_observed_date": max(signal_days).isoformat(),
            },
        )

    def _build_interaction_routines(
        self,
        *,
        signals: tuple[_RawPatternSignal, ...],
        weekday_class: str,
        daypart: str,
        reference_day: date,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        """Build interaction routine objects for one weekday/daypart bucket."""
        observed_days = set(
            _eligible_dates(
                reference_day=reference_day,
                baseline_days=self.baseline_days,
                weekday_class=weekday_class,
            )
        )
        if len(observed_days) < self.min_days_observed:
            return ()

        created: list[LongTermMemoryObjectV1] = []
        weekday_label = _weekday_label(weekday_class)
        for interaction_type in ("conversation_start", "print", "camera_use", "camera_showing"):
            relevant = tuple(
                signal
                for signal in signals
                if signal.routine_type == "interaction"
                and signal.interaction_type == interaction_type
                and signal.daypart == daypart
            )
            if not relevant:
                continue

            signal_days = {
                event_day
                for signal in relevant
                for event_day in signal.event_days
                if event_day in observed_days
            }
            if not signal_days:
                continue

            ratio = len(signal_days) / len(observed_days)
            if ratio < self.min_routine_ratio:
                continue

            created.append(
                LongTermMemoryObjectV1(
                    memory_id=f"routine:interaction:{interaction_type}:{weekday_class}:{daypart}",
                    kind="pattern",
                    summary=f"{interaction_type.replace('_', ' ').title()} is typical in the {daypart} on {weekday_label}.",  # AUDIT-FIX(#8): Use explicit labels so summaries stay grammatical.
                    details="Derived from repeated multimodal interaction signals in the bounded sensor-memory window.",
                    source=LongTermSourceRefV1(
                        source_type="sensor_memory",
                        event_ids=self._window_event_ids(signals=relevant, observed_days=observed_days),
                        modality="sensor",
                    ),
                    status="active",
                    confidence=min(0.9, 0.42 + (ratio * 0.34) + (min(len(signal_days), 8) * 0.015)),
                    sensitivity="low",
                    slot_key=f"routine:interaction:{interaction_type}:{weekday_class}:{daypart}",
                    value_key="interaction_routine",
                    valid_from=min(day.isoformat() for day in signal_days),
                    valid_to=max(day.isoformat() for day in signal_days),
                    attributes={
                        "memory_domain": "sensor_routine",
                        "routine_type": "interaction",
                        "interaction_type": interaction_type,
                        "weekday_class": weekday_class,
                        "daypart": daypart,
                        "days_observed": len(observed_days),
                        "days_with_interaction": len(signal_days),
                        "baseline_window_days": self.baseline_days,
                        "typical_interaction_ratio": round(ratio, 4),
                        "last_observed_date": max(signal_days).isoformat(),
                    },
                )
            )
        return tuple(created)

    def _build_presence_deviations(
        self,
        *,
        routines: tuple[LongTermMemoryObjectV1, ...],
        signals: tuple[_RawPatternSignal, ...],
        reference: datetime,
        reference_day: date,
        current_daypart: str,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        """Build missing-presence deviations for the current daypart."""
        if _daypart_progress(reference, daypart=current_daypart) < self.deviation_min_daypart_progress:
            return ()  # AUDIT-FIX(#2): Do not treat "no presence yet" as abnormal until most of the current daypart has elapsed.

        today_presence = any(
            signal.routine_type == "presence"
            and signal.daypart == current_daypart
            and reference_day in signal.event_days
            for signal in signals
        )
        if today_presence:
            return ()

        current_weekday_class = _weekday_class(reference_day)
        candidate_deviations: list[tuple[int, float, str, LongTermMemoryObjectV1, float]] = []
        for routine in routines:
            attrs = _coerce_attributes(routine.attributes)
            if attrs.get("routine_type") != "presence":
                continue
            if attrs.get("daypart") != current_daypart:
                continue

            weekday_class = _normalize_text(attrs.get("weekday_class", ""))
            if weekday_class not in {current_weekday_class, "all_days"}:
                continue

            expected_ratio = _coerce_ratio(attrs.get("typical_presence_ratio", 0.0), default=0.0)
            delta = max(0.0, expected_ratio)
            if delta < self.deviation_min_delta:
                continue

            specificity = 1 if weekday_class == current_weekday_class else 0
            candidate_deviations.append((specificity, expected_ratio, weekday_class, routine, delta))

        if not candidate_deviations:
            return ()

        _, expected_ratio, weekday_class, routine, delta = max(
            candidate_deviations,
            key=lambda candidate: (candidate[0], candidate[1]),
        )  # AUDIT-FIX(#7): Prefer the most specific applicable routine to avoid duplicate absence summaries for the same daypart.

        weekday_label = _weekday_label(weekday_class)
        return (
            LongTermMemoryObjectV1(
                memory_id=f"deviation:presence:{weekday_class}:{current_daypart}:{reference_day.isoformat()}",
                kind="summary",
                summary=f"Presence seems unusually low in the {current_daypart} for {weekday_label}.",  # AUDIT-FIX(#8): Use explicit labels so summaries stay grammatical.
                details="Derived from the current daypart lacking expected presence against a bounded routine baseline.",
                source=LongTermSourceRefV1(
                    source_type="sensor_memory",
                    event_ids=tuple(getattr(getattr(routine, "source", None), "event_ids", ())[:_MAX_EVENT_IDS]),
                    modality="sensor",
                ),
                status="candidate",
                confidence=min(0.86, 0.42 + (expected_ratio * 0.28) + (delta * 0.2)),
                sensitivity="low",
                slot_key=f"deviation:presence:{weekday_class}:{current_daypart}:{reference_day.isoformat()}",
                value_key="missing_presence",
                valid_from=reference_day.isoformat(),
                valid_to=reference_day.isoformat(),
                attributes={
                    "memory_domain": "sensor_routine",
                    "summary_type": "sensor_deviation",
                    "deviation_type": "missing_presence",
                    "weekday_class": weekday_class,
                    "daypart": current_daypart,
                    "date": reference_day.isoformat(),
                    "expected_ratio": round(expected_ratio, 4),
                    "current_ratio": 0.0,
                    "delta_ratio": round(delta, 4),
                    "baseline_window_days": self.baseline_days,
                    "requires_live_confirmation": True,
                },
            ),
        )

    def _window_event_ids(
        self,
        *,
        signals: tuple[_RawPatternSignal, ...],
        observed_days: set[date],
    ) -> tuple[str, ...]:
        """Collect bounded supporting event IDs for emitted summaries."""
        event_ids: list[str] = []
        seen_event_ids: set[str] = set()
        for signal in signals:
            for event_id, event_day in signal.event_id_days:
                if event_day not in observed_days:
                    continue
                if event_id in seen_event_ids:
                    continue
                event_ids.append(event_id)
                seen_event_ids.add(event_id)
        return tuple(event_ids[-_MAX_EVENT_IDS:])


__all__ = ["LongTermSensorMemoryCompiler"]
