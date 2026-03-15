from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.longterm.models import LongTermMemoryObjectV1, LongTermReflectionResultV1, LongTermSourceRefV1

_DAYPARTS = ("morning", "afternoon", "evening", "night")
_WEEKDAY_CLASSES = ("all_days", "weekday", "weekend")
_MAX_EVENT_IDS = 32


def _normalize_text(value: str | None) -> str:
    return " ".join(str(value or "").split()).strip()


def _daypart_for_datetime(value: datetime) -> str:
    hour = value.hour
    if 5 <= hour < 11:
        return "morning"
    if 11 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "night"


def _weekday_class(value: date) -> str:
    return "weekend" if value.weekday() >= 5 else "weekday"


def _eligible_dates(*, reference_day: date, baseline_days: int, weekday_class: str) -> tuple[date, ...]:
    days: list[date] = []
    for offset in range(1, max(1, baseline_days) + 1):
        candidate = reference_day - timedelta(days=offset)
        if weekday_class != "all_days" and _weekday_class(candidate) != weekday_class:
            continue
        days.append(candidate)
    return tuple(days)


def _parse_event_day(event_id: str) -> date | None:
    clean = _normalize_text(event_id)
    if not clean:
        return None
    parts = clean.split(":", 2)
    if len(parts) < 2:
        return None
    timestamp = parts[1]
    try:
        return datetime.strptime(timestamp, "%Y%m%dT%H%M%S%z").date()
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class _RawPatternSignal:
    routine_type: str
    interaction_type: str | None
    daypart: str
    event_days: frozenset[date]
    event_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LongTermSensorMemoryCompiler:
    timezone_name: str = "Europe/Berlin"
    enabled: bool = False
    baseline_days: int = 21
    min_days_observed: int = 6
    min_routine_ratio: float = 0.55
    deviation_min_delta: float = 0.45

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "LongTermSensorMemoryCompiler":
        return cls(
            timezone_name=config.local_timezone_name,
            enabled=config.long_term_memory_sensor_memory_enabled,
            baseline_days=config.long_term_memory_sensor_baseline_days,
            min_days_observed=config.long_term_memory_sensor_min_days_observed,
            min_routine_ratio=config.long_term_memory_sensor_min_routine_ratio,
            deviation_min_delta=config.long_term_memory_sensor_deviation_min_delta,
        )

    def compile(
        self,
        *,
        objects: tuple[LongTermMemoryObjectV1, ...],
        now: datetime | None = None,
    ) -> LongTermReflectionResultV1:
        if not self.enabled:
            return LongTermReflectionResultV1(reflected_objects=(), created_summaries=())

        reference = now or datetime.now(ZoneInfo(self.timezone_name))
        reference_day = reference.date()
        current_daypart = _daypart_for_datetime(reference)
        signals = tuple(
            signal
            for item in objects
            if (signal := self._classify_raw_signal(item)) is not None
        )

        created: list[LongTermMemoryObjectV1] = []
        presence_routines: list[LongTermMemoryObjectV1] = []
        for weekday_class in _WEEKDAY_CLASSES:
            for daypart in _DAYPARTS:
                presence = self._build_presence_routine(
                    signals=signals,
                    weekday_class=weekday_class,
                    daypart=daypart,
                    reference_day=reference_day,
                )
                if presence is not None:
                    presence_routines.append(presence)
                    created.append(presence)
                created.extend(
                    self._build_interaction_routines(
                        signals=signals,
                        weekday_class=weekday_class,
                        daypart=daypart,
                        reference_day=reference_day,
                    )
                )
        created.extend(
            self._build_presence_deviations(
                routines=tuple(presence_routines),
                signals=signals,
                reference_day=reference_day,
                current_daypart=current_daypart,
            )
        )
        return LongTermReflectionResultV1(reflected_objects=(), created_summaries=tuple(created))

    def _classify_raw_signal(self, item: LongTermMemoryObjectV1) -> _RawPatternSignal | None:
        item = item.canonicalized()
        if item.kind != "pattern" or item.status not in {"active", "candidate", "uncertain"}:
            return None
        attrs = dict(item.attributes or {})
        if _normalize_text(str(attrs.get("memory_domain", ""))) == "sensor_routine":
            return None
        daypart = _normalize_text(str(attrs.get("daypart", ""))).lower()
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
        event_ids = tuple(str(value) for value in item.source.event_ids if _normalize_text(value))
        event_days = frozenset(
            parsed
            for value in event_ids
            if (parsed := _parse_event_day(value)) is not None
        )
        return _RawPatternSignal(
            routine_type=routine_type,
            interaction_type=interaction_type,
            daypart=daypart,
            event_days=event_days,
            event_ids=event_ids,
        )

    def _build_presence_routine(
        self,
        *,
        signals: tuple[_RawPatternSignal, ...],
        weekday_class: str,
        daypart: str,
        reference_day: date,
    ) -> LongTermMemoryObjectV1 | None:
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
        return LongTermMemoryObjectV1(
            memory_id=f"routine:presence:{weekday_class}:{daypart}",
            kind="pattern",
            summary=f"Presence near the device is typical in the {daypart} on {weekday_class.replace('_', ' ')}s.",
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
                    summary=f"{interaction_type.replace('_', ' ').title()} is typical in the {daypart} on {weekday_class.replace('_', ' ')}s.",
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
        reference_day: date,
        current_daypart: str,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        today_presence = any(
            signal.routine_type == "presence"
            and signal.daypart == current_daypart
            and reference_day in signal.event_days
            for signal in signals
        )
        current_weekday_class = _weekday_class(reference_day)
        created: list[LongTermMemoryObjectV1] = []
        for routine in routines:
            attrs = dict(routine.attributes or {})
            if attrs.get("routine_type") != "presence":
                continue
            if attrs.get("daypart") != current_daypart:
                continue
            weekday_class = _normalize_text(str(attrs.get("weekday_class", "")))
            if weekday_class not in {current_weekday_class, "all_days"}:
                continue
            expected_ratio = float(attrs.get("typical_presence_ratio", 0.0) or 0.0)
            current_ratio = 1.0 if today_presence else 0.0
            delta = max(0.0, expected_ratio - current_ratio)
            if today_presence or delta < self.deviation_min_delta:
                continue
            created.append(
                LongTermMemoryObjectV1(
                    memory_id=f"deviation:presence:{weekday_class}:{current_daypart}:{reference_day.isoformat()}",
                    kind="summary",
                    summary=f"Presence seems unusually low in the {current_daypart} for {weekday_class.replace('_', ' ')}s.",
                    details="Derived from the current daypart lacking expected presence against a bounded routine baseline.",
                    source=LongTermSourceRefV1(
                        source_type="sensor_memory",
                        event_ids=tuple(routine.source.event_ids[:_MAX_EVENT_IDS]),
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
                        "current_ratio": current_ratio,
                        "delta_ratio": round(delta, 4),
                        "baseline_window_days": self.baseline_days,
                        "requires_live_confirmation": True,
                    },
                )
            )
        return tuple(created)

    def _window_event_ids(
        self,
        *,
        signals: tuple[_RawPatternSignal, ...],
        observed_days: set[date],
    ) -> tuple[str, ...]:
        event_ids: list[str] = []
        for signal in signals:
            for event_id in signal.event_ids:
                event_day = _parse_event_day(event_id)
                if event_day is None or event_day not in observed_days:
                    continue
                if event_id not in event_ids:
                    event_ids.append(event_id)
        return tuple(event_ids[-_MAX_EVENT_IDS:])


__all__ = ["LongTermSensorMemoryCompiler"]
