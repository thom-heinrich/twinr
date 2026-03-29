# CHANGELOG: 2026-03-29
# BUG-1: Preserve an IANA local timezone instead of forcing astimezone(None) fixed-offset datetimes;
#         fixes DST-sensitive hold/boundary errors around quiet hours and refresh boundaries.
# BUG-2: Passive quiet-hours fills are now capped at the next quiet-state transition (quiet end when
#         already inside quiet hours), so they do not block fresh publishes after quiet hours end.
# BUG-3: Serialize the full publish decision path to prevent duplicate publish/mark races under
#         concurrent callers.
# SEC-1: Added a non-blocking in-process + best-effort POSIX inter-process lock to protect the HDMI
#         surface and planner/history state from concurrent local callers on the same Raspberry Pi.
# IMP-1: Added privacy-conscious structured decision logging suitable for modern telemetry pipelines.
# IMP-2: Added cached config time parsing, DST-safe wall-clock boundary resolution, and centralized
#         request construction to keep the hot path deterministic and Pi-friendly.

"""Publish planned calm reserve-card impulses for the HDMI waiting surface.

This module keeps the live publication path very small. Daily sequencing,
nightly preparation, candidate weighting, and persistence live in the reserve
planner modules. The publisher here only decides whether the current runtime
context may expose the next planned reserve-card item right now.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date as LocalDate, datetime, time as LocalTime, timedelta
import functools
import inspect
import logging
import os
from pathlib import Path
import tempfile
import threading
from types import TracebackType
from typing import Any, Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    import fcntl
except ImportError:  # pragma: no cover - only relevant off Unix platforms.
    fcntl = None

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate
from twinr.display.ambient_impulse_cues import (
    DisplayAmbientImpulseCue,
    DisplayAmbientImpulseCueStore,
)
from twinr.display.ambient_impulse_history import DisplayAmbientImpulseHistoryStore
from twinr.display.emoji_cues import DisplayEmojiCueStore
from twinr.display.presentation_cues import DisplayPresentationStore

from .display_reserve_companion_planner import DisplayReserveCompanionPlanner
from .display_reserve_day_plan import _DEFAULT_REFRESH_AFTER_LOCAL
from .display_reserve_runtime import DisplayReserveRuntimePublisher, DisplayReserveRuntimeRequest
from .display_reserve_support import default_local_now, parse_local_time as _parse_local_time

_DEFAULT_ENABLED = True
_DEFAULT_QUIET_HOURS_START = "21:00"
_DEFAULT_QUIET_HOURS_END = "07:00"
_SOURCE = "proactive_ambient_impulse"

_LOGGER = logging.getLogger(__name__)
_IN_PROCESS_PUBLISH_LOCK = threading.Lock()


class _ReserveDisplayItem(Protocol):
    topic_key: str
    title: str
    source: str
    action: str
    attention_state: str
    eyebrow: str | None
    headline: str
    body: str
    symbol: str | None
    accent: str | None
    hold_seconds: float
    reason: str
    candidate_family: str | None

    def semantic_key(self) -> str | None: ...


@functools.lru_cache(maxsize=64)
def _parse_local_time_cached(raw_value: str, fallback: str) -> LocalTime:
    """Parse one local wall-clock time with a small shared cache."""

    return _parse_local_time(raw_value, fallback=fallback)


@functools.lru_cache(maxsize=1)
def _system_local_zoneinfo() -> ZoneInfo | None:
    """Best-effort resolve the system local IANA timezone.

    `datetime.astimezone()` with no explicit target returns a fixed-offset
    `datetime.timezone` instance, which is not sufficient for future DST-aware
    wall-clock boundary calculations. On Raspberry Pi OS / Linux deployments,
    `/etc/localtime` or `/etc/timezone` usually expose the real IANA key.
    """

    env_tz = str(os.getenv("TZ", "") or "").strip()
    if env_tz:
        try:
            return ZoneInfo(env_tz)
        except ZoneInfoNotFoundError:
            pass

    timezone_file = Path("/etc/timezone")
    try:
        timezone_name = timezone_file.read_text(encoding="utf-8").strip()
    except OSError:
        timezone_name = ""
    if timezone_name:
        try:
            return ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError:
            pass

    localtime_path = Path("/etc/localtime")
    try:
        resolved = localtime_path.resolve(strict=True)
    except OSError:
        resolved = None
    if resolved is not None:
        for base in (
            Path("/usr/share/zoneinfo"),
            Path("/usr/lib/zoneinfo"),
            Path("/usr/share/lib/zoneinfo"),
            Path("/etc/zoneinfo"),
        ):
            try:
                relative = resolved.relative_to(base)
            except ValueError:
                continue
            zone_key = relative.as_posix()
            if zone_key:
                try:
                    return ZoneInfo(zone_key)
                except ZoneInfoNotFoundError:
                    continue
    return None


def _config_local_time(config: TwinrConfig, attr_name: str, fallback: str) -> LocalTime:
    """Read one local-time config field with caching and a safe fallback."""

    raw_value = getattr(config, attr_name, fallback)
    if isinstance(raw_value, LocalTime):
        return raw_value.replace(tzinfo=None)
    normalized = str(raw_value or fallback).strip()
    return _parse_local_time_cached(normalized or fallback, fallback)


def _supports_ambient_impulses(config: TwinrConfig) -> bool:
    """Return whether the current display/runtime setup should allow impulses."""

    if not bool(getattr(config, "display_ambient_impulses_enabled", _DEFAULT_ENABLED)):
        return False
    driver = str(getattr(config, "display_driver", "") or "").strip().lower()
    return driver.startswith("hdmi")


def _normalize_local_datetime(value: datetime) -> datetime:
    """Round-trip one aware datetime through UTC to normalize DST folds/gaps."""

    if value.tzinfo is None or value.utcoffset() is None:
        return value
    return value.astimezone(UTC).astimezone(value.tzinfo)


def _utc_instant(value: datetime) -> datetime:
    """Return one UTC-normalized instant for reliable ordering and deltas."""

    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _wall_clock_candidates_for_date(
    *,
    local_date: LocalDate,
    at_time: LocalTime,
    local_tz: Any,
) -> tuple[datetime, ...]:
    """Return valid normalized candidates for one local wall-clock boundary.

    This considers both fold states so repeated wall-clock times on DST fall-back
    days yield both valid occurrences. For nonexistent times on spring-forward
    days, only normalized candidates that do not land *before* the requested wall
    time are accepted.
    """

    candidates: list[datetime] = []
    seen_utc_instants: set[datetime] = set()
    naive_boundary = datetime.combine(local_date, at_time)

    for fold in (0, 1):
        candidate = naive_boundary.replace(tzinfo=local_tz, fold=fold)
        normalized = _normalize_local_datetime(candidate)
        if normalized.date() != local_date:
            continue
        if normalized.timetz().replace(tzinfo=None) < at_time:
            continue
        utc_instant = normalized.astimezone(UTC)
        if utc_instant in seen_utc_instants:
            continue
        seen_utc_instants.add(utc_instant)
        candidates.append(normalized)

    candidates.sort(key=_utc_instant)
    return tuple(candidates)


def _next_local_boundary(*, local_now: datetime, at_time: LocalTime) -> datetime:
    """Return the next local datetime for one wall-clock boundary.

    When the local timezone has DST transitions, the next wall-clock occurrence
    can be ambiguous (fold) or nonexistent (gap). This helper resolves both
    cases without collapsing the timezone to a fixed offset.
    """

    if local_now.tzinfo is None or local_now.utcoffset() is None:
        candidate = local_now.replace(
            hour=at_time.hour,
            minute=at_time.minute,
            second=0,
            microsecond=0,
        )
        if candidate <= local_now:
            candidate = candidate + timedelta(days=1)
        return candidate

    local_tz = local_now.tzinfo
    for day_offset in (0, 1, 2):
        boundary_date = local_now.date() + timedelta(days=day_offset)
        for candidate in _wall_clock_candidates_for_date(
            local_date=boundary_date,
            at_time=at_time,
            local_tz=local_tz,
        ):
            if _utc_instant(candidate) > _utc_instant(local_now):
                return candidate

    # Defensive fallback: should be unreachable for sane IANA zones.
    fallback_candidate = local_now.replace(
        hour=at_time.hour,
        minute=at_time.minute,
        second=0,
        microsecond=0,
    )
    if _utc_instant(fallback_candidate) <= _utc_instant(local_now):
        fallback_candidate = fallback_candidate + timedelta(days=1)
    return _normalize_local_datetime(fallback_candidate)


def _coerce_effective_local_now(value: datetime) -> datetime:
    """Return one local-aware datetime that preserves DST semantics when possible."""

    local_zone = _system_local_zoneinfo()
    if local_zone is not None:
        return value.astimezone(local_zone)
    return value.astimezone()


def _default_publish_lock_path() -> str:
    """Return a per-user lock path suitable for local Raspberry Pi deployments."""

    runtime_dir = str(os.getenv("XDG_RUNTIME_DIR", "") or "").strip()
    if runtime_dir and os.path.isabs(runtime_dir):
        return os.path.join(runtime_dir, "twinr-display-ambient-impulse.lock")
    get_uid = getattr(os, "getuid", None)
    uid = int(get_uid()) if callable(get_uid) else 0
    return os.path.join(
        tempfile.gettempdir(),
        f"twinr-display-ambient-impulse-{uid}.lock",
    )


class _InterprocessPublishLock:
    """Best-effort non-blocking advisory file lock for local-process serialization."""

    __slots__ = ("_fd", "_path")

    def __init__(self, path: str) -> None:
        self._fd: int | None = None
        self._path = path

    def acquire(self) -> bool:
        """Try to acquire the file lock without blocking."""

        if fcntl is None:
            return True

        try:
            parent = os.path.dirname(self._path) or "."
            os.makedirs(parent, mode=0o700, exist_ok=True)
        except OSError:
            return False

        flags = os.O_CREAT | os.O_RDWR
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)

        try:
            fd = os.open(self._path, flags, 0o600)
        except OSError:
            return False

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False

        self._fd = fd
        return True

    def release(self) -> None:
        """Release the file lock if held."""

        if self._fd is None:
            return
        try:
            if fcntl is not None:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.release()


@dataclass(frozen=True, slots=True)
class DisplayAmbientImpulsePublishResult:
    """Summarize one ambient-impulse publish attempt."""

    action: str
    reason: str
    topic_key: str | None = None
    cue: DisplayAmbientImpulseCue | None = None


@dataclass(slots=True)
class DisplayAmbientImpulsePublisher:
    """Publish the next planned reserve impulse when the runtime is ready."""

    runtime_publisher: DisplayReserveRuntimePublisher
    active_store: DisplayAmbientImpulseCueStore
    emoji_store: DisplayEmojiCueStore
    presentation_store: DisplayPresentationStore
    planner: DisplayReserveCompanionPlanner
    source: str = _SOURCE
    local_now: Callable[[], datetime] = default_local_now

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayAmbientImpulsePublisher":
        """Build one publisher from the configured display cue stores."""

        runtime_publisher = DisplayReserveRuntimePublisher.from_config(
            config,
            default_source=_SOURCE,
        )
        return cls(
            runtime_publisher=runtime_publisher,
            active_store=runtime_publisher.active_store,
            emoji_store=DisplayEmojiCueStore.from_config(config),
            presentation_store=DisplayPresentationStore.from_config(config),
            planner=DisplayReserveCompanionPlanner.from_config(config),
        )

    @property
    def candidate_loader(self) -> Callable[..., tuple[AmbientDisplayImpulseCandidate, ...]]:
        """Expose the planner candidate loader for tests and dependency injection."""

        return self.planner.candidate_loader

    @property
    def history_store(self) -> DisplayAmbientImpulseHistoryStore:
        """Expose the shared reserve history store for tests and observability."""

        return self.runtime_publisher.history_store

    def set_candidate_loader(
        self,
        value: Callable[..., tuple[AmbientDisplayImpulseCandidate, ...]],
    ) -> None:
        """Replace the planner candidate loader for tests or alternate wiring."""

        try:
            signature = inspect.signature(value)
        except (TypeError, ValueError):
            self.planner.candidate_loader = value
            return
        accepts_max_items = "max_items" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if accepts_max_items:
            self.planner.candidate_loader = value
            return

        def _wrapped_candidate_loader(
            config: TwinrConfig,
            *,
            local_now: datetime,
            max_items: int,
        ) -> tuple[AmbientDisplayImpulseCandidate, ...]:
            del max_items
            return tuple(value(config, local_now=local_now))

        self.planner.candidate_loader = _wrapped_candidate_loader

    def publish_if_due(
        self,
        *,
        config: TwinrConfig,
        monotonic_now: float,
        runtime_status: str,
        presence_active: bool,
        local_now: datetime | None = None,
    ) -> DisplayAmbientImpulsePublishResult:
        """Publish the next planned reserve cue when the live context allows it."""

        del monotonic_now
        effective_local_now = _coerce_effective_local_now(local_now or self.local_now())

        thread_lock_acquired = _IN_PROCESS_PUBLISH_LOCK.acquire(blocking=False)
        if not thread_lock_acquired:
            result = DisplayAmbientImpulsePublishResult(
                action="blocked",
                reason="publisher_busy",
            )
            self._log_result(
                result,
                config=config,
                local_now=effective_local_now,
                runtime_status=runtime_status,
                presence_active=presence_active,
            )
            return result

        lock_path = str(
            getattr(config, "proactive_ambient_impulse_publish_lock_path", "") or ""
        ).strip() or _default_publish_lock_path()

        try:
            with _InterprocessPublishLock(lock_path) as process_lock_acquired:
                if not process_lock_acquired:
                    result = DisplayAmbientImpulsePublishResult(
                        action="blocked",
                        reason="publisher_busy",
                    )
                    self._log_result(
                        result,
                        config=config,
                        local_now=effective_local_now,
                        runtime_status=runtime_status,
                        presence_active=presence_active,
                    )
                    return result

                result = self._publish_if_due_unlocked(
                    config=config,
                    runtime_status=runtime_status,
                    presence_active=presence_active,
                    local_now=effective_local_now,
                )
                self._log_result(
                    result,
                    config=config,
                    local_now=effective_local_now,
                    runtime_status=runtime_status,
                    presence_active=presence_active,
                )
                return result
        finally:
            _IN_PROCESS_PUBLISH_LOCK.release()

    def _publish_if_due_unlocked(
        self,
        *,
        config: TwinrConfig,
        runtime_status: str,
        presence_active: bool,
        local_now: datetime,
    ) -> DisplayAmbientImpulsePublishResult:
        """Run one publish attempt with caller-held serialization."""

        if not _supports_ambient_impulses(config):
            return DisplayAmbientImpulsePublishResult(action="inactive", reason="unsupported")
        if runtime_status != "waiting":
            return DisplayAmbientImpulsePublishResult(action="blocked", reason="runtime_not_waiting")
        if self.emoji_store.load_active(now=local_now) is not None:
            return DisplayAmbientImpulsePublishResult(action="blocked", reason="emoji_surface_owned")
        if self.presentation_store.load_active(now=local_now) is not None:
            return DisplayAmbientImpulsePublishResult(
                action="blocked",
                reason="presentation_surface_owned",
            )
        if self.active_store.load_active(now=local_now) is not None:
            return DisplayAmbientImpulsePublishResult(
                action="blocked",
                reason="ambient_impulse_active",
            )
        if self._quiet_hours_active(config=config, local_now=local_now):
            restored = self._restore_passive_fill(
                config=config,
                local_now=local_now,
                reason="quiet_hours_passive_fill",
            )
            if restored is not None:
                return restored
            return DisplayAmbientImpulsePublishResult(action="blocked", reason="quiet_hours")
        if not presence_active:
            restored = self._restore_passive_fill(
                config=config,
                local_now=local_now,
                reason="no_active_presence_passive_fill",
            )
            if restored is not None:
                return restored
            return DisplayAmbientImpulsePublishResult(
                action="blocked",
                reason="no_active_presence",
            )

        item = self.planner.peek_next_item(
            config=config,
            local_now=local_now,
        )
        if item is None:
            fallback_item = self.planner.peek_idle_fill_item(
                config=config,
                local_now=local_now,
            )
            if fallback_item is None:
                return DisplayAmbientImpulsePublishResult(
                    action="inactive",
                    reason="no_planned_item",
                )
            visible_only_result = self.runtime_publisher.show_visible_only(
                self._build_runtime_request(
                    fallback_item,
                    config=config,
                    local_now=local_now,
                    reason=f"{fallback_item.reason}; idle_fill",
                    hold_seconds=self._idle_fill_hold_seconds(
                        config=config,
                        local_now=local_now,
                        fallback_item=fallback_item,
                    ),
                    idle_fill=True,
                ),
                now=local_now,
            )
            return DisplayAmbientImpulsePublishResult(
                action="restored_fill",
                reason="plan_exhausted_idle_fill",
                topic_key=fallback_item.topic_key,
                cue=visible_only_result.cue,
            )

        published = self.runtime_publisher.publish(
            self._build_runtime_request(
                item,
                config=config,
                local_now=local_now,
                reason=item.reason,
                hold_seconds=item.hold_seconds,
                idle_fill=False,
            ),
            now=local_now,
        )
        self.planner.mark_published(
            config=config,
            local_now=local_now,
        )
        return DisplayAmbientImpulsePublishResult(
            action="published",
            reason=item.reason,
            topic_key=item.topic_key,
            cue=published.cue,
        )

    def _build_runtime_request(
        self,
        item: _ReserveDisplayItem,
        *,
        config: TwinrConfig,
        local_now: datetime,
        reason: str,
        hold_seconds: float,
        idle_fill: bool,
    ) -> DisplayReserveRuntimeRequest:
        """Build one runtime request with consistent metadata and anchors."""

        del config, local_now
        metadata: dict[str, Any] = {
            "eyebrow": item.eyebrow,
            "accent": item.accent,
            "symbol": item.symbol,
        }
        if idle_fill:
            metadata["idle_fill"] = True

        return DisplayReserveRuntimeRequest(
            topic_key=item.topic_key,
            title=item.title,
            cue_source=self.source,
            history_source=item.source,
            action=item.action,
            attention_state=item.attention_state,
            eyebrow=item.eyebrow,
            headline=item.headline,
            body=item.body,
            symbol=item.symbol,
            accent=item.accent,
            hold_seconds=float(hold_seconds),
            reason=reason,
            semantic_topic_key=item.semantic_key(),
            candidate_family=item.candidate_family,
            match_anchors=(item.title, item.headline, item.body),
            metadata=metadata,
        )

    def _restore_passive_fill(
        self,
        *,
        config: TwinrConfig,
        local_now: datetime,
        reason: str,
    ) -> DisplayAmbientImpulsePublishResult | None:
        """Restore one passive reserve fill without recording a new exposure.

        Temporary right-lane overrides such as social prompts may expire while
        normal ambient publishing is intentionally blocked. In that state Twinr
        should only restore a card that was already truly shown earlier the
        same day instead of surfacing the next unpublished plan item and
        pinning the rotation on that first topic.
        """

        plan = self.planner.ensure_plan(
            config=config,
            local_now=local_now,
        )
        fill_item = plan.last_shown_item()
        if fill_item is None:
            fill_item = self.planner.peek_idle_fill_item(
                config=config,
                local_now=local_now,
            )
        if fill_item is None:
            return None
        restored = self.runtime_publisher.show_visible_only(
            self._build_runtime_request(
                fill_item,
                config=config,
                local_now=local_now,
                reason=f"{fill_item.reason}; {reason}",
                hold_seconds=self._idle_fill_hold_seconds(
                    config=config,
                    local_now=local_now,
                    fallback_item=fill_item,
                ),
                idle_fill=True,
            ),
            now=local_now,
        )
        return DisplayAmbientImpulsePublishResult(
            action="restored_fill",
            reason=reason,
            topic_key=fill_item.topic_key,
            cue=restored.cue,
        )

    def _quiet_hours_active(
        self,
        *,
        config: TwinrConfig,
        local_now: datetime,
    ) -> bool:
        """Return whether the current local time falls into quiet hours."""

        start = _config_local_time(
            config,
            "proactive_quiet_hours_start_local",
            _DEFAULT_QUIET_HOURS_START,
        )
        end = _config_local_time(
            config,
            "proactive_quiet_hours_end_local",
            _DEFAULT_QUIET_HOURS_END,
        )
        if start == end:
            return False
        current = local_now.timetz().replace(tzinfo=None)
        if start < end:
            return start <= current < end
        return current >= start or current < end

    def _next_quiet_transition(
        self,
        *,
        config: TwinrConfig,
        local_now: datetime,
    ) -> datetime | None:
        """Return the next quiet-hours state transition from the current time."""

        start = _config_local_time(
            config,
            "proactive_quiet_hours_start_local",
            _DEFAULT_QUIET_HOURS_START,
        )
        end = _config_local_time(
            config,
            "proactive_quiet_hours_end_local",
            _DEFAULT_QUIET_HOURS_END,
        )
        if start == end:
            return None
        transition_time = end if self._quiet_hours_active(config=config, local_now=local_now) else start
        return _next_local_boundary(local_now=local_now, at_time=transition_time)

    def _idle_fill_hold_seconds(
        self,
        *,
        config: TwinrConfig,
        local_now: datetime,
        fallback_item: object,
    ) -> float:
        """Return how long one passive exhausted-plan fill may stay visible."""

        refresh_after = _config_local_time(
            config,
            "display_reserve_bus_refresh_after_local",
            _DEFAULT_REFRESH_AFTER_LOCAL,
        )
        next_refresh = _next_local_boundary(local_now=local_now, at_time=refresh_after)
        next_quiet_transition = self._next_quiet_transition(
            config=config,
            local_now=local_now,
        )

        candidate_boundaries = [next_refresh]
        if next_quiet_transition is not None:
            candidate_boundaries.append(next_quiet_transition)

        seconds_until_boundary = min(
            max(1.0, (_utc_instant(boundary) - _utc_instant(local_now)).total_seconds())
            for boundary in candidate_boundaries
        )
        configured_hold_seconds = float(getattr(fallback_item, "hold_seconds", 0.0) or 0.0)
        base_hold_seconds = max(60.0, configured_hold_seconds)
        return max(
            60.0,
            min(max(base_hold_seconds, 15.0 * 60.0), seconds_until_boundary),
        )

    def _log_result(
        self,
        result: DisplayAmbientImpulsePublishResult,
        *,
        config: TwinrConfig,
        local_now: datetime,
        runtime_status: str,
        presence_active: bool,
    ) -> None:
        """Emit one privacy-conscious structured log for observability."""

        if result.action in {"published", "restored_fill"}:
            level = logging.INFO
        elif result.reason in {"publisher_busy"}:
            level = logging.WARNING
        else:
            level = logging.DEBUG

        if not _LOGGER.isEnabledFor(level):
            return

        _LOGGER.log(
            level,
            "display_ambient_impulse_publish_decision",
            extra={
                "action": result.action,
                "reason": result.reason,
                "topic_key": result.topic_key,
                "runtime_status": runtime_status,
                "presence_active": presence_active,
                "quiet_hours_active": self._quiet_hours_active(
                    config=config,
                    local_now=local_now,
                ),
                "display_driver": str(getattr(config, "display_driver", "") or ""),
                "source": self.source,
                "local_now": local_now.isoformat(),
            },
        )