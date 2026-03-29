# CHANGELOG: 2026-03-29
# BUG-1: Prepared plans are no longer cleared before the adopted current-day plan has been durably saved, preventing data loss on I/O errors and power-loss windows.
# BUG-2: Reflection / learning-profile / outcome-summary failures no longer abort next-day preparation; nightly planning now degrades gracefully and stays useful offline.
# BUG-3: Persisted state parsing is hardened against invalid ISO days, malformed counters, and string-as-sequence topic corruption.
# SEC-1: Nightly maintenance now uses an advisory lock, atomic fsync-backed state writes, and private file permissions to reduce race-driven corruption and local data leakage on Raspberry Pi deployments.
# IMP-1: Explicit IANA timezone handling removes dependence on the host system timezone and makes DST/night-window behavior deterministic.
# IMP-2: Existing target-day plans are reused, late prepared plans can replace pristine live-built plans, and non-critical telemetry steps are isolated from the critical prepare-and-adopt path.

"""Own nightly reserve-lane recalibration and prepared next-day plans.

The right-hand HDMI reserve lane already has a deterministic current-day
planner and a slower companion-flow candidate loader. What was still missing
was one explicit overnight step that:

- reviews recent shown-card outcomes
- runs long-term reflection/world refresh once for the coming day
- prepares the next local-day reserve plan ahead of time
- lets the morning runtime adopt that prepared plan without rebuilding it live

This module keeps that slower daily orchestration out of both the hot
display-publish path and the generic realtime loop.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from datetime import date as LocalDate, datetime, timedelta, timezone
import errno
import json
import logging
import os
from pathlib import Path
import tempfile

try:
    import fcntl
except ImportError:  # pragma: no cover - only relevant on non-Unix platforms
    fcntl = None

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:  # pragma: no cover - Python < 3.9 or stripped builds
    ZoneInfo = None  # type: ignore[assignment]

    class ZoneInfoNotFoundError(Exception):
        """Fallback placeholder when zoneinfo is unavailable."""

from twinr.agent.base_agent.config import TwinrConfig
from twinr.display.ambient_impulse_history import DisplayAmbientImpulseHistoryStore
from twinr.memory.longterm.runtime.service import LongTermMemoryService
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError

from .display_reserve_day_plan import (
    DisplayReserveDayPlan,
    DisplayReserveDayPlanStore,
    DisplayReserveDayPlanner,
    DisplayReservePlannedItem,
)
from .display_reserve_learning import (
    DisplayReserveLearningProfile,
    DisplayReserveLearningProfileBuilder,
)
from .display_reserve_support import (
    compact_text,
    default_local_now,
    format_timestamp,
    parse_local_time as _parse_local_time,
    parse_timestamp as _parse_timestamp,
    utc_now,
)

_DEFAULT_PREPARED_PLAN_PATH = "artifacts/stores/ops/display_reserve_bus_plan_prepared.json"
_DEFAULT_MAINTENANCE_STATE_PATH = "artifacts/stores/ops/display_reserve_bus_maintenance.json"
_DEFAULT_MAINTENANCE_LOCK_PATH = "artifacts/stores/ops/display_reserve_bus_maintenance.lock"
_DEFAULT_NIGHTLY_AFTER_LOCAL = "00:30"
_DEFAULT_OUTCOME_LOOKBACK_DAYS = 2.0
_MAX_TOPIC_COUNT = 4
_PRIVATE_FILE_MODE = 0o600

_LOGGER = logging.getLogger(__name__)


def _resolve_store_path(config: TwinrConfig, attr_name: str, default_path: str) -> Path:
    """Resolve one configured artifact path under the current project root."""

    project_root = Path(config.project_root).expanduser().resolve()
    configured = Path(getattr(config, attr_name, default_path) or default_path)
    return configured if configured.is_absolute() else project_root / configured


def _resolve_local_timezone(config: TwinrConfig):
    """Return one explicit IANA timezone from config when available."""

    for attr_name in (
        "display_reserve_bus_timezone",
        "display_reserve_timezone",
        "local_timezone",
        "timezone",
        "time_zone",
    ):
        raw = compact_text(getattr(config, attr_name, None), max_len=128)
        if not raw:
            continue
        if ZoneInfo is None:
            _LOGGER.warning(
                "Configured timezone %r ignored because zoneinfo is unavailable.",
                raw,
            )
            return None
        try:
            return ZoneInfo(raw)
        except ZoneInfoNotFoundError:
            _LOGGER.warning("Ignoring unknown reserve-bus timezone %r.", raw)
            continue
    return None


def _coerce_local_datetime(config: TwinrConfig, value: datetime) -> datetime:
    """Normalize one datetime into the configured local timezone."""

    target_tz = _resolve_local_timezone(config)
    if value.tzinfo is None:
        if target_tz is not None:
            return value.replace(tzinfo=target_tz)
        fallback_tz = datetime.now().astimezone().tzinfo or timezone.utc
        return value.replace(tzinfo=fallback_tz)
    return value.astimezone(target_tz) if target_tz is not None else value.astimezone()


def _coerce_iso_day(value: object | None) -> str | None:
    """Normalize one stored local-day string."""

    text = compact_text(value, max_len=16)
    if not text:
        return None
    try:
        return LocalDate.fromisoformat(text).isoformat()
    except (TypeError, ValueError):
        return None


def _topic_key(value: object | None) -> str:
    """Return one normalized topic key."""

    return compact_text(value, max_len=96).casefold()


def _normalized_topic_keys(value: object | None) -> tuple[str, ...]:
    """Normalize one persisted topic collection without treating strings as iterables."""

    if value is None or isinstance(value, (str, bytes, bytearray)):
        return ()
    if not isinstance(value, Iterable):
        return ()
    normalized: list[str] = []
    for entry in value:
        topic = _topic_key(entry)
        if topic:
            normalized.append(topic)
    return tuple(normalized)


def _safe_non_negative_int(value: object | None) -> int:
    """Parse one possibly malformed integer into a non-negative counter."""

    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _safe_timestamp_text(value: object | None) -> str | None:
    """Normalize one persisted timestamp when it parses."""

    try:
        parsed = _parse_timestamp(value)
    except Exception:
        return None
    return format_timestamp(parsed) if parsed is not None else None


def _set_private_permissions(path: Path) -> None:
    """Best-effort harden one artifact to user-only permissions."""

    with suppress(OSError):
        os.chmod(path, _PRIVATE_FILE_MODE)


def _fsync_directory(path: Path) -> None:
    """Best-effort fsync for the parent directory after one atomic replace."""

    flags = getattr(os, "O_RDONLY", 0)
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        dir_fd = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write one UTF-8 text file atomically enough for crash-prone edge devices."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        _set_private_permissions(tmp_path)
        os.replace(tmp_path, path)
        _set_private_permissions(path)
        _fsync_directory(path.parent)
    finally:
        with suppress(FileNotFoundError):
            tmp_path.unlink()


@contextmanager
def _advisory_file_lock(path: Path, *, blocking: bool) -> Iterator[bool]:
    """Acquire one best-effort cross-process lock around nightly maintenance."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    _set_private_permissions(path)
    if fcntl is None:  # pragma: no cover - Raspberry Pi / Linux normally has fcntl
        try:
            yield True
        finally:
            handle.close()
        return

    try:
        operation = fcntl.LOCK_EX
        if not blocking:
            operation |= fcntl.LOCK_NB
        try:
            fcntl.flock(handle.fileno(), operation)
        except OSError as exc:
            if not blocking and exc.errno in {errno.EACCES, errno.EAGAIN}:
                yield False
                return
            raise
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} acquired_at={format_timestamp(utc_now())}\n")
        handle.flush()
        with suppress(OSError):
            os.fsync(handle.fileno())
        try:
            yield True
        finally:
            handle.seek(0)
            handle.truncate()
            handle.flush()
            with suppress(OSError):
                os.fsync(handle.fileno())
            with suppress(OSError):
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _join_error_messages(parts: Sequence[str]) -> str | None:
    """Compact one list of degraded auxiliary-step errors for persisted state."""

    if not parts:
        return None
    return compact_text("; ".join(part for part in parts if part), max_len=240) or None


@dataclass(frozen=True, slots=True)
class DisplayReserveNightlyOutcomeSummary:
    """Summarize recent reserve-lane outcomes for overnight review."""

    exposure_count: int = 0
    engaged_count: int = 0
    immediate_pickup_count: int = 0
    cooling_count: int = 0
    avoided_count: int = 0
    ignored_count: int = 0
    pending_count: int = 0
    positive_topics: tuple[str, ...] = ()
    cooling_topics: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "DisplayReserveNightlyOutcomeSummary":
        """Build one summary from persisted JSON-safe data."""

        return cls(
            exposure_count=_safe_non_negative_int(payload.get("exposure_count", 0)),
            engaged_count=_safe_non_negative_int(payload.get("engaged_count", 0)),
            immediate_pickup_count=_safe_non_negative_int(
                payload.get("immediate_pickup_count", 0)
            ),
            cooling_count=_safe_non_negative_int(payload.get("cooling_count", 0)),
            avoided_count=_safe_non_negative_int(payload.get("avoided_count", 0)),
            ignored_count=_safe_non_negative_int(payload.get("ignored_count", 0)),
            pending_count=_safe_non_negative_int(payload.get("pending_count", 0)),
            positive_topics=_normalized_topic_keys(payload.get("positive_topics")),
            cooling_topics=_normalized_topic_keys(payload.get("cooling_topics")),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize one summary into JSON-safe data."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class DisplayReserveNightlyMaintenanceState:
    """Persist the last explicit overnight reserve-lane maintenance result."""

    prepared_local_day: str | None = None
    last_attempted_at: str | None = None
    last_completed_at: str | None = None
    last_status: str = "idle"
    last_error: str | None = None
    reflection_reflected_object_count: int = 0
    reflection_created_summary_count: int = 0
    prepared_candidate_count: int = 0
    prepared_item_count: int = 0
    positive_topics: tuple[str, ...] = ()
    cooling_topics: tuple[str, ...] = ()
    outcome_summary: DisplayReserveNightlyOutcomeSummary = DisplayReserveNightlyOutcomeSummary()

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "DisplayReserveNightlyMaintenanceState":
        """Build one state record from JSON-safe persisted data."""

        outcome_payload = payload.get("outcome_summary")
        return cls(
            prepared_local_day=_coerce_iso_day(payload.get("prepared_local_day")),
            last_attempted_at=_safe_timestamp_text(payload.get("last_attempted_at")),
            last_completed_at=_safe_timestamp_text(payload.get("last_completed_at")),
            last_status=compact_text(payload.get("last_status"), max_len=24).lower() or "idle",
            last_error=compact_text(payload.get("last_error"), max_len=240) or None,
            reflection_reflected_object_count=_safe_non_negative_int(
                payload.get("reflection_reflected_object_count", 0)
            ),
            reflection_created_summary_count=_safe_non_negative_int(
                payload.get("reflection_created_summary_count", 0)
            ),
            prepared_candidate_count=_safe_non_negative_int(
                payload.get("prepared_candidate_count", 0)
            ),
            prepared_item_count=_safe_non_negative_int(payload.get("prepared_item_count", 0)),
            positive_topics=_normalized_topic_keys(payload.get("positive_topics")),
            cooling_topics=_normalized_topic_keys(payload.get("cooling_topics")),
            outcome_summary=DisplayReserveNightlyOutcomeSummary.from_dict(outcome_payload)
            if isinstance(outcome_payload, Mapping)
            else DisplayReserveNightlyOutcomeSummary(),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize one maintenance state into JSON-safe data."""

        payload = asdict(self)
        payload["outcome_summary"] = self.outcome_summary.to_dict()
        return payload


@dataclass(slots=True)
class DisplayReserveNightlyMaintenanceStateStore:
    """Persist the last explicit nightly companion-maintenance state."""

    path: Path

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayReserveNightlyMaintenanceStateStore":
        """Resolve the maintenance-state artifact path from configuration."""

        return cls(
            path=_resolve_store_path(
                config,
                "display_reserve_bus_maintenance_state_path",
                _DEFAULT_MAINTENANCE_STATE_PATH,
            )
        )

    def load(self) -> DisplayReserveNightlyMaintenanceState | None:
        """Load the persisted maintenance state when it exists and parses."""

        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            _LOGGER.warning(
                "Failed to read display reserve maintenance state from %s.",
                self.path,
                exc_info=True,
            )
            return None
        if not isinstance(payload, Mapping):
            return None
        try:
            return DisplayReserveNightlyMaintenanceState.from_dict(payload)
        except Exception:
            _LOGGER.warning(
                "Ignoring invalid display reserve maintenance state at %s.",
                self.path,
                exc_info=True,
            )
            return None

    def save(
        self,
        state: DisplayReserveNightlyMaintenanceState,
    ) -> DisplayReserveNightlyMaintenanceState:
        """Persist one maintenance state atomically enough for runtime use."""

        payload = json.dumps(
            state.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
        _atomic_write_text(self.path, payload)
        return state


@dataclass(frozen=True, slots=True)
class DisplayReserveNightlyMaintenanceResult:
    """Describe one explicit overnight reserve-lane maintenance outcome."""

    action: str
    reason: str
    target_local_day: str | None = None
    plan: DisplayReserveDayPlan | None = None
    state: DisplayReserveNightlyMaintenanceState | None = None


@dataclass(slots=True)
class DisplayReserveCompanionPlanner:
    """Coordinate current-day reserve use with explicit nightly preparation."""

    day_planner: DisplayReserveDayPlanner
    prepared_store: DisplayReserveDayPlanStore
    state_store: DisplayReserveNightlyMaintenanceStateStore
    history_store: DisplayAmbientImpulseHistoryStore
    learning_builder_factory: type[DisplayReserveLearningProfileBuilder] = DisplayReserveLearningProfileBuilder
    long_term_memory_factory: Callable[[TwinrConfig], LongTermMemoryService] = LongTermMemoryService.from_config
    local_now: Callable[[], datetime] = default_local_now
    maintenance_lock_path: Path | None = None

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayReserveCompanionPlanner":
        """Build the companion planner and its durable stores from config."""

        return cls(
            day_planner=DisplayReserveDayPlanner.from_config(config),
            prepared_store=DisplayReserveDayPlanStore(
                path=_resolve_store_path(
                    config,
                    "display_reserve_bus_prepared_plan_path",
                    _DEFAULT_PREPARED_PLAN_PATH,
                )
            ),
            state_store=DisplayReserveNightlyMaintenanceStateStore.from_config(config),
            history_store=DisplayAmbientImpulseHistoryStore.from_config(config),
            maintenance_lock_path=_resolve_store_path(
                config,
                "display_reserve_bus_maintenance_lock_path",
                _DEFAULT_MAINTENANCE_LOCK_PATH,
            ),
        )

    @property
    def store(self) -> DisplayReserveDayPlanStore:
        """Expose the active current-day plan store for existing callers/tests."""

        return self.day_planner.store

    @property
    def feedback_store(self):
        """Expose the short-lived feedback store for existing callers/tests."""

        return self.day_planner.feedback_store

    @property
    def candidate_loader(self):
        """Expose the wrapped day-planner candidate loader for tests."""

        return self.day_planner.candidate_loader

    @candidate_loader.setter
    def candidate_loader(self, value) -> None:
        """Replace the wrapped day-planner candidate loader for tests."""

        self.day_planner.candidate_loader = value

    def ensure_plan(
        self,
        *,
        config: TwinrConfig,
        local_now: datetime | None = None,
    ) -> DisplayReserveDayPlan:
        """Return the active plan, adopting a prepared next-day plan when due."""

        effective_now = self._effective_local_now(config=config, local_now=local_now)
        self._clear_stale_prepared_plan(current_day=effective_now.date())
        adopted = self._adopt_prepared_plan_if_due(config=config, local_now=effective_now)
        if adopted is not None:
            self._harden_store_file(self.store)
            return adopted
        plan = self.day_planner.ensure_plan(config=config, local_now=effective_now)
        self._harden_store_file(self.store)
        return plan

    def peek_next_item(
        self,
        *,
        config: TwinrConfig,
        local_now: datetime | None = None,
    ) -> DisplayReservePlannedItem | None:
        """Return the next planned item for the current local day."""

        return self.ensure_plan(config=config, local_now=local_now).current_item()

    def peek_idle_fill_item(
        self,
        *,
        config: TwinrConfig,
        local_now: datetime | None = None,
    ) -> DisplayReservePlannedItem | None:
        """Return one passive same-day fill item when the active plan is exhausted."""

        plan = self.ensure_plan(config=config, local_now=local_now)
        if not plan.is_exhausted():
            return None
        return plan.last_shown_item()

    def mark_published(
        self,
        *,
        config: TwinrConfig,
        local_now: datetime | None = None,
    ) -> DisplayReserveDayPlan:
        """Advance the adopted/current day plan after one successful publish."""

        plan = self.ensure_plan(config=config, local_now=local_now)
        advanced = plan.advance()
        saved = self.store.save(advanced)
        self._harden_store_file(self.store)
        return saved

    def maybe_run_nightly_maintenance(
        self,
        *,
        config: TwinrConfig,
        local_now: datetime | None = None,
        search_backend: object | None = None,
    ) -> DisplayReserveNightlyMaintenanceResult:
        """Run one explicit overnight review and prepare the next local day.

        The planner only runs once per target local day after the configured
        nightly cutoff. It executes long-term reflection/world refresh, derives
        a fresh learning profile, summarizes recent reserve-card outcomes, and
        prepares the next day plan into a separate artifact.
        """

        effective_now = self._effective_local_now(config=config, local_now=local_now)
        if not bool(getattr(config, "display_reserve_bus_nightly_enabled", True)):
            return DisplayReserveNightlyMaintenanceResult(
                action="inactive",
                reason="nightly_disabled",
            )

        target_day = self._nightly_target_local_day(config=config, local_now=effective_now)
        if target_day is None:
            return DisplayReserveNightlyMaintenanceResult(
                action="skipped",
                reason="not_due",
            )
        target_day_text = target_day.isoformat()

        with self._maintenance_lock(blocking=False) as acquired:
            if not acquired:
                return DisplayReserveNightlyMaintenanceResult(
                    action="skipped",
                    reason="locked",
                    target_local_day=target_day_text,
                )

            existing_state = self.state_store.load()
            existing_plan = self._existing_plan_for_local_day(target_day_text)
            if existing_plan is not None:
                return DisplayReserveNightlyMaintenanceResult(
                    action="skipped",
                    reason="already_prepared",
                    target_local_day=target_day_text,
                    plan=existing_plan,
                    state=self._state_for_existing_plan(
                        target_day_text=target_day_text,
                        plan=existing_plan,
                        previous_state=existing_state,
                    ),
                )

            attempted_at = effective_now.astimezone(timezone.utc)
            reflected_count = 0
            created_summary_count = 0
            learning_profile: DisplayReserveLearningProfile | None = None
            outcome_summary = (
                existing_state.outcome_summary
                if existing_state is not None
                else DisplayReserveNightlyOutcomeSummary()
            )
            aux_errors: list[str] = []
            first_aux_exception: Exception | None = None

            try:
                try:
                    memory_service = self.long_term_memory_factory(config)
                    reflection = memory_service.run_reflection(search_backend=search_backend)
                    reflected_count = len(getattr(reflection, "reflected_objects", ()))
                    created_summary_count = len(getattr(reflection, "created_summaries", ()))
                except Exception as exc:
                    first_aux_exception = first_aux_exception or exc
                    aux_errors.append(
                        "reflection:"
                        + (
                            "remote_unavailable"
                            if isinstance(exc, LongTermRemoteUnavailableError)
                            else (compact_text(exc, max_len=80) or exc.__class__.__name__)
                        )
                    )
                    _LOGGER.warning(
                        "Nightly reflection degraded while preparing reserve day %s.",
                        target_day_text,
                        exc_info=True,
                    )

                try:
                    learning_profile = self.learning_builder_factory.from_config(config).build(
                        now=effective_now
                    )
                except Exception as exc:
                    first_aux_exception = first_aux_exception or exc
                    aux_errors.append(
                        f"learning:{compact_text(exc, max_len=80) or exc.__class__.__name__}"
                    )
                    _LOGGER.warning(
                        "Nightly learning-profile build degraded while preparing reserve day %s.",
                        target_day_text,
                        exc_info=True,
                    )

                try:
                    outcome_summary = self._build_outcome_summary(
                        config=config,
                        previous_state=existing_state,
                        local_now=effective_now,
                    )
                except Exception as exc:
                    first_aux_exception = first_aux_exception or exc
                    aux_errors.append(
                        f"outcomes:{compact_text(exc, max_len=80) or exc.__class__.__name__}"
                    )
                    _LOGGER.warning(
                        "Nightly outcome summary degraded while preparing reserve day %s.",
                        target_day_text,
                        exc_info=True,
                    )

                # BREAKING: nightly auxiliary-step failures no longer abort next-day plan
                # preparation by default. Set
                # display_reserve_bus_fail_closed_on_nightly_aux_error=True to restore the
                # previous fail-closed behavior.
                if (
                    aux_errors
                    and bool(
                        getattr(
                            config,
                            "display_reserve_bus_fail_closed_on_nightly_aux_error",
                            False,
                        )
                    )
                    and first_aux_exception is not None
                ):
                    raise first_aux_exception

                plan = self.day_planner.build_plan_for_day(
                    config=config,
                    local_day=target_day,
                    local_now=effective_now,
                )
                prepared_plan = DisplayReserveDayPlan(
                    local_day=target_day_text,
                    generated_at=plan.generated_at,
                    cursor=0,
                    items=plan.items,
                    candidate_count=plan.candidate_count,
                    retired_topic_keys=plan.retired_topic_keys,
                )
                self.prepared_store.save(prepared_plan)
                self._harden_store_file(self.prepared_store)

                state = DisplayReserveNightlyMaintenanceState(
                    prepared_local_day=target_day_text,
                    last_attempted_at=format_timestamp(attempted_at),
                    last_completed_at=format_timestamp(utc_now()),
                    last_status="prepared_degraded" if aux_errors else "prepared",
                    last_error=_join_error_messages(aux_errors),
                    reflection_reflected_object_count=reflected_count,
                    reflection_created_summary_count=created_summary_count,
                    prepared_candidate_count=prepared_plan.candidate_count,
                    prepared_item_count=len(prepared_plan.items),
                    positive_topics=(
                        self._positive_topics_from_profile(learning_profile)
                        if learning_profile is not None
                        else outcome_summary.positive_topics
                    ),
                    cooling_topics=(
                        self._cooling_topics_from_profile(learning_profile)
                        if learning_profile is not None
                        else outcome_summary.cooling_topics
                    ),
                    outcome_summary=outcome_summary,
                )
                saved_state = self._save_state_best_effort(state)

                return DisplayReserveNightlyMaintenanceResult(
                    action="prepared",
                    reason=(
                        "prepared_next_day_degraded"
                        if aux_errors
                        else "prepared_next_day"
                    ),
                    target_local_day=target_day_text,
                    plan=prepared_plan,
                    state=saved_state,
                )
            except Exception as exc:
                self._save_failed_state_best_effort(
                    target_day_text=target_day_text,
                    attempted_at=attempted_at,
                    previous_state=existing_state,
                    error=exc,
                )
                raise

    def _nightly_target_local_day(
        self,
        *,
        config: TwinrConfig,
        local_now: datetime,
    ) -> LocalDate | None:
        """Return the local day that should be prepared now, if any."""

        nightly_after = _parse_local_time(
            getattr(config, "display_reserve_bus_nightly_after_local", _DEFAULT_NIGHTLY_AFTER_LOCAL),
            fallback=_DEFAULT_NIGHTLY_AFTER_LOCAL,
        )
        refresh_after = _parse_local_time(
            getattr(config, "display_reserve_bus_refresh_after_local", "05:30"),
            fallback="05:30",
        )
        current_time = local_now.timetz().replace(tzinfo=None)
        if nightly_after < refresh_after:
            if not (nightly_after <= current_time < refresh_after):
                return None
            return local_now.date()
        if current_time >= nightly_after:
            return local_now.date() + timedelta(days=1)
        if current_time < refresh_after:
            return local_now.date()
        return None

    def _adopt_prepared_plan_if_due(
        self,
        *,
        config: TwinrConfig,
        local_now: datetime,
    ) -> DisplayReserveDayPlan | None:
        """Adopt a prepared next-day plan once the current day should start."""

        refresh_after = _parse_local_time(
            getattr(config, "display_reserve_bus_refresh_after_local", "05:30"),
            fallback="05:30",
        )
        current_time = local_now.timetz().replace(tzinfo=None)
        if current_time < refresh_after:
            return None

        prepared = self._safe_load_plan(self.prepared_store, label="prepared")
        if prepared is None:
            return None
        current_day_text = local_now.date().isoformat()
        prepared_day = _coerce_iso_day(getattr(prepared, "local_day", None))
        if prepared_day != current_day_text:
            return None

        existing = self._safe_load_plan(self.store, label="active")
        if existing is not None and _coerce_iso_day(getattr(existing, "local_day", None)) == current_day_text:
            if self._should_replace_existing_plan(existing=existing, prepared=prepared):
                replacement = DisplayReserveDayPlan(
                    local_day=current_day_text,
                    generated_at=prepared.generated_at,
                    cursor=0,
                    items=prepared.items,
                    candidate_count=prepared.candidate_count,
                    retired_topic_keys=prepared.retired_topic_keys,
                )
                saved = self.store.save(replacement)
                self._harden_store_file(self.store)
                self._safe_clear_store(self.prepared_store, label="prepared")
                return saved
            self._safe_clear_store(self.prepared_store, label="prepared")
            return existing

        adopted = DisplayReserveDayPlan(
            local_day=current_day_text,
            generated_at=prepared.generated_at,
            cursor=0,
            items=prepared.items,
            candidate_count=prepared.candidate_count,
            retired_topic_keys=prepared.retired_topic_keys,
        )
        saved = self.store.save(adopted)
        self._harden_store_file(self.store)
        self._safe_clear_store(self.prepared_store, label="prepared")
        return saved

    def _clear_stale_prepared_plan(self, *, current_day: LocalDate) -> None:
        """Drop prepared plans that missed their day and can no longer be used."""

        prepared = self._safe_load_plan(self.prepared_store, label="prepared")
        if prepared is None:
            return
        prepared_day = _coerce_iso_day(getattr(prepared, "local_day", None))
        if prepared_day is None or prepared_day < current_day.isoformat():
            self._safe_clear_store(self.prepared_store, label="prepared")

    def _build_outcome_summary(
        self,
        *,
        config: TwinrConfig,
        previous_state: DisplayReserveNightlyMaintenanceState | None,
        local_now: datetime,
    ) -> DisplayReserveNightlyOutcomeSummary:
        """Summarize recent shown-card outcomes for the nightly review."""

        if previous_state is not None and previous_state.last_completed_at:
            window_start = _parse_timestamp(previous_state.last_completed_at) or (
                local_now.astimezone(timezone.utc) - timedelta(days=1)
            )
        else:
            window_start = local_now.astimezone(timezone.utc) - timedelta(
                days=float(
                    getattr(
                        config,
                        "display_reserve_bus_outcome_lookback_days",
                        _DEFAULT_OUTCOME_LOOKBACK_DAYS,
                    )
                )
            )
        window_end = local_now.astimezone(timezone.utc) + timedelta(minutes=5)

        positive_topics = Counter[str]()
        cooling_topics = Counter[str]()
        engaged_count = 0
        immediate_pickup_count = 0
        cooling_count = 0
        avoided_count = 0
        ignored_count = 0
        pending_count = 0
        exposure_count = 0

        for exposure in self.history_store.load():
            try:
                shown_at = exposure.shown_at_datetime()
                if shown_at.tzinfo is None:
                    shown_at = shown_at.replace(tzinfo=timezone.utc)
            except Exception:
                _LOGGER.warning(
                    "Skipping reserve exposure with invalid shown_at timestamp during nightly review.",
                    exc_info=True,
                )
                continue
            if shown_at < window_start or shown_at > window_end:
                continue

            exposure_count += 1
            topic = _topic_key(getattr(exposure, "topic_key", None))
            status = compact_text(getattr(exposure, "response_status", None), max_len=24).casefold()
            response_mode = compact_text(getattr(exposure, "response_mode", None), max_len=48).casefold()

            if status == "engaged":
                engaged_count += 1
                if topic:
                    positive_topics[topic] += 1
                if response_mode == "voice_immediate_pickup":
                    immediate_pickup_count += 1
            elif status == "cooled":
                cooling_count += 1
                if topic:
                    cooling_topics[topic] += 1
            elif status == "avoided":
                avoided_count += 1
                if topic:
                    cooling_topics[topic] += 2
            elif status == "ignored":
                ignored_count += 1
                if topic:
                    cooling_topics[topic] += 1
            elif status == "pending":
                pending_count += 1

        return DisplayReserveNightlyOutcomeSummary(
            exposure_count=exposure_count,
            engaged_count=engaged_count,
            immediate_pickup_count=immediate_pickup_count,
            cooling_count=cooling_count,
            avoided_count=avoided_count,
            ignored_count=ignored_count,
            pending_count=pending_count,
            positive_topics=tuple(
                topic for topic, _count in positive_topics.most_common(_MAX_TOPIC_COUNT)
            ),
            cooling_topics=tuple(
                topic for topic, _count in cooling_topics.most_common(_MAX_TOPIC_COUNT)
            ),
        )

    def _positive_topics_from_profile(
        self,
        profile: DisplayReserveLearningProfile,
    ) -> tuple[str, ...]:
        """Return the strongest pulling topics from the long-horizon profile."""

        ranked = sorted(
            profile.topics.values(),
            key=lambda signal: (
                signal.normalized_score,
                signal.immediate_pickup_weight,
                signal.positive_weight,
            ),
            reverse=True,
        )
        selected: list[str] = []
        for signal in ranked:
            if signal.normalized_score <= 0.0 and signal.immediate_pickup_weight <= 0.0:
                continue
            selected.append(signal.key)
            if len(selected) >= _MAX_TOPIC_COUNT:
                break
        return tuple(selected)

    def _cooling_topics_from_profile(
        self,
        profile: DisplayReserveLearningProfile,
    ) -> tuple[str, ...]:
        """Return the strongest cooling topics from the long-horizon profile."""

        ranked = sorted(
            profile.topics.values(),
            key=lambda signal: (
                -signal.normalized_score,
                signal.negative_weight,
                signal.repetition_pressure,
            ),
            reverse=True,
        )
        selected: list[str] = []
        for signal in ranked:
            if signal.normalized_score >= 0.0 and signal.negative_weight <= 0.0:
                continue
            selected.append(signal.key)
            if len(selected) >= _MAX_TOPIC_COUNT:
                break
        return tuple(selected)

    def _effective_local_now(
        self,
        *,
        config: TwinrConfig,
        local_now: datetime | None,
    ) -> datetime:
        """Return one aware local datetime in the configured timezone."""

        return _coerce_local_datetime(config, local_now or self.local_now())

    @contextmanager
    def _maintenance_lock(self, *, blocking: bool) -> Iterator[bool]:
        """Hold one best-effort process lock around nightly maintenance."""

        with _advisory_file_lock(
            self.maintenance_lock_path
            or self.state_store.path.with_suffix(self.state_store.path.suffix + ".lock"),
            blocking=blocking,
        ) as acquired:
            yield acquired

    def _existing_plan_for_local_day(self, target_day_text: str) -> DisplayReserveDayPlan | None:
        """Return one already prepared/active plan for the target local day."""

        for label, store in (("prepared", self.prepared_store), ("active", self.store)):
            plan = self._safe_load_plan(store, label=label)
            if plan is None:
                continue
            if _coerce_iso_day(getattr(plan, "local_day", None)) == target_day_text:
                return plan
        return None

    def _state_for_existing_plan(
        self,
        *,
        target_day_text: str,
        plan: DisplayReserveDayPlan,
        previous_state: DisplayReserveNightlyMaintenanceState | None,
    ) -> DisplayReserveNightlyMaintenanceState:
        """Return one best-effort state when a target-day plan already exists."""

        if (
            previous_state is not None
            and previous_state.prepared_local_day == target_day_text
            and previous_state.last_status in {"prepared", "prepared_degraded"}
        ):
            return previous_state
        return DisplayReserveNightlyMaintenanceState(
            prepared_local_day=target_day_text,
            last_attempted_at=previous_state.last_attempted_at if previous_state else None,
            last_completed_at=previous_state.last_completed_at if previous_state else None,
            last_status="prepared",
            last_error=previous_state.last_error if previous_state else None,
            reflection_reflected_object_count=(
                previous_state.reflection_reflected_object_count if previous_state else 0
            ),
            reflection_created_summary_count=(
                previous_state.reflection_created_summary_count if previous_state else 0
            ),
            prepared_candidate_count=plan.candidate_count,
            prepared_item_count=len(plan.items),
            positive_topics=previous_state.positive_topics if previous_state else (),
            cooling_topics=previous_state.cooling_topics if previous_state else (),
            outcome_summary=(
                previous_state.outcome_summary
                if previous_state is not None
                else DisplayReserveNightlyOutcomeSummary()
            ),
        )

    def _save_state_best_effort(
        self,
        state: DisplayReserveNightlyMaintenanceState,
    ) -> DisplayReserveNightlyMaintenanceState | None:
        """Persist one maintenance state without turning success into failure."""

        try:
            return self.state_store.save(state)
        except Exception:
            _LOGGER.warning(
                "Prepared reserve nightly state but failed to persist maintenance metadata to %s.",
                self.state_store.path,
                exc_info=True,
            )
            return None

    def _save_failed_state_best_effort(
        self,
        *,
        target_day_text: str,
        attempted_at: datetime,
        previous_state: DisplayReserveNightlyMaintenanceState | None,
        error: Exception,
    ) -> None:
        """Persist one failure record without masking the original exception."""

        failed_state = DisplayReserveNightlyMaintenanceState(
            prepared_local_day=target_day_text,
            last_attempted_at=format_timestamp(attempted_at),
            last_completed_at=previous_state.last_completed_at if previous_state is not None else None,
            last_status="failed",
            last_error=compact_text(error, max_len=240) or error.__class__.__name__,
            reflection_reflected_object_count=0,
            reflection_created_summary_count=0,
            prepared_candidate_count=0,
            prepared_item_count=0,
            positive_topics=previous_state.positive_topics if previous_state is not None else (),
            cooling_topics=previous_state.cooling_topics if previous_state is not None else (),
            outcome_summary=(
                previous_state.outcome_summary
                if previous_state is not None
                else DisplayReserveNightlyOutcomeSummary()
            ),
        )
        try:
            self.state_store.save(failed_state)
        except Exception:
            _LOGGER.warning(
                "Failed to persist reserve nightly failure state to %s.",
                self.state_store.path,
                exc_info=True,
            )

    def _safe_load_plan(
        self,
        store: DisplayReserveDayPlanStore,
        *,
        label: str,
    ) -> DisplayReserveDayPlan | None:
        """Load one plan store without letting corrupt artifacts break runtime flow."""

        try:
            return store.load()
        except Exception:
            _LOGGER.warning(
                "Failed to load %s display reserve plan from %s.",
                label,
                getattr(store, "path", "<unknown>"),
                exc_info=True,
            )
            return None

    def _safe_clear_store(self, store: DisplayReserveDayPlanStore, *, label: str) -> None:
        """Clear one plan store best-effort."""

        try:
            store.clear()
        except Exception:
            _LOGGER.warning(
                "Failed to clear %s display reserve plan at %s.",
                label,
                getattr(store, "path", "<unknown>"),
                exc_info=True,
            )

    def _harden_store_file(self, store_or_path: object) -> None:
        """Best-effort harden one plan/store artifact to private permissions."""

        path = store_or_path if isinstance(store_or_path, Path) else getattr(store_or_path, "path", None)
        if isinstance(path, Path) and path.exists():
            _set_private_permissions(path)

    def _should_replace_existing_plan(
        self,
        *,
        existing: DisplayReserveDayPlan,
        prepared: DisplayReserveDayPlan,
    ) -> bool:
        """Return whether a pristine live-built plan should be swapped for prepared."""

        if getattr(existing, "cursor", 0) not in {0, None}:
            return False
        if not getattr(prepared, "items", ()):
            return False
        try:
            existing_ts = _parse_timestamp(getattr(existing, "generated_at", None))
            prepared_ts = _parse_timestamp(getattr(prepared, "generated_at", None))
        except Exception:
            existing_ts = None
            prepared_ts = None
        if existing_ts is None or prepared_ts is None:
            return prepared.candidate_count > existing.candidate_count
        return prepared_ts >= existing_ts


__all__ = [
    "DisplayReserveCompanionPlanner",
    "DisplayReserveNightlyMaintenanceResult",
    "DisplayReserveNightlyMaintenanceState",
    "DisplayReserveNightlyMaintenanceStateStore",
    "DisplayReserveNightlyOutcomeSummary",
]