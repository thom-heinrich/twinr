"""Coordinate Twinr's explicit overnight consolidation and morning prep.

The realtime loop already owns lightweight idle/background helpers such as
reminders, automations, and long-term proactive prompts. This module adds one
canonical overnight orchestration pass that stays off the hot runtime path and
prepares the next local day without speaking or printing during the night.

The nightly pass is intentionally fail-closed for required remote memory:

- if required remote readiness is not currently attested, the run is blocked
- if a remote-memory operation fails mid-run, the caller must escalate Twinr
  into its required-remote error state

For non-critical stages such as weather/news augmentation or digest phrasing,
the run degrades visibly instead of inventing a silent workaround. The durable
artifacts make the overnight outcome auditable by operators and reusable by the
morning-facing runtime surfaces.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, replace
from datetime import date as LocalDate, datetime, tzinfo
import errno
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Protocol, cast

try:
    import fcntl as fcntl
except ImportError:  # pragma: no cover - only relevant on non-Unix platforms
    fcntl = cast(Any, None)

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.contracts import AgentTextProvider
from twinr.agent.personality.evolution import PersonalityEvolutionResult
from twinr.agent.personality.intelligence.models import SituationalAwarenessThread, WorldIntelligenceRefreshResult
from twinr.agent.personality.learning import PersonalityHistoryProbe
from twinr.agent.workflows.realtime_runtime.nightly_digest_refinement import (
    NightlyDigestRefinement,
    build_nightly_digest_refinement,
)
from twinr.agent.workflows.realtime_runtime.nightly_insights import (
    NightlyInsightBundle,
    build_nightly_insight_bundle,
)
from twinr.memory.longterm.core.models import LongTermReflectionResultV1
from twinr.memory.longterm.runtime.service import LongTermMemoryService, _writer_state_details
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError
from twinr.memory.reminders import ReminderEntry
from twinr.ops.remote_memory_watchdog_state import RemoteMemoryWatchdogStore
from twinr.proactive.runtime.display_reserve_companion_planner import (
    DisplayReserveCompanionPlanner,
    DisplayReserveNightlyMaintenanceResult,
)
from twinr.proactive.runtime.display_reserve_support import compact_text, format_timestamp, parse_local_time, utc_now

_DEFAULT_AFTER_LOCAL = "00:30"
_DEFAULT_POLL_INTERVAL_S = 300.0
_DEFAULT_FLUSH_TIMEOUT_S = 15.0
_DEFAULT_REMINDER_LIMIT = 6
_DEFAULT_HEADLINE_LIMIT = 5
_DEFAULT_LIVE_WEB_QUERY_LIMIT = 2
_DEFAULT_STATE_PATH = "artifacts/stores/ops/nightly_run_state.json"
_DEFAULT_DIGEST_PATH = "artifacts/stores/ops/nightly_prepared_digest.json"
_DEFAULT_SUMMARY_PATH = "artifacts/stores/ops/nightly_consolidation_summary.json"
_DEFAULT_PERSONALITY_STATUS_PATH = "artifacts/stores/ops/nightly_personality_status.json"
_DEFAULT_LOCK_SUFFIX = ".lock"
_MAX_TEXT = 4000
_MAX_LINE = 220
_MAX_ERROR = 240
_PRIVATE_FILE_MODE = 0o600

_LOGGER = logging.getLogger(__name__)


class _PersonalityLearningLike(Protocol):
    """Protocol for the personality-learning methods used overnight."""

    def flush_pending(self) -> PersonalityEvolutionResult | None:
        """Commit any queued learning signals."""

    def maybe_refresh_world_intelligence(
        self,
        *,
        force: bool = False,
        search_backend: object | None = None,
    ) -> WorldIntelligenceRefreshResult | None:
        """Refresh due world/place intelligence and commit it when trusted."""

    def probe_history_status(self) -> PersonalityHistoryProbe:
        """Return one authoritative read-status probe for persisted personality history."""


class _LongTermMemoryLike(Protocol):
    """Protocol for the runtime long-term-memory service used overnight."""

    personality_learning: _PersonalityLearningLike | None

    def flush(self, *, timeout_s: float = 2.0) -> bool:
        """Flush active long-term background writers."""

    def run_reflection(
        self,
        *,
        search_backend: object | None = None,
    ) -> LongTermReflectionResultV1:
        """Run one bounded reflection/consolidation pass."""


class _ReminderStoreLike(Protocol):
    """Protocol for reminder-store access used by the digest builder."""

    def load_entries(self) -> tuple[ReminderEntry, ...]:
        """Return the currently persisted reminders."""


class _NightlyRuntimeLike(Protocol):
    """Protocol for the subset of runtime behavior needed overnight."""

    long_term_memory: _LongTermMemoryLike
    reminder_store: _ReminderStoreLike

    def peek_due_reminders(self, *, limit: int = 1) -> tuple[ReminderEntry, ...]:
        """Return due reminders without reserving them."""


def _resolve_store_path(config: TwinrConfig, attr_name: str, default_path: str) -> Path:
    """Resolve one configured artifact path under the current project root."""

    project_root = Path(config.project_root).expanduser().resolve()
    configured = Path(getattr(config, attr_name, default_path) or default_path)
    return configured if configured.is_absolute() else project_root / configured


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
    """Write one UTF-8 file atomically enough for Pi-class edge runtimes."""

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
    except Exception:
        with suppress(OSError):
            tmp_path.unlink()
        raise


@contextmanager
def _advisory_file_lock(path: Path, *, blocking: bool) -> Iterator[bool]:
    """Acquire one best-effort cross-process lock around nightly work."""

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


def _bounded_text(value: object | None, *, max_len: int) -> str:
    """Collapse one value into bounded single-line text."""

    return compact_text(value, max_len=max_len) if value is not None else ""


def _bounded_block_text(value: object | None, *, max_len: int) -> str:
    """Return one bounded text block while preserving intentional newlines."""

    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    if max_len <= 1:
        return text[:max_len]
    trimmed = text[: max_len - 1].rstrip()
    return f"{trimmed}…"


def _bounded_optional_text(value: object | None, *, max_len: int) -> str | None:
    """Return one bounded text value or None when empty."""

    text = _bounded_text(value, max_len=max_len)
    return text or None


def _coerce_nonnegative_int(value: object, *, default: int = 0) -> int:
    """Normalize one optional payload value into a bounded non-negative integer."""

    try:
        result = int(cast(Any, value))
    except (TypeError, ValueError):
        return max(0, default)
    return max(0, result)


def _tuple_from_json_list(payload: Mapping[str, object], key: str) -> tuple[object, ...]:
    """Return one tuple only when the stored payload value is a JSON list."""

    raw = payload.get(key)
    return tuple(raw) if isinstance(raw, list) else ()


def _tuple_from_value(value: object) -> tuple[object, ...]:
    """Return one tuple for generic non-string sequences and tuples."""

    if isinstance(value, (str, bytes, bytearray)):
        return ()
    if isinstance(value, Sequence):
        return tuple(value)
    return ()


def _coerce_tzinfo(value: object | None) -> tzinfo | None:
    """Return one tzinfo object when the input already satisfies that contract."""

    return value if isinstance(value, tzinfo) else None


def _coerce_config_int(config: TwinrConfig, attr_name: str, default: int, *, minimum: int = 0) -> int:
    """Read one integer-like config field and clamp it to a lower bound."""

    raw = getattr(config, attr_name, default) or default
    return max(minimum, _coerce_nonnegative_int(raw, default=default))


def _bounded_text_tuple(values: Sequence[object], *, max_items: int, max_len: int) -> tuple[str, ...]:
    """Normalize one tuple of bounded, deduplicated text items."""

    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _bounded_text(value, max_len=max_len)
        if not text or text.casefold() in seen:
            continue
        seen.add(text.casefold())
        normalized.append(text)
        if len(normalized) >= max_items:
            break
    return tuple(normalized)


def _exception_summary(exc: Exception) -> str:
    """Return one bounded public exception summary."""

    return _bounded_text(f"{type(exc).__name__}: {exc}", max_len=_MAX_ERROR) or type(exc).__name__


def _join_errors(errors: Sequence[str]) -> str | None:
    """Compact one list of degraded stage errors for persisted state."""

    return _bounded_optional_text("; ".join(part for part in errors if part), max_len=_MAX_ERROR)


def _merge_ranked_status(
    current: str,
    candidate: str,
    *,
    priorities: Mapping[str, int],
) -> str:
    """Keep the higher-priority bounded status token."""

    current_rank = priorities.get(current, -1)
    candidate_rank = priorities.get(candidate, -1)
    return candidate if candidate_rank >= current_rank else current


@dataclass(frozen=True, slots=True)
class NightlyPreparedDigest:
    """Persist one prepared morning digest without triggering nighttime output."""

    schema_version: int = 1
    target_local_day: str = ""
    prepared_at: str = ""
    language: str | None = None
    spoken_text: str = ""
    print_text: str = ""
    weather_summary: str | None = None
    reminder_lines: tuple[str, ...] = ()
    headline_lines: tuple[str, ...] = ()
    new_insights: tuple[str, ...] = ()
    continuity_shifts: tuple[str, ...] = ()
    live_news_summary: str | None = None
    weather_sources: tuple[str, ...] = ()
    news_sources: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "NightlyPreparedDigest":
        """Build one prepared digest from persisted JSON-safe data."""

        return cls(
            schema_version=max(1, _coerce_nonnegative_int(payload.get("schema_version", 1), default=1)),
            target_local_day=_bounded_text(payload.get("target_local_day"), max_len=16),
            prepared_at=_bounded_text(payload.get("prepared_at"), max_len=64),
            language=_bounded_optional_text(payload.get("language"), max_len=24),
            spoken_text=_bounded_block_text(payload.get("spoken_text"), max_len=_MAX_TEXT),
            print_text=_bounded_block_text(payload.get("print_text"), max_len=_MAX_TEXT),
            weather_summary=_bounded_optional_text(payload.get("weather_summary"), max_len=600),
            reminder_lines=_bounded_text_tuple(
                _tuple_from_json_list(payload, "reminder_lines"),
                max_items=_DEFAULT_REMINDER_LIMIT,
                max_len=_MAX_LINE,
            ),
            headline_lines=_bounded_text_tuple(
                _tuple_from_json_list(payload, "headline_lines"),
                max_items=_DEFAULT_HEADLINE_LIMIT,
                max_len=_MAX_LINE,
            ),
            new_insights=_bounded_text_tuple(
                _tuple_from_json_list(payload, "new_insights"),
                max_items=_DEFAULT_HEADLINE_LIMIT,
                max_len=_MAX_LINE,
            ),
            continuity_shifts=_bounded_text_tuple(
                _tuple_from_json_list(payload, "continuity_shifts"),
                max_items=_DEFAULT_HEADLINE_LIMIT,
                max_len=_MAX_LINE,
            ),
            live_news_summary=_bounded_optional_text(payload.get("live_news_summary"), max_len=1200),
            weather_sources=_bounded_text_tuple(
                _tuple_from_json_list(payload, "weather_sources"),
                max_items=12,
                max_len=512,
            ),
            news_sources=_bounded_text_tuple(
                _tuple_from_json_list(payload, "news_sources"),
                max_items=12,
                max_len=512,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize one prepared digest into JSON-safe data."""

        payload = asdict(self)
        payload["reminder_lines"] = list(self.reminder_lines)
        payload["headline_lines"] = list(self.headline_lines)
        payload["new_insights"] = list(self.new_insights)
        payload["continuity_shifts"] = list(self.continuity_shifts)
        payload["weather_sources"] = list(self.weather_sources)
        payload["news_sources"] = list(self.news_sources)
        return payload


@dataclass(frozen=True, slots=True)
class NightlyConsolidationSummary:
    """Persist one bounded summary of the overnight consolidation stages."""

    schema_version: int = 3
    target_local_day: str = ""
    prepared_at: str = ""
    long_term_flush_ok: bool = True
    failure_stage: str | None = None
    reflection_reflected_object_count: int = 0
    reflection_created_summary_count: int = 0
    world_refresh_status: str = "unknown"
    world_refresh_refreshed: bool = False
    world_awareness_thread_count: int = 0
    due_reminder_count: int = 0
    target_day_reminder_count: int = 0
    accepted_personality_delta_count: int = 0
    personality_flush_status: str = "not_run"
    personality_history_read_status: str = "not_run"
    last_personality_commit_at: str | None = None
    personality_history_error: str | None = None
    new_insights: tuple[str, ...] = ()
    continuity_shifts: tuple[str, ...] = ()
    operator_new_insights: tuple[str, ...] = ()
    operator_continuity_shifts: tuple[str, ...] = ()
    live_search_queries: int = 0
    weather_sources: tuple[str, ...] = ()
    news_sources: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "NightlyConsolidationSummary":
        """Build one consolidation summary from persisted JSON-safe data."""

        return cls(
            schema_version=max(1, _coerce_nonnegative_int(payload.get("schema_version", 1), default=1)),
            target_local_day=_bounded_text(payload.get("target_local_day"), max_len=16),
            prepared_at=_bounded_text(payload.get("prepared_at"), max_len=64),
            long_term_flush_ok=bool(payload.get("long_term_flush_ok", True)),
            failure_stage=_bounded_optional_text(payload.get("failure_stage"), max_len=96),
            reflection_reflected_object_count=max(
                0, _coerce_nonnegative_int(payload.get("reflection_reflected_object_count", 0))
            ),
            reflection_created_summary_count=max(
                0, _coerce_nonnegative_int(payload.get("reflection_created_summary_count", 0))
            ),
            world_refresh_status=_bounded_text(payload.get("world_refresh_status"), max_len=32) or "unknown",
            world_refresh_refreshed=bool(payload.get("world_refresh_refreshed", False)),
            world_awareness_thread_count=max(
                0, _coerce_nonnegative_int(payload.get("world_awareness_thread_count", 0))
            ),
            due_reminder_count=max(0, _coerce_nonnegative_int(payload.get("due_reminder_count", 0))),
            target_day_reminder_count=max(
                0, _coerce_nonnegative_int(payload.get("target_day_reminder_count", 0))
            ),
            accepted_personality_delta_count=max(
                0, _coerce_nonnegative_int(payload.get("accepted_personality_delta_count", 0))
            ),
            personality_flush_status=(
                _bounded_text(payload.get("personality_flush_status"), max_len=32).lower() or "not_run"
            ),
            personality_history_read_status=(
                _bounded_text(payload.get("personality_history_read_status"), max_len=32).lower() or "not_run"
            ),
            last_personality_commit_at=_bounded_optional_text(
                payload.get("last_personality_commit_at"),
                max_len=64,
            ),
            personality_history_error=_bounded_optional_text(
                payload.get("personality_history_error"),
                max_len=_MAX_ERROR,
            ),
            new_insights=_bounded_text_tuple(
                _tuple_from_json_list(payload, "new_insights"),
                max_items=_DEFAULT_HEADLINE_LIMIT,
                max_len=_MAX_LINE,
            ),
            continuity_shifts=_bounded_text_tuple(
                _tuple_from_json_list(payload, "continuity_shifts"),
                max_items=_DEFAULT_HEADLINE_LIMIT,
                max_len=_MAX_LINE,
            ),
            operator_new_insights=_bounded_text_tuple(
                _tuple_from_json_list(payload, "operator_new_insights"),
                max_items=_DEFAULT_HEADLINE_LIMIT,
                max_len=_MAX_LINE,
            ),
            operator_continuity_shifts=_bounded_text_tuple(
                _tuple_from_json_list(payload, "operator_continuity_shifts"),
                max_items=_DEFAULT_HEADLINE_LIMIT,
                max_len=_MAX_LINE,
            ),
            live_search_queries=max(0, _coerce_nonnegative_int(payload.get("live_search_queries", 0))),
            weather_sources=_bounded_text_tuple(
                _tuple_from_json_list(payload, "weather_sources"),
                max_items=12,
                max_len=512,
            ),
            news_sources=_bounded_text_tuple(
                _tuple_from_json_list(payload, "news_sources"),
                max_items=12,
                max_len=512,
            ),
            errors=_bounded_text_tuple(
                _tuple_from_json_list(payload, "errors"),
                max_items=32,
                max_len=_MAX_ERROR,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize one consolidation summary into JSON-safe data."""

        payload = asdict(self)
        payload["weather_sources"] = list(self.weather_sources)
        payload["news_sources"] = list(self.news_sources)
        payload["new_insights"] = list(self.new_insights)
        payload["continuity_shifts"] = list(self.continuity_shifts)
        payload["operator_new_insights"] = list(self.operator_new_insights)
        payload["operator_continuity_shifts"] = list(self.operator_continuity_shifts)
        payload["errors"] = list(self.errors)
        return payload


@dataclass(frozen=True, slots=True)
class NightlyRunState:
    """Persist the last explicit overnight orchestration result."""

    schema_version: int = 2
    prepared_local_day: str | None = None
    last_attempted_at: str | None = None
    last_completed_at: str | None = None
    last_status: str = "idle"
    last_error: str | None = None
    failure_stage: str | None = None
    remote_ready: bool = False
    remote_status: str | None = None
    digest_ready: bool = False
    display_reserve_status: str | None = None
    display_reserve_reason: str | None = None
    reflection_reflected_object_count: int = 0
    reflection_created_summary_count: int = 0
    world_refresh_status: str | None = None
    live_search_queries: int = 0
    personality_flush_status: str = "not_run"
    personality_history_read_status: str = "not_run"
    last_personality_commit_at: str | None = None
    personality_history_error: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "NightlyRunState":
        """Build one persisted run-state record from JSON-safe data."""

        prepared_day = _bounded_optional_text(payload.get("prepared_local_day"), max_len=16)
        if prepared_day:
            try:
                prepared_day = LocalDate.fromisoformat(prepared_day).isoformat()
            except ValueError:
                prepared_day = None
        return cls(
            schema_version=max(1, _coerce_nonnegative_int(payload.get("schema_version", 1), default=1)),
            prepared_local_day=prepared_day,
            last_attempted_at=_bounded_optional_text(payload.get("last_attempted_at"), max_len=64),
            last_completed_at=_bounded_optional_text(payload.get("last_completed_at"), max_len=64),
            last_status=_bounded_text(payload.get("last_status"), max_len=32).lower() or "idle",
            last_error=_bounded_optional_text(payload.get("last_error"), max_len=_MAX_ERROR),
            failure_stage=_bounded_optional_text(payload.get("failure_stage"), max_len=96),
            remote_ready=bool(payload.get("remote_ready", False)),
            remote_status=_bounded_optional_text(payload.get("remote_status"), max_len=32),
            digest_ready=bool(payload.get("digest_ready", False)),
            display_reserve_status=_bounded_optional_text(payload.get("display_reserve_status"), max_len=32),
            display_reserve_reason=_bounded_optional_text(payload.get("display_reserve_reason"), max_len=64),
            reflection_reflected_object_count=max(
                0, _coerce_nonnegative_int(payload.get("reflection_reflected_object_count", 0))
            ),
            reflection_created_summary_count=max(
                0, _coerce_nonnegative_int(payload.get("reflection_created_summary_count", 0))
            ),
            world_refresh_status=_bounded_optional_text(payload.get("world_refresh_status"), max_len=32),
            live_search_queries=max(0, _coerce_nonnegative_int(payload.get("live_search_queries", 0))),
            personality_flush_status=(
                _bounded_text(payload.get("personality_flush_status"), max_len=32).lower() or "not_run"
            ),
            personality_history_read_status=(
                _bounded_text(payload.get("personality_history_read_status"), max_len=32).lower() or "not_run"
            ),
            last_personality_commit_at=_bounded_optional_text(
                payload.get("last_personality_commit_at"),
                max_len=64,
            ),
            personality_history_error=_bounded_optional_text(
                payload.get("personality_history_error"),
                max_len=_MAX_ERROR,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize one run-state record into JSON-safe data."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class NightlyPersonalityStatus:
    """Persist one explicit attestation for the nightly personality lane."""

    schema_version: int = 1
    target_local_day: str = ""
    attempted_at: str = ""
    flush_status: str = "not_run"
    history_read_status: str = "not_run"
    accepted_personality_delta_count: int = 0
    failure_stage: str | None = None
    last_personality_commit_at: str | None = None
    error: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "NightlyPersonalityStatus":
        """Build one persisted nightly personality status from JSON-safe data."""

        return cls(
            schema_version=max(1, _coerce_nonnegative_int(payload.get("schema_version", 1), default=1)),
            target_local_day=_bounded_text(payload.get("target_local_day"), max_len=16),
            attempted_at=_bounded_text(payload.get("attempted_at"), max_len=64),
            flush_status=_bounded_text(payload.get("flush_status"), max_len=32).lower() or "not_run",
            history_read_status=_bounded_text(payload.get("history_read_status"), max_len=32).lower()
            or "not_run",
            accepted_personality_delta_count=max(
                0, _coerce_nonnegative_int(payload.get("accepted_personality_delta_count", 0))
            ),
            failure_stage=_bounded_optional_text(payload.get("failure_stage"), max_len=96),
            last_personality_commit_at=_bounded_optional_text(
                payload.get("last_personality_commit_at"),
                max_len=64,
            ),
            error=_bounded_optional_text(payload.get("error"), max_len=_MAX_ERROR),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize one nightly personality status into JSON-safe data."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class NightlyOrchestrationResult:
    """Describe one attempt to run the overnight orchestration pass."""

    action: str
    reason: str
    target_local_day: str | None = None
    state: NightlyRunState | None = None
    digest: NightlyPreparedDigest | None = None
    summary: NightlyConsolidationSummary | None = None
    display_reserve_result: DisplayReserveNightlyMaintenanceResult | None = None


@dataclass(frozen=True, slots=True)
class _LongTermFlushOutcome:
    """Describe the explicit result of one long-term flush attempt."""

    ok: bool
    failure_stage: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _PersonalityFlushOutcome:
    """Describe one personality flush call plus its explicit status."""

    status: str = "not_run"
    result: PersonalityEvolutionResult | None = None
    failure_stage: str | None = None
    error: str | None = None


@dataclass(slots=True)
class NightlyRunStateStore:
    """Persist the last explicit overnight orchestration state."""

    path: Path

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "NightlyRunStateStore":
        """Resolve the nightly run-state artifact path from configuration."""

        return cls(
            path=_resolve_store_path(
                config,
                "nightly_orchestration_state_path",
                _DEFAULT_STATE_PATH,
            )
        )

    def load(self) -> NightlyRunState | None:
        """Load one persisted run-state artifact when it exists and parses."""

        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            _LOGGER.warning("Failed to read nightly run state from %s.", self.path, exc_info=True)
            return None
        if not isinstance(payload, Mapping):
            return None
        try:
            return NightlyRunState.from_dict(payload)
        except Exception:
            _LOGGER.warning("Ignoring invalid nightly run state at %s.", self.path, exc_info=True)
            return None

    def save(self, state: NightlyRunState) -> NightlyRunState:
        """Persist one nightly run-state artifact atomically enough for runtime use."""

        payload = json.dumps(state.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        _atomic_write_text(self.path, payload)
        return state


@dataclass(slots=True)
class NightlyPreparedDigestStore:
    """Persist the prepared morning digest artifact."""

    path: Path

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "NightlyPreparedDigestStore":
        """Resolve the prepared-digest artifact path from configuration."""

        return cls(
            path=_resolve_store_path(
                config,
                "nightly_prepared_digest_path",
                _DEFAULT_DIGEST_PATH,
            )
        )

    def load(self) -> NightlyPreparedDigest | None:
        """Load the prepared digest when it exists and parses."""

        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            _LOGGER.warning("Failed to read nightly prepared digest from %s.", self.path, exc_info=True)
            return None
        if not isinstance(payload, Mapping):
            return None
        try:
            return NightlyPreparedDigest.from_dict(payload)
        except Exception:
            _LOGGER.warning("Ignoring invalid nightly prepared digest at %s.", self.path, exc_info=True)
            return None

    def save(self, digest: NightlyPreparedDigest) -> NightlyPreparedDigest:
        """Persist one prepared digest atomically enough for runtime use."""

        payload = json.dumps(digest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        _atomic_write_text(self.path, payload)
        return digest


@dataclass(slots=True)
class NightlyConsolidationSummaryStore:
    """Persist the bounded nightly consolidation summary artifact."""

    path: Path

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "NightlyConsolidationSummaryStore":
        """Resolve the consolidation-summary artifact path from configuration."""

        return cls(
            path=_resolve_store_path(
                config,
                "nightly_consolidation_summary_path",
                _DEFAULT_SUMMARY_PATH,
            )
        )

    def load(self) -> NightlyConsolidationSummary | None:
        """Load the persisted summary when it exists and parses."""

        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            _LOGGER.warning("Failed to read nightly consolidation summary from %s.", self.path, exc_info=True)
            return None
        if not isinstance(payload, Mapping):
            return None
        try:
            return NightlyConsolidationSummary.from_dict(payload)
        except Exception:
            _LOGGER.warning("Ignoring invalid nightly consolidation summary at %s.", self.path, exc_info=True)
            return None

    def save(self, summary: NightlyConsolidationSummary) -> NightlyConsolidationSummary:
        """Persist one nightly summary atomically enough for runtime use."""

        payload = json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        _atomic_write_text(self.path, payload)
        return summary


@dataclass(slots=True)
class NightlyPersonalityStatusStore:
    """Persist one bounded nightly personality-attestation artifact."""

    path: Path

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "NightlyPersonalityStatusStore":
        """Resolve the nightly personality-status artifact path from configuration."""

        return cls(
            path=_resolve_store_path(
                config,
                "nightly_personality_status_path",
                _DEFAULT_PERSONALITY_STATUS_PATH,
            )
        )

    def load(self) -> NightlyPersonalityStatus | None:
        """Load the persisted nightly personality status when it exists and parses."""

        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            _LOGGER.warning("Failed to read nightly personality status from %s.", self.path, exc_info=True)
            return None
        if not isinstance(payload, Mapping):
            return None
        try:
            return NightlyPersonalityStatus.from_dict(payload)
        except Exception:
            _LOGGER.warning("Ignoring invalid nightly personality status at %s.", self.path, exc_info=True)
            return None

    def save(self, status: NightlyPersonalityStatus) -> NightlyPersonalityStatus:
        """Persist one nightly personality status atomically enough for runtime use."""

        payload = json.dumps(status.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        _atomic_write_text(self.path, payload)
        return status


@dataclass(slots=True)
class _PrecomputedReflectionMemoryService:
    """Adapt a precomputed reflection result to the reserve-planner contract."""

    reflection: LongTermReflectionResultV1

    def run_reflection(self, *, search_backend: object | None = None) -> LongTermReflectionResultV1:
        """Return the already computed reflection result."""

        del search_backend
        return self.reflection


@dataclass(slots=True)
class TwinrNightlyOrchestrator:
    """Run one explicit overnight consolidation and preparation pass."""

    config: TwinrConfig
    runtime: _NightlyRuntimeLike
    text_backend: AgentTextProvider | None = None
    search_backend: AgentTextProvider | None = None
    print_backend: AgentTextProvider | None = None
    display_planner: DisplayReserveCompanionPlanner | None = None
    remote_watchdog_store: RemoteMemoryWatchdogStore | None = None
    remote_ready: Callable[[], bool] = lambda: True
    background_allowed: Callable[[], bool] = lambda: True
    state_store: NightlyRunStateStore | None = None
    digest_store: NightlyPreparedDigestStore | None = None
    summary_store: NightlyConsolidationSummaryStore | None = None
    personality_status_store: NightlyPersonalityStatusStore | None = None
    local_now: Callable[[], datetime] = datetime.now

    def __post_init__(self) -> None:
        """Resolve default collaborators from configuration when omitted."""

        if self.state_store is None:
            self.state_store = NightlyRunStateStore.from_config(self.config)
        if self.digest_store is None:
            self.digest_store = NightlyPreparedDigestStore.from_config(self.config)
        if self.summary_store is None:
            self.summary_store = NightlyConsolidationSummaryStore.from_config(self.config)
        if self.personality_status_store is None:
            self.personality_status_store = NightlyPersonalityStatusStore.from_config(self.config)
        if self.display_planner is None:
            self.display_planner = DisplayReserveCompanionPlanner.from_config(self.config)
        if self.remote_watchdog_store is None:
            self.remote_watchdog_store = RemoteMemoryWatchdogStore.from_config(self.config)

    def maybe_run(
        self,
        *,
        local_now: datetime | None = None,
    ) -> NightlyOrchestrationResult:
        """Run the overnight pipeline once for the current local day when due."""

        state_store = self._require_state_store()
        digest_store = self._require_digest_store()
        summary_store = self._require_summary_store()
        personality_status_store = self._require_personality_status_store()
        effective_now = self.local_now() if local_now is None else local_now
        if not bool(getattr(self.config, "nightly_orchestration_enabled", True)):
            return NightlyOrchestrationResult(action="inactive", reason="nightly_disabled")

        target_day = self._target_local_day(effective_now)
        if target_day is None:
            return NightlyOrchestrationResult(action="skipped", reason="not_due")
        target_day_text = target_day.isoformat()

        lock_path = state_store.path.with_suffix(state_store.path.suffix + _DEFAULT_LOCK_SUFFIX)
        with _advisory_file_lock(lock_path, blocking=False) as acquired:
            if not acquired:
                return NightlyOrchestrationResult(
                    action="skipped",
                    reason="locked",
                    target_local_day=target_day_text,
                )

            existing_state = state_store.load()
            if self._already_prepared(existing_state, target_day_text):
                return NightlyOrchestrationResult(
                    action="skipped",
                    reason="already_prepared",
                    target_local_day=target_day_text,
                    state=existing_state,
                )

            attempted_at = format_timestamp(utc_now())
            remote_status = self._remote_status_label()
            personality_status = NightlyPersonalityStatus(
                target_local_day=target_day_text,
                attempted_at=attempted_at,
            )
            if not self.remote_ready():
                personality_status_store.save(
                    replace(
                        personality_status,
                        failure_stage="required_remote_not_ready",
                        error="required_remote_not_ready",
                    )
                )
                state = self._save_state(
                    NightlyRunState(
                        prepared_local_day=None,
                        last_attempted_at=attempted_at,
                        last_completed_at=None,
                        last_status="blocked_remote_not_ready",
                        last_error="required_remote_not_ready",
                        failure_stage="required_remote_not_ready",
                        remote_ready=False,
                        remote_status=remote_status,
                        personality_history_error="required_remote_not_ready",
                    )
                )
                return NightlyOrchestrationResult(
                    action="blocked",
                    reason="remote_not_ready",
                    target_local_day=target_day_text,
                    state=state,
                )

            if not self.background_allowed():
                personality_status_store.save(
                    replace(
                        personality_status,
                        failure_stage="background_not_idle",
                        error="background_not_idle",
                    )
                )
                state = self._save_state(
                    NightlyRunState(
                        prepared_local_day=None,
                        last_attempted_at=attempted_at,
                        last_completed_at=None,
                        last_status="interrupted",
                        last_error="background_not_idle",
                        failure_stage="background_not_idle",
                        remote_ready=True,
                        remote_status=remote_status,
                        personality_history_error="background_not_idle",
                    )
                )
                return NightlyOrchestrationResult(
                    action="skipped",
                    reason="background_not_idle",
                    target_local_day=target_day_text,
                    state=state,
                )

            long_term_memory = self.runtime.long_term_memory
            personality_learning = getattr(long_term_memory, "personality_learning", None)
            errors: list[str] = []
            live_search_queries = 0
            world_refresh_result: WorldIntelligenceRefreshResult | None = None
            reflection_result: LongTermReflectionResultV1 | None = None
            personality_results: list[PersonalityEvolutionResult] = []
            long_term_flush_outcome = _LongTermFlushOutcome(ok=True)
            if personality_learning is None:
                personality_status = replace(
                    personality_status,
                    flush_status="not_configured",
                    history_read_status="not_configured",
                )

            try:
                long_term_flush_outcome = self._flush_long_term_memory(long_term_memory, errors)
                self._ensure_background_allowed(
                    attempted_at=attempted_at,
                    remote_status=remote_status,
                    reason="background_not_idle_after_flush",
                )

                if personality_learning is not None:
                    first_flush = self._flush_personality_learning(
                        personality_learning,
                        results=personality_results,
                        errors=errors,
                    )
                    personality_status = self._merge_personality_status(
                        personality_status,
                        flush_outcome=first_flush,
                    )
                    personality_status = self._merge_personality_status(
                        personality_status,
                        probe=self._probe_personality_history(personality_learning, errors=errors),
                    )
                    self._ensure_background_allowed(
                        attempted_at=attempted_at,
                        remote_status=remote_status,
                        reason="background_not_idle_after_personality_flush",
                    )
                    try:
                        world_refresh_result = personality_learning.maybe_refresh_world_intelligence(
                            search_backend=self.search_backend,
                        )
                    except LongTermRemoteUnavailableError:
                        raise
                    except Exception as exc:
                        errors.append(f"world_refresh:{_exception_summary(exc)}")
                self._ensure_background_allowed(
                    attempted_at=attempted_at,
                    remote_status=remote_status,
                    reason="background_not_idle_after_world_refresh",
                )

                try:
                    reflection_result = long_term_memory.run_reflection(
                        search_backend=self.search_backend,
                    )
                except LongTermRemoteUnavailableError:
                    raise
                except Exception as exc:
                    errors.append(f"reflection:{_exception_summary(exc)}")

                if personality_learning is not None:
                    second_flush = self._flush_personality_learning(
                        personality_learning,
                        results=personality_results,
                        errors=errors,
                    )
                    personality_status = self._merge_personality_status(
                        personality_status,
                        flush_outcome=second_flush,
                    )
                    personality_status = self._merge_personality_status(
                        personality_status,
                        probe=self._probe_personality_history(personality_learning, errors=errors),
                    )
                self._ensure_background_allowed(
                    attempted_at=attempted_at,
                    remote_status=remote_status,
                    reason="background_not_idle_after_reflection",
                )

                digest_inputs = self._build_digest_inputs(
                    target_day=target_day,
                    effective_now=effective_now,
                    world_refresh_result=world_refresh_result,
                )
                if self._should_use_live_web_queries():
                    live_search_queries, weather_summary, weather_sources, live_news_summary, news_sources = (
                        self._augment_digest_inputs(
                            target_day_text=target_day_text,
                            digest_inputs=digest_inputs,
                        )
                    )
                    digest_inputs["weather_summary"] = weather_summary
                    digest_inputs["weather_sources"] = weather_sources
                    digest_inputs["live_news_summary"] = live_news_summary
                    digest_inputs["news_sources"] = news_sources
                nightly_insights = build_nightly_insight_bundle(
                    reflection_result=reflection_result,
                    world_refresh_result=world_refresh_result,
                    personality_results=personality_results,
                )
                digest_refinement = self._refine_digest_inputs(
                    target_day_text=target_day_text,
                    digest_inputs=digest_inputs,
                    nightly_insights=nightly_insights,
                )
                digest_inputs["headline_lines"] = digest_refinement.headline_lines
                digest_inputs["live_news_summary"] = digest_refinement.live_news_summary
                digest_inputs["news_sources"] = digest_refinement.news_sources
                digest_inputs["new_insights"] = digest_refinement.new_insights
                digest_inputs["continuity_shifts"] = digest_refinement.continuity_shifts
                digest = self._prepare_digest(
                    target_day_text=target_day_text,
                    prepared_at=attempted_at,
                    digest_inputs=digest_inputs,
                    errors=errors,
                )
                digest_store.save(digest)
                summary = self._build_summary(
                    target_day_text=target_day_text,
                    prepared_at=attempted_at,
                    long_term_flush_ok=long_term_flush_outcome.ok,
                    failure_stage=(
                        personality_status.failure_stage or long_term_flush_outcome.failure_stage
                    ),
                    reflection_result=reflection_result,
                    world_refresh_result=world_refresh_result,
                    digest=digest,
                    due_reminder_count=_coerce_nonnegative_int(
                        digest_inputs.get("due_reminder_count", 0)
                    ),
                    target_day_reminder_count=_coerce_nonnegative_int(
                        digest_inputs.get("target_day_reminder_count", 0)
                    ),
                    personality_status=personality_status,
                    personality_results=personality_results,
                    nightly_insights=nightly_insights,
                    live_search_queries=live_search_queries,
                    errors=errors,
                )
                summary_store.save(summary)
                personality_status_store.save(
                    replace(
                        personality_status,
                        accepted_personality_delta_count=summary.accepted_personality_delta_count,
                    )
                )

                display_result = self._prepare_display_reserve(
                    effective_now=effective_now,
                    reflection_result=reflection_result,
                    errors=errors,
                )
                completed_at = format_timestamp(utc_now())
                state = self._save_state(
                    NightlyRunState(
                        prepared_local_day=target_day_text,
                        last_attempted_at=attempted_at,
                        last_completed_at=completed_at,
                        last_status="degraded" if errors else "ready",
                        last_error=_join_errors(errors),
                        failure_stage=summary.failure_stage,
                        remote_ready=True,
                        remote_status=remote_status,
                        digest_ready=True,
                        display_reserve_status=(
                            getattr(display_result, "action", None) if display_result is not None else None
                        ),
                        display_reserve_reason=(
                            getattr(display_result, "reason", None) if display_result is not None else None
                        ),
                        reflection_reflected_object_count=len(
                            getattr(reflection_result, "reflected_objects", ())
                        ),
                        reflection_created_summary_count=len(
                            getattr(reflection_result, "created_summaries", ())
                        ),
                        world_refresh_status=(
                            getattr(world_refresh_result, "status", None)
                            if world_refresh_result is not None
                            else None
                        ),
                        live_search_queries=live_search_queries,
                        personality_flush_status=summary.personality_flush_status,
                        personality_history_read_status=summary.personality_history_read_status,
                        last_personality_commit_at=summary.last_personality_commit_at,
                        personality_history_error=summary.personality_history_error,
                    )
                )
                return NightlyOrchestrationResult(
                    action="prepared",
                    reason="prepared_degraded" if errors else "prepared",
                    target_local_day=target_day_text,
                    state=state,
                    digest=digest,
                    summary=summary,
                    display_reserve_result=display_result,
                )
            except LongTermRemoteUnavailableError as exc:
                personality_status_store.save(
                    replace(
                        personality_status,
                        flush_status=_merge_ranked_status(
                            personality_status.flush_status,
                            "remote_unavailable",
                            priorities=self._personality_flush_status_priorities(),
                        ),
                        history_read_status=_merge_ranked_status(
                            personality_status.history_read_status,
                            "remote_unavailable",
                            priorities=self._personality_history_status_priorities(),
                        ),
                        failure_stage=(
                            personality_status.failure_stage
                            or long_term_flush_outcome.failure_stage
                            or "required_remote_failed"
                        ),
                        error=_exception_summary(exc),
                    )
                )
                state = self._save_state(
                    NightlyRunState(
                        prepared_local_day=None,
                        last_attempted_at=attempted_at,
                        last_completed_at=None,
                        last_status="blocked_remote_failed",
                        last_error="required_remote_failed",
                        failure_stage=(
                            personality_status.failure_stage
                            or long_term_flush_outcome.failure_stage
                            or "required_remote_failed"
                        ),
                        remote_ready=False,
                        remote_status=remote_status,
                        digest_ready=False,
                        live_search_queries=live_search_queries,
                        personality_flush_status=personality_status.flush_status,
                        personality_history_read_status=personality_status.history_read_status,
                        last_personality_commit_at=personality_status.last_personality_commit_at,
                        personality_history_error=_exception_summary(exc),
                    )
                )
                raise

    def _target_local_day(self, local_now: datetime) -> LocalDate | None:
        """Return the local day that should be prepared now, if any."""

        cutoff = parse_local_time(
            getattr(self.config, "nightly_orchestration_after_local", _DEFAULT_AFTER_LOCAL),
            fallback=_DEFAULT_AFTER_LOCAL,
        )
        if local_now.timetz().replace(tzinfo=None) < cutoff:
            return None
        return local_now.date()

    def _already_prepared(self, state: NightlyRunState | None, target_day_text: str) -> bool:
        """Return whether the target day already has a complete nightly artifact set."""

        if state is None:
            return False
        if state.prepared_local_day != target_day_text or state.last_status not in {"ready", "degraded"}:
            return False
        digest = self._require_digest_store().load()
        summary = self._require_summary_store().load()
        return bool(
            digest is not None
            and digest.target_local_day == target_day_text
            and summary is not None
            and summary.target_local_day == target_day_text
        )

    def _remote_status_label(self) -> str | None:
        """Return one bounded remote-watchdog status label when available."""

        try:
            snapshot = self._require_remote_watchdog_store().load()
        except Exception:
            return None
        if snapshot is None:
            return None
        return _bounded_optional_text(
            getattr(getattr(snapshot, "current", None), "status", None),
            max_len=32,
        )

    def _save_state(self, state: NightlyRunState) -> NightlyRunState:
        """Persist one run-state artifact and return it."""

        return self._require_state_store().save(state)

    def _require_state_store(self) -> NightlyRunStateStore:
        """Return the configured nightly state store after post-init wiring."""

        if self.state_store is None:
            raise RuntimeError("nightly state store is not configured")
        return self.state_store

    def _require_digest_store(self) -> NightlyPreparedDigestStore:
        """Return the configured prepared-digest store after post-init wiring."""

        if self.digest_store is None:
            raise RuntimeError("nightly digest store is not configured")
        return self.digest_store

    def _require_summary_store(self) -> NightlyConsolidationSummaryStore:
        """Return the configured consolidation-summary store after post-init wiring."""

        if self.summary_store is None:
            raise RuntimeError("nightly summary store is not configured")
        return self.summary_store

    def _require_personality_status_store(self) -> NightlyPersonalityStatusStore:
        """Return the configured nightly personality-status store after post-init wiring."""

        if self.personality_status_store is None:
            raise RuntimeError("nightly personality status store is not configured")
        return self.personality_status_store

    def _require_display_planner(self) -> DisplayReserveCompanionPlanner:
        """Return the configured display planner after post-init wiring."""

        if self.display_planner is None:
            raise RuntimeError("display planner is not configured")
        return self.display_planner

    def _require_remote_watchdog_store(self) -> RemoteMemoryWatchdogStore:
        """Return the configured remote-watchdog store after post-init wiring."""

        if self.remote_watchdog_store is None:
            raise RuntimeError("remote watchdog store is not configured")
        return self.remote_watchdog_store

    def _save_interrupted_state(
        self,
        *,
        attempted_at: str,
        remote_status: str | None,
        reason: str,
    ) -> None:
        """Persist one interrupted state before returning to the idle loop."""

        self._save_state(
            NightlyRunState(
                prepared_local_day=None,
                last_attempted_at=attempted_at,
                last_completed_at=None,
                last_status="interrupted",
                last_error=reason,
                failure_stage=reason,
                remote_ready=True,
                remote_status=remote_status,
                personality_history_error=reason,
            )
        )

    def _ensure_background_allowed(
        self,
        *,
        attempted_at: str,
        remote_status: str | None,
        reason: str,
    ) -> None:
        """Abort the current run cleanly when Twinr stops being idle."""

        if self.background_allowed():
            return
        self._save_interrupted_state(
            attempted_at=attempted_at,
            remote_status=remote_status,
            reason=reason,
        )
        raise RuntimeError(reason)

    def _personality_flush_status_priorities(self) -> dict[str, int]:
        """Return the severity ordering for nightly personality flush statuses."""

        return {
            "not_run": 0,
            "not_configured": 1,
            "no_pending": 2,
            "committed": 3,
            "error": 4,
            "remote_unavailable": 5,
        }

    def _personality_history_status_priorities(self) -> dict[str, int]:
        """Return the severity ordering for nightly personality history statuses."""

        return {
            "not_run": 0,
            "not_configured": 1,
            "missing": 2,
            "loaded": 3,
            "unavailable": 4,
            "error": 5,
            "remote_unavailable": 6,
        }

    def _merge_personality_status(
        self,
        current: NightlyPersonalityStatus,
        *,
        flush_outcome: _PersonalityFlushOutcome | None = None,
        probe: PersonalityHistoryProbe | None = None,
    ) -> NightlyPersonalityStatus:
        """Merge one flush/probe observation into the persisted nightly personality status."""

        merged = current
        if flush_outcome is not None:
            merged = replace(
                merged,
                flush_status=_merge_ranked_status(
                    merged.flush_status,
                    flush_outcome.status,
                    priorities=self._personality_flush_status_priorities(),
                ),
                accepted_personality_delta_count=(
                    merged.accepted_personality_delta_count
                    + len(flush_outcome.result.accepted_deltas)
                    if flush_outcome.result is not None
                    else merged.accepted_personality_delta_count
                ),
            )
            if flush_outcome.failure_stage is not None:
                merged = replace(merged, failure_stage=flush_outcome.failure_stage)
            if flush_outcome.error is not None:
                merged = replace(merged, error=flush_outcome.error)
        if probe is not None:
            merged = replace(
                merged,
                history_read_status=_merge_ranked_status(
                    merged.history_read_status,
                    probe.history_read_status,
                    priorities=self._personality_history_status_priorities(),
                ),
                last_personality_commit_at=probe.last_commit_at or merged.last_personality_commit_at,
            )
            if probe.failure_stage is not None:
                merged = replace(merged, failure_stage=probe.failure_stage)
            if probe.error is not None:
                merged = replace(merged, error=probe.error)
        return merged

    def _flush_long_term_memory(
        self,
        long_term_memory: _LongTermMemoryLike,
        errors: list[str],
    ) -> _LongTermFlushOutcome:
        """Flush the runtime long-term writers within the configured timeout."""

        timeout_s = max(
            1.0,
            float(
                getattr(
                    self.config,
                    "nightly_orchestration_flush_timeout_s",
                    _DEFAULT_FLUSH_TIMEOUT_S,
                )
                or _DEFAULT_FLUSH_TIMEOUT_S
            ),
        )
        try:
            flushed = bool(long_term_memory.flush(timeout_s=timeout_s))
        except LongTermRemoteUnavailableError:
            raise
        except Exception as exc:
            error = _exception_summary(exc)
            errors.append(f"long_term_flush:{error}")
            return _LongTermFlushOutcome(
                ok=False,
                failure_stage="long_term_flush_exception",
                error=error,
            )
        if not flushed:
            failure_stage, failure_detail = self._inspect_long_term_flush_failure(long_term_memory)
            if failure_detail:
                errors.append(f"long_term_flush:{failure_stage}:{failure_detail}")
            else:
                errors.append(f"long_term_flush:{failure_stage}")
            return _LongTermFlushOutcome(
                ok=False,
                failure_stage=failure_stage,
                error=failure_detail,
            )
        return _LongTermFlushOutcome(ok=True)

    def _inspect_long_term_flush_failure(
        self,
        long_term_memory: _LongTermMemoryLike,
    ) -> tuple[str, str | None]:
        """Return one explicit failure stage for an incomplete long-term flush."""

        for label, writer in (
            ("conversation", getattr(long_term_memory, "writer", None)),
            ("multimodal", getattr(long_term_memory, "multimodal_writer", None)),
        ):
            details = _writer_state_details(writer)
            if not isinstance(details, Mapping):
                continue
            last_error_message = _bounded_optional_text(
                details.get("last_error_message"),
                max_len=120,
            )
            if last_error_message:
                return (f"long_term_flush_{label}_writer_error", last_error_message)
            if details.get("worker_alive") is False:
                return (f"long_term_flush_{label}_writer_not_alive", None)
            pending_count = details.get("pending_count")
            try:
                pending = _coerce_nonnegative_int(pending_count or 0)
            except (TypeError, ValueError):
                pending = 0
            if pending > 0:
                return (f"long_term_flush_{label}_writer_pending", f"pending_count={pending}")
        return ("long_term_flush_timeout_or_incomplete", None)

    def _flush_personality_learning(
        self,
        personality_learning: _PersonalityLearningLike,
        *,
        results: list[PersonalityEvolutionResult],
        errors: list[str],
    ) -> _PersonalityFlushOutcome:
        """Flush one queued personality-learning batch and keep degradations visible."""

        try:
            result = personality_learning.flush_pending()
        except LongTermRemoteUnavailableError:
            raise
        except Exception as exc:
            error = _exception_summary(exc)
            errors.append(f"personality_flush:{error}")
            return _PersonalityFlushOutcome(
                status="error",
                failure_stage="personality_flush",
                error=error,
            )
        self._capture_personality_result(results, result)
        if result is None:
            return _PersonalityFlushOutcome(status="no_pending")
        return _PersonalityFlushOutcome(
            status="committed",
            result=result,
        )

    def _probe_personality_history(
        self,
        personality_learning: _PersonalityLearningLike,
        *,
        errors: list[str],
    ) -> PersonalityHistoryProbe:
        """Probe the persisted personality history and surface unreadable states explicitly."""

        try:
            probe = personality_learning.probe_history_status()
        except LongTermRemoteUnavailableError:
            raise
        except Exception as exc:
            error = _exception_summary(exc)
            errors.append(f"personality_history:personality_history_probe:{error}")
            return PersonalityHistoryProbe(
                history_read_status="error",
                failure_stage="personality_history_probe",
                error=error,
            )
        if probe.history_read_status not in {"loaded", "missing", "not_configured"}:
            detail = probe.error or probe.history_read_status
            stage = probe.failure_stage or "personality_history_probe"
            errors.append(f"personality_history:{stage}:{_bounded_text(detail, max_len=120)}")
        return probe

    def _capture_personality_result(
        self,
        results: list[PersonalityEvolutionResult],
        result: PersonalityEvolutionResult | None,
    ) -> None:
        """Keep one committed personality result when present."""

        if result is not None:
            results.append(result)

    def _load_target_day_reminders(
        self,
        *,
        target_day: LocalDate,
        local_tzinfo: tzinfo | None,
    ) -> tuple[ReminderEntry, ...]:
        """Return undelivered reminders relevant to the prepared local day."""

        try:
            entries = tuple(self.runtime.reminder_store.load_entries())
        except Exception:
            entries = ()
        selected: list[ReminderEntry] = []
        seen: set[str] = set()
        for entry in entries:
            reminder_id = _bounded_text(getattr(entry, "reminder_id", None), max_len=80)
            if reminder_id and reminder_id in seen:
                continue
            due_at = getattr(entry, "due_at", None)
            delivered = bool(getattr(entry, "delivered", False))
            if not isinstance(due_at, datetime) or delivered:
                continue
            local_due = due_at.astimezone(local_tzinfo) if due_at.tzinfo is not None and local_tzinfo is not None else due_at
            if local_due.date() != target_day:
                continue
            if reminder_id:
                seen.add(reminder_id)
            selected.append(entry)
        selected.sort(key=lambda item: getattr(item, "due_at", datetime.min).isoformat())
        due_now = tuple(self.runtime.peek_due_reminders(limit=_DEFAULT_REMINDER_LIMIT))
        for entry in due_now:
            reminder_id = _bounded_text(getattr(entry, "reminder_id", None), max_len=80)
            if reminder_id and reminder_id in seen:
                continue
            if reminder_id:
                seen.add(reminder_id)
            selected.insert(0, entry)
        return tuple(selected)

    def _render_reminder_lines(
        self,
        reminders: Sequence[ReminderEntry],
        *,
        local_tzinfo: tzinfo | None,
    ) -> tuple[str, ...]:
        """Render reminder entries into bounded digest lines."""

        lines: list[str] = []
        seen: set[str] = set()
        limit = _coerce_config_int(
            self.config,
            "nightly_digest_reminder_limit",
            _DEFAULT_REMINDER_LIMIT,
            minimum=1,
        )
        for entry in reminders:
            due_at = getattr(entry, "due_at", None)
            summary = _bounded_text(getattr(entry, "summary", None), max_len=160)
            if not summary:
                continue
            time_label = ""
            if isinstance(due_at, datetime):
                local_due = due_at.astimezone(local_tzinfo) if due_at.tzinfo is not None and local_tzinfo is not None else due_at
                time_label = local_due.strftime("%H:%M")
            line = f"{time_label} {summary}".strip()
            normalized = line.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            lines.append(_bounded_text(line, max_len=_MAX_LINE))
            if len(lines) >= limit:
                break
        return tuple(lines)

    def _render_headline_lines(
        self,
        refresh_result: WorldIntelligenceRefreshResult | None,
    ) -> tuple[str, ...]:
        """Render awareness threads into bounded digest headlines."""

        if refresh_result is None:
            return ()
        threads = tuple(getattr(refresh_result, "awareness_threads", ()))
        headline_limit = max(
            1,
            _coerce_config_int(
                self.config,
                "nightly_digest_headline_limit",
                _DEFAULT_HEADLINE_LIMIT,
                minimum=1,
            ),
        )
        ranked = sorted(
            threads,
            key=lambda item: (
                float(getattr(item, "salience", 0.0) or 0.0),
                _bounded_text(getattr(item, "updated_at", None), max_len=64),
            ),
            reverse=True,
        )
        lines: list[str] = []
        seen: set[str] = set()
        for thread in ranked:
            line = self._awareness_thread_line(thread)
            if not line or line.casefold() in seen:
                continue
            seen.add(line.casefold())
            lines.append(line)
            if len(lines) >= headline_limit:
                break
        return tuple(lines)

    def _awareness_thread_line(self, thread: SituationalAwarenessThread) -> str:
        """Render one awareness thread into a bounded calm headline line."""

        title = _bounded_text(getattr(thread, "title", None), max_len=120)
        summary = _bounded_text(getattr(thread, "summary", None), max_len=180)
        if title and summary and summary.casefold() != title.casefold():
            return _bounded_text(f"{title}: {summary}", max_len=_MAX_LINE)
        return title or summary

    def _render_world_news_sources(
        self,
        refresh_result: WorldIntelligenceRefreshResult | None,
    ) -> tuple[str, ...]:
        """Collect bounded source URLs from the refreshed world-intelligence set."""

        if refresh_result is None:
            return ()
        candidates: list[str] = []
        for subscription in tuple(getattr(refresh_result, "subscriptions", ())):
            source_page_url = _bounded_optional_text(getattr(subscription, "source_page_url", None), max_len=512)
            feed_url = _bounded_optional_text(getattr(subscription, "feed_url", None), max_len=512)
            if source_page_url:
                candidates.append(source_page_url)
            if feed_url:
                candidates.append(feed_url)
        return _bounded_text_tuple(candidates, max_items=12, max_len=512)

    def _build_digest_inputs(
        self,
        *,
        target_day: LocalDate,
        effective_now: datetime,
        world_refresh_result: WorldIntelligenceRefreshResult | None,
    ) -> dict[str, object]:
        """Build the structured inputs used for digest preparation."""

        local_tzinfo = _coerce_tzinfo(effective_now.tzinfo)
        reminders = self._load_target_day_reminders(
            target_day=target_day,
            local_tzinfo=local_tzinfo,
        )
        reminder_lines = self._render_reminder_lines(reminders, local_tzinfo=local_tzinfo)
        headline_lines = self._render_headline_lines(world_refresh_result)
        language = _bounded_optional_text(
            getattr(self.config, "openai_realtime_language", None),
            max_len=24,
        )
        return {
            "language": language,
            "target_local_day": target_day.isoformat(),
            "reminders": reminder_lines,
            "headline_lines": headline_lines,
            "world_news_sources": self._render_world_news_sources(world_refresh_result),
            "weather_summary": None,
            "live_news_summary": None,
            "target_day_reminder_count": len(reminders),
            "due_reminder_count": len(tuple(self.runtime.peek_due_reminders(limit=_DEFAULT_REMINDER_LIMIT))),
        }

    def _should_use_live_web_queries(self) -> bool:
        """Return whether bounded live web augmentation is enabled."""

        return bool(getattr(self.config, "nightly_live_web_augmentation_enabled", True))

    def _configured_location_hint(self) -> str | None:
        """Return the configured approximate place context for live web search."""

        parts: list[str] = []
        seen: set[str] = set()
        for attr_name in (
            "openai_web_search_city",
            "openai_web_search_region",
            "openai_web_search_country",
        ):
            text = _bounded_optional_text(getattr(self.config, attr_name, None), max_len=80)
            if text is None:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            parts.append(text)
        if not parts:
            return None
        return ", ".join(parts)

    def _augment_digest_inputs(
        self,
        *,
        target_day_text: str,
        digest_inputs: Mapping[str, object],
    ) -> tuple[int, str | None, tuple[str, ...], str | None, tuple[str, ...]]:
        """Fetch bounded weather/news augmentation for the prepared digest."""

        query_limit = max(
            0,
            _coerce_config_int(
                self.config,
                "nightly_live_web_query_limit",
                _DEFAULT_LIVE_WEB_QUERY_LIMIT,
            ),
        )
        if query_limit <= 0:
            return 0, None, (), None, ()

        language = _bounded_optional_text(digest_inputs.get("language"), max_len=24) or "de"
        weather_summary: str | None = None
        live_news_summary: str | None = None
        weather_sources: tuple[str, ...] = ()
        news_sources: tuple[str, ...] = ()
        queries_used = 0
        location_hint = self._configured_location_hint()

        weather = self._search(
            question=(
                f"Give the weather forecast for {location_hint} on {target_day_text}. "
                if location_hint
                else f"Give the weather forecast for {target_day_text}. "
            )
            + (
                f"Answer briefly in {language}."
            ),
            location_hint=location_hint,
            date_context=target_day_text,
        )
        if weather is not None:
            queries_used += 1
            weather_summary = _bounded_optional_text(getattr(weather, "answer", None), max_len=600)
            weather_sources = _bounded_text_tuple(
                _tuple_from_value(getattr(weather, "sources", ()) or ()),
                max_items=12,
                max_len=512,
            )

        headline_lines = _tuple_from_value(digest_inputs.get("headline_lines", ()))
        world_news_sources = _tuple_from_value(digest_inputs.get("world_news_sources", ()))
        if headline_lines:
            return queries_used, weather_summary, weather_sources, None, _bounded_text_tuple(
                world_news_sources,
                max_items=12,
                max_len=512,
            )
        if queries_used >= query_limit or len(headline_lines) >= max(
            1,
            _coerce_config_int(
                self.config,
                "nightly_digest_headline_limit",
                _DEFAULT_HEADLINE_LIMIT,
                minimum=1,
            ),
        ):
            return queries_used, weather_summary, weather_sources, None, ()

        news = self._search(
            question=(
                (
                    f"List three to five calm, relevant local or world updates for {location_hint} on {target_day_text}. "
                    if location_hint
                    else f"List three to five calm, relevant local or world updates for {target_day_text}. "
                )
                +
                f"Answer briefly in {language}."
            ),
            location_hint=location_hint,
            date_context=target_day_text,
        )
        if news is not None:
            queries_used += 1
            live_news_summary = _bounded_optional_text(getattr(news, "answer", None), max_len=1200)
            news_sources = _bounded_text_tuple(
                _tuple_from_value(getattr(news, "sources", ()) or ()),
                max_items=12,
                max_len=512,
            )
        return queries_used, weather_summary, weather_sources, live_news_summary, news_sources

    def _search(self, *, question: str, date_context: str, location_hint: str | None = None) -> object | None:
        """Run one bounded live-search call when the configured backend supports it."""

        if self.search_backend is None:
            return None
        search = getattr(self.search_backend, "search_live_info_with_metadata", None)
        if not callable(search):
            return None
        search_fn = cast(Callable[..., object], search)
        try:
            return search_fn(  # pylint: disable=not-callable
                question,
                conversation=None,
                location_hint=location_hint,
                date_context=date_context,
            )
        except Exception:
            return None

    def _refine_digest_inputs(
        self,
        *,
        target_day_text: str,
        digest_inputs: Mapping[str, object],
        nightly_insights: NightlyInsightBundle,
    ) -> NightlyDigestRefinement:
        """Refine raw nightly data into calmer user-facing morning-digest content."""

        candidate_news_sources = (
            *_tuple_from_value(digest_inputs.get("world_news_sources", ())),
            *_tuple_from_value(digest_inputs.get("news_sources", ())),
        )
        return build_nightly_digest_refinement(
            compose=self._compose_with_backend if self.text_backend is not None else None,
            language=_bounded_optional_text(digest_inputs.get("language"), max_len=24),
            target_day_text=target_day_text,
            location_hint=self._configured_location_hint(),
            raw_headline_lines=_bounded_text_tuple(
                _tuple_from_value(digest_inputs.get("headline_lines", ())),
                max_items=_coerce_config_int(
                    self.config,
                    "nightly_digest_headline_limit",
                    _DEFAULT_HEADLINE_LIMIT,
                    minimum=1,
                ),
                max_len=_MAX_LINE,
            ),
            raw_live_news_summary=_bounded_optional_text(digest_inputs.get("live_news_summary"), max_len=1200),
            candidate_news_sources=_bounded_text_tuple(candidate_news_sources, max_items=12, max_len=512),
            raw_new_insights=nightly_insights.new_insights,
            raw_continuity_shifts=nightly_insights.continuity_shifts,
            headline_limit=max(
                1,
                int(
                    getattr(self.config, "nightly_digest_headline_limit", _DEFAULT_HEADLINE_LIMIT)
                    or _DEFAULT_HEADLINE_LIMIT
                ),
            ),
            insight_limit=2,
            continuity_limit=2,
            source_limit=5,
        )

    def _prepare_digest(
        self,
        *,
        target_day_text: str,
        prepared_at: str,
        digest_inputs: Mapping[str, object],
        errors: list[str],
    ) -> NightlyPreparedDigest:
        """Compose one spoken and one print-ready morning digest artifact."""

        language = _bounded_optional_text(digest_inputs.get("language"), max_len=24)
        reminder_lines = _bounded_text_tuple(
            _tuple_from_value(digest_inputs.get("reminders", ())),
            max_items=max(
                1,
                _coerce_config_int(
                    self.config,
                    "nightly_digest_reminder_limit",
                    _DEFAULT_REMINDER_LIMIT,
                    minimum=1,
                ),
            ),
            max_len=_MAX_LINE,
        )
        headline_lines = _bounded_text_tuple(
            _tuple_from_value(digest_inputs.get("headline_lines", ())),
            max_items=max(
                1,
                _coerce_config_int(
                    self.config,
                    "nightly_digest_headline_limit",
                    _DEFAULT_HEADLINE_LIMIT,
                    minimum=1,
                ),
            ),
            max_len=_MAX_LINE,
        )
        weather_summary = _bounded_optional_text(digest_inputs.get("weather_summary"), max_len=600)
        live_news_summary = _bounded_optional_text(digest_inputs.get("live_news_summary"), max_len=1200)
        new_insights = _bounded_text_tuple(
            _tuple_from_value(digest_inputs.get("new_insights", ())),
            max_items=max(
                1,
                _coerce_config_int(
                    self.config,
                    "nightly_digest_headline_limit",
                    _DEFAULT_HEADLINE_LIMIT,
                    minimum=1,
                ),
            ),
            max_len=_MAX_LINE,
        )
        continuity_shifts = _bounded_text_tuple(
            _tuple_from_value(digest_inputs.get("continuity_shifts", ())),
            max_items=max(
                1,
                _coerce_config_int(
                    self.config,
                    "nightly_digest_headline_limit",
                    _DEFAULT_HEADLINE_LIMIT,
                    minimum=1,
                ),
            ),
            max_len=_MAX_LINE,
        )
        weather_sources = _bounded_text_tuple(
            _tuple_from_value(digest_inputs.get("weather_sources", ())),
            max_items=12,
            max_len=512,
        )
        news_sources = _bounded_text_tuple(
            _tuple_from_value(digest_inputs.get("news_sources", ())),
            max_items=12,
            max_len=512,
        )

        spoken_text = self._compose_with_backend(
            prompt=self._spoken_digest_prompt(
                target_day_text=target_day_text,
                language=language,
                reminder_lines=reminder_lines,
                headline_lines=headline_lines,
                new_insights=new_insights,
                continuity_shifts=continuity_shifts,
                weather_summary=weather_summary,
                live_news_summary=live_news_summary,
            ),
            max_len=1200,
        )
        if not spoken_text:
            errors.append("spoken_digest:provider_unavailable")
            spoken_text = self._fallback_spoken_digest(
                target_day_text=target_day_text,
                reminder_lines=reminder_lines,
                headline_lines=headline_lines,
                new_insights=new_insights,
                continuity_shifts=continuity_shifts,
                weather_summary=weather_summary,
                live_news_summary=live_news_summary,
            )

        print_source = self._compose_with_backend(
            prompt=self._print_digest_prompt(
                target_day_text=target_day_text,
                language=language,
                reminder_lines=reminder_lines,
                headline_lines=headline_lines,
                new_insights=new_insights,
                continuity_shifts=continuity_shifts,
                weather_summary=weather_summary,
                live_news_summary=live_news_summary,
            ),
            max_len=1200,
        )
        if not print_source:
            errors.append("print_digest:provider_unavailable")
            print_source = self._fallback_print_digest(
                target_day_text=target_day_text,
                reminder_lines=reminder_lines,
                headline_lines=headline_lines,
                new_insights=new_insights,
                continuity_shifts=continuity_shifts,
                weather_summary=weather_summary,
                live_news_summary=live_news_summary,
            )

        print_text = self._format_print_text(print_source)
        if not print_text:
            print_text = print_source

        return NightlyPreparedDigest(
            target_local_day=target_day_text,
            prepared_at=prepared_at,
            language=language,
            spoken_text=_bounded_block_text(spoken_text, max_len=_MAX_TEXT),
            print_text=_bounded_block_text(print_text, max_len=_MAX_TEXT),
            weather_summary=weather_summary,
            reminder_lines=reminder_lines,
            headline_lines=headline_lines,
            new_insights=new_insights,
            continuity_shifts=continuity_shifts,
            live_news_summary=live_news_summary,
            weather_sources=weather_sources,
            news_sources=news_sources,
        )

    def _compose_with_backend(self, *, prompt: str, max_len: int) -> str:
        """Use the configured text backend for one no-web summarization prompt."""

        if self.text_backend is None:
            return ""
        respond = getattr(self.text_backend, "respond_with_metadata", None)
        if not callable(respond):
            return ""
        respond_fn = cast(Callable[..., object], respond)
        try:
            response = respond_fn(  # pylint: disable=not-callable
                prompt,
                conversation=None,
                instructions=None,
                allow_web_search=False,
            )
        except Exception:
            return ""
        return _bounded_block_text(getattr(response, "text", None), max_len=max_len)

    def _format_print_text(self, direct_text: str) -> str:
        """Route one print digest through the print formatter when available."""

        if not direct_text or self.print_backend is None:
            return direct_text
        compose = getattr(self.print_backend, "compose_print_job_with_metadata", None)
        if not callable(compose):
            return direct_text
        compose_fn = cast(Callable[..., object], compose)
        try:
            response = compose_fn(  # pylint: disable=not-callable
                conversation=None,
                focus_hint="prepared_morning_digest",
                direct_text=direct_text,
                request_source="nightly_digest",
            )
        except Exception:
            return direct_text
        return _bounded_block_text(getattr(response, "text", None), max_len=_MAX_TEXT) or direct_text

    def _spoken_digest_prompt(
        self,
        *,
        target_day_text: str,
        language: str | None,
        reminder_lines: Sequence[str],
        headline_lines: Sequence[str],
        new_insights: Sequence[str],
        continuity_shifts: Sequence[str],
        weather_summary: str | None,
        live_news_summary: str | None,
    ) -> str:
        """Build the prompt used to compose the spoken morning digest."""

        language_hint = language or "de"
        return (
            "Prepare Twinr's spoken morning briefing.\n"
            f"Language: {language_hint}\n"
            "Constraints:\n"
            "- Calm, simple, warm wording.\n"
            "- Mention the date once.\n"
            "- Maximum 6 short sentences.\n"
            "- No markdown, no bullets, no technical notes.\n"
            "- Do not mention sections that are missing.\n"
            f"Target local day: {target_day_text}\n"
            f"Weather: {weather_summary or 'none'}\n"
            f"Reminders: {json.dumps(list(reminder_lines), ensure_ascii=False)}\n"
            f"Headlines: {json.dumps(list(headline_lines), ensure_ascii=False)}\n"
            f"New insights: {json.dumps(list(new_insights), ensure_ascii=False)}\n"
            f"Continuity shifts: {json.dumps(list(continuity_shifts), ensure_ascii=False)}\n"
            f"Fallback news summary: {live_news_summary or 'none'}\n"
            "Return only the final spoken briefing text."
        )

    def _print_digest_prompt(
        self,
        *,
        target_day_text: str,
        language: str | None,
        reminder_lines: Sequence[str],
        headline_lines: Sequence[str],
        new_insights: Sequence[str],
        continuity_shifts: Sequence[str],
        weather_summary: str | None,
        live_news_summary: str | None,
    ) -> str:
        """Build the prompt used to compose the print digest source text."""

        language_hint = language or "de"
        return (
            "Prepare Twinr's printed morning digest.\n"
            f"Language: {language_hint}\n"
            "Constraints:\n"
            "- Very short lines.\n"
            "- Plain text only.\n"
            "- Maximum 8 lines.\n"
            "- Keep weather, reminders, and headlines separate when present.\n"
            f"Target local day: {target_day_text}\n"
            f"Weather: {weather_summary or 'none'}\n"
            f"Reminders: {json.dumps(list(reminder_lines), ensure_ascii=False)}\n"
            f"Headlines: {json.dumps(list(headline_lines), ensure_ascii=False)}\n"
            f"New insights: {json.dumps(list(new_insights), ensure_ascii=False)}\n"
            f"Continuity shifts: {json.dumps(list(continuity_shifts), ensure_ascii=False)}\n"
            f"Fallback news summary: {live_news_summary or 'none'}\n"
            "Return only the plain text that should appear on paper."
        )

    def _fallback_spoken_digest(
        self,
        *,
        target_day_text: str,
        reminder_lines: Sequence[str],
        headline_lines: Sequence[str],
        new_insights: Sequence[str],
        continuity_shifts: Sequence[str],
        weather_summary: str | None,
        live_news_summary: str | None,
    ) -> str:
        """Build one deterministic spoken digest when the text backend fails."""

        parts = [f"Guten Morgen. Heute ist {target_day_text}."]
        if weather_summary:
            parts.append(weather_summary)
        if reminder_lines:
            parts.append(f"Heute wichtig: {'; '.join(reminder_lines[:3])}.")
        if headline_lines:
            parts.append(f"Außerdem relevant: {'; '.join(headline_lines[:2])}.")
        if new_insights:
            parts.append(f"Neu aufgefallen: {'; '.join(new_insights[:2])}.")
        if continuity_shifts:
            parts.append(f"Weiter im Blick: {'; '.join(continuity_shifts[:2])}.")
        elif live_news_summary:
            parts.append(live_news_summary)
        return _bounded_text(" ".join(part for part in parts if part), max_len=1200)

    def _fallback_print_digest(
        self,
        *,
        target_day_text: str,
        reminder_lines: Sequence[str],
        headline_lines: Sequence[str],
        new_insights: Sequence[str],
        continuity_shifts: Sequence[str],
        weather_summary: str | None,
        live_news_summary: str | None,
    ) -> str:
        """Build one deterministic print digest when the text backend fails."""

        lines = [f"Morgenbriefing {target_day_text}"]
        if weather_summary:
            lines.append(f"Wetter: {weather_summary}")
        if reminder_lines:
            lines.extend(f"Termin: {line}" for line in reminder_lines[:3])
        if headline_lines:
            lines.extend(f"Aktuell: {line}" for line in headline_lines[:3])
        if new_insights:
            lines.extend(f"Neu: {line}" for line in new_insights[:2])
        if continuity_shifts:
            lines.extend(f"Bleibt dran: {line}" for line in continuity_shifts[:2])
        elif live_news_summary:
            lines.append(f"Aktuell: {live_news_summary}")
        return _bounded_block_text("\n".join(lines), max_len=1200)

    def _build_summary(
        self,
        *,
        target_day_text: str,
        prepared_at: str,
        long_term_flush_ok: bool,
        failure_stage: str | None,
        reflection_result: LongTermReflectionResultV1 | None,
        world_refresh_result: WorldIntelligenceRefreshResult | None,
        digest: NightlyPreparedDigest,
        due_reminder_count: int,
        target_day_reminder_count: int,
        personality_status: NightlyPersonalityStatus,
        personality_results: Sequence[PersonalityEvolutionResult],
        nightly_insights: NightlyInsightBundle,
        live_search_queries: int,
        errors: Sequence[str],
    ) -> NightlyConsolidationSummary:
        """Build the bounded nightly consolidation summary artifact."""

        accepted_delta_count = max(
            personality_status.accepted_personality_delta_count,
            sum(len(result.accepted_deltas) for result in personality_results),
        )
        return NightlyConsolidationSummary(
            target_local_day=target_day_text,
            prepared_at=prepared_at,
            long_term_flush_ok=long_term_flush_ok,
            failure_stage=_bounded_optional_text(failure_stage, max_len=96),
            reflection_reflected_object_count=len(getattr(reflection_result, "reflected_objects", ())),
            reflection_created_summary_count=len(getattr(reflection_result, "created_summaries", ())),
            world_refresh_status=(
                _bounded_text(getattr(world_refresh_result, "status", None), max_len=32) or "unavailable"
            ),
            world_refresh_refreshed=bool(
                getattr(world_refresh_result, "refreshed", False) if world_refresh_result is not None else False
            ),
            world_awareness_thread_count=len(
                getattr(world_refresh_result, "awareness_threads", ()) if world_refresh_result is not None else ()
            ),
            due_reminder_count=max(0, due_reminder_count),
            target_day_reminder_count=max(0, target_day_reminder_count),
            accepted_personality_delta_count=accepted_delta_count,
            personality_flush_status=personality_status.flush_status,
            personality_history_read_status=personality_status.history_read_status,
            last_personality_commit_at=personality_status.last_personality_commit_at,
            personality_history_error=personality_status.error,
            new_insights=digest.new_insights,
            continuity_shifts=digest.continuity_shifts,
            operator_new_insights=nightly_insights.new_insights,
            operator_continuity_shifts=nightly_insights.continuity_shifts,
            live_search_queries=live_search_queries,
            weather_sources=digest.weather_sources,
            news_sources=digest.news_sources,
            errors=_bounded_text_tuple(errors, max_items=32, max_len=_MAX_ERROR),
        )

    def _prepare_display_reserve(
        self,
        *,
        effective_now: datetime,
        reflection_result: LongTermReflectionResultV1 | None,
        errors: list[str],
    ) -> DisplayReserveNightlyMaintenanceResult | None:
        """Prepare the next display reserve plan without duplicating reflection."""

        planner = self._require_display_planner()
        original_factory = planner.long_term_memory_factory
        if reflection_result is not None:
            planner.long_term_memory_factory = cast(
                Callable[[TwinrConfig], LongTermMemoryService],
                lambda _config: cast(
                    LongTermMemoryService,
                    _PrecomputedReflectionMemoryService(reflection=reflection_result),
                ),
            )
        try:
            return planner.maybe_run_nightly_maintenance(
                config=self.config,
                local_now=effective_now,
                search_backend=self.search_backend,
            )
        except LongTermRemoteUnavailableError:
            raise
        except Exception as exc:
            errors.append(f"display_reserve:{_exception_summary(exc)}")
            return None
        finally:
            planner.long_term_memory_factory = original_factory


__all__ = [
    "NightlyConsolidationSummary",
    "NightlyConsolidationSummaryStore",
    "NightlyOrchestrationResult",
    "NightlyPersonalityStatus",
    "NightlyPersonalityStatusStore",
    "NightlyPreparedDigest",
    "NightlyPreparedDigestStore",
    "NightlyRunState",
    "NightlyRunStateStore",
    "TwinrNightlyOrchestrator",
]
