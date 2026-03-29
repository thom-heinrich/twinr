# CHANGELOG: 2026-03-29
# BUG-1: Fixed cursor semantics after topic retirement so the next item no longer skips/repeats cards.
# BUG-2: Fixed last-shown resolution so idle fill still returns the real last published card even when all active topics are retired.
# BUG-3: Fixed plan persistence races/corruption by adding inter-process locking plus atomic fsync+replace writes.
# SEC-1: Refuse oversized/non-regular persisted plan files and write plan artifacts with owner-only permissions.
# IMP-1: Upgraded subset selection from static top-weight picking to adaptive multi-objective re-ranking with relevance/diversity/coverage balancing.
# IMP-2: Upgraded ordering from hard local gap heuristics to intent-aware whole-cycle scheduling with adaptive diversity pressure.

"""Plan one full local day of calm HDMI reserve impulses.

The live proactive monitor should not improvise reserve-card timing on every
tick. This module turns the current expanded reserve card surfaces into one
persistent per-day sequence that can be replayed calmly throughout the day.
That keeps scheduling, candidate weighting, and cursor persistence out of the
hot runtime loop while preserving a visible, personality-shaped reserve
surface beside the face. The planner spaces cards by grouped semantic topics,
normalized seed families, and coarse public/personal/setup axes instead of
only raw source tokens, so sibling cards do not clump and the right lane stays
broader and less repetitive.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date as LocalDate, datetime, time as LocalTime, timedelta, timezone
import errno
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import stat as statmod
import tempfile

try:  # Raspberry Pi 4 / Linux path.
    import fcntl
except Exception:  # pragma: no cover - only relevant on unsupported platforms.
    fcntl = None

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate
from twinr.display.reserve_bus_feedback import (
    DisplayReserveBusFeedbackSignal,
    DisplayReserveBusFeedbackStore,
)

from .display_reserve_candidates import load_display_reserve_candidates
from .display_reserve_diversity import reserve_seed_axis, reserve_seed_family
from .display_reserve_support import (
    compact_text,
    default_local_now,
    format_timestamp,
    parse_local_time as _parse_local_time,
    parse_timestamp as _parse_timestamp,
    utc_now,
)

_DEFAULT_PLAN_PATH = "artifacts/stores/ops/display_reserve_bus_plan.json"
_DEFAULT_REFRESH_AFTER_LOCAL = "05:30"
_DEFAULT_CANDIDATE_LIMIT = 20
_DEFAULT_ITEMS_PER_DAY = 20
_DEFAULT_TOPIC_GAP = 2
_DEFAULT_SOURCE_GAP = 1
_DEFAULT_FAMILY_GAP = 1
_DEFAULT_AXIS_GAP = 1
_EMPTY_PLAN_RETRY_S = 60.0
_DEFAULT_MIN_HOLD_S = 4.0 * 60.0
_DEFAULT_BASE_HOLD_S = 8.0 * 60.0
_DEFAULT_MAX_HOLD_S = 12.0 * 60.0
_DEFAULT_DIVERSITY_PRESSURE = 0.42
_MAX_DIVERSITY_PRESSURE = 0.85
_MIN_DIVERSITY_PRESSURE = 0.08
_PLAN_SCHEMA_VERSION = 2
_MAX_PLAN_BYTES = 1_048_576

_ACTION_BONUS_S = {
    "silent": -2.0 * 60.0,
    "hint": 0.0,
    "brief_update": 1.0 * 60.0,
    "ask_one": 2.0 * 60.0,
    "invite_follow_up": 3.0 * 60.0,
}
_ATTENTION_BONUS_S = {
    "background": 0.0,
    "growing": 1.0 * 60.0,
    "forming": 2.0 * 60.0,
    "shared_thread": 3.0 * 60.0,
}

_LOGGER = logging.getLogger(__name__)


def _bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    """Return one finite bounded integer config value."""

    if not isinstance(value, (int, float, str, bytes, bytearray)):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, number))


def _bounded_seconds(value: object, *, default: float, minimum: float, maximum: float) -> float:
    """Return one finite bounded duration."""

    if not isinstance(value, (int, float, str, bytes, bytearray)):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return max(minimum, min(maximum, number))


def _normalize_topic_keys(values: Iterable[object] | None) -> tuple[str, ...]:
    """Normalize one ordered set of retired topic keys."""

    if values is None:
        return ()
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        compact = compact_text(value, max_len=96).casefold()
        if not compact or compact in seen:
            continue
        seen.add(compact)
        ordered.append(compact)
    return tuple(ordered)


def _normalize_source_key(value: object) -> str:
    """Return one normalized source token."""

    return compact_text(value, max_len=48).casefold() or "unknown"


def _stable_fraction(*parts: object) -> float:
    """Return one deterministic fraction in the inclusive 0..1 range."""

    digest = hashlib.sha1(
        "::".join(compact_text(part, max_len=160) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") / 4_294_967_295.0


def _largest_remainder_allocation(
    *,
    weights: Mapping[str, float],
    slots: int,
    availability: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """Allocate integer targets from fractional weights with bounded availability."""

    if slots <= 0 or not weights:
        return {key: 0 for key in weights}
    positive = {key: max(0.0, float(value)) for key, value in weights.items()}
    total = sum(positive.values())
    if total <= 0.0:
        positive = {key: 1.0 for key in weights}
        total = float(len(positive))
    allocations: dict[str, int] = {key: 0 for key in positive}
    remainders: list[tuple[float, str]] = []
    remaining = slots
    for key, weight in positive.items():
        ideal = (weight / total) * float(slots)
        initial = int(math.floor(ideal))
        if availability is not None:
            initial = min(initial, max(0, int(availability.get(key, 0))))
        allocations[key] = max(0, initial)
        remaining -= allocations[key]
        remainders.append((ideal - initial, key))
    if remaining > 0:
        for _remainder, key in sorted(remainders, reverse=True):
            if remaining <= 0:
                break
            if availability is not None and allocations[key] >= max(0, int(availability.get(key, 0))):
                continue
            allocations[key] += 1
            remaining -= 1
    return allocations


def _best_effort_fsync_directory(path: Path) -> None:
    """Flush one directory entry update when the host platform supports it."""

    try:
        directory_fd = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    except OSError:
        return
    finally:
        os.close(directory_fd)


def _default_candidate_loader(
    config: TwinrConfig,
    *,
    local_now: datetime,
    max_items: int,
) -> tuple[AmbientDisplayImpulseCandidate, ...]:
    """Load the current reserve candidates from structured personality state."""

    return load_display_reserve_candidates(
        config,
        local_now=local_now,
        max_items=max_items,
    )


@dataclass(frozen=True, slots=True)
class DisplayReservePlannedItem:
    """Describe one planned ambient reserve-card publication for the day."""

    topic_key: str
    title: str
    source: str
    action: str
    attention_state: str
    eyebrow: str
    headline: str
    body: str
    symbol: str
    accent: str
    reason: str
    candidate_family: str
    salience: float
    hold_seconds: float
    semantic_topic_key: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "DisplayReservePlannedItem":
        """Build one planned item from JSON-style persisted data."""

        return cls(
            topic_key=compact_text(payload.get("topic_key"), max_len=96).casefold(),
            title=compact_text(payload.get("title"), max_len=72),
            source=compact_text(payload.get("source"), max_len=48),
            action=compact_text(payload.get("action"), max_len=24).lower() or "hint",
            attention_state=compact_text(payload.get("attention_state"), max_len=24).lower() or "background",
            eyebrow=compact_text(payload.get("eyebrow"), max_len=36),
            headline=compact_text(payload.get("headline"), max_len=128),
            body=compact_text(payload.get("body"), max_len=128),
            symbol=compact_text(payload.get("symbol"), max_len=24) or "sparkles",
            accent=compact_text(payload.get("accent"), max_len=24).lower() or "info",
            reason=compact_text(payload.get("reason"), max_len=120),
            candidate_family=compact_text(payload.get("candidate_family"), max_len=40).casefold() or "general",
            salience=_bounded_seconds(
                payload.get("salience"),
                default=0.0,
                minimum=0.0,
                maximum=8.0,
            ),
            hold_seconds=_bounded_seconds(
                payload.get("hold_seconds"),
                default=_DEFAULT_BASE_HOLD_S,
                minimum=_DEFAULT_MIN_HOLD_S,
                maximum=_DEFAULT_MAX_HOLD_S,
            ),
            semantic_topic_key=compact_text(payload.get("semantic_topic_key"), max_len=96).casefold(),
        )

    @classmethod
    def from_candidate(
        cls,
        candidate: AmbientDisplayImpulseCandidate,
        *,
        hold_seconds: float,
        reason: str,
    ) -> "DisplayReservePlannedItem":
        """Build one planned item from a live candidate plus planned timing."""

        return cls(
            topic_key=compact_text(candidate.topic_key, max_len=96).casefold(),
            title=compact_text(candidate.title, max_len=72),
            source=compact_text(candidate.source, max_len=48),
            action=compact_text(candidate.action, max_len=24).lower() or "hint",
            attention_state=compact_text(candidate.attention_state, max_len=24).lower() or "background",
            eyebrow=compact_text(candidate.eyebrow, max_len=36),
            headline=compact_text(candidate.headline, max_len=128),
            body=compact_text(candidate.body, max_len=128),
            symbol=compact_text(candidate.symbol, max_len=24) or "sparkles",
            accent=compact_text(candidate.accent, max_len=24).lower() or "info",
            reason=compact_text(reason, max_len=120),
            candidate_family=compact_text(candidate.candidate_family, max_len=40).casefold() or "general",
            salience=max(0.0, float(candidate.salience)),
            hold_seconds=_bounded_seconds(
                hold_seconds,
                default=_DEFAULT_BASE_HOLD_S,
                minimum=_DEFAULT_MIN_HOLD_S,
                maximum=_DEFAULT_MAX_HOLD_S,
            ),
            semantic_topic_key=candidate.semantic_key(),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the planned item into JSON-safe data."""

        return asdict(self)

    def semantic_key(self) -> str:
        """Return the grouped semantic topic key for this planned item."""

        return compact_text(self.semantic_topic_key, max_len=96).casefold() or self.topic_key


@dataclass(frozen=True, slots=True)
class DisplayReserveDayPlan:
    """Persist the current local-day reserve publication sequence."""

    local_day: str
    generated_at: str
    cursor: int
    items: tuple[DisplayReservePlannedItem, ...]
    candidate_count: int = 0
    retired_topic_keys: tuple[str, ...] = ()
    schema_version: int = _PLAN_SCHEMA_VERSION

    @classmethod
    def empty(cls, *, local_day: LocalDate, generated_at: datetime | None = None) -> "DisplayReserveDayPlan":
        """Return one explicit empty plan for the requested local day."""

        return cls(
            local_day=local_day.isoformat(),
            generated_at=format_timestamp(generated_at or utc_now()),
            cursor=0,
            items=(),
            candidate_count=0,
            retired_topic_keys=(),
            schema_version=_PLAN_SCHEMA_VERSION,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "DisplayReserveDayPlan":
        """Build one day plan from persisted JSON-style data."""

        local_day = compact_text(payload.get("local_day"), max_len=16)
        if not local_day:
            raise ValueError("display reserve day plan requires local_day")
        generated_at = _parse_timestamp(payload.get("generated_at")) or utc_now()
        cursor = _bounded_int(payload.get("cursor"), default=0, minimum=0, maximum=10_000_000)
        raw_items = payload.get("items")
        raw_retired_topic_keys = payload.get("retired_topic_keys")
        if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes, bytearray)):
            raise ValueError("display reserve day plan items must be a sequence")
        items = tuple(
            DisplayReservePlannedItem.from_dict(entry)
            for entry in raw_items
            if isinstance(entry, Mapping)
        )
        return cls(
            local_day=local_day,
            generated_at=format_timestamp(generated_at),
            cursor=cursor,
            items=items,
            candidate_count=_bounded_int(
                payload.get("candidate_count"),
                default=len(items),
                minimum=0,
                maximum=10_000,
            ),
            retired_topic_keys=_normalize_topic_keys(
                raw_retired_topic_keys
                if isinstance(raw_retired_topic_keys, Iterable)
                and not isinstance(raw_retired_topic_keys, (str, bytes, bytearray))
                else None,
            ),
            schema_version=_bounded_int(
                payload.get("schema_version"),
                default=1,
                minimum=1,
                maximum=_PLAN_SCHEMA_VERSION,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the plan into JSON-safe data."""

        return {
            "schema_version": self.schema_version,
            "local_day": self.local_day,
            "generated_at": self.generated_at,
            "cursor": self.cursor,
            "candidate_count": self.candidate_count,
            "retired_topic_keys": list(self.retired_topic_keys),
            "items": [item.to_dict() for item in self.items],
        }

    def active_items(self) -> tuple[DisplayReservePlannedItem, ...]:
        """Return the current same-day rotation after answered topics are retired."""

        if not self.retired_topic_keys:
            return self.items
        retired = set(self.retired_topic_keys)
        return tuple(item for item in self.items if item.semantic_key() not in retired)

    def _next_active_position(self, *, cursor: int | None = None) -> int | None:
        """Return the absolute cursor position of the next active item."""

        if not self.items:
            return None
        retired = set(self.retired_topic_keys)
        start = max(0, int(self.cursor if cursor is None else cursor))
        size = len(self.items)
        for offset in range(size):
            absolute_position = start + offset
            item = self.items[absolute_position % size]
            if item.semantic_key() not in retired:
                return absolute_position
        return None

    def current_item(self) -> DisplayReservePlannedItem | None:
        """Return the current rotating item for the local day."""

        position = self._next_active_position()
        if position is None:
            return None
        return self.items[position % len(self.items)]

    def last_shown_item(self) -> DisplayReservePlannedItem | None:
        """Return the most recently shown item for the current local day."""

        if not self.items or self.cursor <= 0:
            return None
        return self.items[(self.cursor - 1) % len(self.items)]

    def is_exhausted(self) -> bool:
        """Return whether no active same-day rotation items remain."""

        return self._next_active_position() is None

    def advance(self) -> "DisplayReserveDayPlan":
        """Return one copy of the plan with the cursor advanced by one slot."""

        position = self._next_active_position()
        if position is None:
            return self
        return DisplayReserveDayPlan(
            local_day=self.local_day,
            generated_at=self.generated_at,
            cursor=position + 1,
            items=self.items,
            candidate_count=self.candidate_count,
            retired_topic_keys=self.retired_topic_keys,
            schema_version=self.schema_version,
        )

    def retire_topics(self, topic_keys: Sequence[object]) -> "DisplayReserveDayPlan":
        """Return one copy with additional answered same-day topics retired."""

        retired_topic_keys = _normalize_topic_keys((*self.retired_topic_keys, *topic_keys))
        if retired_topic_keys == self.retired_topic_keys:
            return self
        return DisplayReserveDayPlan(
            local_day=self.local_day,
            generated_at=self.generated_at,
            cursor=self.cursor,
            items=self.items,
            candidate_count=self.candidate_count,
            retired_topic_keys=retired_topic_keys,
            schema_version=self.schema_version,
        )


@dataclass(slots=True)
class DisplayReserveDayPlanStore:
    """Read and write the persistent local-day reserve plan artifact."""

    path: Path
    max_bytes: int = _MAX_PLAN_BYTES

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayReserveDayPlanStore":
        """Resolve the reserve-plan artifact path from configuration."""

        project_root = Path(config.project_root).expanduser().resolve()
        configured = Path(
            getattr(config, "display_reserve_bus_plan_path", _DEFAULT_PLAN_PATH) or _DEFAULT_PLAN_PATH
        )
        resolved = configured if configured.is_absolute() else project_root / configured
        return cls(path=resolved)

    @property
    def _lock_path(self) -> Path:
        return self.path.with_name(f"{self.path.name}.lock")

    @contextmanager
    def _locked(self, *, exclusive: bool) -> Iterator[None]:
        """Serialize cross-process access to the persisted plan file."""

        if fcntl is None:
            yield
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = os.open(
            self._lock_path,
            os.O_RDWR | os.O_CREAT,
            0o600,
        )
        try:
            operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(lock_fd, operation)
            yield
        finally:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)

    def _safe_existing_stat(self) -> os.stat_result | None:
        """Return one validated stat result for the persisted plan file."""

        try:
            stat_result = self.path.stat(follow_symlinks=False)
        except FileNotFoundError:
            return None
        except OSError:
            _LOGGER.warning("Failed to stat display reserve day plan at %s.", self.path, exc_info=True)
            return None
        if statmod.S_ISLNK(stat_result.st_mode):
            _LOGGER.warning("Ignoring symlinked display reserve day plan at %s.", self.path)
            return None
        if not statmod.S_ISREG(stat_result.st_mode):
            _LOGGER.warning("Ignoring non-regular display reserve day plan at %s.", self.path)
            return None
        if stat_result.st_size > self.max_bytes:
            _LOGGER.warning(
                "Ignoring oversized display reserve day plan at %s (%s bytes > %s bytes).",
                self.path,
                stat_result.st_size,
                self.max_bytes,
            )
            return None
        return stat_result

    def load(self) -> DisplayReserveDayPlan | None:
        """Load the currently persisted day plan, if one exists and parses."""

        with self._locked(exclusive=False):
            flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                file_descriptor = os.open(self.path, flags)
            except FileNotFoundError:
                return None
            except OSError as exc:
                if getattr(exc, "errno", None) in {getattr(errno, "ELOOP", None), getattr(errno, "EMLINK", None)}:
                    _LOGGER.warning("Ignoring symlinked display reserve day plan at %s.", self.path)
                    return None
                _LOGGER.warning("Failed to open display reserve day plan at %s.", self.path, exc_info=True)
                return None
            try:
                stat_result = os.fstat(file_descriptor)
                if not statmod.S_ISREG(stat_result.st_mode):
                    _LOGGER.warning("Ignoring non-regular display reserve day plan at %s.", self.path)
                    return None
                if stat_result.st_size > self.max_bytes:
                    _LOGGER.warning(
                        "Ignoring oversized display reserve day plan at %s (%s bytes > %s bytes).",
                        self.path,
                        stat_result.st_size,
                        self.max_bytes,
                    )
                    return None
                with os.fdopen(file_descriptor, "r", encoding="utf-8") as handle:
                    file_descriptor = -1
                    payload = json.load(handle)
            except FileNotFoundError:
                return None
            except Exception:
                _LOGGER.warning("Failed to read display reserve day plan from %s.", self.path, exc_info=True)
                return None
            finally:
                if file_descriptor >= 0:
                    try:
                        os.close(file_descriptor)
                    except OSError:
                        pass
        if not isinstance(payload, Mapping):
            _LOGGER.warning(
                "Ignoring invalid display reserve day plan payload at %s because it is not an object.",
                self.path,
            )
            return None
        try:
            return DisplayReserveDayPlan.from_dict(payload)
        except Exception:
            _LOGGER.warning("Ignoring invalid display reserve day plan payload at %s.", self.path, exc_info=True)
            return None

    def save(self, plan: DisplayReserveDayPlan) -> DisplayReserveDayPlan:
        """Persist one full day plan with atomic replacement and inter-process locking."""

        payload = json.dumps(plan.to_dict(), ensure_ascii=False, indent=2) + "\n"
        encoded = payload.encode("utf-8")
        if len(encoded) > self.max_bytes:
            raise ValueError(
                f"display reserve day plan payload exceeds {self.max_bytes} bytes ({len(encoded)} bytes)"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass
        with self._locked(exclusive=True):
            temp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=self.path.parent,
                    prefix=f".{self.path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as handle:
                    temp_path = Path(handle.name)
                    try:
                        os.chmod(temp_path, 0o600)
                    except OSError:
                        pass
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_path, self.path)
                try:
                    os.chmod(self.path, 0o600)
                except OSError:
                    pass
                _best_effort_fsync_directory(self.path)
            except Exception:
                if temp_path is not None:
                    try:
                        temp_path.unlink()
                    except FileNotFoundError:
                        pass
                    except OSError:
                        _LOGGER.warning(
                            "Failed to clean temporary display reserve day plan %s.",
                            temp_path,
                            exc_info=True,
                        )
                raise
        return plan

    def clear(self) -> None:
        """Remove the persisted plan artifact when it exists."""

        with self._locked(exclusive=True):
            try:
                existing = self._safe_existing_stat()
                if existing is not None:
                    self.path.unlink()
            except FileNotFoundError:
                return
            _best_effort_fsync_directory(self.path)


@dataclass(slots=True)
class DisplayReserveDayPlanner:
    """Build and persist one deterministic local-day reserve publication plan."""

    store: DisplayReserveDayPlanStore
    feedback_store: DisplayReserveBusFeedbackStore | None = None
    candidate_loader: Callable[..., tuple[AmbientDisplayImpulseCandidate, ...]] = _default_candidate_loader
    local_now: Callable[[], datetime] = default_local_now

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayReserveDayPlanner":
        """Build the planner and its persistent store from configuration."""

        return cls(
            store=DisplayReserveDayPlanStore.from_config(config),
            feedback_store=DisplayReserveBusFeedbackStore.from_config(config),
        )

    def ensure_plan(
        self,
        *,
        config: TwinrConfig,
        local_now: datetime | None = None,
    ) -> DisplayReserveDayPlan:
        """Load the current plan or build today's plan when none is current."""

        effective_now = (local_now or self.local_now()).astimezone()
        current_day = effective_now.date()
        refresh_after = _parse_local_time(
            getattr(config, "display_reserve_bus_refresh_after_local", _DEFAULT_REFRESH_AFTER_LOCAL),
            fallback=_DEFAULT_REFRESH_AFTER_LOCAL,
        )
        existing = self.store.load()
        feedback_signal = self._load_active_feedback(local_now=effective_now)
        if self._plan_is_current(existing, current_day=current_day, local_now=effective_now, refresh_after=refresh_after):
            if self._feedback_requires_rebuild(existing, feedback_signal=feedback_signal):
                retired_topic_keys = self._retired_topic_keys(
                    existing,
                    feedback_signal=feedback_signal,
                )
                rebuilt = self._build_plan(
                    config=config,
                    local_now=effective_now,
                    local_day=current_day,
                    feedback_signal=feedback_signal,
                    retired_topic_keys=retired_topic_keys,
                )
                if not rebuilt.items and existing is not None and existing.items:
                    rebuilt = DisplayReserveDayPlan(
                        local_day=rebuilt.local_day,
                        generated_at=rebuilt.generated_at,
                        cursor=existing.cursor,
                        items=existing.items,
                        candidate_count=max(existing.candidate_count, len(existing.items)),
                        retired_topic_keys=_normalize_topic_keys(retired_topic_keys),
                        schema_version=_PLAN_SCHEMA_VERSION,
                    )
                return self.store.save(rebuilt)
            if not self._should_retry_empty_current_plan(existing, local_now=effective_now):
                assert existing is not None
                return existing
        rebuilt = self._build_plan(
            config=config,
            local_now=effective_now,
            local_day=current_day,
            feedback_signal=feedback_signal,
        )
        return self.store.save(rebuilt)

    def build_plan_for_day(
        self,
        *,
        config: TwinrConfig,
        local_day: LocalDate,
        local_now: datetime | None = None,
        feedback_signal: DisplayReserveBusFeedbackSignal | None = None,
        retired_topic_keys: frozenset[str] = frozenset(),
    ) -> DisplayReserveDayPlan:
        """Build one unsaved plan for the requested local day.

        The explicit nightly companion planner uses this method to prepare the
        next day ahead of time without replacing the active current-day plan
        immediately. Runtime publication still persists the adopted plan only
        when the next local day actually starts.
        """

        effective_now = (local_now or self.local_now()).astimezone()
        return self._build_plan(
            config=config,
            local_now=effective_now,
            local_day=local_day,
            feedback_signal=feedback_signal,
            retired_topic_keys=retired_topic_keys,
        )

    def peek_next_item(
        self,
        *,
        config: TwinrConfig,
        local_now: datetime | None = None,
    ) -> DisplayReservePlannedItem | None:
        """Return the next planned reserve item for the current local day."""

        return self.ensure_plan(config=config, local_now=local_now).current_item()

    def peek_idle_fill_item(
        self,
        *,
        config: TwinrConfig,
        local_now: datetime | None = None,
    ) -> DisplayReservePlannedItem | None:
        """Return one passive same-day fill item when no active topics remain."""

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
        """Advance the current day plan after one successful publication."""

        plan = self.ensure_plan(config=config, local_now=local_now)
        advanced = plan.advance()
        return self.store.save(advanced)

    def _should_retry_empty_current_plan(
        self,
        plan: DisplayReserveDayPlan | None,
        *,
        local_now: datetime,
    ) -> bool:
        """Return whether one current empty plan should be rebuilt now.

        Empty same-day plans are not allowed to stay sticky for the whole day.
        World/personality state may become available a little later than the
        first proactive tick, so the planner retries after a short bounded
        backoff instead of caching blank reserve space until midnight.
        """

        if plan is None or plan.items:
            return False
        if plan.candidate_count > 0:
            return True
        generated_at = _parse_timestamp(plan.generated_at)
        if generated_at is None:
            return True
        age_s = (local_now.astimezone(timezone.utc) - generated_at).total_seconds()
        return age_s >= _EMPTY_PLAN_RETRY_S

    def _plan_is_current(
        self,
        plan: DisplayReserveDayPlan | None,
        *,
        current_day: LocalDate,
        local_now: datetime,
        refresh_after: LocalTime,
    ) -> bool:
        """Return whether one stored plan is still valid for the current local time."""

        if plan is None:
            return False
        if plan.local_day == current_day.isoformat():
            return True
        previous_day = current_day - timedelta(days=1)
        if plan.local_day != previous_day.isoformat():
            return False
        return local_now.timetz().replace(tzinfo=None) < refresh_after

    def _build_plan(
        self,
        *,
        config: TwinrConfig,
        local_now: datetime,
        local_day: LocalDate,
        feedback_signal: DisplayReserveBusFeedbackSignal | None,
        retired_topic_keys: frozenset[str] = frozenset(),
    ) -> DisplayReserveDayPlan:
        """Build one fresh local-day plan from current reserve candidates."""

        candidate_limit = _bounded_int(
            getattr(config, "display_reserve_bus_candidate_limit", _DEFAULT_CANDIDATE_LIMIT),
            default=_DEFAULT_CANDIDATE_LIMIT,
            minimum=1,
            maximum=96,
        )
        items_per_day = _bounded_int(
            getattr(config, "display_reserve_bus_items_per_day", _DEFAULT_ITEMS_PER_DAY),
            default=_DEFAULT_ITEMS_PER_DAY,
            minimum=1,
            maximum=96,
        )
        topic_gap = _bounded_int(
            getattr(config, "display_reserve_bus_topic_gap", _DEFAULT_TOPIC_GAP),
            default=_DEFAULT_TOPIC_GAP,
            minimum=0,
            maximum=8,
        )
        source_gap = _bounded_int(
            getattr(config, "display_reserve_bus_source_gap", _DEFAULT_SOURCE_GAP),
            default=_DEFAULT_SOURCE_GAP,
            minimum=0,
            maximum=8,
        )
        family_gap = _bounded_int(
            getattr(config, "display_reserve_bus_family_gap", _DEFAULT_FAMILY_GAP),
            default=_DEFAULT_FAMILY_GAP,
            minimum=0,
            maximum=8,
        )
        axis_gap = _bounded_int(
            getattr(config, "display_reserve_bus_axis_gap", _DEFAULT_AXIS_GAP),
            default=_DEFAULT_AXIS_GAP,
            minimum=0,
            maximum=8,
        )
        candidates = tuple(
            self.candidate_loader(
                config,
                local_now=local_now,
                max_items=candidate_limit,
            )
        )
        if retired_topic_keys:
            candidates = tuple(
                candidate
                for candidate in candidates
                if candidate.semantic_key() not in retired_topic_keys
            )
        if not candidates:
            return DisplayReserveDayPlan.empty(
                local_day=local_day,
                generated_at=local_now.astimezone(timezone.utc),
            )
        cycle_slots = self._cycle_slots(
            candidates=candidates,
            configured_slots=items_per_day,
        )
        ordered = self._plan_unique_cycle(
            candidates,
            slots=cycle_slots,
            local_day=local_day,
            topic_gap=topic_gap,
            source_gap=source_gap,
            family_gap=family_gap,
            axis_gap=axis_gap,
            feedback_signal=feedback_signal,
        )
        items = tuple(
            DisplayReservePlannedItem.from_candidate(
                candidate,
                hold_seconds=self._hold_seconds_for_candidate(config=config, candidate=candidate),
                reason=f"plan[{index}] {candidate.reason}",
            )
            for index, candidate in enumerate(ordered)
        )
        return DisplayReserveDayPlan(
            local_day=local_day.isoformat(),
            generated_at=format_timestamp(local_now.astimezone(timezone.utc)),
            cursor=0,
            items=items,
            candidate_count=len(candidates),
            retired_topic_keys=_normalize_topic_keys(retired_topic_keys),
            schema_version=_PLAN_SCHEMA_VERSION,
        )

    def _cycle_slots(
        self,
        *,
        candidates: Sequence[AmbientDisplayImpulseCandidate],
        configured_slots: int,
    ) -> int:
        """Return the unique rotation length for the current candidate pool.

        The planner stores one bounded unique cycle and reuses that cycle
        throughout the same local day. Cards do not disappear merely because
        they were shown once; they stay in rotation until real user feedback
        retires them or the next nightly plan replaces the whole day.
        """

        if not candidates:
            return 0
        return min(max(1, int(configured_slots)), len(candidates))

    def _plan_unique_cycle(
        self,
        candidates: Sequence[AmbientDisplayImpulseCandidate],
        *,
        slots: int,
        local_day: LocalDate,
        topic_gap: int,
        source_gap: int,
        family_gap: int,
        axis_gap: int,
        feedback_signal: DisplayReserveBusFeedbackSignal | None,
    ) -> tuple[AmbientDisplayImpulseCandidate, ...]:
        """Select and order one unique daily cycle with adaptive multi-objective reranking."""

        if slots <= 0 or not candidates:
            return ()
        selected: list[AmbientDisplayImpulseCandidate] = []
        remaining = list(candidates)
        diversity_pressure = self._diversity_pressure(
            candidates,
            feedback_signal=feedback_signal,
        )
        targets = self._coverage_targets(candidates, slots=slots)
        while remaining and len(selected) < slots:
            candidate = max(
                remaining,
                key=lambda item: (
                    self._selection_score(
                        item,
                        selected=selected,
                        targets=targets,
                        local_day=local_day,
                        topic_gap=topic_gap,
                        source_gap=source_gap,
                        family_gap=family_gap,
                        axis_gap=axis_gap,
                        diversity_pressure=diversity_pressure,
                        feedback_signal=feedback_signal,
                    ),
                    _stable_fraction(local_day.isoformat(), item.topic_key, "select", len(selected)),
                ),
            )
            selected.append(candidate)
            remaining.remove(candidate)
        return tuple(selected)

    def _coverage_targets(
        self,
        candidates: Sequence[AmbientDisplayImpulseCandidate],
        *,
        slots: int,
    ) -> dict[str, dict[str, int]]:
        """Estimate intent-aware coverage targets for source/family/axis buckets."""

        weights = {
            candidate.topic_key: self._candidate_weight(candidate, feedback_signal=None)
            for candidate in candidates
        }
        availability = {
            "family": Counter(self._candidate_family(candidate) for candidate in candidates),
            "axis": Counter(self._candidate_axis(candidate) for candidate in candidates),
            "source": Counter(_normalize_source_key(candidate.source) for candidate in candidates),
        }
        bucket_weights: dict[str, defaultdict[str, float]] = {
            "family": defaultdict(float),
            "axis": defaultdict(float),
            "source": defaultdict(float),
        }
        for candidate in candidates:
            weight = weights[candidate.topic_key]
            bucket_weights["family"][self._candidate_family(candidate)] += weight
            bucket_weights["axis"][self._candidate_axis(candidate)] += weight
            bucket_weights["source"][_normalize_source_key(candidate.source)] += weight
        return {
            dimension: _largest_remainder_allocation(
                weights=dict(bucket_weights[dimension]),
                slots=min(slots, len(bucket_weights[dimension]) + max(0, slots // 3)),
                availability=dict(availability[dimension]),
            )
            for dimension in ("family", "axis", "source")
        }

    def _selection_score(
        self,
        candidate: AmbientDisplayImpulseCandidate,
        *,
        selected: Sequence[AmbientDisplayImpulseCandidate],
        targets: Mapping[str, Mapping[str, int]],
        local_day: LocalDate,
        topic_gap: int,
        source_gap: int,
        family_gap: int,
        axis_gap: int,
        diversity_pressure: float,
        feedback_signal: DisplayReserveBusFeedbackSignal | None,
    ) -> float:
        """Score one candidate for the next slot in the unique cycle."""

        base_weight = self._candidate_weight(candidate, feedback_signal=feedback_signal)
        if not selected:
            return base_weight + (0.15 * self._coverage_bonus(candidate, selected=selected, targets=targets))
        recent_topics = tuple(item.semantic_key() for item in selected[-topic_gap:]) if topic_gap > 0 else ()
        if recent_topics and candidate.semantic_key() in recent_topics:
            return -1_000_000.0
        novelty = self._novelty_gain(candidate, selected=selected)
        coverage = self._coverage_bonus(candidate, selected=selected, targets=targets)
        recency = self._recency_spacing_score(
            candidate,
            selected=selected,
            source_gap=source_gap,
            family_gap=family_gap,
            axis_gap=axis_gap,
        )
        position = len(selected)
        return (
            base_weight * (1.0 - (0.48 * diversity_pressure))
            + (1.35 * diversity_pressure * novelty)
            + (0.78 * diversity_pressure * coverage)
            + recency
            + (
                0.05
                * self._candidate_axis_spacing_bonus(
                    candidate,
                    recent_axes=tuple(self._candidate_axis(item) for item in selected[-axis_gap:]) if axis_gap > 0 else (),
                )
            )
            + (0.02 * _stable_fraction(local_day.isoformat(), candidate.topic_key, "score", position))
        )

    def _novelty_gain(
        self,
        candidate: AmbientDisplayImpulseCandidate,
        *,
        selected: Sequence[AmbientDisplayImpulseCandidate],
    ) -> float:
        """Return one whole-list novelty gain for the candidate."""

        if not selected:
            return 1.0
        maximum_similarity = max(self._candidate_similarity(candidate, previous) for previous in selected)
        return max(0.0, 1.0 - maximum_similarity)

    def _candidate_similarity(
        self,
        left: AmbientDisplayImpulseCandidate,
        right: AmbientDisplayImpulseCandidate,
    ) -> float:
        """Return one bounded structural similarity between two candidates."""

        similarity = 0.0
        if left.semantic_key() == right.semantic_key():
            return 1.0
        if self._candidate_family(left) == self._candidate_family(right):
            similarity += 0.44
        if _normalize_source_key(left.source) == _normalize_source_key(right.source):
            similarity += 0.24
        if self._candidate_axis(left) == self._candidate_axis(right):
            similarity += 0.16
        if compact_text(left.action, max_len=24).casefold() == compact_text(right.action, max_len=24).casefold():
            similarity += 0.08
        if compact_text(left.attention_state, max_len=24).casefold() == compact_text(
            right.attention_state,
            max_len=24,
        ).casefold():
            similarity += 0.05
        return max(0.0, min(1.0, similarity))

    def _coverage_bonus(
        self,
        candidate: AmbientDisplayImpulseCandidate,
        *,
        selected: Sequence[AmbientDisplayImpulseCandidate],
        targets: Mapping[str, Mapping[str, int]],
    ) -> float:
        """Return one bonus for under-covered semantic buckets."""

        selected_family = Counter(self._candidate_family(item) for item in selected)
        selected_axis = Counter(self._candidate_axis(item) for item in selected)
        selected_source = Counter(_normalize_source_key(item.source) for item in selected)

        family = self._candidate_family(candidate)
        axis = self._candidate_axis(candidate)
        source = _normalize_source_key(candidate.source)

        family_target = max(0, int(targets.get("family", {}).get(family, 0)))
        axis_target = max(0, int(targets.get("axis", {}).get(axis, 0)))
        source_target = max(0, int(targets.get("source", {}).get(source, 0)))

        family_gap = max(0, family_target - selected_family[family])
        axis_gap_value = max(0, axis_target - selected_axis[axis])
        source_gap_value = max(0, source_target - selected_source[source])

        return (
            (0.54 * min(1.0, float(family_gap)))
            + (0.31 * min(1.0, float(axis_gap_value)))
            + (0.15 * min(1.0, float(source_gap_value)))
        )

    def _recency_spacing_score(
        self,
        candidate: AmbientDisplayImpulseCandidate,
        *,
        selected: Sequence[AmbientDisplayImpulseCandidate],
        source_gap: int,
        family_gap: int,
        axis_gap: int,
    ) -> float:
        """Return one soft spacing score against the recent local context."""

        score = 0.0
        if source_gap > 0:
            recent_sources = tuple(_normalize_source_key(item.source) for item in selected[-source_gap:])
            candidate_source = _normalize_source_key(candidate.source)
            if candidate_source in recent_sources:
                score -= 0.42
            else:
                score += 0.06
        if family_gap > 0:
            recent_families = tuple(self._candidate_family(item) for item in selected[-family_gap:])
            candidate_family = self._candidate_family(candidate)
            if candidate_family in recent_families:
                score -= 0.34
            else:
                score += 0.07
        if axis_gap > 0:
            recent_axes = tuple(self._candidate_axis(item) for item in selected[-axis_gap:])
            candidate_axis = self._candidate_axis(candidate)
            if candidate_axis in recent_axes:
                score -= 0.15
            else:
                score += 0.04
        return score

    def _diversity_pressure(
        self,
        candidates: Sequence[AmbientDisplayImpulseCandidate],
        *,
        feedback_signal: DisplayReserveBusFeedbackSignal | None,
    ) -> float:
        """Return the adaptive quality/diversity trade-off for the current day."""

        pressure = _bounded_seconds(
            _DEFAULT_DIVERSITY_PRESSURE,
            default=_DEFAULT_DIVERSITY_PRESSURE,
            minimum=_MIN_DIVERSITY_PRESSURE,
            maximum=_MAX_DIVERSITY_PRESSURE,
        )
        if not candidates:
            return pressure
        family_counts = Counter(self._candidate_family(candidate) for candidate in candidates)
        axis_counts = Counter(self._candidate_axis(candidate) for candidate in candidates)
        source_counts = Counter(_normalize_source_key(candidate.source) for candidate in candidates)
        total = float(len(candidates))
        concentration = max(
            max(family_counts.values(), default=0) / total,
            max(axis_counts.values(), default=0) / total,
            max(source_counts.values(), default=0) / total,
        )
        pressure += max(0.0, concentration - 0.34) * 0.45
        if feedback_signal is not None:
            intensity = max(0.0, min(1.0, float(feedback_signal.intensity)))
            if feedback_signal.reaction in {"avoided", "cooled", "ignored"}:
                pressure += 0.18 * intensity
            elif feedback_signal.reaction in {"engaged", "immediate_engagement"}:
                pressure -= 0.10 * intensity
        return max(_MIN_DIVERSITY_PRESSURE, min(_MAX_DIVERSITY_PRESSURE, pressure))

    def _candidate_family(self, candidate: AmbientDisplayImpulseCandidate) -> str:
        """Return one generic family token for reserve-plan mixing."""

        return reserve_seed_family(candidate)

    def _candidate_axis(self, candidate: AmbientDisplayImpulseCandidate) -> str:
        """Return one coarse conversation axis for reserve-plan spacing."""

        return reserve_seed_axis(candidate)

    def _candidate_axis_spacing_bonus(
        self,
        candidate: AmbientDisplayImpulseCandidate,
        *,
        recent_axes: Sequence[str],
    ) -> float:
        """Prefer axis alternation when it does not fight stronger spacing rules."""

        if not recent_axes:
            return 0.0
        if self._candidate_axis(candidate) in recent_axes:
            return 0.0
        return 0.08

    def _candidate_weight(
        self,
        candidate: AmbientDisplayImpulseCandidate,
        *,
        feedback_signal: DisplayReserveBusFeedbackSignal | None,
    ) -> float:
        """Return the relative planning weight for one reserve candidate."""

        action_bonus = {
            "hint": 0.10,
            "brief_update": 0.22,
            "ask_one": 0.30,
            "invite_follow_up": 0.40,
        }.get(candidate.action, 0.10)
        attention_bonus = {
            "background": 0.00,
            "growing": 0.08,
            "forming": 0.16,
            "shared_thread": 0.24,
        }.get(candidate.attention_state, 0.00)
        return (
            1.0
            + max(0.0, float(candidate.salience))
            + action_bonus
            + attention_bonus
            + self._feedback_weight_adjustment(candidate, feedback_signal=feedback_signal)
        )

    def _feedback_weight_adjustment(
        self,
        candidate: AmbientDisplayImpulseCandidate,
        *,
        feedback_signal: DisplayReserveBusFeedbackSignal | None,
    ) -> float:
        """Return one short-lived planning bias from recent reserve feedback."""

        if feedback_signal is None:
            return 0.0
        candidate_topic_key = compact_text(candidate.semantic_key(), max_len=96).casefold()
        feedback_topic_key = compact_text(feedback_signal.topic_key, max_len=96).casefold()
        if not feedback_topic_key or candidate_topic_key != feedback_topic_key:
            return 0.0
        intensity = max(0.0, min(1.0, float(feedback_signal.intensity)))
        if feedback_signal.reaction == "immediate_engagement":
            return 0.62 * intensity
        if feedback_signal.reaction == "engaged":
            return 0.38 * intensity
        if feedback_signal.reaction == "cooled":
            return -(0.26 * intensity)
        if feedback_signal.reaction == "avoided":
            return -(0.46 * intensity)
        if feedback_signal.reaction == "ignored":
            return -(0.18 * intensity)
        return 0.0

    def _load_active_feedback(
        self,
        *,
        local_now: datetime,
    ) -> DisplayReserveBusFeedbackSignal | None:
        """Load the current active reserve-bus feedback hint, if any."""

        if self.feedback_store is None:
            return None
        return self.feedback_store.load_active(now=local_now.astimezone(timezone.utc))

    def _feedback_requires_rebuild(
        self,
        plan: DisplayReserveDayPlan | None,
        *,
        feedback_signal: DisplayReserveBusFeedbackSignal | None,
    ) -> bool:
        """Return whether fresh bus feedback should rebuild today's plan."""

        if plan is None or feedback_signal is None:
            return False
        if plan.is_exhausted():
            return False
        generated_at = _parse_timestamp(plan.generated_at)
        if generated_at is None:
            return True
        return feedback_signal.requested_at_datetime() > generated_at

    def _retired_topic_keys(
        self,
        plan: DisplayReserveDayPlan | None,
        *,
        feedback_signal: DisplayReserveBusFeedbackSignal | None,
    ) -> frozenset[str]:
        """Return the same-day topic keys that should leave rotation now.

        Cards stay in the same-day loop until the user actually responds to
        them. Immediate/delayed engagement and explicit cooling/avoidance count
        as answered. Merely being shown once, or later being weakly ignored,
        does not retire the card from today's rotation.
        """

        retired = set(plan.retired_topic_keys if plan is not None else ())
        if feedback_signal is None:
            return frozenset(retired)
        if not feedback_signal.topic_key:
            return frozenset(retired)
        if feedback_signal.reaction in {"immediate_engagement", "engaged", "cooled", "avoided"}:
            retired.add(compact_text(feedback_signal.topic_key, max_len=96).casefold())
        return frozenset(retired)

    def _hold_seconds_for_candidate(
        self,
        *,
        config: TwinrConfig,
        candidate: AmbientDisplayImpulseCandidate,
    ) -> float:
        """Return the desired on-screen duration for one planned candidate."""

        base = _bounded_seconds(
            getattr(config, "display_reserve_bus_base_hold_s", _DEFAULT_BASE_HOLD_S),
            default=_DEFAULT_BASE_HOLD_S,
            minimum=_DEFAULT_MIN_HOLD_S,
            maximum=_DEFAULT_MAX_HOLD_S,
        )
        minimum = _bounded_seconds(
            getattr(config, "display_reserve_bus_min_hold_s", _DEFAULT_MIN_HOLD_S),
            default=_DEFAULT_MIN_HOLD_S,
            minimum=60.0,
            maximum=_DEFAULT_MAX_HOLD_S,
        )
        maximum = _bounded_seconds(
            getattr(config, "display_reserve_bus_max_hold_s", _DEFAULT_MAX_HOLD_S),
            default=_DEFAULT_MAX_HOLD_S,
            minimum=minimum,
            maximum=2.0 * 60.0 * 60.0,
        )
        hold = (
            base
            + _ACTION_BONUS_S.get(candidate.action, 0.0)
            + _ATTENTION_BONUS_S.get(candidate.attention_state, 0.0)
            + (max(0.0, float(candidate.salience)) * 3.0 * 60.0)
        )
        return max(minimum, min(maximum, hold))