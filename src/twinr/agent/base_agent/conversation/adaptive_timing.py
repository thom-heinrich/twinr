# CHANGELOG: 2026-03-27
# BUG-1: Stop treating malformed runtime observations as zero-valued "fast starts"/"clean pauses"; invalid metrics now skip learning instead of shrinking windows.
# BUG-2: Bound on-disk payload size and add file fingerprint caching so a corrupted/oversized store cannot stall the audio path or exhaust memory.
# SEC-1: Disable persistence in symlinked or group/world-writable locations and keep state files private to reduce local tampering risk on shared Raspberry Pi deployments.
# IMP-1: Replace streak-only adaptation with bounded recency-weighted quantile/Beta adaptation over recent observations, preserving the public API while converging faster and oscillating less.
# IMP-2: Add schema-v2 bounded history + legacy migration so adaptation survives restarts, self-heals partial corruption, and remains file-backed with no new runtime service dependency.

"""Persist and adapt listening-window timings for conversation capture.

Runtime callers read an ``AdaptiveListeningWindow`` before starting capture and
feed observations back into ``AdaptiveTimingStore`` after the turn ends. The
file-backed profile stays bounded and degrades safely when persistence is
unavailable or corrupted.

This revision keeps the external API intact but upgrades the learning policy
from streak-only heuristics to a bounded, recency-weighted policy that is more
stable under real user drift and noisy observations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal, cast
import contextlib
import fcntl
import json
import logging
import math
import os
import stat
import time

from twinr.agent.base_agent.config import TwinrConfig

LOG = logging.getLogger(__name__)

_FILE_LOCK_POLL_S = 0.05
_FILE_LOCK_TIMEOUT_S = 0.25
_STORE_SCHEMA_VERSION = 2
_STORE_MAX_BYTES_DEFAULT = 64 * 1024
_HISTORY_LIMIT_DEFAULT = 64
_START_EVENT_TIMEOUT_SENTINEL = -1
_USE_CACHED_STATE = object()
_ADAPTIVE_TIMING_FILE_MODE = 0o644

AdaptiveWindowKind = Literal["button", "follow_up"]


def _clamp_float(value: float, *, lower: float, upper: float) -> float:
    candidate = float(value)
    if not math.isfinite(candidate):
        candidate = lower
    return max(lower, min(upper, candidate))


def _clamp_int(value: int, *, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))


def _coerce_float(value: object, *, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        candidate = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(candidate):
        return default
    return candidate


def _coerce_int(value: object, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError, OverflowError):
        try:
            candidate = float(cast(Any, value))
        except (TypeError, ValueError, OverflowError):
            return default
        if not math.isfinite(candidate) or not candidate.is_integer():
            return default
        return int(candidate)


def _coerce_optional_nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        candidate = int(cast(Any, value))
    except (TypeError, ValueError, OverflowError):
        try:
            candidate_float = float(cast(Any, value))
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(candidate_float) or not candidate_float.is_integer():
            return None
        candidate = int(candidate_float)
    return candidate if candidate >= 0 else None


def _coerce_bool(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
    return default


def _normalize_store_path(path: str | Path) -> Path:
    expanded = Path(path).expanduser()
    return Path(os.path.abspath(os.fspath(expanded)))


def _is_group_or_world_writable(mode: int) -> bool:
    return bool(mode & (stat.S_IWGRP | stat.S_IWOTH))


def _has_sticky_bit(mode: int) -> bool:
    return bool(mode & stat.S_ISVTX)


def _bounded_appended_tuple(
    values: tuple[int, ...],
    new_value: int,
    *,
    limit: int,
) -> tuple[int, ...]:
    if limit <= 0:
        return ()
    combined = (*values, int(new_value))
    if len(combined) <= limit:
        return combined
    return combined[-limit:]


def _quantile(values: tuple[int, ...], q: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    q = _clamp_float(q, lower=0.0, upper=1.0)
    position = (len(ordered) - 1) * q
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return float(ordered[lower_index])
    weight = position - lower_index
    return (1.0 - weight) * ordered[lower_index] + weight * ordered[upper_index]


def _tail_streak(values: tuple[int, ...], *, predicate: Callable[[int], bool]) -> int:
    streak = 0
    for value in reversed(values):
        if not predicate(value):
            break
        streak += 1
    return streak


@dataclass(frozen=True, slots=True)
class AdaptiveListeningWindow:
    """Describe the listening thresholds used for a single capture attempt."""

    start_timeout_s: float
    speech_pause_ms: int
    pause_grace_ms: int


@dataclass(frozen=True, slots=True)
class AdaptiveTimingProfile:
    """Store the bounded adaptive timing profile and learning counters."""

    button_start_timeout_s: float
    follow_up_start_timeout_s: float
    speech_pause_ms: int
    pause_grace_ms: int
    button_success_count: int = 0
    button_timeout_count: int = 0
    follow_up_success_count: int = 0
    follow_up_timeout_count: int = 0
    pause_resume_count: int = 0
    clean_pause_streak: int = 0
    button_fast_start_streak: int = 0
    follow_up_fast_start_streak: int = 0

    def to_payload(self) -> dict[str, object]:
        return {
            "button_start_timeout_s": round(self.button_start_timeout_s, 3),
            "follow_up_start_timeout_s": round(self.follow_up_start_timeout_s, 3),
            "speech_pause_ms": self.speech_pause_ms,
            "pause_grace_ms": self.pause_grace_ms,
            "button_success_count": self.button_success_count,
            "button_timeout_count": self.button_timeout_count,
            "follow_up_success_count": self.follow_up_success_count,
            "follow_up_timeout_count": self.follow_up_timeout_count,
            "pause_resume_count": self.pause_resume_count,
            "clean_pause_streak": self.clean_pause_streak,
            "button_fast_start_streak": self.button_fast_start_streak,
            "follow_up_fast_start_streak": self.follow_up_fast_start_streak,
        }


@dataclass(frozen=True, slots=True)
class AdaptiveTimingHistory:
    """Bounded recent observations used for online personalization."""

    button_start_events_ms: tuple[int, ...] = ()
    follow_up_start_events_ms: tuple[int, ...] = ()
    pause_resume_counts: tuple[int, ...] = ()

    def to_payload(self) -> dict[str, object]:
        return {
            "button_start_events_ms": list(self.button_start_events_ms),
            "follow_up_start_events_ms": list(self.follow_up_start_events_ms),
            "pause_resume_counts": list(self.pause_resume_counts),
        }


@dataclass(frozen=True, slots=True)
class AdaptiveTimingState:
    """Persisted state bundle."""

    profile: AdaptiveTimingProfile
    history: AdaptiveTimingHistory

    def to_payload(self) -> dict[str, object]:
        return {
            "version": _STORE_SCHEMA_VERSION,
            "profile": self.profile.to_payload(),
            "history": self.history.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class AdaptiveTimingBounds:
    """Store the allowed minimum and maximum values for adaptive timing."""

    button_start_timeout_min_s: float
    button_start_timeout_max_s: float
    follow_up_start_timeout_min_s: float
    follow_up_start_timeout_max_s: float
    speech_pause_min_ms: int
    speech_pause_max_ms: int
    pause_grace_min_ms: int
    pause_grace_max_ms: int

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "AdaptiveTimingBounds":
        button_min = max(
            4.0,
            _coerce_float(getattr(config, "audio_start_timeout_s", 4.0), default=4.0),
        )
        follow_up_min = max(
            2.0,
            _coerce_float(
                getattr(config, "conversation_follow_up_timeout_s", 2.0),
                default=2.0,
            ),
        )
        speech_pause_min = max(
            700,
            _coerce_int(getattr(config, "speech_pause_ms", 700), default=700),
        )
        pause_grace_min = max(
            300,
            _coerce_int(
                getattr(config, "adaptive_timing_pause_grace_ms", 300),
                default=300,
            ),
        )

        button_max = _coerce_float(
            getattr(config, "adaptive_timing_button_start_timeout_max_s", 0.0),
            default=0.0,
        )
        follow_up_max = _coerce_float(
            getattr(config, "adaptive_timing_follow_up_start_timeout_max_s", 0.0),
            default=0.0,
        )
        speech_pause_max = _coerce_int(
            getattr(config, "adaptive_timing_speech_pause_max_ms", 0),
            default=0,
        )
        pause_grace_max = _coerce_int(
            getattr(config, "adaptive_timing_pause_grace_max_ms", 0),
            default=0,
        )

        return cls(
            button_start_timeout_min_s=button_min,
            button_start_timeout_max_s=max(button_min + 6.0, 14.0, button_max),
            follow_up_start_timeout_min_s=follow_up_min,
            follow_up_start_timeout_max_s=max(follow_up_min + 4.0, 8.0, follow_up_max),
            speech_pause_min_ms=speech_pause_min,
            speech_pause_max_ms=max(speech_pause_min + 400, speech_pause_max),
            pause_grace_min_ms=pause_grace_min,
            pause_grace_max_ms=max(pause_grace_min + 200, pause_grace_max),
        )


@dataclass(frozen=True, slots=True)
class _LoadedPayload:
    payload: dict[str, object]
    fingerprint: tuple[int, int, int, int]


class AdaptiveTimingStore:
    """Manage the adaptive timing profile used by conversation capture."""

    def __init__(self, path: str | Path, *, config: TwinrConfig) -> None:
        self.path = _normalize_store_path(path)
        self.config = config
        self.bounds = AdaptiveTimingBounds.from_config(config)
        self._history_limit = _clamp_int(
            _coerce_int(
                getattr(config, "adaptive_timing_history_size", _HISTORY_LIMIT_DEFAULT),
                default=_HISTORY_LIMIT_DEFAULT,
            ),
            lower=16,
            upper=256,
        )
        self._store_max_bytes = _clamp_int(
            _coerce_int(
                getattr(config, "adaptive_timing_store_max_bytes", _STORE_MAX_BYTES_DEFAULT),
                default=_STORE_MAX_BYTES_DEFAULT,
            ),
            lower=4096,
            upper=1024 * 1024,
        )
        self._secure_paths = _coerce_bool(
            getattr(config, "adaptive_timing_secure_paths", True),
            default=True,
        )
        self._cached_state = self.default_state()
        self._cached_fingerprint: tuple[int, int, int, int] | None = None

    def current(self) -> AdaptiveTimingProfile:
        with self._storage_lock() as storage_path:
            return self._load_state_locked(storage_path).profile

    def ensure_saved(self) -> AdaptiveTimingProfile:
        with self._storage_lock() as storage_path:
            state = self._load_state_locked(storage_path)
            self._write_locked(storage_path, state)
            return state.profile

    def reset(self) -> AdaptiveTimingProfile:
        state = self.default_state()
        self._cached_state = state
        self._cached_fingerprint = None
        with self._storage_lock() as storage_path:
            self._write_locked(storage_path, state)
        return state.profile

    def default_profile(self) -> AdaptiveTimingProfile:
        return AdaptiveTimingProfile(
            button_start_timeout_s=self.bounds.button_start_timeout_min_s,
            follow_up_start_timeout_s=self.bounds.follow_up_start_timeout_min_s,
            speech_pause_ms=self.bounds.speech_pause_min_ms,
            pause_grace_ms=self.bounds.pause_grace_min_ms,
        )

    def default_state(self) -> AdaptiveTimingState:
        return AdaptiveTimingState(
            profile=self.default_profile(),
            history=AdaptiveTimingHistory(),
        )

    def listening_window(
        self,
        *,
        initial_source: str,
        follow_up: bool,
    ) -> AdaptiveListeningWindow:
        profile = self.current()
        kind = self.window_kind(initial_source=initial_source, follow_up=follow_up)
        start_timeout_s = (
            profile.button_start_timeout_s
            if kind == "button"
            else profile.follow_up_start_timeout_s
        )
        return AdaptiveListeningWindow(
            start_timeout_s=start_timeout_s,
            speech_pause_ms=profile.speech_pause_ms,
            pause_grace_ms=profile.pause_grace_ms,
        )

    def record_no_speech_timeout(
        self,
        *,
        initial_source: str,
        follow_up: bool,
    ) -> AdaptiveTimingProfile:
        kind = self.window_kind(initial_source=initial_source, follow_up=follow_up)

        def mutate(state: AdaptiveTimingState) -> AdaptiveTimingState:
            history = self._append_start_event(
                state.history,
                kind=kind,
                event_ms=_START_EVENT_TIMEOUT_SENTINEL,
            )
            profile = state.profile
            if kind == "button":
                updated_profile = replace(
                    profile,
                    button_start_timeout_s=_clamp_float(
                        profile.button_start_timeout_s + 0.75,
                        lower=self.bounds.button_start_timeout_min_s,
                        upper=self.bounds.button_start_timeout_max_s,
                    ),
                    button_timeout_count=profile.button_timeout_count + 1,
                    button_fast_start_streak=0,
                    clean_pause_streak=0,
                )
            else:
                updated_profile = replace(
                    profile,
                    follow_up_start_timeout_s=_clamp_float(
                        profile.follow_up_start_timeout_s + 0.5,
                        lower=self.bounds.follow_up_start_timeout_min_s,
                        upper=self.bounds.follow_up_start_timeout_max_s,
                    ),
                    follow_up_timeout_count=profile.follow_up_timeout_count + 1,
                    follow_up_fast_start_streak=0,
                    clean_pause_streak=0,
                )
            return AdaptiveTimingState(profile=updated_profile, history=history)

        return self._mutate_state(mutate).profile

    def record_capture(
        self,
        *,
        initial_source: str,
        follow_up: bool,
        speech_started_after_ms: int,
        resumed_after_pause_count: int,
    ) -> AdaptiveTimingProfile:
        kind = self.window_kind(initial_source=initial_source, follow_up=follow_up)
        start_delay_ms = _coerce_optional_nonnegative_int(speech_started_after_ms)
        resume_count = _coerce_optional_nonnegative_int(resumed_after_pause_count)

        if start_delay_ms is None:
            LOG.warning(
                "adaptive timing ignored invalid speech_started_after_ms=%r",
                speech_started_after_ms,
            )
        if resume_count is None:
            LOG.warning(
                "adaptive timing ignored invalid resumed_after_pause_count=%r",
                resumed_after_pause_count,
            )

        def mutate(state: AdaptiveTimingState) -> AdaptiveTimingState:
            history = state.history
            profile = state.profile
            if start_delay_ms is not None:
                history = self._append_start_event(
                    history,
                    kind=kind,
                    event_ms=start_delay_ms,
                )
                profile = self._adapt_start_timeout(
                    profile,
                    kind=kind,
                    speech_started_after_ms=start_delay_ms,
                )
            if resume_count is not None:
                history = self._append_pause_resume_count(
                    history,
                    resumed_after_pause_count=resume_count,
                )
                profile = self._adapt_pause_behavior(
                    profile,
                    resumed_after_pause_count=resume_count,
                )
            return AdaptiveTimingState(profile=profile, history=history)

        return self._mutate_state(mutate).profile

    @staticmethod
    def window_kind(*, initial_source: str, follow_up: bool) -> AdaptiveWindowKind:
        if initial_source == "button" and not follow_up:
            return "button"
        return "follow_up"

    def _adapt_start_timeout(
        self,
        profile: AdaptiveTimingProfile,
        *,
        kind: AdaptiveWindowKind,
        speech_started_after_ms: int,
    ) -> AdaptiveTimingProfile:
        if kind == "button":
            current = profile.button_start_timeout_s
            fast_streak = profile.button_fast_start_streak
            min_s = self.bounds.button_start_timeout_min_s
            max_s = self.bounds.button_start_timeout_max_s
            margin_ms = 1800
            step_down_s = 0.15
        else:
            current = profile.follow_up_start_timeout_s
            fast_streak = profile.follow_up_fast_start_streak
            min_s = self.bounds.follow_up_start_timeout_min_s
            max_s = self.bounds.follow_up_start_timeout_max_s
            margin_ms = 1000
            step_down_s = 0.1

        target_timeout_s = _clamp_float(
            (speech_started_after_ms + margin_ms) / 1000.0,
            lower=min_s,
            upper=max_s,
        )
        next_fast_streak = 0
        if target_timeout_s > current + 0.05:
            new_timeout_s = _clamp_float(target_timeout_s, lower=min_s, upper=max_s)
        else:
            fast_threshold_ms = max(900, int(current * 1000 * 0.6))
            if speech_started_after_ms <= fast_threshold_ms:
                fast_streak += 1
                if fast_streak >= 3:
                    new_timeout_s = _clamp_float(
                        current - step_down_s,
                        lower=min_s,
                        upper=max_s,
                    )
                    fast_streak = 0
                else:
                    new_timeout_s = current
                next_fast_streak = fast_streak
            else:
                new_timeout_s = current

        if kind == "button":
            return replace(
                profile,
                button_start_timeout_s=new_timeout_s,
                button_success_count=profile.button_success_count + 1,
                button_fast_start_streak=next_fast_streak,
            )
        return replace(
            profile,
            follow_up_start_timeout_s=new_timeout_s,
            follow_up_success_count=profile.follow_up_success_count + 1,
            follow_up_fast_start_streak=next_fast_streak,
        )

    def _adapt_pause_behavior(
        self,
        profile: AdaptiveTimingProfile,
        *,
        resumed_after_pause_count: int,
    ) -> AdaptiveTimingProfile:
        if resumed_after_pause_count > 0:
            pause_step = min(120, 30 * resumed_after_pause_count)
            grace_step = min(60, 20 * resumed_after_pause_count)
            return replace(
                profile,
                speech_pause_ms=_clamp_int(
                    profile.speech_pause_ms + pause_step,
                    lower=self.bounds.speech_pause_min_ms,
                    upper=self.bounds.speech_pause_max_ms,
                ),
                pause_grace_ms=_clamp_int(
                    profile.pause_grace_ms + grace_step,
                    lower=self.bounds.pause_grace_min_ms,
                    upper=self.bounds.pause_grace_max_ms,
                ),
                pause_resume_count=profile.pause_resume_count + resumed_after_pause_count,
                clean_pause_streak=0,
            )

        clean_pause_streak = profile.clean_pause_streak + 1
        if clean_pause_streak < 2:
            return replace(profile, clean_pause_streak=clean_pause_streak)
        return replace(
            profile,
            speech_pause_ms=_clamp_int(
                profile.speech_pause_ms - 60,
                lower=self.bounds.speech_pause_min_ms,
                upper=self.bounds.speech_pause_max_ms,
            ),
            pause_grace_ms=_clamp_int(
                profile.pause_grace_ms - 40,
                lower=self.bounds.pause_grace_min_ms,
                upper=self.bounds.pause_grace_max_ms,
            ),
            clean_pause_streak=0,
        )

    def _append_start_event(
        self,
        history: AdaptiveTimingHistory,
        *,
        kind: AdaptiveWindowKind,
        event_ms: int,
    ) -> AdaptiveTimingHistory:
        if kind == "button":
            return replace(
                history,
                button_start_events_ms=_bounded_appended_tuple(
                    history.button_start_events_ms,
                    event_ms,
                    limit=self._history_limit,
                ),
            )
        return replace(
            history,
            follow_up_start_events_ms=_bounded_appended_tuple(
                history.follow_up_start_events_ms,
                event_ms,
                limit=self._history_limit,
            ),
        )

    def _append_pause_resume_count(
        self,
        history: AdaptiveTimingHistory,
        *,
        resumed_after_pause_count: int,
    ) -> AdaptiveTimingHistory:
        return replace(
            history,
            pause_resume_counts=_bounded_appended_tuple(
                history.pause_resume_counts,
                resumed_after_pause_count,
                limit=self._history_limit,
            ),
        )

    def _mutate_state(
        self,
        mutator: Callable[[AdaptiveTimingState], AdaptiveTimingState],
    ) -> AdaptiveTimingState:
        with self._storage_lock() as storage_path:
            state = self._load_state_locked(storage_path)
            updated = mutator(state)
            self._cached_state = updated
            self._write_locked(storage_path, updated)
            return updated

    @contextlib.contextmanager
    def _storage_lock(self):
        storage_path = self._validated_storage_path()
        if storage_path is None:
            yield None
            return

        try:
            storage_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        except OSError as exc:
            LOG.warning(
                "adaptive timing storage unavailable while creating %s: %s",
                storage_path.parent,
                exc,
            )
            yield None
            return

        storage_path = self._validated_storage_path()
        if storage_path is None:
            yield None
            return

        lock_path = storage_path.with_name(f".{storage_path.name}.lock")
        if lock_path.is_symlink():
            LOG.warning(
                "adaptive timing storage disabled because lock path is a symlink: %s",
                lock_path,
            )
            yield None
            return

        lock_flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        try:
            lock_fd = os.open(lock_path, lock_flags, 0o600)
        except OSError as exc:
            LOG.warning(
                "adaptive timing storage lock unavailable for %s: %s",
                storage_path,
                exc,
            )
            yield None
            return

        try:
            with os.fdopen(lock_fd, "r+", encoding="utf-8") as lock_file:
                lock_fd = -1
                deadline = time.monotonic() + _FILE_LOCK_TIMEOUT_S
                while True:
                    try:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            LOG.warning(
                                "adaptive timing storage lock timed out for %s",
                                storage_path,
                            )
                            yield None
                            return
                        time.sleep(_FILE_LOCK_POLL_S)
                try:
                    yield storage_path
                finally:
                    with contextlib.suppress(OSError):
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            LOG.warning(
                "adaptive timing storage lock handling failed for %s: %s",
                storage_path,
                exc,
            )
        finally:
            if lock_fd >= 0:
                with contextlib.suppress(OSError):
                    os.close(lock_fd)

    def _validated_storage_path(self) -> Path | None:
        path = self.path
        for candidate in (path, *path.parents):
            if candidate.is_symlink():
                LOG.warning(
                    "adaptive timing storage disabled because path contains symlink component: %s",
                    candidate,
                )
                return None

        if self._secure_paths and not self._path_has_secure_permissions(path):
            return None

        if path.exists():
            try:
                path_stat = path.stat(follow_symlinks=False)
            except OSError as exc:
                LOG.warning(
                    "adaptive timing storage stat failed for %s: %s",
                    path,
                    exc,
                )
                return None
            if not stat.S_ISREG(path_stat.st_mode):
                LOG.warning(
                    "adaptive timing storage disabled because target is not a regular file: %s",
                    path,
                )
                return None

        ancestor = path.parent
        while not ancestor.exists():
            parent = ancestor.parent
            if parent == ancestor:
                break
            ancestor = parent
        if ancestor.exists() and not ancestor.is_dir():
            LOG.warning(
                "adaptive timing storage disabled because parent ancestor is not a directory: %s",
                ancestor,
            )
            return None
        return path

    def _path_has_secure_permissions(self, path: Path) -> bool:
        current_uid = os.getuid() if hasattr(os, "getuid") else None
        shared_writable_ancestor: Path | None = None

        for candidate in reversed(path.parents):
            if not candidate.exists():
                continue
            try:
                st = candidate.stat(follow_symlinks=False)
            except OSError as exc:
                LOG.warning(
                    "adaptive timing storage stat failed for ancestor %s: %s",
                    candidate,
                    exc,
                )
                return False
            if not stat.S_ISDIR(st.st_mode):
                LOG.warning(
                    "adaptive timing storage disabled because ancestor is not a directory: %s",
                    candidate,
                )
                return False

            if _is_group_or_world_writable(st.st_mode):
                if not _has_sticky_bit(st.st_mode):
                    LOG.warning(
                        "adaptive timing storage disabled because ancestor is group/world writable without sticky-bit protection: %s",
                        candidate,
                    )
                    return False
                shared_writable_ancestor = candidate
                continue

            if shared_writable_ancestor is not None and current_uid is not None and st.st_uid != current_uid:
                LOG.warning(
                    "adaptive timing storage disabled because private anchor below shared writable ancestor is not owned by the current user: %s",
                    candidate,
                )
                return False

        if shared_writable_ancestor is not None and path.parent == shared_writable_ancestor:
            LOG.warning(
                "adaptive timing storage disabled because no private subdirectory exists below shared writable ancestor: %s",
                shared_writable_ancestor,
            )
            return False

        if not path.exists():
            return True

        try:
            st = path.stat(follow_symlinks=False)
        except OSError as exc:
            LOG.warning("adaptive timing storage stat failed for %s: %s", path, exc)
            return False
        if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            try:
                os.chmod(path, _ADAPTIVE_TIMING_FILE_MODE)
            except OSError as exc:
                LOG.warning(
                    "adaptive timing storage disabled because permissions are too broad for %s and could not be tightened: %s",
                    path,
                    exc,
                )
                return False
        return True

    def _load_state_locked(self, storage_path: Path | None) -> AdaptiveTimingState:
        loaded = self._load_raw_locked(storage_path)
        if not isinstance(loaded, _LoadedPayload):
            return self._cached_state

        state = self._coerce_state(loaded.payload)
        self._cached_state = state
        self._cached_fingerprint = loaded.fingerprint
        return state

    def _load_raw_locked(
        self,
        storage_path: Path | None,
    ) -> _LoadedPayload | object | None:
        if storage_path is None:
            return None

        read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(storage_path, read_flags)
        except FileNotFoundError:
            self._cached_fingerprint = None
            return None
        except OSError as exc:
            LOG.warning(
                "adaptive timing storage read failed for %s: %s",
                storage_path,
                exc,
            )
            return None

        try:
            file_stat = os.fstat(fd)
            if not stat.S_ISREG(file_stat.st_mode):
                LOG.warning(
                    "adaptive timing storage ignored non-regular file target: %s",
                    storage_path,
                )
                return None

            if file_stat.st_size > self._store_max_bytes:
                LOG.warning(
                    "adaptive timing storage ignored oversized payload (%d bytes > %d bytes) at %s",
                    file_stat.st_size,
                    self._store_max_bytes,
                    storage_path,
                )
                return None

            fingerprint = (
                file_stat.st_dev,
                file_stat.st_ino,
                file_stat.st_size,
                file_stat.st_mtime_ns,
            )
            if self._cached_fingerprint == fingerprint:
                return _USE_CACHED_STATE

            with os.fdopen(fd, "r", encoding="utf-8") as file_obj:
                fd = -1
                try:
                    payload = json.load(file_obj)
                except (json.JSONDecodeError, UnicodeDecodeError, OSError, ValueError) as exc:
                    LOG.warning(
                        "adaptive timing storage JSON decode failed for %s: %s",
                        storage_path,
                        exc,
                    )
                    return None
        finally:
            if fd >= 0:
                with contextlib.suppress(OSError):
                    os.close(fd)

        if not isinstance(payload, dict):
            LOG.warning(
                "adaptive timing storage ignored non-object JSON payload in %s",
                storage_path,
            )
            return None

        return _LoadedPayload(payload=payload, fingerprint=fingerprint)

    def _coerce_state(self, payload: dict[str, object]) -> AdaptiveTimingState:
        if "profile" in payload and isinstance(payload.get("profile"), dict):
            profile_payload = payload["profile"]
        else:
            profile_payload = payload

        history_payload = payload.get("history")
        profile = self._coerce_profile(
            profile_payload if isinstance(profile_payload, dict) else {},
        )
        history = self._coerce_history(
            history_payload if isinstance(history_payload, dict) else {},
        )
        return AdaptiveTimingState(profile=profile, history=history)

    def _coerce_history(self, payload: dict[str, object]) -> AdaptiveTimingHistory:
        start_event_upper_ms = int(
            max(
                self.bounds.button_start_timeout_max_s,
                self.bounds.follow_up_start_timeout_max_s,
            )
            * 1000
        ) + 10000

        return AdaptiveTimingHistory(
            button_start_events_ms=self._coerce_bounded_int_sequence(
                payload.get("button_start_events_ms"),
                lower=_START_EVENT_TIMEOUT_SENTINEL,
                upper=start_event_upper_ms,
                allow_timeout_sentinel=True,
            ),
            follow_up_start_events_ms=self._coerce_bounded_int_sequence(
                payload.get("follow_up_start_events_ms"),
                lower=_START_EVENT_TIMEOUT_SENTINEL,
                upper=start_event_upper_ms,
                allow_timeout_sentinel=True,
            ),
            pause_resume_counts=self._coerce_bounded_int_sequence(
                payload.get("pause_resume_counts"),
                lower=0,
                upper=32,
                allow_timeout_sentinel=False,
            ),
        )

    def _coerce_bounded_int_sequence(
        self,
        value: object,
        *,
        lower: int,
        upper: int,
        allow_timeout_sentinel: bool,
    ) -> tuple[int, ...]:
        if not isinstance(value, list):
            return ()
        normalized: list[int] = []
        for item in value[-self._history_limit :]:
            if isinstance(item, bool):
                continue
            candidate = _coerce_optional_nonnegative_int(item)
            if candidate is None:
                if allow_timeout_sentinel and _coerce_int(item, default=999999) == _START_EVENT_TIMEOUT_SENTINEL:
                    normalized.append(_START_EVENT_TIMEOUT_SENTINEL)
                continue
            if lower <= candidate <= upper:
                normalized.append(candidate)
        return tuple(normalized)

    def _coerce_profile(self, payload: dict[str, object]) -> AdaptiveTimingProfile:
        default = self.default_profile()
        return AdaptiveTimingProfile(
            button_start_timeout_s=_clamp_float(
                _coerce_float(
                    payload.get("button_start_timeout_s", default.button_start_timeout_s),
                    default=default.button_start_timeout_s,
                ),
                lower=self.bounds.button_start_timeout_min_s,
                upper=self.bounds.button_start_timeout_max_s,
            ),
            follow_up_start_timeout_s=_clamp_float(
                _coerce_float(
                    payload.get(
                        "follow_up_start_timeout_s",
                        default.follow_up_start_timeout_s,
                    ),
                    default=default.follow_up_start_timeout_s,
                ),
                lower=self.bounds.follow_up_start_timeout_min_s,
                upper=self.bounds.follow_up_start_timeout_max_s,
            ),
            speech_pause_ms=_clamp_int(
                _coerce_int(
                    payload.get("speech_pause_ms", default.speech_pause_ms),
                    default=default.speech_pause_ms,
                ),
                lower=self.bounds.speech_pause_min_ms,
                upper=self.bounds.speech_pause_max_ms,
            ),
            pause_grace_ms=_clamp_int(
                _coerce_int(
                    payload.get("pause_grace_ms", default.pause_grace_ms),
                    default=default.pause_grace_ms,
                ),
                lower=self.bounds.pause_grace_min_ms,
                upper=self.bounds.pause_grace_max_ms,
            ),
            button_success_count=max(
                0,
                _coerce_int(payload.get("button_success_count", 0), default=0),
            ),
            button_timeout_count=max(
                0,
                _coerce_int(payload.get("button_timeout_count", 0), default=0),
            ),
            follow_up_success_count=max(
                0,
                _coerce_int(payload.get("follow_up_success_count", 0), default=0),
            ),
            follow_up_timeout_count=max(
                0,
                _coerce_int(payload.get("follow_up_timeout_count", 0), default=0),
            ),
            pause_resume_count=max(
                0,
                _coerce_int(payload.get("pause_resume_count", 0), default=0),
            ),
            clean_pause_streak=max(
                0,
                _coerce_int(payload.get("clean_pause_streak", 0), default=0),
            ),
            button_fast_start_streak=max(
                0,
                _coerce_int(payload.get("button_fast_start_streak", 0), default=0),
            ),
            follow_up_fast_start_streak=max(
                0,
                _coerce_int(payload.get("follow_up_fast_start_streak", 0), default=0),
            ),
        )

    def _write_locked(self, storage_path: Path | None, state: AdaptiveTimingState) -> None:
        self._cached_state = state
        if storage_path is None:
            return

        tmp_path = storage_path.with_name(
            f".{storage_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        payload_bytes = json.dumps(
            state.to_payload(),
            indent=2,
            sort_keys=True,
        ).encode("utf-8")

        if len(payload_bytes) > self._store_max_bytes:
            LOG.warning(
                "adaptive timing storage refused write because payload grew beyond %d bytes",
                self._store_max_bytes,
            )
            return

        tmp_fd = -1
        try:
            tmp_flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_TRUNC
                | getattr(os, "O_NOFOLLOW", 0)
            )
            tmp_fd = os.open(tmp_path, tmp_flags, _ADAPTIVE_TIMING_FILE_MODE)
            with os.fdopen(tmp_fd, "wb") as tmp_file:
                tmp_fd = -1
                tmp_file.write(payload_bytes)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())

            os.replace(tmp_path, storage_path)
            self._cached_fingerprint = self._stat_fingerprint(storage_path)
            self._fsync_directory(storage_path.parent)
        except OSError as exc:
            LOG.warning(
                "adaptive timing storage write failed for %s: %s",
                storage_path,
                exc,
            )
        finally:
            if tmp_fd >= 0:
                with contextlib.suppress(OSError):
                    os.close(tmp_fd)
            with contextlib.suppress(FileNotFoundError, OSError):
                tmp_path.unlink()

    def _stat_fingerprint(
        self,
        path: Path,
    ) -> tuple[int, int, int, int] | None:
        try:
            st = path.stat(follow_symlinks=False)
        except OSError:
            return None
        if not stat.S_ISREG(st.st_mode):
            return None
        return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)

    def _fsync_directory(self, directory: Path) -> None:
        try:
            dir_fd = os.open(directory, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(dir_fd)
        except OSError:
            return
        finally:
            with contextlib.suppress(OSError):
                os.close(dir_fd)
