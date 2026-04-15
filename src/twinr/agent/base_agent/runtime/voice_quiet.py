# CHANGELOG: 2026-03-27
# BUG-1: Quiet-window expiry no longer depends on wall-clock arithmetic alone; a monotonic runtime deadline now prevents early/late expiry after NTP/manual clock jumps.
# BUG-2: Datetime restore/state parsing now rejects malformed or naive values and normalizes every accepted deadline to UTC, preventing crashes and non-UTC until_utc output.
# BUG-3: Expired/corrupt quiet windows now surface malformed persisted state instead of
#        silently mutating it on read.
# SEC-1: restore_voice_quiet re-enforces the bounded max quiet duration and rejects
#        tampered/corrupt far-future deadlines, closing a practical long-lived
#        voice-wake suppression DoS path.
# IMP-1: voice_quiet_active() now uses a monotonic fast path for hot wake-suppression checks on Pi-class hardware.
# IMP-2: Quiet reasons are normalized to a single bounded printable line before storage/exposure, reducing cross-layer log/UI/prompt poisoning and keeping snapshots stable.
# IMP-3: Added explicit host-runtime typing contracts plus configurable VOICE_QUIET_MAX_MINUTES / VOICE_QUIET_REASON_LIMIT knobs without breaking the public API.

"""Own Twinr's bounded temporary quiet window for voice wake suppression.

The user may explicitly ask Twinr to stay quiet for a while, for example while
TV or radio audio is playing in the room. This module keeps that runtime-owned
state out of the workflow/orchestrator layers and exposes one small API:

- set a bounded quiet-until timestamp
- clear it early
- report whether the quiet window is currently active

The quiet window only suppresses transcript-first voice wake and automatic
follow-up reopening. It does not create additional wake paths, and it does not
remove the explicit button/manual path.
"""

from __future__ import annotations

import os
import re
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol


__all__ = ["VoiceQuietState", "TwinrRuntimeVoiceQuietMixin"]


_PROCESS_TOKEN = f"{os.getpid()}:{time.monotonic_ns()}"
_DEFAULT_MAX_QUIET_MINUTES = 720
_DEFAULT_REASON_LIMIT = 120
_NS_PER_SECOND = 1_000_000_000
_REMAINING_SECONDS_ROUND_UP_NS = _NS_PER_SECOND - 1
_CLOCK_REBASE_TOLERANCE_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class VoiceQuietState:
    """Describe the current temporary voice-quiet window."""

    active: bool = False
    until_utc: str | None = None
    reason: str | None = None
    remaining_seconds: int = 0


class _VoiceQuietRuntimeHelpers(Protocol):
    """Typing-only contract for runtimes hosting this mixin."""

    def _compact_runtime_guidance_text(self, value: object | None, *, limit: int) -> str: ...
    def _parse_aware_utc_datetime(self, value: object | None) -> datetime | None: ...
    def _runtime_context_lock(self) -> AbstractContextManager[Any]: ...
    def _require_int(self, value: object, *, field_name: str, minimum: int) -> int: ...
    def _safe_persist_snapshot(self, *, event_on_error: str) -> None: ...


class TwinrRuntimeVoiceQuietMixin:
    """Provide runtime-owned temporary quiet-window helpers."""

    VOICE_QUIET_MAX_MINUTES = _DEFAULT_MAX_QUIET_MINUTES
    VOICE_QUIET_REASON_LIMIT = _DEFAULT_REASON_LIMIT

    def _voice_quiet_now_utc(self) -> datetime:
        """Return the current wall-clock time in UTC."""

        return datetime.now(timezone.utc)

    def _voice_quiet_now_monotonic_ns(self) -> int:
        """Return a monotonic runtime timestamp in nanoseconds."""

        return time.monotonic_ns()

    def _voice_quiet_max_minutes(self) -> int:
        """Return the bounded maximum quiet-window duration in minutes."""

        value = getattr(self, "VOICE_QUIET_MAX_MINUTES", _DEFAULT_MAX_QUIET_MINUTES)
        return value if isinstance(value, int) and value > 0 else _DEFAULT_MAX_QUIET_MINUTES

    def _voice_quiet_max_duration_ns(self) -> int:
        """Return the bounded maximum quiet-window duration in nanoseconds."""

        return self._voice_quiet_max_minutes() * 60 * _NS_PER_SECOND

    def _voice_quiet_reason_limit(self) -> int:
        """Return the maximum stored/exposed quiet-reason length."""

        value = getattr(self, "VOICE_QUIET_REASON_LIMIT", _DEFAULT_REASON_LIMIT)
        return value if isinstance(value, int) and value > 0 else _DEFAULT_REASON_LIMIT

    def _voice_quiet_isoformat_utc(self, value: datetime) -> str:
        """Format an aware datetime as canonical UTC ISO-8601 text."""

        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    def _timedelta_to_ns(self, value: timedelta) -> int:
        """Convert a timedelta to integer nanoseconds without float rounding."""

        return ((value.days * 24 * 60 * 60) + value.seconds) * _NS_PER_SECOND + value.microseconds * 1000

    def _ns_to_timedelta(self, value_ns: int) -> timedelta:
        """Convert integer nanoseconds to a timedelta."""

        return timedelta(microseconds=max(0, value_ns) // 1000)

    def _voice_quiet_deadlines_differ_materially(self, left: datetime, right: datetime) -> bool:
        """Return whether two UTC deadlines differ enough to justify a persisted rebase."""

        return abs((left - right).total_seconds()) > _CLOCK_REBASE_TOLERANCE_SECONDS

    def _compact_voice_quiet_guidance_text(self, value: object | None, *, limit: int) -> str:
        """Delegate bounded reason compaction through the runtime helper contract."""

        compact = getattr(self, "_compact_runtime_guidance_text", None)
        if not callable(compact):
            return str(value or "").strip()[:limit]
        compact_fn = compact
        return str(compact_fn(value, limit=limit) or "").strip()  # pylint: disable=not-callable

    def _parse_voice_quiet_datetime(self, value: object | None) -> datetime | None:
        """Delegate UTC timestamp parsing through the runtime helper contract."""

        parser = getattr(self, "_parse_aware_utc_datetime", None)
        if not callable(parser):
            return None
        parser_fn = parser
        try:
            parsed = parser_fn(value)  # pylint: disable=not-callable
        except (TypeError, ValueError, OverflowError):
            return None
        return parsed if isinstance(parsed, datetime) else None

    def _normalize_voice_quiet_deadline_utc(self, value: object | None) -> datetime | None:
        """Return a validated aware UTC deadline or None."""

        parsed = self._parse_voice_quiet_datetime(value)
        if parsed is None:
            return None
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        return parsed.astimezone(timezone.utc)

    def _voice_quiet_runtime_lock(self) -> Any:
        """Return the shared runtime context lock context manager."""

        lock_factory = getattr(self, "_runtime_context_lock", None)
        if not callable(lock_factory):
            raise RuntimeError("Voice quiet mode requires the runtime context lock")
        lock_factory_fn = lock_factory
        return lock_factory_fn()  # pylint: disable=not-callable

    def _require_voice_quiet_int(self, value: object, *, field_name: str, minimum: int) -> int:
        """Delegate bounded integer validation through the runtime helper contract."""

        require_int = getattr(self, "_require_int", None)
        if not callable(require_int):
            raise RuntimeError("Voice quiet mode requires integer validation support")
        require_int_fn = require_int
        return int(require_int_fn(value, field_name=field_name, minimum=minimum))  # pylint: disable=not-callable

    def _persist_voice_quiet_snapshot(self) -> None:
        """Persist the runtime snapshot through the shared best-effort helper."""

        persist = getattr(self, "_safe_persist_snapshot", None)
        if not callable(persist):
            return
        persist_fn = persist
        persist_fn(event_on_error="voice_quiet_snapshot_persist_failed")  # pylint: disable=not-callable

    def _sanitize_voice_quiet_reason_text(self, text: str, *, limit: int) -> str:
        """Normalize user-controlled reason text to one bounded printable line."""

        if not text:
            return ""
        sanitized = "".join(char if char.isprintable() else " " for char in text)
        sanitized = re.sub(r"\s+", " ", sanitized).strip()
        return sanitized[:limit]

    def _normalize_voice_quiet_reason(self, reason: object | None) -> str | None:
        """Compact and sanitize the quiet reason for storage/exposure."""

        compact = self._compact_voice_quiet_guidance_text(reason, limit=self._voice_quiet_reason_limit())
        sanitized = self._sanitize_voice_quiet_reason_text(compact, limit=self._voice_quiet_reason_limit())
        return sanitized or None

    def _voice_quiet_has_persisted_fields_unlocked(self) -> bool:
        """Return whether persisted quiet fields are currently populated."""

        return (
            getattr(self, "_voice_quiet_until_utc", None) is not None
            or getattr(self, "_voice_quiet_reason", None) is not None
        )

    def _set_voice_quiet_persisted_unlocked(
        self,
        *,
        until_utc: str | None,
        reason: str | None,
    ) -> None:
        """Write persisted quiet fields while the runtime lock is held."""

        setattr(self, "_voice_quiet_until_utc", until_utc)
        setattr(self, "_voice_quiet_reason", reason)

    def _set_voice_quiet_cache_unlocked(self, *, deadline_monotonic_ns: int | None) -> None:
        """Write runtime-only monotonic cache fields while the runtime lock is held."""

        if deadline_monotonic_ns is None:
            setattr(self, "_voice_quiet_deadline_monotonic_ns", None)
            setattr(self, "_voice_quiet_deadline_monotonic_owner", None)
            return
        setattr(self, "_voice_quiet_deadline_monotonic_ns", int(deadline_monotonic_ns))
        setattr(self, "_voice_quiet_deadline_monotonic_owner", _PROCESS_TOKEN)

    def _set_voice_quiet_unlocked(
        self,
        *,
        until_utc: str | None,
        reason: str | None,
        deadline_monotonic_ns: int | None,
    ) -> None:
        """Write quiet-window state while the runtime lock is held."""

        self._set_voice_quiet_persisted_unlocked(until_utc=until_utc, reason=reason)
        self._set_voice_quiet_cache_unlocked(deadline_monotonic_ns=deadline_monotonic_ns)

    def _clear_voice_quiet_unlocked(self) -> None:
        """Clear quiet-window state while the runtime lock is held."""

        self._set_voice_quiet_unlocked(
            until_utc=None,
            reason=None,
            deadline_monotonic_ns=None,
        )

    def _trusted_voice_quiet_deadline_monotonic_ns_unlocked(self) -> int | None:
        """Return the trusted in-process monotonic deadline when available."""

        owner = getattr(self, "_voice_quiet_deadline_monotonic_owner", None)
        raw_deadline = getattr(self, "_voice_quiet_deadline_monotonic_ns", None)
        if owner != _PROCESS_TOKEN:
            return None
        if not isinstance(raw_deadline, int) or raw_deadline <= 0:
            return None
        return raw_deadline

    def _voice_quiet_status_unlocked(
        self,
        *,
        now_utc: datetime,
        now_monotonic_ns: int,
    ) -> tuple[bool, datetime | None, str | None, int, bool]:
        """Return the fully normalized quiet status while the runtime lock is held."""

        raw_reason = getattr(self, "_voice_quiet_reason", None)
        reason = self._normalize_voice_quiet_reason(raw_reason)
        if raw_reason is not None and reason != raw_reason:
            raise ValueError("voice quiet reason is malformed")

        raw_until_utc = getattr(self, "_voice_quiet_until_utc", None)
        if raw_until_utc is None:
            return False, None, None, 0, False
        stored_deadline_utc = self._normalize_voice_quiet_deadline_utc(raw_until_utc)
        if stored_deadline_utc is None:
            raise ValueError("voice quiet deadline is malformed")

        monotonic_deadline_ns = self._trusted_voice_quiet_deadline_monotonic_ns_unlocked()
        max_duration_ns = self._voice_quiet_max_duration_ns()

        if monotonic_deadline_ns is None:
            remaining_ns = self._timedelta_to_ns(stored_deadline_utc - now_utc)
            if remaining_ns <= 0:
                return False, None, None, 0, False

            if remaining_ns > max_duration_ns:
                raise ValueError("voice quiet deadline exceeds the configured maximum duration")

            monotonic_deadline_ns = now_monotonic_ns + remaining_ns
            self._set_voice_quiet_cache_unlocked(deadline_monotonic_ns=monotonic_deadline_ns)
        else:
            remaining_ns = monotonic_deadline_ns - now_monotonic_ns
            if remaining_ns <= 0:
                return False, None, None, 0, False

        remaining_ns = max(0, monotonic_deadline_ns - now_monotonic_ns)
        remaining_seconds = max(1, (remaining_ns + _REMAINING_SECONDS_ROUND_UP_NS) // _NS_PER_SECOND)
        return True, stored_deadline_utc, reason, int(remaining_seconds), False

    def _voice_quiet_active_fast_unlocked(
        self,
        *,
        now_monotonic_ns: int,
    ) -> tuple[bool | None, bool]:
        """Return a hot-path active result from the monotonic cache when possible."""

        monotonic_deadline_ns = self._trusted_voice_quiet_deadline_monotonic_ns_unlocked()
        if monotonic_deadline_ns is None:
            return None, False
        if monotonic_deadline_ns > now_monotonic_ns:
            return True, False
        return False, False

    def voice_quiet_state(self) -> VoiceQuietState:
        """Return the current temporary voice-quiet state."""

        with self._voice_quiet_runtime_lock():
            active, deadline_utc, reason, remaining_seconds, _persist_needed = self._voice_quiet_status_unlocked(
                now_utc=self._voice_quiet_now_utc(),
                now_monotonic_ns=self._voice_quiet_now_monotonic_ns(),
            )
            if not active or deadline_utc is None:
                state = VoiceQuietState()
            else:
                state = VoiceQuietState(
                    active=True,
                    until_utc=self._voice_quiet_isoformat_utc(deadline_utc),
                    reason=reason,
                    remaining_seconds=remaining_seconds,
                )
        return state

    def voice_quiet_active(self) -> bool:
        """Return whether transcript-first wake should currently stay quiet."""

        with self._voice_quiet_runtime_lock():
            now_monotonic_ns = self._voice_quiet_now_monotonic_ns()
            active, _persist_needed = self._voice_quiet_active_fast_unlocked(
                now_monotonic_ns=now_monotonic_ns,
            )
            if active is None:
                active, _, _, _, _persist_needed = self._voice_quiet_status_unlocked(
                    now_utc=self._voice_quiet_now_utc(),
                    now_monotonic_ns=now_monotonic_ns,
                )
        return bool(active)

    def voice_quiet_until_utc(self) -> str | None:
        """Return the current quiet deadline as UTC ISO-8601 text when active."""

        with self._voice_quiet_runtime_lock():
            active, deadline_utc, _, _, _persist_needed = self._voice_quiet_status_unlocked(
                now_utc=self._voice_quiet_now_utc(),
                now_monotonic_ns=self._voice_quiet_now_monotonic_ns(),
            )
            until_utc = self._voice_quiet_isoformat_utc(deadline_utc) if active and deadline_utc is not None else None
        return until_utc

    def set_voice_quiet_minutes(
        self,
        *,
        minutes: int,
        reason: str | None = None,
    ) -> VoiceQuietState:
        """Start or replace a bounded temporary voice-quiet window."""

        safe_minutes = min(
            self._require_voice_quiet_int(minutes, field_name="minutes", minimum=1),
            self._voice_quiet_max_minutes(),
        )
        now_utc = self._voice_quiet_now_utc()
        now_monotonic_ns = self._voice_quiet_now_monotonic_ns()
        deadline_utc = now_utc + timedelta(minutes=safe_minutes)
        deadline_monotonic_ns = now_monotonic_ns + safe_minutes * 60 * _NS_PER_SECOND

        with self._voice_quiet_runtime_lock():
            self._set_voice_quiet_unlocked(
                until_utc=self._voice_quiet_isoformat_utc(deadline_utc),
                reason=self._normalize_voice_quiet_reason(reason),
                deadline_monotonic_ns=deadline_monotonic_ns,
            )

        self._persist_voice_quiet_snapshot()
        return self.voice_quiet_state()

    def clear_voice_quiet(self) -> VoiceQuietState:
        """Clear the current quiet window and return the inactive state."""

        with self._voice_quiet_runtime_lock():
            self._clear_voice_quiet_unlocked()
        self._persist_voice_quiet_snapshot()
        return VoiceQuietState()

    def reset_voice_quiet(self) -> None:
        """Clear the quiet window without triggering a snapshot write."""

        with self._voice_quiet_runtime_lock():
            self._clear_voice_quiet_unlocked()

    def restore_voice_quiet(
        self,
        *,
        until_utc: object | None,
        reason: object | None,
    ) -> None:
        """Restore the quiet window from persisted runtime state."""

        now_utc = self._voice_quiet_now_utc()
        now_monotonic_ns = self._voice_quiet_now_monotonic_ns()

        with self._voice_quiet_runtime_lock():
            restored_deadline_utc = self._normalize_voice_quiet_deadline_utc(until_utc)
            normalized_reason = self._normalize_voice_quiet_reason(reason)
            if reason is not None and normalized_reason != reason:
                raise ValueError("voice quiet reason is malformed")

            if restored_deadline_utc is None:
                if until_utc is not None:
                    raise ValueError("voice quiet deadline is malformed")
                self._clear_voice_quiet_unlocked()
            else:
                remaining_ns = self._timedelta_to_ns(restored_deadline_utc - now_utc)
                if remaining_ns <= 0:
                    self._clear_voice_quiet_unlocked()
                else:
                    max_duration_ns = self._voice_quiet_max_duration_ns()
                    if remaining_ns > max_duration_ns:
                        raise ValueError("voice quiet deadline exceeds the configured maximum duration")

                    self._set_voice_quiet_unlocked(
                        until_utc=self._voice_quiet_isoformat_utc(restored_deadline_utc),
                        reason=normalized_reason,
                        deadline_monotonic_ns=now_monotonic_ns + remaining_ns,
                    )
