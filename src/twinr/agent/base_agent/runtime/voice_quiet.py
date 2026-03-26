"""Own Twinr's bounded temporary quiet window for voice wake suppression.

The user may explicitly ask Twinr to stay quiet for a while, for example while
TV or radio audio is playing in the room. This module keeps that runtime-owned
state out of the workflow/orchestrator layers and exposes one small API:

- set a bounded quiet-until timestamp
- clear it early
- report whether the quiet window is currently active

The quiet window only suppresses transcript-first voice wake and automatic
follow-up reopening. It does not invent fallback wake paths, and it does not
remove the explicit button/manual path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any


_DEFAULT_MAX_QUIET_MINUTES = 720


@dataclass(frozen=True, slots=True)
class VoiceQuietState:
    """Describe the current temporary voice-quiet window."""

    active: bool = False
    until_utc: str | None = None
    reason: str | None = None
    remaining_seconds: int = 0


class TwinrRuntimeVoiceQuietMixin:
    """Provide runtime-owned temporary quiet-window helpers."""

    def _compact_voice_quiet_guidance_text(self, value: object | None, *, limit: int) -> str:
        """Delegate bounded reason compaction through the runtime helper contract."""

        compact = getattr(self, "_compact_runtime_guidance_text", None)
        if not callable(compact):
            return str(value or "").strip()[:limit]
        return str(compact(value, limit=limit) or "").strip()  # pylint: disable=not-callable

    def _parse_voice_quiet_datetime(self, value: object | None) -> datetime | None:
        """Delegate UTC timestamp parsing through the runtime helper contract."""

        parser = getattr(self, "_parse_aware_utc_datetime", None)
        if not callable(parser):
            return None
        parsed = parser(value)  # pylint: disable=not-callable
        return parsed if isinstance(parsed, datetime) else None

    def _voice_quiet_runtime_lock(self) -> Any:
        """Return the shared runtime context lock context manager."""

        lock_factory = getattr(self, "_runtime_context_lock", None)
        if not callable(lock_factory):
            raise RuntimeError("Voice quiet mode requires the runtime context lock")
        return lock_factory()  # pylint: disable=not-callable

    def _require_voice_quiet_int(self, value: object, *, field_name: str, minimum: int) -> int:
        """Delegate bounded integer validation through the runtime helper contract."""

        require_int = getattr(self, "_require_int", None)
        if not callable(require_int):
            raise RuntimeError("Voice quiet mode requires integer validation support")
        return int(require_int(value, field_name=field_name, minimum=minimum))  # pylint: disable=not-callable

    def _persist_voice_quiet_snapshot(self) -> None:
        """Persist the runtime snapshot through the shared best-effort helper."""

        persist = getattr(self, "_safe_persist_snapshot", None)
        if not callable(persist):
            return
        persist(event_on_error="voice_quiet_snapshot_persist_failed")  # pylint: disable=not-callable

    def _normalize_voice_quiet_reason(self, reason: object | None) -> str | None:
        compact = self._compact_voice_quiet_guidance_text(reason, limit=120)
        return compact or None

    def _set_voice_quiet_unlocked(
        self,
        *,
        until_utc: str | None,
        reason: str | None,
    ) -> None:
        """Write the in-memory quiet window fields while the runtime lock is held."""

        setattr(self, "_voice_quiet_until_utc", until_utc)
        setattr(self, "_voice_quiet_reason", reason)

    def _clear_voice_quiet_unlocked(self) -> None:
        """Clear the in-memory quiet window fields while the runtime lock is held."""

        self._set_voice_quiet_unlocked(until_utc=None, reason=None)

    def _voice_quiet_deadline_unlocked(self) -> datetime | None:
        raw_deadline = getattr(self, "_voice_quiet_until_utc", None)
        deadline = self._parse_voice_quiet_datetime(raw_deadline)
        if deadline is None:
            self._clear_voice_quiet_unlocked()
            return None
        now = datetime.now(timezone.utc)
        if deadline <= now:
            self._clear_voice_quiet_unlocked()
            return None
        return deadline

    def voice_quiet_state(self) -> VoiceQuietState:
        """Return the current temporary voice-quiet state."""

        with self._voice_quiet_runtime_lock():
            deadline = self._voice_quiet_deadline_unlocked()
            if deadline is None:
                return VoiceQuietState()
            remaining_seconds = max(
                0,
                int((deadline - datetime.now(timezone.utc)).total_seconds()),
            )
            return VoiceQuietState(
                active=True,
                until_utc=deadline.isoformat().replace("+00:00", "Z"),
                reason=self._normalize_voice_quiet_reason(getattr(self, "_voice_quiet_reason", None)),
                remaining_seconds=remaining_seconds,
            )

    def voice_quiet_active(self) -> bool:
        """Return whether transcript-first wake should currently stay quiet."""

        return self.voice_quiet_state().active

    def voice_quiet_until_utc(self) -> str | None:
        """Return the current quiet deadline as UTC ISO-8601 text when active."""

        return self.voice_quiet_state().until_utc

    def set_voice_quiet_minutes(
        self,
        *,
        minutes: int,
        reason: str | None = None,
    ) -> VoiceQuietState:
        """Start or replace a bounded temporary voice-quiet window."""

        safe_minutes = min(
            self._require_voice_quiet_int(minutes, field_name="minutes", minimum=1),
            _DEFAULT_MAX_QUIET_MINUTES,
        )
        deadline = datetime.now(timezone.utc) + timedelta(minutes=safe_minutes)
        with self._voice_quiet_runtime_lock():
            self._set_voice_quiet_unlocked(
                until_utc=deadline.isoformat().replace("+00:00", "Z"),
                reason=self._normalize_voice_quiet_reason(reason),
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

        with self._voice_quiet_runtime_lock():
            deadline = self._parse_voice_quiet_datetime(until_utc)
            if deadline is None or deadline <= datetime.now(timezone.utc):
                self._clear_voice_quiet_unlocked()
                return
            self._set_voice_quiet_unlocked(
                until_utc=deadline.isoformat().replace("+00:00", "Z"),
                reason=self._normalize_voice_quiet_reason(reason),
            )
