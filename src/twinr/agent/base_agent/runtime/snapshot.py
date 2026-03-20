"""Persist and restore the structured runtime snapshot representation."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from twinr.memory import ConversationTurn, MemoryLedgerItem, MemoryState, SearchMemoryEntry


LOGGER = logging.getLogger(__name__)  # AUDIT-FIX(#1): Capture snapshot failures without crashing the device.
_SNAPSHOT_FALLBACK_AT = datetime(1970, 1, 1, tzinfo=timezone.utc)  # AUDIT-FIX(#6): Invalid timestamps must be treated as stale.


class TwinrRuntimeSnapshotMixin:
    """Provide the canonical runtime snapshot save and restore path."""

    def _persist_snapshot(self, *, error_message: str | None = None) -> None:
        """Save the current runtime state through RuntimeSnapshotStore."""

        try:
            self.snapshot_store.save(
                status=self.status.value,
                memory_turns=tuple(self.memory.turns),  # AUDIT-FIX(#2): Freeze top-level collections before persistence.
                memory_raw_tail=tuple(self.memory.raw_tail),  # AUDIT-FIX(#2): Freeze top-level collections before persistence.
                memory_ledger=tuple(self.memory.ledger),  # AUDIT-FIX(#2): Freeze top-level collections before persistence.
                memory_search_results=tuple(self.memory.search_results),  # AUDIT-FIX(#2): Freeze top-level collections before persistence.
                memory_state=self.memory.state,
                last_transcript=self.last_transcript,
                last_response=self.last_response,
                error_message=error_message,
                user_voice_status=self.user_voice_status,
                user_voice_confidence=self.user_voice_confidence,
                user_voice_checked_at=self._parse_optional_snapshot_timestamp(self.user_voice_checked_at),  # AUDIT-FIX(#3): Persist normalized voice-check timestamps.
                user_voice_user_id=self.user_voice_user_id,
                user_voice_user_display_name=self.user_voice_user_display_name,
                user_voice_match_source=self.user_voice_match_source,
            )
        except Exception:
            LOGGER.exception("Failed to persist runtime snapshot")  # AUDIT-FIX(#2): Persistence must never take down the runtime.

    def _restore_snapshot_context(self) -> None:
        """Restore runtime state from the structured snapshot store."""

        try:
            snapshot = self.snapshot_store.load()  # AUDIT-FIX(#1): Treat persisted snapshots as untrusted input.
        except Exception:
            LOGGER.exception("Failed to load runtime snapshot")  # AUDIT-FIX(#1): Recover from unreadable or corrupted snapshot files.
            self._reset_runtime_snapshot_context()
            return

        if snapshot is None:  # AUDIT-FIX(#1): Cold-start and missing-snapshot cases must stay bootable.
            self._reset_runtime_snapshot_context()
            return

        try:
            self.last_transcript = self._coerce_optional_text(
                self._snapshot_get(snapshot, "last_transcript"),
                empty_to_none=False,
            )
            self.last_response = self._coerce_optional_text(self._snapshot_get(snapshot, "last_response"))
            self.user_voice_status = self._coerce_optional_text(self._snapshot_get(snapshot, "user_voice_status"))
            self.user_voice_confidence = self._coerce_optional_float(self._snapshot_get(snapshot, "user_voice_confidence"))
            restored_voice_checked_at = self._parse_optional_snapshot_timestamp(
                self._snapshot_get(snapshot, "user_voice_checked_at")
            )
            self.user_voice_checked_at = (
                restored_voice_checked_at.isoformat().replace("+00:00", "Z")
                if restored_voice_checked_at is not None
                else None
            )  # AUDIT-FIX(#3): Runtime context expects canonical UTC strings, not datetime objects.
            self.user_voice_user_id = self._coerce_optional_text(self._snapshot_get(snapshot, "user_voice_user_id"))
            self.user_voice_user_display_name = self._coerce_optional_text(
                self._snapshot_get(snapshot, "user_voice_user_display_name")
            )
            self.user_voice_match_source = self._coerce_optional_text(self._snapshot_get(snapshot, "user_voice_match_source"))

            legacy_turns = self._restore_legacy_turns(self._snapshot_get(snapshot, "memory_turns", ()))
            raw_tail = self._restore_raw_tail(self._snapshot_get(snapshot, "memory_raw_tail", ()))
            ledger = self._restore_ledger(self._snapshot_get(snapshot, "memory_ledger", ()))
            search_results = self._restore_search_results(self._snapshot_get(snapshot, "memory_search_results", ()))
            state = self._build_memory_state(self._snapshot_get(snapshot, "memory_state"))

            if self._should_restore_structured(
                raw_tail=raw_tail,
                ledger=ledger,
                search_results=search_results,
                state=state,
            ):
                if not raw_tail and not ledger and not search_results and legacy_turns:
                    raw_tail = legacy_turns  # AUDIT-FIX(#4): Preserve legacy turns when structured state exists but collections are empty.
                self.memory.restore_structured(
                    raw_tail=raw_tail,
                    ledger=ledger,
                    search_results=search_results,
                    state=state,
                )
            else:
                self.memory.restore(legacy_turns)
        except Exception:
            LOGGER.exception("Failed to restore runtime snapshot; resetting runtime context")  # AUDIT-FIX(#1): Fall back to a known-good empty state on any restore failure.
            self._reset_runtime_snapshot_context()
            return

        if not self.last_response:
            self.last_response = self._safe_last_assistant_message()

    def _reset_runtime_snapshot_context(self) -> None:
        self.last_transcript = None  # AUDIT-FIX(#1): Clear partially restored runtime fields before continuing.
        self.last_response = None
        self.user_voice_status = None
        self.user_voice_confidence = None
        self.user_voice_checked_at = None
        self.user_voice_user_id = None
        self.user_voice_user_display_name = None
        self.user_voice_match_source = None
        self._reset_memory_context()

    def _reset_memory_context(self) -> None:
        empty_state = self._build_memory_state(None)  # AUDIT-FIX(#1): Reinitialize memory to a safe empty state after restore failure.
        try:
            self.memory.restore_structured(
                raw_tail=(),
                ledger=(),
                search_results=(),
                state=empty_state,
            )
        except Exception:
            try:
                self.memory.restore(())
            except Exception:
                LOGGER.exception("Failed to reset memory context after snapshot failure")  # AUDIT-FIX(#1): Surface a hard reset failure for operators.

    def _restore_legacy_turns(self, value: Any) -> tuple[ConversationTurn, ...]:
        restored_turns: list[ConversationTurn] = []
        for index, turn in enumerate(self._coerce_sequence(value)):
            try:
                restored_turns.append(
                    ConversationTurn(
                        role=self._snapshot_get(turn, "role"),
                        content=self._snapshot_get(turn, "content"),
                        created_at=self._parse_snapshot_timestamp(self._snapshot_get(turn, "created_at")),
                    )
                )
            except Exception as exc:
                LOGGER.warning("Skipping invalid legacy snapshot turn at index %s: %s", index, exc)  # AUDIT-FIX(#1): Salvage valid entries instead of aborting restore.
        return tuple(restored_turns)

    def _restore_raw_tail(self, value: Any) -> tuple[ConversationTurn, ...]:
        raw_tail: list[ConversationTurn] = []
        for index, turn in enumerate(self._coerce_sequence(value)):
            try:
                raw_tail.append(
                    ConversationTurn(
                        role=self._snapshot_get(turn, "role"),
                        content=self._snapshot_get(turn, "content"),
                        created_at=self._parse_snapshot_timestamp(self._snapshot_get(turn, "created_at")),
                    )
                )
            except Exception as exc:
                LOGGER.warning("Skipping invalid snapshot raw_tail entry at index %s: %s", index, exc)  # AUDIT-FIX(#1): Salvage valid entries instead of aborting restore.
        return tuple(raw_tail)

    def _restore_ledger(self, value: Any) -> tuple[MemoryLedgerItem, ...]:
        ledger: list[MemoryLedgerItem] = []
        for index, item in enumerate(self._coerce_sequence(value)):
            try:
                ledger.append(
                    MemoryLedgerItem(
                        kind=self._snapshot_get(item, "kind"),
                        content=self._snapshot_get(item, "content"),
                        created_at=self._parse_snapshot_timestamp(self._snapshot_get(item, "created_at")),
                        source=self._snapshot_get(item, "source"),
                        metadata=self._coerce_mapping(self._snapshot_get(item, "metadata")),  # AUDIT-FIX(#5): Normalize optional metadata before hydrating ledger entries.
                    )
                )
            except Exception as exc:
                LOGGER.warning("Skipping invalid snapshot ledger entry at index %s: %s", index, exc)  # AUDIT-FIX(#1): Salvage valid entries instead of aborting restore.
        return tuple(ledger)

    def _restore_search_results(self, value: Any) -> tuple[SearchMemoryEntry, ...]:
        search_results: list[SearchMemoryEntry] = []
        for index, item in enumerate(self._coerce_sequence(value)):
            try:
                search_results.append(
                    SearchMemoryEntry(
                        question=self._snapshot_get(item, "question"),
                        answer=self._snapshot_get(item, "answer"),
                        sources=self._coerce_text_tuple(self._snapshot_get(item, "sources")),  # AUDIT-FIX(#5): Prevent strings and None from becoming malformed source tuples.
                        created_at=self._parse_snapshot_timestamp(self._snapshot_get(item, "created_at")),
                        location_hint=self._snapshot_get(item, "location_hint"),
                        date_context=self._snapshot_get(item, "date_context"),
                    )
                )
            except Exception as exc:
                LOGGER.warning("Skipping invalid snapshot search result at index %s: %s", index, exc)  # AUDIT-FIX(#1): Salvage valid entries instead of aborting restore.
        return tuple(search_results)

    @classmethod
    def _build_memory_state(cls, value: Any) -> MemoryState:
        return MemoryState(
            active_topic=cls._snapshot_get(value, "active_topic"),
            last_user_goal=cls._snapshot_get(value, "last_user_goal"),
            pending_printable=cls._snapshot_get(value, "pending_printable"),
            last_search_summary=cls._snapshot_get(value, "last_search_summary"),
            open_loops=cls._coerce_text_tuple(cls._snapshot_get(value, "open_loops", ())),  # AUDIT-FIX(#5): Prevent string open loops from exploding into per-character tuples.
        )

    @staticmethod
    def _should_restore_structured(
        *,
        raw_tail: tuple[ConversationTurn, ...],
        ledger: tuple[MemoryLedgerItem, ...],
        search_results: tuple[SearchMemoryEntry, ...],
        state: MemoryState,
    ) -> bool:
        return bool(raw_tail or ledger or search_results or TwinrRuntimeSnapshotMixin._memory_state_has_data(state))  # AUDIT-FIX(#4): Keep meaningful structured state across restarts even when collections are empty.

    @staticmethod
    def _memory_state_has_data(state: MemoryState) -> bool:
        return (
            state.active_topic is not None
            or state.last_user_goal is not None
            or state.pending_printable is not None
            or state.last_search_summary is not None
            or bool(state.open_loops)
        )

    @staticmethod
    def _snapshot_get(value: Any, name: str, default: Any = None) -> Any:  # AUDIT-FIX(#1): Support object- and dict-backed snapshots safely.
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)

    @staticmethod
    def _coerce_sequence(value: Any) -> tuple[Any, ...]:
        if value is None:  # AUDIT-FIX(#1): Tolerate missing snapshot collections instead of raising.
            return ()
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(value)
        if isinstance(value, str):
            return (value,)
        try:
            return tuple(value)
        except TypeError:
            return (value,)

    @staticmethod
    def _coerce_mapping(value: Any) -> dict[str, Any]:
        if value is None:  # AUDIT-FIX(#5): Treat absent metadata as an empty mapping.
            return {}
        if isinstance(value, dict):
            return dict(value)
        try:
            return dict(value)
        except (TypeError, ValueError):
            return {}

    @classmethod
    def _coerce_text_tuple(cls, value: Any) -> tuple[str, ...]:
        items: list[str] = []
        for item in cls._coerce_sequence(value):  # AUDIT-FIX(#5): Normalize optional text sequences safely.
            if item is None:
                continue
            if isinstance(item, str):
                items.append(item)
            else:
                items.append(str(item))
        return tuple(items)

    @staticmethod
    def _coerce_optional_text(value: Any, *, empty_to_none: bool = True) -> str | None:
        if value is None:  # AUDIT-FIX(#1): Reject missing snapshot text fields without aborting restore.
            return None
        if not isinstance(value, str):
            return None
        if empty_to_none and value == "":
            return None
        return value

    @staticmethod
    def _coerce_optional_float(value: Any) -> float | None:
        if value is None or value == "" or isinstance(value, bool):  # AUDIT-FIX(#1): Reject invalid numeric snapshot data without aborting restore.
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _safe_last_assistant_message(self) -> str | None:
        try:
            return self._coerce_optional_text(self.memory.last_assistant_message())
        except Exception:
            LOGGER.exception("Failed to read last assistant message from restored memory")  # AUDIT-FIX(#1): Keep restore non-fatal even if memory helpers fail.
            return None

    @classmethod
    def _parse_optional_snapshot_timestamp(cls, value: Any) -> datetime | None:
        if value is None or value == "":  # AUDIT-FIX(#3): Preserve missing timestamps as None instead of inventing values.
            return None
        return cls._parse_snapshot_timestamp(value)

    @staticmethod
    def _parse_snapshot_timestamp(value: Any) -> datetime:
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                return _SNAPSHOT_FALLBACK_AT
            try:
                parsed = datetime.fromisoformat(candidate)
            except ValueError:
                return _SNAPSHOT_FALLBACK_AT
        else:
            return _SNAPSHOT_FALLBACK_AT

        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return parsed.replace(tzinfo=timezone.utc)  # AUDIT-FIX(#6): Eliminate naive datetimes during restore.
        return parsed.astimezone(timezone.utc)  # AUDIT-FIX(#6): Normalize restored timestamps to aware UTC.
