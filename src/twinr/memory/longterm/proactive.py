from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import tempfile
from threading import Lock
from zoneinfo import ZoneInfo

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.chonkydb.client import chonkydb_data_path
from twinr.memory.longterm.ontology import normalize_memory_sensitivity
from twinr.memory.longterm.models import LONGTERM_MEMORY_SENSITIVITY, LongTermProactiveCandidateV1, LongTermProactivePlanV1


_STATE_SCHEMA = "twinr_memory_proactive_state"
_STATE_VERSION = 1


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_text(value: str | None) -> str:
    return " ".join(str(value or "").split()).strip()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        temp_path.replace(path)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


@dataclass(frozen=True, slots=True)
class LongTermProactiveHistoryEntryV1:
    candidate_id: str
    kind: str
    summary: str
    sensitivity: str = "normal"
    source_memory_ids: tuple[str, ...] = ()
    first_seen_at: datetime = field(default_factory=_utcnow)
    last_seen_at: datetime = field(default_factory=_utcnow)
    last_reserved_at: datetime | None = None
    last_delivered_at: datetime | None = None
    last_skipped_at: datetime | None = None
    last_skip_reason: str | None = None
    last_prompt_text: str | None = None
    delivery_count: int = 0
    skip_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "sensitivity", normalize_memory_sensitivity(self.sensitivity))
        if not _normalize_text(self.candidate_id):
            raise ValueError("candidate_id is required.")
        if not _normalize_text(self.kind):
            raise ValueError("kind is required.")
        if not _normalize_text(self.summary):
            raise ValueError("summary is required.")
        if self.sensitivity not in LONGTERM_MEMORY_SENSITIVITY:
            raise ValueError(f"sensitivity must be one of: {', '.join(sorted(LONGTERM_MEMORY_SENSITIVITY))}.")

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "candidate_id": self.candidate_id,
            "kind": self.kind,
            "summary": self.summary,
            "sensitivity": self.sensitivity,
            "source_memory_ids": list(self.source_memory_ids),
            "first_seen_at": self.first_seen_at.isoformat(),
            "last_seen_at": self.last_seen_at.isoformat(),
            "delivery_count": self.delivery_count,
            "skip_count": self.skip_count,
        }
        if self.last_reserved_at is not None:
            payload["last_reserved_at"] = self.last_reserved_at.isoformat()
        if self.last_delivered_at is not None:
            payload["last_delivered_at"] = self.last_delivered_at.isoformat()
        if self.last_skipped_at is not None:
            payload["last_skipped_at"] = self.last_skipped_at.isoformat()
        if self.last_skip_reason is not None:
            payload["last_skip_reason"] = self.last_skip_reason
        if self.last_prompt_text is not None:
            payload["last_prompt_text"] = self.last_prompt_text
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "LongTermProactiveHistoryEntryV1":
        return cls(
            candidate_id=str(payload.get("candidate_id", "")),
            kind=str(payload.get("kind", "")),
            summary=str(payload.get("summary", "")),
            sensitivity=normalize_memory_sensitivity(str(payload.get("sensitivity", "normal"))),
            source_memory_ids=tuple(
                str(item) for item in payload.get("source_memory_ids", []) if isinstance(item, str)
            ),
            first_seen_at=datetime.fromisoformat(str(payload.get("first_seen_at"))),
            last_seen_at=datetime.fromisoformat(str(payload.get("last_seen_at"))),
            last_reserved_at=datetime.fromisoformat(str(payload["last_reserved_at"]))
            if payload.get("last_reserved_at")
            else None,
            last_delivered_at=datetime.fromisoformat(str(payload["last_delivered_at"]))
            if payload.get("last_delivered_at")
            else None,
            last_skipped_at=datetime.fromisoformat(str(payload["last_skipped_at"]))
            if payload.get("last_skipped_at")
            else None,
            last_skip_reason=str(payload["last_skip_reason"]) if payload.get("last_skip_reason") is not None else None,
            last_prompt_text=str(payload["last_prompt_text"]) if payload.get("last_prompt_text") is not None else None,
            delivery_count=int(payload.get("delivery_count", 0) or 0),
            skip_count=int(payload.get("skip_count", 0) or 0),
        )


@dataclass(frozen=True, slots=True)
class LongTermProactiveReservationV1:
    candidate: LongTermProactiveCandidateV1
    reserved_at: datetime


@dataclass(slots=True)
class LongTermProactiveStateStore:
    path: Path
    history_limit: int = 128
    _lock: Lock = field(default_factory=Lock, repr=False)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "LongTermProactiveStateStore":
        return cls(
            path=chonkydb_data_path(config) / "twinr_memory_proactive_state_v1.json",
            history_limit=max(16, int(config.long_term_memory_proactive_history_limit)),
        )

    def load_entries(self) -> tuple[LongTermProactiveHistoryEntryV1, ...]:
        if not self.path.exists():
            return ()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        items = payload.get("entries", [])
        if not isinstance(items, list):
            return ()
        entries: list[LongTermProactiveHistoryEntryV1] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            entries.append(LongTermProactiveHistoryEntryV1.from_payload(item))
        return tuple(entries)

    def reserve(self, candidate: LongTermProactiveCandidateV1, *, reserved_at: datetime) -> LongTermProactiveReservationV1:
        with self._lock:
            entries = {entry.candidate_id: entry for entry in self.load_entries()}
            existing = entries.get(candidate.candidate_id)
            entries[candidate.candidate_id] = self._upsert_entry(
                existing=existing,
                candidate=candidate,
                seen_at=reserved_at,
                reserved_at=reserved_at,
            )
            self._write_entries(entries.values())
        return LongTermProactiveReservationV1(candidate=candidate, reserved_at=reserved_at)

    def mark_delivered(
        self,
        *,
        candidate: LongTermProactiveCandidateV1,
        delivered_at: datetime,
        prompt_text: str | None = None,
    ) -> LongTermProactiveHistoryEntryV1:
        with self._lock:
            entries = {entry.candidate_id: entry for entry in self.load_entries()}
            existing = entries.get(candidate.candidate_id)
            current = self._upsert_entry(existing=existing, candidate=candidate, seen_at=delivered_at)
            entries[candidate.candidate_id] = LongTermProactiveHistoryEntryV1(
                candidate_id=current.candidate_id,
                kind=current.kind,
                summary=current.summary,
                sensitivity=current.sensitivity,
                source_memory_ids=current.source_memory_ids,
                first_seen_at=current.first_seen_at,
                last_seen_at=delivered_at,
                last_reserved_at=current.last_reserved_at,
                last_delivered_at=delivered_at,
                last_skipped_at=current.last_skipped_at,
                last_skip_reason=current.last_skip_reason,
                last_prompt_text=_normalize_text(prompt_text) or current.last_prompt_text,
                delivery_count=current.delivery_count + 1,
                skip_count=current.skip_count,
            )
            self._write_entries(entries.values())
            return entries[candidate.candidate_id]

    def mark_skipped(
        self,
        *,
        candidate: LongTermProactiveCandidateV1,
        skipped_at: datetime,
        reason: str,
    ) -> LongTermProactiveHistoryEntryV1:
        clean_reason = _normalize_text(reason) or "unknown"
        with self._lock:
            entries = {entry.candidate_id: entry for entry in self.load_entries()}
            existing = entries.get(candidate.candidate_id)
            current = self._upsert_entry(existing=existing, candidate=candidate, seen_at=skipped_at)
            entries[candidate.candidate_id] = LongTermProactiveHistoryEntryV1(
                candidate_id=current.candidate_id,
                kind=current.kind,
                summary=current.summary,
                sensitivity=current.sensitivity,
                source_memory_ids=current.source_memory_ids,
                first_seen_at=current.first_seen_at,
                last_seen_at=skipped_at,
                last_reserved_at=current.last_reserved_at,
                last_delivered_at=current.last_delivered_at,
                last_skipped_at=skipped_at,
                last_skip_reason=clean_reason,
                last_prompt_text=current.last_prompt_text,
                delivery_count=current.delivery_count,
                skip_count=current.skip_count + 1,
            )
            self._write_entries(entries.values())
            return entries[candidate.candidate_id]

    def _upsert_entry(
        self,
        *,
        existing: LongTermProactiveHistoryEntryV1 | None,
        candidate: LongTermProactiveCandidateV1,
        seen_at: datetime,
        reserved_at: datetime | None = None,
    ) -> LongTermProactiveHistoryEntryV1:
        if existing is None:
            return LongTermProactiveHistoryEntryV1(
                candidate_id=candidate.candidate_id,
                kind=candidate.kind,
                summary=candidate.summary,
                sensitivity=candidate.sensitivity,
                source_memory_ids=candidate.source_memory_ids,
                first_seen_at=seen_at,
                last_seen_at=seen_at,
                last_reserved_at=reserved_at,
            )
        return LongTermProactiveHistoryEntryV1(
            candidate_id=existing.candidate_id,
            kind=candidate.kind,
            summary=candidate.summary,
            sensitivity=candidate.sensitivity,
            source_memory_ids=candidate.source_memory_ids,
            first_seen_at=existing.first_seen_at,
            last_seen_at=seen_at,
            last_reserved_at=reserved_at or existing.last_reserved_at,
            last_delivered_at=existing.last_delivered_at,
            last_skipped_at=existing.last_skipped_at,
            last_skip_reason=existing.last_skip_reason,
            last_prompt_text=existing.last_prompt_text,
            delivery_count=existing.delivery_count,
            skip_count=existing.skip_count,
        )

    def _write_entries(self, entries) -> None:
        ranked = sorted(
            entries,
            key=lambda item: (item.last_seen_at.isoformat(), item.candidate_id),
            reverse=True,
        )[: self.history_limit]
        payload = {
            "schema": _STATE_SCHEMA,
            "version": _STATE_VERSION,
            "entries": [item.to_payload() for item in ranked],
        }
        _write_json_atomic(self.path, payload)


@dataclass(slots=True)
class LongTermProactivePolicy:
    config: TwinrConfig
    state_store: LongTermProactiveStateStore
    blocked_sensitivities: frozenset[str] = frozenset({"private", "sensitive", "critical"})

    def reserve_candidate(
        self,
        *,
        plan: LongTermProactivePlanV1,
        now: datetime | None = None,
    ) -> LongTermProactiveReservationV1 | None:
        candidate = self.preview_candidate(plan=plan, now=now)
        if candidate is None:
            return None
        current_time = now or datetime.now(ZoneInfo(self.config.local_timezone_name))
        return self.state_store.reserve(candidate, reserved_at=current_time)

    def reserve_specific_candidate(
        self,
        candidate: LongTermProactiveCandidateV1,
        *,
        now: datetime | None = None,
    ) -> LongTermProactiveReservationV1:
        current_time = now or datetime.now(ZoneInfo(self.config.local_timezone_name))
        return self.state_store.reserve(candidate, reserved_at=current_time)

    def preview_candidate(
        self,
        *,
        plan: LongTermProactivePlanV1,
        now: datetime | None = None,
    ) -> LongTermProactiveCandidateV1 | None:
        if not self.config.long_term_memory_proactive_enabled:
            return None
        current_time = now or datetime.now(ZoneInfo(self.config.local_timezone_name))
        history = {entry.candidate_id: entry for entry in self.state_store.load_entries()}
        for candidate in plan.candidates:
            if candidate.confidence < self.config.long_term_memory_proactive_min_confidence:
                continue
            if (
                not self.config.long_term_memory_proactive_allow_sensitive
                and candidate.sensitivity in self.blocked_sensitivities
            ):
                continue
            entry = history.get(candidate.candidate_id)
            if self._within_cooldown(
                entry.last_reserved_at if entry is not None else None,
                current_time=current_time,
                cooldown_s=self.config.long_term_memory_proactive_reservation_ttl_s,
            ):
                continue
            if self._within_cooldown(
                entry.last_delivered_at if entry is not None else None,
                current_time=current_time,
                cooldown_s=self.config.long_term_memory_proactive_repeat_cooldown_s,
            ):
                continue
            if self._within_cooldown(
                entry.last_skipped_at if entry is not None else None,
                current_time=current_time,
                cooldown_s=self.config.long_term_memory_proactive_skip_cooldown_s,
            ):
                continue
            return candidate
        return None

    def mark_delivered(
        self,
        reservation: LongTermProactiveReservationV1,
        *,
        delivered_at: datetime | None = None,
        prompt_text: str | None = None,
    ) -> LongTermProactiveHistoryEntryV1:
        return self.state_store.mark_delivered(
            candidate=reservation.candidate,
            delivered_at=delivered_at or datetime.now(ZoneInfo(self.config.local_timezone_name)),
            prompt_text=prompt_text,
        )

    def mark_skipped(
        self,
        reservation: LongTermProactiveReservationV1,
        *,
        reason: str,
        skipped_at: datetime | None = None,
    ) -> LongTermProactiveHistoryEntryV1:
        return self.state_store.mark_skipped(
            candidate=reservation.candidate,
            skipped_at=skipped_at or datetime.now(ZoneInfo(self.config.local_timezone_name)),
            reason=reason,
        )

    def _within_cooldown(
        self,
        value: datetime | None,
        *,
        current_time: datetime,
        cooldown_s: float,
    ) -> bool:
        if value is None:
            return False
        return current_time < value + timedelta(seconds=max(0.0, cooldown_s))


__all__ = [
    "LongTermProactiveHistoryEntryV1",
    "LongTermProactivePolicy",
    "LongTermProactiveReservationV1",
    "LongTermProactiveStateStore",
]
