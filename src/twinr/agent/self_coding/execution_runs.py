"""Persist lifecycle transitions for sandbox and retest execution runs."""

from __future__ import annotations

from collections.abc import Mapping  # AUDIT-FIX(#2): Require real mappings for metadata instead of silent coercion.
from copy import deepcopy  # AUDIT-FIX(#5): Detach nested metadata from caller-owned mutable state.
from dataclasses import replace
from datetime import UTC, datetime
from math import isfinite  # AUDIT-FIX(#4): Reject NaN/inf timeout values before persistence.
from threading import RLock  # AUDIT-FIX(#1): Serialize in-process writes and terminal transitions.
from typing import Any
from uuid import uuid4

from twinr.agent.self_coding.contracts import ExecutionRunStatusRecord
from twinr.agent.self_coding.store import SelfCodingStore


def _utc_now() -> datetime:
    return datetime.now(UTC)


class SelfCodingExecutionRunService:
    """Create and update persisted run-status records for skill execution work."""

    _RUNNING_STATUS = "running"

    def __init__(self, *, store: SelfCodingStore) -> None:
        if store is None:
            raise ValueError("store must not be None")  # AUDIT-FIX(#4): Fail fast on invalid service wiring.
        self.store = store
        self._write_lock = RLock()  # AUDIT-FIX(#1): Serialize writes through this service against the shared file-backed store.
        self._terminal_run_ids: set[str] = set()  # AUDIT-FIX(#1): Remember terminalized runs to reject duplicate finishes.

    @staticmethod
    def _require_non_empty_str(value: str, *, field_name: str) -> str:  # AUDIT-FIX(#4): Normalize required text inputs at the service boundary.
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a non-empty string")
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{field_name} must be a non-empty string")
        return normalized

    @staticmethod
    def _normalize_optional_non_empty_str(value: str | None, *, field_name: str) -> str | None:  # AUDIT-FIX(#4): Reject blank optional text inputs when explicitly provided.
        if value is None:
            return None
        return SelfCodingExecutionRunService._require_non_empty_str(value, field_name=field_name)

    @staticmethod
    def _validate_version(version: int) -> int:  # AUDIT-FIX(#4): Reject bools and negative versions before persistence.
        if isinstance(version, bool) or not isinstance(version, int):
            raise TypeError("version must be an int")
        if version < 0:
            raise ValueError("version must be >= 0")
        return version

    @staticmethod
    def _validate_timeout_seconds(timeout_seconds: float | None) -> float | None:  # AUDIT-FIX(#4): Reject invalid timeout values before they hit scheduling logic.
        if timeout_seconds is None:
            return None
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
            raise TypeError("timeout_seconds must be a finite positive number or None")
        normalized = float(timeout_seconds)
        if not isfinite(normalized) or normalized <= 0:
            raise ValueError("timeout_seconds must be a finite positive number or None")
        return normalized

    @staticmethod
    def _normalize_metadata(  # AUDIT-FIX(#2): Fail fast on invalid metadata instead of silently coercing or crashing later.
        metadata: Mapping[str, Any] | None,
        *,
        field_name: str,
    ) -> dict[str, Any]:
        if metadata is None:
            return {}
        if not isinstance(metadata, Mapping):
            raise TypeError(f"{field_name} must be a mapping with non-empty string keys")
        normalized: dict[str, Any] = {}
        for key, value in metadata.items():
            if not isinstance(key, str):
                raise TypeError(f"{field_name} keys must be strings")
            normalized_key = key.strip()
            if not normalized_key:
                raise ValueError(f"{field_name} keys must be non-empty strings")
            normalized[normalized_key] = deepcopy(value)  # AUDIT-FIX(#5): Break nested shared references before persisting.
        return normalized

    def start_run(
        self,
        *,
        run_kind: str,
        skill_id: str,
        version: int,
        timeout_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionRunStatusRecord:
        """Persist one new running execution record."""

        validated_run_kind = self._require_non_empty_str(run_kind, field_name="run_kind")  # AUDIT-FIX(#4): Reject blank lifecycle identifiers early.
        validated_skill_id = self._require_non_empty_str(skill_id, field_name="skill_id")  # AUDIT-FIX(#4): Reject blank skill identifiers early.
        validated_version = self._validate_version(version)  # AUDIT-FIX(#4): Enforce sane numeric version values.
        validated_timeout_seconds = self._validate_timeout_seconds(timeout_seconds)  # AUDIT-FIX(#4): Prevent invalid timeout records from being persisted.
        normalized_metadata = self._normalize_metadata(metadata, field_name="metadata")  # AUDIT-FIX(#2): Handle None explicitly and require a mapping.
        now = _utc_now()
        record = ExecutionRunStatusRecord(
            run_id=f"run_{uuid4().hex}",
            run_kind=validated_run_kind,
            skill_id=validated_skill_id,
            version=validated_version,
            status=self._RUNNING_STATUS,
            timeout_seconds=validated_timeout_seconds,
            started_at=now,
            updated_at=now,
            metadata=normalized_metadata,
        )
        with self._write_lock:  # AUDIT-FIX(#1): Serialize writes through this service against the shared file-backed store.
            return self.store.save_execution_run(record)

    def finish_run(
        self,
        current: ExecutionRunStatusRecord,
        *,
        status: str,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionRunStatusRecord:
        """Persist one terminal transition for an existing execution run."""

        if current is None:
            raise ValueError("current must not be None")  # AUDIT-FIX(#4): Fail fast on invalid finish inputs.
        normalized_status = self._require_non_empty_str(status, field_name="status")  # AUDIT-FIX(#3): Require a real terminal status token.
        if normalized_status.casefold() == self._RUNNING_STATUS:
            raise ValueError("finish_run requires a terminal status, not 'running'")  # AUDIT-FIX(#3): Prevent impossible completed+running records.
        normalized_reason = self._normalize_optional_non_empty_str(reason, field_name="reason")  # AUDIT-FIX(#4): Reject blank terminal reasons when provided.
        incoming_metadata = self._normalize_metadata(metadata, field_name="metadata")  # AUDIT-FIX(#2): Validate finish metadata before merge.
        with self._write_lock:  # AUDIT-FIX(#1): Serialize terminal transitions and reject duplicate finishes in-process.
            run_id = self._require_non_empty_str(current.run_id, field_name="current.run_id")
            current_status = self._require_non_empty_str(current.status, field_name="current.status")
            if run_id in self._terminal_run_ids or current.completed_at is not None or current_status.casefold() != self._RUNNING_STATUS:
                raise ValueError(
                    f"Execution run {run_id!r} is already terminal or not in a finishable state"
                )  # AUDIT-FIX(#1): Do not overwrite terminal records or stale snapshots.
            now = _utc_now()
            merged_metadata = self._normalize_metadata(current.metadata, field_name="current.metadata")  # AUDIT-FIX(#2): Tolerate legacy or malformed persisted metadata safely.
            merged_metadata.update(incoming_metadata)
            saved_record = self.store.save_execution_run(
                replace(
                    current,
                    status=normalized_status,
                    reason=normalized_reason if normalized_reason is not None else current.reason,
                    updated_at=now,
                    completed_at=now,
                    metadata=merged_metadata,
                )
            )
            self._terminal_run_ids.add(run_id)  # AUDIT-FIX(#1): Remember successful terminalization to block duplicate finishes.
            return saved_record


__all__ = ["SelfCodingExecutionRunService"]