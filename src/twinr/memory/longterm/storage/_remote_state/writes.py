"""Remote snapshot write and attestation helpers for remote state."""
# mypy: disable-error-code="attr-defined,arg-type"

from __future__ import annotations

import math
import time
from typing import Mapping

from twinr.memory.chonkydb import ChonkyDBClient, ChonkyDBError
from twinr.memory.chonkydb.models import ChonkyDBRecordRequest
from twinr.memory.longterm.storage._remote_retry import (
    clone_client_with_capped_timeout,
    remote_write_retry_delay_s,
    retryable_remote_write_attempts,
    should_fallback_async_job_resolution_error,
    should_retry_remote_write_error,
)
from twinr.memory.longterm.storage.remote_read_diagnostics import (
    LongTermRemoteWriteContext,
    record_remote_write_diagnostic,
)
from twinr.memory.longterm.storage.remote_read_observability import record_remote_write_observation

from .shared import (
    LongTermRemoteSnapshotProbe,
    LongTermRemoteUnavailableError,
    _DEFAULT_ASYNC_ATTESTATION_POLL_S,
    _LOGGER,
    _RemoteSnapshotFetchResult,
    _SNAPSHOT_POINTER_SCHEMA,
    _SNAPSHOT_POINTER_VERSION,
    _SNAPSHOT_SCHEMA,
    _extract_store_document_id,
    _extract_store_job_id,
    _mapping_dict,
    _normalize_text,
    _safe_json_text,
    _store_result_failure_detail,
    _utcnow_iso,
)


class LongTermRemoteStateWriteMixin:
    """Write, attest, and bootstrap remote long-term snapshot state."""

    def ensure_snapshot(self, *, snapshot_kind: str, payload: Mapping[str, object]) -> bool:
        """Ensure one snapshot exists remotely, writing it if missing."""

        normalized_snapshot_kind = self._normalize_snapshot_kind(snapshot_kind)
        if not self.enabled:
            return False
        self._require_client(self.read_client, operation="read")
        probe = self.probe_snapshot_load(snapshot_kind=normalized_snapshot_kind)
        if probe.payload is not None:
            return False
        if probe.status == "unavailable":
            raise LongTermRemoteUnavailableError(
                probe.detail or self._remote_failure_detail("read", normalized_snapshot_kind)
            )
        self.save_snapshot(snapshot_kind=normalized_snapshot_kind, payload=payload)
        return True

    def save_snapshot(self, *, snapshot_kind: str, payload: Mapping[str, object]) -> None:
        """Persist one snapshot payload to the remote backend."""

        normalized_snapshot_kind = self._normalize_snapshot_kind(snapshot_kind)
        if not self.enabled:
            return
        write_client = self._require_client(self.write_client, operation="write")
        read_client = self._require_client(self.read_client, operation="read")
        if self._circuit_is_open():
            raise LongTermRemoteUnavailableError(
                "Remote long-term memory is temporarily cooling down after recent failures."
            )
        payload_dict = _mapping_dict(payload) or {}
        retry_attempts = self._retry_attempts()
        retry_backoff_s = self._retry_backoff_s()
        document_id: str | None = None
        attested_payload: dict[str, object] | None = None
        attested_source = "saved_snapshot"
        for attempt in range(retry_attempts):
            try:
                document_id = self._store_snapshot_record(
                    write_client,
                    snapshot_kind=normalized_snapshot_kind,
                    payload=payload_dict,
                    attempts=1,
                    backoff_s=0.0,
                )
            except LongTermRemoteUnavailableError:
                self._forget_snapshot_document_id(snapshot_kind=normalized_snapshot_kind)
                if attempt + 1 >= retry_attempts:
                    raise
                if retry_backoff_s > 0:
                    time.sleep(retry_backoff_s)
                continue
            break
        if not self._is_pointer_snapshot_kind(normalized_snapshot_kind):
            attested_probe = self._attest_saved_snapshot_readback(
                read_client,
                snapshot_kind=normalized_snapshot_kind,
                payload=payload_dict,
                document_id=document_id,
            )
            if attested_probe.payload is not None:
                document_id = attested_probe.document_id or document_id
                attested_payload = dict(attested_probe.payload)
                attested_source = attested_probe.selected_source or "saved_snapshot_attested"
            else:
                self._forget_snapshot_document_id(snapshot_kind=normalized_snapshot_kind)
                if document_id is None:
                    raise LongTermRemoteUnavailableError(
                        attested_probe.detail
                        or (
                            f"Remote long-term snapshot {normalized_snapshot_kind!r} could not be "
                            "read back after write attestation."
                        )
                    )
                if retry_attempts <= 1:
                    raise LongTermRemoteUnavailableError(
                        attested_probe.detail
                        or (
                            f"Remote long-term snapshot {normalized_snapshot_kind!r} could not be "
                            "read back after write attestation."
                        )
                    )
                for retry_attempt in range(1, retry_attempts):
                    if retry_backoff_s > 0:
                        time.sleep(retry_backoff_s)
                    document_id = self._store_snapshot_record(
                        write_client,
                        snapshot_kind=normalized_snapshot_kind,
                        payload=payload_dict,
                        attempts=1,
                        backoff_s=0.0,
                    )
                    attested_probe = self._attest_saved_snapshot_readback(
                        read_client,
                        snapshot_kind=normalized_snapshot_kind,
                        payload=payload_dict,
                        document_id=document_id,
                    )
                    if attested_probe.payload is not None:
                        document_id = attested_probe.document_id or document_id
                        attested_payload = dict(attested_probe.payload)
                        attested_source = attested_probe.selected_source or "saved_snapshot_attested"
                        break
                    self._forget_snapshot_document_id(snapshot_kind=normalized_snapshot_kind)
                    if retry_attempt + 1 >= retry_attempts:
                        raise LongTermRemoteUnavailableError(
                            attested_probe.detail
                            or (
                                f"Remote long-term snapshot {normalized_snapshot_kind!r} could not be "
                                "read back after write attestation."
                            )
                        )
        if document_id and not self._is_pointer_snapshot_kind(normalized_snapshot_kind):
            self._save_snapshot_pointer_with_attestation(
                write_client,
                read_client,
                snapshot_kind=normalized_snapshot_kind,
                document_id=document_id,
            )
            self._remember_snapshot_document_id(
                snapshot_kind=normalized_snapshot_kind,
                document_id=document_id,
                persist=True,
            )
        self._store_snapshot_read(
            snapshot_kind=normalized_snapshot_kind,
            payload=attested_payload or payload_dict,
            document_id=document_id,
        )
        self._clear_cached_probe(snapshot_kind=normalized_snapshot_kind)
        if not self._is_pointer_snapshot_kind(normalized_snapshot_kind):
            self._store_cached_probe(
                LongTermRemoteSnapshotProbe(
                    snapshot_kind=normalized_snapshot_kind,
                    status="found",
                    latency_ms=0.0,
                    document_id=document_id,
                    selected_source=attested_source,
                    payload=attested_payload or dict(payload_dict),
                )
            )

    def _store_snapshot_record(
        self,
        write_client: ChonkyDBClient,
        *,
        snapshot_kind: str,
        payload: Mapping[str, object],
        attempts: int | None = None,
        backoff_s: float | None = None,
    ) -> str | None:
        namespace = self.namespace or "twinr_longterm_v1"
        updated_at = _utcnow_iso()
        started = time.monotonic()
        snapshot_document = {
            "schema": _SNAPSHOT_SCHEMA,
            "namespace": namespace,
            "snapshot_kind": snapshot_kind,
            "updated_at": updated_at,
            "body": dict(payload),
        }
        try:
            snapshot_content = _safe_json_text(snapshot_document)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Snapshot payload must be JSON-serializable and contain only finite JSON values."
            ) from exc

        record = ChonkyDBRecordRequest(
            payload=snapshot_document,
            metadata={
                "twinr_namespace": namespace,
                "twinr_snapshot_kind": snapshot_kind,
                "twinr_snapshot_updated_at": updated_at,
                "twinr_snapshot_schema": _SNAPSHOT_SCHEMA,
            },
            uri=self._snapshot_uri(snapshot_kind),
            content=snapshot_content,
            enable_chunking=False,
            include_insights_in_response=False,
            execution_mode="async",
        )
        last_error: Exception | None = None
        resolved_attempts = max(1, int(attempts or self._retry_attempts()))
        resolved_backoff_s = max(0.0, float(backoff_s if backoff_s is not None else self._retry_backoff_s()))
        attempt = 0
        while attempt < resolved_attempts:
            try:
                result = write_client.store_record(record)
                failure_detail = _store_result_failure_detail(result)
                if failure_detail:
                    raise ChonkyDBError(
                        f"ChonkyDB rejected remote snapshot write: {failure_detail}",
                        response_json=dict(result) if isinstance(result, Mapping) else None,
                    )
                self._note_remote_success()
                document_id = _extract_store_document_id(result)
                resolved_document_id = (
                    document_id
                    if document_id is not None
                    else self._await_async_store_document_id(write_client, result=result)
                )
                record_remote_write_observation(
                    remote_state=self,
                    context=LongTermRemoteWriteContext(
                        snapshot_kind=snapshot_kind,
                        operation="store_snapshot_record",
                        request_method="POST",
                        request_payload_kind="snapshot_record",
                        request_path="/v1/external/record",
                        timeout_s=getattr(getattr(write_client, "config", None), "timeout_s", None),
                        namespace=self.namespace,
                        access_classification="legacy_snapshot_write",
                        document_id_hint=resolved_document_id,
                        uri_hint=record.uri,
                        attempt_count=attempt + 1,
                        request_item_count=1,
                    ),
                    latency_ms=max(0.0, (time.monotonic() - started) * 1000.0),
                    outcome="ok",
                    classification="ok",
                )
                return resolved_document_id
            except Exception as exc:
                last_error = exc
                self._note_remote_failure()
                resolved_attempts = retryable_remote_write_attempts(resolved_attempts, exc=exc)
                if not should_retry_remote_write_error(exc) or attempt + 1 >= resolved_attempts:
                    break
                delay_s = remote_write_retry_delay_s(
                    exc,
                    default_backoff_s=resolved_backoff_s,
                    attempt_index=attempt,
                )
                if delay_s > 0:
                    time.sleep(delay_s)
                attempt += 1
                continue
            attempt += 1
        if last_error is not None:
            record_remote_write_diagnostic(
                remote_state=self,
                context=LongTermRemoteWriteContext(
                    snapshot_kind=snapshot_kind,
                    operation="store_snapshot_record",
                    request_method="POST",
                    request_payload_kind="snapshot_record",
                    request_path="/v1/external/record",
                    timeout_s=getattr(getattr(write_client, "config", None), "timeout_s", None),
                    namespace=self.namespace,
                    access_classification="legacy_snapshot_write",
                    uri_hint=record.uri,
                    attempt_count=attempt + 1,
                    request_item_count=1,
                ),
                exc=last_error,
                started_monotonic=started,
                outcome="failed",
            )
            _LOGGER.warning(
                "Failed to write remote long-term snapshot %r: %s",
                snapshot_kind,
                self._safe_exception_text(last_error),
            )
        raise LongTermRemoteUnavailableError(
            self._remote_failure_detail("write", snapshot_kind, exc=last_error)
        ) from last_error

    def _await_async_store_document_id(
        self,
        write_client: ChonkyDBClient,
        *,
        result: Mapping[str, object] | None,
    ) -> str | None:
        """Resolve one accepted async snapshot write to its exact document id."""

        job_id = _extract_store_job_id(result)
        if job_id is None:
            return None
        job_status_client = self._status_probe_client(write_client)
        poll_interval_s = max(self._retry_backoff_s(), _DEFAULT_ASYNC_ATTESTATION_POLL_S)
        total_timeout_s = self._async_job_visibility_timeout_s()
        deadline = time.monotonic() + total_timeout_s
        while True:
            remaining_timeout_s = deadline - time.monotonic()
            if remaining_timeout_s <= 0.0:
                break
            capped_client = clone_client_with_capped_timeout(job_status_client, timeout_s=remaining_timeout_s)
            try:
                payload = capped_client.job_status(job_id)  # type: ignore[attr-defined]
            except Exception as exc:
                if not should_fallback_async_job_resolution_error(exc):
                    raise
            else:
                if isinstance(payload, Mapping):
                    raw_status = payload.get("status")
                    status = _normalize_text(raw_status if isinstance(raw_status, str) else None).lower()
                    result_payload = payload.get("result")
                    result_mapping = result_payload if isinstance(result_payload, Mapping) else None
                    if status in {"failed", "cancelled", "rejected"}:
                        detail = _store_result_failure_detail(result_mapping)
                        if not detail:
                            raw_error = payload.get("error")
                            raw_error_type = payload.get("error_type")
                            detail = (
                                _normalize_text(raw_error if isinstance(raw_error, str) else None)
                                or _normalize_text(raw_error_type if isinstance(raw_error_type, str) else None)
                                or f"async job status={status}"
                            )
                        raise LongTermRemoteUnavailableError(
                            f"Accepted async remote snapshot write job {job_id!r} failed before readback: {detail}"
                        )
                    document_id = _extract_store_document_id(result_mapping)
                    if document_id is None:
                        document_id = _extract_store_document_id(payload)
                    if document_id is not None:
                        return document_id
                    if status in {"succeeded", "done"}:
                        return None
            remaining_sleep_s = deadline - time.monotonic()
            if remaining_sleep_s <= 0.0:
                break
            if poll_interval_s > 0.0:
                time.sleep(min(poll_interval_s, remaining_sleep_s))
        return None

    def _attest_saved_snapshot_readback(
        self,
        read_client: ChonkyDBClient,
        *,
        snapshot_kind: str,
        payload: Mapping[str, object],
        document_id: str | None,
    ) -> LongTermRemoteSnapshotProbe:
        """Verify one accepted snapshot write resolves to the expected payload."""

        expected_payload = dict(payload)
        return self._attest_expected_snapshot_readback(
            read_client,
            snapshot_kind=snapshot_kind,
            expected_payload=expected_payload,
            document_id=document_id,
            source="saved_document" if document_id else "saved_origin_uri",
            mismatch_detail=(
                f"Remote long-term snapshot {snapshot_kind!r} readback did not match the just-written payload."
            ),
        )

    def _attest_saved_pointer_readback(
        self,
        read_client: ChonkyDBClient,
        *,
        snapshot_kind: str,
        document_id: str,
        pointer_document_id: str | None = None,
    ) -> LongTermRemoteSnapshotProbe:
        """Verify one accepted pointer write resolves to the expected payload."""

        pointer_snapshot_kind = self._pointer_snapshot_kind(snapshot_kind)
        expected_payload = {
            "schema": _SNAPSHOT_POINTER_SCHEMA,
            "version": _SNAPSHOT_POINTER_VERSION,
            "snapshot_kind": snapshot_kind,
            "document_id": document_id,
        }
        return self._attest_expected_snapshot_readback(
            read_client,
            snapshot_kind=pointer_snapshot_kind,
            expected_payload=expected_payload,
            document_id=pointer_document_id,
            source="saved_pointer_document" if pointer_document_id else "saved_pointer",
            mismatch_detail=(
                f"Remote long-term snapshot pointer {snapshot_kind!r} could not be "
                "read back after write attestation."
            ),
        )

    def _attest_expected_snapshot_readback(
        self,
        read_client: ChonkyDBClient,
        *,
        snapshot_kind: str,
        expected_payload: Mapping[str, object],
        document_id: str | None,
        source: str,
        mismatch_detail: str,
    ) -> LongTermRemoteSnapshotProbe:
        """Poll one just-written snapshot until the expected payload is visible."""

        started = time.monotonic()
        attempt_records: list[object] = []
        expected_payload_dict = dict(expected_payload)
        poll_interval_s = max(self._retry_backoff_s(), _DEFAULT_ASYNC_ATTESTATION_POLL_S)
        resolved_attempts = self._retry_attempts()
        if poll_interval_s > 0.0:
            visibility_timeout_s = self._async_attestation_visibility_timeout_s()
            resolved_attempts = max(
                resolved_attempts,
                int(math.ceil(visibility_timeout_s / poll_interval_s)),
            )
        last_fetch: _RemoteSnapshotFetchResult | None = None
        for attempt in range(resolved_attempts):
            result = self._load_snapshot_via_uri(
                read_client,
                snapshot_kind=snapshot_kind,
                document_id=document_id,
                source=source,
                retry_not_found=False,
            )
            attempt_records.extend(result.attempts)
            last_fetch = result
            if result.payload == expected_payload_dict:
                return LongTermRemoteSnapshotProbe(
                    snapshot_kind=snapshot_kind,
                    status="found",
                    latency_ms=round(max(0.0, (time.monotonic() - started) * 1000.0), 3),
                    detail=result.detail,
                    document_id=result.document_id or document_id,
                    selected_source=result.selected_source,
                    payload=dict(result.payload or expected_payload_dict),
                    attempts=tuple(attempt_records),
                )
            if result.status == "unavailable":
                break
            if attempt + 1 >= resolved_attempts:
                break
            if poll_interval_s > 0:
                time.sleep(poll_interval_s)
        detail = mismatch_detail if last_fetch is None else (last_fetch.detail or mismatch_detail)
        return LongTermRemoteSnapshotProbe(
            snapshot_kind=snapshot_kind,
            status="unavailable",
            latency_ms=round(max(0.0, (time.monotonic() - started) * 1000.0), 3),
            detail=detail,
            document_id=None if last_fetch is None else (last_fetch.document_id or document_id),
            selected_source=None if last_fetch is None else last_fetch.selected_source,
            attempts=tuple(attempt_records),
        )

    def _save_snapshot_pointer_with_attestation(
        self,
        write_client: ChonkyDBClient,
        read_client: ChonkyDBClient,
        *,
        snapshot_kind: str,
        document_id: str,
    ) -> None:
        """Persist one pointer snapshot and prove that fresh readers can see it."""

        retry_attempts = self._retry_attempts()
        retry_backoff_s = self._retry_backoff_s()
        for attempt in range(retry_attempts):
            pointer_document_id = self._save_snapshot_pointer(
                write_client,
                snapshot_kind=snapshot_kind,
                document_id=document_id,
            )
            attested_probe = self._attest_saved_pointer_readback(
                read_client,
                snapshot_kind=snapshot_kind,
                document_id=document_id,
                pointer_document_id=pointer_document_id,
            )
            if attested_probe.payload is not None:
                return
            if pointer_document_id is None or attempt + 1 >= retry_attempts:
                raise LongTermRemoteUnavailableError(
                    attested_probe.detail
                    or (
                        f"Remote long-term snapshot pointer {snapshot_kind!r} could not be "
                        "read back after write attestation."
                    )
                )
            if retry_backoff_s > 0:
                time.sleep(retry_backoff_s)

    def _save_snapshot_pointer(
        self,
        write_client: ChonkyDBClient,
        *,
        snapshot_kind: str,
        document_id: str,
    ) -> str | None:
        """Persist one pointer snapshot and return the exact document id when known."""

        pointer_payload = {
            "schema": _SNAPSHOT_POINTER_SCHEMA,
            "version": _SNAPSHOT_POINTER_VERSION,
            "snapshot_kind": snapshot_kind,
            "document_id": document_id,
        }
        return self._store_snapshot_record(
            write_client,
            snapshot_kind=self._pointer_snapshot_kind(snapshot_kind),
            payload=pointer_payload,
        )


__all__ = ["LongTermRemoteStateWriteMixin"]
