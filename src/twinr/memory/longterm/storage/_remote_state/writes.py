"""Remote snapshot write and attestation helpers for remote state."""

from __future__ import annotations

import json
import math
import time
from typing import Mapping
from uuid import uuid4

from twinr.memory.chonkydb import ChonkyDBClient, ChonkyDBError
from twinr.memory.chonkydb.models import ChonkyDBRecordRequest
from twinr.memory.longterm.storage._remote_retry import (
    clone_client_with_capped_timeout,
    exception_chain,
    is_rate_limited_remote_write_error,
    remote_write_retry_delay_s,
    raise_if_remote_operation_cancelled,
    retryable_remote_write_attempts,
    sleep_with_remote_operation_abort,
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

_SYNC_SMALL_SNAPSHOT_WRITE_MAX_BYTES = 16_384
_SYNC_DEFERRED_ID_SNAPSHOT_KINDS = frozenset({"midterm"})


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

    def save_snapshot(
        self,
        *,
        snapshot_kind: str,
        payload: Mapping[str, object],
        skip_async_document_id_wait: bool = False,
    ) -> None:
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
            raise_if_remote_operation_cancelled(operation="Remote snapshot write")
            try:
                document_id = self._store_snapshot_record(
                    write_client,
                    snapshot_kind=normalized_snapshot_kind,
                    payload=payload_dict,
                    attempts=1,
                    backoff_s=0.0,
                    skip_async_document_id_wait=skip_async_document_id_wait,
                )
            except LongTermRemoteUnavailableError:
                self._forget_snapshot_document_id(snapshot_kind=normalized_snapshot_kind)
                if attempt + 1 >= retry_attempts:
                    raise
                if retry_backoff_s > 0:
                    sleep_with_remote_operation_abort(
                        retry_backoff_s,
                        operation="Remote snapshot write retry",
                    )
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
                        skip_async_document_id_wait=skip_async_document_id_wait,
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
        skip_async_document_id_wait: bool = False,
        forced_execution_mode: str | None = None,
        allow_single_item_sync_rescue: bool = True,
    ) -> str | None:
        namespace = self.namespace or "twinr_longterm_v1"
        updated_at = _utcnow_iso()
        started = time.monotonic()
        request_correlation_id = f"ltsw-{uuid4().hex[:12]}"
        execution_mode = forced_execution_mode or "async"
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
            execution_mode=execution_mode,
        )
        request_bytes = len(json.dumps(record.to_payload(), ensure_ascii=False).encode("utf-8"))
        last_error: Exception | None = None
        resolved_attempts = max(1, int(attempts or self._retry_attempts()))
        resolved_backoff_s = max(0.0, float(backoff_s if backoff_s is not None else self._retry_backoff_s()))
        attempt = 0
        while attempt < resolved_attempts:
            raise_if_remote_operation_cancelled(operation="Remote snapshot write")
            store_transport_ms: float | None = None
            async_job_wait_ms: float | None = None
            async_job_resolution_source: str | None = None
            attempt_started = time.monotonic()
            effective_write_client = write_client
            if self._sync_snapshot_write_needs_flush_timeout(
                snapshot_kind=snapshot_kind,
                skip_async_document_id_wait=skip_async_document_id_wait,
                forced_execution_mode=forced_execution_mode,
                execution_mode=execution_mode,
            ):
                effective_write_client = self._clone_snapshot_write_client_with_timeout(
                    write_client,
                    timeout_s=self._remote_flush_timeout_s(),
                )
            try:
                result = effective_write_client.store_record(record)
                store_transport_ms = max(0.0, (time.monotonic() - attempt_started) * 1000.0)
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
                    else None
                )
                if resolved_document_id is not None:
                    async_job_resolution_source = "response"
                elif skip_async_document_id_wait:
                    async_job_resolution_source = "skipped_projection_complete"
                elif document_id is None:
                    async_wait_started = time.monotonic()
                    try:
                        resolved_document_id = self._await_async_store_document_id(write_client, result=result)
                    finally:
                        async_job_wait_ms = max(0.0, (time.monotonic() - async_wait_started) * 1000.0)
                    async_job_resolution_source = (
                        "job_status" if resolved_document_id is not None else "job_status_no_document_id"
                    )
                record_remote_write_observation(
                    remote_state=self,
                    context=LongTermRemoteWriteContext(
                        snapshot_kind=snapshot_kind,
                        operation="store_snapshot_record",
                        request_method="POST",
                        request_payload_kind="snapshot_record",
                        request_path="/v1/external/record",
                        timeout_s=getattr(getattr(effective_write_client, "config", None), "timeout_s", None),
                        namespace=self.namespace,
                        access_classification="legacy_snapshot_write",
                        document_id_hint=resolved_document_id,
                        uri_hint=record.uri,
                        attempt_count=attempt + 1,
                        request_item_count=1,
                        request_correlation_id=request_correlation_id,
                        request_bytes=request_bytes,
                        request_execution_mode=execution_mode,
                        async_job_resolution_source=async_job_resolution_source,
                        store_transport_ms=store_transport_ms,
                        async_job_wait_ms=async_job_wait_ms,
                    ),
                    latency_ms=max(0.0, (time.monotonic() - started) * 1000.0),
                    outcome="ok",
                    classification="ok",
                )
                return resolved_document_id
            except Exception as exc:
                if store_transport_ms is None:
                    store_transport_ms = max(0.0, (time.monotonic() - attempt_started) * 1000.0)
                last_error = exc
                self._note_remote_failure()
                resolved_attempts = retryable_remote_write_attempts(resolved_attempts, exc=exc)
                if (
                    execution_mode == "async"
                    and allow_single_item_sync_rescue
                    and is_rate_limited_remote_write_error(exc)
                    and self._can_rate_limited_snapshot_sync_fallback(
                        snapshot_kind=snapshot_kind,
                        request_bytes=request_bytes,
                        skip_async_document_id_wait=skip_async_document_id_wait,
                    )
                ):
                    return self._store_snapshot_record(
                        write_client,
                        snapshot_kind=snapshot_kind,
                        payload=payload,
                        attempts=resolved_attempts,
                        backoff_s=resolved_backoff_s,
                        skip_async_document_id_wait=skip_async_document_id_wait,
                        forced_execution_mode="sync",
                        allow_single_item_sync_rescue=False,
                    )
                if (
                    not allow_single_item_sync_rescue
                    and forced_execution_mode == "sync"
                    and execution_mode == "sync"
                    and self._can_forced_sync_snapshot_backpressure_async_fallback(
                        snapshot_kind=snapshot_kind,
                        skip_async_document_id_wait=skip_async_document_id_wait,
                        exc=exc,
                    )
                ):
                    return self._store_snapshot_record(
                        write_client,
                        snapshot_kind=snapshot_kind,
                        payload=payload,
                        attempts=resolved_attempts,
                        backoff_s=resolved_backoff_s,
                        skip_async_document_id_wait=skip_async_document_id_wait,
                        forced_execution_mode="async",
                        allow_single_item_sync_rescue=False,
                    )
                if (
                    not allow_single_item_sync_rescue
                    and forced_execution_mode == "sync"
                    and execution_mode == "sync"
                    and self._can_forced_sync_snapshot_timeout_async_fallback(
                        snapshot_kind=snapshot_kind,
                        skip_async_document_id_wait=skip_async_document_id_wait,
                        exc=exc,
                    )
                ):
                    return self._store_snapshot_record(
                        write_client,
                        snapshot_kind=snapshot_kind,
                        payload=payload,
                        attempts=resolved_attempts,
                        backoff_s=resolved_backoff_s,
                        skip_async_document_id_wait=skip_async_document_id_wait,
                        forced_execution_mode="async",
                        allow_single_item_sync_rescue=False,
                    )
                if not should_retry_remote_write_error(exc) or attempt + 1 >= resolved_attempts:
                    break
                delay_s = remote_write_retry_delay_s(
                    exc,
                    default_backoff_s=resolved_backoff_s,
                    attempt_index=attempt,
                )
                if delay_s > 0:
                    sleep_with_remote_operation_abort(
                        delay_s,
                        operation="Remote snapshot write retry",
                    )
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
                    timeout_s=getattr(getattr(effective_write_client, "config", None), "timeout_s", None),
                    namespace=self.namespace,
                    access_classification="legacy_snapshot_write",
                    uri_hint=record.uri,
                    attempt_count=attempt + 1,
                    request_item_count=1,
                    request_correlation_id=request_correlation_id,
                    request_bytes=request_bytes,
                    request_execution_mode=execution_mode,
                    store_transport_ms=store_transport_ms,
                    async_job_wait_ms=async_job_wait_ms,
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

    @staticmethod
    def _clone_snapshot_write_client_with_timeout(
        write_client: ChonkyDBClient,
        *,
        timeout_s: float,
    ) -> ChonkyDBClient:
        """Clone the snapshot write client to the requested transport timeout."""

        clone_with_timeout = getattr(write_client, "clone_with_timeout", None)
        if not callable(clone_with_timeout):
            return write_client
        try:
            current_timeout_s = float(getattr(getattr(write_client, "config", None), "timeout_s"))
            target_timeout_s = max(0.1, float(timeout_s))
        except (AttributeError, TypeError, ValueError):
            return write_client
        if math.isclose(current_timeout_s, target_timeout_s, rel_tol=0.0, abs_tol=1e-9):
            return write_client
        return clone_with_timeout(target_timeout_s)

    @staticmethod
    def _can_rate_limited_snapshot_sync_fallback(
        *,
        snapshot_kind: str,
        request_bytes: int,
        skip_async_document_id_wait: bool,
    ) -> bool:
        """Return whether one tiny deferred-id snapshot write may bypass the async queue once."""

        return (
            skip_async_document_id_wait
            and snapshot_kind in _SYNC_DEFERRED_ID_SNAPSHOT_KINDS
            and request_bytes <= _SYNC_SMALL_SNAPSHOT_WRITE_MAX_BYTES
        )

    @staticmethod
    def _sync_snapshot_write_needs_flush_timeout(
        *,
        snapshot_kind: str,
        skip_async_document_id_wait: bool,
        forced_execution_mode: str | None,
        execution_mode: str,
    ) -> bool:
        """Return whether one sync rescue should inherit the flush transport budget."""

        return (
            execution_mode == "sync"
            and forced_execution_mode == "sync"
            and skip_async_document_id_wait
            and snapshot_kind in _SYNC_DEFERRED_ID_SNAPSHOT_KINDS
        )

    @staticmethod
    def _can_forced_sync_snapshot_backpressure_async_fallback(
        *,
        snapshot_kind: str,
        skip_async_document_id_wait: bool,
        exc: BaseException,
    ) -> bool:
        """Return whether one busy sync rescue may return to bounded async retries."""

        return (
            skip_async_document_id_wait
            and snapshot_kind in _SYNC_DEFERRED_ID_SNAPSHOT_KINDS
            and is_rate_limited_remote_write_error(exc)
        )

    @staticmethod
    def _can_forced_sync_snapshot_timeout_async_fallback(
        *,
        snapshot_kind: str,
        skip_async_document_id_wait: bool,
        exc: BaseException,
    ) -> bool:
        """Return whether one timed-out sync rescue may return to bounded async retries."""

        if not skip_async_document_id_wait or snapshot_kind not in _SYNC_DEFERRED_ID_SNAPSHOT_KINDS:
            return False
        for item in exception_chain(exc):
            if isinstance(item, ChonkyDBError):
                if item.status_code is not None:
                    continue
                message = " ".join(str(item).lower().split())
                if "timed out" in message or "timeout" in message:
                    return True
                continue
            if isinstance(item, (TimeoutError, OSError)):
                return True
        return False

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
            raise_if_remote_operation_cancelled(operation="Remote snapshot async-job wait")
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
                sleep_with_remote_operation_abort(
                    min(poll_interval_s, remaining_sleep_s),
                    operation="Remote snapshot async-job wait",
                )
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
            raise_if_remote_operation_cancelled(operation="Remote snapshot readback attestation")
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
                sleep_with_remote_operation_abort(
                    poll_interval_s,
                    operation="Remote snapshot readback attestation",
                )
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
            raise_if_remote_operation_cancelled(operation="Remote snapshot pointer write")
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
                sleep_with_remote_operation_abort(
                    retry_backoff_s,
                    operation="Remote snapshot pointer write retry",
                )

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
