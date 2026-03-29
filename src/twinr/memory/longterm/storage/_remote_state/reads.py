"""Remote snapshot read and probe helpers for remote state."""
# mypy: disable-error-code=attr-defined,no-redef

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Iterable, Mapping

from twinr.memory.chonkydb import ChonkyDBClient, ChonkyDBError
from twinr.memory.longterm.storage.remote_read_diagnostics import (
    LongTermRemoteReadContext,
    record_remote_read_diagnostic,
)
from twinr.memory.longterm.storage.remote_read_observability import record_remote_read_observation

from .shared import (
    LongTermRemoteFetchAttempt,
    LongTermRemoteSnapshotProbe,
    LongTermRemoteUnavailableError,
    _DEFAULT_REMOTE_READ_TIMEOUT_S,
    _LOGGER,
    _REMOTE_NAMESPACE_PREFIX,
    _RemoteSnapshotCandidate,
    _RemoteSnapshotFetchResult,
    _SNAPSHOT_POINTER_SCHEMA,
    _SNAPSHOT_POINTER_VERSION,
    _SNAPSHOT_SCHEMA,
    _candidate_document_id,
    _coerce_float,
    _normalize_snapshot_document_id,
    _reject_non_finite_json_constant,
    _snapshot_updated_at_sort_key,
)


class LongTermRemoteStateReadMixin:
    """Read, probe, and parse remote long-term snapshot state."""

    _METADATA_ONLY_MAX_CONTENT_CHARS = 1

    namespace: str | None
    read_client: ChonkyDBClient | None
    write_client: ChonkyDBClient | None

    def load_snapshot(
        self,
        *,
        snapshot_kind: str,
        local_path: Path | None = None,
        prefer_cached_document_id: bool = True,
    ) -> dict[str, object] | None:
        """Load one snapshot from remote storage or a safe local fallback."""

        normalized_snapshot_kind = self._normalize_snapshot_kind(snapshot_kind)
        if not self.enabled:
            return None
        cached_payload = self._cached_snapshot_read(snapshot_kind=normalized_snapshot_kind)
        if cached_payload is not None:
            return cached_payload
        try:
            self._require_client(self.read_client, operation="read")
        except LongTermRemoteUnavailableError as exc:
            if self.required:
                raise
            local_payload = None
            if local_path is not None:
                local_payload = self._load_local_snapshot(
                    local_path,
                    snapshot_kind=normalized_snapshot_kind,
                )
            if local_payload is not None:
                return local_payload
            if self.required:
                raise
            _LOGGER.warning(
                "Remote long-term read client unavailable for %r: %s",
                normalized_snapshot_kind,
                self._safe_exception_text(exc),
            )
            return None

        probe = self._probe_snapshot_load_internal(
            snapshot_kind=normalized_snapshot_kind,
            local_path=local_path,
            prefer_cached_document_id=prefer_cached_document_id,
        )
        if probe.payload is not None:
            self._store_snapshot_read(
                snapshot_kind=normalized_snapshot_kind,
                payload=probe.payload,
                document_id=probe.document_id,
            )
            return probe.payload

        local_payload: dict[str, object] | None = None
        if local_path is not None and probe.status in {"not_found", "unavailable"}:
            local_payload = self._load_local_snapshot(
                local_path,
                snapshot_kind=normalized_snapshot_kind,
            )

        if probe.status == "unavailable":
            if self.required:
                raise LongTermRemoteUnavailableError(
                    probe.detail or self._remote_failure_detail("read", normalized_snapshot_kind)
                )
            if local_payload is not None:
                return local_payload
            return None

        if probe.status == "not_found":
            if self.required and not self.config.long_term_memory_migration_enabled:
                return None
            if local_payload is None:
                return None
            if self.config.long_term_memory_migration_enabled:
                try:
                    self.save_snapshot(snapshot_kind=normalized_snapshot_kind, payload=local_payload)
                except LongTermRemoteUnavailableError as exc:
                    if self.required:
                        raise
                    _LOGGER.warning(
                        "Failed to migrate local snapshot %r to remote store: %s",
                        normalized_snapshot_kind,
                        self._safe_exception_text(exc),
                    )
                    return local_payload
            return local_payload

        return None

    def _extract_snapshot_body(
        self,
        payload: Mapping[str, object],
        *,
        snapshot_kind: str,
    ) -> dict[str, object] | None:
        candidate = self._extract_snapshot_candidate(payload, snapshot_kind=snapshot_kind)
        if candidate is None:
            return None
        return dict(candidate.payload)

    def _extract_snapshot_candidate(
        self,
        payload: Mapping[str, object],
        *,
        snapshot_kind: str,
    ) -> _RemoteSnapshotCandidate | None:
        namespace = self.namespace or _REMOTE_NAMESPACE_PREFIX
        latest: tuple[tuple[int, float, int], _RemoteSnapshotCandidate] | None = None
        for ordinal, candidate in enumerate(self._iter_snapshot_candidates(payload)):
            schema = candidate.payload.get("schema")
            if schema is not None and schema != _SNAPSHOT_SCHEMA:
                continue
            if candidate.payload.get("namespace") != namespace:
                continue
            if candidate.payload.get("snapshot_kind") != snapshot_kind:
                continue
            body = candidate.payload.get("body")
            if isinstance(body, Mapping):
                match = _RemoteSnapshotCandidate(
                    payload=dict(body),
                    document_id=candidate.document_id,
                )
                sort_key = (*_snapshot_updated_at_sort_key(candidate.payload.get("updated_at")), ordinal)
                if latest is None:
                    latest = (sort_key, match)
                    continue
                latest_sort_key, _latest_candidate = latest
                if sort_key >= latest_sort_key:
                    latest = (sort_key, match)
        return None if latest is None else latest[1]

    def _iter_snapshot_candidates(self, payload: Mapping[str, object]) -> Iterable[_RemoteSnapshotCandidate]:
        top_level_document_id = _candidate_document_id(payload)
        yield _RemoteSnapshotCandidate(payload=payload, document_id=top_level_document_id)

        direct = payload.get("payload")
        if isinstance(direct, Mapping):
            yield _RemoteSnapshotCandidate(payload=direct, document_id=top_level_document_id)

        nested = payload.get("record")
        if isinstance(nested, Mapping):
            nested_payload = nested.get("payload")
            if isinstance(nested_payload, Mapping):
                yield _RemoteSnapshotCandidate(
                    payload=nested_payload,
                    document_id=_candidate_document_id(nested) or top_level_document_id,
                )
            nested_content = nested.get("content")
            if isinstance(nested_content, str):
                parsed = self._parse_snapshot_content(nested_content)
                if parsed is not None:
                    yield _RemoteSnapshotCandidate(
                        payload=parsed,
                        document_id=_candidate_document_id(nested) or top_level_document_id,
                    )

        content = payload.get("content")
        if isinstance(content, str):
            parsed = self._parse_snapshot_content(content)
            if parsed is not None:
                yield _RemoteSnapshotCandidate(payload=parsed, document_id=top_level_document_id)

        chunks = payload.get("chunks")
        if isinstance(chunks, list):
            for chunk in chunks:
                if not isinstance(chunk, Mapping):
                    continue
                chunk_document_id = _candidate_document_id(chunk) or top_level_document_id
                chunk_payload = chunk.get("payload")
                if isinstance(chunk_payload, Mapping):
                    yield _RemoteSnapshotCandidate(payload=chunk_payload, document_id=chunk_document_id)
                chunk_content = chunk.get("content")
                if isinstance(chunk_content, str):
                    parsed = self._parse_snapshot_content(chunk_content)
                    if parsed is not None:
                        yield _RemoteSnapshotCandidate(payload=parsed, document_id=chunk_document_id)

    def _parse_snapshot_content(self, content: str) -> Mapping[str, object] | None:
        try:
            parsed = json.loads(
                content,
                parse_constant=_reject_non_finite_json_constant,
            )
        except (json.JSONDecodeError, ValueError):
            return None
        if isinstance(parsed, Mapping):
            return parsed
        return None

    def _load_snapshot_via_uri(
        self,
        client: ChonkyDBClient,
        *,
        snapshot_kind: str,
        local_path: Path | None = None,
        document_id: str | None = None,
        source: str,
        retry_not_found: bool = False,
        prefer_metadata_only: bool = False,
        bypass_circuit_open: bool = False,
        fast_fail: bool = False,
    ) -> _RemoteSnapshotFetchResult:
        if not bypass_circuit_open and self._circuit_is_open():
            return _RemoteSnapshotFetchResult(
                status="unavailable",
                detail="Remote long-term memory is temporarily cooling down after recent failures.",
                selected_source=source,
            )

        last_error: Exception | None = None
        attempts = 1 if fast_fail else self._retry_attempts()
        backoff_s = 0.0 if fast_fail else self._retry_backoff_s()
        attempt_records: list[LongTermRemoteFetchAttempt] = []
        started = time.monotonic()
        effective_client = client
        if document_id is None and not fast_fail:
            effective_client = self._origin_resolution_client(client)
        for attempt in range(attempts):
            attempt_started = time.monotonic()
            try:
                payload = self._fetch_snapshot_document(
                    effective_client,
                    snapshot_kind=snapshot_kind,
                    local_path=local_path,
                    document_id=document_id,
                    prefer_metadata_only=prefer_metadata_only,
                )
            except Exception as exc:
                latency_ms = round(max(0.0, (time.monotonic() - attempt_started) * 1000.0), 3)
                status_code = self._status_code_from_exception(exc)
                if isinstance(exc, ChonkyDBError) and status_code == 404:
                    self._note_remote_success()
                    attempt_records.append(
                        LongTermRemoteFetchAttempt(
                            source=source,
                            attempt=attempt + 1,
                            status="not_found",
                            latency_ms=latency_ms,
                            status_code=status_code,
                            document_id=document_id,
                            error_type=type(exc).__name__,
                        )
                    )
                    if retry_not_found and attempt + 1 < attempts:
                        if backoff_s > 0:
                            time.sleep(backoff_s)
                        continue
                    return _RemoteSnapshotFetchResult(
                        status="not_found",
                        latency_ms=round(max(0.0, (time.monotonic() - started) * 1000.0), 3),
                        selected_source=source,
                        attempts=tuple(attempt_records),
                    )
                last_error = exc
                self._note_remote_failure()
                attempt_records.append(
                    LongTermRemoteFetchAttempt(
                        source=source,
                        attempt=attempt + 1,
                        status="error",
                        latency_ms=latency_ms,
                        status_code=status_code,
                        document_id=document_id,
                        error_type=type(exc).__name__,
                        detail=self._safe_exception_text(exc),
                    )
                )
                if attempt + 1 >= attempts:
                    context = self._snapshot_read_context(
                        snapshot_kind=snapshot_kind,
                        source=source,
                        client=effective_client,
                        document_id=document_id,
                        attempt_index=attempt + 1,
                        attempt_count=attempts,
                    )
                    record_remote_read_diagnostic(
                        remote_state=self,
                        context=context,
                        exc=exc,
                        started_monotonic=started,
                        outcome="failed",
                    )
                    _LOGGER.warning(
                        "Failed to read remote long-term snapshot %r: %s",
                        snapshot_kind,
                        self._safe_exception_text(exc),
                    )
                    return _RemoteSnapshotFetchResult(
                        status="unavailable",
                        detail=self._remote_failure_detail("read", snapshot_kind, exc=exc),
                        latency_ms=round(max(0.0, (time.monotonic() - started) * 1000.0), 3),
                        selected_source=source,
                        attempts=tuple(attempt_records),
                    )
                if backoff_s > 0:
                    time.sleep(backoff_s)
                continue

            if not isinstance(payload, Mapping):
                self._note_remote_failure()
                _LOGGER.warning(
                    "Remote long-term snapshot %r returned payload type %s instead of Mapping.",
                    snapshot_kind,
                    type(payload).__name__,
                )
                attempt_records.append(
                    LongTermRemoteFetchAttempt(
                        source=source,
                        attempt=attempt + 1,
                        status="malformed",
                        latency_ms=round(max(0.0, (time.monotonic() - attempt_started) * 1000.0), 3),
                        document_id=document_id,
                        detail=f"payload_type={type(payload).__name__}",
                    )
                )
                if source != "pointer_document" and attempt + 1 < attempts:
                    if backoff_s > 0:
                        time.sleep(backoff_s)
                    continue
                context = self._snapshot_read_context(
                    snapshot_kind=snapshot_kind,
                    source=source,
                    client=effective_client,
                    document_id=document_id,
                    attempt_index=attempt + 1,
                    attempt_count=attempts,
                )
                record_remote_read_diagnostic(
                    remote_state=self,
                    context=context,
                    exc=TypeError(
                        f"Remote long-term snapshot {snapshot_kind!r} returned payload type {type(payload).__name__}."
                    ),
                    started_monotonic=started,
                    outcome="failed",
                )
                return _RemoteSnapshotFetchResult(
                    status="unavailable",
                    detail=(
                        f"Remote long-term snapshot {snapshot_kind!r} returned malformed content "
                        "that Twinr could not parse."
                    ),
                    latency_ms=round(max(0.0, (time.monotonic() - started) * 1000.0), 3),
                    selected_source=source,
                    attempts=tuple(attempt_records),
                )

            self._note_remote_success()
            direct = self._extract_snapshot_candidate(payload, snapshot_kind=snapshot_kind)
            if direct is not None:
                context = self._snapshot_read_context(
                    snapshot_kind=snapshot_kind,
                    source=source,
                    client=effective_client,
                    document_id=document_id,
                    attempt_index=attempt + 1,
                    attempt_count=attempts,
                )
                record_remote_read_observation(
                    remote_state=self,
                    context=context,
                    latency_ms=max(0.0, (time.monotonic() - started) * 1000.0),
                    outcome="ok",
                    classification="ok",
                )
                attempt_records.append(
                    LongTermRemoteFetchAttempt(
                        source=source,
                        attempt=attempt + 1,
                        status="found",
                        latency_ms=round(max(0.0, (time.monotonic() - attempt_started) * 1000.0), 3),
                        document_id=direct.document_id or document_id,
                    )
                )
                return _RemoteSnapshotFetchResult(
                    status="found",
                    payload=dict(direct.payload),
                    document_id=direct.document_id,
                    latency_ms=round(max(0.0, (time.monotonic() - started) * 1000.0), 3),
                    selected_source=source,
                    attempts=tuple(attempt_records),
                )
            attempt_records.append(
                LongTermRemoteFetchAttempt(
                    source=source,
                    attempt=attempt + 1,
                    status="malformed",
                    latency_ms=round(max(0.0, (time.monotonic() - attempt_started) * 1000.0), 3),
                    document_id=document_id,
                    detail="snapshot_candidate_missing",
                )
            )
            if source != "pointer_document" and attempt + 1 < attempts:
                if backoff_s > 0:
                    time.sleep(backoff_s)
                continue
            context = self._snapshot_read_context(
                snapshot_kind=snapshot_kind,
                source=source,
                client=effective_client,
                document_id=document_id,
                attempt_index=attempt + 1,
                attempt_count=attempts,
            )
            record_remote_read_diagnostic(
                remote_state=self,
                context=context,
                exc=ValueError(
                    f"Remote long-term snapshot {snapshot_kind!r} returned malformed content without a parseable snapshot candidate."
                ),
                started_monotonic=started,
                outcome="failed",
            )
            return _RemoteSnapshotFetchResult(
                status="unavailable",
                detail=(
                    f"Remote long-term snapshot {snapshot_kind!r} returned malformed content "
                    "that Twinr could not parse."
                ),
                latency_ms=round(max(0.0, (time.monotonic() - started) * 1000.0), 3),
                selected_source=source,
                attempts=tuple(attempt_records),
            )

        return _RemoteSnapshotFetchResult(
            status="unavailable",
            detail=self._remote_failure_detail("read", snapshot_kind, exc=last_error),
            latency_ms=round(max(0.0, (time.monotonic() - started) * 1000.0), 3),
            selected_source=source,
            attempts=tuple(attempt_records),
        )

    def _snapshot_read_context(
        self,
        *,
        snapshot_kind: str,
        source: str,
        client: ChonkyDBClient,
        document_id: str | None,
        attempt_index: int | None = None,
        attempt_count: int | None = None,
    ) -> LongTermRemoteReadContext:
        """Build structured diagnostics context for one snapshot-read request."""

        timeout_s = _coerce_float(
            getattr(getattr(client, "config", None), "timeout_s", None),
            default=_DEFAULT_REMOTE_READ_TIMEOUT_S,
            minimum=0.1,
            maximum=300.0,
        )
        return LongTermRemoteReadContext(
            snapshot_kind=snapshot_kind,
            operation="snapshot_load",
            request_method="GET",
            request_payload_kind=self._snapshot_request_payload_kind(source=source),
            document_id_hint=document_id,
            uri_hint=None if document_id else self._snapshot_uri(snapshot_kind),
            request_path="/v1/external/documents/full",
            timeout_s=timeout_s,
            namespace=self.namespace,
            attempt_index=attempt_index,
            attempt_count=attempt_count,
            retry_attempts_configured=attempt_count,
            retry_backoff_s=self._retry_backoff_s(),
            retry_mode="bounded_snapshot_read_retry" if (attempt_count or 0) > 1 else "single_attempt",
        )

    @staticmethod
    def _snapshot_request_payload_kind(*, source: str) -> str:
        """Return one bounded request-type label for snapshot-read diagnostics."""

        normalized_source = str(source or "").strip().lower()
        mapping = {
            "cached_document": "document_id_cached_head",
            "pointer_document": "document_id_pointer_head",
            "pointer_lookup": "origin_uri_pointer_lookup",
            "origin_uri": "origin_uri_snapshot_head",
        }
        return mapping.get(normalized_source, "snapshot_lookup")

    def probe_snapshot_load(
        self,
        *,
        snapshot_kind: str,
        local_path: Path | None = None,
        prefer_cached_document_id: bool = False,
        prefer_metadata_only: bool = False,
        fast_fail: bool = False,
    ) -> LongTermRemoteSnapshotProbe:
        """Probe one remote snapshot read and preserve pointer/origin evidence."""

        return self._probe_snapshot_load_internal(
            snapshot_kind=snapshot_kind,
            local_path=local_path,
            prefer_cached_document_id=prefer_cached_document_id,
            prefer_metadata_only=prefer_metadata_only,
            fast_fail=fast_fail,
        )

    def _probe_snapshot_load_internal(
        self,
        *,
        snapshot_kind: str,
        local_path: Path | None = None,
        prefer_cached_document_id: bool,
        prefer_metadata_only: bool = False,
        fast_fail: bool = False,
    ) -> LongTermRemoteSnapshotProbe:
        """Probe one remote snapshot read, optionally reusing a learned document id."""

        normalized_snapshot_kind = self._normalize_snapshot_kind(snapshot_kind)
        if not self.enabled:
            return LongTermRemoteSnapshotProbe(
                snapshot_kind=normalized_snapshot_kind,
                status="disabled",
                latency_ms=0.0,
                detail="Remote-primary long-term memory is disabled.",
            )
        try:
            client = self._require_client(self.read_client, operation="read")
        except LongTermRemoteUnavailableError as exc:
            return LongTermRemoteSnapshotProbe(
                snapshot_kind=normalized_snapshot_kind,
                status="unavailable",
                latency_ms=0.0,
                detail=str(exc),
            )
        cached_probe = self._cached_probe(snapshot_kind=normalized_snapshot_kind)
        if cached_probe is not None:
            return cached_probe
        if prefer_cached_document_id and not self._is_pointer_snapshot_kind(normalized_snapshot_kind):
            probe = self._load_snapshot_with_document_id_hint(
                client,
                snapshot_kind=normalized_snapshot_kind,
                local_path=local_path,
                prefer_metadata_only=prefer_metadata_only,
                fast_fail=fast_fail,
            )
        else:
            probe = self._load_snapshot_with_pointer_fallback(
                client,
                snapshot_kind=normalized_snapshot_kind,
                local_path=local_path,
                prefer_metadata_only=prefer_metadata_only,
                fast_fail=fast_fail,
            )
        self._store_cached_probe(probe)
        return probe

    def _load_snapshot_with_document_id_hint(
        self,
        client: ChonkyDBClient,
        *,
        snapshot_kind: str,
        local_path: Path | None = None,
        prefer_metadata_only: bool = False,
        fast_fail: bool = False,
    ) -> LongTermRemoteSnapshotProbe:
        """Probe one snapshot by a remembered document id before pointer/origin resolution."""

        started = time.monotonic()
        attempt_records: list[LongTermRemoteFetchAttempt] = []
        hinted_document_id = self._cached_snapshot_document_id(snapshot_kind=snapshot_kind)
        if hinted_document_id:
            hinted_result = self._load_snapshot_via_uri(
                client,
                snapshot_kind=snapshot_kind,
                local_path=local_path,
                document_id=hinted_document_id,
                source="cached_document",
                prefer_metadata_only=prefer_metadata_only,
                fast_fail=fast_fail,
            )
            attempt_records.extend(hinted_result.attempts)
            if hinted_result.payload is not None:
                return LongTermRemoteSnapshotProbe(
                    snapshot_kind=snapshot_kind,
                    status=hinted_result.status,
                    latency_ms=round(max(0.0, (time.monotonic() - started) * 1000.0), 3),
                    detail=hinted_result.detail,
                    document_id=hinted_result.document_id or hinted_document_id,
                    selected_source=hinted_result.selected_source,
                    payload=hinted_result.payload,
                    attempts=tuple(attempt_records),
                )
            self._forget_snapshot_document_id(snapshot_kind=snapshot_kind)

        fallback = self._load_snapshot_with_pointer_fallback(
            client,
            snapshot_kind=snapshot_kind,
            local_path=local_path,
            prefer_metadata_only=prefer_metadata_only,
            fast_fail=fast_fail,
        )
        if (
            hinted_document_id
            and hinted_result.status == "unavailable"
            and fallback.payload is None
            and fallback.status == "not_found"
        ):
            return LongTermRemoteSnapshotProbe(
                snapshot_kind=snapshot_kind,
                status="unavailable",
                latency_ms=round(max(0.0, (time.monotonic() - started) * 1000.0), 3),
                detail=hinted_result.detail,
                document_id=hinted_result.document_id or hinted_document_id,
                pointer_document_id=fallback.pointer_document_id,
                selected_source=hinted_result.selected_source,
                attempts=tuple(attempt_records) + fallback.attempts,
            )
        return LongTermRemoteSnapshotProbe(
            snapshot_kind=snapshot_kind,
            status=fallback.status,
            latency_ms=round(max(0.0, (time.monotonic() - started) * 1000.0), 3),
            detail=fallback.detail,
            document_id=fallback.document_id,
            pointer_document_id=fallback.pointer_document_id,
            selected_source=fallback.selected_source,
            payload=fallback.payload,
            attempts=tuple(attempt_records) + fallback.attempts,
        )

    def _load_snapshot_with_pointer_fallback(
        self,
        client: ChonkyDBClient,
        *,
        snapshot_kind: str,
        local_path: Path | None = None,
        prefer_metadata_only: bool = False,
        bypass_circuit_open: bool = False,
        fast_fail: bool = False,
    ) -> LongTermRemoteSnapshotProbe:
        started = time.monotonic()
        attempt_records: list[LongTermRemoteFetchAttempt] = []
        pointer_document_id: str | None = None
        pointer_result: _RemoteSnapshotFetchResult | None = None
        if not self._is_pointer_snapshot_kind(snapshot_kind):
            pointer_lookup_result = self._load_snapshot_via_uri(
                client,
                snapshot_kind=self._pointer_snapshot_kind(snapshot_kind),
                source="pointer_lookup",
                prefer_metadata_only=prefer_metadata_only,
                bypass_circuit_open=bypass_circuit_open,
                fast_fail=fast_fail,
            )
            attempt_records.extend(pointer_lookup_result.attempts)
            pointer_document_id = self._extract_pointer_document_id(
                snapshot_kind=snapshot_kind,
                pointer_payload=pointer_lookup_result.payload,
            )
            if pointer_document_id:
                pointer_result = self._load_snapshot_via_uri(
                    client,
                    snapshot_kind=snapshot_kind,
                    local_path=local_path,
                    document_id=pointer_document_id,
                    source="pointer_document",
                    prefer_metadata_only=prefer_metadata_only,
                    bypass_circuit_open=bypass_circuit_open,
                    fast_fail=fast_fail,
                )
                attempt_records.extend(pointer_result.attempts)
                if pointer_result.payload is not None:
                    return LongTermRemoteSnapshotProbe(
                        snapshot_kind=snapshot_kind,
                        status=pointer_result.status,
                        latency_ms=round(max(0.0, (time.monotonic() - started) * 1000.0), 3),
                        detail=pointer_result.detail,
                        document_id=pointer_result.document_id,
                        pointer_document_id=pointer_document_id,
                        selected_source=pointer_result.selected_source,
                        payload=pointer_result.payload,
                        attempts=tuple(attempt_records),
                    )
                _LOGGER.warning(
                    "Remote snapshot pointer %r targeted unreadable document %r; reloading the latest origin snapshot.",
                    snapshot_kind,
                    pointer_document_id,
                )

        origin_result = self._load_snapshot_via_uri(
            client,
            snapshot_kind=snapshot_kind,
            local_path=local_path,
            source="origin_uri",
            prefer_metadata_only=prefer_metadata_only,
            bypass_circuit_open=bypass_circuit_open or bool(attempt_records),
            fast_fail=fast_fail,
        )
        attempt_records.extend(origin_result.attempts)
        if origin_result.payload is not None and origin_result.document_id and not self._is_pointer_snapshot_kind(snapshot_kind):
            write_client = self.write_client
            if write_client is not None:
                try:
                    self._save_snapshot_pointer(
                        write_client,
                        snapshot_kind=snapshot_kind,
                        document_id=origin_result.document_id,
                    )
                except LongTermRemoteUnavailableError as exc:
                    _LOGGER.warning(
                        "Remote snapshot pointer %r repair failed after a successful origin read: %s",
                        snapshot_kind,
                        self._safe_exception_text(exc),
                    )
        return LongTermRemoteSnapshotProbe(
            snapshot_kind=snapshot_kind,
            status=origin_result.status,
            latency_ms=round(max(0.0, (time.monotonic() - started) * 1000.0), 3),
            detail=origin_result.detail,
            document_id=origin_result.document_id,
            pointer_document_id=pointer_document_id,
            selected_source=origin_result.selected_source,
            payload=origin_result.payload,
            attempts=tuple(attempt_records),
        )

    def _fetch_snapshot_document(
        self,
        client: ChonkyDBClient,
        *,
        snapshot_kind: str,
        local_path: Path | None,
        document_id: str | None,
        prefer_metadata_only: bool,
    ) -> object:
        """Fetch one snapshot document, preferring metadata-only reads when safe.

        Readiness and bootstrap probes only need a parseable snapshot candidate.
        ChonkyDB can often return that directly in record payload/meta without
        streaming the larger ``content`` field, so probe callers can try the
        cheaper metadata-only route first and fall back to the historical
        content-bearing request only when needed.
        """

        selector = {
            "document_id": document_id,
            "origin_uri": None if document_id else self._snapshot_uri(snapshot_kind),
        }
        if prefer_metadata_only:
            try:
                payload = client.fetch_full_document(
                    **selector,
                    include_content=False,
                    max_content_chars=self._METADATA_ONLY_MAX_CONTENT_CHARS,
                )
            except ChonkyDBError as exc:
                if int(exc.status_code or 0) != 400:
                    raise
            else:
                if isinstance(payload, Mapping):
                    candidate = self._extract_snapshot_candidate(payload, snapshot_kind=snapshot_kind)
                    if candidate is not None:
                        return payload
        return client.fetch_full_document(
            **selector,
            include_content=True,
            max_content_chars=self._max_content_chars(local_path=local_path),
        )

    def _extract_pointer_document_id(
        self,
        *,
        snapshot_kind: str,
        pointer_payload: dict[str, object] | None,
    ) -> str | None:
        if pointer_payload is None:
            return None
        if pointer_payload.get("schema") != _SNAPSHOT_POINTER_SCHEMA:
            return None
        if pointer_payload.get("version") != _SNAPSHOT_POINTER_VERSION:
            return None
        if pointer_payload.get("snapshot_kind") != snapshot_kind:
            return None
        return _normalize_snapshot_document_id(pointer_payload.get("document_id"))


__all__ = ["LongTermRemoteStateReadMixin"]
