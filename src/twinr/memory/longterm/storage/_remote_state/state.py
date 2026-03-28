"""State, cache, hint, and local-fallback helpers for remote state."""
# mypy: disable-error-code=attr-defined,call-arg,arg-type

from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import stat
import threading
import time
from typing import Iterator, Self

from twinr.memory.chonkydb import ChonkyDBClient, ChonkyDBConnectionConfig, ChonkyDBError, chonkydb_data_path

from .shared import (
    LongTermRemoteSnapshotProbe,
    LongTermRemoteStatus,
    LongTermRemoteUnavailableError,
    _CachedSnapshotRead,
    _DEFAULT_ASYNC_ATTESTATION_POLL_S,
    _DEFAULT_ASYNC_ATTESTATION_VISIBILITY_TIMEOUT_CAP_S,
    _DEFAULT_ASYNC_ATTESTATION_VISIBILITY_TIMEOUT_S,
    _DEFAULT_ASYNC_JOB_VISIBILITY_TIMEOUT_CAP_S,
    _DEFAULT_ASYNC_JOB_VISIBILITY_TIMEOUT_S,
    _DEFAULT_CIRCUIT_BREAKER_COOLDOWN_S,
    _DEFAULT_LOCAL_SNAPSHOT_MAX_BYTES,
    _DEFAULT_MAX_CONTENT_CHARS,
    _DEFAULT_MAX_CONTENT_CHARS_CAP,
    _DEFAULT_ORIGIN_RESOLUTION_BOOTSTRAP_TIMEOUT_CAP_S,
    _DEFAULT_ORIGIN_RESOLUTION_BOOTSTRAP_TIMEOUT_S,
    _DEFAULT_REMOTE_FLUSH_TIMEOUT_S,
    _DEFAULT_REMOTE_READ_TIMEOUT_S,
    _DEFAULT_REMOTE_WRITE_TIMEOUT_S,
    _DEFAULT_RETRY_ATTEMPTS,
    _DEFAULT_RETRY_BACKOFF_S,
    _DEFAULT_STATUS_PROBE_TIMEOUT_CAP_S,
    _DEFAULT_STATUS_PROBE_TIMEOUT_S,
    _DOCUMENT_ID_HINTS_FILENAME,
    _DOCUMENT_ID_HINTS_SCHEMA,
    _LOGGER,
    _MAX_DOCUMENT_HINTS_BYTES,
    _MAX_NAMESPACE_LENGTH,
    _MAX_SNAPSHOT_KIND_LENGTH,
    _REMOTE_NAMESPACE_PREFIX,
    _SNAPSHOT_POINTER_PREFIX,
    _coerce_float,
    _coerce_int,
    _coerce_timeout_s,
    _encode_uri_path_segment,
    _mapping_dict,
    _normalize_snapshot_document_id,
    _normalize_storage_token,
    _normalize_text,
    _redact_secrets,
    _reject_non_finite_json_constant,
    _remote_namespace_for_config,
    _safe_json_text,
    _safe_resolve_path,
    _strip_text,
    _utcnow_iso,
)

_STATUS_PROBE_SENTINEL_URI = "__status_probe__:remote_status"


class LongTermRemoteStateSupportMixin:  # pylint: disable=too-many-public-methods,no-member
    """Support methods shared by remote-state read and write paths."""

    config: object
    read_client: ChonkyDBClient | None
    write_client: ChonkyDBClient | None
    namespace: str | None
    _state_lock: threading.Lock
    _circuit_open_until_monotonic: float
    _consecutive_failures: int
    _probe_cache_depth: int
    _probe_cache: dict[str, LongTermRemoteSnapshotProbe]
    _document_id_hints: dict[str, str]
    _snapshot_read_cache: dict[str, _CachedSnapshotRead]

    def __post_init__(self) -> None:
        """Normalize the remote namespace once during construction."""

        namespace = self.namespace or _remote_namespace_for_config(self.config)
        self.namespace = _normalize_storage_token(
            namespace,
            field_name="namespace",
            max_length=_MAX_NAMESPACE_LENGTH,
        )
        self._load_persisted_document_id_hints()

    @classmethod
    def from_config(cls, config) -> Self:
        """Build a remote snapshot adapter from Twinr configuration."""

        namespace = _remote_namespace_for_config(config)
        if not (config.long_term_memory_enabled and config.long_term_memory_mode == "remote_primary"):
            return cls(config=config, namespace=namespace)

        base_url = _strip_text(config.chonkydb_base_url)
        api_key = _strip_text(config.chonkydb_api_key)
        if not (base_url and api_key):
            return cls(config=config, namespace=namespace)

        try:
            read_client = ChonkyDBClient(
                ChonkyDBConnectionConfig(
                    base_url=base_url,
                    api_key=api_key,
                    api_key_header=config.chonkydb_api_key_header,
                    allow_bearer_auth=config.chonkydb_allow_bearer_auth,
                    timeout_s=_coerce_timeout_s(
                        config.long_term_memory_remote_read_timeout_s,
                        default=_DEFAULT_REMOTE_READ_TIMEOUT_S,
                    ),
                    max_response_bytes=config.chonkydb_max_response_bytes,
                )
            )
            write_client = ChonkyDBClient(
                ChonkyDBConnectionConfig(
                    base_url=base_url,
                    api_key=api_key,
                    api_key_header=config.chonkydb_api_key_header,
                    allow_bearer_auth=config.chonkydb_allow_bearer_auth,
                    timeout_s=_coerce_timeout_s(
                        config.long_term_memory_remote_write_timeout_s,
                        default=_DEFAULT_REMOTE_WRITE_TIMEOUT_S,
                    ),
                    max_response_bytes=config.chonkydb_max_response_bytes,
                )
            )
        except Exception as exc:
            _LOGGER.warning(
                "Failed to initialize ChonkyDB clients: %s",
                _redact_secrets(f"{type(exc).__name__}: {exc}", secrets=(api_key,)),
            )
            return cls(config=config, namespace=namespace)
        return cls(
            config=config,
            read_client=read_client,
            write_client=write_client,
            namespace=namespace,
        )

    @property
    def enabled(self) -> bool:
        """Return whether remote-primary long-term memory is active."""

        return self.config.long_term_memory_enabled and self.config.long_term_memory_mode == "remote_primary"

    @property
    def required(self) -> bool:
        """Return whether remote-primary failures must fail closed."""

        return self.enabled and self.config.long_term_memory_remote_required

    def status(self) -> LongTermRemoteStatus:
        """Probe whether the remote snapshot backend is ready for use."""

        if not self.enabled:
            return LongTermRemoteStatus(mode="disabled", ready=False)
        if self._circuit_is_open():
            return LongTermRemoteStatus(
                mode="remote_primary",
                ready=False,
                detail="Remote long-term memory is temporarily cooling down after recent failures.",
            )
        if self.read_client is None or self.write_client is None:
            return LongTermRemoteStatus(mode="remote_primary", ready=False, detail="ChonkyDB is not configured.")
        try:
            instance = self._status_probe_client(self.read_client).instance()
        except Exception as exc:
            if self._status_probe_document_liveness():
                self._note_remote_success()
                return LongTermRemoteStatus(mode="remote_primary", ready=True)
            self._note_remote_failure()
            _LOGGER.warning("ChonkyDB health check failed: %s", self._safe_exception_text(exc))
            return LongTermRemoteStatus(
                mode="remote_primary",
                ready=False,
                detail=f"ChonkyDB health check failed ({type(exc).__name__}).",
            )
        self._note_remote_success()
        if not bool(getattr(instance, "ready", False)):
            return LongTermRemoteStatus(
                mode="remote_primary",
                ready=False,
                detail="ChonkyDB instance responded but is not ready.",
            )
        return LongTermRemoteStatus(mode="remote_primary", ready=True)

    def _status_probe_document_liveness(self) -> bool:
        """Return whether one bounded full-document probe proves the backend is live.

        The public ``/v1/external/instance`` route can occasionally stall while the
        normal exact-document read path is already responsive again. In that case,
        a synthetic missing-document lookup is still sufficient to prove that the
        remote API, auth, and routing path are alive. The deeper readiness probe
        still has to prove the real Twinr snapshot namespace afterwards.
        """

        if self.read_client is None:
            return False
        client = self._status_probe_client(self.read_client)
        try:
            client.fetch_full_document(
                origin_uri=_STATUS_PROBE_SENTINEL_URI,
                include_content=False,
                max_content_chars=0,
            )
        except ChonkyDBError as exc:
            return int(exc.status_code or 0) == 404
        except Exception:
            return False
        return True

    @contextmanager
    def cache_probe_reads(self) -> Iterator[None]:
        """Reuse successful snapshot probes within one bounded readiness scope."""

        with self._state_lock:
            self._probe_cache_depth += 1
            if self._probe_cache_depth == 1:
                self._probe_cache.clear()
        try:
            yield
        finally:
            with self._state_lock:
                if self._probe_cache_depth > 0:
                    self._probe_cache_depth -= 1
                if self._probe_cache_depth == 0:
                    self._probe_cache.clear()

    def attest_external_readiness(self) -> None:
        """Clear local cooldown state after an external required-remote proof."""

        with self._state_lock:
            self._consecutive_failures = 0
            self._circuit_open_until_monotonic = 0.0
            self._probe_cache.clear()

    def _require_client(self, client: ChonkyDBClient | None, *, operation: str) -> ChonkyDBClient:
        if client is not None:
            return client
        raise LongTermRemoteUnavailableError(
            f"Remote-primary long-term memory is enabled but ChonkyDB is not configured for {operation} operations."
        )

    def _snapshot_uri(self, snapshot_kind: str) -> str:
        namespace_segment = _encode_uri_path_segment(self.namespace or _REMOTE_NAMESPACE_PREFIX)
        snapshot_segment = _encode_uri_path_segment(self._normalize_snapshot_kind(snapshot_kind))
        return f"twinr://longterm/{namespace_segment}/{snapshot_segment}"

    def _pointer_snapshot_kind(self, snapshot_kind: str) -> str:
        return f"{_SNAPSHOT_POINTER_PREFIX}{snapshot_kind}"

    def _is_pointer_snapshot_kind(self, snapshot_kind: str) -> bool:
        return snapshot_kind.startswith(_SNAPSHOT_POINTER_PREFIX)

    def _retry_attempts(self) -> int:
        return _coerce_int(
            getattr(self.config, "long_term_memory_remote_retry_attempts", _DEFAULT_RETRY_ATTEMPTS),
            default=_DEFAULT_RETRY_ATTEMPTS,
            minimum=1,
            maximum=10,
        )

    def _retry_backoff_s(self) -> float:
        return _coerce_float(
            getattr(self.config, "long_term_memory_remote_retry_backoff_s", _DEFAULT_RETRY_BACKOFF_S),
            default=_DEFAULT_RETRY_BACKOFF_S,
            minimum=0.0,
            maximum=30.0,
        )

    def _max_content_chars(self, *, local_path: Path | None) -> int:
        cap = self._max_content_chars_cap()
        configured = _coerce_int(
            getattr(self.config, "long_term_memory_remote_max_content_chars", _DEFAULT_MAX_CONTENT_CHARS),
            default=_DEFAULT_MAX_CONTENT_CHARS,
            minimum=1,
            maximum=cap,
        )
        if local_path is None:
            return configured
        candidate_path = self._validated_local_snapshot_path(local_path, warn=False)
        if candidate_path is None:
            return configured
        try:
            path_stat = candidate_path.lstat()
        except OSError:
            return configured
        if not stat.S_ISREG(path_stat.st_mode):
            return configured
        requested = max(configured, int(path_stat.st_size) + 131_072)
        return min(requested, cap)

    def _max_content_chars_cap(self) -> int:
        return _coerce_int(
            getattr(
                self.config,
                "long_term_memory_remote_max_content_chars_cap",
                _DEFAULT_MAX_CONTENT_CHARS_CAP,
            ),
            default=_DEFAULT_MAX_CONTENT_CHARS_CAP,
            minimum=_DEFAULT_MAX_CONTENT_CHARS,
            maximum=64 * 1024 * 1024,
        )

    def _local_snapshot_max_bytes(self) -> int:
        return _coerce_int(
            getattr(
                self.config,
                "long_term_memory_remote_local_snapshot_max_bytes",
                _DEFAULT_LOCAL_SNAPSHOT_MAX_BYTES,
            ),
            default=_DEFAULT_LOCAL_SNAPSHOT_MAX_BYTES,
            minimum=1,
            maximum=64 * 1024 * 1024,
        )

    def _circuit_breaker_cooldown_s(self) -> float:
        return _coerce_float(
            getattr(
                self.config,
                "long_term_memory_remote_circuit_breaker_cooldown_s",
                _DEFAULT_CIRCUIT_BREAKER_COOLDOWN_S,
            ),
            default=_DEFAULT_CIRCUIT_BREAKER_COOLDOWN_S,
            minimum=0.0,
            maximum=300.0,
        )

    def _circuit_is_open(self) -> bool:
        with self._state_lock:
            return time.monotonic() < self._circuit_open_until_monotonic

    def _note_remote_success(self) -> None:
        with self._state_lock:
            self._consecutive_failures = 0
            self._circuit_open_until_monotonic = 0.0

    def _note_remote_failure(self) -> None:
        cooldown_s = self._circuit_breaker_cooldown_s()
        if cooldown_s <= 0:
            return
        with self._state_lock:
            self._consecutive_failures += 1
            multiplier = min(self._consecutive_failures, 4)
            self._circuit_open_until_monotonic = max(
                self._circuit_open_until_monotonic,
                time.monotonic() + (cooldown_s * multiplier),
            )

    def _remote_failure_detail(
        self,
        operation: str,
        snapshot_kind: str,
        *,
        exc: Exception | None = None,
    ) -> str:
        if exc is None:
            return f"Failed to {operation} remote long-term snapshot {snapshot_kind!r}."
        return f"Failed to {operation} remote long-term snapshot {snapshot_kind!r} ({type(exc).__name__})."

    @staticmethod
    def _status_code_from_exception(exc: Exception) -> int | None:
        """Extract an HTTP-like status code from ChonkyDB exceptions when present."""

        if isinstance(exc, ChonkyDBError):
            return exc.status_code
        return None

    def _safe_exception_text(self, exc: BaseException) -> str:
        return _redact_secrets(
            f"{type(exc).__name__}: {exc}",
            secrets=(self.config.chonkydb_api_key,),
        )

    def _normalize_snapshot_kind(self, snapshot_kind: str) -> str:
        return _normalize_storage_token(
            snapshot_kind,
            field_name="snapshot_kind",
            max_length=_MAX_SNAPSHOT_KIND_LENGTH,
        )

    def _clear_probe_cache(self) -> None:
        with self._state_lock:
            self._probe_cache.clear()

    def _clear_cached_probe(self, *, snapshot_kind: str) -> None:
        with self._state_lock:
            self._probe_cache.pop(snapshot_kind, None)

    def _remote_read_cache_ttl_s(self) -> float:
        try:
            ttl_s = float(getattr(self.config, "long_term_memory_remote_read_cache_ttl_s", 0.0))
        except (TypeError, ValueError):
            return 0.0
        return ttl_s if math.isfinite(ttl_s) and ttl_s > 0.0 else 0.0

    def _cached_snapshot_read(self, *, snapshot_kind: str) -> dict[str, object] | None:
        if self._remote_read_cache_ttl_s() <= 0.0:
            return None
        now = time.monotonic()
        with self._state_lock:
            cached = self._snapshot_read_cache.get(snapshot_kind)
            if cached is None:
                return None
            if cached.expires_at_monotonic <= now:
                self._snapshot_read_cache.pop(snapshot_kind, None)
                return None
            payload = dict(cached.payload)
            document_id = cached.document_id
        if document_id:
            self._remember_snapshot_document_id(snapshot_kind=snapshot_kind, document_id=document_id)
        return payload

    def _store_snapshot_read(
        self,
        *,
        snapshot_kind: str,
        payload: object,
        document_id: str | None,
    ) -> None:
        ttl_s = self._remote_read_cache_ttl_s()
        normalized_document_id = _normalize_snapshot_document_id(document_id)
        if ttl_s > 0.0 and normalized_document_id is not None:
            self._remember_snapshot_document_id(
                snapshot_kind=snapshot_kind,
                document_id=normalized_document_id,
            )
        if ttl_s <= 0.0:
            return
        payload_dict = _mapping_dict(payload)
        if payload_dict is None:
            return
        entry = _CachedSnapshotRead(
            payload=payload_dict,
            expires_at_monotonic=time.monotonic() + ttl_s,
            document_id=normalized_document_id,
        )
        with self._state_lock:
            self._snapshot_read_cache[snapshot_kind] = entry

    def _clear_snapshot_read(self, *, snapshot_kind: str) -> None:
        with self._state_lock:
            self._snapshot_read_cache.pop(snapshot_kind, None)

    def _cached_snapshot_document_id(self, *, snapshot_kind: str) -> str | None:
        with self._state_lock:
            return self._document_id_hints.get(snapshot_kind)

    def _remember_snapshot_document_id(
        self,
        *,
        snapshot_kind: str,
        document_id: str,
        persist: bool = False,
    ) -> None:
        normalized_document_id = _normalize_snapshot_document_id(document_id)
        if normalized_document_id is None:
            return
        snapshot_hints: dict[str, str] | None = None
        with self._state_lock:
            self._document_id_hints[snapshot_kind] = normalized_document_id
            if persist:
                snapshot_hints = self._persistent_document_id_hints_locked()
        if snapshot_hints is not None:
            self._persist_document_id_hints(snapshot_hints=snapshot_hints)

    def _forget_snapshot_document_id(self, *, snapshot_kind: str) -> None:
        snapshot_hints: dict[str, str] | None = None
        with self._state_lock:
            removed = self._document_id_hints.pop(snapshot_kind, None)
            self._snapshot_read_cache.pop(snapshot_kind, None)
            if removed is not None:
                snapshot_hints = self._persistent_document_id_hints_locked()
        if snapshot_hints is not None:
            self._persist_document_id_hints(snapshot_hints=snapshot_hints)

    def _cached_probe(self, *, snapshot_kind: str) -> LongTermRemoteSnapshotProbe | None:
        with self._state_lock:
            cached = self._probe_cache.get(snapshot_kind)
        if cached is None:
            return None
        payload = dict(cached.payload) if isinstance(cached.payload, dict) else cached.payload
        return LongTermRemoteSnapshotProbe(
            snapshot_kind=cached.snapshot_kind,
            status=cached.status,
            latency_ms=0.0,
            detail=cached.detail,
            document_id=cached.document_id,
            pointer_document_id=cached.pointer_document_id,
            selected_source=cached.selected_source,
            payload=payload,
            attempts=cached.attempts,
        )

    def _origin_resolution_client(self, client: ChonkyDBClient) -> ChonkyDBClient:
        """Return one client tuned for cold `origin_uri` resolution."""

        target_timeout_s = self._origin_resolution_bootstrap_timeout_s()
        if float(client.config.timeout_s) >= target_timeout_s:
            return client
        return client.clone_with_timeout(target_timeout_s)

    def _async_attestation_visibility_timeout_s(self) -> float:
        """Return the bounded visibility window for async same-URI readback."""

        read_timeout_s = _coerce_timeout_s(
            getattr(self.config, "long_term_memory_remote_read_timeout_s", _DEFAULT_REMOTE_READ_TIMEOUT_S),
            default=_DEFAULT_REMOTE_READ_TIMEOUT_S,
        )
        write_timeout_s = _coerce_timeout_s(
            getattr(self.config, "long_term_memory_remote_write_timeout_s", _DEFAULT_REMOTE_WRITE_TIMEOUT_S),
            default=_DEFAULT_REMOTE_WRITE_TIMEOUT_S,
        )
        candidate = max(
            _DEFAULT_ASYNC_ATTESTATION_VISIBILITY_TIMEOUT_S,
            self._status_probe_timeout_s(),
            read_timeout_s,
            min(write_timeout_s * 2.0, _DEFAULT_ASYNC_ATTESTATION_VISIBILITY_TIMEOUT_CAP_S),
        )
        return _coerce_float(
            candidate,
            default=_DEFAULT_ASYNC_ATTESTATION_VISIBILITY_TIMEOUT_S,
            minimum=_DEFAULT_ASYNC_ATTESTATION_POLL_S,
            maximum=_DEFAULT_ASYNC_ATTESTATION_VISIBILITY_TIMEOUT_CAP_S,
        )

    def _remote_flush_timeout_s(self) -> float:
        """Return the bounded end-to-end remote flush budget for one snapshot write."""

        return _coerce_timeout_s(
            getattr(self.config, "long_term_memory_remote_flush_timeout_s", _DEFAULT_REMOTE_FLUSH_TIMEOUT_S),
            default=_DEFAULT_REMOTE_FLUSH_TIMEOUT_S,
        )

    def _async_job_visibility_timeout_s(self) -> float:
        """Return the bounded window for async job completion plus exact-id exposure."""

        read_timeout_s = _coerce_timeout_s(
            getattr(self.config, "long_term_memory_remote_read_timeout_s", _DEFAULT_REMOTE_READ_TIMEOUT_S),
            default=_DEFAULT_REMOTE_READ_TIMEOUT_S,
        )
        flush_timeout_s = self._remote_flush_timeout_s()
        candidate = max(
            _DEFAULT_ASYNC_JOB_VISIBILITY_TIMEOUT_S,
            self._status_probe_timeout_s(),
            flush_timeout_s,
            min(flush_timeout_s + read_timeout_s, _DEFAULT_ASYNC_JOB_VISIBILITY_TIMEOUT_CAP_S),
        )
        return _coerce_float(
            candidate,
            default=_DEFAULT_ASYNC_JOB_VISIBILITY_TIMEOUT_S,
            minimum=_DEFAULT_ASYNC_ATTESTATION_POLL_S,
            maximum=_DEFAULT_ASYNC_JOB_VISIBILITY_TIMEOUT_CAP_S,
        )

    def _status_probe_client(self, client: ChonkyDBClient) -> ChonkyDBClient:
        """Return one client tuned for fail-closed remote health probes."""

        target_timeout_s = self._status_probe_timeout_s()
        if float(client.config.timeout_s) >= target_timeout_s:
            return client
        return client.clone_with_timeout(target_timeout_s)

    def _status_probe_timeout_s(self) -> float:
        """Return the bounded timeout used for remote health and readiness probes."""

        read_timeout_s = _coerce_timeout_s(
            getattr(self.config, "long_term_memory_remote_read_timeout_s", _DEFAULT_REMOTE_READ_TIMEOUT_S),
            default=_DEFAULT_REMOTE_READ_TIMEOUT_S,
        )
        chonky_timeout_s = _coerce_timeout_s(
            getattr(self.config, "chonkydb_timeout_s", _DEFAULT_STATUS_PROBE_TIMEOUT_S),
            default=_DEFAULT_STATUS_PROBE_TIMEOUT_S,
        )
        candidate = max(
            _DEFAULT_STATUS_PROBE_TIMEOUT_S,
            read_timeout_s * 2.0,
            chonky_timeout_s,
        )
        return _coerce_float(
            candidate,
            default=_DEFAULT_STATUS_PROBE_TIMEOUT_S,
            minimum=read_timeout_s,
            maximum=_DEFAULT_STATUS_PROBE_TIMEOUT_CAP_S,
        )

    def _origin_resolution_bootstrap_timeout_s(self) -> float:
        """Return the bounded timeout used for cold current-head resolution."""

        read_timeout_s = _coerce_timeout_s(
            getattr(self.config, "long_term_memory_remote_read_timeout_s", _DEFAULT_REMOTE_READ_TIMEOUT_S),
            default=_DEFAULT_REMOTE_READ_TIMEOUT_S,
        )
        candidate = max(
            _DEFAULT_ORIGIN_RESOLUTION_BOOTSTRAP_TIMEOUT_S,
            read_timeout_s * 3.0,
        )
        return _coerce_float(
            candidate,
            default=_DEFAULT_ORIGIN_RESOLUTION_BOOTSTRAP_TIMEOUT_S,
            minimum=read_timeout_s,
            maximum=_DEFAULT_ORIGIN_RESOLUTION_BOOTSTRAP_TIMEOUT_CAP_S,
        )

    def _document_id_hints_path(self) -> Path | None:
        """Return the local durable hint path when the storage root is valid."""

        try:
            base_dir = chonkydb_data_path(self.config)
        except Exception:
            return None
        return base_dir / _DOCUMENT_ID_HINTS_FILENAME

    def _load_persisted_document_id_hints(self) -> None:
        """Load durable write-learned snapshot document ids from local state."""

        hints_path = self._document_id_hints_path()
        if hints_path is None:
            return
        try:
            path_stat = hints_path.lstat()
        except OSError:
            return
        if not stat.S_ISREG(path_stat.st_mode):
            return
        if path_stat.st_size > _MAX_DOCUMENT_HINTS_BYTES:
            _LOGGER.warning(
                "Ignoring remote snapshot document-id hints at %s because the file exceeds %d bytes.",
                hints_path,
                _MAX_DOCUMENT_HINTS_BYTES,
            )
            return
        try:
            payload = json.loads(
                hints_path.read_text(encoding="utf-8"),
                parse_constant=_reject_non_finite_json_constant,
            )
        except (OSError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        if payload.get("schema") != _DOCUMENT_ID_HINTS_SCHEMA:
            return
        if _normalize_text(payload.get("namespace")) != self.namespace:
            return
        raw_hints = payload.get("document_ids")
        if not isinstance(raw_hints, dict):
            return
        loaded_hints: dict[str, str] = {}
        for raw_kind, raw_document_id in raw_hints.items():
            try:
                normalized_kind = self._normalize_snapshot_kind(str(raw_kind))
            except ValueError:
                continue
            if self._is_pointer_snapshot_kind(normalized_kind):
                continue
            normalized_document_id = _normalize_snapshot_document_id(raw_document_id)
            if normalized_document_id is None:
                continue
            loaded_hints[normalized_kind] = normalized_document_id
        if not loaded_hints:
            return
        with self._state_lock:
            self._document_id_hints.update(loaded_hints)

    def _persistent_document_id_hints_locked(self) -> dict[str, str]:
        """Return the persistable document-id hints while the state lock is held."""

        return {
            snapshot_kind: document_id
            for snapshot_kind, document_id in self._document_id_hints.items()
            if not self._is_pointer_snapshot_kind(snapshot_kind)
        }

    def _persist_document_id_hints(self, *, snapshot_hints: dict[str, str]) -> None:
        """Atomically persist the current write-learned snapshot document ids."""

        hints_path = self._document_id_hints_path()
        if hints_path is None:
            return
        try:
            hints_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            _LOGGER.warning(
                "Failed to create remote snapshot hint directory %s: %s",
                hints_path.parent,
                self._safe_exception_text(exc),
            )
            return
        payload = {
            "schema": _DOCUMENT_ID_HINTS_SCHEMA,
            "namespace": self.namespace,
            "updated_at": _utcnow_iso(),
            "document_ids": dict(sorted(snapshot_hints.items())),
        }
        try:
            json_text = _safe_json_text(payload)
        except (TypeError, ValueError):
            return
        temp_path = hints_path.with_name(
            f"{hints_path.name}.tmp-{os.getpid()}-{threading.get_ident()}-{time.time_ns()}"
        )
        try:
            with open(temp_path, "w", encoding="utf-8") as handle:
                handle.write(json_text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, hints_path)
        except OSError as exc:
            _LOGGER.warning(
                "Failed to persist remote snapshot document-id hints at %s: %s",
                hints_path,
                self._safe_exception_text(exc),
            )
            try:
                temp_path.unlink()
            except OSError:
                pass

    def _store_cached_probe(self, probe: LongTermRemoteSnapshotProbe) -> None:
        if probe.snapshot_kind.startswith(_SNAPSHOT_POINTER_PREFIX):
            return
        with self._state_lock:
            if self._probe_cache_depth <= 0:
                return
            payload = dict(probe.payload) if isinstance(probe.payload, dict) else probe.payload
            self._probe_cache[probe.snapshot_kind] = LongTermRemoteSnapshotProbe(
                snapshot_kind=probe.snapshot_kind,
                status=probe.status,
                latency_ms=probe.latency_ms,
                detail=probe.detail,
                document_id=probe.document_id,
                pointer_document_id=probe.pointer_document_id,
                selected_source=probe.selected_source,
                payload=payload,
                attempts=probe.attempts,
            )

    def _validated_local_snapshot_path(self, local_path: Path, *, warn: bool = True) -> Path | None:
        project_root = _safe_resolve_path(Path(self.config.project_root))
        candidate = local_path if local_path.is_absolute() else (project_root / local_path)
        candidate_absolute = Path(os.path.abspath(os.fspath(candidate)))
        resolved_target = _safe_resolve_path(candidate_absolute)

        configured_memory_path = Path(self.config.long_term_memory_path)
        configured_candidate = (
            configured_memory_path
            if configured_memory_path.is_absolute()
            else (project_root / configured_memory_path)
        )
        configured_absolute = Path(os.path.abspath(os.fspath(configured_candidate)))
        resolved_configured_target = _safe_resolve_path(configured_absolute)

        if configured_absolute.exists() and configured_absolute.is_dir():
            allowed = resolved_target.is_relative_to(resolved_configured_target)
        else:
            allowed = resolved_target == resolved_configured_target

        if not allowed:
            if warn:
                _LOGGER.warning(
                    "Skipped local snapshot fallback %s because the caller supplied a path outside the configured Twinr memory root %s. This is a probe/operator path issue, not corrupted memory data.",
                    candidate_absolute,
                    configured_absolute,
                )
            return None
        return candidate_absolute

    def _load_local_snapshot(self, local_path: Path, *, snapshot_kind: str) -> dict[str, object] | None:
        candidate_path = self._validated_local_snapshot_path(local_path)
        if candidate_path is None:
            return None

        open_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(candidate_path, open_flags)
        except FileNotFoundError:
            return None
        except OSError as exc:
            _LOGGER.warning(
                "Failed to open local snapshot fallback %s for %r: %s",
                candidate_path,
                snapshot_kind,
                self._safe_exception_text(exc),
            )
            return None

        try:
            file_stat = os.fstat(fd)
            if not stat.S_ISREG(file_stat.st_mode):
                _LOGGER.warning(
                    "Rejected local snapshot fallback %s for %r because it is not a regular file.",
                    candidate_path,
                    snapshot_kind,
                )
                return None
            if file_stat.st_size > self._local_snapshot_max_bytes():
                _LOGGER.warning(
                    "Rejected local snapshot fallback %s for %r because it exceeds %s bytes.",
                    candidate_path,
                    snapshot_kind,
                    self._local_snapshot_max_bytes(),
                )
                return None
            with os.fdopen(fd, "r", encoding="utf-8") as handle:
                fd = -1
                try:
                    payload = json.load(
                        handle,
                        parse_constant=_reject_non_finite_json_constant,
                    )
                except (json.JSONDecodeError, UnicodeDecodeError, ValueError, OSError) as exc:
                    _LOGGER.warning(
                        "Failed to parse local snapshot fallback %s for %r: %s",
                        candidate_path,
                        snapshot_kind,
                        self._safe_exception_text(exc),
                    )
                    return None
        finally:
            if fd >= 0:
                os.close(fd)

        if not isinstance(payload, dict):
            _LOGGER.warning(
                "Rejected local snapshot fallback %s for %r because the JSON root is %s, not an object.",
                candidate_path,
                snapshot_kind,
                type(payload).__name__,
            )
            return None
        return dict(payload)


__all__ = ["LongTermRemoteStateSupportMixin"]
