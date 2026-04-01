# CHANGELOG: 2026-03-30
# BUG-1: Dry-run results now populate redacted_fields explicitly, avoiding constructor drift
#        and inconsistent IntegrationResult instances across model implementations.
# BUG-2: Dispatch now validates optional manifest operation catalogs and rejects unknown
#        operation_ids early instead of delegating silent typos to adapters.
# SEC-1: Registry now snapshots manifests, enforces identifier/size limits, and caps
#        warnings/detail fan-out to reduce practical log-injection and memory-DoS risk
#        on Raspberry Pi 4 deployments.
# SEC-2: Policy and adapter calls now run behind bounded execution budgets so a hung or
#        malicious integration cannot stall the assistant thread indefinitely.
# IMP-1: Added async-aware dispatch (sync + async policies/adapters are both supported).
# IMP-2: Added bounded worker execution, optional structured audit hooks, and context
#        management/close() for clean lifecycle handling.
# BREAKING: Manifest changes after register() are no longer relied on for authorization;
#           adapters must re-register to change exposed manifest metadata or operation catalogs.

"""Register adapters and dispatch integration requests safely.

The registry keeps the adapter map, enforces policy decisions, validates
contracts, and normalizes failures into stable integration-layer errors.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import logging
import os
import threading
import time
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from twinr.integrations.adapter import IntegrationAdapter
from twinr.integrations.models import IntegrationDecision, IntegrationRequest, IntegrationResult
from twinr.integrations.policy import SafeIntegrationPolicy

_LOGGER = logging.getLogger(__name__)


class IntegrationRegistryError(RuntimeError):
    """Base error for registry-level integration failures."""

    pass


class IntegrationNotFoundError(IntegrationRegistryError):
    """Raise when a requested integration is not registered."""

    pass


class IntegrationOperationNotFoundError(IntegrationRegistryError):
    """Raise when the manifest declares operations and the request names an unknown one."""

    pass


class IntegrationContractError(IntegrationRegistryError):
    """Raise when adapters, manifests, requests, or policy objects break the contract."""

    pass


class IntegrationPolicyEvaluationError(IntegrationRegistryError):
    """Raise when policy evaluation fails unexpectedly."""

    def __init__(self, integration_id: str, operation_id: str) -> None:
        super().__init__(f"Policy evaluation failed for {integration_id}:{operation_id}.")
        self.integration_id = integration_id
        self.operation_id = operation_id


class IntegrationPolicyTimeoutError(IntegrationRegistryError):
    """Raise when policy evaluation exceeds the configured budget."""

    def __init__(self, integration_id: str, operation_id: str, timeout_s: float) -> None:
        super().__init__(
            f"Policy evaluation timed out for {integration_id}:{operation_id} after {timeout_s:.3f}s."
        )
        self.integration_id = integration_id
        self.operation_id = operation_id
        self.timeout_s = timeout_s


class IntegrationExecutionError(IntegrationRegistryError):
    """Raise when adapter execution fails unexpectedly."""

    def __init__(self, integration_id: str, operation_id: str) -> None:
        super().__init__(f"Integration {integration_id}:{operation_id} failed during execution.")
        self.integration_id = integration_id
        self.operation_id = operation_id


class IntegrationExecutionTimeoutError(IntegrationRegistryError):
    """Raise when adapter execution exceeds the configured budget."""

    def __init__(self, integration_id: str, operation_id: str, timeout_s: float) -> None:
        super().__init__(
            f"Integration {integration_id}:{operation_id} timed out after {timeout_s:.3f}s."
        )
        self.integration_id = integration_id
        self.operation_id = operation_id
        self.timeout_s = timeout_s


class IntegrationPolicyError(IntegrationRegistryError):
    """Raise when policy denies an otherwise valid integration request."""

    def __init__(self, decision: IntegrationDecision) -> None:
        reason = getattr(decision, "reason", None)
        if not isinstance(reason, str) or not reason.strip():
            reason = "Integration request denied by policy."
        super().__init__(reason)
        self.decision = decision


@dataclass(frozen=True, slots=True)
class IntegrationAuditEvent:
    """Structured event emitted at key registry state transitions."""

    phase: str
    integration_id: str
    operation_id: str
    dry_run: bool
    outcome: str
    duration_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _RegisteredAdapter:
    """One adapter plus the manifest snapshot and optional operation catalog."""

    adapter: IntegrationAdapter
    manifest: object
    operation_index: dict[str, dict[str, Any]]


class IntegrationRegistry:
    """Store integration adapters and dispatch requests through policy checks."""

    _DEFAULT_POLICY_TIMEOUT_S = 1.0
    _DEFAULT_EXECUTION_TIMEOUT_S = 15.0
    _DEFAULT_MAX_IDENTIFIER_LENGTH = 128
    _DEFAULT_MAX_WARNING_COUNT = 32
    _DEFAULT_MAX_REDACTED_FIELDS = 64
    _DEFAULT_MAX_DETAIL_KEYS = 256
    _DEFAULT_MAX_STRING_LENGTH = 4096

    def __init__(
        self,
        adapters: tuple[IntegrationAdapter, ...] = (),
        *,
        policy_timeout_s: float | None = _DEFAULT_POLICY_TIMEOUT_S,
        execution_timeout_s: float | None = _DEFAULT_EXECUTION_TIMEOUT_S,
        max_workers: int | None = None,
        max_identifier_length: int = _DEFAULT_MAX_IDENTIFIER_LENGTH,
        max_warning_count: int = _DEFAULT_MAX_WARNING_COUNT,
        max_redacted_fields: int = _DEFAULT_MAX_REDACTED_FIELDS,
        max_detail_keys: int = _DEFAULT_MAX_DETAIL_KEYS,
        max_string_length: int = _DEFAULT_MAX_STRING_LENGTH,
        snapshot_manifests: bool = True,
        audit_hook: Callable[[IntegrationAuditEvent], None] | None = None,
    ) -> None:
        self._lock = RLock()
        self._adapters: dict[str, _RegisteredAdapter] = {}
        self._policy_timeout_s = self._validate_timeout(policy_timeout_s, field_name="policy_timeout_s")
        self._execution_timeout_s = self._validate_timeout(
            execution_timeout_s,
            field_name="execution_timeout_s",
        )
        self._max_identifier_length = self._validate_positive_int(
            max_identifier_length,
            field_name="max_identifier_length",
        )
        self._max_warning_count = self._validate_positive_int(
            max_warning_count,
            field_name="max_warning_count",
        )
        self._max_redacted_fields = self._validate_positive_int(
            max_redacted_fields,
            field_name="max_redacted_fields",
        )
        self._max_detail_keys = self._validate_positive_int(
            max_detail_keys,
            field_name="max_detail_keys",
        )
        self._max_string_length = self._validate_positive_int(
            max_string_length,
            field_name="max_string_length",
        )
        if not isinstance(snapshot_manifests, bool):
            raise IntegrationContractError("snapshot_manifests must be a boolean.")
        self._snapshot_manifests = snapshot_manifests
        if audit_hook is not None and not callable(audit_hook):
            raise IntegrationContractError("audit_hook must be callable when provided.")
        self._audit_hook = audit_hook
        self._closed = False
        self._worker_state = threading.local()
        self._executor = ThreadPoolExecutor(
            max_workers=self._normalize_max_workers(max_workers),
            thread_name_prefix="twinr-integrations",
        )
        for adapter in adapters:
            self.register(adapter)

    @staticmethod
    def _validate_positive_int(value: object, *, field_name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise IntegrationContractError(f"{field_name} must be a positive integer.")
        return value

    @staticmethod
    def _validate_timeout(value: object, *, field_name: str) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise IntegrationContractError(f"{field_name} must be a positive number or None.")
        return float(value)

    @staticmethod
    def _normalize_max_workers(value: object) -> int:
        if value is None:
            cpu_count = os.cpu_count() or 1
            return max(4, min(8, cpu_count))
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise IntegrationContractError("max_workers must be a positive integer or None.")
        return value

    def close(self, *, wait: bool = False, cancel_futures: bool = True) -> None:
        """Release worker resources owned by this registry."""

        if not isinstance(wait, bool):
            raise IntegrationContractError("wait must be a boolean.")
        if not isinstance(cancel_futures, bool):
            raise IntegrationContractError("cancel_futures must be a boolean.")

        with self._lock:
            if self._closed:
                return
            self._closed = True
            executor = self._executor

        executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def __enter__(self) -> IntegrationRegistry:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close(wait=False, cancel_futures=True)

    def _ensure_open(self) -> None:
        if self._closed:
            raise IntegrationRegistryError("IntegrationRegistry is closed.")

    def _emit_audit_event(
        self,
        *,
        phase: str,
        integration_id: str,
        operation_id: str,
        dry_run: bool,
        outcome: str,
        duration_ms: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if self._audit_hook is None:
            return

        event = IntegrationAuditEvent(
            phase=phase,
            integration_id=integration_id,
            operation_id=operation_id,
            dry_run=dry_run,
            outcome=outcome,
            duration_ms=duration_ms,
            metadata=dict(metadata or {}),
        )
        try:
            self._audit_hook(event)
        except Exception:  # pragma: no cover - audit hooks must never break dispatch
            _LOGGER.exception("Integration audit hook failed.")

    def _validate_identifier(self, value: object, *, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise IntegrationContractError(f"{field_name} must be a non-empty string.")
        if len(value) > self._max_identifier_length:
            raise IntegrationContractError(
                f"{field_name} exceeds the maximum length of {self._max_identifier_length}."
            )
        for char in value:
            if unicodedata.category(char).startswith("C"):
                raise IntegrationContractError(
                    f"{field_name} contains unsupported control characters."
                )
        return value

    def _manifest_and_integration_id(self, adapter: IntegrationAdapter) -> tuple[object, str]:
        if adapter is None:
            raise IntegrationContractError("Adapter must not be None.")
        try:
            manifest = adapter.manifest
        except Exception as exc:  # pragma: no cover - defensive contract guard
            raise IntegrationContractError("Adapter must expose a readable manifest.") from exc

        integration_id = self._validate_identifier(
            getattr(manifest, "integration_id", None),
            field_name="adapter.manifest.integration_id",
        )
        return manifest, integration_id

    def _snapshot_manifest(self, manifest: object) -> object:
        if not self._snapshot_manifests:
            return manifest
        try:
            return copy.deepcopy(manifest)
        except Exception as exc:
            raise IntegrationContractError(
                "Adapter manifest must be deepcopyable when snapshot_manifests=True."
            ) from exc

    def _clone_read_only(self, value: object) -> object:
        try:
            return copy.deepcopy(value)
        except Exception:
            return value

    def _validate_request(self, request: IntegrationRequest) -> tuple[str, str, bool]:
        if request is None:
            raise IntegrationContractError("Request must not be None.")

        integration_id = self._validate_identifier(
            getattr(request, "integration_id", None),
            field_name="request.integration_id",
        )
        operation_id = self._validate_identifier(
            getattr(request, "operation_id", None),
            field_name="request.operation_id",
        )
        dry_run = getattr(request, "dry_run", None)
        if not isinstance(dry_run, bool):
            raise IntegrationContractError("request.dry_run must be a boolean.")
        return integration_id, operation_id, dry_run

    @staticmethod
    def _validate_policy(policy: SafeIntegrationPolicy) -> None:
        if policy is None or not callable(getattr(policy, "evaluate", None)):
            raise IntegrationContractError("Policy must expose an evaluate(manifest, request) method.")

    def _normalize_string_tuple(
        self,
        value: object,
        *,
        field_name: str,
        max_items: int,
    ) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            items: Iterable[object] = (value,)
        else:
            if not isinstance(value, Iterable):
                raise IntegrationContractError(f"{field_name} must be an iterable of strings.")
            items = value

        normalized: list[str] = []
        seen: set[str] = set()
        for item in items:
            if len(normalized) >= max_items:
                raise IntegrationContractError(
                    f"{field_name} exceeds the maximum of {max_items} items."
                )
            if not isinstance(item, str):
                raise IntegrationContractError(f"{field_name} must contain only strings.")
            if len(item) > self._max_string_length:
                raise IntegrationContractError(
                    f"{field_name} contains a string longer than {self._max_string_length} characters."
                )
            if item not in seen:
                normalized.append(item)
                seen.add(item)
        return tuple(normalized)

    def _copy_details(self, value: object, *, field_name: str) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            details = dict(value)
        else:
            try:
                details = dict(value)
            except Exception as exc:  # pragma: no cover - defensive contract guard
                raise IntegrationContractError(
                    f"{field_name} must be a mapping or iterable of key/value pairs."
                ) from exc

        if len(details) > self._max_detail_keys:
            raise IntegrationContractError(
                f"{field_name} exceeds the maximum of {self._max_detail_keys} keys."
            )

        copied: dict[str, Any] = {}
        for key, item_value in details.items():
            if not isinstance(key, str):
                raise IntegrationContractError(f"{field_name} keys must be strings.")
            if len(key) > self._max_string_length:
                raise IntegrationContractError(
                    f"{field_name} contains a key longer than {self._max_string_length} characters."
                )
            copied[key] = self._clone_read_only(item_value)
        return copied

    def _validate_decision(
        self,
        decision: IntegrationDecision,
        *,
        integration_id: str,
        operation_id: str,
    ) -> tuple[IntegrationDecision, bool, tuple[str, ...]]:
        if inspect.isawaitable(decision):
            raise IntegrationContractError(
                f"Policy evaluate() returned a nested awaitable for {integration_id}:{operation_id}; "
                "return the resolved decision instead."
            )

        allowed = getattr(decision, "allowed", None)
        if not isinstance(allowed, bool):
            raise IntegrationContractError(
                f"Policy decision for {integration_id}:{operation_id} must provide a boolean allowed flag."
            )

        warnings = self._normalize_string_tuple(
            getattr(decision, "warnings", ()),
            field_name="IntegrationDecision.warnings",
            max_items=self._max_warning_count,
        )
        return decision, allowed, warnings

    def _coerce_result(
        self,
        result: IntegrationResult,
        *,
        integration_id: str,
        operation_id: str,
        additional_warnings: tuple[str, ...] = (),
    ) -> IntegrationResult:
        if inspect.isawaitable(result):
            raise IntegrationContractError(
                f"Adapter execute() returned a nested awaitable for {integration_id}:{operation_id}; "
                "return the resolved IntegrationResult instead."
            )

        ok = getattr(result, "ok", None)
        if not isinstance(ok, bool):
            raise IntegrationContractError(
                f"Integration {integration_id}:{operation_id} returned a non-boolean ok flag."
            )

        summary = getattr(result, "summary", None)
        if not isinstance(summary, str):
            raise IntegrationContractError(
                f"Integration {integration_id}:{operation_id} returned a non-string summary."
            )
        if len(summary) > self._max_string_length:
            raise IntegrationContractError(
                f"Integration {integration_id}:{operation_id} returned an oversized summary."
            )

        details = self._copy_details(
            getattr(result, "details", {}),
            field_name="IntegrationResult.details",
        )
        warnings = self._normalize_string_tuple(
            getattr(result, "warnings", ()),
            field_name="IntegrationResult.warnings",
            max_items=self._max_warning_count,
        )
        merged_warnings = self._normalize_string_tuple(
            warnings + additional_warnings,
            field_name="IntegrationResult.warnings",
            max_items=self._max_warning_count,
        )
        redacted_fields = self._normalize_string_tuple(
            getattr(result, "redacted_fields", ()),
            field_name="IntegrationResult.redacted_fields",
            max_items=self._max_redacted_fields,
        )

        return IntegrationResult(
            ok=ok,
            summary=summary,
            details=details,
            warnings=merged_warnings,
            redacted_fields=redacted_fields,
        )

    def _extract_operation_index(self, manifest: object) -> dict[str, dict[str, Any]]:
        raw_operations = getattr(manifest, "operations", None)
        if raw_operations is None:
            raw_operations = getattr(manifest, "tools", None)
        if raw_operations is None:
            return {}

        operation_index: dict[str, dict[str, Any]] = {}

        if isinstance(raw_operations, Mapping):
            operation_items: Iterable[object] = raw_operations.items()
            for key, value in operation_items:
                operation_id, metadata = self._normalize_operation_descriptor(value, fallback_id=key)
                if operation_id in operation_index:
                    raise IntegrationContractError(
                        f"manifest.operations contains duplicate operation_id {operation_id!r}."
                    )
                operation_index[operation_id] = metadata
            return operation_index

        if isinstance(raw_operations, str) or not isinstance(raw_operations, Iterable):
            raise IntegrationContractError(
                "manifest.operations must be an iterable or mapping when provided."
            )

        for item in raw_operations:
            operation_id, metadata = self._normalize_operation_descriptor(item)
            if operation_id in operation_index:
                raise IntegrationContractError(
                    f"manifest.operations contains duplicate operation_id {operation_id!r}."
                )
            operation_index[operation_id] = metadata
        return operation_index

    def _normalize_operation_descriptor(
        self,
        value: object,
        *,
        fallback_id: object | None = None,
    ) -> tuple[str, dict[str, Any]]:
        if isinstance(value, str):
            operation_id = self._validate_identifier(
                value,
                field_name="manifest.operations[].operation_id",
            )
            return operation_id, {}

        metadata: dict[str, Any] = {}
        candidate_id: object | None = fallback_id

        if isinstance(value, Mapping):
            metadata = dict(value)
            if candidate_id is None:
                candidate_id = (
                    value.get("operation_id")
                    or value.get("id")
                    or value.get("name")
                )
        else:
            if candidate_id is None:
                candidate_id = (
                    getattr(value, "operation_id", None)
                    or getattr(value, "id", None)
                    or getattr(value, "name", None)
                )
            for attribute_name in (
                "title",
                "description",
                "annotations",
                "execution",
                "timeout",
                "timeout_s",
                "input_schema",
                "inputSchema",
                "parameters",
                "output_schema",
                "outputSchema",
            ):
                if hasattr(value, attribute_name):
                    metadata[attribute_name] = getattr(value, attribute_name)

        operation_id = self._validate_identifier(
            candidate_id,
            field_name="manifest.operations[].operation_id",
        )
        return operation_id, self._copy_details(
            metadata,
            field_name="manifest.operations[].metadata",
        )

    def _effective_timeout(
        self,
        *,
        default_timeout_s: float | None,
        operation_metadata: Mapping[str, Any],
    ) -> float | None:
        timeout_candidate = operation_metadata.get("timeout_s", operation_metadata.get("timeout"))
        if timeout_candidate is None:
            return default_timeout_s
        operation_timeout_s = self._validate_timeout(
            timeout_candidate,
            field_name="manifest.operations[].timeout_s",
        )
        if default_timeout_s is None:
            return operation_timeout_s
        return min(default_timeout_s, operation_timeout_s)

    def _require_record(self, integration_id: str) -> _RegisteredAdapter:
        normalized_integration_id = self._validate_identifier(
            integration_id,
            field_name="integration_id",
        )
        with self._lock:
            record = self._adapters.get(normalized_integration_id)
            if record is None:
                raise IntegrationNotFoundError(
                    f"Integration {normalized_integration_id} is not registered."
                )
            return record

    def _require_adapter_and_manifest(
        self,
        integration_id: str,
        operation_id: str,
    ) -> tuple[IntegrationAdapter, object, dict[str, Any]]:
        record = self._require_record(integration_id)
        operation_metadata = record.operation_index.get(operation_id, {})
        if record.operation_index and operation_id not in record.operation_index:
            raise IntegrationOperationNotFoundError(
                f"Integration {integration_id} does not expose operation {operation_id}."
            )
        return record.adapter, self._clone_read_only(record.manifest), dict(operation_metadata)

    def register(self, adapter: IntegrationAdapter, *, replace: bool = False) -> None:
        """Register one adapter under its manifest integration ID."""

        if not isinstance(replace, bool):
            raise IntegrationContractError("replace must be a boolean.")

        manifest, integration_id = self._manifest_and_integration_id(adapter)
        manifest_snapshot = self._snapshot_manifest(manifest)
        operation_index = self._extract_operation_index(manifest_snapshot)
        record = _RegisteredAdapter(
            adapter=adapter,
            manifest=manifest_snapshot,
            operation_index=operation_index,
        )

        with self._lock:
            self._ensure_open()
            if integration_id in self._adapters and not replace:
                raise IntegrationRegistryError(f"Integration {integration_id} is already registered.")
            self._adapters[integration_id] = record

        self._emit_audit_event(
            phase="register",
            integration_id=integration_id,
            operation_id="*",
            dry_run=False,
            outcome="replaced" if replace else "registered",
            metadata={"operation_count": len(operation_index)},
        )

    def get(self, integration_id: str) -> IntegrationAdapter | None:
        """Return one registered adapter or ``None`` when absent."""

        try:
            normalized_integration_id = self._validate_identifier(
                integration_id,
                field_name="integration_id",
            )
        except IntegrationContractError:
            return None

        with self._lock:
            record = self._adapters.get(normalized_integration_id)
            return None if record is None else record.adapter

    def require(self, integration_id: str) -> IntegrationAdapter:
        """Return one registered adapter or raise ``IntegrationNotFoundError``."""

        return self._require_record(integration_id).adapter

    def manifests(self) -> tuple[object, ...]:
        """Return manifest snapshots for all registered adapters in sorted ID order."""

        with self._lock:
            manifests: list[object] = []
            for integration_id in sorted(self._adapters):
                manifests.append(self._clone_read_only(self._adapters[integration_id].manifest))
            return tuple(manifests)

    def descriptors(self) -> tuple[dict[str, Any], ...]:
        """Return normalized tool descriptors for LLM/tool runtimes when metadata exists."""

        with self._lock:
            items = tuple(sorted(self._adapters.items()))

        descriptors: list[dict[str, Any]] = []
        for integration_id, record in items:
            if not record.operation_index:
                descriptors.append({"integration_id": integration_id})
                continue

            for operation_id, metadata in sorted(record.operation_index.items()):
                descriptor = {
                    "integration_id": integration_id,
                    "operation_id": operation_id,
                }
                title = metadata.get("title")
                description = metadata.get("description")
                if isinstance(title, str) and title:
                    descriptor["title"] = title
                if isinstance(description, str) and description:
                    descriptor["description"] = description

                input_schema = (
                    metadata.get("inputSchema")
                    or metadata.get("input_schema")
                    or metadata.get("parameters")
                )
                output_schema = metadata.get("outputSchema") or metadata.get("output_schema")
                if isinstance(input_schema, Mapping):
                    descriptor["input_schema"] = self._copy_details(
                        input_schema,
                        field_name="manifest.operations[].input_schema",
                    )
                if isinstance(output_schema, Mapping):
                    descriptor["output_schema"] = self._copy_details(
                        output_schema,
                        field_name="manifest.operations[].output_schema",
                    )

                annotations = metadata.get("annotations")
                if isinstance(annotations, Mapping):
                    descriptor["annotations"] = self._copy_details(
                        annotations,
                        field_name="manifest.operations[].annotations",
                    )

                descriptors.append(descriptor)
        return tuple(descriptors)

    @staticmethod
    async def _await_any(awaitable: Any) -> Any:
        return await awaitable

    def _mark_worker(self, in_worker: bool) -> None:
        self._worker_state.in_registry_worker = in_worker

    def _in_worker(self) -> bool:
        return bool(getattr(self._worker_state, "in_registry_worker", False))

    def _call_inline(self, func: Callable[..., Any], *args: object) -> Any:
        self._mark_worker(True)
        try:
            result = func(*args)
            if inspect.isawaitable(result):
                try:
                    return asyncio.run(self._await_any(result))
                except RuntimeError as exc:
                    raise IntegrationContractError(
                        "Async adapters/policies must return a fresh awaitable that is safe to "
                        "drive to completion in the registry worker."
                    ) from exc
            return result
        finally:
            self._mark_worker(False)

    def _submit_call(
        self,
        func: Callable[..., Any],
        *args: object,
        timeout_s: float | None,
        timeout_error_factory: Callable[[float], IntegrationRegistryError],
    ) -> Any:
        self._ensure_open()

        if self._in_worker():
            if timeout_s is None:
                return self._call_inline(func, *args)

            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="twinr-int-nested")
            try:
                future = executor.submit(self._call_inline, func, *args)
                try:
                    return future.result(timeout=timeout_s)
                except FutureTimeoutError as exc:
                    future.cancel()
                    raise timeout_error_factory(timeout_s) from exc
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

        try:
            future = self._executor.submit(self._call_inline, func, *args)
        except RuntimeError as exc:
            raise IntegrationRegistryError("IntegrationRegistry is closed.") from exc

        try:
            return future.result(timeout=timeout_s)
        except FutureTimeoutError as exc:
            future.cancel()
            raise timeout_error_factory(timeout_s) from exc

    @staticmethod
    def _is_async_callable(func: Callable[..., Any]) -> bool:
        return inspect.iscoroutinefunction(func) or inspect.iscoroutinefunction(
            getattr(func, "__call__", None)
        )

    async def _submit_call_async(
        self,
        func: Callable[..., Any],
        *args: object,
        timeout_s: float | None,
        timeout_error_factory: Callable[[float], IntegrationRegistryError],
    ) -> Any:
        self._ensure_open()

        if self._is_async_callable(func):
            result = func(*args)
            if inspect.isawaitable(result):
                try:
                    if timeout_s is None:
                        return await result
                    return await asyncio.wait_for(result, timeout=timeout_s)
                except asyncio.TimeoutError as exc:
                    raise timeout_error_factory(timeout_s) from exc
            return result

        if self._in_worker():
            loop = asyncio.get_running_loop()
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="twinr-int-nested")
            try:
                future = executor.submit(self._call_inline, func, *args)
                wrapped = asyncio.wrap_future(future, loop=loop)
                try:
                    if timeout_s is None:
                        return await wrapped
                    return await asyncio.wait_for(wrapped, timeout=timeout_s)
                except asyncio.TimeoutError as exc:
                    future.cancel()
                    raise timeout_error_factory(timeout_s) from exc
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

        loop = asyncio.get_running_loop()
        try:
            future = self._executor.submit(self._call_inline, func, *args)
        except RuntimeError as exc:
            raise IntegrationRegistryError("IntegrationRegistry is closed.") from exc

        wrapped = asyncio.wrap_future(future, loop=loop)
        try:
            if timeout_s is None:
                return await wrapped
            return await asyncio.wait_for(wrapped, timeout=timeout_s)
        except asyncio.TimeoutError as exc:
            future.cancel()
            raise timeout_error_factory(timeout_s) from exc

    def dispatch(self, request: IntegrationRequest, *, policy: SafeIntegrationPolicy) -> IntegrationResult:
        """Validate, authorize, and execute one integration request."""

        return self._dispatch_sync(request, policy=policy)

    async def dispatch_async(
        self,
        request: IntegrationRequest,
        *,
        policy: SafeIntegrationPolicy,
    ) -> IntegrationResult:
        """Async-friendly variant of ``dispatch()`` that avoids blocking the event loop."""

        return await self._dispatch_async(request, policy=policy)

    def _dispatch_sync(self, request: IntegrationRequest, *, policy: SafeIntegrationPolicy) -> IntegrationResult:
        integration_id, operation_id, dry_run = self._validate_request(request)
        self._validate_policy(policy)

        adapter, manifest, operation_metadata = self._require_adapter_and_manifest(
            integration_id,
            operation_id,
        )
        policy_timeout_s = self._effective_timeout(
            default_timeout_s=self._policy_timeout_s,
            operation_metadata=operation_metadata,
        )
        execution_timeout_s = self._effective_timeout(
            default_timeout_s=self._execution_timeout_s,
            operation_metadata=operation_metadata,
        )
        started_ns = time.monotonic_ns()

        try:
            decision = self._submit_call(
                policy.evaluate,
                manifest,
                request,
                timeout_s=policy_timeout_s,
                timeout_error_factory=lambda timeout_s: IntegrationPolicyTimeoutError(
                    integration_id,
                    operation_id,
                    timeout_s,
                ),
            )
        except IntegrationRegistryError:
            self._emit_audit_event(
                phase="policy",
                integration_id=integration_id,
                operation_id=operation_id,
                dry_run=dry_run,
                outcome="error",
                duration_ms=(time.monotonic_ns() - started_ns) / 1_000_000,
            )
            raise
        except Exception as exc:  # pragma: no cover - defensive wrapper for external policy code
            self._emit_audit_event(
                phase="policy",
                integration_id=integration_id,
                operation_id=operation_id,
                dry_run=dry_run,
                outcome="exception",
                duration_ms=(time.monotonic_ns() - started_ns) / 1_000_000,
            )
            raise IntegrationPolicyEvaluationError(integration_id, operation_id) from exc

        decision, allowed, decision_warnings = self._validate_decision(
            decision,
            integration_id=integration_id,
            operation_id=operation_id,
        )
        if not allowed:
            self._emit_audit_event(
                phase="policy",
                integration_id=integration_id,
                operation_id=operation_id,
                dry_run=dry_run,
                outcome="denied",
                duration_ms=(time.monotonic_ns() - started_ns) / 1_000_000,
                metadata={"warning_count": len(decision_warnings)},
            )
            raise IntegrationPolicyError(decision)

        if dry_run:
            redacted_parameters = getattr(request, "redacted_parameters", None)
            if not callable(redacted_parameters):
                raise IntegrationContractError(
                    "Request must expose a callable redacted_parameters() method."
                )
            try:
                request_details = self._copy_details(
                    redacted_parameters(),
                    field_name="IntegrationRequest.redacted_parameters()",
                )
            except IntegrationRegistryError:
                raise
            except Exception as exc:  # pragma: no cover - defensive wrapper for request models
                raise IntegrationContractError(
                    "IntegrationRequest.redacted_parameters() failed."
                ) from exc

            result = IntegrationResult(
                ok=True,
                summary=f"Dry run approved for {integration_id}:{operation_id}.",
                details={"request": request_details},
                warnings=decision_warnings,
                redacted_fields=(),
            )
            self._emit_audit_event(
                phase="dispatch",
                integration_id=integration_id,
                operation_id=operation_id,
                dry_run=True,
                outcome="ok",
                duration_ms=(time.monotonic_ns() - started_ns) / 1_000_000,
                metadata={"warning_count": len(decision_warnings)},
            )
            return result

        execute = getattr(adapter, "execute", None)
        if not callable(execute):
            raise IntegrationContractError("Adapter must expose a callable execute(request) method.")

        try:
            result = self._submit_call(
                execute,
                request,
                timeout_s=execution_timeout_s,
                timeout_error_factory=lambda timeout_s: IntegrationExecutionTimeoutError(
                    integration_id,
                    operation_id,
                    timeout_s,
                ),
            )
        except IntegrationRegistryError:
            self._emit_audit_event(
                phase="execute",
                integration_id=integration_id,
                operation_id=operation_id,
                dry_run=False,
                outcome="error",
                duration_ms=(time.monotonic_ns() - started_ns) / 1_000_000,
            )
            raise
        except Exception as exc:  # pragma: no cover - defensive wrapper for external adapter code
            self._emit_audit_event(
                phase="execute",
                integration_id=integration_id,
                operation_id=operation_id,
                dry_run=False,
                outcome="exception",
                duration_ms=(time.monotonic_ns() - started_ns) / 1_000_000,
            )
            raise IntegrationExecutionError(integration_id, operation_id) from exc

        coerced_result = self._coerce_result(
            result,
            integration_id=integration_id,
            operation_id=operation_id,
            additional_warnings=decision_warnings,
        )
        self._emit_audit_event(
            phase="dispatch",
            integration_id=integration_id,
            operation_id=operation_id,
            dry_run=False,
            outcome="ok",
            duration_ms=(time.monotonic_ns() - started_ns) / 1_000_000,
            metadata={
                "warning_count": len(coerced_result.warnings),
                "ok": coerced_result.ok,
            },
        )
        return coerced_result

    async def _dispatch_async(
        self,
        request: IntegrationRequest,
        *,
        policy: SafeIntegrationPolicy,
    ) -> IntegrationResult:
        integration_id, operation_id, dry_run = self._validate_request(request)
        self._validate_policy(policy)

        adapter, manifest, operation_metadata = self._require_adapter_and_manifest(
            integration_id,
            operation_id,
        )
        policy_timeout_s = self._effective_timeout(
            default_timeout_s=self._policy_timeout_s,
            operation_metadata=operation_metadata,
        )
        execution_timeout_s = self._effective_timeout(
            default_timeout_s=self._execution_timeout_s,
            operation_metadata=operation_metadata,
        )
        started_ns = time.monotonic_ns()

        try:
            decision = await self._submit_call_async(
                policy.evaluate,
                manifest,
                request,
                timeout_s=policy_timeout_s,
                timeout_error_factory=lambda timeout_s: IntegrationPolicyTimeoutError(
                    integration_id,
                    operation_id,
                    timeout_s,
                ),
            )
        except IntegrationRegistryError:
            self._emit_audit_event(
                phase="policy",
                integration_id=integration_id,
                operation_id=operation_id,
                dry_run=dry_run,
                outcome="error",
                duration_ms=(time.monotonic_ns() - started_ns) / 1_000_000,
            )
            raise
        except Exception as exc:  # pragma: no cover - defensive wrapper for external policy code
            self._emit_audit_event(
                phase="policy",
                integration_id=integration_id,
                operation_id=operation_id,
                dry_run=dry_run,
                outcome="exception",
                duration_ms=(time.monotonic_ns() - started_ns) / 1_000_000,
            )
            raise IntegrationPolicyEvaluationError(integration_id, operation_id) from exc

        decision, allowed, decision_warnings = self._validate_decision(
            decision,
            integration_id=integration_id,
            operation_id=operation_id,
        )
        if not allowed:
            self._emit_audit_event(
                phase="policy",
                integration_id=integration_id,
                operation_id=operation_id,
                dry_run=dry_run,
                outcome="denied",
                duration_ms=(time.monotonic_ns() - started_ns) / 1_000_000,
                metadata={"warning_count": len(decision_warnings)},
            )
            raise IntegrationPolicyError(decision)

        if dry_run:
            redacted_parameters = getattr(request, "redacted_parameters", None)
            if not callable(redacted_parameters):
                raise IntegrationContractError(
                    "Request must expose a callable redacted_parameters() method."
                )
            try:
                request_details = self._copy_details(
                    redacted_parameters(),
                    field_name="IntegrationRequest.redacted_parameters()",
                )
            except IntegrationRegistryError:
                raise
            except Exception as exc:  # pragma: no cover - defensive wrapper for request models
                raise IntegrationContractError(
                    "IntegrationRequest.redacted_parameters() failed."
                ) from exc

            result = IntegrationResult(
                ok=True,
                summary=f"Dry run approved for {integration_id}:{operation_id}.",
                details={"request": request_details},
                warnings=decision_warnings,
                redacted_fields=(),
            )
            self._emit_audit_event(
                phase="dispatch",
                integration_id=integration_id,
                operation_id=operation_id,
                dry_run=True,
                outcome="ok",
                duration_ms=(time.monotonic_ns() - started_ns) / 1_000_000,
                metadata={"warning_count": len(decision_warnings)},
            )
            return result

        execute = getattr(adapter, "execute", None)
        if not callable(execute):
            raise IntegrationContractError("Adapter must expose a callable execute(request) method.")

        try:
            result = await self._submit_call_async(
                execute,
                request,
                timeout_s=execution_timeout_s,
                timeout_error_factory=lambda timeout_s: IntegrationExecutionTimeoutError(
                    integration_id,
                    operation_id,
                    timeout_s,
                ),
            )
        except IntegrationRegistryError:
            self._emit_audit_event(
                phase="execute",
                integration_id=integration_id,
                operation_id=operation_id,
                dry_run=False,
                outcome="error",
                duration_ms=(time.monotonic_ns() - started_ns) / 1_000_000,
            )
            raise
        except Exception as exc:  # pragma: no cover - defensive wrapper for external adapter code
            self._emit_audit_event(
                phase="execute",
                integration_id=integration_id,
                operation_id=operation_id,
                dry_run=False,
                outcome="exception",
                duration_ms=(time.monotonic_ns() - started_ns) / 1_000_000,
            )
            raise IntegrationExecutionError(integration_id, operation_id) from exc

        coerced_result = self._coerce_result(
            result,
            integration_id=integration_id,
            operation_id=operation_id,
            additional_warnings=decision_warnings,
        )
        self._emit_audit_event(
            phase="dispatch",
            integration_id=integration_id,
            operation_id=operation_id,
            dry_run=False,
            outcome="ok",
            duration_ms=(time.monotonic_ns() - started_ns) / 1_000_000,
            metadata={
                "warning_count": len(coerced_result.warnings),
                "ok": coerced_result.ok,
            },
        )
        return coerced_result