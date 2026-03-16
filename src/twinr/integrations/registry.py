from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping
from threading import RLock
from typing import Any

from twinr.integrations.adapter import IntegrationAdapter
from twinr.integrations.models import IntegrationDecision, IntegrationRequest, IntegrationResult
from twinr.integrations.policy import SafeIntegrationPolicy


class IntegrationRegistryError(RuntimeError):
    pass


class IntegrationNotFoundError(IntegrationRegistryError):
    pass


# AUDIT-FIX(#4): Distinguish adapter/request/policy contract violations from runtime integration failures.
class IntegrationContractError(IntegrationRegistryError):
    pass


# AUDIT-FIX(#1): Surface policy evaluation failures as deterministic registry errors instead of raw exceptions.
class IntegrationPolicyEvaluationError(IntegrationRegistryError):
    def __init__(self, integration_id: str, operation_id: str) -> None:
        super().__init__(f"Policy evaluation failed for {integration_id}:{operation_id}.")
        self.integration_id = integration_id
        self.operation_id = operation_id


# AUDIT-FIX(#2): Surface adapter execution failures as deterministic registry errors instead of raw exceptions.
class IntegrationExecutionError(IntegrationRegistryError):
    def __init__(self, integration_id: str, operation_id: str) -> None:
        super().__init__(f"Integration {integration_id}:{operation_id} failed during execution.")
        self.integration_id = integration_id
        self.operation_id = operation_id


class IntegrationPolicyError(IntegrationRegistryError):
    def __init__(self, decision: IntegrationDecision) -> None:
        # AUDIT-FIX(#7): Fall back to a stable operator-facing message when the policy omits a usable reason.
        reason = getattr(decision, "reason", None)
        if not isinstance(reason, str) or not reason.strip():
            reason = "Integration request denied by policy."
        super().__init__(reason)
        self.decision = decision


class IntegrationRegistry:
    def __init__(self, adapters: tuple[IntegrationAdapter, ...] = ()) -> None:
        self._lock = RLock()
        self._adapters: dict[str, IntegrationAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    # AUDIT-FIX(#4): Fail closed on malformed identifiers so bad IDs cannot corrupt routing or leak into errors/logs.
    @staticmethod
    def _validate_identifier(value: object, *, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise IntegrationContractError(f"{field_name} must be a non-empty string.")
        if any(char in value for char in ("\x00", "\r", "\n")):
            raise IntegrationContractError(f"{field_name} contains unsupported control characters.")
        return value

    # AUDIT-FIX(#4): Validate that adapters expose a readable manifest with a usable integration ID.
    @classmethod
    def _manifest_and_integration_id(cls, adapter: IntegrationAdapter) -> tuple[object, str]:
        if adapter is None:
            raise IntegrationContractError("Adapter must not be None.")
        try:
            manifest = adapter.manifest
        except Exception as exc:  # pragma: no cover - defensive contract guard
            raise IntegrationContractError("Adapter must expose a readable manifest.") from exc

        integration_id = cls._validate_identifier(
            getattr(manifest, "integration_id", None),
            field_name="adapter.manifest.integration_id",
        )
        return manifest, integration_id

    # AUDIT-FIX(#4): Reject malformed dispatch requests early instead of failing later with AttributeError or truthiness bugs.
    @classmethod
    def _validate_request(cls, request: IntegrationRequest) -> tuple[str, str, bool]:
        if request is None:
            raise IntegrationContractError("Request must not be None.")

        integration_id = cls._validate_identifier(
            getattr(request, "integration_id", None),
            field_name="request.integration_id",
        )
        operation_id = cls._validate_identifier(
            getattr(request, "operation_id", None),
            field_name="request.operation_id",
        )
        dry_run = getattr(request, "dry_run", None)
        if not isinstance(dry_run, bool):
            raise IntegrationContractError("request.dry_run must be a boolean.")
        return integration_id, operation_id, dry_run

    # AUDIT-FIX(#4): Validate the policy object up front so bad dependencies fail closed with a deterministic error.
    @staticmethod
    def _validate_policy(policy: SafeIntegrationPolicy) -> None:
        if policy is None or not callable(getattr(policy, "evaluate", None)):
            raise IntegrationContractError("Policy must expose an evaluate(manifest, request) method.")

    # AUDIT-FIX(#2): Normalize warnings/redacted fields without assuming a specific container implementation.
    @staticmethod
    def _normalize_string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            items: Iterable[object] = (value,)
        else:
            if not isinstance(value, Iterable):
                raise IntegrationContractError(f"{field_name} must be an iterable of strings.")
            items = value

        normalized: list[str] = []
        for item in items:
            if not isinstance(item, str):
                raise IntegrationContractError(f"{field_name} must contain only strings.")
            normalized.append(item)
        return tuple(normalized)

    # AUDIT-FIX(#2): Copy result payloads defensively and reject malformed detail mappings before they trigger TypeError later.
    @staticmethod
    def _copy_details(value: object, *, field_name: str) -> dict[str, Any]:
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

        for key in details:
            if not isinstance(key, str):
                raise IntegrationContractError(f"{field_name} keys must be strings.")
        return details

    # AUDIT-FIX(#1): Detect accidentally-async policies and malformed decision objects before they leak coroutine objects downstream.
    @classmethod
    def _validate_decision(
        cls,
        decision: IntegrationDecision,
        *,
        integration_id: str,
        operation_id: str,
    ) -> tuple[IntegrationDecision, bool, tuple[str, ...]]:
        if inspect.isawaitable(decision):
            raise IntegrationContractError(
                f"Policy evaluate() returned an awaitable for {integration_id}:{operation_id}; "
                "this registry expects a synchronous decision."
            )

        allowed = getattr(decision, "allowed", None)
        if not isinstance(allowed, bool):
            raise IntegrationContractError(
                f"Policy decision for {integration_id}:{operation_id} must provide a boolean allowed flag."
            )

        warnings = cls._normalize_string_tuple(
            getattr(decision, "warnings", ()),
            field_name="IntegrationDecision.warnings",
        )
        return decision, allowed, warnings

    # AUDIT-FIX(#2): Rebuild returned results with validated fields so malformed adapters cannot poison downstream callers.
    @classmethod
    def _coerce_result(
        cls,
        result: IntegrationResult,
        *,
        integration_id: str,
        operation_id: str,
        additional_warnings: tuple[str, ...] = (),
    ) -> IntegrationResult:
        if inspect.isawaitable(result):
            raise IntegrationContractError(
                f"Adapter execute() returned an awaitable for {integration_id}:{operation_id}; "
                "this registry expects a synchronous IntegrationResult."
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

        details = cls._copy_details(
            getattr(result, "details", {}),
            field_name="IntegrationResult.details",
        )
        warnings = cls._normalize_string_tuple(
            getattr(result, "warnings", ()),
            field_name="IntegrationResult.warnings",
        )
        redacted_fields = cls._normalize_string_tuple(
            getattr(result, "redacted_fields", ()),
            field_name="IntegrationResult.redacted_fields",
        )

        return IntegrationResult(
            ok=ok,
            summary=summary,
            details=details,
            warnings=warnings + additional_warnings,
            redacted_fields=redacted_fields,
        )

    # AUDIT-FIX(#6): Re-read the manifest under the registry lock and fail closed if a mutable adapter drifted after registration.
    def _require_adapter_and_manifest(self, integration_id: str) -> tuple[IntegrationAdapter, object]:
        normalized_integration_id = self._validate_identifier(
            integration_id,
            field_name="integration_id",
        )
        with self._lock:
            adapter = self._adapters.get(normalized_integration_id)
            if adapter is None:
                raise IntegrationNotFoundError(
                    f"Integration {normalized_integration_id} is not registered."
                )

            manifest, manifest_integration_id = self._manifest_and_integration_id(adapter)
            if manifest_integration_id != normalized_integration_id:
                raise IntegrationContractError(
                    "Registered adapter manifest.integration_id changed after registration: "
                    f"{normalized_integration_id} -> {manifest_integration_id}."
                )
            return adapter, manifest

    def register(self, adapter: IntegrationAdapter, *, replace: bool = False) -> None:
        # AUDIT-FIX(#5): Enforce an actual boolean so truthy strings like "false" cannot silently replace adapters.
        if not isinstance(replace, bool):
            raise IntegrationContractError("replace must be a boolean.")

        integration_id = self._manifest_and_integration_id(adapter)[1]
        with self._lock:
            if integration_id in self._adapters and not replace:
                raise IntegrationRegistryError(f"Integration {integration_id} is already registered.")
            self._adapters[integration_id] = adapter

    def get(self, integration_id: str) -> IntegrationAdapter | None:
        # AUDIT-FIX(#4): Preserve lookup semantics for callers while refusing malformed identifiers.
        try:
            normalized_integration_id = self._validate_identifier(
                integration_id,
                field_name="integration_id",
            )
        except IntegrationContractError:
            return None

        with self._lock:
            return self._adapters.get(normalized_integration_id)

    def require(self, integration_id: str) -> IntegrationAdapter:
        normalized_integration_id = self._validate_identifier(
            integration_id,
            field_name="integration_id",
        )
        with self._lock:
            adapter = self._adapters.get(normalized_integration_id)
            if adapter is None:
                raise IntegrationNotFoundError(
                    f"Integration {normalized_integration_id} is not registered."
                )
            return adapter

    # AUDIT-FIX(#8): Tighten the manifest collection return type for Python 3.11 callers and static analysis.
    def manifests(self) -> tuple[object, ...]:
        with self._lock:
            manifests: list[object] = []
            for integration_id in sorted(self._adapters):
                manifest, manifest_integration_id = self._manifest_and_integration_id(
                    self._adapters[integration_id]
                )
                # AUDIT-FIX(#6): Detect mutable manifest drift during read paths as well, not only during dispatch.
                if manifest_integration_id != integration_id:
                    raise IntegrationContractError(
                        "Registered adapter manifest.integration_id changed after registration: "
                        f"{integration_id} -> {manifest_integration_id}."
                    )
                manifests.append(manifest)
            return tuple(manifests)

    def dispatch(self, request: IntegrationRequest, *, policy: SafeIntegrationPolicy) -> IntegrationResult:
        integration_id, operation_id, dry_run = self._validate_request(request)
        self._validate_policy(policy)

        adapter, manifest = self._require_adapter_and_manifest(integration_id)

        try:
            # AUDIT-FIX(#1): Contain policy evaluation faults so callers can recover predictably and keep the session alive.
            decision = policy.evaluate(manifest, request)
        except IntegrationRegistryError:
            raise
        except Exception as exc:  # pragma: no cover - defensive wrapper for external policy code
            raise IntegrationPolicyEvaluationError(integration_id, operation_id) from exc

        decision, allowed, decision_warnings = self._validate_decision(
            decision,
            integration_id=integration_id,
            operation_id=operation_id,
        )
        if not allowed:
            raise IntegrationPolicyError(decision)

        if dry_run:
            redacted_parameters = getattr(request, "redacted_parameters", None)
            if not callable(redacted_parameters):
                raise IntegrationContractError(
                    "Request must expose a callable redacted_parameters() method."
                )
            try:
                # AUDIT-FIX(#3): Guard dry-run redaction and copy the payload so malformed request objects fail deterministically.
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

            return IntegrationResult(
                ok=True,
                summary=f"Dry run approved for {integration_id}:{operation_id}.",
                details={"request": request_details},
                warnings=decision_warnings,
            )

        execute = getattr(adapter, "execute", None)
        if not callable(execute):
            raise IntegrationContractError("Adapter must expose a callable execute(request) method.")

        try:
            # AUDIT-FIX(#2): Contain adapter failures and validate the returned result contract.
            result = execute(request)
        except IntegrationRegistryError:
            raise
        except Exception as exc:  # pragma: no cover - defensive wrapper for external adapter code
            raise IntegrationExecutionError(integration_id, operation_id) from exc

        return self._coerce_result(
            result,
            integration_id=integration_id,
            operation_id=operation_id,
            additional_warnings=decision_warnings,
        )