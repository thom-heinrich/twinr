from __future__ import annotations

from threading import RLock

from twinr.integrations.adapter import IntegrationAdapter
from twinr.integrations.models import IntegrationDecision, IntegrationRequest, IntegrationResult
from twinr.integrations.policy import SafeIntegrationPolicy


class IntegrationRegistryError(RuntimeError):
    pass


class IntegrationNotFoundError(IntegrationRegistryError):
    pass


class IntegrationPolicyError(IntegrationRegistryError):
    def __init__(self, decision: IntegrationDecision) -> None:
        super().__init__(decision.reason)
        self.decision = decision


class IntegrationRegistry:
    def __init__(self, adapters: tuple[IntegrationAdapter, ...] = ()) -> None:
        self._lock = RLock()
        self._adapters: dict[str, IntegrationAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: IntegrationAdapter, *, replace: bool = False) -> None:
        integration_id = adapter.manifest.integration_id
        with self._lock:
            if integration_id in self._adapters and not replace:
                raise IntegrationRegistryError(f"Integration {integration_id} is already registered.")
            self._adapters[integration_id] = adapter

    def get(self, integration_id: str) -> IntegrationAdapter | None:
        with self._lock:
            return self._adapters.get(integration_id)

    def require(self, integration_id: str) -> IntegrationAdapter:
        adapter = self.get(integration_id)
        if adapter is None:
            raise IntegrationNotFoundError(f"Integration {integration_id} is not registered.")
        return adapter

    def manifests(self) -> tuple:
        with self._lock:
            return tuple(
                self._adapters[integration_id].manifest
                for integration_id in sorted(self._adapters)
            )

    def dispatch(self, request: IntegrationRequest, *, policy: SafeIntegrationPolicy) -> IntegrationResult:
        adapter = self.require(request.integration_id)
        decision = policy.evaluate(adapter.manifest, request)
        if not decision.allowed:
            raise IntegrationPolicyError(decision)

        if request.dry_run:
            return IntegrationResult(
                ok=True,
                summary=f"Dry run approved for {request.integration_id}:{request.operation_id}.",
                details={"request": request.redacted_parameters()},
                warnings=decision.warnings,
            )

        result = adapter.execute(request)
        if not decision.warnings:
            return result

        return IntegrationResult(
            ok=result.ok,
            summary=result.summary,
            details=dict(result.details),
            warnings=result.warnings + decision.warnings,
            redacted_fields=result.redacted_fields,
        )
