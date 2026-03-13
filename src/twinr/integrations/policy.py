from __future__ import annotations

from dataclasses import dataclass, field

from twinr.integrations.models import (
    ConfirmationMode,
    IntegrationAction,
    IntegrationDecision,
    IntegrationDomain,
    IntegrationManifest,
    IntegrationRequest,
    RequestOrigin,
    RiskLevel,
)

_CONFIRMATION_RANK = {
    ConfirmationMode.NONE: 0,
    ConfirmationMode.USER: 1,
    ConfirmationMode.CAREGIVER: 2,
}


def stricter_confirmation(*modes: ConfirmationMode) -> ConfirmationMode:
    return max(modes, key=lambda mode: _CONFIRMATION_RANK[mode])


@dataclass(slots=True)
class SafeIntegrationPolicy:
    enabled_integrations: frozenset[str] = field(default_factory=frozenset)
    blocked_integrations: frozenset[str] = field(default_factory=frozenset)
    blocked_operations: frozenset[str] = field(default_factory=frozenset)
    allow_remote_requests: bool = False
    allow_background_polling: bool = False
    allow_critical_operations: bool = False

    def evaluate(self, manifest: IntegrationManifest, request: IntegrationRequest) -> IntegrationDecision:
        operation = manifest.operation(request.operation_id)
        if operation is None:
            return IntegrationDecision.deny("The requested integration operation is unknown.")

        if manifest.integration_id in self.blocked_integrations:
            return IntegrationDecision.deny("This integration has been blocked.")

        if request.operation_id in self.blocked_operations or (
            f"{manifest.integration_id}:{request.operation_id}" in self.blocked_operations
        ):
            return IntegrationDecision.deny("This integration operation has been blocked.")

        if manifest.integration_id not in self.enabled_integrations:
            return IntegrationDecision.deny("This integration is not enabled.")

        if request.origin is RequestOrigin.REMOTE_SERVICE:
            if not self.allow_remote_requests or not operation.safety.allow_remote_trigger:
                return IntegrationDecision.deny("Remote-triggered integration requests are disabled.")

        if request.background_trigger:
            if not self.allow_background_polling or not operation.safety.allow_background_polling:
                return IntegrationDecision.deny("Background polling is not allowed for this request.")

        if request.payload_size_bytes() > operation.safety.max_payload_bytes:
            return IntegrationDecision.deny("The integration payload exceeds the allowed size.")

        if operation.safety.risk is RiskLevel.CRITICAL and not self.allow_critical_operations:
            return IntegrationDecision.deny("Critical integration operations are disabled.")

        required_confirmation = stricter_confirmation(
            operation.safety.confirmation,
            self._baseline_confirmation(manifest.domain, operation.action),
        )
        if not self._has_confirmation(request, required_confirmation):
            return IntegrationDecision.deny(
                "This integration request needs more explicit confirmation.",
                required_confirmation=required_confirmation,
            )

        warnings = ()
        if request.dry_run:
            warnings = ("Dry run only. No external action will be executed.",)

        return IntegrationDecision.allow("The integration request passed policy checks.", warnings=warnings)

    def _baseline_confirmation(
        self,
        domain: IntegrationDomain,
        action: IntegrationAction,
    ) -> ConfirmationMode:
        if domain is IntegrationDomain.CALENDAR:
            if action in {IntegrationAction.WRITE, IntegrationAction.SEND, IntegrationAction.CONTROL}:
                return ConfirmationMode.USER
            return ConfirmationMode.NONE

        if domain in {IntegrationDomain.EMAIL, IntegrationDomain.MESSENGER}:
            if action in {IntegrationAction.SEND, IntegrationAction.WRITE}:
                return ConfirmationMode.USER
            return ConfirmationMode.NONE

        if domain is IntegrationDomain.SMART_HOME:
            if action is IntegrationAction.CONTROL:
                return ConfirmationMode.USER
            return ConfirmationMode.NONE

        if domain is IntegrationDomain.SECURITY:
            if action in {IntegrationAction.CONTROL, IntegrationAction.WRITE}:
                return ConfirmationMode.CAREGIVER
            return ConfirmationMode.USER

        if domain is IntegrationDomain.HEALTH:
            if action in {IntegrationAction.CONTROL, IntegrationAction.WRITE}:
                return ConfirmationMode.CAREGIVER
            return ConfirmationMode.USER

        return ConfirmationMode.NONE

    def _has_confirmation(self, request: IntegrationRequest, required_confirmation: ConfirmationMode) -> bool:
        if required_confirmation is ConfirmationMode.NONE:
            return True
        if required_confirmation is ConfirmationMode.USER:
            return request.explicit_user_confirmation or request.explicit_caregiver_confirmation
        return request.explicit_caregiver_confirmation
