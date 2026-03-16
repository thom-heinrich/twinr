"""Evaluate safety policy for Twinr integration requests.

This module turns manifests plus normalized requests into allow or deny
decisions while keeping unknown or malformed state fail-closed.
"""

from __future__ import annotations

import logging  # AUDIT-FIX(#2): Record unexpected policy-evaluation failures and fail closed instead of crashing the caller.
from dataclasses import dataclass, field
from enum import Enum
from typing import Final, TypeVar

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

logger = logging.getLogger(__name__)

EnumT = TypeVar("EnumT", bound=Enum)

_CONFIRMATION_RANK: Final[dict[ConfirmationMode, int]] = {
    ConfirmationMode.NONE: 0,
    ConfirmationMode.USER: 1,
    ConfirmationMode.CAREGIVER: 2,
}
_POLICY_TRUE_VALUES: Final[frozenset[str]] = frozenset(
    {"1", "true", "yes", "on"}
)  # AUDIT-FIX(#1): Parse booleans from .env/file-backed config explicitly instead of trusting Python truthiness.
_POLICY_FALSE_VALUES: Final[frozenset[str]] = frozenset({"", "0", "false", "no", "off"})


def _normalize_policy_bool(value: object, *, field_name: str) -> bool:
    """Parse tolerant config-style booleans into strict ``bool`` values."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in _POLICY_TRUE_VALUES:
            return True
        if normalized in _POLICY_FALSE_VALUES:
            return False
    raise TypeError(f"{field_name} must be a boolean-compatible value.")


def _require_bool(value: object, *, field_name: str) -> bool:
    """Require a real boolean value."""

    if isinstance(value, bool):  # AUDIT-FIX(#3): Confirmation and trigger flags must be real booleans so truthy strings cannot authorize actions.
        return value
    raise TypeError(f"{field_name} must be a boolean.")


def _require_enum(value: object, *, enum_type: type[EnumT], field_name: str) -> EnumT:
    """Require an enum member of the expected type."""

    if isinstance(value, enum_type):
        return value
    raise TypeError(f"{field_name} must be a {enum_type.__name__}.")


def _require_non_negative_int(value: object, *, field_name: str) -> int:
    """Require a non-negative integer."""

    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    raise TypeError(f"{field_name} must be a non-negative integer.")


def _normalize_identifier(value: object, *, field_name: str) -> str:
    """Normalize one identifier-like policy value."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    return normalized


def _normalize_identifier_set(values: object, *, field_name: str) -> frozenset[str]:
    """Normalize one identifier collection into a frozen set."""

    if values is None:
        return frozenset()

    if isinstance(values, str):
        stripped_values = values.strip()
        if not stripped_values:
            return frozenset()
        raw_items = stripped_values.split(",")  # AUDIT-FIX(#1): Support common comma-separated .env representations instead of per-character membership bugs.
    else:
        try:
            raw_items = tuple(values)
        except TypeError as exc:
            raise TypeError(f"{field_name} must be an iterable of identifiers.") from exc

    normalized_items: set[str] = set()
    for raw_item in raw_items:
        if isinstance(raw_item, str) and not raw_item.strip():
            continue
        normalized_items.add(
            _normalize_identifier(raw_item, field_name=field_name)
        )  # AUDIT-FIX(#5): Canonicalize identifiers once so allow/block checks cannot drift on whitespace.
    return frozenset(normalized_items)


def _confirmation_rank(mode: ConfirmationMode) -> int:
    """Map confirmation modes onto an ordering from weaker to stronger."""

    return _CONFIRMATION_RANK.get(
        mode, len(_CONFIRMATION_RANK)
    )  # AUDIT-FIX(#4): Treat unknown future confirmation modes as stricter than every known mode.


def stricter_confirmation(*modes: ConfirmationMode) -> ConfirmationMode:
    """Return the strictest confirmation mode from a set of candidates."""

    return max(modes, key=_confirmation_rank)


@dataclass(slots=True)
class SafeIntegrationPolicy:
    """Evaluate whether a normalized integration request may proceed."""

    enabled_integrations: frozenset[str] = field(default_factory=frozenset)
    blocked_integrations: frozenset[str] = field(default_factory=frozenset)
    blocked_operations: frozenset[str] = field(default_factory=frozenset)
    allow_remote_requests: bool = False
    allow_background_polling: bool = False
    allow_critical_operations: bool = False

    def __post_init__(self) -> None:
        """Normalize allowlists and boolean flags at construction time."""

        self.enabled_integrations = _normalize_identifier_set(
            self.enabled_integrations,
            field_name="enabled_integrations",
        )  # AUDIT-FIX(#1): Freeze and sanitize .env/file-backed allowlists at construction time.
        self.blocked_integrations = _normalize_identifier_set(
            self.blocked_integrations,
            field_name="blocked_integrations",
        )
        self.blocked_operations = _normalize_identifier_set(
            self.blocked_operations,
            field_name="blocked_operations",
        )
        self.allow_remote_requests = _normalize_policy_bool(
            self.allow_remote_requests,
            field_name="allow_remote_requests",
        )
        self.allow_background_polling = _normalize_policy_bool(
            self.allow_background_polling,
            field_name="allow_background_polling",
        )
        self.allow_critical_operations = _normalize_policy_bool(
            self.allow_critical_operations,
            field_name="allow_critical_operations",
        )

    def evaluate(self, manifest: IntegrationManifest, request: IntegrationRequest) -> IntegrationDecision:
        """Evaluate one request and fail closed on unexpected errors."""

        try:
            return self._evaluate_impl(manifest, request)
        except Exception:
            logger.exception(
                "SafeIntegrationPolicy evaluation failed closed."
            )  # AUDIT-FIX(#2): Unexpected manifest/request defects now produce a deterministic deny instead of a caller-visible crash.
            return IntegrationDecision.deny(
                "This action was stopped because the request could not be checked safely."
            )

    def _evaluate_impl(self, manifest: IntegrationManifest, request: IntegrationRequest) -> IntegrationDecision:
        """Evaluate one request against the normalized policy rules."""

        integration_id = _normalize_identifier(
            getattr(manifest, "integration_id"),
            field_name="manifest.integration_id",
        )  # AUDIT-FIX(#5): Canonicalize runtime identifiers before every policy lookup.
        operation_id = _normalize_identifier(
            getattr(request, "operation_id"),
            field_name="request.operation_id",
        )
        origin = _require_enum(
            getattr(request, "origin"),
            enum_type=RequestOrigin,
            field_name="request.origin",
        )
        background_trigger = _require_bool(
            getattr(request, "background_trigger"),
            field_name="request.background_trigger",
        )
        dry_run = _require_bool(
            getattr(request, "dry_run"),
            field_name="request.dry_run",
        )
        explicit_user_confirmation = _require_bool(
            getattr(request, "explicit_user_confirmation"),
            field_name="request.explicit_user_confirmation",
        )
        explicit_caregiver_confirmation = _require_bool(
            getattr(request, "explicit_caregiver_confirmation"),
            field_name="request.explicit_caregiver_confirmation",
        )

        operation = manifest.operation(operation_id)
        if operation is None:
            return IntegrationDecision.deny("The requested integration operation is unknown.")

        if integration_id in self.blocked_integrations:
            return IntegrationDecision.deny("This integration has been blocked.")

        if operation_id in self.blocked_operations or f"{integration_id}:{operation_id}" in self.blocked_operations:
            return IntegrationDecision.deny("This integration operation has been blocked.")

        if integration_id not in self.enabled_integrations:
            return IntegrationDecision.deny("This integration is not enabled.")

        safety = getattr(operation, "safety")
        domain = _require_enum(
            getattr(manifest, "domain"),
            enum_type=IntegrationDomain,
            field_name="manifest.domain",
        )
        action = _require_enum(
            getattr(operation, "action"),
            enum_type=IntegrationAction,
            field_name="operation.action",
        )
        confirmation = _require_enum(
            getattr(safety, "confirmation"),
            enum_type=ConfirmationMode,
            field_name="operation.safety.confirmation",
        )
        allow_remote_trigger = _require_bool(
            getattr(safety, "allow_remote_trigger"),
            field_name="operation.safety.allow_remote_trigger",
        )
        allow_operation_background_polling = _require_bool(
            getattr(safety, "allow_background_polling"),
            field_name="operation.safety.allow_background_polling",
        )
        max_payload_bytes = _require_non_negative_int(
            getattr(safety, "max_payload_bytes"),
            field_name="operation.safety.max_payload_bytes",
        )
        risk = _require_enum(
            getattr(safety, "risk"),
            enum_type=RiskLevel,
            field_name="operation.safety.risk",
        )
        payload_size_bytes = _require_non_negative_int(
            request.payload_size_bytes(),
            field_name="request.payload_size_bytes()",
        )

        if origin is RequestOrigin.REMOTE_SERVICE:
            if not self.allow_remote_requests or not allow_remote_trigger:
                return IntegrationDecision.deny("Remote-triggered integration requests are disabled.")

        if background_trigger:
            if not self.allow_background_polling or not allow_operation_background_polling:
                return IntegrationDecision.deny("Background polling is not allowed for this request.")

        if payload_size_bytes > max_payload_bytes:
            return IntegrationDecision.deny("The integration payload exceeds the allowed size.")

        if risk is RiskLevel.CRITICAL and not self.allow_critical_operations:
            return IntegrationDecision.deny("Critical integration operations are disabled.")

        required_confirmation = stricter_confirmation(
            confirmation,
            self._baseline_confirmation(domain, action),
        )
        if not self._has_confirmation(
            explicit_user_confirmation=explicit_user_confirmation,
            explicit_caregiver_confirmation=explicit_caregiver_confirmation,
            required_confirmation=required_confirmation,
        ):
            return IntegrationDecision.deny(
                "This integration request needs more explicit confirmation.",
                required_confirmation=required_confirmation,
            )

        warnings: tuple[str, ...] = ()
        if dry_run:
            warnings = ("Dry run only. No external action will be executed.",)

        return IntegrationDecision.allow(
            "The integration request passed policy checks.",
            warnings=warnings,
        )

    def _baseline_confirmation(
        self,
        domain: IntegrationDomain,
        action: IntegrationAction,
    ) -> ConfirmationMode:
        """Return the minimum confirmation required for a domain/action pair."""

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

    def _has_confirmation(
        self,
        *,
        explicit_user_confirmation: bool,
        explicit_caregiver_confirmation: bool,
        required_confirmation: ConfirmationMode,
    ) -> bool:
        """Return whether the request carries enough human confirmation."""

        if required_confirmation is ConfirmationMode.NONE:
            return True
        if required_confirmation is ConfirmationMode.USER:
            return explicit_user_confirmation or explicit_caregiver_confirmation
        if required_confirmation is ConfirmationMode.CAREGIVER:
            return explicit_caregiver_confirmation
        return False  # AUDIT-FIX(#4): Unknown future confirmation modes must deny safely instead of being approximated.
