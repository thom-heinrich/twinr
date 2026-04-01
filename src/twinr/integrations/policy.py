# CHANGELOG: 2026-03-30
# BUG-1: Deny blocked/disabled integrations before operation resolution so denied requests cannot force lazy manifest work or plugin side effects on a Raspberry Pi.
# SEC-1: Remote/background requests can no longer self-authorize user/caregiver approval with boolean flags alone; confirmation now requires a trusted, attestable channel by default.
# SEC-2: Policy identifiers now use Unicode NFKC normalization, canonical policy keys, dangerous-character rejection, and max-length limits to stop confusable-ID bypasses and oversized-ID abuse.
# IMP-1: Added risk-adaptive confirmation floors, wildcard policy entries, and internal normalized evaluation contexts aligned with 2026 schema-first authorization patterns.
# IMP-2: Added stable policy fingerprints plus structured audit logging for every allow/deny path to improve diagnostics and incident response.

"""Evaluate safety policy for Twinr integration requests.

This module turns manifests plus normalized requests into allow or deny
decisions while keeping unknown or malformed state fail-closed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
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
_POLICY_TRUE_VALUES: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
_POLICY_FALSE_VALUES: Final[frozenset[str]] = frozenset({"", "0", "false", "no", "off"})
_DEFAULT_MAX_IDENTIFIER_LENGTH: Final[int] = 256
_POLICY_WILDCARD: Final[str] = "*"


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

    if isinstance(value, bool):
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


def _normalize_positive_int(value: object, *, field_name: str, minimum: int = 1) -> int:
    """Parse a positive integer from config-style input."""

    if isinstance(value, int) and not isinstance(value, bool) and value >= minimum:
        return value
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.isdecimal():
            parsed = int(normalized)
            if parsed >= minimum:
                return parsed
    raise TypeError(f"{field_name} must be an integer >= {minimum}.")


def _normalize_identifier(
    value: object,
    *,
    field_name: str,
    max_length: int,
) -> str:
    """Normalize one identifier-like policy value for display and lookup."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")

    normalized = unicodedata.normalize("NFKC", value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    if len(normalized) > max_length:
        raise ValueError(f"{field_name} must be at most {max_length} characters long.")

    for character in normalized:
        if unicodedata.category(character).startswith("C"):
            raise ValueError(
                f"{field_name} contains unsafe control or format characters."
            )

    return normalized


def _canonical_policy_identifier(
    value: object,
    *,
    field_name: str,
    max_length: int,
) -> str:
    """Normalize one identifier-like policy value for stable policy matching."""

    # BREAKING: policy allow/block matching canonicalizes Unicode compatibility forms and case
    # so visually or syntactically equivalent identifiers collapse to one policy identity.
    return _normalize_identifier(
        value,
        field_name=field_name,
        max_length=max_length,
    ).casefold()


def _split_identifier_values(values: object, *, field_name: str) -> tuple[object, ...]:
    """Normalize one identifier collection into a flat tuple of raw values."""

    if values is None:
        return ()

    if isinstance(values, str):
        stripped_values = values.strip()
        if not stripped_values:
            return ()
        return tuple(part for part in stripped_values.split(","))
    try:
        return tuple(values)
    except TypeError as exc:  # pragma: no cover - defensive path
        raise TypeError(f"{field_name} must be an iterable of identifiers.") from exc


def _normalize_identifier_set(
    values: object,
    *,
    field_name: str,
    max_length: int,
    canonical: bool = False,
) -> frozenset[str]:
    """Normalize one identifier collection into a frozen set."""

    normalized_items: set[str] = set()
    for raw_item in _split_identifier_values(values, field_name=field_name):
        if isinstance(raw_item, str) and not raw_item.strip():
            continue

        if canonical:
            normalized_items.add(
                _canonical_policy_identifier(
                    raw_item,
                    field_name=field_name,
                    max_length=max_length,
                )
            )
        else:
            normalized_items.add(
                _normalize_identifier(
                    raw_item,
                    field_name=field_name,
                    max_length=max_length,
                )
            )

    return frozenset(normalized_items)


def _confirmation_rank(mode: ConfirmationMode) -> int:
    """Map confirmation modes onto an ordering from weaker to stronger."""

    return _CONFIRMATION_RANK.get(mode, len(_CONFIRMATION_RANK))


def stricter_confirmation(*modes: ConfirmationMode) -> ConfirmationMode:
    """Return the strictest confirmation mode from a set of candidates."""

    return max(modes, key=_confirmation_rank)


def _enum_name(value: Enum | None) -> str | None:
    """Return an enum name for diagnostics."""

    if value is None:
        return None
    return value.name


@dataclass(frozen=True, slots=True)
class _NormalizedIdentifier:
    """A runtime identifier in lookup and policy-matching form."""

    value: str
    policy_key: str


@dataclass(frozen=True, slots=True)
class _NormalizedRequest:
    """Validated request data used by the policy engine."""

    operation_id: _NormalizedIdentifier
    origin: RequestOrigin
    background_trigger: bool
    dry_run: bool
    explicit_user_confirmation: bool
    explicit_caregiver_confirmation: bool
    payload_size_bytes: int


@dataclass(frozen=True, slots=True)
class _NormalizedManifest:
    """Validated manifest metadata used by the policy engine."""

    integration_id: _NormalizedIdentifier
    domain: IntegrationDomain


@dataclass(frozen=True, slots=True)
class _NormalizedOperationSafety:
    """Validated operation safety metadata used by the policy engine."""

    action: IntegrationAction
    confirmation: ConfirmationMode
    allow_remote_trigger: bool
    allow_background_polling: bool
    max_payload_bytes: int
    risk: RiskLevel


@lru_cache(maxsize=None)
def _baseline_confirmation_for(
    domain: IntegrationDomain,
    action: IntegrationAction,
) -> ConfirmationMode:
    """Return the minimum confirmation required for a domain/action pair."""

    if domain is IntegrationDomain.CALENDAR:
        if action in {
            IntegrationAction.WRITE,
            IntegrationAction.SEND,
            IntegrationAction.CONTROL,
        }:
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


@lru_cache(maxsize=None)
def _risk_confirmation_floor(risk: RiskLevel) -> ConfirmationMode:
    """Return a minimum confirmation mode derived from operation risk."""

    if risk is RiskLevel.CRITICAL:
        return ConfirmationMode.CAREGIVER

    high_risk = getattr(RiskLevel, "HIGH", None)
    if high_risk is not None and risk is high_risk:
        return ConfirmationMode.USER

    return ConfirmationMode.NONE


@dataclass(slots=True)
class SafeIntegrationPolicy:
    """Evaluate whether a normalized integration request may proceed."""

    enabled_integrations: frozenset[str] = field(default_factory=frozenset)
    blocked_integrations: frozenset[str] = field(default_factory=frozenset)
    blocked_operations: frozenset[str] = field(default_factory=frozenset)
    allow_remote_requests: bool = False
    allow_background_polling: bool = False
    allow_critical_operations: bool = False
    allow_remote_confirmations: bool = False  # BREAKING: remote confirmation flags are ignored unless an upstream attestation layer is explicitly trusted.
    allow_background_confirmations: bool = False  # BREAKING: background confirmation flags are ignored unless an upstream attestation layer is explicitly trusted.
    max_identifier_length: int = _DEFAULT_MAX_IDENTIFIER_LENGTH

    _enabled_integrations_canonical: frozenset[str] = field(
        init=False,
        repr=False,
        default_factory=frozenset,
    )
    _blocked_integrations_canonical: frozenset[str] = field(
        init=False,
        repr=False,
        default_factory=frozenset,
    )
    _blocked_operations_canonical: frozenset[str] = field(
        init=False,
        repr=False,
        default_factory=frozenset,
    )
    _allow_all_integrations: bool = field(init=False, repr=False, default=False)
    _deny_all_integrations: bool = field(init=False, repr=False, default=False)
    _block_all_operations: bool = field(init=False, repr=False, default=False)
    _blocked_integration_operation_wildcards: frozenset[str] = field(
        init=False,
        repr=False,
        default_factory=frozenset,
    )
    _policy_fingerprint: str = field(init=False, repr=False, default="")

    def __post_init__(self) -> None:
        """Normalize allowlists, flags, and internal indexes at construction time."""

        self.max_identifier_length = _normalize_positive_int(
            self.max_identifier_length,
            field_name="max_identifier_length",
            minimum=1,
        )

        self.enabled_integrations = _normalize_identifier_set(
            self.enabled_integrations,
            field_name="enabled_integrations",
            max_length=self.max_identifier_length,
        )
        self.blocked_integrations = _normalize_identifier_set(
            self.blocked_integrations,
            field_name="blocked_integrations",
            max_length=self.max_identifier_length,
        )
        self.blocked_operations = _normalize_identifier_set(
            self.blocked_operations,
            field_name="blocked_operations",
            max_length=self.max_identifier_length,
        )

        self._enabled_integrations_canonical = _normalize_identifier_set(
            self.enabled_integrations,
            field_name="enabled_integrations",
            max_length=self.max_identifier_length,
            canonical=True,
        )
        self._blocked_integrations_canonical = _normalize_identifier_set(
            self.blocked_integrations,
            field_name="blocked_integrations",
            max_length=self.max_identifier_length,
            canonical=True,
        )
        self._blocked_operations_canonical = _normalize_identifier_set(
            self.blocked_operations,
            field_name="blocked_operations",
            max_length=self.max_identifier_length,
            canonical=True,
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
        self.allow_remote_confirmations = _normalize_policy_bool(
            self.allow_remote_confirmations,
            field_name="allow_remote_confirmations",
        )
        self.allow_background_confirmations = _normalize_policy_bool(
            self.allow_background_confirmations,
            field_name="allow_background_confirmations",
        )

        self._allow_all_integrations = _POLICY_WILDCARD in self._enabled_integrations_canonical
        self._deny_all_integrations = _POLICY_WILDCARD in self._blocked_integrations_canonical
        self._block_all_operations = _POLICY_WILDCARD in self._blocked_operations_canonical
        self._blocked_integration_operation_wildcards = frozenset(
            value[:-2]
            for value in self._blocked_operations_canonical
            if value.endswith(":*") and len(value) > 2
        )
        self._policy_fingerprint = self._build_policy_fingerprint()

    @property
    def policy_fingerprint(self) -> str:
        """Return a stable fingerprint for the normalized policy."""

        return self._policy_fingerprint

    def evaluate(
        self,
        manifest: IntegrationManifest,
        request: IntegrationRequest,
    ) -> IntegrationDecision:
        """Evaluate one request and fail closed on unexpected errors."""

        try:
            return self._evaluate_impl(manifest, request)
        except Exception:
            logger.exception(
                "SafeIntegrationPolicy evaluation failed closed.",
                extra={"twinr_policy_fingerprint": self.policy_fingerprint},
            )
            return IntegrationDecision.deny(
                "This action was stopped because the request could not be checked safely."
            )

    def _evaluate_impl(
        self,
        manifest: IntegrationManifest,
        request: IntegrationRequest,
    ) -> IntegrationDecision:
        """Evaluate one request against the normalized policy rules."""

        request_ctx = self._normalize_request(request)
        manifest_ctx = self._normalize_manifest(manifest)

        if self._deny_all_integrations or (
            manifest_ctx.integration_id.policy_key in self._blocked_integrations_canonical
        ):
            return self._deny(
                "This integration has been blocked.",
                integration_id=manifest_ctx.integration_id,
                operation_id=request_ctx.operation_id,
                origin=request_ctx.origin,
                background_trigger=request_ctx.background_trigger,
                dry_run=request_ctx.dry_run,
                payload_size_bytes=request_ctx.payload_size_bytes,
            )

        if not self._allow_all_integrations and (
            manifest_ctx.integration_id.policy_key not in self._enabled_integrations_canonical
        ):
            return self._deny(
                "This integration is not enabled.",
                integration_id=manifest_ctx.integration_id,
                operation_id=request_ctx.operation_id,
                origin=request_ctx.origin,
                background_trigger=request_ctx.background_trigger,
                dry_run=request_ctx.dry_run,
                payload_size_bytes=request_ctx.payload_size_bytes,
            )

        operation = self._resolve_operation(manifest, request_ctx.operation_id.value)
        if operation is None:
            return self._deny(
                "The requested integration operation is unknown.",
                integration_id=manifest_ctx.integration_id,
                operation_id=request_ctx.operation_id,
                origin=request_ctx.origin,
                background_trigger=request_ctx.background_trigger,
                dry_run=request_ctx.dry_run,
                payload_size_bytes=request_ctx.payload_size_bytes,
            )

        if self._is_operation_blocked(
            integration_id=manifest_ctx.integration_id,
            operation_id=request_ctx.operation_id,
        ):
            return self._deny(
                "This integration operation has been blocked.",
                integration_id=manifest_ctx.integration_id,
                operation_id=request_ctx.operation_id,
                origin=request_ctx.origin,
                background_trigger=request_ctx.background_trigger,
                dry_run=request_ctx.dry_run,
                payload_size_bytes=request_ctx.payload_size_bytes,
            )

        operation_ctx = self._normalize_operation(operation)

        if request_ctx.origin is RequestOrigin.REMOTE_SERVICE:
            if not self.allow_remote_requests or not operation_ctx.allow_remote_trigger:
                return self._deny(
                    "Remote-triggered integration requests are disabled.",
                    integration_id=manifest_ctx.integration_id,
                    operation_id=request_ctx.operation_id,
                    origin=request_ctx.origin,
                    domain=manifest_ctx.domain,
                    action=operation_ctx.action,
                    risk=operation_ctx.risk,
                    confirmation=operation_ctx.confirmation,
                    background_trigger=request_ctx.background_trigger,
                    dry_run=request_ctx.dry_run,
                    payload_size_bytes=request_ctx.payload_size_bytes,
                )

        if request_ctx.background_trigger:
            if (
                not self.allow_background_polling
                or not operation_ctx.allow_background_polling
            ):
                return self._deny(
                    "Background polling is not allowed for this request.",
                    integration_id=manifest_ctx.integration_id,
                    operation_id=request_ctx.operation_id,
                    origin=request_ctx.origin,
                    domain=manifest_ctx.domain,
                    action=operation_ctx.action,
                    risk=operation_ctx.risk,
                    confirmation=operation_ctx.confirmation,
                    background_trigger=request_ctx.background_trigger,
                    dry_run=request_ctx.dry_run,
                    payload_size_bytes=request_ctx.payload_size_bytes,
                )

        if request_ctx.payload_size_bytes > operation_ctx.max_payload_bytes:
            return self._deny(
                "The integration payload exceeds the allowed size.",
                integration_id=manifest_ctx.integration_id,
                operation_id=request_ctx.operation_id,
                origin=request_ctx.origin,
                domain=manifest_ctx.domain,
                action=operation_ctx.action,
                risk=operation_ctx.risk,
                confirmation=operation_ctx.confirmation,
                background_trigger=request_ctx.background_trigger,
                dry_run=request_ctx.dry_run,
                payload_size_bytes=request_ctx.payload_size_bytes,
            )

        if (
            operation_ctx.risk is RiskLevel.CRITICAL
            and not self.allow_critical_operations
        ):
            return self._deny(
                "Critical integration operations are disabled.",
                integration_id=manifest_ctx.integration_id,
                operation_id=request_ctx.operation_id,
                origin=request_ctx.origin,
                domain=manifest_ctx.domain,
                action=operation_ctx.action,
                risk=operation_ctx.risk,
                confirmation=operation_ctx.confirmation,
                background_trigger=request_ctx.background_trigger,
                dry_run=request_ctx.dry_run,
                payload_size_bytes=request_ctx.payload_size_bytes,
            )

        required_confirmation = stricter_confirmation(
            operation_ctx.confirmation,
            self._baseline_confirmation(manifest_ctx.domain, operation_ctx.action),
            self._risk_confirmation_floor(operation_ctx.risk),
        )

        if (
            required_confirmation is not ConfirmationMode.NONE
            and not self._confirmation_source_is_trustworthy(
                origin=request_ctx.origin,
                background_trigger=request_ctx.background_trigger,
            )
        ):
            return self._deny(
                "This integration request needs attested human confirmation from a trusted interactive channel.",
                required_confirmation=required_confirmation,
                integration_id=manifest_ctx.integration_id,
                operation_id=request_ctx.operation_id,
                origin=request_ctx.origin,
                domain=manifest_ctx.domain,
                action=operation_ctx.action,
                risk=operation_ctx.risk,
                confirmation=required_confirmation,
                background_trigger=request_ctx.background_trigger,
                dry_run=request_ctx.dry_run,
                payload_size_bytes=request_ctx.payload_size_bytes,
            )

        if not self._has_confirmation(
            explicit_user_confirmation=request_ctx.explicit_user_confirmation,
            explicit_caregiver_confirmation=request_ctx.explicit_caregiver_confirmation,
            required_confirmation=required_confirmation,
        ):
            return self._deny(
                "This integration request needs more explicit confirmation.",
                required_confirmation=required_confirmation,
                integration_id=manifest_ctx.integration_id,
                operation_id=request_ctx.operation_id,
                origin=request_ctx.origin,
                domain=manifest_ctx.domain,
                action=operation_ctx.action,
                risk=operation_ctx.risk,
                confirmation=required_confirmation,
                background_trigger=request_ctx.background_trigger,
                dry_run=request_ctx.dry_run,
                payload_size_bytes=request_ctx.payload_size_bytes,
            )

        warnings: tuple[str, ...] = ()
        if request_ctx.dry_run:
            warnings = ("Dry run only. No external action will be executed.",)

        return self._allow(
            "The integration request passed policy checks.",
            warnings=warnings,
            integration_id=manifest_ctx.integration_id,
            operation_id=request_ctx.operation_id,
            origin=request_ctx.origin,
            domain=manifest_ctx.domain,
            action=operation_ctx.action,
            risk=operation_ctx.risk,
            confirmation=required_confirmation,
            background_trigger=request_ctx.background_trigger,
            dry_run=request_ctx.dry_run,
            payload_size_bytes=request_ctx.payload_size_bytes,
        )

    def _normalize_request(self, request: IntegrationRequest) -> _NormalizedRequest:
        """Validate and normalize request inputs used by policy evaluation."""

        operation_id = self._normalize_runtime_identifier(
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
        payload_size_bytes = _require_non_negative_int(
            request.payload_size_bytes(),
            field_name="request.payload_size_bytes()",
        )

        return _NormalizedRequest(
            operation_id=operation_id,
            origin=origin,
            background_trigger=background_trigger,
            dry_run=dry_run,
            explicit_user_confirmation=explicit_user_confirmation,
            explicit_caregiver_confirmation=explicit_caregiver_confirmation,
            payload_size_bytes=payload_size_bytes,
        )

    def _normalize_manifest(self, manifest: IntegrationManifest) -> _NormalizedManifest:
        """Validate and normalize manifest metadata used by policy evaluation."""

        integration_id = self._normalize_runtime_identifier(
            getattr(manifest, "integration_id"),
            field_name="manifest.integration_id",
        )
        domain = _require_enum(
            getattr(manifest, "domain"),
            enum_type=IntegrationDomain,
            field_name="manifest.domain",
        )
        return _NormalizedManifest(integration_id=integration_id, domain=domain)

    def _normalize_operation(self, operation: object) -> _NormalizedOperationSafety:
        """Validate and normalize operation safety metadata."""

        safety = getattr(operation, "safety")
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
        allow_background_polling = _require_bool(
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

        return _NormalizedOperationSafety(
            action=action,
            confirmation=confirmation,
            allow_remote_trigger=allow_remote_trigger,
            allow_background_polling=allow_background_polling,
            max_payload_bytes=max_payload_bytes,
            risk=risk,
        )

    def _normalize_runtime_identifier(
        self,
        value: object,
        *,
        field_name: str,
    ) -> _NormalizedIdentifier:
        """Build lookup and policy keys for one runtime identifier."""

        normalized_value = _normalize_identifier(
            value,
            field_name=field_name,
            max_length=self.max_identifier_length,
        )
        return _NormalizedIdentifier(
            value=normalized_value,
            policy_key=normalized_value.casefold(),
        )

    def _resolve_operation(self, manifest: IntegrationManifest, operation_id: str) -> object | None:
        """Resolve an operation object from the manifest."""

        operation_getter = getattr(manifest, "operation")
        if not callable(operation_getter):
            raise TypeError("manifest.operation must be callable.")
        return operation_getter(operation_id)

    def _is_operation_blocked(
        self,
        *,
        integration_id: _NormalizedIdentifier,
        operation_id: _NormalizedIdentifier,
    ) -> bool:
        """Return whether an operation is blocked by exact or wildcard policy rules."""

        if self._block_all_operations:
            return True

        scoped_operation = f"{integration_id.policy_key}:{operation_id.policy_key}"
        if operation_id.policy_key in self._blocked_operations_canonical:
            return True
        if scoped_operation in self._blocked_operations_canonical:
            return True
        if integration_id.policy_key in self._blocked_integration_operation_wildcards:
            return True
        return False

    def _baseline_confirmation(
        self,
        domain: IntegrationDomain,
        action: IntegrationAction,
    ) -> ConfirmationMode:
        """Return the baseline confirmation required for a domain/action pair."""

        return _baseline_confirmation_for(domain, action)

    def _risk_confirmation_floor(self, risk: RiskLevel) -> ConfirmationMode:
        """Return the confirmation floor implied by the operation risk."""

        return _risk_confirmation_floor(risk)

    def _confirmation_source_is_trustworthy(
        self,
        *,
        origin: RequestOrigin,
        background_trigger: bool,
    ) -> bool:
        """Return whether confirmation flags can be trusted for this request."""

        # BREAKING: Confirmation booleans from REMOTE_SERVICE or background-triggered
        # requests are rejected by default because they are self-asserted unless an
        # upstream attestation layer is explicitly trusted.
        if background_trigger and not self.allow_background_confirmations:
            return False
        if origin is RequestOrigin.REMOTE_SERVICE and not self.allow_remote_confirmations:
            return False
        return True

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
        return False

    def _allow(
        self,
        reason: str,
        *,
        warnings: tuple[str, ...] = (),
        integration_id: _NormalizedIdentifier | None = None,
        operation_id: _NormalizedIdentifier | None = None,
        origin: RequestOrigin | None = None,
        domain: IntegrationDomain | None = None,
        action: IntegrationAction | None = None,
        risk: RiskLevel | None = None,
        confirmation: ConfirmationMode | None = None,
        background_trigger: bool | None = None,
        dry_run: bool | None = None,
        payload_size_bytes: int | None = None,
    ) -> IntegrationDecision:
        """Emit an allow decision with structured audit logging."""

        self._audit_decision(
            allowed=True,
            reason=reason,
            integration_id=integration_id,
            operation_id=operation_id,
            origin=origin,
            domain=domain,
            action=action,
            risk=risk,
            confirmation=confirmation,
            background_trigger=background_trigger,
            dry_run=dry_run,
            payload_size_bytes=payload_size_bytes,
            warnings=warnings,
        )
        return IntegrationDecision.allow(reason, warnings=warnings)

    def _deny(
        self,
        reason: str,
        *,
        required_confirmation: ConfirmationMode | None = None,
        integration_id: _NormalizedIdentifier | None = None,
        operation_id: _NormalizedIdentifier | None = None,
        origin: RequestOrigin | None = None,
        domain: IntegrationDomain | None = None,
        action: IntegrationAction | None = None,
        risk: RiskLevel | None = None,
        confirmation: ConfirmationMode | None = None,
        background_trigger: bool | None = None,
        dry_run: bool | None = None,
        payload_size_bytes: int | None = None,
    ) -> IntegrationDecision:
        """Emit a deny decision with structured audit logging."""

        self._audit_decision(
            allowed=False,
            reason=reason,
            integration_id=integration_id,
            operation_id=operation_id,
            origin=origin,
            domain=domain,
            action=action,
            risk=risk,
            confirmation=confirmation or required_confirmation,
            background_trigger=background_trigger,
            dry_run=dry_run,
            payload_size_bytes=payload_size_bytes,
            warnings=(),
        )

        if required_confirmation is None:
            return IntegrationDecision.deny(reason)
        return IntegrationDecision.deny(
            reason,
            required_confirmation=required_confirmation,
        )

    def _audit_decision(
        self,
        *,
        allowed: bool,
        reason: str,
        integration_id: _NormalizedIdentifier | None,
        operation_id: _NormalizedIdentifier | None,
        origin: RequestOrigin | None,
        domain: IntegrationDomain | None,
        action: IntegrationAction | None,
        risk: RiskLevel | None,
        confirmation: ConfirmationMode | None,
        background_trigger: bool | None,
        dry_run: bool | None,
        payload_size_bytes: int | None,
        warnings: tuple[str, ...],
    ) -> None:
        """Write one structured audit event for the evaluated decision."""

        level = logging.INFO if allowed else logging.WARNING
        if not logger.isEnabledFor(level):
            return

        event = {
            "policy_fingerprint": self.policy_fingerprint,
            "allowed": allowed,
            "reason": reason,
            "integration_id": None if integration_id is None else integration_id.value,
            "operation_id": None if operation_id is None else operation_id.value,
            "origin": _enum_name(origin),
            "domain": _enum_name(domain),
            "action": _enum_name(action),
            "risk": _enum_name(risk),
            "required_confirmation": _enum_name(confirmation),
            "background_trigger": background_trigger,
            "dry_run": dry_run,
            "payload_size_bytes": payload_size_bytes,
            "warnings": warnings,
        }
        logger.log(
            level,
            "SafeIntegrationPolicy decision %s",
            json.dumps(event, sort_keys=True, separators=(",", ":")),
        )

    def _build_policy_fingerprint(self) -> str:
        """Build a stable fingerprint for the normalized policy state."""

        payload = {
            "enabled_integrations": sorted(self._enabled_integrations_canonical),
            "blocked_integrations": sorted(self._blocked_integrations_canonical),
            "blocked_operations": sorted(self._blocked_operations_canonical),
            "allow_remote_requests": self.allow_remote_requests,
            "allow_background_polling": self.allow_background_polling,
            "allow_critical_operations": self.allow_critical_operations,
            "allow_remote_confirmations": self.allow_remote_confirmations,
            "allow_background_confirmations": self.allow_background_confirmations,
            "max_identifier_length": self.max_identifier_length,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]