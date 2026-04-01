# CHANGELOG: 2026-03-30
# BUG-1: Replaced lossy slug-only routing with stable hashed namespaces so distinct integration IDs can no
#        longer collide in block keys, action prefixes, or source prefixes.
# BUG-2: Removed the mixed-snapshot load_all()->get() fallback path; providers are now built from a single
#        consistent integration-store snapshot per request.
# BUG-3: Recover integration IDs from record payloads when store keys are stale or partially corrupted, so
#        manifests and family blocks bind to the intended integration.
# SEC-1: Rejected control-character / whitespace-corrupted IDs and normalized UI text to block log/layout
#        injection from tampered on-disk config or manifest data.
# SEC-2: Disabled ambiguous legacy aliases whenever they would overlap or collide; all providers now expose
#        collision-proof hashed namespaces for future routing.
# IMP-1: Added cached manifest snapshots and structured diagnostics to reduce repeated I/O and exception churn
#        on Raspberry Pi 4 class hardware.
# IMP-2: Added a namespace-planning layer that preserves safe legacy compatibility where possible while making
#        migration state explicit in operator notes.
# BREAKING: block.key and the canonical action/source namespaces now use a stable "<slug>_<hash>" token.
# BREAKING: ambiguous legacy source aliases are intentionally withheld; old automations that relied on
#           overlapping legacy prefixes must migrate to the new hashed namespaces.

"""Expose dashboard automation-family blocks for managed integrations.

This module derives lightweight web-facing family blocks from integration
manifests and store state without taking ownership of full automation flows.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Protocol

try:  # Python 3.11+
    from enum import StrEnum
except ImportError:  # pragma: no cover - compatibility fallback
    from enum import Enum

    class StrEnum(str, Enum):
        """Fallback StrEnum for older runtimes."""


try:  # Python 3.9+
    from functools import cache
except ImportError:  # pragma: no cover - compatibility fallback
    from functools import lru_cache

    def cache(user_function):
        return lru_cache(maxsize=None)(user_function)


from twinr.automations import AutomationStore
from twinr.integrations.catalog import manifest_for_id
from twinr.integrations.runtime import CALENDAR_AGENDA_INTEGRATION_ID, EMAIL_MAILBOX_INTEGRATION_ID
from twinr.integrations.store import ManagedIntegrationConfig, TwinrIntegrationStore

_DEFAULT_MANAGED_INTEGRATION_IDS = (
    EMAIL_MAILBOX_INTEGRATION_ID,
    CALENDAR_AGENDA_INTEGRATION_ID,
)

_LOGGER = logging.getLogger(__name__)
_SAFE_NAMESPACE_COMPONENT_RE = re.compile(r"[^a-z0-9_]+")
_LEGACY_NAMESPACE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_WHITESPACE_RE = re.compile(r"\s+")
_UNKNOWN_INTEGRATION_ID = "unknown_integration"
_DEFAULT_SUMMARY = "Integration-backed automation family."
_DEFAULT_DETAIL = "Future integration-backed automations will be rendered inside this family block."
_NAMESPACE_HASH_BYTES = 6
_NAMESPACE_PERSON = b"twfamv1"
_MAX_TITLE_LENGTH = 96
_MAX_SUMMARY_LENGTH = 240
_MAX_DETAIL_LENGTH = 320


class BlockStatusKey(StrEnum):
    """Known UI status values for family blocks."""

    MUTED = "muted"
    WARN = "warn"


@dataclass(frozen=True, slots=True)
class IntegrationNamespacePlan:
    """Describe the routing namespace for one integration family."""

    integration_id: str
    slug: str
    token: str
    action_prefixes: tuple[str, ...]
    source_prefixes: tuple[str, ...]
    warning_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IntegrationManifestSnapshot:
    """Store the UI-safe manifest metadata for one integration."""

    title: str
    summary: str
    load_error: str | None = None


@dataclass(frozen=True, slots=True)
class IntegrationAutomationFamilyBlock:
    """Describe one integration-backed automation family block for the web UI."""

    key: str
    integration_id: str
    title: str
    summary: str
    detail: str
    status_key: str
    status_label: str
    operator_note: str
    source_prefixes: tuple[str, ...]


class IntegrationAutomationFamilyProvider(Protocol):
    """Describe the provider interface used by the web automations page."""

    def block(self) -> IntegrationAutomationFamilyBlock:
        """Return the static metadata block for one integration family."""

        ...

    def handles_action(self, action: str) -> bool:
        """Return whether this provider owns one incoming action key."""

        ...

    def handle_action(self, action: str, *, form: dict[str, str], automation_store: AutomationStore) -> bool:
        """Handle one action and report whether the provider consumed it."""

        ...


def _truncate_text(value: str, *, max_length: int) -> str:
    """Clamp text to a deterministic UI length budget."""

    if len(value) <= max_length:
        return value
    if max_length <= 1:
        return "…"
    return f"{value[: max_length - 1].rstrip()}…"


def _normalize_ui_text(value: object, *, default: str, max_length: int) -> str:
    """Normalize manifest/store text for UI and log safety."""

    if isinstance(value, str):
        candidate = unicodedata.normalize("NFKC", value)
        candidate = "".join(character if character.isprintable() else " " for character in candidate)
        candidate = _WHITESPACE_RE.sub(" ", candidate).strip()
        if candidate:
            return _truncate_text(candidate, max_length=max_length)
    return default


def _is_usable_integration_id(candidate: str) -> bool:
    """Return whether one integration ID is printable and routing-safe enough to keep."""

    return bool(candidate) and candidate.isprintable() and not any(character.isspace() for character in candidate)


def _coerce_integration_id(value: object) -> str:
    """Coerce store-derived IDs to a stable fallback when malformed."""

    if isinstance(value, str):
        candidate = unicodedata.normalize("NFKC", value).strip()
        if _is_usable_integration_id(candidate):
            return candidate
    return _UNKNOWN_INTEGRATION_ID


def _canonical_namespace_component(integration_id: str) -> str:
    """Convert one integration ID into a safe canonical namespace slug."""

    normalized = unicodedata.normalize("NFKC", integration_id).casefold()
    slug = _SAFE_NAMESPACE_COMPONENT_RE.sub("_", normalized).strip("_")
    return slug or _UNKNOWN_INTEGRATION_ID


def _legacy_namespace_component(integration_id: str) -> str | None:
    """Return a raw historical alias when it is already safe to expose."""

    candidate = unicodedata.normalize("NFKC", integration_id).strip()
    if not _is_usable_integration_id(candidate):
        return None
    return candidate if _LEGACY_NAMESPACE_COMPONENT_RE.fullmatch(candidate) else None


def _stable_namespace_token(integration_id: str) -> str:
    """Return the collision-proof canonical namespace token for one integration."""

    slug = _canonical_namespace_component(integration_id)
    digest = blake2b(
        integration_id.encode("utf-8"),
        digest_size=_NAMESPACE_HASH_BYTES,
        person=_NAMESPACE_PERSON,
    ).hexdigest()
    return f"{slug}_{digest}"


def _humanize_integration_id(integration_id: str) -> str:
    """Convert an integration ID into a user-facing title fragment."""

    text = integration_id.replace("_", " ").replace("-", " ").strip()
    return text.title() if text else "Unknown Integration"


def _coerce_enabled(value: object) -> bool:
    """Treat only real booleans as configured state."""

    return value if isinstance(value, bool) else False


def _resolve_project_path(project_root: str | Path) -> Path | None:
    """Resolve one project root and degrade safely on invalid paths."""

    try:
        project_path = Path(project_root).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, TypeError, ValueError):
        _LOGGER.exception(
            "Failed to resolve integration automation project root",
            extra={"project_root": repr(project_root)},
        )
        return None

    try:
        if project_path.exists() and project_path.is_dir():
            return project_path
    except OSError:
        _LOGGER.exception(
            "Failed to inspect integration automation project root",
            extra={"project_root": str(project_path)},
        )
        return None

    _LOGGER.warning(
        "Integration automation project root is unavailable or not a directory",
        extra={"project_root": str(project_path)},
    )
    return None


def _prefix_has_overlap(token: str, universe: frozenset[str]) -> bool:
    """Return whether one token is a strict prefix of another token."""

    return any(other != token and other.startswith(token) for other in universe)


def _dedupe_preserving_order(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Return one tuple with duplicates removed while preserving order."""

    return tuple(dict.fromkeys(values))


def _effective_record_integration_id(raw_integration_id: object, record: object) -> str:
    """Recover one integration ID from the record payload when the store key is stale."""

    record_integration_id = _coerce_integration_id(getattr(record, "integration_id", None))
    raw_integration_id = _coerce_integration_id(raw_integration_id)

    if record_integration_id != _UNKNOWN_INTEGRATION_ID:
        if raw_integration_id != _UNKNOWN_INTEGRATION_ID and raw_integration_id != record_integration_id:
            _LOGGER.warning(
                "Integration store key does not match record.integration_id; using record value",
                extra={
                    "store_key_integration_id": raw_integration_id,
                    "record_integration_id": record_integration_id,
                },
            )
        return record_integration_id

    return raw_integration_id


@cache
def _manifest_snapshot(integration_id: str) -> IntegrationManifestSnapshot:
    """Load and sanitize one manifest snapshot for repeated block rendering."""

    try:
        manifest = manifest_for_id(integration_id)
    except Exception:
        _LOGGER.exception(
            "Failed to load integration manifest",
            extra={"integration_id": integration_id},
        )
        return IntegrationManifestSnapshot(
            title=_humanize_integration_id(integration_id),
            summary=_DEFAULT_SUMMARY,
            load_error="Integration manifest lookup failed; metadata is being served from safe defaults.",
        )

    return IntegrationManifestSnapshot(
        title=_normalize_ui_text(
            getattr(manifest, "title", None),
            default=_humanize_integration_id(integration_id),
            max_length=_MAX_TITLE_LENGTH,
        ),
        summary=_normalize_ui_text(
            getattr(manifest, "summary", None),
            default=_DEFAULT_SUMMARY,
            max_length=_MAX_SUMMARY_LENGTH,
        ),
        load_error=None,
    )


def clear_integration_automation_family_caches() -> None:
    """Clear cached manifest snapshots after integration package updates."""

    _manifest_snapshot.cache_clear()


def _build_namespace_plans(
    integration_ids: tuple[str, ...],
) -> dict[str, IntegrationNamespacePlan]:
    """Build routing plans for all known integration IDs."""

    unique_ids = tuple(dict.fromkeys(integration_ids))
    canonical_aliases = {integration_id: _canonical_namespace_component(integration_id) for integration_id in unique_ids}
    legacy_aliases = {integration_id: _legacy_namespace_component(integration_id) for integration_id in unique_ids}

    alias_universe = {
        alias
        for alias in canonical_aliases.values()
        if alias
    }
    alias_universe.update(
        alias
        for alias in legacy_aliases.values()
        if alias is not None
    )
    alias_universe = frozenset(alias_universe)
    alias_counts = Counter(
        alias
        for integration_id in unique_ids
        for alias in {canonical_aliases[integration_id], legacy_aliases[integration_id]}
        if alias is not None
    )

    plans: dict[str, IntegrationNamespacePlan] = {}
    for integration_id in unique_ids:
        slug = canonical_aliases[integration_id]
        token = _stable_namespace_token(integration_id)
        warnings: list[str] = []
        action_prefixes = [f"integration_family:{token}:"]
        source_prefixes = [
            f"integration:{token}:",
            f"integration:{token}",
            f"integration_{token}_",
            f"integration_{token}",
        ]

        canonical_collision = alias_counts[slug] > 1
        canonical_prefix_overlap = _prefix_has_overlap(slug, alias_universe)
        if canonical_collision:
            warnings.append(
                "Canonical legacy routing aliases were disabled because another integration resolves to the same namespace."
            )
        else:
            action_prefixes.append(f"integration_family:{slug}:")
            source_prefixes.extend((f"integration:{slug}:", f"integration_{slug}_"))
            if not canonical_prefix_overlap:
                source_prefixes.extend((f"integration:{slug}", f"integration_{slug}"))
            else:
                warnings.append(
                    "Exact canonical source matching was disabled because this namespace is a prefix of another integration."
                )

        legacy_alias = legacy_aliases[integration_id]
        if legacy_alias is not None and legacy_alias != slug:
            legacy_collision = alias_counts[legacy_alias] > 1
            legacy_prefix_overlap = _prefix_has_overlap(legacy_alias, alias_universe)
            if legacy_collision:
                warnings.append(
                    "Historical raw-ID aliases were disabled because they collide with another integration route."
                )
            else:
                action_prefixes.append(f"integration_family:{legacy_alias}:")
                source_prefixes.extend((f"integration:{legacy_alias}:", f"integration_{legacy_alias}_"))
                if not legacy_prefix_overlap:
                    source_prefixes.extend((f"integration:{legacy_alias}", f"integration_{legacy_alias}"))
                else:
                    warnings.append(
                        "Exact raw-ID source matching was disabled because the historical namespace prefixes another integration."
                    )

        plans[integration_id] = IntegrationNamespacePlan(
            integration_id=integration_id,
            slug=slug,
            token=token,
            action_prefixes=_dedupe_preserving_order(action_prefixes),
            source_prefixes=_dedupe_preserving_order(source_prefixes),
            warning_reasons=tuple(dict.fromkeys(warnings)),
        )

    return plans


@dataclass(frozen=True, slots=True)
class ManagedIntegrationAutomationProvider:
    """Build dashboard-family blocks for one managed integration record."""

    project_root: Path
    namespace: IntegrationNamespacePlan
    record: ManagedIntegrationConfig | None = None
    configured: bool | None = None
    load_error: str | None = None

    def _resolved_integration_id(self) -> str:
        """Return the normalized integration ID for this provider."""

        return self.namespace.integration_id

    def _resolved_configured(self) -> bool:
        """Return whether the integration is configured and enabled."""

        if self.configured is not None:
            return _coerce_enabled(self.configured)
        if self.record is None:
            return False
        return _coerce_enabled(getattr(self.record, "enabled", False))

    def _action_prefixes(self) -> tuple[str, ...]:
        """Return action prefixes routed to this family provider."""

        return self.namespace.action_prefixes

    def _source_prefixes(self) -> tuple[str, ...]:
        """Return automation-source prefixes tied to this integration."""

        return self.namespace.source_prefixes

    def block(self) -> IntegrationAutomationFamilyBlock:
        """Build the dashboard block for the managed integration."""

        integration_id = self._resolved_integration_id()
        configured = self._resolved_configured()
        manifest = _manifest_snapshot(integration_id)
        title = f"{manifest.title} automations"
        summary = manifest.summary

        operator_notes: list[str] = []
        detail = _DEFAULT_DETAIL
        status_key = BlockStatusKey.WARN if configured else BlockStatusKey.MUTED
        status_label = "Integration configured" if configured else "Integration not configured"

        if self.load_error:
            status_key = BlockStatusKey.WARN
            status_label = "Integration status unavailable"
            operator_notes.append(
                "Twinr could not read this integration configuration and is showing a safe placeholder instead."
            )
            operator_notes.append("Review the integration store on disk before enabling this automation family.")
            detail = "Configuration could not be loaded; the family is visible in degraded mode."
        elif manifest.load_error:
            status_key = BlockStatusKey.WARN
            status_label = "Integration metadata unavailable"
            operator_notes.append(
                "Twinr could not load the integration manifest for this family, so safe default metadata is being served."
            )
            operator_notes.append("Repair the integration package or clear the manifest cache after deploying a fix.")
            detail = manifest.load_error
        else:
            operator_notes.append(
                "This integration owns its automation family slot and can accept adapter-specific builders without changing the page layout."
            )
            if configured:
                operator_notes.append("The integration is configured and ready for integration-specific create/edit flows.")
            else:
                operator_notes.append("Configure the integration first if you want adapter-specific automation builders to appear here.")

        if self.namespace.warning_reasons:
            status_key = BlockStatusKey.WARN
            operator_notes.extend(self.namespace.warning_reasons)
            if not self.load_error and not manifest.load_error:
                detail = _truncate_text(
                    (
                        "Collision-proof routing is active. Canonical namespaces now use "
                        f"'{self.namespace.token}' and only unambiguous legacy aliases remain enabled."
                    ),
                    max_length=_MAX_DETAIL_LENGTH,
                )

        operator_note = _truncate_text(
            " ".join(operator_notes).strip(),
            max_length=_MAX_DETAIL_LENGTH,
        )

        return IntegrationAutomationFamilyBlock(
            key=f"integration_{self.namespace.token}",
            integration_id=integration_id,
            title=title,
            summary=summary,
            detail=detail,
            status_key=str(status_key),
            status_label=status_label,
            operator_note=operator_note,
            source_prefixes=self._source_prefixes(),
        )

    def handles_action(self, action: str) -> bool:
        """Return whether an action belongs to this integration family."""

        if not isinstance(action, str) or not action:
            return False
        return any(action.startswith(prefix) for prefix in self._action_prefixes())

    def handle_action(self, action: str, *, form: dict[str, str], automation_store: AutomationStore) -> bool:
        """Keep the explicit no-op action handling contract for now."""

        if not self.handles_action(action):
            return False

        _LOGGER.info(
            "Integration family action matched but no adapter-specific handler is registered yet",
            extra={
                "integration_id": self._resolved_integration_id(),
                "action": action,
            },
        )
        return False


def _build_safe_placeholder_providers(
    project_root: Path,
    *,
    reason: str,
) -> tuple[IntegrationAutomationFamilyProvider, ...]:
    """Return placeholder providers when store access is unavailable."""

    namespace_plans = _build_namespace_plans(tuple(sorted(_DEFAULT_MANAGED_INTEGRATION_IDS)))
    return tuple(
        ManagedIntegrationAutomationProvider(
            project_root=project_root,
            namespace=namespace_plans[integration_id],
            configured=False,
            load_error=reason,
        )
        for integration_id in sorted(_DEFAULT_MANAGED_INTEGRATION_IDS)
    )


def integration_automation_family_providers(
    project_root: str | Path,
) -> tuple[IntegrationAutomationFamilyProvider, ...]:
    """Return automation-family providers for all managed integrations."""

    project_path = _resolve_project_path(project_root)
    if project_path is None:
        return _build_safe_placeholder_providers(
            Path.cwd(),
            reason="Integration store path is invalid or unavailable.",
        )

    try:
        store = TwinrIntegrationStore.from_project_root(project_path)
        loaded_records = dict(store.load_all())
    except Exception:
        _LOGGER.exception(
            "Failed to open integration store",
            extra={"project_root": str(project_path)},
        )
        return _build_safe_placeholder_providers(
            project_path,
            reason="Integration store could not be opened.",
        )

    normalized_records: dict[str, ManagedIntegrationConfig] = {}
    for raw_integration_id, record in loaded_records.items():
        if record is None:
            continue

        integration_id = _effective_record_integration_id(raw_integration_id, record)
        if integration_id in normalized_records:
            _LOGGER.warning(
                "Skipping duplicate integration record after normalization",
                extra={"integration_id": integration_id},
            )
            continue

        normalized_records[integration_id] = record

    known_ids = {_coerce_integration_id(integration_id) for integration_id in _DEFAULT_MANAGED_INTEGRATION_IDS}
    known_ids.update(normalized_records.keys())
    ordered_ids = tuple(
        sorted(
            known_ids,
            key=lambda value: (_canonical_namespace_component(value), value),
        )
    )
    namespace_plans = _build_namespace_plans(ordered_ids)

    providers: list[IntegrationAutomationFamilyProvider] = []
    for integration_id in ordered_ids:
        record = normalized_records.get(integration_id)
        if record is not None:
            providers.append(
                ManagedIntegrationAutomationProvider(
                    project_root=project_path,
                    namespace=namespace_plans[integration_id],
                    record=record,
                )
            )
            continue

        if integration_id in _DEFAULT_MANAGED_INTEGRATION_IDS:
            providers.append(
                ManagedIntegrationAutomationProvider(
                    project_root=project_path,
                    namespace=namespace_plans[integration_id],
                    configured=False,
                    load_error="Default integration configuration is missing from the current store snapshot.",
                )
            )
            continue

        providers.append(
            ManagedIntegrationAutomationProvider(
                project_root=project_path,
                namespace=namespace_plans[integration_id],
                configured=False,
                load_error="Integration configuration is unavailable.",
            )
        )

    return tuple(providers)