"""Expose dashboard automation-family blocks for managed integrations.

This module derives lightweight web-facing family blocks from integration
manifests and store state without taking ownership of full automation flows.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

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
_UNKNOWN_INTEGRATION_ID = "unknown_integration"
_DEFAULT_SUMMARY = "Integration-backed automation family."
_DEFAULT_DETAIL = "Future integration-backed automations will be rendered inside this family block."


# AUDIT-FIX(#5): Coerce corrupted or blank persisted identifiers to a stable safe fallback instead of
# trusting arbitrary store data throughout the provider lifecycle.
def _coerce_integration_id(value: object) -> str:
    """Coerce store-derived IDs to a stable fallback when malformed."""

    if isinstance(value, str):
        candidate = value.strip()
        if candidate:
            return candidate
    return _UNKNOWN_INTEGRATION_ID


# AUDIT-FIX(#4): Canonicalize integration IDs before reusing them in keys and action/source namespaces
# so malformed persisted values cannot create ambiguous routing or unsafe identifiers.
def _canonical_namespace_component(integration_id: str) -> str:
    """Convert one integration ID into a safe namespace component."""

    normalized = _SAFE_NAMESPACE_COMPONENT_RE.sub("_", integration_id.strip().lower()).strip("_")
    return normalized or _UNKNOWN_INTEGRATION_ID


# AUDIT-FIX(#4): Only keep legacy namespace aliases for already-safe historical IDs; anything containing
# separators or control characters is rejected from routing identifiers.
def _legacy_namespace_component(integration_id: str) -> str | None:
    """Return a legacy-safe namespace alias when one is still valid."""

    candidate = integration_id.strip()
    if not candidate:
        return None
    return candidate if _LEGACY_NAMESPACE_COMPONENT_RE.fullmatch(candidate) else None


def _humanize_integration_id(integration_id: str) -> str:
    """Convert an integration ID into a user-facing title fragment."""

    text = integration_id.replace("_", " ").replace("-", " ").strip()
    return text.title() if text else "Unknown Integration"


# AUDIT-FIX(#5): Manifest/store text fields are treated defensively so malformed values do not bubble up
# into attribute errors or blank UI strings.
def _coerce_text(value: object, *, default: str) -> str:
    """Coerce a possibly malformed text value to a safe fallback."""

    if isinstance(value, str):
        candidate = value.strip()
        if candidate:
            return candidate
    return default


# AUDIT-FIX(#5): Treat non-boolean persisted "enabled" values as safe False to avoid accidentally exposing
# an integration as configured when the on-disk state is malformed.
def _coerce_enabled(value: object) -> bool:
    """Treat only real booleans as configured state."""

    return value if isinstance(value, bool) else False


# AUDIT-FIX(#1): Resolve and validate the project root before touching the file-backed integration store;
# invalid paths degrade to a safe placeholder instead of crashing this provider listing.
def _resolve_project_path(project_root: str | Path) -> Path | None:
    """Resolve one project root and degrade safely on invalid paths."""

    try:
        project_path = Path(project_root).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, TypeError, ValueError):
        _LOGGER.exception("Failed to resolve integration automation project root: %r", project_root)
        return None

    try:
        if project_path.exists() and project_path.is_dir():
            return project_path
    except OSError:
        _LOGGER.exception("Failed to inspect integration automation project root: %s", project_path)
        return None

    _LOGGER.warning("Integration automation project root is unavailable or not a directory: %s", project_path)
    return None


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


@dataclass(frozen=True, slots=True)
class ManagedIntegrationAutomationProvider:
    """Build dashboard-family blocks for one managed integration record."""

    project_root: Path
    record: ManagedIntegrationConfig | None = None
    integration_id: str | None = None
    configured: bool | None = None
    load_error: str | None = None

    def _resolved_integration_id(self) -> str:
        """Return the normalized integration ID for this provider."""

        if self.integration_id is not None:
            return _coerce_integration_id(self.integration_id)
        if self.record is not None:
            return _coerce_integration_id(getattr(self.record, "integration_id", _UNKNOWN_INTEGRATION_ID))
        return _UNKNOWN_INTEGRATION_ID

    def _resolved_configured(self) -> bool:
        """Return whether the integration is configured and enabled."""

        if self.configured is not None:
            return _coerce_enabled(self.configured)
        if self.record is None:
            return False
        return _coerce_enabled(getattr(self.record, "enabled", False))

    def _action_prefixes(self) -> tuple[str, ...]:
        """Return action prefixes routed to this family provider."""

        integration_id = self._resolved_integration_id()
        safe_component = _canonical_namespace_component(integration_id)
        prefixes = [f"integration_family:{safe_component}:"]
        legacy_component = _legacy_namespace_component(integration_id)
        if legacy_component is not None and legacy_component != safe_component:
            prefixes.append(f"integration_family:{legacy_component}:")
        return tuple(prefixes)

    def _source_prefixes(self) -> tuple[str, ...]:
        """Return automation-source prefixes tied to this integration."""

        integration_id = self._resolved_integration_id()
        safe_component = _canonical_namespace_component(integration_id)
        prefixes = [
            f"integration:{safe_component}",
            f"integration_{safe_component}",
        ]
        legacy_component = _legacy_namespace_component(integration_id)
        if legacy_component is not None and legacy_component != safe_component:
            prefixes.extend(
                (
                    f"integration:{legacy_component}",
                    f"integration_{legacy_component}",
                )
            )
        return tuple(dict.fromkeys(prefixes))

    def block(self) -> IntegrationAutomationFamilyBlock:
        """Build the dashboard block for the managed integration."""

        integration_id = self._resolved_integration_id()
        configured = self._resolved_configured()

        manifest = None
        manifest_error = False
        try:
            # AUDIT-FIX(#2): Manifest lookup failures are isolated per provider so one broken integration
            # cannot take down the entire automations page.
            manifest = manifest_for_id(integration_id)
        except Exception:
            _LOGGER.exception("Failed to load integration manifest for %s", integration_id)
            manifest_error = True

        fallback_title = f"{_humanize_integration_id(integration_id)} automations"
        if manifest is not None:
            # AUDIT-FIX(#5): Manifest text is coerced defensively; malformed values fall back to predictable
            # defaults instead of breaking rendering or showing blank strings.
            manifest_title = _coerce_text(
                getattr(manifest, "title", None),
                default=_humanize_integration_id(integration_id),
            )
            title = f"{manifest_title} automations"
            summary = _coerce_text(getattr(manifest, "summary", None), default=_DEFAULT_SUMMARY)
        else:
            title = fallback_title
            summary = _DEFAULT_SUMMARY

        if self.load_error:
            status_key = "warn"
            status_label = "Integration status unavailable"
            operator_note = (
                "Twinr could not read this integration configuration and is showing a safe placeholder instead. "
                "Review the integration store on disk before enabling this automation family."
            )
            detail = "Configuration could not be loaded; the family is visible in degraded mode."
        elif manifest_error:
            status_key = "warn"
            status_label = "Integration metadata unavailable"
            operator_note = (
                "Twinr could not load the integration manifest for this family. "
                "The slot stays visible so the automations page remains usable while you repair the integration package."
            )
            detail = "Integration manifest lookup failed; metadata is being served from safe defaults."
        else:
            status_key = "warn" if configured else "muted"
            status_label = "Integration configured" if configured else "Integration not configured"
            operator_note = (
                "This integration now owns its own automation family slot. No integration-specific forms are wired yet."
            )
            if configured:
                operator_note += " Future adapter work can add custom create/edit flows here without changing the main automations page layout."
            else:
                operator_note += " Configure the integration first if you want future adapter-specific automation builders to appear here."
            detail = _DEFAULT_DETAIL

        return IntegrationAutomationFamilyBlock(
            # AUDIT-FIX(#4): Use a canonical safe namespace component for block keys so persisted integration
            # IDs cannot inject separators or collide through raw action/source syntax.
            key=f"integration_{_canonical_namespace_component(integration_id)}",
            integration_id=integration_id,
            title=title,
            summary=summary,
            detail=detail,
            status_key=status_key,
            status_label=status_label,
            operator_note=operator_note,
            source_prefixes=self._source_prefixes(),
        )

    def handles_action(self, action: str) -> bool:
        """Return whether an action belongs to this integration family."""

        # AUDIT-FIX(#6): Reject malformed action values instead of assuming the caller always provides a str.
        if not isinstance(action, str) or not action:
            return False
        return any(action.startswith(prefix) for prefix in self._action_prefixes())

    def handle_action(self, action: str, *, form: dict[str, str], automation_store: AutomationStore) -> bool:
        """Keep the explicit no-op action handling contract for now."""

        # AUDIT-FIX(#6): Keep the current explicit no-op behavior, but only after safe action validation so
        # malformed endpoint input does not trigger attribute errors in the provider layer.
        if not self.handles_action(action):
            return False
        return False


# AUDIT-FIX(#2): When the store is unavailable, keep the default family slots visible in a warning state
# so operators can still reach the page and diagnose the backing-store failure.
def _build_safe_placeholder_providers(
    project_root: Path,
    *,
    reason: str,
) -> tuple[IntegrationAutomationFamilyProvider, ...]:
    """Return placeholder providers when store access is unavailable."""

    return tuple(
        ManagedIntegrationAutomationProvider(
            project_root=project_root,
            integration_id=integration_id,
            configured=False,
            load_error=reason,
        )
        for integration_id in sorted(_DEFAULT_MANAGED_INTEGRATION_IDS)
    )


def integration_automation_family_providers(
    project_root: str | Path,
) -> tuple[IntegrationAutomationFamilyProvider, ...]:
    """Return automation-family providers for all managed integrations."""

    # AUDIT-FIX(#1): Validate the project root up front instead of assuming the caller handed us a usable
    # directory for file-backed state.
    project_path = _resolve_project_path(project_root)
    if project_path is None:
        return _build_safe_placeholder_providers(
            Path.cwd(),
            reason="Integration store path is invalid or unavailable.",
        )

    try:
        store = TwinrIntegrationStore.from_project_root(project_path)
        # AUDIT-FIX(#2): Load a single snapshot of the on-disk integration map and degrade cleanly if the
        # file-backed store is unreadable or partially written.
        loaded_records = dict(store.load_all())
    except Exception:
        _LOGGER.exception("Failed to open integration store for project root %s", project_path)
        return _build_safe_placeholder_providers(
            project_path,
            reason="Integration store could not be opened.",
        )

    normalized_records: dict[str, ManagedIntegrationConfig] = {}
    for raw_integration_id, record in loaded_records.items():
        normalized_integration_id = _coerce_integration_id(raw_integration_id)
        if normalized_integration_id not in normalized_records and record is not None:
            normalized_records[normalized_integration_id] = record

    known_ids = {_coerce_integration_id(integration_id) for integration_id in _DEFAULT_MANAGED_INTEGRATION_IDS}
    known_ids.update(normalized_records.keys())

    providers: list[IntegrationAutomationFamilyProvider] = []
    for integration_id in sorted(known_ids, key=lambda value: (_canonical_namespace_component(value), value)):
        record = normalized_records.get(integration_id)
        if record is not None:
            providers.append(
                ManagedIntegrationAutomationProvider(
                    project_root=project_path,
                    record=record,
                    integration_id=integration_id,
                )
            )
            continue

        if integration_id in _DEFAULT_MANAGED_INTEGRATION_IDS:
            try:
                # AUDIT-FIX(#3): Only do a targeted fallback read for missing default IDs, eliminating the
                # original per-record load_all() -> get() TOCTOU path for entries already present in the snapshot.
                fallback_record = store.get(integration_id)
            except Exception:
                _LOGGER.exception("Failed to load default integration config for %s", integration_id)
                providers.append(
                    ManagedIntegrationAutomationProvider(
                        project_root=project_path,
                        integration_id=integration_id,
                        configured=False,
                        load_error="Default integration configuration could not be loaded.",
                    )
                )
            else:
                providers.append(
                    ManagedIntegrationAutomationProvider(
                        project_root=project_path,
                        record=fallback_record,
                        integration_id=integration_id,
                    )
                )
            continue

        providers.append(
            ManagedIntegrationAutomationProvider(
                project_root=project_path,
                integration_id=integration_id,
                configured=False,
                load_error="Integration configuration is unavailable.",
            )
        )

    return tuple(providers)
