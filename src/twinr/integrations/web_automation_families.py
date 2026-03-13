from __future__ import annotations

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


@dataclass(frozen=True, slots=True)
class IntegrationAutomationFamilyBlock:
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
    def block(self) -> IntegrationAutomationFamilyBlock:
        ...

    def handles_action(self, action: str) -> bool:
        ...

    def handle_action(self, action: str, *, form: dict[str, str], automation_store: AutomationStore) -> bool:
        ...


@dataclass(frozen=True, slots=True)
class ManagedIntegrationAutomationProvider:
    project_root: Path
    record: ManagedIntegrationConfig

    def block(self) -> IntegrationAutomationFamilyBlock:
        integration_id = self.record.integration_id
        manifest = manifest_for_id(integration_id)
        configured = self.record.enabled
        if manifest is not None:
            title = f"{manifest.title} automations"
            summary = manifest.summary
        else:
            title = f"{integration_id.replace('_', ' ').title()} automations"
            summary = "Integration-backed automation family."

        status_key = "warn" if configured else "muted"
        status_label = "Integration configured" if configured else "Integration not configured"
        operator_note = (
            "This integration now owns its own automation family slot. No integration-specific forms are wired yet."
        )
        if configured:
            operator_note += " Future adapter work can add custom create/edit flows here without changing the main automations page layout."
        else:
            operator_note += " Configure the integration first if you want future adapter-specific automation builders to appear here."

        return IntegrationAutomationFamilyBlock(
            key=f"integration_{integration_id}",
            integration_id=integration_id,
            title=title,
            summary=summary,
            detail="Future integration-backed automations will be rendered inside this family block.",
            status_key=status_key,
            status_label=status_label,
            operator_note=operator_note,
            source_prefixes=(
                f"integration:{integration_id}",
                f"integration_{integration_id}",
            ),
        )

    def handles_action(self, action: str) -> bool:
        return action.startswith(f"integration_family:{self.record.integration_id}:")

    def handle_action(self, action: str, *, form: dict[str, str], automation_store: AutomationStore) -> bool:
        return False


def integration_automation_family_providers(
    project_root: str | Path,
) -> tuple[IntegrationAutomationFamilyProvider, ...]:
    project_path = Path(project_root).resolve()
    store = TwinrIntegrationStore.from_project_root(project_path)
    known_ids = set(_DEFAULT_MANAGED_INTEGRATION_IDS)
    known_ids.update(store.load_all().keys())
    providers = [
        ManagedIntegrationAutomationProvider(project_root=project_path, record=store.get(integration_id))
        for integration_id in sorted(known_ids)
    ]
    return tuple(providers)
