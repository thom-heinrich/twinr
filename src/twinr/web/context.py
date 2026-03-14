from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from twinr.agent.base_agent import RuntimeSnapshot, RuntimeSnapshotStore, TwinrConfig
from twinr.automations import AutomationStore
from twinr.integrations import TwinrIntegrationStore
from twinr.memory.context_store import ManagedContextFileStore, PersistentMemoryMarkdownStore
from twinr.memory.reminders import ReminderStore
from twinr.ops import TwinrOpsEventStore, TwinrUsageStore
from twinr.web.viewmodels import _nav_items

DEFAULT_RESTART_NOTICE = "Changes to providers, models, or network settings need a Twinr process restart."


@dataclass(slots=True)
class WebAppContext:
    env_path: Path
    project_root: Path
    ops_paths: Any
    templates: Jinja2Templates

    def load_state(self) -> tuple[TwinrConfig, dict[str, str]]:
        from twinr.web.store import read_env_values

        env_values = read_env_values(self.env_path)
        return TwinrConfig.from_env(self.env_path), env_values

    def load_snapshot(self, config: TwinrConfig) -> RuntimeSnapshot:
        return RuntimeSnapshotStore(config.runtime_state_path).load()

    def memory_store(self, config: TwinrConfig) -> PersistentMemoryMarkdownStore:
        return PersistentMemoryMarkdownStore(config.memory_markdown_path)

    def reminder_store(self, config: TwinrConfig) -> ReminderStore:
        return ReminderStore(
            config.reminder_store_path,
            timezone_name=config.local_timezone_name,
            retry_delay_s=config.reminder_retry_delay_s,
            max_entries=config.reminder_max_entries,
        )

    def automation_store(self, config: TwinrConfig) -> AutomationStore:
        return AutomationStore(
            config.automation_store_path,
            timezone_name=config.local_timezone_name,
            max_entries=config.automation_max_entries,
        )

    def user_context_store(self, config: TwinrConfig) -> ManagedContextFileStore:
        return ManagedContextFileStore(
            self.project_root / config.personality_dir / "USER.md",
            section_title="Twinr managed user updates",
        )

    def personality_context_store(self, config: TwinrConfig) -> ManagedContextFileStore:
        return ManagedContextFileStore(
            self.project_root / config.personality_dir / "PERSONALITY.md",
            section_title="Twinr managed personality updates",
        )

    def event_store(self) -> TwinrOpsEventStore:
        return TwinrOpsEventStore.from_project_root(self.project_root)

    def usage_store(self) -> TwinrUsageStore:
        return TwinrUsageStore.from_project_root(self.project_root)

    def integration_store(self) -> TwinrIntegrationStore:
        return TwinrIntegrationStore.from_project_root(self.project_root)

    def render(
        self,
        request: Request,
        template_name: str,
        *,
        page_title: str,
        active_page: str,
        restart_notice: str | None = DEFAULT_RESTART_NOTICE,
        **context: Any,
    ) -> HTMLResponse:
        return self.templates.TemplateResponse(
            request,
            template_name,
            {
                "request": request,
                "page_title": page_title,
                "active_page": active_page,
                "nav_items": _nav_items(),
                "saved": request.query_params.get("saved") == "1",
                "error_message": request.query_params.get("error"),
                "restart_notice": restart_notice,
                "env_path": str(self.env_path),
                **context,
            },
        )
