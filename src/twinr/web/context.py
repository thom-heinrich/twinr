"""Provide shared request-time context for Twinr's web control surface.

`WebAppContext` centralizes stable config loading, store construction, and
safe template rendering so route handlers in `app.py` stay thin.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from markupsafe import Markup

from twinr.agent.base_agent import RuntimeSnapshot, RuntimeSnapshotStore, TwinrConfig
from twinr.agent.self_coding.store import SelfCodingStore
from twinr.automations import AutomationStore
from twinr.integrations import TwinrIntegrationStore
from twinr.memory.context_store import ManagedContextFileStore, PersistentMemoryMarkdownStore
from twinr.memory.reminders import ReminderStore
from twinr.ops import TwinrOpsEventStore, TwinrUsageStore
from twinr.ops.remote_memory_watchdog import RemoteMemoryWatchdogSnapshot, RemoteMemoryWatchdogStore
from twinr.web.presenters import _nav_items
from twinr.web.support.auth import FileBackedWebAuthStore
from twinr.web.support.channel_onboarding import FileBackedChannelOnboardingStore

# AUDIT-FIX(#7): Use plain language in the default restart notice so user-facing pages do not expose internal jargon.
DEFAULT_RESTART_NOTICE = "Some settings only take effect after Twinr is restarted."

logger = logging.getLogger(__name__)

_MAX_STABLE_READ_ATTEMPTS: Final[int] = 3
_MAX_ERROR_MESSAGE_CHARS: Final[int] = 240
_STRICT_RESERVED_CONTEXT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "request",
        "page_title",
        "active_page",
        "nav_items",
        "restart_notice",
        "env_path",
    }
)
_RESERVED_CONTEXT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "request",
        "page_title",
        "active_page",
        "nav_items",
        "saved",
        "error_message",
        "restart_notice",
        "env_path",
    }
)


@dataclass(slots=True)
class WebAppContext:
    """Bundle rooted paths, store factories, and template rendering helpers.

    Attributes:
        env_path: Canonical path to Twinr's `.env` file.
        project_root: Canonical Twinr project root used for rooted file access.
        ops_paths: Resolved ops/support path bundle.
        templates: Jinja loader for package-local web templates.
    """

    env_path: Path
    project_root: Path
    ops_paths: Any
    templates: Jinja2Templates

    def __post_init__(self) -> None:
        """Normalize and validate rooted paths at construction time."""

        # AUDIT-FIX(#8): Normalize and validate path-like constructor inputs early so bad paths fail fast and symlink targets are pinned at startup.
        normalized_env_path = Path(self.env_path).expanduser().resolve(strict=False)
        normalized_project_root = Path(self.project_root).expanduser().resolve(strict=False)

        if normalized_env_path.exists() and normalized_env_path.is_dir():
            raise ValueError(f"env_path must point to a file, not a directory: {normalized_env_path}")
        if normalized_project_root.exists() and not normalized_project_root.is_dir():
            raise ValueError(f"project_root must point to a directory: {normalized_project_root}")

        object.__setattr__(self, "env_path", normalized_env_path)
        object.__setattr__(self, "project_root", normalized_project_root)

    def load_state(self) -> tuple[TwinrConfig, dict[str, str]]:
        """Load config plus raw env values from a stable `.env` snapshot.

        Returns:
            Tuple of parsed `TwinrConfig` and the raw env key/value mapping.

        Raises:
            RuntimeError: If the env file cannot be read consistently.
        """

        from twinr.web.support.store import read_env_values

        last_exception: Exception | None = None

        # AUDIT-FIX(#2): Retry until the .env file is observed as stable across both reads so config and raw env values cannot drift apart mid-request.
        for _attempt in range(_MAX_STABLE_READ_ATTEMPTS):
            try:
                before = self.env_path.stat()
                env_values = read_env_values(self.env_path)
                config = TwinrConfig.from_env(self.env_path)
                after = self.env_path.stat()
            except Exception as exc:
                last_exception = exc
                continue

            if self._stat_fingerprint(before) == self._stat_fingerprint(after):
                return config, env_values

        if last_exception is not None:
            logger.exception("Failed to load Twinr configuration from %s", self.env_path, exc_info=last_exception)
            raise RuntimeError("Twinr could not load its configuration file.") from last_exception

        logger.error("Twinr configuration changed while it was being read from %s", self.env_path)
        raise RuntimeError("Twinr settings were being changed while the page was loading. Please try again.")

    def load_snapshot(self, config: TwinrConfig) -> RuntimeSnapshot:
        """Load the last runtime snapshot with a safe failure contract."""

        try:
            return RuntimeSnapshotStore(config.runtime_state_path).load()
        except Exception as exc:
            # AUDIT-FIX(#3): Convert raw snapshot I/O and parse failures into a safe, plain-language error while keeping the original traceback in logs.
            logger.exception("Failed to load Twinr runtime snapshot from %s", config.runtime_state_path)
            raise RuntimeError("Twinr could not load its saved state.") from exc

    def memory_store(self, config: TwinrConfig) -> PersistentMemoryMarkdownStore:
        """Build the persistent memory markdown store for the current config."""

        return PersistentMemoryMarkdownStore(config.memory_markdown_path)

    def reminder_store(self, config: TwinrConfig) -> ReminderStore:
        """Build the reminder store for the current config."""

        return ReminderStore(
            config.reminder_store_path,
            timezone_name=config.local_timezone_name,
            retry_delay_s=config.reminder_retry_delay_s,
            max_entries=config.reminder_max_entries,
        )

    def automation_store(self, config: TwinrConfig) -> AutomationStore:
        """Build the automation store for the current config."""

        return AutomationStore(
            config.automation_store_path,
            timezone_name=config.local_timezone_name,
            max_entries=config.automation_max_entries,
        )

    def self_coding_store(self) -> SelfCodingStore:
        """Build the self-coding store rooted at the Twinr project."""

        return SelfCodingStore.from_project_root(self.project_root)

    def user_context_store(self, config: TwinrConfig) -> ManagedContextFileStore:
        """Build the managed user-context store for the current config."""

        return ManagedContextFileStore(
            # AUDIT-FIX(#1): Canonicalize the managed file path and reject any personality_dir that escapes the project root.
            self._managed_context_path(config.personality_dir, "USER.md"),
            section_title="Twinr managed user updates",
        )

    def personality_context_store(self, config: TwinrConfig) -> ManagedContextFileStore:
        """Build the managed personality-context store for the current config."""

        return ManagedContextFileStore(
            # AUDIT-FIX(#1): Canonicalize the managed file path and reject any personality_dir that escapes the project root.
            self._managed_context_path(config.personality_dir, "PERSONALITY.md"),
            section_title="Twinr managed personality updates",
        )

    def event_store(self) -> TwinrOpsEventStore:
        """Build the local ops-event store."""

        return TwinrOpsEventStore.from_project_root(self.project_root)

    def usage_store(self) -> TwinrUsageStore:
        """Build the local LLM-usage store."""

        return TwinrUsageStore.from_project_root(self.project_root)

    def load_remote_memory_watchdog(self, config: TwinrConfig) -> RemoteMemoryWatchdogSnapshot | None:
        """Load the last persisted remote-memory watchdog snapshot safely."""

        try:
            return RemoteMemoryWatchdogStore.from_config(config).load()
        except Exception as exc:
            logger.exception("Failed to load remote memory watchdog snapshot for %s", config.project_root)
            raise RuntimeError("Twinr could not load the remote memory watchdog state.") from exc

    def integration_store(self) -> TwinrIntegrationStore:
        """Build the local integration store."""

        return TwinrIntegrationStore.from_project_root(self.project_root)

    def web_auth_store(self) -> FileBackedWebAuthStore:
        """Build the managed web-auth store rooted under the Twinr project."""

        return FileBackedWebAuthStore.from_project_root(self.project_root)

    def channel_onboarding_store(self, channel_id: str) -> FileBackedChannelOnboardingStore:
        """Build one rooted channel-onboarding store."""

        return FileBackedChannelOnboardingStore.from_project_root(self.project_root, channel_id=channel_id)

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
        """Render one template with Twinr-controlled context keys.

        Args:
            request: FastAPI request for the current page render.
            template_name: Package-relative Jinja template name.
            page_title: Human-readable page title.
            active_page: Sidebar key for the active page highlight.
            restart_notice: Optional restart notice shown near the page header.
            **context: Additional template context values that may not override
                Twinr-reserved keys.

        Returns:
            Rendered `HTMLResponse`, or a minimal fallback page when template
            rendering fails.
        """

        # AUDIT-FIX(#5): Prevent caller-supplied context from overriding framework-controlled keys such as request/nav_items/env_path.
        ignored_reserved_keys = sorted(key for key in context if key in _STRICT_RESERVED_CONTEXT_KEYS)
        if ignored_reserved_keys:
            logger.warning(
                "Ignoring reserved template context keys for %s: %s",
                template_name,
                ", ".join(ignored_reserved_keys),
            )

        raw_saved = context.get("saved", request.query_params.get("saved"))
        saved = self._coerce_saved_flag(raw_saved)

        raw_error_message = context.get("error_message", request.query_params.get("error"))
        # AUDIT-FIX(#6): Normalize, length-limit, and HTML-escape the user-visible error string before it reaches the template layer.
        error_message = self._sanitize_error_message(raw_error_message)

        passthrough_context = {
            key: value
            for key, value in context.items()
            if key not in _RESERVED_CONTEXT_KEYS
        }

        try:
            response_context = {
                **passthrough_context,
                "request": request,
                "page_title": page_title,
                "active_page": active_page,
                "nav_items": _nav_items(),
                "saved": saved,
                "error_message": error_message,
                "restart_notice": restart_notice,
                "env_path": str(self.env_path),
                "auth_context": getattr(request.state, "web_auth_context", None),
            }
            return self.templates.TemplateResponse(request, template_name, response_context)
        except Exception:
            # AUDIT-FIX(#4): Degrade to a minimal safe HTML page instead of crashing the request path on template lookup/render failures.
            logger.exception("Failed to render Twinr template %s", template_name)
            return HTMLResponse(
                self._fallback_render_html(page_title),
                status_code=500,
            )

    @staticmethod
    def _stat_fingerprint(stat_result: os.stat_result) -> tuple[int, int, int, int, int]:
        """Reduce `stat()` output to the fields used for change detection."""

        return (
            stat_result.st_dev,
            stat_result.st_ino,
            stat_result.st_size,
            stat_result.st_mtime_ns,
            stat_result.st_ctime_ns,
        )

    def _managed_context_path(self, personality_dir: str | Path, filename: str) -> Path:
        """Resolve one managed context markdown file under `project_root`."""

        # AUDIT-FIX(#1): Resolve the final managed markdown path against the pinned project root and refuse any escaped location.
        managed_path = (self.project_root / Path(personality_dir) / filename).resolve(strict=False)
        if not managed_path.is_relative_to(self.project_root):
            raise ValueError(
                "personality_dir must stay inside project_root for managed context files."
            )

        managed_path.parent.mkdir(parents=True, exist_ok=True)
        return managed_path

    @staticmethod
    def _coerce_saved_flag(value: Any) -> bool:
        """Interpret the saved query flag as a boolean."""

        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).strip() == "1"

    @staticmethod
    def _sanitize_error_message(value: Any) -> Markup | None:
        """Normalize, bound, and escape one flash-style error message."""

        # AUDIT-FIX(#6): Strip control-style formatting, bound the message size, and escape HTML so query-string flash messages stay safe.
        if value is None:
            return None

        normalized = " ".join(str(value).split())
        if not normalized:
            return None

        if len(normalized) > _MAX_ERROR_MESSAGE_CHARS:
            normalized = normalized[: _MAX_ERROR_MESSAGE_CHARS - 1].rstrip() + "…"

        return Markup.escape(normalized)

    @staticmethod
    def _fallback_render_html(page_title: str) -> str:
        """Build a minimal HTML fallback page for template failures."""

        # AUDIT-FIX(#4): Return a plain, dependency-free fallback page when the normal template pipeline fails.
        safe_title = Markup.escape(page_title or "Twinr")
        return (
            "<!doctype html>"
            "<html lang=\"en\">"
            "<head>"
            "<meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>{safe_title}</title>"
            "</head>"
            "<body>"
            "<main>"
            "<h1>Page unavailable</h1>"
            "<p>This page could not be opened right now. Please try again. If the problem continues, restart Twinr.</p>"
            "</main>"
            "</body>"
            "</html>"
        )
