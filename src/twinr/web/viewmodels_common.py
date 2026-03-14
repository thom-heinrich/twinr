from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import HTTPException

from twinr.agent.base_agent import TwinrConfig
from twinr.integrations import EMAIL_APP_PASSWORD_ENV_KEY
from twinr.memory.reminders import format_due_label, now_in_timezone
from twinr.web.store import mask_secret

_PROVIDER_OPTIONS = (
    ("openai", "OpenAI"),
    ("deepinfra", "DeepInfra"),
    ("openrouter", "OpenRouter"),
)
_BOOL_OPTIONS = (("true", "Enabled"), ("false", "Disabled"))
_TRISTATE_BOOL_OPTIONS = (("", "Auto"), ("true", "Always send"), ("false", "Never send"))
_YES_NO_OPTIONS = (("true", "Yes"), ("false", "No"))
_REASONING_EFFORT_OPTIONS = (("low", "Low"), ("medium", "Medium"), ("high", "High"))
_SEARCH_CONTEXT_OPTIONS = (("low", "Low"), ("medium", "Medium"), ("high", "High"))
_CONVERSATION_WEB_SEARCH_OPTIONS = (("auto", "Auto"), ("always", "Always"), ("never", "Never"))
_VISION_DETAIL_OPTIONS = (("auto", "Auto"), ("low", "Low"), ("high", "High"))
_GPIO_BIAS_OPTIONS = (("pull-up", "Pull-up"), ("pull-down", "Pull-down"), ("disabled", "Off"))
_WAKEWORD_BACKEND_OPTIONS = (("stt", "STT"), ("openwakeword", "openWakeWord"))
_EMAIL_PROFILE_OPTIONS = (("gmail", "Gmail"), ("generic_imap_smtp", "Generic IMAP/SMTP"))
_CALENDAR_SOURCE_OPTIONS = (("ics_file", "ICS file"), ("ics_url", "ICS URL"))
_EMAIL_SECRET_KEY = EMAIL_APP_PASSWORD_ENV_KEY


def _nav_items() -> tuple[tuple[str, str, str], ...]:
    return (
        ("dashboard", "Dashboard", "/"),
        ("ops_self_test", "Self-Test", "/ops/self-test"),
        ("ops_devices", "Devices", "/ops/devices"),
        ("ops_usage", "LLM Usage", "/ops/usage"),
        ("ops_health", "System Health", "/ops/health"),
        ("ops_logs", "Ops Logs", "/ops/logs"),
        ("ops_config", "Config Checks", "/ops/config"),
        ("ops_support", "Support", "/ops/support"),
        ("integrations", "Integrations", "/integrations"),
        ("voice_profile", "Voice Profile", "/voice-profile"),
        ("automations", "Automations", "/automations"),
        ("personality", "Personality", "/personality"),
        ("memory", "Memory", "/memory"),
        ("connect", "Connect", "/connect"),
        ("settings", "Settings", "/settings"),
        ("user", "User", "/user"),
    )


def _provider_status(env_values: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return (
        ("OpenAI key", mask_secret(env_values.get("OPENAI_API_KEY"))),
        ("OpenAI project", env_values.get("OPENAI_PROJ_ID", "Not configured") or "Not configured"),
        ("DeepInfra key", mask_secret(env_values.get("DEEPINFRA_API_KEY"))),
        ("OpenRouter key", mask_secret(env_values.get("OPENROUTER_API_KEY"))),
    )


def _default_reminder_due_at(config: TwinrConfig) -> str:
    return now_in_timezone(config.local_timezone_name).replace(second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M")


def _reminder_rows(entries: tuple[Any, ...], *, timezone_name: str) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        status_key = "delivered" if entry.delivered else ("retry" if entry.last_error else "pending")
        if status_key == "delivered":
            status_label = "Delivered"
        elif status_key == "retry":
            status_label = "Retrying"
        else:
            status_label = "Pending"
        rows.append(
            {
                "reminder_id": entry.reminder_id,
                "kind": entry.kind,
                "summary": entry.summary,
                "details": entry.details,
                "source": entry.source,
                "original_request": entry.original_request,
                "delivery_attempts": entry.delivery_attempts,
                "last_error": entry.last_error,
                "created_at": entry.created_at.isoformat(),
                "updated_at": entry.updated_at.isoformat(),
                "due_label": format_due_label(entry.due_at, timezone_name=timezone_name),
                "next_attempt_label": (
                    format_due_label(entry.next_attempt_at, timezone_name=timezone_name)
                    if entry.next_attempt_at is not None
                    else None
                ),
                "delivered_at_label": (
                    format_due_label(entry.delivered_at, timezone_name=timezone_name)
                    if entry.delivered_at is not None
                    else None
                ),
                "delivered": entry.delivered,
                "status_key": status_key,
                "status_label": status_label,
            }
        )
    return tuple(rows)


def _credential_state_label(value: str | None) -> str:
    return "Configured" if value else "Not configured"


def _format_seconds_label(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".") + " s"


def _format_millis_label(value: int) -> str:
    return f"{int(value)} ms"


def _recent_named_files(directory: Path, *, suffix: str) -> tuple[dict[str, str], ...]:
    if not directory.exists():
        return ()
    files = sorted(
        [path for path in directory.iterdir() if path.is_file() and path.name.endswith(suffix)],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return tuple(
        {
            "name": path.name,
            "path": str(path),
            "download_href": (
                f"/ops/support/download/{path.name}" if suffix == ".zip" else f"/ops/self-test/artifacts/{path.name}"
            ),
        }
        for path in files[:8]
    )


def _resolve_named_file(root: Path, name: str) -> Path:
    safe_name = Path(name).name
    if safe_name != name:
        raise HTTPException(status_code=404, detail="File not found")
    candidate = (root / safe_name).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="File not found") from exc
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return candidate
