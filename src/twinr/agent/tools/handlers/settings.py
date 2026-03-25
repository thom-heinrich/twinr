"""Handle bounded setting updates for realtime Twinr sessions.

Applies validated simple-setting changes, persists them to ``.env``, and
reloads live config for the current device.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from twinr.agent.base_agent.settings.simple_settings import update_simple_setting, write_env_updates

from .handler_telemetry import emit_best_effort, record_event_best_effort
from .support import require_sensitive_voice_confirmation


def handle_update_simple_setting(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Update one bounded runtime setting from a tool payload.

    Args:
        owner: Tool executor owner exposing config, runtime, and telemetry
            hooks.
        arguments: Tool payload with required ``setting`` and ``action`` plus
            optional ``value`` and confirmation fields.

    Returns:
        JSON-safe payload with ``status``, normalized setting metadata, and the
        applied value fields returned by the simple-settings helper.

    Raises:
        SensitiveActionConfirmationRequired: If spoken confirmation is required
            before changing device settings.
        RuntimeError: If required fields are missing or the requested setting
            update is invalid.
    """
    require_sensitive_voice_confirmation(owner, arguments, action_label="change Twinr's device settings")
    setting = str(arguments.get("setting", "")).strip()
    action = str(arguments.get("action", "")).strip()
    raw_value = arguments.get("value")
    if not setting or not action:
        raise RuntimeError("update_simple_setting requires `setting` and `action`")

    value: float | int | str | None
    if raw_value is None or raw_value == "":
        value = None
    elif isinstance(raw_value, (int, float)):
        value = float(raw_value)
    elif isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped:
            value = None
        else:
            try:
                value = float(stripped)
            except ValueError:
                value = stripped
    else:
        normalized = str(raw_value).strip()
        value = normalized or None

    try:
        result = update_simple_setting(
            owner.config,
            setting=setting,
            action=action,
            value=value,
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    env_path = Path(owner.config.project_root) / ".env"
    write_env_updates(env_path, result.env_updates)
    owner._reload_live_config_from_env(env_path)

    owner.runtime.remember_note(
        kind="preference",
        content=f"Simple setting update ({result.setting}): {result.summary}",
        source="update_simple_setting",
        metadata={key: str(value) for key, value in result.data.items()},
    )
    emit_best_effort(owner, "simple_setting_tool_call=true")
    emit_best_effort(owner, f"simple_setting={result.setting}")
    emit_best_effort(owner, f"simple_setting_summary={result.summary}")
    record_event_best_effort(
        owner,
        "simple_setting_updated",
        "Twinr updated a bounded runtime setting from an explicit user request.",
        {
            "setting": result.setting,
            "changed": result.changed,
            "summary": result.summary,
        },
    )
    response: dict[str, object] = {
        "status": "updated" if result.changed else "unchanged",
        "setting": result.setting,
        "summary": result.summary,
        "changed": result.changed,
    }
    for key, data_value in result.data.items():
        response[key] = data_value
    return response
