"""Handle temporary voice-quiet tool calls for live Twinr sessions.

This tool gives the runtime one bounded, non-persistent way to suppress the
transcript-first wake path and automatic follow-up reopening for a short time,
for example while TV or radio audio is playing in the room. It does not touch
`.env` settings and it does not remove the manual/button path.
"""

from __future__ import annotations

from math import ceil
from typing import Any

from .handler_telemetry import emit_best_effort, record_event_best_effort

_MAX_DURATION_MINUTES = 720


def _optional_text(value: object | None) -> str | None:
    text = str(value or "").strip()
    return text or None


def _required_action(arguments: dict[str, object]) -> str:
    action = str(arguments.get("action", "")).strip().lower()
    if action in {"set", "clear", "status"}:
        return action
    raise RuntimeError("manage_voice_quiet_mode requires `action` set to set, clear, or status")


def _required_duration_minutes(arguments: dict[str, object]) -> int:
    raw_value = arguments.get("duration_minutes")
    if raw_value is None or raw_value == "":
        raise RuntimeError("manage_voice_quiet_mode action=set requires `duration_minutes`")
    if isinstance(raw_value, bool):
        raise RuntimeError("duration_minutes must be an integer")
    if isinstance(raw_value, int):
        minutes = raw_value
    elif isinstance(raw_value, float):
        if not raw_value.is_integer():
            raise RuntimeError("duration_minutes must be an integer")
        minutes = int(raw_value)
    elif isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped:
            raise RuntimeError("duration_minutes must be an integer")
        try:
            minutes = int(stripped)
        except ValueError as exc:
            raise RuntimeError("duration_minutes must be an integer") from exc
    else:
        raise RuntimeError("duration_minutes must be an integer")
    if minutes <= 0:
        raise RuntimeError("duration_minutes must be greater than zero")
    return min(minutes, _MAX_DURATION_MINUTES)


def _remaining_minutes(remaining_seconds: int) -> int:
    if remaining_seconds <= 0:
        return 0
    return max(1, int(ceil(remaining_seconds / 60.0)))


def _minutes_phrase(minutes: int) -> str:
    return f"{minutes} Minute" if minutes == 1 else f"{minutes} Minuten"


def _spoken_text(*, action: str, active: bool, remaining_minutes: int) -> str:
    if action == "set":
        minutes_text = _minutes_phrase(max(1, remaining_minutes))
        return f"Okay. Ich bin jetzt für {minutes_text} ruhig."
    if action == "clear":
        return "Okay. Ich höre wieder normal zu."
    if active:
        if remaining_minutes > 0:
            return f"Ja. Ich bin ruhig. Noch etwa {_minutes_phrase(remaining_minutes)}."
        return "Ja. Ich bin ruhig."
    return "Nein. Ich höre gerade normal zu."


def _build_response(
    *,
    action: str,
    status: str,
    state: Any,
) -> dict[str, object]:
    active = bool(getattr(state, "active", False))
    remaining_seconds = max(0, int(getattr(state, "remaining_seconds", 0) or 0))
    remaining_minutes = _remaining_minutes(remaining_seconds)
    until_utc = _optional_text(getattr(state, "until_utc", None))
    reason = _optional_text(getattr(state, "reason", None))
    summary = (
        f"Twinr stays quiet until {until_utc}."
        if active and until_utc
        else "Twinr quiet mode is off."
    )
    if active and remaining_minutes > 0:
        summary = f"{summary[:-1]} About {remaining_minutes} minute(s) remaining."
    return {
        "status": status,
        "action": action,
        "active": active,
        "until_utc": until_utc,
        "reason": reason,
        "remaining_seconds": remaining_seconds,
        "remaining_minutes": remaining_minutes,
        "summary": summary,
        "spoken_text": _spoken_text(
            action=action,
            active=active,
            remaining_minutes=remaining_minutes,
        ),
        "suppresses_transcript_first_wake": active,
        "manual_button_path_available": True,
    }


def _refresh_voice_quiet_context(owner: Any) -> None:
    refresh = getattr(owner, "_refresh_voice_orchestrator_sensor_context", None)
    if not callable(refresh):
        return
    try:
        refresh()
    except Exception as exc:
        record_event_best_effort(
            owner,
            "voice_quiet_refresh_failed",
            "Twinr could not replay voice quiet mode into the live voice orchestrator state.",
            {"error_type": type(exc).__name__},
        )


def handle_manage_voice_quiet_mode(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Set, clear, or inspect Twinr's temporary voice-quiet window."""

    if not isinstance(arguments, dict):
        raise RuntimeError("manage_voice_quiet_mode arguments must be an object")
    runtime = getattr(owner, "runtime", None)
    if runtime is None:
        raise RuntimeError("manage_voice_quiet_mode requires an active runtime")

    action = _required_action(arguments)
    reason = _optional_text(arguments.get("reason"))

    if action == "set":
        state = runtime.set_voice_quiet_minutes(
            minutes=_required_duration_minutes(arguments),
            reason=reason,
        )
        _refresh_voice_quiet_context(owner)
        response = _build_response(action=action, status="active", state=state)
    elif action == "clear":
        state = runtime.clear_voice_quiet()
        _refresh_voice_quiet_context(owner)
        response = _build_response(action=action, status="cleared", state=state)
    else:
        voice_quiet_state = getattr(runtime, "voice_quiet_state", None)
        if not callable(voice_quiet_state):
            raise RuntimeError("manage_voice_quiet_mode requires runtime voice quiet support")
        state = voice_quiet_state()
        response = _build_response(
            action=action,
            status="active" if bool(getattr(state, "active", False)) else "inactive",
            state=state,
        )

    emit_best_effort(owner, f"voice_quiet_tool_call={action}")
    emit_best_effort(owner, f"voice_quiet_active={str(bool(response['active'])).lower()}")
    remaining_minutes = response.get("remaining_minutes")
    if isinstance(remaining_minutes, int) and remaining_minutes > 0:
        emit_best_effort(owner, f"voice_quiet_remaining_minutes={remaining_minutes}")
    record_event_best_effort(
        owner,
        "voice_quiet_tool_call",
        "Twinr updated or inspected the temporary voice quiet window.",
        {
            "action": action,
            "status": response["status"],
            "active": response["active"],
            "until_utc": response.get("until_utc"),
            "reason": response.get("reason"),
        },
    )
    return response
