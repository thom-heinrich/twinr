from __future__ import annotations

from typing import Any


def require_current_turn_audio(owner: Any) -> bytes:
    if owner._current_turn_audio_pcm:
        return owner._current_turn_audio_pcm
    raise RuntimeError("Voice profile enrollment needs the current live spoken turn.")


def require_sensitive_voice_confirmation(
    owner: Any,
    arguments: dict[str, object],
    *,
    action_label: str,
) -> None:
    status = (owner.runtime.user_voice_status or "").strip().lower()
    if status not in {"uncertain", "unknown_voice"}:
        return
    if optional_bool(arguments, "confirmed", default=False):
        return
    raise RuntimeError(
        f"The current speaker signal is {status.replace('_', ' ')}. "
        f"Please ask for clear confirmation before you {action_label}, then call the tool again with confirmed=true."
    )


def optional_bool(
    arguments: dict[str, object],
    key: str,
    *,
    default: bool | None,
) -> bool | None:
    if key not in arguments:
        return default
    value = arguments.get(key)
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return bool(value)


def optional_float(arguments: dict[str, object], key: str, *, default: float) -> float:
    if key not in arguments:
        return default
    raw_value = arguments.get(key)
    if raw_value in {None, ""}:
        return default
    try:
        return float(raw_value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{key} must be a number") from exc
