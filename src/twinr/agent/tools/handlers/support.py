"""Provide shared guard helpers for realtime tool handlers.

These helpers normalize primitive arguments, gate sensitive mutations on voice
confirmation, and validate access to current-turn audio.
"""

from __future__ import annotations

from math import isfinite
from typing import Any


# AUDIT-FIX(#5): Use RuntimeError-compatible domain exceptions so callers can
# choose the right recovery prompt without losing backward compatibility.
class VoiceGuardError(RuntimeError):
    """Base error for voice guard helper failures."""


class MissingLiveAudioError(VoiceGuardError):
    """Raised when a live spoken turn is required but no PCM audio is available."""


class SensitiveActionConfirmationRequired(VoiceGuardError):
    """Raised when speaker identity is not strong enough for a sensitive action."""


class ArgumentValidationError(VoiceGuardError):
    """Raised when tool arguments cannot be parsed safely."""


# AUDIT-FIX(#1): Parse booleans strictly so ambiguous truthy strings or numbers
# cannot silently bypass sensitive confirmation gates.
_TRUTHY_STRINGS = frozenset({"true", "1", "yes", "y", "on"})
_FALSY_STRINGS = frozenset({"false", "0", "no", "n", "off"})

# AUDIT-FIX(#3): Normalize risky voice-status spellings to stable tokens and
# default missing/malformed state to unknown_voice, which is the safer path.
_SENSITIVE_CONFIRMATION_STATUSES = frozenset({"uncertain", "unknown_voice"})


def _normalized_voice_status(owner: Any) -> str:
    runtime = getattr(owner, "runtime", None)
    raw_status = getattr(runtime, "user_voice_status", None)
    if not isinstance(raw_status, str):
        return "unknown_voice"
    normalized = "_".join(raw_status.strip().lower().replace("-", "_").split())
    return normalized or "unknown_voice"


def _requires_sensitive_confirmation(status: str) -> bool:
    return (
        status in _SENSITIVE_CONFIRMATION_STATUSES
        or status.startswith("uncertain")
        or status.startswith("unknown")
    )


def require_current_turn_audio(owner: Any) -> bytes:
    """Return validated PCM bytes from the current live spoken turn.

    Args:
        owner: Tool executor owner that may hold ``_current_turn_audio_pcm``.

    Returns:
        Non-empty PCM16 audio bytes for the current turn.

    Raises:
        MissingLiveAudioError: If no usable current-turn audio is available.
    """
    # AUDIT-FIX(#2): Guard attribute access and only return validated bytes-like
    # PCM data so enrollment does not crash on missing or malformed state.
    audio = getattr(owner, "_current_turn_audio_pcm", None)
    if isinstance(audio, memoryview):
        audio = audio.tobytes()
    elif isinstance(audio, bytearray):
        audio = bytes(audio)

    if not isinstance(audio, bytes) or not audio:
        raise MissingLiveAudioError(
            "Voice profile enrollment needs the current live spoken turn."
        )
    return audio


def require_sensitive_voice_confirmation(
    owner: Any,
    arguments: dict[str, object],
    *,
    action_label: str,
) -> None:
    """Require explicit confirmation for sensitive tool actions.

    Args:
        owner: Tool executor owner exposing the current runtime voice status.
        arguments: Tool payload that may already contain ``confirmed=true``.
        action_label: Human-readable description of the pending sensitive
            action.

    Raises:
        ArgumentValidationError: If the ``confirmed`` field is present but not
            boolean-like.
        SensitiveActionConfirmationRequired: If the current speaker identity is
            unknown or uncertain and no confirmation is present.
    """
    # AUDIT-FIX(#3): Read voice state fail-safe and require confirmation whenever
    # the current speaker identity is missing, unknown, or uncertain.
    status = _normalized_voice_status(owner)
    if not _requires_sensitive_confirmation(status):
        return
    if optional_bool(arguments, "confirmed", default=False) is True:
        return

    safe_action_label = str(action_label).strip()
    if not safe_action_label or safe_action_label.lower() == "none":
        safe_action_label = "continue"

    raise SensitiveActionConfirmationRequired(
        f"The current speaker identity is {status.replace('_', ' ')}. "
        f"Please ask for clear confirmation before you {safe_action_label}, "
        "then call the tool again with confirmed=true."
    )


def optional_bool(
    arguments: dict[str, object],
    key: str,
    *,
    default: bool | None,
) -> bool | None:
    """Parse an optional boolean from a tool payload.

    Args:
        arguments: Tool payload to inspect.
        key: Field name to parse.
        default: Value to return when the field is absent, blank, or null.

    Returns:
        The parsed boolean value, or ``default`` when the field is effectively
        unset.

    Raises:
        ArgumentValidationError: If the payload is not an object or the field
            is not boolean-like.
    """
    # AUDIT-FIX(#1): Validate the payload shape and reject ambiguous boolean
    # inputs instead of falling back to Python truthiness.
    if not isinstance(arguments, dict):
        raise ArgumentValidationError("arguments must be an object")
    if key not in arguments:
        return default

    value = arguments.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return default
        if normalized in _TRUTHY_STRINGS:
            return True
        if normalized in _FALSY_STRINGS:
            return False
        raise ArgumentValidationError(f"{key} must be a boolean")
    if isinstance(value, (int, float)):
        if value in (0, 0.0):
            return False
        if value in (1, 1.0):
            return True
        raise ArgumentValidationError(f"{key} must be a boolean")
    raise ArgumentValidationError(f"{key} must be a boolean")


def optional_float(arguments: dict[str, object], key: str, *, default: float) -> float:
    """Parse an optional finite float from a tool payload.

    Args:
        arguments: Tool payload to inspect.
        key: Field name to parse.
        default: Value to return when the field is absent, blank, or null.

    Returns:
        The parsed finite float value, or ``default`` when the field is
        effectively unset.

    Raises:
        ArgumentValidationError: If the payload is not an object or the field
            is not a finite numeric value.
    """
    # AUDIT-FIX(#4): Avoid unhashable-membership crashes and reject bool/NaN/inf
    # so downstream logic only receives real finite numbers.
    if not isinstance(arguments, dict):
        raise ArgumentValidationError("arguments must be an object")
    if key not in arguments:
        return default

    raw_value = arguments.get(key)
    if raw_value is None:
        return default

    if isinstance(raw_value, bool):
        raise ArgumentValidationError(f"{key} must be a number")
    if isinstance(raw_value, (int, float)):
        value = float(raw_value)
    elif isinstance(raw_value, str):
        stripped = raw_value.strip()
        if stripped == "":
            return default
        try:
            value = float(stripped)
        except ValueError as exc:
            raise ArgumentValidationError(f"{key} must be a number") from exc
    else:
        raise ArgumentValidationError(f"{key} must be a number")

    if not isfinite(value):
        raise ArgumentValidationError(f"{key} must be a finite number")

    return value
