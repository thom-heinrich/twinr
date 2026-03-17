"""Shared runtime configuration for local self_coding Codex drivers."""

from __future__ import annotations

import logging  # AUDIT-FIX(#2): Emit operator-visible warnings instead of silently swallowing invalid configuration.
import math
import os

_DEFAULT_TIMEOUT_SECONDS = 900.0
_MAX_TIMEOUT_SECONDS = 3600.0  # AUDIT-FIX(#1): Hard-cap timeouts so bad env values cannot stall a worker for hours.
_DEFAULT_MODEL = "gpt-5-codex"  # AUDIT-FIX(#4): Centralize the safe fallback model used after invalid input.
_MAX_MODEL_NAME_LENGTH = 128  # AUDIT-FIX(#4): Reject obviously malformed model identifiers before downstream API calls.
_VALID_REASONING_EFFORTS = frozenset({"minimal", "low", "medium", "high", "xhigh"})

_LOGGER = logging.getLogger(__name__)  # AUDIT-FIX(#2): Surface misconfiguration details for recovery and remote support.


def _normalize_env_name(env_name: object) -> str | None:
    """Return a safe environment-variable name, or ``None`` if invalid."""

    if not isinstance(env_name, str):  # AUDIT-FIX(#3): Prevent TypeError from os.getenv on runtime misuse.
        _LOGGER.warning("Ignoring non-string environment variable name: %r", env_name)
        return None
    normalized = env_name.strip()
    if not normalized:
        _LOGGER.warning("Ignoring blank environment variable name.")
        return None
    return normalized


def _normalize_timeout(value: object, *, source: str) -> float | None:
    """Return a validated timeout in seconds, or ``None`` if invalid."""

    try:
        timeout_seconds = float(value)
    except (TypeError, ValueError):  # AUDIT-FIX(#1): Reject non-numeric timeout inputs instead of letting bad defaults propagate.
        _LOGGER.warning("Ignoring invalid timeout from %s; expected a numeric value in seconds.", source)
        return None
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        _LOGGER.warning("Ignoring non-finite or non-positive timeout from %s.", source)
        return None
    if timeout_seconds > _MAX_TIMEOUT_SECONDS:
        _LOGGER.warning(
            "Ignoring timeout from %s above hard cap %.1f seconds.",
            source,
            _MAX_TIMEOUT_SECONDS,
        )
        return None
    return timeout_seconds


def _normalize_model(value: object, *, source: str) -> str | None:
    """Return a validated model identifier, or ``None`` if invalid."""

    if value is None:
        return None
    if not isinstance(value, str):  # AUDIT-FIX(#4): Reject invalid runtime defaults instead of coercing arbitrary objects.
        _LOGGER.warning("Ignoring non-string model value from %s.", source)
        return None
    normalized = value.strip()
    if not normalized:
        _LOGGER.warning("Ignoring blank model value from %s.", source)
        return None
    if len(normalized) > _MAX_MODEL_NAME_LENGTH:
        _LOGGER.warning(
            "Ignoring model value from %s longer than %d characters.",
            source,
            _MAX_MODEL_NAME_LENGTH,
        )
        return None
    if any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in normalized):
        _LOGGER.warning("Ignoring model value from %s containing whitespace or control characters.", source)
        return None
    return normalized


def _normalize_reasoning_effort(value: object, *, source: str) -> str | None:
    """Return a validated reasoning effort, or ``None`` if invalid."""

    if value is None:
        return None
    if not isinstance(value, str):  # AUDIT-FIX(#5): Validate caller-provided defaults with the same rules as env values.
        _LOGGER.warning("Ignoring non-string reasoning effort from %s.", source)
        return None
    normalized = value.strip().lower()
    if not normalized:
        _LOGGER.warning("Ignoring blank reasoning effort from %s.", source)
        return None
    if normalized not in _VALID_REASONING_EFFORTS:
        _LOGGER.warning("Ignoring unsupported reasoning effort from %s.", source)
        return None
    return normalized


def codex_timeout_seconds(*env_names: str, default: float = _DEFAULT_TIMEOUT_SECONDS) -> float:
    """Return the first valid timeout from the environment, else a safe default."""

    safe_default = _normalize_timeout(default, source="default")
    if safe_default is None:  # AUDIT-FIX(#1): Ensure the function never returns an invalid caller-provided default.
        _LOGGER.warning(
            "Falling back to built-in timeout default %.1f seconds after invalid caller default.",
            _DEFAULT_TIMEOUT_SECONDS,
        )
        safe_default = _DEFAULT_TIMEOUT_SECONDS

    for env_name in env_names:
        normalized_env_name = _normalize_env_name(env_name)  # AUDIT-FIX(#3): Guard environment lookups against invalid keys.
        if normalized_env_name is None:
            continue
        raw = os.getenv(normalized_env_name)
        if raw is None:
            continue
        timeout_seconds = _normalize_timeout(raw, source=normalized_env_name)
        if timeout_seconds is not None:
            return timeout_seconds
    return safe_default


def codex_optional_model(env_name: str, *, default: str | None = _DEFAULT_MODEL) -> str | None:
    """Return one validated optional model override from the environment."""

    safe_default = _normalize_model(default, source="default")
    if default is not None and safe_default is None:  # AUDIT-FIX(#4): Recover to the built-in model when caller default is malformed.
        _LOGGER.warning("Falling back to built-in model default after invalid caller default.")
        safe_default = _DEFAULT_MODEL

    normalized_env_name = _normalize_env_name(env_name)  # AUDIT-FIX(#3): Prevent invalid env-name crashes in model lookup.
    if normalized_env_name is None:
        return safe_default

    raw = os.getenv(normalized_env_name)
    if raw is None:
        return safe_default
    normalized_model = _normalize_model(raw, source=normalized_env_name)
    return normalized_model if normalized_model is not None else safe_default


def codex_reasoning_effort(env_name: str, *, default: str | None = None) -> str | None:
    """Return one validated reasoning-effort override from the environment."""

    safe_default = _normalize_reasoning_effort(default, source="default")
    if default is not None and safe_default is None:  # AUDIT-FIX(#5): Never return an invalid caller-provided default.
        _LOGGER.warning("Falling back to no reasoning-effort override after invalid caller default.")
        safe_default = None

    normalized_env_name = _normalize_env_name(env_name)  # AUDIT-FIX(#3): Prevent invalid env-name crashes in effort lookup.
    if normalized_env_name is None:
        return safe_default

    raw = os.getenv(normalized_env_name)
    if raw is None:
        return safe_default
    normalized_effort = _normalize_reasoning_effort(raw, source=normalized_env_name)
    return normalized_effort if normalized_effort is not None else safe_default


__all__ = [
    "codex_optional_model",
    "codex_reasoning_effort",
    "codex_timeout_seconds",
]