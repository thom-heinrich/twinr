from __future__ import annotations

import math
from typing import Any

from twinr.agent.base_agent.config import TwinrConfig

# AUDIT-FIX(#1): Default timeout/retry policy is now explicit and tunable so the client does not inherit opaque SDK defaults on flaky home WiFi.
_DEFAULT_OPENAI_TIMEOUT_S = 45.0
_DEFAULT_OPENAI_MAX_RETRIES = 1

# AUDIT-FIX(#2): Bool-like and identifier config values are normalized centrally to avoid whitespace/header bugs and string-truthiness mistakes.
_PROJECT_SCOPED_KEY_PREFIX = "sk-proj-"
_TRUTHY_STRINGS = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSY_STRINGS = frozenset({"0", "false", "f", "no", "n", "off"})


def _default_client_factory(config: TwinrConfig) -> Any:
    # AUDIT-FIX(#2): Normalize and validate the API key so blank/whitespace/control-character values fail fast.
    api_key = _normalize_identifier(
        "OPENAI_API_KEY",
        getattr(config, "openai_api_key", None),
        required=True,
        reject_whitespace=True,
    )

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - exercised when dependency is missing at runtime
        # AUDIT-FIX(#4): Use deployment-agnostic guidance instead of a hard-coded editable-install path.
        raise RuntimeError(
            "The OpenAI SDK is not installed in this environment. Install the `openai` package for the Twinr runtime."
        ) from exc

    # AUDIT-FIX(#2): Normalize the optional project identifier before it can be copied into request headers.
    project_id = _normalize_identifier(
        "OPENAI_PROJECT_ID",
        getattr(config, "openai_project_id", None),
        required=False,
        reject_whitespace=False,
    )
    # AUDIT-FIX(#2): Coerce bool-like config values explicitly so strings like "false" do not become truthy.
    send_project_header = _should_send_project_header_value(
        api_key=api_key,
        project_id=project_id,
        configured_value=getattr(config, "openai_send_project_header", None),
    )

    # AUDIT-FIX(#1): Pin explicit timeout/retry defaults for flaky home WiFi instead of relying on SDK defaults.
    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": _coerce_positive_float(
            "openai_timeout_s",
            getattr(config, "openai_timeout_s", None),
            default=_DEFAULT_OPENAI_TIMEOUT_S,
            minimum=1.0,
        ),
        "max_retries": _coerce_bounded_int(
            "openai_max_retries",
            getattr(config, "openai_max_retries", None),
            default=_DEFAULT_OPENAI_MAX_RETRIES,
            minimum=0,
            maximum=5,
        ),
    }
    if send_project_header and project_id is not None:
        kwargs["project"] = project_id

    try:
        return OpenAI(**kwargs)
    except (TypeError, ValueError) as exc:
        # AUDIT-FIX(#3): Re-raise configuration/constructor failures with an actionable operator-facing message.
        raise RuntimeError(
            "Failed to initialize the OpenAI client from TwinrConfig. Check the OpenAI-related configuration values."
        ) from exc


def _should_send_project_header(config: TwinrConfig) -> bool:
    # AUDIT-FIX(#2): Reuse the same normalization logic as the client factory so heuristic checks see clean input.
    api_key = _normalize_identifier(
        "OPENAI_API_KEY",
        getattr(config, "openai_api_key", None),
        required=False,
        reject_whitespace=True,
    ) or ""
    # AUDIT-FIX(#2): Treat blank/whitespace project IDs as absent instead of sending invalid headers.
    project_id = _normalize_identifier(
        "OPENAI_PROJECT_ID",
        getattr(config, "openai_project_id", None),
        required=False,
        reject_whitespace=False,
    )
    return _should_send_project_header_value(
        api_key=api_key,
        project_id=project_id,
        configured_value=getattr(config, "openai_send_project_header", None),
    )


# AUDIT-FIX(#2): Split out the decision logic so validated values can be reused without duplicating header-selection rules.
def _should_send_project_header_value(*, api_key: str, project_id: str | None, configured_value: Any) -> bool:
    if project_id is None:
        return False

    parsed_flag = _coerce_optional_bool("openai_send_project_header", configured_value)
    if parsed_flag is not None:
        return parsed_flag

    return not api_key.startswith(_PROJECT_SCOPED_KEY_PREFIX)


# AUDIT-FIX(#2): Normalize string identifiers once and reject malformed values before they reach auth/header handling.
def _normalize_identifier(name: str, value: Any, *, required: bool, reject_whitespace: bool) -> str | None:
    if value is None:
        if required:
            raise RuntimeError(f"{name} is required to use the OpenAI backend")
        return None

    if not isinstance(value, str):
        raise RuntimeError(f"{name} must be a string")

    normalized = value.strip()
    if not normalized:
        if required:
            raise RuntimeError(f"{name} is required to use the OpenAI backend")
        return None

    if reject_whitespace and any(character.isspace() for character in normalized):
        raise RuntimeError(f"{name} must not contain whitespace characters")
    if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
        raise RuntimeError(f"{name} contains invalid control characters")

    return normalized


# AUDIT-FIX(#2): Parse bool-like config values defensively so env values such as "false" behave as intended.
def _coerce_optional_bool(name: str, value: Any) -> bool | None:
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, int) and value in (0, 1):
        return bool(value)

    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return None
        if normalized in _TRUTHY_STRINGS:
            return True
        if normalized in _FALSY_STRINGS:
            return False

    allowed_values = ", ".join(sorted(_TRUTHY_STRINGS | _FALSY_STRINGS))
    raise RuntimeError(f"{name} must be a boolean or one of: {allowed_values}")


# AUDIT-FIX(#1): Validate timeout values explicitly so non-finite or undersized numbers do not create pathological network behavior.
def _coerce_positive_float(name: str, value: Any, *, default: float, minimum: float) -> float:
    if value is None:
        return default

    if isinstance(value, bool):
        raise RuntimeError(f"{name} must be a finite number greater than or equal to {minimum}")

    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be a finite number greater than or equal to {minimum}") from exc

    if not math.isfinite(parsed) or parsed < minimum:
        raise RuntimeError(f"{name} must be a finite number greater than or equal to {minimum}")

    return parsed


# AUDIT-FIX(#1): Bound retry counts so bad config cannot create unbounded waits or retry storms on weak home networks.
def _coerce_bounded_int(name: str, value: Any, *, default: int, minimum: int, maximum: int) -> int:
    if value is None:
        return default

    if isinstance(value, bool):
        raise RuntimeError(f"{name} must be an integer between {minimum} and {maximum}")

    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be an integer between {minimum} and {maximum}") from exc

    if parsed < minimum or parsed > maximum:
        raise RuntimeError(f"{name} must be an integer between {minimum} and {maximum}")

    return parsed