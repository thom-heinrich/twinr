"""Build the Twinr web ``/connect`` presenter sections.

This module turns env-backed provider and credential settings into
``SettingsSection`` view models while normalizing malformed persisted values
into stable, operator-safe defaults.
"""

from __future__ import annotations

from collections.abc import Mapping  # AUDIT-FIX(#1): Allow defensive handling of file-backed mappings without assuming a concrete dict type.
from typing import Any, Final  # AUDIT-FIX(#4): Centralize literal defaults and keep helper return typing explicit.

from twinr.web.support.contracts import SettingsSection
from twinr.web.support.forms import _select_field, _text_field
from twinr.web.support.store import FileBackedSetting, mask_secret
from twinr.web.presenters.common import _PROVIDER_OPTIONS, _TRISTATE_BOOL_OPTIONS

_PROVIDER_DEFAULT: Final[str] = "openai"  # AUDIT-FIX(#4): Remove repeated provider default literals.
_PROJECT_HEADER_AUTO_DEFAULT: Final[str] = ""  # AUDIT-FIX(#4): Remove repeated tristate sentinel literals.
_SECRET_PRESENT_HELP_TEXT: Final[str] = "A value is already stored. Leave blank to keep it unchanged."  # AUDIT-FIX(#2): Keep secret UX informative without exposing fingerprints.
_SECRET_MISSING_HELP_TEXT: Final[str] = "No value is stored yet. Enter a value to save it."  # AUDIT-FIX(#2): Keep secret UX informative without exposing fingerprints.
_INVALID_OPTION_HELP_TEXT: Final[str] = "Stored value is unsupported. Showing the safe default until you save."  # AUDIT-FIX(#3): Surface recovery from stale/invalid persisted select values.


def _coerce_env_value(raw_value: object) -> str:
    """Normalize one persisted env value into a trimmed string."""

    # AUDIT-FIX(#1): Normalize malformed file-backed values before they reach shared form builders.
    if raw_value is None:
        return ""
    if isinstance(raw_value, str):
        return raw_value.strip()
    return str(raw_value).strip()


def _normalized_env_values(env_values: Mapping[str, object] | None) -> dict[str, str]:
    """Snapshot mapping-like env values into a string-only dictionary."""

    # AUDIT-FIX(#1): Snapshot and normalize file-backed data so malformed values never crash view-model construction.
    if not env_values:
        return {}
    if not isinstance(env_values, Mapping):
        return {}

    normalized: dict[str, str] = {}
    for key, raw_value in env_values.items():
        if not isinstance(key, str):
            continue
        normalized[key] = _coerce_env_value(raw_value)
    return normalized


def _allowed_option_values(options: object) -> set[str]:
    """Collect canonical option values from select field definitions."""

    # AUDIT-FIX(#3): Extract canonical option values so select fields can recover from stale persisted data.
    allowed: set[str] = set()
    if not isinstance(options, (tuple, list)):
        return allowed

    for option in options:
        if isinstance(option, (tuple, list)) and option:
            allowed.add(_coerce_env_value(option[0]))
            continue

        option_value = getattr(option, "value", None)
        if option_value is not None:
            allowed.add(_coerce_env_value(option_value))

    return {value for value in allowed if value}


def _sanitized_select_env_values(
    key: str,
    env_values: Mapping[str, str],
    options: object,
    default: str,
) -> tuple[dict[str, str], str | None]:
    """Normalize one select-backed setting before building a form field.

    Args:
        key: Env key for the field.
        env_values: Current normalized env snapshot.
        options: Select options accepted by the form control.
        default: Safe default option when the stored value is empty or invalid.

    Returns:
        A per-field env mapping with a canonical value plus optional help text
        describing recovery from invalid persisted state.
    """

    # AUDIT-FIX(#3): Canonicalize select-backed settings and recover from stale/invalid persisted values instead of rendering inconsistent controls.
    field_env_values = dict(env_values)
    allowed_values = _allowed_option_values(options)
    normalized_value = _coerce_env_value(field_env_values.get(key, ""))

    if not allowed_values:
        field_env_values[key] = normalized_value or default
        return field_env_values, None

    if not normalized_value:
        field_env_values[key] = default
        return field_env_values, None

    if normalized_value in allowed_values:
        field_env_values[key] = normalized_value
        return field_env_values, None

    lower_to_actual = {value.lower(): value for value in allowed_values}
    matched_value = lower_to_actual.get(normalized_value.lower())
    if matched_value is not None:
        field_env_values[key] = matched_value
        return field_env_values, None

    field_env_values[key] = default
    return field_env_values, _INVALID_OPTION_HELP_TEXT


def _select_field_safe(
    key: str,
    label: str,
    env_values: Mapping[str, str],
    options: object,
    default: str,
    *,
    help_text: str | None = None,
    tooltip_text: str | None = None,
) -> Any:
    """Build a select field after sanitizing its persisted value."""

    field_env_values, validation_help_text = _sanitized_select_env_values(key, env_values, options, default)  # AUDIT-FIX(#3): Normalize per-field select state before passing it to the shared form builder.
    merged_help_text = help_text
    if validation_help_text:
        merged_help_text = validation_help_text if help_text is None else f"{validation_help_text} {help_text}"

    kwargs: dict[str, str] = {}
    if merged_help_text is not None:
        kwargs["help_text"] = merged_help_text
    if tooltip_text is not None:
        kwargs["tooltip_text"] = tooltip_text

    return _select_field(key, label, field_env_values, options, default, **kwargs)


def _secret_help_text(env_values: Mapping[str, str], key: str) -> str:
    """Return presence-only help text for a stored secret field."""

    raw_value = _coerce_env_value(env_values.get(key, ""))
    if not raw_value:
        return _SECRET_MISSING_HELP_TEXT

    try:
        mask_secret(raw_value)  # AUDIT-FIX(#2): Preserve compatibility with the shared masking path without surfacing any secret preview in the UI.
    except Exception:
        return _SECRET_PRESENT_HELP_TEXT

    return _SECRET_PRESENT_HELP_TEXT


def _connect_sections(env_values: dict[str, str] | Mapping[str, object] | None) -> tuple[SettingsSection, ...]:
    """Build the ``/connect`` settings sections from normalized env values."""

    normalized_env_values = _normalized_env_values(env_values)  # AUDIT-FIX(#1): Build the entire settings page from a stable, sanitized snapshot.
    return (
        SettingsSection(
            title="Provider routing",
            description="Choose which backend handles each pipeline stage.",
            fields=(
                _select_field_safe(  # AUDIT-FIX(#3): Recover gracefully from invalid persisted provider selections.
                    "TWINR_PROVIDER_LLM",
                    "LLM provider",
                    normalized_env_values,
                    _PROVIDER_OPTIONS,
                    _PROVIDER_DEFAULT,
                    tooltip_text="Controls which backend answers normal text questions.",
                ),
                _select_field_safe(  # AUDIT-FIX(#3): Recover gracefully from invalid persisted provider selections.
                    "TWINR_PROVIDER_STT",
                    "STT provider",
                    normalized_env_values,
                    _PROVIDER_OPTIONS,
                    _PROVIDER_DEFAULT,
                    tooltip_text="Controls which backend turns speech into text.",
                ),
                _select_field_safe(  # AUDIT-FIX(#3): Recover gracefully from invalid persisted provider selections.
                    "TWINR_PROVIDER_TTS",
                    "TTS provider",
                    normalized_env_values,
                    _PROVIDER_OPTIONS,
                    _PROVIDER_DEFAULT,
                    tooltip_text="Controls which backend turns replies into spoken audio.",
                ),
                _select_field_safe(  # AUDIT-FIX(#3): Recover gracefully from invalid persisted provider selections.
                    "TWINR_PROVIDER_REALTIME",
                    "Realtime provider",
                    normalized_env_values,
                    _PROVIDER_OPTIONS,
                    _PROVIDER_DEFAULT,
                    tooltip_text="Controls the low-latency voice session backend.",
                ),
            ),
        ),
        SettingsSection(
            title="OpenAI",
            description="Main account and auth settings for the currently active runtime.",
            fields=(
                FileBackedSetting(
                    key="OPENAI_API_KEY",
                    label="API key",
                    value="",
                    help_text=_secret_help_text(normalized_env_values, "OPENAI_API_KEY"),  # AUDIT-FIX(#2): Only expose configured/unconfigured state for secrets.
                    tooltip_text="The main OpenAI secret used for chat, speech, vision, and realtime requests.",
                    input_type="password",
                    placeholder="sk-...",
                    secret=True,
                ),
                _text_field(
                    "OPENAI_PROJ_ID",
                    "Project ID",
                    normalized_env_values,
                    "",
                    placeholder="proj_...",
                    tooltip_text="Optional project id. Only set this when your account setup requires an explicit OpenAI project.",
                ),
                _select_field_safe(  # AUDIT-FIX(#3): Recover gracefully from invalid persisted tristate values.
                    "OPENAI_SEND_PROJECT_HEADER",
                    "Project header",
                    normalized_env_values,
                    _TRISTATE_BOOL_OPTIONS,
                    _PROJECT_HEADER_AUTO_DEFAULT,
                    help_text="Use auto unless you explicitly need to force the header on or off.",
                    tooltip_text="Auto is usually correct. Force this only if your OpenAI key/project setup needs it.",
                ),
            ),
        ),
        SettingsSection(
            title="Other providers",
            description="Stored here now so later provider adapters can use them without editing files by hand.",
            fields=(
                FileBackedSetting(
                    key="DEEPINFRA_API_KEY",
                    label="DeepInfra API key",
                    value="",
                    help_text=_secret_help_text(normalized_env_values, "DEEPINFRA_API_KEY"),  # AUDIT-FIX(#2): Only expose configured/unconfigured state for secrets.
                    tooltip_text="Credential for a future DeepInfra provider integration.",
                    input_type="password",
                    placeholder="DeepInfra key",
                    secret=True,
                ),
                FileBackedSetting(
                    key="OPENROUTER_API_KEY",
                    label="OpenRouter API key",
                    value="",
                    help_text=_secret_help_text(normalized_env_values, "OPENROUTER_API_KEY"),  # AUDIT-FIX(#2): Only expose configured/unconfigured state for secrets.
                    tooltip_text="Credential for a future OpenRouter provider integration.",
                    input_type="password",
                    placeholder="OpenRouter key",
                    secret=True,
                ),
            ),
        ),
    )
