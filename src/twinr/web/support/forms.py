"""Construct and validate declared settings fields for the web UI.

This module centralizes file-backed field registration and form submission
normalization for Twinr's settings pages and managed integration forms. It
keeps accepted keys fail-closed and turns malformed persisted values into
stable display defaults.
"""

from __future__ import annotations

from typing import AbstractSet

from twinr.web.support.store import FileBackedSetting

# AUDIT-FIX(#1): Restrict setting keys to short ASCII identifiers to block confusable / injected keys.
_MAX_SETTING_KEY_LENGTH = 128
# AUDIT-FIX(#3): Bound accepted submitted values so malformed or hostile payloads cannot grow unbounded on a small RPi.
_MAX_SUBMITTED_VALUE_LENGTH = 16_384
# AUDIT-FIX(#6): Clamp textarea height to a sane UI range so one bad caller value cannot break the settings layout.
_MIN_TEXTAREA_ROWS = 1
_MAX_TEXTAREA_ROWS = 12

# AUDIT-FIX(#1): Track only keys explicitly declared through field helpers so form collection can fail closed.
_REGISTERED_SETTING_KEYS: set[str] = set()
# AUDIT-FIX(#5): Remember field kinds to avoid stripping significant whitespace from textarea values during save.
_REGISTERED_FIELD_TYPES: dict[str, str] = {}
# AUDIT-FIX(#2): Remember legal select values so posted select data is validated on both read and write paths.
_REGISTERED_SELECT_OPTIONS: dict[str, frozenset[str]] = {}


def _is_valid_setting_key(key: str) -> bool:
    """Return whether ``key`` is a safe ASCII settings identifier."""

    if not isinstance(key, str) or not key or len(key) > _MAX_SETTING_KEY_LENGTH or not key.isascii():
        return False
    first_char = key[0]
    if first_char != "_" and not first_char.isalpha():
        return False
    return all(character == "_" or character.isalnum() for character in key)


def _register_setting_key(
    key: str,
    *,
    input_type: str,
    options: tuple[tuple[str, str], ...] | None = None,
) -> None:
    """Record one declared setting key and its field metadata."""

    # AUDIT-FIX(#1): Reject unsafe keys at declaration time instead of letting them flow into file-backed configuration.
    if not _is_valid_setting_key(key):
        raise ValueError(
            f"Invalid setting key {key!r}; expected an ASCII identifier starting with a letter or underscore."
        )
    _REGISTERED_SETTING_KEYS.add(key)
    _REGISTERED_FIELD_TYPES[key] = input_type
    if input_type == "select":
        # AUDIT-FIX(#2): Persist the select allowlist so submitted values can be validated later.
        _REGISTERED_SELECT_OPTIONS[key] = frozenset(option_value for option_value, _ in (options or ()))
    else:
        _REGISTERED_SELECT_OPTIONS.pop(key, None)


def _coerce_display_value(value: object, default: str) -> str:
    """Convert persisted values into strings suitable for form rendering."""

    # AUDIT-FIX(#4): Canonicalize corrupted or non-string backing values before handing them to the UI model.
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return default


def _normalize_select_value(
    value: object,
    default: str,
    options: tuple[tuple[str, str], ...],
) -> str:
    """Clamp a persisted select value to the declared option list."""

    # AUDIT-FIX(#2): Keep select state inside the declared option set even when env/state contains stale or tampered data.
    allowed_values = frozenset(option_value for option_value, _ in options)
    fallback = default
    if fallback not in allowed_values and options:
        fallback = options[0][0]
    candidate = _coerce_display_value(value, fallback)
    return candidate if candidate in allowed_values else fallback


def _normalize_textarea_rows(rows: int) -> int:
    """Clamp textarea row counts to the supported UI range."""

    # AUDIT-FIX(#6): Prevent negative/oversized textarea row counts from destabilizing the settings UI.
    if not isinstance(rows, int):
        return 4
    return max(_MIN_TEXTAREA_ROWS, min(rows, _MAX_TEXTAREA_ROWS))


def _normalize_submitted_value(value: object, *, strip_outer_whitespace: bool) -> str | None:
    """Normalize one submitted form value or reject unsupported input."""

    # AUDIT-FIX(#3): Treat runtime types defensively because web-form payloads are not guaranteed to match type hints.
    if value is None:
        candidate = ""
    elif isinstance(value, str):
        candidate = value
    elif isinstance(value, (int, float, bool)):
        candidate = str(value)
    else:
        return None

    candidate = candidate.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    if strip_outer_whitespace:
        candidate = candidate.strip()

    if len(candidate) > _MAX_SUBMITTED_VALUE_LENGTH:
        return None
    return candidate


def _text_field(
    key: str,
    label: str,
    env_values: dict[str, str],
    default: str,
    *,
    help_text: str = "",
    tooltip_text: str = "",
    placeholder: str = "",
    wide: bool = False,
) -> FileBackedSetting:
    """Build a text input definition backed by one env key."""

    # AUDIT-FIX(#1): Register declared keys so update collection can reject undeclared form fields by default.
    _register_setting_key(key, input_type="text")
    return FileBackedSetting(
        key=key,
        label=label,
        # AUDIT-FIX(#4): Ensure text settings always render with a string value even if persisted state is malformed.
        value=_coerce_display_value(env_values.get(key), default),
        help_text=help_text,
        tooltip_text=tooltip_text,
        placeholder=placeholder,
        wide=wide,
    )


def _select_field(
    key: str,
    label: str,
    env_values: dict[str, str],
    options: tuple[tuple[str, str], ...],
    default: str,
    *,
    help_text: str = "",
    tooltip_text: str = "",
) -> FileBackedSetting:
    """Build a select input definition with an allowlisted option set."""

    # AUDIT-FIX(#1): Register declared keys so update collection can reject undeclared form fields by default.
    _register_setting_key(key, input_type="select", options=options)
    return FileBackedSetting(
        key=key,
        label=label,
        # AUDIT-FIX(#2): Never surface or retain a select value that is outside the declared option set.
        value=_normalize_select_value(env_values.get(key), default, options),
        help_text=help_text,
        tooltip_text=tooltip_text,
        input_type="select",
        options=options,
    )


def _textarea_field(
    key: str,
    label: str,
    env_values: dict[str, str],
    default: str,
    *,
    help_text: str = "",
    tooltip_text: str = "",
    placeholder: str = "",
    rows: int = 4,
) -> FileBackedSetting:
    """Build a textarea definition with bounded row counts."""

    # AUDIT-FIX(#1): Register declared keys so update collection can reject undeclared form fields by default.
    _register_setting_key(key, input_type="textarea")
    return FileBackedSetting(
        key=key,
        label=label,
        # AUDIT-FIX(#4): Ensure textarea settings always render with a string value even if persisted state is malformed.
        value=_coerce_display_value(env_values.get(key), default),
        help_text=help_text,
        tooltip_text=tooltip_text,
        input_type="textarea",
        placeholder=placeholder,
        # AUDIT-FIX(#6): Clamp caller-provided row counts to preserve a usable settings layout.
        rows=_normalize_textarea_rows(rows),
        wide=True,
    )


def _collect_standard_updates(
    form: dict[str, str],
    *,
    exclude: set[str] | None = None,
    allowed_keys: AbstractSet[str] | None = None,
) -> dict[str, str]:
    """Collect sanitized updates for declared standard settings fields."""

    blocked = frozenset(exclude or ())
    # AUDIT-FIX(#1): Default to declared helper keys and fail closed when no allowlist is available.
    permitted_keys = frozenset(
        key
        for key in (allowed_keys if allowed_keys is not None else _REGISTERED_SETTING_KEYS)
        if _is_valid_setting_key(key)
    )
    if not permitted_keys:
        return {}

    updates: dict[str, str] = {}
    for raw_key, raw_value in form.items():
        # AUDIT-FIX(#1): Ignore unknown, blocked, or structurally invalid keys instead of accepting every uppercase token.
        if not isinstance(raw_key, str) or raw_key in blocked or raw_key not in permitted_keys:
            continue

        normalized_value = _normalize_submitted_value(
            raw_value,
            # AUDIT-FIX(#5): Preserve intentional leading/trailing whitespace for textarea-backed settings.
            strip_outer_whitespace=_REGISTERED_FIELD_TYPES.get(raw_key) != "textarea",
        )
        if normalized_value is None:
            continue

        # AUDIT-FIX(#2): Reject tampered select submissions that are not part of the declared option set.
        select_options = _REGISTERED_SELECT_OPTIONS.get(raw_key)
        if select_options is not None and normalized_value not in select_options:
            continue

        updates[raw_key] = normalized_value

    return updates
