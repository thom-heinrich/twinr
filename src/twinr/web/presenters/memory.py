"""Build memory-related settings sections for the Twinr web UI."""

from __future__ import annotations

from collections.abc import Mapping

from twinr.agent.base_agent import TwinrConfig
from twinr.web.support.contracts import SettingsSection
from twinr.web.support.forms import _select_field, _text_field
from twinr.web.presenters.common import _BOOL_OPTIONS

# AUDIT-FIX(#1): Canonical boolean tokens used to normalize config-backed defaults for _BOOL_OPTIONS.
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def _config_default(config: TwinrConfig, attr_name: str, fallback: object) -> object:
    """Read a config attribute while treating ``None`` as unset."""

    # AUDIT-FIX(#1): Keep form defaults aligned with TwinrConfig while staying backward-compatible with older config objects.
    value = getattr(config, attr_name, fallback)
    return fallback if value is None else value


def _string_default(config: TwinrConfig, attr_name: str, fallback: object) -> str:
    """Return a stable string default derived from configuration."""

    # AUDIT-FIX(#1): Convert config-backed defaults to stable strings before handing them to field factories.
    return str(_config_default(config, attr_name, fallback))


def _bool_default(config: TwinrConfig, attr_name: str, fallback: bool) -> str:
    """Return canonical ``true`` or ``false`` tokens for boolean defaults."""

    # AUDIT-FIX(#1): Normalize booleans and common env-style string values to the exact tokens expected by _BOOL_OPTIONS.
    value = _config_default(config, attr_name, fallback)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_VALUES:
            return "true"
        if normalized in _FALSE_VALUES:
            return "false"
        return "true" if fallback else "false"
    return "true" if bool(value) else "false"


def _memory_sections(config: TwinrConfig, env_values: Mapping[str, str]) -> tuple[SettingsSection, ...]:
    """Build settings sections for on-device, print, and long-term memory."""

    # AUDIT-FIX(#4): Accept generic read-only mappings such as os.environ without forcing callers into a concrete dict.
    long_term_memory_enabled_default = _bool_default(config, "long_term_memory_enabled", False)
    # AUDIT-FIX(#1): Derive long-term memory defaults from config instead of stale hard-coded UI literals.
    long_term_memory_backend_default = _string_default(config, "long_term_memory_backend", "chonkydb")
    long_term_memory_path_default = _string_default(config, "long_term_memory_path", "state/chonkydb")

    return (
        SettingsSection(
            title="On-device memory",
            # AUDIT-FIX(#2): Use plain operational language to reduce caregiver and installer misconfiguration.
            description="Controls how much recent conversation Twinr keeps on the device before older parts are condensed.",
            fields=(
                _text_field(
                    "TWINR_MEMORY_MAX_TURNS",
                    "Max turns",
                    env_values,
                    str(config.memory_max_turns),
                    # AUDIT-FIX(#2): Replace internal jargon with concrete setup language.
                    tooltip_text="Maximum number of recent conversation turns kept in active memory before older content is condensed.",
                ),
                _text_field(
                    "TWINR_MEMORY_KEEP_RECENT",
                    "Keep recent turns",
                    env_values,
                    str(config.memory_keep_recent),
                    # AUDIT-FIX(#2): Clarify exactly what is preserved during compaction.
                    tooltip_text="Number of newest turns always kept exactly as they were when older content is condensed.",
                ),
            ),
        ),
        SettingsSection(
            title="Print memory",
            # AUDIT-FIX(#2): Keep wording concrete for non-technical operators.
            description="Limits for the print composer so button and tool-based printouts stay short and predictable.",
            fields=(
                _text_field(
                    "TWINR_PRINT_CONTEXT_TURNS",
                    "Print context turns",
                    env_values,
                    str(config.print_context_turns),
                    # AUDIT-FIX(#2): Explain the setting in direct operational terms.
                    tooltip_text="How many recent turns the print composer may read before creating one receipt.",
                ),
                _text_field(
                    "TWINR_PRINT_MAX_LINES",
                    "Print max lines",
                    env_values,
                    str(config.print_max_lines),
                    # AUDIT-FIX(#2): Make the limit explicit to reduce support confusion.
                    tooltip_text="Maximum number of lines allowed on a single printed receipt.",
                ),
                _text_field(
                    "TWINR_PRINT_MAX_CHARS",
                    "Print max chars",
                    env_values,
                    str(config.print_max_chars),
                    # AUDIT-FIX(#2): Keep language simple and concrete.
                    tooltip_text="Maximum number of characters allowed on a single printed receipt.",
                ),
            ),
        ),
        SettingsSection(
            title="Long-term memory",
            # AUDIT-FIX(#2): Remove roadmap/product jargon from operator-facing copy.
            description="Controls the optional long-term memory store used to keep information beyond the active conversation.",
            fields=(
                _select_field(
                    "TWINR_LONG_TERM_MEMORY_ENABLED",
                    "Long-term memory",
                    env_values,
                    _BOOL_OPTIONS,
                    long_term_memory_enabled_default,
                    # AUDIT-FIX(#2): Make enable/disable behavior understandable without internal terminology.
                    tooltip_text="Turn long-term memory on or off. Leave it off unless this device is configured to store long-term memories.",
                ),
                _text_field(
                    "TWINR_LONG_TERM_MEMORY_BACKEND",
                    "Backend",
                    env_values,
                    long_term_memory_backend_default,
                    # AUDIT-FIX(#2): Discourage unsupported hand-edits without leaking roadmap jargon.
                    tooltip_text="Backend identifier for the long-term memory store. Leave the default unless support tells you otherwise.",
                ),
                _text_field(
                    "TWINR_LONG_TERM_MEMORY_PATH",
                    "Storage path",
                    env_values,
                    long_term_memory_path_default,
                    # AUDIT-FIX(#3): Add a defense-in-depth safety cue for a field that can eventually influence disk writes.
                    tooltip_text="Project-local relative path for long-term memory data. Do not use absolute paths or paths outside the Twinr state directory.",
                ),
            ),
        ),
    )
