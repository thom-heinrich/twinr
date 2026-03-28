"""Build memory-related settings sections for the Twinr web UI."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from twinr.agent.base_agent.state.snapshot import RuntimeSnapshot
from twinr.agent.base_agent.config import TwinrConfig
from twinr.web.support.contracts import DashboardCard, DetailMetric, SectionGroup, SettingsSection
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


def _count_label(count: int, singular: str, plural: str | None = None) -> str:
    """Render one compact count label with stable pluralization."""

    suffix = singular if count == 1 else (plural or f"{singular}s")
    return f"{count} {suffix}"


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


def build_activity_overview_cards(
    snapshot: RuntimeSnapshot,
    reminder_entries: Sequence[object],
    delivered_reminder_entries: Sequence[object],
    durable_memory_entries: Sequence[object],
) -> tuple[DashboardCard, ...]:
    """Build the top overview cards for the Activity & Memory page."""

    trace_count = len(snapshot.memory_raw_tail) + len(snapshot.memory_ledger) + len(snapshot.memory_search_results)
    return (
        DashboardCard(
            title="Conversation now",
            value=_count_label(len(snapshot.memory_turns), "context turn"),
            detail=f"{snapshot.status.title()} right now. These turns shape the next reply first.",
            href="#current-context",
        ),
        DashboardCard(
            title="Reminders",
            value=f"{len(reminder_entries)} pending",
            detail=f"{len(delivered_reminder_entries)} delivered reminders stay visible for review.",
            href="#reminders",
        ),
        DashboardCard(
            title="Saved memories",
            value=_count_label(len(durable_memory_entries), "saved item"),
            detail="Only explicit remember-this items belong here.",
            href="#saved-memories",
        ),
        DashboardCard(
            title="Advanced traces",
            value=_count_label(trace_count, "trace item"),
            detail="Raw tail, compact ledger, and stored web answers stay grouped at the end.",
            href="#advanced-memory",
        ),
    )


def build_activity_snapshot_metrics(snapshot: RuntimeSnapshot) -> tuple[DetailMetric, ...]:
    """Describe the current runtime memory snapshot in plain operator language."""

    return (
        DetailMetric(
            label="Status",
            value=snapshot.status.title(),
            detail="Current runtime state for the active memory snapshot.",
        ),
        DetailMetric(
            label="Last heard",
            value=snapshot.last_transcript or "—",
            detail="Most recent transcript kept in runtime state.",
        ),
        DetailMetric(
            label="Last answer",
            value=snapshot.last_response or "—",
            detail="Most recent spoken or generated Twinr reply.",
        ),
        DetailMetric(
            label="Updated",
            value=snapshot.updated_at or "—",
            detail="Last time the runtime snapshot file changed.",
        ),
    )


def build_activity_advanced_metrics(snapshot: RuntimeSnapshot) -> tuple[DetailMetric, ...]:
    """Summarize the deeper runtime traces shown later on the page."""

    return (
        DetailMetric(
            label="Open loops",
            value=str(len(snapshot.memory_state.open_loops)),
            detail="Outstanding follow-ups the runtime still carries forward.",
        ),
        DetailMetric(
            label="Verbatim tail",
            value=str(len(snapshot.memory_raw_tail)),
            detail="Newest uncondensed turns kept before compaction.",
        ),
        DetailMetric(
            label="Compacted ledger",
            value=str(len(snapshot.memory_ledger)),
            detail="Facts and summaries preserved after compaction.",
        ),
        DetailMetric(
            label="Stored search answers",
            value=str(len(snapshot.memory_search_results)),
            detail="Verified web lookups carried into later turns and printing.",
        ),
    )


def build_memory_section_groups(sections: Sequence[SettingsSection]) -> tuple[SectionGroup, ...]:
    """Group memory-related settings into calmer operator-facing buckets."""

    section_by_title = {section.title: section for section in sections}
    groups = (
        SectionGroup(
            key="memory-bounds",
            title="Conversation and print limits",
            description="Keep the active conversation concise and printed output predictable.",
            sections=tuple(
                section_by_title[title]
                for title in ("On-device memory", "Print memory")
                if title in section_by_title
            ),
        ),
        SectionGroup(
            key="long-term-memory",
            title="Long-term storage",
            description="Use long-term storage only when the backend and path are intentionally configured.",
            sections=tuple(
                section_by_title[title]
                for title in ("Long-term memory",)
                if title in section_by_title
            ),
        ),
    )
    return tuple(group for group in groups if group.sections)
