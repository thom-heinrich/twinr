from __future__ import annotations

from twinr.agent.base_agent import TwinrConfig
from twinr.web.contracts import SettingsSection
from twinr.web.forms import _select_field, _text_field
from twinr.web.viewmodels_common import _BOOL_OPTIONS


def _memory_sections(config: TwinrConfig, env_values: dict[str, str]) -> tuple[SettingsSection, ...]:
    return (
        SettingsSection(
            title="On-device memory",
            description="Controls how much rolling conversation state Twinr keeps locally before compacting it.",
            fields=(
                _text_field(
                    "TWINR_MEMORY_MAX_TURNS",
                    "Max turns",
                    env_values,
                    str(config.memory_max_turns),
                    tooltip_text="Upper bound for rolling conversation turns kept in active memory before compaction.",
                ),
                _text_field(
                    "TWINR_MEMORY_KEEP_RECENT",
                    "Keep recent turns",
                    env_values,
                    str(config.memory_keep_recent),
                    tooltip_text="Number of newest turns kept verbatim when older turns are compacted.",
                ),
            ),
        ),
        SettingsSection(
            title="Print memory",
            description="Bounds for the print composer so button and tool-based prints stay short and safe.",
            fields=(
                _text_field(
                    "TWINR_PRINT_CONTEXT_TURNS",
                    "Print context turns",
                    env_values,
                    str(config.print_context_turns),
                    tooltip_text="How many recent turns the print composer may inspect before creating a receipt.",
                ),
                _text_field(
                    "TWINR_PRINT_MAX_LINES",
                    "Print max lines",
                    env_values,
                    str(config.print_max_lines),
                    tooltip_text="Maximum printed line count for one answer receipt.",
                ),
                _text_field(
                    "TWINR_PRINT_MAX_CHARS",
                    "Print max chars",
                    env_values,
                    str(config.print_max_chars),
                    tooltip_text="Hard upper bound for receipt text length.",
                ),
            ),
        ),
        SettingsSection(
            title="Long-term memory",
            description="Configure the active long-term memory path and backend settings used for graph memory and future remote ChonkyDB integration.",
            fields=(
                _select_field(
                    "TWINR_LONG_TERM_MEMORY_ENABLED",
                    "Long-term memory",
                    env_values,
                    _BOOL_OPTIONS,
                    "false",
                    tooltip_text="Enable the active long-term memory layer for graph recall and background episodic persistence.",
                ),
                _text_field(
                    "TWINR_LONG_TERM_MEMORY_BACKEND",
                    "Backend",
                    env_values,
                    "chonkydb",
                    tooltip_text="Identifier for the future long-term memory backend.",
                ),
                _text_field(
                    "TWINR_LONG_TERM_MEMORY_PATH",
                    "Storage path",
                    env_values,
                    "state/chonkydb",
                    tooltip_text="Project-local path for the long-term graph and related on-disk state.",
                ),
            ),
        ),
    )
