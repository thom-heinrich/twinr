"""Expose the stable presenter helper surface for the Twinr web app.

Import presenter builders from this package root so route handlers and context
builders can depend on one curated export surface instead of many individual
modules.
"""

from __future__ import annotations

from twinr.web.presenters.common import (
    _default_reminder_due_at,
    _nav_items,
    _provider_status,
    _recent_named_files,
    _reminder_rows,
    _resolve_named_file,
)
from twinr.web.presenters.connect import _connect_sections
from twinr.web.presenters.debug import build_ops_debug_page_context, coerce_ops_debug_tab
from twinr.web.presenters.integrations import (
    _build_calendar_integration_record,
    _build_email_integration_record,
    _calendar_integration_sections,
    _email_integration_sections,
    _integration_overview_rows,
)
from twinr.web.presenters.memory import _memory_sections
from twinr.web.presenters.ops import _format_log_rows, _format_usage_rows, _health_card_detail
from twinr.web.presenters.self_coding import build_self_coding_ops_page_context
from twinr.web.presenters.settings import _adaptive_timing_view, _settings_sections
from twinr.web.presenters.voice import (
    _capture_voice_profile_sample,
    _voice_action_result,
    _voice_profile_page_context,
    _voice_snapshot_label,
)

__all__ = [
    "_adaptive_timing_view",
    "_build_calendar_integration_record",
    "_build_email_integration_record",
    "_calendar_integration_sections",
    "_capture_voice_profile_sample",
    "_connect_sections",
    "_default_reminder_due_at",
    "_email_integration_sections",
    "_format_log_rows",
    "_format_usage_rows",
    "_health_card_detail",
    "_integration_overview_rows",
    "_memory_sections",
    "_nav_items",
    "_provider_status",
    "_recent_named_files",
    "_reminder_rows",
    "_resolve_named_file",
    "_settings_sections",
    "build_ops_debug_page_context",
    "build_self_coding_ops_page_context",
    "coerce_ops_debug_tab",
    "_voice_action_result",
    "_voice_profile_page_context",
    "_voice_snapshot_label",
]
