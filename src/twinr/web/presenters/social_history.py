"""Build social-history learning sections for the integrations page.

The portal uses this presenter to keep the opt-in history-learning controls
plain and explicit while leaving queue submission and runtime import work in
the route layer and WhatsApp channel service.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from twinr.integrations import (
    SOCIAL_HISTORY_ALLOWED_SOURCES,
    SOCIAL_HISTORY_LOOKBACK_OPTIONS,
    SOCIAL_HISTORY_SOURCE_WHATSAPP,
    ManagedIntegrationConfig,
    SocialHistoryLearningConfig,
)
from twinr.web.presenters.common import _YES_NO_OPTIONS
from twinr.web.support.contracts import SettingsSection, WizardCheckRow
from twinr.web.support.forms import _select_field

_SOCIAL_HISTORY_SOURCE_OPTIONS = (
    (SOCIAL_HISTORY_SOURCE_WHATSAPP, "WhatsApp"),
)
_ALLOWED_ACTIONS = frozenset({"save_social_history", "save_and_import_social_history"})


def _normalize_text(value: object, default: str = "") -> str:
    text = str(value or "").strip()
    return text or default


def _parse_bool_choice(raw: object, *, default: bool = False) -> bool:
    text = _normalize_text(raw).casefold()
    if not text:
        return default
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    raise ValueError("Learn-from-history must be yes or no.")


def _normalize_source(raw: object) -> str:
    candidate = _normalize_text(raw, SOCIAL_HISTORY_SOURCE_WHATSAPP).casefold()
    if candidate not in SOCIAL_HISTORY_ALLOWED_SOURCES:
        raise ValueError("Please choose a supported social-history source.")
    return candidate


def _normalize_lookback(raw: object) -> str:
    candidate = _normalize_text(raw, "3m").casefold()
    valid_values = {value for value, _label in SOCIAL_HISTORY_LOOKBACK_OPTIONS}
    if candidate not in valid_values:
        raise ValueError("Please choose a supported history lookback window.")
    return candidate


def _status_copy(config: SocialHistoryLearningConfig) -> tuple[str, str, str]:
    status = config.last_import_status
    if status == "disabled":
        return "muted", "Off", "Twinr will not read old social-history messages until you allow it here."
    if status == "queued":
        return "warn", "Queued", "Twinr has queued one bounded history import for the selected source and window."
    if status == "running":
        return "warn", "Importing", "Twinr is temporarily reopening the WhatsApp worker in history mode and writing approved messages into shared memory."
    if status == "completed":
        return "ok", "Imported", "The last bounded history import completed and its approved messages were added to shared Twinr memory."
    if status == "partial":
        return "warn", "Partial", "The last import wrote usable history into memory, but WhatsApp only returned part of the requested time window."
    if status == "failed":
        return "fail", "Failed", config.last_import_error or "The last history import failed."
    return "muted", "Idle", "Twinr has consent settings saved but no history import has run yet."


def _format_timestamp(value: str | None) -> str:
    if not value:
        return "Not yet"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return "Not yet"
    return parsed.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")


def _source_label(source: str) -> str:
    for value, label in _SOCIAL_HISTORY_SOURCE_OPTIONS:
        if value == source:
            return label
    return source


def _social_history_learning_sections(record: ManagedIntegrationConfig | None) -> tuple[SettingsSection, ...]:
    """Return the managed consent form sections for social-history learning."""

    config = SocialHistoryLearningConfig.from_record(record)
    values = {
        "enabled": "true" if config.enabled else "false",
        "source": config.source,
        "lookback_key": config.lookback_key,
    }
    return (
        SettingsSection(
            title="Learn from social history",
            description=(
                "Give Twinr explicit permission to import recent direct-message history into the same shared memory system. "
                "This is opt-in, bounded by the selected lookback window, and currently supports WhatsApp only."
            ),
            fields=(
                _select_field(
                    "enabled",
                    "Learn from my social media history",
                    values,
                    _YES_NO_OPTIONS,
                    "false",
                    tooltip_text="When enabled, Twinr may run one bounded import from the selected social source after you explicitly save or start it here.",
                ),
                _select_field(
                    "source",
                    "Source",
                    values,
                    _SOCIAL_HISTORY_SOURCE_OPTIONS,
                    SOCIAL_HISTORY_SOURCE_WHATSAPP,
                    tooltip_text="Choose which connected social source Twinr may read from. More sources can be added later without changing the consent model.",
                ),
                _select_field(
                    "lookback_key",
                    "History window",
                    values,
                    SOCIAL_HISTORY_LOOKBACK_OPTIONS,
                    "3m",
                    tooltip_text="Twinr imports only the selected recent time window, up to a maximum of one year.",
                ),
            ),
        ),
    )


def _social_history_learning_panel(record: ManagedIntegrationConfig | None) -> dict[str, Any]:
    """Build the status panel shown above the social-history form."""

    config = SocialHistoryLearningConfig.from_record(record)
    status, status_label, detail = _status_copy(config)
    return {
        "title": "Social-history learning",
        "status": status,
        "status_label": status_label,
        "detail": detail,
        "checks": (
            WizardCheckRow(
                label="Consent",
                summary="Allowed" if config.enabled else "Off",
                detail="Twinr reads old social messages only after explicit opt-in here.",
                status="ok" if config.enabled else "muted",
            ),
            WizardCheckRow(
                label="Source",
                summary=_source_label(config.source),
                detail="The currently selected social source for bounded history import.",
                status="ok" if config.enabled else "muted",
            ),
            WizardCheckRow(
                label="Window",
                summary=config.lookback_label,
                detail="Maximum age of messages Twinr may import during the next run.",
                status="ok" if config.enabled else "muted",
            ),
            WizardCheckRow(
                label="Last run",
                summary=_format_timestamp(config.last_import_finished_at),
                detail=(
                    f"Imported {config.last_import_messages} messages across {config.last_import_chats} chats "
                    f"and wrote {config.last_import_turns} shared-memory turns."
                    if config.last_import_finished_at and config.last_import_status in {"completed", "partial"}
                    else (config.last_import_error or "No completed history import has been recorded yet.")
                ),
                status=status,
            ),
        ),
        "primary_action": "Save consent",
        "secondary_action": "Save and import now",
    }


def _build_social_history_learning_record(
    form: dict[str, str],
    existing_record: ManagedIntegrationConfig | None,
) -> ManagedIntegrationConfig:
    """Validate one social-history form submission into a managed record."""

    current = SocialHistoryLearningConfig.from_record(existing_record)
    action = _normalize_text(form.get("_integration_action"), "save_social_history")
    if action not in _ALLOWED_ACTIONS:
        raise ValueError("Please choose a valid social-history action.")
    enabled = _parse_bool_choice(form.get("enabled"), default=current.enabled)
    source = _normalize_source(form.get("source"))
    lookback_key = _normalize_lookback(form.get("lookback_key"))
    if source != SOCIAL_HISTORY_SOURCE_WHATSAPP:
        raise ValueError("WhatsApp is the only supported social-history source right now.")

    next_status = current.last_import_status
    if not enabled:
        next_status = "disabled"
    elif current.last_import_status == "disabled":
        next_status = "idle"

    return SocialHistoryLearningConfig(
        enabled=enabled,
        source=source,
        lookback_key=lookback_key,
        last_import_status=next_status,
        last_import_request_id=current.last_import_request_id,
        last_import_started_at=current.last_import_started_at,
        last_import_finished_at=current.last_import_finished_at,
        last_import_error=current.last_import_error,
        last_import_detail=current.last_import_detail,
        last_import_messages=current.last_import_messages,
        last_import_turns=current.last_import_turns,
        last_import_chats=current.last_import_chats,
        last_import_oldest_at=current.last_import_oldest_at,
        last_import_newest_at=current.last_import_newest_at,
    ).to_record()


__all__ = [
    "_build_social_history_learning_record",
    "_social_history_learning_panel",
    "_social_history_learning_sections",
]
