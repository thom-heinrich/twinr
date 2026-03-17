"""Define canonical ack phrase IDs for the orchestrator protocol."""

from __future__ import annotations

_ACK_PAIRS: tuple[tuple[str, str], ...] = (
    ("one_moment", "Einen Moment bitte."),
    ("checking_now", "Ich schaue kurz nach."),
    ("checking_short", "Ich prüfe das kurz."),
    ("one_second", "Einen Augenblick bitte."),
)

ACK_ID_TO_TEXT: dict[str, str] = dict(_ACK_PAIRS)
ACK_TEXT_TO_ID: dict[str, str] = {text: ack_id for ack_id, text in _ACK_PAIRS}


def ack_id_for_text(text: str) -> str | None:
    """Look up the stable ack identifier for a spoken phrase."""

    return ACK_TEXT_TO_ID.get(text.strip())


def ack_text_for_id(ack_id: str) -> str | None:
    """Look up the spoken phrase for a stable ack identifier."""

    return ACK_ID_TO_TEXT.get(ack_id.strip())
