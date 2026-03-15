from __future__ import annotations

from twinr.agent.tools import SUPERVISOR_FAST_ACK_PHRASES

_ACK_PAIRS: tuple[tuple[str, str], ...] = (
    ("one_moment", SUPERVISOR_FAST_ACK_PHRASES[0]),
    ("checking_now", SUPERVISOR_FAST_ACK_PHRASES[1]),
    ("checking_short", SUPERVISOR_FAST_ACK_PHRASES[2]),
    ("one_second", SUPERVISOR_FAST_ACK_PHRASES[3]),
)

ACK_ID_TO_TEXT: dict[str, str] = dict(_ACK_PAIRS)
ACK_TEXT_TO_ID: dict[str, str] = {text: ack_id for ack_id, text in _ACK_PAIRS}


def ack_id_for_text(text: str) -> str | None:
    return ACK_TEXT_TO_ID.get(text.strip())


def ack_text_for_id(ack_id: str) -> str | None:
    return ACK_ID_TO_TEXT.get(ack_id.strip())
