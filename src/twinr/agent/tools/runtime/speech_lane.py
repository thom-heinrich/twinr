"""Model and safely emit speech-lane output callbacks for the dual-lane loop."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable

from .loop_support import coerce_text, strip_text


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SpeechLaneDelta:
    """Describe one speech-output event emitted by the dual-lane loop.

    Attributes:
        text: Spoken text fragment to emit.
        lane: Output lane label such as ``direct``, ``filler``, or ``final``.
        replace_current: Whether this delta should replace the currently spoken
            filler content.
        atomic: Whether the delta should be treated as one indivisible segment.
    """

    text: str
    lane: str
    replace_current: bool = False
    atomic: bool = False


def safe_emit_text_delta(
    on_text_delta: Callable[[str], None] | None,
    text: str,
) -> None:
    """Emit a plain text delta while swallowing callback failures."""

    raw = coerce_text(text)
    if not raw or on_text_delta is None:
        return
    try:
        on_text_delta(raw)
    except Exception:
        logger.exception("on_text_delta callback failed.")


def safe_emit_speech_delta(
    on_lane_text_delta: Callable[[SpeechLaneDelta], None] | None,
    on_text_delta: Callable[[str], None] | None,
    delta: SpeechLaneDelta,
) -> None:
    """Prefer lane-aware speech emission and fall back to plain text callbacks."""

    raw = coerce_text(delta.text)
    if not raw:
        return
    if on_lane_text_delta is not None:
        try:
            on_lane_text_delta(
                SpeechLaneDelta(
                    text=raw,
                    lane=strip_text(delta.lane) or "direct",
                    replace_current=bool(delta.replace_current),
                    atomic=bool(delta.atomic),
                )
            )
            return
        except Exception:
            logger.exception("on_lane_text_delta callback failed.")
    safe_emit_text_delta(on_text_delta, raw)
