"""Assemble hidden instructions for shared external text-channel turns.

This module keeps channel-aware response-style and capability-scope wording out
of the long-lived runtime loop so external text transports can share one
bounded instruction policy without growing orchestration code.
"""

from __future__ import annotations

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.tools.prompting.instructions import build_tool_agent_instructions


def _normalized_channel_label(channel: str) -> str:
    """Return a stable lowercase label for one external text channel."""

    normalized = str(channel or "").strip().lower()
    return normalized or "external"


def _response_style_guidance(channel: str) -> str:
    """Return channel-aware reply-style guidance for one text turn."""

    normalized = _normalized_channel_label(channel)
    if normalized == "whatsapp":
        return (
            "Because this turn came through WhatsApp, reply in a text-native style rather than a spoken style. "
            "You may be a bit more detailed and explicit than in voice, but stay concise, warm, and easy to scan. "
            "Prefer 2 to 5 short sentences or a short flat list when that genuinely helps. "
            "For simple confirmations or yes-no answers, one short line is enough. "
            "Do not pad with spoken filler, theatrical stage directions, or fake typing-style chatter."
        )
    return (
        "Because this is an external text channel, reply in a text-native style rather than as a spoken transcript. "
        "Stay concise and easy to scan, but you do not need to compress the answer as aggressively as voice."
    )


def build_tool_text_channel_turn_instructions(
    config: TwinrConfig,
    *,
    channel: str,
    tool_names: tuple[str, ...],
    pending_action_message: str | None = None,
) -> str:
    """Build the hidden instruction bundle for one external tool text turn."""

    channel_label = _normalized_channel_label(channel)
    available_tools_text = ", ".join(tool_names) if tool_names else "none"
    pending_text = ""
    if pending_action_message:
        pending_text = (
            " There is an active pending structured follow-up for this channel conversation. "
            f"{pending_action_message}"
        )
    return build_tool_agent_instructions(
        config,
        extra_instructions=(
            f"This turn came from the external {channel_label} text channel. "
            "Use the shared Twinr long-term memory, managed user and personality context, and the exposed tool surface just like the main Twinr system. "
            "Relevant memories from other Twinr channels may be recalled into this turn, so do not claim you only know this chat unless the system explicitly tells you shared memory is unavailable. "
            "If the user asks about memory access, explain that you see the current chat plus relevant shared Twinr memory recalled for this turn, not only this channel transcript. "
            f"{_response_style_guidance(channel_label)} "
            "Do not claim a capability is unavailable when the corresponding tool is exposed in this turn. "
            f"The tools exposed in this turn are: {available_tools_text}."
            f"{pending_text}"
        ),
    )


__all__ = ["build_tool_text_channel_turn_instructions"]
