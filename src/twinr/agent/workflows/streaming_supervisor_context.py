"""Build per-turn supervisor instructions for the streaming fast lane.

The streaming supervisor prompt has two layers:

- stable base routing instructions that can be cached across turns
- small per-turn overlays such as the currently visible right-lane card

Keeping this builder separate avoids growing the runtime loops with prompt
assembly logic while still letting the fast lane react to live display state.
"""

from __future__ import annotations

from twinr.agent.base_agent.prompting.personality import merge_instructions
from twinr.agent.base_agent.runtime.display_grounding import (
    build_active_display_grounding_instruction_overlay,
)
from twinr.agent.tools import build_supervisor_decision_instructions


def build_streaming_supervisor_turn_instructions(config) -> str:
    """Return the authoritative supervisor instructions for one live turn.

    The caller gets the normal fast-lane routing instructions plus any active
    display-grounding overlay for the current visible reserve-lane card.
    """

    return build_supervisor_decision_instructions(
        config,
        extra_instructions=merge_instructions(
            config.openai_realtime_instructions,
            build_active_display_grounding_instruction_overlay(config),
        ),
    )


__all__ = ["build_streaming_supervisor_turn_instructions"]
