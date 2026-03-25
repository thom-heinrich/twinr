"""Public baseline personality profiles shared by runtime and evaluation code."""

from __future__ import annotations

from twinr.agent.personality.models import ConversationStyleProfile, HumorProfile


def default_humor_profile() -> HumorProfile:
    """Return the baseline humor stance used before any learning lands."""

    return HumorProfile(
        style="gentle observational humor",
        summary="Use gentle wit sparingly and only when the moment is relaxed.",
        intensity=0.25,
        boundaries=(
            "never mocking",
            "never undercutting serious moments",
            "never sounding flippant about health or distress",
        ),
    )


def default_style_profile() -> ConversationStyleProfile:
    """Return the baseline conversational style before any learning lands."""

    return ConversationStyleProfile(
        verbosity=0.5,
        initiative=0.45,
    )
