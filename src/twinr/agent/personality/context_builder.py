"""Build ordered prompt layers from legacy sections and typed personality state.

The builder keeps authoritative behavior guidance inside ``SYSTEM`` and
``PERSONALITY`` while rendering evolving user/place/world context as explicit
context-data sections. A separate ``MINDSHARE`` section lets Twinr speak more
naturally about what it has been following without promoting those topics to
instruction authority.
"""

from __future__ import annotations

from dataclasses import dataclass

from twinr.agent.personality.intelligence.models import WorldInterestSignal
from twinr.agent.personality.models import (
    ConversationStyleProfile,
    HumorProfile,
    PersonalityPromptLayer,
    PersonalityPromptPlan,
    PersonalitySnapshot,
    PersonalityTrait,
)
from twinr.agent.personality.positive_engagement import render_positive_engagement_policy
from twinr.agent.personality.self_expression import (
    render_mindshare_block,
    render_self_expression_policy,
)
from twinr.agent.personality.steering import render_turn_steering_policy

_LEGACY_PRIORITY_ORDER = ("SYSTEM", "PERSONALITY", "USER")


def _merge_blocks(*blocks: str | None) -> str | None:
    """Join non-empty text blocks with stable blank-line separation."""

    normalized = [str(block).strip() for block in blocks if block and str(block).strip()]
    if not normalized:
        return None
    return "\n\n".join(normalized)


def _format_score(value: float) -> str:
    """Render one salience/confidence-like value for compact prompt display."""

    return f"{value:.2f}"


def _render_trait_lines(traits: tuple[PersonalityTrait, ...]) -> str | None:
    """Render stable character traits for the authoritative personality layer."""

    if not traits:
        return None
    lines = ["## Structured core character"]
    for trait in traits:
        stability = "core" if trait.stable else "evolving"
        lines.append(
            f"- {trait.name}: {trait.summary} (weight {_format_score(trait.weight)}, {stability})"
        )
    return "\n".join(lines)


def _render_humor_block(humor_profile: HumorProfile | None) -> str | None:
    """Render the humor overlay that can safely live inside ``PERSONALITY``."""

    if humor_profile is None:
        return None
    lines = [
        "## Evolving humor stance",
        f"- style: {humor_profile.style}",
        f"- summary: {humor_profile.summary}",
        f"- intensity: {_format_score(humor_profile.intensity)}",
    ]
    if humor_profile.boundaries:
        lines.append(f"- boundaries: {', '.join(humor_profile.boundaries)}")
    return "\n".join(lines)


def _style_band(value: float, *, low: str, medium: str, high: str) -> str:
    """Map one bounded style score onto a compact prompt-facing label."""

    if value < 0.4:
        return low
    if value >= 0.55:
        return high
    return medium


def _render_style_block(style_profile: ConversationStyleProfile | None) -> str | None:
    """Render learned verbosity and initiative into ``PERSONALITY``."""

    if style_profile is None:
        return None
    lines = [
        "## Evolving conversation style",
        (
            f"- verbosity: {_style_band(style_profile.verbosity, low='concise', medium='balanced', high='detailed')} "
            f"({_format_score(style_profile.verbosity)})"
        ),
        (
            f"- initiative: {_style_band(style_profile.initiative, low='mostly reactive', medium='balanced', high='gently proactive')} "
            f"({_format_score(style_profile.initiative)})"
        ),
    ]
    return "\n".join(lines)


def _render_relationship_block(snapshot: PersonalitySnapshot | None) -> str | None:
    """Render learned relationship context into the ``USER`` layer."""

    if snapshot is None or not snapshot.relationship_signals:
        return None
    lines = ["## Learned relationship context"]
    for signal in snapshot.relationship_signals:
        stance_label = signal.stance
        if signal.stance == "aversion":
            stance_label = "aversion"
        lines.append(
            f"- {signal.topic}: {signal.summary} (salience {_format_score(signal.salience)}, stance {stance_label}, source {signal.source})"
        )
    return "\n".join(lines)


def _render_continuity_block(snapshot: PersonalitySnapshot | None) -> str | None:
    """Render active continuity threads into a contextual layer."""

    if snapshot is None or not snapshot.continuity_threads:
        return None
    lines = ["## Active continuity threads"]
    for thread in snapshot.continuity_threads:
        detail = f"- {thread.title}: {thread.summary} (salience {_format_score(thread.salience)})"
        if thread.updated_at:
            detail += f"; updated {thread.updated_at}"
        if thread.expires_at:
            detail += f"; review by {thread.expires_at}"
        lines.append(detail)
    return "\n".join(lines)


def _render_place_block(snapshot: PersonalitySnapshot | None) -> str | None:
    """Render place-awareness context into a contextual layer."""

    if snapshot is None or not snapshot.place_focuses:
        return None
    lines = ["## Place awareness"]
    for focus in snapshot.place_focuses:
        detail = f"- {focus.name}: {focus.summary} (salience {_format_score(focus.salience)})"
        if focus.geography:
            detail += f"; geography {focus.geography}"
        if focus.updated_at:
            detail += f"; updated {focus.updated_at}"
        lines.append(detail)
    return "\n".join(lines)


def _render_world_block(snapshot: PersonalitySnapshot | None) -> str | None:
    """Render world-awareness context into a contextual layer."""

    if snapshot is None or not snapshot.world_signals:
        return None
    lines = ["## World awareness"]
    for signal in snapshot.world_signals:
        detail = f"- {signal.topic}: {signal.summary} (salience {_format_score(signal.salience)}, source {signal.source})"
        if signal.region:
            detail += f"; region {signal.region}"
        if signal.fresh_until:
            detail += f"; fresh until {signal.fresh_until}"
        lines.append(detail)
    return "\n".join(lines)


def _render_reflection_block(snapshot: PersonalitySnapshot | None) -> str | None:
    """Render small reflection deltas into a contextual layer."""

    if snapshot is None or not snapshot.reflection_deltas:
        return None
    lines = ["## Reflection deltas"]
    for delta in snapshot.reflection_deltas:
        detail = (
            f"- {delta.target}: {delta.change} because {delta.reason} "
            f"(confidence {_format_score(delta.confidence)})"
        )
        if delta.review_at:
            detail += f"; review {delta.review_at}"
        lines.append(detail)
    return "\n".join(lines)


@dataclass(slots=True)
class PersonalityContextBuilder:
    """Build ordered prompt layers from legacy personality files and typed state."""

    @staticmethod
    def _normalized_legacy_sections(
        legacy_sections: tuple[tuple[str, str], ...],
    ) -> dict[str, str]:
        """Normalize legacy section tuples into an uppercase-keyed mapping."""

        return {
            str(title).strip().upper(): str(content).strip()
            for title, content in legacy_sections
            if str(title).strip() and str(content).strip()
        }

    def build_prompt_plan(
        self,
        *,
        legacy_sections: tuple[tuple[str, str], ...],
        snapshot: PersonalitySnapshot | None,
        engagement_signals: tuple[WorldInterestSignal, ...] = (),
    ) -> PersonalityPromptPlan:
        """Merge legacy prompt sections with structured personality layers.

        Args:
            legacy_sections: Existing ordered ``(title, content)`` pairs loaded
                from the trusted personality directory.
            snapshot: Optional structured personality snapshot loaded from a
                remote or future ChonkyDB-backed store.

        Returns:
            A prompt plan that preserves the legacy section contract while
            layering in structured personality state when available.
        """

        normalized_legacy = self._normalized_legacy_sections(legacy_sections)
        layers: list[PersonalityPromptLayer] = []

        system_content = normalized_legacy.pop("SYSTEM", None)
        if system_content:
            layers.append(
                PersonalityPromptLayer(
                    layer_id="system",
                    title="SYSTEM",
                    content=system_content,
                    source="legacy_file",
                    instruction_authority=True,
                )
            )

        personality_content = _merge_blocks(
            normalized_legacy.pop("PERSONALITY", None),
            _render_trait_lines(snapshot.core_traits) if snapshot else None,
            _render_style_block(snapshot.style_profile) if snapshot else None,
            _render_humor_block(snapshot.humor_profile) if snapshot else None,
            render_turn_steering_policy(
                snapshot,
                engagement_signals=engagement_signals,
            ),
            render_positive_engagement_policy(
                snapshot,
                engagement_signals=engagement_signals,
            ),
            render_self_expression_policy(
                snapshot,
                engagement_signals=engagement_signals,
            ),
        )
        if personality_content:
            layers.append(
                PersonalityPromptLayer(
                    layer_id="personality",
                    title="PERSONALITY",
                    content=personality_content,
                    source="legacy_plus_structured",
                    instruction_authority=True,
                )
            )

        user_content = _merge_blocks(
            normalized_legacy.pop("USER", None),
            _render_relationship_block(snapshot),
        )
        if user_content:
            layers.append(
                PersonalityPromptLayer(
                    layer_id="user",
                    title="USER",
                    content=user_content,
                    source="legacy_plus_structured",
                )
            )

        mindshare_content = render_mindshare_block(
            snapshot,
            engagement_signals=engagement_signals,
        )
        if mindshare_content:
            layers.append(
                PersonalityPromptLayer(
                    layer_id="mindshare",
                    title="MINDSHARE",
                    content=mindshare_content,
                    source="structured_snapshot",
                )
            )

        for title in _LEGACY_PRIORITY_ORDER:
            normalized_legacy.pop(title, None)
        for title, content in legacy_sections:
            normalized_title = str(title).strip().upper()
            normalized_content = str(content).strip()
            if (
                normalized_title in _LEGACY_PRIORITY_ORDER
                or not normalized_title
                or not normalized_content
            ):
                continue
            layers.append(
                PersonalityPromptLayer(
                    layer_id=f"legacy_{normalized_title.lower()}",
                    title=normalized_title,
                    content=normalized_content,
                    source="legacy_file",
                )
            )

        for layer_id, title, content in (
            ("continuity", "CONTINUITY", _render_continuity_block(snapshot)),
            ("place", "PLACE", _render_place_block(snapshot)),
            ("world", "WORLD", _render_world_block(snapshot)),
            ("reflection", "REFLECTION", _render_reflection_block(snapshot)),
        ):
            if not content:
                continue
            layers.append(
                PersonalityPromptLayer(
                    layer_id=layer_id,
                    title=title,
                    content=content,
                    source="structured_snapshot",
                )
            )

        return PersonalityPromptPlan(layers=tuple(layers))

    def build_supervisor_prompt_plan(
        self,
        *,
        legacy_sections: tuple[tuple[str, str], ...],
        snapshot: PersonalitySnapshot | None,
    ) -> PersonalityPromptPlan:
        """Build a lean supervisor prompt plan without dynamic topic layers.

        The fast supervisor decides whether a turn should route direct or into a
        slower specialist/search handoff. That lane still needs stable Twinr
        character and legacy user context, but it must not receive volatile
        dynamic layers such as `MINDSHARE`, `CONTINUITY`, `PLACE`, `WORLD`, or
        `REFLECTION`, because those can semantically bias noisy search routing.
        """

        normalized_legacy = self._normalized_legacy_sections(legacy_sections)
        layers: list[PersonalityPromptLayer] = []

        system_content = normalized_legacy.pop("SYSTEM", None)
        if system_content:
            layers.append(
                PersonalityPromptLayer(
                    layer_id="system",
                    title="SYSTEM",
                    content=system_content,
                    source="legacy_file",
                    instruction_authority=True,
                )
            )

        personality_content = _merge_blocks(
            normalized_legacy.pop("PERSONALITY", None),
            _render_trait_lines(snapshot.core_traits) if snapshot else None,
            _render_style_block(snapshot.style_profile) if snapshot else None,
            _render_humor_block(snapshot.humor_profile) if snapshot else None,
        )
        if personality_content:
            layers.append(
                PersonalityPromptLayer(
                    layer_id="personality",
                    title="PERSONALITY",
                    content=personality_content,
                    source="legacy_plus_structured",
                    instruction_authority=True,
                )
            )

        user_content = normalized_legacy.pop("USER", None)
        if user_content:
            layers.append(
                PersonalityPromptLayer(
                    layer_id="user",
                    title="USER",
                    content=user_content,
                    source="legacy_file",
                )
            )

        return PersonalityPromptPlan(layers=tuple(layers))
