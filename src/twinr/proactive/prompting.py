from __future__ import annotations

from twinr.proactive.engine import SocialTriggerDecision

_SAFETY_TRIGGER_IDS = frozenset(
    {
        "possible_fall",
        "floor_stillness",
        "distress_possible",
    }
)


def is_safety_trigger(trigger_id: str) -> bool:
    return trigger_id.strip().lower() in _SAFETY_TRIGGER_IDS


def proactive_prompt_mode(trigger: SocialTriggerDecision) -> str:
    return "direct_safety" if is_safety_trigger(trigger.trigger_id) else "llm"


def proactive_observation_facts(
    trigger: SocialTriggerDecision,
    *,
    max_items: int = 5,
) -> tuple[str, ...]:
    facts: list[str] = []
    for item in trigger.evidence[:max_items]:
        detail = item.detail.strip() or "[none]"
        facts.append(
            f"{item.key}: value={item.value:.2f}, weight={item.weight:.2f}, detail={detail}"
        )
    return tuple(facts)


__all__ = [
    "is_safety_trigger",
    "proactive_observation_facts",
    "proactive_prompt_mode",
]
