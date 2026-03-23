"""Turn user-discovery invites into reserve-lane candidate cards."""

from __future__ import annotations

from datetime import datetime

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate
from twinr.memory.user_discovery import UserDiscoveryService


def _compact_text(value: object | None, *, max_len: int) -> str:
    compact = " ".join(str(value or "").split()).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "..."


def load_display_reserve_user_discovery_candidates(
    config: TwinrConfig,
    *,
    local_now: datetime,
    max_items: int,
) -> tuple[AmbientDisplayImpulseCandidate, ...]:
    """Expose at most one due get-to-know-you invitation candidate."""

    del max_items
    invite = UserDiscoveryService.from_config(config).build_invitation(now=local_now)
    if invite is None:
        return ()
    topic_key = f"user_discovery:{invite.phase}:{invite.topic_id}"
    return (
        AmbientDisplayImpulseCandidate(
            topic_key=topic_key,
            title=invite.display_topic_label,
            source="user_discovery",
            action="ask_one",
            attention_state="forming" if invite.phase == "initial_setup" else "growing",
            salience=float(invite.salience),
            eyebrow="",
            headline=_compact_text(invite.headline, max_len=112),
            body=_compact_text(invite.body, max_len=112),
            symbol="question",
            accent="warm",
            reason=_compact_text(invite.reason, max_len=120),
            candidate_family="user_discovery",
            generation_context={
                "candidate_family": "user_discovery",
                "display_goal": "invite_user_discovery",
                "invite_kind": invite.invite_kind,
                "phase": invite.phase,
                "topic_id": invite.topic_id,
                "topic_label": invite.topic_label,
                "display_label": invite.display_topic_label,
                "session_minutes": invite.session_minutes,
                "display_anchor": invite.display_topic_label,
                "hook_hint": invite.body,
            },
        ),
    )


__all__ = ["load_display_reserve_user_discovery_candidates"]
