"""Coordinate feasibility checks and requirements-dialogue persistence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

from .contracts import FeasibilityResult, RequirementsDialogueSession, SkillSpec
from .feasibility import SelfCodingFeasibilityChecker
from .requirements_dialogue import SelfCodingRequirementsDialogue
from .status import FeasibilityOutcome, RequirementsDialogueStatus
from .store import SelfCodingStore


@dataclass(frozen=True, slots=True)
class SelfCodingLearningUpdate:
    """Return the current front-stage state of a self-coding learning flow."""

    phase: str
    feasibility: FeasibilityResult
    recommended_reply: str
    session: RequirementsDialogueSession | None = None
    prompt: str | None = None
    skill_spec: SkillSpec | None = None


class SelfCodingLearningFlow:
    """Persist and advance the self-coding front-stage learning flow."""

    def __init__(
        self,
        store: SelfCodingStore,
        checker: SelfCodingFeasibilityChecker,
        dialogue: SelfCodingRequirementsDialogue | None = None,
    ) -> None:
        self.store = store
        self.checker = checker
        self.dialogue = dialogue or SelfCodingRequirementsDialogue()

    def start_request(self, draft_spec: SkillSpec, *, request_summary: str) -> SelfCodingLearningUpdate:
        """Run the first feasibility check and create a requirements session when allowed."""

        feasibility = self.checker.check(draft_spec)
        if feasibility.outcome == FeasibilityOutcome.RED:
            return SelfCodingLearningUpdate(
                phase="blocked",
                feasibility=feasibility,
                recommended_reply=self._blocked_reply(feasibility),
            )
        session = self.dialogue.create_session(
            draft_spec=draft_spec,
            feasibility=feasibility,
            request_summary=request_summary,
        )
        self.store.save_dialogue_session(session)
        return self._update_from_session(session)

    def answer_question(self, session_id: str, response: dict[str, Any]) -> SelfCodingLearningUpdate:
        """Load one session, apply the answer, and persist the updated state."""

        session = self.store.load_dialogue_session(session_id)
        updated = self.dialogue.answer(session, response)
        updated = replace(
            updated,
            feasibility=self.checker.check(updated.to_skill_spec()),
            updated_at=datetime.now(UTC),
        )
        self.store.save_dialogue_session(updated)
        return self._update_from_session(updated)

    def _update_from_session(self, session: RequirementsDialogueSession) -> SelfCodingLearningUpdate:
        if session.status == RequirementsDialogueStatus.READY_FOR_COMPILE:
            return SelfCodingLearningUpdate(
                phase="ready_for_compile",
                feasibility=session.feasibility,
                recommended_reply="Alright, I have what I need to start learning that.",
                session=session,
                prompt=None,
                skill_spec=session.to_skill_spec(),
            )

        prompt = self.dialogue.prompt_for(session)
        if session.status == RequirementsDialogueStatus.CONFIRMING:
            return SelfCodingLearningUpdate(
                phase="confirming",
                feasibility=session.feasibility,
                recommended_reply=prompt or "Just to make sure, let me repeat that back.",
                session=session,
                prompt=prompt,
            )

        return SelfCodingLearningUpdate(
            phase="questioning",
            feasibility=session.feasibility,
            recommended_reply=self._questioning_reply(session.feasibility),
            session=session,
            prompt=prompt,
        )

    @staticmethod
    def _blocked_reply(feasibility: FeasibilityResult) -> str:
        if feasibility.reasons:
            return f"I cannot learn that yet. {feasibility.reasons[0]}"
        return "I cannot learn that yet because a required capability is missing."

    @staticmethod
    def _questioning_reply(feasibility: FeasibilityResult) -> str:
        if feasibility.outcome == FeasibilityOutcome.YELLOW:
            return "I think I can learn that. I need two or three short questions first."
        return "I can learn that. I need two or three short questions first."
