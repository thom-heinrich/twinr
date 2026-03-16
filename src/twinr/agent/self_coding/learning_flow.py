"""Coordinate feasibility checks and requirements-dialogue persistence."""

from __future__ import annotations

import logging  # AUDIT-FIX(#5): Add structured diagnostics for controlled error handling.
import re  # AUDIT-FIX(#3): Validate untrusted session identifiers before they reach the file-backed store.
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from threading import RLock  # AUDIT-FIX(#2): Serialize in-process mutations for file-backed state.
from typing import Any

from .contracts import CompileJobRecord, FeasibilityResult, RequirementsDialogueSession, SkillSpec
from .feasibility import SelfCodingFeasibilityChecker
from .requirements_dialogue import SelfCodingRequirementsDialogue
from .status import FeasibilityOutcome, RequirementsDialogueStatus
from .store import SelfCodingStore
from .worker import SelfCodingCompileWorker

logger = logging.getLogger(__name__)  # AUDIT-FIX(#5): Emit consistent diagnostics without leaking internals to callers.
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,254}$")  # AUDIT-FIX(#3): Conservative allow-list blocks traversal tokens.
_MAX_FEASIBILITY_REASON_CHARS = 240  # AUDIT-FIX(#8): Keep senior-facing explanations short and stable.


class SelfCodingLearningFlowError(RuntimeError):
    """Raised when the learning flow cannot safely complete."""


@dataclass(frozen=True, slots=True)
class SelfCodingLearningUpdate:
    """Return the current front-stage state of a self-coding learning flow."""

    phase: str
    feasibility: FeasibilityResult
    recommended_reply: str
    session: RequirementsDialogueSession | None = None
    prompt: str | None = None
    skill_spec: SkillSpec | None = None
    compile_job: CompileJobRecord | None = None


class SelfCodingLearningFlow:
    """Persist and advance the self-coding front-stage learning flow."""

    def __init__(
        self,
        store: SelfCodingStore,
        checker: SelfCodingFeasibilityChecker,
        dialogue: SelfCodingRequirementsDialogue | None = None,
        compile_worker: SelfCodingCompileWorker | None = None,
    ) -> None:
        self.store = store
        self.checker = checker
        self.dialogue = (
            dialogue if dialogue is not None else SelfCodingRequirementsDialogue()
        )  # AUDIT-FIX(#9): Preserve valid falsey dialogue implementations.
        self.compile_worker = compile_worker
        self._flow_lock = RLock()  # AUDIT-FIX(#2): Prevent concurrent updates from clobbering file-backed state.

    def start_request(self, draft_spec: SkillSpec, *, request_summary: str) -> SelfCodingLearningUpdate:
        """Run the first feasibility check and create a requirements session when allowed."""

        request_summary = self._validated_request_summary(
            request_summary
        )  # AUDIT-FIX(#7): Validate caller-controlled input before passing it deeper.
        with self._flow_lock:  # AUDIT-FIX(#2): Serialize feasibility check, session creation, and persistence.
            try:  # AUDIT-FIX(#5): Convert lower-level crashes into a stable flow error for the API layer.
                feasibility = self.checker.check(draft_spec)
            except Exception as exc:  # pragma: no cover - defensive boundary
                logger.exception("Self-coding feasibility check failed while starting a learning flow.")
                raise SelfCodingLearningFlowError(
                    "Could not check whether that request can be learned safely."
                ) from exc

            if feasibility.outcome == FeasibilityOutcome.RED:
                return SelfCodingLearningUpdate(
                    phase="blocked",
                    feasibility=feasibility,
                    recommended_reply=self._blocked_reply(feasibility),
                )

            try:  # AUDIT-FIX(#5): Keep persistence failures consistent and caller-safe.
                session = self.dialogue.create_session(
                    draft_spec=draft_spec,
                    feasibility=feasibility,
                    request_summary=request_summary,
                )
                self.store.save_dialogue_session(session)
            except Exception as exc:  # pragma: no cover - defensive boundary
                logger.exception("Self-coding dialogue session creation or persistence failed.")
                raise SelfCodingLearningFlowError(
                    "Could not start the requirements dialogue safely."
                ) from exc

            return self._update_from_session(session)

    def answer_question(self, session_id: str, response: dict[str, Any]) -> SelfCodingLearningUpdate:
        """Load one session, apply the answer, and persist the updated state."""

        session_id = self._validated_session_id(
            session_id
        )  # AUDIT-FIX(#3): Reject unsafe session identifiers before they hit the store.
        response = self._validated_response(
            response
        )  # AUDIT-FIX(#7): Validate caller-controlled answer payloads at the boundary.
        with self._flow_lock:  # AUDIT-FIX(#2): Serialize load-answer-check-save to avoid lost updates.
            try:  # AUDIT-FIX(#5): Surface a stable exception instead of leaking backend-specific failures.
                session = self.store.load_dialogue_session(session_id)
            except Exception as exc:  # pragma: no cover - defensive boundary
                logger.exception("Failed to load self-coding dialogue session %s.", session_id)
                raise SelfCodingLearningFlowError(
                    "Could not load that requirements dialogue safely."
                ) from exc

            if session.status == RequirementsDialogueStatus.READY_FOR_COMPILE:
                return self._update_from_session(
                    session
                )  # AUDIT-FIX(#4): Treat repeated submissions on terminal sessions as idempotent no-ops.

            try:  # AUDIT-FIX(#5): Keep answer processing failures consistent and caller-safe.
                updated = self.dialogue.answer(session, response)
                updated = replace(
                    updated,
                    feasibility=self.checker.check(updated.to_skill_spec()),
                    updated_at=datetime.now(UTC),
                )
                self.store.save_dialogue_session(updated)
            except Exception as exc:  # pragma: no cover - defensive boundary
                logger.exception("Failed to apply an answer to self-coding dialogue session %s.", session_id)
                raise SelfCodingLearningFlowError(
                    "Could not save that answer safely."
                ) from exc

            return self._update_from_session(updated)

    def _update_from_session(self, session: RequirementsDialogueSession) -> SelfCodingLearningUpdate:
        # AUDIT-FIX(#1): A re-checked RED feasibility outcome must fail closed before more questions or compilation.
        if session.feasibility.outcome == FeasibilityOutcome.RED:
            return SelfCodingLearningUpdate(
                phase="blocked",
                feasibility=session.feasibility,
                recommended_reply=self._blocked_reply(session.feasibility),
                session=session,
                prompt=None,
            )

        if session.status == RequirementsDialogueStatus.READY_FOR_COMPILE:
            try:  # AUDIT-FIX(#5): Surface skill-spec reconstruction failures consistently.
                skill_spec = session.to_skill_spec()
            except Exception as exc:  # pragma: no cover - defensive boundary
                logger.exception(
                    "Failed to rebuild a skill spec for self-coding dialogue session %s.",
                    self._session_log_id(session),
                )
                raise SelfCodingLearningFlowError(
                    "Could not prepare the saved requirements for learning."
                ) from exc

            compile_job = None
            if self.compile_worker is None:
                recommended_reply = (
                    "I have what I need. The learning job is ready to start."
                )  # AUDIT-FIX(#6): Do not claim learning already started when no compile worker is configured.
            else:
                try:
                    compile_job = self.compile_worker.ensure_job_for_session(session)
                except Exception:  # pragma: no cover - defensive boundary
                    logger.exception(
                        "Failed to create a compile job for self-coding dialogue session %s.",
                        self._session_log_id(session),
                    )
                    recommended_reply = (
                        "I have what I need, but I could not start the learning job just yet. Please try again."
                    )  # AUDIT-FIX(#6): Fail gracefully when compile job creation breaks after persistence.
                else:
                    recommended_reply = "Alright, I have what I need to start learning that."

            return SelfCodingLearningUpdate(
                phase="ready_for_compile",
                feasibility=session.feasibility,
                recommended_reply=recommended_reply,
                session=session,
                prompt=None,
                skill_spec=skill_spec,
                compile_job=compile_job,
            )

        try:  # AUDIT-FIX(#5): Surface prompt generation failures consistently.
            prompt = self.dialogue.prompt_for(session)
        except Exception as exc:  # pragma: no cover - defensive boundary
            logger.exception(
                "Failed to build the next prompt for self-coding dialogue session %s.",
                self._session_log_id(session),
            )
            raise SelfCodingLearningFlowError(
                "Could not prepare the next requirements question."
            ) from exc

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
    def _validated_request_summary(request_summary: str) -> str:
        # AUDIT-FIX(#7): Enforce the expected boundary type before persisting request metadata.
        if not isinstance(request_summary, str):
            raise SelfCodingLearningFlowError("The request summary is invalid.")
        return request_summary.strip()

    @staticmethod
    def _validated_session_id(session_id: str) -> str:
        # AUDIT-FIX(#3): Restrict session identifiers to a safe character set for file-backed stores.
        if not isinstance(session_id, str):
            raise SelfCodingLearningFlowError("The session identifier is invalid.")
        normalized = session_id.strip()
        if not normalized or _SESSION_ID_RE.fullmatch(normalized) is None:
            raise SelfCodingLearningFlowError("The session identifier is invalid.")
        return normalized

    @staticmethod
    def _validated_response(response: dict[str, Any]) -> dict[str, Any]:
        # AUDIT-FIX(#7): Reject malformed or non-JSON-object answer payloads before dialogue processing.
        if not isinstance(response, dict):
            raise SelfCodingLearningFlowError("The answer payload is invalid.")
        normalized: dict[str, Any] = {}
        for key, value in response.items():
            if not isinstance(key, str) or not key:
                raise SelfCodingLearningFlowError("The answer payload is invalid.")
            normalized[key] = value
        return normalized

    @staticmethod
    def _session_log_id(session: RequirementsDialogueSession) -> str:
        # AUDIT-FIX(#5): Avoid secondary logging crashes when upstream objects are malformed.
        session_id = getattr(session, "session_id", None)
        return session_id if isinstance(session_id, str) and session_id else "<unknown>"

    @staticmethod
    def _blocked_reply(feasibility: FeasibilityResult) -> str:
        reason = SelfCodingLearningFlow._primary_reason(
            feasibility
        )  # AUDIT-FIX(#8): Normalize raw feasibility reasons before exposing them to end users.
        if reason:
            return f"I cannot learn that yet. {reason}"
        return "I cannot learn that yet because a required capability is missing."

    @staticmethod
    def _primary_reason(feasibility: FeasibilityResult) -> str | None:
        # AUDIT-FIX(#8): Collapse multiline/internal text into a short, stable user-facing explanation.
        for reason in feasibility.reasons:
            normalized = " ".join(str(reason).split())
            if normalized:
                if len(normalized) <= _MAX_FEASIBILITY_REASON_CHARS:
                    return normalized
                return normalized[: _MAX_FEASIBILITY_REASON_CHARS - 1].rstrip() + "…"
        return None

    @staticmethod
    def _questioning_reply(feasibility: FeasibilityResult) -> str:
        if feasibility.outcome == FeasibilityOutcome.RED:
            return SelfCodingLearningFlow._blocked_reply(
                feasibility
            )  # AUDIT-FIX(#1): Never emit a positive learning message for a blocked request.
        if feasibility.outcome == FeasibilityOutcome.YELLOW:
            return "I think I can learn that. I need two or three short questions first."
        return "I can learn that. I need two or three short questions first."