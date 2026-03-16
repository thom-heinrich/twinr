"""Manage the deterministic Phase-2 requirements dialogue state machine."""

from __future__ import annotations

from dataclasses import replace
from typing import Any
from uuid import uuid4

from .contracts import FeasibilityResult, RequirementsDialogueSession, SkillSpec
from .status import RequirementsDialogueStatus

_QUESTION_ORDER: tuple[str, ...] = ("when", "what", "how")


class SelfCodingRequirementsDialogue:
    """Advance requirements-gathering sessions without runtime side effects."""

    def create_session(
        self,
        *,
        draft_spec: SkillSpec,
        feasibility: FeasibilityResult,
        request_summary: str,
    ) -> RequirementsDialogueSession:
        """Create a new questioning session from the current draft skill spec."""

        return RequirementsDialogueSession(
            session_id=f"dialogue_{uuid4().hex}",
            request_summary=request_summary,
            skill_name=draft_spec.name,
            action=draft_spec.action,
            capabilities=draft_spec.capabilities,
            feasibility=feasibility,
            skill_id=draft_spec.skill_id,
            status=RequirementsDialogueStatus.QUESTIONING,
            trigger_mode=draft_spec.trigger.mode,
            trigger_conditions=draft_spec.trigger.conditions,
            scope=draft_spec.scope,
            constraints=draft_spec.constraints,
            current_question_id=_QUESTION_ORDER[0],
            answered_question_ids=(),
            answer_summaries={},
            created_at=draft_spec.created_at,
            updated_at=draft_spec.created_at,
            version=draft_spec.version,
        )

    def answer(self, session: RequirementsDialogueSession, response: dict[str, Any]) -> RequirementsDialogueSession:
        """Apply one structured answer to the current session state."""

        if session.status == RequirementsDialogueStatus.QUESTIONING:
            return self._answer_question(session, response)
        if session.status == RequirementsDialogueStatus.CONFIRMING:
            return self._answer_confirmation(session, response)
        raise ValueError("Only active requirements-dialogue sessions can accept answers")

    def prompt_for(self, session: RequirementsDialogueSession) -> str | None:
        """Return the current user-facing prompt for one active session."""

        if session.status == RequirementsDialogueStatus.QUESTIONING:
            assert session.current_question_id is not None
            if session.current_question_id == "when":
                return "Should I do that automatically, or only when you ask me?"
            if session.current_question_id == "what":
                return "Should I do this for everything, or only in certain cases?"
            if session.current_question_id == "how":
                return "How should I do it: just do it, tell you first, or handle it another way?"
            raise ValueError(f"Unsupported question id: {session.current_question_id}")
        if session.status == RequirementsDialogueStatus.CONFIRMING:
            return self.confirmation_prompt(session)
        return None

    def confirmation_prompt(self, session: RequirementsDialogueSession) -> str:
        """Build the deterministic mirror-back confirmation prompt."""

        mode_text = "automatically" if session.trigger_mode == "push" else "only when you ask"
        scope_text = _scope_summary(session.scope)
        constraint_text = _constraint_summary(session.constraints)
        details = f"I should {session.action} {mode_text}".strip()
        if scope_text:
            details = f"{details}, {scope_text}"
        if constraint_text:
            details = f"{details}, {constraint_text}"
        return f"Just to make sure: {details}. Is that right?"

    def remaining_questions(self, session: RequirementsDialogueSession) -> int:
        """Return how many of the three core questions are still open."""

        return max(0, len(_QUESTION_ORDER) - len(session.answered_question_ids))

    def _answer_question(self, session: RequirementsDialogueSession, response: dict[str, Any]) -> RequirementsDialogueSession:
        question_id = session.current_question_id
        if question_id not in _QUESTION_ORDER:
            raise ValueError("Questioning session is missing a valid current_question_id")

        use_default = _coerce_optional_bool(response.get("use_default"))
        answer_summary = _normalize_answer_summary(response.get("answer_summary"))
        updated = session
        if question_id == "when":
            updated = self._apply_when_answer(updated, response, use_default=use_default)
        elif question_id == "what":
            updated = self._apply_what_answer(updated, response, use_default=use_default)
        elif question_id == "how":
            updated = self._apply_how_answer(updated, response, use_default=use_default)

        answered_question_ids = tuple(dict.fromkeys((*updated.answered_question_ids, question_id)))
        answer_summaries = dict(updated.answer_summaries)
        if answer_summary is not None:
            answer_summaries[question_id] = answer_summary
        next_question_id = _next_question_id(answered_question_ids)
        next_status = (
            RequirementsDialogueStatus.CONFIRMING
            if next_question_id is None
            else RequirementsDialogueStatus.QUESTIONING
        )
        return replace(
            updated,
            answered_question_ids=answered_question_ids,
            answer_summaries=answer_summaries,
            status=next_status,
            current_question_id=next_question_id,
        )

    def _answer_confirmation(
        self,
        session: RequirementsDialogueSession,
        response: dict[str, Any],
    ) -> RequirementsDialogueSession:
        confirmed = _coerce_optional_bool(response.get("confirmed"))
        if confirmed is None:
            raise ValueError("The confirmation step requires `confirmed` to be true or false")
        answer_summary = _normalize_answer_summary(response.get("answer_summary"))
        answer_summaries = dict(session.answer_summaries)
        if answer_summary is not None:
            answer_summaries["confirm"] = answer_summary
        if confirmed:
            return replace(
                session,
                status=RequirementsDialogueStatus.READY_FOR_COMPILE,
                current_question_id=None,
                answer_summaries=answer_summaries,
            )
        return replace(
            session,
            status=RequirementsDialogueStatus.QUESTIONING,
            current_question_id=_QUESTION_ORDER[0],
            answered_question_ids=(),
            answer_summaries=answer_summaries,
        )

    def _apply_when_answer(
        self,
        session: RequirementsDialogueSession,
        response: dict[str, Any],
        *,
        use_default: bool | None,
    ) -> RequirementsDialogueSession:
        if use_default:
            return replace(session, trigger_mode="push")
        trigger_mode = _normalize_trigger_mode(response.get("trigger_mode")) or session.trigger_mode
        conditions = _merge_identifiers(session.trigger_conditions, response.get("trigger_conditions"))
        if trigger_mode == "pull" and not conditions:
            conditions = ("on_request",)
        return replace(session, trigger_mode=trigger_mode, trigger_conditions=conditions)

    def _apply_what_answer(
        self,
        session: RequirementsDialogueSession,
        response: dict[str, Any],
        *,
        use_default: bool | None,
    ) -> RequirementsDialogueSession:
        if use_default:
            scope = dict(session.scope)
            scope.setdefault("selection", "all")
            return replace(session, scope=scope)
        merged_scope = dict(session.scope)
        incoming_scope = response.get("scope")
        if incoming_scope is not None:
            if not isinstance(incoming_scope, dict):
                raise ValueError("scope must be a JSON object")
            merged_scope.update(dict(incoming_scope))
        constraints = _merge_texts(session.constraints, response.get("constraints"))
        trigger_conditions = _merge_identifiers(session.trigger_conditions, response.get("trigger_conditions"))
        return replace(
            session,
            scope=merged_scope,
            constraints=constraints,
            trigger_conditions=trigger_conditions,
        )

    def _apply_how_answer(
        self,
        session: RequirementsDialogueSession,
        response: dict[str, Any],
        *,
        use_default: bool | None,
    ) -> RequirementsDialogueSession:
        if use_default:
            return session
        action = str(response.get("action") or "").strip() or session.action
        constraints = _merge_texts(session.constraints, response.get("constraints"))
        scope = dict(session.scope)
        incoming_scope = response.get("scope")
        if incoming_scope is not None:
            if not isinstance(incoming_scope, dict):
                raise ValueError("scope must be a JSON object")
            scope.update(dict(incoming_scope))
        return replace(session, action=action, constraints=constraints, scope=scope)


def _next_question_id(answered_question_ids: tuple[str, ...]) -> str | None:
    for question_id in _QUESTION_ORDER:
        if question_id not in answered_question_ids:
            return question_id
    return None


def _merge_identifiers(existing: tuple[str, ...], raw_value: object) -> tuple[str, ...]:
    if raw_value is None:
        return existing
    if isinstance(raw_value, (str, bytes, bytearray)):
        raw_items = (raw_value,)
    elif isinstance(raw_value, (list, tuple)):
        raw_items = raw_value
    else:
        raise ValueError("trigger_conditions must be a list of identifiers")
    merged: list[str] = []
    seen: set[str] = set()
    for raw_item in (*existing, *raw_items):
        item = str(raw_item or "").strip().lower()
        if not item or item in seen:
            continue
        seen.add(item)
        merged.append(item)
    return tuple(merged)


def _merge_texts(existing: tuple[str, ...], raw_value: object) -> tuple[str, ...]:
    if raw_value is None:
        return existing
    if isinstance(raw_value, (str, bytes, bytearray)):
        raw_items = (raw_value,)
    elif isinstance(raw_value, (list, tuple)):
        raw_items = raw_value
    else:
        raise ValueError("constraints must be a list of short text values")
    merged: list[str] = []
    seen: set[str] = set()
    for raw_item in (*existing, *raw_items):
        item = " ".join(str(raw_item or "").split())
        if not item or item in seen:
            continue
        seen.add(item)
        merged.append(item)
    return tuple(merged)


def _normalize_trigger_mode(raw_value: object) -> str | None:
    text = str(raw_value or "").strip().lower()
    if not text:
        return None
    if text not in {"push", "pull"}:
        raise ValueError("trigger_mode must be `push` or `pull`")
    return text


def _coerce_optional_bool(raw_value: object) -> bool | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, int) and raw_value in (0, 1):
        return bool(raw_value)
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError("Boolean fields must be true or false")


def _normalize_answer_summary(raw_value: object) -> str | None:
    text = " ".join(str(raw_value or "").split())
    return text or None


def _scope_summary(scope: dict[str, Any]) -> str:
    if not scope:
        return ""
    parts: list[str] = []
    for key in sorted(scope):
        value = scope[key]
        if isinstance(value, list):
            render = ", ".join(str(item) for item in value if str(item).strip())
        else:
            render = str(value)
        render = " ".join(render.split()).strip()
        if not render:
            continue
        parts.append(f"{key.replace('_', ' ')}: {render}")
    if not parts:
        return ""
    return "with " + "; ".join(parts)


def _constraint_summary(constraints: tuple[str, ...]) -> str:
    if not constraints:
        return ""
    return "subject to " + "; ".join(constraints)
