"""Manage the deterministic Phase-2 requirements dialogue state machine."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from typing import Any
from uuid import uuid4

from .contracts import (
    FeasibilityResult,
    REQUIREMENTS_DIALOGUE_BASELINE_ACTION_KEY,
    REQUIREMENTS_DIALOGUE_BASELINE_CONSTRAINTS_KEY,
    REQUIREMENTS_DIALOGUE_BASELINE_SCOPE_KEY,
    REQUIREMENTS_DIALOGUE_BASELINE_TRIGGER_CONDITIONS_KEY,
    REQUIREMENTS_DIALOGUE_BASELINE_TRIGGER_MODE_KEY,
    REQUIREMENTS_DIALOGUE_INTERNAL_ANSWER_SUMMARY_KEYS,
    RequirementsDialogueSession,
    SkillSpec,
)
from .status import RequirementsDialogueStatus

_QUESTION_ORDER: tuple[str, ...] = ("when", "what", "how")
# AUDIT-FIX(#5): Constrain identifiers, scope keys, and free-text sizes at the boundary.
_IDENTIFIER_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_.-]{0,63})$")
_SCOPE_KEY_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$")
_MAX_NAME_LEN = 256
_MAX_ACTION_LEN = 280
_MAX_ANSWER_SUMMARY_LEN = 500
_MAX_REQUEST_SUMMARY_LEN = 1000
_MAX_TEXT_ITEM_LEN = 280
_MAX_TEXT_ITEM_COUNT = 32
_MAX_SCOPE_LIST_ITEMS = 32
# AUDIT-FIX(#1): Reuse the contract-owned rollback keys so validation and dialogue mutation stay in sync.
_INTERNAL_BASELINE_ACTION_KEY = REQUIREMENTS_DIALOGUE_BASELINE_ACTION_KEY
_INTERNAL_BASELINE_TRIGGER_MODE_KEY = REQUIREMENTS_DIALOGUE_BASELINE_TRIGGER_MODE_KEY
_INTERNAL_BASELINE_TRIGGER_CONDITIONS_KEY = REQUIREMENTS_DIALOGUE_BASELINE_TRIGGER_CONDITIONS_KEY
_INTERNAL_BASELINE_SCOPE_KEY = REQUIREMENTS_DIALOGUE_BASELINE_SCOPE_KEY
_INTERNAL_BASELINE_CONSTRAINTS_KEY = REQUIREMENTS_DIALOGUE_BASELINE_CONSTRAINTS_KEY
_INTERNAL_ANSWER_SUMMARY_KEYS: frozenset[str] = REQUIREMENTS_DIALOGUE_INTERNAL_ANSWER_SUMMARY_KEYS


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

        # AUDIT-FIX(#4): Stamp a fresh session timestamp instead of reusing the draft-spec timestamp.
        session_timestamp = _now_like(draft_spec.created_at)
        # AUDIT-FIX(#3): Normalize trigger invariants up front so pull-mode sessions have an explicit on-request condition.
        trigger_mode = _normalize_trigger_mode(draft_spec.trigger.mode) or "pull"
        trigger_conditions = _normalize_identifier_tuple(draft_spec.trigger.conditions)
        if trigger_mode == "pull" and not trigger_conditions:
            trigger_conditions = ("on_request",)
        return RequirementsDialogueSession(
            session_id=f"dialogue_{uuid4().hex}",
            # AUDIT-FIX(#5): Normalize user-facing and persisted text before it enters session state.
            request_summary=_normalize_free_text(
                request_summary,
                field_name="request_summary",
                max_length=_MAX_REQUEST_SUMMARY_LEN,
            ),
            skill_name=_normalize_free_text(draft_spec.name, field_name="skill_name", max_length=_MAX_NAME_LEN),
            action=_normalize_action_text(draft_spec.action),
            # AUDIT-FIX(#2): Clone mutable draft containers before storing them in the session.
            capabilities=_copy_sequence_like(draft_spec.capabilities),
            feasibility=feasibility,
            skill_id=draft_spec.skill_id,
            status=RequirementsDialogueStatus.QUESTIONING,
            trigger_mode=trigger_mode,
            trigger_conditions=trigger_conditions,
            scope=_normalize_scope_mapping(draft_spec.scope),
            constraints=_normalize_text_tuple(draft_spec.constraints),
            current_question_id=_QUESTION_ORDER[0],
            answered_question_ids=(),
            answer_summaries={},
            created_at=session_timestamp,
            updated_at=session_timestamp,
            version=draft_spec.version,
        )

    def answer(self, session: RequirementsDialogueSession, response: Mapping[str, Any]) -> RequirementsDialogueSession:
        """Apply one structured answer to the current session state."""

        # AUDIT-FIX(#5): Reject non-object responses before nested access can explode with AttributeError.
        normalized_response = _normalize_response(response)
        if session.status == RequirementsDialogueStatus.QUESTIONING:
            return self._answer_question(session, normalized_response)
        if session.status == RequirementsDialogueStatus.CONFIRMING:
            return self._answer_confirmation(session, normalized_response)
        raise ValueError("Only active requirements-dialogue sessions can accept answers")

    def prompt_for(self, session: RequirementsDialogueSession) -> str | None:
        """Return the current user-facing prompt for one active session."""

        if session.status == RequirementsDialogueStatus.QUESTIONING:
            # AUDIT-FIX(#7): Use an explicit runtime guard instead of assert-based validation.
            if session.current_question_id is None:
                raise ValueError("Questioning session is missing a valid current_question_id")
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

        # AUDIT-FIX(#8): Validate trigger mode and normalize action text before speaking or rendering it back to the user.
        mode_text = _trigger_mode_summary(session.trigger_mode)
        scope_text = _scope_summary(session.scope)
        constraint_text = _constraint_summary(session.constraints)
        details = f"I should {_prompt_action_text(session.action)} {mode_text}".strip()
        if scope_text:
            details = f"{details}, {scope_text}"
        if constraint_text:
            details = f"{details}, {constraint_text}"
        return f"Just to make sure: {details}. Is that right?"

    def remaining_questions(self, session: RequirementsDialogueSession) -> int:
        """Return how many of the three core questions are still open."""

        # AUDIT-FIX(#9): Ignore duplicated or unknown ids when computing remaining work from persisted state.
        answered = tuple(question_id for question_id in session.answered_question_ids if question_id in _QUESTION_ORDER)
        return max(0, len(_QUESTION_ORDER) - len(answered))

    def _answer_question(self, session: RequirementsDialogueSession, response: Mapping[str, Any]) -> RequirementsDialogueSession:
        question_id = session.current_question_id
        if question_id not in _QUESTION_ORDER:
            raise ValueError("Questioning session is missing a valid current_question_id")

        use_default = _coerce_optional_bool(response.get("use_default"))
        # AUDIT-FIX(#6): Fail closed when the extractor did not return a usable payload for the active question.
        _validate_question_payload(question_id, response, use_default=use_default)
        answer_summary = _normalize_answer_summary(response.get("answer_summary"))
        updated = session
        if question_id == "when":
            updated = self._apply_when_answer(updated, response, use_default=use_default)
        elif question_id == "what":
            updated = self._apply_what_answer(updated, response, use_default=use_default)
        elif question_id == "how":
            updated = self._apply_how_answer(updated, response, use_default=use_default)

        answered_question_ids = tuple(dict.fromkeys((*updated.answered_question_ids, question_id)))
        # AUDIT-FIX(#1): Snapshot the pre-dialogue baseline before mutating state so a rejected confirmation can truly roll back.
        answer_summaries = _ensure_restart_baseline(dict(updated.answer_summaries), session)
        if answer_summary is not None:
            answer_summaries[question_id] = answer_summary
        next_question_id = _next_question_id(answered_question_ids)
        next_status = (
            RequirementsDialogueStatus.CONFIRMING
            if next_question_id is None
            else RequirementsDialogueStatus.QUESTIONING
        )
        return _replace_session(
            updated,
            answered_question_ids=answered_question_ids,
            answer_summaries=answer_summaries,
            status=next_status,
            current_question_id=next_question_id,
        )

    def _answer_confirmation(
        self,
        session: RequirementsDialogueSession,
        response: Mapping[str, Any],
    ) -> RequirementsDialogueSession:
        confirmed = _coerce_optional_bool(response.get("confirmed"))
        if confirmed is None:
            raise ValueError("The confirmation step requires `confirmed` to be true or false")
        answer_summary = _normalize_answer_summary(response.get("answer_summary"))
        raw_answer_summaries = dict(session.answer_summaries)
        # AUDIT-FIX(#1): Restore the original pre-dialogue fields when the user rejects the confirmation summary.
        restart_baseline = _extract_restart_baseline(session, raw_answer_summaries)
        answer_summaries = _strip_internal_answer_summaries(raw_answer_summaries)
        if answer_summary is not None:
            answer_summaries["confirm"] = answer_summary
        if confirmed:
            return _replace_session(
                session,
                status=RequirementsDialogueStatus.READY_FOR_COMPILE,
                current_question_id=None,
                answer_summaries=answer_summaries,
            )
        return _replace_session(
            session,
            status=RequirementsDialogueStatus.QUESTIONING,
            current_question_id=_QUESTION_ORDER[0],
            answered_question_ids=(),
            answer_summaries=answer_summaries,
            action=restart_baseline["action"],
            trigger_mode=restart_baseline["trigger_mode"],
            trigger_conditions=restart_baseline["trigger_conditions"],
            scope=restart_baseline["scope"],
            constraints=restart_baseline["constraints"],
        )

    def _apply_when_answer(
        self,
        session: RequirementsDialogueSession,
        response: Mapping[str, Any],
        *,
        use_default: bool | None,
    ) -> RequirementsDialogueSession:
        # AUDIT-FIX(#3): Keep trigger_mode and trigger_conditions consistent when moving between push and pull semantics.
        if use_default:
            trigger_mode = "push"
            conditions = _without_on_request(_normalize_identifier_tuple(session.trigger_conditions))
            return replace(session, trigger_mode=trigger_mode, trigger_conditions=conditions)
        trigger_mode = _normalize_trigger_mode(response.get("trigger_mode")) or session.trigger_mode
        conditions = _merge_identifiers(session.trigger_conditions, response.get("trigger_conditions"))
        if trigger_mode == "push":
            conditions = _without_on_request(conditions)
        elif trigger_mode == "pull" and not conditions:
            conditions = ("on_request",)
        return replace(session, trigger_mode=trigger_mode, trigger_conditions=conditions)

    def _apply_what_answer(
        self,
        session: RequirementsDialogueSession,
        response: Mapping[str, Any],
        *,
        use_default: bool | None,
    ) -> RequirementsDialogueSession:
        if use_default:
            scope = dict(_normalize_scope_mapping(session.scope))
            scope.setdefault("selection", "all")
            return replace(session, scope=scope)
        # AUDIT-FIX(#5): Validate scope shape and text values before merging them into persisted state.
        merged_scope = dict(_normalize_scope_mapping(session.scope))
        merged_scope.update(_normalize_scope_patch(response.get("scope")))
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
        response: Mapping[str, Any],
        *,
        use_default: bool | None,
    ) -> RequirementsDialogueSession:
        if use_default:
            return session
        # AUDIT-FIX(#5): Normalize free text and scope payloads before persisting them or echoing them back later.
        action = _normalize_action_text(response.get("action")) or session.action
        constraints = _merge_texts(session.constraints, response.get("constraints"))
        scope = dict(_normalize_scope_mapping(session.scope))
        scope.update(_normalize_scope_patch(response.get("scope")))
        return replace(session, action=action, constraints=constraints, scope=scope)


def _next_question_id(answered_question_ids: tuple[str, ...]) -> str | None:
    for question_id in _QUESTION_ORDER:
        if question_id not in answered_question_ids:
            return question_id
    return None


# AUDIT-FIX(#5): Centralize shape validation for incoming structured responses and nested payloads.
def _normalize_response(response: Mapping[str, Any] | object) -> dict[str, Any]:
    if not isinstance(response, Mapping):
        raise ValueError("response must be a JSON object")
    normalized: dict[str, Any] = {}
    for key, value in response.items():
        if not isinstance(key, str):
            raise ValueError("response keys must be strings")
        normalized[key] = value
    return normalized


def _validate_question_payload(
    question_id: str,
    response: Mapping[str, Any],
    *,
    use_default: bool | None,
) -> None:
    if use_default:
        return
    relevant_fields: dict[str, tuple[str, ...]] = {
        "when": ("trigger_mode", "trigger_conditions"),
        "what": ("scope", "constraints", "trigger_conditions"),
        "how": ("action", "constraints", "scope"),
    }
    if any(field in response and response[field] is not None for field in relevant_fields[question_id]):
        return
    raise ValueError(f"The `{question_id}` question requires a structured answer or `use_default=true`")


def _merge_identifiers(existing: tuple[str, ...], raw_value: object) -> tuple[str, ...]:
    if raw_value is None:
        return _normalize_identifier_tuple(existing)
    incoming = _normalize_identifier_tuple(raw_value)
    merged: list[str] = []
    seen: set[str] = set()
    for item in (*_normalize_identifier_tuple(existing), *incoming):
        if item in seen:
            continue
        seen.add(item)
        merged.append(item)
    return tuple(merged)


def _merge_texts(existing: tuple[str, ...], raw_value: object) -> tuple[str, ...]:
    if raw_value is None:
        return _normalize_text_tuple(existing)
    incoming = _normalize_text_tuple(raw_value)
    merged: list[str] = []
    seen: set[str] = set()
    for item in (*_normalize_text_tuple(existing), *incoming):
        if item in seen:
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
    text = _normalize_free_text(
        raw_value,
        field_name="answer_summary",
        max_length=_MAX_ANSWER_SUMMARY_LEN,
    )
    return text or None


def _scope_summary(scope: dict[str, Any]) -> str:
    if not scope:
        return ""
    parts: list[str] = []
    for key in sorted(scope):
        value = scope[key]
        if isinstance(value, (list, tuple)):
            rendered_items = [_stringify_scope_scalar(item) for item in value]
            render = ", ".join(item for item in rendered_items if item)
        else:
            render = _stringify_scope_scalar(value)
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
    return "subject to " + "; ".join(_normalize_text_tuple(constraints))


def _normalize_identifier_tuple(raw_value: object) -> tuple[str, ...]:
    raw_items = _coerce_iterable(raw_value, field_name="trigger_conditions", allow_single_string=True)
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        item = _coerce_text(raw_item).lower()
        if not item:
            continue
        if not _IDENTIFIER_RE.fullmatch(item):
            raise ValueError("trigger_conditions must contain only safe identifier values")
        if item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return tuple(normalized)


def _normalize_text_tuple(raw_value: object) -> tuple[str, ...]:
    raw_items = _coerce_iterable(raw_value, field_name="constraints", allow_single_string=True)
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        item = _normalize_free_text(
            raw_item,
            field_name="constraints",
            max_length=_MAX_TEXT_ITEM_LEN,
        )
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        normalized.append(item)
        if len(normalized) > _MAX_TEXT_ITEM_COUNT:
            raise ValueError("constraints contains too many values")
    return tuple(normalized)


def _normalize_scope_patch(raw_value: object) -> dict[str, Any]:
    if raw_value is None:
        return {}
    return _normalize_scope_mapping(raw_value)


def _normalize_scope_mapping(raw_value: object) -> dict[str, Any]:
    if raw_value is None:
        return {}
    if not isinstance(raw_value, Mapping):
        raise ValueError("scope must be a JSON object")
    normalized: dict[str, Any] = {}
    for raw_key, raw_item in raw_value.items():
        if not isinstance(raw_key, str):
            raise ValueError("scope keys must be strings")
        key = raw_key.strip()
        if not key:
            raise ValueError("scope keys must not be empty")
        if not _SCOPE_KEY_RE.fullmatch(key):
            raise ValueError("scope keys must use only letters, numbers, `_`, `-`, or `.`")
        normalized[key] = _normalize_scope_value(raw_item)
    return normalized


def _normalize_scope_value(raw_value: object) -> Any:
    if raw_value is None or isinstance(raw_value, (bool, int, float)):
        return raw_value
    if isinstance(raw_value, str):
        return _normalize_free_text(raw_value, field_name="scope value", max_length=_MAX_TEXT_ITEM_LEN)
    if isinstance(raw_value, (list, tuple)):
        normalized_items = [_normalize_scope_value(item) for item in raw_value]
        normalized_items = [item for item in normalized_items if item not in ("", None)]
        if len(normalized_items) > _MAX_SCOPE_LIST_ITEMS:
            raise ValueError("scope list values contain too many items")
        return normalized_items
    raise ValueError("scope values must be simple JSON scalars or lists of scalars")


def _stringify_scope_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _copy_sequence_like(raw_value: Any) -> Any:
    if isinstance(raw_value, tuple):
        return tuple(raw_value)
    if isinstance(raw_value, list):
        return list(raw_value)
    return raw_value


def _normalize_action_text(raw_value: object) -> str:
    return _normalize_free_text(raw_value, field_name="action", max_length=_MAX_ACTION_LEN)


def _normalize_free_text(raw_value: object, *, field_name: str, max_length: int) -> str:
    text = _coerce_text(raw_value)
    if len(text) > max_length:
        raise ValueError(f"{field_name} must be at most {max_length} characters")
    return text


def _coerce_text(raw_value: object) -> str:
    if raw_value is None:
        return ""
    if isinstance(raw_value, bytes):
        text = raw_value.decode("utf-8", errors="replace")
    elif isinstance(raw_value, bytearray):
        text = bytes(raw_value).decode("utf-8", errors="replace")
    elif isinstance(raw_value, str):
        text = raw_value
    else:
        text = str(raw_value)
    return " ".join(text.split()).strip()


def _coerce_iterable(raw_value: object, *, field_name: str, allow_single_string: bool) -> tuple[object, ...]:
    if raw_value is None:
        return ()
    if allow_single_string and isinstance(raw_value, (str, bytes, bytearray)):
        return (raw_value,)
    if isinstance(raw_value, (list, tuple)):
        return tuple(raw_value)
    raise ValueError(f"{field_name} must be a list of values")


def _trigger_mode_summary(trigger_mode: object) -> str:
    normalized = _normalize_trigger_mode(trigger_mode)
    if normalized == "push":
        return "automatically"
    if normalized == "pull" or normalized is None:
        return "only when you ask"
    raise ValueError(f"Unsupported trigger mode: {trigger_mode}")


def _prompt_action_text(action: object) -> str:
    normalized = _normalize_action_text(action)
    return normalized or "do that"


def _without_on_request(conditions: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(item for item in conditions if item != "on_request")


# AUDIT-FIX(#1): Store the original field values once so a rejected confirmation can restore them exactly.
def _ensure_restart_baseline(
    answer_summaries: dict[str, str],
    session: RequirementsDialogueSession,
) -> dict[str, str]:
    if _INTERNAL_BASELINE_ACTION_KEY not in answer_summaries:
        answer_summaries[_INTERNAL_BASELINE_ACTION_KEY] = _normalize_action_text(session.action)
        answer_summaries[_INTERNAL_BASELINE_TRIGGER_MODE_KEY] = _normalize_trigger_mode(session.trigger_mode) or "pull"
        answer_summaries[_INTERNAL_BASELINE_TRIGGER_CONDITIONS_KEY] = json.dumps(
            list(_normalize_identifier_tuple(session.trigger_conditions)),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        answer_summaries[_INTERNAL_BASELINE_SCOPE_KEY] = json.dumps(
            _normalize_scope_mapping(session.scope),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        answer_summaries[_INTERNAL_BASELINE_CONSTRAINTS_KEY] = json.dumps(
            list(_normalize_text_tuple(session.constraints)),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return answer_summaries


# AUDIT-FIX(#1): Decode the stored rollback baseline with safe fallbacks for pre-fix sessions.
def _extract_restart_baseline(
    session: RequirementsDialogueSession,
    answer_summaries: Mapping[str, str],
) -> dict[str, Any]:
    baseline_action = _normalize_action_text(answer_summaries.get(_INTERNAL_BASELINE_ACTION_KEY)) or session.action
    baseline_trigger_mode = (
        _normalize_trigger_mode(answer_summaries.get(_INTERNAL_BASELINE_TRIGGER_MODE_KEY))
        or _normalize_trigger_mode(session.trigger_mode)
        or "pull"
    )
    baseline_trigger_conditions = _decode_identifier_tuple(
        answer_summaries.get(_INTERNAL_BASELINE_TRIGGER_CONDITIONS_KEY),
        fallback=session.trigger_conditions,
    )
    baseline_scope = _decode_scope_mapping(
        answer_summaries.get(_INTERNAL_BASELINE_SCOPE_KEY),
        fallback=session.scope,
    )
    baseline_constraints = _decode_text_tuple(
        answer_summaries.get(_INTERNAL_BASELINE_CONSTRAINTS_KEY),
        fallback=session.constraints,
    )
    if baseline_trigger_mode == "pull" and not baseline_trigger_conditions:
        baseline_trigger_conditions = ("on_request",)
    return {
        "action": baseline_action,
        "trigger_mode": baseline_trigger_mode,
        "trigger_conditions": baseline_trigger_conditions,
        "scope": baseline_scope,
        "constraints": baseline_constraints,
    }


def _decode_identifier_tuple(raw_json: str | None, *, fallback: object) -> tuple[str, ...]:
    decoded = _decode_json_value(raw_json, fallback=fallback)
    return _normalize_identifier_tuple(decoded)


def _decode_text_tuple(raw_json: str | None, *, fallback: object) -> tuple[str, ...]:
    decoded = _decode_json_value(raw_json, fallback=fallback)
    return _normalize_text_tuple(decoded)


def _decode_scope_mapping(raw_json: str | None, *, fallback: object) -> dict[str, Any]:
    decoded = _decode_json_value(raw_json, fallback=fallback)
    return _normalize_scope_mapping(decoded)


def _decode_json_value(raw_json: str | None, *, fallback: object) -> Any:
    if not raw_json:
        return fallback
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        return fallback


def _strip_internal_answer_summaries(answer_summaries: Mapping[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in answer_summaries.items()
        if key not in _INTERNAL_ANSWER_SUMMARY_KEYS
    }


def _replace_session(session: RequirementsDialogueSession, **changes: Any) -> RequirementsDialogueSession:
    # AUDIT-FIX(#4): Refresh updated_at on every state transition so persistence and recovery can order writes correctly.
    if "updated_at" not in changes:
        changes["updated_at"] = _now_like(getattr(session, "updated_at", None))
    return replace(session, **changes)


# AUDIT-FIX(#4): Preserve the existing datetime shape while generating a fresh timestamp for session writes.
def _now_like(reference: object) -> object:
    if isinstance(reference, datetime):
        if reference.tzinfo is None:
            return datetime.now()
        return datetime.now(reference.tzinfo)
    return reference
