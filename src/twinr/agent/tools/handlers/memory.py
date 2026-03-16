"""Handle durable-memory tool calls for realtime Twinr sessions.

Exports synchronous handlers for facts, contacts, conflict resolution,
preferences, plans, and profile-context updates at the runtime boundary.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping  # AUDIT-FIX(#5): Add runtime-safe collection checks for arguments and runtime payloads.
from typing import Any

from .support import require_sensitive_voice_confirmation


_MAX_IDENTIFIER_LENGTH = 128
_MAX_SHORT_TEXT_LENGTH = 256
_MAX_SUMMARY_LENGTH = 1024
_MAX_MEDIUM_TEXT_LENGTH = 2048
_MAX_LONG_TEXT_LENGTH = 4096
_MAX_EMAIL_LENGTH = 254
_MAX_PHONE_LENGTH = 64
_MAX_EMIT_VALUE_LENGTH = 160
_MAX_EVENT_METADATA_LENGTH = 512


def _ensure_arguments_mapping(arguments: dict[str, object]) -> Mapping[str, object]:
    # AUDIT-FIX(#5): Reject non-object tool arguments early instead of relying on AttributeError from `.get()`.
    if not isinstance(arguments, Mapping):
        raise RuntimeError("Tool arguments must be an object.")
    return arguments


def _get_text_argument(
    arguments: Mapping[str, object],
    key: str,
    *,
    required: bool = False,
    default: str | None = None,
    max_length: int = _MAX_SHORT_TEXT_LENGTH,
) -> str | None:
    # AUDIT-FIX(#5): Reject nested/non-scalar argument payloads and bound field size to avoid silent data corruption.
    raw_value = arguments.get(key)
    if raw_value is None:
        text = ""
    elif isinstance(raw_value, str):
        text = raw_value.strip()
    elif isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
        text = str(raw_value).strip()
    else:
        raise RuntimeError(f"Invalid `{key}` value.")

    if not text:
        if required:
            raise RuntimeError(f"Cannot continue without `{key}`.")
        return default

    if len(text) > max_length:
        raise RuntimeError(f"`{key}` exceeds the maximum supported length.")
    return text


def _require_attr(obj: object, attr_name: str) -> object:
    # AUDIT-FIX(#4): Convert malformed runtime payloads into deterministic tool errors instead of AttributeError crashes.
    if not hasattr(obj, attr_name):
        raise RuntimeError(f"Runtime returned malformed data: missing `{attr_name}`.")
    return getattr(obj, attr_name)


def _get_runtime_text_attr(obj: object, attr_name: str, *, required: bool = False) -> str | None:
    # AUDIT-FIX(#4): Coerce runtime object text fields into JSON-safe strings without assuming exact runtime classes.
    if required:
        raw_value = _require_attr(obj, attr_name)
    else:
        raw_value = getattr(obj, attr_name, None)

    if raw_value is None:
        if required:
            raise RuntimeError(f"Runtime returned malformed data: `{attr_name}` is missing.")
        return None
    return str(raw_value).strip()


def _get_runtime_scalar_attr(obj: object, attr_name: str, *, required: bool = False) -> object:
    # AUDIT-FIX(#4): Keep response payloads JSON-safe even when runtime returns custom scalar-like objects.
    if required:
        raw_value = _require_attr(obj, attr_name)
    else:
        raw_value = getattr(obj, attr_name, None)

    if raw_value is None:
        if required:
            raise RuntimeError(f"Runtime returned malformed data: `{attr_name}` is missing.")
        return None

    if isinstance(raw_value, (str, int, float, bool)):
        return raw_value
    return str(raw_value)


def _coerce_runtime_iterable(
    value: object,
    *,
    field_name: str,
    allow_none: bool = False,
) -> tuple[object, ...]:
    # AUDIT-FIX(#4): Guard list-building code against `None`, scalars, and non-iterables returned by runtime methods.
    if value is None:
        if allow_none:
            return ()
        raise RuntimeError(f"Runtime returned malformed data: `{field_name}` is missing.")

    if isinstance(value, (str, bytes)):
        return (value,)

    if isinstance(value, Mapping):
        raise RuntimeError(f"Runtime returned malformed data: `{field_name}` must be a list-like value.")

    if isinstance(value, Iterable):
        return tuple(value)

    raise RuntimeError(f"Runtime returned malformed data: `{field_name}` must be iterable.")


def _coerce_string_list(value: object, *, field_name: str) -> list[str]:
    # AUDIT-FIX(#4): Build stable string lists for phones/emails/options without leaking TypeError from `list(None)`.
    return [str(item).strip() for item in _coerce_runtime_iterable(value, field_name=field_name, allow_none=True)]


def _sanitize_emit_value(value: object, *, max_length: int = _MAX_EMIT_VALUE_LENGTH) -> str:
    # AUDIT-FIX(#7): Collapse control characters and bound telemetry values before interpolating them into `key=value`.
    if value is None:
        text = ""
    elif isinstance(value, bool):
        text = "true" if value else "false"
    else:
        text = str(value)

    sanitized = "".join(ch if ch.isprintable() and ch not in "\r\n\t" else " " for ch in text)
    sanitized = " ".join(sanitized.split())

    if len(sanitized) > max_length:
        sanitized = sanitized[: max_length - 1].rstrip() + "…"

    return sanitized or "-"


def _sanitize_event_metadata_value(value: object) -> object:
    # AUDIT-FIX(#1): Ensure best-effort audit logging cannot fail on unserializable metadata values.
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return _sanitize_emit_value(value, max_length=_MAX_EVENT_METADATA_LENGTH)


def _safe_emit(owner: Any, key: str, value: object) -> None:
    # AUDIT-FIX(#1,#7): Treat telemetry emission as best-effort so a committed write is not reported as failed.
    try:
        owner.emit(f"{key}={_sanitize_emit_value(value)}")
    except Exception:
        return


def _safe_record_event(owner: Any, event_name: str, message: str, **metadata: object) -> None:
    # AUDIT-FIX(#1): Prevent audit logging failures from aborting successful durable state changes.
    safe_metadata = {key: _sanitize_event_metadata_value(value) for key, value in metadata.items()}
    try:
        owner._record_event(event_name, message, **safe_metadata)
    except Exception:
        return


def _safe_remember_note(
    owner: Any,
    *,
    kind: str,
    content: str,
    source: str,
    metadata: Mapping[str, object],
) -> None:
    # AUDIT-FIX(#1): Persisting secondary notes is useful, but it must not invalidate a write that already succeeded.
    try:
        owner.runtime.remember_note(
            kind=kind,
            content=content,
            source=source,
            metadata={key: _sanitize_event_metadata_value(value) for key, value in metadata.items()},
        )
    except Exception:
        return


def _serialize_conflict_queue(queue: Iterable[object]) -> list[dict[str, object]]:
    serialized: list[dict[str, object]] = []
    for item in _coerce_runtime_iterable(queue, field_name="conflict_queue"):  # AUDIT-FIX(#4): Validate queue shape before dereferencing item attributes.
        options: list[dict[str, object]] = []
        for option in _coerce_runtime_iterable(_require_attr(item, "options"), field_name="conflict_options"):  # AUDIT-FIX(#4): Validate option collections before iterating.
            options.append(
                {
                    "memory_id": _get_runtime_scalar_attr(option, "memory_id", required=True),
                    "summary": _get_runtime_text_attr(option, "summary", required=True),
                    "details": _get_runtime_text_attr(option, "details"),
                    "status": _get_runtime_scalar_attr(option, "status", required=True),
                    "value_key": _get_runtime_scalar_attr(option, "value_key"),
                }
            )
        serialized.append(
            {
                "slot_key": _get_runtime_scalar_attr(item, "slot_key", required=True),
                "question": _get_runtime_text_attr(item, "question", required=True),
                "reason": _get_runtime_text_attr(item, "reason"),
                "candidate_memory_id": _get_runtime_scalar_attr(item, "candidate_memory_id"),
                "options": options,
            }
        )
    return serialized


def handle_remember_memory(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Store a durable free-form memory entry.

    Args:
        owner: Tool executor owner exposing runtime access and telemetry hooks.
        arguments: Tool payload with required ``summary`` and optional
            ``kind``, ``details``, and confirmation fields.

    Returns:
        JSON-safe payload describing the stored durable-memory entry.

    Raises:
        SensitiveActionConfirmationRequired: If spoken confirmation is required
            before saving durable memory.
        RuntimeError: If required fields are missing, too large, or runtime
            persistence returns malformed data.
    """
    arguments = _ensure_arguments_mapping(arguments)  # AUDIT-FIX(#5): Validate top-level tool arguments before field access.
    kind = _get_text_argument(arguments, "kind", default="memory", max_length=_MAX_IDENTIFIER_LENGTH) or "memory"  # AUDIT-FIX(#5): Bound and validate persisted identifiers.
    summary = _get_text_argument(arguments, "summary", required=True, max_length=_MAX_SUMMARY_LENGTH)  # AUDIT-FIX(#5): Prevent empty or oversized durable-memory summaries.
    details = _get_text_argument(arguments, "details", max_length=_MAX_LONG_TEXT_LENGTH)  # AUDIT-FIX(#5): Prevent unbounded detail payloads.
    require_sensitive_voice_confirmation(owner, arguments, action_label="save durable memory")  # AUDIT-FIX(#6): Request confirmation only after the action is locally valid.

    entry = owner.runtime.store_durable_memory(
        kind=kind,
        summary=summary,
        details=details,
    )
    _safe_remember_note(  # AUDIT-FIX(#1): Secondary note persistence must not make an already-saved memory look failed.
        owner,
        kind="fact",
        content=f"Saved memory: {summary}",
        source="remember_memory",
        metadata={
            "memory_kind": _get_runtime_scalar_attr(entry, "kind", required=True),
            "memory_id": _get_runtime_scalar_attr(entry, "entry_id", required=True),
        },
    )
    _safe_emit(owner, "memory_tool_call", True)  # AUDIT-FIX(#1,#7): Emit a sanitized, best-effort telemetry flag.
    _safe_emit(owner, "memory_saved", _get_runtime_text_attr(entry, "summary", required=True))  # AUDIT-FIX(#1,#7): Do not inject raw summary text into telemetry.
    _safe_record_event(  # AUDIT-FIX(#1): Successful writes stay successful even if audit logging fails.
        owner,
        "memory_saved",
        "Important user-requested memory was stored in MEMORY.md.",
        kind=_get_runtime_scalar_attr(entry, "kind", required=True),
        summary=_get_runtime_text_attr(entry, "summary", required=True),
    )
    return {
        "status": "saved",
        "kind": _get_runtime_scalar_attr(entry, "kind", required=True),
        "summary": _get_runtime_text_attr(entry, "summary", required=True),
        "memory_id": _get_runtime_scalar_attr(entry, "entry_id", required=True),
    }


def handle_remember_contact(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Store or clarify a structured contact memory.

    Args:
        owner: Tool executor owner exposing runtime access and telemetry hooks.
        arguments: Tool payload with required ``given_name`` plus optional
            contact details, notes, and confirmation fields.

    Returns:
        JSON-safe payload describing the saved contact or a clarification
        question when multiple contact candidates remain.

    Raises:
        SensitiveActionConfirmationRequired: If spoken confirmation is required
            before saving contact data.
        RuntimeError: If required fields are missing, oversized, or runtime
            results are malformed.
    """
    arguments = _ensure_arguments_mapping(arguments)  # AUDIT-FIX(#5): Validate top-level tool arguments before field access.
    given_name = _get_text_argument(arguments, "given_name", required=True, max_length=_MAX_SHORT_TEXT_LENGTH)  # AUDIT-FIX(#5): Reject empty or oversized contact names.
    family_name = _get_text_argument(arguments, "family_name", max_length=_MAX_SHORT_TEXT_LENGTH)  # AUDIT-FIX(#5): Bound optional text fields before persistence.
    phone = _get_text_argument(arguments, "phone", max_length=_MAX_PHONE_LENGTH)  # AUDIT-FIX(#5): Bound phone-size inputs instead of silently stringifying large objects.
    email = _get_text_argument(arguments, "email", max_length=_MAX_EMAIL_LENGTH)  # AUDIT-FIX(#5): Bound email-size inputs instead of silently stringifying large objects.
    role = _get_text_argument(arguments, "role", max_length=_MAX_SHORT_TEXT_LENGTH)  # AUDIT-FIX(#5): Bound optional role text.
    relation = _get_text_argument(arguments, "relation", max_length=_MAX_SHORT_TEXT_LENGTH)  # AUDIT-FIX(#5): Bound optional relation text.
    notes = _get_text_argument(arguments, "notes", max_length=_MAX_LONG_TEXT_LENGTH)  # AUDIT-FIX(#5): Prevent unbounded note payloads.
    require_sensitive_voice_confirmation(owner, arguments, action_label="save a contact")  # AUDIT-FIX(#6): Defer confirmation until the request is actionable.

    result = owner.runtime.remember_contact(
        given_name=given_name,
        family_name=family_name,
        phone=phone,
        email=email,
        role=role,
        relation=relation,
        notes=notes,
        source="remember_contact",
    )
    _safe_emit(owner, "graph_contact_tool_call", True)  # AUDIT-FIX(#1,#7): Make telemetry emission best-effort and sanitized.
    status = _get_runtime_text_attr(result, "status", required=True)  # AUDIT-FIX(#4): Validate runtime result shape before branching.
    if status == "needs_clarification":
        _safe_emit(owner, "graph_contact_clarification", True)  # AUDIT-FIX(#1,#7): Keep clarification telemetry non-fatal.
        return {
            "status": "needs_clarification",
            "question": _get_runtime_text_attr(result, "question", required=True),
            "options": [
                {
                    "label": _get_runtime_text_attr(option, "label", required=True),
                    "role": _get_runtime_text_attr(option, "role"),
                    "phones": _coerce_string_list(getattr(option, "phones", None), field_name="contact_option_phones"),
                    "emails": _coerce_string_list(getattr(option, "emails", None), field_name="contact_option_emails"),
                }
                for option in _coerce_runtime_iterable(_require_attr(result, "options"), field_name="contact_options")
            ],
        }

    label = _get_runtime_text_attr(result, "label", required=True)  # AUDIT-FIX(#4): Require the success payload fields before emitting success.
    node_id = _get_runtime_scalar_attr(result, "node_id", required=True)  # AUDIT-FIX(#4): Ensure the success response is JSON-safe.
    _safe_emit(owner, "graph_contact_saved", label)  # AUDIT-FIX(#1,#7): Sanitize result labels before telemetry emission.
    _safe_record_event(  # AUDIT-FIX(#1): Audit logging must not trigger duplicate contact writes on retry.
        owner,
        "graph_contact_saved",
        "Structured contact memory was stored.",
        label=label,
        status=status,
    )
    return {
        "status": status,
        "label": label,
        "node_id": node_id,
    }


def handle_lookup_contact(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Look up a saved contact by name and optional filters.

    Args:
        owner: Tool executor owner exposing runtime access and telemetry hooks.
        arguments: Tool payload with required ``name`` and optional
            ``family_name`` or ``role`` filters.

    Returns:
        JSON-safe payload describing a missing match, a clarification prompt,
        or the resolved contact details.

    Raises:
        SensitiveActionConfirmationRequired: If saved contact details would be
            disclosed without trusted speaker confirmation.
        RuntimeError: If required fields are missing, oversized, or runtime
            results are malformed.
    """
    arguments = _ensure_arguments_mapping(arguments)  # AUDIT-FIX(#5): Validate top-level tool arguments before field access.
    name = _get_text_argument(arguments, "name", required=True, max_length=_MAX_SHORT_TEXT_LENGTH)  # AUDIT-FIX(#5): Reject empty or oversized contact lookup names.
    family_name = _get_text_argument(arguments, "family_name", max_length=_MAX_SHORT_TEXT_LENGTH)  # AUDIT-FIX(#5): Bound optional lookup filters.
    role = _get_text_argument(arguments, "role", max_length=_MAX_SHORT_TEXT_LENGTH)  # AUDIT-FIX(#5): Bound optional lookup filters.
    result = owner.runtime.lookup_contact(
        name=name,
        family_name=family_name,
        role=role,
    )
    _safe_emit(owner, "graph_contact_lookup", True)  # AUDIT-FIX(#1,#7): Keep lookup telemetry sanitized and non-fatal.
    status = _get_runtime_text_attr(result, "status", required=True)  # AUDIT-FIX(#4): Validate runtime result shape before branching.
    if status == "not_found":
        return {"status": "not_found", "name": name}
    require_sensitive_voice_confirmation(owner, arguments, action_label="read saved contact details")  # AUDIT-FIX(#10): Gate contact PII disclosure behind the existing sensitive voice confirmation flow.
    if status == "needs_clarification":
        _safe_emit(owner, "graph_contact_clarification", True)  # AUDIT-FIX(#1,#7): Keep clarification telemetry non-fatal.
        return {
            "status": "needs_clarification",
            "question": _get_runtime_text_attr(result, "question", required=True),
            "options": [
                {
                    "label": _get_runtime_text_attr(option, "label", required=True),
                    "role": _get_runtime_text_attr(option, "role"),
                    "phones": _coerce_string_list(getattr(option, "phones", None), field_name="lookup_option_phones"),
                    "emails": _coerce_string_list(getattr(option, "emails", None), field_name="lookup_option_emails"),
                }
                for option in _coerce_runtime_iterable(_require_attr(result, "options"), field_name="lookup_options")
            ],
        }

    match = getattr(result, "match", None)
    if match is None:
        raise RuntimeError("Runtime returned no contact match for a successful lookup.")  # AUDIT-FIX(#3): Replace `assert` with a deterministic production-safe error.
    return {
        "status": "found",
        "label": _get_runtime_text_attr(match, "label", required=True),
        "role": _get_runtime_text_attr(match, "role"),
        "phones": _coerce_string_list(getattr(match, "phones", None), field_name="lookup_match_phones"),
        "emails": _coerce_string_list(getattr(match, "emails", None), field_name="lookup_match_emails"),
    }


def handle_get_memory_conflicts(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """List open long-term memory conflicts for clarification.

    Args:
        owner: Tool executor owner exposing runtime access and telemetry hooks.
        arguments: Tool payload with optional ``query_text`` to narrow the
            returned conflict queue.

    Returns:
        JSON-safe payload with ``status`` and serialized conflict records.

    Raises:
        SensitiveActionConfirmationRequired: If conflict details would be
            disclosed without trusted speaker confirmation.
        RuntimeError: If arguments are invalid or runtime conflict payloads are
            malformed.
    """
    arguments = _ensure_arguments_mapping(arguments)  # AUDIT-FIX(#5): Validate top-level tool arguments before field access.
    query_text = _get_text_argument(arguments, "query_text", max_length=_MAX_MEDIUM_TEXT_LENGTH)  # AUDIT-FIX(#5): Bound free-text filtering input.
    queue = owner.runtime.select_long_term_memory_conflicts(query_text=query_text)
    payload = _serialize_conflict_queue(queue)  # AUDIT-FIX(#8): Validate and serialize the queue before deriving counts or emitting audit events.
    _safe_emit(owner, "memory_conflict_tool_call", True)  # AUDIT-FIX(#1,#7): Keep conflict-inspection telemetry sanitized and non-fatal.
    _safe_record_event(  # AUDIT-FIX(#8): Record counts from the validated payload, not from an unchecked runtime iterable.
        owner,
        "memory_conflicts_inspected",
        "Open long-term memory conflicts were inspected for clarification.",
        conflict_count=len(payload),
        query=query_text or "",
    )
    if not payload:
        return {
            "status": "no_conflicts",
            "conflict_count": 0,
            "conflicts": [],
        }
    require_sensitive_voice_confirmation(owner, arguments, action_label="inspect saved long-term memory details")  # AUDIT-FIX(#10): Gate disclosure of saved-memory details behind the existing sensitive voice confirmation flow.
    return {
        "status": "ok",
        "conflict_count": len(payload),
        "conflicts": payload,
    }


def handle_resolve_memory_conflict(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Resolve a long-term memory conflict in favor of one memory entry.

    Args:
        owner: Tool executor owner exposing runtime access and telemetry hooks.
        arguments: Tool payload with required ``slot_key``,
            ``selected_memory_id``, and optional confirmation fields.

    Returns:
        JSON-safe payload describing the selected memory and remaining conflict
        count after resolution.

    Raises:
        SensitiveActionConfirmationRequired: If spoken confirmation is required
            before changing saved long-term memory.
        RuntimeError: If identifiers are missing, oversized, or runtime
            confirmation data is malformed.
    """
    arguments = _ensure_arguments_mapping(arguments)  # AUDIT-FIX(#5): Validate top-level tool arguments before field access.
    slot_key = _get_text_argument(arguments, "slot_key", required=True, max_length=_MAX_IDENTIFIER_LENGTH)  # AUDIT-FIX(#5): Reject empty or oversized slot identifiers.
    selected_memory_id = _get_text_argument(arguments, "selected_memory_id", required=True, max_length=_MAX_IDENTIFIER_LENGTH)  # AUDIT-FIX(#5): Reject empty or oversized memory identifiers.
    require_sensitive_voice_confirmation(owner, arguments, action_label="change saved long-term memory")  # AUDIT-FIX(#6): Defer confirmation until the request is actionable.

    result = owner.runtime.resolve_long_term_memory_conflict(
        slot_key=slot_key,
        selected_memory_id=selected_memory_id,
    )
    updated_objects = _coerce_runtime_iterable(_require_attr(result, "updated_objects"), field_name="updated_objects")  # AUDIT-FIX(#4): Validate runtime update payload shape before dereferencing it.
    remaining_conflicts = _coerce_runtime_iterable(_require_attr(result, "remaining_conflicts"), field_name="remaining_conflicts")  # AUDIT-FIX(#4): Validate runtime conflict payload shape before counting it.
    updated = {
        str(_get_runtime_scalar_attr(item, "memory_id", required=True)): item
        for item in updated_objects
    }
    selected = updated.get(selected_memory_id)
    if selected is None:
        raise RuntimeError("Runtime did not confirm the selected memory conflict resolution.")  # AUDIT-FIX(#2): Fail closed unless the chosen memory ID is present in the runtime response.

    superseded_memory_ids = sorted(
        str(_get_runtime_scalar_attr(item, "memory_id", required=True))
        for item in updated_objects
        if str(_get_runtime_scalar_attr(item, "memory_id", required=True)) != selected_memory_id
        and _get_runtime_text_attr(item, "status", required=True) == "superseded"
    )
    invalid_memory_ids = sorted(
        str(_get_runtime_scalar_attr(item, "memory_id", required=True))
        for item in updated_objects
        if str(_get_runtime_scalar_attr(item, "memory_id", required=True)) != selected_memory_id
        and _get_runtime_text_attr(item, "status", required=True) == "invalid"
    )
    _safe_emit(owner, "memory_conflict_tool_call", True)  # AUDIT-FIX(#1,#7): Keep conflict-resolution telemetry sanitized and non-fatal.
    _safe_emit(owner, "memory_conflict_resolved", slot_key)  # AUDIT-FIX(#1,#7): Sanitize slot identifiers before telemetry emission.
    _safe_record_event(  # AUDIT-FIX(#1): Audit logging must not make a committed resolution look failed.
        owner,
        "memory_conflict_resolved",
        "A long-term memory conflict was resolved from spoken clarification.",
        slot_key=slot_key,
        selected_memory_id=selected_memory_id,
        remaining_conflicts=len(remaining_conflicts),
    )
    return {
        "status": "resolved",
        "slot_key": slot_key,
        "selected_memory_id": selected_memory_id,
        "selected_summary": _get_runtime_text_attr(selected, "summary"),
        "superseded_memory_ids": superseded_memory_ids,
        "invalid_memory_ids": invalid_memory_ids,
        "remaining_conflict_count": len(remaining_conflicts),
    }


def handle_remember_preference(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Store a structured user preference in long-term memory.

    Args:
        owner: Tool executor owner exposing runtime access and telemetry hooks.
        arguments: Tool payload with required ``category`` and ``value`` plus
            optional sentiment, product scope, details, and confirmation fields.

    Returns:
        JSON-safe payload describing the stored preference node and edge.

    Raises:
        SensitiveActionConfirmationRequired: If spoken confirmation is required
            before saving a preference.
        RuntimeError: If required fields are missing, oversized, or runtime
            results are malformed.
    """
    arguments = _ensure_arguments_mapping(arguments)  # AUDIT-FIX(#5): Validate top-level tool arguments before field access.
    category = _get_text_argument(arguments, "category", required=True, max_length=_MAX_SHORT_TEXT_LENGTH)  # AUDIT-FIX(#5): Reject empty or oversized preference categories.
    value = _get_text_argument(arguments, "value", required=True, max_length=_MAX_MEDIUM_TEXT_LENGTH)  # AUDIT-FIX(#5): Reject empty or oversized preference values.
    sentiment = _get_text_argument(arguments, "sentiment", default="prefer", max_length=_MAX_SHORT_TEXT_LENGTH) or "prefer"  # AUDIT-FIX(#9): Remove the no-op sentiment branch and use the shared parser.
    for_product = _get_text_argument(arguments, "for_product", max_length=_MAX_MEDIUM_TEXT_LENGTH)  # AUDIT-FIX(#5): Bound optional product descriptors.
    details = _get_text_argument(arguments, "details", max_length=_MAX_LONG_TEXT_LENGTH)  # AUDIT-FIX(#5): Prevent unbounded preference detail payloads.
    require_sensitive_voice_confirmation(owner, arguments, action_label="save a preference")  # AUDIT-FIX(#6): Defer confirmation until the request is actionable.

    result = owner.runtime.remember_preference(
        category=category,
        value=value,
        for_product=for_product,
        sentiment=sentiment,
        details=details,
        source="remember_preference",
    )
    status = _get_runtime_text_attr(result, "status", required=True)  # AUDIT-FIX(#4): Validate runtime result shape before success emission.
    label = _get_runtime_text_attr(result, "label", required=True)  # AUDIT-FIX(#4): Require success payload fields explicitly.
    node_id = _get_runtime_scalar_attr(result, "node_id", required=True)  # AUDIT-FIX(#4): Keep returned node IDs JSON-safe.
    edge_type = _get_runtime_scalar_attr(result, "edge_type", required=True)  # AUDIT-FIX(#4): Keep returned edge types JSON-safe.
    _safe_emit(owner, "graph_preference_tool_call", True)  # AUDIT-FIX(#1,#7): Make telemetry emission best-effort and sanitized.
    _safe_emit(owner, "graph_preference_saved", label)  # AUDIT-FIX(#1,#7): Sanitize preference labels before telemetry emission.
    _safe_record_event(  # AUDIT-FIX(#1): Audit logging must not trigger duplicate preference writes on retry.
        owner,
        "graph_preference_saved",
        "Structured preference memory was stored.",
        label=label,
        edge_type=edge_type,
    )
    return {
        "status": status,
        "label": label,
        "node_id": node_id,
        "edge_type": edge_type,
    }


def handle_remember_plan(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Store a future plan in long-term memory.

    Args:
        owner: Tool executor owner exposing runtime access and telemetry hooks.
        arguments: Tool payload with required ``summary`` plus optional
            schedule text, details, and confirmation fields.

    Returns:
        JSON-safe payload describing the stored plan node and edge.

    Raises:
        SensitiveActionConfirmationRequired: If spoken confirmation is required
            before saving a future plan.
        RuntimeError: If required fields are missing, oversized, or runtime
            results are malformed.
    """
    arguments = _ensure_arguments_mapping(arguments)  # AUDIT-FIX(#5): Validate top-level tool arguments before field access.
    summary = _get_text_argument(arguments, "summary", required=True, max_length=_MAX_SUMMARY_LENGTH)  # AUDIT-FIX(#5): Reject empty or oversized plan summaries.
    when_text = _get_text_argument(arguments, "when", max_length=_MAX_MEDIUM_TEXT_LENGTH)  # AUDIT-FIX(#5): Bound free-text schedule descriptors.
    details = _get_text_argument(arguments, "details", max_length=_MAX_LONG_TEXT_LENGTH)  # AUDIT-FIX(#5): Prevent unbounded plan detail payloads.
    require_sensitive_voice_confirmation(owner, arguments, action_label="save a future plan")  # AUDIT-FIX(#6): Defer confirmation until the request is actionable.

    result = owner.runtime.remember_plan(
        summary=summary,
        when_text=when_text,
        details=details,
        source="remember_plan",
    )
    status = _get_runtime_text_attr(result, "status", required=True)  # AUDIT-FIX(#4): Validate runtime result shape before success emission.
    label = _get_runtime_text_attr(result, "label", required=True)  # AUDIT-FIX(#4): Require success payload fields explicitly.
    node_id = _get_runtime_scalar_attr(result, "node_id", required=True)  # AUDIT-FIX(#4): Keep returned node IDs JSON-safe.
    edge_type = _get_runtime_scalar_attr(result, "edge_type", required=True)  # AUDIT-FIX(#4): Keep returned edge types JSON-safe.
    _safe_emit(owner, "graph_plan_tool_call", True)  # AUDIT-FIX(#1,#7): Make telemetry emission best-effort and sanitized.
    _safe_emit(owner, "graph_plan_saved", label)  # AUDIT-FIX(#1,#7): Sanitize plan labels before telemetry emission.
    _safe_record_event(  # AUDIT-FIX(#1): Audit logging must not trigger duplicate plan writes on retry.
        owner,
        "graph_plan_saved",
        "Structured plan memory was stored.",
        label=label,
        edge_type=edge_type,
    )
    return {
        "status": status,
        "label": label,
        "node_id": node_id,
        "edge_type": edge_type,
    }


def handle_update_user_profile(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Update stable user-profile context used by future turns.

    Args:
        owner: Tool executor owner exposing runtime access and telemetry hooks.
        arguments: Tool payload with required ``category`` and ``instruction``
            plus optional confirmation fields.

    Returns:
        JSON-safe payload with ``status="updated"``, the normalized
        ``category``, and stored ``instruction``.

    Raises:
        SensitiveActionConfirmationRequired: If spoken confirmation is required
            before changing the saved user profile.
        RuntimeError: If required fields are missing, oversized, or runtime
            results are malformed.
    """
    arguments = _ensure_arguments_mapping(arguments)  # AUDIT-FIX(#5): Validate top-level tool arguments before field access.
    category = _get_text_argument(arguments, "category", required=True, max_length=_MAX_SHORT_TEXT_LENGTH)  # AUDIT-FIX(#5): Reject empty or oversized profile categories.
    instruction = _get_text_argument(arguments, "instruction", required=True, max_length=_MAX_LONG_TEXT_LENGTH)  # AUDIT-FIX(#5): Reject empty or oversized profile instructions.
    require_sensitive_voice_confirmation(owner, arguments, action_label="change the saved user profile")  # AUDIT-FIX(#6): Defer confirmation until the request is actionable.

    entry = owner.runtime.update_user_profile_context(
        category=category,
        instruction=instruction,
    )
    entry_key = _get_runtime_text_attr(entry, "key", required=True)  # AUDIT-FIX(#4): Validate runtime entry shape before audit/telemetry.
    entry_instruction = _get_runtime_text_attr(entry, "instruction", required=True)  # AUDIT-FIX(#4): Validate runtime entry shape before audit/telemetry.
    _safe_remember_note(  # AUDIT-FIX(#1): Secondary note persistence must not invalidate a committed profile update.
        owner,
        kind="preference",
        content=f"User profile update ({entry_key}): {entry_instruction}",
        source="update_user_profile",
        metadata={"category": entry_key},
    )
    _safe_emit(owner, "user_profile_tool_call", True)  # AUDIT-FIX(#1,#7): Make telemetry emission best-effort and sanitized.
    _safe_emit(owner, "user_profile_update", entry_key)  # AUDIT-FIX(#1,#7): Sanitize profile keys before telemetry emission.
    _safe_record_event(  # AUDIT-FIX(#1): Audit logging must not make a committed profile update look failed.
        owner,
        "user_profile_updated",
        "Stable user profile context was updated from an explicit user request.",
        category=entry_key,
    )
    return {
        "status": "updated",
        "category": entry_key,
        "instruction": entry_instruction,
    }


def handle_update_personality(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Update Twinr personality context used by future turns.

    Args:
        owner: Tool executor owner exposing runtime access and telemetry hooks.
        arguments: Tool payload with required ``category`` and ``instruction``
            plus optional confirmation fields.

    Returns:
        JSON-safe payload with ``status="updated"``, the normalized
        ``category``, and stored ``instruction``.

    Raises:
        SensitiveActionConfirmationRequired: If spoken confirmation is required
            before changing Twinr's future behavior.
        RuntimeError: If required fields are missing, oversized, or runtime
            results are malformed.
    """
    arguments = _ensure_arguments_mapping(arguments)  # AUDIT-FIX(#5): Validate top-level tool arguments before field access.
    category = _get_text_argument(arguments, "category", required=True, max_length=_MAX_SHORT_TEXT_LENGTH)  # AUDIT-FIX(#5): Reject empty or oversized personality categories.
    instruction = _get_text_argument(arguments, "instruction", required=True, max_length=_MAX_LONG_TEXT_LENGTH)  # AUDIT-FIX(#5): Reject empty or oversized personality instructions.
    require_sensitive_voice_confirmation(owner, arguments, action_label="change Twinr's future behavior")  # AUDIT-FIX(#6): Defer confirmation until the request is actionable.

    entry = owner.runtime.update_personality_context(
        category=category,
        instruction=instruction,
    )
    entry_key = _get_runtime_text_attr(entry, "key", required=True)  # AUDIT-FIX(#4): Validate runtime entry shape before audit/telemetry.
    entry_instruction = _get_runtime_text_attr(entry, "instruction", required=True)  # AUDIT-FIX(#4): Validate runtime entry shape before audit/telemetry.
    _safe_remember_note(  # AUDIT-FIX(#1): Secondary note persistence must not invalidate a committed personality update.
        owner,
        kind="preference",
        content=f"Behavior update ({entry_key}): {entry_instruction}",
        source="update_personality",
        metadata={"category": entry_key},
    )
    _safe_emit(owner, "personality_tool_call", True)  # AUDIT-FIX(#1,#7): Make telemetry emission best-effort and sanitized.
    _safe_emit(owner, "personality_update", entry_key)  # AUDIT-FIX(#1,#7): Sanitize personality keys before telemetry emission.
    _safe_record_event(  # AUDIT-FIX(#1): Audit logging must not make a committed personality update look failed.
        owner,
        "personality_updated",
        "Twinr personality context was updated from an explicit user request.",
        category=entry_key,
    )
    return {
        "status": "updated",
        "category": entry_key,
        "instruction": entry_instruction,
    }
