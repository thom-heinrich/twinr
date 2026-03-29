"""Follow-up continuation and closure helpers for the realtime workflow loop."""

from __future__ import annotations

# CHANGELOG: 2026-03-28
# BUG-1: Normalize boolean config values so string/env flags like "false" do not enable follow-up.
# BUG-2: Normalize turn sources before proactive gating so non-canonical values do not bypass policy.
# SEC-1: Sanitize and size-bound emitted reasons; switch to one structured record to prevent log forging/injection.
# IMP-1: Add structural Protocol contracts instead of `Any` for the realtime-loop boundary.
# IMP-2: Fail closed on quiet-mode / closure-runtime faults to avoid unintended reopen-listen behavior.
# IMP-3: Add versioned structured observability payloads ready for OTel-style pipelines.

import json
import math
import re
from typing import Protocol

from twinr.agent.base_agent.conversation.closure import (
    ConversationClosureDecision,
    ConversationClosureEvaluation,
)

PROACTIVE_SOURCE = "proactive"
OBS_CLOSURE_EVENT = "conversation_closure"
OBS_CLOSURE_ERROR_EVENT = "conversation_closure_error"
OBS_SCHEMA_VERSION = 1
MAX_EMITTED_REASON_CHARS = 512

_TRUE_STRINGS = frozenset({"1", "true", "t", "yes", "y", "on", "enabled"})
_FALSE_STRINGS = frozenset({"0", "false", "f", "no", "n", "off", "disabled", ""})
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class LoopConfigProtocol(Protocol):
    conversation_follow_up_enabled: object
    conversation_follow_up_after_proactive_enabled: object


class FollowUpSteeringRuntimeProtocol(Protocol):
    def evaluate_closure(
        self,
        *,
        user_transcript: str,
        assistant_response: str,
        request_source: str,
        proactive_trigger: str | None,
    ) -> ConversationClosureEvaluation: ...

    def apply_closure_evaluation(
        self,
        *,
        evaluation: ConversationClosureEvaluation,
        request_source: str,
        proactive_trigger: str | None,
    ) -> ConversationClosureDecision: ...


class LoopProtocol(Protocol):
    config: LoopConfigProtocol
    runtime: object
    follow_up_steering_runtime: FollowUpSteeringRuntimeProtocol

    def emit(self, event: str) -> None: ...


def _coerce_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
        return default
    try:
        return bool(value)
    except Exception:
        return default


def _read_bool_attr(owner: object, attr_name: str, *, default: bool) -> bool:
    if owner is None:
        return default
    try:
        value = getattr(owner, attr_name)
    except Exception:
        return default
    return _coerce_bool(value, default=default)


def _read_bool_attr_or_predicate(
    owner: object,
    attr_name: str,
    *,
    default_missing: bool,
    default_error: bool,
) -> bool:
    if owner is None:
        return default_missing
    try:
        value = getattr(owner, attr_name)
    except Exception:
        return default_error
    if value is None:
        return default_missing
    if callable(value):
        try:
            value = value()
        except Exception:
            return default_error
    return _coerce_bool(value, default=default_error)


def _normalize_source(source: object) -> str:
    value = getattr(source, "value", source)
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    return str(value).strip().lower()


def _normalize_reason(reason: object) -> str:
    if reason is None:
        text = ""
    elif isinstance(reason, bytes):
        text = reason.decode("utf-8", "replace")
    else:
        text = str(reason)

    text = text.replace("\r", "\n").replace("\t", " ")
    text = _CONTROL_CHAR_RE.sub("", text)
    text = " ".join(text.split())

    if len(text) > MAX_EMITTED_REASON_CHARS:
        text = text[: MAX_EMITTED_REASON_CHARS - 1].rstrip() + "…"
    return text


def _normalize_follow_up_action(value: object) -> str | None:
    text = _normalize_reason(value)
    if text in {"continue", "end"}:
        return text
    return None


def _coerce_confidence(value: object, *, default: float = 0.0) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(confidence):
        return default
    return confidence


def _emit_structured_event(loop: LoopProtocol, event_name: str, payload: dict[str, object]) -> None:
    emitter = getattr(loop, "emit", None)
    if not callable(emitter):
        return
    try:
        emitter(
            f"{event_name}="
            f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'), sort_keys=True)}"
        )
    except Exception:
        return


def _emit_closure_error(loop: LoopProtocol, *, phase: str, error: Exception) -> None:
    _emit_structured_event(
        loop,
        OBS_CLOSURE_ERROR_EVENT,
        {
            "schema_version": OBS_SCHEMA_VERSION,
            "phase": phase,
            "error_type": type(error).__name__,
            "error": _normalize_reason(error),
        },
    )


def _get_steering_runtime(loop: LoopProtocol) -> FollowUpSteeringRuntimeProtocol:
    steering_runtime = getattr(loop, "follow_up_steering_runtime", None)
    if steering_runtime is None:
        raise AttributeError("loop.follow_up_steering_runtime is required")
    return steering_runtime


def follow_up_allowed_for_source(loop: LoopProtocol, *, initial_source: str) -> bool:
    """Report whether the given turn source may reopen a follow-up listen."""

    runtime = getattr(loop, "runtime", None)
    if _read_bool_attr_or_predicate(
        runtime,
        "voice_quiet_active",
        default_missing=False,
        default_error=True,
    ):
        return False

    config = getattr(loop, "config", None)
    if not _read_bool_attr(config, "conversation_follow_up_enabled", default=False):
        return False

    if _normalize_source(initial_source) == PROACTIVE_SOURCE:
        return _read_bool_attr(
            config,
            "conversation_follow_up_after_proactive_enabled",
            default=False,
        )

    return True


def evaluate_follow_up_closure(
    loop: LoopProtocol,
    *,
    user_transcript: str,
    assistant_response: str,
    request_source: str,
    proactive_trigger: str | None,
) -> ConversationClosureEvaluation:
    """Evaluate whether the just-finished turn should force-close follow-up."""

    steering_runtime = _get_steering_runtime(loop)
    return steering_runtime.evaluate_closure(
        user_transcript=user_transcript,
        assistant_response=assistant_response,
        request_source=request_source,
        proactive_trigger=proactive_trigger,
    )


def apply_follow_up_closure_evaluation(
    loop: LoopProtocol,
    *,
    evaluation: ConversationClosureEvaluation,
    request_source: str,
    proactive_trigger: str | None,
) -> bool:
    """Apply one closure evaluation and return whether follow-up is vetoed.

    This helper fails closed: if the steering runtime cannot apply the evaluation,
    the follow-up listen is vetoed instead of reopening the microphone.
    """

    try:
        steering_runtime = _get_steering_runtime(loop)
        decision = steering_runtime.apply_closure_evaluation(
            evaluation=evaluation,
            request_source=request_source,
            proactive_trigger=proactive_trigger,
        )
    except Exception as error:
        _emit_closure_error(loop, phase="apply", error=error)
        return True

    return _read_bool_attr(decision, "force_close", default=True)


def follow_up_vetoed_by_closure(
    loop: LoopProtocol,
    *,
    user_transcript: str,
    assistant_response: str,
    request_source: str,
    proactive_trigger: str | None,
) -> bool:
    """Evaluate and apply closure steering for one finished answer.

    This helper fails closed: if evaluation or application errors occur, it vetoes
    follow-up to avoid unintended reopen-listen behaviour.
    """

    try:
        evaluation = evaluate_follow_up_closure(
            loop,
            user_transcript=user_transcript,
            assistant_response=assistant_response,
            request_source=request_source,
            proactive_trigger=proactive_trigger,
        )
    except Exception as error:
        _emit_closure_error(loop, phase="evaluate", error=error)
        return True

    return apply_follow_up_closure_evaluation(
        loop,
        evaluation=evaluation,
        request_source=request_source,
        proactive_trigger=proactive_trigger,
    )


def emit_closure_decision(loop: LoopProtocol, decision: ConversationClosureDecision) -> None:
    """Emit the normalized closure decision fields for observability."""

    close_now = _read_bool_attr(
        decision,
        "close_now",
        default=_read_bool_attr(decision, "force_close", default=False),
    )
    force_close = _read_bool_attr(decision, "force_close", default=close_now)
    confidence = round(_coerce_confidence(getattr(decision, "confidence", 0.0)), 6)
    reason = _normalize_reason(getattr(decision, "reason", ""))
    follow_up_action = _normalize_follow_up_action(
        getattr(decision, "follow_up_action", None)
    )

    emitter = getattr(loop, "emit", None)
    if callable(emitter):
        try:
            emitter(f"conversation_closure_close_now={str(close_now).lower()}")
            emitter(f"conversation_closure_confidence={confidence:.3f}")
            emitter(f"conversation_closure_reason={reason}")
            if follow_up_action is not None:
                emitter(f"conversation_closure_follow_up_action={follow_up_action}")
        except Exception:
            return

    trace = getattr(loop, "_trace_event", None)
    if callable(trace):
        try:
            trace(
                OBS_CLOSURE_EVENT,
                kind="decision",
                details={
                    "schema_version": OBS_SCHEMA_VERSION,
                    "close_now": close_now,
                    "force_close": force_close,
                    "confidence": confidence,
                    "reason": reason,
                    "follow_up_action": follow_up_action,
                },
            )
        except Exception:
            return


__all__ = [
    "FollowUpSteeringRuntimeProtocol",
    "LoopConfigProtocol",
    "LoopProtocol",
    "apply_follow_up_closure_evaluation",
    "emit_closure_decision",
    "evaluate_follow_up_closure",
    "follow_up_allowed_for_source",
    "follow_up_vetoed_by_closure",
]
