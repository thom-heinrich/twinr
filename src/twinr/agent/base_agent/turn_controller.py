from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Literal, Protocol
import inspect
import json
import math
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.contracts import (
    ConversationLike,
    StreamingSpeechEndpointEvent,
    ToolCallingAgentProvider,
)
from twinr.agent.base_agent.personality import load_turn_controller_instructions

TurnDecisionName = Literal["continue_listening", "end_turn"]

_DEFAULT_CONTEXT_TURNS = 8
_DEFAULT_MAX_TRANSCRIPT_CHARS = 4096
_DEFAULT_MAX_REASON_CHARS = 256
_DEFAULT_MAX_CONVERSATION_ITEM_CHARS = 2000
_DEFAULT_MAX_CONVERSATION_TOTAL_CHARS = 12000
_DEFAULT_PROVIDER_TIMEOUT_SECONDS = 2.0
_DEFAULT_EVALUATION_TIMEOUT_SECONDS = 2.5
_DEFAULT_CIRCUIT_BREAKER_FAILURES = 3
_DEFAULT_CIRCUIT_BREAKER_COOLDOWN_SECONDS = 15.0
_DEFAULT_EMIT_VALUE_MAX_CHARS = 256

_TURN_DECISION_TOOL_SCHEMA: dict[str, object] = {
    "type": "function",
    "name": "submit_turn_decision",
    "description": "Submit the current turn-boundary decision for the active user utterance.",
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["continue_listening", "end_turn"],
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "reason": {
                "type": "string",
            },
            "transcript": {
                "type": "string",
            },
        },
        "required": ["decision", "confidence", "reason", "transcript"],
    },
}


def _safe_getattr(obj: object, name: str, default: object = None) -> object:
    # AUDIT-FIX(#3): Treat external event/provider objects as untrusted and avoid attribute access crashes.
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _coerce_text(
    value: object,
    *,
    default: str = "",
    max_chars: int | None = None,
) -> str:
    # AUDIT-FIX(#3): Normalize external values defensively so None/non-str payloads do not explode inside callbacks.
    if value is None:
        text = default
    else:
        try:
            text = str(value)
        except Exception:
            text = default
    text = text.strip()
    if max_chars is not None and max_chars >= 0 and len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def _coerce_bool(value: object, *, default: bool) -> bool:
    # AUDIT-FIX(#4): Parse env-derived booleans safely instead of relying on Python truthiness for strings like "false".
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    normalized = _coerce_text(value).lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _coerce_int(
    value: object,
    *,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    # AUDIT-FIX(#4): Clamp integer config values so malformed env settings do not raise or create absurd bounds.
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    if minimum is not None:
        number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return number


def _coerce_float(
    value: object,
    *,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    # AUDIT-FIX(#4): Reject malformed and non-finite floats from config/provider responses.
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if not math.isfinite(number):
        number = default
    if minimum is not None:
        number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return number


def _coerce_probability(value: object, *, default: float) -> float:
    # AUDIT-FIX(#4): NaN/Inf must never become fake confidence values like 1.0.
    return _coerce_float(value, default=default, minimum=0.0, maximum=1.0)


def _config_bool(config: object, name: str, default: bool) -> bool:
    return _coerce_bool(_safe_getattr(config, name, default), default=default)


def _config_int(
    config: object,
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    return _coerce_int(_safe_getattr(config, name, default), default=default, minimum=minimum, maximum=maximum)


def _config_float(
    config: object,
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    return _coerce_float(_safe_getattr(config, name, default), default=default, minimum=minimum, maximum=maximum)


def _sanitize_emit_value(value: object, *, max_chars: int) -> str:
    # AUDIT-FIX(#7): Strip control characters so model/output text cannot forge extra log or SSE-style lines.
    sanitized = _coerce_text(value, max_chars=max_chars).replace("\r", " ").replace("\n", " ")
    return " ".join(sanitized.split())


def _extract_json_object(text: object) -> dict[str, object] | None:
    # AUDIT-FIX(#10): Accept fenced/prose-wrapped JSON because LLM fallbacks are often messy in practice.
    raw = _coerce_text(text)
    if not raw:
        return None

    candidates: list[str] = [raw]
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines:
            inner_lines = lines[1:]
            if inner_lines and inner_lines[-1].strip().startswith("```"):
                inner_lines = inner_lines[:-1]
            fenced_body = "\n".join(inner_lines).strip()
            if fenced_body:
                candidates.append(fenced_body)

    start_index = raw.find("{")
    end_index = raw.rfind("}")
    if start_index != -1 and end_index > start_index:
        candidates.append(raw[start_index : end_index + 1])

    seen: set[str] = set()
    for candidate in candidates:
        stripped = candidate.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        try:
            payload = json.loads(stripped)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _compact_conversation(
    conversation: ConversationLike | None,
    *,
    max_turns: int,
    max_item_chars: int,
    max_total_chars: int,
) -> tuple[tuple[str, str], ...]:
    if not conversation:
        return ()
    # AUDIT-FIX(#9): Snapshot conversation best-effort so concurrent mutation does not crash turn evaluation.
    try:
        turns = list(conversation)
    except Exception:
        return ()
    if max_turns > 0 and len(turns) > max_turns:
        turns = turns[-max_turns:]
    compacted: list[tuple[str, str]] = []
    total_chars = 0
    for item in turns:
        try:
            if isinstance(item, tuple) and len(item) == 2:
                role, content = item
            else:
                role = _safe_getattr(item, "role", "")
                content = _safe_getattr(item, "content", "")
        except Exception:
            continue
        role_text = _coerce_text(role, default="user", max_chars=64) or "user"
        content_text = _coerce_text(content, max_chars=max_item_chars)
        if not content_text:
            continue
        projected_chars = total_chars + len(role_text) + len(content_text)
        if max_total_chars > 0 and projected_chars > max_total_chars:
            remaining_chars = max_total_chars - total_chars - len(role_text)
            if remaining_chars <= 0:
                break
            content_text = content_text[:remaining_chars].rstrip()
            if not content_text:
                break
            compacted.append((role_text, content_text))
            break
        compacted.append((role_text, content_text))
        total_chars = projected_chars
    return tuple(compacted)


@dataclass(frozen=True, slots=True)
class TurnEvaluationCandidate:
    transcript: str
    event_type: str
    request_id: str | None = None
    observed_at_monotonic: float = 0.0
    is_final: bool = False
    speech_final: bool = False
    from_finalize: bool = False
    source_revision: int = 0

    @classmethod
    def from_endpoint_event(
        cls,
        event: StreamingSpeechEndpointEvent,
        *,
        transcript: str,
        source_revision: int = 0,
    ) -> "TurnEvaluationCandidate":
        return cls(
            transcript=_coerce_text(transcript),
            event_type=_coerce_text(_safe_getattr(event, "event_type", "endpoint"), default="endpoint", max_chars=64) or "endpoint",
            request_id=_coerce_text(_safe_getattr(event, "request_id", ""), max_chars=128) or None,
            observed_at_monotonic=time.monotonic(),
            is_final=bool(_safe_getattr(event, "is_final", False)),
            speech_final=bool(_safe_getattr(event, "speech_final", False)),
            from_finalize=bool(_safe_getattr(event, "from_finalize", False)),
            source_revision=source_revision,  # AUDIT-FIX(#2): Track input revision so stale decisions can be discarded safely.
        )


@dataclass(frozen=True, slots=True)
class TurnDecision:
    decision: TurnDecisionName
    confidence: float
    reason: str
    transcript: str


class TurnDecisionEvaluator(Protocol):
    def evaluate(
        self,
        *,
        candidate: TurnEvaluationCandidate,
        conversation: ConversationLike | None = None,
    ) -> TurnDecision:
        ...


class ToolCallingTurnDecisionEvaluator:
    def __init__(
        self,
        *,
        config: TwinrConfig,
        provider: ToolCallingAgentProvider,
    ) -> None:
        self.config = config
        self.provider = provider
        # AUDIT-FIX(#5): Keep lightweight provider failure state locally so intermittent Wi-Fi does not thrash the backend.
        self._state_lock = Lock()
        self._consecutive_failures = 0
        self._circuit_open_until_monotonic = 0.0
        self._provider_timeout_kwarg_name = self._detect_provider_timeout_kwarg()

    def evaluate(
        self,
        *,
        candidate: TurnEvaluationCandidate,
        conversation: ConversationLike | None = None,
    ) -> TurnDecision:
        fallback_transcript = _coerce_text(
            candidate.transcript,
            max_chars=_config_int(
                self.config,
                "turn_controller_max_transcript_chars",
                _DEFAULT_MAX_TRANSCRIPT_CHARS,
                minimum=64,
                maximum=32768,
            ),
        )
        if self._is_circuit_open():
            return self._fallback_decision(
                reason="turn_controller_circuit_open",
                transcript=fallback_transcript,
            )
        compact_conversation = _compact_conversation(
            conversation,
            max_turns=_config_int(
                self.config,
                "turn_controller_context_turns",
                _DEFAULT_CONTEXT_TURNS,
                minimum=0,
                maximum=128,
            ),
            max_item_chars=_config_int(
                self.config,
                "turn_controller_max_conversation_item_chars",
                _DEFAULT_MAX_CONVERSATION_ITEM_CHARS,
                minimum=128,
                maximum=8192,
            ),
            max_total_chars=_config_int(
                self.config,
                "turn_controller_max_conversation_total_chars",
                _DEFAULT_MAX_CONVERSATION_TOTAL_CHARS,
                minimum=512,
                maximum=65536,
            ),
        )
        prompt = self._build_prompt(candidate)
        provider_kwargs: dict[str, object] = {}
        if self._provider_timeout_kwarg_name is not None:
            provider_kwargs[self._provider_timeout_kwarg_name] = _config_float(
                self.config,
                "turn_controller_provider_timeout_seconds",
                _DEFAULT_PROVIDER_TIMEOUT_SECONDS,
                minimum=0.25,
                maximum=30.0,
            )

        try:
            response = self.provider.start_turn_streaming(
                prompt,
                conversation=compact_conversation,
                instructions=load_turn_controller_instructions(self.config),
                tool_schemas=(_TURN_DECISION_TOOL_SCHEMA,),
                allow_web_search=False,
                **provider_kwargs,
            )
        except Exception as exc:
            self._note_failure()
            return self._fallback_decision(
                reason=f"turn_controller_error:{type(exc).__name__}",
                transcript=fallback_transcript,
            )

        self._note_success()

        tool_calls = _safe_getattr(response, "tool_calls", ()) or ()
        for tool_call in tool_calls:
            tool_name = _coerce_text(_safe_getattr(tool_call, "name", ""), max_chars=64)
            if tool_name != "submit_turn_decision":
                continue
            raw_arguments = _safe_getattr(tool_call, "arguments", {})
            if isinstance(raw_arguments, dict):
                payload = raw_arguments
            else:
                payload = _extract_json_object(raw_arguments) or {}
            return self._coerce_decision(payload, fallback_transcript=fallback_transcript)

        return self._parse_text_fallback(
            _safe_getattr(response, "text", ""),
            fallback_transcript=fallback_transcript,
        )

    def _build_prompt(self, candidate: TurnEvaluationCandidate) -> str:
        payload = {
            "task": "Decide whether the active user turn should end now or continue listening.",
            "candidate": {
                "transcript": _coerce_text(
                    candidate.transcript,
                    max_chars=_config_int(
                        self.config,
                        "turn_controller_max_transcript_chars",
                        _DEFAULT_MAX_TRANSCRIPT_CHARS,
                        minimum=64,
                        maximum=32768,
                    ),
                ),
                "event_type": _coerce_text(candidate.event_type, default="endpoint", max_chars=64) or "endpoint",
                "request_id": _coerce_text(candidate.request_id, max_chars=128) or None,
                "is_final": bool(candidate.is_final),
                "speech_final": bool(candidate.speech_final),
                "from_finalize": bool(candidate.from_finalize),
            },
        }
        # AUDIT-FIX(#9): Use compact JSON so prompt tokens stay bounded on a Raspberry Pi class device.
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))

    def _coerce_decision(
        self,
        payload: dict[str, object],
        *,
        fallback_transcript: str,
    ) -> TurnDecision:
        raw_decision = _coerce_text(payload.get("decision", ""), max_chars=32).lower()
        decision: TurnDecisionName = "continue_listening"
        if raw_decision == "end_turn":
            decision = "end_turn"
        # AUDIT-FIX(#4): Clamp and validate confidence with finite-number checks.
        confidence_value = _coerce_probability(payload.get("confidence", 0.0), default=0.0)
        reason = _coerce_text(
            payload.get("reason", ""),
            max_chars=_config_int(
                self.config,
                "turn_controller_max_reason_chars",
                _DEFAULT_MAX_REASON_CHARS,
                minimum=32,
                maximum=2048,
            ),
        ) or "unspecified"
        transcript = _coerce_text(
            payload.get("transcript", ""),
            max_chars=_config_int(
                self.config,
                "turn_controller_max_transcript_chars",
                _DEFAULT_MAX_TRANSCRIPT_CHARS,
                minimum=64,
                maximum=32768,
            ),
        ) or fallback_transcript.strip()
        return TurnDecision(
            decision=decision,
            confidence=confidence_value,
            reason=reason,
            transcript=transcript,
        )

    def _parse_text_fallback(
        self,
        text: object,
        *,
        fallback_transcript: str,
    ) -> TurnDecision:
        payload = _extract_json_object(text)
        if payload is None:
            return self._fallback_decision(
                reason="no_structured_turn_decision",
                transcript=fallback_transcript,
            )
        return self._coerce_decision(payload, fallback_transcript=fallback_transcript)

    def _fallback_decision(self, *, reason: str, transcript: str) -> TurnDecision:
        return TurnDecision(
            decision="continue_listening",
            confidence=0.0,
            reason=_coerce_text(
                reason,
                default="turn_controller_fallback",
                max_chars=_config_int(
                    self.config,
                    "turn_controller_max_reason_chars",
                    _DEFAULT_MAX_REASON_CHARS,
                    minimum=32,
                    maximum=2048,
                ),
            ) or "turn_controller_fallback",
            transcript=_coerce_text(
                transcript,
                max_chars=_config_int(
                    self.config,
                    "turn_controller_max_transcript_chars",
                    _DEFAULT_MAX_TRANSCRIPT_CHARS,
                    minimum=64,
                    maximum=32768,
                ),
            ),
        )

    def _detect_provider_timeout_kwarg(self) -> str | None:
        try:
            signature = inspect.signature(self.provider.start_turn_streaming)
        except (TypeError, ValueError):
            return None
        parameter_names = set(signature.parameters)
        if "timeout_seconds" in parameter_names:
            return "timeout_seconds"
        if "timeout" in parameter_names:
            return "timeout"
        if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
            return "timeout_seconds"
        return None

    def _is_circuit_open(self) -> bool:
        # AUDIT-FIX(#5): Simple cooldown breaker prevents repeated failing calls on flaky home Wi-Fi.
        with self._state_lock:
            return self._circuit_open_until_monotonic > time.monotonic()

    def _note_success(self) -> None:
        with self._state_lock:
            self._consecutive_failures = 0
            self._circuit_open_until_monotonic = 0.0

    def _note_failure(self) -> None:
        failure_threshold = _config_int(
            self.config,
            "turn_controller_circuit_breaker_failures",
            _DEFAULT_CIRCUIT_BREAKER_FAILURES,
            minimum=1,
            maximum=100,
        )
        cooldown_seconds = _config_float(
            self.config,
            "turn_controller_circuit_breaker_cooldown_seconds",
            _DEFAULT_CIRCUIT_BREAKER_COOLDOWN_SECONDS,
            minimum=1.0,
            maximum=3600.0,
        )
        with self._state_lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= failure_threshold:
                self._circuit_open_until_monotonic = time.monotonic() + cooldown_seconds


class StreamingTurnController:
    def __init__(
        self,
        *,
        config: TwinrConfig,
        evaluator: TurnDecisionEvaluator,
        conversation_factory: Callable[[], ConversationLike | None],
        emit: Callable[[str], None] | None = None,
    ) -> None:
        self._config = config
        self._evaluator = evaluator
        self._conversation_factory = conversation_factory
        self._emit = emit
        self._lock = Lock()
        self._latest_partial = ""
        self._latest_partial_normalized = ""
        self._pending_candidate: TurnEvaluationCandidate | None = None
        self._last_decision: TurnDecision | None = None
        self._evaluation_inflight = False
        self._stop_requested = False
        self._closed = False
        self._state_revision = 0
        # AUDIT-FIX(#5): Track an outstanding watchdog worker so a hung evaluator call cannot spawn unbounded threads.
        self._evaluator_worker_active = False

    def on_interim(self, transcript: str) -> None:
        cleaned = _coerce_text(
            transcript,
            max_chars=_config_int(
                self._config,
                "turn_controller_max_transcript_chars",
                _DEFAULT_MAX_TRANSCRIPT_CHARS,
                minimum=64,
                maximum=32768,
            ),
        )
        if not cleaned:
            return
        with self._lock:
            # AUDIT-FIX(#6): Ignore late interim updates after shutdown/stop so final state cannot be corrupted.
            if self._closed or self._stop_requested:
                return
            self._state_revision += 1  # AUDIT-FIX(#2): Any interim update invalidates older candidate decisions.
            self._latest_partial = cleaned
            self._latest_partial_normalized = _normalize_turn_text(cleaned)

    def on_endpoint(self, event: StreamingSpeechEndpointEvent) -> None:
        emit_messages: list[str] = []
        should_start_runner = False
        transcript = _coerce_text(
            _safe_getattr(event, "transcript", ""),
            max_chars=_config_int(
                self._config,
                "turn_controller_max_transcript_chars",
                _DEFAULT_MAX_TRANSCRIPT_CHARS,
                minimum=64,
                maximum=32768,
            ),
        )
        fallback_transcript = ""
        with self._lock:
            # AUDIT-FIX(#6): Ignore endpoint callbacks that arrive after closure or once capture should already stop.
            if self._closed or self._stop_requested:
                return
            previous_partial = self._latest_partial
            previous_partial_normalized = self._latest_partial_normalized
            if not transcript:
                transcript = previous_partial
            transcript = _coerce_text(
                transcript,
                max_chars=_config_int(
                    self._config,
                    "turn_controller_max_transcript_chars",
                    _DEFAULT_MAX_TRANSCRIPT_CHARS,
                    minimum=64,
                    maximum=32768,
                ),
            )
            if not transcript:
                return
            next_revision = self._state_revision + 1
            fast_decision = self._fast_endpoint_decision_locked(
                event,
                transcript,
                previous_partial_normalized=previous_partial_normalized,
            )
            self._state_revision = next_revision
            self._latest_partial = transcript
            self._latest_partial_normalized = _normalize_turn_text(transcript)
            fallback_transcript = transcript
            if fast_decision is not None:
                self._last_decision = fast_decision
                self._latest_partial = fast_decision.transcript.strip()
                self._latest_partial_normalized = _normalize_turn_text(self._latest_partial)
                self._stop_requested = True
                emit_messages.extend(self._decision_emit_messages(fast_decision))
            else:
                candidate = TurnEvaluationCandidate.from_endpoint_event(
                    event,
                    transcript=transcript,
                    source_revision=next_revision,
                )
                self._pending_candidate = candidate
                emit_messages.append(self._format_emit("turn_controller_candidate", candidate.event_type))
                if not self._evaluation_inflight:
                    self._evaluation_inflight = True
                    should_start_runner = True

        if should_start_runner:
            try:
                Thread(
                    target=self._run_pending_evaluations,
                    daemon=True,
                    name="twinr-turn-controller",
                ).start()
            except Exception as exc:
                # AUDIT-FIX(#8): Reset inflight state if a worker thread cannot be created.
                start_failure_decision = TurnDecision(
                    decision="continue_listening",
                    confidence=0.0,
                    reason=f"turn_controller_runner_start_error:{type(exc).__name__}",
                    transcript=fallback_transcript,
                )
                with self._lock:
                    self._evaluation_inflight = False
                    self._pending_candidate = None
                    self._last_decision = start_failure_decision
                emit_messages.extend(self._decision_emit_messages(start_failure_decision))

        # AUDIT-FIX(#7): Emit after releasing the controller lock so emitters cannot deadlock capture callbacks.
        self._emit_many(emit_messages)

    def should_stop_capture(self) -> bool:
        with self._lock:
            return self._stop_requested

    def latest_transcript(self) -> str:
        with self._lock:
            # AUDIT-FIX(#2): Only treat the decision transcript as authoritative once capture is actually stopping.
            if (
                self._stop_requested
                and self._last_decision is not None
                and self._last_decision.transcript.strip()
            ):
                return self._last_decision.transcript.strip()
            return self._latest_partial.strip()

    def last_decision(self) -> TurnDecision | None:
        with self._lock:
            return self._last_decision

    def close(self) -> None:
        with self._lock:
            # AUDIT-FIX(#6): Closing the controller must stop capture and discard queued work immediately.
            self._closed = True
            self._stop_requested = True
            self._pending_candidate = None

    def _run_pending_evaluations(self) -> None:
        while True:
            with self._lock:
                if self._closed or self._stop_requested:
                    self._evaluation_inflight = False
                    return
                candidate = self._pending_candidate
                self._pending_candidate = None
            if candidate is None:
                with self._lock:
                    self._evaluation_inflight = False
                return

            decision = self._evaluate_candidate_with_watchdog(candidate)
            emit_messages: list[str] = []
            should_return = False

            with self._lock:
                # AUDIT-FIX(#2): Drop any decision that finishes after a newer state change or an external stop request.
                if self._closed or self._stop_requested:
                    self._evaluation_inflight = False
                    return
                if self._pending_candidate is not None:
                    continue
                if candidate.source_revision != self._state_revision:
                    self._evaluation_inflight = False
                    should_return = True
                    emit_messages.append(self._format_emit("turn_controller_reason", "stale_candidate_discarded"))
                else:
                    self._last_decision = decision
                    if decision.transcript.strip():
                        self._latest_partial = decision.transcript.strip()
                        # AUDIT-FIX(#11): Keep normalized mirror in sync whenever latest_partial changes.
                        self._latest_partial_normalized = _normalize_turn_text(self._latest_partial)
                    if decision.decision == "end_turn":
                        self._stop_requested = True
                    emit_messages.extend(self._decision_emit_messages(decision))

            self._emit_many(emit_messages)

            if should_return:
                return

    def _evaluate_candidate_with_watchdog(
        self,
        candidate: TurnEvaluationCandidate,
    ) -> TurnDecision:
        timeout_seconds = _config_float(
            self._config,
            "turn_controller_eval_timeout_seconds",
            _DEFAULT_EVALUATION_TIMEOUT_SECONDS,
            minimum=0.25,
            maximum=30.0,
        )

        with self._lock:
            # AUDIT-FIX(#5): Do not spawn another evaluator worker while a previous one is still stuck.
            if self._evaluator_worker_active:
                return TurnDecision(
                    decision="continue_listening",
                    confidence=0.0,
                    reason="turn_controller_worker_busy",
                    transcript=candidate.transcript,
                )
            self._evaluator_worker_active = True

        done = Event()
        decision_holder: list[TurnDecision] = []
        error_holder: list[BaseException] = []

        def _worker() -> None:
            try:
                decision_holder.append(
                    self._sanitize_decision(
                        self._evaluator.evaluate(
                            candidate=candidate,
                            conversation=self._conversation_factory(),
                        ),
                        fallback_transcript=candidate.transcript,
                    )
                )
            except Exception as exc:
                error_holder.append(exc)
            finally:
                with self._lock:
                    self._evaluator_worker_active = False
                done.set()

        try:
            Thread(
                target=_worker,
                daemon=True,
                name="twinr-turn-controller-evaluator",
            ).start()
        except Exception as exc:
            with self._lock:
                self._evaluator_worker_active = False
            return TurnDecision(
                decision="continue_listening",
                confidence=0.0,
                reason=f"turn_controller_worker_start_error:{type(exc).__name__}",
                transcript=candidate.transcript,
            )

        if not done.wait(timeout_seconds):
            return TurnDecision(
                decision="continue_listening",
                confidence=0.0,
                reason="turn_controller_timeout",
                transcript=candidate.transcript,
            )

        if error_holder:
            return TurnDecision(
                decision="continue_listening",
                confidence=0.0,
                reason=f"turn_controller_error:{type(error_holder[0]).__name__}",
                transcript=candidate.transcript,
            )

        if not decision_holder:
            return TurnDecision(
                decision="continue_listening",
                confidence=0.0,
                reason="turn_controller_no_result",
                transcript=candidate.transcript,
            )

        return decision_holder[0]

    def _sanitize_decision(
        self,
        decision: TurnDecision,
        *,
        fallback_transcript: str,
    ) -> TurnDecision:
        safe_decision: TurnDecisionName = "end_turn" if decision.decision == "end_turn" else "continue_listening"
        safe_transcript = _coerce_text(
            decision.transcript,
            max_chars=_config_int(
                self._config,
                "turn_controller_max_transcript_chars",
                _DEFAULT_MAX_TRANSCRIPT_CHARS,
                minimum=64,
                maximum=32768,
            ),
        ) or _coerce_text(
            fallback_transcript,
            max_chars=_config_int(
                self._config,
                "turn_controller_max_transcript_chars",
                _DEFAULT_MAX_TRANSCRIPT_CHARS,
                minimum=64,
                maximum=32768,
            ),
        )
        return TurnDecision(
            decision=safe_decision,
            confidence=_coerce_probability(decision.confidence, default=0.0),
            reason=_coerce_text(
                decision.reason,
                default="unspecified",
                max_chars=_config_int(
                    self._config,
                    "turn_controller_max_reason_chars",
                    _DEFAULT_MAX_REASON_CHARS,
                    minimum=32,
                    maximum=2048,
                ),
            ) or "unspecified",
            transcript=safe_transcript,
        )

    def _decision_emit_messages(self, decision: TurnDecision) -> list[str]:
        return [
            self._format_emit("turn_controller_decision", decision.decision),
            self._format_emit("turn_controller_confidence", f"{_coerce_probability(decision.confidence, default=0.0):.2f}"),
            self._format_emit("turn_controller_reason", decision.reason),
        ]

    def _emit_many(self, messages: list[str]) -> None:
        for message in messages:
            self._safe_emit(message)

    def _safe_emit(self, message: str) -> None:
        emit = self._emit
        if emit is None:
            return
        try:
            emit(message)
        except Exception:
            return

    def _format_emit(self, key: str, value: object) -> str:
        return f"{key}={_sanitize_emit_value(value, max_chars=_config_int(self._config, 'turn_controller_emit_value_max_chars', _DEFAULT_EMIT_VALUE_MAX_CHARS, minimum=32, maximum=4096))}"

    def _fast_endpoint_decision_locked(
        self,
        event: StreamingSpeechEndpointEvent,
        transcript: str,
        *,
        previous_partial_normalized: str,
    ) -> TurnDecision | None:
        if not _config_bool(self._config, "turn_controller_fast_endpoint_enabled", False):
            return None
        cleaned = _coerce_text(
            transcript,
            max_chars=_config_int(
                self._config,
                "turn_controller_max_transcript_chars",
                _DEFAULT_MAX_TRANSCRIPT_CHARS,
                minimum=64,
                maximum=32768,
            ),
        )
        if len(cleaned) < _config_int(
            self._config,
            "turn_controller_fast_endpoint_min_chars",
            1,
            minimum=1,
            maximum=32768,
        ):
            return None
        if bool(_safe_getattr(event, "speech_final", False)):
            return TurnDecision(
                decision="end_turn",
                confidence=1.0,
                reason="speech_final_fast_path",
                transcript=cleaned,
            )
        if (
            _coerce_text(_safe_getattr(event, "event_type", ""), max_chars=64) == "utterance_end"
            and _config_bool(self._config, "deepgram_streaming_stop_on_utterance_end", False)
        ):
            normalized = _normalize_turn_text(cleaned)
            # AUDIT-FIX(#1): Compare against the pre-endpoint partial, not the just-overwritten current value.
            if normalized and previous_partial_normalized and normalized == previous_partial_normalized:
                return TurnDecision(
                    decision="end_turn",
                    confidence=_coerce_probability(
                        _safe_getattr(
                            self._config,
                            "turn_controller_fast_endpoint_min_confidence",
                            0.0,
                        ),
                        default=0.0,
                    ),
                    reason="utterance_end_fast_path",
                    transcript=cleaned,
                )
        return None


def _normalize_turn_text(text: str) -> str:
    return " ".join(_coerce_text(text).lower().split())