from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.contracts import ConversationLike, ToolCallingAgentProvider
from twinr.agent.base_agent.personality import load_conversation_closure_instructions
from twinr.agent.base_agent.turn_controller import (
    _coerce_probability,
    _coerce_text,
    _compact_conversation,
    _config_float,
    _config_int,
    _extract_json_object,
)

_DEFAULT_CONTEXT_TURNS = 4
_DEFAULT_MAX_TRANSCRIPT_CHARS = 512
_DEFAULT_MAX_RESPONSE_CHARS = 512
_DEFAULT_MAX_REASON_CHARS = 256
_DEFAULT_PROVIDER_TIMEOUT_SECONDS = 2.0

_CLOSURE_DECISION_TOOL_SCHEMA: dict[str, object] = {
    "type": "function",
    "name": "submit_closure_decision",
    "description": "Decide whether Twinr should stop automatic follow-up listening after the just-finished exchange.",
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "close_now": {
                "type": "boolean",
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "reason": {
                "type": "string",
            },
        },
        "required": ["close_now", "confidence", "reason"],
    },
}


@dataclass(frozen=True, slots=True)
class ConversationClosureDecision:
    close_now: bool
    confidence: float
    reason: str


class ToolCallingConversationClosureEvaluator:
    def __init__(
        self,
        *,
        config: TwinrConfig,
        provider: ToolCallingAgentProvider,
    ) -> None:
        self.config = config
        self.provider = provider
        self._provider_timeout_kwarg_name = self._detect_provider_timeout_kwarg()

    def evaluate(
        self,
        *,
        user_transcript: str,
        assistant_response: str,
        request_source: str,
        proactive_trigger: str | None = None,
        conversation: ConversationLike | None = None,
    ) -> ConversationClosureDecision:
        compact_conversation = _compact_conversation(
            conversation,
            max_turns=_config_int(
                self.config,
                "conversation_closure_context_turns",
                _DEFAULT_CONTEXT_TURNS,
                minimum=0,
                maximum=32,
            ),
            max_item_chars=_config_int(
                self.config,
                "conversation_closure_max_transcript_chars",
                _DEFAULT_MAX_TRANSCRIPT_CHARS,
                minimum=64,
                maximum=4096,
            ),
            max_total_chars=8192,
        )
        prompt = self._build_prompt(
            user_transcript=user_transcript,
            assistant_response=assistant_response,
            request_source=request_source,
            proactive_trigger=proactive_trigger,
            conversation=compact_conversation,
        )
        provider_kwargs: dict[str, object] = {}
        if self._provider_timeout_kwarg_name is not None:
            provider_kwargs[self._provider_timeout_kwarg_name] = _config_float(
                self.config,
                "conversation_closure_provider_timeout_seconds",
                _DEFAULT_PROVIDER_TIMEOUT_SECONDS,
                minimum=0.25,
                maximum=15.0,
            )
        response = self.provider.start_turn_streaming(
            prompt,
            conversation=compact_conversation,
            instructions=load_conversation_closure_instructions(self.config),
            tool_schemas=(_CLOSURE_DECISION_TOOL_SCHEMA,),
            allow_web_search=False,
            **provider_kwargs,
        )
        tool_calls = getattr(response, "tool_calls", ()) or ()
        for tool_call in tool_calls:
            if _coerce_text(getattr(tool_call, "name", ""), max_chars=64) != "submit_closure_decision":
                continue
            raw_arguments = getattr(tool_call, "arguments", {})
            if isinstance(raw_arguments, dict):
                payload = raw_arguments
            else:
                payload = _extract_json_object(raw_arguments) or {}
            return self._coerce_decision(payload)
        payload = _extract_json_object(getattr(response, "text", "")) or {}
        return self._coerce_decision(payload)

    def _build_prompt(
        self,
        *,
        user_transcript: str,
        assistant_response: str,
        request_source: str,
        proactive_trigger: str | None,
        conversation: tuple[tuple[str, str], ...],
    ) -> str:
        payload = {
            "task": "Decide whether Twinr should suppress any automatic follow-up listening because the exchange has clearly ended for now.",
            "exchange": {
                "user_transcript": _coerce_text(
                    user_transcript,
                    max_chars=_config_int(
                        self.config,
                        "conversation_closure_max_transcript_chars",
                        _DEFAULT_MAX_TRANSCRIPT_CHARS,
                        minimum=64,
                        maximum=4096,
                    ),
                ),
                "assistant_response": _coerce_text(
                    assistant_response,
                    max_chars=_config_int(
                        self.config,
                        "conversation_closure_max_response_chars",
                        _DEFAULT_MAX_RESPONSE_CHARS,
                        minimum=64,
                        maximum=4096,
                    ),
                ),
                "request_source": _coerce_text(request_source, default="button", max_chars=64) or "button",
                "proactive_trigger": _coerce_text(proactive_trigger, max_chars=128) or None,
                "recent_turn_count": len(conversation),
            },
        }
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))

    def _coerce_decision(self, payload: dict[str, object]) -> ConversationClosureDecision:
        return ConversationClosureDecision(
            close_now=bool(payload.get("close_now", False)),
            confidence=_coerce_probability(payload.get("confidence", 0.0), default=0.0),
            reason=_coerce_text(
                payload.get("reason", ""),
                default="closure_controller_fallback",
                max_chars=_config_int(
                    self.config,
                    "conversation_closure_max_reason_chars",
                    _DEFAULT_MAX_REASON_CHARS,
                    minimum=32,
                    maximum=1024,
                ),
            )
            or "closure_controller_fallback",
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
