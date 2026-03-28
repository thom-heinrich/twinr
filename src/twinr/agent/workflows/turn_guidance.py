"""Own turn-controller context shaping and label-specific guidance messages."""

from __future__ import annotations

# CHANGELOG: 2026-03-28
# BUG-1: context_for_turn_label() now enforces the configured context bounds; previously provider context could grow without limit and silently diverge from controller_conversation().
# BUG-2: malformed history items and invalid numeric config values no longer crash turn-guidance building; they are sanitized and traced with safe fallbacks.
# SEC-1: bounded provider-context packing closes a practical prompt-amplification / resource-exhaustion path on Raspberry Pi 4 deployments.
# IMP-1: added token/character-aware context packing with optional tiktoken support and a low-cost fallback estimator.
# IMP-2: added pluggable context-selector hooks so higher-level 2026 working-set / compaction / long-term-memory systems can inject dynamic pruning here.
# IMP-3: label guidance is now registry-driven, control-role aware (`developer` when supported, else `system`), and extensible from config.
# IMP-4: leading control messages are pinned and label-aware turns can reserve a minimum recent window so the latest prompt/answer pair is not accidentally dropped.

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Mapping

from twinr.agent.base_agent.conversation.turn_controller import StreamingTurnController


ConversationTurn = tuple[str, str]
Conversation = tuple[ConversationTurn, ...]


@dataclass(frozen=True, slots=True)
class TurnGuidanceContext:
    """Capture the bounded context selected for one guided turn."""

    conversation: Conversation
    max_context_turns: int
    available_context_turns: int
    selected_context_turns: int
    guidance_message_count: int
    turn_label: str | None
    max_context_chars: int = 0
    max_context_tokens: int = 0
    available_context_estimated_tokens: int = 0
    selected_context_estimated_tokens: int = 0
    dropped_context_turns: int = 0
    instruction_role: str = "system"
    selector_name: str = "recency_pack"


class TurnGuidanceRuntime:
    """Build bounded turn-controller context and label-aware guidance."""

    _PINNED_CONTROL_ROLES = frozenset({"system", "developer"})
    _GUIDANCE_LABEL_ALIASES = {
        "ack": "backchannel",
        "acknowledgement": "backchannel",
        "acknowledgment": "backchannel",
        "affirmation": "confirmation",
        "answer": "direct_answer",
        "barge_in": "interruption",
        "barge-in": "interruption",
        "confirm": "confirmation",
        "correction": "repair",
        "direct-answer": "direct_answer",
        "interrupt": "interruption",
        "interrupted": "interruption",
        "repair": "repair",
        "short_answer": "direct_answer",
        "yes_no": "confirmation",
        "yes-no": "confirmation",
    }

    def __init__(self, loop) -> None:
        self._loop = loop

    def controller_conversation(self) -> Conversation:
        """Return the bounded conversation slice used by the turn controller."""

        selected, meta = self._select_bounded_conversation(
            conversation=self._conversation_from_runtime("conversation_context"),
            budget_namespace="turn_controller",
            source_name="controller_conversation",
            turn_label=None,
            reserved_turns=0,
            reserved_chars=0,
            reserved_tokens=0,
            min_recent_turns=self._minimum_recent_turns("turn_controller", None),
        )
        self._trace_event(
            "turn_guidance_controller_context_gate",
            kind="decision",
            details={
                "available_turns": meta["available_turns"],
                "selected_turns": meta["selected_turns"],
                "dropped_turns": meta["dropped_turns"],
                "max_context_turns": meta["max_turns"],
                "max_context_chars": meta["max_chars"],
                "max_context_tokens": meta["max_tokens"],
                "selector_name": meta["selector_name"],
            },
            kpi={
                "selected_turns": meta["selected_turns"],
                "selected_estimated_tokens": meta["selected_tokens"],
            },
        )
        return selected

    def build_streaming_turn_controller(self) -> StreamingTurnController | None:
        """Build the bounded turn controller when the runtime gate is open."""

        loop = self._loop
        enabled = loop.turn_decision_evaluator is not None and self._bool_config("turn_controller_enabled", default=False)
        self._trace_event(
            "turn_guidance_controller_build_gate",
            kind="decision",
            details={"enabled": enabled},
        )
        if not enabled:
            return None
        return StreamingTurnController(
            config=loop.config,
            evaluator=loop.turn_decision_evaluator,
            conversation_factory=self.controller_conversation,
            emit=loop.emit,
        )

    def guidance_messages(self, turn_label: str | None) -> Conversation:
        """Return the label-specific guidance messages for one turn."""

        normalized = self._normalize_label(turn_label)
        if normalized is None:
            return ()
        return self._guidance_registry().get(normalized, ())

    def context_for_turn_label(self, turn_label: str | None) -> TurnGuidanceContext:
        """Return the provider conversation plus any label-specific guidance."""

        normalized = self._normalize_label(turn_label)
        guidance = self.guidance_messages(normalized)
        selected_base, meta = self._select_bounded_conversation(
            conversation=self._conversation_from_runtime("provider_conversation_context"),
            budget_namespace="turn_guidance",
            source_name="provider_conversation_context",
            turn_label=normalized,
            reserved_turns=len(guidance),
            reserved_chars=self._conversation_chars(guidance),
            reserved_tokens=self._estimate_conversation_tokens(guidance),
            min_recent_turns=self._minimum_recent_turns("turn_guidance", normalized),
        )
        conversation = self._compose_conversation(selected_base, guidance)
        context = TurnGuidanceContext(
            conversation=conversation,
            max_context_turns=meta["max_turns"],
            available_context_turns=meta["available_turns"],
            selected_context_turns=meta["selected_turns"],
            guidance_message_count=len(guidance),
            turn_label=normalized,
            max_context_chars=meta["max_chars"],
            max_context_tokens=meta["max_tokens"],
            available_context_estimated_tokens=meta["available_tokens"],
            selected_context_estimated_tokens=meta["selected_tokens"],
            dropped_context_turns=meta["dropped_turns"],
            instruction_role=self._instruction_role(),
            selector_name=meta["selector_name"],
        )
        self._trace_event(
            "turn_guidance_context_gate",
            kind="decision",
            details={
                "turn_label": context.turn_label,
                "available_turns": context.available_context_turns,
                "selected_turns": context.selected_context_turns,
                "guidance_message_count": context.guidance_message_count,
                "max_context_turns": context.max_context_turns,
                "max_context_chars": context.max_context_chars,
                "max_context_tokens": context.max_context_tokens,
                "selector_name": context.selector_name,
            },
            kpi={
                "conversation_turns": len(context.conversation),
                "selected_estimated_tokens": context.selected_context_estimated_tokens,
            },
        )
        return context

    def conversation_context_for_turn_label(self, turn_label: str | None) -> Conversation:
        """Return only the composed conversation payload for one turn label."""

        return self.context_for_turn_label(turn_label).conversation

    def _select_bounded_conversation(
        self,
        *,
        conversation: Conversation,
        budget_namespace: str,
        source_name: str,
        turn_label: str | None,
        reserved_turns: int,
        reserved_chars: int,
        reserved_tokens: int,
        min_recent_turns: int,
    ) -> tuple[Conversation, dict[str, int | str]]:
        max_turns, max_chars, max_tokens = self._context_limits(budget_namespace)

        selected = None
        selector_name = "recency_pack"

        selector = self._context_selector(source_name)
        if selector is not None:
            try:
                selected = self._coerce_conversation(
                    selector(
                        conversation=conversation,
                        max_turns=max_turns,
                        max_chars=max_chars,
                        max_tokens=max_tokens,
                        min_recent_turns=min_recent_turns,
                        turn_label=turn_label,
                        source=source_name,
                    ),
                    source_name=f"{selector_name}",
                )
                selector_name = getattr(selector, "__qualname__", getattr(selector, "__name__", selector_name))
            except Exception as exc:  # pragma: no cover - defensive integration path
                self._trace_event(
                    "turn_guidance_context_selector_error",
                    kind="error",
                    details={
                        "source": source_name,
                        "selector_name": getattr(selector, "__qualname__", getattr(selector, "__name__", type(selector).__name__)),
                        "error": repr(exc),
                    },
                )
                selected = None
                selector_name = "recency_pack"

        if selected is None:
            selected = self._recency_pack_with_pinned_prefix(
                conversation=conversation,
                max_turns=max_turns,
                max_chars=max_chars,
                max_tokens=max_tokens,
                reserved_turns=reserved_turns,
                reserved_chars=reserved_chars,
                reserved_tokens=reserved_tokens,
                min_recent_turns=min_recent_turns,
            )

        available_tokens = self._estimate_conversation_tokens(conversation)
        selected_tokens = self._estimate_conversation_tokens(selected)
        meta: dict[str, int | str] = {
            "available_turns": len(conversation),
            "selected_turns": len(selected),
            "dropped_turns": max(0, len(conversation) - len(selected)),
            "available_tokens": available_tokens,
            "selected_tokens": selected_tokens,
            "max_turns": max_turns,
            "max_chars": max_chars,
            "max_tokens": max_tokens,
            "selector_name": selector_name,
        }
        return selected, meta

    def _recency_pack_with_pinned_prefix(
        self,
        *,
        conversation: Conversation,
        max_turns: int,
        max_chars: int,
        max_tokens: int,
        reserved_turns: int,
        reserved_chars: int,
        reserved_tokens: int,
        min_recent_turns: int,
    ) -> Conversation:
        pinned_prefix, tail = self._split_pinned_prefix(conversation)
        pinned_chars = self._conversation_chars(pinned_prefix)
        pinned_tokens = self._estimate_conversation_tokens(pinned_prefix)

        effective_turns = self._reduce_budget(max_turns, reserved_turns + len(pinned_prefix), keep_one=bool(tail))
        effective_chars = self._reduce_budget(max_chars, reserved_chars + pinned_chars, keep_one=False)
        effective_tokens = self._reduce_budget(max_tokens, reserved_tokens + pinned_tokens, keep_one=False)

        packed_tail = self._recency_pack(
            tail,
            max_turns=effective_turns,
            max_chars=effective_chars,
            max_tokens=effective_tokens,
            min_turns=min_recent_turns,
        )
        return pinned_prefix + packed_tail

    def _recency_pack(
        self,
        conversation: Conversation,
        *,
        max_turns: int | None,
        max_chars: int | None,
        max_tokens: int | None,
        min_turns: int,
    ) -> Conversation:
        if not conversation:
            return ()

        selected_reversed: list[ConversationTurn] = []
        running_chars = 0
        running_tokens = 0

        for role, content in reversed(conversation):
            turn_chars = len(role) + len(content)
            turn_tokens = self._estimate_turn_tokens(role, content)

            if len(selected_reversed) >= max(1, min_turns):
                if max_turns is not None and len(selected_reversed) >= max_turns:
                    break
                if max_chars is not None and running_chars + turn_chars > max_chars:
                    break
                if max_tokens is not None and running_tokens + turn_tokens > max_tokens:
                    break

            selected_reversed.append((role, content))
            running_chars += turn_chars
            running_tokens += turn_tokens

        return tuple(reversed(selected_reversed))

    def _compose_conversation(self, base_conversation: Conversation, guidance: Conversation) -> Conversation:
        if not guidance:
            return base_conversation
        if base_conversation[-len(guidance) :] == guidance:
            return base_conversation
        return base_conversation + guidance

    def _conversation_from_runtime(self, method_name: str) -> Conversation:
        runtime = getattr(self._loop, "runtime", None)
        getter = getattr(runtime, method_name, None)
        if not callable(getter):
            self._trace_event(
                "turn_guidance_runtime_context_missing",
                kind="error",
                details={"method_name": method_name},
            )
            return ()
        try:
            raw_conversation = getter()
        except Exception as exc:  # pragma: no cover - defensive integration path
            self._trace_event(
                "turn_guidance_runtime_context_error",
                kind="error",
                details={"method_name": method_name, "error": repr(exc)},
            )
            return ()
        return self._coerce_conversation(raw_conversation, source_name=method_name)

    def _guidance_registry(self) -> dict[str, Conversation]:
        instruction_role = self._instruction_role()
        registry = self._default_guidance_registry(instruction_role)

        configured = getattr(getattr(self._loop, "config", None), "turn_guidance_label_messages", None)
        if isinstance(configured, Mapping):
            for label, payload in configured.items():
                normalized_label = self._normalize_label(label)
                if normalized_label is None:
                    continue
                messages = self._coerce_guidance_payload(payload, default_role=instruction_role)
                if messages:
                    registry[normalized_label] = messages
        return registry

    def _default_guidance_registry(self, instruction_role: str) -> dict[str, Conversation]:
        return {
            "backchannel": (
                (
                    instruction_role,
                    "The current user turn is a short backchannel or direct answer to the latest assistant prompt. "
                    "If you answer, keep it very short, direct, and do not restate the whole context. "
                    "Prefer a single sentence and avoid follow-up questions unless they are required to unblock the user.",
                ),
            ),
            "confirmation": (
                (
                    instruction_role,
                    "The current user turn is a brief confirmation or yes/no answer. "
                    "Acknowledge briefly and continue with the already established next step without re-explaining prior context.",
                ),
            ),
            "direct_answer": (
                (
                    instruction_role,
                    "The current user turn directly answers the assistant's latest question. "
                    "Use the answer, continue the task, and avoid restating the earlier conversation.",
                ),
            ),
            "interruption": (
                (
                    instruction_role,
                    "The user interrupted or barged in while the assistant was speaking. "
                    "Treat the interrupting content as higher priority than the assistant's unfinished answer. "
                    "Do not resume the old answer unless the user explicitly asks for it.",
                ),
            ),
            "repair": (
                (
                    instruction_role,
                    "The current user turn corrects or repairs earlier content. "
                    "Treat the latest correction as authoritative, acknowledge briefly, and continue from the corrected state.",
                ),
            ),
        }

    def _coerce_guidance_payload(self, payload: Any, *, default_role: str) -> Conversation:
        if payload is None:
            return ()

        items: Iterable[Any]
        if isinstance(payload, str):
            items = ((default_role, payload),)
        elif self._looks_like_turn(payload):
            items = (payload,)
        elif isinstance(payload, Iterable):
            items = payload
        else:
            return ()

        messages: list[ConversationTurn] = []
        for item in items:
            if isinstance(item, str):
                turn = (default_role, item)
            else:
                turn = self._coerce_turn(item, default_role=default_role)
            if turn is not None:
                messages.append(turn)
        return tuple(messages)

    def _coerce_conversation(self, conversation: Any, *, source_name: str) -> Conversation:
        if conversation is None:
            return ()

        items: Iterable[Any]
        if self._looks_like_turn(conversation):
            items = (conversation,)
        elif isinstance(conversation, Iterable):
            items = conversation
        else:
            self._trace_event(
                "turn_guidance_invalid_conversation_container",
                kind="error",
                details={"source_name": source_name, "container_type": type(conversation).__name__},
            )
            return ()

        normalized: list[ConversationTurn] = []
        dropped = 0
        for item in items:
            turn = self._coerce_turn(item)
            if turn is None:
                dropped += 1
                continue
            normalized.append(turn)

        if dropped:
            self._trace_event(
                "turn_guidance_dropped_invalid_turns",
                kind="decision",
                details={"source_name": source_name, "dropped_turns": dropped},
                kpi={"dropped_turns": dropped},
            )
        return tuple(normalized)

    def _coerce_turn(self, item: Any, *, default_role: str | None = None) -> ConversationTurn | None:
        role: Any
        content: Any

        if isinstance(item, Mapping):
            role = item.get("role", default_role)
            content = item.get("content", "")
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            role, content = item[0], item[1]
        else:
            return None

        role_str = str(role or default_role or "").strip().lower()
        if not role_str:
            return None

        if isinstance(content, (tuple, list, dict)):
            content_str = str(content)
        else:
            content_str = "" if content is None else str(content)

        return (role_str, content_str)

    def _looks_like_turn(self, value: Any) -> bool:
        if isinstance(value, Mapping):
            return "role" in value and "content" in value
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            return False
        role, _content = value
        return not isinstance(role, (tuple, list, dict))

    def _normalize_label(self, turn_label: str | None) -> str | None:
        normalized = str(turn_label or "").strip().lower().replace(" ", "_")
        if not normalized:
            return None
        return self._GUIDANCE_LABEL_ALIASES.get(normalized, normalized)

    def _minimum_recent_turns(self, budget_namespace: str, turn_label: str | None) -> int:
        default = 1
        if budget_namespace == "turn_guidance" and turn_label in {"backchannel", "confirmation", "direct_answer"}:
            default = 2
        return self._int_config(
            f"{budget_namespace}_min_recent_turns",
            fallback_name="turn_controller_min_recent_turns" if budget_namespace == "turn_guidance" else None,
            default=default,
            minimum=1,
        )

    def _context_limits(self, budget_namespace: str) -> tuple[int, int, int]:
        if budget_namespace == "turn_guidance":
            fallback_prefix = "turn_controller"
        else:
            fallback_prefix = None

        max_turns = self._int_config(
            f"{budget_namespace}_context_turns",
            fallback_name=f"{fallback_prefix}_context_turns" if fallback_prefix else None,
            default=0,
            minimum=0,
        )
        max_chars = self._int_config(
            f"{budget_namespace}_context_max_chars",
            fallback_name=f"{fallback_prefix}_context_max_chars" if fallback_prefix else None,
            default=0,
            minimum=0,
        )
        max_tokens = self._int_config(
            f"{budget_namespace}_context_max_tokens",
            fallback_name=f"{fallback_prefix}_context_max_tokens" if fallback_prefix else None,
            default=0,
            minimum=0,
        )
        return max_turns, max_chars, max_tokens

    def _context_selector(self, source_name: str):
        runtime = getattr(self._loop, "runtime", None)
        candidates = []
        if source_name == "controller_conversation":
            candidates.extend(
                (
                    getattr(runtime, "select_turn_controller_context", None),
                    getattr(runtime, "select_conversation_context", None),
                    getattr(self._loop, "select_turn_controller_context", None),
                )
            )
        else:
            candidates.extend(
                (
                    getattr(runtime, "select_turn_guidance_context", None),
                    getattr(runtime, "select_provider_conversation_context", None),
                    getattr(runtime, "select_conversation_context", None),
                    getattr(self._loop, "select_turn_guidance_context", None),
                )
            )

        for candidate in candidates:
            if callable(candidate):
                return candidate
        return None

    def _split_pinned_prefix(self, conversation: Conversation) -> tuple[Conversation, Conversation]:
        if not conversation:
            return (), ()

        index = 0
        while index < len(conversation) and conversation[index][0] in self._PINNED_CONTROL_ROLES:
            index += 1
        return conversation[:index], conversation[index:]

    def _instruction_role(self) -> str:
        explicit = str(
            getattr(getattr(self._loop, "config", None), "turn_guidance_instruction_role", "") or ""
        ).strip().lower()
        if explicit in {"developer", "system"}:
            return explicit

        for roles in self._supported_roles():
            if "developer" in roles:
                return "developer"
            if "system" in roles:
                return "system"
        return "system"

    def _supported_roles(self) -> list[set[str]]:
        role_sets: list[set[str]] = []
        for owner in (getattr(self._loop, "runtime", None), self._loop, getattr(self._loop, "config", None)):
            if owner is None:
                continue
            for attr_name in (
                "supported_roles",
                "provider_supported_roles",
                "input_roles",
                "message_roles",
            ):
                raw = getattr(owner, attr_name, None)
                if isinstance(raw, Mapping):
                    values = tuple(raw.keys()) + tuple(raw.values())
                elif isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)):
                    values = raw
                else:
                    continue
                role_sets.append({str(value).strip().lower() for value in values if str(value).strip()})
        return role_sets

    def _reduce_budget(self, budget: int, reserved: int, *, keep_one: bool) -> int | None:
        if budget <= 0:
            return None
        remaining = budget - max(0, reserved)
        if keep_one:
            return max(1, remaining)
        return max(0, remaining)

    def _conversation_chars(self, conversation: Conversation) -> int:
        return sum(len(role) + len(content) for role, content in conversation)

    def _estimate_conversation_tokens(self, conversation: Conversation) -> int:
        return sum(self._estimate_turn_tokens(role, content) for role, content in conversation)

    def _estimate_turn_tokens(self, role: str, content: str) -> int:
        encoder = self._tiktoken_encoder()
        payload = f"{role}\n{content}"
        if encoder is not None:
            try:
                return max(1, len(encoder.encode(payload)) + 2)
            except Exception:  # pragma: no cover - defensive third-party integration path
                pass
        return max(1, (len(payload) + 3) // 4)

    def _tiktoken_encoder(self):
        model_name = self._string_config(
            "turn_guidance_tokenizer_model",
            fallback_name="turn_controller_tokenizer_model",
            default=self._string_config(
                "model",
                fallback_name="llm_model",
                default=self._string_config("provider_model", default=""),
            ),
        )
        encoding_name = self._string_config(
            "turn_guidance_tokenizer_encoding",
            fallback_name="turn_controller_tokenizer_encoding",
            default="",
        )
        return self._load_tiktoken_encoder(model_name or "", encoding_name or "")

    @staticmethod
    @lru_cache(maxsize=8)
    def _load_tiktoken_encoder(model_name: str, encoding_name: str):
        try:
            import tiktoken
        except Exception:  # pragma: no cover - optional dependency
            return None

        if encoding_name:
            try:
                return tiktoken.get_encoding(encoding_name)
            except Exception:
                pass

        if model_name:
            try:
                return tiktoken.encoding_for_model(model_name)
            except Exception:
                pass

        try:
            return tiktoken.get_encoding("o200k_base")
        except Exception:
            return None

    def _bool_config(self, name: str, *, default: bool) -> bool:
        config = getattr(self._loop, "config", None)
        raw = getattr(config, name, default)
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            normalized = raw.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return bool(raw)

    def _int_config(
        self,
        name: str,
        *,
        fallback_name: str | None = None,
        default: int,
        minimum: int = 0,
    ) -> int:
        config = getattr(self._loop, "config", None)
        for candidate_name in (name, fallback_name):
            if not candidate_name:
                continue
            raw = getattr(config, candidate_name, None)
            if raw is None:
                continue
            try:
                return max(minimum, int(raw))
            except (TypeError, ValueError):
                self._trace_event(
                    "turn_guidance_invalid_numeric_config",
                    kind="error",
                    details={"name": candidate_name, "value": repr(raw), "default": default},
                )
        return max(minimum, default)

    def _string_config(
        self,
        name: str,
        *,
        fallback_name: str | None = None,
        default: str = "",
    ) -> str:
        config = getattr(self._loop, "config", None)
        for candidate_name in (name, fallback_name):
            if not candidate_name:
                continue
            raw = getattr(config, candidate_name, None)
            if raw is None:
                continue
            text = str(raw).strip()
            if text:
                return text
        return default

    def _trace_event(
        self,
        name: str,
        *,
        kind: str = "decision",
        details: Mapping[str, Any] | None = None,
        kpi: Mapping[str, Any] | None = None,
    ) -> None:
        tracer = getattr(self._loop, "_trace_event", None)
        if not callable(tracer):
            return
        try:
            tracer(name, kind=kind, details=dict(details or {}), kpi=dict(kpi or {}))
        except Exception:  # pragma: no cover - tracing should never break runtime
            return
    