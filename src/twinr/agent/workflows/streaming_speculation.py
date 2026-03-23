"""Own speculative bridge warmups for the streaming voice loop.

The streaming loop keeps the public hook methods on the loop itself so tests
and higher-level orchestration can still override them. This controller holds
the underlying speculative state and the actual warmup logic so
``streaming_runner.py`` stays focused on orchestration.
"""

from __future__ import annotations

from threading import Event, Lock, Thread

from twinr.agent.base_agent.contracts import (
    FirstWordReply,
    SupervisorDecision,
    supervisor_decision_requires_full_context,
)
from twinr.agent.base_agent.conversation.turn_controller import _normalize_turn_text
from twinr.agent.tools import DualLaneToolLoop


class StreamingSpeculationController:
    """Manage speculative first-word and supervisor warmups for one loop."""

    def __init__(self, loop) -> None:
        self._loop = loop
        self._ensure_state()
        self.reset()

    def _ensure_state(self) -> None:
        loop = self._loop
        if getattr(loop, "_speculative_supervisor_lock", None) is None:
            loop._speculative_supervisor_lock = Lock()
        if getattr(loop, "_speculative_first_word_lock", None) is None:
            loop._speculative_first_word_lock = Lock()
        if getattr(loop, "_speculative_supervisor_done", None) is None:
            loop._speculative_supervisor_done = Event()
        if getattr(loop, "_speculative_first_word_done", None) is None:
            loop._speculative_first_word_done = Event()
        if getattr(loop, "_supervisor_cache_prewarmed", None) is None:
            loop._supervisor_cache_prewarmed = False
        if getattr(loop, "_first_word_cache_prewarmed", None) is None:
            loop._first_word_cache_prewarmed = False

    def reset(self) -> None:
        """Reset all speculative state for the next turn."""

        loop = self._loop
        with loop._speculative_supervisor_lock:
            loop._speculative_supervisor_done = Event()
            loop._speculative_supervisor_started = False
            loop._speculative_supervisor_transcript = ""
            loop._speculative_supervisor_decision = None
        with loop._speculative_first_word_lock:
            loop._speculative_first_word_done = Event()
            loop._speculative_first_word_started = False
            loop._speculative_first_word_transcript = ""
            loop._speculative_first_word_reply = None

    def maybe_start_first_word(self, text: str) -> None:
        """Start the speculative first-word worker once the transcript is meaningful."""

        loop = self._loop
        if self.dual_lane_prefers_supervisor_bridge():
            loop._trace_event(
                "speculative_first_word_skipped_supervisor_bridge",
                kind="branch",
                details={"text_len": len(text.strip())},
            )
            return
        if not loop.config.streaming_first_word_enabled:
            return
        if not loop.config.streaming_first_word_prefetch_enabled:
            return
        provider = loop.first_word_provider
        if provider is None:
            return
        cleaned = text.strip()
        word_count = len(cleaned.split())
        min_chars = max(1, int(loop.config.streaming_first_word_prefetch_min_chars))
        min_words = max(1, int(loop.config.streaming_first_word_prefetch_min_words))
        if len(cleaned) < min_chars or word_count < min_words:
            loop._trace_event(
                "speculative_first_word_skipped_short_text",
                kind="branch",
                details={
                    "text_len": len(cleaned),
                    "word_count": word_count,
                    "min_chars": min_chars,
                    "min_words": min_words,
                },
            )
            return
        with loop._speculative_first_word_lock:
            if loop._speculative_first_word_started:
                loop._trace_event(
                    "speculative_first_word_skipped_already_started",
                    kind="branch",
                    details={"text_len": len(cleaned)},
                )
                return
            loop._speculative_first_word_started = True
            loop._speculative_first_word_transcript = cleaned
            done_event = loop._speculative_first_word_done
            with loop._trace_span(
                name="speculative_first_word_context_build",
                kind="span",
                details={"text_len": len(cleaned)},
            ):
                conversation = loop.runtime.first_word_provider_conversation_context()
        loop._trace_event(
            "speculative_first_word_worker_started",
            kind="queue",
            details={"text_len": len(cleaned)},
        )
        worker = Thread(
            target=self._speculative_first_word_worker,
            args=(provider, cleaned, conversation, done_event),
            daemon=True,
        )
        worker.start()

    def _speculative_first_word_worker(
        self,
        provider,
        transcript: str,
        conversation,
        done_event: Event,
    ) -> None:
        loop = self._loop
        reply: FirstWordReply | None = None
        try:
            reply = provider.reply(
                transcript,
                conversation=conversation,
                instructions=None,
            )
        except Exception as exc:
            loop.emit(f"speculative_first_word_failed={type(exc).__name__}")
            loop._trace_event(
                "speculative_first_word_worker_failed",
                kind="exception",
                level="WARN",
                details={"error_type": type(exc).__name__, "text_len": len(transcript)},
            )
        finally:
            with loop._speculative_first_word_lock:
                if loop._speculative_first_word_transcript == transcript:
                    loop._speculative_first_word_reply = reply
            done_event.set()
            loop._trace_event(
                "speculative_first_word_worker_completed",
                kind="queue",
                details={"hit": reply is not None, "text_len": len(transcript)},
            )

    def consume_first_word(self, transcript: str) -> FirstWordReply | None:
        """Return the speculative first-word reply when it still matches the transcript."""

        loop = self._loop
        if not loop.config.streaming_first_word_enabled:
            return None
        with loop._speculative_first_word_lock:
            if not loop._speculative_first_word_started:
                return None
            done_event = loop._speculative_first_word_done
            seeded_transcript = loop._speculative_first_word_transcript
        wait_ms = max(0, int(loop.config.streaming_first_word_prefetch_wait_ms))
        if wait_ms > 0 and not done_event.is_set():
            done_event.wait(wait_ms / 1000.0)
        with loop._speculative_first_word_lock:
            reply = loop._speculative_first_word_reply
        if reply is None:
            return None
        normalized_seed = _normalize_turn_text(seeded_transcript)
        normalized_final = _normalize_turn_text(transcript)
        if not normalized_seed or not normalized_final:
            return None
        if not (
            normalized_final.startswith(normalized_seed)
            or normalized_seed.startswith(normalized_final)
        ):
            return None
        loop.emit("speculative_first_word_hit=true")
        loop._trace_event(
            "speculative_first_word_consumed",
            kind="cache",
            details={"seed_len": len(seeded_transcript), "final_len": len(transcript)},
        )
        return reply

    def maybe_start_supervisor_decision(self, text: str) -> None:
        """Start speculative supervisor routing once enough transcript is present."""

        loop = self._loop
        if not loop.config.streaming_supervisor_prefetch_enabled:
            return
        if not isinstance(loop.streaming_turn_loop, DualLaneToolLoop):
            return
        provider = getattr(loop.streaming_turn_loop, "supervisor_decision_provider", None)
        if provider is None:
            return
        cleaned = text.strip()
        if len(cleaned) < max(1, int(loop.config.streaming_supervisor_prefetch_min_chars)):
            loop._trace_event(
                "speculative_supervisor_skipped_short_text",
                kind="branch",
                details={"text_len": len(cleaned)},
            )
            return
        with loop._speculative_supervisor_lock:
            if loop._speculative_supervisor_started:
                loop._trace_event(
                    "speculative_supervisor_skipped_already_started",
                    kind="branch",
                    details={"text_len": len(cleaned)},
                )
                return
            loop._speculative_supervisor_started = True
            loop._speculative_supervisor_transcript = cleaned
            done_event = loop._speculative_supervisor_done
            with loop._trace_span(
                name="speculative_supervisor_context_build",
                kind="span",
                details={"text_len": len(cleaned)},
            ):
                supervisor_conversation = loop.runtime.supervisor_provider_conversation_context()
            supervisor_instructions = loop.streaming_turn_loop.supervisor_instructions
        loop._trace_event(
            "speculative_supervisor_worker_started",
            kind="queue",
            details={"text_len": len(cleaned)},
        )
        worker = Thread(
            target=self._speculative_supervisor_worker,
            args=(provider, cleaned, supervisor_conversation, supervisor_instructions, done_event),
            daemon=True,
        )
        worker.start()

    def _speculative_supervisor_worker(
        self,
        provider,
        transcript: str,
        conversation,
        instructions: str,
        done_event: Event,
    ) -> None:
        loop = self._loop
        decision: SupervisorDecision | None = None
        try:
            decision = provider.decide(
                transcript,
                conversation=conversation,
                instructions=instructions,
            )
        except Exception as exc:
            loop.emit(f"speculative_supervisor_failed={type(exc).__name__}")
            loop._trace_event(
                "speculative_supervisor_worker_failed",
                kind="exception",
                level="WARN",
                details={"error_type": type(exc).__name__, "text_len": len(transcript)},
            )
        finally:
            with loop._speculative_supervisor_lock:
                if loop._speculative_supervisor_transcript == transcript:
                    loop._speculative_supervisor_decision = decision
            done_event.set()
            loop._trace_event(
                "speculative_supervisor_worker_completed",
                kind="queue",
                details={
                    "decision_action": getattr(decision, "action", None) if decision is not None else None,
                    "text_len": len(transcript),
                },
            )

    def consume_supervisor_decision(self, transcript: str) -> SupervisorDecision | None:
        """Return one speculative supervisor decision when it still matches."""

        loop = self._loop
        if not loop.config.streaming_supervisor_prefetch_enabled:
            return None
        decision = self._matching_supervisor_decision(
            transcript,
            wait_ms=max(0, int(loop.config.streaming_supervisor_prefetch_wait_ms)),
        )
        if decision is None:
            return None
        loop.emit("speculative_supervisor_hit=true")
        loop._trace_event(
            "speculative_supervisor_consumed",
            kind="cache",
            details={
                "action": str(getattr(decision, "action", "") or "").strip().lower(),
                "seed_len": len(getattr(loop, "_speculative_supervisor_transcript", "") or ""),
                "final_len": len(transcript),
            },
        )
        return decision

    def wait_for_supervisor_decision(
        self,
        transcript: str,
        *,
        wait_ms: int | None = None,
    ) -> SupervisorDecision | None:
        """Wait for the shared speculative supervisor decision without re-calling the provider."""

        cleaned = str(transcript or "").strip()
        if not cleaned:
            return None
        self.maybe_start_supervisor_decision(cleaned)
        return self._matching_supervisor_decision(cleaned, wait_ms=wait_ms)

    def has_shared_supervisor_decision(self, transcript: str) -> bool:
        """Return whether a matching shared supervisor-decision worker already exists."""

        cleaned = str(transcript or "").strip()
        if not cleaned:
            return False
        return self._matching_supervisor_transcript(cleaned)

    def prime_supervisor_decision_cache(self) -> None:
        """Prewarm the supervisor cache once so the first live turn is not cold."""

        loop = self._loop
        if loop._supervisor_cache_prewarmed:
            return
        if not isinstance(loop.streaming_turn_loop, DualLaneToolLoop):
            return
        provider = getattr(loop.streaming_turn_loop, "supervisor_decision_provider", None)
        if provider is None:
            return
        try:
            provider.decide(
                "Sag bitte nur kurz Hallo.",
                conversation=(),
                instructions=loop.streaming_turn_loop.supervisor_instructions,
            )
            loop._supervisor_cache_prewarmed = True
            loop._trace_event("supervisor_cache_prewarmed", kind="cache", details={})
        except Exception as exc:
            loop.emit(f"supervisor_cache_prewarm_failed={type(exc).__name__}")
            loop._trace_event(
                "supervisor_cache_prewarm_failed",
                kind="exception",
                level="WARN",
                details={"error_type": type(exc).__name__},
            )

    def prime_first_word_cache(self) -> None:
        """Prewarm the standalone first-word path when it is active."""

        loop = self._loop
        if self.dual_lane_prefers_supervisor_bridge():
            loop._trace_event("first_word_cache_skip_supervisor_bridge", kind="cache", details={})
            return
        if loop._first_word_cache_prewarmed:
            return
        provider = loop.first_word_provider
        if provider is None:
            return
        try:
            provider.reply(
                "Sag bitte nur kurz Hallo.",
                conversation=(),
                instructions=None,
            )
            loop._first_word_cache_prewarmed = True
            loop._trace_event("first_word_cache_prewarmed", kind="cache", details={})
        except Exception as exc:
            loop.emit(f"first_word_cache_prewarm_failed={type(exc).__name__}")
            loop._trace_event(
                "first_word_cache_prewarm_failed",
                kind="exception",
                level="WARN",
                details={"error_type": type(exc).__name__},
            )

    def generate_first_word_reply(
        self,
        transcript: str,
        *,
        instructions: str | None = None,
    ) -> FirstWordReply | None:
        """Synchronously build a first-word reply when speculation missed."""

        loop = self._loop
        if not loop.config.streaming_first_word_enabled:
            return None
        provider = loop.first_word_provider
        if provider is None:
            return None
        try:
            with loop._trace_span(
                name="first_word_context_build_sync",
                kind="span",
                details={"text_len": len(transcript)},
            ):
                conversation = loop.runtime.first_word_provider_conversation_context()
            reply = provider.reply(
                transcript,
                conversation=conversation,
                instructions=instructions,
            )
            loop._trace_event(
                "first_word_reply_generated",
                kind="llm_call",
                details={
                    "text_len": len(transcript),
                    "mode": getattr(reply, "mode", None),
                    "route_overlay": bool(str(instructions or "").strip()),
                },
            )
            return reply
        except Exception as exc:
            loop.emit(f"first_word_sync_failed={type(exc).__name__}")
            loop._trace_event(
                "first_word_reply_failed",
                kind="exception",
                level="WARN",
                details={
                    "error_type": type(exc).__name__,
                    "text_len": len(transcript),
                    "route_overlay": bool(str(instructions or "").strip()),
                },
            )
            return None

    def dual_lane_prefers_supervisor_bridge(self) -> bool:
        """Return whether the bridge speech should come from supervisor routing."""

        if not isinstance(self._loop.streaming_turn_loop, DualLaneToolLoop):
            return False
        provider = getattr(self._loop.streaming_turn_loop, "supervisor_decision_provider", None)
        return provider is not None

    def store_supervisor_decision(
        self,
        *,
        transcript: str,
        decision: SupervisorDecision | None,
    ) -> None:
        """Persist one supervisor decision so the final lane can reuse it."""

        loop = self._loop
        cleaned = str(transcript or "").strip()
        if not cleaned:
            return
        with loop._speculative_supervisor_lock:
            loop._speculative_supervisor_started = True
            loop._speculative_supervisor_transcript = cleaned
            loop._speculative_supervisor_decision = decision
            loop._speculative_supervisor_done.set()

    def generate_supervisor_bridge_reply(
        self,
        transcript: str,
        *,
        instructions: str | None,
    ) -> FirstWordReply | None:
        """Resolve the fast spoken bridge via the supervisor decision provider."""

        loop = self._loop
        if not self.dual_lane_prefers_supervisor_bridge():
            return None
        provider = getattr(loop.streaming_turn_loop, "supervisor_decision_provider", None)
        if provider is None:
            return None
        cleaned = str(transcript or "").strip()
        if not cleaned:
            return None
        self.maybe_start_supervisor_decision(cleaned)
        decision = self.wait_for_supervisor_decision(cleaned, wait_ms=None)
        if decision is None:
            loop._trace_event(
                "supervisor_bridge_reply_unavailable",
                kind="branch",
                details={"text_len": len(cleaned), "route_overlay": bool(str(instructions or "").strip())},
            )
            return None
        reply = self.dual_lane_bridge_reply_from_decision(decision)
        loop._trace_event(
            "supervisor_bridge_reply_generated",
            kind="llm_call",
            details={
                "text_len": len(cleaned),
                "decision_action": getattr(decision, "action", None),
                "reply_mode": getattr(reply, "mode", None) if reply is not None else None,
            },
        )
        return reply

    def _matching_supervisor_decision(
        self,
        transcript: str,
        *,
        wait_ms: int | None,
    ) -> SupervisorDecision | None:
        """Return the shared speculative decision when the transcript still matches it."""

        loop = self._loop
        with loop._speculative_supervisor_lock:
            if not loop._speculative_supervisor_started:
                return None
            done_event = loop._speculative_supervisor_done
            seeded_transcript = loop._speculative_supervisor_transcript
        if wait_ms is None:
            if not done_event.is_set():
                done_event.wait()
        elif wait_ms > 0 and not done_event.is_set():
            done_event.wait(wait_ms / 1000.0)
        with loop._speculative_supervisor_lock:
            decision = loop._speculative_supervisor_decision
        if decision is None:
            return None
        action = str(getattr(decision, "action", "") or "").strip().lower()
        if action not in {"direct", "handoff", "end_conversation"}:
            return None
        normalized_seed = _normalize_turn_text(seeded_transcript)
        normalized_final = _normalize_turn_text(transcript)
        if not normalized_seed or not normalized_final:
            return None
        if not (
            normalized_final.startswith(normalized_seed)
            or normalized_seed.startswith(normalized_final)
        ):
            return None
        return decision

    def _matching_supervisor_transcript(self, transcript: str) -> bool:
        """Return whether the stored speculative transcript still matches the active transcript."""

        loop = self._loop
        with loop._speculative_supervisor_lock:
            if not loop._speculative_supervisor_started:
                return False
            seeded_transcript = loop._speculative_supervisor_transcript
        normalized_seed = _normalize_turn_text(seeded_transcript)
        normalized_final = _normalize_turn_text(transcript)
        if not normalized_seed or not normalized_final:
            return False
        return (
            normalized_final.startswith(normalized_seed)
            or normalized_seed.startswith(normalized_final)
        )

    def dual_lane_bridge_reply_from_decision(
        self,
        prefetched_decision: SupervisorDecision | None,
    ) -> FirstWordReply | None:
        """Build a bridge reply directly from one resolved supervisor decision."""

        fallback_text = ""
        fallback_mode = "filler"
        if prefetched_decision is not None:
            action = str(getattr(prefetched_decision, "action", "") or "").strip().lower()
            if action == "direct" and supervisor_decision_requires_full_context(prefetched_decision):
                fallback_text = str(getattr(prefetched_decision, "spoken_ack", None) or "").strip()
                fallback_mode = "filler"
            elif action == "end_conversation":
                fallback_text = str(
                    getattr(prefetched_decision, "spoken_reply", None)
                    or getattr(prefetched_decision, "spoken_ack", None)
                    or ""
                ).strip()
                fallback_mode = "direct" if fallback_text else fallback_mode
            elif action == "handoff":
                fallback_text = str(getattr(prefetched_decision, "spoken_ack", None) or "").strip()
        if not fallback_text:
            return None
        return FirstWordReply(
            mode=fallback_mode,
            spoken_text=fallback_text,
            response_id=getattr(prefetched_decision, "response_id", None),
            request_id=getattr(prefetched_decision, "request_id", None),
            model=getattr(prefetched_decision, "model", None),
            token_usage=getattr(prefetched_decision, "token_usage", None),
        )
