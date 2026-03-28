# CHANGELOG: 2026-03-28
# BUG-1: Add per-lane generation tokens so stale speculative results cannot leak
#        across reset()/turn boundaries when the next turn reuses the same text.
# BUG-2: Bound the shared supervisor wait path so a hung provider no longer
#        freezes the live bridge reply indefinitely.
# BUG-3: Make context-build failures non-fatal and avoid wedging a lane in
#        `_started=True` without a worker or completion signal.
# SEC-1: Bound speculative background concurrency and add per-lane cooldowns to
#        reduce practical Pi 4 denial-of-service risk from slow/hung providers.
# SEC-2: Cap speculative transcript length so untrusted long partials cannot
#        amplify speculative CPU/network spend.
# IMP-1: Use token-aware transcript compatibility checks to reduce false cache
#        hits from naive raw-string prefix matching on partial ASR text.
# IMP-2: Move context building out of locks and allow safe refresh of completed
#        stale speculation without spawning duplicate in-flight work.
# IMP-3: Name background threads for observability and preserve drop-in loop
#        fields while making speculation best-effort and failure-tolerant.

"""Own speculative bridge warmups for the streaming voice loop.

The streaming loop keeps the public hook methods on the loop itself so tests
and higher-level orchestration can still override them. This controller holds
the underlying speculative state and the actual warmup logic so
``streaming_runner.py`` stays focused on orchestration.
"""

from __future__ import annotations

from threading import BoundedSemaphore, Event, Lock, Thread
from time import monotonic

from twinr.agent.base_agent.contracts import (
    FirstWordReply,
    SupervisorDecision,
    normalize_supervisor_decision_runtime_tool_name,
    supervisor_decision_requires_full_context,
)
from twinr.agent.base_agent.conversation.decision_core import normalize_turn_text
from twinr.agent.tools import DualLaneToolLoop
from twinr.agent.workflows.streaming_supervisor_context import (
    build_streaming_supervisor_turn_instructions,
)


class StreamingSpeculationController:
    """Manage speculative first-word and supervisor warmups for one loop."""

    _DEFAULT_BACKGROUND_WORKERS = 4
    _DEFAULT_FAILURE_THRESHOLD = 3
    _DEFAULT_COOLDOWN_S = 30.0
    _DEFAULT_FIRST_WORD_MAX_CHARS = 320
    _DEFAULT_SUPERVISOR_MAX_CHARS = 1024

    # BREAKING: wait_for_supervisor_decision(wait_ms=None) no longer waits
    # forever by default. Live voice code now uses a bounded best-effort wait.
    # Set `config.streaming_supervisor_prefetch_hard_timeout_ms = None` to
    # restore the legacy unbounded behavior.
    _DEFAULT_SUPERVISOR_HARD_TIMEOUT_MS = 1200

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

        if getattr(loop, "_speculative_first_word_generation", None) is None:
            loop._speculative_first_word_generation = 0
        if getattr(loop, "_speculative_supervisor_generation", None) is None:
            loop._speculative_supervisor_generation = 0

        if getattr(loop, "_speculative_first_word_worker", None) is None:
            loop._speculative_first_word_worker = None
        if getattr(loop, "_speculative_supervisor_worker", None) is None:
            loop._speculative_supervisor_worker = None

        if getattr(loop, "_speculative_first_word_timeout_generation", None) is None:
            loop._speculative_first_word_timeout_generation = -1
        if getattr(loop, "_speculative_supervisor_timeout_generation", None) is None:
            loop._speculative_supervisor_timeout_generation = -1

        if getattr(loop, "_speculative_first_word_consecutive_failures", None) is None:
            loop._speculative_first_word_consecutive_failures = 0
        if getattr(loop, "_speculative_supervisor_consecutive_failures", None) is None:
            loop._speculative_supervisor_consecutive_failures = 0

        if getattr(loop, "_speculative_first_word_disabled_until", None) is None:
            loop._speculative_first_word_disabled_until = 0.0
        if getattr(loop, "_speculative_supervisor_disabled_until", None) is None:
            loop._speculative_supervisor_disabled_until = 0.0

        if getattr(loop, "_speculative_background_slots", None) is None:
            max_workers = self._get_int_config(
                "streaming_speculative_max_background_workers",
                self._DEFAULT_BACKGROUND_WORKERS,
                minimum=1,
            )
            loop._speculative_background_slots = BoundedSemaphore(max_workers)
            loop._speculative_background_slots_capacity = max_workers

    def reset(self) -> None:
        """Reset all speculative state for the next turn."""

        loop = self._loop
        with loop._speculative_supervisor_lock:
            loop._speculative_supervisor_generation += 1
            loop._speculative_supervisor_done = Event()
            loop._speculative_supervisor_started = False
            loop._speculative_supervisor_transcript = ""
            loop._speculative_supervisor_decision = None
            loop._speculative_supervisor_worker = None
            loop._speculative_supervisor_timeout_generation = -1

        with loop._speculative_first_word_lock:
            loop._speculative_first_word_generation += 1
            loop._speculative_first_word_done = Event()
            loop._speculative_first_word_started = False
            loop._speculative_first_word_transcript = ""
            loop._speculative_first_word_reply = None
            loop._speculative_first_word_worker = None
            loop._speculative_first_word_timeout_generation = -1

    def maybe_start_first_word(self, text: str) -> None:
        """Start the speculative first-word worker once the transcript is meaningful."""

        loop = self._loop
        if self.dual_lane_prefers_supervisor_bridge():
            loop._trace_event(
                "speculative_first_word_skipped_supervisor_bridge",
                kind="branch",
                details={"text_len": len(self._coerce_text(text))},
            )
            return
        if not loop.config.streaming_first_word_enabled:
            return
        if not loop.config.streaming_first_word_prefetch_enabled:
            return

        provider = loop.first_word_provider
        if provider is None:
            return

        if self._lane_disabled("first_word"):
            loop._trace_event(
                "speculative_first_word_skipped_circuit_open",
                kind="branch",
                details={},
            )
            return

        cleaned = self._coerce_text(text)
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

        cleaned = self._trim_speculative_text(cleaned, lane="first_word")
        if not cleaned:
            return

        with loop._speculative_first_word_lock:
            generation = loop._speculative_first_word_generation
            if loop._speculative_first_word_started:
                done_event = loop._speculative_first_word_done
                seeded_transcript = loop._speculative_first_word_transcript
                existing_reply = loop._speculative_first_word_reply

                if not done_event.is_set():
                    loop._trace_event(
                        "speculative_first_word_skipped_already_started",
                        kind="branch",
                        details={"text_len": len(cleaned)},
                    )
                    return

                if self._transcripts_compatible(seeded_transcript, cleaned):
                    if existing_reply is not None or len(cleaned) <= len(seeded_transcript):
                        loop._trace_event(
                            "speculative_first_word_skipped_completed_match",
                            kind="branch",
                            details={
                                "seed_len": len(seeded_transcript),
                                "text_len": len(cleaned),
                            },
                        )
                        return
                    restart_reason = "miss"
                else:
                    restart_reason = "stale"

                generation += 1
                loop._speculative_first_word_generation = generation
                loop._speculative_first_word_done = Event()
                loop._trace_event(
                    f"speculative_first_word_restarted_completed_{restart_reason}",
                    kind="branch",
                    details={
                        "old_seed_len": len(seeded_transcript),
                        "text_len": len(cleaned),
                    },
                )

            loop._speculative_first_word_started = True
            loop._speculative_first_word_transcript = cleaned
            loop._speculative_first_word_reply = None
            loop._speculative_first_word_timeout_generation = -1
            done_event = loop._speculative_first_word_done

        try:
            with loop._trace_span(
                name="speculative_first_word_context_build",
                kind="span",
                details={"text_len": len(cleaned)},
            ):
                conversation = loop.runtime.first_word_provider_conversation_context()
        except Exception as exc:
            self._handle_context_build_failure(
                lane="first_word",
                generation=generation,
                transcript=cleaned,
                done_event=done_event,
                error=exc,
            )
            return

        if not self._launch_background_worker(
            lane="first_word",
            generation=generation,
            transcript=cleaned,
            target=self._speculative_first_word_worker,
            args=(generation, provider, cleaned, conversation, done_event),
        ):
            self._abort_lane_launch(
                lane="first_word",
                generation=generation,
                transcript=cleaned,
                done_event=done_event,
            )
            return

        loop._trace_event(
            "speculative_first_word_worker_started",
            kind="queue",
            details={"text_len": len(cleaned)},
        )

    def _speculative_first_word_worker(
        self,
        generation: int,
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
            self._mark_lane_success("first_word")
        except Exception as exc:
            self._mark_lane_failure("first_word")
            loop.emit(f"speculative_first_word_failed={type(exc).__name__}")
            loop._trace_event(
                "speculative_first_word_worker_failed",
                kind="exception",
                level="WARN",
                details={"error_type": type(exc).__name__, "text_len": len(transcript)},
            )
        finally:
            with loop._speculative_first_word_lock:
                if (
                    loop._speculative_first_word_generation == generation
                    and loop._speculative_first_word_transcript == transcript
                ):
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

        cleaned = self._coerce_text(transcript)
        with loop._speculative_first_word_lock:
            if not loop._speculative_first_word_started:
                return None
            generation = loop._speculative_first_word_generation
            done_event = loop._speculative_first_word_done

        wait_ms = max(0, int(loop.config.streaming_first_word_prefetch_wait_ms))
        if wait_ms > 0 and not done_event.is_set():
            if not done_event.wait(wait_ms / 1000.0):
                self._mark_lane_timeout_once("first_word", generation)
                return None

        with loop._speculative_first_word_lock:
            if generation != loop._speculative_first_word_generation:
                return None
            reply = loop._speculative_first_word_reply
            seeded_transcript = loop._speculative_first_word_transcript

        if reply is None:
            return None
        if not self._transcripts_compatible(seeded_transcript, cleaned):
            return None

        loop.emit("speculative_first_word_hit=true")
        loop._trace_event(
            "speculative_first_word_consumed",
            kind="cache",
            details={"seed_len": len(seeded_transcript), "final_len": len(cleaned)},
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

        if self._lane_disabled("supervisor"):
            loop._trace_event(
                "speculative_supervisor_skipped_circuit_open",
                kind="branch",
                details={},
            )
            return

        cleaned = self._coerce_text(text)
        if len(cleaned) < max(1, int(loop.config.streaming_supervisor_prefetch_min_chars)):
            loop._trace_event(
                "speculative_supervisor_skipped_short_text",
                kind="branch",
                details={"text_len": len(cleaned)},
            )
            return

        cleaned = self._trim_speculative_text(cleaned, lane="supervisor")
        if not cleaned:
            return

        with loop._speculative_supervisor_lock:
            generation = loop._speculative_supervisor_generation
            if loop._speculative_supervisor_started:
                done_event = loop._speculative_supervisor_done
                seeded_transcript = loop._speculative_supervisor_transcript
                existing_decision = loop._speculative_supervisor_decision

                if not done_event.is_set():
                    loop._trace_event(
                        "speculative_supervisor_skipped_already_started",
                        kind="branch",
                        details={"text_len": len(cleaned)},
                    )
                    return

                if self._transcripts_compatible(seeded_transcript, cleaned):
                    if existing_decision is not None or len(cleaned) <= len(seeded_transcript):
                        loop._trace_event(
                            "speculative_supervisor_skipped_completed_match",
                            kind="branch",
                            details={
                                "seed_len": len(seeded_transcript),
                                "text_len": len(cleaned),
                            },
                        )
                        return
                    restart_reason = "miss"
                else:
                    restart_reason = "stale"

                generation += 1
                loop._speculative_supervisor_generation = generation
                loop._speculative_supervisor_done = Event()
                loop._trace_event(
                    f"speculative_supervisor_restarted_completed_{restart_reason}",
                    kind="branch",
                    details={
                        "old_seed_len": len(seeded_transcript),
                        "text_len": len(cleaned),
                    },
                )

            loop._speculative_supervisor_started = True
            loop._speculative_supervisor_transcript = cleaned
            loop._speculative_supervisor_decision = None
            loop._speculative_supervisor_timeout_generation = -1
            done_event = loop._speculative_supervisor_done

        try:
            with loop._trace_span(
                name="speculative_supervisor_context_build",
                kind="span",
                details={"text_len": len(cleaned)},
            ):
                supervisor_conversation = loop.runtime.supervisor_provider_conversation_context()
            supervisor_instructions = build_streaming_supervisor_turn_instructions(loop.config)
        except Exception as exc:
            self._handle_context_build_failure(
                lane="supervisor",
                generation=generation,
                transcript=cleaned,
                done_event=done_event,
                error=exc,
            )
            return

        if not self._launch_background_worker(
            lane="supervisor",
            generation=generation,
            transcript=cleaned,
            target=self._speculative_supervisor_worker,
            args=(
                generation,
                provider,
                cleaned,
                supervisor_conversation,
                supervisor_instructions,
                done_event,
            ),
        ):
            self._abort_lane_launch(
                lane="supervisor",
                generation=generation,
                transcript=cleaned,
                done_event=done_event,
            )
            return

        loop._trace_event(
            "speculative_supervisor_worker_started",
            kind="queue",
            details={"text_len": len(cleaned)},
        )

    def _speculative_supervisor_worker(
        self,
        generation: int,
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
            self._mark_lane_success("supervisor")
        except Exception as exc:
            self._mark_lane_failure("supervisor")
            loop.emit(f"speculative_supervisor_failed={type(exc).__name__}")
            loop._trace_event(
                "speculative_supervisor_worker_failed",
                kind="exception",
                level="WARN",
                details={"error_type": type(exc).__name__, "text_len": len(transcript)},
            )
        finally:
            with loop._speculative_supervisor_lock:
                if (
                    loop._speculative_supervisor_generation == generation
                    and loop._speculative_supervisor_transcript == transcript
                ):
                    loop._speculative_supervisor_decision = decision
            done_event.set()
            loop._trace_event(
                "speculative_supervisor_worker_completed",
                kind="queue",
                details={
                    "decision_action": getattr(decision, "action", None)
                    if decision is not None
                    else None,
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
                "final_len": len(self._coerce_text(transcript)),
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

        cleaned = self._coerce_text(transcript)
        if not cleaned:
            return None
        self.maybe_start_supervisor_decision(cleaned)
        return self._matching_supervisor_decision(
            cleaned,
            wait_ms=self._resolve_supervisor_wait_ms(wait_ms),
        )

    def has_shared_supervisor_decision(self, transcript: str) -> bool:
        """Return whether a matching shared supervisor-decision worker already exists."""

        cleaned = self._coerce_text(transcript)
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

        cleaned = self._coerce_text(transcript)
        if not cleaned:
            return None

        try:
            with loop._trace_span(
                name="first_word_context_build_sync",
                kind="span",
                details={"text_len": len(cleaned)},
            ):
                conversation = loop.runtime.first_word_provider_conversation_context()

            reply = provider.reply(
                cleaned,
                conversation=conversation,
                instructions=instructions,
            )
            loop._trace_event(
                "first_word_reply_generated",
                kind="llm_call",
                details={
                    "text_len": len(cleaned),
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
                    "text_len": len(cleaned),
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
        cleaned = self._coerce_text(transcript)
        if not cleaned:
            return

        with loop._speculative_supervisor_lock:
            # Invalidate any older in-flight worker so it cannot overwrite the
            # authoritative decision later.
            loop._speculative_supervisor_generation += 1
            loop._speculative_supervisor_started = True
            loop._speculative_supervisor_transcript = cleaned
            loop._speculative_supervisor_decision = decision
            loop._speculative_supervisor_timeout_generation = -1
            loop._speculative_supervisor_done = Event()
            loop._speculative_supervisor_done.set()
            loop._speculative_supervisor_worker = None

        self._mark_lane_success("supervisor")

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

        cleaned = self._coerce_text(transcript)
        if not cleaned:
            return None

        self.maybe_start_supervisor_decision(cleaned)
        decision = self.wait_for_supervisor_decision(cleaned, wait_ms=None)
        if decision is None:
            loop._trace_event(
                "supervisor_bridge_reply_unavailable",
                kind="branch",
                details={
                    "text_len": len(cleaned),
                    "route_overlay": bool(str(instructions or "").strip()),
                },
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
        cleaned = self._coerce_text(transcript)
        with loop._speculative_supervisor_lock:
            if not loop._speculative_supervisor_started:
                return None
            generation = loop._speculative_supervisor_generation
            done_event = loop._speculative_supervisor_done

        if wait_ms is None:
            if not done_event.is_set():
                done_event.wait()
        elif wait_ms > 0 and not done_event.is_set():
            if not done_event.wait(wait_ms / 1000.0):
                self._mark_lane_timeout_once("supervisor", generation)
                return None

        with loop._speculative_supervisor_lock:
            if generation != loop._speculative_supervisor_generation:
                return None
            decision = loop._speculative_supervisor_decision
            seeded_transcript = loop._speculative_supervisor_transcript

        if decision is None:
            return None

        action = str(getattr(decision, "action", "") or "").strip().lower()
        if action not in {"direct", "handoff", "end_conversation"}:
            return None

        if not self._transcripts_compatible(seeded_transcript, cleaned):
            return None

        return decision

    def _matching_supervisor_transcript(self, transcript: str) -> bool:
        """Return whether the stored speculative transcript still matches the active transcript."""

        loop = self._loop
        cleaned = self._coerce_text(transcript)
        with loop._speculative_supervisor_lock:
            if not loop._speculative_supervisor_started:
                return False
            seeded_transcript = loop._speculative_supervisor_transcript
        return self._transcripts_compatible(seeded_transcript, cleaned)

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
                # One-shot runtime-local handoffs already speak through the
                # direct tool result; replaying spoken_ack here duplicates the
                # confirmation and widens the speaking window unnecessarily.
                runtime_tool_name = normalize_supervisor_decision_runtime_tool_name(
                    getattr(prefetched_decision, "runtime_tool_name", None)
                )
                if runtime_tool_name is None:
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

    def _resolve_supervisor_wait_ms(self, wait_ms: int | None) -> int | None:
        if wait_ms is not None:
            return max(0, int(wait_ms))
        configured = getattr(
            self._loop.config,
            "streaming_supervisor_prefetch_hard_timeout_ms",
            self._DEFAULT_SUPERVISOR_HARD_TIMEOUT_MS,
        )
        if configured is None:
            return None
        return max(0, int(configured))

    def _handle_context_build_failure(
        self,
        *,
        lane: str,
        generation: int,
        transcript: str,
        done_event: Event,
        error: Exception,
    ) -> None:
        loop = self._loop
        self._mark_lane_failure(lane)
        loop.emit(f"speculative_{lane}_context_failed={type(error).__name__}")
        loop._trace_event(
            f"speculative_{lane}_context_build_failed",
            kind="exception",
            level="WARN",
            details={"error_type": type(error).__name__, "text_len": len(transcript)},
        )
        self._complete_lane_without_result(
            lane=lane,
            generation=generation,
            transcript=transcript,
            done_event=done_event,
            keep_started=True,
        )

    def _abort_lane_launch(
        self,
        *,
        lane: str,
        generation: int,
        transcript: str,
        done_event: Event,
    ) -> None:
        loop = self._loop
        self._mark_lane_failure(lane)
        self._complete_lane_without_result(
            lane=lane,
            generation=generation,
            transcript=transcript,
            done_event=done_event,
            keep_started=False,
        )
        loop._trace_event(
            f"speculative_{lane}_worker_not_started",
            kind="branch",
            level="WARN",
            details={"text_len": len(transcript)},
        )

    def _complete_lane_without_result(
        self,
        *,
        lane: str,
        generation: int,
        transcript: str,
        done_event: Event,
        keep_started: bool,
    ) -> None:
        loop = self._loop
        lock = self._lane_lock(lane)
        result_attr = self._lane_result_attr(lane)
        with lock:
            if (
                self._lane_generation(lane) == generation
                and self._lane_transcript(lane) == transcript
            ):
                setattr(loop, result_attr, None)
                if not keep_started:
                    self._set_lane_started(lane, False)
        done_event.set()

    def _launch_background_worker(
        self,
        *,
        lane: str,
        generation: int,
        transcript: str,
        target,
        args: tuple,
    ) -> bool:
        loop = self._loop
        slots = loop._speculative_background_slots
        if not slots.acquire(blocking=False):
            loop.emit(f"speculative_{lane}_backpressure=true")
            loop._trace_event(
                f"speculative_{lane}_backpressure",
                kind="queue",
                level="WARN",
                details={
                    "text_len": len(transcript),
                    "max_workers": getattr(loop, "_speculative_background_slots_capacity", None),
                },
            )
            return False

        worker_name = f"twinr-spec-{lane}-{generation}"

        def runner() -> None:
            try:
                target(*args)
            finally:
                with self._lane_lock(lane):
                    if self._lane_generation(lane) == generation:
                        setattr(loop, self._lane_worker_attr(lane), None)
                slots.release()

        worker = Thread(target=runner, name=worker_name, daemon=True)

        with self._lane_lock(lane):
            if (
                self._lane_generation(lane) != generation
                or self._lane_transcript(lane) != transcript
                or not self._lane_started(lane)
            ):
                slots.release()
                return False
            setattr(loop, self._lane_worker_attr(lane), worker)

        try:
            worker.start()
        except Exception as exc:
            with self._lane_lock(lane):
                if self._lane_generation(lane) == generation:
                    setattr(loop, self._lane_worker_attr(lane), None)
            slots.release()
            loop.emit(f"speculative_{lane}_start_failed={type(exc).__name__}")
            loop._trace_event(
                f"speculative_{lane}_worker_start_failed",
                kind="exception",
                level="WARN",
                details={"error_type": type(exc).__name__, "text_len": len(transcript)},
            )
            return False
        return True

    def _lane_disabled(self, lane: str) -> bool:
        disabled_until = float(
            getattr(self._loop, self._lane_disabled_until_attr(lane), 0.0) or 0.0
        )
        return monotonic() < disabled_until

    def _mark_lane_success(self, lane: str) -> None:
        loop = self._loop
        lock = self._lane_lock(lane)
        with lock:
            setattr(loop, self._lane_failure_attr(lane), 0)
            setattr(loop, self._lane_disabled_until_attr(lane), 0.0)
            setattr(loop, self._lane_timeout_generation_attr(lane), -1)

    def _mark_lane_failure(self, lane: str) -> None:
        loop = self._loop
        threshold = self._get_int_config(
            "streaming_speculative_failure_threshold",
            self._DEFAULT_FAILURE_THRESHOLD,
            minimum=1,
        )
        cooldown_s = self._get_float_config(
            "streaming_speculative_cooldown_s",
            self._DEFAULT_COOLDOWN_S,
            minimum=0.0,
        )
        lock = self._lane_lock(lane)
        with lock:
            failures = int(getattr(loop, self._lane_failure_attr(lane), 0) or 0) + 1
            setattr(loop, self._lane_failure_attr(lane), failures)
            if failures >= threshold:
                disabled_until = monotonic() + cooldown_s
                setattr(loop, self._lane_disabled_until_attr(lane), disabled_until)
                loop._trace_event(
                    f"speculative_{lane}_circuit_opened",
                    kind="branch",
                    level="WARN",
                    details={"consecutive_failures": failures, "cooldown_s": cooldown_s},
                )

    def _mark_lane_timeout_once(self, lane: str, generation: int) -> None:
        loop = self._loop
        lock = self._lane_lock(lane)
        with lock:
            if self._lane_generation(lane) != generation:
                return
            if getattr(loop, self._lane_timeout_generation_attr(lane), -1) == generation:
                return
            setattr(loop, self._lane_timeout_generation_attr(lane), generation)

        loop.emit(f"speculative_{lane}_timeout=true")
        loop._trace_event(
            f"speculative_{lane}_timeout",
            kind="exception",
            level="WARN",
            details={"generation": generation},
        )
        self._mark_lane_failure(lane)

    def _trim_speculative_text(self, text: str, *, lane: str) -> str:
        loop = self._loop
        cleaned = self._coerce_text(text)
        if not cleaned:
            return ""

        default_limit = (
            self._DEFAULT_FIRST_WORD_MAX_CHARS
            if lane == "first_word"
            else self._DEFAULT_SUPERVISOR_MAX_CHARS
        )
        config_name = f"streaming_{lane}_prefetch_max_chars"
        limit = getattr(loop.config, config_name, default_limit)
        if limit is None:
            return cleaned

        limit = max(32, int(limit))
        if len(cleaned) <= limit:
            return cleaned

        trimmed = cleaned[:limit].rsplit(" ", 1)[0].strip()
        if not trimmed:
            trimmed = cleaned[:limit].strip()

        loop._trace_event(
            f"speculative_{lane}_transcript_trimmed",
            kind="branch",
            details={"original_len": len(cleaned), "trimmed_len": len(trimmed)},
        )
        return trimmed

    def _transcripts_compatible(self, seeded: str, final: str) -> bool:
        normalized_seed = normalize_turn_text(self._coerce_text(seeded))
        normalized_final = normalize_turn_text(self._coerce_text(final))
        if not normalized_seed or not normalized_final:
            return False
        if normalized_seed == normalized_final:
            return True

        seed_tokens = normalized_seed.split()
        final_tokens = normalized_final.split()
        shorter, longer = (
            (seed_tokens, final_tokens)
            if len(seed_tokens) <= len(final_tokens)
            else (final_tokens, seed_tokens)
        )
        if not shorter:
            return False
        if len(shorter) == 1 and len(longer) == 1:
            return longer[0].startswith(shorter[0])

        if len(longer) < len(shorter):
            return False
        if longer[: len(shorter) - 1] != shorter[: len(shorter) - 1]:
            return False
        return longer[len(shorter) - 1].startswith(shorter[-1])

    @staticmethod
    def _coerce_text(text: str | None) -> str:
        return str(text or "").strip()

    def _get_int_config(self, name: str, default: int, *, minimum: int | None = None) -> int:
        raw = getattr(self._loop.config, name, default)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = int(default)
        if minimum is not None:
            value = max(minimum, value)
        return value

    def _get_float_config(
        self,
        name: str,
        default: float,
        *,
        minimum: float | None = None,
    ) -> float:
        raw = getattr(self._loop.config, name, default)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = float(default)
        if minimum is not None:
            value = max(minimum, value)
        return value

    def _lane_lock(self, lane: str):
        return getattr(self._loop, f"_speculative_{lane}_lock")

    def _lane_generation(self, lane: str) -> int:
        return int(getattr(self._loop, f"_speculative_{lane}_generation"))

    def _lane_transcript(self, lane: str) -> str:
        return str(getattr(self._loop, f"_speculative_{lane}_transcript", "") or "")

    def _lane_started(self, lane: str) -> bool:
        return bool(getattr(self._loop, f"_speculative_{lane}_started", False))

    def _set_lane_started(self, lane: str, started: bool) -> None:
        setattr(self._loop, f"_speculative_{lane}_started", bool(started))

    @staticmethod
    def _lane_result_attr(lane: str) -> str:
        return (
            "_speculative_first_word_reply"
            if lane == "first_word"
            else "_speculative_supervisor_decision"
        )

    @staticmethod
    def _lane_worker_attr(lane: str) -> str:
        return f"_speculative_{lane}_worker"

    @staticmethod
    def _lane_failure_attr(lane: str) -> str:
        return f"_speculative_{lane}_consecutive_failures"

    @staticmethod
    def _lane_disabled_until_attr(lane: str) -> str:
        return f"_speculative_{lane}_disabled_until"

    @staticmethod
    def _lane_timeout_generation_attr(lane: str) -> str:
        return f"_speculative_{lane}_timeout_generation"