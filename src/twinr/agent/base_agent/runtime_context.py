from __future__ import annotations

import math
import threading
from datetime import datetime, timedelta, timezone

from twinr.agent.base_agent.adaptive_timing import AdaptiveListeningWindow, AdaptiveTimingProfile
from twinr.agent.base_agent.language import memory_and_response_contract
from twinr.memory import LongTermMemoryService, TwinrPersonalGraphStore
from twinr.proactive import ProactiveGovernor


_ALLOWED_VOICE_STATUSES = frozenset({"likely_user", "uncertain", "unknown_voice"})
_DEFAULT_VOICE_ASSESSMENT_MAX_AGE_S = 120
_MAX_FUTURE_SKEW_S = 5
_LOCK_INIT_GUARD = threading.Lock()


class TwinrRuntimeContextMixin:
    def _runtime_context_lock(self) -> threading.RLock:
        lock = getattr(self, "_twinr_runtime_context_lock", None)
        if lock is None:
            with _LOCK_INIT_GUARD:
                lock = getattr(self, "_twinr_runtime_context_lock", None)
                if lock is None:
                    # AUDIT-FIX(#3): Serialize live state access so config swaps and prompt/timing reads cannot interleave.
                    lock = threading.RLock()
                    setattr(self, "_twinr_runtime_context_lock", lock)
        return lock

    def _safe_append_ops_event(
        self,
        *,
        event: str,
        message: str,
        data: dict[str, object] | None = None,
    ) -> None:
        sink = getattr(self, "ops_events", None)
        if sink is None:
            return
        try:
            # AUDIT-FIX(#5): Telemetry must never break the main senior interaction path.
            sink.append(
                event=event,
                message=message,
                data=data or {},
            )
        except Exception:
            return

    def _safe_persist_snapshot(self, *, event_on_error: str) -> None:
        persist = getattr(self, "_persist_snapshot", None)
        if not callable(persist):
            return
        try:
            # AUDIT-FIX(#5): Snapshot persistence is best-effort; runtime state must survive transient file-write failures.
            persist()
        except Exception as exc:
            self._safe_append_ops_event(
                event=event_on_error,
                message="Twinr could not persist runtime state and continued with in-memory state.",
                data={"error_type": type(exc).__name__},
            )

    def _best_effort_cleanup(self, resource: object | None, *, timeout_s: float | None = None) -> None:
        if resource is None:
            return
        for method_name in ("shutdown", "close"):
            method = getattr(resource, method_name, None)
            if not callable(method):
                continue
            try:
                # AUDIT-FIX(#8): Clean up swapped-out or abandoned resources to avoid leaking file handles or workers.
                if method_name == "shutdown" and timeout_s is not None:
                    method(timeout_s=timeout_s)
                else:
                    method()
                return
            except TypeError:
                try:
                    method()
                    return
                except Exception as exc:
                    self._safe_append_ops_event(
                        event="resource_cleanup_failed",
                        message="Twinr could not fully clean up a replaced runtime resource.",
                        data={
                            "error_type": type(exc).__name__,
                            "resource_type": resource.__class__.__name__,
                            "method": method_name,
                        },
                    )
                    return
            except Exception as exc:
                self._safe_append_ops_event(
                    event="resource_cleanup_failed",
                    message="Twinr could not fully clean up a replaced runtime resource.",
                    data={
                        "error_type": type(exc).__name__,
                        "resource_type": resource.__class__.__name__,
                        "method": method_name,
                    },
                )
                return

    @staticmethod
    def _parse_aware_utc_datetime(value: str | None) -> datetime | None:
        raw = (value or "").strip()
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            # AUDIT-FIX(#2): Reject timezone-naive timestamps so voice-trust freshness cannot drift on DST/local clock assumptions.
            return None
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _normalize_voice_status(status: str | None) -> str | None:
        normalized = (status or "").strip().lower()
        if not normalized:
            return None
        if normalized in _ALLOWED_VOICE_STATUSES:
            return normalized
        # AUDIT-FIX(#1): Collapse unknown values to a conservative state instead of injecting raw text into a system prompt.
        return "unknown_voice"

    @staticmethod
    def _normalize_confidence(confidence: float | None) -> float | None:
        if confidence is None:
            return None
        try:
            value = float(confidence)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        # AUDIT-FIX(#7): Clamp confidence to a real percentage range before it influences provider authorization guidance.
        return max(0.0, min(1.0, value))

    def _normalize_checked_at(self, checked_at: str | None) -> str | None:
        parsed = self._parse_aware_utc_datetime(checked_at)
        if parsed is None:
            return None
        return parsed.isoformat().replace("+00:00", "Z")

    def _voice_assessment_max_age_s(self) -> int:
        raw = getattr(getattr(self, "config", None), "voice_assessment_max_age_s", _DEFAULT_VOICE_ASSESSMENT_MAX_AGE_S)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return _DEFAULT_VOICE_ASSESSMENT_MAX_AGE_S
        return max(1, value)

    def _voice_assessment_is_fresh_unlocked(self) -> bool:
        checked_at = self._parse_aware_utc_datetime(getattr(self, "user_voice_checked_at", None))
        if checked_at is None:
            return False
        now = datetime.now(timezone.utc)
        if checked_at > now + timedelta(seconds=_MAX_FUTURE_SKEW_S):
            return False
        age_s = max(0.0, (now - checked_at).total_seconds())
        return age_s <= float(self._voice_assessment_max_age_s())

    @staticmethod
    def _coerce_non_negative_int(value: object, *, default: int = 0) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        # AUDIT-FIX(#9): Prevent invalid negative timing observations from poisoning adaptive listening behavior.
        return max(0, parsed)

    @staticmethod
    def _require_int(value: object, *, field_name: str, minimum: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be an integer") from exc
        if parsed < minimum:
            raise ValueError(f"{field_name} must be >= {minimum}")
        return parsed

    def _conversation_context_unlocked(self) -> tuple[tuple[str, str], ...]:
        turns = tuple(getattr(getattr(self, "memory", None), "turns", ()) or ())
        messages: list[tuple[str, str]] = []
        for turn in turns:
            role = getattr(turn, "role", None)
            if role is None:
                continue
            content = getattr(turn, "content", "")
            if content is None:
                content = ""
            # AUDIT-FIX(#4): Build a defensive snapshot so malformed turns do not abort prompt assembly.
            messages.append((str(role), str(content)))
        return tuple(messages)

    def _raw_tail_context_unlocked(self, *, limit: int | None = None) -> tuple[tuple[str, str], ...]:
        turns = tuple(getattr(getattr(self, "memory", None), "raw_tail", ()) or ())
        if limit is not None:
            turns = turns[-max(limit, 0) :]
        messages: list[tuple[str, str]] = []
        for turn in turns:
            role = getattr(turn, "role", None)
            if role is None:
                continue
            content = getattr(turn, "content", "")
            if content is None:
                content = ""
            messages.append((str(role), str(content)))
        return tuple(messages)

    def conversation_context(self) -> tuple[tuple[str, str], ...]:
        with self._runtime_context_lock():
            return self._conversation_context_unlocked()

    def _provider_context_messages(self, *, tool_context: bool) -> tuple[tuple[str, str], ...]:
        with self._runtime_context_lock():
            messages: list[tuple[str, str]] = []
            try:
                contract = memory_and_response_contract(self.config.openai_realtime_language)
            except Exception as exc:
                self._safe_append_ops_event(
                    event="provider_context_contract_failed",
                    message="Twinr could not build the base language contract and continued with reduced context.",
                    data={"error_type": type(exc).__name__},
                )
            else:
                messages.append(("system", contract))

            guidance = self._voice_guidance_message()
            if guidance:
                messages.append(("system", guidance))

            last_transcript = str(getattr(self, "last_transcript", "") or "")
            try:
                context_builder = (
                    self.long_term_memory.build_tool_provider_context(last_transcript)
                    if tool_context
                    else self.long_term_memory.build_provider_context(last_transcript)
                )
                for context_message in context_builder.system_messages():
                    messages.append(("system", str(context_message)))
            except Exception as exc:
                # AUDIT-FIX(#4): Long-term-memory failure should degrade to short-term context, not crash the turn.
                self._safe_append_ops_event(
                    event="provider_context_memory_failed",
                    message="Twinr skipped long-term memory context for this turn after a runtime error.",
                    data={
                        "error_type": type(exc).__name__,
                        "tool_context": tool_context,
                    },
                )

            messages.extend(self._conversation_context_unlocked())
            return tuple(messages)

    def provider_conversation_context(self) -> tuple[tuple[str, str], ...]:
        return self._provider_context_messages(tool_context=False)

    def tool_provider_conversation_context(self) -> tuple[tuple[str, str], ...]:
        return self._provider_context_messages(tool_context=True)

    def supervisor_provider_conversation_context(self) -> tuple[tuple[str, str], ...]:
        with self._runtime_context_lock():
            messages: list[tuple[str, str]] = []
            try:
                contract = memory_and_response_contract(self.config.openai_realtime_language)
            except Exception as exc:
                self._safe_append_ops_event(
                    event="supervisor_context_contract_failed",
                    message="Twinr could not build the fast-lane language contract and continued with reduced context.",
                    data={"error_type": type(exc).__name__},
                )
            else:
                messages.append(("system", contract))

            guidance = self._voice_guidance_message()
            if guidance:
                messages.append(("system", guidance))

            messages.extend(
                self._raw_tail_context_unlocked(limit=max(int(self.config.streaming_supervisor_context_turns), 0))
            )
            return tuple(messages)

    def update_user_voice_assessment(
        self,
        *,
        status: str | None,
        confidence: float | None,
        checked_at: str | None,
    ) -> None:
        with self._runtime_context_lock():
            normalized_status = self._normalize_voice_status(status)
            normalized_confidence = self._normalize_confidence(confidence)
            normalized_checked_at = self._normalize_checked_at(checked_at)
            # AUDIT-FIX(#1): Persist only validated voice-state enums so provider prompts cannot be text-injected.
            self.user_voice_status = normalized_status
            # AUDIT-FIX(#7): Persist bounded confidence only.
            self.user_voice_confidence = normalized_confidence
            # AUDIT-FIX(#2): Persist only timezone-aware UTC timestamps for freshness checks.
            self.user_voice_checked_at = normalized_checked_at if normalized_status else None
            self._safe_persist_snapshot(event_on_error="voice_assessment_snapshot_failed")

    def apply_live_config(self, config) -> None:
        new_graph_memory = None
        new_long_term_memory = None
        new_proactive_governor = None
        new_adaptive_timing_store = None

        with self._runtime_context_lock():
            old_graph_memory = getattr(self, "graph_memory", None)
            old_long_term_memory = getattr(self, "long_term_memory", None)
            old_proactive_governor = getattr(self, "proactive_governor", None)
            old_adaptive_timing_store = getattr(self, "adaptive_timing_store", None)

            if old_adaptive_timing_store is None:
                raise RuntimeError("adaptive_timing_store must be initialized before apply_live_config")

            try:
                memory_max_turns = self._require_int(
                    config.memory_max_turns,
                    field_name="memory_max_turns",
                    minimum=1,
                )
                memory_keep_recent = self._require_int(
                    config.memory_keep_recent,
                    field_name="memory_keep_recent",
                    minimum=0,
                )
                # AUDIT-FIX(#6): Build all replacement dependencies before mutating the live runtime.
                new_graph_memory = TwinrPersonalGraphStore.from_config(config)
                new_long_term_memory = LongTermMemoryService.from_config(
                    config,
                    graph_store=new_graph_memory,
                )
                new_proactive_governor = ProactiveGovernor.from_config(config)
                new_adaptive_timing_store = old_adaptive_timing_store.__class__(
                    config.adaptive_timing_store_path,
                    config=config,
                )
                if config.adaptive_timing_enabled:
                    new_adaptive_timing_store.ensure_saved()
                self.memory.reconfigure(
                    max_turns=memory_max_turns,
                    keep_recent=memory_keep_recent,
                )
            except Exception as exc:
                self._best_effort_cleanup(new_long_term_memory, timeout_s=1.0)
                self._best_effort_cleanup(new_graph_memory)
                self._best_effort_cleanup(new_proactive_governor)
                self._best_effort_cleanup(new_adaptive_timing_store)
                self._safe_append_ops_event(
                    event="live_config_apply_failed",
                    message="Twinr rejected a live config update and kept the previous runtime configuration.",
                    data={"error_type": type(exc).__name__},
                )
                raise

            # AUDIT-FIX(#6): Swap the runtime atomically only after every replacement dependency is ready.
            self.config = config
            self.graph_memory = new_graph_memory
            self.long_term_memory = new_long_term_memory
            self.proactive_governor = new_proactive_governor
            self.adaptive_timing_store = new_adaptive_timing_store

            self._safe_persist_snapshot(event_on_error="live_config_snapshot_failed")

        # AUDIT-FIX(#8): Clean up replaced resources after the successful swap so they do not linger in memory or on disk.
        if old_long_term_memory is not None and old_long_term_memory is not new_long_term_memory:
            self._best_effort_cleanup(old_long_term_memory, timeout_s=1.0)
        if old_graph_memory is not None and old_graph_memory is not new_graph_memory:
            self._best_effort_cleanup(old_graph_memory)
        if old_proactive_governor is not None and old_proactive_governor is not new_proactive_governor:
            self._best_effort_cleanup(old_proactive_governor)
        if old_adaptive_timing_store is not None and old_adaptive_timing_store is not new_adaptive_timing_store:
            self._best_effort_cleanup(old_adaptive_timing_store)

    def _fallback_listening_window(self, *, initial_source: str, follow_up: bool) -> AdaptiveListeningWindow:
        return AdaptiveListeningWindow(
            start_timeout_s=(
                self.config.audio_start_timeout_s
                if initial_source == "button" and not follow_up
                else self.config.conversation_follow_up_timeout_s
            ),
            speech_pause_ms=self.config.speech_pause_ms,
            pause_grace_ms=self.config.adaptive_timing_pause_grace_ms,
        )

    def listening_window(self, *, initial_source: str, follow_up: bool) -> AdaptiveListeningWindow:
        with self._runtime_context_lock():
            if self.config.adaptive_timing_enabled:
                try:
                    return self.adaptive_timing_store.listening_window(
                        initial_source=initial_source,
                        follow_up=follow_up,
                    )
                except Exception as exc:
                    # AUDIT-FIX(#5): Adaptive timing failure must fall back to static timing instead of killing audio capture.
                    self._safe_append_ops_event(
                        event="adaptive_timing_window_failed",
                        message="Twinr fell back to static listening timing after an adaptive timing error.",
                        data={"error_type": type(exc).__name__},
                    )
            return self._fallback_listening_window(
                initial_source=initial_source,
                follow_up=follow_up,
            )

    def remember_listen_timeout(self, *, initial_source: str, follow_up: bool) -> AdaptiveTimingProfile | None:
        with self._runtime_context_lock():
            if not self.config.adaptive_timing_enabled:
                return None
            try:
                previous = self.adaptive_timing_store.current()
                updated = self.adaptive_timing_store.record_no_speech_timeout(
                    initial_source=initial_source,
                    follow_up=follow_up,
                )
            except Exception as exc:
                # AUDIT-FIX(#5): Learning failures must not abort the live conversation loop.
                self._safe_append_ops_event(
                    event="adaptive_timing_record_timeout_failed",
                    message="Twinr could not learn from a listen-timeout event and continued with the current timing profile.",
                    data={"error_type": type(exc).__name__},
                )
                return None
            self._record_adaptive_timing_event(
                previous,
                updated,
                reason="timeout",
                initial_source=initial_source,
                follow_up=follow_up,
            )
            return updated

    def remember_listen_capture(
        self,
        *,
        initial_source: str,
        follow_up: bool,
        speech_started_after_ms: int,
        resumed_after_pause_count: int,
    ) -> AdaptiveTimingProfile | None:
        safe_speech_started_after_ms = self._coerce_non_negative_int(speech_started_after_ms)
        safe_resumed_after_pause_count = self._coerce_non_negative_int(resumed_after_pause_count)

        with self._runtime_context_lock():
            if not self.config.adaptive_timing_enabled:
                return None
            try:
                previous = self.adaptive_timing_store.current()
                updated = self.adaptive_timing_store.record_capture(
                    initial_source=initial_source,
                    follow_up=follow_up,
                    speech_started_after_ms=safe_speech_started_after_ms,
                    resumed_after_pause_count=safe_resumed_after_pause_count,
                )
            except Exception as exc:
                # AUDIT-FIX(#5): Learning failures must not abort the live conversation loop.
                self._safe_append_ops_event(
                    event="adaptive_timing_record_capture_failed",
                    message="Twinr could not learn from a captured utterance and continued with the current timing profile.",
                    data={"error_type": type(exc).__name__},
                )
                return None
            self._record_adaptive_timing_event(
                previous,
                updated,
                reason="capture",
                initial_source=initial_source,
                follow_up=follow_up,
                speech_started_after_ms=safe_speech_started_after_ms,
                resumed_after_pause_count=safe_resumed_after_pause_count,
            )
            return updated

    def _record_adaptive_timing_event(
        self,
        previous: AdaptiveTimingProfile | None,
        updated: AdaptiveTimingProfile | None,
        *,
        reason: str,
        initial_source: str,
        follow_up: bool,
        speech_started_after_ms: int | None = None,
        resumed_after_pause_count: int | None = None,
    ) -> None:
        if updated is None or updated == previous:
            return
        try:
            data: dict[str, object] = {
                "reason": reason,
                "initial_source": initial_source,
                "follow_up": follow_up,
                "button_start_timeout_s": round(updated.button_start_timeout_s, 2),
                "follow_up_start_timeout_s": round(updated.follow_up_start_timeout_s, 2),
                "speech_pause_ms": updated.speech_pause_ms,
                "pause_grace_ms": updated.pause_grace_ms,
            }
            if speech_started_after_ms is not None:
                data["speech_started_after_ms"] = self._coerce_non_negative_int(speech_started_after_ms)
            if resumed_after_pause_count is not None:
                data["resumed_after_pause_count"] = self._coerce_non_negative_int(resumed_after_pause_count)
        except Exception as exc:
            self._safe_append_ops_event(
                event="adaptive_timing_event_build_failed",
                message="Twinr skipped adaptive timing telemetry after a profile-serialization error.",
                data={"error_type": type(exc).__name__},
            )
            return

        self._safe_append_ops_event(
            event="adaptive_timing_updated",
            message="Twinr adjusted listening timing from observed button and pause behavior.",
            data=data,
        )

    def _voice_guidance_message(self) -> str | None:
        with self._runtime_context_lock():
            status = self._normalize_voice_status(getattr(self, "user_voice_status", None))
            if not status:
                return None

            if not self._voice_assessment_is_fresh_unlocked():
                # AUDIT-FIX(#2): Expire stale voice verification state before it can influence prompt-level authorization hints.
                self.user_voice_status = None
                self.user_voice_confidence = None
                self.user_voice_checked_at = None
                self._safe_persist_snapshot(event_on_error="voice_assessment_expiry_snapshot_failed")
                return None

            if status == "likely_user":
                signal = "likely match to the enrolled main-user voice profile"
            elif status == "uncertain":
                signal = "partial match to the enrolled main-user voice profile"
            else:
                signal = "does not match the enrolled main-user voice profile closely enough"

            parts = [
                "Live speaker signal for this turn. Treat it as a local verification signal, not proof of identity.",
                f"Speaker signal: {signal}.",
            ]
            confidence = self._normalize_confidence(getattr(self, "user_voice_confidence", None))
            if confidence is not None:
                parts.append(f"Confidence: {confidence * 100:.0f}%.")

            if status in {"uncertain", "unknown_voice"}:
                parts.append(
                    "For persistent or security-sensitive changes, first ask for explicit confirmation. "
                    "Only call tools with confirmed=true after the user clearly confirms in the current conversation."
                )
            else:
                parts.append(
                    "You may use this signal for calmer personalization, but never as the only authorization for a sensitive action."
                )
            return " ".join(parts)
