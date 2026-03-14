from __future__ import annotations

from twinr.agent.base_agent.adaptive_timing import AdaptiveListeningWindow, AdaptiveTimingProfile
from twinr.agent.base_agent.language import memory_and_response_contract
from twinr.memory import LongTermMemoryService, TwinrPersonalGraphStore


class TwinrRuntimeContextMixin:
    def conversation_context(self) -> tuple[tuple[str, str], ...]:
        return tuple((turn.role, turn.content) for turn in self.memory.turns)

    def provider_conversation_context(self) -> tuple[tuple[str, str], ...]:
        messages: list[tuple[str, str]] = []
        messages.append(("system", memory_and_response_contract(self.config.openai_realtime_language)))
        guidance = self._voice_guidance_message()
        if guidance:
            messages.append(("system", guidance))
        for context_message in self.long_term_memory.build_provider_context(self.last_transcript).system_messages():
            messages.append(("system", context_message))
        messages.extend(self.conversation_context())
        return tuple(messages)

    def update_user_voice_assessment(
        self,
        *,
        status: str | None,
        confidence: float | None,
        checked_at: str | None,
    ) -> None:
        self.user_voice_status = (status or "").strip() or None
        self.user_voice_confidence = confidence
        self.user_voice_checked_at = (checked_at or "").strip() or None
        self._persist_snapshot()

    def apply_live_config(self, config) -> None:
        self.config = config
        self.memory.reconfigure(
            max_turns=config.memory_max_turns,
            keep_recent=config.memory_keep_recent,
        )
        self.long_term_memory.shutdown(timeout_s=1.0)
        self.graph_memory = TwinrPersonalGraphStore.from_config(config)
        self.long_term_memory = LongTermMemoryService.from_config(
            config,
            graph_store=self.graph_memory,
        )
        self.adaptive_timing_store = self.adaptive_timing_store.__class__(
            config.adaptive_timing_store_path,
            config=config,
        )
        if config.adaptive_timing_enabled:
            self.adaptive_timing_store.ensure_saved()
        self._persist_snapshot()

    def listening_window(self, *, initial_source: str, follow_up: bool) -> AdaptiveListeningWindow:
        if self.config.adaptive_timing_enabled:
            return self.adaptive_timing_store.listening_window(
                initial_source=initial_source,
                follow_up=follow_up,
            )
        return AdaptiveListeningWindow(
            start_timeout_s=(
                self.config.audio_start_timeout_s
                if initial_source == "button" and not follow_up
                else self.config.conversation_follow_up_timeout_s
            ),
            speech_pause_ms=self.config.speech_pause_ms,
            pause_grace_ms=self.config.adaptive_timing_pause_grace_ms,
        )

    def remember_listen_timeout(self, *, initial_source: str, follow_up: bool) -> AdaptiveTimingProfile | None:
        if not self.config.adaptive_timing_enabled:
            return None
        previous = self.adaptive_timing_store.current()
        updated = self.adaptive_timing_store.record_no_speech_timeout(
            initial_source=initial_source,
            follow_up=follow_up,
        )
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
        if not self.config.adaptive_timing_enabled:
            return None
        previous = self.adaptive_timing_store.current()
        updated = self.adaptive_timing_store.record_capture(
            initial_source=initial_source,
            follow_up=follow_up,
            speech_started_after_ms=speech_started_after_ms,
            resumed_after_pause_count=resumed_after_pause_count,
        )
        self._record_adaptive_timing_event(
            previous,
            updated,
            reason="capture",
            initial_source=initial_source,
            follow_up=follow_up,
            speech_started_after_ms=speech_started_after_ms,
            resumed_after_pause_count=resumed_after_pause_count,
        )
        return updated

    def _record_adaptive_timing_event(
        self,
        previous: AdaptiveTimingProfile,
        updated: AdaptiveTimingProfile,
        *,
        reason: str,
        initial_source: str,
        follow_up: bool,
        speech_started_after_ms: int | None = None,
        resumed_after_pause_count: int | None = None,
    ) -> None:
        if updated == previous:
            return
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
            data["speech_started_after_ms"] = int(speech_started_after_ms)
        if resumed_after_pause_count is not None:
            data["resumed_after_pause_count"] = int(resumed_after_pause_count)
        self.ops_events.append(
            event="adaptive_timing_updated",
            message="Twinr adjusted listening timing from observed button and pause behavior.",
            data=data,
        )

    def _voice_guidance_message(self) -> str | None:
        status = (self.user_voice_status or "").strip().lower()
        if not status:
            return None
        if status == "likely_user":
            signal = "likely match to the enrolled main-user voice profile"
        elif status == "uncertain":
            signal = "partial match to the enrolled main-user voice profile"
        elif status == "unknown_voice":
            signal = "does not match the enrolled main-user voice profile closely enough"
        else:
            signal = status.replace("_", " ")

        parts = [
            "Live speaker signal for this turn. Treat it as a local heuristic, not proof of identity.",
            f"Speaker signal: {signal}.",
        ]
        if self.user_voice_confidence is not None:
            parts.append(f"Confidence: {self.user_voice_confidence * 100:.0f}%.")
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
