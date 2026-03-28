"""Runtime-state lifecycle helpers for the orchestrator voice session."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from twinr.hardware.household_voice_identity import HouseholdVoiceProfile
from twinr.orchestrator.voice_runtime_intent import VoiceRuntimeIntentContext

from ..voice_contracts import (
    OrchestratorVoiceFollowUpClosedEvent,
    OrchestratorVoiceHelloRequest,
    OrchestratorVoiceIdentityProfilesEvent,
    OrchestratorVoiceReadyEvent,
    OrchestratorVoiceRuntimeStateEvent,
)


class VoiceSessionRuntimeStateMixin:
    """Own session hello, runtime state, and follow-up timeout behavior."""

    def handle_hello(self, request: OrchestratorVoiceHelloRequest) -> list[dict[str, Any]]:
        """Accept one new edge voice session and validate stream metadata."""

        self._session_id = request.session_id
        self._trace_id = str(request.trace_id or request.session_id or self._trace_id).strip() or self._trace_id
        self._received_frame_bucket = self._received_frame_bucket.__class__(
            chunk_ms=self.chunk_ms,
            speech_threshold=self.speech_threshold,
        )
        if int(request.sample_rate) != self.sample_rate:
            return [
                {
                    "type": "voice_error",
                    "error": (
                        "Voice session sample rate mismatch. "
                        f"Expected {self.sample_rate} Hz but received {request.sample_rate} Hz."
                    ),
                }
            ]
        if int(request.channels) != self.channels:
            return [
                {
                    "type": "voice_error",
                    "error": (
                        "Voice session channel mismatch. "
                        f"Expected {self.channels} channel(s) but received {request.channels}."
                    ),
                }
            ]
        raw_initial_state = request.initial_state or "waiting"
        self._state = self._normalize_runtime_state(raw_initial_state)
        self._follow_up_allowed = bool(getattr(request, "follow_up_allowed", False))
        self._runtime_state_attested = bool(getattr(request, "state_attested", False))
        self._voice_quiet_until_utc = self._normalize_voice_quiet_until_utc(
            getattr(request, "voice_quiet_until_utc", None)
        )
        self._intent_context = VoiceRuntimeIntentContext.from_runtime_event(request)
        if self._state == "follow_up_open" and self._follow_up_allowed:
            now = self._monotonic()
            self._follow_up_opened_at = now
            self._follow_up_deadline_at = now + self._effective_follow_up_timeout_s()
        else:
            self._follow_up_deadline_at = None
            self._follow_up_opened_at = None
        self._refresh_waiting_visibility_anchor()
        self._trace_event(
            "voice_session_hello_accepted",
            kind="run_start",
            details={
                "sample_rate": request.sample_rate,
                "channels": request.channels,
                "chunk_ms": request.chunk_ms,
                "initial_state": self._state,
                "raw_initial_state": raw_initial_state,
                "detail": getattr(request, "detail", None),
                "follow_up_allowed": self._follow_up_allowed,
                "state_attested": self._runtime_state_attested,
                **self._intent_context.trace_details(),
            },
        )
        return [OrchestratorVoiceReadyEvent(session_id=request.session_id, backend=self.backend_name).to_payload()]

    def handle_runtime_state(self, event: OrchestratorVoiceRuntimeStateEvent) -> list[dict[str, Any]]:
        """Update explicit edge runtime state and drain any timeout-based events."""

        return self._apply_runtime_state(
            event,
            trace_event_name="voice_runtime_state_received",
            trace_kind="mutation",
        )

    def handle_identity_profiles(
        self,
        event: OrchestratorVoiceIdentityProfilesEvent,
    ) -> list[dict[str, Any]]:
        """Update the read-only household voice profiles used for wake bias."""

        profiles: list[HouseholdVoiceProfile] = []
        for profile_event in event.profiles:
            profile = HouseholdVoiceProfile.from_dict(
                {
                    "user_id": profile_event.user_id,
                    "display_name": profile_event.display_name,
                    "primary_user": profile_event.primary_user,
                    "embedding": list(profile_event.embedding),
                    "sample_count": profile_event.sample_count,
                    "average_duration_ms": profile_event.average_duration_ms,
                    "updated_at": profile_event.updated_at,
                }
            )
            if profile is not None:
                profiles.append(profile)
        self._voice_identity_profiles = tuple(
            sorted(profiles, key=lambda item: (not item.primary_user, item.user_id))
        )
        self._voice_identity_profiles_revision = str(event.revision or "").strip() or None
        self._trace_event(
            "voice_identity_profiles_received",
            kind="mutation",
            details={
                "voice_identity_profiles_revision": self._voice_identity_profiles_revision,
                "voice_identity_profiles_count": len(self._voice_identity_profiles),
            },
        )
        return []

    def _apply_runtime_state(
        self,
        event: OrchestratorVoiceRuntimeStateEvent,
        *,
        trace_event_name: str,
        trace_kind: str,
    ) -> list[dict[str, Any]]:
        """Apply one runtime-state snapshot and drain any timeout-based events."""

        previous_state = self._state
        raw_state = event.state or "waiting"
        self._state = self._normalize_runtime_state(raw_state)
        self._follow_up_allowed = bool(event.follow_up_allowed)
        self._runtime_state_attested = True
        self._voice_quiet_until_utc = self._normalize_voice_quiet_until_utc(
            getattr(event, "voice_quiet_until_utc", None)
        )
        self._intent_context = VoiceRuntimeIntentContext.from_runtime_event(event)
        if self._state == "follow_up_open" and self._follow_up_allowed:
            now = self._monotonic()
            if previous_state != "follow_up_open" or self._follow_up_opened_at is None:
                self._follow_up_opened_at = now
            self._follow_up_deadline_at = self._follow_up_opened_at + self._effective_follow_up_timeout_s()
        else:
            self._follow_up_deadline_at = None
            self._follow_up_opened_at = None
        self._refresh_waiting_visibility_anchor()
        if previous_state != self._state and self._uses_remote_asr_utterance_path():
            self._reset_remote_asr_utterance_history(
                previous_state=previous_state,
                detail=event.detail,
            )
        if self._state != "speaking":
            self._barge_in_sent = False
        self._cancel_blocked_waiting_activation_buffers(
            previous_state=previous_state,
            detail=event.detail,
        )
        if not self._uses_remote_asr_utterance_path():
            self._pending_transcript_utterance = None
        if self._state not in self._ACTIVE_STATES:
            self._pending_transcript_utterance = None
        self._trace_event(
            trace_event_name,
            kind=trace_kind,
            details={
                "previous_state": previous_state,
                "new_state": self._state,
                "raw_state": raw_state,
                "detail": event.detail,
                "follow_up_allowed": self._follow_up_allowed,
                "state_attested": self._runtime_state_attested,
                **self._intent_context.trace_details(),
            },
        )
        return self._drain_timeouts()

    def _runtime_state_event_matches_current(self, event: OrchestratorVoiceRuntimeStateEvent) -> bool:
        """Return whether one incoming runtime snapshot matches current session state."""

        if not self._runtime_state_attested:
            return False
        if self._normalize_runtime_state(event.state or "waiting") != self._state:
            return False
        if bool(event.follow_up_allowed) != self._follow_up_allowed:
            return False
        if self._normalize_voice_quiet_until_utc(getattr(event, "voice_quiet_until_utc", None)) != self._voice_quiet_until_utc:
            return False
        return VoiceRuntimeIntentContext.from_runtime_event(event) == self._intent_context

    def _normalize_runtime_state(self, state: str | None) -> str:
        """Map retired runtime-state labels onto the supported remote-only set."""

        normalized = str(state or "").strip() or "waiting"
        if normalized == "wake_armed":
            return "waiting"
        if normalized in self._SUPPORTED_RUNTIME_STATES:
            return normalized
        return normalized

    def _reset_remote_asr_utterance_history(
        self,
        *,
        previous_state: str,
        detail: str | None,
    ) -> None:
        """Drop stale stream history before a fresh remote utterance window starts."""

        preserved_pending = self._preserve_pending_waiting_utterance_on_listening_handoff(
            previous_state=previous_state,
            detail=detail,
        )
        self._history.clear()
        self._pending_transcript_utterance = preserved_pending
        self._trace_event(
            "voice_remote_asr_utterance_history_reset",
            kind="mutation",
            details={
                "previous_state": previous_state,
                "new_state": self._state,
                "detail": detail,
                "preserved_pending_utterance": preserved_pending is not None,
                "preserved_pending_active_ms": (
                    preserved_pending.active_ms if preserved_pending is not None else 0
                ),
                "preserved_pending_captured_ms": (
                    preserved_pending.captured_ms if preserved_pending is not None else 0
                ),
            },
        )

    def _preserve_pending_waiting_utterance_on_listening_handoff(
        self,
        *,
        previous_state: str,
        detail: str | None,
    ):
        """Carry a just-started waiting utterance across the listen-beep handoff."""

        pending = self._pending_transcript_utterance
        if pending is None:
            return None
        if previous_state != "waiting" or self._state != "listening":
            return None
        if pending.origin_state != "waiting":
            return None
        if pending.active_ms <= 0:
            return None
        if pending.trailing_silence_ms >= self.wake_tail_endpoint_silence_ms:
            return None
        carried_frames = self._latest_active_speech_burst_frames(tuple(pending.frames))
        if not carried_frames:
            return None
        promoted = self._pending_transcript_utterance_from_frames(
            origin_state="listening",
            frames=carried_frames,
            max_capture_ms=pending.max_capture_ms,
        )
        self._trace_event(
            "voice_remote_asr_listening_handoff_preserved",
            kind="mutation",
            details={
                "previous_state": previous_state,
                "new_state": self._state,
                "detail": detail,
                "carried_frame_count": len(carried_frames),
                "carried_active_ms": promoted.active_ms,
                "carried_captured_ms": promoted.captured_ms,
            },
        )
        return promoted

    def _cancel_blocked_waiting_activation_buffers(
        self,
        *,
        previous_state: str,
        detail: str | None,
    ) -> None:
        """Drop buffered waiting audio once live context explicitly blocks speech."""

        if self._state != "waiting":
            return
        if self._waiting_activation_allowed():
            return
        if self._pending_transcript_utterance is None:
            return
        self._history.clear()
        self._pending_transcript_utterance = None
        details = {
            "previous_state": previous_state,
            "new_state": self._state,
            "detail": detail,
            **self._intent_context.trace_details(),
        }
        self._record_transcript_debug(
            stage="activation_utterance",
            outcome="cancelled_context_blocked",
            details=details,
        )
        self._trace_event(
            "voice_waiting_activation_cancelled_context_blocked",
            kind="mutation",
            details=details,
        )

    def _uses_remote_asr_utterance_path(self) -> bool:
        """Return whether the same-stream remote ASR utterance scanner owns routing."""

        if self._state in self._REMOTE_ASR_UTTERANCE_STATES:
            return True
        return self._state == "follow_up_open" and self._follow_up_allowed

    def _intent_audio_bias_active(self) -> bool:
        """Return whether compact multimodal context may relax audio-owned gates."""

        return self._intent_context.audio_bias_allowed()

    def _refresh_waiting_visibility_anchor(self) -> None:
        """Remember the last attested visible waiting context."""

        if not self._runtime_state_attested:
            return
        if self._state != "waiting":
            self._last_waiting_visible_at = None
            return
        if self._intent_context.person_visible is True:
            self._last_waiting_visible_at = self._monotonic()

    def _normalize_voice_quiet_until_utc(self, value: object | None) -> str | None:
        """Return a canonical future UTC quiet deadline or ``None``."""

        text = str(value or "").strip()
        if not text:
            return None
        normalized_text = f"{text[:-1]}+00:00" if text.endswith("Z") else text
        try:
            deadline = datetime.fromisoformat(normalized_text)
        except ValueError:
            return None
        if deadline.tzinfo is None:
            return None
        deadline = deadline.astimezone(timezone.utc)
        if deadline <= datetime.now(timezone.utc):
            return None
        return deadline.isoformat().replace("+00:00", "Z")

    def _voice_quiet_active(self) -> bool:
        """Return whether the temporary voice-quiet window is still active."""

        normalized_deadline = self._normalize_voice_quiet_until_utc(self._voice_quiet_until_utc)
        self._voice_quiet_until_utc = normalized_deadline
        return normalized_deadline is not None

    def _close_follow_up_open(self, *, reason: str) -> list[dict[str, Any]]:
        """Close the current remote follow-up window with an explicit reason."""

        self._follow_up_deadline_at = None
        self._follow_up_opened_at = None
        self._pending_transcript_utterance = None
        self._state = "waiting"
        self._trace_event(
            "voice_follow_up_closed",
            kind="mutation",
            details={"reason": reason},
        )
        return [OrchestratorVoiceFollowUpClosedEvent(reason=reason).to_payload()]

    def _waiting_visibility_grace_active(self) -> bool:
        """Return whether a recent visible waiting context is still fresh."""

        if self._state != "waiting":
            return False
        last_visible_at = self._last_waiting_visible_at
        if last_visible_at is None:
            return False
        return (self._monotonic() - last_visible_at) <= self._WAITING_VISIBILITY_GRACE_S

    def _waiting_activation_allowed(self) -> bool:
        """Return whether idle transcript-first scanning may open a new utterance."""

        if self._state != "waiting":
            return True
        if self._voice_quiet_active():
            return False
        if not self._runtime_state_attested:
            return False
        if self._intent_context.waiting_activation_allowed():
            return True
        return self._waiting_visibility_grace_active()

    def _effective_remote_asr_stage1_window_ms(self) -> int:
        """Return the current transcript-first stage-one scan window."""

        window_ms = self.remote_asr_stage1_window_ms
        if self._intent_audio_bias_active():
            window_ms = min(
                self.history_ms,
                window_ms + self.intent_stage1_window_bonus_ms,
            )
        return window_ms

    def _effective_remote_asr_min_activation_duration_ms(self) -> int:
        """Return the bounded minimum activation duration for the current context."""

        required_ms = self.remote_asr_min_wake_duration_ms
        if self._intent_audio_bias_active() or self._waiting_visibility_grace_active():
            required_ms -= self.intent_min_wake_duration_relief_ms
        return max(self.chunk_ms, required_ms)

    def _effective_follow_up_timeout_s(self) -> float:
        """Return the bounded follow-up timeout for the current multimodal context."""

        if not self._intent_audio_bias_active():
            return self.follow_up_timeout_s
        return self.follow_up_timeout_s + self.intent_follow_up_timeout_bonus_s

    def _drain_timeouts(self) -> list[dict[str, Any]]:
        if self._state != "follow_up_open":
            return []
        if self._voice_quiet_active():
            return self._close_follow_up_open(reason="voice_quiet_active")
        if not self._follow_up_allowed:
            return self._close_follow_up_open(reason="follow_up_disabled")
        if self._follow_up_deadline_at is None:
            return []
        if self._monotonic() < self._follow_up_deadline_at:
            return []
        return self._close_follow_up_open(reason="timeout")
