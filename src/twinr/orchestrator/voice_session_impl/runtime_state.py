# CHANGELOG: 2026-03-29
# BUG-1: Validate hello metadata before mutating session state and reject chunk_ms mismatches.
# BUG-2: Reset per-session audio/follow-up buffers on hello and clear stale follow-up history on close.
# BUG-3: Fail closed for unknown runtime states and downgrade unattested/inconsistent hello snapshots to waiting.
# SEC-1: Fence runtime-state/profile events to the active session when session_id is present, preventing stale cross-session mutation.
# SEC-2: Bound identity-profile ingestion and temporary voice-quiet windows to reduce practical DoS risk on Raspberry Pi deployments.
# IMP-1: Track quiet windows with monotonic deadlines so NTP / RTC clock jumps do not prolong or prematurely expire gates.
# IMP-2: Add frontier-ready runtime policy hooks (semantic turn eagerness / idle timeout / per-session overrides) with richer tracing.

"""Runtime-state lifecycle helpers for the orchestrator voice session."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from itertools import islice
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

    _DEFAULT_MAX_VOICE_IDENTITY_PROFILES = 32
    _DEFAULT_MAX_VOICE_IDENTITY_EMBEDDING_VALUES = 2048
    _DEFAULT_MAX_VOICE_QUIET_WINDOW_S = 900.0
    _SEMANTIC_EAGERNESS_TIMEOUT_S = {
        "high": 2.0,
        "medium": 4.0,
        "auto": 4.0,
        "low": 8.0,
    }

    def handle_hello(self, request: OrchestratorVoiceHelloRequest) -> list[dict[str, Any]]:
        """Accept one new edge voice session and validate stream metadata."""

        error = self._validate_hello_request(request)
        if error is not None:
            return [self._voice_error_payload(error)]

        self._session_id = request.session_id
        self._trace_id = str(request.trace_id or request.session_id or self._trace_id).strip() or self._trace_id
        self._reset_session_runtime_buffers()

        raw_initial_state = request.initial_state or "waiting"
        self._state = self._normalize_runtime_state(raw_initial_state)
        self._follow_up_allowed = bool(getattr(request, "follow_up_allowed", False))
        self._runtime_state_attested = bool(getattr(request, "state_attested", False))
        quiet_details = self._set_voice_quiet_until_utc(getattr(request, "voice_quiet_until_utc", None))
        self._intent_context = VoiceRuntimeIntentContext.from_runtime_event(request)
        self._apply_runtime_policy_overrides(request)

        hello_state_downgrade_reason: str | None = None
        if not self._runtime_state_attested and self._state != "waiting":
            # BREAKING: unattested hello state is now treated as speculative and downgraded to waiting.
            hello_state_downgrade_reason = "state_not_attested"
            self._state = "waiting"
        elif self._state == "follow_up_open" and not self._follow_up_allowed:
            hello_state_downgrade_reason = "follow_up_disabled"
            self._state = "waiting"
        elif self._state == "follow_up_open" and self._voice_quiet_active():
            hello_state_downgrade_reason = "voice_quiet_active"
            self._state = "waiting"

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
                "hello_state_downgrade_reason": hello_state_downgrade_reason,
                **quiet_details,
                **self._runtime_policy_trace_details(),
                **self._intent_context.trace_details(),
            },
        )
        return [OrchestratorVoiceReadyEvent(session_id=request.session_id, backend=self.backend_name).to_payload()]

    def handle_runtime_state(self, event: OrchestratorVoiceRuntimeStateEvent) -> list[dict[str, Any]]:
        """Update explicit edge runtime state and drain any timeout-based events."""

        if not self._event_matches_active_session(event):
            self._trace_event(
                "voice_runtime_state_ignored_session_mismatch",
                kind="mutation",
                details={
                    "current_session_id": self._current_session_id(),
                    "event_session_id": self._event_session_id(event),
                },
            )
            return self._drain_timeouts()

        incoming_order = self._runtime_state_event_order(event)
        current_order = getattr(self, "_runtime_state_order", None)
        if incoming_order is not None and current_order is not None:
            if incoming_order < current_order:
                self._trace_event(
                    "voice_runtime_state_ignored_stale_order",
                    kind="mutation",
                    details={
                        "current_state_order": current_order,
                        "incoming_state_order": incoming_order,
                        "state": getattr(event, "state", None),
                        "detail": getattr(event, "detail", None),
                    },
                )
                return self._drain_timeouts()
            if incoming_order == current_order:
                if self._runtime_state_event_matches_current(event):
                    self._trace_event(
                        "voice_runtime_state_ignored_duplicate",
                        kind="mutation",
                        details={
                            "state_order": incoming_order,
                            "state": getattr(event, "state", None),
                            "detail": getattr(event, "detail", None),
                        },
                    )
                    return self._drain_timeouts()
                self._trace_event(
                    "voice_runtime_state_ignored_conflicting_duplicate_order",
                    kind="mutation",
                    details={
                        "state_order": incoming_order,
                        "state": getattr(event, "state", None),
                        "detail": getattr(event, "detail", None),
                    },
                )
                return self._drain_timeouts()
        elif self._runtime_state_event_matches_current(event):
            self._trace_event(
                "voice_runtime_state_ignored_duplicate",
                kind="mutation",
                details={
                    "state": getattr(event, "state", None),
                    "detail": getattr(event, "detail", None),
                },
            )
            return self._drain_timeouts()

        return self._apply_runtime_state(
            event,
            trace_event_name="voice_runtime_state_received",
            trace_kind="mutation",
            incoming_order=incoming_order,
        )

    def handle_identity_profiles(
        self,
        event: OrchestratorVoiceIdentityProfilesEvent,
    ) -> list[dict[str, Any]]:
        """Update the read-only household voice profiles used for wake bias."""

        if not self._event_matches_active_session(event):
            self._trace_event(
                "voice_identity_profiles_ignored_session_mismatch",
                kind="mutation",
                details={
                    "current_session_id": self._current_session_id(),
                    "event_session_id": self._event_session_id(event),
                },
            )
            return []

        profiles: list[HouseholdVoiceProfile] = []
        dropped_invalid = 0
        dropped_oversized_embeddings = 0
        dropped_due_to_profile_limit = 0
        max_profiles = self._max_voice_identity_profiles()
        max_embedding_values = self._max_voice_identity_embedding_values()
        profile_events = iter(getattr(event, "profiles", ()) or ())
        for profile_event in islice(profile_events, max_profiles):
            try:
                embedding_iter = iter(getattr(profile_event, "embedding", ()) or ())
            except TypeError:
                dropped_invalid += 1
                continue
            embedding_values = tuple(islice(embedding_iter, max_embedding_values + 1))
            if len(embedding_values) > max_embedding_values:
                dropped_oversized_embeddings += 1
                continue
            profile = HouseholdVoiceProfile.from_dict(
                {
                    "user_id": profile_event.user_id,
                    "display_name": profile_event.display_name,
                    "primary_user": profile_event.primary_user,
                    "embedding": list(embedding_values),
                    "sample_count": profile_event.sample_count,
                    "average_duration_ms": profile_event.average_duration_ms,
                    "updated_at": profile_event.updated_at,
                }
            )
            if profile is None:
                dropped_invalid += 1
                continue
            profiles.append(profile)
        if next(profile_events, None) is not None:
            dropped_due_to_profile_limit = 1
        self._voice_identity_profiles = tuple(
            sorted(profiles, key=lambda item: (not item.primary_user, str(item.user_id or "")))
        )
        self._voice_identity_profiles_revision = str(event.revision or "").strip() or None
        self._trace_event(
            "voice_identity_profiles_received",
            kind="mutation",
            details={
                "voice_identity_profiles_revision": self._voice_identity_profiles_revision,
                "voice_identity_profiles_count": len(self._voice_identity_profiles),
                "voice_identity_profiles_dropped_invalid": dropped_invalid,
                "voice_identity_profiles_dropped_oversized_embeddings": dropped_oversized_embeddings,
                "voice_identity_profiles_dropped_due_to_profile_limit": dropped_due_to_profile_limit,
            },
        )
        return []

    def _apply_runtime_state(
        self,
        event: OrchestratorVoiceRuntimeStateEvent,
        *,
        trace_event_name: str,
        trace_kind: str,
        incoming_order: int | None,
    ) -> list[dict[str, Any]]:
        """Apply one runtime-state snapshot and drain any timeout-based events."""

        previous_state = self._state
        raw_state = event.state or "waiting"
        self._state = self._normalize_runtime_state(raw_state)
        self._follow_up_allowed = bool(event.follow_up_allowed)
        self._runtime_state_attested = True
        quiet_details = self._set_voice_quiet_until_utc(getattr(event, "voice_quiet_until_utc", None))
        self._intent_context = VoiceRuntimeIntentContext.from_runtime_event(event)
        self._apply_runtime_policy_overrides(event)
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
        if incoming_order is not None:
            self._runtime_state_order = incoming_order
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
                "state_order": getattr(self, "_runtime_state_order", None),
                **quiet_details,
                **self._runtime_policy_trace_details(),
                **self._intent_context.trace_details(),
            },
        )
        return self._drain_timeouts()

    def _runtime_state_event_matches_current(self, event: OrchestratorVoiceRuntimeStateEvent) -> bool:
        """Return whether one incoming runtime snapshot matches current session state."""

        if not self._event_matches_active_session(event):
            return False
        if not self._runtime_state_attested:
            return False
        if self._normalize_runtime_state(event.state or "waiting") != self._state:
            return False
        if bool(event.follow_up_allowed) != self._follow_up_allowed:
            return False
        if self._normalize_voice_quiet_until_utc(getattr(event, "voice_quiet_until_utc", None)) != self._voice_quiet_until_utc:
            return False
        if self._derive_runtime_policy_overrides(event) != self._runtime_policy_overrides_snapshot():
            return False
        return VoiceRuntimeIntentContext.from_runtime_event(event) == self._intent_context

    def _normalize_runtime_state(self, state: str | None) -> str:
        """Map retired or unknown runtime-state labels onto the supported remote-only set."""

        normalized = str(state or "").strip() or "waiting"
        if normalized == "wake_armed":
            return "waiting"
        if normalized in self._SUPPORTED_RUNTIME_STATES:
            return normalized
        # BREAKING: unknown runtime states now fail closed to waiting instead of being accepted verbatim.
        return "waiting"

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

        normalized_deadline, _, _ = self._parse_voice_quiet_until_utc(value)
        return normalized_deadline

    def _voice_quiet_active(self) -> bool:
        """Return whether the temporary voice-quiet window is still active."""

        deadline_at = getattr(self, "_voice_quiet_deadline_at", None)
        if deadline_at is None and getattr(self, "_voice_quiet_until_utc", None):
            self._set_voice_quiet_until_utc(self._voice_quiet_until_utc)
            deadline_at = getattr(self, "_voice_quiet_deadline_at", None)
        if deadline_at is None:
            return False
        if self._monotonic() < deadline_at:
            return True
        self._voice_quiet_until_utc = None
        self._voice_quiet_deadline_at = None
        return False

    def _close_follow_up_open(self, *, reason: str) -> list[dict[str, Any]]:
        """Close the current remote follow-up window with an explicit reason."""

        previous_state = self._state
        self._follow_up_deadline_at = None
        self._follow_up_opened_at = None
        self._pending_transcript_utterance = None
        self._history.clear()
        self._barge_in_sent = False
        self._state = "waiting"
        self._refresh_waiting_visibility_anchor()
        self._trace_event(
            "voice_follow_up_closed",
            kind="mutation",
            details={
                "reason": reason,
                "previous_state": previous_state,
                "new_state": self._state,
            },
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

        runtime_override = self._runtime_policy_overrides_snapshot().get("remote_asr_stage1_window_ms")
        if runtime_override is not None:
            return runtime_override
        window_ms = min(self.history_ms, max(self.chunk_ms, self.remote_asr_stage1_window_ms))
        if self._intent_audio_bias_active():
            window_ms = min(
                self.history_ms,
                window_ms + self.intent_stage1_window_bonus_ms,
            )
        return window_ms

    def _effective_remote_asr_min_activation_duration_ms(self) -> int:
        """Return the bounded minimum activation duration for the current context."""

        runtime_override = self._runtime_policy_overrides_snapshot().get("remote_asr_min_activation_duration_ms")
        if runtime_override is not None:
            return min(runtime_override, self._effective_remote_asr_stage1_window_ms())
        required_ms = max(self.chunk_ms, self.remote_asr_min_wake_duration_ms)
        if self._intent_audio_bias_active() or self._waiting_visibility_grace_active():
            required_ms -= self.intent_min_wake_duration_relief_ms
        required_ms = max(self.chunk_ms, required_ms)
        return min(required_ms, self._effective_remote_asr_stage1_window_ms())

    def _effective_follow_up_timeout_s(self) -> float:
        """Return the bounded follow-up timeout for the current multimodal context."""

        runtime_override = self._runtime_policy_overrides_snapshot().get("follow_up_timeout_s")
        if runtime_override is not None:
            return runtime_override
        base_timeout_s = max(0.1, float(self.follow_up_timeout_s))
        if not self._intent_audio_bias_active():
            return base_timeout_s
        return max(0.1, base_timeout_s + self.intent_follow_up_timeout_bonus_s)

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

    def _voice_error_payload(self, message: str) -> dict[str, Any]:
        return {"type": "voice_error", "error": message}

    def _validate_hello_request(self, request: OrchestratorVoiceHelloRequest) -> str | None:
        sample_rate = self._coerce_positive_int(getattr(request, "sample_rate", None))
        if sample_rate is None:
            return "Voice session sample rate metadata is missing or invalid."
        if sample_rate != self.sample_rate:
            return (
                "Voice session sample rate mismatch. "
                f"Expected {self.sample_rate} Hz but received {request.sample_rate} Hz."
            )
        channels = self._coerce_positive_int(getattr(request, "channels", None))
        if channels is None:
            return "Voice session channel metadata is missing or invalid."
        if channels != self.channels:
            return (
                "Voice session channel mismatch. "
                f"Expected {self.channels} channel(s) but received {request.channels}."
            )
        raw_chunk_ms = getattr(request, "chunk_ms", None)
        if raw_chunk_ms is not None:
            chunk_ms = self._coerce_positive_int(raw_chunk_ms)
            if chunk_ms is None:
                return "Voice session chunk duration metadata is missing or invalid."
            if chunk_ms != self.chunk_ms:
                return (
                    "Voice session chunk duration mismatch. "
                    f"Expected {self.chunk_ms} ms but received {request.chunk_ms} ms."
                )
        return None

    def _reset_session_runtime_buffers(self) -> None:
        self._received_frame_bucket = self._received_frame_bucket.__class__(
            chunk_ms=self.chunk_ms,
            speech_threshold=self.speech_threshold,
        )
        self._history.clear()
        self._pending_transcript_utterance = None
        self._follow_up_deadline_at = None
        self._follow_up_opened_at = None
        self._barge_in_sent = False
        self._last_waiting_visible_at = None
        self._voice_quiet_until_utc = None
        self._voice_quiet_deadline_at = None
        self._runtime_state_order = None
        self._clear_runtime_policy_overrides()

    def _current_session_id(self) -> str | None:
        session_id = str(getattr(self, "_session_id", "") or "").strip()
        return session_id or None

    def _event_session_id(self, event: object) -> str | None:
        session_id = str(getattr(event, "session_id", "") or "").strip()
        return session_id or None

    def _event_matches_active_session(self, event: object) -> bool:
        event_session_id = self._event_session_id(event)
        if event_session_id is None:
            return True
        current_session_id = str(getattr(self, "_session_id", "") or "").strip() or None
        if current_session_id is None:
            return True
        return event_session_id == current_session_id

    def _runtime_state_event_order(self, event: object) -> int | None:
        for attribute_name in (
            "runtime_state_revision",
            "state_revision",
            "sequence",
            "sequence_no",
            "sequence_id",
            "version",
            "ordinal",
        ):
            value = getattr(event, attribute_name, None)
            parsed = self._coerce_non_negative_int(value)
            if parsed is not None:
                return parsed
        return None

    def _clear_runtime_policy_overrides(self) -> None:
        self._runtime_policy_overrides = {
            "turn_detection_mode": None,
            "semantic_turn_eagerness": None,
            "follow_up_timeout_s": None,
            "remote_asr_stage1_window_ms": None,
            "remote_asr_min_activation_duration_ms": None,
        }

    def _runtime_policy_overrides_snapshot(self) -> dict[str, Any]:
        overrides = getattr(self, "_runtime_policy_overrides", None)
        if isinstance(overrides, dict):
            return overrides
        self._clear_runtime_policy_overrides()
        return self._runtime_policy_overrides

    def _runtime_policy_trace_details(self) -> dict[str, Any]:
        overrides = self._runtime_policy_overrides_snapshot()
        return {
            "turn_detection_mode": overrides.get("turn_detection_mode"),
            "semantic_turn_eagerness": overrides.get("semantic_turn_eagerness"),
            "follow_up_timeout_override_s": overrides.get("follow_up_timeout_s"),
            "remote_asr_stage1_window_override_ms": overrides.get("remote_asr_stage1_window_ms"),
            "remote_asr_min_activation_override_ms": overrides.get(
                "remote_asr_min_activation_duration_ms"
            ),
        }

    def _apply_runtime_policy_overrides(self, source: object) -> None:
        self._runtime_policy_overrides = self._derive_runtime_policy_overrides(source)

    def _derive_runtime_policy_overrides(self, source: object) -> dict[str, Any]:
        turn_detection_mode = self._normalize_turn_detection_mode(
            getattr(source, "turn_detection_mode", None) or getattr(source, "turn_detection", None)
        )
        semantic_turn_eagerness = self._normalize_semantic_turn_eagerness(
            getattr(source, "semantic_turn_eagerness", None)
            or getattr(source, "turn_eagerness", None)
            or getattr(source, "eagerness", None)
        )
        follow_up_timeout_s = self._coerce_positive_float(getattr(source, "follow_up_timeout_s", None))
        if follow_up_timeout_s is None:
            idle_timeout_ms = self._coerce_positive_int(getattr(source, "idle_timeout_ms", None))
            if idle_timeout_ms is not None:
                follow_up_timeout_s = idle_timeout_ms / 1000.0
        if follow_up_timeout_s is None and semantic_turn_eagerness is not None:
            follow_up_timeout_s = self._SEMANTIC_EAGERNESS_TIMEOUT_S.get(semantic_turn_eagerness)
        max_follow_up_timeout_s = self._max_follow_up_timeout_override_s()
        if follow_up_timeout_s is not None:
            follow_up_timeout_s = max(0.1, min(follow_up_timeout_s, max_follow_up_timeout_s))

        stage1_window_ms = self._coerce_positive_int(
            getattr(source, "remote_asr_stage1_window_ms", None)
            or getattr(source, "stage1_window_ms", None)
        )
        if stage1_window_ms is not None:
            stage1_window_ms = max(self.chunk_ms, min(stage1_window_ms, self.history_ms))

        min_activation_duration_ms = self._coerce_positive_int(
            getattr(source, "remote_asr_min_wake_duration_ms", None)
            or getattr(source, "min_activation_duration_ms", None)
        )
        max_activation_duration_ms = stage1_window_ms or self.history_ms
        if min_activation_duration_ms is not None:
            min_activation_duration_ms = max(
                self.chunk_ms,
                min(min_activation_duration_ms, max_activation_duration_ms),
            )

        return {
            "turn_detection_mode": turn_detection_mode,
            "semantic_turn_eagerness": semantic_turn_eagerness,
            "follow_up_timeout_s": follow_up_timeout_s,
            "remote_asr_stage1_window_ms": stage1_window_ms,
            "remote_asr_min_activation_duration_ms": min_activation_duration_ms,
        }

    def _normalize_turn_detection_mode(self, value: object | None) -> str | None:
        text = str(value or "").strip().lower()
        if not text:
            return None
        if text in {"semantic_vad", "server_vad", "vad", "stt", "manual"}:
            return text
        return text

    def _normalize_semantic_turn_eagerness(self, value: object | None) -> str | None:
        text = str(value or "").strip().lower()
        if text in self._SEMANTIC_EAGERNESS_TIMEOUT_S:
            return text
        return None

    def _set_voice_quiet_until_utc(self, value: object | None) -> dict[str, Any]:
        normalized_deadline, monotonic_deadline, capped = self._parse_voice_quiet_until_utc(value)
        self._voice_quiet_until_utc = normalized_deadline
        self._voice_quiet_deadline_at = monotonic_deadline
        return {
            "voice_quiet_until_utc": normalized_deadline,
            "voice_quiet_window_capped": capped,
        }

    def _parse_voice_quiet_until_utc(self, value: object | None) -> tuple[str | None, float | None, bool]:
        text = str(value or "").strip()
        if not text:
            return None, None, False
        normalized_text = f"{text[:-1]}+00:00" if text.endswith("Z") else text
        try:
            deadline = datetime.fromisoformat(normalized_text)
        except ValueError:
            return None, None, False
        if deadline.tzinfo is None or deadline.utcoffset() is None:
            return None, None, False
        deadline = deadline.astimezone(timezone.utc)
        now_utc = datetime.now(timezone.utc)
        remaining_s = (deadline - now_utc).total_seconds()
        if remaining_s <= 0:
            return None, None, False
        capped = False
        max_window_s = self._max_voice_quiet_window_s()
        if max_window_s is not None and remaining_s > max_window_s:
            deadline = now_utc + timedelta(seconds=max_window_s)
            remaining_s = max_window_s
            capped = True
        return (
            deadline.isoformat().replace("+00:00", "Z"),
            self._monotonic() + remaining_s,
            capped,
        )

    def _max_voice_quiet_window_s(self) -> float | None:
        configured = getattr(self, "max_voice_quiet_window_s", None)
        if configured is None:
            return self._DEFAULT_MAX_VOICE_QUIET_WINDOW_S
        value = self._coerce_positive_float(configured)
        return value

    def _max_follow_up_timeout_override_s(self) -> float:
        configured = self._coerce_positive_float(getattr(self, "max_follow_up_timeout_s", None))
        if configured is not None:
            return configured
        return max(30.0, float(self.follow_up_timeout_s) + max(0.0, float(self.intent_follow_up_timeout_bonus_s)))

    def _max_voice_identity_profiles(self) -> int:
        configured = self._coerce_non_negative_int(getattr(self, "max_voice_identity_profiles", None))
        if configured is not None:
            return configured
        return self._DEFAULT_MAX_VOICE_IDENTITY_PROFILES

    def _max_voice_identity_embedding_values(self) -> int:
        configured = self._coerce_non_negative_int(
            getattr(self, "max_voice_identity_embedding_values", None)
        )
        if configured is not None:
            return configured
        return self._DEFAULT_MAX_VOICE_IDENTITY_EMBEDDING_VALUES

    def _coerce_positive_int(self, value: object | None) -> int | None:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        if parsed <= 0:
            return None
        return parsed

    def _coerce_non_negative_int(self, value: object | None) -> int | None:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        if parsed < 0:
            return None
        return parsed

    def _coerce_positive_float(self, value: object | None) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if parsed <= 0:
            return None
        return parsed