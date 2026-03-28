# CHANGELOG: 2026-03-27
# BUG-1: Non-empty prompt fallback added so empty/missing social prompts no longer fail delivery.
# BUG-2: _BackgroundDeliveryBlocked is now handled as a normal skip for speech delivery too, not as an error.
# BUG-3: LLM phrasing is now latency-bounded so a hung phrasing call cannot stall the realtime background loop indefinitely.
# SEC-1: Telemetry/log payloads are now neutralized and truncated to reduce log injection, UI injection, and sensitive-data leakage risk.
# IMP-1: Non-safety triggers can now expire when stale, preventing outdated proactive nudges from being delivered late.
# IMP-2: Spoken prompts are now duration-budgeted and trimmed for low-latency, interruption-tolerant delivery on constrained devices.
"""Proactive social-trigger delivery helpers for the realtime background loop."""

# mypy: ignore-errors

from __future__ import annotations

from contextvars import copy_context
from datetime import datetime, timedelta, timezone
import math
import queue
import re
import threading
import time
import unicodedata

from twinr.agent.workflows.realtime_runtime.background_delivery import (
    BackgroundDeliveryBlocked as _BackgroundDeliveryBlocked,
)
from twinr.agent.workflows.realtime_runtime.proactive_delivery import (
    ProactiveDeliveryDecision,
)
from twinr.proactive.governance.governor import ProactiveGovernorCandidate
from twinr.proactive.social.engine import SocialTriggerDecision
from twinr.proactive.social.prompting import (
    is_safety_trigger,
    proactive_observation_facts,
    proactive_prompt_mode,
)

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class BackgroundSocialMixin:
    """Handle proactive social-trigger phrasing and delivery."""

    _SOCIAL_PROMPT_TELEMETRY_MAX_CHARS = 160
    _SOCIAL_REASON_TELEMETRY_MAX_CHARS = 160
    _SOCIAL_ERROR_TELEMETRY_MAX_CHARS = 240
    _SOCIAL_GOVERNOR_SUMMARY_MAX_CHARS = 180
    _SOCIAL_PROMPT_INPUT_MAX_CHARS = 600
    _SOCIAL_DISPLAY_PROMPT_MAX_CHARS = 180
    _SOCIAL_MIN_SPEECH_WORDS = 8
    _SOCIAL_BASE_SPEECH_SECONDS = 8.0
    _SOCIAL_HIGH_PRIORITY_SPEECH_SECONDS = 10.0
    _SOCIAL_SAFETY_SPEECH_SECONDS = 12.0
    _SOCIAL_LOW_PRIORITY_SPEECH_SECONDS = 6.0
    _SOCIAL_ESTIMATED_WORDS_PER_SECOND = 2.6
    _SOCIAL_TRIGGER_STALE_AFTER_S = 30.0
    _SOCIAL_PHRASE_TIMEOUT_S = 2.5
    _SOCIAL_MAX_FUTURE_SKEW_S = 300.0
    _SOCIAL_MIN_EPOCH_S = 946684800.0

    def handle_social_trigger(self, trigger: SocialTriggerDecision) -> bool:
        trigger_id = self._normalize_social_trigger_id(getattr(trigger, "trigger_id", None)) or "unknown"
        priority = self._coerce_priority(getattr(trigger, "priority", None), default=50)
        default_prompt = self._clean_social_user_text(
            getattr(trigger, "prompt", None),
            max_chars=self._SOCIAL_PROMPT_INPUT_MAX_CHARS,
        )
        trigger_reason = self._clean_social_user_text(
            getattr(trigger, "reason", None),
            max_chars=self._SOCIAL_PROMPT_INPUT_MAX_CHARS,
        )
        safety_trigger = is_safety_trigger(trigger_id)

        trigger_age_s = self._social_trigger_age_seconds(trigger)
        # BREAKING: non-safety triggers older than the configured TTL are now dropped
        # instead of being delivered late, because stale proactive nudges degrade trust
        # and can become contextually wrong in mixed-initiative systems.
        if trigger_age_s is not None and not safety_trigger and trigger_age_s > self._social_trigger_stale_after_s():
            self._safe_emit("social_trigger_skipped=stale")
            self._safe_emit(f"social_trigger_age_s={trigger_age_s:.3f}")
            self._safe_record_event(
                "social_trigger_skipped",
                "Social trigger prompt was skipped because the trigger was already stale.",
                trigger=trigger_id,
                reason=self._social_telemetry_text(
                    trigger_reason,
                    max_chars=self._SOCIAL_REASON_TELEMETRY_MAX_CHARS,
                ),
                prompt=self._social_telemetry_text(default_prompt),
                priority=priority,
                skip_reason="stale",
                trigger_age_s=round(trigger_age_s, 3),
            )
            return False

        delivery_decision = self._decide_social_trigger_delivery(
            trigger_id=trigger_id,
            safety_trigger=safety_trigger,
        )
        delivery_channel = self._normalize_social_delivery_channel(
            getattr(delivery_decision, "channel", None)
        )
        governor_inputs = self._current_governor_inputs(requested_channel=delivery_channel)

        initial_prompt_text, _, _ = self._prepare_social_prompt(
            prompt_text=default_prompt,
            channel=delivery_channel,
            trigger_id=trigger_id,
            trigger_reason=trigger_reason,
            priority=priority,
            safety_trigger=safety_trigger,
        )

        if not self._background_work_allowed():
            skip_reason = (
                "busy"
                if getattr(getattr(self.runtime, "status", None), "value", None) != "waiting"
                else "conversation_active"
            )
            self._safe_emit(f"social_trigger_skipped={skip_reason}")
            self._safe_record_event(
                "social_trigger_skipped",
                "Social trigger prompt was skipped because Twinr was not idle for background prompts.",
                trigger=trigger_id,
                reason=self._social_telemetry_text(
                    trigger_reason,
                    max_chars=self._SOCIAL_REASON_TELEMETRY_MAX_CHARS,
                ),
                prompt=self._social_telemetry_text(initial_prompt_text),
                priority=priority,
                skip_reason=skip_reason,
            )
            return False

        governor_reservation = self._reserve_governed_prompt(
            ProactiveGovernorCandidate(
                source_kind="social",
                source_id=trigger_id,
                summary=self._social_governor_summary(initial_prompt_text),
                channel=governor_inputs.channel,
                priority=priority,
                presence_session_id=governor_inputs.presence_session_id,
                safety_exempt=safety_trigger,
                counts_toward_presence_budget=not safety_trigger,
            ),
            governor_inputs=governor_inputs,
        )
        if governor_reservation is None:
            return False

        blocked_reason = self._background_block_reason()
        if blocked_reason is not None:
            self._safe_cancel_governor_reservation(governor_reservation)
            self._safe_emit(f"social_trigger_skipped={blocked_reason}")
            self._safe_record_event(
                "social_trigger_skipped",
                "Social trigger prompt was skipped because Twinr stopped being idle before delivery started.",
                trigger=trigger_id,
                reason=self._social_telemetry_text(
                    trigger_reason,
                    max_chars=self._SOCIAL_REASON_TELEMETRY_MAX_CHARS,
                ),
                prompt=self._social_telemetry_text(initial_prompt_text),
                priority=priority,
                skip_reason=blocked_reason,
            )
            return False

        phrase_response = None
        prompt_mode = proactive_prompt_mode(trigger)
        prompt_text = initial_prompt_text
        prompt_trimmed = False
        prompt_estimate_s = None

        try:
            if delivery_channel == "display":
                try:
                    self._begin_background_delivery(
                        lambda lease: lease.run_locked(
                            lambda: self._show_social_trigger_display_cue(
                                trigger_id=trigger_id,
                                prompt_text=prompt_text,
                                reason=delivery_decision.reason,
                                cue_hold_seconds=delivery_decision.cue_hold_seconds,
                            )
                        )
                    )
                except _BackgroundDeliveryBlocked as blocked:
                    self._safe_cancel_governor_reservation(governor_reservation)
                    self._safe_emit(f"social_trigger_skipped={blocked.reason}")
                    self._safe_record_event(
                        "social_trigger_skipped",
                        "Social trigger display cue was skipped because Twinr stopped being idle before delivery started.",
                        trigger=trigger_id,
                        reason=self._social_telemetry_text(
                            trigger_reason,
                            max_chars=self._SOCIAL_REASON_TELEMETRY_MAX_CHARS,
                        ),
                        prompt=self._social_telemetry_text(prompt_text),
                        priority=priority,
                        skip_reason=blocked.reason,
                    )
                    return False

                self._safe_mark_governor_delivered(governor_reservation)
                self._safe_emit(f"social_trigger={trigger_id}")
                self._safe_emit(f"social_trigger_priority={priority}")
                self._safe_emit("social_prompt_mode=display_first")
                self._safe_emit(f"social_prompt={self._social_telemetry_text(prompt_text)}")
                self._safe_record_event(
                    "social_trigger_displayed",
                    "Twinr kept a proactive social prompt visual-first instead of speaking it aloud.",
                    trigger=trigger_id,
                    reason=self._social_telemetry_text(
                        trigger_reason,
                        max_chars=self._SOCIAL_REASON_TELEMETRY_MAX_CHARS,
                    ),
                    priority=priority,
                    prompt=self._social_telemetry_text(prompt_text),
                    default_prompt=self._social_telemetry_text(default_prompt),
                    display_reason=self._social_telemetry_text(
                        delivery_decision.reason,
                        max_chars=self._SOCIAL_REASON_TELEMETRY_MAX_CHARS,
                    ),
                )
                return True

            # BREAKING: safety triggers now bypass LLM rephrasing so delivery stays
            # deterministic and low-latency even when the phrasing provider stalls.
            if safety_trigger and prompt_mode == "llm":
                prompt_mode = "deterministic_safety"

            if prompt_mode == "llm":
                stop_processing_feedback = self._start_working_feedback_loop("processing")
                try:
                    try:
                        phrase_response = self._phrase_social_prompt_with_timeout(
                            trigger_id=trigger_id,
                            reason=trigger_reason,
                            default_prompt=default_prompt,
                            priority=priority,
                            conversation=self.runtime.conversation_context(),
                            recent_prompts=self._recent_proactive_prompts(trigger_id=trigger_id),
                            observation_facts=proactive_observation_facts(trigger),
                        )
                    finally:
                        stop_processing_feedback()

                    candidate_prompt = self._clean_social_user_text(
                        getattr(phrase_response, "text", None),
                        max_chars=self._SOCIAL_PROMPT_INPUT_MAX_CHARS,
                    )
                    if candidate_prompt:
                        prompt_text = candidate_prompt
                    else:
                        prompt_mode = "default_fallback"
                        self._safe_emit("social_prompt_fallback=empty_phrase")
                except TimeoutError as exc:
                    prompt_mode = "default_fallback_timeout"
                    self._safe_emit("social_prompt_fallback=timeout")
                    self._safe_emit(
                        f"social_prompt_phrase_error={self._social_error_text(exc)}"
                    )
                    self._safe_record_event(
                        "social_trigger_phrase_fallback",
                        "Twinr fell back to the default proactive prompt after proactive phrasing timed out.",
                        level="warning",
                        trigger=trigger_id,
                        error=self._social_error_text(exc),
                    )
                except Exception as exc:
                    prompt_mode = "default_fallback"
                    self._safe_emit("social_prompt_fallback=default")
                    self._safe_emit(
                        f"social_prompt_phrase_error={self._social_error_text(exc)}"
                    )
                    self._safe_record_event(
                        "social_trigger_phrase_fallback",
                        "Twinr fell back to the default proactive prompt after proactive phrasing failed.",
                        level="warning",
                        trigger=trigger_id,
                        error=self._social_error_text(exc),
                    )

            prompt_text, prompt_trimmed, prompt_estimate_s = self._prepare_social_prompt(
                prompt_text=prompt_text,
                channel="speech",
                trigger_id=trigger_id,
                trigger_reason=trigger_reason,
                priority=priority,
                safety_trigger=safety_trigger,
            )

            if prompt_estimate_s is not None:
                self._safe_emit(f"social_prompt_estimate_s={prompt_estimate_s:.3f}")
                self._safe_emit(
                    f"social_prompt_budget_s={self._social_speech_budget_seconds(priority, safety_trigger):.3f}"
                )
            if prompt_trimmed:
                self._safe_emit("social_prompt_trimmed=1")

            try:
                prompt = self._begin_background_delivery(
                    lambda lease: lease.run_locked(
                        lambda: self.runtime.begin_proactive_prompt(
                            self._require_non_empty_text(
                                prompt_text,
                                context=f"social trigger {trigger_id} prompt",
                            )
                        )
                    )
                )
            except _BackgroundDeliveryBlocked as blocked:
                self._safe_cancel_governor_reservation(governor_reservation)
                self._safe_emit(f"social_trigger_skipped={blocked.reason}")
                self._safe_record_event(
                    "social_trigger_skipped",
                    "Social trigger speech prompt was skipped because Twinr stopped being idle before delivery started.",
                    trigger=trigger_id,
                    reason=self._social_telemetry_text(
                        trigger_reason,
                        max_chars=self._SOCIAL_REASON_TELEMETRY_MAX_CHARS,
                    ),
                    prompt=self._social_telemetry_text(prompt_text),
                    priority=priority,
                    skip_reason=blocked.reason,
                )
                return False

            self._safe_emit_status(force=True)
            tts_started = time.monotonic()
            tts_ms, first_audio_ms = self._play_streaming_tts_with_feedback(prompt, turn_started=tts_started)
            self._finalize_speaking_output()
            self._safe_mark_governor_delivered(governor_reservation)
            self._safe_emit(f"social_trigger={trigger_id}")
            self._safe_emit(f"social_trigger_priority={priority}")
            self._safe_emit(f"social_prompt_mode={prompt_mode}")
            self._safe_emit(
                f"social_prompt={self._social_telemetry_text(self._coerce_text(prompt) or prompt_text)}"
            )
            if phrase_response is not None:
                if getattr(phrase_response, "response_id", None):
                    self._safe_emit(
                        f"social_response_id={self._social_telemetry_text(phrase_response.response_id, max_chars=96)}"
                    )
                if getattr(phrase_response, "request_id", None):
                    self._safe_emit(
                        f"social_request_id={self._social_telemetry_text(phrase_response.request_id, max_chars=96)}"
                    )
                self._safe_record_usage(
                    request_kind="proactive_prompt",
                    source="realtime_loop",
                    model=getattr(phrase_response, "model", "unknown"),
                    response_id=getattr(phrase_response, "response_id", None),
                    request_id=getattr(phrase_response, "request_id", None),
                    used_web_search=False,
                    token_usage=getattr(phrase_response, "token_usage", None),
                    proactive_trigger=trigger_id,
                )
            self._safe_emit(f"timing_social_tts_ms={tts_ms}")
            if first_audio_ms is not None:
                self._safe_emit(f"timing_social_first_audio_ms={first_audio_ms}")
            self._safe_record_event(
                "social_trigger_prompted",
                "Twinr spoke a proactive social prompt.",
                trigger=trigger_id,
                reason=self._social_telemetry_text(
                    trigger_reason,
                    max_chars=self._SOCIAL_REASON_TELEMETRY_MAX_CHARS,
                ),
                priority=priority,
                prompt=self._social_telemetry_text(self._coerce_text(prompt) or prompt_text),
                default_prompt=self._social_telemetry_text(default_prompt),
                prompt_mode=prompt_mode,
                prompt_trimmed=prompt_trimmed,
                prompt_estimate_s=round(prompt_estimate_s, 3) if prompt_estimate_s is not None else None,
            )
            follow_up_engaged = self._safe_run_proactive_follow_up(trigger)
            latest_audio_policy = self._current_audio_policy_snapshot()
            if latest_audio_policy is not None and latest_audio_policy.barge_in_recent is True:
                self._proactive_delivery_policy().note_interrupted(
                    source_id=trigger_id,
                    monotonic_now=self._social_monotonic_now(),
                )
                self._safe_record_event(
                    "social_trigger_visual_first_cooldown_started",
                    "Twinr switched future proactive prompts to visual-first after an interrupted prompt.",
                    trigger=trigger_id,
                    reason=self._social_telemetry_text(
                        trigger_reason,
                        max_chars=self._SOCIAL_REASON_TELEMETRY_MAX_CHARS,
                    ),
                    cooldown_reason="interrupted",
                )
            elif not follow_up_engaged:
                self._proactive_delivery_policy().note_ignored(
                    source_id=trigger_id,
                    monotonic_now=self._social_monotonic_now(),
                )
                self._safe_record_event(
                    "social_trigger_visual_first_cooldown_started",
                    "Twinr switched future proactive prompts to visual-first after a prompt was ignored.",
                    trigger=trigger_id,
                    reason=self._social_telemetry_text(
                        trigger_reason,
                        max_chars=self._SOCIAL_REASON_TELEMETRY_MAX_CHARS,
                    ),
                    cooldown_reason="ignored",
                )
            return True
        except Exception as exc:
            self._recover_speaking_output_state()
            self._safe_mark_governor_skipped(
                governor_reservation,
                reason=f"delivery_failed: {self._social_error_text(exc)}",
            )
            self._safe_emit(f"social_trigger_error={self._social_error_text(exc)}")
            self._safe_record_event(
                "social_trigger_failed",
                "A proactive social trigger failed during delivery.",
                level="error",
                trigger=trigger_id,
                reason=self._social_telemetry_text(
                    trigger_reason,
                    max_chars=self._SOCIAL_REASON_TELEMETRY_MAX_CHARS,
                ),
                priority=priority,
                error=self._social_error_text(exc),
            )
            return False

    def _decide_social_trigger_delivery(
        self,
        *,
        trigger_id: str,
        safety_trigger: bool,
    ) -> ProactiveDeliveryDecision:
        """Choose whether one social trigger should speak or stay visual."""

        decision = self._proactive_delivery_policy().decide(
            monotonic_now=self._social_monotonic_now(),
            local_now=self._local_now(),
            source_id=trigger_id,
            safety_exempt=safety_trigger,
            audio_policy_snapshot=self._current_audio_policy_snapshot(),
        )
        normalized_channel = self._normalize_social_delivery_channel(
            getattr(decision, "channel", None)
        )
        raw_hold_s = getattr(decision, "cue_hold_seconds", None)
        default_hold_s = (
            self._proactive_delivery_policy().visual_first_cue_hold_s
            if normalized_channel == "display"
            else None
        )
        normalized_hold_s = self._coerce_non_negative_float(
            raw_hold_s,
            default=default_hold_s,
        )
        if (
            normalized_channel != getattr(decision, "channel", None)
            or normalized_hold_s != raw_hold_s
        ):
            decision = ProactiveDeliveryDecision(
                channel=normalized_channel,
                reason=self._clean_social_user_text(getattr(decision, "reason", None)),
                cue_hold_seconds=normalized_hold_s,
            )

        if safety_trigger:
            return decision

        initiative_snapshot = self._current_multimodal_initiative_snapshot()
        if (
            initiative_snapshot is not None
            and not initiative_snapshot.ready
            and initiative_snapshot.recommended_channel == "display"
            and decision.channel == "speech"
        ):
            return ProactiveDeliveryDecision(
                channel="display",
                reason=self._clean_social_user_text(getattr(initiative_snapshot, "block_reason", None))
                or "low_multimodal_initiative_confidence",
                cue_hold_seconds=self._proactive_delivery_policy().visual_first_cue_hold_s,
            )
        return decision

    def _show_social_trigger_display_cue(
        self,
        *,
        trigger_id: str,
        prompt_text: str,
        reason: str | None,
        cue_hold_seconds: float | None,
    ) -> None:
        """Render one visual-first proactive cue on the display layer."""

        publisher = self._display_social_reserve_publisher()
        publisher.publish(
            trigger_id=trigger_id,
            prompt_text=self._require_non_empty_text(
                self._clean_social_user_text(
                    prompt_text,
                    max_chars=self._SOCIAL_DISPLAY_PROMPT_MAX_CHARS,
                )
                or self._fallback_social_prompt_text(
                    trigger_id=trigger_id,
                    trigger_reason=None,
                    channel="display",
                    safety_trigger=is_safety_trigger(trigger_id),
                ),
                context=f"social trigger {trigger_id} display prompt",
            ),
            display_reason=self._social_telemetry_text(
                reason,
                max_chars=self._SOCIAL_REASON_TELEMETRY_MAX_CHARS,
            ),
            hold_seconds=self._coerce_non_negative_float(cue_hold_seconds, default=None),
            now=self._social_utc_now(),
        )
        self._safe_emit(
            f"social_trigger_display={self._social_telemetry_text(trigger_id, max_chars=96) or 'unknown'}"
        )
        if reason:
            self._safe_emit(
                f"social_display_reason={self._social_telemetry_text(reason, max_chars=self._SOCIAL_REASON_TELEMETRY_MAX_CHARS)}"
            )

    def _phrase_social_prompt_with_timeout(self, **kwargs):
        """Bound proactive social phrasing latency so the background loop cannot hang indefinitely."""

        timeout_s = self._social_phrase_timeout_s()
        active_worker = getattr(self, "_social_phrase_worker", None)
        if active_worker is not None and active_worker.is_alive():
            raise TimeoutError("previous social prompt phrasing call is still running")

        result_queue = queue.Queue(maxsize=1)
        caller_context = copy_context()

        def run_phrase_call() -> None:
            try:
                response = caller_context.run(
                    self.agent_provider.phrase_proactive_prompt_with_metadata,
                    **kwargs,
                )
            except BaseException as exc:
                result_queue.put(("error", exc))
                return
            result_queue.put(("ok", response))

        # BREAKING: social prompt phrasing is now performed behind a strict latency budget.
        # A timed-out phrasing call falls back to deterministic prompt text instead of blocking
        # the realtime background loop until the model/provider eventually returns.
        worker = threading.Thread(
            target=run_phrase_call,
            name="twinr-social-phrase",
            daemon=True,
        )
        self._social_phrase_worker = worker
        worker.start()
        worker.join(timeout=timeout_s)

        if worker.is_alive():
            raise TimeoutError(f"social prompt phrasing exceeded {timeout_s:.2f}s")

        self._social_phrase_worker = None

        try:
            result_kind, result_payload = result_queue.get_nowait()
        except queue.Empty as exc:
            raise RuntimeError("social prompt phrasing finished without returning a result") from exc

        if result_kind == "error":
            raise result_payload
        return result_payload

    def _prepare_social_prompt(
        self,
        *,
        prompt_text: str | None,
        channel: str,
        trigger_id: str,
        trigger_reason: str | None,
        priority: int,
        safety_trigger: bool,
    ) -> tuple[str, bool, float | None]:
        """Normalize, validate, and budget prompt text for the selected delivery channel."""

        channel = self._normalize_social_delivery_channel(channel)
        cleaned = self._clean_social_user_text(
            prompt_text,
            max_chars=self._SOCIAL_PROMPT_INPUT_MAX_CHARS,
        )
        if not cleaned:
            cleaned = self._fallback_social_prompt_text(
                trigger_id=trigger_id,
                trigger_reason=trigger_reason,
                channel=channel,
                safety_trigger=safety_trigger,
            )

        if channel == "display":
            fitted = self._trim_social_display_text(cleaned)
            return fitted, fitted != cleaned, None

        fitted = self._trim_social_speech_text(
            cleaned,
            max_seconds=self._social_speech_budget_seconds(priority, safety_trigger),
        )
        return fitted, fitted != cleaned, self._estimate_social_speech_seconds(fitted)

    def _trim_social_display_text(self, text: str) -> str:
        return self._trim_text_at_boundary(
            text,
            max_chars=self._SOCIAL_DISPLAY_PROMPT_MAX_CHARS,
            append_ellipsis=True,
        )

    def _trim_social_speech_text(self, text: str, *, max_seconds: float) -> str:
        estimated_s = self._estimate_social_speech_seconds(text)
        if estimated_s <= max_seconds:
            return self._ensure_terminal_punctuation(text)

        max_words = max(
            self._SOCIAL_MIN_SPEECH_WORDS,
            int(math.floor(max_seconds * self._SOCIAL_ESTIMATED_WORDS_PER_SECOND)),
        )

        sentence_parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]
        if sentence_parts:
            kept_parts = []
            total_words = 0
            for part in sentence_parts:
                part_words = len(part.split())
                if kept_parts and total_words + part_words > max_words:
                    break
                if not kept_parts and part_words > max_words:
                    break
                kept_parts.append(part)
                total_words += part_words
            if kept_parts:
                return self._ensure_terminal_punctuation(" ".join(kept_parts))

        words = text.split()
        if len(words) <= max_words:
            return self._ensure_terminal_punctuation(text)

        trimmed = " ".join(words[:max_words]).rstrip(" ,;:-")
        return self._ensure_terminal_punctuation(trimmed)

    def _estimate_social_speech_seconds(self, text: str) -> float:
        words = len(text.split())
        if words <= 0:
            return 0.0
        return words / self._SOCIAL_ESTIMATED_WORDS_PER_SECOND

    def _social_speech_budget_seconds(self, priority: int, safety_trigger: bool) -> float:
        if safety_trigger:
            return self._SOCIAL_SAFETY_SPEECH_SECONDS
        if priority >= 80:
            return self._SOCIAL_HIGH_PRIORITY_SPEECH_SECONDS
        if priority <= 25:
            return self._SOCIAL_LOW_PRIORITY_SPEECH_SECONDS
        return self._SOCIAL_BASE_SPEECH_SECONDS

    def _fallback_social_prompt_text(
        self,
        *,
        trigger_id: str,
        trigger_reason: str | None,
        channel: str,
        safety_trigger: bool,
    ) -> str:
        readable_reason = self._social_reason_for_user(trigger_reason)
        if readable_reason:
            if safety_trigger:
                return self._ensure_terminal_punctuation(f"Please pay attention. {readable_reason}")
            if channel == "display":
                return self._ensure_terminal_punctuation(readable_reason)
            return self._ensure_terminal_punctuation(f"I have a quick thought: {readable_reason}")

        if safety_trigger:
            return "Please pay attention to this important update."
        if channel == "display":
            return "A quick suggestion is ready."
        return "I have a quick suggestion for you."

    def _social_reason_for_user(self, reason: str | None) -> str | None:
        cleaned = self._clean_social_user_text(reason, max_chars=140)
        if not cleaned:
            return None
        lowered = cleaned.lower()
        if "_" in cleaned or lowered.startswith("low_") or lowered.endswith("_confidence"):
            return None
        if cleaned.count(" ") == 0 and "-" not in cleaned:
            return None
        return cleaned

    def _social_governor_summary(self, text: str | None) -> str:
        summary = self._social_telemetry_text(
            text,
            max_chars=self._SOCIAL_GOVERNOR_SUMMARY_MAX_CHARS,
        )
        return summary or "social-trigger"

    def _social_trigger_stale_after_s(self) -> float:
        value = getattr(self, "social_trigger_stale_after_s", None)
        return self._coerce_non_negative_float(value, default=self._SOCIAL_TRIGGER_STALE_AFTER_S)

    def _social_phrase_timeout_s(self) -> float:
        value = getattr(self, "social_phrase_timeout_s", None)
        return self._coerce_non_negative_float(value, default=self._SOCIAL_PHRASE_TIMEOUT_S)

    def _social_max_future_skew_s(self) -> float:
        value = getattr(self, "social_trigger_max_future_skew_s", None)
        return self._coerce_non_negative_float(value, default=self._SOCIAL_MAX_FUTURE_SKEW_S)

    def _social_monotonic_now(self) -> float:
        return time.monotonic()

    def _social_utc_now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _social_trigger_age_seconds(self, trigger: SocialTriggerDecision) -> float | None:
        trigger_ts = self._extract_social_trigger_timestamp(trigger)
        if trigger_ts is None:
            return None
        now_utc = self._social_utc_now()
        delta_s = (now_utc - trigger_ts).total_seconds()
        return max(0.0, delta_s)

    def _extract_social_trigger_timestamp(
        self,
        trigger: SocialTriggerDecision,
    ) -> datetime | None:
        for field_name in (
            "created_at",
            "triggered_at",
            "observed_at",
            "occurred_at",
            "timestamp",
            "ts",
            "decided_at",
            "generated_at",
        ):
            value = getattr(trigger, field_name, None)
            dt_value = self._coerce_datetime_utc(value)
            if dt_value is not None:
                return dt_value
        return None

    def _coerce_datetime_utc(self, value) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        if isinstance(value, (int, float)):
            numeric = self._coerce_non_negative_float(value, default=None)
            if numeric is None:
                return None
            return self._coerce_numeric_datetime_utc(numeric)
        text = self._coerce_text(value)
        if not text:
            return None
        try:
            numeric = self._coerce_non_negative_float(text, default=None)
        except (OverflowError, OSError, ValueError):
            numeric = None
        if numeric is not None:
            return self._coerce_numeric_datetime_utc(numeric)
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None

    def _coerce_numeric_datetime_utc(self, numeric: float) -> datetime | None:
        now_utc = self._social_utc_now()
        future_skew = self._social_max_future_skew_s()
        monotonic_now = self._social_monotonic_now()
        if 0.0 <= numeric <= monotonic_now + future_skew and abs(monotonic_now - numeric) <= future_skew:
            delta_s = max(-monotonic_now, numeric - monotonic_now)
            return now_utc + timedelta(seconds=delta_s)

        if self._SOCIAL_MIN_EPOCH_S <= numeric <= now_utc.timestamp() + future_skew:
            try:
                return datetime.fromtimestamp(numeric, tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                return None
        for scale in (1_000.0, 1_000_000.0, 1_000_000_000.0):
            scaled = numeric / scale
            if self._SOCIAL_MIN_EPOCH_S <= scaled <= now_utc.timestamp() + future_skew:
                try:
                    return datetime.fromtimestamp(scaled, tz=timezone.utc)
                except (OverflowError, OSError, ValueError):
                    return None
        return None

    def _normalize_social_delivery_channel(self, channel: str | None) -> str:
        text = self._coerce_text(channel)
        if text == "display":
            return "display"
        return "speech"

    def _normalize_social_trigger_id(self, value) -> str | None:
        cleaned = self._clean_social_user_text(value, max_chars=96)
        if not cleaned:
            return None
        return cleaned.replace(" ", "_")

    # BREAKING: emitted and recorded social prompt/reason/error fields are now
    # sanitized and truncated before they reach telemetry sinks. This prevents
    # log forging/control-character injection and reduces accidental leakage of
    # sensitive social/health context through operational logs.
    def _social_telemetry_text(self, value, *, max_chars: int | None = None) -> str | None:
        return self._clean_social_user_text(
            value,
            max_chars=max_chars or self._SOCIAL_PROMPT_TELEMETRY_MAX_CHARS,
        )

    def _social_error_text(self, value) -> str:
        return (
            self._clean_social_user_text(
                value,
                max_chars=self._SOCIAL_ERROR_TELEMETRY_MAX_CHARS,
            )
            or "unknown_error"
        )

    def _clean_social_user_text(self, value, *, max_chars: int | None = None) -> str | None:
        text = self._coerce_text(value)
        if text is None:
            return None

        text = _ANSI_ESCAPE_RE.sub("", text)
        normalized_chars = []
        for char in text:
            if char in ("\r", "\n", "\t"):
                normalized_chars.append(" ")
                continue
            category = unicodedata.category(char)
            if category in {"Cc", "Cf", "Cs"}:
                continue
            normalized_chars.append(char)

        cleaned = "".join(normalized_chars)
        cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
        if not cleaned:
            return None

        if max_chars is not None and len(cleaned) > max_chars:
            cleaned = self._trim_text_at_boundary(
                cleaned,
                max_chars=max_chars,
                append_ellipsis=True,
            )
        return cleaned

    def _trim_text_at_boundary(
        self,
        text: str,
        *,
        max_chars: int,
        append_ellipsis: bool,
    ) -> str:
        if len(text) <= max_chars:
            return text
        if max_chars <= 1:
            return "…" if append_ellipsis and max_chars == 1 else text[:max_chars]

        slice_len = max_chars - 1 if append_ellipsis else max_chars
        candidate = text[:slice_len].rstrip()
        last_boundary = max(
            candidate.rfind(". "),
            candidate.rfind("! "),
            candidate.rfind("? "),
            candidate.rfind("; "),
            candidate.rfind(", "),
            candidate.rfind(" "),
        )
        if last_boundary >= max(12, slice_len // 2):
            candidate = candidate[:last_boundary].rstrip(" ,;:-")
        if append_ellipsis:
            return f"{candidate}…"
        return candidate

    def _ensure_terminal_punctuation(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return cleaned
        if cleaned[-1] in ".!?":
            return cleaned
        return f"{cleaned}."

    def _coerce_non_negative_float(self, value, *, default):
        if value is None:
            return default
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        if math.isnan(number) or math.isinf(number) or number < 0:
            return default
        return number
