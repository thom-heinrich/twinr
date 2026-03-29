# mypy: disable-error-code="attr-defined,has-type,assignment"
# CHANGELOG: 2026-03-29
# BUG-1: Recompute pending utterance timing after buffer trims so `active_ms`,
#        `trailing_silence_ms`, and `speech_active` never drift away from the
#        actual retained frame window.
# BUG-2: Compute capture `active_ratio` by retained speech duration rather than
#        frame count so jittery/variable websocket chunking cannot bias VAD
#        gating decisions.
# BUG-3: Guard direct barge-in transcription against transient backend errors so
#        a network/provider blip does not tear down the voice loop.
# BUG-4: Ignore zero-byte PCM ingress instead of fabricating a full silent chunk;
#        treating empty frames as real time could force false endpointing.
# SEC-1: Bound oversized PCM ingress and retain only the newest aligned tail so
#        malformed or malicious websocket frames cannot force large allocations
#        or oversized remote uploads on a Raspberry Pi.
# SEC-2: Redact transcript previews in generic trace events by default to avoid
#        leaking sensitive user speech into routine logs; raw previews remain
#        opt-in via `voice_trace_include_transcripts=True`.
# IMP-1: Add optional neural-VAD speech-probability support (e.g. Silero/openWakeWord
#        side-channel scores) while preserving RMS fallback for drop-in compatibility.
# IMP-2: Add duplicate-window suppression for remote stage-1 and barge-in scans to
#        avoid rescanning identical audio and wasting latency, CPU, and API budget.
# IMP-3: Preserve a small true pre-roll before the latest active burst, including
#        zero-RMS frames, to reduce clipped wake-word/utterance onsets.
# IMP-4: Add an optional semantic endpoint hook so newer 2026-style turn-state
#        predictors can override fixed trailing-silence endpointing without
#        changing this mixin's public API.
"""Frame buffering, utterance assembly, and candidate scanning helpers."""

from __future__ import annotations

import hashlib

from twinr.hardware.audio import AmbientAudioCaptureWindow, AmbientAudioLevelSample
from twinr.orchestrator.voice_activation import VoiceActivationMatch

from ..voice_contracts import (
    OrchestratorVoiceBargeInInterruptEvent,
    OrchestratorVoiceTranscriptCommittedEvent,
    OrchestratorVoiceWakeConfirmedEvent,
)
from .builders import pcm_capture_to_wav_bytes
from .types import (
    _PendingTranscriptUtterance,
    _RecentFrame,
    _normalize_text_length,
    _pcm16_rms,
)


class VoiceSessionScannerMixin:
    """Own bounded frame history, utterance buffering, and candidate scans."""

    def _remote_asr_speech_continue_threshold(self) -> int:
        """Return the bounded lower threshold for one already-open speech burst."""

        return max(
            1,
            min(
                self.speech_threshold,
                int(round(self.speech_threshold * self._REMOTE_ASR_SPEECH_CONTINUE_RATIO)),
            ),
        )

    def _remote_asr_speech_probability_threshold(self) -> float | None:
        """Return an optional neural-VAD threshold if the runtime exposes one."""

        raw_threshold = getattr(self, "remote_asr_speech_probability_threshold", None)
        if raw_threshold is None:
            raw_threshold = getattr(
                getattr(self, "config", None),
                "remote_asr_speech_probability_threshold",
                None,
            )
        if raw_threshold is None:
            return None
        try:
            threshold = float(raw_threshold)
        except (TypeError, ValueError):
            return None
        return min(1.0, max(0.0, threshold))

    def _remote_asr_speech_continue_probability_threshold(self) -> float | None:
        """Return the lower neural-VAD threshold for already-open speech bursts."""

        threshold = self._remote_asr_speech_probability_threshold()
        if threshold is None:
            return None
        return min(
            threshold,
            max(0.0, threshold * float(self._REMOTE_ASR_SPEECH_CONTINUE_RATIO)),
        )

    def _frame_speech_probability(self, frame: _RecentFrame) -> float | None:
        """Resolve an optional per-frame speech probability from runtime hooks."""

        raw_probability = None
        for attribute in (
            "speech_probability",
            "vad_probability",
            "speech_prob",
            "vad_score",
        ):
            raw_probability = getattr(frame, attribute, None)
            if raw_probability is not None:
                break
        if raw_probability is None:
            resolver = getattr(self, "_resolve_frame_speech_probability", None)
            if callable(resolver):
                try:
                    raw_probability = resolver(frame)  # pylint: disable=not-callable
                except Exception:
                    raw_probability = None
        if raw_probability is None:
            return None
        try:
            probability_input = (
                raw_probability
                if isinstance(raw_probability, (bool, int, float, str, bytes, bytearray))
                else str(raw_probability)
            )
            probability = float(probability_input)
        except (TypeError, ValueError):
            return None
        return min(1.0, max(0.0, probability))

    def _frame_counts_as_remote_asr_speech(
        self,
        frame: _RecentFrame,
        *,
        continuing: bool = False,
    ) -> bool:
        """Treat speech bursts with bounded hysteresis for wake-duration accounting."""

        speech_probability = self._frame_speech_probability(frame)
        probability_threshold = self._remote_asr_speech_probability_threshold()
        if speech_probability is not None and probability_threshold is not None:
            if speech_probability >= probability_threshold:
                return True
            continue_threshold = self._remote_asr_speech_continue_probability_threshold()
            return continuing and continue_threshold is not None and speech_probability >= continue_threshold
        if frame.rms >= self.speech_threshold:
            return True
        return continuing and frame.rms >= self._remote_asr_speech_continue_threshold()

    def _remote_asr_speech_flags(
        self,
        frames: tuple[_RecentFrame, ...],
    ) -> tuple[bool, ...]:
        """Return one forward hysteresis speech mask for the provided frames."""

        flags: list[bool] = []
        continuing = False
        for frame in frames:
            continuing = self._frame_counts_as_remote_asr_speech(
                frame,
                continuing=continuing,
            )
            flags.append(continuing)
        return tuple(flags)

    def _pending_utterance_details(
        self,
        pending: _PendingTranscriptUtterance,
    ) -> dict[str, int]:
        """Expose compact buffered-utterance metrics for transcript debug traces."""

        return {
            "pending_captured_ms": int(pending.captured_ms),
            "pending_active_ms": int(pending.active_ms),
            "pending_trailing_silence_ms": int(pending.trailing_silence_ms),
        }

    def _bytes_per_pcm_sample(self) -> int:
        """Return the PCM alignment size for one interleaved sample across channels."""

        return max(2, int(self.channels) * 2)

    def _expected_chunk_pcm_bytes(self) -> int:
        """Return the nominal byte size for one configured audio chunk."""

        bytes_per_second = max(1, int(self.sample_rate) * self._bytes_per_pcm_sample())
        expected = int(round((bytes_per_second * max(1, int(self.chunk_ms))) / 1000.0))
        alignment = self._bytes_per_pcm_sample()
        expected -= expected % alignment
        return max(alignment, expected)

    def _max_ingress_pcm_bytes(self) -> int:
        """Return the maximum PCM payload accepted from a single ingress frame."""

        configured_limit = getattr(self, "max_ingress_pcm_bytes", None)
        if configured_limit is None:
            configured_limit = getattr(
                getattr(self, "config", None),
                "max_ingress_pcm_bytes",
                None,
            )
        if configured_limit is not None:
            try:
                limit = int(configured_limit)
            except (TypeError, ValueError):
                limit = 0
            if limit > 0:
                alignment = self._bytes_per_pcm_sample()
                limit -= limit % alignment
                return max(alignment, limit)
        return max(
            self._expected_chunk_pcm_bytes(),
            self._expected_chunk_pcm_bytes() * 32,
        )

    def _bounded_ingress_pcm_fragments(self, pcm_bytes: bytes) -> tuple[bytes, ...]:
        """Normalize one ingress payload into aligned, bounded PCM chunks."""

        if not pcm_bytes:
            return ()
        alignment = self._bytes_per_pcm_sample()
        normalized = pcm_bytes
        max_ingress_bytes = self._max_ingress_pcm_bytes()
        if len(normalized) > max_ingress_bytes:
            # BREAKING: Oversized ingress payloads are truncated to the newest
            # aligned tail instead of being buffered whole. This prevents
            # oversized websocket/audio frames from causing memory spikes or
            # oversized uploads on constrained devices.
            retained = max_ingress_bytes - (max_ingress_bytes % alignment)
            normalized = normalized[-retained:]
            self._trace_event(
                "voice_frame_oversize_truncated",
                kind="security",
                details={
                    "received_bytes": len(pcm_bytes),
                    "retained_bytes": len(normalized),
                    "max_ingress_pcm_bytes": max_ingress_bytes,
                },
            )
        retained_bytes = len(normalized) - (len(normalized) % alignment)
        if retained_bytes <= 0:
            return ()
        normalized = normalized[-retained_bytes:]
        expected_chunk_bytes = self._expected_chunk_pcm_bytes()
        if len(normalized) <= expected_chunk_bytes:
            return (normalized,)
        return tuple(
            normalized[index : index + expected_chunk_bytes]
            for index in range(0, len(normalized), expected_chunk_bytes)
            if normalized[index : index + expected_chunk_bytes]
        )

    def _remember_frame(self, pcm_bytes: bytes) -> None:
        """Append one PCM chunk into the bounded recent-history buffer."""

        for fragment in self._bounded_ingress_pcm_fragments(pcm_bytes):
            duration_ms = int(
                round(
                    (len(fragment) / max(1, self.channels * 2 * self.sample_rate)) * 1000.0,
                ),
            )
            self._history.append(
                _RecentFrame(
                    pcm_bytes=fragment,
                    rms=_pcm16_rms(fragment),
                    duration_ms=max(1, duration_ms),
                )
            )

    def _flush_received_frame_bucket(self) -> None:
        """Persist one bounded summary of recently received websocket frames."""

        if not self._received_frame_bucket.has_data():
            return
        self._trace_event(
            "voice_server_frame_window_received",
            kind="io",
            details=self._received_frame_bucket.flush_details(),
        )

    def _trace_transcript_previews_enabled(self) -> bool:
        """Return whether routine traces may include raw transcript previews."""

        return bool(
            getattr(self, "voice_trace_include_transcripts", False)
            or getattr(
                getattr(self, "config", None),
                "voice_trace_include_transcripts",
                False,
            )
        )

    def _safe_transcript_preview(self, text: str | None, *, limit: int) -> str:
        """Return a preview that is privacy-safe for generic trace events."""

        resolved_text = str(text or "").strip()
        if not resolved_text:
            return ""
        if self._trace_transcript_previews_enabled():
            return resolved_text[:limit]
        # BREAKING: Routine trace events now redact transcript previews by
        # default. Set `voice_trace_include_transcripts=True` to restore raw
        # previews for explicit debugging sessions.
        digest = hashlib.blake2s(resolved_text.encode("utf-8"), digest_size=6).hexdigest()
        return f"[redacted chars={len(resolved_text)} hash={digest}]"

    def _scan_cache(self) -> dict[str, tuple[str, int, float]]:
        """Return the mutable cache used to suppress duplicate remote scans."""

        cache = getattr(self, "_recent_remote_scan_cache", None)
        if cache is None:
            cache = {}
            self._recent_remote_scan_cache = cache
        return cache

    def _duplicate_remote_scan_ttl_s(self) -> float:
        """Return how long an identical window should be considered a duplicate."""

        raw_value = getattr(self, "duplicate_remote_scan_ttl_s", None)
        if raw_value is None:
            raw_value = getattr(
                getattr(self, "config", None),
                "duplicate_remote_scan_ttl_s",
                None,
            )
        if raw_value is not None:
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                value = 0.0
            if value > 0.0:
                return value
        return max(0.5, (max(1, int(self.chunk_ms)) * 4) / 1000.0)

    def _capture_fingerprint(self, capture: AmbientAudioCaptureWindow) -> str:
        """Build one stable content fingerprint for one capture window."""

        digest = hashlib.blake2s(digest_size=12)
        digest.update(int(capture.sample.duration_ms).to_bytes(4, "little", signed=False))
        digest.update(int(capture.sample_rate).to_bytes(4, "little", signed=False))
        digest.update(int(capture.channels).to_bytes(2, "little", signed=False))
        digest.update(capture.pcm_bytes)
        return digest.hexdigest()

    def _duplicate_remote_scan_min_new_audio_ms(self) -> int:
        """Return the minimum audio novelty required before re-scanning a window."""

        raw_value = getattr(self, "duplicate_remote_scan_min_new_audio_ms", None)
        if raw_value is None:
            raw_value = getattr(
                getattr(self, "config", None),
                "duplicate_remote_scan_min_new_audio_ms",
                None,
            )
        if raw_value is None:
            return max(1, int(self.chunk_ms))
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            return max(1, int(self.chunk_ms))
        return max(1, value)

    def _should_skip_duplicate_remote_scan(
        self,
        *,
        scan_kind: str,
        capture: AmbientAudioCaptureWindow,
    ) -> bool:
        """Return whether an identical capture window was already scanned recently."""

        fingerprint = self._capture_fingerprint(capture)
        cache = self._scan_cache()
        previous = cache.get(scan_kind)
        now = self._monotonic()
        cache[scan_kind] = (fingerprint, int(capture.sample.duration_ms), now)
        if previous is None:
            return False
        previous_fingerprint, previous_duration_ms, previous_at = previous
        return (
            previous_fingerprint == fingerprint
            and abs(previous_duration_ms - int(capture.sample.duration_ms))
            < self._duplicate_remote_scan_min_new_audio_ms()
            and (now - previous_at) <= self._duplicate_remote_scan_ttl_s()
        )

    def _recompute_pending_utterance_metrics(
        self,
        pending: _PendingTranscriptUtterance,
    ) -> None:
        """Rebuild utterance timing from retained frames after every trim/mutation."""

        frames = tuple(pending.frames)
        speech_flags = self._remote_asr_speech_flags(frames)
        pending.captured_ms = 0
        pending.active_ms = 0
        pending.trailing_silence_ms = 0
        pending.speech_active = False
        for frame, is_active in zip(frames, speech_flags, strict=False):
            frame_ms = max(0, int(frame.duration_ms))
            pending.captured_ms += frame_ms
            if is_active:
                pending.active_ms += frame_ms
                pending.trailing_silence_ms = 0
            else:
                pending.trailing_silence_ms += frame_ms
            pending.speech_active = bool(is_active)
        if not frames:
            pending.trailing_silence_ms = 0
            pending.speech_active = False

    def _maybe_semantic_endpoint_ready(
        self,
        pending: _PendingTranscriptUtterance,
    ) -> bool | None:
        """Ask an optional semantic endpointing hook whether the utterance is done."""

        resolver = getattr(self, "_semantic_endpoint_ready", None)
        if not callable(resolver):
            return None
        capture = self._pending_transcript_capture_window(pending)
        try:
            decision = resolver(  # pylint: disable=not-callable
                capture=capture,
                pending=pending,
                origin_state=pending.origin_state,
                active_ms=int(pending.active_ms),
                trailing_silence_ms=int(pending.trailing_silence_ms),
            )
        except TypeError:
            decision = resolver(capture)  # pylint: disable=not-callable
        except Exception:
            return None
        if decision is None:
            return None
        return bool(decision)

    def _safe_backend_transcribe_capture(
        self,
        *,
        capture: AmbientAudioCaptureWindow,
        stage: str,
        filename: str = "voice-window.wav",
    ) -> str | None:
        """Run one backend transcription without letting transient errors kill the loop."""

        try:
            with self._backend_request_context(stage=stage, capture=capture):
                transcript = self.backend.transcribe(
                    pcm_capture_to_wav_bytes(capture),
                    filename=filename,
                    content_type="audio/wav",
                    language=self.config.openai_realtime_language,
                )
        except Exception as exc:
            self._record_transcript_debug(
                stage=stage,
                outcome="backend_error",
                capture=capture,
                details={"error_type": type(exc).__name__},
            )
            self._trace_event(
                "voice_backend_transcribe_error",
                kind="error",
                details={
                    "stage": stage,
                    "error_type": type(exc).__name__,
                    "window_ms": int(capture.sample.duration_ms),
                },
            )
            return None
        return str(transcript or "").strip()

    def _advance_remote_asr_utterance(
        self,
    ) -> OrchestratorVoiceWakeConfirmedEvent | OrchestratorVoiceTranscriptCommittedEvent | None:
        """Drive one same-stream utterance until endpointing decides it is complete."""

        latest_frame = self._history[-1] if self._history else None
        if latest_frame is None:
            return None
        if not self._runtime_state_attested:
            return None
        pending = self._pending_transcript_utterance
        if pending is None:
            if self._state == "waiting" and not self._waiting_activation_allowed():
                return None
            if not self._frame_counts_as_remote_asr_speech(latest_frame):
                return None
            pending = self._new_pending_transcript_utterance(origin_state=self._state)
            self._pending_transcript_utterance = pending
            if pending.active_ms < self._effective_remote_asr_min_activation_duration_ms():
                self._record_transcript_debug(
                    stage="activation_utterance",
                    outcome="buffering_short_utterance",
                    capture=self._pending_transcript_capture_window(pending),
                    details={
                        "origin_state": pending.origin_state,
                        "active_ms": pending.active_ms,
                        "required_active_ms": self._effective_remote_asr_min_activation_duration_ms(),
                        **self._pending_utterance_details(pending),
                        **self._wake_bias_details(origin_state=pending.origin_state),
                        **self._intent_context.trace_details(),
                    },
                )
            return None
        if pending.origin_state == "waiting" and not self._waiting_activation_allowed():
            self._cancel_blocked_waiting_activation_buffers(
                previous_state=pending.origin_state,
                detail="runtime_context_blocked",
            )
            return None
        self._append_pending_frame(pending, latest_frame)
        should_finalize = False
        if pending.captured_ms >= pending.max_capture_ms:
            should_finalize = True
        elif pending.active_ms > 0:
            semantic_endpoint = self._maybe_semantic_endpoint_ready(pending)
            if semantic_endpoint is True:
                should_finalize = True
            elif semantic_endpoint is None and (
                pending.trailing_silence_ms >= self.wake_tail_endpoint_silence_ms
            ):
                should_finalize = True
        if not should_finalize:
            return None
        self._pending_transcript_utterance = None
        capture = self._pending_transcript_capture_window(pending)
        if pending.active_ms < self._effective_remote_asr_min_activation_duration_ms():
            self._record_transcript_debug(
                stage="activation_utterance",
                outcome="rejected_short_utterance",
                capture=capture,
                details={
                    "origin_state": pending.origin_state,
                    "active_ms": pending.active_ms,
                    "required_active_ms": self._effective_remote_asr_min_activation_duration_ms(),
                    **self._pending_utterance_details(pending),
                    **self._intent_context.trace_details(),
                },
            )
            return None
        familiar_speaker_assessment = (
            self._assess_familiar_speaker_capture(capture, origin_state=pending.origin_state)
            if pending.origin_state == "waiting"
            else None
        )
        match_details = {
            "origin_state": pending.origin_state,
            **self._pending_utterance_details(pending),
        }
        if pending.origin_state == "waiting":
            match = self._detect_wake_capture(
                capture=capture,
                stage="activation_utterance",
                details=match_details,
            )
        else:
            supports_transcript_matching = self._wake_phrase_spotter_supports_transcript_matching(
                capture=capture,
                origin_state=pending.origin_state,
            )
            if supports_transcript_matching:
                transcript = self._transcribe_capture(
                    capture=capture,
                    stage="activation_utterance",
                    details=match_details,
                )
                if transcript is None:
                    return None
                match = self._match_transcribed_wake(
                    transcript=transcript,
                    capture=capture,
                    stage="activation_utterance",
                    details=match_details,
                )
            else:
                match = self._detect_wake_capture(
                    capture=capture,
                    stage="activation_utterance",
                    details=match_details,
                )
                if match is None:
                    return None
                if not match.detected:
                    transcript = self._transcribe_capture(
                        capture=capture,
                        stage="activation_utterance",
                        details=match_details,
                    )
                    if transcript is None:
                        return None
                    match = VoiceActivationMatch(
                        detected=False,
                        transcript=transcript,
                        normalized_transcript=str(transcript or "").strip().lower(),
                        backend=self.backend_name,
                    )
        if match is None:
            return None
        outcome = "matched" if match.detected else "no_match"
        self._record_transcript_debug(
            stage="activation_utterance",
            outcome=outcome,
            transcript=match.transcript,
            matched_phrase=match.matched_phrase,
            remaining_text=match.remaining_text,
            detector_label=match.detector_label,
            score=match.score,
            capture=capture,
            details={
                "origin_state": pending.origin_state,
                **self._pending_utterance_details(pending),
                **self._wake_bias_details(
                    origin_state=pending.origin_state,
                    familiar_speaker_assessment=familiar_speaker_assessment,
                ),
            },
        )
        if match.detected:
            self._state = "thinking"
            self._follow_up_deadline_at = None
            self._follow_up_opened_at = None
            self._trace_event(
                "voice_remote_asr_utterance_matched",
                kind="decision",
                details={
                    "origin_state": pending.origin_state,
                    "matched_phrase": match.matched_phrase,
                    "remaining_text_chars": len(str(match.remaining_text or "").strip()),
                    "transcript_chars": len(str(match.transcript or "").strip()),
                    **self._wake_bias_details(
                        origin_state=pending.origin_state,
                        familiar_speaker_assessment=familiar_speaker_assessment,
                    ),
                },
            )
            return OrchestratorVoiceWakeConfirmedEvent(
                matched_phrase=match.matched_phrase,
                remaining_text=str(match.remaining_text or "").strip(),
                backend=match.backend or self.backend_name,
                detector_label=match.detector_label,
                score=match.score,
            )
        transcript = str(match.transcript or "").strip()
        if pending.origin_state == "follow_up_open" and _normalize_text_length(transcript) >= self.follow_up_min_transcript_chars:
            self._state = "waiting"
            self._follow_up_deadline_at = None
            self._follow_up_opened_at = None
            self._record_transcript_debug(
                stage="activation_utterance",
                outcome="follow_up_committed",
                transcript=transcript,
                capture=capture,
                details={"origin_state": pending.origin_state},
            )
            self._trace_event(
                "voice_remote_asr_follow_up_committed",
                kind="decision",
                details={
                    "transcript_chars": len(transcript),
                    "transcript_preview": self._safe_transcript_preview(transcript, limit=80),
                },
            )
            return OrchestratorVoiceTranscriptCommittedEvent(
                transcript=transcript,
                source="follow_up",
            )
        if pending.origin_state == "listening" and _normalize_text_length(transcript) >= self.follow_up_min_transcript_chars:
            self._state = "thinking"
            self._record_transcript_debug(
                stage="activation_utterance",
                outcome="listening_committed",
                transcript=transcript,
                capture=capture,
                details={"origin_state": pending.origin_state},
            )
            return OrchestratorVoiceTranscriptCommittedEvent(
                transcript=transcript,
                source="listening",
            )
        return None

    def _new_pending_transcript_utterance(self, *, origin_state: str) -> _PendingTranscriptUtterance:
        """Seed one same-stream utterance from the latest active speech burst."""

        seed_frames = self._latest_active_speech_burst_frames(self._recent_frames_window(self.history_ms))
        if not seed_frames:
            seed_frames = self._recent_frames_window(self.chunk_ms)
        return self._pending_transcript_utterance_from_frames(
            origin_state=origin_state,
            frames=seed_frames,
        )

    def _pending_transcript_utterance_from_frames(
        self,
        *,
        origin_state: str,
        frames: tuple[_RecentFrame, ...],
        max_capture_ms: int | None = None,
    ) -> _PendingTranscriptUtterance:
        """Build one bounded pending utterance from the provided frame slice."""

        pending = _PendingTranscriptUtterance(
            origin_state=origin_state,
            max_capture_ms=max_capture_ms
            if max_capture_ms is not None
            else max(self.history_ms, self.wake_candidate_window_ms + self.wake_tail_max_ms),
        )
        for frame in frames:
            self._append_pending_frame(pending, frame)
        return pending

    def _recent_frames_window(self, duration_ms: int) -> tuple[_RecentFrame, ...]:
        """Return the newest bounded frame sequence for one duration window."""

        target_ms = max(self.chunk_ms, int(duration_ms))
        frames: list[_RecentFrame] = []
        collected_ms = 0
        for frame in reversed(self._history):
            frames.append(frame)
            collected_ms += frame.duration_ms
            if collected_ms >= target_ms:
                break
        frames.reverse()
        return tuple(frames)

    def _append_pending_frame(
        self,
        pending: _PendingTranscriptUtterance,
        frame: _RecentFrame,
    ) -> None:
        """Append one frame into the wake buffer and trim to the bounded budget."""

        pending.frames.append(frame)
        pending.captured_ms += frame.duration_ms
        while pending.captured_ms > pending.max_capture_ms and pending.frames:
            removed = pending.frames.popleft()
            pending.captured_ms = max(0, pending.captured_ms - removed.duration_ms)
        self._recompute_pending_utterance_metrics(pending)

    def _pending_transcript_capture_window(
        self,
        pending: _PendingTranscriptUtterance,
    ) -> AmbientAudioCaptureWindow:
        """Build one capture window from the bounded same-stream utterance buffer."""

        if pending.frames:
            frames = tuple(pending.frames)
        else:
            frames = self._recent_frames_window(self.history_ms)
        return self._capture_window_from_frames(frames)

    def _recent_remote_asr_stage1_capture(
        self,
        *,
        window_ms: int | None = None,
    ) -> AmbientAudioCaptureWindow:
        """Prefer the start of the latest active speech burst over the newest tail."""

        target_window_ms = (
            self._effective_remote_asr_stage1_window_ms()
            if window_ms is None
            else max(self.chunk_ms, min(self.history_ms, int(window_ms)))
        )
        recent_frames = self._recent_frames_window(self.history_ms)
        burst_frames = self._latest_active_speech_burst_frames(recent_frames)
        if burst_frames:
            return self._capture_window_from_frames(
                self._leading_frames_window(
                    burst_frames,
                    duration_ms=target_window_ms,
                )
            )
        return self._capture_window_from_frames(
            self._leading_frames_window(
                recent_frames,
                duration_ms=target_window_ms,
            )
        )

    def _latest_active_speech_burst_frames(
        self,
        frames: tuple[_RecentFrame, ...],
    ) -> tuple[_RecentFrame, ...]:
        """Return the latest speech burst plus contiguous non-silent onset frames."""

        resolved_frames = tuple(frames)
        if not resolved_frames:
            return ()
        speech_flags = self._remote_asr_speech_flags(resolved_frames)
        last_active_index: int | None = None
        for index in range(len(resolved_frames) - 1, -1, -1):
            if speech_flags[index]:
                last_active_index = index
                break
        if last_active_index is None:
            return ()
        max_silence_ms = max(
            self.chunk_ms,
            min(self.follow_up_window_ms, self._effective_remote_asr_stage1_window_ms()),
        )
        silence_ms = 0
        start_index = last_active_index
        have_active = False
        for index, frame in enumerate(resolved_frames[: last_active_index + 1]):
            if speech_flags[index]:
                if not have_active or silence_ms >= max_silence_ms:
                    start_index = index
                have_active = True
                silence_ms = 0
                continue
            if have_active:
                silence_ms += max(0, int(frame.duration_ms))
        pre_roll_ms = min(
            self._effective_remote_asr_stage1_window_ms(),
            max(self.chunk_ms, self.chunk_ms * 2),
        )
        pre_roll_start_index = start_index
        collected_pre_roll_ms = 0
        while pre_roll_start_index > 0 and collected_pre_roll_ms < pre_roll_ms:
            previous_frame = resolved_frames[pre_roll_start_index - 1]
            if max(0, int(getattr(previous_frame, "rms", 0))) <= 0:
                break
            collected_pre_roll_ms += max(0, int(previous_frame.duration_ms))
            pre_roll_start_index -= 1
        return resolved_frames[pre_roll_start_index : last_active_index + 1]

    def _leading_frames_window(
        self,
        frames: tuple[_RecentFrame, ...],
        *,
        duration_ms: int,
    ) -> tuple[_RecentFrame, ...]:
        """Return the earliest bounded prefix from one recent frame sequence."""

        target_ms = max(self.chunk_ms, int(duration_ms))
        selected: list[_RecentFrame] = []
        collected_ms = 0
        for frame in frames:
            selected.append(frame)
            collected_ms += max(0, int(frame.duration_ms))
            if collected_ms >= target_ms:
                break
        return tuple(selected)

    def _maybe_detect_remote_asr_activation_candidate(self) -> VoiceActivationMatch | None:
        """Transcribe a recent speech window remotely and match activation phrases."""

        now = self._monotonic()
        if now < self._next_wake_candidate_check_at:
            return None
        try:
            with self._trace_span(
                name="voice_remote_asr_stage1_scan",
                kind="llm_call",
                details={
                    "window_ms": self._effective_remote_asr_stage1_window_ms(),
                    **self._intent_context.trace_details(),
                },
            ):
                capture = self._recent_remote_asr_stage1_capture()
            if capture.sample.peak_rms <= 0 and capture.sample.average_rms <= 0:
                self._record_transcript_debug(
                    stage="activation_stage1",
                    outcome="buffering_no_audio_energy",
                    capture=capture,
                    details={"reason": "nonzero_audio_energy_required"},
                )
                self._trace_decision(
                    "voice_remote_asr_stage1_buffering",
                    question="Should this remote-ASR activation candidate be scanned already?",
                    selected={"id": "buffer", "summary": "Wait for non-empty audio before scanning"},
                    options=[
                        {"id": "buffer", "summary": "Keep buffering until non-empty audio arrives"},
                        {"id": "scan", "summary": "Transcribe the current silent/empty window now"},
                    ],
                    context={
                        "window_ms": capture.sample.duration_ms,
                        "peak_rms": int(capture.sample.peak_rms),
                        "average_rms": int(capture.sample.average_rms),
                    },
                    confidence="high",
                    guardrails=["nonzero_audio_energy_required"],
                )
                return None
            if capture.sample.duration_ms < self._effective_remote_asr_min_activation_duration_ms():
                self._record_transcript_debug(
                    stage="activation_stage1",
                    outcome="buffering_short_wake_burst",
                    capture=capture,
                    details={
                        "reason": "remote_asr_min_activation_duration_ms",
                        "required_window_ms": self._effective_remote_asr_min_activation_duration_ms(),
                        **self._intent_context.trace_details(),
                    },
                )
                self._trace_decision(
                    "voice_remote_asr_stage1_buffering",
                    question="Should this remote-ASR activation candidate be scanned already?",
                    selected={"id": "buffer", "summary": "Wait for a longer wake burst before scanning"},
                    options=[
                        {"id": "buffer", "summary": "Keep buffering wake speech"},
                        {"id": "scan", "summary": "Transcribe the short wake burst now"},
                    ],
                    context={
                        "window_ms": capture.sample.duration_ms,
                        "required_window_ms": self._effective_remote_asr_min_activation_duration_ms(),
                        "active_ratio": round(float(capture.sample.active_ratio), 4),
                    },
                    confidence="high",
                    guardrails=["remote_asr_min_activation_duration_ms"],
                )
                return None
            if (
                self.wake_candidate_min_active_ratio > 0.0
                and capture.sample.active_ratio < self.wake_candidate_min_active_ratio
            ):
                self._record_transcript_debug(
                    stage="activation_stage1",
                    outcome="rejected_low_activity",
                    capture=capture,
                    details={
                        "reason": "wake_candidate_min_active_ratio",
                        "required_active_ratio": self.wake_candidate_min_active_ratio,
                    },
                )
                self._trace_decision(
                    "voice_remote_asr_stage1_rejected",
                    question="Should this remote-ASR activation candidate be scanned?",
                    selected={"id": "reject", "summary": "Active ratio below configured threshold"},
                    options=[
                        {"id": "reject", "summary": "Skip low-activity candidate"},
                        {"id": "scan", "summary": "Run transcript-first wake scan"},
                    ],
                    context={"active_ratio": round(float(capture.sample.active_ratio), 4)},
                    confidence="high",
                    guardrails=["wake_candidate_min_active_ratio"],
                )
                return None
            familiar_speaker_assessment = self._assess_familiar_speaker_capture(
                capture,
                origin_state="waiting",
            )
            if self._familiar_speaker_wake_bias_active(
                familiar_speaker_assessment,
                origin_state="waiting",
            ):
                expanded_capture = self._recent_remote_asr_stage1_capture(
                    window_ms=(
                        self._effective_remote_asr_stage1_window_ms()
                        + self._STRONG_SPEAKER_STAGE1_WINDOW_BONUS_MS
                    )
                )
                if expanded_capture.sample.duration_ms > capture.sample.duration_ms:
                    capture = expanded_capture
                    familiar_speaker_assessment = self._assess_familiar_speaker_capture(
                        capture,
                        origin_state="waiting",
                    )
            if self._should_skip_duplicate_remote_scan(
                scan_kind="remote_asr_stage1",
                capture=capture,
            ):
                self._record_transcript_debug(
                    stage="activation_stage1",
                    outcome="buffering_duplicate_window",
                    capture=capture,
                    details={"reason": "duplicate_remote_scan"},
                )
                return None
            match = self._detect_wake_capture(
                capture=capture,
                stage="activation_stage1",
                details={"origin_state": "waiting"},
            )
            if match is None:
                return None
            if not match.detected:
                self._record_transcript_debug(
                    stage="activation_stage1",
                    outcome="no_match",
                    transcript=match.transcript,
                    matched_phrase=match.matched_phrase,
                    remaining_text=match.remaining_text,
                    detector_label=match.detector_label,
                    score=match.score,
                    capture=capture,
                    details=self._wake_bias_details(
                        origin_state="waiting",
                        familiar_speaker_assessment=familiar_speaker_assessment,
                    ),
                )
                self._trace_event(
                    "voice_remote_asr_stage1_no_match",
                    kind="branch",
                    details={
                        "active_ratio": round(float(capture.sample.active_ratio), 4),
                        "transcript_chars": len(str(match.transcript or "").strip()),
                        **self._wake_bias_details(
                            origin_state="waiting",
                            familiar_speaker_assessment=familiar_speaker_assessment,
                        ),
                    },
                )
                return None
            if _normalize_text_length(match.transcript) < self.wake_candidate_min_transcript_chars:
                self._record_transcript_debug(
                    stage="wake_stage1",
                    outcome="rejected_short_transcript",
                    transcript=match.transcript,
                    matched_phrase=match.matched_phrase,
                    remaining_text=match.remaining_text,
                    detector_label=match.detector_label,
                    score=match.score,
                    capture=capture,
                    details={
                        "reason": "wake_candidate_min_transcript_chars",
                        "required_transcript_chars": self.wake_candidate_min_transcript_chars,
                        **self._wake_bias_details(
                            origin_state="waiting",
                            familiar_speaker_assessment=familiar_speaker_assessment,
                        ),
                    },
                )
                self._trace_decision(
                    "voice_remote_asr_stage1_rejected",
                    question="Should this remote-ASR activation candidate be accepted?",
                    selected={"id": "reject", "summary": "Transcript too short after wake scan"},
                    options=[
                        {"id": "reject", "summary": "Reject low-evidence transcript"},
                        {"id": "accept", "summary": "Accept wake candidate"},
                    ],
                    context={
                        "transcript_chars": len(str(match.transcript or "").strip()),
                        "matched_phrase": match.matched_phrase,
                    },
                    confidence="medium",
                    guardrails=["wake_candidate_min_transcript_chars"],
                )
                return None
            self._record_transcript_debug(
                stage="activation_stage1",
                outcome="matched",
                transcript=match.transcript,
                matched_phrase=match.matched_phrase,
                remaining_text=match.remaining_text,
                detector_label=match.detector_label,
                score=match.score,
                capture=capture,
                details=self._wake_bias_details(
                    origin_state="waiting",
                    familiar_speaker_assessment=familiar_speaker_assessment,
                ),
            )
            self._trace_event(
                "voice_remote_asr_stage1_match",
                kind="decision",
                details={
                    "matched_phrase": match.matched_phrase,
                    "transcript_chars": len(str(match.transcript or "").strip()),
                    "remaining_text_chars": len(str(match.remaining_text or "").strip()),
                    **self._wake_bias_details(
                        origin_state="waiting",
                        familiar_speaker_assessment=familiar_speaker_assessment,
                    ),
                },
            )
            return match
        finally:
            self._next_wake_candidate_check_at = self._monotonic() + self._wake_candidate_cooldown_s

    def _maybe_detect_barge_in_candidate(self) -> OrchestratorVoiceBargeInInterruptEvent | None:
        """Transcribe one bounded speaking window to detect a user interruption."""

        if self._monotonic() < self._next_barge_in_candidate_check_at:
            return None
        capture = self._recent_capture_window(self.barge_in_window_ms)
        self._next_barge_in_candidate_check_at = self._monotonic() + self.candidate_cooldown_s
        if (
            capture.sample.active_chunk_count <= 0
            or capture.sample.active_ratio < self.barge_in_min_active_ratio
        ):
            self._record_transcript_debug(
                stage="barge_in_candidate",
                outcome="rejected_low_activity",
                capture=capture,
                details={"required_active_ratio": self.barge_in_min_active_ratio},
            )
            self._trace_decision(
                "voice_barge_in_candidate_rejected",
                question="Should this speech candidate trigger a remote action?",
                selected={"id": "reject", "summary": "Insufficient active speech evidence"},
                options=[
                    {"id": "reject", "summary": "Ignore low-activity window"},
                    {"id": "transcribe", "summary": "Transcribe candidate window"},
                ],
                context={
                    "active_chunk_count": capture.sample.active_chunk_count,
                    "active_ratio": round(float(capture.sample.active_ratio), 4),
                },
                confidence="high",
                guardrails=["speech_activity_threshold"],
            )
            return None
        if self._should_skip_duplicate_remote_scan(scan_kind="barge_in", capture=capture):
            self._record_transcript_debug(
                stage="barge_in_candidate",
                outcome="buffering_duplicate_window",
                capture=capture,
                details={"reason": "duplicate_remote_scan"},
            )
            return None
        with self._trace_span(
            name="voice_barge_in_candidate_transcribe",
            kind="llm_call",
            details={"window_ms": capture.sample.duration_ms},
        ):
            transcript = self._safe_backend_transcribe_capture(
                capture=capture,
                stage="barge_in_candidate",
            )
        if transcript is None:
            return None
        self._record_transcript_debug(
            stage="barge_in_candidate",
            outcome="transcribed",
            transcript=transcript,
            capture=capture,
        )
        if _normalize_text_length(transcript) < self.barge_in_min_transcript_chars:
            self._record_transcript_debug(
                stage="barge_in_candidate",
                outcome="rejected_short_transcript",
                transcript=transcript,
                capture=capture,
                details={"required_transcript_chars": self.barge_in_min_transcript_chars},
            )
            self._trace_decision(
                "voice_barge_in_candidate_rejected",
                question="Should this transcribed speech candidate trigger a remote action?",
                selected={"id": "reject", "summary": "Transcript did not contain enough speech"},
                options=[
                    {"id": "reject", "summary": "Ignore low-evidence transcript"},
                    {"id": "accept", "summary": "Trigger remote action"},
                ],
                context={
                    "transcript_chars": len(transcript),
                    "active_ratio": round(float(capture.sample.active_ratio), 4),
                },
                confidence="medium",
                guardrails=["min_transcript_chars"],
            )
            return None
        self._record_transcript_debug(
            stage="barge_in_candidate",
            outcome="interrupt_requested",
            transcript=transcript,
            capture=capture,
        )
        self._trace_event(
            "voice_barge_in_candidate_triggered",
            kind="decision",
            details={
                "transcript_chars": len(transcript),
                "transcript_preview": self._safe_transcript_preview(transcript, limit=80),
            },
        )
        return OrchestratorVoiceBargeInInterruptEvent(transcript_preview=transcript[:160])

    def _recent_capture_window(self, duration_ms: int) -> AmbientAudioCaptureWindow:
        """Return one bounded ambient window from the recent frame history."""

        frames = self._recent_frames_window(duration_ms)
        return self._capture_window_from_frames(frames)

    def _capture_window_from_frames(
        self,
        frames: tuple[_RecentFrame, ...] | list[_RecentFrame],
    ) -> AmbientAudioCaptureWindow:
        """Assemble one ambient capture window from a bounded frame sequence."""

        resolved_frames = tuple(frames)
        if not resolved_frames:
            resolved_frames = (_RecentFrame(pcm_bytes=b"", rms=0, duration_ms=0),)
        pcm_fragments = [frame.pcm_bytes for frame in resolved_frames]
        rms_values = [int(frame.rms) for frame in resolved_frames]
        duration_values = [max(0, int(frame.duration_ms)) for frame in resolved_frames]
        collected_ms = sum(duration_values)
        speech_flags = self._remote_asr_speech_flags(resolved_frames)
        active_chunk_count = sum(1 for active in speech_flags if active)
        active_ms = sum(
            duration_ms
            for duration_ms, is_active in zip(duration_values, speech_flags, strict=False)
            if is_active
        )
        sample = AmbientAudioLevelSample(
            duration_ms=collected_ms,
            chunk_count=len(rms_values),
            active_chunk_count=active_chunk_count,
            average_rms=int(sum(rms_values) / max(1, len(rms_values))),
            peak_rms=max(rms_values),
            # BREAKING: `active_ratio` is now duration-weighted rather than
            # frame-count-weighted so variable-size chunks cannot skew VAD
            # gating decisions.
            active_ratio=(active_ms / collected_ms) if collected_ms > 0 else 0.0,
        )
        return AmbientAudioCaptureWindow(
            sample=sample,
            pcm_bytes=b"".join(pcm_fragments),
            sample_rate=self.sample_rate,
            channels=self.channels,
        )
