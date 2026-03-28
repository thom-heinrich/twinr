"""Tracing and transcript-debug helpers for the orchestrator voice session."""

from __future__ import annotations

from contextlib import nullcontext

from twinr.agent.workflows.forensics import WorkflowForensics
from twinr.hardware.audio import AmbientAudioCaptureWindow, pcm16_signal_profile
from twinr.orchestrator.voice_activation import VoiceActivationPhraseMatcher
from twinr.orchestrator.voice_familiarity import FamiliarSpeakerWakeAssessment
from twinr.orchestrator.voice_forensics import prefixed_signal_profile_details


class VoiceSessionObservabilityMixin:
    """Own the trace/debug surfaces shared by the voice session helpers."""

    def set_forensics(
        self,
        tracer: WorkflowForensics | None,
        *,
        trace_id: str | None = None,
    ) -> None:
        """Bind one shared forensic tracer for this websocket voice session."""

        if isinstance(tracer, WorkflowForensics) and tracer.enabled:
            self._forensics = tracer
        else:
            self._forensics = None
        configure_backend_forensics = getattr(self.backend, "set_forensics", None)
        if callable(configure_backend_forensics):
            configure_backend_forensics(self._forensics)
        if trace_id:
            self._trace_id = str(trace_id)

    def _trace_details(self, details: dict[str, object] | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "session_id": self._session_id or None,
            "state": self._state,
            "backend": self.backend_name,
            "follow_up_allowed": self._follow_up_allowed,
            "voice_quiet_active": self._voice_quiet_active(),
            "voice_quiet_until_utc": self._voice_quiet_until_utc,
        }
        payload.update(self._intent_context.trace_details())
        if details:
            payload.update(details)
        return payload

    def _strong_speaker_wake_bias_active(self, *, origin_state: str | None = None) -> bool:
        """Return whether strong local speaker context exists for waiting wake."""

        resolved_origin_state = str(origin_state or "").strip().lower() or self._state
        if resolved_origin_state != "waiting":
            return False
        return self._intent_context.familiar_speaker_bias_allowed()

    def _familiar_speaker_wake_bias_active(
        self,
        assessment: FamiliarSpeakerWakeAssessment | None,
        *,
        origin_state: str | None = None,
    ) -> bool:
        """Return whether contextual alias expansion may use familiar-speaker bias."""

        if not self._strong_speaker_wake_bias_active(origin_state=origin_state):
            return False
        return bool(assessment is not None and assessment.familiar)

    def _wake_phrase_spotter_for_origin_state(
        self,
        *,
        origin_state: str | None = None,
        allow_contextual_aliases: bool = False,
    ) -> VoiceActivationPhraseMatcher:
        """Return the wake matcher appropriate for the current bias context."""

        del origin_state
        if allow_contextual_aliases:
            return self._strong_bias_wake_phrase_spotter
        return self._wake_phrase_spotter

    def _wake_bias_details(
        self,
        *,
        origin_state: str | None = None,
        familiar_speaker_assessment: FamiliarSpeakerWakeAssessment | None = None,
    ) -> dict[str, object]:
        """Return trace/debug details for the current wake bias tier."""

        return {
            "wake_audio_bias_active": self._intent_audio_bias_active(),
            "wake_strong_speaker_bias_active": self._strong_speaker_wake_bias_active(
                origin_state=origin_state
            ),
            "wake_alias_expansion_active": self._familiar_speaker_wake_bias_active(
                familiar_speaker_assessment,
                origin_state=origin_state,
            ),
            "wake_speaker_associated": self._intent_context.speaker_associated,
            "wake_speaker_association_confidence": self._intent_context.speaker_association_confidence,
            **(
                {}
                if familiar_speaker_assessment is None
                else familiar_speaker_assessment.trace_details()
            ),
        }

    def _trace_event(
        self,
        msg: str,
        *,
        kind: str,
        details: dict[str, object] | None = None,
        reason: dict[str, object] | None = None,
        kpi: dict[str, object] | None = None,
        level: str = "INFO",
    ) -> None:
        tracer = self._forensics
        if not isinstance(tracer, WorkflowForensics):
            return
        tracer.event(
            kind=kind,
            msg=msg,
            details=self._trace_details(details),
            reason=reason,
            kpi=kpi,
            level=level,
            trace_id=self._trace_id,
            loc_skip=2,
        )

    def _trace_decision(
        self,
        msg: str,
        *,
        question: str,
        selected: dict[str, object],
        options: list[dict[str, object]],
        context: dict[str, object] | None = None,
        confidence: object | None = None,
        guardrails: list[str] | None = None,
        kpi_impact_estimate: dict[str, object] | None = None,
    ) -> None:
        tracer = self._forensics
        if not isinstance(tracer, WorkflowForensics):
            return
        tracer.decision(
            msg=msg,
            question=question,
            selected=selected,
            options=options,
            context=self._trace_details(context),
            confidence=confidence,
            guardrails=guardrails,
            kpi_impact_estimate=kpi_impact_estimate,
            trace_id=self._trace_id,
        )

    def _trace_span(self, *, name: str, kind: str, details: dict[str, object] | None = None):
        tracer = self._forensics
        if not isinstance(tracer, WorkflowForensics):
            return nullcontext()
        return tracer.span(
            name=name,
            kind=kind,
            details=self._trace_details(details),
            trace_id=self._trace_id,
        )

    def _capture_sample_details(
        self,
        capture: AmbientAudioCaptureWindow | None,
    ) -> dict[str, object]:
        """Return compact capture metrics for transcript debug entries."""

        if capture is None:
            return {}
        sample = capture.sample
        signal_profile = pcm16_signal_profile(capture.pcm_bytes)
        return {
            "duration_ms": int(sample.duration_ms),
            "chunk_count": int(sample.chunk_count),
            "active_chunk_count": int(sample.active_chunk_count),
            "average_rms": int(sample.average_rms),
            "peak_rms": int(sample.peak_rms),
            "active_ratio": round(float(sample.active_ratio), 6),
            **prefixed_signal_profile_details(signal_profile, prefix="signal"),
        }

    def _record_transcript_debug(
        self,
        *,
        stage: str,
        outcome: str,
        transcript: str | None = None,
        matched_phrase: str | None = None,
        remaining_text: str | None = None,
        detector_label: str | None = None,
        score: float | None = None,
        capture: AmbientAudioCaptureWindow | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        """Persist one raw transcript/debug record for the live gateway."""

        resolved_details = self._intent_context.trace_details()
        if details:
            resolved_details.update(details)
        if capture is not None:
            audio_artifact = self._audio_debug_store.persist_capture(
                capture=capture,
                session_id=self._session_id or None,
                trace_id=self._trace_id,
                stage=stage,
                outcome=outcome,
            )
            if audio_artifact:
                resolved_details.update(audio_artifact)
        self._transcript_debug_stream.append_entry(
            session_id=self._session_id or None,
            trace_id=self._trace_id,
            state=self._state,
            backend=self.backend_name,
            stage=stage,
            outcome=outcome,
            transcript=transcript,
            matched_phrase=matched_phrase,
            remaining_text=remaining_text,
            detector_label=detector_label,
            score=score,
            sample=self._capture_sample_details(capture),
            details=resolved_details,
        )
