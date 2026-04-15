# CHANGELOG: 2026-03-29
# BUG-1: Fallback to raw-audio wake detection when transcript matching raises, instead of silently dropping a valid wake event.
# BUG-2: Accept structured/None transcription payloads and normalize blank transcripts to None, avoiding false backend errors and empty-string downstream behavior.
# BUG-3: Catch wake-spotter / familiarity-resolution failures inside the helper surface, so misconfigured selectors do not crash the whole voice session.
# SEC-1: Stop persisting raw user transcripts in matcher-error debug paths by default; log length/hash/preview only unless plaintext debug is explicitly enabled.
# SEC-2: Reject empty / too-short / oversized captures before backend submission to reduce practical DoS and privacy exposure on Raspberry Pi deployments.
# IMP-1: Add capability-aware transcription routing hooks (`transcribe_<route>`, `transcribe_capture`, `select_transcription_route`) for 2026 hybrid edge/cloud backends.
# IMP-2: Add capability-aware transcription options parsing for modern ASR backends (structured responses, server-VAD chunking, explicit model hints) while staying drop-in compatible.
# IMP-3: Add transcript normalization, safe error/detail sanitization, richer request-context metadata, and optional capture-preparation hooks for privacy-preserving / denoised backend ASR.
"""Backend-facing wake detection and transcription helpers."""

from __future__ import annotations

import hashlib
import inspect
import re
import unicodedata
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

from twinr.hardware.audio import AmbientAudioCaptureWindow, pcm16_signal_profile
from twinr.orchestrator.voice_activation import VoiceActivationMatch
from twinr.orchestrator.voice_familiarity import (
    FamiliarSpeakerWakeAssessment,
    assess_familiar_speaker_pcm16,
)
from twinr.orchestrator.voice_forensics import prefixed_signal_profile_details

from .builders import pcm_capture_to_wav_bytes

_CONTROL_CHARS_RE = re.compile(r"[\u0000-\u001f\u007f-\u009f]")
_ZERO_WIDTH_CHARS_RE = re.compile(r"[\u200b-\u200f\u2060\ufeff]")
_WHITESPACE_RE = re.compile(r"\s+")


class VoiceSessionTranscriptionMixin:
    """Own the backend-ASR and wake-transcript request surface for the voice session."""

    _MIN_CAPTURE_MS = 80
    _MAX_CAPTURE_MS = 30_000
    _MAX_CAPTURE_BYTES = 4 * 1024 * 1024
    _MAX_TRANSCRIPT_CHARS = 4_096
    _MAX_ERROR_CHARS = 240
    _DEBUG_PREVIEW_CHARS = 96

    def _assess_familiar_speaker_capture(
        self,
        capture: AmbientAudioCaptureWindow,
        *,
        origin_state: str | None,
    ) -> FamiliarSpeakerWakeAssessment | None:
        resolved_origin_state = self._normalize_origin_state(origin_state) or self._state
        if resolved_origin_state != "waiting":
            return None

        if not self._voice_identity_profiles:
            return FamiliarSpeakerWakeAssessment(
                assessment=None,
                familiar=False,
                revision=self._voice_identity_profiles_revision,
                profile_count=0,
            )

        pcm_bytes = self._capture_pcm_bytes(capture)
        sample_rate = int(getattr(capture, "sample_rate", 0) or 0)
        channels = int(getattr(capture, "channels", 0) or 0)
        if not pcm_bytes or sample_rate <= 0 or channels <= 0:
            return FamiliarSpeakerWakeAssessment(
                assessment=None,
                familiar=False,
                revision=self._voice_identity_profiles_revision,
                profile_count=len(self._voice_identity_profiles),
            )

        return assess_familiar_speaker_pcm16(
            self.config,
            pcm_bytes=pcm_bytes,
            sample_rate=sample_rate,
            channels=channels,
            profiles=self._voice_identity_profiles,
            revision=self._voice_identity_profiles_revision,
        )

    def _detect_wake_capture(
        self,
        *,
        capture: AmbientAudioCaptureWindow,
        stage: str,
        details: dict[str, object] | None = None,
    ) -> VoiceActivationMatch | None:
        rejected_details = self._capture_rejection_details(capture)
        if rejected_details is not None:
            self._record_capture_rejection(
                stage=stage,
                capture=capture,
                details=details,
                rejection_details=rejected_details,
                mode="wake_detection",
            )
            return None

        origin_state = self._normalize_origin_state(
            None if details is None else details.get("origin_state")
        )
        try:
            familiar_speaker_assessment, wake_phrase_spotter = self._resolve_wake_phrase_spotter(
                capture,
                origin_state=origin_state,
            )
            with self._backend_request_context(
                stage=stage,
                capture=capture,
                origin_state=origin_state,
            ):
                return wake_phrase_spotter.detect(capture)
        except Exception as exc:
            resolved_details = self._build_error_details(
                exc,
                details=details,
                extra=self._wake_bias_details(
                    origin_state=origin_state,
                    familiar_speaker_assessment=locals().get("familiar_speaker_assessment"),
                ),
            )
            self._record_transcript_debug(
                stage=stage,
                outcome="backend_error",
                capture=capture,
                details=resolved_details,
            )
            self._trace_event(
                "voice_activation_backend_error",
                kind="warning",
                level="WARN",
                details={"stage": stage, **resolved_details},
            )
            return None

    def _transcribe_capture(
        self,
        *,
        capture: AmbientAudioCaptureWindow,
        stage: str,
        details: dict[str, object] | None = None,
        prompt: str | None = None,
    ) -> str | None:
        rejected_details = self._capture_rejection_details(capture)
        if rejected_details is not None:
            self._record_capture_rejection(
                stage=stage,
                capture=capture,
                details=details,
                rejection_details=rejected_details,
                mode="transcription",
            )
            return None

        origin_state = self._normalize_origin_state(
            None if details is None else details.get("origin_state")
        )
        prepared_capture = self._prepare_capture_for_backend_asr(
            capture=capture,
            stage=stage,
            origin_state=origin_state,
        )
        prepared_rejection = self._capture_rejection_details(prepared_capture)
        if prepared_rejection is not None:
            self._record_capture_rejection(
                stage=stage,
                capture=prepared_capture,
                details=details,
                rejection_details=prepared_rejection,
                mode="transcription_prepared",
            )
            return None

        resolved_prompt = self._normalize_prompt_text(prompt)

        try:
            with self._backend_request_context(
                stage=stage,
                capture=prepared_capture,
                origin_state=origin_state,
            ):
                response = self._call_backend_transcription(
                    capture=prepared_capture,
                    stage=stage,
                    origin_state=origin_state,
                    prompt=resolved_prompt,
                )
            transcript = self._extract_transcription_text(response)
            if transcript is None:
                resolved_details = self._sanitize_details(details)
                resolved_details["response_type"] = type(response).__name__ if response is not None else "NoneType"
                if resolved_prompt is not None:
                    resolved_details["prompt_chars"] = len(resolved_prompt)
                self._record_transcript_debug(
                    stage=stage,
                    outcome="empty_transcript",
                    capture=prepared_capture,
                    details=resolved_details,
                )
            return transcript
        except Exception as exc:
            resolved_details = self._build_error_details(exc, details=details)
            if resolved_prompt is not None:
                resolved_details["prompt_chars"] = len(resolved_prompt)
            self._record_transcript_debug(
                stage=stage,
                outcome="backend_error",
                capture=prepared_capture,
                details=resolved_details,
            )
            self._trace_event(
                "voice_transcription_backend_error",
                kind="warning",
                level="WARN",
                details={"stage": stage, **resolved_details},
            )
            return None

    def _match_transcribed_wake(
        self,
        *,
        transcript: str,
        capture: AmbientAudioCaptureWindow,
        stage: str,
        details: dict[str, object] | None = None,
    ) -> VoiceActivationMatch | None:
        normalized_transcript = self._normalize_transcript_text(transcript)
        if normalized_transcript is None:
            self._record_transcript_debug(
                stage=stage,
                outcome="empty_transcript",
                capture=capture,
                details={
                    **self._sanitize_details(details),
                    **self._transcript_debug_details(transcript),
                },
            )
            return None

        origin_state = self._normalize_origin_state(
            None if details is None else details.get("origin_state")
        )
        try:
            familiar_speaker_assessment, wake_phrase_spotter = self._resolve_wake_phrase_spotter(
                capture,
                origin_state=origin_state,
            )
        except Exception as exc:
            resolved_details = self._build_error_details(
                exc,
                details=details,
                extra=self._transcript_debug_details(normalized_transcript),
            )
            self._record_transcript_debug(
                stage=stage,
                outcome="matcher_error",
                capture=capture,
                details=resolved_details,
            )
            self._trace_event(
                "voice_activation_matcher_error",
                kind="error",
                level="ERROR",
                details={"stage": stage, **resolved_details},
            )
            raise RuntimeError(
                "Wake transcript matching could not resolve the active wake phrase matcher."
            ) from exc

        match_transcript = getattr(wake_phrase_spotter, "match_transcript", None)
        if not callable(match_transcript):
            resolved_details = {
                **self._sanitize_details(details),
                **self._wake_bias_details(
                    origin_state=origin_state,
                    familiar_speaker_assessment=familiar_speaker_assessment,
                ),
                **self._transcript_debug_details(normalized_transcript),
                "matcher_type": type(wake_phrase_spotter).__name__,
            }
            self._record_transcript_debug(
                stage=stage,
                outcome="matcher_error",
                capture=capture,
                details=resolved_details,
            )
            self._trace_event(
                "voice_activation_matcher_missing_transcript_path",
                kind="error",
                level="ERROR",
                details={"stage": stage, **resolved_details},
            )
            raise RuntimeError(
                "Wake transcript matching requires a wake phrase matcher with match_transcript()."
            )

        try:
            return match_transcript(normalized_transcript)
        except Exception as exc:
            resolved_details = self._build_error_details(
                exc,
                details=details,
                extra={
                    **self._wake_bias_details(
                        origin_state=origin_state,
                        familiar_speaker_assessment=familiar_speaker_assessment,
                    ),
                    **self._transcript_debug_details(normalized_transcript),
                },
            )
            self._record_transcript_debug(
                stage=stage,
                outcome="matcher_error",
                capture=capture,
                details=resolved_details,
            )
            self._trace_event(
                "voice_activation_matcher_error",
                kind="error",
                level="ERROR",
                details={"stage": stage, **resolved_details},
            )
            raise RuntimeError(
                "Wake transcript matching failed while evaluating the committed utterance."
            ) from exc

    def _backend_request_context(
        self,
        *,
        stage: str,
        capture: AmbientAudioCaptureWindow,
        origin_state: str | None = None,
    ):
        bind_context = getattr(self.backend, "bind_request_context", None)
        if not callable(bind_context):
            return nullcontext()
        return bind_context(
            self._backend_request_context_payload(
                stage=stage,
                capture=capture,
                origin_state=origin_state,
            )
        )

    def _resolve_wake_phrase_spotter(
        self,
        capture: AmbientAudioCaptureWindow,
        *,
        origin_state: str | None,
    ) -> tuple[FamiliarSpeakerWakeAssessment | None, Any]:
        familiar_speaker_assessment = self._assess_familiar_speaker_capture(
            capture,
            origin_state=origin_state,
        )
        wake_phrase_spotter = self._wake_phrase_spotter_for_origin_state(
            origin_state=origin_state,
            allow_contextual_aliases=self._familiar_speaker_wake_bias_active(
                familiar_speaker_assessment,
                origin_state=origin_state,
            ),
        )
        return familiar_speaker_assessment, wake_phrase_spotter

    def _prepare_capture_for_backend_asr(
        self,
        *,
        capture: AmbientAudioCaptureWindow,
        stage: str,
        origin_state: str | None,
    ) -> AmbientAudioCaptureWindow:
        for owner in (self, getattr(self, "backend", None)):
            prepare_capture = None if owner is None else getattr(owner, "sanitize_capture_for_backend_asr", None)
            if not callable(prepare_capture):
                continue
            prepared_capture = self._call_capture_callable(
                prepare_capture,
                capture=capture,
                stage=stage,
                origin_state=origin_state,
            )
            if isinstance(prepared_capture, AmbientAudioCaptureWindow):
                return prepared_capture
        return capture

    def _call_backend_transcription(
        self,
        *,
        capture: AmbientAudioCaptureWindow,
        stage: str,
        origin_state: str | None,
        prompt: str | None,
    ) -> Any:
        route = self._select_transcription_route(
            capture=capture,
            stage=stage,
            origin_state=origin_state,
            prompt=prompt,
        )
        filename = "voice-window.wav"
        content_type = "audio/wav"
        language = getattr(self.config, "openai_realtime_language", None)
        core_kwargs = {
            "filename": filename,
            "content_type": content_type,
            "language": language,
            "prompt": prompt,
        }
        optional_kwargs = self._transcription_optional_kwargs(
            capture=capture,
            origin_state=origin_state,
            route=route,
        )

        if route:
            routed_transcribe = getattr(self.backend, f"transcribe_{route}", None)
            if callable(routed_transcribe):
                return self._call_capture_callable(
                    routed_transcribe,
                    capture=capture,
                    stage=stage,
                    origin_state=origin_state,
                    **core_kwargs,
                    **optional_kwargs,
                )

        transcribe_capture = getattr(self.backend, "transcribe_capture", None)
        if callable(transcribe_capture):
            return self._call_capture_callable(
                transcribe_capture,
                capture=capture,
                stage=stage,
                origin_state=origin_state,
                **core_kwargs,
                **optional_kwargs,
            )

        wav_bytes = pcm_capture_to_wav_bytes(capture)
        return self._call_audio_transcribe_callable(
            self.backend.transcribe,
            wav_bytes,
            filename=filename,
            content_type=content_type,
            language=language,
            prompt=prompt,
            optional_kwargs=optional_kwargs,
        )

    def _select_transcription_route(
        self,
        *,
        capture: AmbientAudioCaptureWindow,
        stage: str,
        origin_state: str | None,
        prompt: str | None,
    ) -> str | None:
        for owner in (self, getattr(self, "backend", None)):
            selector = None if owner is None else getattr(owner, "select_transcription_route", None)
            if not callable(selector):
                continue
            route = self._normalize_route_name(
                self._call_capture_callable(
                    selector,
                    capture=capture,
                    stage=stage,
                    origin_state=origin_state,
                    prompt=prompt,
                )
            )
            if route is not None:
                return route
        return self._normalize_route_name(
            self._config_value(
                "voice_backend_default_route",
                "openai_transcription_route",
                default=None,
            )
        )

    def _transcription_optional_kwargs(
        self,
        *,
        capture: AmbientAudioCaptureWindow,
        origin_state: str | None,
        route: str | None,
    ) -> dict[str, object]:
        optional_kwargs: dict[str, object] = {}

        model_hint = self._config_value(
            "openai_audio_transcription_model",
            "openai_transcription_model",
            "openai_realtime_transcription_model",
            default=None,
        )
        if model_hint:
            optional_kwargs["model"] = model_hint

        response_format = self._config_value(
            "openai_audio_transcription_response_format",
            "openai_transcription_response_format",
            default=None,
        )
        if response_format:
            optional_kwargs["response_format"] = response_format
        elif bool(
            self._config_value(
                "voice_backend_prefer_structured_transcription_response",
                default=False,
            )
        ):
            optional_kwargs["response_format"] = "verbose_json"

        if self._capture_duration_ms(capture) >= 30_000 or bool(
            self._config_value("voice_backend_force_server_vad_chunking", default=False)
        ):
            chunking_strategy = self._config_value(
                "openai_audio_transcription_chunking_strategy",
                "openai_transcription_chunking_strategy",
                default="auto",
            )
            if chunking_strategy:
                optional_kwargs["chunking_strategy"] = chunking_strategy

        if bool(self._config_value("voice_backend_request_word_timestamps", default=False)):
            optional_kwargs["timestamp_granularities"] = ["word", "segment"]

        return optional_kwargs

    def _call_capture_callable(self, func, /, **kwargs) -> Any:
        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            try:
                return func(**kwargs)
            except TypeError:
                return func(kwargs["capture"])

        params = signature.parameters
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
            return func(**kwargs)

        filtered_kwargs = {name: value for name, value in kwargs.items() if name in params}
        if "capture" not in filtered_kwargs and len(params) == 1:
            return func(kwargs["capture"])
        return func(**filtered_kwargs)

    def _call_audio_transcribe_callable(
        self,
        func,
        audio_bytes: bytes,
        *,
        filename: str,
        content_type: str,
        language: str | None,
        prompt: str | None,
        optional_kwargs: Mapping[str, object] | None = None,
    ) -> Any:
        required_kwargs = {
            "filename": filename,
            "content_type": content_type,
            "language": language,
            "prompt": prompt,
        }
        extras = dict(optional_kwargs or {})
        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            try:
                return func(audio_bytes, **required_kwargs, **extras)
            except TypeError:
                return func(audio_bytes, **required_kwargs)

        params = signature.parameters
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
            return func(audio_bytes, **required_kwargs, **extras)

        filtered_extras = {name: value for name, value in extras.items() if name in params}
        return func(audio_bytes, **required_kwargs, **filtered_extras)

    def _backend_request_context_payload(
        self,
        *,
        stage: str,
        capture: AmbientAudioCaptureWindow,
        origin_state: str | None,
    ) -> dict[str, object]:
        pcm_bytes = self._capture_pcm_bytes(capture)
        sample = getattr(capture, "sample", None)
        signal_profile = pcm16_signal_profile(pcm_bytes)
        payload: dict[str, object] = {
            "session_id": getattr(self, "_session_id", None),
            "trace_id": getattr(self, "_trace_id", None),
            "stage": stage,
            "state": getattr(self, "_state", None),
            "origin_state": origin_state,
            "capture_duration_ms": self._capture_duration_ms(capture),
            "capture_bytes": len(pcm_bytes),
            "capture_sample_rate": int(getattr(capture, "sample_rate", 0) or 0),
            "capture_channels": int(getattr(capture, "channels", 0) or 0),
            "capture_average_rms": int(getattr(sample, "average_rms", 0) or 0),
            "capture_peak_rms": int(getattr(sample, "peak_rms", 0) or 0),
            "capture_active_ratio": round(float(getattr(sample, "active_ratio", 0.0) or 0.0), 6),
            "capture_sha256_prefix": hashlib.sha256(pcm_bytes).hexdigest()[:16] if pcm_bytes else None,
            "transcription_route_hint": self._normalize_route_name(
                self._config_value(
                    "voice_backend_default_route",
                    "openai_transcription_route",
                    default=None,
                )
            ),
            "privacy_mode": self._config_value("voice_backend_privacy_mode", default="standard"),
            **prefixed_signal_profile_details(signal_profile, prefix="capture_signal"),
        }
        return {key: value for key, value in payload.items() if value is not None}

    def _capture_rejection_details(
        self,
        capture: AmbientAudioCaptureWindow,
    ) -> dict[str, object] | None:
        pcm_bytes = self._capture_pcm_bytes(capture)
        if not pcm_bytes:
            return {"capture_bytes": 0, "reason": "empty_capture"}

        duration_ms = self._capture_duration_ms(capture)
        min_capture_ms = self._config_int_value(
            "voice_backend_min_capture_duration_ms",
            "voice_capture_min_duration_ms",
            default=self._MIN_CAPTURE_MS,
        )
        if duration_ms > 0 and min_capture_ms > 0 and duration_ms < min_capture_ms:
            return {
                "reason": "capture_too_short",
                "capture_duration_ms": duration_ms,
                "min_capture_duration_ms": min_capture_ms,
            }

        max_capture_ms = self._config_int_value(
            "voice_backend_max_capture_duration_ms",
            "voice_capture_max_duration_ms",
            default=self._MAX_CAPTURE_MS,
        )
        if duration_ms > 0 and max_capture_ms > 0 and duration_ms > max_capture_ms:
            return {
                "reason": "capture_too_long",
                "capture_duration_ms": duration_ms,
                "max_capture_duration_ms": max_capture_ms,
            }

        max_capture_bytes = self._config_int_value(
            "voice_backend_max_capture_bytes",
            "voice_capture_max_backend_bytes",
            default=self._MAX_CAPTURE_BYTES,
        )
        if max_capture_bytes > 0 and len(pcm_bytes) > max_capture_bytes:
            return {
                "reason": "capture_too_large",
                "capture_bytes": len(pcm_bytes),
                "max_capture_bytes": max_capture_bytes,
            }

        return None

    def _record_capture_rejection(
        self,
        *,
        stage: str,
        capture: AmbientAudioCaptureWindow,
        details: Mapping[str, object] | None,
        rejection_details: Mapping[str, object],
        mode: str,
    ) -> None:
        resolved_details = {
            **self._sanitize_details(details),
            **self._sanitize_details(rejection_details),
            "mode": mode,
        }
        self._record_transcript_debug(
            stage=stage,
            outcome=str(rejection_details.get("reason", "capture_rejected")),
            capture=capture,
            details=resolved_details,
        )
        self._trace_event(
            "voice_capture_rejected",
            kind="warning",
            level="WARN",
            details={"stage": stage, **resolved_details},
        )

    def _extract_transcription_text(self, response: Any) -> str | None:
        if response is None:
            return None
        if isinstance(response, bytes):
            return self._normalize_transcript_text(response.decode("utf-8", "replace"))
        if isinstance(response, str):
            return self._normalize_transcript_text(response)

        mapping_response: Mapping[str, Any] | None = None
        if isinstance(response, Mapping):
            mapping_response = response
        else:
            for method_name in ("model_dump", "dict"):
                dump_method = getattr(response, method_name, None)
                if callable(dump_method):
                    dumped = dump_method()
                    if isinstance(dumped, Mapping):
                        mapping_response = dumped
                        break

        if mapping_response is not None:
            for key in ("text", "transcript", "output_text"):
                transcript = self._extract_transcription_text(mapping_response.get(key))
                if transcript is not None:
                    return transcript
            for key in ("output", "content", "segments", "results", "alternatives", "choices", "items"):
                transcript = self._extract_output_text_container(mapping_response.get(key))
                if transcript is not None:
                    return transcript

        for attr_name in ("text", "transcript", "output_text"):
            transcript = self._extract_transcription_text(getattr(response, attr_name, None))
            if transcript is not None:
                return transcript

        if isinstance(response, (list, tuple)):
            return self._extract_output_text_container(response)
        return None

    def _extract_output_text_container(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, (str, bytes)):
            return self._extract_transcription_text(value)
        if isinstance(value, Mapping):
            for key in ("text", "transcript", "output_text"):
                transcript = self._extract_transcription_text(value.get(key))
                if transcript is not None:
                    return transcript
            for key in ("content", "segments", "results", "alternatives", "choices", "items"):
                transcript = self._extract_output_text_container(value.get(key))
                if transcript is not None:
                    return transcript
            return None
        if isinstance(value, (list, tuple)):
            parts: list[str] = []
            for item in value:
                part = self._extract_output_text_container(item)
                if part:
                    parts.append(part)
            return self._normalize_transcript_text(" ".join(parts)) if parts else None
        return None

    def _normalize_transcript_text(self, transcript: Any) -> str | None:
        if transcript is None:
            return None
        text = unicodedata.normalize("NFKC", str(transcript))
        text = _ZERO_WIDTH_CHARS_RE.sub("", text)
        text = _CONTROL_CHARS_RE.sub(" ", text)
        text = _WHITESPACE_RE.sub(" ", text).strip()
        if not text:
            return None
        max_chars = self._config_int_value(
            "voice_backend_max_transcript_chars",
            default=self._MAX_TRANSCRIPT_CHARS,
        )
        return text[:max_chars] if max_chars > 0 else text

    def _normalize_prompt_text(self, prompt: str | None) -> str | None:
        if prompt is None:
            return None
        return _ZERO_WIDTH_CHARS_RE.sub(
            "",
            unicodedata.normalize("NFKC", str(prompt)),
        ).strip() or None

    def _transcript_debug_details(self, transcript: Any) -> dict[str, object]:
        normalized_transcript = self._normalize_transcript_text(transcript)
        if normalized_transcript is None:
            return {"transcript_chars": 0}
        details: dict[str, object] = {
            "transcript_chars": len(normalized_transcript),
            "transcript_sha256_prefix": hashlib.sha256(
                normalized_transcript.encode("utf-8")
            ).hexdigest()[:16],
        }
        if bool(self._config_value("voice_debug_log_plaintext_transcripts", default=False)):
            details["transcript_preview"] = normalized_transcript[: self._DEBUG_PREVIEW_CHARS]
        return details

    def _build_error_details(
        self,
        exc: Exception,
        *,
        details: Mapping[str, object] | None = None,
        extra: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        resolved_details = self._sanitize_details(details)
        if extra:
            resolved_details.update(self._sanitize_details(extra))
        resolved_details["error_type"] = type(exc).__name__
        error_message = self._sanitize_text(str(exc), max_chars=self._MAX_ERROR_CHARS)
        if error_message:
            resolved_details["error_message"] = error_message
        return resolved_details

    def _sanitize_details(self, details: Mapping[str, object] | None) -> dict[str, object]:
        if not details:
            return {}
        return {str(key): self._sanitize_detail_value(value) for key, value in details.items()}

    def _sanitize_detail_value(self, value: object) -> object:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, bytes):
            return f"<{len(value)} bytes>"
        if isinstance(value, str):
            return self._sanitize_text(value, max_chars=256)
        if isinstance(value, Mapping):
            return {
                self._sanitize_text(key, max_chars=64): self._sanitize_detail_value(item)
                for key, item in list(value.items())[:16]
            }
        if isinstance(value, (list, tuple, set)):
            return [self._sanitize_detail_value(item) for item in list(value)[:16]]
        return self._sanitize_text(repr(value), max_chars=256)

    def _sanitize_text(self, value: object, *, max_chars: int) -> str:
        text = _ZERO_WIDTH_CHARS_RE.sub("", str(value))
        text = _CONTROL_CHARS_RE.sub(" ", text)
        text = _WHITESPACE_RE.sub(" ", text).strip()
        return text[:max_chars] if max_chars > 0 else text

    def _capture_pcm_bytes(self, capture: AmbientAudioCaptureWindow) -> bytes:
        return bytes(getattr(capture, "pcm_bytes", b"") or b"")

    def _capture_duration_ms(self, capture: AmbientAudioCaptureWindow) -> int:
        sample = getattr(capture, "sample", None)
        try:
            duration_ms = int(getattr(sample, "duration_ms", 0) or 0)
        except (TypeError, ValueError):
            duration_ms = 0
        if duration_ms > 0:
            return duration_ms

        pcm_bytes = self._capture_pcm_bytes(capture)
        sample_rate = int(getattr(capture, "sample_rate", 0) or 0)
        channels = int(getattr(capture, "channels", 0) or 0)
        if not pcm_bytes or sample_rate <= 0 or channels <= 0:
            return 0
        return int((len(pcm_bytes) / (sample_rate * channels * 2)) * 1000)

    def _normalize_origin_state(self, origin_state: object | None) -> str | None:
        normalized = str(origin_state or "").strip().lower()
        return normalized or None

    def _normalize_route_name(self, route: object | None) -> str | None:
        normalized = str(route or "").strip().lower().replace("-", "_").replace(" ", "_")
        return normalized or None

    def _config_value(self, *names: str, default: object = None) -> object:
        config = getattr(self, "config", None)
        if config is None:
            return default
        for name in names:
            if hasattr(config, name):
                value = getattr(config, name)
                if value is not None:
                    return value
        return default

    def _config_int_value(self, *names: str, default: int) -> int:
        value = self._config_value(*names, default=default)
        try:
            if isinstance(value, (bool, int, float, str, bytes, bytearray)):
                return int(value)
        except (TypeError, ValueError):
            pass
        return int(default)
