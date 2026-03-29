# CHANGELOG: 2026-03-29
# BUG-1: Observability sinks are now fail-open; tracer/debug-store failures no longer break the live voice session.
# BUG-2: `_wake_phrase_spotter_for_origin_state()` now honors the origin state, preventing contextual alias expansion outside the waiting state.
# BUG-3: `set_forensics()` now clears stale trace IDs between sessions instead of silently reusing an old trace.
# SEC-1: Transcript/debug payloads are sanitized, control-character-normalized, and size-bounded to reduce log-injection and memory/disk exhaustion risk.
# SEC-2: Raw audio persistence is now policy-gated and size-limited.  # BREAKING: by default audio captures persist only when forensics is explicitly enabled or policy opts in.
# IMP-1: Trace/debug payloads are now deep-copied, JSON-safe, datetime-safe, and conflict-safe for modern OTel-style exporters and processors.
# IMP-2: Transcript records now include monotonic ordering metadata and cached signal-profile reuse to improve correlation and reduce edge-device overhead.

"""Tracing and transcript-debug helpers for the orchestrator voice session."""

from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from datetime import date, datetime, timezone
from typing import Any

from twinr.agent.workflows.forensics import WorkflowForensics
from twinr.hardware.audio import AmbientAudioCaptureWindow, pcm16_signal_profile
from twinr.orchestrator.voice_activation import VoiceActivationPhraseMatcher
from twinr.orchestrator.voice_familiarity import FamiliarSpeakerWakeAssessment
from twinr.orchestrator.voice_forensics import prefixed_signal_profile_details

_LOGGER = logging.getLogger(__name__)

_DEFAULT_OBSERVABILITY_CONFIG: dict[str, object] = {
    "include_monotonic_ns": True,
    "max_depth": 5,
    "max_mapping_items": 64,
    "max_sequence_items": 64,
    "max_text_chars": 4096,
    "max_capture_bytes": 1_048_576,
    "max_capture_duration_ms": 15_000,
    "persist_audio_captures": None,
    "record_raw_transcripts": True,
    "record_raw_matched_phrase": True,
    "record_raw_remaining_text": True,
    "normalize_control_chars": True,
}

_TRUNCATION_SENTINEL = "…<truncated>"
_TRANSCRIPT_DEBUG_PRIORITY_KEYS = (
    "error_type",
    "error_message",
    "required_active_ms",
    "audio_artifact_path",
    "audio_artifact_manifest_path",
    "audio_artifact_id",
    "audio_artifact_duration_ms",
    "audio_artifact_bytes",
    "audio_artifact_metadata_path",
    "audio_store_error",
    "audio_store_error_type",
    "audio_capture_persist_allowed",
    "audio_capture_persist_reason",
    "origin_state",
    "pending_active_ms",
    "pending_captured_ms",
    "pending_trailing_silence_ms",
    "trace_id",
    "trace.id",
    "observed_at_utc",
    "telemetry.sequence",
    "telemetry.monotonic_ns",
)


class VoiceSessionObservabilityMixin:
    """Own the trace/debug surfaces shared by the voice session helpers."""

    def set_forensics(
        self,
        tracer: WorkflowForensics | None,
        *,
        trace_id: str | None = None,
    ) -> None:
        """Bind one shared forensic tracer for this websocket voice session."""

        self._forensics = tracer if isinstance(tracer, WorkflowForensics) and tracer.enabled else None
        # BUGFIX: clear stale trace IDs when a session is rebound without an explicit replacement.
        self._trace_id = (
            self._observability_sanitize_text(trace_id, limit=128)
            if trace_id is not None
            else None
        )
        configure_backend_forensics = getattr(self.backend, "set_forensics", None)
        if callable(configure_backend_forensics):
            self._safe_observability_call(
                "backend.set_forensics",
                configure_backend_forensics,
                self._forensics,
            )

    def _trace_details(self, details: dict[str, object] | None = None) -> dict[str, object]:
        session_id = (
            self._observability_sanitize_text(self._session_id, limit=128)
            if self._session_id
            else None
        )
        state = self._observability_sanitize_text(self._state, limit=128)
        backend_name = self._observability_sanitize_text(self.backend_name, limit=128)
        quiet_active = self._voice_quiet_active()
        payload: dict[str, object] = {
            "session_id": session_id,
            "session.id": session_id,
            "trace_id": self._trace_id or None,
            "trace.id": self._trace_id or None,
            "state": state,
            "voice.state": state,
            "backend": backend_name,
            "voice.backend": backend_name,
            "follow_up_allowed": self._follow_up_allowed,
            "voice.follow_up_allowed": self._follow_up_allowed,
            "voice_quiet_active": quiet_active,
            "voice.quiet.active": quiet_active,
            "voice_quiet_until_utc": self._normalize_observability_value(
                self._voice_quiet_until_utc,
                depth=0,
                path="voice_quiet_until_utc",
            ),
            "voice.quiet.until_utc": self._normalize_observability_value(
                self._voice_quiet_until_utc,
                depth=0,
                path="voice.quiet.until_utc",
            ),
            "observed_at_utc": self._utc_now_iso(),
            "telemetry.sequence": self._next_observability_sequence(),
        }
        if self._config_bool("include_monotonic_ns", True):
            payload["telemetry.monotonic_ns"] = time.monotonic_ns()
        self._merge_trace_payload(
            payload,
            self._safe_intent_trace_details(),
            conflict_prefix="intent",
        )
        self._merge_trace_payload(payload, details, conflict_prefix="detail")
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

        if allow_contextual_aliases and self._strong_speaker_wake_bias_active(
            origin_state=origin_state
        ):
            return getattr(self, "_strong_bias_wake_phrase_spotter", self._wake_phrase_spotter)
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
                else self._normalize_mapping_or_wrap(
                    familiar_speaker_assessment.trace_details(),
                    field_name="speaker_assessment",
                )
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
        self._safe_observability_call(
            "forensics.event",
            tracer.event,
            kind=self._observability_sanitize_identifier(kind, default="event", limit=64),
            msg=self._observability_sanitize_text(msg, limit=1024),
            details=self._trace_details(details),
            reason=self._normalize_mapping_or_wrap(reason, field_name="reason"),
            kpi=self._normalize_mapping_or_wrap(kpi, field_name="kpi"),
            level=self._observability_sanitize_identifier(level, default="INFO", limit=16).upper(),
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
        self._safe_observability_call(
            "forensics.decision",
            tracer.decision,
            msg=self._observability_sanitize_text(msg, limit=1024),
            question=self._observability_sanitize_text(question, limit=1024),
            selected=self._normalize_mapping_or_wrap(selected, field_name="selected"),
            options=self._normalize_sequence(options, path="options"),
            context=self._trace_details(context),
            confidence=self._normalize_observability_value(confidence, depth=0, path="confidence"),
            guardrails=self._normalize_sequence(guardrails, path="guardrails"),
            kpi_impact_estimate=self._normalize_mapping_or_wrap(
                kpi_impact_estimate,
                field_name="kpi_impact_estimate",
            ),
            trace_id=self._trace_id,
        )

    def _trace_span(self, *, name: str, kind: str, details: dict[str, object] | None = None):
        tracer = self._forensics
        if not isinstance(tracer, WorkflowForensics):
            return nullcontext()
        return self._safe_observability_call(
            "forensics.span",
            tracer.span,
            name=self._observability_sanitize_text(name, limit=256),
            kind=self._observability_sanitize_identifier(kind, default="span", limit=64),
            details=self._trace_details(details),
            trace_id=self._trace_id,
            fallback=nullcontext(),
        )

    def _capture_sample_details(
        self,
        capture: AmbientAudioCaptureWindow | None,
    ) -> dict[str, object]:
        """Return compact capture metrics for transcript debug entries."""

        if capture is None:
            return {}
        try:
            sample = capture.sample
            signal_profile = getattr(capture, "signal_profile", None)
            if signal_profile is None:
                signal_profile = getattr(sample, "signal_profile", None)
            if signal_profile is None:
                signal_profile = pcm16_signal_profile(capture.pcm_bytes)
            payload: dict[str, object] = {
                "duration_ms": int(sample.duration_ms),
                "chunk_count": int(sample.chunk_count),
                "active_chunk_count": int(sample.active_chunk_count),
                "average_rms": int(sample.average_rms),
                "peak_rms": int(sample.peak_rms),
                "active_ratio": round(float(sample.active_ratio), 6),
                "pcm_byte_count": len(capture.pcm_bytes),
                **prefixed_signal_profile_details(signal_profile, prefix="signal"),
            }
            sample_rate_hz = getattr(sample, "sample_rate_hz", None)
            if sample_rate_hz is not None:
                payload["sample_rate_hz"] = int(sample_rate_hz)
            return self._normalize_mapping_or_wrap(payload, field_name="sample")
        except Exception as exc:  # pragma: no cover - defensive fail-open path
            self._log_observability_failure("capture.sample_details", exc)
            return {
                "capture_profile_error": exc.__class__.__name__,
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

        resolved_details = self._safe_intent_trace_details()
        resolved_details.setdefault("observed_at_utc", self._utc_now_iso())
        resolved_details.setdefault("telemetry.sequence", self._next_observability_sequence())
        resolved_details.setdefault("trace_id", self._trace_id or None)
        resolved_details.setdefault("trace.id", self._trace_id or None)
        if self._config_bool("include_monotonic_ns", True):
            resolved_details.setdefault("telemetry.monotonic_ns", time.monotonic_ns())
        self._merge_trace_payload(resolved_details, details, conflict_prefix="detail")

        audio_artifact: dict[str, object] | None = None
        persist_capture, persist_reason = self._audio_capture_persistence_decision(capture)
        resolved_details["audio_capture_persist_allowed"] = persist_capture
        resolved_details["audio_capture_persist_reason"] = persist_reason
        if capture is not None and persist_capture:
            persist_audio = getattr(self._audio_debug_store, "persist_capture", None)
            if callable(persist_audio):
                audio_artifact = self._safe_observability_call(
                    "audio_debug_store.persist_capture",
                    persist_audio,
                    capture=capture,
                    session_id=self._session_id or None,
                    trace_id=self._trace_id,
                    stage=self._observability_sanitize_identifier(stage, default="stage", limit=64),
                    outcome=self._observability_sanitize_identifier(outcome, default="outcome", limit=64),
                    fallback=None,
                )
            if audio_artifact:
                self._merge_trace_payload(
                    resolved_details,
                    audio_artifact,
                    conflict_prefix="audio",
                )
        resolved_details = self._prioritize_transcript_debug_details(resolved_details)

        append_entry = getattr(self._transcript_debug_stream, "append_entry", None)
        if not callable(append_entry):
            return
        self._safe_observability_call(
            "transcript_debug_stream.append_entry",
            append_entry,
            session_id=self._session_id or None,
            trace_id=self._trace_id,
            state=self._state,
            backend=self.backend_name,
            stage=self._observability_sanitize_text(stage, limit=128),
            outcome=self._observability_sanitize_text(outcome, limit=128),
            transcript=self._observable_text_value(
                transcript,
                field_name="transcript",
                raw_allowed_key="record_raw_transcripts",
            ),
            matched_phrase=self._observable_text_value(
                matched_phrase,
                field_name="matched_phrase",
                raw_allowed_key="record_raw_matched_phrase",
            ),
            remaining_text=self._observable_text_value(
                remaining_text,
                field_name="remaining_text",
                raw_allowed_key="record_raw_remaining_text",
            ),
            detector_label=self._observability_sanitize_text(detector_label, limit=256),
            score=self._normalize_numeric(score),
            sample=self._capture_sample_details(capture),
            details=resolved_details,
        )

    def _prioritize_transcript_debug_details(
        self,
        details: dict[str, object],
    ) -> dict[str, object]:
        """Keep high-signal debug keys ahead of low-priority intent context."""

        prioritized: dict[str, object] = {}
        for key in _TRANSCRIPT_DEBUG_PRIORITY_KEYS:
            if key in details:
                prioritized[key] = details[key]
        for key, value in details.items():
            if key not in prioritized:
                prioritized[key] = value
        return prioritized

    def _audio_capture_persistence_decision(
        self,
        capture: AmbientAudioCaptureWindow | None,
    ) -> tuple[bool, str]:
        if capture is None:
            return False, "no_capture"
        explicit_policy = self._resolved_observability_config().get("persist_audio_captures")
        if explicit_policy is None:
            audio_debug_store = getattr(self, "_audio_debug_store", None)
            audio_debug_config = getattr(audio_debug_store, "config", None)
            if bool(getattr(audio_debug_config, "enabled", False)):
                explicit_policy = True
        if explicit_policy is None:
            # BREAKING: safer default for production deployments; raw audio is only persisted
            # when forensic tracing is explicitly enabled unless a policy overrides it.
            explicit_policy = bool(isinstance(self._forensics, WorkflowForensics) and self._forensics.enabled)
        if not bool(explicit_policy):
            return False, "policy_disabled"
        pcm_bytes = getattr(capture, "pcm_bytes", b"")
        if len(pcm_bytes) > self._config_int("max_capture_bytes", 1_048_576):
            return False, "capture_too_large"
        sample = getattr(capture, "sample", None)
        duration_ms = getattr(sample, "duration_ms", None)
        if isinstance(duration_ms, (int, float)) and duration_ms > self._config_int(
            "max_capture_duration_ms",
            15_000,
        ):
            return False, "capture_too_long"
        return True, "enabled"

    def _safe_intent_trace_details(self) -> dict[str, object]:
        intent_context = getattr(self, "_intent_context", None)
        trace_details = getattr(intent_context, "trace_details", None)
        if not callable(trace_details):
            return {}
        result = self._safe_observability_call(
            "intent_context.trace_details",
            trace_details,
            fallback={},
        )
        return self._normalize_mapping_or_wrap(result, field_name="intent_context")

    def _resolved_observability_config(self) -> dict[str, object]:
        config = dict(_DEFAULT_OBSERVABILITY_CONFIG)
        for candidate in self._iter_observability_config_candidates():
            if isinstance(candidate, Mapping):
                config.update(candidate)
        return config

    def _iter_observability_config_candidates(self) -> list[Mapping[str, object]]:
        candidates: list[Mapping[str, object]] = []
        for owner in (self, getattr(self, "backend", None)):
            if owner is None:
                continue
            for attr_name in ("observability_config", "_observability_config"):
                candidate = getattr(owner, attr_name, None)
                if isinstance(candidate, Mapping):
                    candidates.append(candidate)
            for method_name in ("get_observability_config", "_get_observability_config"):
                resolver = getattr(owner, method_name, None)
                if callable(resolver):
                    resolved = self._safe_observability_call(
                        f"{owner.__class__.__name__}.{method_name}",
                        resolver,
                        fallback=None,
                    )
                    if isinstance(resolved, Mapping):
                        candidates.append(resolved)
        return candidates

    def _config_bool(self, key: str, default: bool) -> bool:
        value = self._resolved_observability_config().get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return bool(value)

    def _config_int(self, key: str, default: int) -> int:
        value = self._resolved_observability_config().get(key, default)
        if isinstance(value, bool):
            return default
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return default

    def _utc_now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _next_observability_sequence(self) -> int:
        lock = getattr(self, "_observability_sequence_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._observability_sequence_lock = lock
        with lock:
            current = int(getattr(self, "_observability_sequence", 0)) + 1
            self._observability_sequence = current
            return current

    def _safe_observability_call(
        self,
        label: str,
        func,
        *args,
        fallback: Any = None,
        **kwargs,
    ) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive fail-open path
            self._log_observability_failure(label, exc)
            return fallback

    def _log_observability_failure(self, label: str, exc: Exception) -> None:
        _LOGGER.warning(
            "voice-session observability failure in %s [session=%s trace=%s]: %s: %s",
            label,
            getattr(self, "_session_id", None),
            getattr(self, "_trace_id", None),
            exc.__class__.__name__,
            exc,
            exc_info=True,
        )

    def _normalize_mapping_or_wrap(
        self,
        value: object | None,
        *,
        field_name: str,
    ) -> dict[str, object] | None:
        if value is None:
            return None
        normalized = self._normalize_observability_value(value, depth=0, path=field_name)
        if isinstance(normalized, dict):
            return normalized
        return {field_name: normalized}

    def _normalize_sequence(self, value: object | None, *, path: str) -> list[object] | None:
        if value is None:
            return None
        normalized = self._normalize_observability_value(value, depth=0, path=path)
        if isinstance(normalized, list):
            return normalized
        return [normalized]

    def _normalize_numeric(self, value: object | None) -> float | int | None | str:
        normalized = self._normalize_observability_value(value, depth=0, path="numeric")
        if normalized is None:
            return None
        if isinstance(normalized, (int, float, str)):
            return normalized
        return self._observability_sanitize_text(normalized, limit=64)

    def _normalize_observability_value(
        self,
        value: object,
        *,
        depth: int,
        path: str,
    ) -> object:
        max_depth = self._config_int("max_depth", 5)
        if depth >= max_depth:
            return self._observability_truncate_repr(value)
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return str(value)
            return value
        if isinstance(value, str):
            return self._observability_sanitize_text(
                value,
                limit=self._config_int("max_text_chars", 4096),
            )
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, bytes):
            return {"type": "bytes", "length": len(value)}
        if isinstance(value, bytearray):
            return {"type": "bytearray", "length": len(value)}
        if isinstance(value, Mapping):
            output: dict[str, object] = {}
            max_items = self._config_int("max_mapping_items", 64)
            for index, (raw_key, raw_item) in enumerate(value.items()):
                if index >= max_items:
                    output["_truncated_items"] = index
                    break
                key = self._observability_sanitize_key(raw_key)
                output[key] = self._normalize_observability_value(
                    raw_item,
                    depth=depth + 1,
                    path=f"{path}.{key}" if path else key,
                )
            return output
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            output_list: list[object] = []
            max_items = self._config_int("max_sequence_items", 64)
            for index, item in enumerate(value):
                if index >= max_items:
                    output_list.append({"_truncated_items": index})
                    break
                output_list.append(
                    self._normalize_observability_value(
                        item,
                        depth=depth + 1,
                        path=f"{path}[{index}]",
                    )
                )
            return output_list
        trace_details = getattr(value, "trace_details", None)
        if callable(trace_details):
            nested_details = self._safe_observability_call(
                f"{value.__class__.__name__}.trace_details",
                trace_details,
                fallback=None,
            )
            if nested_details is not None:
                return self._normalize_observability_value(
                    nested_details,
                    depth=depth + 1,
                    path=f"{path}.trace_details" if path else "trace_details",
                )
        return self._observability_truncate_repr(value)

    def _merge_trace_payload(
        self,
        target: dict[str, object],
        extra: object | None,
        *,
        conflict_prefix: str,
    ) -> None:
        if extra is None:
            return
        normalized = self._normalize_observability_value(extra, depth=0, path=conflict_prefix)
        if not isinstance(normalized, dict):
            target[f"{conflict_prefix}_payload"] = normalized
            return
        for raw_key, raw_value in normalized.items():
            key = self._observability_sanitize_key(raw_key)
            candidate = key
            suffix = 1
            while candidate in target:
                candidate = f"{conflict_prefix}_{key}_{suffix}"
                suffix += 1
            target[candidate] = raw_value

    def _observable_text_value(
        self,
        value: str | None,
        *,
        field_name: str,
        raw_allowed_key: str,
    ) -> str | None:
        if value is None:
            return None
        text = self._observability_sanitize_text(
            value,
            limit=self._config_int("max_text_chars", 4096),
        )
        if text is None:
            return None
        redactor = getattr(self, "redact_observability_text", None)
        if callable(redactor):
            redacted = self._safe_observability_call(
                "redact_observability_text",
                redactor,
                text,
                field_name=field_name,
                fallback=None,
            )
            if redacted is None:
                redacted = self._safe_observability_call(
                    "redact_observability_text",
                    redactor,
                    text,
                    field_name,
                    fallback=text,
                )
            if isinstance(redacted, str):
                text = self._observability_sanitize_text(
                    redacted,
                    limit=self._config_int("max_text_chars", 4096),
                )
        if self._config_bool(raw_allowed_key, True):
            return text
        return (
            f"[redacted:{field_name}:chars={len(text)}:fp={self._text_fingerprint(text)}]"
        )

    def _text_fingerprint(self, text: str) -> str:
        return hashlib.blake2s(text.encode("utf-8"), digest_size=6).hexdigest()

    def _observability_sanitize_text(self, value: object | None, *, limit: int) -> str | None:
        if value is None:
            return None
        text = str(value)
        if self._config_bool("normalize_control_chars", True):
            text = text.replace("\x00", " ")
            text = text.replace("\r", " ")
            text = text.replace("\n", " ")
            text = "".join(ch if ch.isprintable() else " " for ch in text)
            text = " ".join(text.split())
        if len(text) > limit:
            available = max(0, limit - len(_TRUNCATION_SENTINEL))
            text = f"{text[:available]}{_TRUNCATION_SENTINEL}"
        return text

    def _observability_sanitize_key(self, value: object) -> str:
        key = self._observability_sanitize_text(value, limit=128) or "field"
        return key.replace(" ", "_")

    def _observability_sanitize_identifier(
        self,
        value: object | None,
        *,
        default: str,
        limit: int,
    ) -> str:
        text = self._observability_sanitize_text(value, limit=limit) or default
        normalized = []
        for char in text:
            if char.isalnum() or char in {"-", "_", "."}:
                normalized.append(char)
            elif char.isspace():
                normalized.append("_")
            else:
                normalized.append("_")
        cleaned = "".join(normalized).strip("._-")
        return cleaned or default

    def _observability_truncate_repr(self, value: object) -> str:
        return self._observability_sanitize_text(
            repr(value),
            limit=self._config_int("max_text_chars", 4096),
        ) or value.__class__.__name__
