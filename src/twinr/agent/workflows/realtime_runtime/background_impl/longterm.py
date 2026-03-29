# CHANGELOG: 2026-03-27
# BUG-1: Safety-critical long-term candidates were treated like ordinary ones in initiative/governor gating, so urgent reminders could be suppressed by low-confidence initiative state or presence budgets.
# BUG-2: Reservation mismatches and required-remote failure paths could leak candidate reservations, causing silent starvation of later proactive prompts.
# BUG-3: Exceptions during nightly reserve maintenance could escape the background loop and stop future maintenance runs.
# BUG-4: Remote-memory failures while building observation facts were swallowed by the phrasing fallback path instead of honoring fail-closed remote error handling.
# SEC-1: Raw prompt/summary/rationale/error strings were emitted into telemetry, leaking sensitive household data and allowing control-character/log-injection from memory or sensor-derived strings.
# IMP-1: Poll scheduling and cache TTLs now use monotonic_ns-compatible helpers for better long-uptime behavior on always-on Raspberry Pi deployments.
# IMP-2: Observation facts are now bounded, deduplicated, uncertainty-aware, and backed by a small TTL cache for source-memory attributes to reduce prompt noise and remote/object-store latency.
# IMP-3: Reservation release paths, telemetry sanitization, and error reporting are unified for more robust background-loop behavior.

"""Long-term proactive delivery helpers for the realtime background loop."""

# mypy: ignore-errors

from __future__ import annotations

from datetime import datetime
import hashlib
import re
import time
from typing import Any

from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError
from twinr.proactive.governance.governor import ProactiveGovernorCandidate, ProactiveGovernorReservation
from twinr.proactive.runtime.ambiguous_room_guard import (
    AmbiguousRoomGuardSnapshot,
    ambiguous_room_guard_requires_hard_block,
    derive_ambiguous_room_guard,
)
from twinr.proactive.runtime.multimodal_initiative import ReSpeakerMultimodalInitiativeSnapshot
from twinr.proactive.runtime.sensitive_behavior_gate import evaluate_respeaker_sensitive_behavior_gate


_MONOTONIC_NS_FALLBACK_THRESHOLD = 1_000_000_000_000
_DEFAULT_LONGTERM_POLL_INTERVAL_S = 30.0
_DEFAULT_DISPLAY_RESERVE_NIGHTLY_POLL_INTERVAL_S = 300.0
_DEFAULT_SOURCE_ATTR_CACHE_TTL_S = 300.0
_MAX_SOURCE_ATTR_CACHE_ITEMS = 256
_MAX_SOURCE_MEMORY_IDS = 4
_MAX_OBSERVATION_FACTS = 32
_MAX_OBSERVATION_FACT_CHARS = 160
_TELEMETRY_PREVIEW_CHARS = 160
_ERROR_PREVIEW_CHARS = 240
_MAX_SPOKEN_PROMPT_CHARS = 320
_SAFETY_EXEMPT_SENSITIVITIES = frozenset({"safety", "critical", "emergency", "urgent", "medical_urgent"})
_SAFETY_EXEMPT_KIND_PREFIXES = (
    "safety_",
    "critical_",
    "emergency_",
    "urgent_",
    "fall_",
)
_DISPLAY_CHANNEL = "display"
_TRUTHY = {"1", "true", "yes", "on"}
_FALSEY = {"0", "false", "no", "off"}


class BackgroundLongTermMixin:
    """Own long-term proactive gating, phrasing, and reserve maintenance."""

    def _longterm_config_bool(self, name: str, default: bool = False) -> bool:
        cfg = getattr(self, "config", None)
        value = default
        if isinstance(cfg, dict):
            value = cfg.get(name, default)
        elif cfg is not None and hasattr(cfg, name):
            value = getattr(cfg, name)
        if isinstance(value, bool):
            return value
        text = self._coerce_text(value)
        if text is None:
            return default
        lowered = text.strip().lower()
        if lowered in _TRUTHY:
            return True
        if lowered in _FALSEY:
            return False
        return bool(value)

    def _longterm_config_int(self, name: str, default: int) -> int:
        cfg = getattr(self, "config", None)
        value: Any = default
        if isinstance(cfg, dict):
            value = cfg.get(name, default)
        elif cfg is not None and hasattr(cfg, name):
            value = getattr(cfg, name)
        try:
            return int(value)
        except Exception:
            return default

    def _longterm_interval_to_ns(self, seconds: float) -> int:
        try:
            return max(0, int(float(seconds) * 1_000_000_000))
        except Exception:
            return 0

    def _longterm_deadline_due(self, attr_name: str, interval_s: float) -> bool:
        now_ns = time.monotonic_ns()
        raw_deadline = getattr(self, attr_name, 0)
        due = True
        if isinstance(raw_deadline, int) and raw_deadline >= _MONOTONIC_NS_FALLBACK_THRESHOLD:
            due = now_ns >= raw_deadline
        else:
            try:
                due = time.monotonic() >= float(raw_deadline)
            except Exception:
                due = True
        if not due:
            return False
        setattr(self, attr_name, now_ns + self._longterm_interval_to_ns(interval_s))
        return True

    def _longterm_normalize_telemetry_text(self, value: object, *, limit: int = _TELEMETRY_PREVIEW_CHARS) -> str:
        text = self._coerce_text(value)
        if not text:
            return ""
        text = text.replace("\r", " ").replace("\n", " ")
        text = "".join(ch if ch.isprintable() else " " for ch in text)
        text = re.sub(r"\s+", " ", text).strip()
        if limit > 0 and len(text) > limit:
            text = text[: max(0, limit - 1)].rstrip() + "…"
        return text

    def _longterm_exception_summary(self, exc: Exception) -> str:
        return self._longterm_normalize_telemetry_text(
            f"{type(exc).__name__}: {exc}",
            limit=_ERROR_PREVIEW_CHARS,
        ) or type(exc).__name__

    def _longterm_text_fingerprint(self, value: object) -> str | None:
        text = self._coerce_text(value)
        if not text:
            return None
        return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]

    # BREAKING: long-term telemetry no longer emits raw prompt/summary/rationale text by default.
    # Set allow_sensitive_longterm_text_telemetry=true in config if downstream tooling explicitly requires raw text.
    def _longterm_telemetry_text_fields(
        self,
        field_name: str,
        value: object,
        *,
        preview_limit: int = _TELEMETRY_PREVIEW_CHARS,
    ) -> dict[str, object]:
        text = self._coerce_text(value)
        if not text:
            return {}
        fields: dict[str, object] = {
            f"{field_name}_preview": self._longterm_normalize_telemetry_text(text, limit=preview_limit),
            f"{field_name}_chars": len(text),
        }
        fingerprint = self._longterm_text_fingerprint(text)
        if fingerprint is not None:
            fields[f"{field_name}_sha256"] = fingerprint
        if self._longterm_config_bool("allow_sensitive_longterm_text_telemetry", False):
            fields[field_name] = text
        return fields

    def _longterm_sanitize_event_value(self, value: object):
        if isinstance(value, str):
            return self._longterm_normalize_telemetry_text(value, limit=_TELEMETRY_PREVIEW_CHARS)
        if isinstance(value, dict):
            sanitized: dict[str, object] = {}
            for key, item in value.items():
                sanitized_key = self._longterm_normalize_telemetry_text(key, limit=64) or "field"
                sanitized[sanitized_key] = self._longterm_sanitize_event_value(item)
            return sanitized
        if isinstance(value, (list, tuple, set)):
            return [self._longterm_sanitize_event_value(item) for item in value]
        return value

    def _longterm_sanitize_event_kwargs(self, **kwargs: object) -> dict[str, object]:
        return {key: self._longterm_sanitize_event_value(value) for key, value in kwargs.items()}

    def _longterm_safe_emit_kv(self, key: str, value: object, *, limit: int = _TELEMETRY_PREVIEW_CHARS) -> None:
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, (int, float)):
            rendered = str(value)
        else:
            rendered = self._longterm_normalize_telemetry_text(value, limit=limit)
            if not rendered:
                rendered = "none"
        self._safe_emit(f"{key}={rendered}")

    def _longterm_candidate_is_safety_exempt(self, candidate: object) -> bool:
        if bool(getattr(candidate, "safety_exempt", False)):
            return True
        sensitivity = (self._coerce_text(getattr(candidate, "sensitivity", None)) or "").strip().lower()
        if sensitivity in _SAFETY_EXEMPT_SENSITIVITIES:
            return True
        kind = (self._coerce_text(getattr(candidate, "kind", None)) or "").strip().lower()
        return any(kind.startswith(prefix) for prefix in _SAFETY_EXEMPT_KIND_PREFIXES)

    def _longterm_prepare_spoken_prompt(self, text: object, *, context: str) -> str:
        max_chars = max(32, self._longterm_config_int("long_term_proactive_max_prompt_chars", _MAX_SPOKEN_PROMPT_CHARS))
        cleaned = self._longterm_normalize_telemetry_text(text, limit=max_chars)
        return self._require_non_empty_text(cleaned, context=context)

    def _longterm_append_observation_fact(self, facts: list[str], key: str, value: object) -> None:
        rendered = self._longterm_normalize_telemetry_text(f"{key}={value}", limit=_MAX_OBSERVATION_FACT_CHARS)
        if rendered and rendered not in facts and len(facts) < _MAX_OBSERVATION_FACTS:
            facts.append(rendered)

    def _longterm_source_attributes_cache(self) -> dict[str, tuple[int, dict[str, object]]]:
        cache = getattr(self, "_longterm_source_attributes_cache_store", None)
        if not isinstance(cache, dict):
            cache = {}
            self._longterm_source_attributes_cache_store = cache
        return cache

    def _longterm_cached_source_attributes(self, memory_id: object) -> dict[str, object]:
        source_id = self._coerce_text(memory_id)
        if not source_id:
            return {}
        cache = self._longterm_source_attributes_cache()
        now_ns = time.monotonic_ns()
        entry = cache.get(source_id)
        if isinstance(entry, tuple) and len(entry) == 2:
            expires_at_ns, cached_attrs = entry
            if isinstance(expires_at_ns, int) and now_ns < expires_at_ns and isinstance(cached_attrs, dict):
                return dict(cached_attrs)

        try:
            source = self.runtime.long_term_memory.object_store.get_object(source_id)
        except LongTermRemoteUnavailableError:
            raise
        except Exception as exc:
            self._remember_background_fault("load_longterm_source_attributes", exc)
            return {}

        raw_attrs = getattr(source, "attributes", None) if source is not None else None
        attrs = dict(raw_attrs) if isinstance(raw_attrs, dict) else {}
        ttl_s = self._config_interval_seconds(
            "long_term_source_attributes_cache_ttl_s",
            _DEFAULT_SOURCE_ATTR_CACHE_TTL_S,
        )
        cache[source_id] = (now_ns + self._longterm_interval_to_ns(ttl_s), attrs)
        while len(cache) > _MAX_SOURCE_ATTR_CACHE_ITEMS:
            cache.pop(next(iter(cache)))
        return dict(attrs)

    def _longterm_reserve_and_skip_candidate(
        self,
        *,
        candidate: object,
        current_time: datetime,
        reason: object,
        reserve_fault_context: str,
    ):
        reservation = None
        skip_reason = self._longterm_normalize_telemetry_text(reason, limit=96) or "blocked"
        try:
            reservation = self.runtime.reserve_specific_long_term_proactive_candidate(
                candidate,
                now=current_time,
            )
        except Exception as exc:
            self._remember_background_fault(reserve_fault_context, exc)
        if reservation is not None:
            self._safe_mark_long_term_proactive_candidate_skipped(
                reservation,
                reason=skip_reason,
            )
        return reservation, skip_reason

    def _longterm_safe_release_candidate_reservation(
        self,
        reservation,
        *,
        reason: object,
        prefer_skip: bool = True,
    ) -> None:
        if reservation is None:
            return
        release_reason = self._longterm_normalize_telemetry_text(reason, limit=96) or "released"
        if prefer_skip:
            try:
                self._safe_mark_long_term_proactive_candidate_skipped(
                    reservation,
                    reason=release_reason,
                )
                return
            except Exception as exc:
                self._remember_background_fault("release_longterm_candidate_reservation_skip", exc)
        runtime = getattr(self, "runtime", None)
        for method_name in (
            "cancel_specific_long_term_proactive_candidate_reservation",
            "cancel_long_term_proactive_candidate_reservation",
            "release_long_term_proactive_candidate_reservation",
        ):
            method = getattr(runtime, method_name, None)
            if not callable(method):
                continue
            try:
                try:
                    method(reservation, reason=release_reason)  # pylint: disable=not-callable
                except TypeError:
                    method(reservation)  # pylint: disable=not-callable
                return
            except Exception as exc:
                self._remember_background_fault(method_name, exc)
        if not prefer_skip:
            try:
                self._safe_mark_long_term_proactive_candidate_skipped(
                    reservation,
                    reason=release_reason,
                )
                return
            except Exception as exc:
                self._remember_background_fault("release_longterm_candidate_reservation_skip", exc)

    def _block_sensitive_longterm_candidate_if_needed(
        self,
        *,
        candidate,
        live_facts: dict[str, object] | None,
        current_time: datetime,
    ) -> bool:
        """Skip one sensitive long-term candidate when room context is ambiguous."""

        decision = evaluate_respeaker_sensitive_behavior_gate(
            candidate_sensitivity=getattr(candidate, "sensitivity", None),
            live_facts=live_facts,
        )
        if decision.allowed:
            return False

        reservation, skip_reason = self._longterm_reserve_and_skip_candidate(
            candidate=candidate,
            current_time=current_time,
            reason=decision.reason or "sensitive_audio_context_blocked",
            reserve_fault_context="reserve_sensitive_longterm_candidate",
        )

        candidate_id = self._longterm_normalize_telemetry_text(getattr(candidate, "candidate_id", None), limit=128) or "unknown"
        self._longterm_safe_emit_kv("longterm_proactive_skipped", skip_reason)
        self._safe_record_event(
            "longterm_proactive_skipped",
            "Twinr blocked a sensitive proactive memory prompt because current room context was too ambiguous.",
            candidate_id=candidate_id,
            candidate_kind=self._longterm_normalize_telemetry_text(getattr(candidate, "kind", None), limit=96) or "unknown",
            skip_reason=skip_reason,
            skip_recorded=reservation is not None,
            **self._longterm_telemetry_text_fields("summary", getattr(candidate, "summary", None)),
            **self._longterm_sanitize_event_kwargs(**decision.event_data()),
        )
        return True

    def _block_longterm_candidate_for_ambiguous_room_guard_if_needed(
        self,
        *,
        candidate,
        live_facts: dict[str, object] | None,
        current_time: datetime,
    ) -> bool:
        """Skip one long-term candidate when the room context is not target-safe."""

        if not isinstance(live_facts, dict):
            return False
        sensor = live_facts.get("sensor")
        observed_at = None
        if isinstance(sensor, dict):
            observed_at = sensor.get("observed_at")
        snapshot = AmbiguousRoomGuardSnapshot.from_fact_map(
            live_facts.get("ambiguous_room_guard"),
        ) or derive_ambiguous_room_guard(
            observed_at=None if observed_at is None else observed_at,
            live_facts=live_facts,
        )
        if not snapshot.guard_active or not ambiguous_room_guard_requires_hard_block(snapshot.reason):
            return False

        reservation, skip_reason = self._longterm_reserve_and_skip_candidate(
            candidate=candidate,
            current_time=current_time,
            reason=self._coerce_text(snapshot.reason) or "ambiguous_room_guard_blocked",
            reserve_fault_context="reserve_ambiguous_room_longterm_candidate",
        )

        candidate_id = self._longterm_normalize_telemetry_text(getattr(candidate, "candidate_id", None), limit=128) or "unknown"
        self._longterm_safe_emit_kv("longterm_proactive_skipped", skip_reason)
        self._safe_record_event(
            "longterm_proactive_skipped",
            "Twinr blocked a long-term proactive prompt because the room context was not safe for targeted inference.",
            candidate_id=candidate_id,
            candidate_kind=self._longterm_normalize_telemetry_text(getattr(candidate, "kind", None), limit=96) or "unknown",
            skip_reason=skip_reason,
            skip_recorded=reservation is not None,
            **self._longterm_telemetry_text_fields("summary", getattr(candidate, "summary", None)),
            **self._longterm_sanitize_event_kwargs(**snapshot.event_data()),
        )
        return True

    def _block_longterm_candidate_for_multimodal_initiative_if_needed(
        self,
        *,
        candidate,
        live_facts: dict[str, object] | None,
        current_time: datetime,
    ) -> bool:
        """Skip one non-safety long-term candidate when initiative confidence is low."""

        if self._longterm_candidate_is_safety_exempt(candidate):
            return False
        if not isinstance(live_facts, dict):
            return False
        snapshot = ReSpeakerMultimodalInitiativeSnapshot.from_fact_map(
            live_facts.get("multimodal_initiative"),
        )
        if snapshot is None or snapshot.ready or snapshot.recommended_channel != _DISPLAY_CHANNEL:
            return False

        reservation, skip_reason = self._longterm_reserve_and_skip_candidate(
            candidate=candidate,
            current_time=current_time,
            reason=self._coerce_text(snapshot.block_reason) or "low_multimodal_initiative_confidence",
            reserve_fault_context="reserve_multimodal_longterm_candidate",
        )

        candidate_id = self._longterm_normalize_telemetry_text(getattr(candidate, "candidate_id", None), limit=128) or "unknown"
        self._longterm_safe_emit_kv("longterm_proactive_skipped", skip_reason)
        self._safe_record_event(
            "longterm_proactive_skipped",
            "Twinr blocked a long-term proactive prompt because multimodal initiative confidence was too low.",
            candidate_id=candidate_id,
            candidate_kind=self._longterm_normalize_telemetry_text(getattr(candidate, "kind", None), limit=96) or "unknown",
            skip_reason=skip_reason,
            skip_recorded=reservation is not None,
            safety_exempt=False,
            **self._longterm_telemetry_text_fields("summary", getattr(candidate, "summary", None)),
            **self._longterm_sanitize_event_kwargs(**snapshot.event_data()),
        )
        return True

    def _longterm_proactive_observation_facts(
        self,
        *,
        candidate,
        live_facts: dict[str, object] | None,
    ) -> tuple[str, ...]:
        candidate_kind = self._coerce_text(getattr(candidate, "kind", None)) or "unknown"
        facts: list[str] = []
        self._longterm_append_observation_fact(facts, "candidate_kind", candidate_kind)
        self._longterm_append_observation_fact(
            facts,
            "sensitivity",
            self._coerce_text(getattr(candidate, "sensitivity", None)) or "unknown",
        )
        self._longterm_append_observation_fact(facts, "safety_exempt", self._longterm_candidate_is_safety_exempt(candidate))

        sensor = live_facts.get("sensor") if isinstance(live_facts, dict) else None
        observed_at = sensor.get("observed_at") if isinstance(sensor, dict) else None

        ambiguous_snapshot = None
        if isinstance(live_facts, dict):
            ambiguous_snapshot = AmbiguousRoomGuardSnapshot.from_fact_map(
                live_facts.get("ambiguous_room_guard"),
            ) or derive_ambiguous_room_guard(
                observed_at=None if observed_at is None else observed_at,
                live_facts=live_facts,
            )
        if ambiguous_snapshot is not None:
            self._longterm_append_observation_fact(facts, "ambiguous_room_guard_active", ambiguous_snapshot.guard_active)
            self._longterm_append_observation_fact(
                facts,
                "ambiguous_room_guard_reason",
                self._coerce_text(ambiguous_snapshot.reason) or "none",
            )

        initiative_snapshot = None
        if isinstance(live_facts, dict):
            initiative_snapshot = ReSpeakerMultimodalInitiativeSnapshot.from_fact_map(
                live_facts.get("multimodal_initiative"),
            )
        if initiative_snapshot is not None:
            self._longterm_append_observation_fact(facts, "multimodal_initiative_ready", initiative_snapshot.ready)
            self._longterm_append_observation_fact(
                facts,
                "multimodal_recommended_channel",
                self._coerce_text(initiative_snapshot.recommended_channel) or "unknown",
            )
            self._longterm_append_observation_fact(
                facts,
                "multimodal_block_reason",
                self._coerce_text(initiative_snapshot.block_reason) or "none",
            )

        if candidate_kind.startswith("routine_"):
            attrs: dict[str, object] = {}
            for memory_id in tuple(getattr(candidate, "source_memory_ids", ()) or ())[:_MAX_SOURCE_MEMORY_IDS]:
                attrs = self._longterm_cached_source_attributes(memory_id)
                if attrs:
                    break
            if attrs.get("routine_type"):
                self._longterm_append_observation_fact(facts, "sensor_routine_type", attrs.get("routine_type"))
            if attrs.get("interaction_type"):
                self._longterm_append_observation_fact(facts, "sensor_interaction_type", attrs.get("interaction_type"))
            if attrs.get("deviation_type"):
                self._longterm_append_observation_fact(facts, "sensor_deviation_type", attrs.get("deviation_type"))
            if attrs.get("daypart"):
                self._longterm_append_observation_fact(facts, "sensor_daypart", attrs.get("daypart"))
            if attrs.get("weekday_class"):
                self._longterm_append_observation_fact(facts, "sensor_weekday_class", attrs.get("weekday_class"))
            if isinstance(live_facts, dict):
                camera = live_facts.get("camera")
                vad = live_facts.get("vad")
                if isinstance(camera, dict):
                    self._longterm_append_observation_fact(facts, "live_person_visible", bool(camera.get("person_visible")))
                    self._longterm_append_observation_fact(
                        facts,
                        "live_looking_toward_device",
                        bool(camera.get("looking_toward_device")),
                    )
                    self._longterm_append_observation_fact(
                        facts,
                        "live_hand_or_object_near_camera",
                        bool(camera.get("hand_or_object_near_camera")),
                    )
                    self._longterm_append_observation_fact(
                        facts,
                        "live_body_pose",
                        camera.get("body_pose") or "unknown",
                    )
                if isinstance(vad, dict):
                    self._longterm_append_observation_fact(facts, "live_quiet", bool(vad.get("quiet")))
                    self._longterm_append_observation_fact(facts, "live_speech_detected", bool(vad.get("speech_detected")))
                self._longterm_append_observation_fact(
                    facts,
                    "last_response_available",
                    bool(live_facts.get("last_response_available")),
                )
                self._longterm_append_observation_fact(
                    facts,
                    "recent_print_completed",
                    bool(live_facts.get("recent_print_completed")),
                )
        return tuple(facts)

    def _maybe_run_long_term_memory_proactive(self) -> bool:
        if not self._longterm_deadline_due(
            "_next_long_term_memory_proactive_check_at",
            self._config_interval_seconds(
                "long_term_memory_proactive_poll_interval_s",
                _DEFAULT_LONGTERM_POLL_INTERVAL_S,
            ),
        ):
            return False
        if not self._background_work_allowed():
            return False

        governor_reservation: ProactiveGovernorReservation | None = None
        reservation = None
        candidate_id: str | None = None
        try:
            live_facts = self._current_longterm_live_facts()
            preview = self.runtime.preview_long_term_proactive_candidate(live_facts=live_facts)
            if preview is None:
                return False

            candidate_id = self._longterm_normalize_telemetry_text(getattr(preview, "candidate_id", None), limit=128) or "unknown"
            current_time = datetime.now(self._local_timezone())
            safety_exempt = self._longterm_candidate_is_safety_exempt(preview)
            if self._block_longterm_candidate_for_ambiguous_room_guard_if_needed(
                candidate=preview,
                live_facts=live_facts,
                current_time=current_time,
            ):
                return False
            if self._block_sensitive_longterm_candidate_if_needed(
                candidate=preview,
                live_facts=live_facts,
                current_time=current_time,
            ):
                return False
            if self._block_longterm_candidate_for_multimodal_initiative_if_needed(
                candidate=preview,
                live_facts=live_facts,
                current_time=current_time,
            ):
                return False
            governor_inputs = self._current_governor_inputs(requested_channel="speech")
            governor_reservation = self._reserve_governed_prompt(
                ProactiveGovernorCandidate(
                    source_kind="longterm",
                    source_id=candidate_id,
                    summary=self._coerce_text(getattr(preview, "summary", None)),
                    priority=self._confidence_to_priority(
                        getattr(preview, "confidence", None),
                        default=50,
                    ),
                    presence_session_id=governor_inputs.presence_session_id,
                    safety_exempt=safety_exempt,
                    counts_toward_presence_budget=not safety_exempt,
                ),
                governor_inputs=governor_inputs,
            )
            if governor_reservation is None:
                return False

            reservation = self.runtime.reserve_specific_long_term_proactive_candidate(
                preview,
                now=current_time,
            )
            if reservation is None or reservation.candidate.candidate_id != getattr(
                preview,
                "candidate_id",
                None,
            ):
                if reservation is not None:
                    reserved_candidate_id = self._longterm_normalize_telemetry_text(
                        getattr(reservation.candidate, "candidate_id", None),
                        limit=128,
                    ) or "unknown"
                    self._longterm_safe_release_candidate_reservation(
                        reservation,
                        reason="stale_preview_candidate_mismatch",
                        prefer_skip=False,
                    )
                    self._safe_record_event(
                        "longterm_proactive_reservation_mismatch",
                        "Twinr discarded a stale preview because the reserved long-term candidate changed before delivery.",
                        level="warning",
                        preview_candidate_id=candidate_id,
                        reserved_candidate_id=reserved_candidate_id,
                    )
                self._safe_cancel_governor_reservation(governor_reservation)
                return False

            candidate = reservation.candidate
            response = None
            prompt_mode = "default"
            prompt_text = self._coerce_text(getattr(candidate, "summary", None))
            trigger_candidate_id = self._longterm_normalize_telemetry_text(
                getattr(candidate, "candidate_id", None),
                limit=128,
            ) or "unknown"
            trigger_id = f"longterm:{trigger_candidate_id}"
            try:
                stop_processing_feedback = self._start_working_feedback_loop("processing")
                try:
                    response = self.agent_provider.phrase_proactive_prompt_with_metadata(
                        trigger_id=trigger_id,
                        reason=self._coerce_text(getattr(candidate, "rationale", None)),
                        default_prompt=prompt_text,
                        priority=self._confidence_to_priority(
                            getattr(candidate, "confidence", None),
                            default=50,
                        ),
                        conversation=self.runtime.conversation_context(),
                        recent_prompts=self._recent_proactive_prompts(trigger_id=trigger_id),
                        observation_facts=self._longterm_proactive_observation_facts(
                            candidate=candidate,
                            live_facts=live_facts,
                        ),
                    )
                finally:
                    stop_processing_feedback()
                candidate_prompt = self._coerce_text(getattr(response, "text", None))
                if candidate_prompt:
                    prompt_text = candidate_prompt
                    prompt_mode = "llm"
                else:
                    prompt_mode = "default_fallback"
            except LongTermRemoteUnavailableError:
                raise
            except Exception as exc:
                prompt_mode = "default_fallback"
                self._longterm_safe_emit_kv("longterm_proactive_phrase_fallback", "default")
                self._longterm_safe_emit_kv("longterm_proactive_phrase_error", self._longterm_exception_summary(exc))
                self._safe_record_event(
                    "longterm_proactive_phrase_fallback",
                    "Twinr fell back to the default long-term proactive prompt after phrasing failed.",
                    level="warning",
                    candidate_id=trigger_candidate_id,
                    error_type=type(exc).__name__,
                    error=self._longterm_exception_summary(exc),
                )

            spoken_prompt = self.runtime.begin_proactive_prompt(
                self._longterm_prepare_spoken_prompt(
                    prompt_text,
                    context=(
                        "long-term proactive candidate "
                        f"{trigger_candidate_id} prompt"
                    ),
                )
            )
            self._safe_emit_status(force=True)
            tts_started = time.perf_counter()
            tts_ms, first_audio_ms = self._play_streaming_tts_with_feedback(
                spoken_prompt,
                turn_started=tts_started,
            )
            self._finalize_speaking_output()
            self.runtime.mark_long_term_proactive_candidate_delivered(
                reservation,
                prompt_text=spoken_prompt,
            )
            self._safe_mark_governor_delivered(governor_reservation)
            self._longterm_safe_emit_kv(
                "longterm_proactive_candidate",
                self._coerce_text(getattr(candidate, "candidate_id", None)) or "unknown",
            )
            self._longterm_safe_emit_kv(
                "longterm_proactive_kind",
                self._coerce_text(getattr(candidate, "kind", None)) or "unknown",
            )
            self._longterm_safe_emit_kv("longterm_proactive_prompt_mode", prompt_mode)
            prompt_fingerprint = self._longterm_text_fingerprint(spoken_prompt)
            if prompt_fingerprint is not None:
                self._longterm_safe_emit_kv("longterm_proactive_prompt_sha256", prompt_fingerprint)
            self._longterm_safe_emit_kv(
                "longterm_proactive_prompt_preview",
                self._longterm_normalize_telemetry_text(spoken_prompt, limit=_TELEMETRY_PREVIEW_CHARS),
            )
            self._longterm_safe_emit_kv("longterm_proactive_prompt_chars", len(spoken_prompt))
            self._longterm_safe_emit_kv("timing_longterm_proactive_tts_ms", tts_ms)
            if first_audio_ms is not None:
                self._longterm_safe_emit_kv("timing_longterm_proactive_first_audio_ms", first_audio_ms)
            if response is not None:
                self._safe_record_usage(
                    request_kind="longterm_proactive_prompt",
                    source="realtime_loop",
                    model=getattr(response, "model", "unknown"),
                    response_id=getattr(response, "response_id", None),
                    request_id=getattr(response, "request_id", None),
                    used_web_search=False,
                    token_usage=getattr(response, "token_usage", None),
                    proactive_trigger=trigger_id,
                )
            self._safe_record_event(
                "longterm_proactive_prompted",
                "Twinr spoke a proactive prompt derived from long-term memory.",
                trigger=trigger_id,
                candidate_id=trigger_candidate_id,
                candidate_kind=self._longterm_normalize_telemetry_text(getattr(candidate, "kind", None), limit=96) or "unknown",
                prompt_mode=prompt_mode,
                safety_exempt=safety_exempt,
                **self._longterm_telemetry_text_fields("prompt", spoken_prompt),
                **self._longterm_telemetry_text_fields("rationale", getattr(candidate, "rationale", None)),
            )
            return True
        except LongTermRemoteUnavailableError as exc:
            self._recover_speaking_output_state()
            error_summary = self._longterm_exception_summary(exc)
            if reservation is not None:
                self._longterm_safe_release_candidate_reservation(
                    reservation,
                    reason=f"required_remote_failed: {error_summary}",
                )
                reservation = None
            if self._enter_required_remote_error(exc):
                if governor_reservation is not None:
                    self._safe_cancel_governor_reservation(governor_reservation)
                self._safe_record_event(
                    "longterm_proactive_required_remote_failed",
                    "A long-term proactive probe hit a required remote-memory failure and forced Twinr into fail-closed error state.",
                    level="error",
                    candidate_id=candidate_id,
                    error_type=type(exc).__name__,
                    error=error_summary,
                )
                return False
            if governor_reservation is not None:
                self._safe_mark_governor_skipped(
                    governor_reservation,
                    reason=f"delivery_failed: {error_summary}",
                )
            self._longterm_safe_emit_kv("longterm_proactive_error", error_summary)
            self._safe_record_event(
                "longterm_proactive_failed",
                "A long-term proactive prompt failed during delivery.",
                level="error",
                candidate_id=candidate_id,
                error_type=type(exc).__name__,
                error=error_summary,
            )
            return False
        except Exception as exc:
            self._recover_speaking_output_state()
            error_summary = self._longterm_exception_summary(exc)
            if reservation is not None:
                self._longterm_safe_release_candidate_reservation(
                    reservation,
                    reason=f"delivery_failed: {error_summary}",
                )
                if governor_reservation is not None:
                    self._safe_mark_governor_skipped(
                        governor_reservation,
                        reason=f"delivery_failed: {error_summary}",
                    )
            elif governor_reservation is not None:
                self._safe_cancel_governor_reservation(governor_reservation)
            self._longterm_safe_emit_kv("longterm_proactive_error", error_summary)
            self._safe_record_event(
                "longterm_proactive_failed",
                "A long-term proactive prompt failed during delivery.",
                level="error",
                candidate_id=candidate_id,
                error_type=type(exc).__name__,
                error=error_summary,
            )
            return False

    def _maybe_run_display_reserve_nightly_maintenance(self) -> bool:
        """Prepare the next reserve-lane day plan during quiet idle windows."""

        if not self._longterm_deadline_due(
            "_next_display_reserve_nightly_check_at",
            self._config_interval_seconds(
                "display_reserve_bus_nightly_poll_interval_s",
                _DEFAULT_DISPLAY_RESERVE_NIGHTLY_POLL_INTERVAL_S,
            ),
        ):
            return False
        if not self._background_work_allowed():
            return False
        try:
            planner = self._display_reserve_companion_planner()
            result = planner.maybe_run_nightly_maintenance(
                config=self.config,
                local_now=datetime.now(self._local_timezone()),
                search_backend=self.print_backend,
            )
        except Exception as exc:
            self._remember_background_fault("display_reserve_nightly_maintenance", exc)
            self._longterm_safe_emit_kv("display_reserve_nightly_error", self._longterm_exception_summary(exc))
            self._safe_record_event(
                "display_reserve_nightly_failed",
                "Nightly HDMI reserve maintenance failed inside the realtime background loop.",
                level="error",
                error_type=type(exc).__name__,
                error=self._longterm_exception_summary(exc),
            )
            return False
        if result.action != "prepared":
            return False
        state = result.state
        plan = result.plan
        self._longterm_safe_emit_kv("display_reserve_nightly_prepared", result.target_local_day)
        self._safe_record_event(
            "display_reserve_nightly_prepared",
            "Prepared the next local-day HDMI reserve plan after nightly reflection.",
            level="info",
            target_local_day=result.target_local_day,
            prepared_item_count=(len(plan.items) if plan is not None else 0),
            prepared_candidate_count=(plan.candidate_count if plan is not None else 0),
            reflection_reflected_object_count=(
                state.reflection_reflected_object_count if state is not None else 0
            ),
            reflection_created_summary_count=(
                state.reflection_created_summary_count if state is not None else 0
            ),
            positive_topics=self._longterm_sanitize_event_value(list(state.positive_topics if state is not None else ())),
            cooling_topics=self._longterm_sanitize_event_value(list(state.cooling_topics if state is not None else ())),
        )
        return True
