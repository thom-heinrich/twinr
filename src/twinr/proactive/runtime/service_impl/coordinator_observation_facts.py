# CHANGELOG: 2026-03-29
# BUG-1: Avoid dispatcher crashes when `audio_snapshot` is absent or lacks `.observation`;
#        attention observation now degrades safely instead of dereferencing a nullable snapshot.
# BUG-2: Replace `assert perception_runtime.attention is not None` with an explicit fallback path;
#        `assert` can disappear under `python -O`, turning a controlled invariant into a later crash.
# BUG-3: `sensor_event` derivation is now duplicate-safe and key-safe; partial fact payloads no longer
#        raise `KeyError`, and repeated external camera events no longer double-trigger automations.
# SEC-1: Bound and sanitize automation-facing payloads and external event names to reduce practical
#        resource-exhaustion / event-injection risks on Raspberry Pi 4 deployments.
# IMP-1: Use a monotonic clock for durations and modality staleness so NTP / RTC jumps do not freeze
#        or rewind speech/quiet timers or portrait-match freshness checks.
# IMP-2: Add frontier-style degraded-mode / modality-health facts with last-good snapshot reuse, so
#        missing or unreliable modalities fail soft instead of taking down the automation loop.
# IMP-3: Preserve legacy fact keys while adding explicit pipeline provenance, fallback sources,
#        bounded serialization, and uncertainty/degradation metadata for dynamic multimodal fusion.

"""Automation-fact assembly helpers for proactive observation dispatch.

Purpose: keep fact payload construction and rising-edge event derivation
separate from ops recording and display bridge helpers.

Invariants: automation fact schemas, snapshot side effects, and derived sensor
event names remain backward-compatible with the legacy observation mixin.
This version adds bounded serialization, monotonic timing, and degraded-mode
fallbacks so modality dropouts do not collapse the automation loop.
"""

# mypy: ignore-errors

from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from twinr.hardware.respeaker import build_respeaker_claim_payloads, resolve_respeaker_indicator_state

from ...social.camera_surface import ProactiveCameraSnapshot
from ...social.engine import SocialObservation
from ..affect_proxy import derive_affect_proxy
from ..ambiguous_room_guard import derive_ambiguous_room_guard
from ..audio_policy import ReSpeakerAudioPolicySnapshot
from ..known_user_hint import derive_known_user_hint
from ..multimodal_initiative import derive_respeaker_multimodal_initiative
from ..person_state import derive_person_state
from ..portrait_match import derive_portrait_match
from ..speaker_association import derive_respeaker_speaker_association
from .compat import _round_optional_seconds


_LOGGER = logging.getLogger(__name__)
_SAFE_EVENT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,127}$")
_DEFAULT_FACT_STRING_LIMIT = 512
_DEFAULT_FACT_ITEMS_LIMIT = 96
_DEFAULT_FACT_DEPTH_LIMIT = 8
_DEFAULT_EXTERNAL_EVENT_LIMIT = 32
_DEFAULT_SNAPSHOT_REUSE_TTL_S = 2.5
_AUTOMATION_FACT_SCHEMA_VERSION = 2
_AUTOMATION_FACT_BUILD = "2026-03-29"


@dataclass(slots=True)
class _AutomationFactsSnapshot:
    """Lightweight drop-in snapshot used for conservative degraded-mode fallbacks."""

    payload: dict[str, Any]

    def to_automation_facts(self) -> dict[str, Any]:
        return dict(self.payload)


@dataclass(slots=True)
class _FallbackAttention:
    attention_target: _AutomationFactsSnapshot


@dataclass(slots=True)
class _FallbackPerceptionRuntime:
    attention: _FallbackAttention | None
    degraded: bool = True
    reason: str = "attention_unavailable"

    def to_automation_facts(self) -> dict[str, Any]:
        return {
            "degraded": self.degraded,
            "reason": self.reason,
            "attention": None if self.attention is None else self.attention.attention_target.to_automation_facts(),
        }


class ProactiveCoordinatorObservationFactsMixin:
    """Build automation-facing facts and rising-edge events from observations."""

    def _build_automation_facts(
        self,
        observation: SocialObservation,
        *,
        inspected: bool,
        audio_snapshot=None,
        camera_snapshot: ProactiveCameraSnapshot,
        audio_policy_snapshot: ReSpeakerAudioPolicySnapshot | None,
        presence_snapshot,
    ) -> dict[str, Any]:
        """Build the automation-facing fact payload for one observation."""

        observed_at = self._coerce_float(getattr(observation, "observed_at", None), default=time.time())
        observed_monotonic_ns = self._resolve_monotonic_ns(observation=observation, camera_snapshot=camera_snapshot)
        observed_monotonic_s = observed_monotonic_ns / 1_000_000_000.0

        audio = getattr(observation, "audio", None)
        if audio is None:
            audio = self._null_audio_observation()

        runtime_status = getattr(getattr(getattr(self, "runtime", None), "status", None), "value", None)
        config = getattr(self, "config", None)

        speech_detected = getattr(audio, "speech_detected", None) is True
        quiet = getattr(audio, "speech_detected", None) is False
        self._speech_detected_since = self._next_since(
            speech_detected,
            getattr(self, "_speech_detected_since", None),
            observed_monotonic_s,
        )
        self._quiet_since = self._next_since(
            quiet,
            getattr(self, "_quiet_since", None),
            observed_monotonic_s,
        )

        no_motion_for_s = 0.0
        last_motion_at = getattr(self, "_last_motion_at", None)
        if getattr(observation, "pir_motion_detected", None) is False and last_motion_at is not None:
            no_motion_for_s = max(0.0, observed_at - self._coerce_float(last_motion_at, default=observed_at))

        presence_session_id = None if presence_snapshot is None else getattr(presence_snapshot, "session_id", None)
        presence_session_ref = self._redact_session_id(presence_session_id)

        signal_snapshot = None if audio_snapshot is None else getattr(audio_snapshot, "signal_snapshot", None)
        respeaker_claim_contract = self._sanitize_automation_value(
            self._call_with_optional_cache(
                "latest_respeaker_claim_contract",
                lambda: build_respeaker_claim_payloads(
                    signal_snapshot=signal_snapshot,
                    session_id=presence_session_id,
                    non_speech_audio_likely=getattr(audio, "non_speech_audio_likely", None),
                    background_media_likely=getattr(audio, "background_media_likely", None),
                ),
                observed_monotonic_ns=observed_monotonic_ns,
                fallback_factory=lambda: {},
            )[0]
        )

        indicator_state = self._sanitize_automation_value(
            self._call_with_optional_cache(
                "latest_respeaker_indicator_state",
                lambda: resolve_respeaker_indicator_state(
                    runtime_status=runtime_status,
                    runtime_alert_code=(
                        None if audio_policy_snapshot is None else audio_policy_snapshot.runtime_alert_code
                    ),
                    mute_active=getattr(audio, "mute_active", None),
                ).event_data(),
                observed_monotonic_ns=observed_monotonic_ns,
                fallback_factory=lambda: {},
            )[0]
        )

        camera_facts, camera_facts_source = self._call_with_optional_cache(
            "latest_camera_facts_snapshot",
            lambda: self._sanitize_automation_value(camera_snapshot.to_automation_facts()),
            observed_monotonic_ns=observed_monotonic_ns,
            fallback_factory=self._fallback_camera_facts,
        )

        modality_health = self._build_modality_health(
            observation=observation,
            observed_at=observed_at,
            observed_monotonic_ns=observed_monotonic_ns,
            camera_snapshot=camera_snapshot,
            camera_facts=camera_facts,
            camera_facts_source=camera_facts_source,
            audio_snapshot=audio_snapshot,
            audio_policy_snapshot=audio_policy_snapshot,
            presence_snapshot=presence_snapshot,
        )

        facts = {
            "sensor": {
                "inspected": inspected,
                "observed_at": observed_at,
                "captured_at": observed_at,
                "observed_monotonic_ns": observed_monotonic_ns,
                "schema_version": _AUTOMATION_FACT_SCHEMA_VERSION,
                "fact_build": _AUTOMATION_FACT_BUILD,
                # BREAKING: automation-facing payload now exports a redacted session reference by default.
                # Set `config.automation_export_raw_session_id = True` to preserve the legacy raw session id.
                "presence_session_id": presence_session_ref,
                "voice_activation_armed": None if presence_snapshot is None else getattr(presence_snapshot, "armed", None),
                "voice_activation_presence_reason": (
                    None if presence_snapshot is None else getattr(presence_snapshot, "reason", None)
                ),
            },
            "pir": {
                "motion_detected": getattr(observation, "pir_motion_detected", None),
                "low_motion": getattr(observation, "low_motion", None),
                "no_motion_for_s": round(no_motion_for_s, 3),
            },
            "camera": camera_facts,
            "vad": {
                "speech_detected": speech_detected,
                "speech_detected_for_s": round(
                    self._duration_since(getattr(self, "_speech_detected_since", None), observed_monotonic_s),
                    3,
                ),
                "quiet": quiet,
                "quiet_for_s": round(
                    self._duration_since(getattr(self, "_quiet_since", None), observed_monotonic_s),
                    3,
                ),
                "distress_detected": getattr(audio, "distress_detected", None) is True,
                "room_quiet": getattr(audio, "room_quiet", None),
                "recent_speech_age_s": _round_optional_seconds(getattr(audio, "recent_speech_age_s", None)),
                "assistant_output_active": getattr(audio, "assistant_output_active", None),
                "signal_source": getattr(audio, "signal_source", None),
            },
            "respeaker": {
                "runtime_mode": getattr(audio, "device_runtime_mode", None),
                "host_control_ready": getattr(audio, "host_control_ready", None),
                "transport_reason": getattr(audio, "transport_reason", None),
                "azimuth_deg": getattr(audio, "azimuth_deg", None),
                "direction_confidence": getattr(audio, "direction_confidence", None),
                "non_speech_audio_likely": getattr(audio, "non_speech_audio_likely", None),
                "background_media_likely": getattr(audio, "background_media_likely", None),
                "speech_overlap_likely": getattr(audio, "speech_overlap_likely", None),
                "barge_in_detected": getattr(audio, "barge_in_detected", None),
                "mute_active": getattr(audio, "mute_active", None),
                **indicator_state,
                "claim_contract": respeaker_claim_contract,
            },
            "audio_policy": {
                "presence_audio_active": (
                    None if audio_policy_snapshot is None else getattr(audio_policy_snapshot, "presence_audio_active", None)
                ),
                "recent_follow_up_speech": (
                    None if audio_policy_snapshot is None else getattr(audio_policy_snapshot, "recent_follow_up_speech", None)
                ),
                "room_busy_or_overlapping": (
                    None if audio_policy_snapshot is None else getattr(audio_policy_snapshot, "room_busy_or_overlapping", None)
                ),
                "quiet_window_open": (
                    None if audio_policy_snapshot is None else getattr(audio_policy_snapshot, "quiet_window_open", None)
                ),
                "non_speech_audio_likely": (
                    None if audio_policy_snapshot is None else getattr(audio_policy_snapshot, "non_speech_audio_likely", None)
                ),
                "background_media_likely": (
                    None if audio_policy_snapshot is None else getattr(audio_policy_snapshot, "background_media_likely", None)
                ),
                "barge_in_recent": (
                    None if audio_policy_snapshot is None else getattr(audio_policy_snapshot, "barge_in_recent", None)
                ),
                "speaker_direction_stable": (
                    None if audio_policy_snapshot is None else getattr(audio_policy_snapshot, "speaker_direction_stable", None)
                ),
                "mute_blocks_voice_capture": (
                    None if audio_policy_snapshot is None else getattr(audio_policy_snapshot, "mute_blocks_voice_capture", None)
                ),
                "resume_window_open": (
                    None if audio_policy_snapshot is None else getattr(audio_policy_snapshot, "resume_window_open", None)
                ),
                "initiative_block_reason": (
                    None if audio_policy_snapshot is None else getattr(audio_policy_snapshot, "initiative_block_reason", None)
                ),
                "speech_delivery_defer_reason": (
                    None if audio_policy_snapshot is None else getattr(audio_policy_snapshot, "speech_delivery_defer_reason", None)
                ),
                "runtime_alert_code": (
                    None if audio_policy_snapshot is None else getattr(audio_policy_snapshot, "runtime_alert_code", None)
                ),
            },
            "modality_health": modality_health,
            "pipeline": {
                "schema_version": _AUTOMATION_FACT_SCHEMA_VERSION,
                "fact_build": _AUTOMATION_FACT_BUILD,
                "degraded": False,
                "degrade_reasons": [],
                "fallback_sources": {
                    "camera_facts": camera_facts_source,
                },
            },
        }

        speaker_association, speaker_association_source = self._call_with_optional_cache(
            "latest_speaker_association_snapshot",
            lambda: derive_respeaker_speaker_association(
                observed_at=observed_at,
                live_facts=facts,
            ),
            observed_monotonic_ns=observed_monotonic_ns,
            fallback_factory=lambda: self._fallback_speaker_association(facts),
        )

        multimodal_initiative, multimodal_initiative_source = self._call_with_optional_cache(
            "latest_multimodal_initiative_snapshot",
            lambda: derive_respeaker_multimodal_initiative(
                observed_at=observed_at,
                live_facts=facts,
                speaker_association=speaker_association,
            ),
            observed_monotonic_ns=observed_monotonic_ns,
            fallback_factory=lambda: self._fallback_multimodal_initiative(facts),
        )

        ambiguous_room_guard, ambiguous_room_guard_source = self._call_with_optional_cache(
            "latest_ambiguous_room_guard_snapshot",
            lambda: derive_ambiguous_room_guard(
                observed_at=observed_at,
                live_facts=facts,
            ),
            observed_monotonic_ns=observed_monotonic_ns,
            fallback_factory=lambda: self._fallback_ambiguous_room_guard(facts),
        )

        portrait_match, portrait_match_source = self._call_with_optional_cache(
            "latest_portrait_match_snapshot",
            lambda: derive_portrait_match(
                observed_at=observed_at,
                live_facts=facts,
                provider=getattr(self, "portrait_match_provider", None),
                ambiguous_room_guard=ambiguous_room_guard,
                now_monotonic=observed_monotonic_s,
            ),
            observed_monotonic_ns=observed_monotonic_ns,
            fallback_factory=lambda: self._fallback_portrait_match(facts),
        )

        identity_fusion, identity_fusion_source = self._call_with_optional_cache(
            "latest_identity_fusion_snapshot",
            lambda: getattr(self, "identity_fusion_tracker").observe(
                observed_at=observed_at,
                live_facts=facts,
                voice_status=getattr(getattr(self, "runtime", None), "user_voice_status", None),
                voice_confidence=getattr(getattr(self, "runtime", None), "user_voice_confidence", None),
                voice_checked_at=getattr(getattr(self, "runtime", None), "user_voice_checked_at", None),
                voice_matched_user_id=getattr(getattr(self, "runtime", None), "user_voice_user_id", None),
                voice_matched_user_display_name=getattr(
                    getattr(self, "runtime", None), "user_voice_user_display_name", None
                ),
                voice_match_source=getattr(getattr(self, "runtime", None), "user_voice_match_source", None),
                max_voice_age_s=int(getattr(config, "voice_assessment_max_age_s", 120) or 120),
                presence_session_id=presence_session_id,
                ambiguous_room_guard=ambiguous_room_guard,
                speaker_association=speaker_association,
                portrait_match=portrait_match,
            ),
            observed_monotonic_ns=observed_monotonic_ns,
            fallback_factory=lambda: self._fallback_identity_fusion(facts),
        )

        known_user_hint, known_user_hint_source = self._call_with_optional_cache(
            "latest_known_user_hint_snapshot",
            lambda: derive_known_user_hint(
                observed_at=observed_at,
                live_facts=facts,
                voice_status=getattr(getattr(self, "runtime", None), "user_voice_status", None),
                voice_confidence=getattr(getattr(self, "runtime", None), "user_voice_confidence", None),
                voice_checked_at=getattr(getattr(self, "runtime", None), "user_voice_checked_at", None),
                max_voice_age_s=int(getattr(config, "voice_assessment_max_age_s", 120) or 120),
                ambiguous_room_guard=ambiguous_room_guard,
                speaker_association=speaker_association,
                portrait_match=portrait_match,
                identity_fusion=identity_fusion,
            ),
            observed_monotonic_ns=observed_monotonic_ns,
            fallback_factory=lambda: self._fallback_known_user_hint(facts),
        )

        affect_proxy, affect_proxy_source = self._call_with_optional_cache(
            "latest_affect_proxy_snapshot",
            lambda: derive_affect_proxy(
                observed_at=observed_at,
                live_facts=facts,
                ambiguous_room_guard=ambiguous_room_guard,
            ),
            observed_monotonic_ns=observed_monotonic_ns,
            fallback_factory=lambda: self._fallback_affect_proxy(facts),
        )

        audio_observation = None if audio_snapshot is None else getattr(audio_snapshot, "observation", None)
        perception_runtime, perception_runtime_source = self._call_with_optional_cache(
            "latest_perception_runtime_snapshot",
            lambda: getattr(self, "perception_orchestrator").observe_attention(
                observed_at=observed_at,
                source="automation_observation",
                captured_at=getattr(camera_snapshot, "last_camera_frame_at", None),
                camera_snapshot=camera_snapshot,
                audio_observation=audio_observation,
                audio_policy_snapshot=audio_policy_snapshot,
                runtime_status=runtime_status,
                presence_session_id=presence_session_id,
                speaker_association=speaker_association,
                identity_fusion=identity_fusion,
            ),
            observed_monotonic_ns=observed_monotonic_ns,
            fallback_factory=lambda: self._fallback_perception_runtime(reason="observe_attention_failed"),
        )

        attention = None if perception_runtime is None else getattr(perception_runtime, "attention", None)
        attention_target = None if attention is None else getattr(attention, "attention_target", None)
        if attention_target is None:
            cached_attention_target = self._reuse_recent_cache("latest_attention_target_snapshot", observed_monotonic_ns)
            if cached_attention_target is not None:
                attention_target = cached_attention_target
                attention_target_source = "cache"
            else:
                attention_target = self._fallback_attention_target(
                    facts=facts,
                    speaker_association=speaker_association,
                    identity_fusion=identity_fusion,
                    reason="attention_unavailable",
                )
                attention_target_source = "fallback"
            perception_runtime = self._fallback_perception_runtime(
                attention_target=attention_target,
                reason="attention_unavailable",
            )
            perception_runtime_source = "fallback" if perception_runtime_source == "live" else perception_runtime_source
        else:
            self._remember_cache("latest_attention_target_snapshot", attention_target, observed_monotonic_ns)
            attention_target_source = "live"

        enriched_facts = {
            **facts,
            "speaker_association": self._snapshot_facts(speaker_association),
            "multimodal_initiative": self._snapshot_facts(multimodal_initiative),
            "ambiguous_room_guard": self._snapshot_facts(ambiguous_room_guard),
            "identity_fusion": self._snapshot_facts(identity_fusion),
            "portrait_match": self._snapshot_facts(portrait_match),
            "known_user_hint": self._snapshot_facts(known_user_hint),
            "affect_proxy": self._snapshot_facts(affect_proxy),
            "attention_target": self._snapshot_facts(attention_target),
        }

        person_state, person_state_source = self._call_with_optional_cache(
            "latest_person_state_snapshot",
            lambda: derive_person_state(
                observed_at=observed_at,
                live_facts=enriched_facts,
            ),
            observed_monotonic_ns=observed_monotonic_ns,
            fallback_factory=lambda: self._fallback_person_state(enriched_facts),
        )

        facts["speaker_association"] = enriched_facts["speaker_association"]
        facts["multimodal_initiative"] = enriched_facts["multimodal_initiative"]
        facts["ambiguous_room_guard"] = enriched_facts["ambiguous_room_guard"]
        facts["identity_fusion"] = enriched_facts["identity_fusion"]
        facts["portrait_match"] = enriched_facts["portrait_match"]
        facts["known_user_hint"] = enriched_facts["known_user_hint"]
        facts["affect_proxy"] = enriched_facts["affect_proxy"]
        facts["attention_target"] = enriched_facts["attention_target"]
        facts["person_state"] = self._snapshot_facts(person_state)

        pipeline_fallback_sources = facts["pipeline"]["fallback_sources"]
        pipeline_fallback_sources.update(
            {
                "speaker_association": speaker_association_source,
                "multimodal_initiative": multimodal_initiative_source,
                "ambiguous_room_guard": ambiguous_room_guard_source,
                "portrait_match": portrait_match_source,
                "identity_fusion": identity_fusion_source,
                "known_user_hint": known_user_hint_source,
                "affect_proxy": affect_proxy_source,
                "perception_runtime": perception_runtime_source,
                "attention_target": attention_target_source,
                "person_state": person_state_source,
            }
        )

        degrade_reasons = sorted(
            {
                stage
                for stage, source in pipeline_fallback_sources.items()
                if source in {"cache", "fallback"}
            }
        )
        facts["pipeline"]["degraded"] = bool(degrade_reasons)
        facts["pipeline"]["degrade_reasons"] = degrade_reasons
        facts["modality_health"]["degraded"] = facts["pipeline"]["degraded"]
        facts["modality_health"]["degrade_reasons"] = degrade_reasons

        self.latest_speaker_association_snapshot = speaker_association
        self.latest_multimodal_initiative_snapshot = multimodal_initiative
        self.latest_ambiguous_room_guard_snapshot = ambiguous_room_guard
        self.latest_identity_fusion_snapshot = identity_fusion
        self.latest_portrait_match_snapshot = portrait_match
        self.latest_known_user_hint_snapshot = known_user_hint
        self.latest_affect_proxy_snapshot = affect_proxy
        self.latest_perception_runtime_snapshot = perception_runtime
        self.latest_attention_target_snapshot = attention_target
        self.latest_person_state_snapshot = person_state

        return self._sanitize_automation_value(facts)

    def _derive_sensor_events(
        self,
        facts: dict[str, Any],
        *,
        camera_event_names: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        """Return rising-edge event names derived from the latest fact payload."""

        current_flags = {
            "pir.motion_detected": self._as_flag(self._nested_get(facts, "pir", "motion_detected")),
            "vad.speech_detected": self._as_flag(self._nested_get(facts, "vad", "speech_detected")),
            "audio_policy.presence_audio_active": self._as_flag(
                self._nested_get(facts, "audio_policy", "presence_audio_active")
            ),
            "audio_policy.quiet_window_open": self._as_flag(
                self._nested_get(facts, "audio_policy", "quiet_window_open")
            ),
            "audio_policy.resume_window_open": self._as_flag(
                self._nested_get(facts, "audio_policy", "resume_window_open")
            ),
            "audio_policy.room_busy_or_overlapping": self._as_flag(
                self._nested_get(facts, "audio_policy", "room_busy_or_overlapping")
            ),
            "audio_policy.barge_in_recent": self._as_flag(
                self._nested_get(facts, "audio_policy", "barge_in_recent")
            ),
            "speaker_association.associated": self._as_flag(
                self._nested_get(facts, "speaker_association", "associated")
            ),
            "multimodal_initiative.ready": self._as_flag(
                self._nested_get(facts, "multimodal_initiative", "ready")
            ),
            "ambiguous_room_guard.guard_active": self._as_flag(
                self._nested_get(facts, "ambiguous_room_guard", "guard_active")
            ),
            "identity_fusion.matches_main_user": self._as_flag(
                self._nested_get(facts, "identity_fusion", "matches_main_user")
            ),
            "portrait_match.matches_reference_user": self._as_flag(
                self._nested_get(facts, "portrait_match", "matches_reference_user")
            ),
            "known_user_hint.matches_main_user": self._as_flag(
                self._nested_get(facts, "known_user_hint", "matches_main_user")
            ),
            "affect_proxy.concern_cue": self._nested_get(facts, "affect_proxy", "state") == "concern_cue",
            "attention_target.session_focus_active": self._as_flag(
                self._nested_get(facts, "attention_target", "session_focus_active")
            ),
            "person_state.interaction_ready": self._as_flag(
                self._nested_get(facts, "person_state", "interaction_ready")
            ),
            "person_state.safety_concern_active": self._as_flag(
                self._nested_get(facts, "person_state", "safety_concern_active")
            ),
            "person_state.calm_personalization_allowed": self._as_flag(
                self._nested_get(facts, "person_state", "calm_personalization_allowed")
            ),
        }

        # BREAKING: invalid or oversized external camera event names are now dropped instead of being
        # forwarded verbatim into the automation bus.
        event_names: list[str] = list(self._normalize_external_event_names(camera_event_names))
        last_sensor_flags = getattr(self, "_last_sensor_flags", {})
        for key, value in current_flags.items():
            previous = last_sensor_flags.get(key)
            if value and previous is not True:
                event_names.append(key)
        self._last_sensor_flags = current_flags
        return tuple(self._dedupe_strings(event_names))

    def _next_since(self, active: bool, since: float | None, now: float) -> float | None:
        """Advance or clear one duration anchor depending on activity."""

        if active:
            return now if since is None else since
        return None

    def _duration_since(self, since: float | None, now: float) -> float:
        """Return the elapsed duration for one optional activity anchor."""

        if since is None:
            return 0.0
        return max(0.0, now - since)

    def _cfg(self, name: str, default: Any) -> Any:
        config = getattr(self, "config", None)
        value = getattr(config, name, default) if config is not None else default
        return default if value is None else value

    def _coerce_float(self, value: Any, *, default: float) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(result):
            return default
        return result

    def _resolve_monotonic_ns(self, *, observation, camera_snapshot) -> int:
        candidates = (
            ("observed_monotonic_ns", 1),
            ("monotonic_ns", 1),
            ("observed_monotonic_s", 1_000_000_000),
            ("monotonic_s", 1_000_000_000),
            ("observed_monotonic_at", 1_000_000_000),
            ("monotonic_at", 1_000_000_000),
        )
        for source in (observation, camera_snapshot, getattr(self, "runtime", None)):
            for attr_name, multiplier in candidates:
                value = getattr(source, attr_name, None)
                if value is None:
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(numeric) and numeric > 0:
                    return int(numeric * multiplier)
        return time.monotonic_ns()

    def _call_with_optional_cache(
        self,
        cache_name: str,
        producer: Callable[[], Any],
        *,
        observed_monotonic_ns: int,
        fallback_factory: Callable[[], Any] | None = None,
    ) -> tuple[Any, str]:
        try:
            value = producer()
        except Exception:
            _LOGGER.exception("automation facts stage failed", extra={"stage": cache_name})
            cached = self._reuse_recent_cache(cache_name, observed_monotonic_ns)
            if cached is not None:
                return cached, "cache"
            if fallback_factory is not None:
                return fallback_factory(), "fallback"
            raise
        self._remember_cache(cache_name, value, observed_monotonic_ns)
        return value, "live"

    def _cache_meta(self) -> dict[str, int]:
        cache_meta = getattr(self, "_automation_fact_cache_observed_at_ns", None)
        if cache_meta is None:
            cache_meta = {}
            setattr(self, "_automation_fact_cache_observed_at_ns", cache_meta)
        return cache_meta

    def _remember_cache(self, cache_name: str, value: Any, observed_monotonic_ns: int) -> None:
        setattr(self, cache_name, value)
        self._cache_meta()[cache_name] = observed_monotonic_ns

    def _reuse_recent_cache(self, cache_name: str, observed_monotonic_ns: int) -> Any | None:
        ttl_s = self._coerce_float(
            self._cfg("automation_snapshot_reuse_ttl_s", _DEFAULT_SNAPSHOT_REUSE_TTL_S),
            default=_DEFAULT_SNAPSHOT_REUSE_TTL_S,
        )
        if ttl_s <= 0.0:
            return None
        value = getattr(self, cache_name, None)
        cached_at = self._cache_meta().get(cache_name)
        if value is None or cached_at is None:
            return None
        age_s = max(0.0, (observed_monotonic_ns - cached_at) / 1_000_000_000.0)
        if age_s <= ttl_s:
            return value
        return None

    def _snapshot_facts(self, snapshot: Any) -> dict[str, Any]:
        if snapshot is None:
            return {}
        if isinstance(snapshot, Mapping):
            return self._sanitize_automation_value(dict(snapshot))
        to_automation_facts = getattr(snapshot, "to_automation_facts", None)
        if callable(to_automation_facts):
            return self._sanitize_automation_value(to_automation_facts())
        return self._sanitize_automation_value({"value": repr(snapshot)})

    def _build_modality_health(
        self,
        *,
        observation: SocialObservation,
        observed_at: float,
        observed_monotonic_ns: int,
        camera_snapshot,
        camera_facts: dict[str, Any],
        camera_facts_source: str,
        audio_snapshot,
        audio_policy_snapshot,
        presence_snapshot,
    ) -> dict[str, Any]:
        camera_frame_at = getattr(camera_snapshot, "last_camera_frame_at", None)
        camera_frame_age_s = None
        if camera_frame_at is not None:
            camera_frame_age_s = round(
                max(0.0, observed_at - self._coerce_float(camera_frame_at, default=observed_at)),
                3,
            )

        audio_observation = None if audio_snapshot is None else getattr(audio_snapshot, "observation", None)
        audio_signal_snapshot = None if audio_snapshot is None else getattr(audio_snapshot, "signal_snapshot", None)

        camera_available = camera_facts_source in {"live", "cache"} and camera_facts.get("available", True) is not False

        available_modalities = []
        missing_modalities = []
        for name, available in (
            ("pir", True),
            ("camera", camera_available),
            ("audio", audio_observation is not None or getattr(observation, "audio", None) is not None),
            ("audio_signal", audio_signal_snapshot is not None),
            ("audio_policy", audio_policy_snapshot is not None),
            ("presence", presence_snapshot is not None),
        ):
            (available_modalities if available else missing_modalities).append(name)

        return {
            "available_modalities": available_modalities,
            "missing_modalities": missing_modalities,
            "camera_available": "camera" in available_modalities,
            "camera_frame_age_s": camera_frame_age_s,
            "audio_available": "audio" in available_modalities,
            "audio_signal_available": "audio_signal" in available_modalities,
            "audio_policy_available": "audio_policy" in available_modalities,
            "presence_available": "presence" in available_modalities,
            "observation_age_s": round(
                max(0.0, (time.monotonic_ns() - observed_monotonic_ns) / 1_000_000_000.0),
                6,
            ),
            "degraded": False,
            "degrade_reasons": [],
        }

    def _fallback_camera_facts(self) -> dict[str, Any]:
        return {
            "available": False,
            "degraded": True,
            "degrade_reason": "camera_facts_unavailable",
        }

    def _fallback_speaker_association(self, facts: dict[str, Any]) -> _AutomationFactsSnapshot:
        return _AutomationFactsSnapshot(
            {
                "associated": False,
                "available": False,
                "degraded": True,
                "degrade_reason": "speaker_association_unavailable",
            }
        )

    def _fallback_multimodal_initiative(self, facts: dict[str, Any]) -> _AutomationFactsSnapshot:
        return _AutomationFactsSnapshot(
            {
                "ready": False,
                "available": False,
                "degraded": True,
                "degrade_reason": "multimodal_initiative_unavailable",
            }
        )

    def _fallback_ambiguous_room_guard(self, facts: dict[str, Any]) -> _AutomationFactsSnapshot:
        return _AutomationFactsSnapshot(
            {
                "guard_active": True,
                "available": False,
                "degraded": True,
                "degrade_reason": "ambiguous_room_guard_unavailable",
            }
        )

    def _fallback_portrait_match(self, facts: dict[str, Any]) -> _AutomationFactsSnapshot:
        return _AutomationFactsSnapshot(
            {
                "matches_reference_user": False,
                "available": False,
                "confidence": 0.0,
                "degraded": True,
                "degrade_reason": "portrait_match_unavailable",
            }
        )

    def _fallback_identity_fusion(self, facts: dict[str, Any]) -> _AutomationFactsSnapshot:
        return _AutomationFactsSnapshot(
            {
                "matches_main_user": False,
                "available": False,
                "confidence": 0.0,
                "fusion_state": "unknown",
                "degraded": True,
                "degrade_reason": "identity_fusion_unavailable",
            }
        )

    def _fallback_known_user_hint(self, facts: dict[str, Any]) -> _AutomationFactsSnapshot:
        return _AutomationFactsSnapshot(
            {
                "matches_main_user": False,
                "available": False,
                "confidence": 0.0,
                "hint_state": "unknown",
                "degraded": True,
                "degrade_reason": "known_user_hint_unavailable",
            }
        )

    def _fallback_affect_proxy(self, facts: dict[str, Any]) -> _AutomationFactsSnapshot:
        return _AutomationFactsSnapshot(
            {
                "state": "unknown",
                "available": False,
                "degraded": True,
                "degrade_reason": "affect_proxy_unavailable",
            }
        )

    def _fallback_attention_target(
        self,
        *,
        facts: dict[str, Any],
        speaker_association,
        identity_fusion,
        reason: str,
    ) -> _AutomationFactsSnapshot:
        speaker_facts = self._snapshot_facts(speaker_association)
        fusion_facts = self._snapshot_facts(identity_fusion)
        session_focus_active = bool(speaker_facts.get("associated")) and bool(fusion_facts.get("matches_main_user"))
        return _AutomationFactsSnapshot(
            {
                "session_focus_active": session_focus_active,
                "available": False,
                "degraded": True,
                "degrade_reason": reason,
                "target_source": "fallback",
            }
        )

    def _fallback_perception_runtime(
        self,
        *,
        attention_target: Any | None = None,
        reason: str,
    ) -> _FallbackPerceptionRuntime:
        wrapped_attention = None
        if attention_target is not None:
            if not isinstance(attention_target, _AutomationFactsSnapshot):
                attention_target = _AutomationFactsSnapshot(self._snapshot_facts(attention_target))
            wrapped_attention = _FallbackAttention(attention_target=attention_target)
        return _FallbackPerceptionRuntime(
            attention=wrapped_attention,
            degraded=True,
            reason=reason,
        )

    def _fallback_person_state(self, enriched_facts: dict[str, Any]) -> _AutomationFactsSnapshot:
        attention_target = enriched_facts.get("attention_target", {})
        ambiguous_room_guard = enriched_facts.get("ambiguous_room_guard", {})
        known_user_hint = enriched_facts.get("known_user_hint", {})
        affect_proxy = enriched_facts.get("affect_proxy", {})
        vad = enriched_facts.get("vad", {})

        safety_concern_active = bool(vad.get("distress_detected")) or affect_proxy.get("state") == "concern_cue"
        interaction_ready = bool(attention_target.get("session_focus_active")) and not bool(
            ambiguous_room_guard.get("guard_active")
        )
        calm_personalization_allowed = (
            interaction_ready
            and bool(known_user_hint.get("matches_main_user"))
            and not safety_concern_active
        )
        return _AutomationFactsSnapshot(
            {
                "interaction_ready": interaction_ready,
                "safety_concern_active": safety_concern_active,
                "calm_personalization_allowed": calm_personalization_allowed,
                "available": False,
                "degraded": True,
                "degrade_reason": "person_state_unavailable",
            }
        )

    def _as_flag(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            return normalized in {"1", "true", "yes", "on", "active"}
        return bool(value)

    def _nested_get(self, payload: Mapping[str, Any] | None, *path: str) -> Any:
        current: Any = payload
        for key in path:
            if not isinstance(current, Mapping):
                return None
            current = current.get(key)
        return current

    def _normalize_external_event_names(self, event_names: Iterable[str]) -> tuple[str, ...]:
        max_events = int(
            self._cfg("automation_max_external_events", _DEFAULT_EXTERNAL_EVENT_LIMIT)
            or _DEFAULT_EXTERNAL_EVENT_LIMIT
        )
        allowed_prefixes = tuple(self._cfg("automation_allowed_event_prefixes", ()) or ())
        normalized: list[str] = []
        for raw_name in event_names:
            if len(normalized) >= max_events:
                break
            if not isinstance(raw_name, str):
                continue
            event_name = raw_name.strip()
            if not event_name or not _SAFE_EVENT_NAME_RE.fullmatch(event_name):
                _LOGGER.warning("dropping invalid automation event name")
                continue
            if allowed_prefixes and not event_name.startswith(allowed_prefixes):
                _LOGGER.warning("dropping non-allowlisted automation event name")
                continue
            normalized.append(event_name)
        return tuple(self._dedupe_strings(normalized))

    def _dedupe_strings(self, values: Sequence[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for value in values:
            if value not in seen:
                seen.add(value)
                deduped.append(value)
        return deduped

    def _sanitize_automation_value(self, value: Any, *, depth: int = 0) -> Any:
        max_depth = int(self._cfg("automation_facts_max_depth", _DEFAULT_FACT_DEPTH_LIMIT) or _DEFAULT_FACT_DEPTH_LIMIT)
        max_items = int(self._cfg("automation_facts_max_items", _DEFAULT_FACT_ITEMS_LIMIT) or _DEFAULT_FACT_ITEMS_LIMIT)
        max_string = int(
            self._cfg("automation_facts_max_string_len", _DEFAULT_FACT_STRING_LIMIT) or _DEFAULT_FACT_STRING_LIMIT
        )

        if depth >= max_depth:
            return {"truncated": True}

        if value is None or isinstance(value, (bool, int)):
            return value

        if isinstance(value, float):
            return value if math.isfinite(value) else None

        if isinstance(value, str):
            return value if len(value) <= max_string else f"{value[: max_string - 3]}..."

        if isinstance(value, bytes):
            text = value.hex()
            return text if len(text) <= max_string else f"{text[: max_string - 3]}..."

        if isinstance(value, Mapping):
            sanitized: dict[str, Any] = {}
            for index, (key, child) in enumerate(value.items()):
                if index >= max_items:
                    sanitized["__truncated__"] = True
                    break
                sanitized[str(key)] = self._sanitize_automation_value(child, depth=depth + 1)
            return sanitized

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            sanitized_items: list[Any] = []
            for index, item in enumerate(value):
                if index >= max_items:
                    sanitized_items.append({"truncated": True})
                    break
                sanitized_items.append(self._sanitize_automation_value(item, depth=depth + 1))
            return sanitized_items

        if hasattr(value, "to_automation_facts") and callable(value.to_automation_facts):
            return self._sanitize_automation_value(value.to_automation_facts(), depth=depth + 1)

        return self._sanitize_automation_value(str(value), depth=depth + 1)

    def _redact_session_id(self, session_id: Any) -> Any:
        if self._cfg("automation_export_raw_session_id", False):
            return session_id
        if session_id in (None, ""):
            return None
        text = str(session_id)
        hash_key = self._cfg("automation_session_id_hash_key", None)
        if hash_key not in (None, ""):
            digest = hashlib.blake2s(
                text.encode("utf-8", errors="ignore"),
                key=str(hash_key).encode("utf-8", errors="ignore")[:32],
                digest_size=8,
            ).hexdigest()
        else:
            digest = hashlib.blake2s(text.encode("utf-8", errors="ignore"), digest_size=8).hexdigest()
        return f"sid:{digest}"

    def _null_audio_observation(self):
        class _NullAudioObservation:
            speech_detected = None
            distress_detected = False
            room_quiet = None
            recent_speech_age_s = None
            assistant_output_active = None
            signal_source = None
            device_runtime_mode = None
            host_control_ready = None
            transport_reason = None
            azimuth_deg = None
            direction_confidence = None
            non_speech_audio_likely = None
            background_media_likely = None
            speech_overlap_likely = None
            barge_in_detected = None
            mute_active = None

        return _NullAudioObservation()