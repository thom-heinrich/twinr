# CHANGELOG: 2026-03-27
# BUG-1: Fixed a race in lazy lock/singleton initialization that could create multiple locks or duplicate
#        policy helper instances under concurrent background threads.
# BUG-2: Fixed finalize_* paths so status emission still happens when runtime.finish_* raises, preventing
#        stuck "answering"/"printing" state propagation in background delivery.
# BUG-3: Fixed local timezone validation to handle malformed or oversized zone keys that can raise OSError,
#        and cache the validated ZoneInfo object to avoid repeated lookup churn.
# SEC-1: Stopped emitting raw exception text and unsanitized reasons to outward-facing channels; helper-
#        generated emits are now normalized and length-bounded to reduce information leakage and log injection.
# SEC-2: Bounded and pruned automation failure backoff state so unique failing automation IDs cannot grow
#        memory without bound on Raspberry Pi deployments.
# IMP-1: Hardened compound shared-state mutations with explicit locks so behavior stays correct on modern
#        free-threaded Python builds as well as traditional GIL-enabled builds.
# IMP-2: Upgraded background fault retention to a bounded ring-buffer mirror and added thread-safe cached
#        instance helpers to reduce allocation churn on proactive background paths.

"""Shared support helpers for realtime background delivery."""

# mypy: ignore-errors

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from math import isfinite
from threading import RLock
import time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from twinr.agent.tools.runtime.broker_policy import (
    AutomationToolBrokerPolicy,
    default_automation_tool_broker_policy,
)
from twinr.automations import AutomationAction
from twinr.agent.workflows.realtime_runtime.background_delivery import (
    background_block_reason,
    background_block_reason_locked,
    background_work_allowed,
    begin_background_delivery,
)
from twinr.agent.workflows.realtime_runtime.proactive_delivery import (
    ProactiveDeliveryPolicy,
)
from twinr.agent.workflows.realtime_runtime.reminder_delivery import (
    default_due_reminder_text,
    safe_format_due_label,
)
from twinr.proactive.governance.governor import ProactiveGovernorCandidate, ProactiveGovernorReservation
from twinr.proactive.runtime.audio_policy import ReSpeakerAudioPolicySnapshot
from twinr.proactive.runtime.display_reserve_companion_planner import DisplayReserveCompanionPlanner
from twinr.proactive.runtime.display_social_reserve import DisplaySocialReservePublisher
from twinr.proactive.runtime.governor_inputs import ReSpeakerGovernorInputs, build_respeaker_governor_inputs
from twinr.proactive.runtime.multimodal_initiative import ReSpeakerMultimodalInitiativeSnapshot
from twinr.proactive.social.engine import SocialTriggerDecision
from twinr.proactive.runtime.smart_home_context import SmartHomeContextTracker


_BACKGROUND_SUPPORT_BOOTSTRAP_LOCK = RLock()
_BACKGROUND_FAULT_LIMIT = 20
_DEFAULT_AUTOMATION_BACKOFF_STATE_LIMIT = 256
_MAX_PUBLIC_EMIT_VALUE_CHARS = 160
_MAX_DIAGNOSTIC_TEXT_CHARS = 512


def _is_lock_like(value: object) -> bool:
    return callable(getattr(value, "acquire", None)) and callable(getattr(value, "release", None))


class BackgroundSupportMixin:
    """Provide shared safety, coercion, and policy helpers."""

    def _get_lock(self, name: str) -> RLock:
        """Return one lazily created re-entrant lock for background state."""

        lock = getattr(self, name, None)
        if _is_lock_like(lock):
            return lock

        with _BACKGROUND_SUPPORT_BOOTSTRAP_LOCK:
            lock = getattr(self, name, None)
            if _is_lock_like(lock):
                return lock
            lock = RLock()
            setattr(self, name, lock)
            return lock

    def _get_cached_instance_of_type(self, attr_name: str, expected_type, factory):
        """Return one lazily created cached instance with per-attribute locking."""

        value = getattr(self, attr_name, None)
        if isinstance(value, expected_type):
            return value

        with self._get_lock(f"{attr_name}_lock"):
            value = getattr(self, attr_name, None)
            if isinstance(value, expected_type):
                return value
            value = factory()
            setattr(self, attr_name, value)
            return value

    def _normalize_diagnostic_text(self, value: object, *, max_length: int = _MAX_DIAGNOSTIC_TEXT_CHARS) -> str:
        text = "" if value is None else str(value)
        if not text:
            return ""
        text = text.replace("\r", "\\r").replace("\n", "\\n")
        if len(text) > max_length:
            return f"{text[: max(0, max_length - 3)]}..."
        return text

    def _public_emit_value(self, value: object, *, default: str = "unknown") -> str:
        if isinstance(value, BaseException):
            text = type(value).__name__
        else:
            text = self._coerce_text(value)
        if not text:
            return default

        normalized: list[str] = []
        for char in text:
            if char.isalnum() or char in {"_", "-", ".", ":"}:
                normalized.append(char)
            elif char.isspace() or char in {"=", ",", ";", "/", "\\"}:
                normalized.append("_")
        token = "".join(normalized).strip("_")
        if not token:
            token = default
        if len(token) > _MAX_PUBLIC_EMIT_VALUE_CHARS:
            token = token[:_MAX_PUBLIC_EMIT_VALUE_CHARS]
        return token

    # BREAKING: helper-generated outward emits now normalize values into bounded tokens instead of
    # exposing raw exception or reason strings. This intentionally trades exact text fidelity for
    # safer, machine-parseable telemetry on shared household devices.
    def _safe_emit_key_value(self, key: str, value: object, *, default: str = "unknown") -> None:
        self._safe_emit(f"{key}={self._public_emit_value(value, default=default)}")

    def _background_fault_ring(self) -> deque[dict[str, str]]:
        ring = getattr(self, "_background_fault_ring_store", None)
        if isinstance(ring, deque) and ring.maxlen == _BACKGROUND_FAULT_LIMIT:
            return ring

        with self._get_lock("_background_faults_lock"):
            ring = getattr(self, "_background_fault_ring_store", None)
            if isinstance(ring, deque) and ring.maxlen == _BACKGROUND_FAULT_LIMIT:
                return ring

            legacy_faults = getattr(self, "_background_faults", None)
            seed = legacy_faults[-_BACKGROUND_FAULT_LIMIT:] if isinstance(legacy_faults, list) else ()
            ring = deque(seed, maxlen=_BACKGROUND_FAULT_LIMIT)
            setattr(self, "_background_fault_ring_store", ring)
            legacy_faults_list = legacy_faults if isinstance(legacy_faults, list) else []
            legacy_faults_list.clear()
            legacy_faults_list.extend(ring)
            setattr(self, "_background_faults", legacy_faults_list)
            return ring

    def _remember_background_fault(self, source: str, error: Exception | str) -> None:
        entry = {
            "source": self._normalize_diagnostic_text(source, max_length=128) or "unknown",
            "error": self._normalize_diagnostic_text(error),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._get_lock("_background_faults_lock"):
            ring = self._background_fault_ring()
            ring.append(entry)
            # Maintain the legacy list mirror for drop-in compatibility with callers
            # that read `_background_faults` directly.
            legacy_faults = getattr(self, "_background_faults", None)
            if not isinstance(legacy_faults, list):
                legacy_faults = []
                setattr(self, "_background_faults", legacy_faults)
            legacy_faults.clear()
            legacy_faults.extend(ring)

    def _safe_emit(self, message: str) -> None:
        try:
            self.emit(message)
        except Exception as exc:
            self._remember_background_fault("emit", exc)

    def _safe_record_event(self, event: str, message: str, **kwargs: object) -> None:
        try:
            self._record_event(event, message, **kwargs)
        except Exception as exc:
            self._remember_background_fault(f"record_event:{event}", exc)

    def _safe_record_usage(self, **kwargs: object) -> None:
        try:
            self._record_usage(**kwargs)
        except Exception as exc:
            self._remember_background_fault("record_usage", exc)

    def _safe_emit_status(self, *, force: bool = False) -> None:
        try:
            self._emit_status(force=force)
        except Exception as exc:
            self._remember_background_fault("emit_status", exc)

    def _safe_enqueue_multimodal_evidence(self, **kwargs: object) -> None:
        try:
            self.runtime.long_term_memory.enqueue_multimodal_evidence(**kwargs)
        except Exception as exc:
            self._remember_background_fault("enqueue_multimodal_evidence", exc)

    def _safe_run_proactive_follow_up(self, trigger: SocialTriggerDecision) -> bool:
        try:
            return bool(self._run_proactive_follow_up(trigger))
        except Exception as exc:
            self._remember_background_fault("run_proactive_follow_up", exc)
            self._safe_emit_key_value("social_follow_up_error", exc, default="internal_error")
            self._safe_record_event(
                "social_trigger_follow_up_failed",
                "A proactive social follow-up failed after the prompt was already delivered.",
                level="error",
                trigger=getattr(trigger, "trigger_id", None),
                error_class=type(exc).__name__,
                error=self._normalize_diagnostic_text(exc),
            )
            return False

    def _proactive_delivery_policy(self) -> ProactiveDeliveryPolicy:
        """Return the cached proactive delivery policy instance."""

        return self._get_cached_instance_of_type(
            "_proactive_delivery_policy_instance",
            ProactiveDeliveryPolicy,
            lambda: ProactiveDeliveryPolicy.from_config(self.config),
        )

    def _display_social_reserve_publisher(self) -> DisplaySocialReservePublisher:
        """Return the cached reserve-lane publisher for display-first prompts."""

        return self._get_cached_instance_of_type(
            "_display_social_reserve_publisher_instance",
            DisplaySocialReservePublisher,
            lambda: DisplaySocialReservePublisher.from_config(self.config),
        )

    def _display_reserve_companion_planner(self) -> DisplayReserveCompanionPlanner:
        """Return the cached nightly reserve-lane planner."""

        return self._get_cached_instance_of_type(
            "_display_reserve_companion_planner_instance",
            DisplayReserveCompanionPlanner,
            lambda: DisplayReserveCompanionPlanner.from_config(self.config),
        )

    def _finalize_speaking_output(self) -> None:
        try:
            self.runtime.finish_speaking()
        except Exception as exc:
            self._remember_background_fault("finish_speaking", exc)
            if getattr(getattr(self.runtime, "status", None), "value", None) == "answering":
                self._recover_speaking_output_state()
                return
        self._safe_emit_status(force=True)

    def _recover_speaking_output_state(self) -> None:
        if getattr(getattr(self.runtime, "status", None), "value", None) != "answering":
            return
        try:
            self.runtime.finish_speaking()
        except Exception as exc:
            self._remember_background_fault("finish_speaking", exc)
        self._safe_emit_status(force=True)

    def _finalize_printing_output(self) -> None:
        try:
            self.runtime.finish_printing()
        except Exception as exc:
            self._remember_background_fault("finish_printing", exc)
            if getattr(getattr(self.runtime, "status", None), "value", None) == "printing":
                self._recover_printing_output_state()
                return
        self._safe_emit_status(force=True)

    def _recover_printing_output_state(self) -> None:
        if getattr(getattr(self.runtime, "status", None), "value", None) != "printing":
            return
        try:
            self.runtime.finish_printing()
        except Exception as exc:
            self._remember_background_fault("finish_printing", exc)
        self._safe_emit_status(force=True)

    def _safe_cancel_governor_reservation(self, reservation: ProactiveGovernorReservation) -> None:
        try:
            self.runtime.proactive_governor.cancel(reservation)
        except Exception as exc:
            self._remember_background_fault("governor_cancel", exc)

    def _safe_mark_governor_delivered(self, reservation: ProactiveGovernorReservation) -> None:
        try:
            self.runtime.proactive_governor.mark_delivered(reservation)
        except Exception as exc:
            self._remember_background_fault("governor_mark_delivered", exc)

    def _safe_mark_governor_skipped(
        self,
        reservation: ProactiveGovernorReservation,
        *,
        reason: str,
    ) -> None:
        try:
            self.runtime.proactive_governor.mark_skipped(reservation, reason=reason)
        except Exception as exc:
            self._remember_background_fault("governor_mark_skipped", exc)

    def _safe_mark_long_term_proactive_candidate_skipped(self, reservation, *, reason: str) -> None:
        try:
            self.runtime.mark_long_term_proactive_candidate_skipped(reservation, reason=reason)
        except Exception as exc:
            self._remember_background_fault("mark_long_term_proactive_candidate_skipped", exc)

    def _safe_mark_reminder_failed(self, reminder_id: str, *, error: str) -> None:
        try:
            self.runtime.mark_reminder_failed(reminder_id, error=error)
        except Exception as exc:
            self._remember_background_fault("mark_reminder_failed", exc)

    def _safe_release_reminder_reservation(self, reminder_id: str) -> None:
        try:
            self.runtime.release_reminder_reservation(reminder_id)
        except Exception as exc:
            self._remember_background_fault("release_reminder_reservation", exc)

    def _smart_home_context_tracker(self) -> SmartHomeContextTracker:
        """Return the cached smart-home context tracker instance."""

        return self._get_cached_instance_of_type(
            "_smart_home_context_tracker_instance",
            SmartHomeContextTracker,
            lambda: SmartHomeContextTracker.from_config(self.config),
        )

    def _automation_tool_broker_policy(self) -> AutomationToolBrokerPolicy:
        return self._get_cached_instance_of_type(
            "_automation_tool_broker_policy_instance",
            AutomationToolBrokerPolicy,
            default_automation_tool_broker_policy,
        )

    def _config_interval_seconds(self, attr_name: str, default: float) -> float:
        raw_value = getattr(self.config, attr_name, default)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            self._remember_background_fault(f"config_interval:{attr_name}", f"invalid interval {raw_value!r}")
            return max(0.25, default)
        if not isfinite(value) or value <= 0.0:
            self._remember_background_fault(f"config_interval:{attr_name}", f"invalid interval {raw_value!r}")
            return max(0.25, default)
        return max(0.25, value)

    def _monotonic_deadline(self, attr_name: str) -> float:
        raw_value = getattr(self, attr_name, 0.0)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return 0.0
        if not isfinite(value):
            return 0.0
        return value

    def _local_timezone_name(self) -> str:
        configured = str(getattr(self.config, "local_timezone_name", "") or "").strip()
        effective_name = configured or "UTC"

        cached_key = getattr(self, "_local_timezone_cache_key", None)
        cached_name = getattr(self, "_local_timezone_cache_name", None)
        if cached_key == effective_name and isinstance(cached_name, str):
            return cached_name

        with self._get_lock("_local_timezone_cache_lock"):
            cached_key = getattr(self, "_local_timezone_cache_key", None)
            cached_name = getattr(self, "_local_timezone_cache_name", None)
            if cached_key == effective_name and isinstance(cached_name, str):
                return cached_name

            try:
                zone = ZoneInfo(effective_name)
            except (ZoneInfoNotFoundError, ValueError, OSError) as exc:
                setattr(self, "_local_timezone_cache_key", effective_name)
                setattr(self, "_local_timezone_cache_name", "UTC")
                setattr(self, "_local_timezone_cache_zoneinfo", ZoneInfo("UTC"))

                if configured:
                    notified_for = getattr(self, "_invalid_local_timezone_notified_for", None)
                    if notified_for != configured:
                        setattr(self, "_invalid_local_timezone_notified_for", configured)
                        self._remember_background_fault("local_timezone_name", exc)
                        self._safe_record_event(
                            "invalid_local_timezone_fallback",
                            "Twinr fell back to UTC because the configured local timezone name was invalid.",
                            level="warning",
                            configured_timezone=configured,
                            fallback_timezone="UTC",
                            error_class=type(exc).__name__,
                            error=self._normalize_diagnostic_text(exc),
                        )
                return "UTC"

            setattr(self, "_local_timezone_cache_key", effective_name)
            setattr(self, "_local_timezone_cache_name", effective_name)
            setattr(self, "_local_timezone_cache_zoneinfo", zone)
            if configured:
                setattr(self, "_invalid_local_timezone_notified_for", None)
            return effective_name

    def _local_timezone(self) -> ZoneInfo:
        effective_name = self._local_timezone_name()
        cached_key = getattr(self, "_local_timezone_cache_key", None)
        cached_zone = getattr(self, "_local_timezone_cache_zoneinfo", None)
        if cached_key == effective_name and isinstance(cached_zone, ZoneInfo):
            return cached_zone

        with self._get_lock("_local_timezone_cache_lock"):
            cached_key = getattr(self, "_local_timezone_cache_key", None)
            cached_zone = getattr(self, "_local_timezone_cache_zoneinfo", None)
            if cached_key == effective_name and isinstance(cached_zone, ZoneInfo):
                return cached_zone
            zone = ZoneInfo(effective_name)
            setattr(self, "_local_timezone_cache_key", effective_name)
            setattr(self, "_local_timezone_cache_name", effective_name)
            setattr(self, "_local_timezone_cache_zoneinfo", zone)
            return zone

    def _coerce_text(self, value: object) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _require_non_empty_text(self, value: object, *, context: str) -> str:
        text = self._coerce_text(value)
        if text:
            return text
        raise RuntimeError(f"{context} is empty")

    def _coerce_bool(self, value: object, *, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off", ""}:
                return False
        return default

    def _coerce_priority(
        self,
        value: object,
        *,
        default: int,
        minimum: int = 1,
        maximum: int = 99,
    ) -> int:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        if not isfinite(number):
            return default
        return max(minimum, min(maximum, int(number)))

    def _confidence_to_priority(self, confidence: object, *, default: int = 50) -> int:
        try:
            scaled = float(confidence) * 100.0
        except (TypeError, ValueError):
            return default
        if not isfinite(scaled):
            return default
        return self._coerce_priority(scaled, default=default)

    def _automation_payload(self, action: AutomationAction) -> dict[str, object]:
        payload = getattr(action, "payload", None)
        return payload if isinstance(payload, dict) else {}

    def _default_due_reminder_text(self, reminder) -> str:
        return default_due_reminder_text(self, reminder)

    def _safe_format_due_label(self, value: object) -> str:
        return safe_format_due_label(self, value)

    def _automation_failure_backoff_seconds(self) -> float:
        default = max(30.0, self._config_interval_seconds("automation_poll_interval_s", 5.0) * 2.0)
        raw_value = getattr(self.config, "automation_failure_retry_backoff_s", default)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return default
        if not isfinite(value) or value < 1.0:
            return default
        return min(value, 3600.0)

    def _automation_failure_backoff_capacity(self) -> int:
        raw_value = getattr(
            self.config,
            "automation_failure_backoff_state_max_entries",
            _DEFAULT_AUTOMATION_BACKOFF_STATE_LIMIT,
        )
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            return _DEFAULT_AUTOMATION_BACKOFF_STATE_LIMIT
        return max(32, min(value, 4096))

    def _automation_failure_backoff_map_locked(self) -> dict[str, float]:
        state = getattr(self, "_automation_failure_backoff_until", None)
        if isinstance(state, dict):
            return state
        state = {}
        setattr(self, "_automation_failure_backoff_until", state)
        return state

    def _prune_automation_failure_backoff_locked(self, *, now: float | None = None) -> dict[str, float]:
        backoff_map = self._automation_failure_backoff_map_locked()
        current = time.monotonic() if now is None else now

        expired_ids = [automation_id for automation_id, until in backoff_map.items() if not isfinite(until) or until <= current]
        for automation_id in expired_ids:
            backoff_map.pop(automation_id, None)

        max_entries = self._automation_failure_backoff_capacity()
        while len(backoff_map) > max_entries:
            oldest_automation_id = next(iter(backoff_map), None)
            if oldest_automation_id is None:
                break
            backoff_map.pop(oldest_automation_id, None)

        return backoff_map

    def _automation_failure_backoff_map(self) -> dict[str, float]:
        with self._get_lock("_automation_failure_backoff_lock"):
            return self._prune_automation_failure_backoff_locked()

    def _automation_retry_blocked(self, automation_id: str) -> bool:
        current = time.monotonic()
        with self._get_lock("_automation_failure_backoff_lock"):
            backoff_map = self._prune_automation_failure_backoff_locked(now=current)
            until = backoff_map.get(automation_id)
            return until is not None and isfinite(until) and current < until

    def _register_automation_failure_backoff(self, automation_id: str) -> None:
        if not automation_id:
            return
        current = time.monotonic()
        with self._get_lock("_automation_failure_backoff_lock"):
            backoff_map = self._prune_automation_failure_backoff_locked(now=current)
            backoff_map[automation_id] = current + self._automation_failure_backoff_seconds()
            self._prune_automation_failure_backoff_locked(now=current)

    def _clear_automation_failure_backoff(self, automation_id: str) -> None:
        with self._get_lock("_automation_failure_backoff_lock"):
            self._automation_failure_backoff_map_locked().pop(automation_id, None)

    def _background_block_reason_locked(self) -> str | None:
        return background_block_reason_locked(self)

    def _background_block_reason(self) -> str | None:
        return background_block_reason(self)

    def _background_work_allowed(self) -> bool:
        return background_work_allowed(self)

    def _begin_background_delivery(self, action):
        return begin_background_delivery(self, action)

    def _current_presence_session_id(self) -> int | None:
        monitor = getattr(self, "proactive_monitor", None)
        coordinator = None if monitor is None else getattr(monitor, "coordinator", None)
        snapshot = None if coordinator is None else getattr(coordinator, "latest_presence_snapshot", None)
        if snapshot is None or not getattr(snapshot, "armed", False):
            return None
        session_id = getattr(snapshot, "session_id", None)
        try:
            return int(session_id) if session_id is not None else None
        except (TypeError, ValueError):
            return None

    def _current_audio_policy_snapshot(self) -> ReSpeakerAudioPolicySnapshot | None:
        """Return the latest proactive audio-policy snapshot when available."""

        monitor = getattr(self, "proactive_monitor", None)
        coordinator = None if monitor is None else getattr(monitor, "coordinator", None)
        snapshot = None if coordinator is None else getattr(coordinator, "latest_audio_policy_snapshot", None)
        if isinstance(snapshot, ReSpeakerAudioPolicySnapshot):
            return snapshot
        return None

    def _current_multimodal_initiative_snapshot(self) -> ReSpeakerMultimodalInitiativeSnapshot | None:
        """Return the latest multimodal initiative snapshot when available."""

        facts = getattr(self, "_latest_sensor_observation_facts", None)
        if not isinstance(facts, dict):
            return None
        return ReSpeakerMultimodalInitiativeSnapshot.from_fact_map(
            facts.get("multimodal_initiative"),
        )

    def _current_governor_inputs(self, *, requested_channel: str) -> ReSpeakerGovernorInputs:
        """Return the current ReSpeaker-aware governor input bundle."""

        monitor = getattr(self, "proactive_monitor", None)
        coordinator = None if monitor is None else getattr(monitor, "coordinator", None)
        presence_snapshot = None if coordinator is None else getattr(coordinator, "latest_presence_snapshot", None)
        return build_respeaker_governor_inputs(
            requested_channel=requested_channel,
            presence_snapshot=presence_snapshot,
            audio_policy_snapshot=self._current_audio_policy_snapshot(),
            multimodal_initiative_snapshot=self._current_multimodal_initiative_snapshot(),
        )

    def _local_now(self) -> datetime:
        """Return the current wall clock in the configured local timezone."""

        return datetime.now(self._local_timezone())

    def _reserve_governed_prompt(
        self,
        candidate: ProactiveGovernorCandidate,
        *,
        governor_inputs: ReSpeakerGovernorInputs | None = None,
    ) -> ProactiveGovernorReservation | None:
        emit_prefix = {
            "social": "social_trigger",
            "longterm": "longterm_proactive",
            "reminder": "reminder",
            "automation": "automation",
        }.get(candidate.source_kind, candidate.source_kind)

        try:
            decision = self.runtime.proactive_governor.try_reserve(candidate)
        except Exception as exc:
            self._safe_emit_key_value(f"{emit_prefix}_reservation_error", exc, default="internal_error")
            self._safe_record_event(
                "proactive_governor_error",
                "The shared proactive governor raised an error while evaluating a delivery candidate.",
                level="error",
                source_kind=candidate.source_kind,
                source_id=candidate.source_id,
                summary=candidate.summary,
                error_class=type(exc).__name__,
                error=self._normalize_diagnostic_text(exc),
                **({} if governor_inputs is None else governor_inputs.event_data()),
            )
            return None

        if decision.allowed:
            reservation = getattr(decision, "reservation", None)
            if reservation is not None:
                return reservation
            self._remember_background_fault(
                "proactive_governor_contract",
                "Governor allowed a reservation but returned no reservation object.",
            )
            self._safe_emit_key_value(f"{emit_prefix}_reservation_error", "missing_reservation", default="internal_error")
            self._safe_record_event(
                "proactive_governor_contract_error",
                "The shared proactive governor allowed delivery without returning a reservation object.",
                level="error",
                source_kind=candidate.source_kind,
                source_id=candidate.source_id,
                summary=candidate.summary,
                **({} if governor_inputs is None else governor_inputs.event_data()),
            )
            return None

        self._safe_emit_key_value(f"{emit_prefix}_skipped", decision.reason, default="blocked")
        self._safe_record_event(
            "proactive_governor_blocked",
            "Proactive delivery was blocked by the shared governor policy.",
            source_kind=candidate.source_kind,
            source_id=candidate.source_id,
            summary=candidate.summary,
            reason=self._normalize_diagnostic_text(decision.reason, max_length=160),
            channel=getattr(candidate, "channel", None),
            priority=int(candidate.priority),
            presence_session_id=candidate.presence_session_id,
            **({} if governor_inputs is None else governor_inputs.event_data()),
        )
        return None

    def _recent_proactive_prompts(
        self,
        *,
        trigger_id: str | None = None,
        limit: int = 3,
    ) -> tuple[str, ...]:
        prompts: list[str] = []
        try:
            bounded_limit = max(1, int(limit))
        except (TypeError, ValueError):
            bounded_limit = 3
        for entry in reversed(tuple(self.runtime.ops_events.tail(limit=100) or ())):
            if not isinstance(entry, dict):
                continue
            if entry.get("event") not in {"social_trigger_prompted", "longterm_proactive_prompted"}:
                continue
            data = entry.get("data")
            if not isinstance(data, dict):
                data = {}
            if trigger_id is not None and data.get("trigger") != trigger_id:
                continue
            prompt = self._coerce_text(data.get("prompt", ""))
            if not prompt:
                continue
            prompts.append(prompt)
            if len(prompts) >= bounded_limit:
                break
        prompts.reverse()
        return tuple(prompts)
