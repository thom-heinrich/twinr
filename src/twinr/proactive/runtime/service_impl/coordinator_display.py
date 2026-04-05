# CHANGELOG: 2026-03-29
# BUG-1: Refresh cadence now gates on a monotonic clock instead of wall time so NTP/RTC jumps cannot stall or burst the HDMI refresh loops.
# BUG-2: Last-refresh markers are now advanced only after a refresh path completes, so mid-pipeline faults no longer suppress the next retry window.
# BUG-3: Missing/malformed audio snapshots and missing gesture runtime payloads now degrade safely instead of crashing the coordinator tick.
# SEC-1: Added fault barriers and non-blocking per-route reentrancy locks so one bad observer/orchestrator/publisher result cannot take down the refresh path or duplicate side effects under concurrent ticks.
# IMP-1: Added freshness validation and missing-modality degradation for attention/gesture refreshes to avoid acting on stale sensor data.
# IMP-2: Added stale/over-budget wakeup suppression so gesture wakeups are not dispatched from expired context on slow or overloaded edge hardware.
# IMP-3: Added richer stage-level fault telemetry that preserves forensic tracing while containing component failures.

"""HDMI attention and gesture refresh paths for the proactive coordinator.

Purpose: keep the dedicated display-attention and gesture-refresh workflows in
one focused module so the main coordinator file stays centered on monitor ticks
and trigger policy.

Invariants: refresh cadence, runtime gating, forensic tracing, and publish
outcomes must stay behavior-identical to the legacy service implementation,
while downstream attention/gesture semantics now come from the shared runtime
perception orchestrator instead of lane-local temporal truth.
"""


from __future__ import annotations

from dataclasses import replace
from threading import Lock
from types import SimpleNamespace
from typing import Any, Callable, TypeVar
import math
import time

from twinr.agent.workflows.forensics import workflow_decision, workflow_event, workflow_span

from twinr.proactive.runtime.service_impl.compat import (
    _ATTENTION_REFRESH_AUDIO_CACHE_MAX_AGE_S,
    _display_attention_refresh_allowed_runtime_status,
)

from ...social.engine import SocialAudioObservation, SocialObservation, SocialVisionObservation
from ..audio_policy import ReSpeakerAudioPolicySnapshot
from ..display_attention import (
    display_attention_refresh_supported,
    resolve_display_attention_refresh_interval,
)
from ..display_gesture_emoji import (
    display_gesture_refresh_supported,
    resolve_display_gesture_refresh_interval,
)
from ..gesture_wakeup_lane import GestureWakeupDecision
from ..gesture_wakeup_priority import decide_gesture_wakeup_priority
from .. import service_attention_helpers
from .. import service_gesture_helpers


_T = TypeVar("_T")


class ProactiveCoordinatorDisplayMixin:
    """Provide the dedicated HDMI attention and gesture refresh workflows."""

    def refresh_display_attention(self) -> bool:
        """Refresh HDMI attention-follow from the cheap local camera path."""

        refresh_started_ns = time.monotonic_ns()
        stage_ms: dict[str, float] = {}
        self._last_display_attention_fusion_debug = None
        now: float | None = None
        runtime_status_value: str | None = None

        lock = self._get_refresh_route_lock("_display_attention_refresh_route_lock")
        if not lock.acquire(blocking=False):
            self._safe_record_attention_debug_tick(
                observed_at=self._safe_clock_now(),
                outcome="refresh_in_progress",
                runtime_status_value=None,
                stage_ms=stage_ms,
            )
            return False

        try:
            if not display_attention_refresh_supported(
                config=self.config,
                vision_observer=self.vision_observer,
            ):
                self._safe_record_attention_debug_tick(
                    observed_at=self._safe_clock_now(),
                    outcome="unsupported",
                    runtime_status_value=None,
                    stage_ms=stage_ms,
                )
                return False

            now = self._safe_clock_now()
            interval_s = resolve_display_attention_refresh_interval(self.config)
            if interval_s is None:
                self._safe_record_attention_debug_tick(
                    observed_at=now,
                    outcome="no_refresh_interval",
                    runtime_status_value=None,
                    stage_ms=stage_ms,
                )
                return False
            if self._refresh_interval_pending(
                monotonic_attr="_last_display_attention_refresh_monotonic_s",
                wall_attr="_last_display_attention_refresh_at",
                observed_at=now,
                interval_s=interval_s,
            ):
                return False
            if self._failure_backoff_pending(
                monotonic_attr="_last_display_attention_refresh_failure_monotonic_s",
                interval_s=interval_s,
                config_attr="display_attention_refresh_failure_backoff_s",
            ):
                return False

            ok, runtime_status_value = self._read_runtime_status_value(
                fault_event="proactive_display_attention_runtime_status_failed",
                fault_message="Failed to read runtime status for HDMI attention refresh.",
            )
            if not ok:
                self._mark_refresh_failure("_last_display_attention_refresh_failure_monotonic_s")
                self._safe_record_attention_debug_tick(
                    observed_at=now,
                    outcome="runtime_status_failed",
                    runtime_status_value=None,
                    stage_ms=stage_ms,
                )
                return False

            if not _display_attention_refresh_allowed_runtime_status(runtime_status_value):
                self._safe_record_attention_debug_tick(
                    observed_at=now,
                    outcome="runtime_status_blocked",
                    runtime_status_value=runtime_status_value,
                    stage_ms=stage_ms,
                )
                return False

            stage_started_ns = time.monotonic_ns()
            snapshot = self._fault_barrier(
                event="proactive_display_attention_vision_observe_failed",
                message="Failed to observe vision for HDMI attention refresh.",
                func=self._observe_vision_for_attention_refresh,
            )
            stage_ms["vision_observe"] = self._elapsed_ms(stage_started_ns)
            if snapshot is None:
                stage_ms["total"] = self._elapsed_ms(refresh_started_ns)
                self._mark_refresh_failure("_last_display_attention_refresh_failure_monotonic_s")
                self._safe_record_attention_debug_tick(
                    observed_at=now,
                    outcome="vision_snapshot_missing",
                    runtime_status_value=runtime_status_value,
                    stage_ms=stage_ms,
                )
                return False

            vision_observation = getattr(snapshot, "observation", None)
            if vision_observation is None:
                stage_ms["total"] = self._elapsed_ms(refresh_started_ns)
                self._mark_refresh_failure("_last_display_attention_refresh_failure_monotonic_s")
                self._safe_record_attention_debug_tick(
                    observed_at=now,
                    outcome="vision_observation_missing",
                    runtime_status_value=runtime_status_value,
                    stage_ms=stage_ms,
                )
                return False

            if self._snapshot_is_stale(
                snapshot=snapshot,
                observed_at=now,
                max_age_s=self._resolve_vision_snapshot_max_age_s(
                    kind="attention",
                    interval_s=interval_s,
                ),
            ):
                stage_ms["total"] = self._elapsed_ms(refresh_started_ns)
                # BREAKING: stale attention frames are now dropped instead of being published.
                self._mark_refresh_failure("_last_display_attention_refresh_failure_monotonic_s")
                self._safe_record_attention_debug_tick(
                    observed_at=now,
                    outcome="vision_snapshot_stale",
                    runtime_status_value=runtime_status_value,
                    stage_ms=stage_ms,
                )
                return False

            stage_started_ns = time.monotonic_ns()
            fused_vision = self._fault_barrier(
                event="proactive_display_attention_camera_fusion_failed",
                message="Failed to fuse camera observation for HDMI attention refresh.",
                func=lambda: self._fuse_display_attention_camera_observation(
                    observed_at=now,
                    observation=vision_observation,
                ),
            )
            stage_ms["camera_fusion"] = self._elapsed_ms(stage_started_ns)
            if fused_vision is None:
                stage_ms["total"] = self._elapsed_ms(refresh_started_ns)
                self._mark_refresh_failure("_last_display_attention_refresh_failure_monotonic_s")
                self._safe_record_attention_debug_tick(
                    observed_at=now,
                    outcome="camera_fusion_failed",
                    runtime_status_value=runtime_status_value,
                    stage_ms=stage_ms,
                )
                return False

            self._fault_barrier(
                event="proactive_display_attention_record_vision_snapshot_failed",
                message="Failed to record HDMI attention vision snapshot for forensics.",
                func=lambda: self._record_vision_snapshot_safe(snapshot),
                default=None,
            )

            stage_started_ns = time.monotonic_ns()
            camera_update = self._fault_barrier(
                event="proactive_display_attention_camera_surface_failed",
                message="Failed to project HDMI attention camera surface.",
                func=lambda: self._observe_display_attention_camera_surface(
                    SocialObservation(
                        observed_at=now,
                        inspected=True,
                        pir_motion_detected=False,
                        low_motion=False,
                        vision=fused_vision,
                        audio=SocialAudioObservation(),
                    ),
                    inspected=True,
                ),
            )
            stage_ms["camera_surface"] = self._elapsed_ms(stage_started_ns)
            camera_snapshot = getattr(camera_update, "snapshot", None)
            if camera_update is None or camera_snapshot is None:
                stage_ms["total"] = self._elapsed_ms(refresh_started_ns)
                self._mark_refresh_failure("_last_display_attention_refresh_failure_monotonic_s")
                self._safe_record_attention_debug_tick(
                    observed_at=now,
                    outcome="camera_surface_failed",
                    runtime_status_value=runtime_status_value,
                    stage_ms=stage_ms,
                )
                return False

            stage_started_ns = time.monotonic_ns()
            raw_audio_snapshot = self._fault_barrier(
                event="proactive_display_attention_audio_observe_failed",
                message="Failed to observe audio for HDMI attention refresh.",
                func=lambda: self._observe_audio_for_attention_refresh(now=now),
            )
            stage_ms["audio_observe"] = self._elapsed_ms(stage_started_ns)
            audio_snapshot, audio_degraded = self._normalize_attention_audio_snapshot(
                observed_at=now,
                interval_s=interval_s,
                audio_snapshot=raw_audio_snapshot,
            )

            stage_started_ns = time.monotonic_ns()
            audio_policy_snapshot = self._fault_barrier(
                event="proactive_display_attention_audio_policy_failed",
                message="Failed to derive audio policy for HDMI attention refresh.",
                func=lambda: self._observe_audio_policy(
                    now=now,
                    audio_observation=audio_snapshot.observation,
                ),
                default=None,
            )
            stage_ms["audio_policy"] = self._elapsed_ms(stage_started_ns)

            self._fault_barrier(
                event="proactive_display_attention_live_context_publish_failed",
                message="Failed to export HDMI attention live context.",
                func=lambda: self._publish_display_attention_live_context(
                    observed_at=now,
                    vision_observation=fused_vision,
                    camera_snapshot=camera_snapshot,
                    audio_snapshot=audio_snapshot,
                    audio_policy_snapshot=audio_policy_snapshot,
                ),
                default=None,
            )

            stage_started_ns = time.monotonic_ns()
            publish_result = self._fault_barrier(
                event="proactive_display_attention_follow_publish_failed",
                message="Failed to publish HDMI attention follow state.",
                func=lambda: self._update_display_attention_follow(
                    source="display_attention_refresh",
                    observed_at=now,
                    camera_snapshot=camera_snapshot,
                    audio_observation=audio_snapshot.observation,
                    audio_policy_snapshot=audio_policy_snapshot,
                ),
                default=None,
            )
            stage_ms["publish"] = self._elapsed_ms(stage_started_ns)

            stage_started_ns = time.monotonic_ns()
            self._fault_barrier(
                event="proactive_display_attention_debug_signals_failed",
                message="Failed to update HDMI attention debug signals.",
                func=lambda: self._update_display_debug_signals(camera_update),
                default=None,
            )
            stage_ms["debug_signals"] = self._elapsed_ms(stage_started_ns)
            stage_ms["total"] = self._elapsed_ms(refresh_started_ns)

            self._fault_barrier(
                event="proactive_display_attention_follow_change_record_failed",
                message="Failed to record HDMI attention follow change.",
                func=lambda: self._record_display_attention_follow_if_changed(
                    observed_at=now,
                    runtime_status_value=runtime_status_value,
                    camera_snapshot=camera_snapshot,
                    publish_result=publish_result,
                ),
                default=None,
            )

            self._safe_record_attention_debug_tick(
                observed_at=now,
                outcome=self._resolve_attention_outcome(
                    publish_result=publish_result,
                    audio_degraded=audio_degraded,
                ),
                runtime_status_value=runtime_status_value,
                camera_snapshot=camera_snapshot,
                audio_observation=audio_snapshot.observation,
                publish_result=publish_result,
                stage_ms=stage_ms,
            )
            self._mark_refresh_complete(
                monotonic_attr="_last_display_attention_refresh_monotonic_s",
                wall_attr="_last_display_attention_refresh_at",
                observed_at=now,
                failure_attr="_last_display_attention_refresh_failure_monotonic_s",
            )
            return True
        except Exception as exc:
            self._record_fault_safe(
                event="proactive_display_attention_refresh_failed",
                message="Unhandled exception in HDMI attention refresh.",
                error=exc,
            )
            self._mark_refresh_failure("_last_display_attention_refresh_failure_monotonic_s")
            fallback_now = now if now is not None else self._safe_clock_now()
            stage_ms["total"] = self._elapsed_ms(refresh_started_ns)
            self._safe_record_attention_debug_tick(
                observed_at=fallback_now,
                outcome="unhandled_exception",
                runtime_status_value=runtime_status_value,
                stage_ms=stage_ms,
            )
            return False
        finally:
            lock.release()

    def refresh_display_gesture_emoji(self) -> bool:
        """Refresh HDMI gesture acknowledgements from the dedicated gesture path."""

        refresh_started_ns = time.monotonic_ns()
        stage_ms: dict[str, float] = {}
        now: float | None = None
        runtime_status_value: str | None = None

        if self.display_gesture_emoji_publisher is None:
            return False

        lock = self._get_refresh_route_lock("_display_gesture_refresh_route_lock")
        if not lock.acquire(blocking=False):
            self._safe_record_gesture_debug_tick(
                observed_at=self._safe_clock_now(),
                outcome="refresh_in_progress",
                runtime_status_value=None,
                stage_ms=stage_ms,
            )
            return False

        try:
            if not display_gesture_refresh_supported(
                config=self.config,
                vision_observer=self.vision_observer,
            ):
                self._safe_record_gesture_debug_tick(
                    observed_at=self._safe_clock_now(),
                    outcome="unsupported",
                    runtime_status_value=None,
                    stage_ms=stage_ms,
                )
                return False

            now = self._safe_clock_now()
            interval_s = resolve_display_gesture_refresh_interval(self.config)
            if interval_s is None:
                self._safe_record_gesture_debug_tick(
                    observed_at=now,
                    outcome="no_refresh_interval",
                    runtime_status_value=None,
                    stage_ms=stage_ms,
                )
                return False
            if self._refresh_interval_pending(
                monotonic_attr="_last_display_gesture_refresh_monotonic_s",
                wall_attr="_last_display_gesture_refresh_at",
                observed_at=now,
                interval_s=interval_s,
            ):
                return False
            if self._failure_backoff_pending(
                monotonic_attr="_last_display_gesture_refresh_failure_monotonic_s",
                interval_s=interval_s,
                config_attr="display_gesture_refresh_failure_backoff_s",
            ):
                return False

            ok, runtime_status_value = self._read_runtime_status_value(
                fault_event="proactive_display_gesture_runtime_status_failed",
                fault_message="Failed to read runtime status for HDMI gesture refresh.",
            )
            if not ok:
                self._mark_refresh_failure("_last_display_gesture_refresh_failure_monotonic_s")
                self._safe_record_gesture_debug_tick(
                    observed_at=now,
                    outcome="runtime_status_failed",
                    runtime_status_value=None,
                    stage_ms=stage_ms,
                )
                return False

            if not _display_attention_refresh_allowed_runtime_status(runtime_status_value):
                self._safe_record_gesture_debug_tick(
                    observed_at=now,
                    outcome="runtime_status_blocked",
                    runtime_status_value=runtime_status_value,
                    stage_ms=stage_ms,
                )
                return False

            with self._gesture_forensics.bind_refresh(
                observed_at=now,
                runtime_status_value=runtime_status_value,
                vision_mode=self._last_gesture_vision_refresh_mode,
                refresh_interval_s=interval_s,
            ):
                workflow_decision(
                    msg="gesture_refresh_runtime_status_gate",
                    question="Should the dedicated HDMI gesture refresh run on this tick?",
                    selected={
                        "id": "allowed",
                        "summary": "The runtime state allows the dedicated gesture refresh to execute.",
                    },
                    options=[
                        {"id": "allowed", "summary": "Proceed with the dedicated gesture refresh."},
                        {"id": "blocked", "summary": "Skip the gesture refresh because the runtime state forbids it."},
                    ],
                    context={
                        "runtime_status": str(runtime_status_value or "").strip().lower() or None,
                        "refresh_interval_s": interval_s,
                    },
                    confidence="forensic",
                    guardrails=["display_gesture_runtime_status_gate"],
                    kpi_impact_estimate={"latency": "low", "user_feedback": "high"},
                )
                with workflow_span(
                    name="proactive_display_gesture_refresh",
                    kind="turn",
                    details={
                        "runtime_status": str(runtime_status_value or "").strip().lower() or None,
                        "refresh_interval_s": interval_s,
                    },
                ):
                    stage_started_ns = time.monotonic_ns()
                    with workflow_span(
                        name="proactive_display_gesture_observe_vision",
                        kind="io",
                        details={"vision_observer_type": type(self.vision_observer).__name__},
                    ):
                        snapshot = self._fault_barrier(
                            event="proactive_display_gesture_vision_observe_failed",
                            message="Failed to observe vision for HDMI gesture refresh.",
                            func=self._observe_vision_for_gesture_refresh,
                        )
                    stage_ms["vision_observe"] = self._elapsed_ms(stage_started_ns)
                    if snapshot is None:
                        stage_ms["total"] = self._elapsed_ms(refresh_started_ns)
                        workflow_event(
                            kind="branch",
                            msg="gesture_refresh_snapshot_missing",
                            details={"stage_ms": dict(stage_ms)},
                            reason={
                                "selected": {
                                    "id": "snapshot_missing",
                                    "justification": "The gesture refresh could not continue because the vision observer returned no snapshot.",
                                    "expected_outcome": "Abort the current refresh without touching HDMI gesture state.",
                                },
                                "options": [
                                    {"id": "snapshot_present", "summary": "Continue with the captured snapshot."},
                                    {"id": "snapshot_missing", "summary": "Abort because no snapshot was returned."},
                                ],
                                "confidence": "forensic",
                                "guardrails": ["vision_snapshot_required"],
                                "kpi_impact_estimate": {"latency": "low", "display_side_effect": "none"},
                            },
                        )
                        self._mark_refresh_failure("_last_display_gesture_refresh_failure_monotonic_s")
                        self._safe_record_gesture_debug_tick(
                            observed_at=now,
                            outcome="vision_snapshot_missing",
                            runtime_status_value=runtime_status_value,
                            stage_ms=stage_ms,
                        )
                        return False

                    vision_observation = getattr(snapshot, "observation", None)
                    captured_at = self._coerce_float(getattr(snapshot, "captured_at", None)) or now
                    if vision_observation is None:
                        stage_ms["total"] = self._elapsed_ms(refresh_started_ns)
                        workflow_event(
                            kind="branch",
                            msg="gesture_refresh_observation_missing",
                            details={"stage_ms": dict(stage_ms)},
                            reason={
                                "selected": {
                                    "id": "observation_missing",
                                    "justification": "The gesture refresh snapshot is missing its vision observation payload.",
                                    "expected_outcome": "Abort the refresh to avoid invalid HDMI gesture or wakeup decisions.",
                                },
                                "options": [
                                    {"id": "observation_present", "summary": "Continue with the current vision observation."},
                                    {"id": "observation_missing", "summary": "Abort because the vision observation payload is missing."},
                                ],
                                "confidence": "forensic",
                                "guardrails": ["vision_observation_required"],
                                "kpi_impact_estimate": {"latency": "low", "display_side_effect": "none"},
                            },
                        )
                        self._mark_refresh_failure("_last_display_gesture_refresh_failure_monotonic_s")
                        self._safe_record_gesture_debug_tick(
                            observed_at=now,
                            outcome="vision_observation_missing",
                            runtime_status_value=runtime_status_value,
                            stage_ms=stage_ms,
                        )
                        return False

                    if self._snapshot_is_stale(
                        snapshot=snapshot,
                        observed_at=now,
                        max_age_s=self._resolve_vision_snapshot_max_age_s(
                            kind="gesture",
                            interval_s=interval_s,
                        ),
                    ):
                        stage_ms["total"] = self._elapsed_ms(refresh_started_ns)
                        workflow_event(
                            kind="branch",
                            msg="gesture_refresh_snapshot_stale",
                            details={"stage_ms": dict(stage_ms)},
                            reason={
                                "selected": {
                                    "id": "snapshot_stale",
                                    "justification": "The gesture refresh snapshot is older than the configured freshness budget.",
                                    "expected_outcome": "Abort the refresh to avoid stale HDMI or wakeup side effects.",
                                },
                                "options": [
                                    {"id": "snapshot_fresh", "summary": "Continue with the current snapshot."},
                                    {"id": "snapshot_stale", "summary": "Abort because the snapshot is stale."},
                                ],
                                "confidence": "forensic",
                                "guardrails": ["fresh_vision_snapshot_required"],
                                "kpi_impact_estimate": {"latency": "medium", "display_side_effect": "low"},
                            },
                        )
                        # BREAKING: stale gesture frames are now dropped instead of being published.
                        self._mark_refresh_failure("_last_display_gesture_refresh_failure_monotonic_s")
                        self._safe_record_gesture_debug_tick(
                            observed_at=now,
                            outcome="vision_snapshot_stale",
                            runtime_status_value=runtime_status_value,
                            stage_ms=stage_ms,
                        )
                        return False

                    self._fault_barrier(
                        event="proactive_display_gesture_record_vision_snapshot_failed",
                        message="Failed to record HDMI gesture vision snapshot for forensics.",
                        func=lambda: self._record_vision_snapshot_safe(snapshot),
                        default=None,
                    )
                    self._fault_barrier(
                        event="proactive_display_gesture_semantics_remember_failed",
                        message="Failed to remember gesture-derived display attention semantics.",
                        func=lambda: self._remember_display_attention_camera_semantics(
                            observed_at=now,
                            observation=vision_observation,
                            source="gesture",
                        ),
                        default=None,
                    )
                    workflow_event(
                        kind="io",
                        msg="gesture_refresh_observation",
                        details=(
                            self._fault_barrier(
                                event="proactive_display_gesture_observation_trace_failed",
                                message="Failed to derive gesture observation trace details.",
                                func=lambda: self._gesture_observation_trace_details(vision_observation),
                                default={},
                            )
                            or {}
                        ),
                    )

                    stage_started_ns = time.monotonic_ns()
                    with workflow_span(
                        name="proactive_display_gesture_perception_orchestrator",
                        kind="decision",
                    ):
                        perception_runtime = self._fault_barrier(
                            event="proactive_display_gesture_perception_failed",
                            message="Failed to derive gesture semantics from the perception orchestrator.",
                            func=lambda: self.perception_orchestrator.observe_gesture(
                                observed_at=now,
                                source=self._last_gesture_vision_refresh_mode,
                                captured_at=captured_at,
                                vision_observation=vision_observation,
                            ),
                        )
                    stage_ms["gesture_orchestrator"] = self._elapsed_ms(stage_started_ns)
                    gesture_runtime = getattr(perception_runtime, "gesture", None)
                    if perception_runtime is None or gesture_runtime is None:
                        stage_ms["total"] = self._elapsed_ms(refresh_started_ns)
                        workflow_event(
                            kind="branch",
                            msg="gesture_refresh_runtime_missing",
                            details={"stage_ms": dict(stage_ms)},
                            reason={
                                "selected": {
                                    "id": "gesture_runtime_missing",
                                    "justification": "The perception orchestrator returned no gesture payload.",
                                    "expected_outcome": "Abort the refresh to avoid publishing invalid HDMI gesture state.",
                                },
                                "options": [
                                    {"id": "gesture_runtime_present", "summary": "Continue with the orchestrator payload."},
                                    {"id": "gesture_runtime_missing", "summary": "Abort because no gesture payload was returned."},
                                ],
                                "confidence": "forensic",
                                "guardrails": ["gesture_runtime_required"],
                                "kpi_impact_estimate": {"latency": "low", "display_side_effect": "none"},
                            },
                        )
                        self._mark_refresh_failure("_last_display_gesture_refresh_failure_monotonic_s")
                        self._safe_record_gesture_debug_tick(
                            observed_at=now,
                            outcome="gesture_runtime_missing",
                            runtime_status_value=runtime_status_value,
                            observation=vision_observation,
                            stage_ms=stage_ms,
                        )
                        return False

                    decision = getattr(gesture_runtime, "ack_decision", None)
                    wakeup_decision = getattr(gesture_runtime, "wakeup_decision", None)
                    if decision is None or wakeup_decision is None:
                        stage_ms["total"] = self._elapsed_ms(refresh_started_ns)
                        workflow_event(
                            kind="branch",
                            msg="gesture_refresh_decisions_missing",
                            details={"stage_ms": dict(stage_ms)},
                            reason={
                                "selected": {
                                    "id": "gesture_decisions_missing",
                                    "justification": "The gesture runtime payload is missing an acknowledgement or wakeup decision.",
                                    "expected_outcome": "Abort the refresh without mutating HDMI gesture state.",
                                },
                                "options": [
                                    {"id": "gesture_decisions_present", "summary": "Continue with valid decisions."},
                                    {"id": "gesture_decisions_missing", "summary": "Abort because required decisions are missing."},
                                ],
                                "confidence": "forensic",
                                "guardrails": ["gesture_decisions_required"],
                                "kpi_impact_estimate": {"latency": "low", "display_side_effect": "none"},
                            },
                        )
                        self._mark_refresh_failure("_last_display_gesture_refresh_failure_monotonic_s")
                        self._safe_record_gesture_debug_tick(
                            observed_at=now,
                            outcome="gesture_decisions_missing",
                            runtime_status_value=runtime_status_value,
                            observation=vision_observation,
                            stage_ms=stage_ms,
                        )
                        return False

                    self.latest_perception_runtime_snapshot = perception_runtime
                    self._fault_barrier(
                        event="proactive_display_gesture_ack_trace_failed",
                        message="Failed to trace gesture acknowledgement lane decision.",
                        func=lambda: self._trace_gesture_ack_lane_decision(
                            observation=vision_observation,
                            decision=decision,
                        ),
                        default=None,
                    )
                    self._fault_barrier(
                        func=lambda: self._trace_gesture_wakeup_lane_decision(decision=wakeup_decision),
                        default=None,
                        event="proactive_display_gesture_wakeup_trace_failed",
                        message="Failed to trace gesture wakeup lane decision.",
                    )

                    if wakeup_decision.active:
                        stage_started_ns = time.monotonic_ns()
                        with workflow_span(
                            name="proactive_display_gesture_wakeup_priority",
                            kind="decision",
                        ):
                            wakeup_priority = self._fault_barrier(
                                event="proactive_display_gesture_wakeup_priority_failed",
                                message="Failed to evaluate gesture wakeup priority.",
                                func=lambda: decide_gesture_wakeup_priority(
                                    runtime_status_value=runtime_status_value,
                                    voice_path_enabled=bool(getattr(self.config, "voice_orchestrator_enabled", False)),
                                    presence_snapshot=self.latest_presence_snapshot,
                                    recent_speech_guard_s=_ATTENTION_REFRESH_AUDIO_CACHE_MAX_AGE_S,
                                ),
                            )
                        if wakeup_priority is None or not wakeup_priority.allow:
                            wakeup_decision = replace(
                                wakeup_decision,
                                active=False,
                                reason=(
                                    getattr(wakeup_priority, "reason", None)
                                    or "wakeup_priority_blocked"
                                ),
                            )
                        stage_ms["wakeup_priority"] = self._elapsed_ms(stage_started_ns)
                        workflow_decision(
                            msg="gesture_wakeup_priority_gate",
                            question="Should the active gesture wake candidate be allowed to start the voice path now?",
                            selected={
                                "id": "allow" if getattr(wakeup_priority, "allow", False) else "block",
                                "summary": (
                                    "Allow the wake request to proceed."
                                    if getattr(wakeup_priority, "allow", False)
                                    else "Suppress the wake request because the current runtime priority forbids it."
                                ),
                            },
                            options=[
                                {"id": "allow", "summary": "Allow the wake request to proceed."},
                                {"id": "block", "summary": "Block the wake request due to current runtime policy."},
                            ],
                            context={
                                "runtime_status": str(runtime_status_value or "").strip().lower() or None,
                                "voice_path_enabled": bool(getattr(self.config, "voice_orchestrator_enabled", False)),
                                "wakeup_reason": wakeup_decision.reason,
                                "block_reason": getattr(wakeup_priority, "reason", None),
                            },
                            confidence="forensic",
                            guardrails=["gesture_wakeup_priority"],
                            kpi_impact_estimate={"voice_turn": "high"},
                        )

                    # BREAKING: accepted gesture wakeups are now suppressed when their context is stale
                    # or the refresh path exceeded its latency budget.
                    if wakeup_decision.active and self._gesture_wakeup_context_expired(
                        observed_at=now,
                        refresh_started_ns=refresh_started_ns,
                        interval_s=interval_s,
                        vision_snapshot=snapshot,
                    ):
                        wakeup_decision = replace(
                            wakeup_decision,
                            active=False,
                            reason="stale_refresh_context",
                        )
                        workflow_event(
                            kind="branch",
                            msg="gesture_refresh_wakeup_suppressed_stale_context",
                            details={"stage_ms": dict(stage_ms)},
                            reason={
                                "selected": {
                                    "id": "suppressed",
                                    "justification": "The wakeup side effect was suppressed because the sensor context exceeded the freshness or latency budget.",
                                    "expected_outcome": "Do not start the voice path from stale gesture evidence.",
                                },
                                "options": [
                                    {"id": "dispatch", "summary": "Dispatch the gesture wakeup now."},
                                    {"id": "suppressed", "summary": "Suppress the wakeup because the context is stale or over budget."},
                                ],
                                "confidence": "forensic",
                                "guardrails": ["fresh_wakeup_context_required"],
                                "kpi_impact_estimate": {"voice_turn": "high", "false_wakeup": "high"},
                            },
                        )

                    stage_started_ns = time.monotonic_ns()
                    with workflow_span(
                        name="proactive_display_gesture_publish",
                        kind="mutation",
                        details={
                            "decision_reason": getattr(decision, "reason", None),
                            "decision_active": getattr(decision, "active", None),
                        },
                    ):
                        publish_result = self._fault_barrier(
                            event="proactive_display_gesture_publish_failed",
                            message="Failed to publish HDMI gesture decision.",
                            func=lambda: self._publish_display_gesture_decision(decision),
                            default=None,
                        )
                    stage_ms["publish"] = self._elapsed_ms(stage_started_ns)

                    stage_started_ns = time.monotonic_ns()
                    with workflow_span(
                        name="proactive_display_gesture_wakeup_handler",
                        kind="mutation",
                        details={
                            "wakeup_reason": getattr(wakeup_decision, "reason", None),
                            "wakeup_active": getattr(wakeup_decision, "active", None),
                        },
                    ):
                        wakeup_handled = bool(
                            self._fault_barrier(
                                event="proactive_display_gesture_wakeup_handler_failed",
                                message="Failed to dispatch HDMI gesture wakeup.",
                                func=lambda: self._dispatch_gesture_wakeup_with_fresh_context(
                                    observed_at=now,
                                    vision_snapshot=snapshot,
                                    decision=wakeup_decision,
                                ),
                                default=False,
                            )
                        )
                    stage_ms["wakeup_handler"] = self._elapsed_ms(stage_started_ns)
                    stage_ms["total"] = self._elapsed_ms(refresh_started_ns)

                    self._fault_barrier(
                        event="proactive_display_gesture_publish_trace_failed",
                        message="Failed to trace HDMI gesture publish decision.",
                        func=lambda: self._trace_gesture_publish_decision(
                            decision=decision,
                            publish_result=publish_result,
                            wakeup_decision=wakeup_decision,
                            wakeup_handled=wakeup_handled,
                        ),
                        default=None,
                    )
                    workflow_event(
                        kind="metric",
                        msg="gesture_refresh_stage_metrics",
                        details={"stage_ms": dict(stage_ms)},
                        kpi={f"stage_{key}_ms": value for key, value in stage_ms.items()},
                    )
                    self._safe_record_gesture_debug_tick(
                        observed_at=now,
                        outcome=(
                            "publish_failed"
                            if publish_result is None
                            else getattr(publish_result, "action", "published")
                        ),
                        runtime_status_value=runtime_status_value,
                        observation=vision_observation,
                        decision=decision,
                        publish_result=publish_result,
                        wakeup_decision=wakeup_decision,
                        wakeup_handled=wakeup_handled,
                        stage_ms=stage_ms,
                    )
                    self._mark_refresh_complete(
                        monotonic_attr="_last_display_gesture_refresh_monotonic_s",
                        wall_attr="_last_display_gesture_refresh_at",
                        observed_at=now,
                        failure_attr="_last_display_gesture_refresh_failure_monotonic_s",
                    )
                    return True
        except Exception as exc:
            self._record_fault_safe(
                event="proactive_display_gesture_refresh_failed",
                message="Unhandled exception in HDMI gesture refresh.",
                error=exc,
            )
            self._mark_refresh_failure("_last_display_gesture_refresh_failure_monotonic_s")
            fallback_now = now if now is not None else self._safe_clock_now()
            stage_ms["total"] = self._elapsed_ms(refresh_started_ns)
            self._safe_record_gesture_debug_tick(
                observed_at=fallback_now,
                outcome="unhandled_exception",
                runtime_status_value=runtime_status_value,
                stage_ms=stage_ms,
            )
            return False
        finally:
            lock.release()

    def _dispatch_gesture_wakeup_with_fresh_context(
        self,
        *,
        observed_at: float,
        vision_snapshot,
        decision: GestureWakeupDecision,
    ) -> bool:
        """Prime current gesture facts before dispatching an accepted wakeup."""

        return service_gesture_helpers.dispatch_gesture_wakeup_with_fresh_context(
            self,
            observed_at=observed_at,
            vision_snapshot=vision_snapshot,
            decision=decision,
        )

    def _prime_gesture_wakeup_sensor_context(
        self,
        *,
        observed_at: float,
        vision_snapshot,
    ) -> None:
        """Export one fresh sensor/person-state payload from the active gesture tick."""

        service_gesture_helpers.prime_gesture_wakeup_sensor_context(
            self,
            observed_at=observed_at,
            vision_snapshot=vision_snapshot,
        )

    def _publish_display_attention_live_context(
        self,
        *,
        observed_at: float,
        vision_observation: SocialVisionObservation,
        camera_snapshot,
        audio_snapshot,
        audio_policy_snapshot: ReSpeakerAudioPolicySnapshot | None,
    ) -> None:
        """Export authoritative HDMI-attention facts to the live voice context."""

        service_attention_helpers.publish_display_attention_live_context(
            self,
            observed_at=observed_at,
            vision_observation=vision_observation,
            camera_snapshot=camera_snapshot,
            audio_snapshot=audio_snapshot,
            audio_policy_snapshot=audio_policy_snapshot,
        )

    def _get_refresh_route_lock(self, attr_name: str) -> Lock:
        lock = getattr(self, attr_name, None)
        if lock is None:
            lock = Lock()
            setattr(self, attr_name, lock)
        return lock

    def _safe_clock_now(self) -> float:
        try:
            return float(self.clock())
        except Exception:
            return time.time()

    def _elapsed_ms(self, started_at: int | float) -> float:
        """Return one elapsed duration across the coordinator mixin stack.

        This helper is intentionally compatible with both monotonic-ns starts
        from the display refresh paths and perf-counter starts from the
        observation-dispatch mixin, because ``ProactiveCoordinatorDisplayMixin``
        sits first in the final coordinator MRO.
        """

        if isinstance(started_at, float):
            if not math.isfinite(started_at):
                return 0.0
            return round(max(0.0, time.perf_counter() - started_at) * 1000.0, 3)
        return round(max(0, time.monotonic_ns() - int(started_at)) / 1_000_000.0, 3)

    def _record_fault_safe(self, *, event: str, message: str, error: Exception) -> None:
        try:
            self._record_fault(event=event, message=message, error=error)
        except Exception:
            pass

    def _fault_barrier(
        self,
        *,
        event: str,
        message: str,
        func: Callable[[], _T],
        default: _T | None = None,
    ) -> _T | None:
        try:
            return func()
        except Exception as exc:
            self._record_fault_safe(event=event, message=message, error=exc)
            return default

    def _read_runtime_status_value(
        self,
        *,
        fault_event: str,
        fault_message: str,
    ) -> tuple[bool, str | None]:
        try:
            return True, self.runtime.status.value
        except Exception as exc:
            self._record_fault_safe(event=fault_event, message=fault_message, error=exc)
            return False, None

    def _refresh_interval_pending(
        self,
        *,
        monotonic_attr: str,
        wall_attr: str,
        observed_at: float,
        interval_s: float,
    ) -> bool:
        now_monotonic_s = time.monotonic()
        last_monotonic_s = self._coerce_float(getattr(self, monotonic_attr, None))
        if last_monotonic_s is not None:
            return (now_monotonic_s - last_monotonic_s) < interval_s

        last_wall = self._coerce_float(getattr(self, wall_attr, None))
        if last_wall is not None:
            return (observed_at - last_wall) < interval_s
        return False

    def _mark_refresh_complete(
        self,
        *,
        monotonic_attr: str,
        wall_attr: str,
        observed_at: float,
        failure_attr: str | None = None,
    ) -> None:
        setattr(self, monotonic_attr, time.monotonic())
        setattr(self, wall_attr, observed_at)
        if failure_attr is not None:
            setattr(self, failure_attr, None)

    def _mark_refresh_failure(self, monotonic_attr: str) -> None:
        setattr(self, monotonic_attr, time.monotonic())

    def _failure_backoff_pending(
        self,
        *,
        monotonic_attr: str,
        interval_s: float,
        config_attr: str,
    ) -> bool:
        last_failure_monotonic_s = self._coerce_float(getattr(self, monotonic_attr, None))
        if last_failure_monotonic_s is None:
            return False
        configured = self._coerce_float(getattr(self.config, config_attr, None))
        backoff_s = configured if configured is not None else min(max(interval_s * 0.5, 0.1), 1.0)
        return (time.monotonic() - last_failure_monotonic_s) < backoff_s

    def _coerce_float(self, value: Any, *, default: float | None = None) -> float | None:
        """Coerce one optional numeric value across the coordinator mixin stack.

        The observation/facts mixins pass ``default=...`` while the display
        helpers use the legacy ``None``-on-failure behavior. Keep both call
        shapes valid because the display mixin resolves first in the final MRO.
        """

        try:
            if value is None:
                return default
            result = float(value)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(result):
            return default
        return result

    def _extract_snapshot_timestamp(self, snapshot: Any) -> float | None:
        candidates = (
            getattr(snapshot, "captured_at", None),
            getattr(snapshot, "observed_at", None),
            getattr(getattr(snapshot, "observation", None), "observed_at", None),
        )
        for candidate in candidates:
            ts = self._coerce_float(candidate)
            if ts is not None:
                return ts
        return None

    def _snapshot_is_stale(
        self,
        *,
        snapshot: Any,
        observed_at: float,
        max_age_s: float | None,
    ) -> bool:
        if max_age_s is None or max_age_s <= 0:
            return False
        captured_at = self._extract_snapshot_timestamp(snapshot)
        if captured_at is None:
            return False
        return (observed_at - captured_at) > max_age_s

    def _resolve_vision_snapshot_max_age_s(
        self,
        *,
        kind: str,
        interval_s: float,
    ) -> float | None:
        config = getattr(self, "config", None)
        specific = self._coerce_float(
            getattr(config, f"display_{kind}_vision_max_age_s", None)
        )
        if specific is not None:
            return specific
        legacy = self._coerce_float(
            getattr(config, f"display_{kind}_max_snapshot_age_s", None)
        )
        if legacy is not None:
            return legacy
        floor = 1.0 if kind == "attention" else 0.75
        return max(floor, interval_s * 2.0)

    def _resolve_gesture_wakeup_max_context_age_s(self, *, interval_s: float) -> float | None:
        config = getattr(self, "config", None)
        configured = self._coerce_float(
            getattr(config, "display_gesture_wakeup_max_context_age_s", None)
        )
        if configured is not None:
            return configured
        return max(0.75, interval_s * 2.0)

    def _resolve_gesture_wakeup_max_latency_ms(self, *, interval_s: float) -> float | None:
        config = getattr(self, "config", None)
        configured = self._coerce_float(
            getattr(config, "display_gesture_wakeup_max_pipeline_latency_ms", None)
        )
        if configured is not None:
            return configured
        return min(max(interval_s * 2_000.0, 400.0), 1_500.0)

    def _gesture_wakeup_context_expired(
        self,
        *,
        observed_at: float,
        refresh_started_ns: int,
        interval_s: float,
        vision_snapshot: Any,
    ) -> bool:
        max_context_age_s = self._resolve_gesture_wakeup_max_context_age_s(interval_s=interval_s)
        max_latency_ms = self._resolve_gesture_wakeup_max_latency_ms(interval_s=interval_s)
        if self._snapshot_is_stale(
            snapshot=vision_snapshot,
            observed_at=observed_at,
            max_age_s=max_context_age_s,
        ):
            return True
        if max_latency_ms is None or max_latency_ms <= 0:
            return False
        return self._elapsed_ms(refresh_started_ns) > max_latency_ms

    def _empty_attention_audio_snapshot(self, *, observed_at: float):
        return SimpleNamespace(
            observed_at=observed_at,
            captured_at=observed_at,
            observation=SocialAudioObservation(),
        )

    def _normalize_attention_audio_snapshot(
        self,
        *,
        observed_at: float,
        interval_s: float,
        audio_snapshot: Any,
    ) -> tuple[Any, bool]:
        if audio_snapshot is None or getattr(audio_snapshot, "observation", None) is None:
            return self._empty_attention_audio_snapshot(observed_at=observed_at), True

        max_age_s = self._coerce_float(
            getattr(
                self.config,
                "display_attention_audio_max_age_s",
                _ATTENTION_REFRESH_AUDIO_CACHE_MAX_AGE_S,
            )
        )
        if max_age_s is None:
            max_age_s = max(_ATTENTION_REFRESH_AUDIO_CACHE_MAX_AGE_S, interval_s)
        if self._snapshot_is_stale(
            snapshot=audio_snapshot,
            observed_at=observed_at,
            max_age_s=max_age_s,
        ):
            return self._empty_attention_audio_snapshot(observed_at=observed_at), True
        return audio_snapshot, False

    def _resolve_attention_outcome(self, *, publish_result, audio_degraded: bool) -> str:
        published = publish_result is not None
        if published and audio_degraded:
            return "published_audio_degraded"
        if published:
            return "published"
        if audio_degraded:
            return "no_publish_result_audio_degraded"
        return "no_publish_result"

    def _safe_record_attention_debug_tick(self, **kwargs: Any) -> None:
        try:
            self._record_attention_debug_tick(**kwargs)
        except Exception:
            pass

    def _safe_record_gesture_debug_tick(self, **kwargs: Any) -> None:
        try:
            self._record_gesture_debug_tick(**kwargs)
        except Exception:
            pass
