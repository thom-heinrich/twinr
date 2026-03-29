"""HDMI attention and gesture refresh paths for the proactive coordinator.

Purpose: keep the dedicated display-attention and gesture-refresh workflows in
one focused module so the main coordinator file stays centered on monitor ticks
and trigger policy.

Invariants: refresh cadence, runtime gating, forensic tracing, and publish
outcomes must stay behavior-identical to the legacy service implementation,
while downstream attention/gesture semantics now come from the shared runtime
perception orchestrator instead of lane-local temporal truth.
"""

# mypy: ignore-errors

from __future__ import annotations

from dataclasses import replace
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


class ProactiveCoordinatorDisplayMixin:
    """Provide the dedicated HDMI attention and gesture refresh workflows."""

    def refresh_display_attention(self) -> bool:
        """Refresh HDMI attention-follow from the cheap local camera path."""

        refresh_started_ns = time.monotonic_ns()
        stage_ms: dict[str, float] = {}
        self._last_display_attention_fusion_debug = None

        if not display_attention_refresh_supported(
            config=self.config,
            vision_observer=self.vision_observer,
        ):
            self._record_attention_debug_tick(
                observed_at=self.clock(),
                outcome="unsupported",
                runtime_status_value=None,
                stage_ms=stage_ms,
            )
            return False
        now = self.clock()
        interval_s = resolve_display_attention_refresh_interval(self.config)
        if interval_s is None:
            self._record_attention_debug_tick(
                observed_at=now,
                outcome="no_refresh_interval",
                runtime_status_value=None,
                stage_ms=stage_ms,
            )
            return False
        if (
            self._last_display_attention_refresh_at is not None
            and (now - self._last_display_attention_refresh_at) < interval_s
        ):
            return False
        try:
            runtime_status_value = self.runtime.status.value
        except Exception as exc:
            self._record_fault(
                event="proactive_display_attention_runtime_status_failed",
                message="Failed to read runtime status for HDMI attention refresh.",
                error=exc,
            )
            self._record_attention_debug_tick(
                observed_at=now,
                outcome="runtime_status_failed",
                runtime_status_value=None,
                stage_ms=stage_ms,
            )
            return False
        if not _display_attention_refresh_allowed_runtime_status(runtime_status_value):
            self._record_attention_debug_tick(
                observed_at=now,
                outcome="runtime_status_blocked",
                runtime_status_value=runtime_status_value,
                stage_ms=stage_ms,
            )
            return False
        stage_started_ns = time.monotonic_ns()
        snapshot = self._observe_vision_for_attention_refresh()
        stage_ms["vision_observe"] = round((time.monotonic_ns() - stage_started_ns) / 1_000_000.0, 3)
        if snapshot is None:
            stage_ms["total"] = round((time.monotonic_ns() - refresh_started_ns) / 1_000_000.0, 3)
            self._record_attention_debug_tick(
                observed_at=now,
                outcome="vision_snapshot_missing",
                runtime_status_value=runtime_status_value,
                stage_ms=stage_ms,
            )
            return False
        stage_started_ns = time.monotonic_ns()
        fused_vision = self._fuse_display_attention_camera_observation(
            observed_at=now,
            observation=snapshot.observation,
        )
        stage_ms["camera_fusion"] = round((time.monotonic_ns() - stage_started_ns) / 1_000_000.0, 3)
        self._last_display_attention_refresh_at = now
        self._record_vision_snapshot_safe(snapshot)
        stage_started_ns = time.monotonic_ns()
        camera_update = self._observe_display_attention_camera_surface(
            SocialObservation(
                observed_at=now,
                inspected=True,
                pir_motion_detected=False,
                low_motion=False,
                vision=fused_vision,
                audio=SocialAudioObservation(),
            ),
            inspected=True,
        )
        stage_ms["camera_surface"] = round((time.monotonic_ns() - stage_started_ns) / 1_000_000.0, 3)
        stage_started_ns = time.monotonic_ns()
        audio_snapshot = self._observe_audio_for_attention_refresh(now=now)
        stage_ms["audio_observe"] = round((time.monotonic_ns() - stage_started_ns) / 1_000_000.0, 3)
        stage_started_ns = time.monotonic_ns()
        audio_policy_snapshot = self._observe_audio_policy(
            now=now,
            audio_observation=audio_snapshot.observation,
        )
        stage_ms["audio_policy"] = round((time.monotonic_ns() - stage_started_ns) / 1_000_000.0, 3)
        self._publish_display_attention_live_context(
            observed_at=now,
            vision_observation=fused_vision,
            camera_snapshot=camera_update.snapshot,
            audio_snapshot=audio_snapshot,
            audio_policy_snapshot=audio_policy_snapshot,
        )
        stage_started_ns = time.monotonic_ns()
        publish_result = self._update_display_attention_follow(
            source="display_attention_refresh",
            observed_at=now,
            camera_snapshot=camera_update.snapshot,
            audio_observation=audio_snapshot.observation,
            audio_policy_snapshot=audio_policy_snapshot,
        )
        stage_ms["publish"] = round((time.monotonic_ns() - stage_started_ns) / 1_000_000.0, 3)
        stage_started_ns = time.monotonic_ns()
        self._update_display_debug_signals(camera_update)
        stage_ms["debug_signals"] = round((time.monotonic_ns() - stage_started_ns) / 1_000_000.0, 3)
        stage_ms["total"] = round((time.monotonic_ns() - refresh_started_ns) / 1_000_000.0, 3)
        self._record_display_attention_follow_if_changed(
            observed_at=now,
            runtime_status_value=runtime_status_value,
            camera_snapshot=camera_update.snapshot,
            publish_result=publish_result,
        )
        self._record_attention_debug_tick(
            observed_at=now,
            outcome="published" if publish_result is not None else "no_publish_result",
            runtime_status_value=runtime_status_value,
            camera_snapshot=camera_update.snapshot,
            audio_observation=audio_snapshot.observation,
            publish_result=publish_result,
            stage_ms=stage_ms,
        )
        return True

    def refresh_display_gesture_emoji(self) -> bool:
        """Refresh HDMI gesture acknowledgements from the dedicated gesture path."""

        refresh_started_ns = time.monotonic_ns()
        stage_ms: dict[str, float] = {}
        if self.display_gesture_emoji_publisher is None:
            return False
        if not display_gesture_refresh_supported(
            config=self.config,
            vision_observer=self.vision_observer,
        ):
            self._record_gesture_debug_tick(
                observed_at=self.clock(),
                outcome="unsupported",
                runtime_status_value=None,
                stage_ms=stage_ms,
            )
            return False
        now = self.clock()
        interval_s = resolve_display_gesture_refresh_interval(self.config)
        if interval_s is None:
            self._record_gesture_debug_tick(
                observed_at=now,
                outcome="no_refresh_interval",
                runtime_status_value=None,
                stage_ms=stage_ms,
            )
            return False
        if (
            self._last_display_gesture_refresh_at is not None
            and (now - self._last_display_gesture_refresh_at) < interval_s
        ):
            return False
        try:
            runtime_status_value = self.runtime.status.value
        except Exception as exc:
            self._record_fault(
                event="proactive_display_gesture_runtime_status_failed",
                message="Failed to read runtime status for HDMI gesture refresh.",
                error=exc,
            )
            self._record_gesture_debug_tick(
                observed_at=now,
                outcome="runtime_status_failed",
                runtime_status_value=None,
                stage_ms=stage_ms,
            )
            return False
        if not _display_attention_refresh_allowed_runtime_status(runtime_status_value):
            self._record_gesture_debug_tick(
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
                    snapshot = self._observe_vision_for_gesture_refresh()
                stage_ms["vision_observe"] = round((time.monotonic_ns() - stage_started_ns) / 1_000_000.0, 3)
                if snapshot is None:
                    stage_ms["total"] = round((time.monotonic_ns() - refresh_started_ns) / 1_000_000.0, 3)
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
                    self._record_gesture_debug_tick(
                        observed_at=now,
                        outcome="vision_snapshot_missing",
                        runtime_status_value=runtime_status_value,
                        stage_ms=stage_ms,
                    )
                    return False
                self._last_display_gesture_refresh_at = now
                self._record_vision_snapshot_safe(snapshot)
                self._remember_display_attention_camera_semantics(
                    observed_at=now,
                    observation=snapshot.observation,
                    source="gesture",
                )
                workflow_event(
                    kind="io",
                    msg="gesture_refresh_observation",
                    details=self._gesture_observation_trace_details(snapshot.observation),
                )
                stage_started_ns = time.monotonic_ns()
                with workflow_span(
                    name="proactive_display_gesture_perception_orchestrator",
                    kind="decision",
                ):
                    perception_runtime = self.perception_orchestrator.observe_gesture(
                        observed_at=now,
                        source=self._last_gesture_vision_refresh_mode,
                        captured_at=snapshot.captured_at,
                        vision_observation=snapshot.observation,
                    )
                self.latest_perception_runtime_snapshot = perception_runtime
                assert perception_runtime.gesture is not None
                decision = perception_runtime.gesture.ack_decision
                wakeup_decision = perception_runtime.gesture.wakeup_decision
                stage_ms["gesture_orchestrator"] = round((time.monotonic_ns() - stage_started_ns) / 1_000_000.0, 3)
                self._trace_gesture_ack_lane_decision(
                    observation=snapshot.observation,
                    decision=decision,
                )
                self._trace_gesture_wakeup_lane_decision(decision=wakeup_decision)
                if wakeup_decision.active:
                    stage_started_ns = time.monotonic_ns()
                    with workflow_span(
                        name="proactive_display_gesture_wakeup_priority",
                        kind="decision",
                    ):
                        wakeup_priority = decide_gesture_wakeup_priority(
                            runtime_status_value=runtime_status_value,
                            voice_path_enabled=bool(getattr(self.config, "voice_orchestrator_enabled", False)),
                            presence_snapshot=self.latest_presence_snapshot,
                            recent_speech_guard_s=_ATTENTION_REFRESH_AUDIO_CACHE_MAX_AGE_S,
                        )
                    if not wakeup_priority.allow:
                        wakeup_decision = replace(
                            wakeup_decision,
                            active=False,
                            reason=wakeup_priority.reason,
                        )
                    stage_ms["wakeup_priority"] = round((time.monotonic_ns() - stage_started_ns) / 1_000_000.0, 3)
                    workflow_decision(
                        msg="gesture_wakeup_priority_gate",
                        question="Should the active gesture wake candidate be allowed to start the voice path now?",
                        selected={
                            "id": "allow" if wakeup_priority.allow else wakeup_priority.reason,
                            "summary": (
                                "Allow the wake request to proceed."
                                if wakeup_priority.allow
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
                        },
                        confidence="forensic",
                        guardrails=["gesture_wakeup_priority"],
                        kpi_impact_estimate={"voice_turn": "high"},
                    )
                stage_started_ns = time.monotonic_ns()
                with workflow_span(
                    name="proactive_display_gesture_publish",
                    kind="mutation",
                    details={"decision_reason": decision.reason, "decision_active": decision.active},
                ):
                    publish_result = self._publish_display_gesture_decision(decision)
                stage_ms["publish"] = round((time.monotonic_ns() - stage_started_ns) / 1_000_000.0, 3)
                stage_started_ns = time.monotonic_ns()
                with workflow_span(
                    name="proactive_display_gesture_wakeup_handler",
                    kind="mutation",
                    details={"wakeup_reason": wakeup_decision.reason, "wakeup_active": wakeup_decision.active},
                ):
                    wakeup_handled = self._dispatch_gesture_wakeup_with_fresh_context(
                        observed_at=now,
                        vision_snapshot=snapshot,
                        decision=wakeup_decision,
                    )
                stage_ms["wakeup_handler"] = round((time.monotonic_ns() - stage_started_ns) / 1_000_000.0, 3)
                stage_ms["total"] = round((time.monotonic_ns() - refresh_started_ns) / 1_000_000.0, 3)
                self._trace_gesture_publish_decision(
                    decision=decision,
                    publish_result=publish_result,
                    wakeup_decision=wakeup_decision,
                    wakeup_handled=wakeup_handled,
                )
                workflow_event(
                    kind="metric",
                    msg="gesture_refresh_stage_metrics",
                    details={"stage_ms": dict(stage_ms)},
                    kpi={f"stage_{key}_ms": value for key, value in stage_ms.items()},
                )
                self._record_gesture_debug_tick(
                    observed_at=now,
                    outcome=("publish_failed" if publish_result is None else publish_result.action),
                    runtime_status_value=runtime_status_value,
                    observation=snapshot.observation,
                    decision=decision,
                    publish_result=publish_result,
                    wakeup_decision=wakeup_decision,
                    wakeup_handled=wakeup_handled,
                    stage_ms=stage_ms,
                )
                return True

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
