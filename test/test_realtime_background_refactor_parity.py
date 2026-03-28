from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
from queue import Queue
import re
import sys
import tempfile
from types import SimpleNamespace
from typing import Any
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.runtime.runtime import TwinrRuntime
from twinr.automations import AutomationAction
from twinr.display.ambient_impulse_cues import DisplayAmbientImpulseCueStore
from twinr.memory.reminders import now_in_timezone
from twinr.proactive.social.engine import SocialTriggerDecision, SocialTriggerPriority
from twinr.proactive.runtime.audio_policy import ReSpeakerAudioPolicySnapshot
from twinr.agent.workflows.realtime_runtime.background import TwinrRealtimeBackgroundMixin
from twinr.agent.workflows.realtime_runtime.background_impl import TwinrRealtimeBackgroundMixinImpl
from test.test_realtime_runner import FakePlayer, FakePrintBackend, FakePrinter


_EXPECTED_GOLDEN_DIGESTS = {
    "sensor_merge": "ac597e383247c2c23e8b89386ff3213248fa5e87258d2eaec413ff3dff29f8e8",
    "social_visual_first": "6669363a676429b09571fd61ba4a91905e82aa5e704ed9538e62db08354738c3",
    "printed_automation": "c97aafeb579d9b71b432a96ace3bee0ded3482985101e670b7cb117c6b718228",
    "longterm_observation_facts": "95d7023310911e6be86f9792a5678954dab37814ec5197a0ed171faefdc2f9a9",
}
_WORKING_FEEDBACK_OWNER_RE = re.compile(
    r"(working_feedback:[a-z_]+:)\d+(?::\d+)?"
)


def _normalize_payload(value: Any):
    if is_dataclass(value):
        return {key: _normalize_payload(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_payload(item) for item in value]
    if isinstance(value, set):
        return sorted(_normalize_payload(item) for item in value)
    if isinstance(value, bytes):
        return {"len": len(value)}
    if hasattr(value, "isoformat") and callable(getattr(value, "isoformat", None)):
        try:
            return value.isoformat()
        except TypeError:
            pass
    if hasattr(value, "__dict__") and not isinstance(value, (int, float, bool, str, type(None))):
        payload = {
            key: _normalize_payload(item)
            for key, item in value.__dict__.items()
            if not key.startswith("_")
        }
        payload["__class__"] = type(value).__name__
        return payload
    return value


def _stable_line(line: str) -> str:
    if line.startswith("automation_id="):
        return "automation_id=<normalized>"
    if line.startswith("automation_execution_key="):
        return "automation_execution_key=<normalized>"
    if line.startswith("automation_print_job="):
        return "automation_print_job=<normalized>"
    if line.startswith("automation_scheduled_for_at="):
        return "automation_scheduled_for_at=<normalized>"
    line = _WORKING_FEEDBACK_OWNER_RE.sub(r"\1<normalized>", line)
    return line


class _UsageStore:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def append(self, **kwargs: object) -> None:
        self.records.append(dict(kwargs))


def _make_harness(
    mixin_cls: type,
    *,
    proactive_monitor: object | None = None,
):
    class _Harness(mixin_cls):
        def __init__(self, root: str, *, proactive_monitor: object | None = None) -> None:
            self.config = TwinrConfig(
                project_root=root,
                personality_dir="personality",
                runtime_state_path=str(Path(root) / "state" / "runtime-state.json"),
                reminder_store_path=str(Path(root) / "state" / "reminders.json"),
                automation_store_path=str(Path(root) / "state" / "automations.json"),
                long_term_memory_enabled=True,
                long_term_memory_path=str(Path(root) / "state" / "chonkydb"),
                long_term_memory_proactive_enabled=True,
                proactive_quiet_hours_visual_only_enabled=False,
                automation_poll_interval_s=0.0,
                reminder_poll_interval_s=0.0,
            )
            self.runtime = TwinrRuntime(config=self.config)
            self.emit_lines: list[str] = []
            self.emit = self.emit_lines.append
            self.agent_provider = FakePrintBackend()
            self.print_backend = self.agent_provider
            self.tts_provider = self.agent_provider
            self.player = FakePlayer()
            self.printer = FakePrinter()
            self.usage_store = _UsageStore()
            self.tool_executor = SimpleNamespace()
            self.proactive_monitor = proactive_monitor
            self._sensor_observation_queue: Queue[tuple[dict[str, object], tuple[str, ...]]] = Queue(maxsize=1)
            self._conversation_session_active = False
            self.feedback_kinds: list[str] = []
            self.multimodal_evidence: list[dict[str, object]] = []
            self.voice_context_refreshes = 0
            self.required_remote_errors: list[str] = []
            self.sleep = lambda _seconds: None
            self._background_now_calls = 0
            self._social_now_utc = datetime(2026, 3, 22, 14, 30, tzinfo=timezone.utc)
            self._social_monotonic = 45.0
            self.runtime.long_term_memory.enqueue_multimodal_evidence = (  # type: ignore[method-assign]
                lambda **kwargs: self.multimodal_evidence.append(_normalize_payload(kwargs))
            )

        def _record_event(self, event: str, message: str, *, level: str = "info", **data: object) -> None:
            self.runtime.ops_events.append(event=event, message=message, level=level, data=data)

        def _record_usage(self, **kwargs: object) -> None:
            self.usage_store.append(**kwargs)

        def _emit_status(self, *, force: bool = False) -> None:
            status = getattr(getattr(self.runtime, "status", None), "value", "unknown")
            if force or status != getattr(self, "_last_status", None):
                self.emit(f"status={status}")
                self._last_status = status

        def _start_working_feedback_loop(self, kind: str):
            self.feedback_kinds.append(kind)
            return lambda: None

        def _play_streaming_tts_with_feedback(self, text: str, *, turn_started: float):
            del turn_started
            self.feedback_kinds.append("answering")
            rendered = b"".join(self.tts_provider.synthesize_stream(text))
            self.player.played.append(rendered)
            return (1, None)

        def _run_proactive_follow_up(self, trigger: SocialTriggerDecision) -> bool:
            del trigger
            return False

        def _enter_required_remote_error(self, exc: BaseException | str) -> bool:
            self.required_remote_errors.append(str(exc))
            return True

        def _refresh_voice_orchestrator_sensor_context(self) -> None:
            self.voice_context_refreshes += 1

        def _background_now(self) -> tuple[float, float]:
            monotonic_now = 100.0 + float(self._background_now_calls)
            epoch_now = 1_710_000_000.0 + float(self._background_now_calls)
            self._background_now_calls += 1
            return monotonic_now, epoch_now

        def _social_monotonic_now(self) -> float:
            return self._social_monotonic

        def _social_utc_now(self) -> datetime:
            return self._social_now_utc

        def _local_now(self) -> datetime:
            return self._social_now_utc + timedelta(hours=1)

    return _Harness


def _shutdown_runtime(loop: object) -> None:
    runtime = getattr(loop, "runtime", None)
    shutdown = getattr(runtime, "shutdown", None)
    if callable(shutdown):
        shutdown(timeout_s=0.2)


def _digest_payload(payload: dict[str, object]) -> str:
    serialized = json.dumps(
        _normalize_payload(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


def _capture_sensor_merge_payload(mixin_cls: type) -> dict[str, object]:
    harness_cls = _make_harness(mixin_cls)
    with tempfile.TemporaryDirectory() as temp_dir:
        loop = harness_cls(temp_dir)
        try:
            loop.handle_sensor_observation(
                {
                    "sensor": {"observed_at": 10.0},
                    "camera": {"person_visible": True},
                    "vad": {"speech_detected": False, "quiet": True},
                },
                ("camera.visible",),
            )
            loop.handle_sensor_observation(
                {
                    "sensor": {"observed_at": 12.0},
                    "camera": {"looking_toward_device": True, "body_pose": "standing"},
                    "pir": {"motion_detected": True},
                },
                ("pir.motion",),
            )
            queued_facts, queued_events = loop._sensor_observation_queue.get_nowait()
            return {
                "latest_facts": loop._latest_sensor_observation_facts,
                "queued_facts": queued_facts,
                "queued_events": queued_events,
                "voice_context_refreshes": loop.voice_context_refreshes,
                "fault_count": len(getattr(loop, "_background_faults", [])),
            }
        finally:
            _shutdown_runtime(loop)


def _capture_social_visual_first_payload(mixin_cls: type) -> dict[str, object]:
    proactive_monitor = SimpleNamespace(
        coordinator=SimpleNamespace(
            latest_audio_policy_snapshot=ReSpeakerAudioPolicySnapshot(
                observed_at=42.0,
                speech_delivery_defer_reason="background_media_active",
                background_media_likely=True,
            ),
        )
    )
    harness_cls = _make_harness(mixin_cls, proactive_monitor=proactive_monitor)
    with tempfile.TemporaryDirectory() as temp_dir:
        loop = harness_cls(temp_dir, proactive_monitor=proactive_monitor)
        try:
            spoke = loop.handle_social_trigger(
                SocialTriggerDecision(
                    trigger_id="attention_window",
                    prompt="Soll ich helfen?",
                    reason="quiet room",
                    observed_at=42.0,
                    priority=SocialTriggerPriority.ATTENTION_WINDOW,
                )
            )
            raw_cue_payload = json.loads(
                DisplayAmbientImpulseCueStore.from_config(loop.config).path.read_text(encoding="utf-8")
            )
            cue_payload = {
                "headline": raw_cue_payload.get("headline"),
                "body": raw_cue_payload.get("body"),
                "source": raw_cue_payload.get("source"),
            }
            event_summary = [
                {
                    "event": entry["event"],
                    "display_reason": entry.get("data", {}).get("display_reason"),
                    "trigger": entry.get("data", {}).get("trigger"),
                }
                for entry in loop.runtime.ops_events.tail(limit=20)
                if entry["event"].startswith("social_trigger")
            ]
            return {
                "result": spoke,
                "lines": [_stable_line(line) for line in loop.emit_lines],
                "cue": cue_payload,
                "events": event_summary,
                "played": list(loop.player.played),
                "proactive_calls": list(loop.agent_provider.proactive_calls),
            }
        finally:
            _shutdown_runtime(loop)


def _capture_printed_automation_payload(mixin_cls: type) -> dict[str, object]:
    harness_cls = _make_harness(mixin_cls)
    with tempfile.TemporaryDirectory() as temp_dir:
        loop = harness_cls(temp_dir)
        try:
            entry = loop.runtime.create_time_automation(
                name="Daily briefing",
                schedule="daily",
                time_of_day=now_in_timezone(loop.config.local_timezone_name).strftime("%H:%M"),
                actions=(
                    AutomationAction(
                        kind="llm_prompt",
                        text="Print the daily briefing.",
                        payload={"delivery": "printed", "allow_web_search": "false"},
                        enabled=True,
                    ),
                ),
                source="test",
            )
            executed = loop._maybe_run_due_automation()
            stored = loop.runtime.automation_store.get(entry.automation_id)
            event_summary = [
                {
                    "event": item["event"],
                    "request_source": item.get("data", {}).get("request_source"),
                }
                for item in loop.runtime.ops_events.tail(limit=20)
                if item["event"].startswith("automation_")
            ]
            return {
                "result": executed,
                "lines": [_stable_line(line) for line in loop.emit_lines],
                "automation_calls": list(loop.agent_provider.automation_calls),
                "printed": list(loop.printer.printed),
                "feedback": list(loop.feedback_kinds),
                "usage_count": len(loop.usage_store.records),
                "stored_triggered": bool(stored is not None and stored.last_triggered_at is not None),
                "events": event_summary,
                "multimodal_evidence": [
                    {
                        "event_name": item.get("event_name"),
                        "source": item.get("source"),
                        "request_source": item.get("data", {}).get("request_source"),
                        "job": "<normalized>" if item.get("data", {}).get("job") else None,
                        "queue": item.get("data", {}).get("queue"),
                    }
                    for item in loop.multimodal_evidence
                ],
            }
        finally:
            _shutdown_runtime(loop)


def _capture_longterm_observation_facts_payload(mixin_cls: type) -> dict[str, object]:
    harness_cls = _make_harness(mixin_cls)
    with tempfile.TemporaryDirectory() as temp_dir:
        loop = harness_cls(temp_dir)
        try:
            facts = loop._longterm_proactive_observation_facts(
                candidate=SimpleNamespace(kind="routine_breakfast", sensitivity="normal", source_memory_ids=()),
                live_facts={
                    "camera": {
                        "person_visible": True,
                        "looking_toward_device": True,
                        "hand_or_object_near_camera": False,
                        "body_pose": "standing",
                    },
                    "vad": {
                        "quiet": True,
                        "speech_detected": False,
                    },
                    "multimodal_initiative": {
                        "ready": False,
                        "block_reason": "background_media_active",
                    },
                    "last_response_available": True,
                    "recent_print_completed": False,
                },
            )
            return {"facts": facts}
        finally:
            _shutdown_runtime(loop)


class RealtimeBackgroundRefactorParityTests(unittest.TestCase):
    def test_public_wrapper_preserves_class_module(self) -> None:
        self.assertEqual(
            TwinrRealtimeBackgroundMixin.__module__,
            "twinr.agent.workflows.realtime_runtime.background",
        )

    def test_public_wrapper_is_legacy_subclass_of_internal_impl(self) -> None:
        self.assertTrue(issubclass(TwinrRealtimeBackgroundMixin, TwinrRealtimeBackgroundMixinImpl))

    def test_golden_master_hashes_remain_stable(self) -> None:
        cases = {
            "sensor_merge": _capture_sensor_merge_payload(TwinrRealtimeBackgroundMixin),
            "social_visual_first": _capture_social_visual_first_payload(TwinrRealtimeBackgroundMixin),
            "printed_automation": _capture_printed_automation_payload(TwinrRealtimeBackgroundMixin),
            "longterm_observation_facts": _capture_longterm_observation_facts_payload(TwinrRealtimeBackgroundMixin),
        }
        for name, payload in cases.items():
            with self.subTest(case=name):
                self.assertEqual(_digest_payload(payload), _EXPECTED_GOLDEN_DIGESTS[name])

    def test_public_wrapper_matches_internal_implementation_payloads(self) -> None:
        cases = (
            ("sensor_merge", _capture_sensor_merge_payload),
            ("social_visual_first", _capture_social_visual_first_payload),
            ("printed_automation", _capture_printed_automation_payload),
            ("longterm_observation_facts", _capture_longterm_observation_facts_payload),
        )
        for name, builder in cases:
            with self.subTest(case=name):
                wrapped = _normalize_payload(builder(TwinrRealtimeBackgroundMixin))
                internal = _normalize_payload(builder(TwinrRealtimeBackgroundMixinImpl))
                self.assertEqual(wrapped, internal)

    def test_recent_print_completed_returns_true_for_recent_print_event(self) -> None:
        harness_cls = _make_harness(TwinrRealtimeBackgroundMixin)
        with tempfile.TemporaryDirectory() as temp_dir:
            loop = harness_cls(temp_dir)
            try:
                loop.runtime.ops_events.append(
                    event="print_finished",
                    message="Printed a background response.",
                )

                self.assertTrue(loop._recent_print_completed())
            finally:
                _shutdown_runtime(loop)
