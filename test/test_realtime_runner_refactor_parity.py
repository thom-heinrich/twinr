from __future__ import annotations

from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
from typing import Any
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.runtime.runtime import TwinrRuntime
from twinr.agent.workflows.realtime_runner import TwinrRealtimeHardwareLoop
from twinr.agent.workflows.realtime_runner_impl import TwinrRealtimeHardwareLoopImpl
from twinr.orchestrator.voice_activation import VoiceActivationMatch
from test.test_realtime_runner import (
    FakeCamera,
    FakeIdleButtonMonitor,
    FakePlayer,
    FakePrintBackend,
    FakePrinter,
    FakeProactiveMonitor,
    FakeRealtimeSession,
    FakeRecorder,
    FakeTurnStreamingSpeechToTextProvider,
    FakeTurnToolAgentProvider,
)

_TIMING_LINE_RE = re.compile(r"^(timing_[a-z_]+)=.+$")
_WORKING_FEEDBACK_OWNER_RE = re.compile(
    r"(working_feedback:[a-z_]+:)\d+(?::\d+)?"
)
_EXPECTED_GOLDEN_DIGESTS = {
    "voice_activation_seed_text": "0ca2c99c8671621a3d317cf028966d3dbb712bc2218ca7ab4277bb1dbd324fe8",
    "button_audio_turn": "79cd2f07c522acca5b1d42f1357dfe27c3d45bdda86ff216eacc2bd167661cd9",
}


def _normalize_payload(value):
    if is_dataclass(value):
        return {key: _normalize_payload(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return "<PATH>"
    if isinstance(value, dict):
        return {str(key): _normalize_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_payload(item) for item in value]
    if isinstance(value, bytes):
        return {"len": len(value)}
    if isinstance(value, str):
        match = _TIMING_LINE_RE.match(value)
        if match:
            return f"{match.group(1)}=<normalized>"
        value = _WORKING_FEEDBACK_OWNER_RE.sub(r"\1<normalized>", value)
        return value
    if hasattr(value, "__dict__") and not isinstance(value, (int, float, bool, type(None))):
        payload = {
            key: _normalize_payload(item)
            for key, item in value.__dict__.items()
            if not key.startswith("_")
        }
        payload["__class__"] = type(value).__name__
        return payload
    return value


def _snapshot_payload(
    loop: Any,
    *,
    lines: list[str],
    session: FakeRealtimeSession,
    result: bool,
) -> dict[str, object]:
    return {
        "result": result,
        "runtime_status": getattr(loop.runtime.status, "value", None),
        "last_transcript": loop.runtime.last_transcript,
        "conversation": _normalize_payload(loop.runtime.conversation_context()),
        "session_audio_calls": [len(item) for item in session.calls],
        "session_text_calls": list(session.text_calls),
        "session_conversations": _normalize_payload(session.conversations),
        "player_tones": _normalize_payload(loop.player.tones),
        "player_played": [len(item) for item in loop.player.played],
        "lines": _normalize_payload(list(lines)),
    }


class TwinrRealtimeRunnerRefactorParityTests(unittest.TestCase):
    def _make_loop(
        self,
        *,
        loop_cls=TwinrRealtimeHardwareLoop,
        recorder: FakeRecorder | None = None,
        turn_stt_provider=None,
        turn_tool_agent_provider=None,
    ) -> tuple[TwinrRealtimeHardwareLoop, list[str], FakeRealtimeSession]:
        temp_dir = tempfile.TemporaryDirectory()
        config = TwinrConfig(
            openai_api_key="test-key",
            project_root=temp_dir.name,
            personality_dir="personality",
            turn_controller_enabled=turn_stt_provider is not None,
            conversation_closure_guard_enabled=False,
            streaming_early_transcript_enabled=True,
            turn_controller_interrupt_enabled=False,
        )
        lines: list[str] = []
        session = FakeRealtimeSession()
        print_backend = FakePrintBackend()
        loop = loop_cls(
            config=config,
            runtime=TwinrRuntime(config=config),
            realtime_session=session,
            print_backend=print_backend,
            stt_provider=print_backend,
            agent_provider=print_backend,
            tts_provider=print_backend,
            turn_stt_provider=turn_stt_provider,
            turn_tool_agent_provider=turn_tool_agent_provider,
            button_monitor=FakeIdleButtonMonitor(),
            recorder=recorder or FakeRecorder(),
            player=FakePlayer(),
            printer=FakePrinter(),
            camera=FakeCamera(),
            voice_profile_monitor=SimpleNamespace(),
            usage_store=SimpleNamespace(),
            proactive_monitor=FakeProactiveMonitor(),
            emit=lines.append,
            sleep=lambda _seconds: None,
            error_reset_seconds=0.0,
        )
        self.addCleanup(temp_dir.cleanup)
        self.addCleanup(loop.close, timeout_s=0.2)
        loop._refactor_parity_temp_dir = temp_dir
        return loop, lines, session

    def _voice_activation_seed_text_payload(self, *, loop_cls=TwinrRealtimeHardwareLoop) -> dict[str, object]:
        loop, lines, session = self._make_loop(loop_cls=loop_cls)
        result = loop.handle_voice_activation(
            VoiceActivationMatch(
                detected=True,
                matched_phrase="hallo twinr",
                transcript="hallo twinr wie geht es dir",
                remaining_text="wie geht es dir",
                score=0.97,
            )
        )
        return _snapshot_payload(loop, lines=lines, session=session, result=result)

    def _button_audio_turn_payload(self, *, loop_cls=TwinrRealtimeHardwareLoop) -> dict[str, object]:
        turn_config = TwinrConfig(openai_api_key="test-key")
        loop, lines, session = self._make_loop(
            loop_cls=loop_cls,
            recorder=FakeRecorder(),
            turn_stt_provider=FakeTurnStreamingSpeechToTextProvider(turn_config),
            turn_tool_agent_provider=FakeTurnToolAgentProvider(turn_config),
        )
        result = loop._run_conversation_session(initial_source="button")
        return _snapshot_payload(loop, lines=lines, session=session, result=result)

    def test_public_wrapper_preserves_class_module(self) -> None:
        self.assertEqual(
            TwinrRealtimeHardwareLoop.__module__,
            "twinr.agent.workflows.realtime_runner",
        )

    def test_golden_master_hashes_remain_stable(self) -> None:
        cases = {
            "voice_activation_seed_text": self._voice_activation_seed_text_payload(),
            "button_audio_turn": self._button_audio_turn_payload(),
        }
        for name, payload in cases.items():
            with self.subTest(case=name):
                serialized = json.dumps(
                    _normalize_payload(payload),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
                digest = sha256(serialized.encode("utf-8")).hexdigest()
                self.assertEqual(digest, _EXPECTED_GOLDEN_DIGESTS[name])

    def test_public_wrapper_matches_internal_implementation_payloads(self) -> None:
        cases = (
            ("voice_activation_seed_text", self._voice_activation_seed_text_payload),
            ("button_audio_turn", self._button_audio_turn_payload),
        )
        for name, builder in cases:
            with self.subTest(case=name):
                wrapped = _normalize_payload(builder(loop_cls=TwinrRealtimeHardwareLoop))
                internal = _normalize_payload(builder(loop_cls=TwinrRealtimeHardwareLoopImpl))
                self.assertEqual(wrapped, internal)

    def test_wrapper_boot_sound_patch_surface_stays_on_legacy_module_path(self) -> None:
        loop, _lines, _session = self._make_loop()
        with mock.patch(
            "twinr.agent.workflows.realtime_runner.start_startup_boot_sound"
        ) as start_mock:
            result = loop.run(duration_s=0.01, poll_timeout=0.001)
        self.assertEqual(result, 0)
        start_mock.assert_called_once_with(
            config=loop.config,
            playback_coordinator=loop.playback_coordinator,
            emit=loop.emit,
        )

    def test_wrapper_smart_home_builder_patch_surface_stays_on_legacy_module_path(self) -> None:
        loop, _lines, _session = self._make_loop()
        sentinel = object()
        with mock.patch(
            "twinr.agent.workflows.realtime_runner.build_smart_home_hub_adapter",
            return_value=sentinel,
        ) as builder_mock:
            adapter = loop._build_managed_smart_home_adapter()
        self.assertIs(adapter, sentinel)
        builder_mock.assert_called_once_with(Path(loop.config.project_root))
