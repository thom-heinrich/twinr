from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from typing import Any, cast
import sys
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.proactive.runtime import service as proactive_service_mod
from twinr.proactive.runtime.service import (
    _proactive_respeaker_host_control_conflicts_with_voice_orchestrator,
    build_default_proactive_monitor,
)


class ReSpeakerVoiceRuntimeAudioGateTests(unittest.TestCase):
    def test_respeaker_host_control_conflict_detects_same_card(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                voice_orchestrator_enabled=True,
                voice_orchestrator_ws_url="ws://192.168.1.10:8797/ws/orchestrator/voice",
                voice_orchestrator_audio_device="sysdefault:CARD=Array",
                audio_input_device="plughw:CARD=Array,DEV=0",
                proactive_audio_enabled=True,
            )

            self.assertTrue(
                _proactive_respeaker_host_control_conflicts_with_voice_orchestrator(config)
            )

    def test_respeaker_host_control_conflict_ignores_other_cards(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                voice_orchestrator_enabled=True,
                voice_orchestrator_ws_url="ws://192.168.1.10:8797/ws/orchestrator/voice",
                voice_orchestrator_audio_device="plughw:CARD=USB,DEV=0",
                audio_input_device="plughw:CARD=Array,DEV=0",
                proactive_audio_enabled=True,
            )

            self.assertFalse(
                _proactive_respeaker_host_control_conflicts_with_voice_orchestrator(config)
            )

    def test_build_default_monitor_disables_respeaker_audio_when_voice_runtime_owns_same_device(
        self,
    ) -> None:
        recorded_ops_events: list[dict[str, object]] = []
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                display_driver="hdmi_fbdev",
                proactive_vision_provider="local_first",
                voice_orchestrator_enabled=True,
                voice_orchestrator_ws_url="ws://192.168.1.10:8797/ws/orchestrator/voice",
                voice_orchestrator_audio_device="sysdefault:CARD=Array",
                audio_input_device="plughw:CARD=Array,DEV=0",
                proactive_audio_enabled=True,
                display_attention_refresh_interval_s=0.6,
                display_gesture_refresh_interval_s=0.0,
            )
            runtime = SimpleNamespace(
                ops_events=SimpleNamespace(append=lambda **kwargs: recorded_ops_events.append(kwargs)),
                fail=lambda _detail: None,
                status=SimpleNamespace(value="waiting"),
            )
            sentinel_provider = object()

            with (
                patch.object(
                    proactive_service_mod.LocalAICameraObservationProvider,
                    "from_config",
                    return_value=sentinel_provider,
                ),
                patch.object(proactive_service_mod, "ReSpeakerSignalProvider") as signal_provider_cls,
            ):
                monitor = build_default_proactive_monitor(
                    config=config,
                    runtime=runtime,
                    backend=cast(Any, object()),
                    camera=cast(Any, object()),
                    camera_lock=None,
                    audio_lock=None,
                    trigger_handler=lambda _decision: False,
                )

            self.assertIsNotNone(monitor)
            assert monitor is not None
            self.assertIs(monitor.coordinator.vision_observer, sentinel_provider)
            self.assertIsInstance(
                monitor.coordinator.audio_observer,
                proactive_service_mod.NullAudioObservationProvider,
            )
            signal_provider_cls.assert_not_called()

        warning_found = False
        for entry in recorded_ops_events:
            if entry.get("event") != "proactive_component_warning":
                continue
            raw_data = entry.get("data")
            if not isinstance(raw_data, dict):
                continue
            data = cast(dict[str, object], raw_data)
            if data.get("reason") == "respeaker_signal_provider_disabled_for_voice_runtime":
                warning_found = True
                break
        self.assertTrue(warning_found)


if __name__ == "__main__":
    unittest.main()
