from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import base64
import json
import os
import socket
import stat
import sys
import tempfile
import unittest
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.state.snapshot import RuntimeSnapshotStore
from twinr.hardware.audio import AmbientAudioLevelSample
from twinr.memory.on_device import ConversationTurn
from twinr.ops.checks import run_config_checks
from twinr.ops.events import TwinrOpsEventStore
from twinr.ops.health import collect_system_health
from twinr.ops.locks import loop_instance_lock
from twinr.ops.runtime_scope import build_scoped_runtime_config, resolve_scoped_runtime_state_path
from twinr.ops.self_test import TwinrSelfTestRunner
from twinr.ops.support import build_support_bundle
from twinr.ops.usage import TokenUsage, TwinrUsageStore

_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl9l2sAAAAASUVORK5CYII="
)


def _fake_respeaker_snapshot(
    *,
    capture_ready: bool,
    usb_visible: bool,
    host_control_ready: bool,
    arecord_available: bool = True,
    lsusb_available: bool = True,
    transport_reason: str | None = None,
    requires_elevated_permissions: bool = False,
):
    capture_device = (
        SimpleNamespace(card_label="reSpeaker XVF3800 4-Mic Array")
        if capture_ready
        else None
    )
    usb_device = (
        SimpleNamespace(description="Seeed Technology Co., Ltd. reSpeaker XVF3800 4-Mic Array")
        if usb_visible
        else None
    )
    probe = SimpleNamespace(
        capture_ready=capture_ready,
        capture_device=capture_device,
        usb_visible=usb_visible,
        usb_device=usb_device,
        arecord_available=arecord_available,
        lsusb_available=lsusb_available,
        state=(
            "audio_ready"
            if capture_ready
            else "usb_visible_no_capture"
            if usb_visible and arecord_available
            else "not_detected"
        ),
    )
    return SimpleNamespace(
        probe=probe,
        host_control_ready=host_control_ready,
        transport=SimpleNamespace(
            reason=transport_reason,
            requires_elevated_permissions=requires_elevated_permissions,
        ),
        firmware_version=(2, 0, 7),
        direction=SimpleNamespace(
            doa_degrees=277,
            speech_detected=True,
            room_quiet=False,
            beam_azimuth_degrees=(90.0, 270.0, 180.0, 277.0),
            beam_speech_energies=(0.1, 0.2, 0.0, 0.0),
            selected_azimuth_degrees=(277.0, 277.0),
        ),
        mute=SimpleNamespace(
            mute_active=None,
            gpo_logic_levels=(0, 0, 0, 1, 0),
        ),
    )


class OpsModuleTests(unittest.TestCase):
    def test_build_scoped_runtime_config_isolates_auxiliary_runtime_snapshot(self) -> None:
        config = TwinrConfig(
            project_root="/tmp/twinr-project",
            runtime_state_path="state/runtime-state.json",
            restore_runtime_state_on_startup=True,
        )

        scoped = build_scoped_runtime_config(
            config,
            scope_name="whatsapp-channel",
            restore_runtime_state_on_startup=False,
        )
        scoped_path = resolve_scoped_runtime_state_path(config, scope_name="whatsapp-channel")

        self.assertEqual(scoped.runtime_state_path, str(scoped_path))
        self.assertIn("runtime-scopes/whatsapp-channel/runtime-state.json", scoped.runtime_state_path)
        self.assertFalse(scoped.restore_runtime_state_on_startup)

    def test_event_store_tail_returns_latest_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TwinrOpsEventStore(Path(temp_dir) / "events.jsonl")
            for index in range(5):
                store.append(event=f"event_{index}", message=f"message {index}")

            entries = store.tail(limit=2)

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["event"], "event_3")
        self.assertEqual(entries[1]["event"], "event_4")

    def test_event_store_uses_shared_writer_modes_for_current_file_lock_and_created_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store_path = Path(temp_dir) / "artifacts" / "stores" / "ops" / "events.jsonl"
            store = TwinrOpsEventStore(store_path)

            store.append(event="deploy_probe", message="one event")
            store.tail(limit=1)

            file_mode = stat.S_IMODE(os.stat(store_path).st_mode)
            lock_mode = stat.S_IMODE(os.stat(store_path.with_name(".events.jsonl.lock")).st_mode)
            parent_mode = stat.S_IMODE(os.stat(store_path.parent).st_mode)

        self.assertEqual(file_mode, 0o666)
        self.assertEqual(lock_mode, 0o666)
        self.assertEqual(parent_mode, 0o755)

    def test_event_store_append_repairs_cross_service_modes_on_existing_file_and_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store_path = Path(temp_dir) / "artifacts" / "stores" / "ops" / "events.jsonl"
            store = TwinrOpsEventStore(store_path)

            store.append(event="before_repair", message="seed")
            lock_path = store_path.with_name(".events.jsonl.lock")
            os.chmod(store_path, 0o600)
            os.chmod(lock_path, 0o600)

            store.append(event="after_repair", message="repair modes")
            store.tail(limit=2)

            file_mode = stat.S_IMODE(os.stat(store_path).st_mode)
            lock_mode = stat.S_IMODE(os.stat(lock_path).st_mode)

        self.assertEqual(file_mode, 0o666)
        self.assertEqual(lock_mode, 0o666)

    def test_event_store_rotation_downgrades_archives_back_to_read_only_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store_path = Path(temp_dir) / "artifacts" / "stores" / "ops" / "events.jsonl"
            store = TwinrOpsEventStore(store_path, max_file_bytes=64, compression="none")

            store.append(event="first", message="x" * 80)
            store.append(event="second", message="y" * 80)

            archive_names = sorted(name for name in os.listdir(store_path.parent) if name.startswith("events.jsonl."))
            self.assertTrue(archive_names)
            archive_mode = stat.S_IMODE(os.stat(store_path.parent / archive_names[-1]).st_mode)

        self.assertEqual(archive_mode, 0o644)

    def test_usage_store_summarizes_requests_and_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TwinrUsageStore(Path(temp_dir) / "usage.jsonl")
            store.append(
                source="hardware_loop",
                request_kind="conversation",
                model="gpt-5.2",
                response_id="resp_1",
                token_usage=TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150),
            )
            store.append(
                source="realtime_tool",
                request_kind="search",
                model="gpt-5.2-chat-latest",
                request_id="req_2",
                used_web_search=True,
                token_usage=TokenUsage(input_tokens=80, output_tokens=20, total_tokens=100, reasoning_tokens=4),
            )

            summary = store.summary()
            records = store.tail(limit=5)

        self.assertEqual(summary.requests_total, 2)
        self.assertEqual(summary.total_tokens, 250)
        self.assertEqual((summary.by_kind or {})["conversation"], 1)
        self.assertEqual((summary.by_kind or {})["search"], 1)
        self.assertEqual(records[-1].request_id, "req_2")

    def test_build_support_bundle_redacts_secrets_and_includes_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            runtime_state_path = root / "runtime-state.json"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENAI_API_KEY=sk-test-1234",
                        "OPENAI_MODEL=gpt-5.2",
                        f"TWINR_RUNTIME_STATE_PATH={runtime_state_path}",
                        "TWINR_PRINTER_QUEUE=Thermal_GP58",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config = TwinrConfig.from_env(env_path)
            RuntimeSnapshotStore(config.runtime_state_path).save(
                status="waiting",
                memory_turns=(
                    ConversationTurn(
                        "user",
                        "Hallo Twinr",
                        datetime(2026, 3, 13, 8, 0, tzinfo=timezone.utc),
                    ),
                ),
                last_transcript="Hallo Twinr",
                last_response="Guten Morgen",
                user_voice_status="likely_user",
                user_voice_confidence=0.82,
                user_voice_checked_at="2026-03-13T08:05:00+00:00",
            )
            event_store = TwinrOpsEventStore.from_config(config)
            event_store.append(event="turn_started", message="Green button started a turn.")
            event_store.append(event="error", message="Printer error.", level="error")
            TwinrUsageStore.from_config(config).append(
                source="hardware_loop",
                request_kind="conversation",
                model="gpt-5.2",
                response_id="resp_123",
                token_usage=TokenUsage(input_tokens=100, output_tokens=40, total_tokens=140),
            )

            bundle = build_support_bundle(config, env_path=env_path)

            with ZipFile(bundle.bundle_path) as archive:
                redacted_env = json.loads(archive.read("redacted_env.json"))
                events = json.loads(archive.read("events.json"))
                snapshot = json.loads(archive.read("runtime_snapshot.json"))
                health = json.loads(archive.read("system_health.json"))
                usage_summary = json.loads(archive.read("usage_summary.json"))

        self.assertEqual(redacted_env["OPENAI_API_KEY"], "sk-t…1234")
        self.assertEqual(events[-1]["event"], "error")
        self.assertEqual(snapshot["last_response"], "Guten Morgen")
        self.assertNotIn("user_voice_status", snapshot)
        self.assertNotIn("user_voice_confidence", snapshot)
        self.assertNotIn("user_voice_checked_at", snapshot)
        self.assertIn("status", health)
        self.assertEqual(usage_summary["total_tokens"], 140)

    def test_collect_system_health_returns_core_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            snapshot = RuntimeSnapshotStore(Path(temp_dir) / "runtime.json").save(
                status="waiting",
                memory_turns=(),
                last_transcript=None,
                last_response=None,
            )

            health = collect_system_health(config, snapshot=snapshot)

        self.assertIn(health.status, {"ok", "warn", "fail"})
        self.assertTrue(health.hostname)
        self.assertGreaterEqual(len(health.services), 1)

    def test_run_config_checks_reports_missing_key(self) -> None:
        config = TwinrConfig(openai_api_key=None, printer_queue="")

        checks = run_config_checks(config)

        by_key = {check.key: check for check in checks}
        self.assertEqual(by_key["openai_key"].status, "fail")
        self.assertEqual(by_key["printer_queue"].status, "fail")
        self.assertEqual(by_key["pir"].status, "warn")

    def test_run_config_checks_reports_configured_pir(self) -> None:
        config = TwinrConfig(
            openai_api_key="sk-test",
            printer_queue="Thermal_GP58",
            gpio_chip="/dev/gpiochip4",
            green_button_gpio=23,
            yellow_button_gpio=22,
            pir_motion_gpio=26,
            pir_active_high=True,
            pir_bias="pull-down",
        )

        with patch(
            "twinr.ops.checks._probe_gpio_lines",
            return_value=SimpleNamespace(
                status="ok",
                detail="GPIO chip exposes the configured line offsets.",
            ),
        ):
            checks = run_config_checks(config)

        by_key = {check.key: check for check in checks}
        self.assertEqual(by_key["pir"].status, "ok")
        self.assertIn("GPIO 26", by_key["pir"].detail)

    def test_run_config_checks_reports_proactive_audio_device(self) -> None:
        config = TwinrConfig(
            audio_input_device="default",
            proactive_audio_enabled=True,
            proactive_audio_input_device="plughw:CARD=CameraB409241,DEV=0",
        )

        with patch(
            "twinr.ops.checks._probe_alsa_device",
            return_value=SimpleNamespace(
                status="ok",
                detail="ALSA lists the configured input device.",
            ),
        ):
            checks = run_config_checks(config)

        by_key = {check.key: check for check in checks}
        self.assertEqual(by_key["proactive_audio_input"].status, "ok")
        self.assertIn("CameraB409241", by_key["proactive_audio_input"].detail)

    def test_run_config_checks_reports_respeaker_runtime_ready(self) -> None:
        config = TwinrConfig(
            audio_input_device="plughw:CARD=Array,DEV=0",
            proactive_audio_enabled=True,
            proactive_audio_input_device="plughw:CARD=Array,DEV=0",
        )

        fake_snapshot = _fake_respeaker_snapshot(
            capture_ready=True,
            usb_visible=True,
            host_control_ready=True,
        )

        with patch("twinr.ops.checks.capture_respeaker_primitive_snapshot", return_value=fake_snapshot):
            checks = run_config_checks(config)

        by_key = {check.key: check for check in checks}
        self.assertEqual(by_key["respeaker_xvf3800"].status, "ok")
        self.assertIn("host-control primitives are readable", by_key["respeaker_xvf3800"].detail)

    def test_run_config_checks_warns_when_respeaker_is_usb_visible_without_capture(self) -> None:
        config = TwinrConfig(
            audio_input_device="plughw:CARD=Array,DEV=0",
            proactive_audio_enabled=True,
            proactive_audio_input_device="plughw:CARD=Array,DEV=0",
        )

        fake_snapshot = _fake_respeaker_snapshot(
            capture_ready=False,
            usb_visible=True,
            host_control_ready=False,
        )

        with patch("twinr.ops.checks.capture_respeaker_primitive_snapshot", return_value=fake_snapshot):
            checks = run_config_checks(config)

        by_key = {check.key: check for check in checks}
        self.assertEqual(by_key["respeaker_xvf3800"].status, "warn")
        self.assertIn("DFU/safe mode", by_key["respeaker_xvf3800"].detail)

    def test_run_config_checks_warns_when_host_control_permissions_are_missing(self) -> None:
        config = TwinrConfig(
            audio_input_device="plughw:CARD=Array,DEV=0",
            proactive_audio_enabled=True,
            proactive_audio_input_device="plughw:CARD=Array,DEV=0",
        )

        fake_snapshot = _fake_respeaker_snapshot(
            capture_ready=True,
            usb_visible=True,
            host_control_ready=False,
            transport_reason="permission_denied_or_transport_blocked",
            requires_elevated_permissions=True,
        )

        with patch("twinr.ops.checks.capture_respeaker_primitive_snapshot", return_value=fake_snapshot):
            checks = run_config_checks(config)

        by_key = {check.key: check for check in checks}
        self.assertEqual(by_key["respeaker_xvf3800"].status, "warn")
        self.assertIn("runtime user likely lacks", by_key["respeaker_xvf3800"].detail.lower())

    def test_run_config_checks_rejects_pir_button_collision(self) -> None:
        config = TwinrConfig(
            green_button_gpio=23,
            yellow_button_gpio=22,
            pir_motion_gpio=22,
        )

        checks = run_config_checks(config)

        by_key = {check.key: check for check in checks}
        self.assertEqual(by_key["pir"].status, "fail")

    def test_run_config_checks_rejects_display_button_collision(self) -> None:
        config = TwinrConfig(
            display_driver="waveshare_4in2_v2",
            green_button_gpio=23,
            yellow_button_gpio=24,
            display_busy_gpio=24,
        )

        checks = run_config_checks(config)

        by_key = {check.key: check for check in checks}
        self.assertEqual(by_key["display_gpio"].status, "fail")
        self.assertIn("BUSY GPIO 24 collides with button `yellow`", by_key["display_gpio"].detail)

    def test_run_config_checks_accepts_hdmi_display_character_device_path(self) -> None:
        config = TwinrConfig(
            display_driver="hdmi_fbdev",
            display_fb_path="/dev/null",
        )

        checks = run_config_checks(config)

        by_key = {check.key: check for check in checks}
        self.assertEqual(by_key["display_gpio"].status, "ok")
        self.assertIn("does not require display GPIO pins", by_key["display_gpio"].detail)

    def test_run_config_checks_accepts_wayland_display_socket(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime_dir = Path(temp_dir)
            socket_path = runtime_dir / "wayland-0"
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                server.bind(str(socket_path))
                config = TwinrConfig(
                    display_driver="hdmi_wayland",
                    display_wayland_display="wayland-0",
                    display_wayland_runtime_dir=str(runtime_dir),
                )

                checks = run_config_checks(config)
            finally:
                server.close()

        by_key = {check.key: check for check in checks}
        self.assertEqual(by_key["display_gpio"].status, "ok")
        self.assertIn(str(socket_path), by_key["display_gpio"].detail)

    def test_run_config_checks_includes_self_coding_codex_readiness(self) -> None:
        config = TwinrConfig()
        fake_report = SimpleNamespace(
            ready=False,
            detail="codex auth file is missing: /tmp/.codex/auth.json",
        )

        with patch(
            "twinr.agent.self_coding.codex_driver.environment.collect_codex_sdk_environment_report",
            return_value=fake_report,
        ):
            checks = run_config_checks(config)

        by_key = {check.key: check for check in checks}
        self.assertEqual(by_key["self_coding_codex"].status, "fail")
        self.assertIn("codex auth file is missing", by_key["self_coding_codex"].detail)

    def test_pir_self_test_succeeds_with_motion_event(self) -> None:
        class FakePirMonitor:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def snapshot_value(self) -> int:
                return 0

            def wait_for_motion(self, *, duration_s: float, poll_timeout: float):
                return type(
                    "Event",
                    (),
                    {
                        "raw_edge": "rising",
                        "motion_detected": True,
                    },
                )()

        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir, pir_motion_gpio=26)
            runner = TwinrSelfTestRunner(
                config,
                pir_monitor_factory=lambda _config: FakePirMonitor(),
            )

            result = runner.run("pir")

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.summary, "PIR motion detected.")

    def test_button_self_test_is_blocked_while_realtime_loop_lock_is_held(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
                green_button_gpio=23,
                yellow_button_gpio=22,
            )
            runner = TwinrSelfTestRunner(config)

            with loop_instance_lock(config, "realtime-loop"):
                result = runner.run("buttons")

        self.assertEqual(result.status, "blocked")
        self.assertIn("realtime loop", result.summary)
        self.assertIn("Stop it first", result.summary)

    def test_pir_self_test_is_blocked_while_hardware_loop_lock_is_held(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
                pir_motion_gpio=26,
            )
            runner = TwinrSelfTestRunner(config)

            with loop_instance_lock(config, "hardware-loop"):
                result = runner.run("pir")

        self.assertEqual(result.status, "blocked")
        self.assertIn("hardware loop", result.summary)
        self.assertIn("Stop it first", result.summary)

    def test_proactive_mic_self_test_uses_configured_background_device(self) -> None:
        class FakeAmbientSampler:
            def __init__(self) -> None:
                self.calls: list[int] = []

            def sample_levels(self, *, duration_ms: int | None = None):
                self.calls.append(duration_ms or 0)
                return AmbientAudioLevelSample(
                    duration_ms=duration_ms or 900,
                    chunk_count=5,
                    active_chunk_count=2,
                    average_rms=640,
                    peak_rms=1200,
                    active_ratio=0.4,
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                proactive_audio_enabled=True,
                proactive_audio_input_device="plughw:CARD=CameraB409241,DEV=0",
                proactive_audio_sample_ms=900,
            )
            sampler = FakeAmbientSampler()
            runner = TwinrSelfTestRunner(
                config,
                ambient_sampler_factory=lambda _config: sampler,
            )

            result = runner.run("proactive_mic")

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.summary, "Proactive background-microphone sample captured.")
        self.assertIn("plughw:CARD=CameraB409241,DEV=0", result.details[0])
        self.assertIn("Speech-like activity: yes", result.details)

    def test_proactive_mic_self_test_is_blocked_when_voice_orchestrator_owns_same_device(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                proactive_audio_enabled=True,
                voice_orchestrator_enabled=True,
                voice_orchestrator_ws_url="ws://127.0.0.1:8765/voice",
                audio_input_device="plughw:CARD=Array,DEV=0",
                proactive_audio_input_device="plughw:CARD=Array,DEV=0",
                voice_orchestrator_audio_device="plughw:CARD=Array,DEV=0",
            )
            runner = TwinrSelfTestRunner(
                config,
                ambient_sampler_factory=lambda _config: self.fail("ambient sampler must not be constructed"),
            )

            with patch("twinr.proactive.runtime.service_impl.compat.loop_lock_owner", side_effect=[9876]):
                result = runner.run("proactive_mic")

        self.assertEqual(result.status, "blocked")
        self.assertIn("voice orchestrator owns the same capture device", result.summary)

    def test_proactive_mic_self_test_uses_pcm_when_voice_orchestrator_is_inactive(self) -> None:
        class FakeAmbientSampler:
            def sample_levels(self, *, duration_ms: int | None = None):
                return AmbientAudioLevelSample(
                    duration_ms=duration_ms or 900,
                    chunk_count=4,
                    active_chunk_count=0,
                    average_rms=220,
                    peak_rms=400,
                    active_ratio=0.0,
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                proactive_audio_enabled=True,
                proactive_audio_sample_ms=900,
                voice_orchestrator_enabled=True,
                voice_orchestrator_ws_url="ws://127.0.0.1:8765/voice",
                audio_input_device="plughw:CARD=Array,DEV=0",
                proactive_audio_input_device="plughw:CARD=Array,DEV=0",
                voice_orchestrator_audio_device="plughw:CARD=Array,DEV=0",
            )
            runner = TwinrSelfTestRunner(
                config,
                ambient_sampler_factory=lambda _config: FakeAmbientSampler(),
            )

            with patch("twinr.proactive.runtime.service_impl.compat.loop_lock_owner", side_effect=[None, None]):
                result = runner.run("proactive_mic")

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.summary, "Proactive background-microphone sample captured.")
        self.assertIn("plughw:CARD=Array,DEV=0", result.details[0])

    def test_printer_self_test_requires_manual_confirmation(self) -> None:
        class FakePrinter:
            def __init__(self) -> None:
                self.payloads: list[str] = []

            def print_text(self, text: str) -> str:
                self.payloads.append(text)
                return "request id is Thermal_GP58-31 (1 file(s))"

        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir, printer_queue="Thermal_GP58")
            fake_printer = FakePrinter()
            runner = TwinrSelfTestRunner(
                config,
                printer_factory=lambda _config: fake_printer,
            )

            result = runner.run("printer")

        self.assertEqual(fake_printer.payloads, ["Twinr self-test\nPrinter path OK."])
        self.assertEqual(result.status, "warn")
        self.assertEqual(result.summary, "Printer job submitted. Confirm the paper output on the device.")
        self.assertIn("Queue: Thermal_GP58", result.details)
        self.assertIn("CUPS response: request id is Thermal_GP58-31 (1 file(s))", result.details)
        self.assertIn("Physical paper output must be confirmed at the printer.", result.details)

    def test_aideck_camera_self_test_reports_image_sanity(self) -> None:
        class FakeCamera:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def capture_photo(self, *, output_path, filename: str):
                self.calls.append(
                    {
                        "output_path": output_path,
                        "filename": filename,
                    }
                )
                return SimpleNamespace(
                    data=_TINY_PNG,
                    content_type="image/png",
                    filename=filename,
                    source_device="aideck://192.168.4.1:5000",
                    input_format="aideck-cpx-raw-bayer",
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                camera_device="aideck://192.168.4.1:5000",
            )
            fake_camera = FakeCamera()
            runner = TwinrSelfTestRunner(
                config,
                camera_factory=lambda _config: fake_camera,
            )
            with patch.object(runner, "_probe_tcp_endpoint", return_value=None):
                with patch.object(runner, "_validate_image_artifact", return_value=None):
                        result = runner.run("aideck_camera")

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.summary, "AI-Deck frame captured.")
        self.assertIn("AI-Deck stream became reachable and returned one frame.", result.details)
        self.assertIn("Input format: aideck-cpx-raw-bayer", result.details)
        self.assertIn("Image size: 1x1", result.details)

    def test_aideck_camera_self_test_writes_artifact_from_in_memory_capture(self) -> None:
        class FakeCamera:
            def __init__(self) -> None:
                self.output_paths: list[object] = []

            def capture_photo(self, *, output_path, filename: str):
                self.output_paths.append(output_path)
                return SimpleNamespace(
                    data=_TINY_PNG,
                    content_type="image/png",
                    filename=filename,
                    source_device="aideck://192.168.4.1:5000",
                    input_format="aideck-cpx-raw-bayer",
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                camera_device="aideck://192.168.4.1:5000",
            )
            fake_camera = FakeCamera()
            runner = TwinrSelfTestRunner(
                config,
                camera_factory=lambda _config: fake_camera,
            )

            with patch.object(runner, "_probe_tcp_endpoint", return_value=None):
                with patch.object(runner, "_validate_image_artifact", return_value=None):
                        result = runner.run("aideck_camera")

        self.assertEqual(result.status, "ok")
        self.assertEqual(fake_camera.output_paths, [None])

    def test_aideck_camera_self_test_requires_aideck_device_uri(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir, camera_device="/dev/video0")
            runner = TwinrSelfTestRunner(config)

            result = runner.run("aideck_camera")

        self.assertEqual(result.status, "fail")
        self.assertIn("aideck://", result.summary)

    def test_drone_stack_self_test_queues_and_cancels_pending_manual_arm_mission(self) -> None:
        class FakeDroneMission:
            def __init__(self, *, mission_id: str, state: str, summary: str) -> None:
                self.mission_id = mission_id
                self.state = state
                self.summary = summary

        class FakeDroneState:
            def __init__(self) -> None:
                self.service_status = "ready"
                self.skill_layer_mode = "stationary_observe_only"
                self.pose = SimpleNamespace(tracking_state="tracking")
                self.safety = SimpleNamespace(
                    radio_ready=True,
                    pose_ready=True,
                    can_arm=True,
                    reasons=(),
                )

        class FakeDroneClient:
            def __init__(self, *, base_url: str, timeout_s: float) -> None:
                self.base_url = base_url
                self.timeout_s = timeout_s

            def health_payload(self) -> dict[str, object]:
                return {"ok": True, "can_arm": True}

            def state(self) -> FakeDroneState:
                return FakeDroneState()

            def create_inspect_mission(self, **kwargs) -> FakeDroneMission:
                self.last_request = kwargs
                return FakeDroneMission(
                    mission_id="DRN-123",
                    state="pending_manual_arm",
                    summary="queued",
                )

            def cancel_mission(self, mission_id: str) -> FakeDroneMission:
                return FakeDroneMission(
                    mission_id=mission_id,
                    state="cancelled",
                    summary="cancelled",
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                drone_enabled=True,
                drone_base_url="http://127.0.0.1:8791",
            )
            runner = TwinrSelfTestRunner(config)
            with patch("twinr.ops.self_test.RemoteDroneServiceClient", FakeDroneClient):
                result = runner.run("drone_stack")

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.summary, "Drone daemon reached and bounded inspect mission queued safely.")
        self.assertIn("Mission state: pending_manual_arm", result.details)
        self.assertIn("Cancel state: cancelled", result.details)


if __name__ == "__main__":
    unittest.main()
