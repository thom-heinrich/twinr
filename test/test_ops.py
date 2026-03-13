from datetime import datetime, timezone
from pathlib import Path
import json
import sys
import tempfile
import unittest
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import RuntimeSnapshotStore, TwinrConfig
from twinr.hardware.audio import AmbientAudioLevelSample
from twinr.memory import ConversationTurn
from twinr.ops import (
    TwinrOpsEventStore,
    TwinrSelfTestRunner,
    TwinrUsageStore,
    TokenUsage,
    build_support_bundle,
    collect_system_health,
    run_config_checks,
)
from twinr.ops.locks import loop_instance_lock


class OpsModuleTests(unittest.TestCase):
    def test_event_store_tail_returns_latest_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TwinrOpsEventStore(Path(temp_dir) / "events.jsonl")
            for index in range(5):
                store.append(event=f"event_{index}", message=f"message {index}")

            entries = store.tail(limit=2)

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["event"], "event_3")
        self.assertEqual(entries[1]["event"], "event_4")

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
        self.assertEqual(summary.by_kind["conversation"], 1)
        self.assertEqual(summary.by_kind["search"], 1)
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
            green_button_gpio=23,
            yellow_button_gpio=22,
            pir_motion_gpio=26,
            pir_active_high=True,
            pir_bias="pull-down",
        )

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

        checks = run_config_checks(config)

        by_key = {check.key: check for check in checks}
        self.assertEqual(by_key["proactive_audio_input"].status, "ok")
        self.assertIn("CameraB409241", by_key["proactive_audio_input"].detail)

    def test_run_config_checks_rejects_pir_button_collision(self) -> None:
        config = TwinrConfig(
            green_button_gpio=23,
            yellow_button_gpio=22,
            pir_motion_gpio=22,
        )

        checks = run_config_checks(config)

        by_key = {check.key: check for check in checks}
        self.assertEqual(by_key["pir"].status, "fail")

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
        self.assertIn("Speech-like activity: yes", result.details[-1])

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


if __name__ == "__main__":
    unittest.main()
