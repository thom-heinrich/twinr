from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.ops.process_memory import ProcessMemoryMetrics, StreamingMemoryAttributionStore
from twinr.ops.streaming_memory_probe import StreamingMemoryProbe


class StreamingMemoryProbeTests(unittest.TestCase):
    def test_probe_records_first_sample_and_then_waits_for_time_or_growth(self) -> None:
        calls: list[dict[str, object]] = []
        probe_path = Path("/tmp/fake-streaming-memory.json")
        recorded_anons = iter((128_000, 220_000))

        class _FakeStore:
            def __init__(self, path: Path) -> None:
                self.path = path

            def record_phase(
                self,
                *,
                label: str,
                owner_label: str | None = None,
                owner_detail: str | None = None,
                replace: bool = False,
            ):
                calls.append(
                    {
                        "label": label,
                        "owner_label": owner_label,
                        "owner_detail": owner_detail,
                        "replace": replace,
                    }
                )
                return SimpleNamespace(
                    current_metrics=ProcessMemoryMetrics(anonymous_kb=next(recorded_anons))
                )

        probe = StreamingMemoryProbe(
            path=probe_path,
            label="voice_orchestrator.capture_loop",
            owner_label="voice_orchestrator.capture_loop",
            owner_detail="Voice orchestrator capture loop is active.",
            sample_interval_s=5.0,
            growth_threshold_kb=64_000,
        )

        with mock.patch(
            "twinr.ops.streaming_memory_probe.ProcessMemoryMetrics.from_proc",
            side_effect=(
                ProcessMemoryMetrics(anonymous_kb=128_000),
                ProcessMemoryMetrics(anonymous_kb=132_000),
                ProcessMemoryMetrics(anonymous_kb=220_000),
            ),
        ):
            with mock.patch(
                "twinr.ops.streaming_memory_probe.StreamingMemoryAttributionStore",
                _FakeStore,
            ):
                with mock.patch(
                    "twinr.ops.streaming_memory_probe.time.monotonic",
                    side_effect=(10.0, 12.0, 13.0),
                ):
                    self.assertTrue(probe.maybe_record())
                    self.assertFalse(probe.maybe_record())
                    self.assertTrue(probe.maybe_record())

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["label"], "voice_orchestrator.capture_loop")
        self.assertEqual(calls[0]["owner_label"], "voice_orchestrator.capture_loop")
        self.assertTrue(bool(calls[0]["replace"]))
        self.assertEqual(calls[1]["owner_detail"], "Voice orchestrator capture loop is active.")

    def test_probe_from_config_uses_default_streaming_memory_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            probe = StreamingMemoryProbe.from_config(
                config,
                label="proactive_monitor.tick",
                owner_label="proactive_monitor.tick",
                owner_detail="Proactive monitor tick is active.",
            )

            self.assertEqual(
                probe.path,
                StreamingMemoryAttributionStore.from_config(config).path,
            )


if __name__ == "__main__":
    unittest.main()
