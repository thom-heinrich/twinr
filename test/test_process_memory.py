from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.ops.process_memory import (
    ProcessMemoryMetrics,
    StreamingMemoryAttributionStore,
    load_current_streaming_memory_snapshot,
)


class ProcessMemoryTests(unittest.TestCase):
    def test_store_persists_operator_readable_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            store = StreamingMemoryAttributionStore.from_config(config)

            with mock.patch("twinr.ops.process_memory.os.getpid", return_value=4242):
                with mock.patch("twinr.ops.process_memory._read_boot_id", return_value="boot-1"):
                    with mock.patch("twinr.ops.process_memory._read_process_start_ticks", return_value=777):
                        with mock.patch(
                            "twinr.ops.process_memory.ProcessMemoryMetrics.from_proc",
                            return_value=ProcessMemoryMetrics(vm_rss_kb=240_000, anonymous_kb=180_000),
                        ):
                            store.record_phase(
                                label="streaming_loop.lock_acquired",
                                owner_label="streaming_loop.lock",
                                reset=True,
                            )

            self.assertEqual(stat.S_IMODE(store.path.stat().st_mode), 0o644)

    def test_store_selects_largest_anonymous_delta_as_owner(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            store = StreamingMemoryAttributionStore.from_config(config)
            metrics = (
                ProcessMemoryMetrics(vm_rss_kb=240_000, anonymous_kb=180_000),
                ProcessMemoryMetrics(vm_rss_kb=1_900_000, anonymous_kb=1_760_000),
                ProcessMemoryMetrics(vm_rss_kb=620_000, anonymous_kb=420_000),
            )

            with mock.patch("twinr.ops.process_memory.os.getpid", return_value=4242):
                with mock.patch("twinr.ops.process_memory._read_boot_id", return_value="boot-1"):
                    with mock.patch("twinr.ops.process_memory._read_process_start_ticks", return_value=777):
                        with mock.patch(
                            "twinr.ops.process_memory.ProcessMemoryMetrics.from_proc",
                            side_effect=metrics,
                        ):
                            store.record_phase(
                                label="streaming_loop.lock_acquired",
                                owner_label="streaming_loop.lock",
                                reset=True,
                            )
                            store.record_phase(
                                label="display.hdmi_wayland.native_window_frame_presented",
                                owner_label="display_companion.hdmi_wayland.native_window_frame",
                            )
                            snapshot = store.record_phase(
                                label="streaming_loop.provider_bundle_ready",
                                owner_label="streaming_providers.bundle",
                            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.pid, 4242)
        self.assertEqual(
            snapshot.owner_label,
            "display_companion.hdmi_wayland.native_window_frame",
        )
        self.assertGreater(snapshot.owner_anonymous_delta_kb or 0, 1_500_000)
        self.assertEqual(len(snapshot.phases), 3)

    def test_load_current_snapshot_rejects_pid_reuse_when_start_ticks_change(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            store = StreamingMemoryAttributionStore.from_config(config)

            with mock.patch("twinr.ops.process_memory.os.getpid", return_value=31337):
                with mock.patch("twinr.ops.process_memory._read_boot_id", return_value="boot-1"):
                    with mock.patch("twinr.ops.process_memory._read_process_start_ticks", return_value=123):
                        with mock.patch(
                            "twinr.ops.process_memory.ProcessMemoryMetrics.from_proc",
                            return_value=ProcessMemoryMetrics(vm_rss_kb=240_000, anonymous_kb=180_000),
                        ):
                            store.record_phase(
                                label="streaming_loop.lock_acquired",
                                owner_label="streaming_loop.lock",
                                reset=True,
                            )

            with mock.patch("twinr.ops.process_memory._read_boot_id", return_value="boot-1"):
                with mock.patch("twinr.ops.process_memory._read_process_start_ticks", return_value=999):
                    loaded = load_current_streaming_memory_snapshot(config, expected_pid=31337)

        self.assertIsNone(loaded)


if __name__ == "__main__":
    unittest.main()
