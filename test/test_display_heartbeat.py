from pathlib import Path
import stat
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.display.heartbeat import DisplayHeartbeatStore, build_display_heartbeat


class DisplayHeartbeatStoreTests(unittest.TestCase):
    def test_save_persists_operator_readable_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            store = DisplayHeartbeatStore.from_config(config)

            store.save(
                build_display_heartbeat(
                    pid=1234,
                    runtime_status="waiting",
                    phase="rendered",
                    seq=1,
                    updated_at=datetime(2026, 4, 5, 16, 30, tzinfo=timezone.utc),
                    updated_monotonic_ns=123_456_789,
                )
            )

            self.assertEqual(stat.S_IMODE(store.path.stat().st_mode), 0o644)

    def test_save_mirrors_runtime_heartbeat_to_legacy_fallback_with_operator_readable_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime_dir = Path(temp_dir) / "runtime"
            config = TwinrConfig(project_root=temp_dir)
            with mock.patch.dict("os.environ", {"RUNTIME_DIRECTORY": str(runtime_dir)}, clear=False):
                store = DisplayHeartbeatStore.from_config(config)

                store.save(
                    build_display_heartbeat(
                        pid=1234,
                        runtime_status="waiting",
                        phase="rendered",
                        seq=1,
                        updated_at=datetime(2026, 4, 9, 10, 8, tzinfo=timezone.utc),
                        updated_monotonic_ns=123_456_789,
                    )
                )

            legacy_path = Path(temp_dir) / "artifacts" / "stores" / "ops" / "display_heartbeat.json"
            self.assertEqual(store.path, runtime_dir / "display_heartbeat.json")
            self.assertEqual(store.fallback_paths, (legacy_path,))
            self.assertTrue(legacy_path.exists())
            self.assertEqual(stat.S_IMODE(store.path.stat().st_mode), 0o644)
            self.assertEqual(stat.S_IMODE(legacy_path.stat().st_mode), 0o644)
            self.assertEqual(store.path.read_text(encoding="utf-8"), legacy_path.read_text(encoding="utf-8"))

    def test_save_surfaces_mode_write_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            store = DisplayHeartbeatStore.from_config(config)

            with mock.patch(
                "twinr.display.heartbeat.os.fchmod",
                side_effect=PermissionError("chmod denied"),
            ):
                with self.assertRaises(PermissionError):
                    store.save(
                        build_display_heartbeat(
                            pid=1234,
                            runtime_status="waiting",
                            phase="rendered",
                            seq=1,
                            updated_at=datetime(2026, 4, 7, 15, 50, tzinfo=timezone.utc),
                            updated_monotonic_ns=123_456_789,
                        )
                    )


if __name__ == "__main__":
    unittest.main()
