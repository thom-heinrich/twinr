from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.runtime import TwinrRuntime
from twinr.agent.base_agent.runtime_state import RuntimeSnapshotStore


class RuntimeContextTests(unittest.TestCase):
    def _config(self, temp_dir: str) -> TwinrConfig:
        root = Path(temp_dir)
        return TwinrConfig(
            project_root=temp_dir,
            long_term_memory_path=str(root / "state" / "chonkydb"),
            runtime_state_path=str(root / "state" / "runtime-state.json"),
        )

    def test_provider_context_accepts_datetime_voice_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = TwinrRuntime(config=self._config(temp_dir))
            try:
                runtime.user_voice_status = "likely_user"
                runtime.user_voice_confidence = 0.81
                runtime.user_voice_checked_at = datetime.now(timezone.utc)

                context = runtime.provider_conversation_context()

                self.assertGreater(len(context), 0)
                self.assertTrue(all(isinstance(item, tuple) and len(item) == 2 for item in context))
            finally:
                runtime.shutdown(timeout_s=1.0)

    def test_snapshot_restore_normalizes_voice_timestamp_to_string(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)
            RuntimeSnapshotStore(config.runtime_state_path).save(
                status="waiting",
                memory_turns=(),
                last_transcript=None,
                last_response="Schon gespeichert.",
                user_voice_status="likely_user",
                user_voice_confidence=0.93,
                user_voice_checked_at="2026-03-15T18:57:56+00:00",
            )

            runtime = TwinrRuntime(
                config=TwinrConfig(
                    project_root=temp_dir,
                    long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                    runtime_state_path=str(Path(temp_dir) / "state" / "runtime-state.json"),
                    restore_runtime_state_on_startup=True,
                )
            )
            try:
                self.assertEqual(runtime.user_voice_checked_at, "2026-03-15T18:57:56Z")
                runtime.provider_conversation_context()
            finally:
                runtime.shutdown(timeout_s=1.0)


if __name__ == "__main__":
    unittest.main()
