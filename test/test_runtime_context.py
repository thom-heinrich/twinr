from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.runtime import TwinrRuntime
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError
from twinr.agent.base_agent.state.snapshot import RuntimeSnapshotStore


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

    def test_search_provider_context_skips_long_term_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = TwinrRuntime(config=self._config(temp_dir))
            try:
                runtime.memory.remember("user", "Erster Turn")
                runtime.memory.remember("assistant", "Zweiter Turn")
                runtime.memory.remember("user", "Letzte Frage")
                runtime.memory.remember("assistant", "Letzte Antwort")

                class _FailingLongTermMemory:
                    def build_provider_context(self, query_text):
                        raise AssertionError("search context must not query long-term provider context")

                    def build_tool_provider_context(self, query_text):
                        raise AssertionError("search context must not query long-term tool context")

                runtime.long_term_memory = _FailingLongTermMemory()

                context = runtime.search_provider_conversation_context()

                self.assertGreater(len(context), 0)
                self.assertEqual(
                    [(role, content) for role, content in context if role != "system"],
                    [("user", "Erster Turn"), ("assistant", "Zweiter Turn"), ("user", "Letzte Frage"), ("assistant", "Letzte Antwort")][-3:],
                )
            finally:
                runtime.shutdown(timeout_s=1.0)

    def test_provider_context_degrades_when_remote_long_term_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = TwinrRuntime(config=self._config(temp_dir))
            try:
                runtime.memory.remember("user", "Letzte Frage")

                class _UnavailableLongTermMemory:
                    def build_provider_context(self, query_text):
                        raise LongTermRemoteUnavailableError("remote unavailable")

                    def build_tool_provider_context(self, query_text):
                        raise LongTermRemoteUnavailableError("remote unavailable")

                runtime.long_term_memory = _UnavailableLongTermMemory()

                provider_context = runtime.provider_conversation_context()
                tool_context = runtime.tool_provider_conversation_context()

                self.assertIn(("user", "Letzte Frage"), provider_context)
                self.assertIn(("user", "Letzte Frage"), tool_context)
            finally:
                runtime.shutdown(timeout_s=1.0)

    def test_provider_context_raises_when_required_remote_long_term_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = TwinrRuntime(
                config=TwinrConfig(
                    project_root=temp_dir,
                    long_term_memory_enabled=True,
                    long_term_memory_mode="remote_primary",
                    long_term_memory_remote_required=True,
                    long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                    runtime_state_path=str(Path(temp_dir) / "state" / "runtime-state.json"),
                )
            )
            try:
                runtime.memory.remember("user", "Letzte Frage")
                runtime.last_transcript = "Was haben wir heute besprochen?"

                class _UnavailableLongTermMemory:
                    def build_provider_context(self, query_text):
                        raise LongTermRemoteUnavailableError("remote unavailable")

                    def build_tool_provider_context(self, query_text):
                        raise LongTermRemoteUnavailableError("remote unavailable")

                runtime.long_term_memory = _UnavailableLongTermMemory()

                with self.assertRaises(LongTermRemoteUnavailableError):
                    runtime.provider_conversation_context()
                with self.assertRaises(LongTermRemoteUnavailableError):
                    runtime.tool_provider_conversation_context()
                first_word_context = runtime.first_word_provider_conversation_context()
                self.assertIn(("user", "Letzte Frage"), first_word_context)
            finally:
                runtime.shutdown(timeout_s=1.0)

    def test_first_word_context_uses_local_summary_and_skips_remote_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = TwinrRuntime(config=self._config(temp_dir))
            try:
                runtime.memory.remember("user", "Mir ging es gestern nicht gut.")
                runtime.memory.remember_note(
                    kind="fact",
                    content="Die Nutzerin hatte gestern Kopfweh.",
                    source="memory",
                )

                class _FailingLongTermMemory:
                    def build_provider_context(self, query_text):
                        raise AssertionError("first-word context must not query remote long-term memory")

                runtime.long_term_memory = _FailingLongTermMemory()

                context = runtime.first_word_provider_conversation_context()

                summary_messages = [content for role, content in context if role == "system"]
                self.assertTrue(
                    any(
                        "Twinr memory summary:" in message
                        and "Die Nutzerin hatte gestern Kopfweh." in message
                        for message in summary_messages
                    )
                )
                self.assertIn(("user", "Mir ging es gestern nicht gut."), context)
            finally:
                runtime.shutdown(timeout_s=1.0)

    def test_supervisor_context_uses_local_summary_and_skips_remote_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = TwinrRuntime(config=self._config(temp_dir))
            try:
                runtime.memory.remember("user", "Ich bin morgen in Schwarzenbek.")
                runtime.memory.remember_note(
                    kind="fact",
                    content="Die Nutzerin fährt morgen nach Schwarzenbek.",
                    source="memory",
                )

                class _FailingLongTermMemory:
                    def build_provider_context(self, query_text):
                        raise AssertionError("supervisor context must not query remote long-term memory")

                runtime.long_term_memory = _FailingLongTermMemory()

                context = runtime.supervisor_provider_conversation_context()

                summary_messages = [content for role, content in context if role == "system"]
                self.assertTrue(
                    any(
                        "Twinr memory summary:" in message
                        and "Die Nutzerin fährt morgen nach Schwarzenbek." in message
                        for message in summary_messages
                    )
                )
                self.assertIn(("user", "Ich bin morgen in Schwarzenbek."), context)
            finally:
                runtime.shutdown(timeout_s=1.0)

    def test_supervisor_direct_context_uses_full_provider_memory_for_current_query(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = TwinrRuntime(config=self._config(temp_dir))
            try:
                runtime.memory.remember("user", "Wir haben über Medikamente gesprochen.")
                recorded_queries: list[str] = []

                class _RecordingLongTermMemory:
                    def build_provider_context(self, query_text):
                        recorded_queries.append(str(query_text))
                        return SimpleNamespace(
                            system_messages=lambda: ("Structured memory for direct supervisor reply.",)
                        )

                    def build_tool_provider_context(self, query_text):
                        raise AssertionError("direct supervisor context must use normal provider memory")

                runtime.long_term_memory = _RecordingLongTermMemory()

                context = runtime.supervisor_direct_provider_conversation_context(
                    "Worüber haben wir heute gesprochen?"
                )

                self.assertEqual(recorded_queries, ["Worüber haben wir heute gesprochen?"])
                self.assertIn(
                    ("system", "Structured memory for direct supervisor reply."),
                    context,
                )
                self.assertIn(("user", "Wir haben über Medikamente gesprochen."), context)
            finally:
                runtime.shutdown(timeout_s=1.0)

    def test_runtime_startup_degrades_when_remote_long_term_bootstrap_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)

            class _FailingLongTermMemory:
                def ensure_remote_ready(self):
                    raise LongTermRemoteUnavailableError("remote unavailable at startup")

                def shutdown(self, *, timeout_s: float = 0.0):
                    return None

                def close(self):
                    return None

            with patch("twinr.agent.base_agent.runtime.base.LongTermMemoryService.from_config", return_value=_FailingLongTermMemory()):
                runtime = TwinrRuntime(config=config)
            try:
                self.assertIsNotNone(runtime.ops_events)
                self.assertEqual(runtime.status.value, "waiting")
            finally:
                runtime.shutdown(timeout_s=1.0)

    def test_runtime_startup_enters_error_when_required_remote_bootstrap_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_required=True,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                runtime_state_path=str(Path(temp_dir) / "state" / "runtime-state.json"),
            )

            class _FailingLongTermMemory:
                def ensure_remote_ready(self):
                    raise LongTermRemoteUnavailableError("remote unavailable at startup")

                def remote_required(self):
                    return True

                def remote_status(self):
                    raise AssertionError("startup error path should use ensure_remote_ready")

                def shutdown(self, *, timeout_s: float = 0.0):
                    return None

                def close(self):
                    return None

            with patch(
                "twinr.agent.base_agent.runtime.base.LongTermMemoryService.from_config",
                return_value=_FailingLongTermMemory(),
            ):
                runtime = TwinrRuntime(config=config)
            try:
                self.assertEqual(runtime.status.value, "error")
                snapshot = RuntimeSnapshotStore(config.runtime_state_path).load()
                self.assertEqual(snapshot.status, "error")
                self.assertEqual(snapshot.error_message, "remote unavailable at startup")
            finally:
                runtime.shutdown(timeout_s=1.0)

    def test_check_required_remote_dependency_uses_deep_ready_check(self) -> None:
        runtime = TwinrRuntime.__new__(TwinrRuntime)
        runtime.config = TwinrConfig(
            long_term_memory_enabled=True,
            long_term_memory_mode="remote_primary",
            long_term_memory_remote_required=True,
        )

        class _FalsePositiveRemote:
            def remote_required(self):
                return True

            def ensure_remote_ready(self):
                raise LongTermRemoteUnavailableError("archive shard unavailable")

            def remote_status(self):
                return SimpleNamespace(ready=True, detail=None)

        runtime.long_term_memory = _FalsePositiveRemote()

        with self.assertRaises(LongTermRemoteUnavailableError):
            runtime.check_required_remote_dependency()


if __name__ == "__main__":
    unittest.main()
