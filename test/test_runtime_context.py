from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import stat
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.runtime.display_grounding import (
    build_active_display_grounding_instruction_overlay,
)
from twinr.agent.base_agent.runtime import TwinrRuntime
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError
from twinr.agent.base_agent.state.snapshot import RuntimeSnapshotStore
from twinr.display.ambient_impulse_cues import DisplayAmbientImpulseController


class RuntimeContextTests(unittest.TestCase):
    def _config(self, temp_dir: str) -> TwinrConfig:
        root = Path(temp_dir)
        return TwinrConfig(
            project_root=temp_dir,
            long_term_memory_path=str(root / "state" / "chonkydb"),
            runtime_state_path=str(root / "state" / "runtime-state.json"),
        )

    def _write_active_display_cue(self, config: TwinrConfig) -> None:
        controller = DisplayAmbientImpulseController.from_config(config)
        controller.show_impulse(
            topic_key="hamburg local politics",
            eyebrow="",
            headline="Hamburger Lokalpolitik zieht wieder an",
            body="Ich halte kurz die Augen auf dem Rathaus.",
            hold_seconds=300.0,
            source="test",
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

    def test_identified_voice_guidance_allows_low_risk_discovery_without_identity_recheck(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = TwinrRuntime(config=self._config(temp_dir))
            try:
                runtime.user_voice_status = "likely_user"
                runtime.user_voice_confidence = 0.91
                runtime.user_voice_checked_at = datetime.now(timezone.utc)

                context = runtime.provider_conversation_context()

                system_messages = [content for role, content in context if role == "system"]
                self.assertTrue(
                    any(
                        "you do not need to ask who the person is again" in message.lower()
                        and "low-risk guided user-discovery" in message.lower()
                        for message in system_messages
                    )
                )
            finally:
                runtime.shutdown(timeout_s=1.0)

    def test_legacy_identified_voice_status_maps_to_likely_user_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = TwinrRuntime(config=self._config(temp_dir))
            try:
                runtime.user_voice_status = "identified"
                runtime.user_voice_confidence = 0.91
                runtime.user_voice_checked_at = datetime.now(timezone.utc)

                context = runtime.provider_conversation_context()

                system_messages = [content for role, content in context if role == "system"]
                self.assertTrue(
                    any(
                        "low-risk guided user-discovery" in message.lower()
                        and "you do not need to ask who the person is again" in message.lower()
                        for message in system_messages
                    )
                )
                self.assertFalse(
                    any(
                        "for persistent or security-sensitive changes, first ask for explicit confirmation"
                        in message.lower()
                        for message in system_messages
                    )
                )
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

    def test_runtime_snapshot_store_writes_world_readable_snapshot_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)
            store = RuntimeSnapshotStore(config.runtime_state_path)

            store.save(
                status="waiting",
                memory_turns=(),
                last_transcript=None,
                last_response="Bereit.",
            )

            primary_mode = stat.S_IMODE(Path(config.runtime_state_path).stat().st_mode)
            backup_mode = stat.S_IMODE(Path(f"{config.runtime_state_path}.bak").stat().st_mode)

        self.assertEqual(primary_mode, 0o644)
        self.assertEqual(backup_mode, 0o644)

    def test_snapshot_restore_preserves_household_voice_identity_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)
            snapshot = RuntimeSnapshotStore(config.runtime_state_path).save(
                status="waiting",
                memory_turns=(),
                last_transcript=None,
                last_response="Das klingt nach dir.",
                user_voice_status="known_other_user",
                user_voice_confidence=0.88,
                user_voice_checked_at="2026-03-15T19:01:00+00:00",
                user_voice_user_id="eva",
                user_voice_user_display_name="Eva",
                user_voice_match_source="household_voice_identity",
            )

            self.assertEqual(snapshot.user_voice_user_id, "eva")
            self.assertEqual(snapshot.user_voice_user_display_name, "Eva")
            self.assertEqual(snapshot.user_voice_match_source, "household_voice_identity")

            runtime = TwinrRuntime(
                config=TwinrConfig(
                    project_root=temp_dir,
                    long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                    runtime_state_path=str(Path(temp_dir) / "state" / "runtime-state.json"),
                    restore_runtime_state_on_startup=True,
                )
            )
            try:
                self.assertEqual(runtime.user_voice_status, "known_other_user")
                self.assertEqual(runtime.user_voice_user_id, "eva")
                self.assertEqual(runtime.user_voice_user_display_name, "Eva")
                self.assertEqual(runtime.user_voice_match_source, "household_voice_identity")
                self.assertEqual(runtime.user_voice_checked_at, "2026-03-15T19:01:00Z")
            finally:
                runtime.shutdown(timeout_s=1.0)

    def test_snapshot_restore_preserves_temporary_voice_quiet_window(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)
            quiet_until = (datetime.now(timezone.utc) + timedelta(minutes=25)).replace(microsecond=0)
            RuntimeSnapshotStore(config.runtime_state_path).save(
                status="waiting",
                memory_turns=(),
                last_transcript=None,
                last_response="Ich bleibe kurz ruhig.",
                voice_quiet_until_utc=quiet_until.isoformat(),
                voice_quiet_reason="tv news",
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
                state = runtime.voice_quiet_state()
                self.assertTrue(state.active)
                self.assertEqual(state.reason, "tv news")
                self.assertEqual(state.until_utc, quiet_until.isoformat().replace("+00:00", "Z"))
                self.assertGreater(state.remaining_seconds, 0)
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

    def test_search_provider_context_does_not_use_fast_topic_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = TwinrRuntime(
                config=TwinrConfig(
                    project_root=temp_dir,
                    long_term_memory_enabled=True,
                    long_term_memory_fast_topic_enabled=True,
                    long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                    runtime_state_path=str(Path(temp_dir) / "state" / "runtime-state.json"),
                )
            )
            try:
                runtime.memory.remember("user", "Erster Turn")
                runtime.memory.remember("assistant", "Zweiter Turn")
                runtime.last_transcript = "Was weisst du ueber Janina?"

                class _FastLongTermMemory:
                    def build_fast_provider_context(self, query_text):
                        raise AssertionError("search context must not query fast topic memory")

                runtime.long_term_memory = _FastLongTermMemory()

                context = runtime.search_provider_conversation_context()

                self.assertNotIn(("system", "Fast topic hint."), context)
                self.assertIn(("user", "Erster Turn"), context)
                self.assertIn(("assistant", "Zweiter Turn"), context)
            finally:
                runtime.shutdown(timeout_s=1.0)

    def test_search_provider_context_includes_active_display_grounding(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)
            self._write_active_display_cue(config)
            runtime = TwinrRuntime(config=config)
            try:
                context = runtime.search_provider_conversation_context()
                system_messages = [content for role, content in context if role == "system"]
                self.assertTrue(
                    any(
                        "AUF DEINEM SCREEN STEHT GERADE" in message
                        and "Sichtbarer Themenanker: hamburg local politics." in message
                        and "Sichtbare Überschrift: Hamburger Lokalpolitik zieht wieder an." in message
                        and "formuliere goal und prompt mit dem sichtbaren Thema" in message
                        for message in system_messages
                    )
                )
            finally:
                runtime.shutdown(timeout_s=1.0)

    def test_supervisor_context_includes_active_display_grounding(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)
            self._write_active_display_cue(config)
            runtime = TwinrRuntime(config=config)
            try:
                context = runtime.supervisor_provider_conversation_context()
                system_messages = [content for role, content in context if role == "system"]
                self.assertTrue(
                    any(
                        "Sichtbarer Themenanker: hamburg local politics." in message
                        and "behandle diesen Screen-Inhalt als primären Deutungsanker" in message
                        for message in system_messages
                    )
                )
                direct_context = runtime.supervisor_direct_provider_conversation_context(
                    "Was ist denn heute so in der Hamburger Lokalpolitik?"
                )
                direct_system_messages = [content for role, content in direct_context if role == "system"]
                self.assertTrue(
                    any("Sichtbarer Themenanker: hamburg local politics." in message for message in direct_system_messages)
                )
            finally:
                runtime.shutdown(timeout_s=1.0)

    def test_display_grounding_overlay_marks_visible_card_as_authoritative(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)
            self._write_active_display_cue(config)

            overlay = build_active_display_grounding_instruction_overlay(config)

            self.assertIsNotNone(overlay)
            self.assertIn("AUF DEINEM SCREEN STEHT GERADE", overlay)
            self.assertIn("Sichtbarer Themenanker: hamburg local politics.", overlay)
            self.assertIn("autoritativer situativer Kontext", overlay)
            self.assertIn("nicht bloß wegen einer kurzen Rückfrage zur sichtbaren Karte", overlay)

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

    def test_tool_provider_context_includes_active_user_discovery_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = TwinrRuntime(config=self._config(temp_dir))
            try:
                runtime.manage_user_discovery(action="start_or_resume", topic_id="basics")

                context = runtime.tool_provider_conversation_context()

                system_messages = [content for role, content in context if role == "system"]
                self.assertTrue(
                    any(
                        "Guided user-discovery state for this turn." in message
                        and "Session state: active." in message
                        and "Current discovery topic: Basisinfos." in message
                        and "next discovery answer" in message
                        and "review_profile plus replace_fact or delete_fact in the same turn" in message
                        for message in system_messages
                    )
                )
            finally:
                runtime.shutdown(timeout_s=1.0)

    def test_tool_provider_context_includes_idle_user_discovery_availability_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = TwinrRuntime(config=self._config(temp_dir))
            try:
                context = runtime.tool_provider_conversation_context()

                system_messages = [content for role, content in context if role == "system"]
                self.assertTrue(
                    any(
                        "Guided user-discovery is available for this turn." in message
                        and "start_or_resume or answer directly" in message
                        and "Do not ask a separate save-permission question" in message
                        for message in system_messages
                    )
                )
            finally:
                runtime.shutdown(timeout_s=1.0)

    def test_tiny_recent_tool_context_skips_remote_lookup_and_keeps_bounded_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = TwinrRuntime(config=self._config(temp_dir))
            try:
                runtime.memory.remember("user", "Bitte sei kurz ruhig.")
                runtime.memory.remember_note(
                    kind="fact",
                    content="Twinr memory summary: Die Nutzerin wollte vorhin Ruhe beim Fernsehen.",
                    source="memory",
                )
                runtime.manage_user_discovery(action="start_or_resume", topic_id="basics")

                class _FailingLongTermMemory:
                    def build_tool_provider_context(self, query_text):
                        raise AssertionError("tiny recent tool context must not query remote long-term memory")

                runtime.long_term_memory = _FailingLongTermMemory()

                context = runtime.tool_provider_tiny_recent_conversation_context()

                system_messages = [content for role, content in context if role == "system"]
                self.assertTrue(
                    any("Guided user-discovery state for this turn." in message for message in system_messages)
                )
                self.assertTrue(
                    any("Twinr memory summary:" in message for message in system_messages)
                )
                self.assertIn(("user", "Bitte sei kurz ruhig."), context)
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

    def test_supervisor_direct_context_uses_fast_topic_memory_for_current_query(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = TwinrRuntime(
                config=TwinrConfig(
                    project_root=temp_dir,
                    long_term_memory_enabled=True,
                    long_term_memory_fast_topic_enabled=True,
                    long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                    runtime_state_path=str(Path(temp_dir) / "state" / "runtime-state.json"),
                )
            )
            try:
                runtime.memory.remember("user", "Wir haben über Medikamente gesprochen.")
                recorded_queries: list[str] = []

                class _RecordingLongTermMemory:
                    def build_fast_provider_context(self, query_text):
                        recorded_queries.append(str(query_text))
                        return SimpleNamespace(
                            system_messages=lambda: ("Fast topic memory for direct supervisor reply.",)
                        )

                runtime.long_term_memory = _RecordingLongTermMemory()

                context = runtime.supervisor_direct_provider_conversation_context(
                    "Worüber haben wir heute gesprochen?"
                )

                self.assertEqual(recorded_queries, ["Worüber haben wir heute gesprochen?"])
                self.assertIn(
                    ("system", "Fast topic memory for direct supervisor reply."),
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

    def test_check_required_remote_dependency_uses_watchdog_artifact_when_configured(self) -> None:
        runtime = TwinrRuntime.__new__(TwinrRuntime)
        runtime.config = TwinrConfig(
            long_term_memory_enabled=True,
            long_term_memory_mode="remote_primary",
            long_term_memory_remote_required=True,
            long_term_memory_remote_runtime_check_mode="watchdog_artifact",
        )

        class _GuardedLongTermMemory:
            def remote_required(self):
                return True

            def ensure_remote_ready(self):
                raise AssertionError("deep remote check must not run in watchdog_artifact mode")

        runtime.long_term_memory = _GuardedLongTermMemory()

        with patch(
            "twinr.agent.workflows.required_remote_snapshot.ensure_required_remote_watchdog_snapshot_ready"
        ) as guard_mock:
            runtime.check_required_remote_dependency()

        guard_mock.assert_called_once_with(runtime.config)


if __name__ == "__main__":
    unittest.main()
