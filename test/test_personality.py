import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.personality import DEFAULT_PERSONALITY_SNAPSHOT_KIND
from twinr.agent.personality.intelligence import DEFAULT_WORLD_INTELLIGENCE_STATE_KIND
from twinr.memory.longterm.storage.remote_state import (
    LongTermRemoteUnavailableError,
    remote_snapshot_document_hints_path,
)
from twinr.memory.context_store import ManagedContextFileStore, PersistentMemoryMarkdownStore
from twinr.agent.base_agent.prompting.personality import (
    load_conversation_closure_instructions,
    load_personality_instructions,
    load_supervisor_loop_instructions,
    load_tool_loop_instructions,
    merge_instructions,
)


class _FakeRemoteState:
    def __init__(self) -> None:
        self.enabled = True
        self.config = SimpleNamespace(long_term_memory_migration_enabled=False)
        self.snapshots: dict[str, dict[str, object]] = {}

    def load_snapshot(self, *, snapshot_kind: str, local_path=None):
        del local_path
        payload = self.snapshots.get(snapshot_kind)
        return dict(payload) if isinstance(payload, dict) else None

    def save_snapshot(self, *, snapshot_kind: str, payload):
        self.snapshots[snapshot_kind] = dict(payload)


class _FailingRemoteState(_FakeRemoteState):
    def load_snapshot(self, *, snapshot_kind: str, local_path=None):
        del local_path
        raise LongTermRemoteUnavailableError(
            f"Failed to read remote long-term snapshot {snapshot_kind!r}: status=503"
        )


class _CountingRemoteState(_FakeRemoteState):
    def __init__(self) -> None:
        super().__init__()
        self.load_calls: list[str] = []

    def load_snapshot(self, *, snapshot_kind: str, local_path=None):
        self.load_calls.append(snapshot_kind)
        return super().load_snapshot(snapshot_kind=snapshot_kind, local_path=local_path)


def _section_marker(title: str, authority: str) -> str:
    return f'<section title="{title}" authority="{authority}" encoding="verbatim_text">'


class PersonalityTests(unittest.TestCase):
    def test_load_personality_instructions_orders_sections_for_stable_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            personality_dir = Path(tmpdir) / "personality"
            personality_dir.mkdir()
            state_dir = Path(tmpdir) / "state"
            state_dir.mkdir()
            (personality_dir / "SYSTEM.md").write_text("System context", encoding="utf-8")
            (personality_dir / "PERSONALITY.md").write_text("Style context", encoding="utf-8")
            (personality_dir / "USER.md").write_text("User profile", encoding="utf-8")
            (state_dir / "MEMORY.md").write_text(
                "\n".join(
                    [
                        "# Twinr Memory",
                        "",
                        "## Entries",
                        "",
                        "### MEM-20260313T120000Z",
                        "- kind: appointment",
                        "- created_at: 2026-03-13T12:00:00+00:00",
                        "- updated_at: 2026-03-13T12:00:00+00:00",
                        "- summary: Arzttermin am Montag um 14 Uhr.",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (state_dir / "reminders.json").write_text(
                (
                    '{\n'
                    '  "entries": [\n'
                    '    {\n'
                    '      "reminder_id": "REM-20260314T110000000000Z",\n'
                    '      "kind": "reminder",\n'
                    '      "summary": "Muell rausstellen",\n'
                    '      "due_at": "2026-03-14T11:00:00+01:00",\n'
                    '      "created_at": "2026-03-13T12:00:00+01:00",\n'
                    '      "updated_at": "2026-03-13T12:00:00+01:00",\n'
                    '      "source": "tool",\n'
                    '      "delivery_attempts": 0\n'
                    '    }\n'
                    '  ]\n'
                    '}\n'
                ),
                encoding="utf-8",
            )

            instructions = load_personality_instructions(
                TwinrConfig(
                    project_root=tmpdir,
                    personality_dir="personality",
                    memory_markdown_path=str(state_dir / "MEMORY.md"),
                    reminder_store_path=str(state_dir / "reminders.json"),
                )
            )

        self.assertIsNotNone(instructions)
        self.assertTrue(instructions.startswith('<assistant_context_bundle version="2">'))
        self.assertLess(
            instructions.index(_section_marker("SYSTEM", "configuration")),
            instructions.index(_section_marker("PERSONALITY", "configuration")),
        )
        self.assertLess(
            instructions.index(_section_marker("PERSONALITY", "configuration")),
            instructions.index(_section_marker("USER", "context_data")),
        )
        self.assertLess(
            instructions.index(_section_marker("USER", "context_data")),
            instructions.index(_section_marker("MEMORY", "context_data")),
        )
        self.assertLess(
            instructions.index(_section_marker("MEMORY", "context_data")),
            instructions.index(_section_marker("REMINDERS", "context_data")),
        )
        self.assertIn("System context", instructions)
        self.assertIn("Style context", instructions)
        self.assertIn("User profile", instructions)
        self.assertIn("Durable remembered items explicitly saved for future turns:", instructions)
        self.assertIn("Scheduled reminders and timers:", instructions)
        self.assertIn("Arzttermin am Montag um 14 Uhr.", instructions)
        self.assertIn("Muell rausstellen", instructions)

    def test_merge_instructions_skips_empty_parts(self) -> None:
        merged = merge_instructions("Base", None, " ", "Task")
        self.assertEqual(merged, "Base\n\nTask")

    def test_tool_loop_instructions_exclude_automation_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            personality_dir = Path(tmpdir) / "personality"
            personality_dir.mkdir()
            state_dir = Path(tmpdir) / "state"
            state_dir.mkdir()
            (personality_dir / "SYSTEM.md").write_text("System context", encoding="utf-8")
            (state_dir / "automations.json").write_text(
                (
                    '{\n'
                    '  "entries": [\n'
                    '    {\n'
                    '      "automation_id": "AUTO-1",\n'
                    '      "name": "Morgenwetter",\n'
                    '      "enabled": true,\n'
                    '      "trigger": {"kind": "time", "schedule": "daily", "time_of_day": "08:00", "weekdays": [], "timezone_name": "Europe/Berlin"},\n'
                    '      "actions": [{"kind": "say", "text": "Wetterbericht", "payload": {}, "enabled": true}],\n'
                    '      "created_at": "2026-03-13T12:00:00+01:00",\n'
                    '      "updated_at": "2026-03-13T12:00:00+01:00",\n'
                    '      "source": "tool",\n'
                    '      "tags": []\n'
                    '    }\n'
                    '  ]\n'
                    '}\n'
                ),
                encoding="utf-8",
            )

            instructions = load_tool_loop_instructions(
                TwinrConfig(
                    project_root=tmpdir,
                    personality_dir="personality",
                    automation_store_path=str(state_dir / "automations.json"),
                )
            )
            full_instructions = load_personality_instructions(
                TwinrConfig(
                    project_root=tmpdir,
                    personality_dir="personality",
                    automation_store_path=str(state_dir / "automations.json"),
                )
            )

        self.assertNotIn("AUTOMATIONS:", instructions)
        self.assertNotIn("Active automations:", instructions)
        self.assertNotIn(_section_marker("AUTOMATIONS", "context_data"), instructions)
        self.assertIn(_section_marker("AUTOMATIONS", "context_data"), full_instructions)
        self.assertIn("Active automations:", full_instructions)
        self.assertIn("Morgenwetter", full_instructions)

    def test_supervisor_loop_instructions_exclude_memory_and_reminders(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            personality_dir = Path(tmpdir) / "personality"
            personality_dir.mkdir()
            state_dir = Path(tmpdir) / "state"
            state_dir.mkdir()
            (personality_dir / "SYSTEM.md").write_text("System context", encoding="utf-8")
            (personality_dir / "PERSONALITY.md").write_text("Style context", encoding="utf-8")
            (personality_dir / "USER.md").write_text("User profile", encoding="utf-8")
            (state_dir / "MEMORY.md").write_text("# Twinr Memory\n", encoding="utf-8")
            (state_dir / "reminders.json").write_text('{"entries":[]}\n', encoding="utf-8")

            instructions = load_supervisor_loop_instructions(
                TwinrConfig(
                    project_root=tmpdir,
                    personality_dir="personality",
                    memory_markdown_path=str(state_dir / "MEMORY.md"),
                    reminder_store_path=str(state_dir / "reminders.json"),
                )
            )

        self.assertIn(_section_marker("SYSTEM", "configuration"), instructions)
        self.assertIn(_section_marker("PERSONALITY", "configuration"), instructions)
        self.assertIn(_section_marker("USER", "context_data"), instructions)
        self.assertIn("System context", instructions)
        self.assertIn("Style context", instructions)
        self.assertIn("User profile", instructions)
        self.assertNotIn(_section_marker("MEMORY", "context_data"), instructions)
        self.assertNotIn(_section_marker("REMINDERS", "context_data"), instructions)

    def test_supervisor_loop_instructions_exclude_dynamic_topic_layers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            personality_dir = Path(tmpdir) / "personality"
            personality_dir.mkdir()
            (personality_dir / "SYSTEM.md").write_text("System context", encoding="utf-8")
            (personality_dir / "PERSONALITY.md").write_text("Style context", encoding="utf-8")
            (personality_dir / "USER.md").write_text("User profile", encoding="utf-8")

            remote_state = _FakeRemoteState()
            remote_state.snapshots[DEFAULT_PERSONALITY_SNAPSHOT_KIND] = {
                "schema_version": 1,
                "core_traits": [
                    {
                        "name": "laid_back",
                        "summary": "Stay relaxed and grounded.",
                        "weight": 0.86,
                        "stable": True,
                    }
                ],
                "style_profile": {
                    "verbosity": 0.34,
                    "initiative": 0.61,
                },
                "humor_profile": {
                    "style": "dry",
                    "summary": "Quietly dry humor when it lands.",
                    "intensity": 0.58,
                    "boundaries": ["sensitive_context"],
                },
                "relationship_signals": [
                    {
                        "topic": "AI companions",
                        "summary": "Strong shared thread.",
                        "salience": 0.88,
                        "stance": "affinity",
                        "source": "conversation_turn",
                    }
                ],
                "continuity_threads": [
                    {
                        "thread_id": "thread:janina",
                        "title": "Janina",
                        "summary": "Check in about Janina.",
                        "salience": 0.78,
                        "last_updated": "2026-03-20T18:00:00+00:00",
                    }
                ],
                "place_focuses": [
                    {
                        "name": "Schwarzenbek",
                        "summary": "Treat Schwarzenbek as home base.",
                        "geography": "city",
                        "salience": 0.98,
                    }
                ],
                "world_signals": [
                    {
                        "signal_id": "world:hamburg_local_politics",
                        "topic": "Hamburg local politics",
                        "summary": "Recent local-politics developments in Hamburg.",
                        "scope": "local",
                        "salience": 0.79,
                        "confidence": 0.84,
                        "source": "live_search",
                        "updated_at": "2026-03-20T19:25:00+00:00",
                    }
                ],
            }
            remote_state.snapshots[DEFAULT_WORLD_INTELLIGENCE_STATE_KIND] = {
                "schema_version": 1,
                "interest_signals": [
                    {
                        "signal_id": "interest:hamburg_local_politics",
                        "topic": "Hamburg local politics",
                        "summary": "Repeated user follow-ups show strong durable engagement.",
                        "scope": "topic",
                        "salience": 0.86,
                        "confidence": 0.84,
                        "engagement_score": 0.95,
                        "evidence_count": 3,
                        "engagement_count": 5,
                        "updated_at": "2026-03-20T19:25:00+00:00",
                    }
                ],
            }

            config = TwinrConfig(
                project_root=tmpdir,
                personality_dir="personality",
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_required=False,
            )
            with patch(
                "twinr.agent.base_agent.prompting.personality.LongTermRemoteStateStore.from_config",
                return_value=remote_state,
            ):
                instructions = load_supervisor_loop_instructions(config)

        self.assertIn(_section_marker("SYSTEM", "configuration"), instructions)
        self.assertIn(_section_marker("PERSONALITY", "configuration"), instructions)
        self.assertIn("System context", instructions)
        self.assertIn("Style context", instructions)
        self.assertIn("## Structured core character", instructions)
        self.assertIn("## Evolving conversation style", instructions)
        self.assertIn("## Evolving humor stance", instructions)
        self.assertNotIn(_section_marker("MINDSHARE", "context_data"), instructions)
        self.assertNotIn(_section_marker("CONTINUITY", "context_data"), instructions)
        self.assertNotIn(_section_marker("PLACE", "context_data"), instructions)
        self.assertNotIn(_section_marker("WORLD", "context_data"), instructions)
        self.assertNotIn(_section_marker("REFLECTION", "context_data"), instructions)
        self.assertNotIn("Hamburg local politics", instructions)
        self.assertNotIn("Schwarzenbek", instructions)

    def test_conversation_closure_instructions_stay_dedicated_and_lean(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            personality_dir = Path(tmpdir) / "personality"
            personality_dir.mkdir()
            state_dir = Path(tmpdir) / "state"
            state_dir.mkdir()
            (personality_dir / "SYSTEM.md").write_text("System context", encoding="utf-8")
            (personality_dir / "PERSONALITY.md").write_text("Style context", encoding="utf-8")
            (personality_dir / "CONVERSATION_CLOSURE.md").write_text("Closure-only instructions", encoding="utf-8")
            (state_dir / "MEMORY.md").write_text("# Twinr Memory\n", encoding="utf-8")
            (state_dir / "reminders.json").write_text('{"entries":[]}\n', encoding="utf-8")

            instructions = load_conversation_closure_instructions(
                TwinrConfig(
                    project_root=tmpdir,
                    personality_dir="personality",
                    memory_markdown_path=str(state_dir / "MEMORY.md"),
                    reminder_store_path=str(state_dir / "reminders.json"),
                )
            )

        self.assertEqual(instructions, "Closure-only instructions")

    def test_load_personality_instructions_reads_remote_prompt_memory_and_user_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            personality_dir = Path(tmpdir) / "personality"
            personality_dir.mkdir()
            state_dir = Path(tmpdir) / "state"
            state_dir.mkdir()
            (personality_dir / "SYSTEM.md").write_text("System context", encoding="utf-8")
            (personality_dir / "PERSONALITY.md").write_text("Style context", encoding="utf-8")
            (personality_dir / "USER.md").write_text("Base user profile", encoding="utf-8")
            remote_state = _FakeRemoteState()
            PersistentMemoryMarkdownStore(
                state_dir / "MEMORY.md",
                remote_state=remote_state,
            ).remember(
                kind="appointment",
                summary="Eye doctor on Tuesday at 10:30.",
            )
            ManagedContextFileStore(
                personality_dir / "USER.md",
                section_title="Twinr managed user updates",
                remote_state=remote_state,
                remote_snapshot_kind="user_context",
            ).upsert(
                category="pets",
                instruction="Thom has two dogs.",
            )

            config = TwinrConfig(
                project_root=tmpdir,
                personality_dir="personality",
                memory_markdown_path=str(state_dir / "MEMORY.md"),
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_required=False,
            )
            with patch("twinr.memory.longterm.storage.remote_state.LongTermRemoteStateStore.from_config", return_value=remote_state), patch(
                "twinr.agent.base_agent.prompting.personality.LongTermRemoteStateStore.from_config",
                return_value=remote_state,
            ):
                instructions = load_personality_instructions(config)

        self.assertIsNotNone(instructions)
        self.assertIn("Eye doctor on Tuesday at 10:30.", instructions)
        self.assertIn("Base user profile", instructions)
        self.assertIn("pets: Thom has two dogs.", instructions)

    def test_load_personality_instructions_includes_structured_mindshare_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            personality_dir = Path(tmpdir) / "personality"
            personality_dir.mkdir()
            (personality_dir / "SYSTEM.md").write_text("System context", encoding="utf-8")
            (personality_dir / "PERSONALITY.md").write_text("Style context", encoding="utf-8")
            (personality_dir / "USER.md").write_text("Base user profile", encoding="utf-8")

            remote_state = _FakeRemoteState()
            remote_state.snapshots[DEFAULT_PERSONALITY_SNAPSHOT_KIND] = {
                "schema_version": 1,
                "core_traits": [
                    {
                        "name": "attentive companion",
                        "summary": "Stay warm and aware of the user's real world.",
                        "weight": 0.9,
                    }
                ],
                "style_profile": {
                    "verbosity": 0.48,
                    "initiative": 0.58,
                },
                "humor_profile": {
                    "style": "dry observational humor",
                    "summary": "Use occasional understated wit.",
                    "intensity": 0.32,
                },
                "relationship_signals": [
                    {
                        "topic": "AI companions",
                        "summary": "This remains part of the user's durable interest.",
                        "salience": 0.91,
                        "source": "installer_seed",
                        "stance": "affinity",
                    }
                ],
                "continuity_threads": [
                    {
                        "title": "Hamburg local politics",
                        "summary": "Twinr has been keeping an eye on local civic changes that affect daily life.",
                        "salience": 0.79,
                        "updated_at": "2026-03-20T19:18:07+00:00",
                    }
                ],
                "place_focuses": [
                    {
                        "name": "Schwarzenbek",
                        "summary": "Treat Schwarzenbek as the user's home anchor for local context.",
                        "geography": "city",
                        "salience": 0.98,
                    },
                    {
                        "name": "Hamburg",
                        "summary": "Treat Hamburg as the main nearby urban context.",
                        "geography": "city",
                        "salience": 0.93,
                    },
                ],
            }
            remote_state.snapshots[DEFAULT_WORLD_INTELLIGENCE_STATE_KIND] = {
                "schema_version": 1,
                "interest_signals": [
                    {
                        "signal_id": "interest:ai_companions",
                        "topic": "AI companions",
                        "summary": "Repeated user follow-ups show strong durable engagement.",
                        "scope": "topic",
                        "salience": 0.86,
                        "confidence": 0.84,
                        "engagement_score": 0.95,
                        "evidence_count": 3,
                        "engagement_count": 5,
                        "updated_at": "2026-03-20T19:25:00+00:00",
                    }
                ],
            }

            config = TwinrConfig(
                project_root=tmpdir,
                personality_dir="personality",
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_required=False,
            )
            with patch(
                "twinr.agent.base_agent.prompting.personality.LongTermRemoteStateStore.from_config",
                return_value=remote_state,
            ):
                instructions = load_personality_instructions(config)

        self.assertIsNotNone(instructions)
        self.assertIn("Conversational self-expression", instructions)
        self.assertIn("Current conversation steering", instructions)
        self.assertIn("Positive engagement policy", instructions)
        self.assertIn(
            "this has genuinely caught Twinr's ongoing attention",
            instructions,
        )
        self.assertIn("Use each surfaced topic's appetite cue", instructions)
        self.assertIn("Shared-thread topics may guide an open conversation", instructions)
        self.assertIn("Use these bounded actions to encourage welcomed conversation growth", instructions)
        self.assertIn(_section_marker("MINDSHARE", "context_data"), instructions)
        self.assertIn("Current companion mindshare", instructions)
        self.assertIn("Schwarzenbek / Hamburg", instructions)
        self.assertIn("Hamburg local politics", instructions)
        self.assertIn("appetite_state=resonant", instructions)
        self.assertIn(
            'interest="this has genuinely caught Twinr\'s ongoing attention"',
            instructions,
        )
        self.assertIn("one short concrete update is okay", instructions)
        self.assertLess(instructions.index("- AI companions:"), instructions.index("- Schwarzenbek / Hamburg:"))

    def test_load_personality_instructions_fails_closed_when_remote_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            personality_dir = Path(tmpdir) / "personality"
            personality_dir.mkdir()
            state_dir = Path(tmpdir) / "state"
            state_dir.mkdir()
            (personality_dir / "SYSTEM.md").write_text("System context", encoding="utf-8")
            (personality_dir / "PERSONALITY.md").write_text(
                "\n".join(
                    [
                        "Style context",
                        "",
                        "<!-- TWINR_MANAGED_CONTEXT_START -->",
                        "## Twinr managed personality updates",
                        "_This section is managed by Twinr. Keep entries short and stable._",
                        "- response_style: Keep answers calm.",
                        "<!-- TWINR_MANAGED_CONTEXT_END -->",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (personality_dir / "USER.md").write_text(
                "\n".join(
                    [
                        "Base user profile",
                        "",
                        "<!-- TWINR_MANAGED_CONTEXT_START -->",
                        "## Twinr managed user updates",
                        "_This section is managed by Twinr. Keep entries short and stable._",
                        "- pets: Thom has two dogs.",
                        "<!-- TWINR_MANAGED_CONTEXT_END -->",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            config = TwinrConfig(
                project_root=tmpdir,
                personality_dir="personality",
                memory_markdown_path=str(state_dir / "MEMORY.md"),
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_required=True,
            )
            failing_remote = _FailingRemoteState()
            with patch("twinr.agent.base_agent.prompting.personality.LongTermRemoteStateStore.from_config", return_value=failing_remote):
                instructions = load_personality_instructions(config)

        self.assertIsNotNone(instructions)
        self.assertTrue(instructions.startswith('<assistant_context_bundle version="2">'))
        self.assertIn(_section_marker("SYSTEM", "configuration"), instructions)
        self.assertIn("System context", instructions)
        self.assertNotIn(_section_marker("PERSONALITY", "configuration"), instructions)
        self.assertNotIn(_section_marker("USER", "context_data"), instructions)
        self.assertNotIn("Style context", instructions)
        self.assertNotIn("Base user profile", instructions)
        self.assertNotIn("Thom has two dogs.", instructions)
        self.assertNotIn(_section_marker("REMINDERS", "context_data"), instructions)
        self.assertNotIn(_section_marker("AUTOMATIONS", "context_data"), instructions)

    def test_tool_loop_instructions_cache_skips_repeated_remote_reads_when_sources_stay_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            personality_dir = Path(tmpdir) / "personality"
            personality_dir.mkdir()
            state_dir = Path(tmpdir) / "state"
            state_dir.mkdir()
            (personality_dir / "SYSTEM.md").write_text("System context", encoding="utf-8")
            (personality_dir / "PERSONALITY.md").write_text("Base personality", encoding="utf-8")
            (personality_dir / "USER.md").write_text("Base user", encoding="utf-8")
            (state_dir / "MEMORY.md").write_text("# Twinr Memory\n", encoding="utf-8")
            (state_dir / "reminders.json").write_text('{"entries":[]}\n', encoding="utf-8")

            remote_state = _CountingRemoteState()
            ManagedContextFileStore(
                personality_dir / "USER.md",
                section_title="Twinr managed user updates",
                remote_state=remote_state,
                remote_snapshot_kind="user_context",
            ).upsert(category="pets", instruction="Thom has two dogs.")
            ManagedContextFileStore(
                personality_dir / "PERSONALITY.md",
                section_title="Twinr managed personality updates",
                remote_state=remote_state,
                remote_snapshot_kind="personality_context",
            ).upsert(category="style", instruction="Keep answers calm.")

            config = TwinrConfig(
                project_root=tmpdir,
                personality_dir="personality",
                memory_markdown_path=str(state_dir / "MEMORY.md"),
                reminder_store_path=str(state_dir / "reminders.json"),
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_required=False,
            )
            with patch(
                "twinr.memory.longterm.storage.remote_state.LongTermRemoteStateStore.from_config",
                return_value=remote_state,
            ), patch(
                "twinr.agent.base_agent.prompting.personality.LongTermRemoteStateStore.from_config",
                return_value=remote_state,
            ):
                first = load_tool_loop_instructions(config)
                first_remote_calls = len(remote_state.load_calls)
                second = load_tool_loop_instructions(config)

        self.assertIsNotNone(first)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first_remote_calls, 2)
        self.assertEqual(len(remote_state.load_calls), first_remote_calls)

    def test_tool_loop_instructions_cache_invalidates_when_remote_hint_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            personality_dir = Path(tmpdir) / "personality"
            personality_dir.mkdir()
            state_dir = Path(tmpdir) / "state"
            state_dir.mkdir()
            remote_dir = state_dir / "chonkydb"
            remote_dir.mkdir()
            (personality_dir / "SYSTEM.md").write_text("System context", encoding="utf-8")
            (personality_dir / "PERSONALITY.md").write_text("Base personality", encoding="utf-8")
            (personality_dir / "USER.md").write_text("Base user", encoding="utf-8")
            (state_dir / "MEMORY.md").write_text("# Twinr Memory\n", encoding="utf-8")
            (state_dir / "reminders.json").write_text('{"entries":[]}\n', encoding="utf-8")

            remote_state = _CountingRemoteState()
            ManagedContextFileStore(
                personality_dir / "USER.md",
                section_title="Twinr managed user updates",
                remote_state=remote_state,
                remote_snapshot_kind="user_context",
            ).upsert(category="pets", instruction="Thom has two dogs.")

            config = TwinrConfig(
                project_root=tmpdir,
                personality_dir="personality",
                memory_markdown_path=str(state_dir / "MEMORY.md"),
                reminder_store_path=str(state_dir / "reminders.json"),
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_required=False,
                long_term_memory_path=str(remote_dir),
            )
            with patch(
                "twinr.memory.longterm.storage.remote_state.LongTermRemoteStateStore.from_config",
                return_value=remote_state,
            ), patch(
                "twinr.agent.base_agent.prompting.personality.LongTermRemoteStateStore.from_config",
                return_value=remote_state,
            ):
                first = load_tool_loop_instructions(config)
                first_remote_calls = len(remote_state.load_calls)
                remote_state.snapshots["user_context"] = {
                    "schema": "twinr_managed_context",
                    "version": 1,
                    "entries": [
                        {
                            "key": "pets",
                            "instruction": "Thom now also helps his daughter with her cat.",
                            "updated_at": "2026-03-21T20:15:00+00:00",
                        }
                    ],
                }
                hints_path = remote_snapshot_document_hints_path(config)
                assert hints_path is not None
                hints_path.parent.mkdir(parents=True, exist_ok=True)
                hints_path.write_text('{"schema":"twinr_remote_snapshot_document_hints_v1"}', encoding="utf-8")
                hint_stat = hints_path.stat()
                os.utime(
                    hints_path,
                    ns=(hint_stat.st_atime_ns, hint_stat.st_mtime_ns + 1_000_000),
                )
                second = load_tool_loop_instructions(config)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertNotEqual(first, second)
        self.assertIn("Thom now also helps his daughter with her cat.", second)
        self.assertGreater(len(remote_state.load_calls), first_remote_calls)


if __name__ == "__main__":
    unittest.main()
