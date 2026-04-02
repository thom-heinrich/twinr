"""Regression tests for the remote prompt current-head repair helper."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from twinr.ops.remote_prompt_current_head_repair import repair_prompt_current_heads_from_store


class _FakeRemoteRecords:
    def __init__(self, *, statuses: list[str]) -> None:
        self._statuses = list(statuses)
        self.probe_calls = 0
        self.save_calls: list[str] = []

    def probe_current_head_result(self, *, snapshot_kind: str) -> tuple[str, dict[str, object] | None]:
        self.probe_calls += 1
        status = self._statuses.pop(0)
        payload = {"schema": f"{snapshot_kind}_catalog"} if status == "found" else None
        return status, payload

    def save_empty_collection_head(
        self,
        *,
        snapshot_kind: str,
        attest_readback: bool = True,
    ) -> dict[str, object]:
        del attest_readback
        self.save_calls.append(snapshot_kind)
        return {"schema": f"{snapshot_kind}_catalog", "version": 3, "items_count": 0, "segments": []}


class RemotePromptCurrentHeadRepairTests(unittest.TestCase):
    """Cover the bounded operator repair logic without touching a live backend."""

    def test_force_publish_skips_probe_and_marks_success_without_verify(self) -> None:
        user_records = _FakeRemoteRecords(statuses=["found"])
        prompt_store = SimpleNamespace(
            memory_store=SimpleNamespace(_remote_records=_FakeRemoteRecords(statuses=["found"])),
            user_store=SimpleNamespace(_remote_records=user_records),
            personality_store=SimpleNamespace(_remote_records=_FakeRemoteRecords(statuses=["found"])),
        )

        items = repair_prompt_current_heads_from_store(
            prompt_store=prompt_store,
            snapshot_kinds=("user_context",),
            force=True,
            verify=False,
        )

        self.assertEqual(len(items), 1)
        self.assertTrue(items[0].ok)
        self.assertEqual(items[0].action, "force_publish_empty")
        self.assertEqual(items[0].before_status, "skipped_force")
        self.assertEqual(items[0].after_status, "not_checked")
        self.assertEqual(user_records.probe_calls, 0)
        self.assertEqual(user_records.save_calls, ["user_context"])

    def test_invalid_head_is_repaired_and_verified(self) -> None:
        personality_records = _FakeRemoteRecords(statuses=["invalid", "found"])
        prompt_store = SimpleNamespace(
            memory_store=SimpleNamespace(_remote_records=_FakeRemoteRecords(statuses=["found"])),
            user_store=SimpleNamespace(_remote_records=_FakeRemoteRecords(statuses=["found"])),
            personality_store=SimpleNamespace(_remote_records=personality_records),
        )

        items = repair_prompt_current_heads_from_store(
            prompt_store=prompt_store,
            snapshot_kinds=("personality_context",),
            verify=True,
        )

        self.assertEqual(len(items), 1)
        self.assertTrue(items[0].ok)
        self.assertEqual(items[0].action, "repair_invalid")
        self.assertEqual(items[0].before_status, "invalid")
        self.assertEqual(items[0].after_status, "found")
        self.assertEqual(personality_records.probe_calls, 2)
        self.assertEqual(personality_records.save_calls, ["personality_context"])

    def test_missing_head_stays_red_without_explicit_repair_flag(self) -> None:
        memory_records = _FakeRemoteRecords(statuses=["missing", "missing"])
        prompt_store = SimpleNamespace(
            memory_store=SimpleNamespace(_remote_records=memory_records),
            user_store=SimpleNamespace(_remote_records=_FakeRemoteRecords(statuses=["found"])),
            personality_store=SimpleNamespace(_remote_records=_FakeRemoteRecords(statuses=["found"])),
        )

        items = repair_prompt_current_heads_from_store(
            prompt_store=prompt_store,
            snapshot_kinds=("prompt_memory",),
            verify=True,
        )

        self.assertEqual(len(items), 1)
        self.assertFalse(items[0].ok)
        self.assertEqual(items[0].action, "missing_unrepaired")
        self.assertEqual(items[0].before_status, "missing")
        self.assertEqual(items[0].after_status, "missing")
        self.assertEqual(memory_records.probe_calls, 2)
        self.assertEqual(memory_records.save_calls, [])
