"""Regression coverage for the live Pi tool-matrix harness."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast
import tempfile

import test.run_pi_tool_matrix as matrix
from twinr.agent.workflows.streaming_runner import TwinrStreamingHardwareLoop


def test_assert_tools_allows_specialist_handoff_wrapper() -> None:
    artifact = matrix.TurnArtifact(
        prompt="prompt",
        answer="answer",
        tool_calls=["handoff_specialist_worker", "lookup_contact"],
        raw_tool_calls=[],
        emitted=[],
        status="waiting",
        keep_listening=True,
    )

    ok, detail = matrix._assert_tools(artifact, required=("lookup_contact",))

    assert ok is True
    assert "lookup_contact" in detail


def test_assert_tools_allows_sensor_stream_companion_entity_listing() -> None:
    artifact = matrix.TurnArtifact(
        prompt="prompt",
        answer="answer",
        tool_calls=[
            "handoff_specialist_worker",
            "read_smart_home_sensor_stream",
            "list_smart_home_entities",
        ],
        raw_tool_calls=[],
        emitted=[],
        status="waiting",
        keep_listening=True,
    )

    ok, detail = matrix._assert_tools(
        artifact,
        required=("read_smart_home_sensor_stream",),
        allowed=("list_smart_home_entities",),
    )

    assert ok is True
    assert "read_smart_home_sensor_stream" in detail


def test_load_static_camera_image_prefers_repo_asset_over_tiny_fallback() -> None:
    image_bytes = matrix._load_static_camera_image()

    assert isinstance(image_bytes, bytes)
    assert len(image_bytes) > len(matrix._FALLBACK_TINY_PNG)


def test_tool_matrix_context_uses_unique_remote_memory_namespace() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        (root / "personality").mkdir()
        env_path = root / ".env"
        env_path.write_text("OPENAI_API_KEY=test\n", encoding="utf-8")

        context = matrix.ToolMatrixContext(base_env_path=env_path)
        try:
            env_text = context.env_path.read_text(encoding="utf-8")
        finally:
            context.close()

    assert "TWINR_LONG_TERM_MEMORY_REMOTE_NAMESPACE=pi_tool_matrix_" in env_text


def test_tool_matrix_context_make_loop_authorizes_sensitive_tools_by_default(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        (root / "personality").mkdir()
        env_path = root / ".env"
        env_path.write_text("OPENAI_API_KEY=test\n", encoding="utf-8")

        authorizations: list[str] = []

        class FakeRuntime:
            def __init__(self, *, config) -> None:
                self.config = config

            def update_user_voice_assessment(self, **kwargs) -> None:
                del kwargs

        class FakeLoop:
            def __init__(self, **kwargs) -> None:
                self.config = kwargs["config"]
                self.runtime = kwargs["runtime"]
                self._runtime_tool_names = ()
                self._tool_handlers = {}
                self.realtime_session = SimpleNamespace(_tool_handlers={})

            def authorize_realtime_sensitive_tools(self, reason: str = "explicit") -> tuple[str, ...]:
                authorizations.append(reason)
                return ()

        monkeypatch.setattr(matrix, "TwinrRuntime", FakeRuntime)
        monkeypatch.setattr(matrix, "TwinrStreamingHardwareLoop", FakeLoop)

        context = matrix.ToolMatrixContext(base_env_path=env_path)
        try:
            loop = context.make_loop(emitted=[])
        finally:
            context.close()

    assert authorizations == ["pi_tool_matrix"]
    assert isinstance(loop, FakeLoop)


def test_seed_conflict_reuses_existing_loop_without_booting_a_second_runtime() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        (root / "personality").mkdir()
        env_path = root / ".env"
        env_path.write_text("OPENAI_API_KEY=test\n", encoding="utf-8")

        context = matrix.ToolMatrixContext(base_env_path=env_path)
        applied: list[object] = []
        fake_loop = SimpleNamespace(
            runtime=SimpleNamespace(
                long_term_memory=SimpleNamespace(
                    object_store=SimpleNamespace(
                        write_snapshot=lambda *, objects, conflicts=(), archived_objects=(): applied.append(
                            SimpleNamespace(
                                objects=objects,
                                conflicts=conflicts,
                                archived_objects=archived_objects,
                            )
                        )
                    )
                )
            )
        )
        setattr(
            context,
            "make_loop",
            lambda *, emitted: (_ for _ in ()).throw(AssertionError("make_loop should not be called")),
        )
        try:
            returned_loop = context.seed_conflict(
                loop=cast(TwinrStreamingHardwareLoop, fake_loop),
            )
        finally:
            context.close()

    assert returned_loop is fake_loop
    assert len(applied) == 1
    result = cast(SimpleNamespace, applied[0])
    assert result.conflicts[0].candidate_memory_id == "fact:corinna_phone_new"
    assert {item.memory_id for item in result.objects} == {"fact:corinna_phone_old", "fact:corinna_phone_new"}


def test_sensor_automation_matches_requires_no_motion_print_contract() -> None:
    record = {
        "name": "Besuchergruss",
        "sensor_trigger_kind": "pir_no_motion",
        "sensor_hold_seconds": 45.0,
        "actions": [{"kind": "print", "text": "Hallo! Schön, dass Sie da sind."}],
    }

    assert matrix._sensor_automation_matches(
        record,
        trigger_kind="pir_no_motion",
        hold_seconds=45.0,
        delivery="printed",
    )
    assert not matrix._sensor_automation_matches(
        record,
        trigger_kind="vad_quiet",
        hold_seconds=45.0,
        delivery="printed",
    )


def test_managed_context_contains_matches_expected_keys_and_fragments() -> None:
    entries = (
        SimpleNamespace(key="response_style", instruction="Future replies should be very short and calm."),
        SimpleNamespace(key="verbosity", instruction="Keep future replies very brief."),
    )

    assert matrix._managed_context_contains(
        entries,
        expected_fragments_by_key={
            "response_style": ("short and calm",),
            "verbosity": ("brief",),
        },
    )
    assert not matrix._managed_context_contains(
        entries,
        expected_fragments_by_key={
            "response_style": ("playful",),
        },
    )


def test_user_discovery_review_items_contain_matches_object_and_mapping_summaries() -> None:
    object_items = (
        SimpleNamespace(summary="User prefers black coffee in the morning."),
        SimpleNamespace(summary="User prefers a quiet start to the day."),
    )
    mapping_items = (
        {"summary": "User prefers black coffee in the morning."},
        {"summary": "User prefers a quiet start to the day."},
    )

    assert matrix._user_discovery_review_items_contain(
        object_items,
        ("coffee", "quiet", "morning"),
    )
    assert matrix._user_discovery_review_items_contain(
        mapping_items,
        ("coffee", "quiet", "morning"),
    )
    assert not matrix._user_discovery_review_items_contain(
        object_items,
        ("tea",),
    )


def test_voice_profile_status_answer_matches_enrolled_and_missing_states() -> None:
    assert matrix._voice_profile_status_answer_matches(
        "Ja. Es ist ein lokales Stimmprofil von Ihnen gespeichert.",
        enrolled=True,
    )
    assert matrix._voice_profile_status_answer_matches(
        "Nein, aktuell ist kein lokales Stimmprofil gespeichert.",
        enrolled=False,
    )


def test_safe_load_self_coding_activation_returns_detail_for_missing_record() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir) / "state" / "self_coding"
        store = SimpleNamespace(
            load_activation=lambda skill_id, *, version: (_ for _ in ()).throw(
                FileNotFoundError(f"{skill_id}@{version} missing under {root}")
            )
        )

        activation, detail = matrix._safe_load_self_coding_activation(
            store,
            skill_id="announce_family_updates",
            version=1,
        )

    assert activation is None
    assert "announce_family_updates@v1" in detail
    assert "missing" in detail
