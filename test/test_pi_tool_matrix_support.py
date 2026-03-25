"""Regression coverage for deterministic Pi tool-matrix support backends."""

from __future__ import annotations

from test.pi_tool_matrix_support import (
    MatrixHouseholdIdentityManager,
    MatrixPortraitProvider,
    MatrixSelfCodingCompileDriver,
    MatrixSmartHomeProvider,
    MatrixWorldRemoteState,
)
from twinr.agent.self_coding.status import ArtifactKind
from twinr.integrations.smarthome.models import SmartHomeCommand, SmartHomeEntityClass


def test_matrix_portrait_provider_enrolls_observes_and_resets() -> None:
    provider = MatrixPortraitProvider()

    enrolled = provider.capture_and_enroll_reference(
        user_id="main_user",
        display_name="Thom",
        source="matrix",
    )
    summary = provider.summary(user_id="main_user")
    observation = provider.observe()
    cleared = provider.clear_identity_profile(user_id="main_user")

    assert enrolled.status == "enrolled"
    assert summary.enrolled is True
    assert observation.state == "likely_reference_user"
    assert observation.matched_user_display_name == "Thom"
    assert cleared.status == "cleared"
    assert provider.summary(user_id="main_user").enrolled is False


def test_matrix_household_identity_manager_tracks_face_voice_and_feedback() -> None:
    manager = MatrixHouseholdIdentityManager()

    enrollment, member_after_face = manager.enroll_face(user_id="main_user", display_name="Thom")
    voice_summary, member_after_voice = manager.enroll_voice(
        b"\x00\x01" * 24000,
        sample_rate=24000,
        channels=1,
        user_id="main_user",
        display_name="Thom",
    )
    feedback, member_after_feedback = manager.record_feedback(
        outcome="confirm",
        user_id="main_user",
        display_name="Thom",
    )
    status = manager.status()

    assert enrollment.status == "enrolled"
    assert member_after_face.portrait_reference_count == 1
    assert voice_summary.sample_count == 1
    assert member_after_voice.voice_sample_count == 1
    assert feedback.outcome == "confirm"
    assert member_after_feedback.confirm_count == 1
    assert len(status.members) == 1
    assert status.current_observation is not None
    assert status.current_observation.matched_user_display_name == "Thom"


def test_matrix_smart_home_provider_lists_controls_and_streams() -> None:
    provider = MatrixSmartHomeProvider()
    adapter = provider.build_adapter()

    listed = adapter.execute(
        request=type(
            "_Req",
            (),
            {
                "operation_id": "list_entities",
                "parameters": {"entity_class": SmartHomeEntityClass.LIGHT.value},
                "integration_id": "smart_home_hub",
                "origin": "test",
                "explicit_user_confirmation": False,
                "explicit_caregiver_confirmation": False,
                "dry_run": False,
                "background_trigger": False,
            },
        )()
    )
    controlled = adapter.execute(
        request=type(
            "_Req",
            (),
            {
                "operation_id": "control_entities",
                "parameters": {"entity_ids": ["light.hallway"], "command": SmartHomeCommand.TURN_ON.value},
                "integration_id": "smart_home_hub",
                "origin": "test",
                "explicit_user_confirmation": True,
                "explicit_caregiver_confirmation": False,
                "dry_run": False,
                "background_trigger": False,
            },
        )()
    )
    stream = adapter.execute(
        request=type(
            "_Req",
            (),
            {
                "operation_id": "read_sensor_stream",
                "parameters": {"limit": 5},
                "integration_id": "smart_home_hub",
                "origin": "test",
                "explicit_user_confirmation": False,
                "explicit_caregiver_confirmation": False,
                "dry_run": False,
                "background_trigger": False,
            },
        )()
    )

    assert listed.ok is True
    assert controlled.ok is True
    assert provider.state_for("light.hallway")["power"] == "on"
    assert stream.ok is True
    assert stream.details["count"] >= 1


def test_matrix_world_remote_state_round_trips_snapshots() -> None:
    remote_state = MatrixWorldRemoteState()

    remote_state.save_snapshot(snapshot_kind="world", payload={"count": 2})

    assert remote_state.load_snapshot(snapshot_kind="world") == {"count": 2}
    assert remote_state.load_snapshot(snapshot_kind="missing") is None


def test_matrix_self_coding_compile_driver_returns_automation_manifest() -> None:
    driver = MatrixSelfCodingCompileDriver()

    result = driver.run_compile(object())

    assert result.status == "ok"
    assert result.artifacts[0].kind == ArtifactKind.AUTOMATION_MANIFEST
    assert "Announce Family Updates" in result.artifacts[0].content
