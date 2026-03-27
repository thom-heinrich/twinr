"""Prepare operator-facing self-coding telemetry rows for the local web UI."""

from __future__ import annotations

from typing import Any

from twinr.agent.self_coding.operator_status import build_self_coding_operator_status
from twinr.agent.self_coding.store import SelfCodingStore
from twinr.agent.self_coding.status import ArtifactKind, LearnedSkillStatus
from twinr.agent.self_coding.watchdog import SelfCodingRunWatchdog


def build_self_coding_ops_page_context(store: SelfCodingStore) -> dict[str, Any]:
    """Build the template context for the self-coding operations page."""

    latest_compile = _first_or_none(store.list_compile_statuses())
    latest_diagnostics = getattr(latest_compile, "diagnostics", {})
    if not isinstance(latest_diagnostics, dict):
        latest_diagnostics = {}
    latest_live_e2e = _first_or_none(store.list_live_e2e_statuses())
    watchdog_snapshot = SelfCodingRunWatchdog(store=store).build_snapshot()
    activations = store.list_activations()
    versions_by_skill: dict[str, tuple[Any, ...]] = {}
    for activation in activations:
        versions_by_skill.setdefault(activation.skill_id, ())
        versions_by_skill[activation.skill_id] = tuple((*versions_by_skill[activation.skill_id], activation))
    skill_rows = []
    for activation in activations:
        try:
            health = store.load_skill_health(activation.skill_id, version=activation.version)
        except (FileNotFoundError, PermissionError):
            health = None
        metadata = dict(getattr(activation, "metadata", {}) or {})
        artifact_kind = _activation_artifact_kind(store, activation=activation, metadata=metadata)
        rollback_target_version = _rollback_target_version(versions_by_skill[activation.skill_id], activation=activation)
        sandbox_allowed_methods = tuple(
            str(item).strip()
            for item in metadata.get("sandbox_allowed_methods", ())
            if str(item).strip()
        )
        skill_rows.append(
            {
                "skill_id": activation.skill_id,
                "skill_name": activation.skill_name,
                "version": activation.version,
                "status": activation.status.value,
                "artifact_kind": artifact_kind,
                "job_id": activation.job_id,
                "feedback_due_at": activation.feedback_due_at,
                "pause_reason": metadata.get("pause_reason"),
                "health_status": None if health is None else health.status,
                "trigger_count": None if health is None else health.trigger_count,
                "delivered_count": None if health is None else health.delivered_count,
                "error_count": None if health is None else health.error_count,
                "consecutive_error_count": None if health is None else health.consecutive_error_count,
                "auto_pause_count": None if health is None else health.auto_pause_count,
                "sandbox_allowed_methods": sandbox_allowed_methods,
                "can_pause": activation.status == LearnedSkillStatus.ACTIVE,
                "can_reactivate": activation.status == LearnedSkillStatus.PAUSED,
                "can_rollback": rollback_target_version is not None,
                "rollback_target_version": rollback_target_version,
                "can_retest": activation.status == LearnedSkillStatus.ACTIVE and artifact_kind == ArtifactKind.SKILL_PACKAGE.value,
                "can_cleanup": activation.status in {LearnedSkillStatus.PAUSED, LearnedSkillStatus.SOFT_LAUNCH_READY},
            }
        )
    return {
        "operator_status": build_self_coding_operator_status(store),
        "latest_compile_status": latest_compile,
        "latest_compile_diagnostics": latest_diagnostics,
        "latest_live_e2e": latest_live_e2e,
        "watchdog_snapshot": watchdog_snapshot,
        "skill_rows": tuple(skill_rows),
    }


def _first_or_none(items: tuple[Any, ...]) -> Any | None:
    return items[0] if items else None


def _activation_artifact_kind(store: SelfCodingStore, *, activation: Any, metadata: dict[str, Any]) -> str | None:
    text = str(metadata.get("artifact_kind") or "").strip()
    if text:
        return text
    artifact_id = str(getattr(activation, "artifact_id", "") or "").strip()
    if not artifact_id:
        return None
    try:
        artifact = store.load_artifact(artifact_id)
    except FileNotFoundError:
        return None
    return artifact.kind.value


def _rollback_target_version(items: tuple[Any, ...], *, activation: Any) -> int | None:
    if activation.status != LearnedSkillStatus.ACTIVE:
        return None
    candidates = [
        item.version
        for item in items
        if item.version < activation.version and item.status in {LearnedSkillStatus.ACTIVE, LearnedSkillStatus.PAUSED}
    ]
    if not candidates:
        return None
    return max(candidates)


__all__ = ["build_self_coding_ops_page_context"]
