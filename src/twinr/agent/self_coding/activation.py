"""Activate, version, and roll back learned self-coding automation skills."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
from typing import Any

from twinr.agent.self_coding.contracts import ActivationRecord, CompileArtifactRecord, CompileJobRecord
from twinr.agent.self_coding.status import ArtifactKind, CompileJobStatus, LearnedSkillStatus
from twinr.agent.self_coding.store import SelfCodingStore
from twinr.automations import (
    AutomationAction,
    AutomationCondition,
    AutomationDefinition,
    AutomationStore,
    IfThenAutomationTrigger,
    TimeAutomationTrigger,
)

_ROLLBACK_READY_STATUSES = frozenset(
    {
        LearnedSkillStatus.ACTIVE,
        LearnedSkillStatus.PAUSED,
    }
)


class SelfCodingActivationService:
    """Stage, confirm, and roll back activation-ready automation manifests."""

    def __init__(
        self,
        *,
        store: SelfCodingStore,
        automation_store: AutomationStore,
        feedback_window: timedelta = timedelta(hours=72),
    ) -> None:
        self.store = store
        self.automation_store = automation_store
        self.feedback_window = feedback_window

    def prepare_soft_launch(self, job_id: str) -> ActivationRecord:
        """Stage the canonical manifest from a completed compile job."""

        existing = self.store.find_activation_for_job(job_id)
        if existing is not None:
            return existing

        job = self.store.load_job(job_id)
        if job.status != CompileJobStatus.SOFT_LAUNCH_READY:
            raise ValueError("soft launch requires a job in soft_launch_ready status")
        artifact = self._activation_artifact_for_job(job)
        manifest = self._load_manifest_payload(artifact)
        version = self._next_version(job.skill_id)
        staged_automation = self._versioned_automation_from_manifest(manifest, version=version)
        self.automation_store.upsert(staged_automation)
        activation = ActivationRecord(
            skill_id=job.skill_id,
            skill_name=job.skill_name,
            version=version,
            status=LearnedSkillStatus.SOFT_LAUNCH_READY,
            job_id=job.job_id,
            artifact_id=artifact.artifact_id,
            metadata={
                "automation_id": staged_automation.automation_id,
                "base_automation_id": str(manifest.get("automation", {}).get("automation_id") or ""),
                "manifest_schema": str(manifest.get("schema") or ""),
            },
        )
        return self.store.save_activation(activation)

    def confirm_activation(self, *, job_id: str, confirmed: bool) -> ActivationRecord:
        """Enable one staged version after explicit user confirmation."""

        if not confirmed:
            raise ValueError("activation requires explicit confirmation")
        activation = self.prepare_soft_launch(job_id)
        if activation.status == LearnedSkillStatus.ACTIVE:
            return activation

        now = datetime.now(UTC)
        for existing in self.store.list_activations(skill_id=activation.skill_id):
            if existing.version == activation.version:
                continue
            if existing.status != LearnedSkillStatus.ACTIVE:
                continue
            self._set_automation_enabled(existing, enabled=False)
            self.store.save_activation(
                replace(
                    existing,
                    status=LearnedSkillStatus.PAUSED,
                    updated_at=now,
                )
            )

        self._set_automation_enabled(activation, enabled=True)
        active = self.store.save_activation(
            replace(
                activation,
                status=LearnedSkillStatus.ACTIVE,
                updated_at=now,
                activated_at=now,
                feedback_due_at=now + self.feedback_window,
            )
        )
        return active

    def rollback_activation(self, *, skill_id: str, target_version: int | None = None) -> ActivationRecord:
        """Pause the current active version and restore the previous stable one."""

        activations = self.store.list_activations(skill_id=skill_id)
        current = next((item for item in activations if item.status == LearnedSkillStatus.ACTIVE), None)
        if current is None:
            raise ValueError("rollback requires one active learned skill version")

        if target_version is None:
            candidates = [
                item
                for item in activations
                if item.version < current.version and item.status in _ROLLBACK_READY_STATUSES
            ]
            if not candidates:
                raise ValueError("rollback requires an earlier paused or active version")
            target = max(candidates, key=lambda item: item.version)
        else:
            target = self.store.load_activation(skill_id, version=target_version)
            if target.status not in _ROLLBACK_READY_STATUSES:
                raise ValueError("rollback target must be paused or active")

        now = datetime.now(UTC)
        self._set_automation_enabled(current, enabled=False)
        self.store.save_activation(
            replace(
                current,
                status=LearnedSkillStatus.PAUSED,
                updated_at=now,
            )
        )

        self._set_automation_enabled(target, enabled=True)
        restored = self.store.save_activation(
            replace(
                target,
                status=LearnedSkillStatus.ACTIVE,
                updated_at=now,
                activated_at=now,
                feedback_due_at=now + self.feedback_window,
            )
        )
        return restored

    def load_activation(self, *, skill_id: str, version: int) -> ActivationRecord:
        """Load one persisted activation record."""

        return self.store.load_activation(skill_id, version=version)

    def _activation_artifact_for_job(self, job: CompileJobRecord) -> CompileArtifactRecord:
        artifacts = self.store.list_artifacts(job_id=job.job_id)
        for artifact in artifacts:
            if artifact.kind == ArtifactKind.AUTOMATION_MANIFEST:
                return artifact
        raise ValueError(f"compile job {job.job_id!r} has no automation_manifest artifact")

    def _load_manifest_payload(self, artifact: CompileArtifactRecord) -> dict[str, Any]:
        try:
            payload = json.loads(self.store.read_text_artifact(artifact.artifact_id))
        except json.JSONDecodeError as exc:
            raise ValueError("stored automation manifest is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("stored automation manifest must be a JSON object")
        automation_payload = payload.get("automation")
        if not isinstance(automation_payload, dict):
            raise ValueError("stored automation manifest is missing an automation object")
        return payload

    def _next_version(self, skill_id: str) -> int:
        versions = [record.version for record in self.store.list_activations(skill_id=skill_id)]
        return (max(versions) + 1) if versions else 1

    def _versioned_automation_from_manifest(self, manifest: dict[str, Any], *, version: int) -> AutomationDefinition:
        automation_payload = dict(manifest.get("automation", {}))
        automation = _automation_definition_from_payload(
            automation_payload,
            timezone_name=self.automation_store.timezone_name,
        )
        base_automation_id = str(automation_payload.get("automation_id") or automation.automation_id).strip() or automation.automation_id
        return replace(
            automation,
            automation_id=_versioned_automation_id(base_automation_id, version),
            enabled=False,
        )

    def _set_automation_enabled(self, activation: ActivationRecord, *, enabled: bool) -> None:
        automation_id = str(activation.metadata.get("automation_id", "") or "").strip()
        if not automation_id:
            raise ValueError("activation record is missing automation_id metadata")
        entry = self.automation_store.get(automation_id)
        if entry is None:
            raise ValueError(f"missing staged automation {automation_id!r}")
        self.automation_store.update(automation_id, enabled=enabled)


def _versioned_automation_id(base_automation_id: str, version: int) -> str:
    normalized_base = str(base_automation_id or "").strip() or "ase_skill"
    suffix = f"_v{int(version)}"
    if normalized_base.endswith(suffix):
        return normalized_base
    return f"{normalized_base}{suffix}"


def _automation_definition_from_payload(payload: dict[str, Any], *, timezone_name: str) -> AutomationDefinition:
    automation_id = str(payload.get("automation_id") or "").strip() or "ase_skill"
    name = str(payload.get("name") or "").strip()
    if not name:
        raise ValueError("automation payload must include a name")
    trigger_payload = payload.get("trigger")
    actions_payload = payload.get("actions")
    if not isinstance(trigger_payload, dict):
        raise ValueError("automation payload must include a trigger object")
    if not isinstance(actions_payload, list) or not actions_payload:
        raise ValueError("automation payload must include at least one action")
    trigger = _automation_trigger_from_payload(trigger_payload, timezone_name=timezone_name)
    actions = tuple(_automation_action_from_payload(item) for item in actions_payload)
    return AutomationDefinition(
        automation_id=automation_id,
        name=name,
        description=str(payload.get("description") or "").strip() or None,
        enabled=bool(payload.get("enabled", False)),
        trigger=trigger,
        actions=actions,
        source=str(payload.get("source") or "self_coding"),
        tags=tuple(str(item).strip() for item in payload.get("tags", ()) if str(item).strip()),
    )


def _automation_trigger_from_payload(payload: dict[str, Any], *, timezone_name: str) -> TimeAutomationTrigger | IfThenAutomationTrigger:
    kind = str(payload.get("kind") or "").strip().lower()
    if kind == "time":
        weekdays_payload = payload.get("weekdays", ())
        if not isinstance(weekdays_payload, (list, tuple)):
            raise ValueError("time trigger weekdays must be a list")
        return TimeAutomationTrigger(
            schedule=str(payload.get("schedule", "once")),
            due_at=payload.get("due_at"),
            time_of_day=payload.get("time_of_day"),
            weekdays=tuple(int(item) for item in weekdays_payload),
            timezone_name=str(payload.get("timezone_name") or timezone_name),
        )
    if kind == "if_then":
        all_conditions_payload = payload.get("all_conditions", ())
        any_conditions_payload = payload.get("any_conditions", ())
        if not isinstance(all_conditions_payload, (list, tuple)):
            raise ValueError("if_then all_conditions must be a list")
        if not isinstance(any_conditions_payload, (list, tuple)):
            raise ValueError("if_then any_conditions must be a list")
        return IfThenAutomationTrigger(
            event_name=payload.get("event_name"),
            all_conditions=tuple(_automation_condition_from_payload(item) for item in all_conditions_payload),
            any_conditions=tuple(_automation_condition_from_payload(item) for item in any_conditions_payload),
            cooldown_seconds=payload.get("cooldown_seconds", 0.0) or 0.0,
        )
    raise ValueError(f"unsupported automation trigger kind: {kind or 'unknown'}")


def _automation_condition_from_payload(payload: Any) -> AutomationCondition:
    if not isinstance(payload, dict):
        raise ValueError("automation condition must be an object")
    return AutomationCondition(
        key=str(payload.get("key", "")),
        operator=str(payload.get("operator", "eq")),
        value=payload.get("value"),
    )


def _automation_action_from_payload(payload: Any) -> AutomationAction:
    if not isinstance(payload, dict):
        raise ValueError("automation action must be an object")
    payload_mapping = payload.get("payload", {})
    if payload_mapping is None:
        payload_mapping = {}
    if not isinstance(payload_mapping, dict):
        raise ValueError("automation action payload must be an object")
    return AutomationAction(
        kind=str(payload.get("kind", "")),
        text=payload.get("text"),
        tool_name=payload.get("tool_name"),
        payload=dict(payload_mapping),
        enabled=bool(payload.get("enabled", True)),
    )
