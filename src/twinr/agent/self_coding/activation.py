"""Activate, version, and roll back learned self-coding automation skills."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
import logging
from threading import RLock
from typing import TYPE_CHECKING, Any

from twinr.agent.self_coding.contracts import ActivationRecord, CompileArtifactRecord, CompileJobRecord
from twinr.agent.self_coding.runtime import (
    skill_package_activation_metadata,
    skill_package_automation_entries,
    skill_package_document_from_document,
)
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

if TYPE_CHECKING:
    from twinr.agent.self_coding.runtime.state import SelfCodingSkillRuntimeStore

_ROLLBACK_READY_STATUSES = frozenset(
    {
        LearnedSkillStatus.ACTIVE,
        LearnedSkillStatus.PAUSED,
    }
)

LOGGER = logging.getLogger(__name__)


class SelfCodingActivationService:
    """Stage, confirm, roll back, and retire activation-ready learned skills."""

    def __init__(
        self,
        *,
        store: SelfCodingStore,
        automation_store: AutomationStore,
        feedback_window: timedelta = timedelta(hours=72),
    ) -> None:
        self.store = store
        self.automation_store = automation_store
        # AUDIT-FIX(#9): Reject invalid feedback windows so newly activated skills never become immediately overdue.
        if feedback_window <= timedelta(0):
            raise ValueError("feedback_window must be a positive timedelta")
        self.feedback_window = feedback_window
        # AUDIT-FIX(#1): Serialize multi-step state mutations within this process to avoid duplicate versions and split-brain activations.
        self._mutation_lock = RLock()

    def prepare_soft_launch(self, job_id: str) -> ActivationRecord:
        """Stage the canonical manifest from a completed compile job."""

        # AUDIT-FIX(#1): Guard read-modify-write staging so version assignment and automation upserts stay consistent.
        with self._mutation_lock:
            existing = self.store.find_activation_for_job(job_id)
            # AUDIT-FIX(#4): Allow a previously retired job to be staged again as a fresh version instead of returning a cleaned-up record.
            if existing is not None and existing.status != LearnedSkillStatus.RETIRED:
                return existing

            job = self.store.load_job(job_id)
            if job.status != CompileJobStatus.SOFT_LAUNCH_READY:
                raise ValueError("soft launch requires a job in soft_launch_ready status")
            version = self._next_version(job.skill_id)
            artifact = self._activation_artifact_for_job(job)

            staged_automations: tuple[AutomationDefinition, ...]
            if artifact.kind == ArtifactKind.AUTOMATION_MANIFEST:
                manifest = self._load_manifest_payload(artifact)
                # AUDIT-FIX(#2): Derive a per-skill fallback automation id to prevent cross-skill collisions when manifests omit automation_id.
                staged_automation = self._versioned_automation_from_manifest(
                    manifest,
                    version=version,
                    skill_id=job.skill_id,
                )
                staged_automations = (staged_automation,)
                activation = ActivationRecord(
                    skill_id=job.skill_id,
                    skill_name=job.skill_name,
                    version=version,
                    status=LearnedSkillStatus.SOFT_LAUNCH_READY,
                    job_id=job.job_id,
                    artifact_id=artifact.artifact_id,
                    metadata={
                        "artifact_kind": ArtifactKind.AUTOMATION_MANIFEST.value,
                        "automation_id": staged_automation.automation_id,
                        "automation_ids": [staged_automation.automation_id],
                        "base_automation_id": str(manifest.get("automation", {}).get("automation_id") or ""),
                        "manifest_schema": str(manifest.get("schema") or ""),
                    },
                )
            else:
                document = skill_package_document_from_document(
                    self.store.read_text_artifact(artifact.artifact_id),
                    fallback_capabilities=job.required_capabilities,
                )
                package = document.package
                entries = tuple(
                    skill_package_automation_entries(
                        skill_id=job.skill_id,
                        skill_name=job.skill_name,
                        version=version,
                        package=package,
                    )
                )
                staged_automations = entries
                activation = ActivationRecord(
                    skill_id=job.skill_id,
                    skill_name=job.skill_name,
                    version=version,
                    status=LearnedSkillStatus.SOFT_LAUNCH_READY,
                    job_id=job.job_id,
                    artifact_id=artifact.artifact_id,
                    metadata=skill_package_activation_metadata(
                        package=package,
                        automation_entries=entries,
                        policy_manifest=document.policy_manifest,
                    ),
                )

            # AUDIT-FIX(#7): Normalize activation metadata so later enable/disable paths do not crash on missing or malformed automation_ids.
            activation = self._activation_with_automation_ids(activation, staged_automations)
            automation_ids = tuple(entry.automation_id for entry in staged_automations)
            snapshot = self._snapshot_automation_entries(automation_ids)
            try:
                # AUDIT-FIX(#3): Restore automation definitions if activation persistence fails after partial staging.
                for entry in staged_automations:
                    self.automation_store.upsert(entry)
                return self.store.save_activation(activation)
            except Exception:
                self._restore_automation_entries(snapshot)
                raise

    def confirm_activation(self, *, job_id: str, confirmed: bool) -> ActivationRecord:
        """Enable one staged version after explicit user confirmation."""

        if not confirmed:
            raise ValueError("activation requires explicit confirmation")

        # AUDIT-FIX(#1): Perform enable/disable transitions under a shared mutation lock with best-effort rollback.
        with self._mutation_lock:
            activation = self.prepare_soft_launch(job_id)
            if activation.status == LearnedSkillStatus.ACTIVE:
                return activation
            # AUDIT-FIX(#4): Do not allow confirmation to reactivate paused/retired records that were merely found by job_id.
            if activation.status != LearnedSkillStatus.SOFT_LAUNCH_READY:
                raise ValueError("activation confirmation requires a soft_launch_ready learned skill version")

            now = datetime.now(UTC)
            competing = [
                existing
                for existing in self.store.list_activations(skill_id=activation.skill_id)
                if existing.version != activation.version and existing.status == LearnedSkillStatus.ACTIVE
            ]
            originals = tuple(replace(item) for item in competing) + (replace(activation),)
            automation_ids = self._combined_activation_automation_ids((*competing, activation))
            snapshot = self._snapshot_automation_entries(automation_ids)

            try:
                for existing in competing:
                    self._set_automation_enabled(existing, enabled=False)
                for existing in competing:
                    self.store.save_activation(
                        replace(
                            existing,
                            status=LearnedSkillStatus.PAUSED,
                            updated_at=now,
                        )
                    )

                active = replace(
                    activation,
                    status=LearnedSkillStatus.ACTIVE,
                    updated_at=now,
                    activated_at=now,
                    feedback_due_at=now + self.feedback_window,
                )
                active = self.store.save_activation(active)
                self._set_automation_enabled(active, enabled=True)
                return active
            except Exception:
                self._restore_automation_entries(snapshot)
                self._restore_activation_records(originals)
                raise

    def rollback_activation(self, *, skill_id: str, target_version: int | None = None) -> ActivationRecord:
        """Pause the current active version and restore the previous stable one."""

        # AUDIT-FIX(#1): Serialize rollback so active-version selection and status flips cannot interleave with other mutations.
        with self._mutation_lock:
            activations = self.store.list_activations(skill_id=skill_id)
            # AUDIT-FIX(#6): Resolve the "current" version deterministically and normalize any corrupted multi-active state.
            active_records = [item for item in activations if item.status == LearnedSkillStatus.ACTIVE]
            if not active_records:
                raise ValueError("rollback requires one active learned skill version")
            current = max(active_records, key=lambda item: item.version)

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

            competing = [item for item in active_records if item.version != target.version]
            if target.version == current.version and not competing:
                return current

            now = datetime.now(UTC)
            originals = tuple(replace(item) for item in competing) + (replace(target),)
            automation_ids = self._combined_activation_automation_ids((*competing, target))
            snapshot = self._snapshot_automation_entries(automation_ids)

            try:
                for existing in competing:
                    self._set_automation_enabled(existing, enabled=False)
                for existing in competing:
                    self.store.save_activation(
                        replace(
                            existing,
                            status=LearnedSkillStatus.PAUSED,
                            updated_at=now,
                        )
                    )

                restored = replace(
                    target,
                    status=LearnedSkillStatus.ACTIVE,
                    updated_at=now,
                    activated_at=now,
                    feedback_due_at=now + self.feedback_window,
                )
                restored = self.store.save_activation(restored)
                self._set_automation_enabled(restored, enabled=True)
                return restored
            except Exception:
                self._restore_automation_entries(snapshot)
                self._restore_activation_records(originals)
                raise

    def pause_activation(self, *, skill_id: str, version: int, reason: str = "operator_pause") -> ActivationRecord:
        """Pause one active learned skill version and disable its automations."""

        # AUDIT-FIX(#1): Lock pause so automation disablement and persisted status cannot diverge under concurrent requests.
        with self._mutation_lock:
            activation = self.store.load_activation(skill_id, version=version)
            if activation.status == LearnedSkillStatus.PAUSED:
                return activation
            if activation.status != LearnedSkillStatus.ACTIVE:
                raise ValueError("pause requires an active learned skill version")

            now = datetime.now(UTC)
            # AUDIT-FIX(#7): Normalize metadata before mutation to survive malformed file-backed records.
            metadata = self._activation_metadata(activation)
            metadata["pause_reason"] = str(reason or "operator_pause").strip() or "operator_pause"
            metadata["paused_at"] = _utc_isoformat(now)
            snapshot = self._snapshot_automation_entries(self._activation_automation_ids(activation))

            try:
                self._set_automation_enabled(activation, enabled=False)
                return self.store.save_activation(
                    replace(
                        activation,
                        status=LearnedSkillStatus.PAUSED,
                        updated_at=now,
                        metadata=metadata,
                    )
                )
            except Exception:
                self._restore_automation_entries(snapshot)
                self._restore_activation_records((replace(activation),))
                raise

    def reactivate_activation(self, *, skill_id: str, version: int) -> ActivationRecord:
        """Re-enable one paused learned skill version and pause competing active ones."""

        # AUDIT-FIX(#1): Lock reactivation so status persistence and automation toggles remain coordinated.
        with self._mutation_lock:
            activation = self.store.load_activation(skill_id, version=version)
            if activation.status == LearnedSkillStatus.ACTIVE:
                return activation
            if activation.status != LearnedSkillStatus.PAUSED:
                raise ValueError("reactivation requires a paused learned skill version")

            now = datetime.now(UTC)
            competing = [
                existing
                for existing in self.store.list_activations(skill_id=activation.skill_id)
                if existing.version != activation.version and existing.status == LearnedSkillStatus.ACTIVE
            ]
            originals = tuple(replace(item) for item in competing) + (replace(activation),)
            automation_ids = self._combined_activation_automation_ids((*competing, activation))
            snapshot = self._snapshot_automation_entries(automation_ids)

            try:
                for existing in competing:
                    self._set_automation_enabled(existing, enabled=False)
                for existing in competing:
                    self.store.save_activation(
                        replace(
                            existing,
                            status=LearnedSkillStatus.PAUSED,
                            updated_at=now,
                        )
                    )

                reactivated = replace(
                    activation,
                    status=LearnedSkillStatus.ACTIVE,
                    updated_at=now,
                    activated_at=now,
                    feedback_due_at=now + self.feedback_window,
                )
                reactivated = self.store.save_activation(reactivated)
                self._set_automation_enabled(reactivated, enabled=True)
                return reactivated
            except Exception:
                self._restore_automation_entries(snapshot)
                self._restore_activation_records(originals)
                raise

    def cleanup_activation(
        self,
        *,
        skill_id: str,
        version: int,
        runtime_store: "SelfCodingSkillRuntimeStore",
        reason: str = "operator_cleanup",
    ) -> ActivationRecord:
        """Retire one inactive learned skill version and remove its runtime artifacts."""

        # AUDIT-FIX(#1): Serialize cleanup so retirement metadata and artifact removal do not race with reactivation.
        with self._mutation_lock:
            activation = self.store.load_activation(skill_id, version=version)
            if activation.status == LearnedSkillStatus.ACTIVE:
                raise ValueError("cleanup requires a non-active learned skill version")

            now = datetime.now(UTC)
            # AUDIT-FIX(#7): Normalize metadata before writing cleanup markers.
            metadata = self._activation_metadata(activation)
            metadata.update(
                {
                    "cleanup_reason": str(reason or "operator_cleanup").strip() or "operator_cleanup",
                    "cleaned_up_at": _utc_isoformat(now),
                }
            )

            # AUDIT-FIX(#3): Persist RETIRED first so restart recovery never leaves a cleaned-up version looking runnable.
            retired = self.store.save_activation(
                replace(
                    activation,
                    status=LearnedSkillStatus.RETIRED,
                    updated_at=now,
                    metadata=metadata,
                )
            )

            automation_ids = self._activation_automation_ids(retired)
            snapshot = self._snapshot_automation_entries(automation_ids)
            removed_automation_ids: list[str] = []
            cleanup_errors: list[str] = []

            for automation_id in automation_ids:
                try:
                    self.automation_store.delete(automation_id)
                except KeyError:
                    continue
                except Exception as exc:
                    cleanup_errors.append(f"automation:{automation_id}:{exc}")
                else:
                    removed_automation_ids.append(automation_id)

            removed_materialized: Any = False
            try:
                removed_materialized = runtime_store.delete_materialized_package(skill_id=skill_id, version=version)
            except Exception as exc:
                cleanup_errors.append(f"materialized_package:{exc}")

            removed_state: Any = False
            try:
                removed_state = runtime_store.delete_state(skill_id=skill_id, version=version)
            except Exception as exc:
                cleanup_errors.append(f"runtime_state:{exc}")

            removed_health: Any = False
            try:
                removed_health = self.store.delete_skill_health(skill_id, version=version)
            except Exception as exc:
                cleanup_errors.append(f"skill_health:{exc}")

            final_metadata = self._activation_metadata(retired)
            final_metadata.update(
                {
                    "cleanup_removed_automation_ids": removed_automation_ids,
                    "cleanup_removed_materialized": removed_materialized,
                    "cleanup_removed_state": removed_state,
                    "cleanup_removed_health": removed_health,
                }
            )
            if cleanup_errors:
                final_metadata["cleanup_errors"] = cleanup_errors

            try:
                return self.store.save_activation(
                    replace(
                        retired,
                        updated_at=now,
                        metadata=final_metadata,
                    )
                )
            except Exception:
                self._restore_automation_entries(snapshot)
                raise

    def load_activation(self, *, skill_id: str, version: int) -> ActivationRecord:
        """Load one persisted activation record."""

        return self.store.load_activation(skill_id, version=version)

    def _activation_artifact_for_job(self, job: CompileJobRecord) -> CompileArtifactRecord:
        artifacts = self.store.list_artifacts(job_id=job.job_id)
        for artifact in artifacts:
            if artifact.kind == ArtifactKind.AUTOMATION_MANIFEST:
                return artifact
        for artifact in artifacts:
            if artifact.kind == ArtifactKind.SKILL_PACKAGE:
                return artifact
        raise ValueError(f"compile job {job.job_id!r} has no activatable artifact")

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

    def _versioned_automation_from_manifest(
        self,
        manifest: dict[str, Any],
        *,
        version: int,
        skill_id: str,
    ) -> AutomationDefinition:
        automation_payload = dict(manifest.get("automation", {}))
        fallback_automation_id = _default_automation_base_id(skill_id)
        automation = _automation_definition_from_payload(
            automation_payload,
            timezone_name=self.automation_store.timezone_name,
            fallback_automation_id=fallback_automation_id,
        )
        base_automation_id = (
            str(automation_payload.get("automation_id") or automation.automation_id).strip()
            or automation.automation_id
        )
        return replace(
            automation,
            automation_id=_versioned_automation_id(base_automation_id, version),
            enabled=False,
        )

    def _set_automation_enabled(self, activation: ActivationRecord, *, enabled: bool) -> None:
        automation_ids = self._activation_automation_ids(activation)
        if not automation_ids:
            raise ValueError("activation record is missing automation_ids metadata")
        for automation_id in automation_ids:
            entry = self.automation_store.get(automation_id)
            if entry is None:
                raise ValueError(f"missing staged automation {automation_id!r}")
            self.automation_store.update(automation_id, enabled=enabled)

    # AUDIT-FIX(#3): Snapshot automation definitions by value so failed stages/transitions can restore the exact previous entries.
    def _snapshot_automation_entries(
        self,
        automation_ids: tuple[str, ...],
    ) -> dict[str, AutomationDefinition | None]:
        snapshots: dict[str, AutomationDefinition | None] = {}
        for automation_id in automation_ids:
            entry = self.automation_store.get(automation_id)
            snapshots[automation_id] = replace(entry) if entry is not None else None
        return snapshots

    # AUDIT-FIX(#1): Best-effort restoration narrows the blast radius when a multi-step mutation fails halfway through.
    def _restore_automation_entries(
        self,
        snapshot: dict[str, AutomationDefinition | None],
    ) -> None:
        for automation_id, entry in snapshot.items():
            try:
                if entry is None:
                    self.automation_store.delete(automation_id)
                else:
                    self.automation_store.upsert(entry)
            except KeyError:
                continue
            except Exception:
                LOGGER.warning(
                    "Self-coding activation failed to restore automation entry %s during rollback.",
                    automation_id,
                    exc_info=True,
                )
                continue

    # AUDIT-FIX(#1): Restore persisted activation records after partial failures so status storage matches automation state again.
    def _restore_activation_records(self, records: tuple[ActivationRecord, ...]) -> None:
        for record in records:
            try:
                self.store.save_activation(record)
            except Exception:
                LOGGER.warning(
                    "Self-coding activation failed to restore activation record %s v%s during rollback.",
                    record.skill_id,
                    record.version,
                    exc_info=True,
                )
                continue

    @staticmethod
    # AUDIT-FIX(#7): Fail with a clear validation error instead of AttributeError/TypeError when file-backed metadata is malformed.
    def _activation_metadata(activation: ActivationRecord) -> dict[str, Any]:
        raw_metadata = activation.metadata
        if raw_metadata is None:
            return {}
        if not isinstance(raw_metadata, dict):
            raise ValueError("activation record metadata must be a JSON object")
        return dict(raw_metadata)

    @classmethod
    # AUDIT-FIX(#7): Backfill normalized automation ids into every activation record so later enable/disable paths stay deterministic.
    def _activation_with_automation_ids(
        cls,
        activation: ActivationRecord,
        automations: tuple[AutomationDefinition, ...],
    ) -> ActivationRecord:
        metadata = cls._activation_metadata(activation)
        normalized_ids = tuple(
            dict.fromkeys(
                str(entry.automation_id).strip()
                for entry in automations
                if str(entry.automation_id).strip()
            )
        )
        if normalized_ids:
            metadata["automation_ids"] = list(normalized_ids)
            metadata.setdefault("automation_id", normalized_ids[0])
        return replace(activation, metadata=metadata)

    @classmethod
    # AUDIT-FIX(#6): De-duplicate automation ids across all affected activations while normalizing corrupted multi-active states.
    def _combined_activation_automation_ids(
        cls,
        activations: tuple[ActivationRecord, ...],
    ) -> tuple[str, ...]:
        combined: list[str] = []
        for activation in activations:
            combined.extend(cls._activation_automation_ids(activation))
        return tuple(dict.fromkeys(combined))

    @classmethod
    def _activation_automation_ids(cls, activation: ActivationRecord) -> tuple[str, ...]:
        metadata = cls._activation_metadata(activation)
        raw_ids = metadata.get("automation_ids")
        if isinstance(raw_ids, (list, tuple)):
            normalized = tuple(dict.fromkeys(str(item).strip() for item in raw_ids if str(item).strip()))
            if normalized:
                return normalized
        automation_id = str(metadata.get("automation_id", "") or "").strip()
        return (automation_id,) if automation_id else ()


def _versioned_automation_id(base_automation_id: str, version: int) -> str:
    normalized_base = str(base_automation_id or "").strip() or "ase_skill"
    suffix = f"_v{int(version)}"
    if normalized_base.endswith(suffix):
        return normalized_base
    return f"{normalized_base}{suffix}"


# AUDIT-FIX(#2): Generate per-skill fallback ids so missing automation_id values cannot collide across unrelated skills.
def _default_automation_base_id(skill_id: str) -> str:
    normalized = "".join(
        character.lower() if character.isalnum() else "_"
        for character in str(skill_id or "").strip()
    ).strip("_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized or "ase_skill"


def _automation_definition_from_payload(
    payload: dict[str, Any],
    *,
    timezone_name: str,
    fallback_automation_id: str,
) -> AutomationDefinition:
    automation_id = str(payload.get("automation_id") or "").strip() or fallback_automation_id
    name = str(payload.get("name") or "").strip()
    if not name:
        raise ValueError("automation payload must include a name")
    trigger_payload = payload.get("trigger")
    actions_payload = payload.get("actions")
    if not isinstance(trigger_payload, dict):
        raise ValueError("automation payload must include a trigger object")
    if not isinstance(actions_payload, (list, tuple)) or not actions_payload:
        raise ValueError("automation payload must include at least one action")
    trigger = _automation_trigger_from_payload(trigger_payload, timezone_name=timezone_name)
    actions = tuple(_automation_action_from_payload(item) for item in actions_payload)
    return AutomationDefinition(
        automation_id=automation_id,
        name=name,
        description=_optional_string(payload.get("description"), field_name="automation.description"),
        # AUDIT-FIX(#5): Parse boolean flags strictly so string values like "false" never become True.
        enabled=_coerce_bool(payload.get("enabled"), field_name="automation.enabled", default=False),
        trigger=trigger,
        actions=actions,
        source=str(payload.get("source") or "self_coding"),
        # AUDIT-FIX(#8): Reject string/scalar tags payloads so malformed manifests do not turn one tag into character-by-character tags.
        tags=_string_tuple(payload.get("tags", ()), field_name="automation.tags"),
    )


def _automation_trigger_from_payload(
    payload: dict[str, Any],
    *,
    timezone_name: str,
) -> TimeAutomationTrigger | IfThenAutomationTrigger:
    kind = str(payload.get("kind") or "").strip().lower()
    if kind == "time":
        weekdays_payload = payload.get("weekdays", ())
        if not isinstance(weekdays_payload, (list, tuple)):
            raise ValueError("time trigger weekdays must be a list")
        return TimeAutomationTrigger(
            schedule=str(payload.get("schedule", "once")),
            due_at=_optional_string(payload.get("due_at"), field_name="time_trigger.due_at"),
            time_of_day=_optional_string(payload.get("time_of_day"), field_name="time_trigger.time_of_day"),
            weekdays=_weekday_tuple(weekdays_payload),
            timezone_name=str(payload.get("timezone_name") or timezone_name),
        )
    if kind == "if_then":
        all_conditions_payload = payload.get("all_conditions", ())
        any_conditions_payload = payload.get("any_conditions", ())
        if not isinstance(all_conditions_payload, (list, tuple)):
            raise ValueError("if_then all_conditions must be a list")
        if not isinstance(any_conditions_payload, (list, tuple)):
            raise ValueError("if_then any_conditions must be a list")
        # AUDIT-FIX(#8): Require a non-empty event name so invalid event-driven automations fail at activation time instead of much later at runtime.
        event_name = _required_string(payload.get("event_name"), field_name="if_then.event_name")
        return IfThenAutomationTrigger(
            event_name=event_name,
            all_conditions=tuple(_automation_condition_from_payload(item) for item in all_conditions_payload),
            any_conditions=tuple(_automation_condition_from_payload(item) for item in any_conditions_payload),
            # AUDIT-FIX(#5): Normalize cooldown values to a non-negative float instead of leaking strings into runtime triggers.
            cooldown_seconds=_coerce_non_negative_float(
                payload.get("cooldown_seconds"),
                field_name="if_then.cooldown_seconds",
                default=0.0,
            ),
        )
    raise ValueError(f"unsupported automation trigger kind: {kind or 'unknown'}")


def _automation_condition_from_payload(payload: Any) -> AutomationCondition:
    if not isinstance(payload, dict):
        raise ValueError("automation condition must be an object")
    # AUDIT-FIX(#8): Validate required string fields so malformed conditions do not surface as late runtime errors.
    key = _required_string(payload.get("key"), field_name="condition.key")
    operator = _required_string(payload.get("operator", "eq"), field_name="condition.operator")
    return AutomationCondition(
        key=key,
        operator=operator,
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
    # AUDIT-FIX(#8): Require an action kind so invalid manifests fail fast instead of creating unusable automation actions.
    kind = _required_string(payload.get("kind"), field_name="action.kind")
    return AutomationAction(
        kind=kind,
        text=payload.get("text"),
        tool_name=payload.get("tool_name"),
        payload=dict(payload_mapping),
        # AUDIT-FIX(#5): Parse per-action enabled flags strictly to avoid accidentally enabling actions from stringly-typed JSON.
        enabled=_coerce_bool(payload.get("enabled"), field_name="action.enabled", default=True),
    )


# AUDIT-FIX(#5): Parse stringly-typed JSON booleans safely instead of relying on Python truthiness.
def _coerce_bool(value: Any, *, field_name: str, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean")


# AUDIT-FIX(#5): Normalize numeric trigger fields eagerly so runtime triggers never receive strings or negative cooldowns.
def _coerce_non_negative_float(value: Any, *, field_name: str, default: float) -> float:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc
    if result < 0:
        raise ValueError(f"{field_name} must be greater than or equal to 0")
    return result


# AUDIT-FIX(#8): Validate required manifest strings at activation time for fast, localized failures.
def _required_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


# AUDIT-FIX(#8): Keep optional manifest strings typed and trimmed instead of letting arbitrary JSON values leak downstream.
def _optional_string(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip()
    return normalized or None


# AUDIT-FIX(#8): Require list-like string collections so malformed scalar payloads cannot silently reshape metadata.
def _string_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list")
    normalized = tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))
    return normalized


# AUDIT-FIX(#8): Parse weekday collections strictly to avoid late trigger-construction crashes.
def _weekday_tuple(value: list[Any] | tuple[Any, ...]) -> tuple[int, ...]:
    weekdays: list[int] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError("time trigger weekdays must contain integers")
        try:
            weekdays.append(int(item))
        except (TypeError, ValueError) as exc:
            raise ValueError("time trigger weekdays must contain integers") from exc
    return tuple(weekdays)


# AUDIT-FIX(#7): Emit a single normalized UTC timestamp format for pause/cleanup metadata.
def _utc_isoformat(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
