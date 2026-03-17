from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.self_coding import (
    ArtifactKind,
    CompileJobStatus,
    CompileTarget,
    FeasibilityOutcome,
    FeasibilityResult,
    RequirementsDialogueSession,
    RequirementsDialogueStatus,
    SelfCodingStore,
)
from twinr.agent.self_coding.activation import SelfCodingActivationService
from twinr.agent.self_coding.codex_driver import CodexCompileArtifact, CodexCompileResult
from twinr.agent.self_coding.contracts import CompileJobRecord, CompileRunStatusRecord
from twinr.agent.self_coding.runtime import SelfCodingSkillExecutionService
from twinr.agent.self_coding.runtime.contracts import SkillPackage, SkillPackageFile, SkillPackageScheduledTrigger, canonical_skill_package_document
from twinr.agent.self_coding.sandbox import SelfCodingSandboxLimits, SelfCodingSandboxRunner
from twinr.agent.self_coding.watchdog import (
    SelfCodingRunWatchdog,
    SelfCodingWatchdogThresholds,
    cleanup_stale_compile_status,
    cleanup_stale_execution_run,
)
from twinr.agent.self_coding.worker import SelfCodingCompileWorker
from twinr.automations import AutomationStore


def _ready_session(
    *,
    session_id: str,
    capabilities: tuple[str, ...],
    skill_name: str = "Sandbox Watchdog Probe",
) -> RequirementsDialogueSession:
    return RequirementsDialogueSession(
        session_id=session_id,
        request_summary="Compile a watchdog probe skill.",
        skill_name=skill_name,
        action="Run one sandboxed handler.",
        capabilities=capabilities,
        feasibility=FeasibilityResult(
            outcome=FeasibilityOutcome.YELLOW,
            summary="Needs the skill-package path.",
            suggested_target=CompileTarget.SKILL_PACKAGE,
        ),
        status=RequirementsDialogueStatus.READY_FOR_COMPILE,
        trigger_mode="push",
        trigger_conditions=("daily_0800",),
        scope={"channel": "voice"},
        constraints=(),
    )


def _package_payload(skill_code: str, *, name: str = "Sandbox Watchdog Probe") -> str:
    return json.dumps(
        {
            "skill_package": {
                "name": name,
                "description": "Probe the sandboxed runtime.",
                "entry_module": "skill_main.py",
                "scheduled_triggers": [
                    {
                        "trigger_id": "refresh",
                        "schedule": "daily",
                        "time_of_day": "08:00",
                        "timezone_name": "Europe/Berlin",
                        "handler": "refresh",
                    }
                ],
                "sensor_triggers": [],
                "files": [
                    {
                        "path": "skill_main.py",
                        "content": skill_code.strip(),
                    }
                ],
            }
        }
    )


class _CompileDriver:
    def __init__(self, skill_code: str) -> None:
        self.skill_code = skill_code

    def run_compile(self, request, *, event_sink=None) -> CodexCompileResult:
        del request, event_sink
        return CodexCompileResult(
            status="ok",
            summary="Compiled a sandbox watchdog probe skill package.",
            artifacts=(
                CodexCompileArtifact(
                    kind=ArtifactKind.SKILL_PACKAGE,
                    artifact_name="skill_package.json",
                    media_type="application/json",
                    content=_package_payload(self.skill_code),
                    summary="Sandbox watchdog probe package.",
                ),
            ),
        )


class _Owner:
    def __init__(self) -> None:
        self.config = type(
            "_Config",
            (),
            {
                "local_timezone_name": "Europe/Berlin",
                "project_root": ".",
            },
        )()
        self.print_backend = None
        self.agent_provider = None
        self.runtime = type("_Runtime", (), {"search_provider_conversation_context": lambda self: ()})()


class SelfCodingWatchdogTests(unittest.TestCase):
    def test_canonical_skill_package_embeds_capability_policy_manifest(self) -> None:
        session = _ready_session(
            session_id="dialogue_policy_manifest",
            capabilities=("speaker", "memory", "scheduler"),
            skill_name="Policy Manifest Probe",
        )
        job = CompileJobRecord(
            job_id="job_policy_manifest",
            skill_id="policy_manifest_probe",
            skill_name="Policy Manifest Probe",
            status=CompileJobStatus.COMPILING,
            requested_target=CompileTarget.SKILL_PACKAGE,
            spec_hash="spec_policy_manifest",
            required_capabilities=session.capabilities,
        )
        compiled = canonical_skill_package_document(
            job=job,
            session=session,
            package=SkillPackage(
                name="Policy Manifest Probe",
                description="Probe the policy manifest.",
                entry_module="skill_main.py",
                scheduled_triggers=(
                    SkillPackageScheduledTrigger(
                        trigger_id="refresh",
                        schedule="daily",
                        time_of_day="08:00",
                        timezone_name="Europe/Berlin",
                        handler="refresh",
                    ),
                ),
                files=(
                    SkillPackageFile(
                        path="skill_main.py",
                        content="def refresh(ctx):\n    ctx.say('Hallo')\n",
                    ),
                ),
            ),
        )

        payload = json.loads(compiled.content)
        manifest = payload["sandbox"]["policy_manifest"]

        self.assertEqual(tuple(manifest["required_capabilities"]), ("speaker", "memory", "scheduler"))
        self.assertEqual(
            tuple(manifest["allowed_methods"]),
            (
                "delete_json",
                "list_json_keys",
                "load_json",
                "merge_json",
                "now_iso",
                "say",
                "store_json",
                "today_local_date",
            ),
        )
        self.assertNotIn("search_web", manifest["allowed_methods"])
        self.assertEqual(manifest["capability_methods"]["speaker"], ["say"])

    def test_runtime_blocks_ctx_methods_not_allowed_by_policy_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = SelfCodingStore.from_project_root(root)
            automation_store = AutomationStore(root / "state" / "automations.json", timezone_name="Europe/Berlin")
            worker = SelfCodingCompileWorker(
                store=store,
                driver=_CompileDriver(
                    """
def refresh(ctx):
    result = ctx.search_web("weather")
    ctx.store_json("answer", {"summary": result.answer})
                    """
                ),
            )
            activation = SelfCodingActivationService(store=store, automation_store=automation_store)
            runtime = SelfCodingSkillExecutionService(store=store)

            job = worker.ensure_job_for_session(
                _ready_session(
                    session_id="dialogue_policy_block",
                    capabilities=("memory", "scheduler"),
                )
            )
            completed = worker.run_job(job.job_id)
            active = activation.confirm_activation(job_id=completed.job_id, confirmed=True)

            with self.assertRaisesRegex(RuntimeError, "search_web|sandbox broker"):
                runtime.execute_scheduled(
                    _Owner(),
                    skill_id=active.skill_id,
                    version=active.version,
                    trigger_id="refresh",
                    now=datetime(2026, 3, 17, 8, 0, tzinfo=UTC),
                )

    def test_timeout_run_is_persisted_and_marked_by_watchdog(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = SelfCodingStore.from_project_root(root)
            automation_store = AutomationStore(root / "state" / "automations.json", timezone_name="Europe/Berlin")
            worker = SelfCodingCompileWorker(
                store=store,
                driver=_CompileDriver(
                    """
def refresh(ctx):
    while True:
        pass
                    """
                ),
            )
            activation = SelfCodingActivationService(store=store, automation_store=automation_store)
            runtime = SelfCodingSkillExecutionService(
                store=store,
                sandbox_runner=SelfCodingSandboxRunner(
                    limits=SelfCodingSandboxLimits(timeout_seconds=0.2, cpu_seconds=60)
                ),
            )

            job = worker.ensure_job_for_session(
                _ready_session(
                    session_id="dialogue_policy_timeout",
                    capabilities=("scheduler",),
                )
            )
            completed = worker.run_job(job.job_id)
            active = activation.confirm_activation(job_id=completed.job_id, confirmed=True)

            with self.assertRaisesRegex(RuntimeError, "timed out|sandbox"):
                runtime.execute_scheduled(
                    _Owner(),
                    skill_id=active.skill_id,
                    version=active.version,
                    trigger_id="refresh",
                    now=datetime(2026, 3, 17, 8, 0, tzinfo=UTC),
                )

            runs = store.list_execution_runs(skill_id=active.skill_id)
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0].status, "timed_out")
            self.assertEqual(runs[0].run_kind, "scheduled_trigger")
            self.assertEqual(runs[0].metadata["timeout_seconds"], 0.2)
            self.assertIn("hardening", runs[0].metadata)

    def test_watchdog_surfaces_stale_compile_and_execution_runs_and_cleanup_marks_them(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = SelfCodingStore.from_project_root(root)
            now = datetime(2026, 3, 17, 8, 30, tzinfo=UTC)
            stale_at = now - timedelta(minutes=25)
            store.save_job(
                CompileJobRecord(
                    job_id="job_watchdog_stale",
                    skill_id="morning_briefing",
                    skill_name="Morning Briefing",
                    status=CompileJobStatus.COMPILING,
                    requested_target=CompileTarget.SKILL_PACKAGE,
                    spec_hash="spec_watchdog_stale",
                    required_capabilities=("speaker", "web_search"),
                    created_at=stale_at,
                    updated_at=stale_at,
                )
            )
            store.save_compile_status(
                CompileRunStatusRecord(
                    job_id="job_watchdog_stale",
                    phase="streaming",
                    driver_name="CodexSdkDriver",
                    event_count=4,
                    last_event_kind="assistant_delta",
                    started_at=stale_at,
                    updated_at=stale_at,
                )
            )
            running = store.save_execution_run(
                run_id="run_watchdog_stale",
                run_kind="retest",
                skill_id="morning_briefing",
                version=2,
                status="running",
                started_at=stale_at,
                updated_at=stale_at,
                metadata={"environment": "web"},
            )

            watchdog = SelfCodingRunWatchdog(
                store=store,
                thresholds=SelfCodingWatchdogThresholds(
                    stale_compile_seconds=300,
                    stale_execution_seconds=300,
                ),
            )
            snapshot = watchdog.build_snapshot(now=now)

            self.assertEqual(len(snapshot.stale_compile_runs), 1)
            self.assertEqual(snapshot.stale_compile_runs[0].job_id, "job_watchdog_stale")
            self.assertEqual(len(snapshot.stale_execution_runs), 1)
            self.assertEqual(snapshot.stale_execution_runs[0].run_id, "run_watchdog_stale")

            compile_record = cleanup_stale_compile_status(store=store, job_id="job_watchdog_stale", reason="operator_cleanup")
            execution_record = cleanup_stale_execution_run(store=store, run_id=running.run_id, reason="operator_cleanup")
            updated_job = store.load_job("job_watchdog_stale")

            self.assertEqual(compile_record.phase, "aborted")
            self.assertEqual(updated_job.status, CompileJobStatus.FAILED)
            self.assertEqual(execution_record.status, "cleaned")
            self.assertEqual(execution_record.reason, "operator_cleanup")


if __name__ == "__main__":
    unittest.main()
