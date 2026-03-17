from datetime import UTC, datetime
from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.self_coding import (
    ArtifactKind,
    CompileTarget,
    FeasibilityOutcome,
    FeasibilityResult,
    RequirementsDialogueSession,
    RequirementsDialogueStatus,
    SelfCodingStore,
)
from twinr.agent.self_coding.activation import SelfCodingActivationService
from twinr.agent.self_coding.codex_driver import CodexCompileArtifact, CodexCompileResult
from twinr.agent.self_coding.runtime import SelfCodingSkillExecutionService
from twinr.agent.self_coding.worker import SelfCodingCompileWorker
from twinr.automations import AutomationStore


def _ready_session(*, session_id: str, skill_name: str = "Sandbox Probe") -> RequirementsDialogueSession:
    return RequirementsDialogueSession(
        session_id=session_id,
        request_summary="Compile a sandbox probe skill.",
        skill_name=skill_name,
        action="Run one sandboxed scheduled handler.",
        capabilities=("memory", "scheduler"),
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


def _package_payload(skill_code: str) -> str:
    normalized_code = skill_code.strip()
    if not normalized_code.startswith("from __future__ import annotations"):
        normalized_code = "from __future__ import annotations\n\n" + normalized_code
    return json.dumps(
        {
            "skill_package": {
                "name": "Sandbox Probe",
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
                        "content": normalized_code,
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
            summary="Compiled a sandbox probe skill package.",
            artifacts=(
                CodexCompileArtifact(
                    kind=ArtifactKind.SKILL_PACKAGE,
                    artifact_name="skill_package.json",
                    media_type="application/json",
                    content=_package_payload(self.skill_code),
                    summary="Sandbox probe package.",
                ),
            ),
        )


class _Owner:
    def __init__(self) -> None:
        self.config = type("_Config", (), {"local_timezone_name": "Europe/Berlin"})()
        self.print_backend = None
        self.agent_provider = None
        self.runtime = type("_Runtime", (), {"search_provider_conversation_context": lambda self: ()})()


class SelfCodingSandboxTests(unittest.TestCase):
    def test_runtime_rejects_unsafe_imports_inside_skill_package(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = SelfCodingStore.from_project_root(root)
            automation_store = AutomationStore(root / "state" / "automations.json", timezone_name="Europe/Berlin")
            worker = SelfCodingCompileWorker(
                store=store,
                driver=_CompileDriver(
                    """
from __future__ import annotations
import os


def refresh(ctx):
    ctx.store_json("cwd", os.getcwd())
                    """
                ),
            )
            activation = SelfCodingActivationService(store=store, automation_store=automation_store)
            runtime = SelfCodingSkillExecutionService(store=store)

            job = worker.ensure_job_for_session(_ready_session(session_id="dialogue_sandbox_import"))
            completed = worker.run_job(job.job_id)
            active = activation.confirm_activation(job_id=completed.job_id, confirmed=True)

            with self.assertRaisesRegex(RuntimeError, "sandbox|import"):
                runtime.execute_scheduled(
                    _Owner(),
                    skill_id=active.skill_id,
                    version=active.version,
                    trigger_id="refresh",
                    now=datetime(2026, 3, 17, 8, 0, tzinfo=UTC),
                )

    def test_runtime_rejects_dunder_attribute_escapes_inside_skill_package(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = SelfCodingStore.from_project_root(root)
            automation_store = AutomationStore(root / "state" / "automations.json", timezone_name="Europe/Berlin")
            worker = SelfCodingCompileWorker(
                store=store,
                driver=_CompileDriver(
                    """
def refresh(ctx):
    payload = (1).__class__.__mro__
    ctx.store_json("payload", str(payload))
                    """
                ),
            )
            activation = SelfCodingActivationService(store=store, automation_store=automation_store)
            runtime = SelfCodingSkillExecutionService(store=store)

            job = worker.ensure_job_for_session(_ready_session(session_id="dialogue_sandbox_dunder"))
            completed = worker.run_job(job.job_id)
            active = activation.confirm_activation(job_id=completed.job_id, confirmed=True)

            with self.assertRaisesRegex(RuntimeError, "sandbox|dunder|attribute"):
                runtime.execute_scheduled(
                    _Owner(),
                    skill_id=active.skill_id,
                    version=active.version,
                    trigger_id="refresh",
                    now=datetime(2026, 3, 17, 8, 0, tzinfo=UTC),
                )


if __name__ == "__main__":
    unittest.main()
