from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.self_coding import (
    CompileJobStatus,
    CompileTarget,
    RequirementsDialogueStatus,
    SelfCodingCapabilityRegistry,
    SelfCodingCompileWorker,
    SelfCodingFeasibilityChecker,
    SelfCodingLearningFlow,
    SelfCodingRequirementsDialogue,
    SelfCodingStore,
    SkillSpec,
    SkillTriggerSpec,
)
from twinr.agent.self_coding.codex_driver import CodexCompileResult


@dataclass(frozen=True, slots=True)
class _FakeIntegrationReadiness:
    integration_id: str
    label: str
    status: str
    summary: str
    detail: str


@dataclass(frozen=True, slots=True)
class _FakeManagedIntegrationsRuntime:
    readiness: tuple[object, ...] = ()


class _FakeCompileDriver:
    def __init__(self) -> None:
        self.requests: list[object] = []

    def run_compile(self, request) -> CodexCompileResult:
        self.requests.append(request)
        return CodexCompileResult(
            status="ok",
            summary="Compile completed.",
            review="Artifacts drafted.",
        )


class SelfCodingDialogueTests(unittest.TestCase):
    def make_flow(
        self,
        project_root: str,
        *,
        compile_worker: SelfCodingCompileWorker | None = None,
    ) -> SelfCodingLearningFlow:
        store = SelfCodingStore.from_project_root(project_root)
        registry = SelfCodingCapabilityRegistry(
            project_root=project_root,
            integration_runtime_factory=lambda *args, **kwargs: _FakeManagedIntegrationsRuntime(
                readiness=(
                    _FakeIntegrationReadiness(
                        integration_id="email_mailbox",
                        label="Email",
                        status="ok",
                        summary="Email ready.",
                        detail="Configured mailbox is ready.",
                    ),
                    _FakeIntegrationReadiness(
                        integration_id="calendar_agenda",
                        label="Calendar",
                        status="ok",
                        summary="Calendar ready.",
                        detail="Configured calendar is ready.",
                    ),
                )
            ),
        )
        checker = SelfCodingFeasibilityChecker(registry)
        return SelfCodingLearningFlow(
            store=store,
            checker=checker,
            dialogue=SelfCodingRequirementsDialogue(),
            compile_worker=compile_worker,
        )

    def test_learning_flow_runs_question_sequence_and_produces_skill_spec(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            flow = self.make_flow(temp_dir)
            update = flow.start_request(
                SkillSpec(
                    name="Read New Emails",
                    action="Read new email aloud",
                    trigger=SkillTriggerSpec(mode="push", conditions=("new_email",)),
                    scope={"channel": "email"},
                    constraints=("not_during_night_mode",),
                    capabilities=("email", "speaker", "safety"),
                ),
                request_summary="Read out new emails.",
            )

            assert update.session is not None
            self.assertEqual(update.phase, "questioning")
            self.assertEqual(update.session.status, RequirementsDialogueStatus.QUESTIONING)
            self.assertEqual(update.prompt, "Should I do that automatically, or only when you ask me?")
            self.assertEqual(update.feasibility.suggested_target, CompileTarget.AUTOMATION_MANIFEST)

            update = flow.answer_question(
                update.session.session_id,
                {"trigger_mode": "push", "trigger_conditions": ["user_visible"], "answer_summary": "Only when I am nearby."},
            )
            assert update.session is not None
            self.assertEqual(update.prompt, "Should I do this for everything, or only in certain cases?")

            update = flow.answer_question(
                update.session.session_id,
                {"scope": {"contacts": ["family"]}, "answer_summary": "Only from family."},
            )
            assert update.session is not None
            self.assertEqual(update.prompt, "How should I do it: just do it, tell you first, or handle it another way?")

            update = flow.answer_question(
                update.session.session_id,
                {"action": "Ask first, then read the email aloud", "constraints": ["ask_first"]},
            )
            assert update.session is not None
            self.assertEqual(update.phase, "confirming")
            self.assertIn("Just to make sure:", update.prompt or "")

            update = flow.answer_question(update.session.session_id, {"confirmed": True})

            assert update.skill_spec is not None
            self.assertEqual(update.phase, "ready_for_compile")
            self.assertEqual(update.skill_spec.trigger.conditions, ("new_email", "user_visible"))
            self.assertEqual(update.skill_spec.scope["contacts"], ["family"])
            self.assertIn("ask_first", update.skill_spec.constraints)
            self.assertEqual(update.skill_spec.action, "Ask first, then read the email aloud")

    def test_rejected_confirmation_restarts_question_flow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            flow = self.make_flow(temp_dir)
            update = flow.start_request(
                SkillSpec(
                    name="Morning Calendar",
                    action="Read today's calendar aloud",
                    trigger=SkillTriggerSpec(mode="push", conditions=("morning_window",)),
                    scope={"channel": "calendar"},
                    capabilities=("calendar", "speaker"),
                ),
                request_summary="Tell me my calendar in the morning.",
            )
            assert update.session is not None

            for response in (
                {"use_default": True},
                {"use_default": True},
                {"use_default": True},
            ):
                update = flow.answer_question(update.session.session_id, response)
                assert update.session is not None

            self.assertEqual(update.phase, "confirming")
            update = flow.answer_question(update.session.session_id, {"confirmed": False})
            assert update.session is not None
            self.assertEqual(update.phase, "questioning")
            self.assertEqual(update.session.current_question_id, "when")
            self.assertEqual(update.session.answered_question_ids, ())

    def test_learning_flow_creates_compile_job_when_session_becomes_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SelfCodingStore.from_project_root(temp_dir)
            driver = _FakeCompileDriver()
            worker = SelfCodingCompileWorker(store=store, driver=driver)
            flow = self.make_flow(temp_dir, compile_worker=worker)
            update = flow.start_request(
                SkillSpec(
                    name="Read New Emails",
                    action="Read new email aloud",
                    trigger=SkillTriggerSpec(mode="push", conditions=("new_email",)),
                    scope={"channel": "email"},
                    capabilities=("email", "speaker", "safety"),
                ),
                request_summary="Read out new emails.",
            )
            assert update.session is not None

            for response in (
                {"use_default": True},
                {"use_default": True},
                {"use_default": True},
                {"confirmed": True},
            ):
                update = flow.answer_question(update.session.session_id, response)
                assert update.session is not None

            assert update.compile_job is not None
            self.assertEqual(update.phase, "ready_for_compile")
            self.assertEqual(update.compile_job.status, CompileJobStatus.QUEUED)
            self.assertEqual(store.load_job(update.compile_job.job_id).metadata["session_id"], update.session.session_id)
            self.assertEqual(driver.requests, [])


if __name__ == "__main__":
    unittest.main()
