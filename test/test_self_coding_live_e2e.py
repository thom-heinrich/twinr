"""Opt-in live end-to-end self_coding test backed by the real codex-sdk path."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.self_coding import (
    CompileJobStatus,
    SelfCodingActivationService,
    SelfCodingCapabilityRegistry,
    SelfCodingCompileWorker,
    SelfCodingFeasibilityChecker,
    SelfCodingLearningFlow,
    SelfCodingRequirementsDialogue,
    SelfCodingStore,
    SkillSpec,
    SkillTriggerSpec,
)
from twinr.automations import AutomationStore


@unittest.skipUnless(
    os.environ.get("TWINR_RUN_LIVE_CODEX_E2E") == "1",
    "Set TWINR_RUN_LIVE_CODEX_E2E=1 to run the live codex-sdk self_coding test.",
)
class SelfCodingLiveEndToEndTests(unittest.TestCase):
    def test_request_to_soft_launch_activation_with_real_codex_sdk(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = SelfCodingStore.from_project_root(root)
            automation_store = AutomationStore(root / "state" / "automations.json", timezone_name="Europe/Berlin")
            registry = SelfCodingCapabilityRegistry(project_root=root)
            checker = SelfCodingFeasibilityChecker(registry)
            worker = SelfCodingCompileWorker(store=store)
            flow = SelfCodingLearningFlow(
                store=store,
                checker=checker,
                dialogue=SelfCodingRequirementsDialogue(),
                compile_worker=worker,
            )
            activation = SelfCodingActivationService(store=store, automation_store=automation_store)

            update = flow.start_request(
                SkillSpec(
                    name="Visible Hello",
                    action="Say Hello there aloud.",
                    trigger=SkillTriggerSpec(mode="push", conditions=("user_visible",)),
                    scope={"channel": "voice"},
                    constraints=("ask_first",),
                    capabilities=("camera", "speaker", "safety"),
                ),
                request_summary="Learn a simple greeting skill when you can see me.",
            )
            assert update.session is not None

            for response in (
                {
                    "trigger_mode": "push",
                    "trigger_conditions": ["user_visible"],
                    "answer_summary": "Whenever you can see me.",
                },
                {
                    "scope": {"selection": "all"},
                    "answer_summary": "Always.",
                },
                {
                    "action": "Say Hello there aloud.",
                    "constraints": ["ask_first"],
                    "answer_summary": "Speak the greeting.",
                },
                {"confirmed": True},
            ):
                update = flow.answer_question(update.session.session_id, response)
                assert update.session is not None

            assert update.compile_job is not None
            self.assertEqual(update.compile_job.status, CompileJobStatus.QUEUED)

            completed_job = worker.run_job(update.compile_job.job_id)
            self.assertEqual(completed_job.status, CompileJobStatus.SOFT_LAUNCH_READY)
            manifest_artifact = next(
                artifact
                for artifact in store.list_artifacts(job_id=completed_job.job_id)
                if artifact.kind.value == "automation_manifest"
            )
            manifest_payload = json.loads(store.read_text_artifact(manifest_artifact.artifact_id))
            self.assertEqual(manifest_payload["automation"]["trigger"]["kind"], "if_then")
            self.assertTrue(manifest_payload["automation"]["actions"])

            staged = activation.prepare_soft_launch(completed_job.job_id)
            self.assertEqual(staged.status.value, "soft_launch_ready")

            active = activation.confirm_activation(job_id=completed_job.job_id, confirmed=True)
            self.assertEqual(active.status.value, "active")

            automation_id = str(active.metadata["automation_id"])
            staged_entry = automation_store.get(automation_id)
            self.assertIsNotNone(staged_entry)
            assert staged_entry is not None
            self.assertTrue(staged_entry.enabled)


if __name__ == "__main__":
    unittest.main()
