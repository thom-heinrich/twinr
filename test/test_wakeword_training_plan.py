from pathlib import Path
import tempfile
import unittest

from twinr.proactive.wakeword.training_plan import (
    build_default_wakeword_training_plan,
    render_wakeword_training_plan_markdown,
)


class WakewordTrainingPlanTests(unittest.TestCase):
    def test_default_training_plan_uses_family_stage1_and_confusion_guard(self) -> None:
        plan = build_default_wakeword_training_plan(project_root=Path("/tmp/twinr"))

        self.assertEqual(plan.stage1_phrase_profile, "family")
        self.assertEqual(plan.positive_families, ("twinr", "twinna", "twina", "twinner"))
        self.assertEqual(plan.confusion_families, ("twin", "winner", "winter", "tina", "timer", "twitter"))
        self.assertIn("critical16", plan.holdout_manifests)
        self.assertIn("family34", plan.holdout_manifests)
        self.assertTrue(any(metric.name == "ambient_false_accepts_per_hour" for metric in plan.acceptance_metrics))
        self.assertTrue(any(command.name == "Build Stage-1 Dataset" for command in plan.commands))

    def test_render_training_plan_markdown_includes_commands_and_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = build_default_wakeword_training_plan(project_root=Path(temp_dir))

        markdown = render_wakeword_training_plan_markdown(plan)

        self.assertIn("# Twinr Wakeword Training Plan", markdown)
        self.assertIn("critical16_false_negatives", markdown)
        self.assertIn("ambient_false_accepts_per_hour", markdown)
        self.assertIn("--phrase-profile family", markdown)
        self.assertIn("--hard-negative-manifest <pi_room_capture_manifest.json>", markdown)
        self.assertIn("Twin, Winner, Winter, Tina, Timer, Twitter", markdown)


if __name__ == "__main__":
    unittest.main()
