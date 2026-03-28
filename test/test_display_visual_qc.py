from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.display.visual_qc import (
    DisplayVisualQcCapture,
    DisplayVisualQcDiffMetric,
    DisplayVisualQcRunner,
    DisplayVisualQcRunResult,
    build_visual_qc_report_markdown,
    build_visual_qc_report_payload,
)


class DisplayVisualQcTests(unittest.TestCase):
    def test_default_steps_cover_idle_face_and_presentation_flow(self) -> None:
        runner = DisplayVisualQcRunner(
            TwinrConfig(display_driver="hdmi_wayland"),
            screenshot_func=lambda _: None,
            sleep_func=lambda _: None,
        )

        steps = runner.default_steps(sample_image_path=Path("/tmp/sample.png"))

        self.assertEqual([step.key for step in steps], [
            "idle_home",
            "face_react",
            "presentation_mid",
            "presentation_focused",
            "restored_home",
        ])
        self.assertEqual(steps[1].action, "face_expression")
        self.assertEqual(steps[2].action, "presentation_scene")
        self.assertEqual(steps[3].action, "hold")
        self.assertEqual(steps[2].active_card_key, "family_photo")

    def test_run_writes_captures_diffs_and_summary(self) -> None:
        colors = [
            (0, 0, 0),
            (40, 40, 40),
            (120, 50, 40),
            (230, 180, 80),
            (0, 0, 0),
        ]
        capture_index = {"value": 0}

        def fake_capture(path: Path) -> None:
            color = colors[capture_index["value"]]
            capture_index["value"] += 1
            Image.new("RGB", (800, 480), color).save(path)

        runner = DisplayVisualQcRunner(
            TwinrConfig(display_driver="hdmi_wayland"),
            screenshot_func=fake_capture,
            sleep_func=lambda _: None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.run(Path(tmpdir))

            self.assertEqual(len(result.captures), 5)
            self.assertEqual(len(result.diffs), 4)
            self.assertTrue(Path(result.summary_path).exists())
            self.assertTrue(Path(result.contact_sheet_path).exists())
            self.assertTrue(Path(result.sample_image_path).exists())
            self.assertTrue(all(Path(path).exists() for path in result.attachment_paths()))
            self.assertTrue(all(diff.changed_pixels > 0 for diff in result.diffs))

    def test_run_retries_until_a_visible_scene_change_appears(self) -> None:
        colors = [
            (0, 0, 0),
            (0, 0, 0),
            (40, 40, 40),
            (120, 50, 40),
            (230, 180, 80),
            (0, 0, 0),
        ]
        capture_index = {"value": 0}

        def fake_capture(path: Path) -> None:
            color = colors[capture_index["value"]]
            capture_index["value"] += 1
            Image.new("RGB", (800, 480), color).save(path)

        runner = DisplayVisualQcRunner(
            TwinrConfig(display_driver="hdmi_wayland", display_poll_interval_s=0.5),
            screenshot_func=fake_capture,
            sleep_func=lambda _: None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.run(Path(tmpdir))

            self.assertEqual(len(result.captures), 5)
            self.assertGreater(capture_index["value"], len(result.captures))
            self.assertTrue(all(diff.changed_pixels > 0 for diff in result.diffs))

    def test_report_payload_and_markdown_include_visual_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            capture_path = root / "scene_01_idle.png"
            diff_path = root / "diff_01_idle__face.png"
            sample_path = root / "sample.png"
            summary_path = root / "summary.json"
            contact_sheet_path = root / "sheet.png"
            Image.new("RGB", (800, 480), (0, 0, 0)).save(capture_path)
            Image.new("RGB", (800, 480), (255, 255, 255)).save(diff_path)
            Image.new("RGB", (640, 360), (20, 20, 20)).save(sample_path)
            Image.new("RGB", (800, 480), (10, 10, 10)).save(contact_sheet_path)
            summary_path.write_text("{}", encoding="utf-8")
            result = DisplayVisualQcRunResult(
                generated_at="2026-03-18T18:00:00+00:00",
                workdir=str(root),
                sample_image_path=str(sample_path),
                summary_path=str(summary_path),
                contact_sheet_path=str(contact_sheet_path),
                captures=(
                    DisplayVisualQcCapture(
                        key="idle_home",
                        label="Idle home",
                        description="Idle state",
                        image_path=str(capture_path),
                        delay_s=0.45,
                        captured_at="2026-03-18T18:00:01+00:00",
                        width=800,
                        height=480,
                    ),
                ),
                diffs=(
                    DisplayVisualQcDiffMetric(
                        from_key="idle_home",
                        to_key="face_react",
                        diff_image_path=str(diff_path),
                        changed_pixels=1234,
                        changed_ratio=0.12,
                        bbox=(1, 2, 3, 4),
                    ),
                ),
            )

            payload = build_visual_qc_report_payload(result, title="QC report", task_id="808cb48943b5")
            markdown = build_visual_qc_report_markdown(result)

            self.assertEqual(payload["title"], "QC report")
            self.assertIn("visual-qc", payload["tags"])
            self.assertEqual(payload["links"][-1], "task:808cb48943b5")
            self.assertIn("attachment:sheet.png", {item["ref"] for item in payload["evidence"]})
            self.assertIn("./assets/sheet.png", markdown)
            self.assertIn("./assets/scene_01_idle.png", markdown)
            self.assertIn("./assets/diff_01_idle__face.png", markdown)
