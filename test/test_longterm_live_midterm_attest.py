from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.memory.longterm.evaluation.live_midterm_attest import (
    LiveMidtermAttestResult,
    LiveMidtermSeedTurn,
    default_live_midterm_attest_path,
    load_latest_live_midterm_attest,
    write_live_midterm_attest_artifacts,
)


class LiveMidtermAttestArtifactTests(unittest.TestCase):
    def test_write_and_load_latest_live_midterm_attest_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            result = LiveMidtermAttestResult(
                probe_id="midterm_live_20260318T100000Z",
                status="ok",
                started_at="2026-03-18T10:00:00Z",
                finished_at="2026-03-18T10:00:09Z",
                env_path=str(project_root / ".env"),
                base_project_root=str(project_root),
                runtime_namespace="twinr_midterm_attest_midterm_live_20260318t100000z",
                writer_root=str(project_root / "writer"),
                fresh_reader_root=str(project_root / "reader"),
                flush_ok=True,
                midterm_context_present=True,
                follow_up_query="Was bringt mir Lea heute Abend um 19 Uhr vorbei?",
                follow_up_answer_text="Lea bringt dir eine Thermoskanne mit selbstgemachter Linsensuppe vorbei.",
                follow_up_model="gpt-5.2",
                expected_answer_terms=("thermoskanne", "linsensuppe"),
                matched_answer_terms=("thermoskanne", "linsensuppe"),
                writer_packet_ids=(
                    "midterm:20260318_lea_visit_thermos_lentil_soup_1900",
                    "midterm:user_has_daughter_lea",
                ),
                remote_packet_ids=(
                    "midterm:20260318_lea_visit_thermos_lentil_soup_1900",
                    "midterm:user_has_daughter_lea",
                ),
                fresh_reader_packet_ids=(
                    "midterm:20260318_lea_visit_thermos_lentil_soup_1900",
                    "midterm:user_has_daughter_lea",
                ),
                seed_turns=(
                    LiveMidtermSeedTurn(
                        prompt="Meine Tochter Lea bringt mir heute Abend eine Thermoskanne mit Linsensuppe vorbei.",
                        response_text="Ich merke mir das für später.",
                        model="gpt-5.2",
                    ),
                ),
                last_path_warning_class="outside_root_local_fallback_skipped",
                last_path_warning_message="Skipped local snapshot fallback because the path is outside the configured Twinr memory root.",
            )

            persisted = write_live_midterm_attest_artifacts(result, project_root=project_root)
            latest = load_latest_live_midterm_attest(project_root)

            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest.probe_id, result.probe_id)
            self.assertEqual(latest.writer_packet_ids, result.writer_packet_ids)
            self.assertEqual(latest.remote_packet_ids, result.remote_packet_ids)
            self.assertEqual(latest.fresh_reader_packet_ids, result.fresh_reader_packet_ids)
            self.assertEqual(latest.last_path_warning_class, "outside_root_local_fallback_skipped")
            self.assertTrue(Path(persisted.report_path or "").exists())
            self.assertTrue(default_live_midterm_attest_path(project_root).exists())


if __name__ == "__main__":
    unittest.main()
