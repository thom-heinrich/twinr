from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.memory.longterm.evaluation.live_memory_acceptance import LiveMemoryAcceptanceResult
from twinr.orchestrator.non_voice_acceptance import run_non_voice_acceptance
from twinr.orchestrator.probe_turn import OrchestratorProbeStageResult


class NonVoiceAcceptanceTests(unittest.TestCase):
    def test_run_non_voice_acceptance_combines_direct_tool_and_memory_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        f"PROJECT_ROOT={root}",
                        "PERSONALITY_DIR=personality",
                    ]
                ),
                encoding="utf-8",
            )
            (root / "personality").mkdir(parents=True, exist_ok=True)

            fake_probe_outcome = SimpleNamespace(
                result=SimpleNamespace(
                    text="ok",
                    rounds=1,
                    used_web_search=False,
                    model="gpt-5.4-mini",
                    request_id="req-1",
                    response_id="resp-1",
                ),
                deltas=("ok",),
                tool_handler_count=1,
                stage_results=(OrchestratorProbeStageResult(stage="websocket_turn", status="ok", elapsed_ms=12),),
            )
            fake_memory_result = LiveMemoryAcceptanceResult(
                probe_id="memory_probe",
                status="ok",
                started_at="2026-04-01T10:00:00Z",
                finished_at="2026-04-01T10:00:10Z",
                env_path=str(env_path),
                base_project_root=str(root),
                runtime_namespace="memory_ns",
                queue_before_count=1,
                queue_after_count=0,
                restart_queue_count=0,
                case_results=(
                    {
                        "case_id": "memory_case_1",
                        "phase": "after_restart",
                        "query_text": "Was weisst du?",
                        "answer_text": "ok",
                        "passed": True,
                    },
                ),
            )

            with patch("twinr.orchestrator.non_voice_acceptance.TwinrRuntime", return_value=SimpleNamespace(shutdown=lambda timeout_s=1.0: None)):
                with patch("twinr.orchestrator.non_voice_acceptance.OpenAIBackend", return_value=SimpleNamespace(close=lambda: None)):
                    with patch("twinr.orchestrator.non_voice_acceptance.run_orchestrator_probe_turn", return_value=fake_probe_outcome):
                        with patch("twinr.orchestrator.non_voice_acceptance.run_live_memory_acceptance", return_value=fake_memory_result):
                            result = run_non_voice_acceptance(env_path=env_path, emit_line=lambda line: None)
            self.assertTrue(result.ready)
            self.assertEqual(result.status, "ok")
            self.assertEqual(result.direct_case.status, "ok")
            self.assertEqual(result.tool_case.status, "ok")
            self.assertTrue(result.memory_result.ready)
            self.assertTrue(Path(result.artifact_path or "").is_file())
            self.assertTrue(Path(result.report_path or "").is_file())


if __name__ == "__main__":
    unittest.main()
