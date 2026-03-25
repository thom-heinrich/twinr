from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.orchestrator.server import create_app as create_orchestrator_app


class OrchestratorGatewayPolicyTests(unittest.TestCase):
    def test_create_app_requires_remote_asr_service_url(self) -> None:
        with TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENAI_API_KEY=test-key",
                        "TWINR_ORCHESTRATOR_SHARED_SECRET=secret-token",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "TWINR_VOICE_ORCHESTRATOR_REMOTE_ASR_URL",
            ):
                create_orchestrator_app(env_path)

    def test_create_app_rejects_retired_voice_gateway_env(self) -> None:
        with TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENAI_API_KEY=test-key",
                        "TWINR_ORCHESTRATOR_SHARED_SECRET=secret-token",
                        "TWINR_VOICE_ORCHESTRATOR_REMOTE_ASR_URL=http://127.0.0.1:18090",
                        "TWINR_VOICE_ORCHESTRATOR_WAKE_STAGE1_MODE=remote_asr",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "TWINR_VOICE_ORCHESTRATOR_WAKE_STAGE1_MODE is retired",
            ):
                create_orchestrator_app(env_path)


if __name__ == "__main__":
    unittest.main()
