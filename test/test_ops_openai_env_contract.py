from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.ops.openai_env_contract import check_openai_env_contract


class OpenAIEnvContractTests(unittest.TestCase):
    def test_contract_passes_for_parseable_env_with_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        'OPENAI_API_KEY="sk-proj-test-1234567890"',
                        'OPENAI_PROJ_ID="proj_test_123"',
                        'OPENAI_MODEL="gpt-5.2"',
                        "OPENAI_SEND_PROJECT_HEADER=false",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            status = check_openai_env_contract(env_path)

        self.assertTrue(status.ok)
        self.assertTrue(status.openai_api_key_present)
        self.assertTrue(status.openai_project_id_present)
        self.assertTrue(status.config_loaded)
        self.assertFalse(status.literal_newline_collapse_detected)
        self.assertEqual(status.issues, ())

    def test_contract_fails_when_openai_key_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                'OPENAI_PROJ_ID="proj_test_123"\nOPENAI_MODEL="gpt-5.2"\n',
                encoding="utf-8",
            )

            status = check_openai_env_contract(env_path)

        self.assertFalse(status.ok)
        self.assertIn("openai_api_key_missing", status.issues)
        self.assertTrue(status.openai_project_id_present)

    def test_contract_flags_literal_newline_collapsed_env(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                'OPENAI_PROJ_ID="proj_test_123"\\nOPENAI_API_KEY="sk-proj-test-1234567890"\\nOPENAI_MODEL="gpt-5.2"\n',
                encoding="utf-8",
            )

            status = check_openai_env_contract(env_path)

        self.assertFalse(status.ok)
        self.assertTrue(status.literal_newline_collapse_detected)
        self.assertIn("env_literal_newline_collapse", status.issues)


if __name__ == "__main__":
    unittest.main()
