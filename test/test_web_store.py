from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.web.support.store import parse_urlencoded_form, read_env_values, write_env_updates


class WebStoreTests(unittest.TestCase):
    def test_parse_urlencoded_form_handles_multiline_fields(self) -> None:
        form = parse_urlencoded_form(b"name=Twinr&note=Hello%0AWorld")
        self.assertEqual(form, {"name": "Twinr", "note": "Hello\nWorld"})

    def test_write_env_updates_preserves_comments_and_quotes_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text("# Twinr\nOPENAI_MODEL=gpt-5.2\n", encoding="utf-8")

            write_env_updates(
                env_path,
                {
                    "OPENAI_MODEL": "gpt-4o-mini",
                    "TWINR_PRINTER_HEADER_TEXT": "TWINR.com",
                    "OPENAI_TTS_INSTRUCTIONS": "Speak natural German.",
                },
            )

            text = env_path.read_text(encoding="utf-8")
            self.assertIn("# Twinr", text)
            self.assertIn("OPENAI_MODEL=gpt-4o-mini", text)
            self.assertIn('TWINR_PRINTER_HEADER_TEXT=TWINR.com', text)
            self.assertIn('OPENAI_TTS_INSTRUCTIONS=\"Speak natural German.\"', text)
            self.assertEqual(read_env_values(env_path)["OPENAI_MODEL"], "gpt-4o-mini")


if __name__ == "__main__":
    unittest.main()
