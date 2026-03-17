from importlib import import_module
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.self_coding.modules import (
    MODULE_LIBRARY,
    SelfCodingModuleRuntimeUnavailableError,
    module_spec_for,
)


class SelfCodingModuleLibraryTests(unittest.TestCase):
    def test_module_library_exposes_expected_mvp_specs(self) -> None:
        self.assertEqual(
            tuple(spec.module_name for spec in MODULE_LIBRARY),
            (
                "camera",
                "pir",
                "speaker",
                "web_search",
                "llm_call",
                "memory",
                "scheduler",
                "rules",
                "safety",
                "email",
                "calendar",
            ),
        )
        self.assertEqual(
            tuple(spec.capability_definition().capability_id for spec in MODULE_LIBRARY),
            (
                "camera",
                "pir",
                "speaker",
                "web_search",
                "llm_call",
                "memory",
                "scheduler",
                "rules",
                "safety",
                "email",
                "calendar",
            ),
        )

    def test_module_specs_expose_doc_header_and_public_api(self) -> None:
        speaker = module_spec_for("speaker")

        assert speaker is not None
        self.assertIn("Public API", speaker.doc_header)
        self.assertEqual(
            tuple(function.name for function in speaker.public_api),
            ("say", "play_sound", "ask_and_wait"),
        )
        self.assertTrue(all(function.summary for function in speaker.public_api))

    def test_importable_stub_module_raises_runtime_unavailable(self) -> None:
        camera_module = import_module("twinr.agent.self_coding.modules.camera")

        self.assertEqual(camera_module.MODULE_SPEC.module_name, "camera")
        with self.assertRaises(SelfCodingModuleRuntimeUnavailableError):
            camera_module.is_anyone_visible()


if __name__ == "__main__":
    unittest.main()
