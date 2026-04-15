from pathlib import Path
from types import SimpleNamespace
import importlib
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class MainRuntimeScopeTests(unittest.TestCase):
    def test_orchestrator_probe_scope_restores_its_isolated_runtime_state(self) -> None:
        main_mod = importlib.import_module("twinr.__main__")
        try:
            config = main_mod.TwinrConfig(
                project_root="/tmp/twinr-project",
                runtime_state_path="state/runtime-state.json",
                restore_runtime_state_on_startup=True,
            )
            probe_args = SimpleNamespace(
                run_whatsapp_channel=False,
                demo_transcript=None,
                openai_prompt=None,
                vision_prompt=None,
                orchestrator_probe_turn="Hallo",
            )
            probe_runtime_config = main_mod._resolve_runtime_config(config, probe_args)
        finally:
            sys.modules.pop("twinr.__main__", None)

        self.assertIn("runtime-scopes/orchestrator-probe-turn/runtime-state.json", probe_runtime_config.runtime_state_path)
        self.assertTrue(probe_runtime_config.restore_runtime_state_on_startup)
