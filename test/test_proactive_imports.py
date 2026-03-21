from pathlib import Path
import builtins
import importlib
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class ProactiveImportTests(unittest.TestCase):
    def test_proactive_import_survives_missing_optional_wekws_yaml_dependency(self) -> None:
        saved_modules = {
            name: module
            for name, module in list(sys.modules.items())
            if name == "twinr.proactive" or name.startswith("twinr.proactive.")
        }
        for name in list(saved_modules):
            sys.modules.pop(name, None)

        original_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "yaml":
                raise ModuleNotFoundError("No module named 'yaml'")
            return original_import(name, globals, locals, fromlist, level)

        try:
            with patch("builtins.__import__", side_effect=fake_import):
                proactive = importlib.import_module("twinr.proactive")
            self.assertTrue(callable(proactive.build_default_proactive_monitor))
            self.assertTrue(callable(proactive.WakewordPhraseSpotter))
            self.assertEqual(proactive.WakewordMatch.__name__, "WakewordMatch")
            with self.assertRaises(ModuleNotFoundError):
                proactive.WakewordWekwsSpotter()
        finally:
            for name in list(sys.modules):
                if name == "twinr.proactive" or name.startswith("twinr.proactive."):
                    sys.modules.pop(name, None)
            sys.modules.update(saved_modules)

    def test_runtime_service_import_survives_missing_optional_wekws_yaml_dependency(self) -> None:
        saved_modules = {
            name: module
            for name, module in list(sys.modules.items())
            if name == "twinr.proactive" or name.startswith("twinr.proactive.")
        }
        for name in list(saved_modules):
            sys.modules.pop(name, None)

        original_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "yaml":
                raise ModuleNotFoundError("No module named 'yaml'")
            return original_import(name, globals, locals, fromlist, level)

        try:
            with patch("builtins.__import__", side_effect=fake_import):
                service = importlib.import_module("twinr.proactive.runtime.service")
            self.assertTrue(callable(service.build_default_proactive_monitor))
            self.assertTrue(callable(service.open_pir_monitor_with_busy_retry))
        finally:
            for name in list(sys.modules):
                if name == "twinr.proactive" or name.startswith("twinr.proactive."):
                    sys.modules.pop(name, None)
            sys.modules.update(saved_modules)


if __name__ == "__main__":
    unittest.main()
