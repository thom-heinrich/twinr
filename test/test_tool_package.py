from pathlib import Path
import importlib
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class ToolPackageImportTests(unittest.TestCase):
    def test_schema_builders_import_without_eager_runtime_modules(self) -> None:
        sys.modules.pop("twinr.agent.tools", None)
        sys.modules.pop("twinr.agent.tools.runtime.executor", None)
        sys.modules.pop("twinr.agent.tools.runtime.dual_lane_loop", None)

        module = importlib.import_module("twinr.agent.tools")

        self.assertTrue(callable(module.build_realtime_tool_schemas))
        self.assertNotIn("twinr.agent.tools.runtime.executor", sys.modules)
        self.assertNotIn("twinr.agent.tools.runtime.dual_lane_loop", sys.modules)

    def test_runtime_exports_load_lazily_on_first_access(self) -> None:
        sys.modules.pop("twinr.agent.tools", None)
        sys.modules.pop("twinr.agent.tools.runtime.registry", None)

        module = importlib.import_module("twinr.agent.tools")

        self.assertNotIn("twinr.agent.tools.runtime.registry", sys.modules)
        handler_binder = module.bind_realtime_tool_handlers
        self.assertTrue(callable(handler_binder))
        self.assertIn("twinr.agent.tools.runtime.registry", sys.modules)


if __name__ == "__main__":
    unittest.main()
