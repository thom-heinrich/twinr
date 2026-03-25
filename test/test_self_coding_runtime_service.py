from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.self_coding.runtime.service import SkillContext


class _RuntimeWithInterruptingClose:
    def close(self) -> None:
        raise KeyboardInterrupt("stop")


class SkillContextTests(unittest.TestCase):
    def test_close_does_not_silence_base_exceptions(self) -> None:
        context = object.__new__(SkillContext)
        context._managed_integrations_runtime = _RuntimeWithInterruptingClose()
        context.log_event = lambda *args, **kwargs: None

        with self.assertRaises(KeyboardInterrupt):
            context.close()

        self.assertIsNone(context._managed_integrations_runtime)


if __name__ == "__main__":
    unittest.main()
