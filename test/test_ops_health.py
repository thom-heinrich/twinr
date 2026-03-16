from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.ops import health as health_mod


class OpsHealthTests(unittest.TestCase):
    def test_collect_service_health_treats_streaming_loop_as_conversation_loop(self) -> None:
        def fake_find_processes(pattern: str) -> tuple[str, ...]:
            if "--run-streaming-loop" in pattern:
                return ("123 python -u -m twinr --run-streaming-loop",)
            return ()

        with mock.patch.object(health_mod, "_find_processes", side_effect=fake_find_processes):
            services = health_mod._collect_service_health()

        conversation = next(service for service in services if service.key == "conversation_loop")
        self.assertTrue(conversation.running)
        self.assertEqual(conversation.count, 1)
        self.assertIn("--run-streaming-loop", conversation.detail)


if __name__ == "__main__":
    unittest.main()
