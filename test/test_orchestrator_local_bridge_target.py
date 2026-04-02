from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.orchestrator.local_bridge_target import resolve_local_orchestrator_probe_target


class LocalOrchestratorBridgeTargetTests(unittest.TestCase):
    def test_host_side_probe_rewrites_non_loopback_target_to_local_bridge(self) -> None:
        config = TwinrConfig(
            project_root="/home/thh/twinr",
            orchestrator_ws_url="ws://192.168.1.154:8797/ws/orchestrator",
            orchestrator_allow_insecure_ws=True,
            orchestrator_port=8797,
        )

        with patch(
            "twinr.orchestrator.local_bridge_target._loopback_port_reachable",
            return_value=True,
        ):
            target = resolve_local_orchestrator_probe_target(config)

        self.assertTrue(target.rewritten)
        self.assertEqual(target.reason, "host_loopback_bridge_override")
        self.assertEqual(target.url, "ws://127.0.0.1:8797/ws/orchestrator")

    def test_pi_runtime_probe_keeps_configured_remote_target(self) -> None:
        config = TwinrConfig(
            project_root="/twinr",
            orchestrator_ws_url="ws://192.168.1.154:8797/ws/orchestrator",
            orchestrator_allow_insecure_ws=True,
            orchestrator_port=8797,
        )

        with patch(
            "twinr.orchestrator.local_bridge_target._loopback_port_reachable",
            return_value=True,
        ):
            target = resolve_local_orchestrator_probe_target(config)

        self.assertFalse(target.rewritten)
        self.assertIsNone(target.reason)
        self.assertEqual(target.url, "ws://192.168.1.154:8797/ws/orchestrator")

    def test_host_side_probe_keeps_configured_target_when_local_bridge_is_absent(self) -> None:
        config = TwinrConfig(
            project_root="/home/thh/twinr",
            orchestrator_ws_url="ws://192.168.1.154:8797/ws/orchestrator",
            orchestrator_allow_insecure_ws=True,
            orchestrator_port=8797,
        )

        with patch(
            "twinr.orchestrator.local_bridge_target._loopback_port_reachable",
            return_value=False,
        ):
            target = resolve_local_orchestrator_probe_target(config)

        self.assertFalse(target.rewritten)
        self.assertIsNone(target.reason)
        self.assertEqual(target.url, "ws://192.168.1.154:8797/ws/orchestrator")


if __name__ == "__main__":
    unittest.main()
