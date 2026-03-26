from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
from unittest import mock
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.channels.onboarding import ChannelPairingSnapshot
from twinr.channels.service_connect import start_service_connect_flow
from twinr.display.service_connect_cues import DisplayServiceConnectCueStore


class _FakeWhatsAppCoordinator:
    def __init__(self, *, store, registry, snapshot_observer=None, pairing_window_s: float = 90.0) -> None:
        del store, registry, snapshot_observer
        self.pairing_window_s = pairing_window_s

    def load_snapshot(self) -> ChannelPairingSnapshot:
        return ChannelPairingSnapshot.initial("whatsapp")

    def start_pairing(self, _config: TwinrConfig) -> bool:
        return True


class ServiceConnectTests(unittest.TestCase):
    def make_config(self, project_root: str) -> TwinrConfig:
        return TwinrConfig(
            openai_api_key="sk-test",
            project_root=project_root,
            personality_dir="personality",
        )

    def test_start_service_connect_flow_reports_unsupported_service_and_writes_cue(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.make_config(temp_dir)

            result = start_service_connect_flow(config, service="Signal")
            cue = DisplayServiceConnectCueStore.from_config(config).load()

        self.assertEqual(result.status, "unsupported")
        self.assertEqual(result.service_id, "signal")
        self.assertFalse(result.supported)
        self.assertIsNotNone(cue)
        assert cue is not None
        self.assertEqual(cue.service_id, "signal")
        self.assertEqual(cue.phase, "unsupported")
        self.assertEqual(cue.accent, "alert")

    def test_start_service_connect_flow_accepts_service_alias_and_starts_whatsapp_pairing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.make_config(temp_dir)
            probe = SimpleNamespace(
                node_ready=True,
                node_detail="Node ok",
                worker_ready=True,
                worker_detail="Worker ok",
                paired=False,
                pair_detail="No linked session yet.",
            )

            with mock.patch("twinr.channels.whatsapp.pairing.probe_whatsapp_runtime", return_value=probe), mock.patch(
                "twinr.channels.whatsapp.pairing.WhatsAppPairingCoordinator",
                _FakeWhatsAppCoordinator,
            ):
                result = start_service_connect_flow(config, service="WA")
            cue = DisplayServiceConnectCueStore.from_config(config).load()

        self.assertEqual(result.status, "started")
        self.assertEqual(result.service_id, "whatsapp")
        self.assertEqual(result.service_label, "WhatsApp")
        self.assertTrue(result.started)
        self.assertTrue(result.running)
        self.assertFalse(result.qr_visible)
        self.assertIsNotNone(cue)
        assert cue is not None
        self.assertEqual(cue.service_id, "whatsapp")
        self.assertEqual(cue.phase, "starting")
        self.assertIn("QR", cue.detail)


if __name__ == "__main__":
    unittest.main()
