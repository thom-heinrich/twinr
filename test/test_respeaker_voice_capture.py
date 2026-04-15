from array import array
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.respeaker.voice_capture import (
    project_respeaker_capture_frame,
    resolve_respeaker_voice_capture_contract,
)
from twinr.hardware.respeaker.voice_mux import ReSpeakerVoiceMuxState


class ReSpeakerVoiceCaptureTests(unittest.TestCase):
    def _live_mux_state(self) -> ReSpeakerVoiceMuxState:
        return ReSpeakerVoiceMuxState(
            asr_output_enabled=True,
            output_pairs=((8, 0), (1, 0), (1, 2), (7, 3), (1, 1), (1, 3)),
        )

    def test_respeaker_mono_transport_uses_live_mux_attested_asr_lane(self) -> None:
        contract = resolve_respeaker_voice_capture_contract(
            capture_device="plughw:CARD=Array,DEV=0",
            transport_channels=1,
            mux_state=self._live_mux_state(),
        )

        self.assertEqual(contract.capture_channels, 6)
        self.assertEqual(contract.transport_channels, 1)
        self.assertEqual(contract.extract_channel_index, 1)
        self.assertEqual(contract.route_label, "respeaker_usb_asr_lane")

    def test_non_respeaker_capture_keeps_direct_contract(self) -> None:
        contract = resolve_respeaker_voice_capture_contract(
            capture_device="default",
            transport_channels=1,
        )

        self.assertEqual(contract.capture_channels, 1)
        self.assertEqual(contract.transport_channels, 1)
        self.assertIsNone(contract.extract_channel_index)
        self.assertEqual(contract.route_label, "direct_pcm")

    def test_project_capture_frame_extracts_live_asr_channel_from_interleaved_pcm(self) -> None:
        contract = resolve_respeaker_voice_capture_contract(
            capture_device="plughw:CARD=Array,DEV=0",
            transport_channels=1,
            mux_state=self._live_mux_state(),
        )
        interleaved = array(
            "h",
            [
                100,
                200,
                300,
                400,
                500,
                600,
                110,
                210,
                310,
                410,
                510,
                610,
            ],
        ).tobytes()

        projected = project_respeaker_capture_frame(interleaved, contract=contract)
        selected = array("h")
        selected.frombytes(projected)

        self.assertEqual(list(selected), [200, 210])

    def test_respeaker_contract_fails_closed_without_resolved_asr_lane(self) -> None:
        with self.assertRaises(RuntimeError):
            resolve_respeaker_voice_capture_contract(
                capture_device="plughw:CARD=Array,DEV=0",
                transport_channels=1,
                mux_state=ReSpeakerVoiceMuxState(
                    asr_output_enabled=True,
                    output_pairs=((8, 0), (1, 0), (1, 2), (0, 0), (1, 1), (1, 3)),
                ),
            )

    def test_respeaker_contract_fails_closed_when_usb_capture_width_does_not_match_productive_firmware(self) -> None:
        with self.assertRaises(RuntimeError):
            resolve_respeaker_voice_capture_contract(
                capture_device="plughw:CARD=Array,DEV=0",
                transport_channels=1,
                mux_state=ReSpeakerVoiceMuxState(
                    asr_output_enabled=True,
                    output_pairs=((8, 0), (1, 0)),
                ),
            )


if __name__ == "__main__":
    unittest.main()
