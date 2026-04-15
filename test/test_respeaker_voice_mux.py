from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.respeaker.models import (
    ReSpeakerParameterRead,
    ReSpeakerTransportAvailability,
)
from twinr.hardware.respeaker.voice_mux import (
    AEC_ASROUTONOFF_PARAMETER,
    AUDIO_MGR_OP_ALL_PARAMETER,
    AUDIO_MGR_OP_R_PARAMETER,
    capture_respeaker_voice_mux_state,
    ensure_respeaker_voice_mux_contract,
    resolve_respeaker_usb_capture_channel_index,
)


def _successful_read(spec, decoded_value):
    return ReSpeakerParameterRead(
        spec=spec,
        captured_at=1.0,
        ok=True,
        attempt_count=1,
        status_code=0,
        decoded_value=decoded_value,
    )


class _FakeVoiceMuxTransport:
    def __init__(self, *read_batches):
        self._read_batches = list(read_batches)
        self.write_calls: list[tuple[str, tuple[object, ...]]] = []

    def capture_reads(self, specs, *, probe=None):
        del specs, probe
        return self._read_batches.pop(0)

    def write_parameter(self, spec, values, *, probe=None):
        del probe
        normalized_values = tuple(values) if not isinstance(values, tuple) else values
        self.write_calls.append((spec.name, normalized_values))
        return ReSpeakerTransportAvailability(backend="libusb", available=True)

    def close(self) -> None:
        return


class ReSpeakerVoiceMuxTests(unittest.TestCase):
    def test_capture_state_resolves_live_asr_pair_and_usb_capture_lane(self) -> None:
        transport = _FakeVoiceMuxTransport(
            (
                ReSpeakerTransportAvailability(backend="libusb", available=True),
                {
                    AEC_ASROUTONOFF_PARAMETER.name: _successful_read(
                        AEC_ASROUTONOFF_PARAMETER,
                        (1,),
                    ),
                    AUDIO_MGR_OP_ALL_PARAMETER.name: _successful_read(
                        AUDIO_MGR_OP_ALL_PARAMETER,
                        (8, 0, 1, 0, 1, 2, 7, 3, 1, 1, 1, 3),
                    ),
                },
            ),
        )

        state = capture_respeaker_voice_mux_state(transport=transport)

        self.assertTrue(state.asr_output_enabled)
        self.assertEqual(state.voice_capture_channels, 6)
        self.assertEqual(state.asr_output_pair_index, 3)
        self.assertEqual(state.asr_capture_channel_index, 1)

    def test_usb_capture_channel_mapping_interleaves_xmos_triplets(self) -> None:
        self.assertEqual(
            [resolve_respeaker_usb_capture_channel_index(index) for index in range(6)],
            [0, 2, 4, 1, 3, 5],
        )

    def test_ensure_contract_enables_asr_output_and_routes_right_primary_lane(self) -> None:
        transport = _FakeVoiceMuxTransport(
            (
                ReSpeakerTransportAvailability(backend="libusb", available=True),
                {
                    AEC_ASROUTONOFF_PARAMETER.name: _successful_read(
                        AEC_ASROUTONOFF_PARAMETER,
                        (0,),
                    ),
                    AUDIO_MGR_OP_ALL_PARAMETER.name: _successful_read(
                        AUDIO_MGR_OP_ALL_PARAMETER,
                        (8, 0, 1, 0, 1, 2, 0, 0, 1, 1, 1, 3),
                    ),
                },
            ),
            (
                ReSpeakerTransportAvailability(backend="libusb", available=True),
                {
                    AEC_ASROUTONOFF_PARAMETER.name: _successful_read(
                        AEC_ASROUTONOFF_PARAMETER,
                        (1,),
                    ),
                    AUDIO_MGR_OP_ALL_PARAMETER.name: _successful_read(
                        AUDIO_MGR_OP_ALL_PARAMETER,
                        (8, 0, 1, 0, 1, 2, 7, 3, 1, 1, 1, 3),
                    ),
                },
            ),
        )

        state = ensure_respeaker_voice_mux_contract(transport=transport)

        self.assertEqual(
            transport.write_calls,
            [
                (AEC_ASROUTONOFF_PARAMETER.name, (1,)),
                (AUDIO_MGR_OP_R_PARAMETER.name, (7, 3)),
            ],
        )
        self.assertTrue(state.asr_output_enabled)
        self.assertEqual(state.asr_output_pair_index, 3)
        self.assertEqual(state.asr_capture_channel_index, 1)


if __name__ == "__main__":
    unittest.main()
