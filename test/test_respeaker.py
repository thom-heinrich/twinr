from pathlib import Path
import math
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.respeaker import (
    AEC_AZIMUTH_VALUES_PARAMETER,
    AEC_SPENERGY_VALUES_PARAMETER,
    AUDIO_MGR_SELECTED_AZIMUTHS_PARAMETER,
    DOA_VALUE_PARAMETER,
    GPO_READ_VALUES_PARAMETER,
    VERSION_PARAMETER,
    ReSpeakerLibusbTransport,
    ReSpeakerParameterRead,
    ReSpeakerProbeResult,
    ReSpeakerSignalProvider,
    ReSpeakerTransportAvailability,
    ReSpeakerUsbDevice,
    capture_respeaker_primitive_snapshot,
    config_targets_respeaker,
    probe_respeaker_xvf3800,
)
from twinr.hardware.respeaker.models import (
    ReSpeakerCaptureDevice,
    ReSpeakerDirectionSnapshot,
    ReSpeakerMuteSnapshot,
    ReSpeakerPrimitiveSnapshot,
)


class ReSpeakerProbeTests(unittest.TestCase):
    def test_config_targets_respeaker_detects_known_device_strings(self) -> None:
        self.assertTrue(config_targets_respeaker("plughw:CARD=Array,DEV=0"))
        self.assertTrue(config_targets_respeaker("alsa_input.usb-Seeed_Studio_reSpeaker_XVF3800_4-Mic_Array-00.mono-fallback"))
        self.assertFalse(config_targets_respeaker("default", "hw:CARD=Headphones"))

    def test_probe_detects_audio_ready_runtime(self) -> None:
        def fake_run(command, *, capture_output: bool, text: bool, check: bool, timeout: float, env, encoding: str, errors: str):
            if command[:3] == ["/usr/bin/lsusb", "-d", "2886:001a"]:
                return type(
                    "CompletedProcess",
                    (),
                    {
                        "returncode": 0,
                        "stdout": "Bus 001 Device 005: ID 2886:001a Seeed Technology Co., Ltd. reSpeaker XVF3800 4-Mic Array\n",
                        "stderr": "",
                    },
                )()
            if command[:2] == ["/usr/bin/arecord", "-l"]:
                return type(
                    "CompletedProcess",
                    (),
                    {
                        "returncode": 0,
                        "stdout": "card 4: Array [reSpeaker XVF3800 4-Mic Array], device 0: USB Audio [USB Audio]\n",
                        "stderr": "",
                    },
                )()
            raise AssertionError(f"Unexpected command: {command}")

        with patch("twinr.hardware.respeaker.probe.which", side_effect=lambda name: f"/usr/bin/{name}" if name in {"lsusb", "arecord"} else None):
            with patch("twinr.hardware.respeaker.probe.subprocess.run", side_effect=fake_run):
                probe = probe_respeaker_xvf3800()

        self.assertEqual(probe.state, "audio_ready")
        self.assertTrue(probe.usb_visible)
        self.assertTrue(probe.capture_ready)
        self.assertEqual(probe.capture_device.hw_identifier, "hw:CARD=Array,DEV=0")

    def test_probe_detects_usb_visible_without_capture(self) -> None:
        def fake_run(command, *, capture_output: bool, text: bool, check: bool, timeout: float, env, encoding: str, errors: str):
            if command[:3] == ["/usr/bin/lsusb", "-d", "2886:001a"]:
                return type(
                    "CompletedProcess",
                    (),
                    {
                        "returncode": 0,
                        "stdout": "Bus 001 Device 007: ID 2886:001a Seeed Technology Co., Ltd. reSpeaker XVF3800 4-Mic Array\n",
                        "stderr": "",
                    },
                )()
            if command[:2] == ["/usr/bin/arecord", "-l"]:
                return type(
                    "CompletedProcess",
                    (),
                    {
                        "returncode": 0,
                        "stdout": "card 3: Headphones [bcm2835 Headphones], device 0: bcm2835 Headphones [bcm2835 Headphones]\n",
                        "stderr": "",
                    },
                )()
            raise AssertionError(f"Unexpected command: {command}")

        with patch("twinr.hardware.respeaker.probe.which", side_effect=lambda name: f"/usr/bin/{name}" if name in {"lsusb", "arecord"} else None):
            with patch("twinr.hardware.respeaker.probe.subprocess.run", side_effect=fake_run):
                probe = probe_respeaker_xvf3800()

        self.assertEqual(probe.state, "usb_visible_no_capture")
        self.assertTrue(probe.usb_visible)
        self.assertFalse(probe.capture_ready)

    def test_libusb_transport_retries_and_decodes_version(self) -> None:
        class FakeBindings:
            def __init__(self) -> None:
                self.responses = {
                    (VERSION_PARAMETER.resid, VERSION_PARAMETER.request_value): [
                        bytes([64, 0, 0, 0]),
                        bytes([0, 2, 0, 7]),
                    ],
                }

            def init_context(self):
                return object()

            def open_device(self, context, vendor_id, product_id):
                return object()

            def close(self, handle) -> None:
                return None

            def exit(self, context) -> None:
                return None

            def control_transfer(self, handle, *, request_type, request, value, index, buffer, timeout_ms):
                response = self.responses[(index, value)].pop(0)
                for offset, byte in enumerate(response):
                    buffer[offset] = byte
                return len(response)

            def error_name(self, code: int) -> str:
                return f"err_{code}"

        transport = ReSpeakerLibusbTransport(bindings=FakeBindings(), sleep_fn=lambda _: None)
        availability, reads = transport.capture_reads([VERSION_PARAMETER])

        self.assertTrue(availability.available)
        self.assertIn("VERSION", reads)
        self.assertTrue(reads["VERSION"].ok)
        self.assertEqual(reads["VERSION"].attempt_count, 2)
        self.assertEqual(reads["VERSION"].decoded_value, (2, 0, 7))

    def test_libusb_transport_marks_open_failure_as_permission_issue(self) -> None:
        class FakeBindings:
            def init_context(self):
                return object()

            def open_device(self, context, vendor_id, product_id):
                return None

            def close(self, handle) -> None:
                return None

            def exit(self, context) -> None:
                return None

        transport = ReSpeakerLibusbTransport(bindings=FakeBindings())
        probe = ReSpeakerProbeResult(
            usb_device=ReSpeakerUsbDevice(
                bus="001",
                device="005",
                vendor_id="2886",
                product_id="001a",
                description="Seeed Technology Co., Ltd. reSpeaker XVF3800 4-Mic Array",
                raw_line="Bus 001 Device 005: ID 2886:001a Seeed Technology Co., Ltd. reSpeaker XVF3800 4-Mic Array",
            ),
            capture_device=None,
            lsusb_available=True,
            arecord_available=True,
        )

        availability, reads = transport.capture_reads([VERSION_PARAMETER], probe=probe)

        self.assertFalse(availability.available)
        self.assertEqual(availability.reason, "permission_denied_or_transport_blocked")
        self.assertTrue(availability.requires_elevated_permissions)
        self.assertEqual(reads, {})

    def test_snapshot_service_interprets_directional_primitives(self) -> None:
        class FakeTransport:
            def capture_reads(self, specs, *, probe=None):
                return (
                    ReSpeakerTransportAvailability(backend="libusb", available=True),
                    {
                        "VERSION": ReSpeakerParameterRead(
                            spec=VERSION_PARAMETER,
                            captured_at=10.0,
                            ok=True,
                            attempt_count=1,
                            status_code=0,
                            decoded_value=(2, 0, 7),
                        ),
                        "DOA_VALUE": ReSpeakerParameterRead(
                            spec=DOA_VALUE_PARAMETER,
                            captured_at=11.0,
                            ok=True,
                            attempt_count=1,
                            status_code=0,
                            decoded_value=(277, 1),
                        ),
                        "AEC_AZIMUTH_VALUES": ReSpeakerParameterRead(
                            spec=AEC_AZIMUTH_VALUES_PARAMETER,
                            captured_at=12.0,
                            ok=True,
                            attempt_count=1,
                            status_code=0,
                            decoded_value=(0.0, math.pi / 2.0, math.pi, (3.0 * math.pi) / 2.0),
                        ),
                        "AEC_SPENERGY_VALUES": ReSpeakerParameterRead(
                            spec=AEC_SPENERGY_VALUES_PARAMETER,
                            captured_at=13.0,
                            ok=True,
                            attempt_count=1,
                            status_code=0,
                            decoded_value=(0.25, 0.5, 0.0, 0.75),
                        ),
                        "AUDIO_MGR_SELECTED_AZIMUTHS": ReSpeakerParameterRead(
                            spec=AUDIO_MGR_SELECTED_AZIMUTHS_PARAMETER,
                            captured_at=14.0,
                            ok=True,
                            attempt_count=1,
                            status_code=0,
                            decoded_value=(math.pi / 4.0, float("nan")),
                        ),
                        "GPO_READ_VALUES": ReSpeakerParameterRead(
                            spec=GPO_READ_VALUES_PARAMETER,
                            captured_at=15.0,
                            ok=True,
                            attempt_count=1,
                            status_code=0,
                            decoded_value=(0, 0, 0, 1, 0),
                        ),
                    },
                )

        probe = ReSpeakerProbeResult(
            usb_device=ReSpeakerUsbDevice(
                bus="001",
                device="005",
                vendor_id="2886",
                product_id="001a",
                description="Seeed Technology Co., Ltd. reSpeaker XVF3800 4-Mic Array",
                raw_line="Bus 001 Device 005: ID 2886:001a Seeed Technology Co., Ltd. reSpeaker XVF3800 4-Mic Array",
            ),
            capture_device=ReSpeakerCaptureDevice(
                card_index=4,
                card_name="Array",
                card_label="reSpeaker XVF3800 4-Mic Array",
                device_index=0,
                raw_line="card 4: Array [reSpeaker XVF3800 4-Mic Array], device 0: USB Audio [USB Audio]",
            ),
            lsusb_available=True,
            arecord_available=True,
        )

        snapshot = capture_respeaker_primitive_snapshot(transport=FakeTransport(), probe=probe)

        self.assertEqual(snapshot.firmware_version, (2, 0, 7))
        self.assertEqual(snapshot.device_runtime_mode, "audio_ready")
        self.assertEqual(snapshot.direction.doa_degrees, 277)
        self.assertTrue(snapshot.direction.speech_detected)
        self.assertFalse(snapshot.direction.room_quiet)
        self.assertEqual(snapshot.direction.beam_azimuth_degrees, (0.0, 90.0, 180.0, 270.0))
        self.assertEqual(snapshot.direction.beam_speech_energies, (0.25, 0.5, 0.0, 0.75))
        self.assertEqual(snapshot.direction.selected_azimuth_degrees, (45.0, None))
        self.assertEqual(snapshot.mute.gpo_logic_levels, (0, 0, 0, 1, 0))
        self.assertIsNone(snapshot.mute.mute_active)

    def test_snapshot_service_degrades_when_transport_is_unavailable(self) -> None:
        class FakeTransport:
            def capture_reads(self, specs, *, probe=None):
                return (
                    ReSpeakerTransportAvailability(
                        backend="libusb",
                        available=False,
                        reason="permission_denied_or_transport_blocked",
                        requires_elevated_permissions=True,
                    ),
                    {},
                )

        probe = ReSpeakerProbeResult(
            usb_device=ReSpeakerUsbDevice(
                bus="001",
                device="005",
                vendor_id="2886",
                product_id="001a",
                description="Seeed Technology Co., Ltd. reSpeaker XVF3800 4-Mic Array",
                raw_line="Bus 001 Device 005: ID 2886:001a Seeed Technology Co., Ltd. reSpeaker XVF3800 4-Mic Array",
            ),
            capture_device=None,
            lsusb_available=True,
            arecord_available=True,
        )

        snapshot = capture_respeaker_primitive_snapshot(transport=FakeTransport(), probe=probe)

        self.assertFalse(snapshot.transport.available)
        self.assertIsNone(snapshot.firmware_version)
        self.assertIsNone(snapshot.direction.doa_degrees)
        self.assertIsNone(snapshot.direction.speech_detected)
        self.assertIsNone(snapshot.mute.gpo_logic_levels)

    def test_signal_provider_tracks_recent_speech_age(self) -> None:
        probe = ReSpeakerProbeResult(
            usb_device=ReSpeakerUsbDevice(
                bus="001",
                device="005",
                vendor_id="2886",
                product_id="001a",
                description="Seeed Technology Co., Ltd. reSpeaker XVF3800 4-Mic Array",
                raw_line="Bus 001 Device 005: ID 2886:001a Seeed Technology Co., Ltd. reSpeaker XVF3800 4-Mic Array",
            ),
            capture_device=ReSpeakerCaptureDevice(
                card_index=4,
                card_name="Array",
                card_label="reSpeaker XVF3800 4-Mic Array",
                device_index=0,
                raw_line="card 4: Array [reSpeaker XVF3800 4-Mic Array], device 0: USB Audio [USB Audio]",
            ),
            lsusb_available=True,
            arecord_available=True,
        )

        snapshots = iter(
            (
                ReSpeakerPrimitiveSnapshot(
                    captured_at=10.0,
                    probe=probe,
                    transport=ReSpeakerTransportAvailability(backend="libusb", available=True),
                    firmware_version=(2, 0, 7),
                    direction=ReSpeakerDirectionSnapshot(
                        captured_at=10.0,
                        speech_detected=True,
                        room_quiet=False,
                        doa_degrees=277,
                    ),
                    mute=ReSpeakerMuteSnapshot(
                        captured_at=10.0,
                        mute_active=None,
                        gpo_logic_levels=(0, 0, 0, 1, 0),
                    ),
                ),
                ReSpeakerPrimitiveSnapshot(
                    captured_at=11.0,
                    probe=probe,
                    transport=ReSpeakerTransportAvailability(backend="libusb", available=True),
                    firmware_version=(2, 0, 7),
                    direction=ReSpeakerDirectionSnapshot(
                        captured_at=11.0,
                        speech_detected=False,
                        room_quiet=True,
                        doa_degrees=15,
                    ),
                    mute=ReSpeakerMuteSnapshot(
                        captured_at=11.0,
                        mute_active=None,
                        gpo_logic_levels=(0, 0, 0, 1, 0),
                    ),
                ),
            )
        )
        clock_values = iter((100.0, 103.25))
        provider = ReSpeakerSignalProvider(
            snapshot_factory=lambda **_kwargs: next(snapshots),
            monotonic_clock=lambda: next(clock_values),
        )

        first = provider.observe()
        second = provider.observe()

        self.assertEqual(first.recent_speech_age_s, 0.0)
        self.assertEqual(first.azimuth_deg, 277)
        self.assertEqual(second.recent_speech_age_s, 3.25)
        self.assertTrue(second.room_quiet)
        self.assertEqual(second.azimuth_deg, 15)
        self.assertTrue(second.host_control_ready)

    def test_signal_provider_degrades_conservatively_on_snapshot_failure(self) -> None:
        provider = ReSpeakerSignalProvider(
            snapshot_factory=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        snapshot = provider.observe()

        self.assertFalse(snapshot.host_control_ready)
        self.assertEqual(snapshot.device_runtime_mode, "signal_provider_error")
        self.assertEqual(snapshot.transport_reason, "signal_provider_error:RuntimeError")
        self.assertIsNone(snapshot.speech_detected)


if __name__ == "__main__":
    unittest.main()
