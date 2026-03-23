from pathlib import Path
import math
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.respeaker import (
    AEC_AZIMUTH_VALUES_PARAMETER,
    AEC_SPENERGY_VALUES_PARAMETER,
    AUDIO_MGR_SELECTED_AZIMUTHS_PARAMETER,
    DOA_VALUE_PARAMETER,
    LED_COLOR_PARAMETER,
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
    derive_respeaker_signal_state,
    probe_respeaker_xvf3800,
)
from twinr.hardware.audio import AmbientAudioLevelSample
from twinr.hardware.respeaker.ambient_classification import classify_respeaker_ambient_audio
from twinr.hardware.respeaker.pcm_content_classifier import classify_pcm_speech_likeness
from twinr.hardware.respeaker.models import (
    ReSpeakerCaptureDevice,
    ReSpeakerDirectionSnapshot,
    ReSpeakerMuteSnapshot,
    ReSpeakerPrimitiveSnapshot,
)


def _encode_pcm16(samples: np.ndarray) -> bytes:
    """Return one clipped PCM16 payload from normalized float samples."""

    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def _speech_like_pcm_bytes(*, sample_rate: int = 16_000) -> bytes:
    """Return one bounded speech-like harmonic burst pattern."""

    rng = np.random.default_rng(7)
    duration_s = 0.96
    t = np.arange(int(sample_rate * duration_s), dtype=np.float32) / float(sample_rate)
    fundamental_hz = 120.0 + 30.0 * np.sin(2.0 * math.pi * 0.5 * t)
    phase = np.cumsum((2.0 * math.pi * fundamental_hz) / float(sample_rate))
    voiced = (
        np.sin(phase)
        + 0.50 * np.sin(2.0 * phase)
        + 0.25 * np.sin(3.0 * phase)
        + 0.10 * np.sin(4.0 * phase)
    )
    syllable_mask = (np.sin(2.0 * math.pi * 4.3 * t) > -0.2).astype(np.float32)
    smoothing = np.hanning(401).astype(np.float32)
    smoothing /= float(np.sum(smoothing))
    envelope = np.convolve(syllable_mask, smoothing, mode="same")
    envelope /= float(np.max(envelope) + 1e-6)
    samples = (0.65 * voiced * envelope) + (0.01 * rng.normal(size=t.shape[0]))
    return _encode_pcm16(samples)


def _music_like_pcm_bytes(*, sample_rate: int = 16_000) -> bytes:
    """Return one bounded steady tonal media-like pattern."""

    duration_s = 0.64
    t = np.arange(int(sample_rate * duration_s), dtype=np.float32) / float(sample_rate)
    tremolo = 0.78 + (0.18 * np.sin(2.0 * math.pi * 2.0 * t))
    chord = (
        0.38 * np.sin(2.0 * math.pi * 220.0 * t)
        + 0.26 * np.sin(2.0 * math.pi * 330.0 * t)
        + 0.18 * np.sin(2.0 * math.pi * 440.0 * t)
    )
    return _encode_pcm16(chord * tremolo)


def _noise_like_pcm_bytes(*, sample_rate: int = 16_000) -> bytes:
    """Return one bounded colored-noise pattern."""

    rng = np.random.default_rng(11)
    duration_s = 0.64
    sample_count = int(sample_rate * duration_s)
    noise = rng.normal(size=sample_count).astype(np.float32)
    kernel = np.ones(48, dtype=np.float32) / 48.0
    colored = np.convolve(noise, kernel, mode="same")
    colored /= float(np.max(np.abs(colored)) + 1e-6)
    return _encode_pcm16(0.55 * colored)


class ReSpeakerProbeTests(unittest.TestCase):
    def test_pcm_speech_likeness_ranks_speech_above_music_and_noise(self) -> None:
        speech = classify_pcm_speech_likeness(
            _speech_like_pcm_bytes(),
            sample_rate=16_000,
            channels=1,
        )
        music = classify_pcm_speech_likeness(
            _music_like_pcm_bytes(),
            sample_rate=16_000,
            channels=1,
        )
        noise = classify_pcm_speech_likeness(
            _noise_like_pcm_bytes(),
            sample_rate=16_000,
            channels=1,
        )

        assert speech.speech_probability is not None
        assert music.speech_probability is not None
        assert noise.speech_probability is not None
        self.assertGreater(speech.speech_probability, 0.50)
        self.assertLess(music.speech_probability, 0.30)
        self.assertLess(noise.speech_probability, 0.30)
        self.assertGreater(speech.speech_probability, music.speech_probability)
        self.assertGreater(speech.speech_probability, noise.speech_probability)
        self.assertTrue(music.strong_non_speech)
        self.assertTrue(noise.strong_non_speech)
        self.assertTrue(speech.speech_likely)

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

    def test_libusb_transport_writes_led_color_payload(self) -> None:
        class FakeBindings:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def init_context(self):
                return object()

            def open_device(self, context, vendor_id, product_id):
                return object()

            def close(self, handle) -> None:
                return None

            def exit(self, context) -> None:
                return None

            def control_transfer(self, handle, *, request_type, request, value, index, buffer, timeout_ms):
                self.calls.append(
                    {
                        "request_type": request_type,
                        "request": request,
                        "value": value,
                        "index": index,
                        "payload": bytes(buffer),
                        "timeout_ms": timeout_ms,
                    }
                )
                return len(buffer)

            def error_name(self, code: int) -> str:
                return f"err_{code}"

        bindings = FakeBindings()
        transport = ReSpeakerLibusbTransport(bindings=bindings)

        availability = transport.write_parameter(LED_COLOR_PARAMETER, (0xFFAA33,))

        self.assertTrue(availability.available)
        self.assertEqual(len(bindings.calls), 1)
        self.assertEqual(bindings.calls[0]["request_type"], 0x40)
        self.assertEqual(bindings.calls[0]["value"], LED_COLOR_PARAMETER.cmdid)
        self.assertEqual(bindings.calls[0]["index"], LED_COLOR_PARAMETER.resid)
        self.assertEqual(bindings.calls[0]["payload"], b"\x33\xaa\xff\x00")

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
        self.assertEqual(
            snapshot.raw_reads["AUDIO_MGR_SELECTED_AZIMUTHS"].decoded_value,
            (math.pi / 4.0, None),
        )
        self.assertEqual(snapshot.mute.gpo_logic_levels, (0, 0, 0, 1, 0))
        self.assertFalse(snapshot.mute.mute_active)

    def test_derive_signal_state_uses_fixed_beam_speech_for_confidence_overlap_and_barge_in(self) -> None:
        direction = ReSpeakerDirectionSnapshot(
            captured_at=10.0,
            speech_detected=True,
            room_quiet=False,
            doa_degrees=88,
            beam_azimuth_degrees=(90.0, 270.0, 180.0, 90.0),
            beam_speech_energies=(0.5, 0.25, 0.0, 0.5),
            selected_azimuth_degrees=(92.0, 90.0),
        )

        derived = derive_respeaker_signal_state(direction, assistant_output_active=True)

        self.assertEqual(derived.fixed_beam_speech_count, 2)
        self.assertTrue(derived.near_end_speech_detected)
        self.assertAlmostEqual(derived.direction_confidence, 0.483, places=3)
        self.assertTrue(derived.speech_overlap_likely)
        self.assertTrue(derived.barge_in_detected)

    def test_derive_signal_state_uses_first_valid_selected_azimuth_slot(self) -> None:
        direction = ReSpeakerDirectionSnapshot(
            captured_at=10.0,
            speech_detected=True,
            room_quiet=False,
            doa_degrees=88,
            beam_azimuth_degrees=(90.0, 270.0, 180.0, 90.0),
            beam_speech_energies=(0.5, 0.0, 0.0, 0.5),
            selected_azimuth_degrees=(None, 92.0),
        )

        derived = derive_respeaker_signal_state(direction, assistant_output_active=False)

        self.assertIsNotNone(derived.direction_confidence)
        assert derived.direction_confidence is not None
        self.assertGreater(derived.direction_confidence, 0.95)

    def test_classify_respeaker_ambient_audio_marks_background_media_without_speech(self) -> None:
        classification = classify_respeaker_ambient_audio(
            signal_snapshot=type(
                "SignalSnapshot",
                (),
                {
                    "assistant_output_active": False,
                    "host_control_ready": True,
                    "speech_detected": False,
                },
            )(),
            sample=AmbientAudioLevelSample(
                duration_ms=1000,
                chunk_count=10,
                active_chunk_count=8,
                average_rms=1800,
                peak_rms=3200,
                active_ratio=0.8,
            ),
        )

        self.assertTrue(classification.audio_activity_detected)
        self.assertTrue(classification.non_speech_audio_likely)
        self.assertTrue(classification.background_media_likely)

    def test_classify_respeaker_ambient_audio_treats_uncorroborated_speech_as_non_speech(self) -> None:
        classification = classify_respeaker_ambient_audio(
            signal_snapshot=type(
                "SignalSnapshot",
                (),
                {
                    "assistant_output_active": False,
                    "host_control_ready": True,
                    "speech_detected": True,
                    "direction_confidence": None,
                    "beam_activity": (0.0, 0.0, 0.0, 0.0),
                },
            )(),
            sample=AmbientAudioLevelSample(
                duration_ms=1000,
                chunk_count=10,
                active_chunk_count=8,
                average_rms=1800,
                peak_rms=3200,
                active_ratio=0.8,
            ),
        )

        self.assertTrue(classification.audio_activity_detected)
        self.assertTrue(classification.non_speech_audio_likely)
        self.assertTrue(classification.background_media_likely)

    def test_classify_respeaker_ambient_audio_uses_rms_fallback_for_low_level_music(self) -> None:
        classification = classify_respeaker_ambient_audio(
            signal_snapshot=type(
                "SignalSnapshot",
                (),
                {
                    "assistant_output_active": False,
                    "host_control_ready": True,
                    "speech_detected": True,
                    "direction_confidence": None,
                    "beam_activity": (0.0, 0.0, 0.0, 0.0),
                },
            )(),
            sample=AmbientAudioLevelSample(
                duration_ms=1000,
                chunk_count=10,
                active_chunk_count=0,
                average_rms=400,
                peak_rms=550,
                active_ratio=0.0,
            ),
        )

        self.assertTrue(classification.audio_activity_detected)
        self.assertTrue(classification.non_speech_audio_likely)
        self.assertTrue(classification.background_media_likely)

    def test_classify_respeaker_ambient_audio_keeps_low_rms_quiet_unknown(self) -> None:
        classification = classify_respeaker_ambient_audio(
            signal_snapshot=type(
                "SignalSnapshot",
                (),
                {
                    "assistant_output_active": False,
                    "host_control_ready": True,
                    "speech_detected": True,
                    "direction_confidence": None,
                    "beam_activity": (0.0, 0.0, 0.0, 0.0),
                },
            )(),
            sample=AmbientAudioLevelSample(
                duration_ms=1000,
                chunk_count=10,
                active_chunk_count=0,
                average_rms=180,
                peak_rms=260,
                active_ratio=0.0,
            ),
        )

        self.assertFalse(classification.audio_activity_detected)
        self.assertFalse(classification.non_speech_audio_likely)
        self.assertFalse(classification.background_media_likely)

    def test_classify_respeaker_ambient_audio_keeps_corroborated_speech_suppressed(self) -> None:
        classification = classify_respeaker_ambient_audio(
            signal_snapshot=type(
                "SignalSnapshot",
                (),
                {
                    "assistant_output_active": False,
                    "host_control_ready": True,
                    "speech_detected": True,
                    "direction_confidence": 0.91,
                    "beam_activity": (1200.0, 0.0, 0.0, 1200.0),
                },
            )(),
            sample=AmbientAudioLevelSample(
                duration_ms=1000,
                chunk_count=10,
                active_chunk_count=8,
                average_rms=1800,
                peak_rms=3200,
                active_ratio=0.8,
            ),
        )

        self.assertTrue(classification.audio_activity_detected)
        self.assertFalse(classification.non_speech_audio_likely)
        self.assertFalse(classification.background_media_likely)

    def test_classify_respeaker_ambient_audio_rejects_low_confidence_overlap_as_speech(self) -> None:
        classification = classify_respeaker_ambient_audio(
            signal_snapshot=type(
                "SignalSnapshot",
                (),
                {
                    "assistant_output_active": False,
                    "host_control_ready": True,
                    "speech_detected": True,
                    "speech_overlap_likely": True,
                    "direction_confidence": 0.49,
                    "beam_activity": (2480953.5, 405761.25, 1759086.375, 2480953.5),
                },
            )(),
            sample=AmbientAudioLevelSample(
                duration_ms=1000,
                chunk_count=10,
                active_chunk_count=4,
                average_rms=758,
                peak_rms=2186,
                active_ratio=0.4,
            ),
        )

        self.assertTrue(classification.audio_activity_detected)
        self.assertTrue(classification.non_speech_audio_likely)
        self.assertFalse(classification.background_media_likely)

    def test_classify_respeaker_ambient_audio_pcm_vetoes_host_control_false_speech(self) -> None:
        classification = classify_respeaker_ambient_audio(
            signal_snapshot=type(
                "SignalSnapshot",
                (),
                {
                    "assistant_output_active": False,
                    "host_control_ready": True,
                    "speech_detected": True,
                    "direction_confidence": 0.98,
                    "beam_activity": (1200.0, 0.0, 0.0, 0.0),
                },
            )(),
            sample=AmbientAudioLevelSample(
                duration_ms=1000,
                chunk_count=10,
                active_chunk_count=8,
                average_rms=1800,
                peak_rms=3200,
                active_ratio=0.8,
            ),
            pcm_bytes=_music_like_pcm_bytes(),
            sample_rate=16_000,
            channels=1,
        )

        self.assertTrue(classification.audio_activity_detected)
        self.assertTrue(classification.non_speech_audio_likely)
        self.assertTrue(classification.background_media_likely)

    def test_classify_respeaker_ambient_audio_keeps_pcm_backed_speech(self) -> None:
        classification = classify_respeaker_ambient_audio(
            signal_snapshot=type(
                "SignalSnapshot",
                (),
                {
                    "assistant_output_active": False,
                    "host_control_ready": True,
                    "speech_detected": True,
                    "direction_confidence": None,
                    "beam_activity": (0.0, 0.0, 0.0, 0.0),
                },
            )(),
            sample=AmbientAudioLevelSample(
                duration_ms=1000,
                chunk_count=10,
                active_chunk_count=8,
                average_rms=1800,
                peak_rms=3200,
                active_ratio=0.8,
            ),
            pcm_bytes=_speech_like_pcm_bytes(),
            sample_rate=16_000,
            channels=1,
        )

        self.assertTrue(classification.audio_activity_detected)
        self.assertFalse(classification.non_speech_audio_likely)
        self.assertFalse(classification.background_media_likely)

    def test_classify_respeaker_ambient_audio_pcm_non_speech_can_raise_activity_without_sample_gate(self) -> None:
        classification = classify_respeaker_ambient_audio(
            signal_snapshot=type(
                "SignalSnapshot",
                (),
                {
                    "assistant_output_active": False,
                    "host_control_ready": True,
                    "speech_detected": True,
                    "direction_confidence": 0.97,
                    "beam_activity": (1200.0, 0.0, 0.0, 0.0),
                },
            )(),
            sample=AmbientAudioLevelSample(
                duration_ms=1000,
                chunk_count=10,
                active_chunk_count=0,
                average_rms=120,
                peak_rms=180,
                active_ratio=0.0,
            ),
            pcm_bytes=_music_like_pcm_bytes(),
            sample_rate=16_000,
            channels=1,
        )

        self.assertTrue(classification.audio_activity_detected)
        self.assertTrue(classification.non_speech_audio_likely)
        self.assertFalse(classification.background_media_likely)

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
        self.assertIn("speech_detected", first.claim_contract)
        self.assertEqual(first.claim_contract["speech_detected"].source, "respeaker_xvf3800")
        self.assertGreater(first.claim_contract["speech_detected"].confidence, 0.7)
        self.assertIn("azimuth_deg", first.claim_contract)
        self.assertEqual(second.recent_speech_age_s, 3.25)
        self.assertTrue(second.room_quiet)
        self.assertEqual(second.azimuth_deg, 15)
        self.assertTrue(second.host_control_ready)

    def test_signal_provider_requires_fixed_beam_speech_for_busy_state_barge_in(self) -> None:
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
                        doa_degrees=270,
                        beam_azimuth_degrees=(90.0, 270.0, 180.0, 148.0),
                        beam_speech_energies=(0.0, 18_710_022.0, 9_360_130.0, 18_710_022.0),
                        selected_azimuth_degrees=(148.0, 149.0),
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
                        speech_detected=True,
                        room_quiet=False,
                        doa_degrees=92,
                        beam_azimuth_degrees=(90.0, 270.0, 180.0, 92.0),
                        beam_speech_energies=(0.7, 0.65, 0.0, 0.7),
                        selected_azimuth_degrees=(91.0, 92.0),
                    ),
                    mute=ReSpeakerMuteSnapshot(
                        captured_at=11.0,
                        mute_active=None,
                        gpo_logic_levels=(0, 0, 0, 1, 0),
                    ),
                ),
            )
        )
        clock_values = iter((100.0, 101.5))
        provider = ReSpeakerSignalProvider(
            snapshot_factory=lambda **_kwargs: next(snapshots),
            monotonic_clock=lambda: next(clock_values),
            assistant_output_active_predicate=lambda: True,
        )

        playback_only = provider.observe()
        interruption = provider.observe()

        self.assertIsNone(playback_only.recent_speech_age_s)
        self.assertGreater(playback_only.direction_confidence, 0.49)
        self.assertFalse(playback_only.speech_overlap_likely)
        self.assertFalse(playback_only.barge_in_detected)
        self.assertEqual(interruption.recent_speech_age_s, 0.0)
        self.assertGreater(interruption.direction_confidence, 0.45)
        self.assertTrue(interruption.speech_overlap_likely)
        self.assertTrue(interruption.barge_in_detected)

    def test_signal_provider_degrades_conservatively_on_snapshot_failure(self) -> None:
        provider = ReSpeakerSignalProvider(
            snapshot_factory=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        snapshot = provider.observe()

        self.assertFalse(snapshot.host_control_ready)
        self.assertEqual(snapshot.device_runtime_mode, "signal_provider_error")
        self.assertEqual(snapshot.transport_reason, "signal_provider_error")
        self.assertNotIn("RuntimeError", snapshot.transport_reason)
        self.assertIsNone(snapshot.speech_detected)
        self.assertEqual(dict(snapshot.claim_contract), {})


if __name__ == "__main__":
    unittest.main()
