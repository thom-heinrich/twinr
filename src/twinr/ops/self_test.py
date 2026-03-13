from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import errno

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.audio import AmbientAudioSampler, SilenceDetectedRecorder, WaveAudioPlayer
from twinr.hardware.buttons import configured_button_monitor
from twinr.hardware.camera import V4L2StillCamera
from twinr.hardware.pir import configured_pir_monitor
from twinr.hardware.printer import RawReceiptPrinter
from twinr.ops.events import TwinrOpsEventStore
from twinr.ops.locks import loop_lock_owner
from twinr.ops.paths import resolve_ops_paths_for_config


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass(frozen=True, slots=True)
class SelfTestResult:
    test_name: str
    status: str
    summary: str
    details: tuple[str, ...] = ()
    artifact_name: str | None = None
    finished_at: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class SelfTestBlockedError(RuntimeError):
    pass


class TwinrSelfTestRunner:
    def __init__(
        self,
        config: TwinrConfig,
        *,
        event_store: TwinrOpsEventStore | None = None,
        recorder_factory=None,
        ambient_sampler_factory=None,
        player_factory=None,
        printer_factory=None,
        camera_factory=None,
        pir_monitor_factory=None,
    ) -> None:
        self.config = config
        self.paths = resolve_ops_paths_for_config(config)
        self.event_store = event_store or TwinrOpsEventStore.from_config(config)
        self.recorder_factory = recorder_factory or self._build_mic_test_recorder
        self.ambient_sampler_factory = ambient_sampler_factory or AmbientAudioSampler.from_config
        self.player_factory = player_factory or WaveAudioPlayer.from_config
        self.printer_factory = printer_factory or RawReceiptPrinter.from_config
        self.camera_factory = camera_factory or V4L2StillCamera.from_config
        self.pir_monitor_factory = pir_monitor_factory or configured_pir_monitor

    @staticmethod
    def available_tests() -> tuple[tuple[str, str, str], ...]:
        return (
            ("mic", "Mic aufnehmen", "Record a short speech sample and store the WAV artifact."),
            (
                "proactive_mic",
                "Proaktives Mikrofon",
                "Sample a short bounded window from the proactive background microphone and report activity levels.",
            ),
            ("speaker", "Speaker-Beep", "Play a local confirmation beep through the configured output."),
            (
                "printer",
                "Printer-Testdruck",
                "Submit a short service ticket to the configured receipt printer and then confirm the paper output on the device.",
            ),
            ("camera", "Kamera-Testbild", "Capture a fresh still image and store it as a PNG artifact."),
            ("buttons", "Button-State", "Read the configured GPIO button levels once."),
            ("pir", "PIR motion", "Wait for a motion trigger on the configured PIR input."),
        )

    def run(self, test_name: str) -> SelfTestResult:
        normalized = (test_name or "").strip().lower()
        self.paths.self_tests_root.mkdir(parents=True, exist_ok=True)
        self.event_store.append(
            event="self_test_started",
            message=f"Self-test `{normalized}` started.",
            data={"test_name": normalized},
        )
        try:
            if normalized == "mic":
                result = self._run_mic_test()
            elif normalized == "proactive_mic":
                result = self._run_proactive_mic_test()
            elif normalized == "speaker":
                result = self._run_speaker_test()
            elif normalized == "printer":
                result = self._run_printer_test()
            elif normalized == "camera":
                result = self._run_camera_test()
            elif normalized == "buttons":
                result = self._run_button_test()
            elif normalized == "pir":
                result = self._run_pir_test()
            else:
                raise ValueError(f"Unsupported self-test: {test_name}")
        except SelfTestBlockedError as exc:
            self.event_store.append(
                event="self_test_blocked",
                level="warn",
                message=f"Self-test `{normalized}` blocked.",
                data={"test_name": normalized, "reason": str(exc)},
            )
            return SelfTestResult(
                test_name=normalized,
                status="blocked",
                summary=str(exc),
                finished_at=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as exc:
            self.event_store.append(
                event="self_test_failed",
                level="error",
                message=f"Self-test `{normalized}` failed.",
                data={"test_name": normalized, "error": str(exc)},
            )
            return SelfTestResult(
                test_name=normalized,
                status="fail",
                summary=str(exc),
                finished_at=datetime.now(timezone.utc).isoformat(),
            )

        self.event_store.append(
            event="self_test_finished",
            message=f"Self-test `{normalized}` finished.",
            data={
                "test_name": normalized,
                "status": result.status,
                "artifact_name": result.artifact_name or "",
            },
        )
        return result

    def _run_mic_test(self) -> SelfTestResult:
        recorder = self.recorder_factory(self.config)
        artifact_name = f"mic-{_utc_stamp()}.wav"
        artifact_path = self.paths.self_tests_root / artifact_name
        audio_bytes = recorder.record_until_pause(pause_ms=min(self.config.speech_pause_ms, 900))
        artifact_path.write_bytes(audio_bytes)
        return SelfTestResult(
            test_name="mic",
            status="ok",
            summary="Speech sample recorded.",
            details=(
                f"Saved WAV recording to {artifact_path}.",
                f"Captured {len(audio_bytes)} bytes.",
            ),
            artifact_name=artifact_name,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )

    def _run_speaker_test(self) -> SelfTestResult:
        player = self.player_factory(self.config)
        player.play_tone(
            frequency_hz=self.config.audio_beep_frequency_hz,
            duration_ms=self.config.audio_beep_duration_ms,
            volume=self.config.audio_beep_volume,
            sample_rate=self.config.openai_realtime_input_sample_rate,
        )
        return SelfTestResult(
            test_name="speaker",
            status="ok",
            summary="Confirmation beep played.",
            details=(f"Output device: {self.config.audio_output_device}",),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )

    def _run_proactive_mic_test(self) -> SelfTestResult:
        if not self.config.proactive_audio_enabled and not (self.config.proactive_audio_input_device or "").strip():
            raise RuntimeError("No proactive background-audio device is configured.")
        sampler = self.ambient_sampler_factory(self.config)
        sample_duration_ms = max(300, min(self.config.proactive_audio_sample_ms, 1500))
        sample = sampler.sample_levels(duration_ms=sample_duration_ms)
        speech_detected = sample.active_chunk_count > 0 and sample.active_ratio >= 0.2
        configured_device = (self.config.proactive_audio_input_device or "").strip()
        device_label = configured_device or self.config.audio_input_device
        reuse_note = ""
        if not configured_device:
            reuse_note = "Proactive path currently reuses the primary input device."
        details = [
            f"Input device: {device_label}",
            f"Sample window: {sample.duration_ms} ms",
            f"Average RMS: {sample.average_rms}",
            f"Peak RMS: {sample.peak_rms}",
            f"Active chunks: {sample.active_chunk_count}/{sample.chunk_count}",
            f"Speech-like activity: {'yes' if speech_detected else 'no'}",
        ]
        if reuse_note:
            details.append(reuse_note)
        return SelfTestResult(
            test_name="proactive_mic",
            status="ok",
            summary="Proactive background-microphone sample captured.",
            details=tuple(details),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )

    def _run_printer_test(self) -> SelfTestResult:
        printer = self.printer_factory(self.config)
        print_job = printer.print_text("Twinr self-test\nPrinter path OK.")
        details = [f"Queue: {self.config.printer_queue}"]
        if print_job:
            details.append(f"CUPS response: {print_job}")
        details.append("Physical paper output must be confirmed at the printer.")
        details.append("If nothing prints, check paper, cabling, and the printer's own power supply.")
        return SelfTestResult(
            test_name="printer",
            status="warn",
            summary="Printer job submitted. Confirm the paper output on the device.",
            details=tuple(details),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )

    def _run_camera_test(self) -> SelfTestResult:
        camera = self.camera_factory(self.config)
        artifact_name = f"camera-{_utc_stamp()}.png"
        artifact_path = self.paths.self_tests_root / artifact_name
        capture = camera.capture_photo(output_path=artifact_path, filename=artifact_name)
        return SelfTestResult(
            test_name="camera",
            status="ok",
            summary="Camera frame captured.",
            details=(
                f"Saved PNG to {artifact_path}.",
                f"Source device: {capture.source_device}",
                f"Input format: {capture.input_format or 'default'}",
                f"Bytes: {len(capture.data)}",
            ),
            artifact_name=artifact_name,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )

    def _run_button_test(self) -> SelfTestResult:
        if not self.config.button_gpios:
            raise RuntimeError("No configured button GPIOs are available for probing.")
        self._ensure_gpio_self_test_available("Button self-test")
        try:
            with configured_button_monitor(self.config) as monitor:
                levels = monitor.snapshot_values()
        except OSError as exc:
            self._raise_gpio_busy("Button self-test", exc)
            raise
        details = tuple(
            f"{name}: GPIO {line_offset} raw={levels.get(line_offset, '?')}"
            for name, line_offset in sorted(self.config.button_gpios.items())
        )
        return SelfTestResult(
            test_name="buttons",
            status="ok",
            summary="GPIO button levels read successfully.",
            details=details,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )

    def _run_pir_test(self) -> SelfTestResult:
        if not self.config.pir_enabled:
            raise RuntimeError("No PIR motion GPIO is configured.")
        self._ensure_gpio_self_test_available("PIR self-test")
        try:
            with self.pir_monitor_factory(self.config) as monitor:
                current_value = monitor.snapshot_value()
                event = monitor.wait_for_motion(duration_s=12.0, poll_timeout=0.2)
        except OSError as exc:
            self._raise_gpio_busy("PIR self-test", exc)
            raise
        if event is None:
            raise RuntimeError("No PIR motion event detected within 12 seconds.")
        trigger_kind = "current high level" if event.raw_edge == "level" else f"{event.raw_edge} edge"
        return SelfTestResult(
            test_name="pir",
            status="ok",
            summary="PIR motion detected.",
            details=(
                f"GPIO: {self.config.pir_motion_gpio}",
                f"Bias: {self.config.pir_bias}",
                f"Active high: {str(self.config.pir_active_high).lower()}",
                f"Initial value: {current_value}",
                f"Trigger: {trigger_kind}",
            ),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )

    def _ensure_gpio_self_test_available(self, label: str) -> None:
        active_loops: list[str] = []
        for loop_name, loop_label in (("realtime-loop", "realtime loop"), ("hardware-loop", "hardware loop")):
            owner = loop_lock_owner(self.config, loop_name)
            if owner is None:
                continue
            active_loops.append(f"{loop_label} (pid {owner})")
        if not active_loops:
            return
        joined = ", ".join(active_loops)
        raise SelfTestBlockedError(
            f"{label} is unavailable while the Twinr {joined} is running. Stop it first."
        )

    def _raise_gpio_busy(self, label: str, exc: OSError) -> None:
        if exc.errno != errno.EBUSY:
            return
        raise SelfTestBlockedError(
            f"{label} could not access the configured GPIO line because it is busy. Stop the active Twinr loop or other GPIO consumer and try again."
        ) from exc

    def _build_mic_test_recorder(self, config: TwinrConfig) -> SilenceDetectedRecorder:
        return SilenceDetectedRecorder(
            device=config.audio_input_device,
            sample_rate=config.audio_sample_rate,
            channels=config.audio_channels,
            chunk_ms=config.audio_chunk_ms,
            preroll_ms=config.audio_preroll_ms,
            speech_threshold=config.audio_speech_threshold,
            speech_start_chunks=1,
            start_timeout_s=min(config.audio_start_timeout_s, 4.0),
            max_record_seconds=min(config.audio_max_record_seconds, 6.0),
        )
