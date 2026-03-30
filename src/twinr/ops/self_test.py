# CHANGELOG: 2026-03-30
# BUG-1: Added a process-wide file lock so parallel self-tests from different processes cannot race on the same Pi hardware.
# BUG-2: Fixed camera artifact handling so file-backed captures are accepted, image format is sniffed correctly, and AI-Deck JPEG frames are no longer mislabeled as PNG.
# BUG-3: Fixed drone self-test safety so a queued mission must be cancelled successfully; otherwise the self-test fails instead of reporting a false green.
# SEC-1: Hardened artifact storage with symlink checks, private 0700/0600 permissions, bounded retention, and free-space guards for sensitive mic/camera evidence.
# SEC-2: Tightened AI-Deck URI validation and added a bounded TCP preflight so malformed or hostile endpoint strings do not flow unchecked into the camera path.
# IMP-1: Added short-lived artifact pruning and storage quotas, matching the module contract and reducing SD-card exhaustion on Raspberry Pi 4 deployments.
# IMP-2: Upgraded self-tests to emit richer validation data (WAV metadata, image format/dimensions, elapsed metrics) while remaining externally drop-in.
"""Run bounded self-tests for Twinr peripherals and operator workflows.

This module executes one-at-a-time diagnostics for microphone, speaker,
printer, camera, buttons, and PIR hardware and stores short-lived artifacts
when a test needs evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import errno
import fcntl
import io
import os
from pathlib import Path
import shutil
import socket
import struct
import tempfile
import threading
import time
from typing import Any, Callable
from urllib.parse import urlsplit
from uuid import uuid4
import wave

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.audio import AmbientAudioSampler, SilenceDetectedRecorder, WaveAudioPlayer
from twinr.hardware.buttons import configured_button_monitor
from twinr.hardware.camera import V4L2StillCamera
from twinr.hardware.drone_service import DroneServiceConfig, RemoteDroneServiceClient
from twinr.hardware.pir import configured_pir_monitor
from twinr.hardware.printer import RawReceiptPrinter
from twinr.ops.events import TwinrOpsEventStore
from twinr.ops.locks import loop_lock_owner
from twinr.ops.paths import resolve_ops_paths_for_config
from twinr.proactive.runtime.service_impl.compat import (
    _proactive_pcm_capture_conflicts_with_voice_orchestrator,
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_AIDECK_DEVICE_SCHEME = "aideck://"
_AIDECK_DEFAULT_PORT = 5000
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SIGNATURE = b"\xff\xd8"
_DIR_MODE = 0o700
_FILE_MODE = 0o600
_LOCKFILE_NAME = ".twinr-self-tests.lock"
_MIN_STORAGE_FREE_BYTES_DEFAULT = 64 * 1024 * 1024
_MAX_STORAGE_BYTES_DEFAULT = 256 * 1024 * 1024
_MAX_STORAGE_FILES_DEFAULT = 32
_ARTIFACT_TTL_SECONDS_DEFAULT = 24 * 60 * 60
_DEFAULT_TCP_PROBE_TIMEOUT_S = 1.5
_MIN_IMAGE_BYTES_DEFAULT = 1024
_MIN_IMAGE_DIMENSION_DEFAULT = 32
_MIN_WAV_DURATION_MS_DEFAULT = 250


@dataclass(frozen=True, slots=True)
class SelfTestResult:
    """Represent the outcome of one bounded Twinr self-test."""

    test_name: str
    status: str
    summary: str
    details: tuple[str, ...] = ()
    artifact_name: str | None = None
    finished_at: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class SelfTestBlockedError(RuntimeError):
    """Raised when a self-test cannot run because the hardware path is busy."""

    pass


class TwinrSelfTestRunner:
    """Run Twinr hardware self-tests with bounded concurrency and artifacts."""

    _RUN_LOCK = threading.Lock()

    def __init__(
        self,
        config: TwinrConfig,
        *,
        event_store: TwinrOpsEventStore | None = None,
        recorder_factory: Callable[[TwinrConfig], Any] | None = None,
        ambient_sampler_factory: Callable[[TwinrConfig], Any] | None = None,
        player_factory: Callable[[TwinrConfig], Any] | None = None,
        printer_factory: Callable[[TwinrConfig], Any] | None = None,
        camera_factory: Callable[[TwinrConfig], Any] | None = None,
        button_monitor_factory: Callable[[TwinrConfig], Any] | None = None,
        pir_monitor_factory: Callable[[TwinrConfig], Any] | None = None,
    ) -> None:
        self.config = config
        self.paths = resolve_ops_paths_for_config(config)
        self.event_store = event_store or TwinrOpsEventStore.from_config(config)
        self.recorder_factory = recorder_factory or self._build_mic_test_recorder
        self.ambient_sampler_factory = ambient_sampler_factory or AmbientAudioSampler.from_config
        self.player_factory = player_factory or WaveAudioPlayer.from_config
        self.printer_factory = printer_factory or RawReceiptPrinter.from_config
        self.camera_factory = camera_factory or V4L2StillCamera.from_config
        self.button_monitor_factory = button_monitor_factory or configured_button_monitor
        self.pir_monitor_factory = pir_monitor_factory or configured_pir_monitor

    @staticmethod
    def available_tests() -> tuple[tuple[str, str, str], ...]:
        """Return the supported self-test identifiers and operator labels."""

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
            ("camera", "Kamera-Testbild", "Capture a fresh still image and store a viewable image artifact."),
            (
                "aideck_camera",
                "AI-Deck Kamera",
                "Probe the AI-Deck stream endpoint, capture one frame, and verify the image artifact looks sane.",
            ),
            (
                "drone_stack",
                "Drohnen-Stack",
                "Reach the bounded drone daemon, queue one inspect mission, verify manual-arm gating, and cancel it safely.",
            ),
            ("buttons", "Button-State", "Read the configured GPIO button levels once."),
            ("pir", "PIR motion", "Wait for a motion trigger on the configured PIR input."),
        )

    def run(self, test_name: str) -> SelfTestResult:
        """Execute one supported self-test and return its normalized result.

        Args:
            test_name: Requested self-test identifier.

        Returns:
            A ``SelfTestResult`` with ``ok``, ``warn``, ``fail``, or ``blocked`` status.
        """

        normalized = self._normalize_test_name(test_name)
        available_tests = {name for name, _, _ in self.available_tests()}
        if normalized not in available_tests:
            summary = (
                f"Unsupported self-test `{normalized or '<empty>'}`. "
                f"Available tests: {', '.join(sorted(available_tests))}."
            )
            self._append_event_safely(
                event="self_test_failed",
                level="error",
                message="Self-test request rejected.",
                data={"test_name": normalized, "error": summary},
            )
            return SelfTestResult(
                test_name=normalized,
                status="fail",
                summary=summary,
                finished_at=_utc_now_iso(),
            )

        if not self.__class__._RUN_LOCK.acquire(blocking=False):
            summary = "Another self-test is already running in this service process. Wait for it to finish and try again."
            self._append_event_safely(
                event="self_test_blocked",
                level="warn",
                message=f"Self-test `{normalized}` blocked.",
                data={"test_name": normalized, "reason": summary},
            )
            return SelfTestResult(
                test_name=normalized,
                status="blocked",
                summary=summary,
                finished_at=_utc_now_iso(),
            )

        lock_handle: io.BufferedRandom | None = None
        started_monotonic = time.monotonic()
        try:
            try:
                self._ensure_self_tests_root()
                lock_handle = self._acquire_process_run_lock()
                self._prune_artifacts_best_effort()
                self._append_event_safely(
                    event="self_test_started",
                    message=f"Self-test `{normalized}` started.",
                    data={"test_name": normalized},
                )
                result = self._dispatch_test(normalized)
            except SelfTestBlockedError as exc:
                reason = self._describe_exception(exc)
                self._append_event_safely(
                    event="self_test_blocked",
                    level="warn",
                    message=f"Self-test `{normalized}` blocked.",
                    data={"test_name": normalized, "reason": reason},
                )
                return SelfTestResult(
                    test_name=normalized,
                    status="blocked",
                    summary=reason,
                    finished_at=_utc_now_iso(),
                )
            except Exception as exc:
                reason = self._describe_exception(exc)
                self._append_event_safely(
                    event="self_test_failed",
                    level="error",
                    message=f"Self-test `{normalized}` failed.",
                    data={"test_name": normalized, "error": reason},
                )
                return SelfTestResult(
                    test_name=normalized,
                    status="fail",
                    summary=reason,
                    finished_at=_utc_now_iso(),
                )

            elapsed_ms = int((time.monotonic() - started_monotonic) * 1000)
            self._append_event_safely(
                event="self_test_finished",
                message=f"Self-test `{normalized}` finished.",
                data={
                    "test_name": normalized,
                    "status": result.status,
                    "artifact_name": result.artifact_name or "",
                    "elapsed_ms": elapsed_ms,
                },
            )
            if elapsed_ms > 0:
                result = SelfTestResult(
                    test_name=result.test_name,
                    status=result.status,
                    summary=result.summary,
                    details=result.details + (f"Elapsed: {elapsed_ms} ms",),
                    artifact_name=result.artifact_name,
                    finished_at=result.finished_at,
                )
            return result
        finally:
            self._release_process_run_lock(lock_handle)
            self.__class__._RUN_LOCK.release()

    def _dispatch_test(self, normalized: str) -> SelfTestResult:
        if normalized == "mic":
            return self._run_mic_test()
        if normalized == "proactive_mic":
            return self._run_proactive_mic_test()
        if normalized == "speaker":
            return self._run_speaker_test()
        if normalized == "printer":
            return self._run_printer_test()
        if normalized == "camera":
            return self._run_camera_test()
        if normalized == "aideck_camera":
            return self._run_aideck_camera_test()
        if normalized == "drone_stack":
            return self._run_drone_stack_test()
        if normalized == "buttons":
            return self._run_button_test()
        return self._run_pir_test()

    def _run_mic_test(self) -> SelfTestResult:
        recorder = self.recorder_factory(self.config)
        try:
            pause_ms = self._bounded_int(
                getattr(self.config, "self_test_mic_pause_ms", self.config.speech_pause_ms),
                minimum=300,
                maximum=2500,
                default=1200,
            )
            try:
                audio_bytes = recorder.record_until_pause(pause_ms=pause_ms)
            except OSError as exc:
                self._raise_device_busy("Microphone self-test", exc)
                raise
        finally:
            self._close_quietly(recorder)

        payload = bytes(audio_bytes)
        wav_meta = self._wav_metadata(payload)
        min_duration_ms = self._bounded_int(
            getattr(self.config, "self_test_min_wav_duration_ms", _MIN_WAV_DURATION_MS_DEFAULT),
            minimum=100,
            maximum=5000,
            default=_MIN_WAV_DURATION_MS_DEFAULT,
        )
        if wav_meta is None:
            raise RuntimeError("Microphone self-test did not return a valid WAV payload.")
        if wav_meta["frame_count"] <= 0 or wav_meta["duration_ms"] < min_duration_ms:
            raise RuntimeError(
                f"Microphone self-test captured too little usable audio ({wav_meta['duration_ms']} ms)."
            )

        self._prepare_artifact_storage(estimated_bytes=len(payload))
        artifact_name = self._make_artifact_name("mic", ".wav")
        self._write_artifact_bytes(artifact_name, payload)
        return SelfTestResult(
            test_name="mic",
            status="ok",
            summary="Speech sample recorded.",
            details=(
                f"Saved WAV recording as {artifact_name}.",
                f"Duration: {wav_meta['duration_ms']} ms",
                f"Sample rate: {wav_meta['sample_rate']} Hz",
                f"Channels: {wav_meta['channels']}",
                f"Sample width: {wav_meta['sample_width_bytes']} bytes",
                f"Bytes: {len(payload)}",
                "Stored in the self-test artifact directory.",
            ),
            artifact_name=artifact_name,
            finished_at=_utc_now_iso(),
        )

    def _run_speaker_test(self) -> SelfTestResult:
        player = self.player_factory(self.config)
        speaker_sample_rate = int(
            getattr(self.config, "audio_output_sample_rate", self.config.openai_realtime_input_sample_rate)
        )
        try:
            try:
                player.play_tone(
                    frequency_hz=self.config.audio_beep_frequency_hz,
                    duration_ms=self.config.audio_beep_duration_ms,
                    volume=self.config.audio_beep_volume,
                    sample_rate=speaker_sample_rate,
                )
            except OSError as exc:
                self._raise_device_busy("Speaker self-test", exc)
                raise
        finally:
            self._close_quietly(player)
        return SelfTestResult(
            test_name="speaker",
            status="ok",
            summary="Confirmation beep played.",
            details=(
                f"Output device: {self.config.audio_output_device}",
                f"Frequency: {self.config.audio_beep_frequency_hz} Hz",
                f"Duration: {self.config.audio_beep_duration_ms} ms",
                f"Volume: {self.config.audio_beep_volume}",
                f"Sample rate: {speaker_sample_rate} Hz",
            ),
            finished_at=_utc_now_iso(),
        )

    def _run_proactive_mic_test(self) -> SelfTestResult:
        if not self.config.proactive_audio_enabled and not (self.config.proactive_audio_input_device or "").strip():
            raise RuntimeError("No proactive background-audio device is configured.")
        if _proactive_pcm_capture_conflicts_with_voice_orchestrator(
            self.config,
            require_active_owner=True,
        ):
            raise SelfTestBlockedError(
                "Proactive microphone self-test is unavailable while the voice orchestrator owns the same capture device. Stop the active streaming runtime first."
            )

        sampler = self.ambient_sampler_factory(self.config)
        try:
            sample_duration_ms = self._bounded_int(
                getattr(self.config, "self_test_proactive_audio_sample_ms", self.config.proactive_audio_sample_ms),
                minimum=300,
                maximum=3000,
                default=1000,
            )
            try:
                sample = sampler.sample_levels(duration_ms=sample_duration_ms)
            except OSError as exc:
                self._raise_device_busy("Proactive microphone self-test", exc)
                raise
        finally:
            self._close_quietly(sampler)

        if getattr(sample, "chunk_count", 0) <= 0 or getattr(sample, "duration_ms", 0) <= 0:
            raise RuntimeError("Proactive microphone self-test captured no sample frames.")

        active_ratio = float(getattr(sample, "active_ratio", 0.0) or 0.0)
        speech_detected = getattr(sample, "active_chunk_count", 0) > 0 and active_ratio >= 0.2
        configured_device = (self.config.proactive_audio_input_device or "").strip()
        device_label = configured_device or (self.config.audio_input_device or "default")
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
            finished_at=_utc_now_iso(),
        )

    def _run_printer_test(self) -> SelfTestResult:
        printer = self.printer_factory(self.config)
        try:
            try:
                print_job = printer.print_text("Twinr self-test\nPrinter path OK.")
            except OSError as exc:
                self._raise_device_busy("Printer self-test", exc)
                raise
        finally:
            self._close_quietly(printer)

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
            finished_at=_utc_now_iso(),
        )

    def _run_camera_test(self) -> SelfTestResult:
        camera = self.camera_factory(self.config)
        try:
            try:
                capture, _artifact_path, artifact_name, artifact_size, image_format, image_size = self._capture_camera_artifact(
                    camera,
                    artifact_prefix="camera",
                )
            except OSError as exc:
                self._raise_device_busy("Camera self-test", exc)
                raise
        finally:
            self._close_quietly(camera)

        source_device = getattr(capture, "source_device", "unknown") if capture is not None else "unknown"
        input_format = getattr(capture, "input_format", None) if capture is not None else None
        return SelfTestResult(
            test_name="camera",
            status="ok",
            summary="Camera frame captured.",
            details=(
                f"Saved image as {artifact_name}.",
                "Stored in the self-test artifact directory.",
                f"Image format: {image_format}",
                f"Image size: {image_size}",
                f"Source device: {source_device}",
                f"Input format: {input_format or 'default'}",
                f"Bytes: {artifact_size}",
            ),
            artifact_name=artifact_name,
            finished_at=_utc_now_iso(),
        )

    def _run_aideck_camera_test(self) -> SelfTestResult:
        device = str(getattr(self.config, "camera_device", "") or "").strip()
        host, port = self._parse_aideck_device(device)
        self._probe_tcp_endpoint(
            host,
            port,
            timeout_s=self._bounded_float(
                getattr(self.config, "self_test_aideck_probe_timeout_s", _DEFAULT_TCP_PROBE_TIMEOUT_S),
                minimum=0.2,
                maximum=5.0,
                default=_DEFAULT_TCP_PROBE_TIMEOUT_S,
            ),
            label="AI-Deck stream endpoint",
        )
        camera = self.camera_factory(self.config)
        try:
            try:
                capture, _artifact_path, artifact_name, artifact_size, image_format, image_size = self._capture_camera_artifact(
                    camera,
                    artifact_prefix="aideck-camera",
                )
            except OSError as exc:
                self._raise_device_busy("AI-Deck camera self-test", exc)
                raise
        finally:
            self._close_quietly(camera)

        source_device = getattr(capture, "source_device", device) if capture is not None else device
        input_format = getattr(capture, "input_format", None) if capture is not None else None
        return SelfTestResult(
            test_name="aideck_camera",
            status="ok",
            summary="AI-Deck frame captured.",
            details=(
                f"Saved image as {artifact_name}.",
                "Stored in the self-test artifact directory.",
                "AI-Deck stream became reachable and returned one frame.",
                f"Image format: {image_format}",
                f"Image size: {image_size}",
                f"Source device: {source_device}",
                f"Input format: {input_format or 'default'}",
                f"Bytes: {artifact_size}",
            ),
            artifact_name=artifact_name,
            finished_at=_utc_now_iso(),
        )

    def _run_drone_stack_test(self) -> SelfTestResult:
        drone_config = DroneServiceConfig.from_config(self.config)
        if not drone_config.enabled:
            raise RuntimeError("Twinr drone support is disabled. Set TWINR_DRONE_ENABLED=true first.")
        if not drone_config.base_url:
            raise RuntimeError("Twinr drone support requires TWINR_DRONE_BASE_URL.")
        client = RemoteDroneServiceClient(
            base_url=drone_config.base_url,
            timeout_s=drone_config.request_timeout_s,
        )
        health = client.health_payload()
        state = client.state()
        mission = client.create_inspect_mission(
            target_hint="self test",
            capture_intent="scene",
            max_duration_s=drone_config.mission_timeout_s,
        )
        cancelled = None
        cancel_error: str | None = None
        try:
            cancelled = client.cancel_mission(mission.mission_id)
        except Exception as exc:
            cancel_error = self._describe_exception(exc)

        details = [
            f"Base URL: {drone_config.base_url}",
            f"Skill layer mode: {state.skill_layer_mode}",
            f"Radio ready: {str(state.safety.radio_ready).lower()}",
            f"Pose ready: {str(state.safety.pose_ready).lower()}",
            f"Can arm: {str(state.safety.can_arm).lower()}",
            f"Mission id: {mission.mission_id}",
            f"Mission state: {mission.state}",
            f"Health can_arm: {str(bool(health.get('can_arm'))).lower()}",
        ]
        if state.pose.tracking_state:
            details.append(f"Tracking state: {state.pose.tracking_state}")
        if state.safety.reasons:
            details.append(f"Safety reasons: {', '.join(state.safety.reasons)}")
        if cancelled is not None:
            details.append(f"Cancel state: {getattr(cancelled, 'state', 'unknown')}")
        if cancel_error:
            details.append(f"Cancel error: {cancel_error}")

        expected_state = "pending_manual_arm" if drone_config.require_manual_arm else "running"
        if mission.state != expected_state:
            raise RuntimeError(
                f"Drone daemon returned mission state `{mission.state}` instead of `{expected_state}`."
            )
        if cancel_error or cancelled is None:
            raise RuntimeError(
                "Drone self-test created a mission but could not confirm cancellation. Inspect the drone daemon before retrying."
            )
        cancel_state = str(getattr(cancelled, "state", "") or "").strip().lower()
        if cancel_state and "cancel" not in cancel_state and cancel_state not in {
            "aborted",
            "stopped",
            "terminated",
        }:
            raise RuntimeError(f"Drone self-test cancel returned unexpected state `{cancelled.state}`.")

        return SelfTestResult(
            test_name="drone_stack",
            status="ok",
            summary="Drone daemon reached and bounded inspect mission queued safely.",
            details=tuple(details),
            finished_at=_utc_now_iso(),
        )

    def _run_button_test(self) -> SelfTestResult:
        if not self.config.button_gpios:
            raise RuntimeError("No configured button GPIOs are available for probing.")
        self._ensure_gpio_self_test_available("Button self-test")
        try:
            with self.button_monitor_factory(self.config) as monitor:
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
            finished_at=_utc_now_iso(),
        )

    def _run_pir_test(self) -> SelfTestResult:
        if not self.config.pir_enabled:
            raise RuntimeError("No PIR motion GPIO is configured.")
        self._ensure_gpio_self_test_available("PIR self-test")
        wait_seconds = self._bounded_float(
            getattr(self.config, "self_test_pir_wait_seconds", 20.0),
            minimum=5.0,
            maximum=60.0,
            default=20.0,
        )
        try:
            with self.pir_monitor_factory(self.config) as monitor:
                current_value = monitor.snapshot_value()
                event = monitor.wait_for_motion(duration_s=wait_seconds, poll_timeout=0.2)
        except OSError as exc:
            self._raise_gpio_busy("PIR self-test", exc)
            raise
        if event is None:
            raise RuntimeError(f"No PIR motion event detected within {wait_seconds:g} seconds.")
        trigger_kind = "current high level" if event.raw_edge == "level" else f"{event.raw_edge} edge"
        details = [
            f"GPIO: {self.config.pir_motion_gpio}",
            f"Bias: {self.config.pir_bias}",
            f"Active high: {str(self.config.pir_active_high).lower()}",
            f"Initial value: {current_value}",
            f"Trigger: {trigger_kind}",
        ]
        event_timestamp_ns = getattr(event, "timestamp_ns", None)
        if event_timestamp_ns is not None:
            details.append(f"Event timestamp ns: {event_timestamp_ns}")
        return SelfTestResult(
            test_name="pir",
            status="ok",
            summary="PIR motion detected.",
            details=tuple(details),
            finished_at=_utc_now_iso(),
        )

    def _ensure_gpio_self_test_available(self, label: str) -> None:
        active_loops: list[str] = []
        for loop_name, loop_label in (("realtime-loop", "realtime loop"), ("hardware-loop", "hardware loop")):
            try:
                owner = loop_lock_owner(self.config, loop_name)
            except OSError as exc:
                raise RuntimeError("Could not inspect active Twinr loop locks.") from exc
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

    def _raise_device_busy(self, label: str, exc: OSError) -> None:
        if exc.errno != errno.EBUSY:
            return
        raise SelfTestBlockedError(
            f"{label} could not access the configured device because it is busy. Stop the active Twinr runtime or other device consumer and try again."
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
            start_timeout_s=self._bounded_float(
                getattr(config, "self_test_mic_start_timeout_s", config.audio_start_timeout_s),
                minimum=1.0,
                maximum=8.0,
                default=4.0,
            ),
            max_record_seconds=self._bounded_float(
                getattr(config, "self_test_mic_max_record_seconds", config.audio_max_record_seconds),
                minimum=2.0,
                maximum=15.0,
                default=8.0,
            ),
        )

    def _normalize_test_name(self, test_name: str) -> str:
        return " ".join((test_name or "").strip().lower().split())

    def _append_event_safely(self, **kwargs: Any) -> None:
        try:
            self.event_store.append(**kwargs)
        except Exception:
            return

    def _describe_exception(self, exc: Exception) -> str:
        text = " ".join(part.strip() for part in str(exc).splitlines() if part.strip())
        if not text:
            text = exc.__class__.__name__
        if len(text) > 500:
            text = f"{text[:497]}..."
        return text

    def _close_quietly(self, resource: Any) -> None:
        if resource is None:
            return
        close = getattr(resource, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception:
            return

    def _ensure_self_tests_root(self) -> Path:
        root = Path(self.paths.self_tests_root)
        self._ensure_no_symlink_components(root)
        parent = root.parent
        parent.mkdir(parents=True, exist_ok=True)
        if root.exists():
            if root.is_symlink():
                raise RuntimeError(f"Self-test storage path `{root}` must not be a symlink.")
            if not root.is_dir():
                raise RuntimeError(f"Self-test storage path `{root}` is not a directory.")
        else:
            root.mkdir(mode=_DIR_MODE, exist_ok=True)
        self._chmod_if_possible(root, _DIR_MODE)
        return root

    def _ensure_no_symlink_components(self, target: Path) -> None:
        for candidate in (target, *target.parents):
            try:
                if candidate.exists() and candidate.is_symlink():
                    raise RuntimeError(f"Path component `{candidate}` must not be a symlink.")
            except OSError as exc:
                raise RuntimeError(f"Could not validate path component `{candidate}`.") from exc
            if candidate == candidate.parent:
                break

    def _make_artifact_name(self, prefix: str, suffix: str) -> str:
        return f"{prefix}-{_utc_stamp()}-{uuid4().hex[:8]}{suffix}"

    def _make_artifact_stem(self, prefix: str) -> str:
        return f"{prefix}-{_utc_stamp()}-{uuid4().hex[:8]}"

    def _artifact_target_path(self, artifact_name: str) -> Path:
        if Path(artifact_name).name != artifact_name:
            raise RuntimeError("Artifact name must be a plain filename.")
        return self._ensure_self_tests_root() / artifact_name

    def _write_artifact_bytes(self, artifact_name: str, payload: bytes) -> Path:
        target_path = self._artifact_target_path(artifact_name)
        with tempfile.TemporaryDirectory(prefix=".self-test-", dir=str(target_path.parent)) as staging_dir_raw:
            staging_dir = Path(staging_dir_raw)
            self._chmod_if_possible(staging_dir, _DIR_MODE)
            staged_path = staging_dir / artifact_name
            self._write_private_file(staged_path, payload)
            os.replace(staged_path, target_path)
        self._chmod_if_possible(target_path, _FILE_MODE)
        return target_path

    def _write_private_file(self, path: Path, payload: bytes) -> None:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _FILE_MODE)
        try:
            with os.fdopen(fd, "wb", closefd=False) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.fchmod(fd, _FILE_MODE)
        finally:
            os.close(fd)

    def _capture_camera_artifact(
        self,
        camera: Any,
        *,
        artifact_prefix: str,
    ) -> tuple[Any, Path, str, int, str, str]:
        self._prepare_artifact_storage(
            estimated_bytes=self._bounded_int(
                getattr(self.config, "self_test_camera_estimated_bytes", 512 * 1024),
                minimum=16 * 1024,
                maximum=64 * 1024 * 1024,
                default=512 * 1024,
            )
        )
        stem = self._make_artifact_stem(artifact_prefix)
        with tempfile.TemporaryDirectory(prefix=".self-test-camera-", dir=str(self._ensure_self_tests_root())) as staging_dir_raw:
            staging_dir = Path(staging_dir_raw)
            self._chmod_if_possible(staging_dir, _DIR_MODE)
            staged_name = f"{stem}.capture"
            attempts: tuple[dict[str, object], ...] = (
                {"output_path": None, "filename": staged_name},
                {"output_path": str(staging_dir), "filename": staged_name},
            )
            capture: Any = None
            staged_path: Path | None = None
            last_type_error: TypeError | None = None
            for kwargs in attempts:
                try:
                    capture = camera.capture_photo(**kwargs)
                except TypeError as exc:
                    last_type_error = exc
                    continue
                staged_path = self._materialize_capture_artifact(
                    capture,
                    staging_dir=staging_dir,
                    staged_name=staged_name,
                )
                if staged_path is not None:
                    break

            if staged_path is None:
                if last_type_error is not None and capture is None:
                    raise last_type_error
                raise RuntimeError("Camera self-test did not produce a usable image artifact.")

            payload = staged_path.read_bytes()
            image_format = self._sniff_image_format(payload)
            if image_format not in {"png", "jpeg"}:
                # BREAKING: camera self-test artifacts must now be directly viewable PNG/JPEG files.
                raise RuntimeError(
                    "Camera self-test produced an unsupported image format. Configure the backend to emit PNG or JPEG."
                )
            dimensions = self._image_dimensions(payload, image_format)
            self._validate_image_artifact(payload, image_format=image_format, dimensions=dimensions)
            # BREAKING: camera self-test artifacts now keep their real suffix (.png or .jpg)
            # instead of always forcing .png for every backend.
            artifact_name = f"{stem}{'.png' if image_format == 'png' else '.jpg'}"
            final_staged_path = staging_dir / artifact_name
            if final_staged_path != staged_path:
                staged_path.rename(final_staged_path)
            target_path = self._artifact_target_path(artifact_name)
            os.replace(final_staged_path, target_path)
        self._chmod_if_possible(target_path, _FILE_MODE)
        artifact_size = target_path.stat().st_size
        image_size = "unknown" if dimensions is None else f"{dimensions[0]}x{dimensions[1]}"
        return capture, target_path, artifact_name, artifact_size, image_format, image_size

    def _materialize_capture_artifact(
        self,
        capture: Any,
        *,
        staging_dir: Path,
        staged_name: str,
    ) -> Path | None:
        staged_path = staging_dir / staged_name
        data = getattr(capture, "data", None) if capture is not None else None
        if data:
            self._write_private_file(staged_path, bytes(data))
            return staged_path

        candidate_paths: list[Path] = [staging_dir / staged_name]
        if capture is not None:
            for attr_name in (
                "artifact_path",
                "output_path",
                "path",
                "file_path",
                "saved_path",
                "image_path",
            ):
                value = getattr(capture, attr_name, None)
                if not value:
                    continue
                candidate = Path(str(value))
                if candidate.is_dir():
                    candidate = candidate / staged_name
                candidate_paths.append(candidate)
            filename = getattr(capture, "filename", None)
            if filename:
                candidate_paths.append(staging_dir / str(filename))
                candidate_paths.append(Path.cwd() / str(filename))
        candidate_paths.append(Path.cwd() / staged_name)

        seen: set[str] = set()
        for candidate in candidate_paths:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            try:
                if not candidate.exists() or not candidate.is_file() or candidate.stat().st_size <= 0:
                    continue
            except OSError:
                continue
            if candidate.resolve() == staged_path.resolve():
                return staged_path
            payload = candidate.read_bytes()
            self._write_private_file(staged_path, payload)
            self._cleanup_foreign_capture(candidate, staged_path=staged_path, staged_name=staged_name)
            return staged_path
        return None

    def _cleanup_foreign_capture(self, candidate: Path, *, staged_path: Path, staged_name: str) -> None:
        try:
            if candidate.resolve() == staged_path.resolve():
                return
        except OSError:
            return
        if candidate.parent == Path.cwd() and candidate.name == staged_name:
            try:
                candidate.unlink(missing_ok=True)
            except Exception:
                return

    def _sniff_image_format(self, payload: bytes) -> str | None:
        if len(payload) >= len(_PNG_SIGNATURE) and payload.startswith(_PNG_SIGNATURE):
            return "png"
        if len(payload) >= len(_JPEG_SIGNATURE) and payload.startswith(_JPEG_SIGNATURE):
            return "jpeg"
        return None

    def _image_dimensions(self, payload: bytes, image_format: str) -> tuple[int, int] | None:
        if image_format == "png":
            return self._png_dimensions(payload)
        if image_format == "jpeg":
            return self._jpeg_dimensions(payload)
        return None

    def _validate_image_artifact(
        self,
        payload: bytes,
        *,
        image_format: str,
        dimensions: tuple[int, int] | None,
    ) -> None:
        min_bytes = self._bounded_int(
            getattr(self.config, "self_test_min_image_bytes", _MIN_IMAGE_BYTES_DEFAULT),
            minimum=128,
            maximum=16 * 1024 * 1024,
            default=_MIN_IMAGE_BYTES_DEFAULT,
        )
        if len(payload) < min_bytes:
            raise RuntimeError(f"Camera self-test produced an unexpectedly small {image_format.upper()} artifact.")
        min_dimension = self._bounded_int(
            getattr(self.config, "self_test_min_image_dimension_px", _MIN_IMAGE_DIMENSION_DEFAULT),
            minimum=8,
            maximum=4096,
            default=_MIN_IMAGE_DIMENSION_DEFAULT,
        )
        if dimensions is not None and (dimensions[0] < min_dimension or dimensions[1] < min_dimension):
            raise RuntimeError(
                f"Camera self-test produced an implausibly small image ({dimensions[0]}x{dimensions[1]})."
            )

    @staticmethod
    def _png_dimensions(payload: bytes) -> tuple[int, int] | None:
        if len(payload) < 24 or not payload.startswith(_PNG_SIGNATURE):
            return None
        if payload[12:16] != b"IHDR":
            return None
        width, height = struct.unpack(">II", payload[16:24])
        if width <= 0 or height <= 0:
            return None
        return width, height

    @staticmethod
    def _jpeg_dimensions(payload: bytes) -> tuple[int, int] | None:
        if len(payload) < 4 or not payload.startswith(_JPEG_SIGNATURE):
            return None
        index = 2
        sof_markers = {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }
        while index < len(payload) - 1:
            if payload[index] != 0xFF:
                index += 1
                continue
            while index < len(payload) and payload[index] == 0xFF:
                index += 1
            if index >= len(payload):
                return None
            marker = payload[index]
            index += 1
            if marker in {0xD8, 0xD9, 0x01}:
                continue
            if marker == 0xDA:
                return None
            if index + 2 > len(payload):
                return None
            segment_length = int.from_bytes(payload[index:index + 2], "big")
            if segment_length < 2 or index + segment_length > len(payload):
                return None
            if marker in sof_markers:
                if segment_length < 7:
                    return None
                height = int.from_bytes(payload[index + 3:index + 5], "big")
                width = int.from_bytes(payload[index + 5:index + 7], "big")
                if width <= 0 or height <= 0:
                    return None
                return width, height
            index += segment_length
        return None

    def _parse_aideck_device(self, device: str) -> tuple[str, int]:
        normalized = str(device or "").strip()
        if not normalized.lower().startswith(_AIDECK_DEVICE_SCHEME):
            raise RuntimeError("AI-Deck self-test requires TWINR_CAMERA_DEVICE to use aideck://host[:port].")
        try:
            parsed = urlsplit(normalized)
            host = str(parsed.hostname or "").strip()
            port = int(parsed.port or _AIDECK_DEFAULT_PORT)
        except ValueError as exc:
            raise RuntimeError("AI-Deck self-test requires a valid aideck://host[:port] device URI.") from exc
        if not host:
            raise RuntimeError("AI-Deck self-test requires an aideck://host[:port] device URI.")
        if parsed.username or parsed.password:
            raise RuntimeError("AI-Deck device URI must not include credentials.")
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise RuntimeError("AI-Deck device URI must be exactly aideck://host[:port] with no path or query.")
        if port <= 0 or port > 65535:
            raise RuntimeError("AI-Deck device URI contains an invalid TCP port.")
        return host, port

    def _probe_tcp_endpoint(self, host: str, port: int, *, timeout_s: float, label: str) -> None:
        try:
            with socket.create_connection((host, port), timeout=timeout_s):
                return
        except OSError as exc:
            message = self._describe_exception(exc)
            raise RuntimeError(f"{label} `{host}:{port}` is unreachable: {message}") from exc

    def _wav_metadata(self, payload: bytes) -> dict[str, int] | None:
        try:
            with wave.open(io.BytesIO(payload), "rb") as wav_file:
                frame_count = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
        except (wave.Error, EOFError):
            return None
        duration_ms = 0
        if sample_rate > 0:
            duration_ms = int(round(frame_count * 1000 / sample_rate))
        return {
            "frame_count": int(frame_count),
            "sample_rate": int(sample_rate),
            "channels": int(channels),
            "sample_width_bytes": int(sample_width),
            "duration_ms": int(duration_ms),
        }

    def _process_lock_path(self) -> Path:
        root = Path(self.paths.self_tests_root)
        self._ensure_no_symlink_components(root.parent)
        root.parent.mkdir(parents=True, exist_ok=True)
        lock_path = root.parent / _LOCKFILE_NAME
        if lock_path.exists() and lock_path.is_symlink():
            raise RuntimeError(f"Self-test lock path `{lock_path}` must not be a symlink.")
        return lock_path

    def _acquire_process_run_lock(self) -> io.BufferedRandom:
        lock_path = self._process_lock_path()
        handle = open(lock_path, "a+b")
        try:
            self._chmod_if_possible(Path(lock_path), _FILE_MODE)
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise SelfTestBlockedError(
                "Another self-test is already running in a different Twinr process. Wait for it to finish and try again."
            ) from exc
        except Exception:
            handle.close()
            raise
        return handle

    def _release_process_run_lock(self, handle: io.BufferedRandom | None) -> None:
        if handle is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            handle.close()
        except Exception:
            pass

    def _prepare_artifact_storage(self, *, estimated_bytes: int) -> Path:
        root = self._ensure_self_tests_root()
        self._prune_artifacts(root)
        self._ensure_minimum_free_space(root, estimated_bytes=estimated_bytes)
        return root

    def _artifact_ttl_seconds(self) -> float:
        # BREAKING: self-test evidence is automatically pruned after a bounded TTL by default.
        return self._bounded_float(
            getattr(self.config, "self_test_artifact_ttl_seconds", _ARTIFACT_TTL_SECONDS_DEFAULT),
            minimum=60.0,
            maximum=14 * 24 * 60 * 60,
            default=float(_ARTIFACT_TTL_SECONDS_DEFAULT),
        )

    def _artifact_max_total_bytes(self) -> int:
        return self._bounded_int(
            getattr(self.config, "self_test_artifact_max_total_bytes", _MAX_STORAGE_BYTES_DEFAULT),
            minimum=4 * 1024 * 1024,
            maximum=4 * 1024 * 1024 * 1024,
            default=_MAX_STORAGE_BYTES_DEFAULT,
        )

    def _artifact_max_count(self) -> int:
        return self._bounded_int(
            getattr(self.config, "self_test_artifact_max_count", _MAX_STORAGE_FILES_DEFAULT),
            minimum=4,
            maximum=512,
            default=_MAX_STORAGE_FILES_DEFAULT,
        )

    def _minimum_free_bytes(self) -> int:
        return self._bounded_int(
            getattr(self.config, "self_test_min_free_bytes", _MIN_STORAGE_FREE_BYTES_DEFAULT),
            minimum=8 * 1024 * 1024,
            maximum=4 * 1024 * 1024 * 1024,
            default=_MIN_STORAGE_FREE_BYTES_DEFAULT,
        )

    def _iter_artifact_files(self, root: Path) -> list[Path]:
        files: list[Path] = []
        for child in root.iterdir():
            try:
                if child.is_file():
                    files.append(child)
            except OSError:
                continue
        return files

    def _prune_artifacts_best_effort(self) -> None:
        try:
            self._prune_artifacts(self._ensure_self_tests_root())
        except Exception:
            return

    def _prune_artifacts(self, root: Path) -> None:
        now = time.time()
        ttl_seconds = self._artifact_ttl_seconds()
        files = self._iter_artifact_files(root)

        for path in files:
            try:
                if now - path.stat().st_mtime > ttl_seconds:
                    path.unlink(missing_ok=True)
            except OSError:
                continue

        files = self._iter_artifact_files(root)
        if not files:
            return

        files.sort(key=self._path_mtime)
        max_total_bytes = self._artifact_max_total_bytes()
        max_count = self._artifact_max_count()
        total_bytes = 0
        sizes: dict[str, int] = {}
        for path in files:
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            sizes[str(path)] = size
            total_bytes += size

        while files and (len(files) > max_count or total_bytes > max_total_bytes):
            oldest = files.pop(0)
            size = sizes.get(str(oldest), 0)
            try:
                oldest.unlink(missing_ok=True)
                total_bytes -= size
            except OSError:
                continue

    def _ensure_minimum_free_space(self, root: Path, *, estimated_bytes: int) -> None:
        usage = shutil.disk_usage(root)
        required = self._minimum_free_bytes() + max(0, estimated_bytes)
        if usage.free < required:
            raise RuntimeError(
                f"Self-test storage is low on free space ({usage.free} bytes free, {required} required). "
                "Clear disk space and retry."
            )

    def _path_mtime(self, path: Path) -> float:
        try:
            return float(path.stat().st_mtime)
        except OSError:
            return 0.0

    def _chmod_if_possible(self, path: Path, mode: int) -> None:
        try:
            path.chmod(mode)
        except Exception:
            return

    @staticmethod
    def _bounded_int(value: Any, *, minimum: int, maximum: int, default: int) -> int:
        try:
            candidate = int(value)
        except (TypeError, ValueError):
            return default
        return max(minimum, min(candidate, maximum))

    @staticmethod
    def _bounded_float(value: Any, *, minimum: float, maximum: float, default: float) -> float:
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            return default
        return max(minimum, min(candidate, maximum))