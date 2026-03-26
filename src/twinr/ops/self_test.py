"""Run bounded self-tests for Twinr peripherals and operator workflows.

This module executes one-at-a-time diagnostics for microphone, speaker,
printer, camera, buttons, and PIR hardware and stores short-lived artifacts
when a test needs evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import errno
import os
import shutil
import struct
import tempfile
import threading
from typing import Any, Callable
from urllib.parse import urlsplit
from uuid import uuid4

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


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_AIDECK_DEVICE_SCHEME = "aideck://"
_AIDECK_DEFAULT_PORT = 5000
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


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

    # AUDIT-FIX(#1): Serialize self-tests across all runner instances in-process so parallel requests
    # do not contend for the same GPIO/audio/video devices or overwrite same-test artifacts.
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
            ("camera", "Kamera-Testbild", "Capture a fresh still image and store it as a PNG artifact."),
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
            A ``SelfTestResult`` with ``ok``, ``fail``, or ``blocked`` status.
        """

        normalized = self._normalize_test_name(test_name)
        finished_at = _utc_now_iso()
        available_tests = {name for name, _, _ in self.available_tests()}
        if normalized not in available_tests:
            # AUDIT-FIX(#8): Reject unsupported names before logging "started" events so invalid input
            # cannot create misleading audit records and returns an actionable list of valid tests.
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
                finished_at=finished_at,
            )

        if not self.__class__._RUN_LOCK.acquire(blocking=False):
            # AUDIT-FIX(#1): Return a deterministic blocked result instead of letting two self-tests
            # race each other and produce EBUSY errors or corrupted device state.
            summary = "Another self-test is already running. Wait for it to finish and try again."
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
                finished_at=finished_at,
            )

        try:
            try:
                # AUDIT-FIX(#2): Validate and create the artifact root inside the guarded execution path
                # so storage failures become structured self-test failures instead of escaping the runner.
                self._ensure_self_tests_root()
                # AUDIT-FIX(#3): Event logging is best-effort; telemetry failure must never crash the
                # self-test path because the appliance still needs to return a user-visible result.
                self._append_event_safely(
                    event="self_test_started",
                    message=f"Self-test `{normalized}` started.",
                    data={"test_name": normalized},
                )
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
                elif normalized == "aideck_camera":
                    result = self._run_aideck_camera_test()
                elif normalized == "drone_stack":
                    result = self._run_drone_stack_test()
                elif normalized == "buttons":
                    result = self._run_button_test()
                else:
                    result = self._run_pir_test()
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
                # AUDIT-FIX(#7): Normalize exception text before logging/returning it so blank messages
                # become actionable and multi-line values cannot pollute logs or operator UIs.
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

            self._append_event_safely(
                event="self_test_finished",
                message=f"Self-test `{normalized}` finished.",
                data={
                    "test_name": normalized,
                    "status": result.status,
                    "artifact_name": result.artifact_name or "",
                },
            )
            return result
        finally:
            self.__class__._RUN_LOCK.release()

    def _run_mic_test(self) -> SelfTestResult:
        recorder = self.recorder_factory(self.config)
        try:
            # AUDIT-FIX(#6): Use gentler self-test timing bounds so slower senior speech patterns do not
            # get truncated by the hard-coded 900 ms / 4 s / 6 s caps from the original implementation.
            pause_ms = self._bounded_int(
                getattr(self.config, "self_test_mic_pause_ms", self.config.speech_pause_ms),
                minimum=300,
                maximum=2500,
                default=1200,
            )
            audio_bytes = recorder.record_until_pause(pause_ms=pause_ms)
        finally:
            # AUDIT-FIX(#4): Quietly close hardware objects when they expose close() so one failed test
            # does not leave ALSA or device handles dangling and block later runs.
            self._close_quietly(recorder)

        payload = bytes(audio_bytes)
        if len(payload) <= 44:
            # AUDIT-FIX(#5): Do not report success for empty/header-only WAV payloads because that hides
            # broken microphone captures behind a false green status.
            raise RuntimeError("Microphone self-test captured no usable audio data.")

        artifact_name = self._make_artifact_name("mic", ".wav")
        self._write_artifact_bytes(artifact_name, payload)
        return SelfTestResult(
            test_name="mic",
            status="ok",
            summary="Speech sample recorded.",
            details=(
                f"Saved WAV recording as {artifact_name}.",
                f"Captured {len(payload)} bytes.",
                "Stored in the self-test artifact directory.",
            ),
            artifact_name=artifact_name,
            finished_at=_utc_now_iso(),
        )

    def _run_speaker_test(self) -> SelfTestResult:
        player = self.player_factory(self.config)
        try:
            player.play_tone(
                frequency_hz=self.config.audio_beep_frequency_hz,
                duration_ms=self.config.audio_beep_duration_ms,
                volume=self.config.audio_beep_volume,
                sample_rate=self.config.openai_realtime_input_sample_rate,
            )
        finally:
            # AUDIT-FIX(#4): Release the player backend when supported to avoid sticky output-device state
            # after repeated self-tests on small Linux audio stacks.
            self._close_quietly(player)
        return SelfTestResult(
            test_name="speaker",
            status="ok",
            summary="Confirmation beep played.",
            details=(f"Output device: {self.config.audio_output_device}",),
            finished_at=_utc_now_iso(),
        )

    def _run_proactive_mic_test(self) -> SelfTestResult:
        if not self.config.proactive_audio_enabled and not (self.config.proactive_audio_input_device or "").strip():
            raise RuntimeError("No proactive background-audio device is configured.")

        sampler = self.ambient_sampler_factory(self.config)
        try:
            sample_duration_ms = self._bounded_int(
                getattr(self.config, "self_test_proactive_audio_sample_ms", self.config.proactive_audio_sample_ms),
                minimum=300,
                maximum=3000,
                default=1000,
            )
            sample = sampler.sample_levels(duration_ms=sample_duration_ms)
        finally:
            # AUDIT-FIX(#4): Close ambient samplers after use when the backend exposes close() to prevent
            # background-input handles from surviving the self-test lifecycle.
            self._close_quietly(sampler)

        if getattr(sample, "chunk_count", 0) <= 0 or getattr(sample, "duration_ms", 0) <= 0:
            # AUDIT-FIX(#9): A zero-frame ambient sample is not a valid success path; surface it as a
            # failure so hardware/config problems are visible to the operator.
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
            print_job = printer.print_text("Twinr self-test\nPrinter path OK.")
        finally:
            # AUDIT-FIX(#4): Release printer backends when supported so transport handles are not kept
            # open across service actions on low-resource Raspberry Pi deployments.
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
            artifact_name = self._make_artifact_name("camera", ".png")
            capture, artifact_path, artifact_size = self._capture_camera_artifact(camera, artifact_name)
        finally:
            # AUDIT-FIX(#4): Close camera backends on best effort so /dev/video resources are released
            # promptly after a self-test instead of causing later EBUSY failures.
            self._close_quietly(camera)

        source_device = getattr(capture, "source_device", "unknown") if capture is not None else "unknown"
        input_format = getattr(capture, "input_format", None) if capture is not None else None
        return SelfTestResult(
            test_name="camera",
            status="ok",
            summary="Camera frame captured.",
            details=(
                f"Saved PNG as {artifact_name}.",
                "Stored in the self-test artifact directory.",
                f"Source device: {source_device}",
                f"Input format: {input_format or 'default'}",
                f"Bytes: {artifact_size}",
            ),
            artifact_name=artifact_name,
            finished_at=_utc_now_iso(),
        )

    def _run_aideck_camera_test(self) -> SelfTestResult:
        device = str(getattr(self.config, "camera_device", "") or "").strip()
        self._parse_aideck_device(device)
        camera = self.camera_factory(self.config)
        try:
            artifact_name = self._make_artifact_name("aideck-camera", ".png")
            capture, artifact_path, artifact_size = self._capture_camera_artifact(camera, artifact_name)
        finally:
            self._close_quietly(camera)

        source_device = getattr(capture, "source_device", device) if capture is not None else device
        input_format = getattr(capture, "input_format", None) if capture is not None else None
        image_size = self._summarize_image_dimensions(capture, artifact_path=artifact_path)
        return SelfTestResult(
            test_name="aideck_camera",
            status="ok",
            summary="AI-Deck frame captured.",
            details=(
                f"Saved PNG as {artifact_name}.",
                "Stored in the self-test artifact directory.",
                "AI-Deck stream became reachable and returned one frame.",
                f"Source device: {source_device}",
                f"Input format: {input_format or 'default'}",
                f"Image size: {image_size}",
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
        try:
            cancelled = client.cancel_mission(mission.mission_id)
        except Exception:
            cancelled = None
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
            details.append(f"Cancel state: {cancelled.state}")
        expected_state = "pending_manual_arm" if drone_config.require_manual_arm else "running"
        if mission.state != expected_state:
            raise RuntimeError(
                f"Drone daemon returned mission state `{mission.state}` instead of `{expected_state}`."
            )
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
            # AUDIT-FIX(#10): Use the injected button monitor factory just like the other hardware
            # dependencies so tests and alternate GPIO backends stay consistent.
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
                # AUDIT-FIX(#6): Allow a longer configurable PIR window because seniors often move more
                # slowly and the hard-coded 12 seconds caused avoidable false failures.
                event = monitor.wait_for_motion(duration_s=wait_seconds, poll_timeout=0.2)
        except OSError as exc:
            self._raise_gpio_busy("PIR self-test", exc)
            raise
        if event is None:
            raise RuntimeError(f"No PIR motion event detected within {wait_seconds:g} seconds.")
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

    def _build_mic_test_recorder(self, config: TwinrConfig) -> SilenceDetectedRecorder:
        return SilenceDetectedRecorder(
            device=config.audio_input_device,
            sample_rate=config.audio_sample_rate,
            channels=config.audio_channels,
            chunk_ms=config.audio_chunk_ms,
            preroll_ms=config.audio_preroll_ms,
            speech_threshold=config.audio_speech_threshold,
            speech_start_chunks=1,
            # AUDIT-FIX(#6): Keep self-test recorder limits bounded but no longer overly aggressive
            # versus the main device config, preserving slower user interaction patterns.
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
        # AUDIT-FIX(#8): Collapse embedded whitespace/newlines before validation so unsupported
        # names cannot inject formatting noise into event logs or UI surfaces.
        return " ".join((test_name or "").strip().lower().split())

    def _append_event_safely(self, **kwargs: Any) -> None:
        # AUDIT-FIX(#3): Ops telemetry is intentionally best-effort so event-store outages do not
        # bubble back into the user-facing self-test result path.
        try:
            self.event_store.append(**kwargs)
        except Exception:
            return

    def _describe_exception(self, exc: Exception) -> str:
        # AUDIT-FIX(#7): Collapse multi-line/empty exceptions into bounded one-line text for safer
        # logging and clearer operator feedback.
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
        # AUDIT-FIX(#2): Refuse a symlinked final artifact root and create the directory with
        # restrictive permissions so staged atomic writes land in a predictable on-device location.
        if root.exists():
            if root.is_symlink():
                raise RuntimeError(f"Self-test storage path `{root}` must not be a symlink.")
            if not root.is_dir():
                raise RuntimeError(f"Self-test storage path `{root}` is not a directory.")
            return root

        parent = root.parent
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        root.mkdir(exist_ok=True, mode=0o700)
        if root.is_symlink() or not root.is_dir():
            raise RuntimeError(f"Self-test storage path `{root}` could not be created safely.")
        return root

    def _make_artifact_name(self, prefix: str, suffix: str) -> str:
        # AUDIT-FIX(#2): Add microsecond timestamping plus a random suffix so repeated same-second runs
        # do not collide on the same filename and silently overwrite prior artifacts.
        return f"{prefix}-{_utc_stamp()}-{uuid4().hex[:8]}{suffix}"

    def _artifact_target_path(self, artifact_name: str) -> Path:
        if Path(artifact_name).name != artifact_name:
            raise RuntimeError("Artifact name must be a plain filename.")
        return self._ensure_self_tests_root() / artifact_name

    def _write_artifact_bytes(self, artifact_name: str, payload: bytes) -> Path:
        # AUDIT-FIX(#2): Write into a private staging directory and promote with os.replace() so
        # partially written files and target-path races do not leak into the final artifact name.
        target_path = self._artifact_target_path(artifact_name)
        staging_dir = Path(tempfile.mkdtemp(prefix=".self-test-", dir=str(target_path.parent)))
        staged_path = staging_dir / artifact_name
        try:
            with staged_path.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(staged_path, target_path)
            return target_path
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

    def _capture_camera_artifact(self, camera: Any, artifact_name: str) -> tuple[Any, Path, int]:
        # AUDIT-FIX(#2): Capture into a private staging area first, then atomically promote the
        # finished image so camera backends never write straight into the public artifact target.
        target_path = self._artifact_target_path(artifact_name)
        staging_dir = Path(tempfile.mkdtemp(prefix=".self-test-camera-", dir=str(target_path.parent)))
        staged_path = staging_dir / artifact_name
        try:
            capture = camera.capture_photo(output_path=None, filename=artifact_name)
            data = getattr(capture, "data", None) if capture is not None else None
            if data:
                with staged_path.open("wb") as handle:
                    handle.write(bytes(data))
                    handle.flush()
                    os.fsync(handle.fileno())
            if not staged_path.exists() or staged_path.stat().st_size <= 0:
                # AUDIT-FIX(#5): Accept either a file-backed capture or an in-memory payload, but never
                # mark the camera test successful when no image artifact was actually produced.
                raise RuntimeError("Camera self-test did not produce a usable image artifact.")
            artifact_size = staged_path.stat().st_size
            os.replace(staged_path, target_path)
            return capture, target_path, artifact_size
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

    def _parse_aideck_device(self, device: str) -> tuple[str, int]:
        normalized = str(device or "").strip()
        if not normalized.lower().startswith(_AIDECK_DEVICE_SCHEME):
            raise RuntimeError("AI-Deck self-test requires TWINR_CAMERA_DEVICE to use aideck://host[:port].")
        parsed = urlsplit(normalized)
        host = str(parsed.hostname or "").strip()
        if not host:
            raise RuntimeError("AI-Deck self-test requires an aideck://host[:port] device URI.")
        port = int(parsed.port or _AIDECK_DEFAULT_PORT)
        return host, port

    def _summarize_image_dimensions(self, capture: Any, *, artifact_path: Path) -> str:
        payload = b""
        data = getattr(capture, "data", None) if capture is not None else None
        if data:
            payload = bytes(data)
        elif artifact_path.exists():
            payload = artifact_path.read_bytes()
        dimensions = self._png_dimensions(payload)
        if dimensions is None:
            return "unknown"
        width, height = dimensions
        return f"{width}x{height}"

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
