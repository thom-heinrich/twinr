# CHANGELOG: 2026-03-30
# BUG-1: Audio checks now verify configured ALSA devices against the live ALSA inventory instead of only checking for arecord/aplay on PATH.
# BUG-2: Camera checks now probe the active Raspberry Pi camera stack (rpicam/libcamera) and V4L2 device readability instead of treating any character device as camera-ready.
# BUG-3: GPIO, display GPIO, and PIR checks now verify the configured gpiochip and line offsets through libgpiod/gpioinfo, eliminating false-green hardware checks.
# BUG-4: Printer queues that are missing or unreachable now fail instead of warning, and CUPS scheduler state is checked explicitly.
# BUG-5: Optional deep probes (ReSpeaker/Codex) are time-bounded so one hung probe cannot stall the whole audit.
# SEC-1: Runtime-state paths must now be absolute and are rejected when placed in shared/world-writable directories or writable non-owner files.
# SEC-2: Probe subprocesses now resolve binaries to absolute paths and scrub dangerous loader-related environment variables before execution.
# IMP-1: Added first-class Raspberry Pi camera-stack support for rpicam/libcamera and modern Pi OS camera deployments.
# IMP-2: Added active GPIO line availability checks using libgpiod v2 when available, with gpioinfo fallback for Pi deployments without Python bindings.

"""Run configuration checks for Twinr's operational surfaces.

This module inspects API keys, audio devices, printer, camera, GPIO, PIR, and
runtime-state paths and returns normalized check records for ops tooling.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
import os
from pathlib import Path
from queue import Empty, Queue
import re
from shutil import which
import stat
import subprocess
import threading
from typing import TypeVar, cast

from twinr.agent.base_agent.config import TwinrConfig
from twinr.display.wayland_env import resolve_wayland_socket
from twinr.hardware.respeaker import capture_respeaker_primitive_snapshot, config_targets_respeaker


_VALID_STATUSES = frozenset({"ok", "warn", "fail"})
_ALLOWED_PIR_BIASES = frozenset({"pull-up", "pull-down", "disable", "disabled", "as-is", "none"})
_ALLOWED_CAMERA_BACKENDS = frozenset({"", "auto", "v4l2", "usb", "libcamera", "rpicam", "picamera2"})
_PRINTER_CHECK_TIMEOUT_SECONDS = 5.0
_AUDIO_CHECK_TIMEOUT_SECONDS = 2.5
_CAMERA_CHECK_TIMEOUT_SECONDS = 3.5
_GPIO_CHECK_TIMEOUT_SECONDS = 2.0
_RESPEAKER_CHECK_TIMEOUT_SECONDS = 3.0
_CODEX_CHECK_TIMEOUT_SECONDS = 6.0
_TOOL_ENV_DENYLIST = frozenset({"LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONHOME", "PYTHONPATH"})
_GPIOD_LINE_BUSY_CONSUMER_ALLOWLIST = frozenset({"", "twinr", "twinr-agent", "twinr-display", "twinr-gpio"})


@dataclass(frozen=True, slots=True)
class ConfigCheck:
    """Represent one normalized configuration audit result."""

    key: str
    label: str
    status: str
    detail: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _normalize_status(self.status))
        object.__setattr__(self, "key", _sanitize_identifier(self.key))
        object.__setattr__(self, "label", _sanitize_detail(self.label, default="Unknown check"))
        object.__setattr__(self, "detail", _sanitize_detail(self.detail, default="No additional detail."))

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _ProbeOutcome:
    status: str
    detail: str


_T = TypeVar("_T")


def run_config_checks(config: TwinrConfig) -> tuple[ConfigCheck, ...]:
    """Run the standard Twinr ops configuration checks."""

    checks: tuple[tuple[str, str, Callable[[TwinrConfig], ConfigCheck]], ...] = (
        ("openai_key", "OpenAI key", _openai_key_check),
        ("audio_input", "Audio input", _audio_input_check),
        ("proactive_audio_input", "Proactive audio input", _proactive_audio_input_check),
        ("respeaker_xvf3800", "ReSpeaker XVF3800", _respeaker_xvf3800_check),
        ("audio_output", "Audio output", _audio_output_check),
        ("printer_queue", "Printer queue", _printer_check),
        ("camera", "Camera", _camera_check),
        ("gpio", "GPIO buttons", _gpio_check),
        ("display_gpio", "Display GPIO", _display_gpio_check),
        ("pir", "PIR motion sensor", _pir_check),
        ("runtime_state", "Runtime snapshot", _runtime_state_check),
        ("self_coding_codex", "Self-coding Codex", _self_coding_codex_check),
    )
    return tuple(_run_check_safely(key, label, check_fn, config) for key, label, check_fn in checks)


def check_summary(checks: tuple[ConfigCheck, ...] | list[ConfigCheck]) -> dict[str, int]:
    """Count configuration checks by normalized status."""

    summary = {"ok": 0, "warn": 0, "fail": 0}
    for check in checks:
        status = _normalize_status(getattr(check, "status", None))
        summary[status] += 1
    return summary


def _openai_key_check(config: TwinrConfig) -> ConfigCheck:
    api_key = _clean_config_text(getattr(config, "openai_api_key", ""))
    if api_key:
        return ConfigCheck("openai_key", "OpenAI key", "ok", "API key is configured.")
    return ConfigCheck("openai_key", "OpenAI key", "fail", "OPENAI_API_KEY is missing.")


def _audio_input_check(config: TwinrConfig) -> ConfigCheck:
    device = _clean_config_text(getattr(config, "audio_input_device", ""))
    if not device:
        return ConfigCheck("audio_input", "Audio input", "fail", "No input device is configured.")

    probe = _probe_alsa_device("arecord", device)
    if probe.status == "ok":
        return ConfigCheck(
            "audio_input",
            "Audio input",
            "ok",
            f"Input device `{_display_value(device)}` is visible to ALSA. {probe.detail}",
        )
    return ConfigCheck("audio_input", "Audio input", probe.status, probe.detail)


def _audio_output_check(config: TwinrConfig) -> ConfigCheck:
    device = _clean_config_text(getattr(config, "audio_output_device", ""))
    if not device:
        return ConfigCheck("audio_output", "Audio output", "fail", "No output device is configured.")

    probe = _probe_alsa_device("aplay", device)
    if probe.status == "ok":
        return ConfigCheck(
            "audio_output",
            "Audio output",
            "ok",
            f"Output device `{_display_value(device)}` is visible to ALSA. {probe.detail}",
        )
    return ConfigCheck("audio_output", "Audio output", probe.status, probe.detail)


def _proactive_audio_input_check(config: TwinrConfig) -> ConfigCheck:
    proactive_audio_enabled = _coerce_bool(getattr(config, "proactive_audio_enabled", False))
    if not proactive_audio_enabled:
        return ConfigCheck(
            "proactive_audio_input",
            "Proactive audio input",
            "ok",
            "Proactive background-audio sampling is disabled.",
        )

    proactive_device = _clean_config_text(getattr(config, "proactive_audio_input_device", ""))
    if proactive_device:
        probe = _probe_alsa_device("arecord", proactive_device)
        if probe.status == "ok":
            return ConfigCheck(
                "proactive_audio_input",
                "Proactive audio input",
                "ok",
                f"Proactive mic `{_display_value(proactive_device)}` is visible to ALSA. {probe.detail}",
            )
        return ConfigCheck("proactive_audio_input", "Proactive audio input", probe.status, probe.detail)

    primary_device = _clean_config_text(getattr(config, "audio_input_device", ""))
    if not primary_device:
        return ConfigCheck(
            "proactive_audio_input",
            "Proactive audio input",
            "fail",
            "Proactive background audio is enabled, but no input device is available.",
        )

    probe = _probe_alsa_device("arecord", primary_device)
    if probe.status == "ok":
        return ConfigCheck(
            "proactive_audio_input",
            "Proactive audio input",
            "ok",
            f"Proactive audio reuses primary input `{_display_value(primary_device)}`. {probe.detail}",
        )
    return ConfigCheck(
        "proactive_audio_input",
        "Proactive audio input",
        probe.status,
        f"Proactive audio reuses `{_display_value(primary_device)}`, but readiness could not be confirmed. {probe.detail}",
    )


def _respeaker_xvf3800_check(config: TwinrConfig) -> ConfigCheck:
    """Check whether a configured or attached XVF3800 is capture-ready."""

    primary_device = _clean_config_text(getattr(config, "audio_input_device", ""))
    proactive_device = _clean_config_text(getattr(config, "proactive_audio_input_device", ""))
    configured = config_targets_respeaker(primary_device, proactive_device)

    try:
        snapshot = _call_with_timeout(
            capture_respeaker_primitive_snapshot,
            timeout=_RESPEAKER_CHECK_TIMEOUT_SECONDS,
            operation_label="ReSpeaker XVF3800 probe",
        )
    except TimeoutError:
        if configured:
            return ConfigCheck(
                "respeaker_xvf3800",
                "ReSpeaker XVF3800",
                "warn",
                f"ReSpeaker XVF3800 probing exceeded {_RESPEAKER_CHECK_TIMEOUT_SECONDS:.1f}s and readiness could not be confirmed.",
            )
        return ConfigCheck(
            "respeaker_xvf3800",
            "ReSpeaker XVF3800",
            "warn",
            f"ReSpeaker probe exceeded {_RESPEAKER_CHECK_TIMEOUT_SECONDS:.1f}s; no configured XVF3800 path was verified.",
        )

    probe = snapshot.probe

    if probe.capture_ready and snapshot.host_control_ready:
        capture = probe.capture_device
        capture_label = capture.card_label if capture is not None else "capture ready"
        detail = (
            f"ReSpeaker XVF3800 is capture-ready as `{_display_value(capture_label)}` "
            "and host-control primitives are readable."
        )
        firmware_label = _respeaker_firmware_label(snapshot.firmware_version)
        if firmware_label is not None:
            detail += f" Firmware `{firmware_label}` is visible."
        return ConfigCheck("respeaker_xvf3800", "ReSpeaker XVF3800", "ok", detail)

    if probe.capture_ready and not snapshot.host_control_ready:
        reason = _display_value(snapshot.transport.reason, unset="unknown_transport_state")
        detail = (
            "ReSpeaker XVF3800 is ALSA capture-ready, but host-control primitives are degraded "
            f"(`{reason}`)."
        )
        if snapshot.transport.requires_elevated_permissions:
            detail += " The runtime user likely lacks the required USB permissions."
        return ConfigCheck("respeaker_xvf3800", "ReSpeaker XVF3800", "warn", detail)

    if probe.usb_visible and probe.arecord_available:
        return ConfigCheck(
            "respeaker_xvf3800",
            "ReSpeaker XVF3800",
            "warn",
            "ReSpeaker XVF3800 is USB-visible, but no ALSA capture card was reported. This usually means DFU/safe mode or incomplete runtime.",
        )

    if probe.usb_visible:
        return ConfigCheck(
            "respeaker_xvf3800",
            "ReSpeaker XVF3800",
            "warn",
            "ReSpeaker XVF3800 is USB-visible, but `arecord` is unavailable, so capture readiness cannot be verified.",
        )

    if configured:
        if not probe.lsusb_available and not probe.arecord_available:
            return ConfigCheck(
                "respeaker_xvf3800",
                "ReSpeaker XVF3800",
                "warn",
                "A ReSpeaker XVF3800 path is configured, but neither `lsusb` nor `arecord` is available for verification.",
            )
        return ConfigCheck(
            "respeaker_xvf3800",
            "ReSpeaker XVF3800",
            "fail",
            "A ReSpeaker XVF3800 capture path is configured, but the device was not detected.",
        )

    return ConfigCheck(
        "respeaker_xvf3800",
        "ReSpeaker XVF3800",
        "ok",
        "No ReSpeaker XVF3800 capture path is configured or detected.",
    )


def _printer_check(config: TwinrConfig) -> ConfigCheck:
    queue = _clean_config_text(getattr(config, "printer_queue", ""))
    if not queue:
        return ConfigCheck("printer_queue", "Printer queue", "fail", "TWINR_PRINTER_QUEUE is missing.")

    if _resolve_executable("lp") is None:
        return ConfigCheck(
            "printer_queue",
            "Printer queue",
            "warn",
            f"Queue `{_display_value(queue)}` is configured, but the `lp` command is not available.",
        )

    if _resolve_executable("lpstat") is None:
        return ConfigCheck(
            "printer_queue",
            "Printer queue",
            "warn",
            f"Queue `{_display_value(queue)}` is configured, but queue validation via `lpstat` is unavailable.",
        )

    try:
        scheduler = _run_command(["lpstat", "-r"], timeout=_PRINTER_CHECK_TIMEOUT_SECONDS)
        queue_state = _run_command(["lpstat", "-p", queue], timeout=_PRINTER_CHECK_TIMEOUT_SECONDS)
        device_state = _run_command(["lpstat", "-v", queue], timeout=_PRINTER_CHECK_TIMEOUT_SECONDS)
        accepting_state = _run_command(["lpstat", "-a", queue], timeout=_PRINTER_CHECK_TIMEOUT_SECONDS)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return ConfigCheck(
            "printer_queue",
            "Printer queue",
            "warn",
            f"Queue `{_display_value(queue)}` is configured, but CUPS probing failed: {exc}",
        )

    scheduler_text = _combine_command_output(scheduler).lower()
    if scheduler.returncode != 0 or "not running" in scheduler_text:
        return ConfigCheck(
            "printer_queue",
            "Printer queue",
            "fail",
            "The CUPS scheduler is not running, so the printer queue is unavailable.",
        )

    if queue_state.returncode != 0:
        return ConfigCheck(
            "printer_queue",
            "Printer queue",
            "fail",
            _command_error_detail(queue_state, default=f"Queue `{_display_value(queue)}` is not visible to CUPS."),
        )

    if device_state.returncode != 0:
        return ConfigCheck(
            "printer_queue",
            "Printer queue",
            "fail",
            _command_error_detail(device_state, default=f"Queue `{_display_value(queue)}` is configured but no device binding was reported."),
        )

    queue_text = _combine_command_output(queue_state).lower()
    accepting_text = _combine_command_output(accepting_state).lower()
    device_text = _sanitize_detail(_combine_command_output(device_state), default="")

    if "disabled" in queue_text:
        return ConfigCheck(
            "printer_queue",
            "Printer queue",
            "warn",
            f"Queue `{_display_value(queue)}` exists but is currently disabled. {_trim_detail(device_text)}",
        )

    if accepting_state.returncode != 0 or "not accepting" in accepting_text:
        return ConfigCheck(
            "printer_queue",
            "Printer queue",
            "warn",
            f"Queue `{_display_value(queue)}` exists but is not accepting jobs. {_trim_detail(device_text)}",
        )

    return ConfigCheck(
        "printer_queue",
        "Printer queue",
        "ok",
        f"Queue `{_display_value(queue)}` is enabled, accepting jobs, and visible to CUPS. {_trim_detail(device_text)}",
    )


def _camera_check(config: TwinrConfig) -> ConfigCheck:
    backend = _clean_config_text(getattr(config, "camera_backend", "auto")).lower() or "auto"
    if backend not in _ALLOWED_CAMERA_BACKENDS:
        return ConfigCheck(
            "camera",
            "Camera",
            "fail",
            f"Unsupported camera backend `{_display_value(backend)}`.",
        )

    raw_device = _clean_config_text(getattr(config, "camera_device", ""))
    ffmpeg_path = _clean_config_text(getattr(config, "camera_ffmpeg_path", "")) or "ffmpeg"
    rpicam_binary = _detect_rpicam_binary(_clean_config_text(getattr(config, "camera_rpicam_binary", "")))

    if backend in {"libcamera", "rpicam", "picamera2"}:
        return _libcamera_camera_check(rpicam_binary)

    if backend in {"v4l2", "usb"}:
        return _v4l2_camera_check(raw_device, ffmpeg_path)

    libcamera_ready = _probe_rpicam_stack(rpicam_binary)
    if libcamera_ready.status == "ok":
        return ConfigCheck("camera", "Camera", "ok", libcamera_ready.detail)

    if raw_device:
        v4l2_check = _v4l2_camera_check(raw_device, ffmpeg_path)
        if v4l2_check.status == "ok":
            return v4l2_check
        if libcamera_ready.status == "warn":
            return ConfigCheck(
                "camera",
                "Camera",
                "warn",
                f"{v4l2_check.detail} Also, Raspberry Pi camera-stack probing was inconclusive: {libcamera_ready.detail}",
            )
        return v4l2_check

    if libcamera_ready.status in {"warn", "fail"} and rpicam_binary is not None:
        return ConfigCheck("camera", "Camera", libcamera_ready.status, libcamera_ready.detail)

    if raw_device:
        return _v4l2_camera_check(raw_device, ffmpeg_path)

    return ConfigCheck(
        "camera",
        "Camera",
        "fail",
        "No camera device or supported Raspberry Pi camera backend is configured.",
    )


def _libcamera_camera_check(rpicam_binary: str | None) -> ConfigCheck:
    probe = _probe_rpicam_stack(rpicam_binary)
    return ConfigCheck("camera", "Camera", probe.status, probe.detail)


def _v4l2_camera_check(raw_device: str, ffmpeg_value: str) -> ConfigCheck:
    if not raw_device:
        return ConfigCheck("camera", "Camera", "fail", "No V4L2 camera device is configured.")

    device = Path(raw_device).expanduser()
    if not device.exists():
        return ConfigCheck(
            "camera",
            "Camera",
            "fail",
            f"Camera device `{_display_value(device)}` does not exist on this machine.",
        )
    if not device.is_char_device():
        return ConfigCheck(
            "camera",
            "Camera",
            "fail",
            f"Camera device `{_display_value(device)}` exists, but it is not a character device.",
        )
    if not os.access(device, os.R_OK):
        return ConfigCheck(
            "camera",
            "Camera",
            "fail",
            f"Camera device `{_display_value(device)}` exists, but it is not readable by the runtime user.",
        )

    v4l2_path = _resolve_executable("v4l2-ctl")
    if v4l2_path is not None:
        try:
            v4l2_result = _run_command(["v4l2-ctl", "-d", str(device), "-D"], timeout=_CAMERA_CHECK_TIMEOUT_SECONDS)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            return ConfigCheck(
                "camera",
                "Camera",
                "warn",
                f"Camera device `{_display_value(device)}` exists, but V4L2 probing failed: {exc}",
            )
        if v4l2_result.returncode != 0:
            return ConfigCheck(
                "camera",
                "Camera",
                "fail",
                _command_error_detail(v4l2_result, default=f"Camera device `{_display_value(device)}` is not queryable via V4L2."),
            )
    else:
        return ConfigCheck(
            "camera",
            "Camera",
            "warn",
            f"Camera device `{_display_value(device)}` exists, but `v4l2-ctl` is unavailable so V4L2 readiness could not be confirmed.",
        )

    ffmpeg_resolved = _resolve_executable(ffmpeg_value)
    if ffmpeg_resolved is None:
        return ConfigCheck(
            "camera",
            "Camera",
            "warn",
            f"Camera device `{_display_value(device)}` is queryable via V4L2, but ffmpeg `{_display_value(ffmpeg_value)}` is not available.",
        )

    return ConfigCheck(
        "camera",
        "Camera",
        "ok",
        f"Camera device `{_display_value(device)}` is queryable via V4L2 and ffmpeg `{_display_value(ffmpeg_resolved)}` is available.",
    )


def _probe_rpicam_stack(rpicam_binary: str | None) -> _ProbeOutcome:
    if rpicam_binary is None:
        return _ProbeOutcome(
            "warn",
            "No `rpicam-*` camera probe binary is available. Install `rpicam-apps` on modern Raspberry Pi OS camera deployments.",
        )

    try:
        result = _run_command([rpicam_binary, "--list-cameras"], timeout=_CAMERA_CHECK_TIMEOUT_SECONDS)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return _ProbeOutcome(
            "warn",
            f"Raspberry Pi camera-stack probing via `{_display_value(rpicam_binary)}` failed: {exc}",
        )

    output = _combine_command_output(result)
    lowered = output.lower()
    if result.returncode != 0:
        return _ProbeOutcome(
            "fail",
            _command_error_detail(result, default=f"`{_display_value(rpicam_binary)}` could not enumerate cameras."),
        )
    if "no cameras available" in lowered:
        return _ProbeOutcome(
            "fail",
            f"`{_display_value(rpicam_binary)}` ran successfully but reported no cameras available.",
        )

    camera_count = len(re.findall(r"^\s*\d+\s*:", output, flags=re.MULTILINE))
    if camera_count == 0 and "available cameras" not in lowered:
        return _ProbeOutcome(
            "warn",
            f"`{_display_value(rpicam_binary)}` executed, but its output did not clearly enumerate attached cameras.",
        )

    count_label = f"{camera_count} camera{'s' if camera_count != 1 else ''}" if camera_count else "at least one camera"
    return _ProbeOutcome(
        "ok",
        f"Raspberry Pi camera stack is available via `{_display_value(rpicam_binary)}` and enumerated {count_label}.",
    )


def _self_coding_codex_check(_config: TwinrConfig) -> ConfigCheck:
    """Check whether the local self_coding Codex runtime is runnable."""

    def _collect_report() -> object:
        from twinr.agent.self_coding.codex_driver.environment import collect_codex_sdk_environment_report

        return collect_codex_sdk_environment_report(
            run_local_self_test=True,
            run_live_auth_check=False,
        )

    try:
        report = _call_with_timeout(
            _collect_report,
            timeout=_CODEX_CHECK_TIMEOUT_SECONDS,
            operation_label="Self-coding Codex probe",
        )
    except TimeoutError:
        return ConfigCheck(
            "self_coding_codex",
            "Self-coding Codex",
            "warn",
            f"Self-coding Codex probe exceeded {_CODEX_CHECK_TIMEOUT_SECONDS:.1f}s and was aborted.",
        )

    ready = bool(getattr(report, "ready", False))
    detail = _sanitize_detail(getattr(report, "detail", ""), default="No additional detail.")
    return ConfigCheck(
        "self_coding_codex",
        "Self-coding Codex",
        "ok" if ready else "fail",
        detail,
    )


def _gpio_check(config: TwinrConfig) -> ConfigCheck:
    raw_gpios = getattr(config, "button_gpios", {})
    if not isinstance(raw_gpios, Mapping):
        return ConfigCheck(
            "gpio",
            "GPIO buttons",
            "fail",
            "Button GPIO mapping must be a dictionary-like object.",
        )

    gpios = dict(raw_gpios)
    chip = _clean_config_text(getattr(config, "gpio_chip", ""))
    required_keys = ("green", "yellow")
    missing = [key for key in required_keys if key not in gpios]
    valid_required_values = {
        key: _coerce_gpio_number(gpios[key])
        for key in required_keys
        if key in gpios and _coerce_gpio_number(gpios[key]) is not None
    }
    invalid = [key for key in required_keys if key in gpios and key not in valid_required_values]

    if not chip:
        return ConfigCheck("gpio", "GPIO buttons", "fail", "No GPIO chip is configured.")

    if missing:
        if gpios:
            return ConfigCheck(
                "gpio",
                "GPIO buttons",
                "warn",
                f"Only part of the button mapping is configured: {_format_mapping(gpios)}.",
            )
        return ConfigCheck("gpio", "GPIO buttons", "fail", "No green/yellow button GPIOs are configured.")

    if invalid:
        return ConfigCheck(
            "gpio",
            "GPIO buttons",
            "fail",
            f"Invalid GPIO number configured for: {', '.join(sorted(invalid))}.",
        )

    green_gpio = cast(int, valid_required_values["green"])
    yellow_gpio = cast(int, valid_required_values["yellow"])
    if green_gpio == yellow_gpio:
        return ConfigCheck("gpio", "GPIO buttons", "fail", "Green and yellow buttons cannot share the same GPIO.")

    probe = _probe_gpio_lines(chip, {"green": green_gpio, "yellow": yellow_gpio})
    if probe.status == "fail":
        return ConfigCheck("gpio", "GPIO buttons", "fail", probe.detail)
    if probe.status == "warn":
        return ConfigCheck(
            "gpio",
            "GPIO buttons",
            "warn",
            f"Green={green_gpio} and yellow={yellow_gpio} are configured on `{_display_value(chip)}`, but active verification was degraded. {probe.detail}",
        )

    return ConfigCheck(
        "gpio",
        "GPIO buttons",
        "ok",
        f"Green={green_gpio} and yellow={yellow_gpio} are configured on `{_display_value(chip)}` and the lines are reachable. {probe.detail}",
    )


def _display_gpio_check(config: TwinrConfig) -> ConfigCheck:
    driver = _clean_config_text(getattr(config, "display_driver", "")).lower() or "hdmi_fbdev"
    if not getattr(config, "display_uses_gpio", False):
        if driver == "hdmi_wayland":
            display_name = _clean_config_text(getattr(config, "display_wayland_display", "")) or "wayland-0"
            runtime_dir = _clean_config_text(getattr(config, "display_wayland_runtime_dir", "")) or None
            socket_path = resolve_wayland_socket(
                display_name,
                configured_runtime_dir=runtime_dir,
            )
            if socket_path is None:
                return ConfigCheck(
                    "display_gpio",
                    "Display output",
                    "fail",
                    f"Wayland socket `{_display_value(display_name)}` was not found. Configure `TWINR_DISPLAY_WAYLAND_RUNTIME_DIR`.",
                )
            if not os.access(socket_path, os.R_OK | os.W_OK):
                return ConfigCheck(
                    "display_gpio",
                    "Display output",
                    "fail",
                    f"Wayland socket `{_display_value(socket_path)}` exists, but the runtime user cannot access it.",
                )
            return ConfigCheck(
                "display_gpio",
                "Display output",
                "ok",
                f"Display driver `{driver}` targets `{_display_value(socket_path)}` and does not require display GPIO pins.",
            )

        fb_path = Path(_clean_config_text(getattr(config, "display_fb_path", "")) or "/dev/fb0").expanduser()
        if not fb_path.exists():
            return ConfigCheck(
                "display_gpio",
                "Display output",
                "fail",
                f"HDMI framebuffer path `{_display_value(fb_path)}` does not exist.",
            )
        if not fb_path.is_char_device():
            return ConfigCheck(
                "display_gpio",
                "Display output",
                "fail",
                f"HDMI framebuffer path `{_display_value(fb_path)}` is not a character device.",
            )
        if not os.access(fb_path, os.W_OK):
            return ConfigCheck(
                "display_gpio",
                "Display output",
                "fail",
                f"HDMI framebuffer path `{_display_value(fb_path)}` is not writable by the runtime user.",
            )
        return ConfigCheck(
            "display_gpio",
            "Display output",
            "ok",
            f"Display driver `{driver}` writes to `{_display_value(fb_path)}` and does not require display GPIO pins.",
        )

    chip = _clean_config_text(getattr(config, "gpio_chip", ""))
    if not chip:
        return ConfigCheck("display_gpio", "Display GPIO", "fail", "Display GPIO requires a configured GPIO chip.")

    display_pins = _raw_display_gpio_map(config)
    missing = [name for name, value in display_pins.items() if value is None]
    if missing:
        return ConfigCheck(
            "display_gpio",
            "Display GPIO",
            "fail",
            f"Missing display GPIO configuration for: {', '.join(missing)}.",
        )

    invalid = [name for name, value in display_pins.items() if _coerce_gpio_number(value) is None]
    if invalid:
        return ConfigCheck(
            "display_gpio",
            "Display GPIO",
            "fail",
            f"Invalid display GPIO number configured for: {', '.join(invalid)}.",
        )

    valid_display_pins = {name: cast(int, _coerce_gpio_number(value)) for name, value in display_pins.items()}
    duplicate_groups = _duplicate_gpio_groups(valid_display_pins)
    if duplicate_groups:
        return ConfigCheck(
            "display_gpio",
            "Display GPIO",
            "fail",
            "; ".join(duplicate_groups),
        )

    button_conflicts = _gpio_map_collisions(valid_display_pins, _valid_button_gpio_map(config), other_label="button")
    if button_conflicts:
        return ConfigCheck(
            "display_gpio",
            "Display GPIO",
            "fail",
            "; ".join(button_conflicts),
        )

    pir_gpio = _coerce_gpio_number(getattr(config, "pir_motion_gpio", None))
    if pir_gpio is not None and pir_gpio in set(valid_display_pins.values()):
        return ConfigCheck(
            "display_gpio",
            "Display GPIO",
            "fail",
            f"Display GPIO collides with PIR GPIO {pir_gpio}.",
        )

    raw_conflicts = config.display_gpio_conflicts()
    if isinstance(raw_conflicts, str):
        conflicts = [raw_conflicts]
    elif isinstance(raw_conflicts, Iterable) and not isinstance(raw_conflicts, Mapping):
        conflicts = [str(conflict) for conflict in raw_conflicts if _clean_config_text(conflict)]
    else:
        conflicts = []
    if conflicts:
        return ConfigCheck(
            "display_gpio",
            "Display GPIO",
            "fail",
            "; ".join(conflicts),
        )

    probe = _probe_gpio_lines(chip, valid_display_pins)
    if probe.status == "fail":
        return ConfigCheck("display_gpio", "Display GPIO", "fail", probe.detail)
    if probe.status == "warn":
        return ConfigCheck(
            "display_gpio",
            "Display GPIO",
            "warn",
            (
                f"Display pins are configured as "
                f"CS={valid_display_pins['CS']}, DC={valid_display_pins['DC']}, "
                f"RESET={valid_display_pins['RESET']}, BUSY={valid_display_pins['BUSY']}, "
                f"but active verification was degraded. {probe.detail}"
            ),
        )
    return ConfigCheck(
        "display_gpio",
        "Display GPIO",
        "ok",
        (
            f"Display pins are configured as "
            f"CS={valid_display_pins['CS']}, DC={valid_display_pins['DC']}, "
            f"RESET={valid_display_pins['RESET']}, BUSY={valid_display_pins['BUSY']}. "
            f"{probe.detail}"
        ),
    )


def _pir_check(config: TwinrConfig) -> ConfigCheck:
    pir_gpio = _coerce_gpio_number(getattr(config, "pir_motion_gpio", None))
    if pir_gpio is None and getattr(config, "pir_motion_gpio", None) is None:
        return ConfigCheck(
            "pir",
            "PIR motion sensor",
            "warn",
            "TWINR_PIR_MOTION_GPIO is not configured.",
        )

    button_lines = set(_valid_button_gpio_map(config).values())
    if pir_gpio in button_lines:
        return ConfigCheck(
            "pir",
            "PIR motion sensor",
            "fail",
            f"PIR GPIO {pir_gpio} collides with a button input.",
        )

    display_lines = set(_valid_display_gpio_map(config).values())
    if pir_gpio in display_lines:
        return ConfigCheck(
            "pir",
            "PIR motion sensor",
            "fail",
            f"PIR GPIO {pir_gpio} collides with a display GPIO.",
        )

    chip = _clean_config_text(getattr(config, "gpio_chip", ""))
    if not chip:
        return ConfigCheck("pir", "PIR motion sensor", "fail", "PIR requires a configured GPIO chip.")

    probe = _probe_gpio_lines(chip, {"pir": cast(int, pir_gpio)})
    if probe.status == "fail":
        return ConfigCheck("pir", "PIR motion sensor", "fail", probe.detail)

    raw_bias = _clean_config_text(getattr(config, "pir_bias", ""))
    normalized_bias = raw_bias.lower()
    if normalized_bias not in _ALLOWED_PIR_BIASES:
        return ConfigCheck(
            "pir",
            "PIR motion sensor",
            "fail",
            f"PIR bias `{_display_value(raw_bias)}` is unsupported.",
        )

    active_high = _coerce_bool(getattr(config, "pir_active_high", False))
    status = "warn" if probe.status == "warn" else "ok"
    detail = (
        f"PIR is configured on GPIO {pir_gpio} "
        f"(active_high={str(active_high).lower()}, bias={normalized_bias}). {probe.detail}"
    )
    return ConfigCheck("pir", "PIR motion sensor", status, detail)


def _runtime_state_check(config: TwinrConfig) -> ConfigCheck:
    raw_path = _clean_config_text(getattr(config, "runtime_state_path", ""))
    if not raw_path:
        return ConfigCheck("runtime_state", "Runtime snapshot", "fail", "TWINR_RUNTIME_STATE_PATH is missing.")

    path = Path(raw_path).expanduser()

    # BREAKING: Relative runtime-state paths are now rejected because service working directories vary across deployments and can silently redirect state files.
    if not path.is_absolute():
        return ConfigCheck(
            "runtime_state",
            "Runtime snapshot",
            "fail",
            f"Snapshot path `{_display_value(path)}` must be absolute.",
        )

    parent = path.parent
    if _path_has_symlink_component(path):
        return ConfigCheck(
            "runtime_state",
            "Runtime snapshot",
            "fail",
            f"Snapshot path `{_display_value(path)}` must not use symlink components.",
        )

    if path.exists() and path.is_dir():
        return ConfigCheck(
            "runtime_state",
            "Runtime snapshot",
            "fail",
            f"Snapshot path `{_display_value(path)}` points to a directory, not a file.",
        )

    if path.exists() and not path.is_file():
        return ConfigCheck(
            "runtime_state",
            "Runtime snapshot",
            "fail",
            f"Snapshot path `{_display_value(path)}` is not a regular file.",
        )

    if parent.exists() and not parent.is_dir():
        return ConfigCheck(
            "runtime_state",
            "Runtime snapshot",
            "fail",
            f"Snapshot parent `{_display_value(parent)}` is not a directory.",
        )

    insecure_parent = _insecure_permissions_detail(parent if parent.exists() else _nearest_existing_ancestor(parent))
    if insecure_parent is not None:
        return ConfigCheck(
            "runtime_state",
            "Runtime snapshot",
            "fail",
            insecure_parent,
        )

    if path.exists():
        insecure_path = _insecure_permissions_detail(path)
        if insecure_path is not None:
            return ConfigCheck(
                "runtime_state",
                "Runtime snapshot",
                "fail",
                insecure_path,
            )

    if not parent.exists():
        nearest_existing = _nearest_existing_ancestor(parent)
        if not nearest_existing.exists() or not nearest_existing.is_dir():
            return ConfigCheck(
                "runtime_state",
                "Runtime snapshot",
                "fail",
                f"Snapshot ancestor `{_display_value(nearest_existing)}` is not a writable directory.",
            )
        if not os.access(nearest_existing, os.W_OK | os.X_OK):
            return ConfigCheck(
                "runtime_state",
                "Runtime snapshot",
                "fail",
                f"Snapshot directory `{_display_value(parent)}` cannot be created from `{_display_value(nearest_existing)}`.",
            )
        return ConfigCheck(
            "runtime_state",
            "Runtime snapshot",
            "warn",
            f"Snapshot directory `{_display_value(parent)}` does not exist yet. It will be created on first write.",
        )

    if not os.access(parent, os.W_OK | os.X_OK):
        return ConfigCheck(
            "runtime_state",
            "Runtime snapshot",
            "fail",
            f"Snapshot directory `{_display_value(parent)}` is not writable.",
        )

    if path.exists() and not os.access(path, os.W_OK):
        return ConfigCheck(
            "runtime_state",
            "Runtime snapshot",
            "fail",
            f"Snapshot file `{_display_value(path)}` is not writable.",
        )

    return ConfigCheck(
        "runtime_state",
        "Runtime snapshot",
        "ok",
        f"Snapshot path is `{_display_value(path)}`.",
    )


def _binary_or_path_exists(value: str) -> bool:
    return _resolve_executable(value) is not None


def _resolve_executable(command: str) -> str | None:
    candidate = _clean_config_text(command)
    if not candidate:
        return None

    resolved = which(candidate)
    if resolved:
        return str(Path(resolved).resolve())

    path = Path(candidate).expanduser()
    if path.exists() and path.is_file() and os.access(path, os.X_OK):
        return str(path.resolve())
    return None


def _run_command(args: Sequence[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    if not args:
        raise ValueError("Command arguments must not be empty.")

    resolved = _resolve_executable(str(args[0]))
    if resolved is None:
        raise FileNotFoundError(f"Executable `{_display_value(args[0])}` is not available.")

    env = {
        key: value
        for key, value in os.environ.items()
        if key not in _TOOL_ENV_DENYLIST
    }
    env.setdefault("LC_ALL", "C")

    normalized_args = [resolved, *(str(argument) for argument in args[1:])]
    return subprocess.run(
        normalized_args,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout,
        env=env,
    )


def _call_with_timeout(
    callback: Callable[[], _T],
    *,
    timeout: float,
    operation_label: str,
) -> _T:
    result_queue: Queue[tuple[bool, object]] = Queue(maxsize=1)

    def _runner() -> None:
        try:
            result_queue.put((True, callback()))
        except BaseException as exc:  # noqa: BLE001
            result_queue.put((False, exc))

    thread = threading.Thread(target=_runner, name=f"twinr-check-{operation_label}", daemon=True)
    thread.start()
    try:
        ok, payload = result_queue.get(timeout=timeout)
    except Empty as exc:
        raise TimeoutError(f"{operation_label} exceeded {timeout:.1f}s") from exc

    if ok:
        return cast(_T, payload)
    raise cast(BaseException, payload)


def _probe_alsa_device(binary: str, device: str) -> _ProbeOutcome:
    resolved = _resolve_executable(binary)
    if resolved is None:
        return _ProbeOutcome(
            "warn",
            f"Device `{_display_value(device)}` is configured, but `{_display_value(binary)}` is not on PATH.",
        )

    try:
        pcm_result = _run_command([binary, "-L"], timeout=_AUDIO_CHECK_TIMEOUT_SECONDS)
        hw_result = _run_command([binary, "-l"], timeout=_AUDIO_CHECK_TIMEOUT_SECONDS)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return _ProbeOutcome(
            "warn",
            f"Device `{_display_value(device)}` is configured, but ALSA probing via `{_display_value(resolved)}` failed: {exc}",
        )

    if pcm_result.returncode != 0 and hw_result.returncode != 0:
        return _ProbeOutcome(
            "warn",
            f"Device `{_display_value(device)}` is configured, but ALSA inventory probing failed.",
        )

    pcm_visible = _alsa_pcm_list_contains(device, pcm_result.stdout)
    hw_visible = _alsa_hw_list_contains(device, hw_result.stdout)
    if pcm_visible or hw_visible:
        return _ProbeOutcome(
            "ok",
            f"ALSA inventory confirms `{_display_value(device)}`.",
        )

    if pcm_result.returncode == 0 and hw_result.returncode == 0:
        return _ProbeOutcome(
            "fail",
            f"Device `{_display_value(device)}` is configured, but it was not found in ALSA PCM or hardware listings.",
        )

    return _ProbeOutcome(
        "warn",
        f"Device `{_display_value(device)}` is configured, but ALSA probing was only partially available and readiness could not be fully confirmed.",
    )


def _detect_rpicam_binary(configured: str) -> str | None:
    if configured:
        resolved = _resolve_executable(configured)
        if resolved is not None:
            return resolved
        return None

    for candidate in ("rpicam-hello", "rpicam-still", "libcamera-hello", "libcamera-still"):
        resolved = _resolve_executable(candidate)
        if resolved is not None:
            return resolved
    return None


def _probe_gpio_lines(chip: str, lines: Mapping[str, int]) -> _ProbeOutcome:
    chip_path = _normalize_gpio_chip_path(chip)
    if chip_path is None:
        return _ProbeOutcome(
            "fail",
            f"GPIO chip `{_display_value(chip)}` is invalid. Use `/dev/gpiochipN`, `gpiochipN`, or `N`.",
        )

    if not chip_path.exists():
        return _ProbeOutcome(
            "fail",
            f"GPIO chip `{_display_value(chip_path)}` does not exist.",
        )
    if not chip_path.is_char_device():
        return _ProbeOutcome(
            "fail",
            f"GPIO chip `{_display_value(chip_path)}` is not a character device.",
        )
    if not os.access(chip_path, os.R_OK):
        return _ProbeOutcome(
            "fail",
            f"GPIO chip `{_display_value(chip_path)}` is not readable by the runtime user.",
        )

    try:
        import gpiod  # type: ignore[import-not-found]  # pylint: disable=import-error
    except Exception:  # noqa: BLE001
        gpiod = None

    if gpiod is not None:
        try:
            chip_handle = gpiod.Chip(str(chip_path))
            warnings: list[str] = []
            with chip_handle:
                for name, offset in lines.items():
                    info = chip_handle.get_line_info(offset)
                    consumer = _clean_config_text(getattr(info, "consumer", ""))
                    used = bool(getattr(info, "used", False))
                    if used and consumer.lower() not in _GPIOD_LINE_BUSY_CONSUMER_ALLOWLIST:
                        warnings.append(
                            f"GPIO {offset} for `{_display_value(name)}` is already in use by `{_display_value(consumer, unset='another consumer')}`."
                        )
            if warnings:
                return _ProbeOutcome("warn", " ".join(warnings))
            return _ProbeOutcome(
                "ok",
                f"GPIO chip `{_display_value(chip_path)}` exposes the configured line offsets.",
            )
        except Exception as exc:  # noqa: BLE001
            return _ProbeOutcome(
                "warn",
                f"libgpiod probing on `{_display_value(chip_path)}` failed: {exc}",
            )

    gpioinfo_path = _resolve_executable("gpioinfo")
    if gpioinfo_path is None:
        return _ProbeOutcome(
            "warn",
            f"GPIO chip `{_display_value(chip_path)}` exists, but neither Python libgpiod nor `gpioinfo` is available for active probing.",
        )

    args = ["gpioinfo", "-c", str(chip_path), *(str(offset) for offset in lines.values())]
    try:
        result = _run_command(args, timeout=_GPIO_CHECK_TIMEOUT_SECONDS)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return _ProbeOutcome(
            "warn",
            f"`gpioinfo` probing failed for `{_display_value(chip_path)}`: {exc}",
        )

    if result.returncode != 0:
        return _ProbeOutcome(
            "fail",
            _command_error_detail(result, default=f"`gpioinfo` could not inspect `{_display_value(chip_path)}`."),
        )

    output = _combine_command_output(result).lower()
    warnings = []
    for offset in lines.values():
        if f"line {offset}:" not in output:
            return _ProbeOutcome(
                "fail",
                f"GPIO line {offset} was not reported on chip `{_display_value(chip_path)}`.",
            )
        if "used" in output and f"line {offset}:" in output:
            match = re.search(
                rf"line\s+{offset}:\s+.*?(?:consumer=(?P<consumer>[^,\n]+))?",
                output,
                flags=re.IGNORECASE,
            )
            if match is not None:
                consumer = _clean_config_text(match.groupdict().get("consumer", ""))
                if consumer and consumer.lower() not in _GPIOD_LINE_BUSY_CONSUMER_ALLOWLIST:
                    warnings.append(
                        f"GPIO {offset} appears busy on `{_display_value(chip_path)}` (consumer `{_display_value(consumer)}`)."
                    )

    if warnings:
        return _ProbeOutcome("warn", " ".join(warnings))
    return _ProbeOutcome(
        "ok",
        f"`gpioinfo` confirmed the configured line offsets on `{_display_value(chip_path)}`.",
    )


def _normalize_gpio_chip_path(chip: str) -> Path | None:
    candidate = _clean_config_text(chip)
    if not candidate:
        return None

    if candidate.isdigit():
        return Path(f"/dev/gpiochip{candidate}")

    if re.fullmatch(r"gpiochip\d+", candidate):
        return Path("/dev") / candidate

    path = Path(candidate).expanduser()
    if path.name and re.fullmatch(r"gpiochip\d+", path.name):
        return path

    return None


def _combine_command_output(result: subprocess.CompletedProcess[str]) -> str:
    stdout = _clean_config_text(result.stdout)
    stderr = _clean_config_text(result.stderr)
    if stdout and stderr:
        return f"{stdout} {stderr}"
    return stdout or stderr


def _command_error_detail(result: subprocess.CompletedProcess[str], *, default: str) -> str:
    output = _sanitize_detail(_combine_command_output(result), default=default)
    return output if output else default


def _trim_detail(text: str, *, limit: int = 180) -> str:
    cleaned = _sanitize_detail(text, default="")
    if not cleaned:
        return ""
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: limit - 3]}..."


def _insecure_permissions_detail(path: Path) -> str | None:
    try:
        mode = path.stat().st_mode
    except OSError:
        return None

    if stat.S_ISDIR(mode):
        if mode & (stat.S_IWGRP | stat.S_IWOTH):
            return f"Snapshot directory `{_display_value(path)}` is group/world-writable and is not safe for runtime state."
        return None

    if stat.S_ISREG(mode):
        if mode & (stat.S_IWGRP | stat.S_IWOTH):
            return f"Snapshot file `{_display_value(path)}` is group/world-writable and is not safe for runtime state."
        return None

    return f"Snapshot path `{_display_value(path)}` is not a regular file or directory."


def _run_check_safely(
    key: str,
    label: str,
    check_fn: Callable[[TwinrConfig], ConfigCheck],
    config: TwinrConfig,
) -> ConfigCheck:
    try:
        result = check_fn(config)
    except Exception as exc:
        return ConfigCheck(
            key,
            label,
            "fail",
            f"{label} check could not be completed because the configuration could not be read safely ({exc.__class__.__name__}).",
        )
    if not isinstance(result, ConfigCheck):
        return ConfigCheck(
            key,
            label,
            "fail",
            f"{label} check returned an invalid result type `{type(result).__name__}`.",
        )
    return result


def _normalize_status(value: object) -> str:
    text = _clean_config_text(value).lower()
    if text in _VALID_STATUSES:
        return text
    return "fail"


def _clean_config_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _sanitize_identifier(value: object) -> str:
    text = _clean_config_text(value)
    if not text:
        return "unknown"
    return _sanitize_detail(text, default="unknown").replace(" ", "_")


def _sanitize_detail(value: object, *, default: str) -> str:
    if value is None:
        return default
    text = str(value)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = "".join(character if character.isprintable() else " " for character in text)
    text = " ".join(text.split())
    if not text:
        return default
    if len(text) > 400:
        return f"{text[:397]}..."
    return text


def _display_value(value: object, unset: str = "<unset>") -> str:
    text = _clean_config_text(value)
    if not text:
        return unset
    return _sanitize_detail(text, default=unset)


def _respeaker_firmware_label(version: tuple[int, int, int] | None) -> str | None:
    if version is None or len(version) != 3:
        return None
    return ".".join(str(int(part)) for part in version)


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = _clean_config_text(value).lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return bool(value)


def _coerce_gpio_number(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None

    text = _clean_config_text(value)
    if not text:
        return None
    if text.isdigit():
        return int(text)
    return None


def _is_valid_gpio_number(value: object) -> bool:
    return _coerce_gpio_number(value) is not None


def _raw_display_gpio_map(config: TwinrConfig) -> dict[str, object]:
    return {
        "CS": getattr(config, "display_cs_gpio", None),
        "DC": getattr(config, "display_dc_gpio", None),
        "RESET": getattr(config, "display_reset_gpio", None),
        "BUSY": getattr(config, "display_busy_gpio", None),
    }


def _valid_display_gpio_map(config: TwinrConfig) -> dict[str, int]:
    valid: dict[str, int] = {}
    for name, value in _raw_display_gpio_map(config).items():
        gpio = _coerce_gpio_number(value)
        if gpio is not None:
            valid[name] = gpio
    return valid


def _valid_button_gpio_map(config: TwinrConfig) -> dict[str, int]:
    raw_gpios = getattr(config, "button_gpios", {})
    if not isinstance(raw_gpios, Mapping):
        return {}

    valid: dict[str, int] = {}
    for key, value in raw_gpios.items():
        gpio = _coerce_gpio_number(value)
        if gpio is not None:
            valid[str(key)] = gpio
    return valid


def _duplicate_gpio_groups(gpios: Mapping[str, int]) -> list[str]:
    by_gpio: dict[int, list[str]] = {}
    for name, gpio in gpios.items():
        by_gpio.setdefault(gpio, []).append(name)
    return [
        f"{'/'.join(sorted(names))} share GPIO {gpio}"
        for gpio, names in sorted(by_gpio.items())
        if len(names) > 1
    ]


def _gpio_map_collisions(
    gpios: Mapping[str, int],
    other_gpios: Mapping[str, int],
    *,
    other_label: str,
) -> list[str]:
    collisions: list[str] = []
    for name, gpio in gpios.items():
        for other_name, other_gpio in other_gpios.items():
            if gpio == other_gpio:
                collisions.append(f"{name} GPIO {gpio} collides with {other_label} `{_display_value(other_name)}`")
    return collisions


def _format_mapping(mapping: Mapping[object, object]) -> str:
    return ", ".join(
        f"{_display_value(key)}={_display_value(value)}"
        for key, value in sorted(mapping.items(), key=lambda item: str(item[0]))
    )


def _nearest_existing_ancestor(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _path_has_symlink_component(path: Path) -> bool:
    for candidate in (path, *path.parents):
        if candidate.exists() and candidate.is_symlink():
            return True
    return False


def _alsa_pcm_list_contains(device: str, pcm_listing: str) -> bool:
    normalized = _clean_config_text(device).lower()
    if not normalized:
        return False

    for line in pcm_listing.splitlines():
        line = line.strip().lower()
        if not line:
            continue
        if line == normalized:
            return True
        if line.startswith(f"{normalized}:") or normalized in {segment.strip() for segment in line.split(",")}:
            return True
    return False


def _alsa_hw_list_contains(device: str, hw_listing: str) -> bool:
    normalized = _clean_config_text(device).lower()
    if not normalized:
        return False

    if normalized.startswith(("hw:", "plughw:")):
        spec = normalized.split(":", 1)[1]
        parts = [part.strip() for part in spec.split(",", 1)]
        if len(parts) == 2 and all(part.isdigit() for part in parts):
            card_index, device_index = parts
            lowered = hw_listing.lower()
            return f"card {card_index}:" in lowered and f"device {device_index}:" in lowered

    return False
