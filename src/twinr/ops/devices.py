"""Collect a bounded operational overview of Twinr's local devices.

This module combines live system probes with recent ops and self-test events
to build dashboard-friendly device status snapshots.
"""

# CHANGELOG: 2026-03-30
# BUG-1: Fix false-positive audio health by validating configured ALSA/PipeWire devices instead of treating any reported capture card as success.
# BUG-2: Fix camera false negatives on Raspberry Pi OS Bookworm+ by probing the supported rpicam/libcamera/Picamera2 stack and by defaulting blank ffmpeg paths to the system ffmpeg.
# BUG-3: Fix printer live-state misreporting by using lpstat's accepting/device views for current queue state instead of inferring it from lpoptions defaults.
# BUG-4: Fix impossible PIR/button GPIO values reporting as healthy by bounding Raspberry Pi 4 BCM GPIO validation to the externally exposed range.
# SEC-1: Stop resolving bare executables through an attacker-controlled PATH; only trusted system paths or explicit absolute executable paths are accepted.
# SEC-2: Reject relative executable paths and force a trusted PATH plus DEVNULL stdin for subprocess probes to reduce path-hijack and probe-hang risk on shared Pis.
# IMP-1: Run independent device probes concurrently so the dashboard stays responsive under slow peripherals and command timeouts.
# IMP-2: Upgrade camera probing for Bookworm-era rpicam/libcamera/Picamera2 and audio probing for PipeWire/WirePlumber-era Raspberry Pi deployments.

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.util import find_spec
from pathlib import Path
import os
import re
import subprocess
from urllib.parse import urlsplit, urlunsplit

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.respeaker import capture_respeaker_primitive_snapshot, config_targets_respeaker
from twinr.hardware.respeaker.derived_signals import derive_respeaker_signal_state
from twinr.ops.events import TwinrOpsEventStore
from twinr.ops.locks import loop_lock_owner


_COMMAND_TIMEOUT_SECONDS = 2.0
_CAMERA_PROBE_TIMEOUT_SECONDS = 1.5
_PIPEWIRE_PROBE_TIMEOUT_SECONDS = 1.0
_MAX_CONCURRENT_PROBES = 4
_GPIO_MIN_BCM = 0
_GPIO_MAX_BCM_PI4 = 27
_TRUSTED_EXECUTABLE_DIRS = (
    "/usr/local/sbin",
    "/usr/local/bin",
    "/usr/sbin",
    "/usr/bin",
    "/sbin",
    "/bin",
)
_TRUSTED_PATH = os.pathsep.join(_TRUSTED_EXECUTABLE_DIRS)
_OPS_HISTORY_UNAVAILABLE = "ops history unavailable"
_CAMERA_HEADER_RE = re.compile(r"^\s*(\d+)\s*:\s*(.+?)\s*$")
_DEVICE_URI_RE = re.compile(r"^\s*device for .*?:\s*(.+?)\s*$", re.IGNORECASE)
_AUDIO_CARD_ALIAS_RE = re.compile(r"card\s+(\d+).*?device\s+(\d+)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class DeviceFact:
    """Represent one short key/value fact shown for a device probe."""

    label: str
    value: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DeviceStatus:
    """Represent the current ops-facing state of one device category."""

    key: str
    label: str
    status: str
    summary: str
    facts: tuple[DeviceFact, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["facts"] = [fact.to_dict() for fact in self.facts]
        return payload


@dataclass(frozen=True, slots=True)
class DeviceOverview:
    """Collect the device statuses captured for one ops snapshot."""

    captured_at: str
    devices: tuple[DeviceStatus, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "captured_at": self.captured_at,
            "devices": [device.to_dict() for device in self.devices],
        }


@dataclass(frozen=True, slots=True)
class _CommandResult:
    ok: bool
    returncode: int | None
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class _SelfTestSnapshot:
    status: str
    created_at: str
    message: str


def collect_device_overview(
    config: TwinrConfig,
    *,
    event_store: TwinrOpsEventStore | None = None,
) -> DeviceOverview:
    """Collect a dashboard-ready overview of Twinr device health.

    Args:
        config: Twinr runtime configuration to probe against.
        event_store: Optional event store override used to enrich the overview
            with recent motion and self-test evidence.

    Returns:
        A ``DeviceOverview`` containing bounded statuses for printer, camera,
        audio, PIR, and button hardware.
    """

    latest_self_tests: dict[str, _SelfTestSnapshot] | None
    last_motion_at: str | None
    try:
        store = event_store if event_store is not None else TwinrOpsEventStore.from_config(config)
        events = _normalize_event_entries(store.tail(limit=200))
        latest_self_tests = _latest_self_tests(events)
        last_motion_at = _latest_motion_timestamp(events)
    except Exception:
        latest_self_tests = None
        last_motion_at = _OPS_HISTORY_UNAVAILABLE

    collectors: tuple[tuple[str, str, Callable[[], DeviceStatus]], ...] = (
        ("printer", "Printer", lambda: _collect_printer_status(config, latest_self_tests)),
        ("camera", "Camera", lambda: _collect_camera_status(config, latest_self_tests)),
        ("audio_input", "Primary Audio", lambda: _collect_primary_audio_status(config, latest_self_tests)),
        ("proactive_audio", "Background Audio", lambda: _collect_proactive_audio_status(config, latest_self_tests)),
        ("respeaker", "ReSpeaker XVF3800", lambda: _collect_respeaker_status(config, latest_self_tests)),
        ("pir", "PIR Motion", lambda: _collect_pir_status(config, latest_self_tests, last_motion_at=last_motion_at)),
        ("buttons", "Buttons", lambda: _collect_button_status(config, latest_self_tests)),
    )

    return DeviceOverview(
        captured_at=_captured_at(),
        devices=_collect_device_statuses_concurrently(collectors),
    )


def _collect_device_statuses_concurrently(
    collectors: tuple[tuple[str, str, Callable[[], DeviceStatus]], ...],
) -> tuple[DeviceStatus, ...]:
    if not collectors:
        return ()

    max_workers = max(1, min(len(collectors), _MAX_CONCURRENT_PROBES))
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="twinr-device-overview") as executor:
        futures = [
            executor.submit(_safe_collect_device_status, key, label, collector)
            for key, label, collector in collectors
        ]
        return tuple(future.result() for future in futures)


def _collect_printer_status(
    config: TwinrConfig,
    latest_self_tests: dict[str, _SelfTestSnapshot] | None,
) -> DeviceStatus:
    queue = _strip_text(getattr(config, "printer_queue", ""))
    if not queue:
        return DeviceStatus(
            key="printer",
            label="Printer",
            status="fail",
            summary="No printer queue is configured.",
            facts=(DeviceFact("Queue", "missing"),),
        )

    lpstat_path = _resolve_executable("lpstat")
    if lpstat_path is None:
        return DeviceStatus(
            key="printer",
            label="Printer",
            status="warn",
            summary="Printer queue is configured, but `lpstat` is unavailable on this machine.",
            facts=(
                DeviceFact("Queue", queue),
                DeviceFact("Accepting jobs", "unknown"),
                DeviceFact("Device URI", "unknown"),
                DeviceFact("Paper status", "unknown on the current path"),
                DeviceFact("Last self-test", _self_test_label(latest_self_tests, "printer")),
            ),
        )

    detail_result = _run_command([lpstat_path, "-l", "-p", queue])
    accepting_result = _run_command([lpstat_path, "-a", queue])
    device_result = _run_command([lpstat_path, "-v", queue])

    detail_visible = detail_result.ok and detail_result.returncode == 0
    accepting_jobs, accepting_detail = _parse_lpstat_accepting_detail(accepting_result.stdout or accepting_result.stderr)
    device_uri = _parse_lpstat_device_uri(device_result.stdout)

    if not detail_visible and accepting_jobs == "unknown" and not device_uri:
        detail = _display_text(
            detail_result.stderr or detail_result.stdout or accepting_result.stderr or "Queue is not visible to CUPS.",
        )
        return DeviceStatus(
            key="printer",
            label="Printer",
            status="warn",
            summary=f"Queue `{queue}` is configured but not visible to CUPS right now.",
            facts=(
                DeviceFact("Queue", queue),
                DeviceFact("CUPS detail", detail),
                DeviceFact("Paper status", "unknown on the current raw USB/CUPS path"),
                DeviceFact("Last self-test", _self_test_label(latest_self_tests, "printer")),
            ),
        )

    lpstat_info = _parse_lpstat_printer_detail(detail_result.stdout)
    state_reasons = _display_text(
        lpstat_info.get("printer_state_reasons") or lpstat_info.get("alerts") or "unknown",
        default="unknown",
    )
    headline = _display_text(lpstat_info.get("headline") or f"Queue `{queue}` is visible to CUPS.")
    description = _display_text(lpstat_info.get("description") or "—")
    connection = _display_text(lpstat_info.get("connection") or "—")
    location = _display_text(lpstat_info.get("location") or "—")
    enabled_label = "no" if "disabled" in headline.lower() else "yes"

    status = "ok"
    if not detail_visible or accepting_jobs == "no" or state_reasons not in {"none", "unknown"} or enabled_label == "no":
        status = "warn"

    return DeviceStatus(
        key="printer",
        label="Printer",
        status=status,
        summary=headline,
        facts=(
            DeviceFact("Queue", queue),
            DeviceFact("Description", description),
            DeviceFact("Location", location),
            DeviceFact("Connection", connection),
            DeviceFact("Enabled", enabled_label),
            DeviceFact("Accepting jobs", accepting_jobs),
            DeviceFact("Accepting detail", accepting_detail),
            DeviceFact("State reasons", state_reasons),
            DeviceFact("Device URI", _redact_device_uri(device_uri or "—")),
            DeviceFact("Paper status", "unknown on the current raw USB/CUPS path"),
            DeviceFact("Last self-test", _self_test_label(latest_self_tests, "printer")),
        ),
        notes=(
            "Twinr can see queue and live CUPS state here, but it cannot prove real paper output from this printer path.",
        ),
    )


def _collect_camera_status(
    config: TwinrConfig,
    latest_self_tests: dict[str, _SelfTestSnapshot] | None,
) -> DeviceStatus:
    device_value = _strip_text(getattr(config, "camera_device", ""))
    device_path = Path(device_value) if device_value else None
    device_present = _path_is_char_device(device_path)

    ffmpeg_candidate = _strip_text(getattr(config, "camera_ffmpeg_path", "")) or "ffmpeg"
    ffmpeg_path = _resolve_executable(ffmpeg_candidate)
    camera_probe_path = _resolve_executable("rpicam-hello") or _resolve_executable("libcamera-hello")
    detected_cameras, camera_probe_detail = _probe_camera_inventory(camera_probe_path)
    picamera2_available = _python_module_available("picamera2")
    modern_stack_ready = camera_probe_path is not None or picamera2_available

    notes: list[str] = []
    if detected_cameras and not device_present:
        notes.append(
            "rpicam/libcamera can see a camera even though the configured fixed V4L2 node is missing; this is common with CSI/libcamera-first setups.",
        )
    if device_present and not modern_stack_ready and ffmpeg_path is None:
        notes.append(
            "A V4L2 device exists, but Twinr cannot see rpicam/libcamera, Picamera2, or ffmpeg tooling on this runtime.",
        )

    if device_present and (modern_stack_ready or ffmpeg_path is not None):
        status = "ok"
        summary = f"Camera device `{device_value}` is present."
    elif detected_cameras and modern_stack_ready and device_value:
        probe_name = Path(camera_probe_path).name if camera_probe_path else "camera probe"
        status = "warn"
        summary = f"Camera(s) are detected via `{probe_name}`, but configured device `{device_value}` is missing or is not a character device."
    elif detected_cameras and modern_stack_ready:
        status = "warn"
        summary = "Camera hardware is detected via rpicam/libcamera, but no fixed V4L2 camera device is configured."
    elif device_value and not device_present:
        status = "warn"
        summary = f"Camera device `{device_value}` is missing or is not a character device."
    elif device_present:
        status = "warn"
        summary = f"Camera device `{device_value}` exists, but no supported userspace camera stack is visible."
    else:
        status = "fail"
        summary = "No camera device is configured and rpicam/libcamera did not detect a camera."

    return DeviceStatus(
        key="camera",
        label="Camera",
        status=status,
        summary=summary,
        facts=(
            DeviceFact("Device", device_value or "not configured"),
            DeviceFact("Present", "yes" if device_present else "no"),
            DeviceFact("Resolution", f"{config.camera_width}x{config.camera_height}"),
            DeviceFact("Framerate", f"{config.camera_framerate} fps"),
            DeviceFact("Input format", _display_text(getattr(config, "camera_input_format", "") or "default")),
            DeviceFact("rpicam/libcamera probe", Path(camera_probe_path).name if camera_probe_path else "missing"),
            DeviceFact("Detected cameras", _join_limited(detected_cameras, default=camera_probe_detail or "none detected")),
            DeviceFact("Picamera2", "available" if picamera2_available else "missing"),
            DeviceFact("ffmpeg", ffmpeg_path or f"missing: {_display_text(ffmpeg_candidate)}"),
            DeviceFact("Reference image", _display_text(getattr(config, "vision_reference_image_path", "") or "not configured")),
            DeviceFact("Last self-test", _self_test_label(latest_self_tests, "camera")),
        ),
        notes=tuple(notes),
    )


def _collect_primary_audio_status(
    config: TwinrConfig,
    latest_self_tests: dict[str, _SelfTestSnapshot] | None,
) -> DeviceStatus:
    arecord_path = _resolve_executable("arecord")
    aplay_path = _resolve_executable("aplay")
    wpctl_path = _resolve_executable("wpctl")

    capture_devices = _list_arecord_capture_devices(arecord_path)
    capture_pcms = _list_audio_pcm_targets(arecord_path)
    playback_pcms = _list_audio_pcm_targets(aplay_path)
    pipewire_ready = _probe_pipewire_status(wpctl_path) or _has_pipewire_pcm(capture_pcms) or _has_pipewire_pcm(playback_pcms)

    configured_input = _display_text(getattr(config, "audio_input_device", "") or "not configured")
    configured_output = _display_text(getattr(config, "audio_output_device", "") or "not configured")
    input_match = _configured_audio_device_matches(
        getattr(config, "audio_input_device", ""),
        capture_devices,
        capture_pcms,
        pipewire_ready=pipewire_ready,
    )
    output_match = _configured_audio_device_matches(
        getattr(config, "audio_output_device", ""),
        (),
        playback_pcms,
        pipewire_ready=pipewire_ready,
    )

    if arecord_path is None:
        status = "warn"
        summary = "Audio capture cannot be inspected because `arecord` is unavailable."
    elif not capture_devices and not capture_pcms and not pipewire_ready:
        status = "warn"
        summary = "No ALSA or PipeWire capture targets were reported."
    elif input_match is False or output_match is False:
        status = "warn"
        summary = "Configured audio devices are not fully visible in ALSA/PipeWire right now."
    else:
        status = "ok"
        summary = f"{max(len(capture_devices), len(capture_pcms), 1)} capture target(s) reported by ALSA/PipeWire."

    return DeviceStatus(
        key="audio_input",
        label="Primary Audio",
        status=status,
        summary=summary,
        facts=(
            DeviceFact("Configured input", configured_input),
            DeviceFact("Configured output", configured_output),
            DeviceFact("Input visible", _match_label(input_match)),
            DeviceFact("Output visible", _match_label(output_match)),
            DeviceFact("Sample rate", f"{config.audio_sample_rate} Hz"),
            DeviceFact("Channels", str(config.audio_channels)),
            DeviceFact("ALSA capture cards", _join_limited(capture_devices, default="none reported")),
            DeviceFact("ALSA capture PCMs", _join_limited(capture_pcms, default="none reported")),
            DeviceFact("ALSA playback PCMs", _join_limited(playback_pcms, default="none reported")),
            DeviceFact("PipeWire/WirePlumber", "available" if pipewire_ready else "not detected"),
            DeviceFact("Last self-test", _self_test_label(latest_self_tests, "mic")),
        ),
    )


def _collect_proactive_audio_status(
    config: TwinrConfig,
    latest_self_tests: dict[str, _SelfTestSnapshot] | None,
) -> DeviceStatus:
    proactive_device = _strip_text(getattr(config, "proactive_audio_input_device", ""))
    primary_device = _strip_text(getattr(config, "audio_input_device", ""))
    enabled = bool(getattr(config, "proactive_audio_enabled", False)) or bool(proactive_device)
    if not enabled:
        return DeviceStatus(
            key="proactive_audio",
            label="Background Audio",
            status="muted",
            summary="The proactive background-microphone path is disabled.",
            facts=(
                DeviceFact("Configured device", "disabled"),
                DeviceFact("Last self-test", _self_test_label(latest_self_tests, "proactive_mic")),
            ),
        )

    arecord_path = _resolve_executable("arecord")
    wpctl_path = _resolve_executable("wpctl")
    capture_devices = _list_arecord_capture_devices(arecord_path)
    capture_pcms = _list_audio_pcm_targets(arecord_path)
    pipewire_ready = _probe_pipewire_status(wpctl_path) or _has_pipewire_pcm(capture_pcms)

    target_device = proactive_device or primary_device
    device_label = proactive_device or (f"reuse primary input ({primary_device})" if primary_device else "missing")
    device_match = _configured_audio_device_matches(
        target_device,
        capture_devices,
        capture_pcms,
        pipewire_ready=pipewire_ready,
    )

    if arecord_path is None and not pipewire_ready:
        status = "warn"
        summary = "Background-audio capture is enabled, but ALSA/PipeWire capture tooling is unavailable."
    elif not target_device:
        status = "warn"
        summary = "Background-audio capture is enabled, but no input device is configured."
    elif device_match is False:
        status = "warn"
        summary = f"Background-audio capture is enabled, but device `{target_device}` is not visible in ALSA/PipeWire."
    else:
        status = "ok"
        summary = "Background-audio capture is configured."

    return DeviceStatus(
        key="proactive_audio",
        label="Background Audio",
        status=status,
        summary=summary,
        facts=(
            DeviceFact("Configured device", device_label),
            DeviceFact("Device visible", _match_label(device_match)),
            DeviceFact("Sample window", f"{config.proactive_audio_sample_ms} ms"),
            DeviceFact("Distress detector", "enabled" if config.proactive_audio_distress_enabled else "disabled"),
            DeviceFact("ALSA capture PCMs", _join_limited(capture_pcms, default="none reported")),
            DeviceFact("PipeWire/WirePlumber", "available" if pipewire_ready else "not detected"),
            DeviceFact("Last self-test", _self_test_label(latest_self_tests, "proactive_mic")),
        ),
    )


def _collect_respeaker_status(
    config: TwinrConfig,
    latest_self_tests: dict[str, _SelfTestSnapshot] | None,
) -> DeviceStatus:
    primary_device = _strip_text(getattr(config, "audio_input_device", ""))
    proactive_device = _strip_text(getattr(config, "proactive_audio_input_device", ""))
    configured = config_targets_respeaker(primary_device, proactive_device)
    snapshot = capture_respeaker_primitive_snapshot()
    probe = snapshot.probe

    usb_label = "not detected"
    if probe.usb_device is not None:
        usb_label = probe.usb_device.description or probe.usb_device.raw_line
    elif not probe.lsusb_available:
        usb_label = "lsusb unavailable"

    alsa_label = "not detected"
    if probe.capture_device is not None:
        capture = probe.capture_device
        alsa_label = capture.hw_identifier or capture.card_label or capture.raw_line
    elif not probe.arecord_available:
        alsa_label = "arecord unavailable"

    status = "muted"
    summary = "No ReSpeaker XVF3800 is configured or detected."
    notes: list[str] = []
    if probe.capture_ready and snapshot.host_control_ready:
        status = "ok"
        summary = "ReSpeaker XVF3800 is visible, ALSA capture is ready, and host-control primitives are readable."
    elif probe.capture_ready:
        status = "warn"
        summary = "ReSpeaker XVF3800 is ALSA capture-ready, but host-control primitives are degraded."
        if snapshot.transport.requires_elevated_permissions:
            notes.append("The runtime user likely lacks the USB permissions required for XVF3800 host-control reads.")
    elif probe.usb_visible and probe.arecord_available:
        status = "warn"
        summary = "ReSpeaker XVF3800 is USB-visible, but no ALSA capture card is ready."
    elif probe.usb_visible:
        status = "warn"
        summary = "ReSpeaker XVF3800 is USB-visible, but ALSA capture readiness could not be verified."
    elif configured:
        status = "fail"
        summary = "Twinr is configured for ReSpeaker XVF3800 capture, but the device is not detected."

    firmware_label = _respeaker_firmware_label(snapshot.firmware_version)
    derived = derive_respeaker_signal_state(snapshot.direction, assistant_output_active=None)
    azimuth_label = _display_optional_number(snapshot.direction.doa_degrees, default="unknown")
    direction_confidence_label = _display_optional_number(derived.direction_confidence, default="unknown")
    beam_energy_label = _display_float_tuple(snapshot.direction.beam_speech_energies)
    gpo_label = _display_int_tuple(snapshot.mute.gpo_logic_levels)
    host_control_label = "yes" if snapshot.host_control_ready else "no"
    transport_reason = _display_text(snapshot.transport.reason, default="none")

    if probe.usb_visible and not probe.capture_ready and probe.arecord_available:
        notes.append("USB-visible without ALSA capture usually means DFU/safe mode or incomplete runtime.")

    return DeviceStatus(
        key="respeaker",
        label="ReSpeaker XVF3800",
        status=status,
        summary=summary,
        facts=(
            DeviceFact("Configured for Twinr", "yes" if configured else "no"),
            DeviceFact("Primary route", primary_device or "not configured"),
            DeviceFact("Background route", proactive_device or "not configured"),
            DeviceFact("Probe state", probe.state),
            DeviceFact("Host control", host_control_label),
            DeviceFact("Transport reason", transport_reason),
            DeviceFact("Firmware", firmware_label or "unknown"),
            DeviceFact("Speech detected", _display_optional_bool(snapshot.direction.speech_detected)),
            DeviceFact("Room quiet", _display_optional_bool(snapshot.direction.room_quiet)),
            DeviceFact("DOA azimuth", azimuth_label),
            DeviceFact("Direction confidence", direction_confidence_label),
            DeviceFact("Beam speech energy", beam_energy_label),
            DeviceFact("Speech overlap", _display_optional_bool(derived.speech_overlap_likely)),
            DeviceFact("Barge-in", _display_optional_bool(derived.barge_in_detected)),
            DeviceFact("Mute state", _display_optional_bool(snapshot.mute.mute_active)),
            DeviceFact("GPO levels", gpo_label),
            DeviceFact("USB device", usb_label),
            DeviceFact("ALSA capture", alsa_label),
            DeviceFact("Last self-test", _self_test_label(latest_self_tests, "proactive_mic")),
        ),
        notes=tuple(notes),
    )


def _collect_pir_status(
    config: TwinrConfig,
    latest_self_tests: dict[str, _SelfTestSnapshot] | None,
    *,
    last_motion_at: str | None,
) -> DeviceStatus:
    if not config.pir_enabled:
        return DeviceStatus(
            key="pir",
            label="PIR Motion",
            status="muted",
            summary="No PIR motion sensor is configured.",
            facts=(DeviceFact("GPIO", "not configured"),),
        )

    pir_gpio = _normalize_gpio(getattr(config, "pir_motion_gpio", None))
    if pir_gpio is None:
        return DeviceStatus(
            key="pir",
            label="PIR Motion",
            status="warn",
            summary="PIR motion input is enabled, but the GPIO value is missing or invalid for Raspberry Pi 4 header GPIOs.",
            facts=(
                DeviceFact("GPIO", _display_text(getattr(config, "pir_motion_gpio", "missing"), default="missing")),
                DeviceFact("Active high", "yes" if config.pir_active_high else "no"),
                DeviceFact("Bias", _display_text(getattr(config, "pir_bias", "") or "default")),
                DeviceFact("Debounce", f"{config.pir_debounce_ms} ms"),
                DeviceFact("Last motion seen", last_motion_at or "not recorded in recent ops events"),
                DeviceFact("Last self-test", _self_test_label(latest_self_tests, "pir")),
            ),
        )

    return DeviceStatus(
        key="pir",
        label="PIR Motion",
        status="ok",
        summary=f"PIR motion input is configured on GPIO {pir_gpio}.",
        facts=(
            DeviceFact("GPIO", str(pir_gpio)),
            DeviceFact("Active high", "yes" if config.pir_active_high else "no"),
            DeviceFact("Bias", _display_text(getattr(config, "pir_bias", "") or "default")),
            DeviceFact("Debounce", f"{config.pir_debounce_ms} ms"),
            DeviceFact("Last motion seen", last_motion_at or "not recorded in recent ops events"),
            DeviceFact("Last self-test", _self_test_label(latest_self_tests, "pir")),
        ),
    )


def _collect_button_status(
    config: TwinrConfig,
    latest_self_tests: dict[str, _SelfTestSnapshot] | None,
) -> DeviceStatus:
    raw_button_gpios = getattr(config, "button_gpios", {})
    button_gpios = raw_button_gpios if isinstance(raw_button_gpios, dict) else {}
    if not button_gpios:
        return DeviceStatus(
            key="buttons",
            label="Buttons",
            status="muted",
            summary="No green/yellow button GPIOs are configured.",
            facts=(DeviceFact("Configured buttons", "none"),),
        )

    owners: list[str] = []
    for loop_name, label in (("realtime-loop", "realtime"), ("hardware-loop", "hardware")):
        try:
            owner = loop_lock_owner(config, loop_name)
        except Exception as exc:
            owners.append(f"{label} owner unavailable ({exc.__class__.__name__})")
            continue
        if owner is not None:
            owners.append(f"{label} pid {owner}")

    owner_label = ", ".join(owners) if owners else "idle"
    facts = [DeviceFact("GPIO owner", owner_label), DeviceFact("Last self-test", _self_test_label(latest_self_tests, "buttons"))]

    missing_buttons: list[str] = []
    invalid_buttons: list[str] = []
    normalized_button_gpios: dict[str, int] = {}
    for name in ("green", "yellow"):
        if name not in button_gpios:
            missing_buttons.append(name)
            continue
        gpio = _normalize_gpio(button_gpios.get(name))
        if gpio is None:
            invalid_buttons.append(name)
            facts.append(DeviceFact(f"{name.title()} button GPIO", _display_text(button_gpios.get(name), default="invalid")))
            continue
        normalized_button_gpios[name] = gpio
        facts.append(DeviceFact(f"{name.title()} button GPIO", str(gpio)))

    duplicate_gpio = len(set(normalized_button_gpios.values())) != len(normalized_button_gpios)
    status = "ok"
    summary = "Hardware button GPIO mappings are configured."
    notes: list[str] = []
    if missing_buttons or invalid_buttons or duplicate_gpio:
        status = "warn"
        problems: list[str] = []
        if missing_buttons:
            problems.append(f"missing: {', '.join(missing_buttons)}")
        if invalid_buttons:
            problems.append(f"invalid: {', '.join(invalid_buttons)}")
        if duplicate_gpio:
            problems.append("green and yellow share the same GPIO")
        summary = f"Button GPIO mappings are incomplete or invalid ({'; '.join(problems)})."
        notes.append("Twinr expects separate green and yellow hardware buttons for reliable senior interaction.")
    return DeviceStatus(
        key="buttons",
        label="Buttons",
        status=status,
        summary=summary,
        facts=tuple(facts),
        notes=tuple(notes),
    )


def _captured_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_command(command: list[str], *, timeout: float = _COMMAND_TIMEOUT_SECONDS) -> _CommandResult:
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    env["LANG"] = "C"
    env["PATH"] = _TRUSTED_PATH
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env=env,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return _CommandResult(ok=False, returncode=None, stdout="", stderr=f"{exc.__class__.__name__}: {_display_text(exc)}")
    return _CommandResult(
        ok=True,
        returncode=result.returncode,
        stdout=result.stdout.strip(),
        stderr=result.stderr.strip(),
    )


def _parse_lpstat_printer_detail(raw_text: str) -> dict[str, str]:
    lines = [line.rstrip() for line in raw_text.splitlines() if line.strip()]
    info: dict[str, str] = {}
    if not lines:
        return info
    info["headline"] = lines[0].strip()
    for line in lines[1:]:
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        normalized_key = key.strip().lower().replace(" ", "_").replace("-", "_")
        info[normalized_key] = value.strip()
    return info


def _parse_lpstat_accepting_detail(raw_text: str) -> tuple[str, str]:
    detail = _display_text(raw_text, default="unknown")
    lowered = detail.lower()
    if "not accepting requests" in lowered:
        return "no", detail
    if "accepting requests" in lowered:
        return "yes", detail
    return "unknown", detail


def _parse_lpstat_device_uri(raw_text: str) -> str:
    for line in raw_text.splitlines():
        match = _DEVICE_URI_RE.match(line)
        if match is not None:
            return match.group(1).strip()
    return ""


def _probe_camera_inventory(camera_probe_path: str | None) -> tuple[tuple[str, ...], str | None]:
    if camera_probe_path is None:
        return (), "probe unavailable"
    result = _run_command([camera_probe_path, "--list-cameras"], timeout=_CAMERA_PROBE_TIMEOUT_SECONDS)
    if not result.ok:
        return (), _display_text(result.stderr or result.stdout or "camera probe failed")

    combined = "\n".join(part for part in (result.stdout, result.stderr) if part)
    cameras = _parse_list_cameras_output(combined)
    if cameras:
        return cameras, None
    if "no cameras available" in combined.lower() or result.returncode == 0:
        return (), "no cameras available"
    return (), _display_text(combined or "camera probe failed")


def _parse_list_cameras_output(raw_text: str) -> tuple[str, ...]:
    cameras: list[str] = []
    for line in raw_text.splitlines():
        match = _CAMERA_HEADER_RE.match(line)
        if match is None:
            continue
        index, description = match.groups()
        cameras.append(f"{index}: {_display_text(description, max_length=120)}")
    return tuple(cameras)


def _python_module_available(module_name: str) -> bool:
    try:
        return find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _list_arecord_capture_devices(arecord_path: str | None = None) -> tuple[str, ...]:
    if arecord_path is None:
        arecord_path = _resolve_executable("arecord")
    if arecord_path is None:
        return ()
    result = _run_command([arecord_path, "-l"])
    if not result.ok or result.returncode != 0:
        return ()
    rows = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("card "):
            rows.append(stripped)
    return tuple(rows)


def _list_audio_pcm_targets(command_path: str | None) -> tuple[str, ...]:
    if command_path is None:
        return ()
    result = _run_command([command_path, "-L"])
    if not result.ok or result.returncode != 0:
        return ()
    names: list[str] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        if line[0].isspace():
            continue
        names.append(line.strip())
    return tuple(names)


def _probe_pipewire_status(wpctl_path: str | None) -> bool:
    if wpctl_path is None:
        return False
    result = _run_command([wpctl_path, "status"], timeout=_PIPEWIRE_PROBE_TIMEOUT_SECONDS)
    return result.ok and result.returncode == 0 and bool(result.stdout)


def _has_pipewire_pcm(names: tuple[str, ...]) -> bool:
    return any(_normalize_compare_text(name) == "pipewire" for name in names)


def _configured_audio_device_matches(
    configured_value: object,
    physical_rows: tuple[str, ...],
    logical_names: tuple[str, ...],
    *,
    pipewire_ready: bool,
) -> bool | None:
    configured = _strip_text(configured_value)
    if not configured:
        return None

    normalized_configured = _normalize_compare_text(configured)
    candidate_names = {_normalize_compare_text(name) for name in logical_names}
    candidate_names.update(_audio_aliases_from_cards(physical_rows))

    if normalized_configured in candidate_names:
        return True
    if any(
        normalized_configured == candidate
        or normalized_configured.startswith(f"{candidate}:")
        or candidate.startswith(f"{normalized_configured}:")
        for candidate in candidate_names
        if candidate
    ):
        return True
    if any(
        normalized_configured in _normalize_compare_text(row) or _normalize_compare_text(row) in normalized_configured
        for row in physical_rows
    ):
        return True
    if normalized_configured in {"default", "sysdefault", "pipewire"} and (
        pipewire_ready or "default" in candidate_names or "pipewire" in candidate_names
    ):
        return True
    if pipewire_ready and normalized_configured.startswith("pipewire"):
        return True
    return False


def _audio_aliases_from_cards(physical_rows: tuple[str, ...]) -> set[str]:
    aliases: set[str] = set()
    for row in physical_rows:
        normalized_row = _normalize_compare_text(row)
        if normalized_row:
            aliases.add(normalized_row)
        match = _AUDIO_CARD_ALIAS_RE.search(row)
        if match is not None:
            card_index, device_index = match.groups()
            aliases.add(f"hw:{card_index},{device_index}")
            aliases.add(f"plughw:{card_index},{device_index}")
    return aliases


def _normalize_compare_text(value: object) -> str:
    return _strip_text(value).strip().lower()


def _join_limited(items: tuple[str, ...] | list[str], *, default: str = "—", limit: int = 4) -> str:
    if not items:
        return default
    displayed = [_display_text(item, max_length=80) for item in items[:limit]]
    if len(items) > limit:
        displayed.append(f"… +{len(items) - limit} more")
    return " | ".join(displayed)


def _match_label(value: bool | None) -> str:
    if value is None:
        return "not configured"
    return "yes" if value else "no"


def _latest_self_tests(entries: list[dict[str, object]]) -> dict[str, _SelfTestSnapshot]:
    latest: dict[str, _SelfTestSnapshot] = {}
    for entry in reversed(entries):
        if not isinstance(entry, dict):
            continue
        event_name = str(entry.get("event", "")).strip().lower()
        data = entry.get("data")
        if not isinstance(data, dict):
            continue
        test_name = str(data.get("test_name", "")).strip().lower()
        if not test_name or test_name in latest:
            continue
        if event_name == "self_test_finished":
            status = str(data.get("status", "ok")).strip().lower() or "ok"
        elif event_name == "self_test_failed":
            status = "fail"
        elif event_name == "self_test_blocked":
            status = "blocked"
        else:
            continue
        latest[test_name] = _SelfTestSnapshot(
            status=status,
            created_at=str(entry.get("created_at", "")).strip() or "unknown time",
            message=str(entry.get("message", "")).strip(),
        )
    return latest


def _latest_motion_timestamp(entries: list[dict[str, object]]) -> str | None:
    for entry in reversed(entries):
        if not isinstance(entry, dict):
            continue
        if str(entry.get("event", "")).strip().lower() != "proactive_observation":
            continue
        data = entry.get("data")
        if not isinstance(data, dict):
            continue
        if _as_bool(data.get("pir_motion_detected")) is True:
            return str(entry.get("created_at", "")).strip() or None
    return None


def _self_test_label(latest_self_tests: dict[str, _SelfTestSnapshot] | None, test_name: str) -> str:
    if latest_self_tests is None:
        return _OPS_HISTORY_UNAVAILABLE
    snapshot = latest_self_tests.get(test_name)
    if snapshot is None:
        return "never run"
    return f"{snapshot.status} at {snapshot.created_at}"


def _safe_collect_device_status(
    key: str,
    label: str,
    collector: Callable[[], DeviceStatus],
) -> DeviceStatus:
    try:
        result = collector()
    except Exception as exc:
        return DeviceStatus(
            key=key,
            label=label,
            status="fail",
            summary=f"{label} status could not be collected.",
            facts=(DeviceFact("Error", f"{exc.__class__.__name__}: {_display_text(exc)}"),),
        )
    return result


def _normalize_event_entries(entries: object) -> list[dict[str, object]]:
    if isinstance(entries, list):
        raw_entries = entries
    elif isinstance(entries, Iterable):
        raw_entries = list(entries)
    else:
        return []
    normalized: list[dict[str, object]] = []
    for entry in raw_entries:
        if isinstance(entry, dict):
            normalized.append(entry)
    return normalized


def _resolve_executable(command: str) -> str | None:
    candidate = _strip_text(command)
    if not candidate:
        return None

    if os.sep in candidate or (os.altsep and os.altsep in candidate):
        path = Path(candidate)
        # BREAKING: relative executable paths are rejected so probe commands cannot be shadowed from the service working directory.
        if not path.is_absolute():
            return None
        return str(path) if _path_is_executable_file(path) else None

    for directory in _TRUSTED_EXECUTABLE_DIRS:
        path = Path(directory) / candidate
        if _path_is_executable_file(path):
            return str(path)
    return None


def _path_is_executable_file(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        return path.is_file() and os.access(path, os.X_OK)
    except OSError:
        return False


def _path_is_char_device(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        return path.is_char_device()
    except OSError:
        return False


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    if isinstance(value, str) and any(character in value for character in (".", "e", "E")):
        return None
    return coerced


def _normalize_gpio(value: object) -> int | None:
    gpio = _coerce_int(value)
    if gpio is None:
        return None
    if gpio < _GPIO_MIN_BCM or gpio > _GPIO_MAX_BCM_PI4:
        return None
    return gpio


def _strip_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _display_text(value: object, *, default: str = "—", max_length: int = 240) -> str:
    text = _strip_text(value)
    if not text:
        return default
    text = "".join(character if character.isprintable() else " " for character in text)
    text = " ".join(text.split())
    if len(text) > max_length:
        return f"{text[: max_length - 1].rstrip()}…"
    return text


def _display_optional_bool(value: object, *, default: str = "unknown") -> str:
    normalized = _as_bool(value)
    if normalized is None:
        return default
    return "yes" if normalized else "no"


def _display_optional_number(value: object, *, default: str = "—") -> str:
    parsed = _coerce_int(value)
    if parsed is None:
        return default
    return str(parsed)


def _display_float_tuple(values: tuple[float | None, ...] | None, *, default: str = "unknown") -> str:
    if not values:
        return default
    rendered: list[str] = []
    for value in values:
        if value is None:
            rendered.append("—")
            continue
        rendered.append(f"{float(value):.3f}")
    return ", ".join(rendered)


def _display_int_tuple(values: tuple[int, ...] | None, *, default: str = "unknown") -> str:
    if not values:
        return default
    return ", ".join(str(int(value)) for value in values)


def _respeaker_firmware_label(version: tuple[int, int, int] | None) -> str | None:
    if version is None or len(version) != 3:
        return None
    return ".".join(str(int(part)) for part in version)


def _as_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    text = _strip_text(value).lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return None


def _redact_device_uri(value: object) -> str:
    raw_value = _display_text(value, default="—")
    if raw_value == "—":
        return raw_value
    try:
        parts = urlsplit(raw_value)
    except ValueError:
        return raw_value
    if not parts.scheme:
        return raw_value

    hostname = parts.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    userinfo = ""
    if parts.username:
        userinfo = parts.username
        if parts.password is not None:
            userinfo += ":***"
        userinfo += "@"
    elif parts.password is not None:
        userinfo = "***@"
    netloc = f"{userinfo}{hostname}"
    if parts.port is not None:
        netloc = f"{netloc}:{parts.port}"
    query = "***" if parts.query else ""
    fragment = "***" if parts.fragment else ""
    return urlunsplit((parts.scheme, netloc or parts.netloc, parts.path, query, fragment))