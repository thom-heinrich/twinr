"""Collect a bounded operational overview of Twinr's local devices.

This module combines live system probes with recent ops and self-test events
to build dashboard-friendly device status snapshots.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from shutil import which
import os
import shlex
import subprocess
from urllib.parse import urlsplit, urlunsplit

from twinr.agent.base_agent.config import TwinrConfig
from twinr.ops.events import TwinrOpsEventStore
from twinr.ops.locks import loop_lock_owner


_COMMAND_TIMEOUT_SECONDS = 2.0
_OPS_HISTORY_UNAVAILABLE = "ops history unavailable"


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
        # AUDIT-FIX(#2): Use explicit None-check instead of truthiness so an injected store instance is never ignored accidentally.
        store = event_store if event_store is not None else TwinrOpsEventStore.from_config(config)
        events = _normalize_event_entries(store.tail(limit=200))
        latest_self_tests = _latest_self_tests(events)
        last_motion_at = _latest_motion_timestamp(events)
    except Exception:
        # AUDIT-FIX(#1): Degrade gracefully when the file-backed ops store is unreadable or corrupted instead of failing the whole overview.
        latest_self_tests = None
        last_motion_at = _OPS_HISTORY_UNAVAILABLE

    return DeviceOverview(
        captured_at=_captured_at(),
        devices=(
            # AUDIT-FIX(#1): Isolate each probe so one broken subsystem does not blank the full dashboard.
            _safe_collect_device_status("printer", "Printer", lambda: _collect_printer_status(config, latest_self_tests)),
            # AUDIT-FIX(#1): Isolate each probe so one broken subsystem does not blank the full dashboard.
            _safe_collect_device_status("camera", "Camera", lambda: _collect_camera_status(config, latest_self_tests)),
            # AUDIT-FIX(#1): Isolate each probe so one broken subsystem does not blank the full dashboard.
            _safe_collect_device_status("audio_input", "Primary Audio", lambda: _collect_primary_audio_status(config, latest_self_tests)),
            # AUDIT-FIX(#1): Isolate each probe so one broken subsystem does not blank the full dashboard.
            _safe_collect_device_status("proactive_audio", "Background Audio", lambda: _collect_proactive_audio_status(config, latest_self_tests)),
            # AUDIT-FIX(#1): Isolate each probe so one broken subsystem does not blank the full dashboard.
            _safe_collect_device_status("pir", "PIR Motion", lambda: _collect_pir_status(config, latest_self_tests, last_motion_at=last_motion_at)),
            # AUDIT-FIX(#1): Isolate each probe so one broken subsystem does not blank the full dashboard.
            _safe_collect_device_status("buttons", "Buttons", lambda: _collect_button_status(config, latest_self_tests)),
        ),
    )


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

    # AUDIT-FIX(#3): Resolve the exact executable once and run that path, instead of re-searching PATH during subprocess execution.
    lpstat_path = _resolve_executable("lpstat")
    if lpstat_path is None:
        return DeviceStatus(
            key="printer",
            label="Printer",
            status="warn",
            summary="Printer queue is configured, but `lpstat` is unavailable on this machine.",
            facts=(
                DeviceFact("Queue", queue),
                DeviceFact("Paper status", "unknown on the current path"),
                DeviceFact("Last self-test", _self_test_label(latest_self_tests, "printer")),
            ),
        )

    lpstat_result = _run_command([lpstat_path, "-l", "-p", queue])
    if not lpstat_result.ok or lpstat_result.returncode != 0:
        detail = _display_text(lpstat_result.stderr or lpstat_result.stdout or "Queue is not visible to CUPS.")
        return DeviceStatus(
            key="printer",
            label="Printer",
            status="warn",
            summary=f"Queue `{queue}` is configured but not visible to CUPS right now.",
            facts=(
                DeviceFact("Queue", queue),
                DeviceFact("CUPS detail", detail),
                DeviceFact("Paper status", "unknown on the current path"),
                DeviceFact("Last self-test", _self_test_label(latest_self_tests, "printer")),
            ),
        )

    lpstat_info = _parse_lpstat_printer_detail(lpstat_result.stdout)
    # AUDIT-FIX(#3): Resolve the exact executable once and run that path, instead of re-searching PATH during subprocess execution.
    lpoptions_path = _resolve_executable("lpoptions")
    lpoptions_info = _parse_lpoptions(lpoptions_path, queue) if lpoptions_path is not None else {}
    accepting_jobs = _display_text(lpoptions_info.get("printer-is-accepting-jobs", "unknown"), default="unknown")
    state_reasons = _display_text(
        lpoptions_info.get("printer-state-reasons") or lpstat_info.get("alerts") or "unknown",
        default="unknown",
    )
    status = "ok"
    if accepting_jobs not in {"true", "unknown"} or state_reasons not in {"none", "unknown"}:
        status = "warn"
    summary = _display_text(lpstat_info.get("headline") or f"Queue `{queue}` is visible to CUPS.")
    return DeviceStatus(
        key="printer",
        label="Printer",
        status=status,
        summary=summary,
        facts=(
            DeviceFact("Queue", queue),
            DeviceFact("Description", _display_text(lpstat_info.get("description") or lpoptions_info.get("printer-info") or "—")),
            DeviceFact("Connection", _display_text(lpstat_info.get("connection") or "—")),
            # AUDIT-FIX(#4): Redact credentials and query strings from device URIs before returning them to callers.
            DeviceFact("Device URI", _redact_device_uri(lpoptions_info.get("device-uri", "—"))),
            DeviceFact("Accepting jobs", accepting_jobs),
            DeviceFact("State reasons", state_reasons),
            DeviceFact("Paper status", "unknown on the current raw USB/CUPS path"),
            DeviceFact("Last self-test", _self_test_label(latest_self_tests, "printer")),
        ),
        notes=(
            "Twinr can see queue and USB/CUPS state here, but it cannot prove real paper output from this printer path.",
        ),
    )


def _collect_camera_status(
    config: TwinrConfig,
    latest_self_tests: dict[str, _SelfTestSnapshot] | None,
) -> DeviceStatus:
    # AUDIT-FIX(#5): Treat blank paths as missing config instead of mapping '' to the current working directory.
    device_value = _strip_text(getattr(config, "camera_device", ""))
    device_path = Path(device_value) if device_value else None
    # AUDIT-FIX(#5): Require a character device for camera presence so regular files/directories do not report as healthy.
    device_present = _path_is_char_device(device_path)
    # AUDIT-FIX(#5): Require a real executable file for ffmpeg instead of any existing filesystem path.
    ffmpeg_path = _resolve_executable(getattr(config, "camera_ffmpeg_path", ""))
    ffmpeg_ok = ffmpeg_path is not None

    if not device_value:
        status = "fail"
        summary = "No camera device is configured."
    elif device_present and ffmpeg_ok:
        status = "ok"
        summary = f"Camera device `{device_value}` is present."
    elif not device_present:
        status = "warn"
        summary = f"Camera device `{device_value}` is missing or is not a character device."
    else:
        status = "warn"
        summary = f"Camera device `{device_value}` exists, but ffmpeg `{getattr(config, 'camera_ffmpeg_path', '')}` is unavailable."

    return DeviceStatus(
        key="camera",
        label="Camera",
        status=status,
        summary=summary,
        facts=(
            DeviceFact("Device", device_value or "missing"),
            DeviceFact("Present", "yes" if device_present else "no"),
            DeviceFact("Resolution", f"{config.camera_width}x{config.camera_height}"),
            DeviceFact("Framerate", f"{config.camera_framerate} fps"),
            DeviceFact("Input format", _display_text(getattr(config, "camera_input_format", "") or "default")),
            DeviceFact("ffmpeg", ffmpeg_path or f"missing: {_display_text(getattr(config, 'camera_ffmpeg_path', '') or 'not configured')}"),
            DeviceFact("Reference image", _display_text(getattr(config, "vision_reference_image_path", "") or "not configured")),
            DeviceFact("Last self-test", _self_test_label(latest_self_tests, "camera")),
        ),
    )


def _collect_primary_audio_status(
    config: TwinrConfig,
    latest_self_tests: dict[str, _SelfTestSnapshot] | None,
) -> DeviceStatus:
    # AUDIT-FIX(#3): Resolve once and pass the exact binary path into the subprocess probe.
    arecord_path = _resolve_executable("arecord")
    capture_devices = _list_arecord_capture_devices(arecord_path)
    if arecord_path is None:
        status = "warn"
        summary = "Audio capture cannot be inspected because `arecord` is unavailable."
    elif capture_devices:
        status = "ok"
        summary = f"{len(capture_devices)} capture device(s) reported by ALSA."
    else:
        status = "warn"
        summary = "No ALSA capture devices were reported by `arecord -l`."
    return DeviceStatus(
        key="audio_input",
        label="Primary Audio",
        status=status,
        summary=summary,
        facts=(
            DeviceFact("Configured input", _display_text(getattr(config, "audio_input_device", "") or "not configured")),
            DeviceFact("Configured output", _display_text(getattr(config, "audio_output_device", "") or "not configured")),
            DeviceFact("Sample rate", f"{config.audio_sample_rate} Hz"),
            DeviceFact("Channels", str(config.audio_channels)),
            DeviceFact("Detected capture devices", " | ".join(capture_devices) if capture_devices else "none reported"),
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

    # AUDIT-FIX(#3): Resolve once and pass the exact binary path into the subprocess probe.
    arecord_path = _resolve_executable("arecord")
    if arecord_path is None:
        status = "warn"
        summary = "Background-audio capture is enabled, but `arecord` is unavailable."
    elif proactive_device or primary_device:
        status = "ok"
        summary = "Background-audio capture is configured."
    else:
        # AUDIT-FIX(#6): Do not report healthy background audio when capture is enabled but no input device is configured at all.
        status = "warn"
        summary = "Background-audio capture is enabled, but no input device is configured."
    device_label = proactive_device or (f"reuse primary input ({primary_device})" if primary_device else "missing")
    return DeviceStatus(
        key="proactive_audio",
        label="Background Audio",
        status=status,
        summary=summary,
        facts=(
            DeviceFact("Configured device", device_label),
            DeviceFact("Sample window", f"{config.proactive_audio_sample_ms} ms"),
            DeviceFact("Distress detector", "enabled" if config.proactive_audio_distress_enabled else "disabled"),
            DeviceFact("Last self-test", _self_test_label(latest_self_tests, "proactive_mic")),
        ),
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

    # AUDIT-FIX(#7): Validate GPIO values before marking the PIR path as healthy.
    pir_gpio = _normalize_gpio(getattr(config, "pir_motion_gpio", None))
    if pir_gpio is None:
        return DeviceStatus(
            key="pir",
            label="PIR Motion",
            status="warn",
            summary="PIR motion input is enabled, but the GPIO value is missing or invalid.",
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
            # AUDIT-FIX(#1): Lock-inspection failures should degrade this device status, not crash the whole overview.
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
        # AUDIT-FIX(#7): Require both expected buttons and unique valid GPIOs before reporting a healthy button path.
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


def _command_available_or_path_exists(command: str) -> bool:
    # AUDIT-FIX(#5): This helper now only returns true for non-empty executable files, not arbitrary existing paths or directories.
    return _resolve_executable(command) is not None


def _run_command(command: list[str], *, timeout: float = _COMMAND_TIMEOUT_SECONDS) -> _CommandResult:
    env = os.environ.copy()
    # AUDIT-FIX(#3): Force a stable C locale because downstream parsers depend on English command output tokens.
    env["LC_ALL"] = "C"
    # AUDIT-FIX(#3): Force a stable C locale because downstream parsers depend on English command output tokens.
    env["LANG"] = "C"
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env=env,
            # AUDIT-FIX(#8): Decode defensively so non-UTF-8 device descriptions cannot raise and abort health collection.
            encoding="utf-8",
            # AUDIT-FIX(#8): Replace undecodable bytes instead of throwing UnicodeDecodeError inside the health probe.
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
        normalized_key = key.strip().lower().replace(" ", "_")
        info[normalized_key] = value.strip()
    return info


def _parse_lpoptions(lpoptions_path: str, queue: str) -> dict[str, str]:
    result = _run_command([lpoptions_path, "-p", queue])
    if not result.ok or result.returncode != 0 or not result.stdout:
        return {}
    values: dict[str, str] = {}
    try:
        tokens = shlex.split(result.stdout)
    except ValueError:
        # AUDIT-FIX(#9): Malformed or localized lpoptions output must not crash the printer overview.
        tokens = result.stdout.split()
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        values[key.strip()] = value.strip()
    return values


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


def _latest_self_tests(entries: list[dict[str, object]]) -> dict[str, _SelfTestSnapshot]:
    latest: dict[str, _SelfTestSnapshot] = {}
    for entry in reversed(entries):
        # AUDIT-FIX(#10): Skip malformed rows from the file-backed event store instead of assuming every item is a dict.
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
        # AUDIT-FIX(#10): Skip malformed rows from the file-backed event store instead of assuming every item is a dict.
        if not isinstance(entry, dict):
            continue
        if str(entry.get("event", "")).strip().lower() != "proactive_observation":
            continue
        data = entry.get("data")
        if not isinstance(data, dict):
            continue
        # AUDIT-FIX(#11): Parse truthy/falsey strings explicitly so 'false' does not become True via bool('false').
        if _as_bool(data.get("pir_motion_detected")) is True:
            return str(entry.get("created_at", "")).strip() or None
    return None


def _self_test_label(latest_self_tests: dict[str, _SelfTestSnapshot] | None, test_name: str) -> str:
    if latest_self_tests is None:
        # AUDIT-FIX(#1): Preserve a truthful degraded state when ops history could not be read at all.
        return _OPS_HISTORY_UNAVAILABLE
    snapshot = latest_self_tests.get(test_name)
    if snapshot is None:
        return "never run"
    return f"{snapshot.status} at {snapshot.created_at}"


# AUDIT-FIX(#1): Centralize per-device exception isolation so a single probe failure cannot blank the full overview.
def _safe_collect_device_status(
    key: str,
    label: str,
    collector,
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


# AUDIT-FIX(#10): Normalize and filter raw event rows before parsing so corrupted file-backed records are skipped safely.
def _normalize_event_entries(entries: object) -> list[dict[str, object]]:
    if isinstance(entries, list):
        raw_entries = entries
    else:
        try:
            raw_entries = list(entries)
        except Exception:
            return []
    normalized: list[dict[str, object]] = []
    for entry in raw_entries:
        if isinstance(entry, dict):
            normalized.append(entry)
    return normalized


# AUDIT-FIX(#3): Resolve executables once up front to avoid PATH re-resolution at subprocess execution time.
def _resolve_executable(command: str) -> str | None:
    candidate = _strip_text(command)
    if not candidate:
        return None
    resolved = which(candidate)
    if resolved is not None:
        return resolved
    path = Path(candidate)
    if _path_is_executable_file(path):
        return str(path)
    return None


# AUDIT-FIX(#5): Distinguish real executable files from arbitrary existing paths such as directories.
def _path_is_executable_file(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        return path.is_file() and os.access(path, os.X_OK)
    except OSError:
        return False


# AUDIT-FIX(#5): Validate camera nodes as character devices, not merely existing filesystem entries.
def _path_is_char_device(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        return path.is_char_device()
    except OSError:
        return False


# AUDIT-FIX(#7): Reject bools, negative numbers, and non-integer GPIO values before reporting hardware as healthy.
def _normalize_gpio(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        gpio = int(value)
    except (TypeError, ValueError):
        return None
    if gpio < 0:
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


# AUDIT-FIX(#4): Strip credentials and query/fragment material from printer URIs before exposing them downstream.
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
