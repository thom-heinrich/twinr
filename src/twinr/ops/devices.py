from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from shutil import which
import shlex
import subprocess

from twinr.agent.base_agent.config import TwinrConfig
from twinr.ops.events import TwinrOpsEventStore
from twinr.ops.locks import loop_lock_owner


@dataclass(frozen=True, slots=True)
class DeviceFact:
    label: str
    value: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DeviceStatus:
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
    events = (event_store or TwinrOpsEventStore.from_config(config)).tail(limit=200)
    latest_self_tests = _latest_self_tests(events)
    last_motion_at = _latest_motion_timestamp(events)
    return DeviceOverview(
        captured_at=_captured_at(),
        devices=(
            _collect_printer_status(config, latest_self_tests),
            _collect_camera_status(config, latest_self_tests),
            _collect_primary_audio_status(config, latest_self_tests),
            _collect_proactive_audio_status(config, latest_self_tests),
            _collect_pir_status(config, latest_self_tests, last_motion_at=last_motion_at),
            _collect_button_status(config, latest_self_tests),
        ),
    )


def _collect_printer_status(
    config: TwinrConfig,
    latest_self_tests: dict[str, _SelfTestSnapshot],
) -> DeviceStatus:
    queue = config.printer_queue.strip()
    if not queue:
        return DeviceStatus(
            key="printer",
            label="Printer",
            status="fail",
            summary="No printer queue is configured.",
            facts=(DeviceFact("Queue", "missing"),),
        )

    lpstat_path = which("lpstat")
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

    lpstat_result = _run_command(["lpstat", "-l", "-p", queue])
    if not lpstat_result.ok or lpstat_result.returncode != 0:
        detail = lpstat_result.stderr or lpstat_result.stdout or "Queue is not visible to CUPS."
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
    lpoptions_info = _parse_lpoptions(queue) if which("lpoptions") is not None else {}
    accepting_jobs = lpoptions_info.get("printer-is-accepting-jobs", "unknown")
    state_reasons = lpoptions_info.get("printer-state-reasons") or lpstat_info.get("alerts") or "unknown"
    status = "ok"
    if accepting_jobs not in {"true", "unknown"} or state_reasons not in {"none", "unknown"}:
        status = "warn"
    summary = lpstat_info.get("headline") or f"Queue `{queue}` is visible to CUPS."
    return DeviceStatus(
        key="printer",
        label="Printer",
        status=status,
        summary=summary,
        facts=(
            DeviceFact("Queue", queue),
            DeviceFact("Description", lpstat_info.get("description") or lpoptions_info.get("printer-info") or "—"),
            DeviceFact("Connection", lpstat_info.get("connection") or "—"),
            DeviceFact("Device URI", lpoptions_info.get("device-uri", "—")),
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
    latest_self_tests: dict[str, _SelfTestSnapshot],
) -> DeviceStatus:
    device = Path(config.camera_device)
    ffmpeg_ok = _command_available_or_path_exists(config.camera_ffmpeg_path)
    if device.exists() and ffmpeg_ok:
        status = "ok"
        summary = f"Camera device `{device}` is present."
    elif not device.exists():
        status = "warn"
        summary = f"Camera device `{device}` does not exist on this machine."
    else:
        status = "warn"
        summary = f"Camera device `{device}` exists, but ffmpeg `{config.camera_ffmpeg_path}` is unavailable."
    return DeviceStatus(
        key="camera",
        label="Camera",
        status=status,
        summary=summary,
        facts=(
            DeviceFact("Device", str(device)),
            DeviceFact("Present", "yes" if device.exists() else "no"),
            DeviceFact("Resolution", f"{config.camera_width}x{config.camera_height}"),
            DeviceFact("Framerate", f"{config.camera_framerate} fps"),
            DeviceFact("Input format", config.camera_input_format or "default"),
            DeviceFact("ffmpeg", config.camera_ffmpeg_path if ffmpeg_ok else f"missing: {config.camera_ffmpeg_path}"),
            DeviceFact("Reference image", config.vision_reference_image_path or "not configured"),
            DeviceFact("Last self-test", _self_test_label(latest_self_tests, "camera")),
        ),
    )


def _collect_primary_audio_status(
    config: TwinrConfig,
    latest_self_tests: dict[str, _SelfTestSnapshot],
) -> DeviceStatus:
    capture_devices = _list_arecord_capture_devices()
    if which("arecord") is None:
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
            DeviceFact("Configured input", config.audio_input_device),
            DeviceFact("Configured output", config.audio_output_device),
            DeviceFact("Sample rate", f"{config.audio_sample_rate} Hz"),
            DeviceFact("Channels", str(config.audio_channels)),
            DeviceFact("Detected capture devices", " | ".join(capture_devices) if capture_devices else "none reported"),
            DeviceFact("Last self-test", _self_test_label(latest_self_tests, "mic")),
        ),
    )


def _collect_proactive_audio_status(
    config: TwinrConfig,
    latest_self_tests: dict[str, _SelfTestSnapshot],
) -> DeviceStatus:
    proactive_device = (config.proactive_audio_input_device or "").strip()
    enabled = config.proactive_audio_enabled or bool(proactive_device)
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

    if which("arecord") is None:
        status = "warn"
        summary = "Background-audio capture is enabled, but `arecord` is unavailable."
    else:
        status = "ok"
        summary = "Background-audio capture is configured."
    device_label = proactive_device or f"reuse primary input ({config.audio_input_device})"
    return DeviceStatus(
        key="proactive_audio",
        label="Background Audio",
        status=status,
        summary=summary,
        facts=(
            DeviceFact("Configured device", device_label),
            DeviceFact("Sample window", f"{config.proactive_audio_sample_ms} ms"),
            DeviceFact("Distress heuristic", "enabled" if config.proactive_audio_distress_enabled else "disabled"),
            DeviceFact("Last self-test", _self_test_label(latest_self_tests, "proactive_mic")),
        ),
    )


def _collect_pir_status(
    config: TwinrConfig,
    latest_self_tests: dict[str, _SelfTestSnapshot],
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

    return DeviceStatus(
        key="pir",
        label="PIR Motion",
        status="ok",
        summary=f"PIR motion input is configured on GPIO {config.pir_motion_gpio}.",
        facts=(
            DeviceFact("GPIO", str(config.pir_motion_gpio)),
            DeviceFact("Active high", "yes" if config.pir_active_high else "no"),
            DeviceFact("Bias", config.pir_bias),
            DeviceFact("Debounce", f"{config.pir_debounce_ms} ms"),
            DeviceFact("Last motion seen", last_motion_at or "not recorded in recent ops events"),
            DeviceFact("Last self-test", _self_test_label(latest_self_tests, "pir")),
        ),
    )


def _collect_button_status(
    config: TwinrConfig,
    latest_self_tests: dict[str, _SelfTestSnapshot],
) -> DeviceStatus:
    button_gpios = config.button_gpios
    if not button_gpios:
        return DeviceStatus(
            key="buttons",
            label="Buttons",
            status="muted",
            summary="No green/yellow button GPIOs are configured.",
            facts=(DeviceFact("Configured buttons", "none"),),
        )

    owners = []
    for loop_name, label in (("realtime-loop", "realtime"), ("hardware-loop", "hardware")):
        owner = loop_lock_owner(config, loop_name)
        if owner is not None:
            owners.append(f"{label} pid {owner}")
    owner_label = ", ".join(owners) if owners else "idle"
    facts = [DeviceFact("GPIO owner", owner_label), DeviceFact("Last self-test", _self_test_label(latest_self_tests, "buttons"))]
    for name in ("green", "yellow"):
        if name in button_gpios:
            facts.append(DeviceFact(f"{name.title()} button GPIO", str(button_gpios[name])))
    return DeviceStatus(
        key="buttons",
        label="Buttons",
        status="ok",
        summary="Hardware button GPIO mappings are configured.",
        facts=tuple(facts),
    )


def _captured_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _command_available_or_path_exists(command: str) -> bool:
    return which(command) is not None or Path(command).exists()


def _run_command(command: list[str], *, timeout: float = 2.0) -> _CommandResult:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except Exception as exc:
        return _CommandResult(ok=False, returncode=None, stdout="", stderr=str(exc))
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


def _parse_lpoptions(queue: str) -> dict[str, str]:
    result = _run_command(["lpoptions", "-p", queue])
    if not result.ok or result.returncode != 0 or not result.stdout:
        return {}
    values: dict[str, str] = {}
    for token in shlex.split(result.stdout):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _list_arecord_capture_devices() -> tuple[str, ...]:
    if which("arecord") is None:
        return ()
    result = _run_command(["arecord", "-l"])
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
        if str(entry.get("event", "")).strip().lower() != "proactive_observation":
            continue
        data = entry.get("data")
        if not isinstance(data, dict):
            continue
        if bool(data.get("pir_motion_detected")):
            return str(entry.get("created_at", "")).strip() or None
    return None


def _self_test_label(latest_self_tests: dict[str, _SelfTestSnapshot], test_name: str) -> str:
    snapshot = latest_self_tests.get(test_name)
    if snapshot is None:
        return "never run"
    return f"{snapshot.status} at {snapshot.created_at}"
