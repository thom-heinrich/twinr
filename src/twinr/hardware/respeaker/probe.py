"""Probe ReSpeaker XVF3800 USB and capture readiness for Twinr."""

from __future__ import annotations

from pathlib import Path
from shutil import which
import os
import subprocess

from twinr.hardware.respeaker.models import ReSpeakerCaptureDevice, ReSpeakerProbeResult, ReSpeakerUsbDevice


_COMMAND_TIMEOUT_SECONDS = 2.0
_XVF3800_VENDOR_ID = "2886"
_XVF3800_PRODUCT_ID = "001a"
_CAPTURE_MARKERS = (
    "respeaker xvf3800",
    "seeed studio respeaker xvf3800",
    "xvf3800 4-mic array",
)
_CONFIG_MARKERS = (
    "card=array",
    "respeaker",
    "xvf3800",
    "seeed_studio_respeaker_xvf3800",
)


def config_targets_respeaker(*device_values: str | None) -> bool:
    """Return whether any configured device string explicitly targets XVF3800.

    Args:
        *device_values: One or more configured audio-device strings.

    Returns:
        True when at least one device string contains an explicit XVF3800/
        ReSpeaker marker used by Twinr's current ALSA or PipeWire paths.
    """

    for value in device_values:
        normalized = str(value or "").strip().lower()
        if any(marker in normalized for marker in _CONFIG_MARKERS):
            return True
    return False


def probe_respeaker_xvf3800() -> ReSpeakerProbeResult:
    """Probe the current host-visible XVF3800 USB and ALSA state."""

    lsusb_path = _resolve_executable("lsusb")
    arecord_path = _resolve_executable("arecord")
    usb_device = _probe_usb_device(lsusb_path)
    capture_device = _probe_capture_device(arecord_path)
    return ReSpeakerProbeResult(
        usb_device=usb_device,
        capture_device=capture_device,
        lsusb_available=lsusb_path is not None,
        arecord_available=arecord_path is not None,
    )


def _probe_usb_device(lsusb_path: str | None) -> ReSpeakerUsbDevice | None:
    if lsusb_path is None:
        return None
    result = _run_command([lsusb_path, "-d", f"{_XVF3800_VENDOR_ID}:{_XVF3800_PRODUCT_ID}"])
    if not result.ok or result.returncode != 0 or not result.stdout:
        return None
    for line in result.stdout.splitlines():
        device = _parse_lsusb_line(line)
        if device is not None:
            return device
    return None


def _probe_capture_device(arecord_path: str | None) -> ReSpeakerCaptureDevice | None:
    if arecord_path is None:
        return None
    result = _run_command([arecord_path, "-l"])
    if not result.ok or result.returncode != 0 or not result.stdout:
        return None
    for line in result.stdout.splitlines():
        device = _parse_arecord_line(line)
        if device is not None:
            return device
    return None


def _parse_lsusb_line(line: str) -> ReSpeakerUsbDevice | None:
    stripped = " ".join(str(line).split()).strip()
    if not stripped.startswith("Bus "):
        return None
    parts = stripped.split(" ")
    if len(parts) < 6 or parts[2] != "Device" or parts[4] != "ID":
        return None
    vendor_product = parts[5].lower()
    if ":" not in vendor_product:
        return None
    vendor_id, product_id = vendor_product.split(":", 1)
    if vendor_id != _XVF3800_VENDOR_ID or product_id != _XVF3800_PRODUCT_ID:
        return None
    description = " ".join(parts[6:]).strip()
    return ReSpeakerUsbDevice(
        bus=parts[1],
        device=parts[3].rstrip(":"),
        vendor_id=vendor_id,
        product_id=product_id,
        description=description,
        raw_line=stripped,
    )


def _parse_arecord_line(line: str) -> ReSpeakerCaptureDevice | None:
    stripped = " ".join(str(line).split()).strip()
    if not stripped.startswith("card "):
        return None
    lowered = stripped.lower()
    if not any(marker in lowered for marker in _CAPTURE_MARKERS):
        return None

    header, _, remainder = stripped.partition(":")
    if not remainder:
        return None
    card_index = _parse_prefixed_int(header, prefix="card ")
    card_section, _, device_section = remainder.partition(", device ")
    card_name = card_section.split("[", 1)[0].strip()
    card_label = _bracket_text(card_section) or card_section.strip()
    device_index = _parse_prefixed_int(device_section, prefix="")
    return ReSpeakerCaptureDevice(
        card_index=card_index,
        card_name=card_name,
        card_label=card_label,
        device_index=device_index,
        raw_line=stripped,
    )


def _parse_prefixed_int(value: str, *, prefix: str) -> int | None:
    normalized = value.strip()
    if prefix and not normalized.startswith(prefix):
        return None
    if prefix:
        normalized = normalized[len(prefix):]
    digits: list[str] = []
    for character in normalized:
        if character.isdigit():
            digits.append(character)
            continue
        break
    if not digits:
        return None
    try:
        return int("".join(digits))
    except ValueError:
        return None


def _bracket_text(value: str) -> str:
    left = value.find("[")
    right = value.find("]", left + 1) if left >= 0 else -1
    if left < 0 or right <= left:
        return ""
    return value[left + 1 : right].strip()


def _resolve_executable(command: str) -> str | None:
    candidate = str(command or "").strip()
    if not candidate:
        return None
    resolved = which(candidate)
    if resolved is not None:
        return resolved
    path = Path(candidate)
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


def _run_command(command: list[str]) -> _CommandResult:
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    env["LANG"] = "C"
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=_COMMAND_TIMEOUT_SECONDS,
            env=env,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return _CommandResult(ok=False, returncode=None, stdout="", stderr="")
    return _CommandResult(
        ok=True,
        returncode=result.returncode,
        stdout=result.stdout.strip(),
        stderr=result.stderr.strip(),
    )


class _CommandResult:
    """Capture one bounded command execution result."""

    __slots__ = ("ok", "returncode", "stdout", "stderr")

    def __init__(self, *, ok: bool, returncode: int | None, stdout: str, stderr: str) -> None:
        self.ok = ok
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
