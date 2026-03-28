#!/usr/bin/env python3
# CHANGELOG: 2026-03-27
# BUG-1: Fixed stale Crazyradio 2 native USB-ID handling and mode detection; the probe now
# BUG-1: accepts current native IDs, gives UF2 markers precedence, and avoids false negatives.
# BUG-2: Fixed workspace probing so a broken venv/interpreter no longer aborts the whole probe.
# BUG-3: Fixed misleading cflib access validation; PA-compatible probing is now reported
# BUG-3: explicitly and no longer mislabels native Crazyradio 2 mode as a generic access failure.
# SEC-1: Run workspace Python snippets in isolated mode (-I) to block PYTHON* env and unsafe path injection.
# IMP-1: Switched primary USB enumeration to pyudev/libudev with sysfs fallback for 2026 Linux practice.
# IMP-2: Added richer structured reporting (filesystem label, bcdDevice, probe scope/backend, workspace errors).
# IMP-3: Updated recommendations to reflect the 2025/2026 Crazyradio 2 landscape
# IMP-3: (PA-compatible path vs native firmware / inline-mode path).
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pyudev>=0.24.4",
# ]
# ///
"""Inspect Bitcraze USB state and validate the local workspace.

Purpose
-------
Detect connected Bitcraze USB devices, classify whether the Crazyradio is in
PA emulation, native Crazyradio 2.0, or UF2 bootloader mode, and optionally
prove that the configured workspace can open the radio through ``cflib``.

Usage
-----
Command-line examples::

    python3 hardware/bitcraze/probe_crazyradio.py --workspace /twinr/bitcraze
    python3 hardware/bitcraze/probe_crazyradio.py --workspace /twinr/bitcraze --json
    python3 hardware/bitcraze/probe_crazyradio.py --workspace /twinr/bitcraze --expect-mode pa_emulation --require-cflib-access

Outputs
-------
- Human-readable status lines by default
- JSON report with ``--json``
- Exit code 0 on success and 1 on failed expectations
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

PA_EMULATION_USB_ID = ("1915", "7777")
# Current Bitcraze firmware repo points to 35f0:ad20 for the native USB protocol.
# Keep 35f0:bad2 as a compatibility alias because older field scripts have used it.
CRAZYRADIO2_NATIVE_USB_IDS = {
    ("35f0", "ad20"),
    ("35f0", "bad2"),
}
# Treat classic Bitcraze bootloader USB IDs only as weak hints; UF2 markers win.
UF2_BOOTLOADER_USB_IDS = {
    ("1915", "0101"),
}
UF2_FILESYSTEM_LABELS = {"crazyradio2"}
UF2_INFO_FILENAMES = ("INFO_UF2.TXT", "info_uf2.txt")


@dataclass(frozen=True)
class BitcrazeUsbDevice:
    """Describe one detected Bitcraze USB device."""

    sysfs_path: str
    vendor_id: str
    product_id: str
    bcd_device: str | None
    manufacturer: str | None
    product: str | None
    serial: str | None
    tty_nodes: tuple[str, ...]
    block_devices: tuple[str, ...]
    mountpoints: tuple[str, ...]
    filesystem_label: str | None
    info_uf2_present: bool
    mode: str
    cflib_probe_compatible: bool
    recommendation: str


@dataclass(frozen=True)
class WorkspaceProbe:
    """Describe the Bitcraze Python workspace state."""

    workspace: str
    exists: bool
    venv_python: str | None
    cflib_version: str | None
    cfclient_version: str | None
    workspace_probe_error: str | None
    radio_access_attempted: bool
    radio_access_scope: str
    radio_access_ok: bool
    radio_access_error: str | None
    detected_radio_serials: tuple[str, ...]
    radio_version: str | None


def _read_text(path: Path) -> str | None:
    """Read and strip a small text file if it exists."""

    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError:
        return None


def _coerce_text(value: Any) -> str | None:
    """Convert pyudev / sysfs values to clean text."""

    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace").strip()
        except Exception:
            return value.decode(errors="replace").strip()
    return str(value).strip()


def _normalize_usb_hex(value: Any) -> str:
    """Normalize USB identifiers to lowercase hex without ``0x``."""

    text = _coerce_text(value)
    if not text:
        return ""
    normalized = text.lower()
    if normalized.startswith("0x"):
        normalized = normalized[2:]
    return normalized


def parse_proc_mounts(contents: str) -> dict[str, list[str]]:
    """Parse ``/proc/mounts`` into a device-to-mountpoints map."""

    mounts: dict[str, list[str]] = {}
    for line in contents.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        source = parts[0].replace("\\040", " ")
        target = parts[1].replace("\\040", " ")
        mounts.setdefault(source, []).append(target)
    return mounts


def _load_mount_map(proc_mounts_path: Path) -> dict[str, list[str]]:
    """Best-effort mount table loader."""

    try:
        return parse_proc_mounts(proc_mounts_path.read_text(encoding="utf-8"))
    except OSError:
        return {}


def _mount_has_uf2_info(mountpoint: str) -> bool:
    """Return whether a mountpoint looks like a UF2 bootloader volume."""

    base = Path(mountpoint)
    for filename in UF2_INFO_FILENAMES:
        try:
            if (base / filename).exists():
                return True
        except OSError:
            continue
    return False


def _label_is_uf2(filesystem_label: str | None) -> bool:
    """Return whether a filesystem label matches the Crazyradio UF2 volume."""

    return (filesystem_label or "").strip().lower() in UF2_FILESYSTEM_LABELS


def classify_usb_mode(
    vendor_id: str,
    product_id: str,
    *,
    info_uf2_present: bool,
    filesystem_label: str | None,
    manufacturer: str | None,
    product: str | None,
) -> str:
    """Classify the current Crazyradio compatibility mode."""

    manufacturer_l = (manufacturer or "").lower()
    product_l = (product or "").lower()

    if info_uf2_present or _label_is_uf2(filesystem_label):
        return "uf2_bootloader"
    if (vendor_id, product_id) == PA_EMULATION_USB_ID:
        return "pa_emulation"
    if (vendor_id, product_id) in CRAZYRADIO2_NATIVE_USB_IDS:
        return "crazyradio2_native"
    if (vendor_id, product_id) in UF2_BOOTLOADER_USB_IDS and "crazyradio" in f"{manufacturer_l} {product_l}":
        return "uf2_bootloader"
    return "unknown"


def recommendation_for_mode(mode: str) -> str:
    """Return the next concrete action for the detected mode."""

    if mode == "pa_emulation":
        return "legacy-compatible path; current cflib/cfclient can validate and use this mode"
    if mode == "uf2_bootloader":
        return "flash the intended UF2: PA emulation for current cflib/cfclient, native firmware for native/swarm stacks"
    if mode == "crazyradio2_native":
        return "native Crazyradio 2 mode detected; keep for native/swarm workflows, but the cflib low-level radio probe is PA-compatible only"
    return "verify the attached device, firmware, and USB permissions"


def _is_relevant_bitcraze_device(
    vendor_id: str,
    product_id: str,
    manufacturer: str | None,
    product: str | None,
    *,
    filesystem_label: str | None = None,
    info_uf2_present: bool = False,
) -> bool:
    """Return whether a USB device looks Crazyradio/Bitcraze-related."""

    manufacturer_l = (manufacturer or "").lower()
    product_l = (product or "").lower()
    if (vendor_id, product_id) == PA_EMULATION_USB_ID:
        return True
    if (vendor_id, product_id) in CRAZYRADIO2_NATIVE_USB_IDS:
        return True
    if info_uf2_present or _label_is_uf2(filesystem_label):
        return True
    if (vendor_id, product_id) in UF2_BOOTLOADER_USB_IDS and "crazyradio" in f"{manufacturer_l} {product_l}":
        return True
    return "bitcraze" in manufacturer_l or "crazyradio" in product_l or "crazyradio2" in product_l


def _iter_usb_sysfs_devices(sysfs_root: Path) -> Iterable[Path]:
    """Yield sysfs USB devices."""

    try:
        children = sorted(sysfs_root.iterdir(), key=lambda path: path.name)
    except OSError:
        return

    for device_path in children:
        if (device_path / "idVendor").exists() and (device_path / "idProduct").exists():
            yield device_path


def _maybe_import_pyudev() -> Any | None:
    """Import pyudev lazily so the sysfs fallback still works."""

    try:
        import pyudev  # type: ignore[import-not-found]  # pylint: disable=import-error

        return pyudev
    except Exception:
        return None


def _pyudev_find_usb_parent(device: Any) -> Any | None:
    """Walk the udev parent chain until the owning USB device is reached."""

    current = device
    while current is not None:
        subsystem = getattr(current, "subsystem", None)
        device_type = getattr(current, "device_type", None)
        if subsystem == "usb" and device_type == "usb_device":
            return current
        current = getattr(current, "parent", None)
    return None


def _lookup_label_for_device_node(device_node: str) -> str | None:
    """Best-effort reverse lookup from a block node to /dev/disk/by-label."""

    sys_name = Path(device_node).name
    labels_dir = Path("/dev/disk/by-label")
    try:
        for entry in labels_dir.iterdir():
            try:
                if entry.resolve().name == sys_name:
                    return entry.name
            except OSError:
                continue
    except (FileNotFoundError, OSError):
        return None
    return None


def _build_devices_from_pyudev(proc_mounts_path: Path) -> list[BitcrazeUsbDevice]:
    """Enumerate Crazyradio-related devices through pyudev/libudev."""

    pyudev = _maybe_import_pyudev()
    if pyudev is None:
        raise RuntimeError("pyudev not available")

    context = pyudev.Context()
    mount_map = _load_mount_map(proc_mounts_path)

    devices_by_sysfs: dict[str, dict[str, Any]] = {}
    for usb_device in context.list_devices(subsystem="usb", DEVTYPE="usb_device"):
        vendor_id = _normalize_usb_hex(usb_device.get("ID_VENDOR_ID") or usb_device.attributes.get("idVendor"))
        product_id = _normalize_usb_hex(usb_device.get("ID_MODEL_ID") or usb_device.attributes.get("idProduct"))
        manufacturer = _coerce_text(usb_device.attributes.get("manufacturer")) or _coerce_text(
            usb_device.get("ID_VENDOR_FROM_DATABASE")
        )
        product = _coerce_text(usb_device.attributes.get("product")) or _coerce_text(
            usb_device.get("ID_MODEL_FROM_DATABASE")
        )
        serial = _coerce_text(usb_device.attributes.get("serial")) or _coerce_text(usb_device.get("ID_SERIAL_SHORT"))
        bcd_device = _normalize_usb_hex(usb_device.attributes.get("bcdDevice")) or None

        if not vendor_id or not product_id:
            continue

        devices_by_sysfs[str(usb_device.sys_path)] = {
            "sysfs_path": str(usb_device.sys_path),
            "vendor_id": vendor_id,
            "product_id": product_id,
            "bcd_device": bcd_device,
            "manufacturer": manufacturer,
            "product": product,
            "serial": serial,
            "tty_nodes": set(),
            "block_devices": set(),
            "mountpoints": set(),
            "filesystem_label": None,
            "info_uf2_present": False,
        }

    if not devices_by_sysfs:
        return []

    for tty_device in context.list_devices(subsystem="tty"):
        usb_parent = _pyudev_find_usb_parent(tty_device)
        if usb_parent is None:
            continue
        record = devices_by_sysfs.get(str(usb_parent.sys_path))
        if record is None:
            continue
        if tty_device.device_node:
            record["tty_nodes"].add(str(tty_device.device_node))

    for block_device in context.list_devices(subsystem="block"):
        usb_parent = _pyudev_find_usb_parent(block_device)
        if usb_parent is None:
            continue
        record = devices_by_sysfs.get(str(usb_parent.sys_path))
        if record is None:
            continue

        if block_device.device_node:
            node = str(block_device.device_node)
            record["block_devices"].add(node)
            for mountpoint in mount_map.get(node, ()):
                record["mountpoints"].add(mountpoint)
                if _mount_has_uf2_info(mountpoint):
                    record["info_uf2_present"] = True
            if not record["filesystem_label"]:
                record["filesystem_label"] = _coerce_text(block_device.get("ID_FS_LABEL")) or _lookup_label_for_device_node(
                    node
                )

    devices: list[BitcrazeUsbDevice] = []
    for record in sorted(devices_by_sysfs.values(), key=lambda item: str(item["sysfs_path"])):
        manufacturer = record["manufacturer"] if isinstance(record["manufacturer"], str) else None
        product = record["product"] if isinstance(record["product"], str) else None
        filesystem_label = record["filesystem_label"] if isinstance(record["filesystem_label"], str) else None
        info_uf2_present = bool(record["info_uf2_present"])
        vendor_id = str(record["vendor_id"])
        product_id = str(record["product_id"])

        if not _is_relevant_bitcraze_device(
            vendor_id,
            product_id,
            manufacturer,
            product,
            filesystem_label=filesystem_label,
            info_uf2_present=info_uf2_present,
        ):
            continue

        mode = classify_usb_mode(
            vendor_id,
            product_id,
            info_uf2_present=info_uf2_present,
            filesystem_label=filesystem_label,
            manufacturer=manufacturer,
            product=product,
        )

        devices.append(
            BitcrazeUsbDevice(
                sysfs_path=str(record["sysfs_path"]),
                vendor_id=str(record["vendor_id"]),
                product_id=str(record["product_id"]),
                bcd_device=record["bcd_device"] if isinstance(record["bcd_device"], str) else None,
                manufacturer=manufacturer,
                product=product,
                serial=record["serial"] if isinstance(record["serial"], str) else None,
                tty_nodes=tuple(sorted(str(item) for item in record["tty_nodes"])),
                block_devices=tuple(sorted(str(item) for item in record["block_devices"])),
                mountpoints=tuple(sorted(str(item) for item in record["mountpoints"])),
                filesystem_label=filesystem_label,
                info_uf2_present=info_uf2_present,
                mode=mode,
                cflib_probe_compatible=(mode == "pa_emulation"),
                recommendation=recommendation_for_mode(mode),
            )
        )

    return devices


def _build_devices_from_sysfs(
    *,
    sysfs_root: Path,
    proc_mounts_path: Path,
) -> list[BitcrazeUsbDevice]:
    """Enumerate Crazyradio-related devices from sysfs."""

    mount_map = _load_mount_map(proc_mounts_path)
    devices: list[BitcrazeUsbDevice] = []

    for device_path in _iter_usb_sysfs_devices(sysfs_root):
        vendor_id = _normalize_usb_hex(_read_text(device_path / "idVendor"))
        product_id = _normalize_usb_hex(_read_text(device_path / "idProduct"))
        bcd_device = _normalize_usb_hex(_read_text(device_path / "bcdDevice")) or None
        manufacturer = _read_text(device_path / "manufacturer")
        product = _read_text(device_path / "product")
        serial = _read_text(device_path / "serial")

        tty_nodes = tuple(
            sorted(f"/dev/{entry.name}" for entry in device_path.glob("**/tty*") if entry.name.startswith("tty"))
        )
        block_devices = tuple(sorted(f"/dev/{entry.name}" for entry in device_path.glob("**/block/*")))

        mountpoints: list[str] = []
        filesystem_label: str | None = None
        for block_device in block_devices:
            mountpoints.extend(mount_map.get(block_device, ()))
            if filesystem_label is None:
                filesystem_label = _lookup_label_for_device_node(block_device)
        deduped_mountpoints = tuple(sorted(dict.fromkeys(mountpoints)))
        info_uf2_present = any(_mount_has_uf2_info(mountpoint) for mountpoint in deduped_mountpoints)

        if not _is_relevant_bitcraze_device(
            vendor_id,
            product_id,
            manufacturer,
            product,
            filesystem_label=filesystem_label,
            info_uf2_present=info_uf2_present,
        ):
            continue

        mode = classify_usb_mode(
            vendor_id,
            product_id,
            info_uf2_present=info_uf2_present,
            filesystem_label=filesystem_label,
            manufacturer=manufacturer,
            product=product,
        )

        devices.append(
            BitcrazeUsbDevice(
                sysfs_path=str(device_path),
                vendor_id=vendor_id,
                product_id=product_id,
                bcd_device=bcd_device,
                manufacturer=manufacturer,
                product=product,
                serial=serial,
                tty_nodes=tty_nodes,
                block_devices=block_devices,
                mountpoints=deduped_mountpoints,
                filesystem_label=filesystem_label,
                info_uf2_present=info_uf2_present,
                mode=mode,
                cflib_probe_compatible=(mode == "pa_emulation"),
                recommendation=recommendation_for_mode(mode),
            )
        )

    return devices


def enumerate_bitcraze_devices(
    *,
    sysfs_root: Path = Path("/sys/bus/usb/devices"),
    proc_mounts_path: Path = Path("/proc/mounts"),
) -> tuple[list[BitcrazeUsbDevice], str]:
    """Enumerate connected Bitcraze USB devices."""

    try:
        devices = _build_devices_from_pyudev(proc_mounts_path)
        return devices, "pyudev"
    except Exception:
        devices = _build_devices_from_sysfs(sysfs_root=sysfs_root, proc_mounts_path=proc_mounts_path)
        return devices, "sysfs"


def _isolated_subprocess_env() -> dict[str, str]:
    """Return a conservative environment for child Python probes."""

    env: dict[str, str] = {}
    for key in ("HOME", "LANG", "LC_ALL", "PATH", "TMPDIR"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    if "LANG" not in env and "LC_ALL" not in env:
        env["LANG"] = "C.UTF-8"
    if "PATH" not in env:
        env["PATH"] = "/usr/bin:/bin"
    return env


def _run_python_json(python_bin: Path, code: str) -> dict[str, object]:
    """Run a Python snippet and parse its JSON stdout."""

    completed = subprocess.run(
        [str(python_bin), "-I", "-c", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
        env=_isolated_subprocess_env(),
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise ValueError("child probe did not return a JSON object")
    return payload


def probe_workspace(workspace: Path, *, try_cflib_access: bool) -> WorkspaceProbe:
    """Inspect the Bitcraze workspace and optionally open the radio."""

    venv_python = workspace / ".venv" / "bin" / "python"
    if not workspace.exists() or not venv_python.exists():
        return WorkspaceProbe(
            workspace=str(workspace),
            exists=workspace.exists(),
            venv_python=str(venv_python) if venv_python.exists() else None,
            cflib_version=None,
            cfclient_version=None,
            workspace_probe_error=None,
            radio_access_attempted=False,
            radio_access_scope="not_attempted",
            radio_access_ok=False,
            radio_access_error=None,
            detected_radio_serials=(),
            radio_version=None,
        )

    versions: dict[str, object] | None = None
    workspace_probe_error: str | None = None
    try:
        versions = _run_python_json(
            venv_python,
            """
import importlib.metadata
import json

def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

print(json.dumps({
    "cflib_version": version("cflib"),
    "cfclient_version": version("cfclient"),
}))
""",
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError, OSError, ValueError) as exc:
        if isinstance(exc, subprocess.CalledProcessError):
            workspace_probe_error = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        else:
            workspace_probe_error = str(exc)

    cflib_version = None
    cfclient_version = None
    if versions is not None:
        cflib_value = versions.get("cflib_version")
        cfclient_value = versions.get("cfclient_version")
        cflib_version = str(cflib_value) if isinstance(cflib_value, str) else None
        cfclient_version = str(cfclient_value) if isinstance(cfclient_value, str) else None

    radio_access_attempted = False
    radio_access_scope = "not_attempted"
    radio_access_ok = False
    radio_access_error: str | None = None
    detected_radio_serials: tuple[str, ...] = ()
    radio_version: str | None = None

    if try_cflib_access and cflib_version and workspace_probe_error is None:
        radio_access_attempted = True
        radio_access_scope = "pa_compatible_usb"
        try:
            radio_probe = _run_python_json(
                venv_python,
                """
import json
from cflib.drivers.crazyradio import Crazyradio, get_serials

radio = Crazyradio()
result = {
    "serials": list(get_serials()),
    "version": str(radio.version),
}
radio.close()
print(json.dumps(result))
""",
            )
            radio_access_ok = True
            serials_value = radio_probe.get("serials", ())
            if isinstance(serials_value, list):
                detected_radio_serials = tuple(str(item) for item in serials_value)
            version_value = radio_probe.get("version")
            if isinstance(version_value, str):
                radio_version = version_value
        except subprocess.CalledProcessError as exc:
            radio_access_error = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        except (json.JSONDecodeError, subprocess.TimeoutExpired, OSError, ValueError) as exc:
            radio_access_error = str(exc)

    return WorkspaceProbe(
        workspace=str(workspace),
        exists=True,
        venv_python=str(venv_python),
        cflib_version=cflib_version,
        cfclient_version=cfclient_version,
        workspace_probe_error=workspace_probe_error,
        radio_access_attempted=radio_access_attempted,
        radio_access_scope=radio_access_scope,
        radio_access_ok=radio_access_ok,
        radio_access_error=radio_access_error,
        detected_radio_serials=detected_radio_serials,
        radio_version=radio_version,
    )


def validate_report(
    *,
    devices: list[BitcrazeUsbDevice],
    workspace_probe: WorkspaceProbe,
    expect_mode: str | None,
    require_cflib_access: bool,
) -> list[str]:
    """Validate expectations against the current report."""

    failures: list[str] = []
    if not devices:
        failures.append("no Bitcraze USB device detected")
        return failures

    if expect_mode is not None and not any(device.mode == expect_mode for device in devices):
        modes = ", ".join(device.mode for device in devices)
        failures.append(f"expected mode {expect_mode}, found {modes}")

    if require_cflib_access:
        pa_devices = [device for device in devices if device.cflib_probe_compatible]
        if not pa_devices:
            failures.append(
                "require-cflib-access validates the PA-compatible cflib/crazyradio path, "
                "but no PA-compatible Crazyradio device is currently connected"
            )
        elif workspace_probe.workspace_probe_error:
            failures.append(f"workspace probe failed: {workspace_probe.workspace_probe_error}")
        elif not workspace_probe.cflib_version:
            failures.append("workspace does not have cflib installed")
        elif not workspace_probe.radio_access_ok:
            failures.append(workspace_probe.radio_access_error or "cflib could not open the Crazyradio")

    return failures


def format_human_report(
    devices: list[BitcrazeUsbDevice],
    workspace_probe: WorkspaceProbe,
    failures: list[str],
    *,
    enumeration_backend: str,
) -> str:
    """Render the report in a compact human-readable form."""

    lines = [
        f"Detected Bitcraze USB devices: {len(devices)}",
        f"USB enumeration backend: {enumeration_backend}",
    ]
    for device in devices:
        lines.append(
            " - "
            f"{device.product or 'unknown'} {device.vendor_id}:{device.product_id} "
            f"bcd={device.bcd_device or '-'} "
            f"mode={device.mode} "
            f"serial={device.serial or '-'} "
            f"label={device.filesystem_label or '-'} "
            f"mount={','.join(device.mountpoints) or '-'} "
            f"tty={','.join(device.tty_nodes) or '-'}"
        )
        lines.append(
            "   "
            f"cflib_probe_compatible={device.cflib_probe_compatible} "
            f"recommendation={device.recommendation}"
        )

    lines.append(
        "Workspace "
        f"{workspace_probe.workspace}: "
        f"exists={workspace_probe.exists} "
        f"cflib={workspace_probe.cflib_version or '-'} "
        f"cfclient={workspace_probe.cfclient_version or '-'} "
        f"venv_python={workspace_probe.venv_python or '-'}"
    )
    if workspace_probe.workspace_probe_error:
        lines.append(f"workspace probe error: {workspace_probe.workspace_probe_error}")
    if workspace_probe.radio_access_attempted:
        lines.append(
            "cflib radio access: "
            f"scope={workspace_probe.radio_access_scope} "
            f"ok={workspace_probe.radio_access_ok} "
            f"version={workspace_probe.radio_version or '-'} "
            f"serials={','.join(workspace_probe.detected_radio_serials) or '-'}"
        )
    if workspace_probe.radio_access_error:
        lines.append(f"cflib error: {workspace_probe.radio_access_error}")

    if failures:
        lines.append("Failures:")
        for failure in failures:
            lines.append(f" - {failure}")

    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("/twinr/bitcraze"),
        help="Bitcraze workspace root to inspect",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full report as JSON",
    )
    parser.add_argument(
        "--expect-mode",
        choices=["pa_emulation", "uf2_bootloader", "crazyradio2_native", "unknown"],
        help="Fail if no detected Bitcraze device matches the expected mode",
    )
    parser.add_argument(
        "--require-cflib-access",
        action="store_true",
        # BREAKING: This flag now explicitly validates only the PA-compatible cflib/crazyradio
        # access path. Native Crazyradio 2 mode is reported separately and is no longer treated
        # as an implicit target for this probe.
        help="Fail unless the workspace can open a PA-compatible Crazyradio through cflib",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Crazyradio probe CLI."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)

    devices, enumeration_backend = enumerate_bitcraze_devices()
    try_cflib_access = any(device.cflib_probe_compatible for device in devices)
    workspace_probe = probe_workspace(args.workspace, try_cflib_access=try_cflib_access)
    failures = validate_report(
        devices=devices,
        workspace_probe=workspace_probe,
        expect_mode=args.expect_mode,
        require_cflib_access=args.require_cflib_access,
    )

    # BREAKING: JSON output now contains ``enumeration_backend`` and richer per-device /
    # workspace fields (for example ``bcd_device``, ``filesystem_label``,
    # ``cflib_probe_compatible``, ``workspace_probe_error`` and ``radio_access_scope``).
    payload = {
        "enumeration_backend": enumeration_backend,
        "devices": [asdict(device) for device in devices],
        "workspace": asdict(workspace_probe),
        "failures": failures,
    }

    if args.json:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(format_human_report(devices, workspace_probe, failures, enumeration_backend=enumeration_backend))

    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
