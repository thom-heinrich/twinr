#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
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
from pathlib import Path
import subprocess
import sys
from typing import Iterable


PA_EMULATION_USB_ID = ("1915", "7777")
CRAZYRADIO2_USB_ID = ("35f0", "bad2")


@dataclass(frozen=True)
class BitcrazeUsbDevice:
    """Describe one detected Bitcraze USB device."""

    sysfs_path: str
    vendor_id: str
    product_id: str
    manufacturer: str | None
    product: str | None
    serial: str | None
    tty_nodes: tuple[str, ...]
    block_devices: tuple[str, ...]
    mountpoints: tuple[str, ...]
    info_uf2_present: bool
    mode: str
    recommendation: str


@dataclass(frozen=True)
class WorkspaceProbe:
    """Describe the Bitcraze Python workspace state."""

    workspace: str
    exists: bool
    venv_python: str | None
    cflib_version: str | None
    cfclient_version: str | None
    radio_access_attempted: bool
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


def parse_proc_mounts(contents: str) -> dict[str, list[str]]:
    """Parse ``/proc/mounts`` into a device-to-mountpoints map.

    Args:
        contents: Raw contents of ``/proc/mounts``.

    Returns:
        Mapping of source device path to one or more mountpoints.
    """

    mounts: dict[str, list[str]] = {}
    for line in contents.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        source = parts[0].replace("\\040", " ")
        target = parts[1].replace("\\040", " ")
        mounts.setdefault(source, []).append(target)
    return mounts


def classify_usb_mode(
    vendor_id: str,
    product_id: str,
    *,
    info_uf2_present: bool,
) -> str:
    """Classify the current Crazyradio compatibility mode.

    Args:
        vendor_id: Lowercase hexadecimal USB vendor id without the ``0x`` prefix.
        product_id: Lowercase hexadecimal USB product id without the ``0x`` prefix.
        info_uf2_present: Whether the mounted volume exposes ``INFO_UF2.TXT``.

    Returns:
        One of ``pa_emulation``, ``uf2_bootloader``, ``crazyradio2_native``,
        or ``unknown``.
    """

    if (vendor_id, product_id) == PA_EMULATION_USB_ID:
        return "pa_emulation"
    if (vendor_id, product_id) == CRAZYRADIO2_USB_ID:
        if info_uf2_present:
            return "uf2_bootloader"
        return "crazyradio2_native"
    return "unknown"


def recommendation_for_mode(mode: str) -> str:
    """Return the next concrete action for the detected mode."""

    if mode == "pa_emulation":
        return "ready for cflib/cfclient"
    if mode == "uf2_bootloader":
        return "flash the pinned PA emulation UF2 for cflib compatibility"
    if mode == "crazyradio2_native":
        return "switch to PA emulation if you want current cflib/cfclient compatibility"
    return "verify the attached device and firmware"


def _iter_bitcraze_sysfs_devices(sysfs_root: Path) -> Iterable[Path]:
    """Yield sysfs USB devices that look Bitcraze-related."""

    for device_path in sorted(sysfs_root.iterdir()):
        vendor_id = _read_text(device_path / "idVendor")
        product_id = _read_text(device_path / "idProduct")
        manufacturer = (_read_text(device_path / "manufacturer") or "").lower()
        product = (_read_text(device_path / "product") or "").lower()

        if vendor_id is None or product_id is None:
            continue
        if (vendor_id, product_id) in {PA_EMULATION_USB_ID, CRAZYRADIO2_USB_ID}:
            yield device_path
            continue
        if "bitcraze" in manufacturer or "crazyradio" in product:
            yield device_path


def enumerate_bitcraze_devices(
    *,
    sysfs_root: Path = Path("/sys/bus/usb/devices"),
    proc_mounts_path: Path = Path("/proc/mounts"),
) -> list[BitcrazeUsbDevice]:
    """Enumerate connected Bitcraze USB devices from sysfs."""

    mount_map = parse_proc_mounts(proc_mounts_path.read_text(encoding="utf-8"))
    devices: list[BitcrazeUsbDevice] = []

    for device_path in _iter_bitcraze_sysfs_devices(sysfs_root):
        vendor_id = _read_text(device_path / "idVendor") or ""
        product_id = _read_text(device_path / "idProduct") or ""
        manufacturer = _read_text(device_path / "manufacturer")
        product = _read_text(device_path / "product")
        serial = _read_text(device_path / "serial")

        tty_nodes = tuple(
            sorted(f"/dev/{entry.name}" for entry in device_path.glob("**/tty*") if entry.name.startswith("tty"))
        )
        block_devices = tuple(
            sorted(f"/dev/{entry.name}" for entry in device_path.glob("**/block/*"))
        )

        mountpoints: list[str] = []
        for block_device in block_devices:
            mountpoints.extend(mount_map.get(block_device, ()))
        deduped_mountpoints = tuple(sorted(dict.fromkeys(mountpoints)))
        info_uf2_present = any((Path(mountpoint) / "INFO_UF2.TXT").exists() for mountpoint in deduped_mountpoints)
        mode = classify_usb_mode(vendor_id, product_id, info_uf2_present=info_uf2_present)

        devices.append(
            BitcrazeUsbDevice(
                sysfs_path=str(device_path),
                vendor_id=vendor_id,
                product_id=product_id,
                manufacturer=manufacturer,
                product=product,
                serial=serial,
                tty_nodes=tty_nodes,
                block_devices=block_devices,
                mountpoints=deduped_mountpoints,
                info_uf2_present=info_uf2_present,
                mode=mode,
                recommendation=recommendation_for_mode(mode),
            )
        )

    return devices


def _run_python_json(python_bin: Path, code: str) -> dict[str, object]:
    """Run a Python snippet and parse its JSON stdout."""

    completed = subprocess.run(
        [str(python_bin), "-c", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return json.loads(completed.stdout)


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
            radio_access_attempted=False,
            radio_access_ok=False,
            radio_access_error=None,
            detected_radio_serials=(),
            radio_version=None,
        )

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

    radio_access_attempted = False
    radio_access_ok = False
    radio_access_error: str | None = None
    detected_radio_serials: tuple[str, ...] = ()
    radio_version: str | None = None

    if try_cflib_access and versions.get("cflib_version"):
        radio_access_attempted = True
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
            radio_version = str(radio_probe.get("version"))
        except subprocess.CalledProcessError as exc:
            radio_access_error = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        except (json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
            radio_access_error = str(exc)

    return WorkspaceProbe(
        workspace=str(workspace),
        exists=True,
        venv_python=str(venv_python),
        cflib_version=versions.get("cflib_version"),  # type: ignore[arg-type]
        cfclient_version=versions.get("cfclient_version"),  # type: ignore[arg-type]
        radio_access_attempted=radio_access_attempted,
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
        if not workspace_probe.cflib_version:
            failures.append("workspace does not have cflib installed")
        elif not workspace_probe.radio_access_ok:
            failures.append(workspace_probe.radio_access_error or "cflib could not open the Crazyradio")

    return failures


def format_human_report(
    devices: list[BitcrazeUsbDevice],
    workspace_probe: WorkspaceProbe,
    failures: list[str],
) -> str:
    """Render the report in a compact human-readable form."""

    lines = [f"Detected Bitcraze USB devices: {len(devices)}"]
    for device in devices:
        lines.append(
            " - "
            f"{device.product or 'unknown'} {device.vendor_id}:{device.product_id} "
            f"mode={device.mode} mount={','.join(device.mountpoints) or '-'} "
            f"tty={','.join(device.tty_nodes) or '-'}"
        )
        lines.append(f"   recommendation={device.recommendation}")

    lines.append(
        "Workspace "
        f"{workspace_probe.workspace}: "
        f"exists={workspace_probe.exists} "
        f"cflib={workspace_probe.cflib_version or '-'} "
        f"cfclient={workspace_probe.cfclient_version or '-'}"
    )
    if workspace_probe.radio_access_attempted:
        lines.append(
            "cflib radio access: "
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
        help="Fail unless the workspace can open the Crazyradio through cflib",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Crazyradio probe CLI."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)

    devices = enumerate_bitcraze_devices()
    try_cflib_access = bool(args.require_cflib_access or any(device.mode == "pa_emulation" for device in devices))
    workspace_probe = probe_workspace(args.workspace, try_cflib_access=try_cflib_access)
    failures = validate_report(
        devices=devices,
        workspace_probe=workspace_probe,
        expect_mode=args.expect_mode,
        require_cflib_access=args.require_cflib_access,
    )

    payload = {
        "devices": [asdict(device) for device in devices],
        "workspace": asdict(workspace_probe),
        "failures": failures,
    }

    if args.json:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(format_human_report(devices, workspace_probe, failures))

    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
