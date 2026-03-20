#!/usr/bin/env python3
"""Install the pinned Twinr-local Node.js runtime for the WhatsApp worker.

Usage:
    python3 hardware/ops/install_whatsapp_node_runtime.py

The script downloads the official Node.js archive for the current host, verifies
the archive checksum against the release manifest, and stages the runtime under
``state/tools/node-v<version>-<platform>-<arch>`` so Twinr can launch the
Baileys worker without depending on a separately managed system ``node``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
from urllib import request

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = PROJECT_ROOT / "src" / "twinr" / "channels" / "whatsapp" / "node_runtime.py"
_SPEC = importlib.util.spec_from_file_location("twinr_channels_whatsapp_node_runtime", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load Node runtime helper from {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
PINNED_WHATSAPP_NODE_VERSION = _MODULE.PINNED_WHATSAPP_NODE_VERSION
detect_whatsapp_node_runtime_spec = _MODULE.detect_whatsapp_node_runtime_spec


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the WhatsApp Node runtime installer."""

    parser = argparse.ArgumentParser(description="Install the pinned Twinr-local Node.js runtime for WhatsApp.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Twinr project root that should receive the runtime under state/tools.",
    )
    parser.add_argument(
        "--version",
        default=PINNED_WHATSAPP_NODE_VERSION,
        help="Pinned Node.js version to install, without the leading v.",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="Optional Node.js platform override such as linux or darwin.",
    )
    parser.add_argument(
        "--machine",
        default=None,
        help="Optional architecture override such as arm64 or x86_64.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing staged runtime.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=30.0,
        help="Network timeout for archive and checksum downloads.",
    )
    return parser


def install_whatsapp_node_runtime(
    *,
    project_root: str | Path,
    version: str = PINNED_WHATSAPP_NODE_VERSION,
    system_name: str | None = None,
    machine_name: str | None = None,
    force: bool = False,
    timeout_s: float = 30.0,
) -> dict[str, object]:
    """Download, verify, and stage the pinned Node.js runtime."""

    spec = detect_whatsapp_node_runtime_spec(
        project_root,
        version=version,
        system_name=system_name,
        machine_name=machine_name,
    )
    install_root = spec.install_root
    binary_path = spec.binary_path
    if binary_path.is_file() and not force:
        return _result_payload(spec, status="reused")

    install_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="twinr-node-runtime-", dir=str(install_root.parent)) as temp_dir:
        temp_root = Path(temp_dir)
        archive_path = temp_root / spec.archive_name
        _download_file(spec.download_url, archive_path, timeout_s=timeout_s)
        archive_sha256 = _sha256_file(archive_path)
        _verify_archive_checksum(spec.shasums_url, spec.archive_name, archive_sha256, timeout_s=timeout_s)
        extracted_root = _extract_archive(archive_path, temp_root=temp_root, expected_slug=spec.runtime_slug)
        if force and install_root.exists():
            shutil.rmtree(install_root)
        shutil.move(str(extracted_root), str(install_root))

    if not binary_path.is_file():
        raise RuntimeError(f"Installed archive did not provide the expected node binary: {binary_path}")
    return _result_payload(spec, status="installed")


def _download_file(url: str, destination: Path, *, timeout_s: float) -> None:
    """Download one remote file to disk without buffering the whole payload in RAM."""

    with request.urlopen(url, timeout=timeout_s) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 hex digest for one local file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_archive_checksum(url: str, archive_name: str, archive_sha256: str, *, timeout_s: float) -> None:
    """Verify the downloaded archive against the official Node.js checksum manifest."""

    with request.urlopen(url, timeout=timeout_s) as response:
        manifest = response.read().decode("utf-8")
    expected_sha256 = None
    for raw_line in manifest.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        if parts[-1].lstrip("*") == archive_name:
            expected_sha256 = parts[0].lower()
            break
    if not expected_sha256:
        raise RuntimeError(f"Could not find {archive_name} in the Node.js checksum manifest.")
    if expected_sha256 != archive_sha256.lower():
        raise RuntimeError(
            f"Checksum mismatch for {archive_name}: expected {expected_sha256}, got {archive_sha256.lower()}"
        )


def _extract_archive(archive_path: Path, *, temp_root: Path, expected_slug: str) -> Path:
    """Extract one Node.js archive into a temporary directory and return its top-level folder."""

    extract_root = temp_root / "extract"
    extract_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:xz") as archive:
        for member in archive.getmembers():
            destination = (extract_root / member.name).resolve(strict=False)
            if not _path_is_within(destination, extract_root.resolve(strict=False)):
                raise RuntimeError(f"Refusing to extract archive member outside the target root: {member.name}")
        archive.extractall(path=extract_root)
    extracted_root = extract_root / expected_slug
    if not extracted_root.is_dir():
        raise RuntimeError(f"Expected extracted Node.js directory {expected_slug!r} under {extract_root}")
    return extracted_root


def _path_is_within(candidate: Path, root: Path) -> bool:
    """Return whether ``candidate`` stays inside ``root``."""

    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _result_payload(spec, *, status: str) -> dict[str, object]:
    """Render one compact machine-readable install result."""

    return {
        "status": status,
        "version": spec.version,
        "platform": spec.platform_name,
        "arch": spec.arch,
        "install_root": str(spec.install_root),
        "node_binary": str(spec.binary_path),
        "download_url": spec.download_url,
    }


def main() -> int:
    """Run the bounded installer and print a compact JSON result."""

    args = build_parser().parse_args()
    result = install_whatsapp_node_runtime(
        project_root=args.project_root,
        version=args.version,
        system_name=args.system,
        machine_name=args.machine,
        force=args.force,
        timeout_s=args.timeout_s,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
