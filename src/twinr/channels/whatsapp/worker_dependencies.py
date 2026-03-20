"""Validate and install the WhatsApp worker's locked Node dependencies."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from twinr.channels.contracts import ChannelTransportError


_SubprocessRunner = Any
_LOCK_MARKER_NAME = ".twinr-package-lock.sha256"
_INSTALL_TIMEOUT_S = 300.0


@dataclass(frozen=True, slots=True)
class WhatsAppWorkerDependencyProbe:
    """Summarize whether the worker's pinned npm dependencies are usable."""

    ready: bool
    detail: str
    package_json_path: Path
    package_lock_path: Path
    node_modules_path: Path
    marker_path: Path
    dependency_names: tuple[str, ...]
    lock_sha256: str | None


def probe_whatsapp_worker_dependencies(worker_root: Path) -> WhatsAppWorkerDependencyProbe:
    """Inspect the worker folder and report whether its locked packages are ready."""

    package_json_path = worker_root / "package.json"
    package_lock_path = worker_root / "package-lock.json"
    node_modules_path = worker_root / "node_modules"
    marker_path = node_modules_path / _LOCK_MARKER_NAME

    if not worker_root.is_dir():
        return WhatsAppWorkerDependencyProbe(
            ready=False,
            detail=f"WhatsApp worker folder is missing: {worker_root}",
            package_json_path=package_json_path,
            package_lock_path=package_lock_path,
            node_modules_path=node_modules_path,
            marker_path=marker_path,
            dependency_names=(),
            lock_sha256=None,
        )

    if not package_json_path.is_file():
        return WhatsAppWorkerDependencyProbe(
            ready=False,
            detail=f"WhatsApp worker package.json is missing: {package_json_path}",
            package_json_path=package_json_path,
            package_lock_path=package_lock_path,
            node_modules_path=node_modules_path,
            marker_path=marker_path,
            dependency_names=(),
            lock_sha256=None,
        )

    dependency_names, dependency_error = _load_dependency_names(package_json_path)
    if dependency_error is not None:
        return WhatsAppWorkerDependencyProbe(
            ready=False,
            detail=dependency_error,
            package_json_path=package_json_path,
            package_lock_path=package_lock_path,
            node_modules_path=node_modules_path,
            marker_path=marker_path,
            dependency_names=(),
            lock_sha256=None,
        )

    if not dependency_names:
        return WhatsAppWorkerDependencyProbe(
            ready=True,
            detail=f"WhatsApp worker declares no npm dependencies under {worker_root}.",
            package_json_path=package_json_path,
            package_lock_path=package_lock_path,
            node_modules_path=node_modules_path,
            marker_path=marker_path,
            dependency_names=(),
            lock_sha256=None,
        )

    if not package_lock_path.is_file():
        return WhatsAppWorkerDependencyProbe(
            ready=False,
            detail=f"WhatsApp worker package-lock.json is missing: {package_lock_path}",
            package_json_path=package_json_path,
            package_lock_path=package_lock_path,
            node_modules_path=node_modules_path,
            marker_path=marker_path,
            dependency_names=dependency_names,
            lock_sha256=None,
        )

    lock_sha256 = _sha256_file(package_lock_path)
    missing_dependencies = tuple(
        dependency_name
        for dependency_name in dependency_names
        if not _installed_dependency_package_path(node_modules_path, dependency_name).is_file()
    )
    if missing_dependencies:
        return WhatsAppWorkerDependencyProbe(
            ready=False,
            detail=(
                "WhatsApp worker npm dependencies are missing under "
                f"{node_modules_path}; run `npm ci` in {worker_root}. "
                f"Missing: {', '.join(missing_dependencies)}"
            ),
            package_json_path=package_json_path,
            package_lock_path=package_lock_path,
            node_modules_path=node_modules_path,
            marker_path=marker_path,
            dependency_names=dependency_names,
            lock_sha256=lock_sha256,
        )

    try:
        marker_text = marker_path.read_text(encoding="utf-8").strip()
    except OSError:
        marker_text = ""

    if marker_text != lock_sha256:
        return WhatsAppWorkerDependencyProbe(
            ready=False,
            detail=(
                "WhatsApp worker package-lock marker is stale or missing under "
                f"{marker_path}; run `npm ci` in {worker_root} to realign node_modules."
            ),
            package_json_path=package_json_path,
            package_lock_path=package_lock_path,
            node_modules_path=node_modules_path,
            marker_path=marker_path,
            dependency_names=dependency_names,
            lock_sha256=lock_sha256,
        )

    return WhatsAppWorkerDependencyProbe(
        ready=True,
        detail=f"WhatsApp worker npm dependencies are ready under {worker_root}.",
        package_json_path=package_json_path,
        package_lock_path=package_lock_path,
        node_modules_path=node_modules_path,
        marker_path=marker_path,
        dependency_names=dependency_names,
        lock_sha256=lock_sha256,
    )


def ensure_whatsapp_worker_dependencies(
    *,
    worker_root: Path,
    node_binary: str,
    subprocess_runner: _SubprocessRunner = subprocess.run,
) -> WhatsAppWorkerDependencyProbe:
    """Install the worker's pinned packages when the lockfile state is not ready."""

    probe = probe_whatsapp_worker_dependencies(worker_root)
    if probe.ready:
        return probe

    npm_binary = resolve_whatsapp_worker_npm_binary(node_binary)
    if not npm_binary:
        raise ChannelTransportError(
            "WhatsApp worker npm dependencies are not ready and no npm binary is available. "
            f"Expected a sibling npm next to {node_binary!r} or one on PATH."
        )

    try:
        completed = subprocess_runner(
            [npm_binary, "ci", "--no-audit", "--no-fund"],
            cwd=str(worker_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            timeout=_INSTALL_TIMEOUT_S,
            env=_build_whatsapp_worker_npm_env(node_binary),
        )
    except OSError as exc:
        raise ChannelTransportError(
            f"Could not execute npm ci for the WhatsApp worker via {npm_binary!r}: {exc}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ChannelTransportError(
            "Timed out while running `npm ci` for the WhatsApp worker "
            f"under {worker_root} after {_INSTALL_TIMEOUT_S:.0f}s."
        ) from exc

    if completed.returncode != 0:
        detail = _normalize_command_detail(completed.stderr or completed.stdout or "")
        raise ChannelTransportError(
            f"`npm ci` failed for the WhatsApp worker under {worker_root}: {detail or completed.returncode}"
        )

    post_probe = probe_whatsapp_worker_dependencies(worker_root)
    if post_probe.lock_sha256:
        post_probe.marker_path.parent.mkdir(parents=True, exist_ok=True)
        post_probe.marker_path.write_text(post_probe.lock_sha256 + "\n", encoding="utf-8")
        post_probe = probe_whatsapp_worker_dependencies(worker_root)
    if not post_probe.ready:
        raise ChannelTransportError(
            "WhatsApp worker dependencies are still not ready after `npm ci`: "
            f"{post_probe.detail}"
        )
    return post_probe


def resolve_whatsapp_worker_npm_binary(node_binary: str) -> str | None:
    """Resolve the npm executable that matches the configured Node runtime."""

    normalized = str(node_binary or "").strip()
    if normalized:
        node_path = Path(normalized).expanduser()
        for candidate in (
            node_path.with_name("npm"),
            node_path.with_name("npm.cmd"),
            node_path.with_name("npm-cli.js"),
        ):
            if candidate.is_file():
                return str(candidate)
    resolved = shutil.which("npm")
    if resolved:
        return str(resolved).strip() or None
    return None


def _build_whatsapp_worker_npm_env(node_binary: str) -> dict[str, str]:
    """Build the subprocess environment for npm commands.

    The staged Node runtime ships `npm` as a launcher script whose shebang uses
    `/usr/bin/env node`. If PATH still points at the system runtime first, npm
    falls back to the wrong Node major even when Twinr resolved the correct
    local `node` binary. Prefix the configured node directory so `npm ci` and
    the actual worker launch run under the same Node runtime.
    """

    environment = dict(os.environ)
    normalized = str(node_binary or "").strip()
    if not normalized:
        return environment
    node_dir = Path(normalized).expanduser().parent
    current_path = str(environment.get("PATH", "") or "")
    prefixed_entries = [str(node_dir)]
    if current_path:
        prefixed_entries.append(current_path)
    environment["PATH"] = os.pathsep.join(prefixed_entries)
    return environment


def _load_dependency_names(package_json_path: Path) -> tuple[tuple[str, ...], str | None]:
    try:
        payload = json.loads(package_json_path.read_text(encoding="utf-8"))
    except OSError as exc:
        return (), f"Could not read WhatsApp worker package.json at {package_json_path}: {exc}"
    except json.JSONDecodeError as exc:
        return (), f"Could not parse WhatsApp worker package.json at {package_json_path}: {exc}"

    dependencies = payload.get("dependencies")
    if dependencies is None:
        return (), None
    if not isinstance(dependencies, dict):
        return (), f"WhatsApp worker dependencies must be a JSON object in {package_json_path}"
    names = tuple(sorted(str(name).strip() for name in dependencies if str(name).strip()))
    return names, None


def _installed_dependency_package_path(node_modules_path: Path, dependency_name: str) -> Path:
    return node_modules_path.joinpath(*dependency_name.split("/"), "package.json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_command_detail(value: str, *, limit: int = 400) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
