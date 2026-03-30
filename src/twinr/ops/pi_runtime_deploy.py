# CHANGELOG: 2026-03-30
# BUG-1: Resolve repo-backed Pi units against the requested remote_root instead of hard-coding /twinr.
# BUG-2: Use systemctl is-enabled semantics so enabled-runtime/linked-runtime units are not silently skipped.
# BUG-3: Parse unit files with systemd.syntax-compatible whitespace/comment/continuation handling so valid Pi units are discovered reliably.
# BUG-4: Snapshot mutable local deploy artifacts and abort when the authoritative repo changes mid-deploy, preventing mixed-state rollouts.
# SEC-1: Reject unsafe remote paths and malformed service identifiers before they reach remote shell/systemctl commands.
# SEC-2: Serialize deploys on the Pi with a remote lock file to prevent interleaved installs/restarts from corrupting the runtime.
# IMP-1: Verify repo-backed units on the Pi with systemd-analyze verify before install/restart.
# IMP-2: Record the unit-verification summary in the deploy result for operator observability.

"""Deploy the leading Twinr repo and runtime config to the Pi acceptance host.

This module owns the operator workflow that turns the current authoritative
repo state in ``/home/thh/twinr`` into a restarted and verified Pi acceptance
runtime under ``/twinr``. It reuses the one-way repo mirror for code sync,
optionally overwrites the Pi runtime ``.env`` from the leading repo, refreshes
the editable install in the Pi virtualenv, explicitly syncs optional local
workspace runtime manifests such as ``browser_automation/runtime_requirements``
before installing them on the Pi, independently attests the mirrored repo
contents before restart, installs the productive base units plus any
repo-backed Pi runtime units that are already enabled on the host, supports
explicit first-rollout activation for optional Pi units, restarts the selected
services, and checks that they came back healthy.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any, Protocol, Sequence

from twinr.ops.pi_repo_mirror import (
    PiRepoMirrorCycleResult,
    PiRepoMirrorWatchdog,
    build_authoritative_repo_entry_digests,
)
from twinr.ops.pi_runtime_deploy_remote import (
    PiRemoteExecutor as _PiRemoteExecutor,
    PiPythonImportContractResult,
    PiRemoteRepoAttestationResult,
    PiSyncedFileResult,
    PiSystemdServiceState,
    attest_remote_repo_entries as _attest_remote_repo_entries,
    install_browser_automation_runtime_support as _install_browser_automation_runtime_support,
    install_editable_package as _install_editable_package,
    install_python_requirements_manifest as _install_python_requirements_manifest,
    install_service_units as _install_service_units,
    repair_runtime_state_permissions as _repair_runtime_state_permissions,
    refresh_python_bytecode as _refresh_python_bytecode,
    restart_services as _restart_services,
    run_env_contract_probe as _run_env_contract_probe,
    sync_authoritative_file as _sync_authoritative_file,
    verify_runtime_state_permissions as _verify_runtime_state_permissions,
    verify_python_import_contract as _verify_python_import_contract,
    wait_for_services as _wait_for_services,
)
from twinr.ops.self_coding_pi import load_pi_connection_settings

_SubprocessRunner = Any

DEFAULT_DEPLOY_SERVICES: tuple[str, ...] = (
    "twinr-remote-memory-watchdog.service",
    "twinr-runtime-supervisor.service",
    "twinr-web.service",
)
_BROWSER_AUTOMATION_RUNTIME_REQUIREMENTS = Path("browser_automation/runtime_requirements.txt")
_BROWSER_AUTOMATION_PLAYWRIGHT_BROWSERS = Path("browser_automation/playwright_browsers.txt")
_PI_RUNTIME_REQUIREMENTS = Path("hardware/ops/pi_runtime_requirements.txt")
_PI_RUNTIME_BYTECODE_RELATIVE_ROOTS: tuple[str, ...] = (
    "src",
    "hardware",
)
_PI_RUNTIME_IMPORT_MODULES: tuple[str, ...] = (
    "dateutil",
    "markupsafe",
    "starlette",
    "pydantic",
    "pydantic_core",
    "urllib3",
    "gpiod",
    "lgpio",
    "pigpio",
    "rapidfuzz",
    "wcwidth",
    "onnx",
    "msgspec",
    "orjson",
    "portalocker",
    "zstandard",
    "h2",
    "opentelemetry.trace",
    "twinr.memory.context_store",
    "twinr.memory.longterm.storage._remote_current_records",
    "twinr.memory.longterm.runtime.health",
)
_PI_RUNTIME_ATTRIBUTE_CONTRACTS: dict[str, tuple[str, ...]] = {
    "twinr.hardware.camera_ai.adapter_impl.observe:AICameraAdapterObserveMixin": (
        "observe_attention_stream",
        "observe_attention_from_frame_stream",
        "observe_gesture_stream",
        "observe_gesture_from_frame_stream",
    ),
    "twinr.hardware.camera_ai.adapter_impl.perception:AICameraAdapterPerceptionMixin": (
        "observe_perception_stream",
    ),
    "twinr.hardware.camera_ai.adapter_impl.core:LocalAICameraAdapter": (
        "observe_perception_stream",
        "observe_attention_stream",
        "observe_attention_from_frame_stream",
        "observe_gesture_stream",
        "observe_gesture_from_frame_stream",
    ),
}
_ENABLED_DEPLOY_UNIT_STATES: frozenset[str] = frozenset(
    {
        "enabled",
        "enabled-runtime",
        "linked",
        "linked-runtime",
        "alias",
        "indirect",
    }
)
_REMOTE_PATH_RE = re.compile(r"^/[A-Za-z0-9._/-]+$")
# BREAKING: remote service identifiers must now be valid, shell-safe systemd unit names.
_SERVICE_NAME_RE = re.compile(r"^[A-Za-z0-9_.@:-]+(?:\.service)?$")
_KNOWN_NON_SERVICE_SUFFIXES: tuple[str, ...] = (
    ".socket",
    ".timer",
    ".path",
    ".target",
    ".mount",
    ".slice",
    ".scope",
)
_DANGEROUS_REMOTE_ROOTS: frozenset[str] = frozenset(
    {
        "/",
        "/bin",
        "/boot",
        "/dev",
        "/etc",
        "/lib",
        "/lib64",
        "/proc",
        "/root",
        "/run",
        "/sbin",
        "/sys",
        "/usr",
        "/var",
    }
)
_VERIFY_FATAL_MARKERS: tuple[str, ...] = (
    "Unknown lvalue",
    "Unknown key name",
    "Unknown section",
    "cannot be started",
    "failed to load",
    "No such file or directory",
    "not executable",
    "Exec format error",
)


class _MirrorWatchdog(Protocol):
    """Describe the minimal mirror-watchdog contract used by deploys."""

    def probe_once(
        self,
        *,
        apply_sync: bool = True,
        checksum: bool = True,
        max_change_lines: int = 40,
    ) -> PiRepoMirrorCycleResult:
        """Run one mirror cycle and optionally heal Pi drift."""


@dataclass(frozen=True, slots=True)
class PiRuntimeDeployResult:
    """Summarize one completed Pi deploy run."""

    ok: bool
    host: str
    remote_root: str
    repo_mirror: PiRepoMirrorCycleResult
    repo_attestation: PiRemoteRepoAttestationResult
    env_sync: PiSyncedFileResult | None
    editable_install_summary: str | None
    bytecode_refresh_summary: str | None
    installed_units: tuple[str, ...]
    restarted_services: tuple[str, ...]
    service_states: tuple[PiSystemdServiceState, ...]
    import_contract: PiPythonImportContractResult | None
    env_contract: dict[str, object] | None
    duration_s: float
    unit_verification_summary: str | None = None


class PiRuntimeDeployError(RuntimeError):
    """Raise when one deploy phase fails with a phase-specific message."""

    def __init__(self, phase: str, message: str) -> None:
        super().__init__(message)
        self.phase = phase


def deploy_pi_runtime(
    *,
    project_root: str | Path,
    pi_env_path: str | Path,
    remote_root: str = "/twinr",
    services: Sequence[str] | None = None,
    rollout_services: Sequence[str] = (),
    env_source_path: str | Path | None = None,
    remote_env_path: str | None = None,
    timeout_s: float = 180.0,
    service_wait_s: float = 30.0,
    sync_env: bool = True,
    install_editable: bool = True,
    install_with_deps: bool = False,
    install_systemd_units: bool = True,
    verify_env_contract: bool = True,
    live_text: str | None = None,
    live_search: str | None = None,
    subprocess_runner: _SubprocessRunner = subprocess.run,
    mirror_watchdog: _MirrorWatchdog | None = None,
) -> PiRuntimeDeployResult:
    """Deploy the current leading-repo state to the Pi and verify it.

    Args:
        project_root: Leading Twinr repo root.
        pi_env_path: Path to the Pi SSH credential dotenv file.
        remote_root: Twinr runtime checkout root on the Pi.
        services: Productive systemd units to install, restart, and verify.
            When omitted, the deploy always manages the base productive units
            and also picks up any repo-backed Pi runtime units that are already
            enabled on the acceptance host.
        rollout_services: Additional repo-backed Pi runtime units that should
            join the deploy target set even when they are not enabled on the
            host yet. Use this for first rollout of optional services such as
            ``twinr-whatsapp-channel.service`` without replacing the default
            base-runtime target set.
        env_source_path: Optional authoritative local env file to copy to the Pi.
        remote_env_path: Optional target env path on the Pi.
        timeout_s: Per-subprocess timeout in seconds.
        service_wait_s: Maximum wait time for restarted services to become healthy.
        sync_env: Whether to overwrite the Pi runtime env file from the leading repo.
        install_editable: Whether to refresh the editable Twinr install in the Pi venv.
        install_with_deps: Whether the editable refresh should also resolve and
            reinstall runtime dependencies. The default keeps dependency state
            unchanged and updates only the editable package, which avoids
            rebuilding Pi-host packages such as PyQt5 on every deploy.
            When the local ignored ``browser_automation/`` workspace exposes
            runtime manifests, the deploy also syncs and installs those
            requirements after the editable refresh so the Pi can execute the
            local adapter code without relying on the repo mirror to carry
            ignored workspace files.
        install_systemd_units: Whether to copy the service unit files from
            ``/twinr/hardware/ops`` into ``/etc/systemd/system`` and reload systemd.
        verify_env_contract: Whether to run the bounded Pi env-contract probe.
        live_text: Optional real non-search provider probe for env-contract verification.
        live_search: Optional real search-backed provider probe for env-contract verification.
        subprocess_runner: Injectable subprocess runner for tests.
        mirror_watchdog: Optional prebuilt mirror watchdog for tests.

        The deploy also runs a fixed remote import contract against the
        preserved Pi venv so critical direct-import modules must remain
        importable under ``/twinr/.venv/bin/python`` after every rollout, and
        it independently attests the mirrored repo contents on the resolved
        ``remote_root`` so a stale checkout cannot still produce a green deploy.

    Returns:
        A structured deploy result for operator logging and script output.

    Raises:
        ValueError: If inputs are invalid.
        PiRuntimeDeployError: If any deploy phase fails.
    """

    if live_text is not None and live_search is not None:
        raise ValueError("live_text and live_search are mutually exclusive")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be greater than zero")
    if service_wait_s <= 0:
        raise ValueError("service_wait_s must be greater than zero")

    started = time.monotonic()
    resolved_root = Path(project_root).resolve()
    if not resolved_root.exists() or not resolved_root.is_dir():
        raise ValueError(f"project root does not exist: {resolved_root}")

    expected_repo_entries = build_authoritative_repo_entry_digests(resolved_root)
    settings = load_pi_connection_settings(pi_env_path)
    # BREAKING: remote_root and remote_env_path are now validated as absolute, shell-safe POSIX paths.
    resolved_remote_root = _normalize_remote_path(remote_root, field_name="remote_root", reject_dangerous_root=True)
    env_source = (resolved_root / ".env") if env_source_path is None else Path(env_source_path).resolve()
    env_target = _normalize_remote_path(
        remote_env_path or f"{resolved_remote_root}/.env",
        field_name="remote_env_path",
        reject_dangerous_root=False,
    )
    remote_python = f"{resolved_remote_root}/.venv/bin/python"
    install_browser_requirements = _has_nonempty_local_file(
        resolved_root / _BROWSER_AUTOMATION_RUNTIME_REQUIREMENTS
    )
    install_playwright_browsers = _has_nonempty_local_file(
        resolved_root / _BROWSER_AUTOMATION_PLAYWRIGHT_BROWSERS
    )
    install_pi_runtime_requirements = _has_nonempty_local_file(
        resolved_root / _PI_RUNTIME_REQUIREMENTS
    )

    remote = _PiRemoteExecutor(
        settings=settings,
        subprocess_runner=subprocess_runner,
        timeout_s=timeout_s,
    )

    with tempfile.TemporaryDirectory(prefix="twinr-pi-runtime-deploy-") as snapshot_dir_str:
        snapshot_dir = Path(snapshot_dir_str)
        env_source_for_sync = env_source
        if sync_env:
            if not env_source.exists() or not env_source.is_file():
                raise ValueError(f"authoritative env file does not exist: {env_source}")
            env_source_for_sync = _snapshot_local_file(
                source_path=env_source,
                snapshot_dir=snapshot_dir,
                snapshot_name="authoritative.env",
            )

        optional_manifest_snapshots: dict[Path, Path] = {}
        if install_browser_requirements:
            optional_manifest_snapshots[_BROWSER_AUTOMATION_RUNTIME_REQUIREMENTS] = _snapshot_local_file(
                source_path=(resolved_root / _BROWSER_AUTOMATION_RUNTIME_REQUIREMENTS).resolve(),
                snapshot_dir=snapshot_dir,
                snapshot_name="browser_automation-runtime_requirements.txt",
            )
        if install_playwright_browsers:
            optional_manifest_snapshots[_BROWSER_AUTOMATION_PLAYWRIGHT_BROWSERS] = _snapshot_local_file(
                source_path=(resolved_root / _BROWSER_AUTOMATION_PLAYWRIGHT_BROWSERS).resolve(),
                snapshot_dir=snapshot_dir,
                snapshot_name="browser_automation-playwright_browsers.txt",
            )

        deploy_lock_ttl_s = _compute_remote_lock_ttl(timeout_s=timeout_s, service_wait_s=service_wait_s)
        with _remote_deploy_lock(
            remote=remote,
            remote_root=resolved_remote_root,
            owner=f"{settings.user}@{settings.host}",
            ttl_s=deploy_lock_ttl_s,
        ):
            normalized_services = _resolve_deploy_services(
                project_root=resolved_root,
                remote=remote,
                remote_root=resolved_remote_root,
                requested_services=services,
                rollout_services=rollout_services,
            )
            _run_phase(
                "service_root_contract",
                lambda: _assert_selected_repo_units_are_remote_root_compatible(
                    project_root=resolved_root,
                    services=normalized_services,
                    remote_root=resolved_remote_root,
                ),
            )
            watchdog = mirror_watchdog or PiRepoMirrorWatchdog.from_env(
                project_root=resolved_root,
                pi_env_path=pi_env_path,
                remote_root=resolved_remote_root,
                timeout_s=timeout_s,
                subprocess_runner=subprocess_runner,
            )

            repo_mirror = _run_phase(
                "repo_mirror",
                lambda: watchdog.probe_once(apply_sync=True, checksum=True, max_change_lines=40),
            )

            _run_phase(
                "repo_stability",
                lambda: _assert_authoritative_repo_unchanged(
                    project_root=resolved_root,
                    expected_entries=expected_repo_entries,
                ),
            )

            env_sync_result: PiSyncedFileResult | None = None
            if sync_env:
                env_sync_result = _run_phase(
                    "env_sync",
                    lambda: _sync_authoritative_file(
                        remote=remote,
                        local_path=env_source_for_sync,
                        remote_path=env_target,
                        mode="600",
                    ),
                )

            if install_browser_requirements:
                _run_phase(
                    "browser_automation_requirements_sync",
                    lambda: _sync_optional_manifest(
                        remote=remote,
                        remote_root=resolved_remote_root,
                        local_path=optional_manifest_snapshots[_BROWSER_AUTOMATION_RUNTIME_REQUIREMENTS],
                        manifest_relpath=_BROWSER_AUTOMATION_RUNTIME_REQUIREMENTS,
                    ),
                )
            if install_playwright_browsers:
                _run_phase(
                    "browser_automation_browsers_sync",
                    lambda: _sync_optional_manifest(
                        remote=remote,
                        remote_root=resolved_remote_root,
                        local_path=optional_manifest_snapshots[_BROWSER_AUTOMATION_PLAYWRIGHT_BROWSERS],
                        manifest_relpath=_BROWSER_AUTOMATION_PLAYWRIGHT_BROWSERS,
                    ),
                )

            repo_attestation_result = _run_phase(
                "repo_attestation",
                lambda: _attest_remote_repo_entries(
                    remote=remote,
                    remote_root=resolved_remote_root,
                    entries=expected_repo_entries,
                ),
            )

            editable_install_summary: str | None = None
            if install_editable:
                editable_install_summary = _run_phase(
                    "editable_install",
                    lambda: _install_editable_package(
                        remote=remote,
                        remote_root=resolved_remote_root,
                        install_with_deps=install_with_deps,
                    ),
                )
                if install_pi_runtime_requirements:
                    pi_runtime_requirements_summary = _run_phase(
                        "pi_runtime_requirements",
                        lambda: _install_python_requirements_manifest(
                            remote=remote,
                            remote_root=resolved_remote_root,
                            manifest_relpath=_PI_RUNTIME_REQUIREMENTS.as_posix(),
                            label="pi_runtime",
                        ),
                    )
                    editable_install_summary = "\n".join(
                        part
                        for part in (editable_install_summary, pi_runtime_requirements_summary)
                        if part and part.strip()
                    )
                if install_browser_requirements or install_playwright_browsers:
                    _run_phase(
                        "browser_automation_runtime",
                        lambda: _install_browser_automation_runtime_support(
                            remote=remote,
                            remote_root=resolved_remote_root,
                            install_python_requirements=install_browser_requirements,
                            install_playwright_browsers=install_playwright_browsers,
                        ),
                    )

            bytecode_refresh_summary = _run_phase(
                "python_bytecode_refresh",
                lambda: _refresh_python_bytecode(
                    remote=remote,
                    remote_python=remote_python,
                    roots=tuple(
                        f"{resolved_remote_root}/{relative_root.strip('/')}"
                        for relative_root in _PI_RUNTIME_BYTECODE_RELATIVE_ROOTS
                    ),
                ),
            )

            import_contract_result = _run_phase(
                "python_import_contract",
                lambda: _verify_python_import_contract(
                    remote=remote,
                    remote_python=remote_python,
                    modules=_PI_RUNTIME_IMPORT_MODULES,
                    attribute_contracts=_PI_RUNTIME_ATTRIBUTE_CONTRACTS,
                ),
            )

            _run_phase(
                "state_permissions",
                lambda: _repair_runtime_state_permissions(
                    remote=remote,
                    remote_root=resolved_remote_root,
                    owner_user=settings.user,
                ),
            )

            unit_verification_summary = _run_phase(
                "systemd_unit_verify",
                lambda: _verify_repo_service_units(
                    remote=remote,
                    remote_root=resolved_remote_root,
                    services=normalized_services,
                ),
            )

            if install_systemd_units:
                _run_phase(
                    "systemd_install",
                    lambda: _install_service_units(
                        remote=remote,
                        remote_root=resolved_remote_root,
                        services=normalized_services,
                    ),
                )

            _run_phase(
                "systemd_restart",
                lambda: _restart_services(
                    remote=remote,
                    services=normalized_services,
                ),
            )

            service_states = _run_phase(
                "service_verification",
                lambda: _wait_for_services(
                    remote=remote,
                    services=normalized_services,
                    wait_timeout_s=service_wait_s,
                ),
            )

            _run_phase(
                "state_permissions_postcheck",
                lambda: _verify_runtime_state_permissions(
                    remote=remote,
                    remote_root=resolved_remote_root,
                    owner_user=settings.user,
                ),
            )

            env_contract_result: dict[str, object] | None = None
            if verify_env_contract:
                env_contract_result = _run_phase(
                    "env_contract",
                    lambda: _run_env_contract_probe(
                        remote=remote,
                        remote_root=resolved_remote_root,
                        env_path=env_target,
                        live_text=live_text,
                        live_search=live_search,
                    ),
                )

    return PiRuntimeDeployResult(
        ok=True,
        host=settings.host,
        remote_root=resolved_remote_root,
        repo_mirror=repo_mirror,
        repo_attestation=repo_attestation_result,
        env_sync=env_sync_result,
        editable_install_summary=editable_install_summary,
        bytecode_refresh_summary=bytecode_refresh_summary,
        installed_units=normalized_services if install_systemd_units else (),
        restarted_services=normalized_services,
        service_states=service_states,
        import_contract=import_contract_result,
        env_contract=env_contract_result,
        duration_s=round(time.monotonic() - started, 3),
        unit_verification_summary=unit_verification_summary,
    )


def _run_phase(phase: str, fn: Any) -> Any:
    """Wrap one deploy phase and normalize errors to a phase-specific type."""

    try:
        return fn()
    except PiRuntimeDeployError:
        raise
    except Exception as exc:  # pragma: no cover - thin error normalization.
        raise PiRuntimeDeployError(phase, str(exc)) from exc


def _has_nonempty_local_file(path: Path) -> bool:
    """Return whether one local deploy manifest exists and contains data."""

    return path.is_file() and bool(path.read_bytes().strip())


def _sync_optional_manifest(
    *,
    remote: _PiRemoteExecutor,
    remote_root: str,
    local_path: Path,
    manifest_relpath: Path,
) -> PiSyncedFileResult:
    """Sync one optional local manifest into the Pi runtime checkout."""

    if not _has_nonempty_local_file(local_path):
        raise ValueError(f"optional manifest is missing or empty: {local_path}")
    remote_manifest_path = f"{remote_root.rstrip('/')}/{manifest_relpath.as_posix()}"
    return _sync_authoritative_file(
        remote=remote,
        local_path=local_path,
        remote_path=remote_manifest_path,
        mode="644",
    )


def _normalize_services(services: Sequence[str]) -> tuple[str, ...]:
    """Return a stable de-duplicated tuple of systemd service names."""

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_name in services:
        name = _normalize_service_name(raw_name)
        if name in seen:
            continue
        seen.add(name)
        normalized.append(name)
    if not normalized:
        raise ValueError("at least one service is required")
    return tuple(normalized)


def _resolve_deploy_services(
    *,
    project_root: Path,
    remote: _PiRemoteExecutor,
    remote_root: str,
    requested_services: Sequence[str] | None,
    rollout_services: Sequence[str] = (),
) -> tuple[str, ...]:
    """Resolve the service set that this deploy run must manage."""

    repo_services = _discover_repo_pi_service_candidates(project_root, remote_root=remote_root)
    normalized_rollout_services = _normalize_rollout_services(
        rollout_services=rollout_services,
        repo_services=repo_services,
    )
    if requested_services is not None:
        return _normalize_services((*requested_services, *normalized_rollout_services))
    enabled_repo_services = _load_enabled_repo_services(
        remote=remote,
        services=repo_services,
    )
    return _normalize_services((*DEFAULT_DEPLOY_SERVICES, *enabled_repo_services, *normalized_rollout_services))


def _normalize_rollout_services(
    *,
    rollout_services: Sequence[str],
    repo_services: Sequence[str],
) -> tuple[str, ...]:
    """Validate and normalize explicit first-rollout Pi service requests."""

    normalized_rollout_services = _normalize_services(rollout_services) if rollout_services else ()
    if not normalized_rollout_services:
        return ()
    allowed_services = set(_normalize_services(repo_services))
    invalid_services = tuple(service for service in normalized_rollout_services if service not in allowed_services)
    if invalid_services:
        invalid_list = ", ".join(invalid_services)
        raise ValueError(
            "rollout_services must be repo-backed Pi runtime units under hardware/ops: "
            f"{invalid_list}"
        )
    return normalized_rollout_services


def _discover_repo_pi_service_candidates(project_root: Path, *, remote_root: str) -> tuple[str, ...]:
    """Return repo-backed systemd units that target the requested Pi runtime checkout."""

    ops_root = project_root / "hardware" / "ops"
    if not ops_root.is_dir():
        return ()
    discovered: list[str] = []
    for service_path in sorted(ops_root.glob("*.service")):
        try:
            service_text = service_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not _is_repo_pi_runtime_service_unit(service_text, remote_root=remote_root):
            continue
        discovered.append(service_path.name)
    return tuple(discovered)


def _is_repo_pi_runtime_service_unit(service_text: str, *, remote_root: str) -> bool:
    """Return whether one repo unit file targets the Pi runtime checkout."""

    normalized_remote_root = _normalize_remote_path(
        remote_root,
        field_name="remote_root",
        reject_dangerous_root=False,
    )
    for _, key, value in _iter_systemd_assignments(service_text):
        normalized_key = key.strip()
        if normalized_key == "WorkingDirectory" and _matches_runtime_root(value, normalized_remote_root):
            return True
        if normalized_key != "ExecStart":
            continue
        if _execstart_targets_runtime_root(value, normalized_remote_root):
            return True
    return False


def _load_enabled_repo_services(
    *,
    remote: _PiRemoteExecutor,
    services: Sequence[str],
) -> tuple[str, ...]:
    """Return repo-backed Pi runtime units that are currently enabled on the Pi."""

    normalized_services = _normalize_services(services) if services else ()
    if not normalized_services:
        return ()
    payload = {"services": list(normalized_services)}
    completed = remote.run_ssh(
        _remote_python_json_script(
            payload=payload,
            body_lines=(
                "import subprocess",
                "results = []",
                "for name in payload['services']:",
                "    completed = subprocess.run(",
                "        ['systemctl', 'is-enabled', name],",
                "        check=False,",
                "        capture_output=True,",
                "        text=True,",
                "        encoding='utf-8',",
                "        errors='replace',",
                "    )",
                "    combined = '\\n'.join(part for part in (completed.stdout, completed.stderr) if part)",
                "    state = ''",
                "    for raw_line in combined.splitlines():",
                "        candidate = raw_line.strip()",
                "        if candidate:",
                "            state = candidate",
                "            break",
                "    results.append({'name': name, 'state': state, 'returncode': completed.returncode})",
                "print(json.dumps(results, ensure_ascii=False))",
            ),
        )
    )
    payload = json.loads((completed.stdout or "[]").strip() or "[]")
    enabled: list[str] = []
    for item in payload:
        state = str(item.get("state", "") or "").strip().lower()
        if state not in _ENABLED_DEPLOY_UNIT_STATES:
            continue
        enabled.append(str(item.get("name", "") or "").strip())
    return tuple(enabled)


def _verify_repo_service_units(
    *,
    remote: _PiRemoteExecutor,
    remote_root: str,
    services: Sequence[str],
) -> str:
    """Verify repo-backed service unit files on the Pi before restart."""

    normalized_services = _normalize_services(services) if services else ()
    if not normalized_services:
        return "no services selected"
    payload = {
        "remote_root": remote_root,
        "services": list(normalized_services),
        "fatal_markers": list(_VERIFY_FATAL_MARKERS),
    }
    completed = remote.run_ssh(
        _remote_python_json_script(
            payload=payload,
            body_lines=(
                "from pathlib import Path",
                "import subprocess",
                "def run_verify(command):",
                "    completed = subprocess.run(",
                "        command,",
                "        check=False,",
                "        capture_output=True,",
                "        text=True,",
                "        encoding='utf-8',",
                "        errors='replace',",
                "    )",
                "    output = '\\n'.join(part for part in (completed.stdout, completed.stderr) if part).strip()",
                "    return completed.returncode, output",
                "results = []",
                "for name in payload['services']:",
                "    unit_path = Path(payload['remote_root']) / 'hardware' / 'ops' / name",
                "    if not unit_path.is_file():",
                "        results.append({'name': name, 'skipped': True, 'ok': True, 'path': str(unit_path), 'output': ''})",
                "        continue",
                "    command = ['systemd-analyze', 'verify', '--recursive-errors=yes', f'{unit_path}:{name}']",
                "    returncode, output = run_verify(command)",
                "    combined_lower = output.lower()",
                "    unsupported_recursive = '--recursive-errors' in output and ('unknown option' in combined_lower or 'unrecognized option' in combined_lower or 'invalid option' in combined_lower)",
                "    if unsupported_recursive:",
                "        returncode, output = run_verify(['systemd-analyze', 'verify', f'{unit_path}:{name}'])",
                "    fatal = any(marker.lower() in combined_lower for marker in payload['fatal_markers'])",
                "    results.append(",
                "        {",
                "            'name': name,",
                "            'path': str(unit_path),",
                "            'skipped': False,",
                "            'returncode': returncode,",
                "            'ok': (returncode == 0) and not fatal,",
                "            'output': output,",
                "        }",
                "    )",
                "print(json.dumps(results, ensure_ascii=False))",
            ),
        )
    )
    results = json.loads((completed.stdout or "[]").strip() or "[]")
    failures = [item for item in results if not bool(item.get("ok", False))]
    if failures:
        details = []
        for item in failures:
            name = str(item.get("name", "") or "unknown.service")
            output = str(item.get("output", "") or "").strip()
            detail = name if not output else f"{name}: {output}"
            details.append(detail)
        raise PiRuntimeDeployError(
            "systemd_unit_verify",
            "repo-backed service unit verification failed: " + " | ".join(details),
        )
    verified_count = sum(1 for item in results if not bool(item.get("skipped", False)))
    skipped_count = sum(1 for item in results if bool(item.get("skipped", False)))
    return f"verified={verified_count}, skipped_external={skipped_count}"


def _normalize_remote_path(
    raw_path: str | Path,
    *,
    field_name: str,
    reject_dangerous_root: bool,
) -> str:
    """Validate one remote path before it reaches shell-backed remote helpers."""

    value = str(raw_path or "").strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    if any(ch in value for ch in ("\x00", "\n", "\r", "\t")):
        raise ValueError(f"{field_name} contains control characters")
    if not _REMOTE_PATH_RE.fullmatch(value):
        raise ValueError(
            f"{field_name} must be an absolute shell-safe POSIX path containing only /, letters, digits, ., _, and -"
        )
    raw_parts = PurePosixPath(value).parts
    if not raw_parts or raw_parts[0] != "/":
        raise ValueError(f"{field_name} must be an absolute POSIX path: {value}")
    if any(part in {".", ".."} for part in raw_parts):
        raise ValueError(f"{field_name} must not contain '.' or '..' path segments: {value}")
    normalized = posixpath.normpath(value)
    if not normalized.startswith("/"):
        raise ValueError(f"{field_name} must resolve to an absolute POSIX path: {value}")
    if reject_dangerous_root and normalized in _DANGEROUS_REMOTE_ROOTS:
        raise ValueError(f"refusing dangerous {field_name}: {normalized}")
    return normalized


def _normalize_service_name(raw_name: str) -> str:
    """Return one validated systemd service name."""

    name = str(raw_name or "").strip()
    if not name:
        raise ValueError("service names must not be empty")
    if not name.endswith(".service"):
        if any(name.endswith(suffix) for suffix in _KNOWN_NON_SERVICE_SUFFIXES):
            raise ValueError(f"expected a .service unit, got {raw_name!r}")
        name += ".service"
    if not _SERVICE_NAME_RE.fullmatch(name):
        raise ValueError(
            f"invalid systemd service name: {raw_name!r}; expected a shell-safe unit identifier"
        )
    return name


def _assert_selected_repo_units_are_remote_root_compatible(
    *,
    project_root: Path,
    services: Sequence[str],
    remote_root: str,
) -> None:
    """Fail fast when selected repo units still point at the legacy /twinr root."""

    if remote_root == "/twinr":
        return
    ops_root = project_root / "hardware" / "ops"
    mismatched: list[str] = []
    for service_name in _normalize_services(services):
        service_path = ops_root / service_name
        if not service_path.is_file():
            continue
        try:
            service_text = service_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "/twinr" not in service_text:
            continue
        if _is_repo_pi_runtime_service_unit(service_text, remote_root=remote_root):
            continue
        mismatched.append(service_name)
    if mismatched:
        raise PiRuntimeDeployError(
            "service_root_contract",
            "selected repo unit files still reference /twinr and must be updated for the requested remote_root: "
            + ", ".join(mismatched),
        )


def _assert_authoritative_repo_unchanged(*, project_root: Path, expected_entries: Any) -> None:
    """Abort when the authoritative repo changed after the deploy snapshot."""

    current_entries = build_authoritative_repo_entry_digests(project_root)
    if current_entries != expected_entries:
        raise PiRuntimeDeployError(
            "repo_stability",
            "authoritative repo changed during deploy; rerun against a stable working tree",
        )


def _snapshot_local_file(*, source_path: Path, snapshot_dir: Path, snapshot_name: str) -> Path:
    """Copy one mutable local file into an immutable deploy snapshot directory."""

    if not source_path.exists() or not source_path.is_file():
        raise ValueError(f"snapshot source file does not exist: {source_path}")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    target_path = snapshot_dir / snapshot_name
    shutil.copy2(source_path, target_path)
    return target_path


def _compute_remote_lock_ttl(*, timeout_s: float, service_wait_s: float) -> int:
    """Return a bounded TTL for stale remote deploy locks."""

    return max(600, int((timeout_s * 4) + (service_wait_s * 4) + 120))


@contextmanager
def _remote_deploy_lock(
    *,
    remote: _PiRemoteExecutor,
    remote_root: str,
    owner: str,
    ttl_s: int,
):
    """Serialize deploys per remote runtime root using a remote lock file."""

    lock_suffix = re.sub(r"[^A-Za-z0-9_.-]+", "-", remote_root.strip("/") or "root")
    lock_path = f"/tmp/twinr-runtime-deploy-{lock_suffix}.lock"
    acquire_payload = {
        "lock_path": lock_path,
        "owner": owner,
        "remote_root": remote_root,
        "stale_after_s": int(ttl_s),
    }
    acquire = json.loads(
        (
            remote.run_ssh(
                _remote_python_json_script(
                    payload=acquire_payload,
                    body_lines=(
                        "from pathlib import Path",
                        "import os",
                        "import socket",
                        "token = os.urandom(16).hex()",
                        "lock_path = Path(payload['lock_path'])",
                        "import time",
                        "requested_at = time.time()",
                        "def read_existing():",
                        "    try:",
                        "        return json.loads(lock_path.read_text(encoding='utf-8'))",
                        "    except FileNotFoundError:",
                        "        return None",
                        "    except Exception:",
                        "        return {'raw': lock_path.read_text(encoding='utf-8', errors='replace')}",
                        "def is_stale(existing):",
                        "    try:",
                        "        started = float(existing.get('requested_at', 0.0))",
                        "    except Exception:",
                        "        return True",
                        "    return (requested_at - started) > float(payload['stale_after_s'])",
                        "metadata = {",
                        "    'token': token,",
                        "    'owner': payload['owner'],",
                        "    'remote_root': payload['remote_root'],",
                        "    'requested_at': requested_at,",
                        "    'hostname': socket.gethostname(),",
                        "}",
                        "result = None",
                        "for _ in range(2):",
                        "    try:",
                        "        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)",
                        "    except FileExistsError:",
                        "        existing = read_existing() or {}",
                        "        if existing and is_stale(existing):",
                        "            try:",
                        "                lock_path.unlink()",
                        "            except FileNotFoundError:",
                        "                continue",
                        "            except OSError as exc:",
                        "                result = {'acquired': False, 'reason': 'stale-lock-unremovable', 'lock_path': str(lock_path), 'existing': existing, 'error': str(exc)}",
                        "                break",
                        "            continue",
                        "        result = {'acquired': False, 'reason': 'locked', 'lock_path': str(lock_path), 'existing': existing}",
                        "        break",
                        "    else:",
                        "        with os.fdopen(fd, 'w', encoding='utf-8') as handle:",
                        "            json.dump(metadata, handle, ensure_ascii=False)",
                        "        result = {'acquired': True, 'lock_path': str(lock_path), 'token': token}",
                        "        break",
                        "if result is None:",
                        "    result = {'acquired': False, 'reason': 'unknown', 'lock_path': str(lock_path)}",
                        "print(json.dumps(result, ensure_ascii=False))",
                    ),
                )
            ).stdout
            or "{}"
        ).strip()
        or "{}"
    )
    if not bool(acquire.get("acquired", False)):
        existing = acquire.get("existing") or {}
        owner_text = str(existing.get("owner", "") or "unknown-owner")
        age_suffix = ""
        raise PiRuntimeDeployError(
            "deploy_lock",
            f"another deploy already holds {lock_path} (owner={owner_text}{age_suffix})",
        )

    try:
        yield
    finally:
        release_payload = {
            "lock_path": lock_path,
            "token": str(acquire.get("token", "") or ""),
        }
        try:
            remote.run_ssh(
                _remote_python_json_script(
                    payload=release_payload,
                    body_lines=(
                        "from pathlib import Path",
                        "lock_path = Path(payload['lock_path'])",
                        "released = {'released': False, 'lock_path': str(lock_path)}",
                        "try:",
                        "    existing = json.loads(lock_path.read_text(encoding='utf-8'))",
                        "except FileNotFoundError:",
                        "    released = {'released': True, 'lock_path': str(lock_path), 'missing': True}",
                        "except Exception:",
                        "    existing = None",
                        "if existing is not None:",
                        "    if str(existing.get('token', '')) == str(payload['token']):",
                        "        lock_path.unlink(missing_ok=True)",
                        "        released = {'released': True, 'lock_path': str(lock_path)}",
                        "    else:",
                        "        released = {'released': False, 'lock_path': str(lock_path), 'reason': 'token-mismatch'}",
                        "print(json.dumps(released, ensure_ascii=False))",
                    ),
                )
            )
        except Exception:
            pass


def _remote_python_json_script(*, payload: dict[str, Any], body_lines: Sequence[str]) -> str:
    """Build one remote python heredoc with a URL-safe base64 JSON payload."""

    encoded_payload = base64.urlsafe_b64encode(
        json.dumps(payload, ensure_ascii=False).encode("utf-8")
    ).decode("ascii")
    lines = [
        "python3 - <<'PY'",
        "import base64",
        "import json",
        f"payload = json.loads(base64.urlsafe_b64decode({encoded_payload!r}).decode('utf-8'))",
        *body_lines,
        "PY",
    ]
    return "\n".join(lines)


def _iter_systemd_assignments(service_text: str) -> tuple[tuple[str | None, str, str], ...]:
    """Yield logical ``(section, key, value)`` assignments from one systemd unit file."""

    section: str | None = None
    logical_lines: list[str] = []
    buffer = ""
    for raw_line in str(service_text or "").splitlines():
        line = raw_line.rstrip("\n")
        stripped = line.lstrip()
        if buffer:
            if not stripped or stripped.startswith(("#", ";")):
                continue
            continuation = line.rstrip()
            if continuation.endswith("\\"):
                buffer += continuation[:-1] + " "
                continue
            logical_lines.append(buffer + continuation)
            buffer = ""
            continue
        if not stripped or stripped.startswith(("#", ";")):
            continue
        if line.rstrip().endswith("\\"):
            buffer = line.rstrip()[:-1] + " "
            continue
        logical_lines.append(line)
    if buffer:
        logical_lines.append(buffer)

    assignments: list[tuple[str | None, str, str]] = []
    for raw_line in logical_lines:
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped
            continue
        if "=" not in raw_line:
            continue
        raw_key, raw_value = raw_line.split("=", 1)
        key = raw_key.strip()
        if not key:
            continue
        value = raw_value.strip()
        assignments.append((section, key, value))
    return tuple(assignments)


def _matches_runtime_root(raw_value: str, remote_root: str) -> bool:
    """Return whether one unit value resolves to the runtime root."""

    value = _strip_matching_quotes(str(raw_value or "").strip())
    if not value or not value.startswith("/"):
        return False
    return posixpath.normpath(value) == remote_root


def _execstart_targets_runtime_root(raw_value: str, remote_root: str) -> bool:
    """Return whether one ExecStart line targets or enters the runtime root."""

    candidate = str(raw_value or "").strip()
    if not candidate:
        return False
    while candidate.startswith(("@", "-", ":", "+", "!")):
        candidate = candidate[2:] if candidate.startswith("!!") else candidate[1:]
        candidate = candidate.lstrip()
    first_token = _first_command_token(candidate)
    if first_token:
        first_token = _strip_matching_quotes(first_token)
        if first_token.startswith("/"):
            normalized = posixpath.normpath(first_token)
            if normalized == remote_root or normalized.startswith(remote_root.rstrip("/") + "/"):
                return True
    cd_pattern = re.compile(
        rf"\bcd\s+(?P<quote>['\"]?){re.escape(remote_root)}(?P=quote)(?:\s|;|&&|\|\||$)"
    )
    return bool(cd_pattern.search(candidate))


def _first_command_token(command: str) -> str | None:
    """Return the first token from one ExecStart-style command string."""

    if not command:
        return None
    quoted_match = re.match(r"^(?P<quote>['\"])(?P<token>.*?)(?P=quote)(?:\s|;|$)", command)
    if quoted_match:
        return quoted_match.group("token")
    unquoted_match = re.match(r"^(?P<token>[^\s;]+)", command)
    if unquoted_match:
        return unquoted_match.group("token")
    return None


def _strip_matching_quotes(value: str) -> str:
    """Remove one matching pair of surrounding quotes from a string."""

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


__all__ = [
    "DEFAULT_DEPLOY_SERVICES",
    "PiRuntimeDeployError",
    "PiPythonImportContractResult",
    "PiRuntimeDeployResult",
    "PiSyncedFileResult",
    "PiSystemdServiceState",
    "deploy_pi_runtime",
]