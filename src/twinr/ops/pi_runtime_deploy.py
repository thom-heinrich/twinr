"""Deploy the leading Twinr repo and runtime config to the Pi acceptance host.

This module owns the operator workflow that turns the current authoritative
repo state in ``/home/thh/twinr`` into a restarted and verified Pi acceptance
runtime under ``/twinr``. It reuses the one-way repo mirror for code sync,
optionally overwrites the Pi runtime ``.env`` from the leading repo, refreshes
the editable install in the Pi virtualenv, installs optional mirrored local
workspace runtime manifests such as ``browser_automation/runtime_requirements``
before restart, installs the productive base units plus any repo-backed Pi
runtime units that are already enabled on the host, supports explicit
first-rollout activation for optional Pi units, restarts the selected
services, and checks that they came back healthy.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Protocol, Sequence

from twinr.ops.pi_repo_mirror import PiRepoMirrorCycleResult, PiRepoMirrorWatchdog
from twinr.ops.pi_runtime_deploy_remote import (
    PiRemoteExecutor as _PiRemoteExecutor,
    PiSyncedFileResult,
    PiSystemdServiceState,
    install_browser_automation_runtime_support as _install_browser_automation_runtime_support,
    install_editable_package as _install_editable_package,
    install_service_units as _install_service_units,
    restart_services as _restart_services,
    run_env_contract_probe as _run_env_contract_probe,
    sync_authoritative_file as _sync_authoritative_file,
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
    env_sync: PiSyncedFileResult | None
    editable_install_summary: str | None
    installed_units: tuple[str, ...]
    restarted_services: tuple[str, ...]
    service_states: tuple[PiSystemdServiceState, ...]
    env_contract: dict[str, object] | None
    duration_s: float


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
            runtime manifests, the deploy also installs those mirrored
            requirements after the editable refresh so the Pi can execute the
            mirrored adapter code.
        install_systemd_units: Whether to copy the service unit files from
            ``/twinr/hardware/ops`` into ``/etc/systemd/system`` and reload systemd.
        verify_env_contract: Whether to run the bounded Pi env-contract probe.
        live_text: Optional real non-search provider probe for env-contract verification.
        live_search: Optional real search-backed provider probe for env-contract verification.
        subprocess_runner: Injectable subprocess runner for tests.
        mirror_watchdog: Optional prebuilt mirror watchdog for tests.

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

    settings = load_pi_connection_settings(pi_env_path)
    env_source = (resolved_root / ".env") if env_source_path is None else Path(env_source_path).resolve()
    resolved_remote_root = remote_root.rstrip("/") or "/"
    env_target = remote_env_path or f"{resolved_remote_root}/.env"
    install_browser_requirements = _has_nonempty_local_file(
        resolved_root / _BROWSER_AUTOMATION_RUNTIME_REQUIREMENTS
    )
    install_playwright_browsers = _has_nonempty_local_file(
        resolved_root / _BROWSER_AUTOMATION_PLAYWRIGHT_BROWSERS
    )

    remote = _PiRemoteExecutor(
        settings=settings,
        subprocess_runner=subprocess_runner,
        timeout_s=timeout_s,
    )
    normalized_services = _resolve_deploy_services(
        project_root=resolved_root,
        remote=remote,
        requested_services=services,
        rollout_services=rollout_services,
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

    env_sync_result: PiSyncedFileResult | None = None
    if sync_env:
        if not env_source.exists() or not env_source.is_file():
            raise ValueError(f"authoritative env file does not exist: {env_source}")
        env_sync_result = _run_phase(
            "env_sync",
            lambda: _sync_authoritative_file(
                remote=remote,
                local_path=env_source,
                remote_path=env_target,
                mode="600",
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
        env_sync=env_sync_result,
        editable_install_summary=editable_install_summary,
        installed_units=normalized_services if install_systemd_units else (),
        restarted_services=normalized_services,
        service_states=service_states,
        env_contract=env_contract_result,
        duration_s=round(time.monotonic() - started, 3),
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


def _normalize_services(services: Sequence[str]) -> tuple[str, ...]:
    """Return a stable de-duplicated tuple of systemd service names."""

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_name in services:
        name = str(raw_name).strip()
        if not name:
            continue
        if not name.endswith(".service"):
            name += ".service"
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
    requested_services: Sequence[str] | None,
    rollout_services: Sequence[str] = (),
) -> tuple[str, ...]:
    """Resolve the service set that this deploy run must manage."""

    repo_services = _discover_repo_pi_service_candidates(project_root)
    normalized_rollout_services = _normalize_rollout_services(
        rollout_services=rollout_services,
        repo_services=repo_services,
    )
    if requested_services is not None:
        return _normalize_services((*requested_services, *normalized_rollout_services))
    enabled_repo_services = _load_enabled_repo_services(
        project_root=project_root,
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


def _discover_repo_pi_service_candidates(project_root: Path) -> tuple[str, ...]:
    """Return repo-backed systemd units that target the Pi runtime checkout."""

    ops_root = project_root / "hardware" / "ops"
    if not ops_root.is_dir():
        return ()
    discovered: list[str] = []
    for service_path in sorted(ops_root.glob("*.service")):
        try:
            service_text = service_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not _is_repo_pi_runtime_service_unit(service_text):
            continue
        discovered.append(service_path.name)
    return tuple(discovered)


def _is_repo_pi_runtime_service_unit(service_text: str) -> bool:
    """Return whether one repo unit file targets the Pi runtime checkout."""

    for raw_line in str(service_text or "").splitlines():
        line = raw_line.strip()
        if line == "WorkingDirectory=/twinr":
            return True
        if not line.startswith("ExecStart="):
            continue
        if line.startswith("ExecStart=/twinr/"):
            return True
        if "cd /twinr" in line:
            return True
    return False


def _load_enabled_repo_services(
    *,
    project_root: Path,
    remote: _PiRemoteExecutor,
    services: Sequence[str],
) -> tuple[str, ...]:
    """Return repo-backed Pi runtime units that are currently enabled on the Pi."""

    normalized_services = _normalize_services(services) if services else ()
    if not normalized_services:
        return ()
    install_targets_by_service = _load_repo_service_install_targets(
        project_root=project_root,
        services=normalized_services,
    )
    script = "\n".join(
        (
            "python3 - <<'PY'",
            "import json",
            "from pathlib import Path",
            "import subprocess",
            f"services = {json.dumps(list(normalized_services), ensure_ascii=False)}",
            f"install_targets_by_service = {json.dumps(install_targets_by_service, ensure_ascii=False)}",
            "payload = []",
            "for name in services:",
            "    completed = subprocess.run(",
            "        ['systemctl', 'show', name, '--property=UnitFileState'],",
            "        check=False,",
            "        capture_output=True,",
            "        text=True,",
            "        encoding='utf-8',",
            "        errors='replace',",
            "    )",
            "    state = ''",
            "    for raw_line in completed.stdout.splitlines():",
            "        if raw_line.startswith('UnitFileState='):",
            "            state = raw_line.split('=', 1)[1].strip()",
            "            break",
            "    install_link_present = False",
            "    for target in install_targets_by_service.get(name, []):",
            "        wants_path = Path('/etc/systemd/system') / f'{target}.wants' / name",
            "        if wants_path.is_symlink() or wants_path.exists():",
            "            install_link_present = True",
            "            break",
            "    payload.append(",
            "        {",
            "            'name': name,",
            "            'unit_file_state': state,",
            "            'install_link_present': install_link_present,",
            "        }",
            "    )",
            "print(json.dumps(payload, ensure_ascii=False))",
            "PY",
        )
    )
    completed = remote.run_ssh(script)
    payload = json.loads((completed.stdout or "[]").strip() or "[]")
    enabled: list[str] = []
    for item in payload:
        state = str(item.get("unit_file_state", "") or "").strip().lower()
        install_link_present = bool(item.get("install_link_present"))
        if not _is_enabled_unit_file_state(state) and not install_link_present:
            continue
        enabled.append(str(item.get("name", "") or "").strip())
    return tuple(enabled)


def _load_repo_service_install_targets(
    *,
    project_root: Path,
    services: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    """Return the repo-declared WantedBy install targets for each Pi service."""

    ops_root = project_root / "hardware" / "ops"
    install_targets: dict[str, tuple[str, ...]] = {}
    for service_name in services:
        service_path = ops_root / service_name
        targets = _parse_service_install_targets(service_path)
        if targets:
            install_targets[service_name] = targets
    return install_targets


def _parse_service_install_targets(service_path: Path) -> tuple[str, ...]:
    """Return the ordered WantedBy targets declared in one systemd unit file."""

    try:
        service_text = service_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ()

    in_install_section = False
    targets: list[str] = []
    seen: set[str] = set()
    for raw_line in service_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_install_section = line == "[Install]"
            continue
        if not in_install_section:
            continue
        if not line.startswith("WantedBy="):
            continue
        _, raw_targets = line.split("=", 1)
        for raw_target in raw_targets.split():
            target = raw_target.strip()
            if not target or target in seen:
                continue
            seen.add(target)
            targets.append(target)
    return tuple(targets)


def _is_enabled_unit_file_state(state: str) -> bool:
    """Return whether one systemd unit-file state should join default deploys."""

    normalized = str(state or "").strip().lower()
    if not normalized:
        return False
    return normalized.startswith("enabled") or normalized.startswith("linked")


__all__ = [
    "DEFAULT_DEPLOY_SERVICES",
    "PiRuntimeDeployError",
    "PiRuntimeDeployResult",
    "PiSyncedFileResult",
    "PiSystemdServiceState",
    "deploy_pi_runtime",
]
