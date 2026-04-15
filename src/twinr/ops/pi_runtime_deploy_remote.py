# CHANGELOG: 2026-03-30
# BUG-1: Fixed silent dependency drift in the fast `--no-deps` deploy path. The old top-level
# BUG-1: version check missed extras/transitives and could leave the Pi venv broken while reporting success.
# BUG-2: Fixed service-health false negatives by using richer systemd state (`LoadState`, `Type`, `Result`)
# BUG-2: instead of requiring only `active/running`.
# SEC-1: SSH host-key verification is no longer disabled by default. The default policy is now
# SEC-1: `accept-new` with a dedicated known_hosts file, with explicit opt-in required for insecure mode.
# SEC-2: Runtime state files stay owner-only, while the shared ops event stream now has an explicit
# SEC-2: cross-user writer contract so Pi services and operator probes append to the same diagnostics file.
# IMP-1: Remote file sync now stages in the target directory and atomically renames into place after checksum verification.
# IMP-2: Deploy-time package validation now uses pip's modern dry-run/report flow when available and always runs `pip check`.
# IMP-3: systemd units are verified with `systemd-analyze verify` before enable/restart.
# IMP-4: Playwright browser installs now support OS dependency installation and a shared browser cache path.

"""Remote execution and verification helpers for Pi runtime deploys.

This module owns the SSH/SCP-side primitives used by the operator-facing Pi
deploy flow. Besides core repo/env/systemd steps, it also installs optional
browser-automation runtime support from manifests that are already present
inside the mirrored authoritative release snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence

from twinr.ops.deploy_progress import ProgressCallback, progress_span
from twinr.ops.pi_repo_mirror import PiRepoMirrorEntryDigest
from twinr.ops.self_coding_pi import PiConnectionSettings
from twinr.ops.venv_bridged_system_cleanup import BridgedSystemShadowedDistribution
from twinr.ops.venv_system_site_bridge import PiVenvSystemSiteBridgeResult
from twinr.ops.venv_wrapper_repair import PiVenvScriptRepairResult, repair_venv_python_shebangs


_SubprocessRunner = Any
_DEFAULT_SSH_CONNECT_TIMEOUT_S = 10
_DEFAULT_SSH_SERVER_ALIVE_INTERVAL_S = 15
_DEFAULT_SSH_SERVER_ALIVE_COUNT_MAX = 3
# BREAKING: the secure default is now `accept-new`; use the `insecure` policy only for
# explicitly trusted throwaway environments.
_DEFAULT_SSH_HOST_KEY_POLICY = "accept-new"
_DEFAULT_KNOWN_HOSTS_PATH = Path.home() / ".cache" / "twinr" / "known_hosts"
# BREAKING: runtime state is now owner-only instead of world-writable/readable.
_STATE_PERMISSION_SPECS: tuple[tuple[str, str], ...] = (
    ("automations.json", "600"),
    ("automations.json.bak", "600"),
    ("automations.json.lock", "600"),
    ("user_discovery.json", "600"),
)
_STATE_DIR_MODE = "700"
_OPS_DIR_MODE = "700"
_OPS_ARTIFACT_PERMISSION_SPECS: tuple[tuple[str, str], ...] = (
    ("events.jsonl", "666"),
    (".events.jsonl.lock", "666"),
    ("usage.jsonl.sqlite3", "600"),
    ("usage.jsonl.sqlite3.lock", "600"),
    ("remote_memory_watchdog.json", "644"),
    ("current_release_manifest.json", "644"),
    ("display_ambient_impulse.json", "644"),
    ("display_heartbeat.json", "644"),
    ("display_render_state.json", "644"),
    ("streaming_memory_segments.json", "644"),
)
_SERVICE_HEALTHY_SUBSTATES = frozenset({"running", "listening", "exited"})


@dataclass(frozen=True, slots=True)
class PiSyncedFileResult:
    """Describe one authoritative file sync from the leading repo to the Pi."""

    local_path: str
    remote_path: str
    sha256: str
    changed: bool
    backup_path: str | None


@dataclass(frozen=True, slots=True)
class PiSystemdServiceState:
    """Summarize one productive service state on the Pi."""

    name: str
    active_state: str
    sub_state: str
    unit_file_state: str
    main_pid: int | None
    exec_main_status: int | None
    healthy: bool
    load_state: str = ""
    service_type: str = ""
    service_result: str = ""


@dataclass(frozen=True, slots=True)
class PiPythonImportContractResult:
    """Summarize one remote Python import attestation."""

    python_path: str
    checked_modules: tuple[str, ...]
    imported_modules: tuple[str, ...]
    checked_attribute_contracts: tuple[str, ...]
    validated_attribute_contracts: tuple[str, ...]
    elapsed_s: float


@dataclass(frozen=True, slots=True)
class PiRemoteRepoAttestationResult:
    """Summarize one remote authoritative-repo content attestation."""

    verified_entry_count: int
    verified_file_count: int
    verified_symlink_count: int
    missing_count: int
    mismatch_count: int
    sampled_missing_paths: tuple[str, ...]
    sampled_mismatch_details: tuple[str, ...]
    elapsed_s: float


class PiRemoteExecutor:
    """Run bounded SSH/SCP commands against the Pi acceptance host."""

    def __init__(
        self,
        *,
        settings: PiConnectionSettings,
        subprocess_runner: _SubprocessRunner,
        timeout_s: float,
    ) -> None:
        self.settings = settings
        self._subprocess_runner = subprocess_runner
        self.timeout_s = timeout_s

    @property
    def remote_spec(self) -> str:
        """Return the canonical ``user@host`` SSH target string."""

        return f"{self.settings.user}@{self.settings.host}"

    def run_ssh(
        self,
        script: str,
        *,
        timeout_s: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run one remote bash script over SSH and capture UTF-8 text output."""

        return self._run_local(
            [
                *self._sshpass_prefix(use_stdin=True),
                "ssh",
                *self._ssh_common_args(for_scp=False),
                self.remote_spec,
                "bash -lc " + shlex.quote("set -euo pipefail; " + script),
            ],
            timeout_s=timeout_s,
        )

    def run_scp(
        self,
        local_path: Path,
        remote_path: str,
        *,
        timeout_s: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Copy one local file to the Pi over SCP."""

        return self._run_local(
            [
                *self._sshpass_prefix(use_stdin=True),
                "scp",
                *self._ssh_common_args(for_scp=True),
                str(local_path),
                f"{self.remote_spec}:{remote_path}",
            ],
            timeout_s=timeout_s,
        )

    def _run_local(
        self,
        args: list[str],
        *,
        timeout_s: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        effective_timeout_s = self.timeout_s if timeout_s is None else float(timeout_s)
        if effective_timeout_s <= 0:
            raise ValueError("timeout_s must be greater than zero")
        kwargs = {
            "check": False,
            "capture_output": True,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "timeout": effective_timeout_s,
        }
        password = str(self.settings.password)
        try:
            completed = self._subprocess_runner(args, input=password, **kwargs)
        except TypeError:
            completed = self._subprocess_runner(
                self._fallback_sshpass_env_args(args),
                env=sshpass_env(password),
                **kwargs,
            )
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "").strip()
            if not message:
                message = f"command failed: {' '.join(args)}"
            raise RuntimeError(message)
        return completed

    def _sshpass_prefix(self, *, use_stdin: bool) -> list[str]:
        return ["sshpass", "-d", "0"] if use_stdin else ["sshpass", "-e"]

    def _fallback_sshpass_env_args(self, args: Sequence[str]) -> list[str]:
        if len(args) >= 3 and args[0] == "sshpass" and args[1] == "-d" and args[2] == "0":
            return ["sshpass", "-e", *args[3:]]
        return list(args)

    def _ssh_common_args(self, *, for_scp: bool) -> list[str]:
        args: list[str] = []
        port = self._ssh_port()
        if port is not None:
            args.extend(["-P" if for_scp else "-p", str(port)])

        for option_name, option_value in self._ssh_option_items():
            args.extend(["-o", f"{option_name}={option_value}"])
        return args

    def _ssh_option_items(self) -> tuple[tuple[str, str], ...]:
        policy = self._ssh_host_key_policy()
        option_items: list[tuple[str, str]] = [
            ("ConnectTimeout", str(_DEFAULT_SSH_CONNECT_TIMEOUT_S)),
            ("ServerAliveInterval", str(_DEFAULT_SSH_SERVER_ALIVE_INTERVAL_S)),
            ("ServerAliveCountMax", str(_DEFAULT_SSH_SERVER_ALIVE_COUNT_MAX)),
            ("LogLevel", "ERROR"),
        ]
        if policy == "insecure":
            option_items.extend(
                (
                    ("StrictHostKeyChecking", "no"),
                    ("UserKnownHostsFile", "/dev/null"),
                )
            )
            return tuple(option_items)

        known_hosts_path = self._known_hosts_path()
        known_hosts_path.parent.mkdir(parents=True, exist_ok=True)
        known_hosts_path.touch(exist_ok=True)
        option_items.extend(
            (
                ("StrictHostKeyChecking", policy),
                ("UserKnownHostsFile", str(known_hosts_path)),
            )
        )
        return tuple(option_items)

    def _ssh_port(self) -> int | None:
        raw_port = (
            getattr(self.settings, "port", None)
            or getattr(self.settings, "ssh_port", None)
            or os.environ.get("TWINR_PI_SSH_PORT")
        )
        if raw_port in (None, ""):
            return None
        try:
            port = int(str(raw_port).strip())
        except ValueError:
            raise ValueError(f"Invalid SSH port: {raw_port!r}") from None
        if port <= 0:
            raise ValueError(f"Invalid SSH port: {raw_port!r}")
        return port

    def _ssh_host_key_policy(self) -> str:
        raw_policy = (
            getattr(self.settings, "ssh_host_key_policy", None)
            or getattr(self.settings, "host_key_policy", None)
            or os.environ.get("TWINR_PI_SSH_HOST_KEY_POLICY")
            or _DEFAULT_SSH_HOST_KEY_POLICY
        )
        policy = str(raw_policy).strip().lower()
        aliases = {
            "yes": "strict",
            "strict": "strict",
            "accept-new": "accept-new",
            "accept_new": "accept-new",
            "ask": "strict",
            "no": "insecure",
            "off": "insecure",
            "insecure": "insecure",
        }
        normalized = aliases.get(policy)
        if normalized is None:
            allowed = ", ".join(sorted({"strict", "accept-new", "insecure"}))
            raise ValueError(
                "Unsupported SSH host key policy "
                f"{raw_policy!r}; expected one of {allowed}"
            )
        return normalized

    def _known_hosts_path(self) -> Path:
        raw_path = (
            getattr(self.settings, "ssh_known_hosts_file", None)
            or getattr(self.settings, "known_hosts_file", None)
            or os.environ.get("TWINR_PI_SSH_KNOWN_HOSTS_FILE")
        )
        if raw_path is None or not str(raw_path).strip():
            return _DEFAULT_KNOWN_HOSTS_PATH
        return Path(str(raw_path)).expanduser()


class RetentionCanaryProbeError(RuntimeError):
    """Carry the structured canary payload when the remote proof fails."""

    def __init__(
        self,
        message: str,
        *,
        payload: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.payload = dict(payload) if isinstance(payload, Mapping) else None


def sync_authoritative_file(
    *,
    remote: PiRemoteExecutor,
    local_path: Path,
    remote_path: str,
    mode: str,
) -> PiSyncedFileResult:
    """Copy one authoritative local file onto the Pi with checksum verification."""

    local_bytes = local_path.read_bytes()
    local_sha = hashlib.sha256(local_bytes).hexdigest()
    remote_sha = read_remote_sha256(remote=remote, remote_path=remote_path)
    if remote_sha == local_sha:
        return PiSyncedFileResult(
            local_path=str(local_path),
            remote_path=remote_path,
            sha256=local_sha,
            changed=False,
            backup_path=None,
        )

    remote_temp = f"/tmp/twinr-deploy-{time.time_ns()}-{local_path.name}"
    backup_suffix = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    remote.run_scp(local_path, remote_temp)
    completed = remote.run_ssh(
        "\n".join(
            (
                f"target={shlex.quote(remote_path)}",
                f"tmp={shlex.quote(remote_temp)}",
                f"expected_sha={shlex.quote(local_sha)}",
                f"mode={shlex.quote(mode)}",
                f"backup=\"$target.deploy-backup-{backup_suffix}\"",
                "target_dir=$(dirname -- \"$target\")",
                "target_base=$(basename -- \"$target\")",
                "install -d \"$target_dir\"",
                "staged_target=$(mktemp \"$target_dir/.${target_base}.twinr-staged.XXXXXX\")",
                "cleanup() { rm -f \"$tmp\" \"$staged_target\"; }",
                "trap cleanup EXIT",
                "if [ -e \"$target\" ]; then cp -p -- \"$target\" \"$backup\"; else backup=\"\"; fi",
                "install -m \"$mode\" \"$tmp\" \"$staged_target\"",
                "mv -f \"$staged_target\" \"$target\"",
                "actual_sha=$(sha256sum \"$target\" | awk '{print $1}')",
                "if [ \"$actual_sha\" != \"$expected_sha\" ]; then",
                "  echo \"remote checksum mismatch after sync\" >&2",
                "  exit 1",
                "fi",
                "printf '%s' \"$backup\"",
            )
        )
    )
    backup_path = (completed.stdout or "").strip() or None
    return PiSyncedFileResult(
        local_path=str(local_path),
        remote_path=remote_path,
        sha256=local_sha,
        changed=True,
        backup_path=backup_path,
    )


def attest_remote_repo_entries(
    *,
    remote: PiRemoteExecutor,
    remote_root: str,
    entries: Sequence[PiRepoMirrorEntryDigest],
    max_samples: int = 20,
) -> PiRemoteRepoAttestationResult:
    """Independently attest the mirrored repo scope on the Pi.

    This phase is intentionally separate from the rsync mirror result so the
    operator-facing deploy cannot declare success solely because rsync believed
    the sync converged. The deploy uploads a manifest of authoritative local
    repo entries and compares the actual Pi checkout against it.
    """

    normalized_entries = tuple(
        entry
        for entry in entries
        if str(entry.relative_path or "").strip()
    )
    if max_samples <= 0:
        raise ValueError("max_samples must be greater than zero")
    if not normalized_entries:
        return PiRemoteRepoAttestationResult(
            verified_entry_count=0,
            verified_file_count=0,
            verified_symlink_count=0,
            missing_count=0,
            mismatch_count=0,
            sampled_missing_paths=(),
            sampled_mismatch_details=(),
            elapsed_s=0.0,
        )

    payload: dict[str, object] = {
        "entries": [
            {
                "relative_path": entry.relative_path,
                "kind": entry.kind,
                "sha256": entry.sha256,
                "link_target": entry.link_target,
            }
            for entry in normalized_entries
        ]
    }
    manifest_path = _write_repo_attestation_manifest(payload)
    remote_manifest_path = f"/tmp/twinr-deploy-repo-attestation-{time.time_ns()}.json"
    completed: subprocess.CompletedProcess[str] | None = None
    try:
        remote.run_scp(manifest_path, remote_manifest_path)
        completed = remote.run_ssh(
            "\n".join(
                (
                    f"repo_attestation_manifest_path={shlex.quote(remote_manifest_path)}",
                    f"repo_attestation_remote_root={shlex.quote(remote_root)}",
                    f"repo_attestation_max_samples={max_samples}",
                    "export repo_attestation_manifest_path repo_attestation_remote_root repo_attestation_max_samples",
                    "python3 - <<'PY'",
                    "from __future__ import annotations",
                    "import hashlib",
                    "import json",
                    "import os",
                    "from pathlib import Path",
                    "import time",
                    "manifest_path = Path(os.environ['repo_attestation_manifest_path'])",
                    "remote_root = Path(os.environ['repo_attestation_remote_root'])",
                    "max_samples = max(1, int(os.environ['repo_attestation_max_samples']))",
                    "payload = json.loads(manifest_path.read_text(encoding='utf-8'))",
                    "entries = payload.get('entries', [])",
                    "missing_paths = []",
                    "mismatch_details = []",
                    "verified_files = 0",
                    "verified_symlinks = 0",
                    "started = time.perf_counter()",
                    "def sha256_for_file(path: Path) -> str:",
                    "    digest = hashlib.sha256()",
                    "    with path.open('rb') as handle:",
                    "        while True:",
                    "            chunk = handle.read(1024 * 1024)",
                    "            if not chunk:",
                    "                break",
                    "            digest.update(chunk)",
                    "    return digest.hexdigest()",
                    "for item in entries:",
                    "    relative_path = str(item.get('relative_path', '') or '').strip().lstrip('/')",
                    "    kind = str(item.get('kind', '') or '').strip()",
                    "    expected_sha = str(item.get('sha256', '') or '').strip()",
                    "    expected_link_target = str(item.get('link_target', '') or '')",
                    "    if not relative_path or kind not in {'file', 'symlink'}:",
                    "        continue",
                    "    target_path = remote_root / relative_path",
                    "    path_exists = target_path.exists() or target_path.is_symlink()",
                    "    if kind == 'file':",
                    "        if not path_exists:",
                    "            missing_paths.append(relative_path)",
                    "            continue",
                    "        if target_path.is_symlink():",
                    "            mismatch_details.append(",
                    "                f\"{relative_path}: expected file sha256 {expected_sha}, found symlink -> {os.readlink(target_path)}\"",
                    "            )",
                    "            continue",
                    "        if not target_path.is_file():",
                    "            mismatch_details.append(",
                    "                f\"{relative_path}: expected file sha256 {expected_sha}, found non-file entry\"",
                    "            )",
                    "            continue",
                    "        actual_sha = sha256_for_file(target_path)",
                    "        if actual_sha != expected_sha:",
                    "            mismatch_details.append(",
                    "                f\"{relative_path}: expected sha256 {expected_sha}, got {actual_sha}\"",
                    "            )",
                    "            continue",
                    "        verified_files += 1",
                    "        continue",
                    "    if not path_exists:",
                    "        missing_paths.append(relative_path)",
                    "        continue",
                    "    if not target_path.is_symlink():",
                    "        mismatch_details.append(",
                    "            f\"{relative_path}: expected symlink -> {expected_link_target}, found regular entry\"",
                    "        )",
                    "        continue",
                    "    actual_link_target = os.readlink(target_path)",
                    "    if actual_link_target != expected_link_target:",
                    "        mismatch_details.append(",
                    "            f\"{relative_path}: expected symlink -> {expected_link_target}, got -> {actual_link_target}\"",
                    "        )",
                    "        continue",
                    "    verified_symlinks += 1",
                    "result = {",
                    "    'verified_entry_count': len(entries),",
                    "    'verified_file_count': verified_files,",
                    "    'verified_symlink_count': verified_symlinks,",
                    "    'missing_count': len(missing_paths),",
                    "    'mismatch_count': len(mismatch_details),",
                    "    'sampled_missing_paths': missing_paths[:max_samples],",
                    "    'sampled_mismatch_details': mismatch_details[:max_samples],",
                    "    'elapsed_s': round(time.perf_counter() - started, 3),",
                    "}",
                    "print(json.dumps(result, ensure_ascii=False))",
                    "PY",
                    "rm -f \"$repo_attestation_manifest_path\"",
                )
            )
        )
    finally:
        manifest_path.unlink(missing_ok=True)
        try:
            remote.run_ssh(f"rm -f {shlex.quote(remote_manifest_path)}")
        except Exception:
            pass
    payload = json.loads((completed.stdout if completed is not None else "{}").strip() or "{}")
    sampled_missing_paths_raw = payload.get("sampled_missing_paths", ())
    sampled_mismatch_details_raw = payload.get("sampled_mismatch_details", ())
    result = PiRemoteRepoAttestationResult(
        verified_entry_count=max(0, int(str(payload.get("verified_entry_count", 0) or 0).strip())),
        verified_file_count=max(0, int(str(payload.get("verified_file_count", 0) or 0).strip())),
        verified_symlink_count=max(
            0, int(str(payload.get("verified_symlink_count", 0) or 0).strip())
        ),
        missing_count=max(0, int(str(payload.get("missing_count", 0) or 0).strip())),
        mismatch_count=max(0, int(str(payload.get("mismatch_count", 0) or 0).strip())),
        sampled_missing_paths=(
            tuple(str(path).strip() for path in sampled_missing_paths_raw if str(path).strip())
            if isinstance(sampled_missing_paths_raw, list)
            else ()
        ),
        sampled_mismatch_details=(
            tuple(str(detail).strip() for detail in sampled_mismatch_details_raw if str(detail).strip())
            if isinstance(sampled_mismatch_details_raw, list)
            else ()
        ),
        elapsed_s=max(0.0, float(str(payload.get("elapsed_s", 0.0) or 0.0).strip())),
    )
    if result.missing_count or result.mismatch_count:
        detail_parts: list[str] = []
        if result.sampled_missing_paths:
            detail_parts.append("missing " + ", ".join(result.sampled_missing_paths))
        if result.sampled_mismatch_details:
            detail_parts.append("mismatched " + "; ".join(result.sampled_mismatch_details))
        detail_summary = "; ".join(detail_parts) if detail_parts else "remote authoritative repo mismatch"
        raise RuntimeError(
            "remote authoritative repo attestation failed: "
            f"{result.missing_count} missing, {result.mismatch_count} mismatched; {detail_summary}"
        )
    return result


def _write_repo_attestation_manifest(payload: dict[str, object]) -> Path:
    """Persist one temporary JSON manifest for remote repo attestation."""

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix="twinr-pi-attestation-",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.write("\n")
        return Path(handle.name)


def read_remote_sha256(*, remote: PiRemoteExecutor, remote_path: str) -> str | None:
    """Return the remote file checksum if the file exists, otherwise ``None``."""

    completed = remote.run_ssh(
        f"if [ -f {shlex.quote(remote_path)} ]; then sha256sum {shlex.quote(remote_path)} | awk '{{print $1}}'; fi"
    )
    checksum = (completed.stdout or "").strip()
    return checksum or None


def install_editable_package(
    *,
    remote: PiRemoteExecutor,
    remote_root: str,
    install_with_deps: bool,
    progress_callback: ProgressCallback | None = None,
) -> str:
    """Ensure the Pi venv exists and refresh the editable Twinr install.

    The default deploy path keeps ``pip install -e`` in ``--no-deps`` mode so
    stable Pi-host packages such as ``PyQt5`` are not rebuilt on every rollout.
    To keep that fast path integration-safe when the project adds new runtime
    dependencies, the deploy first performs the cheap editable refresh and then
    uses pip's modern dry-run/report flow (with a legacy fallback for older pip)
    to decide whether a full resolver-backed sync is required.
    """

    remote_python = f"{remote_root}/.venv/bin/python"
    with progress_span(
        progress_callback,
        phase="editable_install",
        step="ensure_remote_venv",
        detail=remote_root,
    ):
        remote.run_ssh(
            "\n".join(
                (
                    f"remote_root={shlex.quote(remote_root)}",
                    f"remote_python={shlex.quote(remote_python)}",
                    "if [ ! -x \"$remote_python\" ]; then python3 -m venv \"$remote_root/.venv\"; fi",
                )
            )
        )
    with progress_span(
        progress_callback,
        phase="editable_install",
        step="bridge_system_site_packages",
    ):
        site_bridge_result = _bridge_remote_venv_system_site_packages(
            remote=remote,
            remote_root=remote_root,
            remote_python=remote_python,
        )
    bridge_summary = ""
    if site_bridge_result.active_paths:
        path_count = len(site_bridge_result.active_paths)
        noun = "path" if path_count == 1 else "paths"
        if site_bridge_result.changed:
            bridge_summary = f"bridged {path_count} Pi system site-package {noun} into the venv"
        else:
            bridge_summary = f"verified Pi system site-package bridge for {path_count} {noun}"
    with progress_span(
        progress_callback,
        phase="editable_install",
        step="cleanup_shadowed_system_packages",
    ):
        shadowed_cleanup_summary = _cleanup_remote_venv_shadowed_system_distributions(
            remote=remote,
            remote_root=remote_root,
            remote_python=remote_python,
            active_system_paths=site_bridge_result.active_paths,
        )
    pip_args = "-e \"$remote_root\"" if install_with_deps else "--no-deps -e \"$remote_root\""
    with progress_span(
        progress_callback,
        phase="editable_install",
        step="pip_install_editable",
        detail="with_deps" if install_with_deps else "no_deps",
    ):
        completed = remote.run_ssh(
            "\n".join(
                (
                    f"remote_root={shlex.quote(remote_root)}",
                    f"remote_python={shlex.quote(remote_python)}",
                    "export PIP_DISABLE_PIP_VERSION_CHECK=1",
                    "export PYTHONUTF8=1",
                    f"\"$remote_python\" -m pip install {pip_args}",
                )
            )
        )
    dependency_sync_summary = ""
    if not install_with_deps:
        with progress_span(
            progress_callback,
            phase="editable_install",
            step="sync_runtime_dependencies",
        ):
            dependency_sync_summary = sync_project_runtime_dependencies(
                remote=remote,
                remote_root=remote_root,
                remote_python=remote_python,
            )
    with progress_span(
        progress_callback,
        phase="editable_install",
        step="pip_check",
    ):
        package_graph_summary = verify_remote_python_package_graph(
            remote=remote,
            remote_python=remote_python,
        )
    with progress_span(
        progress_callback,
        phase="editable_install",
        step="repair_venv_entrypoints",
    ):
        repair_result = _repair_remote_venv_python_shebangs(
            remote=remote,
            remote_root=remote_root,
            remote_python=remote_python,
        )
    summary_parts = [
        summarize_output(completed),
        bridge_summary,
        shadowed_cleanup_summary,
        dependency_sync_summary,
        package_graph_summary,
    ]
    if repair_result.rewritten_files:
        repaired = ", ".join(repair_result.sample_paths) or "stale wrappers"
        summary_parts.append(
            "normalized "
            f"{repair_result.rewritten_files} stale venv entrypoint file(s): {repaired}"
        )
    elif repair_result.checked_files:
        summary_parts.append(
            "verified "
            f"{repair_result.checked_files} venv entrypoint file(s); no stale paths found"
        )
    return summarize_text("\n".join(part for part in summary_parts if part))


def _cleanup_remote_venv_shadowed_system_distributions(
    *,
    remote: PiRemoteExecutor,
    remote_root: str,
    remote_python: str,
    active_system_paths: Sequence[str],
) -> str:
    """Remove venv copies when bridged Pi packages already satisfy direct requirements."""

    normalized_system_paths = tuple(
        str(path).strip()
        for path in active_system_paths
        if str(path).strip()
    )
    if not normalized_system_paths:
        return ""
    shadowed_distributions = _load_remote_shadowed_system_distributions(
        remote=remote,
        remote_root=remote_root,
        remote_python=remote_python,
        active_system_paths=normalized_system_paths,
    )
    if not shadowed_distributions:
        return ""

    uninstall_names = tuple(
        dict.fromkeys(distribution.name for distribution in shadowed_distributions if distribution.name)
    )
    if not uninstall_names:
        return ""
    uninstall_args = " ".join(shlex.quote(name) for name in uninstall_names)
    completed = remote.run_ssh(
        "\n".join(
            (
                f"remote_python={shlex.quote(remote_python)}",
                "export PIP_DISABLE_PIP_VERSION_CHECK=1",
                "export PYTHONUTF8=1",
                f"\"$remote_python\" -m pip uninstall -y {uninstall_args}",
            )
        )
    )
    cleanup_targets = ", ".join(
        f"{distribution.name} (venv {distribution.venv_version} -> system {distribution.system_version})"
        for distribution in shadowed_distributions
    )
    noun = "dependency" if len(uninstall_names) == 1 else "dependencies"
    prefix = (
        f"removed {len(uninstall_names)} venv-shadowed direct {noun} in favor of bridged system packages: "
        f"{cleanup_targets}"
    )
    details = summarize_output(completed)
    return prefix if not details else f"{prefix}\n{details}"


def _load_remote_shadowed_system_distributions(
    *,
    remote: PiRemoteExecutor,
    remote_root: str,
    remote_python: str,
    active_system_paths: Sequence[str],
) -> tuple[BridgedSystemShadowedDistribution, ...]:
    """Return direct dependencies duplicated in the venv and bridged system paths."""

    helper_path = f"{remote_root}/src/twinr/ops/venv_bridged_system_cleanup.py"
    script = "\n".join(
        (
            f"{shlex.quote(remote_python)} - <<'PY'",
            "import importlib.util",
            "import json",
            "import sys",
            "import sysconfig",
            "from pathlib import Path",
            f"helper_path = Path({json.dumps(helper_path)})",
            "spec = importlib.util.spec_from_file_location('twinr_ops_venv_bridged_system_cleanup', helper_path)",
            "if spec is None or spec.loader is None:",
            "    raise RuntimeError(f'Could not load venv cleanup helper from {helper_path}')",
            "module = importlib.util.module_from_spec(spec)",
            "sys.modules[spec.name] = module",
            "spec.loader.exec_module(module)",
            (
                "result = module.find_shadowed_direct_dependency_distributions("
                f"project_pyproject=Path({json.dumps(f'{remote_root}/pyproject.toml')}), "
                "venv_site_packages_dir=Path(sysconfig.get_path('purelib')), "
                f"bridged_system_paths=tuple({json.dumps(list(active_system_paths), ensure_ascii=False)})"
                ")"
            ),
            "print(json.dumps([",
            "    {",
            "        'name': item.name,",
            "        'venv_version': item.venv_version,",
            "        'venv_path': item.venv_path,",
            "        'system_version': item.system_version,",
            "        'system_path': item.system_path,",
            "    }",
            "    for item in result",
            "], ensure_ascii=False))",
            "PY",
        )
    )
    completed = remote.run_ssh(script)
    raw_payload = json.loads((completed.stdout or "[]").strip() or "[]")
    if not isinstance(raw_payload, list):
        return ()
    shadowed_distributions: list[BridgedSystemShadowedDistribution] = []
    for item in raw_payload:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        shadowed_distributions.append(
            BridgedSystemShadowedDistribution(
                name=name,
                venv_version=str(item.get("venv_version", "")).strip(),
                venv_path=str(item.get("venv_path", "")).strip(),
                system_version=str(item.get("system_version", "")).strip(),
                system_path=str(item.get("system_path", "")).strip(),
            )
        )
    return tuple(shadowed_distributions)


def sync_project_runtime_dependencies(
    *,
    remote: PiRemoteExecutor,
    remote_root: str,
    remote_python: str,
) -> str:
    """Resolve only when the mirrored project would change runtime dependencies."""

    pending_resolved_distributions = load_pending_project_runtime_distributions(
        remote=remote,
        remote_root=remote_root,
        remote_python=remote_python,
    )
    if pending_resolved_distributions == ():
        return "verified mirrored project dependencies; no resolver changes needed"

    if pending_resolved_distributions is None:
        pending_requirements = _load_pending_project_runtime_requirements_legacy(
            remote=remote,
            remote_root=remote_root,
            remote_python=remote_python,
        )
        if not pending_requirements:
            return "verified mirrored project dependencies via legacy check; no changes needed"
        requirement_args = " ".join(shlex.quote(requirement) for requirement in pending_requirements)
        completed = remote.run_ssh(
            "\n".join(
                (
                    f"remote_python={shlex.quote(remote_python)}",
                    "export PIP_DISABLE_PIP_VERSION_CHECK=1",
                    "export PYTHONUTF8=1",
                    f"\"$remote_python\" -m pip install {requirement_args}",
                )
            )
        )
        dependency_count = len(pending_requirements)
        noun = "dependency" if dependency_count == 1 else "dependencies"
        installed = ", ".join(pending_requirements)
        prefix = f"installed {dependency_count} mirrored project {noun} via legacy fallback: {installed}"
        details = summarize_output(completed)
        return prefix if not details else f"{prefix}\n{details}"

    completed = remote.run_ssh(
        "\n".join(
            (
                f"remote_python={shlex.quote(remote_python)}",
                f"remote_root={shlex.quote(remote_root)}",
                "export PIP_DISABLE_PIP_VERSION_CHECK=1",
                "export PYTHONUTF8=1",
                "\"$remote_python\" -m pip install -e \"$remote_root\"",
            )
        )
    )
    distribution_count = len(pending_resolved_distributions)
    noun = "distribution" if distribution_count == 1 else "distributions"
    installed = ", ".join(pending_resolved_distributions)
    prefix = (
        "resolved mirrored project runtime drift with pip's dependency resolver: "
        f"{distribution_count} {noun} would have changed ({installed})"
    )
    details = summarize_output(completed)
    return prefix if not details else f"{prefix}\n{details}"


def load_pending_project_runtime_distributions(
    *,
    remote: PiRemoteExecutor,
    remote_root: str,
    remote_python: str,
) -> tuple[str, ...] | None:
    """Return resolved non-project distributions pip would change, or ``None`` if unavailable."""

    script = "\n".join(
        (
            f"{shlex.quote(remote_python)} - <<'PY'",
            "from __future__ import annotations",
            "import json",
            "import os",
            "from pathlib import Path",
            "import re",
            "import subprocess",
            "import tomllib",
            "from urllib.parse import unquote, urlparse",
            f"remote_root = Path({json.dumps(remote_root)}).resolve()",
            f"remote_python = {json.dumps(remote_python)}",
            "def normalize_name(value: str) -> str:",
            "    return re.sub(r'[-_.]+', '-', value).strip().lower()",
            "payload = tomllib.loads((remote_root / 'pyproject.toml').read_text(encoding='utf-8'))",
            "project_name = normalize_name(str(payload.get('project', {}).get('name', '') or ''))",
            "command = [remote_python, '-m', 'pip', 'install', '--dry-run', '--quiet', '--report', '-', '-e', str(remote_root)]",
            "completed = subprocess.run(",
            "    command,",
            "    check=False,",
            "    capture_output=True,",
            "    text=True,",
            "    encoding='utf-8',",
            "    errors='replace',",
            "    env={**dict(os.environ), 'PIP_DISABLE_PIP_VERSION_CHECK': '1', 'PYTHONUTF8': '1'},",
            ")",
            "if completed.returncode != 0:",
            "    error_text = (completed.stderr or completed.stdout or '').strip()",
            "    print(json.dumps({'ok': False, 'error': error_text}, ensure_ascii=False))",
            "    raise SystemExit(0)",
            "try:",
            "    report = json.loads((completed.stdout or '{}').strip() or '{}')",
            "except json.JSONDecodeError as exc:",
            "    print(json.dumps({'ok': False, 'error': f'could not parse pip report: {exc}'}, ensure_ascii=False))",
            "    raise SystemExit(0)",
            "install_items = report.get('install', [])",
            "pending = []",
            "for item in install_items:",
            "    if not isinstance(item, dict):",
            "        continue",
            "    metadata = item.get('metadata', {}) if isinstance(item.get('metadata'), dict) else {}",
            "    raw_name = str(metadata.get('name', '') or '').strip()",
            "    normalized_name = normalize_name(raw_name)",
            "    version = str(metadata.get('version', '') or '').strip()",
            "    if project_name and normalized_name == project_name:",
            "        continue",
            "    download_info = item.get('download_info', {}) if isinstance(item.get('download_info'), dict) else {}",
            "    direct_url = str(download_info.get('url', '') or '').strip()",
            "    if direct_url.startswith('file:'):",
            "        try:",
            "            direct_path = Path(unquote(urlparse(direct_url).path)).resolve()",
            "        except Exception:",
            "            direct_path = None",
            "        if direct_path is not None and direct_path == remote_root:",
            "            continue",
            "    if raw_name and version:",
            "        pending.append(f'{raw_name}=={version}')",
            "        continue",
            "    if raw_name:",
            "        pending.append(raw_name)",
            "        continue",
            "    if direct_url:",
            "        pending.append(direct_url)",
            "result = {",
            "    'ok': True,",
            "    'pending': list(dict.fromkeys(pending)),",
            "}",
            "print(json.dumps(result, ensure_ascii=False))",
            "PY",
        )
    )
    completed = remote.run_ssh(script)
    payload = json.loads((completed.stdout or "{}").strip() or "{}")
    if not bool(payload.get("ok", False)):
        return None
    raw_pending = payload.get("pending", ())
    if not isinstance(raw_pending, list):
        return ()
    return tuple(
        str(item).strip()
        for item in raw_pending
        if str(item).strip()
    )


def _load_pending_project_runtime_requirements_legacy(
    *,
    remote: PiRemoteExecutor,
    remote_root: str,
    remote_python: str,
) -> tuple[str, ...]:
    """Return mirrored project requirements missing or out of spec on the Pi."""

    project_pyproject = f"{remote_root}/pyproject.toml"
    script = "\n".join(
        (
            f"{shlex.quote(remote_python)} - <<'PY'",
            "from __future__ import annotations",
            "import importlib.metadata as importlib_metadata",
            "import json",
            "from pathlib import Path",
            "import tomllib",
            "try:",
            "    from packaging.markers import default_environment",
            "    from packaging.requirements import Requirement",
            "except Exception:",
            "    from pip._vendor.packaging.markers import default_environment",
            "    from pip._vendor.packaging.requirements import Requirement",
            f"project_pyproject = Path({json.dumps(project_pyproject)})",
            "project_payload = tomllib.loads(project_pyproject.read_text(encoding='utf-8'))",
            "dependencies = project_payload.get('project', {}).get('dependencies', [])",
            "environment = default_environment()",
            "pending_requirements = []",
            "for raw_dependency in dependencies:",
            "    requirement = Requirement(raw_dependency)",
            "    if requirement.marker is not None and not requirement.marker.evaluate(environment):",
            "        continue",
            "    try:",
            "        installed_version = importlib_metadata.version(requirement.name)",
            "    except importlib_metadata.PackageNotFoundError:",
            "        pending_requirements.append(raw_dependency)",
            "        continue",
            "    try:",
            "        if requirement.specifier and not requirement.specifier.contains(installed_version, prereleases=True):",
            "            pending_requirements.append(raw_dependency)",
            "    except Exception:",
            "        pending_requirements.append(raw_dependency)",
            "print(json.dumps({'requirements': pending_requirements}, ensure_ascii=False))",
            "PY",
        )
    )
    completed = remote.run_ssh(script)
    payload = json.loads((completed.stdout or "{}").strip() or "{}")
    raw_requirements = payload.get("requirements", ())
    if not isinstance(raw_requirements, list):
        return ()
    pending_requirements = tuple(
        str(requirement).strip()
        for requirement in raw_requirements
        if str(requirement).strip()
    )
    return pending_requirements


def verify_remote_python_package_graph(
    *,
    remote: PiRemoteExecutor,
    remote_python: str,
) -> str:
    """Run `pip check` inside the remote venv and raise on a broken package graph."""

    completed = remote.run_ssh(
        "\n".join(
            (
                f"remote_python={shlex.quote(remote_python)}",
                "export PIP_DISABLE_PIP_VERSION_CHECK=1",
                "export PYTHONUTF8=1",
                "\"$remote_python\" -m pip check",
            )
        )
    )
    summary = summarize_output(completed)
    return summary or "verified remote Python package graph; no broken requirements found"


def _repair_remote_venv_python_shebangs(
    *,
    remote: PiRemoteExecutor,
    remote_root: str,
    remote_python: str,
) -> PiVenvScriptRepairResult:
    """Run the venv wrapper-shebang repair in the Pi runtime environment."""

    remote_bin_dir = f"{remote_root}/.venv/bin"
    helper_path = f"{remote_root}/src/twinr/ops/venv_wrapper_repair.py"
    script = "\n".join(
        (
            f"{shlex.quote(remote_python)} - <<'PY'",
            "import importlib.util",
            "import sys",
            "from pathlib import Path",
            "import json",
            f"helper_path = Path({json.dumps(helper_path)})",
            "spec = importlib.util.spec_from_file_location('twinr_ops_venv_wrapper_repair', helper_path)",
            "if spec is None or spec.loader is None:",
            "    raise RuntimeError(f'Could not load venv repair helper from {helper_path}')",
            "module = importlib.util.module_from_spec(spec)",
            "sys.modules[spec.name] = module",
            "spec.loader.exec_module(module)",
            (
                "result = module.repair_venv_python_shebangs("
                f"bin_dir=Path({json.dumps(remote_bin_dir)}), "
                f"expected_interpreter={json.dumps(remote_python)}"
                ")"
            ),
            "print(json.dumps({",
            "    'checked_files': result.checked_files,",
            "    'rewritten_files': result.rewritten_files,",
            "    'sample_paths': list(result.sample_paths),",
            "}, ensure_ascii=False))",
            "PY",
        )
    )
    completed = remote.run_ssh(script)
    payload = json.loads((completed.stdout or "{}").strip() or "{}")
    sample_paths_raw = payload.get("sample_paths", ())
    sample_paths = (
        tuple(str(item).strip() for item in sample_paths_raw if str(item).strip())
        if isinstance(sample_paths_raw, list)
        else ()
    )
    return PiVenvScriptRepairResult(
        checked_files=max(0, int(payload.get("checked_files", 0) or 0)),
        rewritten_files=max(0, int(payload.get("rewritten_files", 0) or 0)),
        sample_paths=sample_paths,
    )


def _bridge_remote_venv_system_site_packages(
    *,
    remote: PiRemoteExecutor,
    remote_root: str,
    remote_python: str,
) -> PiVenvSystemSiteBridgeResult:
    """Expose Pi OS-managed `dist-packages` inside the preserved venv."""

    helper_path = f"{remote_root}/src/twinr/ops/venv_system_site_bridge.py"
    script = "\n".join(
        (
            f"{shlex.quote(remote_python)} - <<'PY'",
            "import importlib.util",
            "import json",
            "import sys",
            "import sysconfig",
            "from pathlib import Path",
            f"helper_path = Path({json.dumps(helper_path)})",
            "spec = importlib.util.spec_from_file_location('twinr_ops_venv_system_site_bridge', helper_path)",
            "if spec is None or spec.loader is None:",
            "    raise RuntimeError(f'Could not load venv bridge helper from {helper_path}')",
            "module = importlib.util.module_from_spec(spec)",
            "sys.modules[spec.name] = module",
            "spec.loader.exec_module(module)",
            (
                "result = module.ensure_pi_system_site_packages_bridge("
                "site_packages_dir=Path(sysconfig.get_path('purelib'))"
                ")"
            ),
            "print(json.dumps({",
            "    'bridge_path': result.bridge_path,",
            "    'active_paths': list(result.active_paths),",
            "    'changed': result.changed,",
            "}, ensure_ascii=False))",
            "PY",
        )
    )
    completed = remote.run_ssh(script)
    payload = json.loads((completed.stdout or "{}").strip() or "{}")
    active_paths_raw = payload.get("active_paths", ())
    active_paths = (
        tuple(str(item).strip() for item in active_paths_raw if str(item).strip())
        if isinstance(active_paths_raw, list)
        else ()
    )
    return PiVenvSystemSiteBridgeResult(
        bridge_path=str(payload.get("bridge_path", "")).strip(),
        active_paths=active_paths,
        changed=bool(payload.get("changed", False)),
    )


def install_python_requirements_manifest(
    *,
    remote: PiRemoteExecutor,
    remote_root: str,
    manifest_relpath: str,
    label: str,
    progress_callback: ProgressCallback | None = None,
    progress_phase: str | None = None,
) -> str:
    """Install one mirrored Python requirements manifest into the Pi venv."""

    remote_python = f"{remote_root}/.venv/bin/python"
    requirements_path = f"{remote_root}/{manifest_relpath.lstrip('/')}"
    phase_name = progress_phase or f"{label}_requirements"
    with progress_span(
        progress_callback,
        phase=phase_name,
        step="pip_install_requirements",
        detail=manifest_relpath,
    ):
        completed = remote.run_ssh(
            "\n".join(
                (
                    f"remote_root={shlex.quote(remote_root)}",
                    f"remote_python={shlex.quote(remote_python)}",
                    f"requirements_path={shlex.quote(requirements_path)}",
                    "export PIP_DISABLE_PIP_VERSION_CHECK=1",
                    "export PYTHONUTF8=1",
                    "if [ ! -x \"$remote_python\" ]; then python3 -m venv \"$remote_root/.venv\"; fi",
                    'if [ ! -s "$requirements_path" ]; then',
                    '  echo "missing or empty Python requirements manifest: $requirements_path" >&2',
                    '  requirements_dir=$(dirname -- "$requirements_path")',
                    '  if [ -d "$requirements_dir" ]; then ls -la "$requirements_dir" >&2; fi',
                    '  exit 1',
                    'fi',
                    f'echo "[{label}] installing python requirements"',
                    '"$remote_python" -m pip install -r "$requirements_path"',
                )
            )
        )
    summary_parts = [summarize_output(completed)]
    with progress_span(
        progress_callback,
        phase=phase_name,
        step="pip_check",
    ):
        summary_parts.append(
            verify_remote_python_package_graph(
                remote=remote,
                remote_python=remote_python,
            )
        )
    return summarize_text("\n".join(part for part in summary_parts if part))


def verify_python_import_contract(
    *,
    remote: PiRemoteExecutor,
    remote_python: str,
    modules: Sequence[str],
    attribute_contracts: dict[str, Sequence[str]] | None = None,
) -> PiPythonImportContractResult:
    """Import and attest a fixed Python surface inside the remote Pi venv.

    ``attribute_contracts`` maps one import target spec to the attributes that
    must exist on that target after import. A target spec is either one module
    path (for module-level attributes) or ``module.path:Symbol.path`` for a
    nested object inside that module.
    """

    normalized_modules = tuple(str(module).strip() for module in modules if str(module).strip())
    normalized_attribute_contracts: dict[str, tuple[str, ...]] = {}
    for raw_target, raw_attributes in (attribute_contracts or {}).items():
        target = str(raw_target).strip()
        if not target:
            continue
        attributes = tuple(
            dict.fromkeys(str(attribute).strip() for attribute in raw_attributes if str(attribute).strip())
        )
        if not attributes:
            continue
        normalized_attribute_contracts[target] = attributes

    expected_attribute_checks = tuple(
        f"{target}.{attribute}"
        for target, attributes in normalized_attribute_contracts.items()
        for attribute in attributes
    )

    if not normalized_modules and not normalized_attribute_contracts:
        return PiPythonImportContractResult(
            python_path=remote_python,
            checked_modules=(),
            imported_modules=(),
            checked_attribute_contracts=(),
            validated_attribute_contracts=(),
            elapsed_s=0.0,
        )

    script = "\n".join(
        (
            f"{shlex.quote(remote_python)} - <<'PY'",
            "import importlib",
            "import json",
            "import time",
            f"modules = {json.dumps(list(normalized_modules), ensure_ascii=False)}",
            "attribute_contracts = "
            + json.dumps(
                {target: list(attributes) for target, attributes in normalized_attribute_contracts.items()},
                ensure_ascii=False,
            ),
            "started = time.perf_counter()",
            "imported = []",
            "failed = {}",
            "checked_attribute_contracts = []",
            "validated_attribute_contracts = []",
            "failed_attribute_contracts = {}",
            "missing_attributes = {}",
            "module_cache = {}",
            "def load_module(name):",
            "    module = module_cache.get(name)",
            "    if module is not None:",
            "        return module",
            "    module = importlib.import_module(name)",
            "    module_cache[name] = module",
            "    return module",
            "def resolve_target(spec):",
            "    module_name, separator, symbol_path = spec.partition(':')",
            "    module = load_module(module_name if separator else spec)",
            "    target = module",
            "    if separator and symbol_path:",
            "        for segment in symbol_path.split('.'):",
            "            target = getattr(target, segment)",
            "    return target",
            "for name in modules:",
            "    try:",
            "        load_module(name)",
            "    except Exception as exc:",
            "        failed[name] = f'{type(exc).__name__}: {exc}'",
            "    else:",
            "        imported.append(name)",
            "for target_spec, required_attributes in attribute_contracts.items():",
            "    checked_attribute_contracts.extend(f'{target_spec}.{attribute}' for attribute in required_attributes)",
            "    try:",
            "        target = resolve_target(target_spec)",
            "    except Exception as exc:",
            "        failed_attribute_contracts[target_spec] = f'{type(exc).__name__}: {exc}'",
            "        continue",
            "    missing = []",
            "    for attribute in required_attributes:",
            "        if hasattr(target, attribute):",
            "            validated_attribute_contracts.append(f'{target_spec}.{attribute}')",
            "            continue",
            "        missing.append(attribute)",
            "    if missing:",
            "        missing_attributes[target_spec] = missing",
            "payload = {",
            "    'python_path': " + json.dumps(remote_python) + ",",
            "    'checked_modules': modules,",
            "    'imported_modules': imported,",
            "    'failed_imports': failed,",
            "    'checked_attribute_contracts': checked_attribute_contracts,",
            "    'validated_attribute_contracts': validated_attribute_contracts,",
            "    'failed_attribute_contracts': failed_attribute_contracts,",
            "    'missing_attributes': missing_attributes,",
            "    'elapsed_s': round(time.perf_counter() - started, 3),",
            "}",
            "print(json.dumps(payload, ensure_ascii=False))",
            "PY",
        )
    )
    completed = remote.run_ssh(script)
    payload = json.loads((completed.stdout or "{}").strip() or "{}")

    checked_modules_raw = payload.get("checked_modules", ())
    imported_modules_raw = payload.get("imported_modules", ())
    failed_imports_raw = payload.get("failed_imports", {})
    checked_attribute_contracts_raw = payload.get("checked_attribute_contracts", ())
    validated_attribute_contracts_raw = payload.get("validated_attribute_contracts", ())
    failed_attribute_contracts_raw = payload.get("failed_attribute_contracts", {})
    missing_attributes_raw = payload.get("missing_attributes", {})

    checked_modules = (
        tuple(str(item).strip() for item in checked_modules_raw if str(item).strip())
        if isinstance(checked_modules_raw, list)
        else normalized_modules
    )
    imported_modules = (
        tuple(str(item).strip() for item in imported_modules_raw if str(item).strip())
        if isinstance(imported_modules_raw, list)
        else ()
    )
    failed_imports = (
        {
            str(module).strip(): str(error).strip()
            for module, error in failed_imports_raw.items()
            if str(module).strip()
        }
        if isinstance(failed_imports_raw, dict)
        else {}
    )
    if failed_imports:
        details = "; ".join(f"{module} -> {error}" for module, error in sorted(failed_imports.items()))
        raise RuntimeError(f"remote python import contract failed for {remote_python}: {details}")
    missing_modules = tuple(module for module in checked_modules if module not in set(imported_modules))
    if missing_modules:
        missing_list = ", ".join(missing_modules)
        raise RuntimeError(
            "remote python import contract returned an incomplete result for "
            f"{remote_python}: missing {missing_list}"
        )

    checked_attribute_contracts = (
        tuple(str(item).strip() for item in checked_attribute_contracts_raw if str(item).strip())
        if isinstance(checked_attribute_contracts_raw, list)
        else ()
    )
    validated_attribute_contracts = (
        tuple(str(item).strip() for item in validated_attribute_contracts_raw if str(item).strip())
        if isinstance(validated_attribute_contracts_raw, list)
        else ()
    )
    failed_attribute_contracts = (
        {
            str(target).strip(): str(error).strip()
            for target, error in failed_attribute_contracts_raw.items()
            if str(target).strip()
        }
        if isinstance(failed_attribute_contracts_raw, dict)
        else {}
    )
    missing_attributes = (
        {
            str(target).strip(): tuple(
                str(attribute).strip()
                for attribute in attributes
                if str(attribute).strip()
            )
            for target, attributes in missing_attributes_raw.items()
            if str(target).strip() and isinstance(attributes, list)
        }
        if isinstance(missing_attributes_raw, dict)
        else {}
    )
    if failed_attribute_contracts:
        details = "; ".join(
            f"{target} -> {error}" for target, error in sorted(failed_attribute_contracts.items())
        )
        raise RuntimeError(f"remote python attribute contract failed for {remote_python}: {details}")
    if expected_attribute_checks:
        missing_checked_contracts = tuple(
            contract for contract in expected_attribute_checks if contract not in set(checked_attribute_contracts)
        )
        if missing_checked_contracts:
            missing_list = ", ".join(missing_checked_contracts)
            raise RuntimeError(
                "remote python import contract returned an incomplete attribute result for "
                f"{remote_python}: missing {missing_list}"
            )
        if missing_attributes:
            details = "; ".join(
                f"{target} -> {', '.join(attributes)}"
                for target, attributes in sorted(missing_attributes.items())
            )
            raise RuntimeError(f"remote python attribute contract failed for {remote_python}: {details}")
        missing_validated_contracts = tuple(
            contract for contract in expected_attribute_checks if contract not in set(validated_attribute_contracts)
        )
        if missing_validated_contracts:
            missing_list = ", ".join(missing_validated_contracts)
            raise RuntimeError(
                "remote python import contract returned an incomplete validation result for "
                f"{remote_python}: missing {missing_list}"
            )

    return PiPythonImportContractResult(
        python_path=str(payload.get("python_path", remote_python)).strip() or remote_python,
        checked_modules=checked_modules,
        imported_modules=imported_modules,
        checked_attribute_contracts=checked_attribute_contracts,
        validated_attribute_contracts=validated_attribute_contracts,
        elapsed_s=max(0.0, float(payload.get("elapsed_s", 0.0) or 0.0)),
    )


def refresh_python_bytecode(
    *,
    remote: PiRemoteExecutor,
    remote_python: str,
    roots: Sequence[str],
) -> str:
    """Rebuild checked-hash bytecode for mirrored Python sources on the Pi.

    The repo mirror intentionally excludes ``__pycache__`` and ``*.pyc`` from
    authoritative sync, so the deploy must refresh mirrored bytecode explicitly
    to prevent stale timestamp-based caches from shadowing newer source files.
    Older Pi imports can also leave those cache directories owned by ``root``,
    so the refresh first hands any existing mirrored ``__pycache__`` trees back
    to the runtime user before `compileall` rebuilds checked-hash bytecode.
    """

    normalized_roots = tuple(dict.fromkeys(str(root).strip() for root in roots if str(root).strip()))
    if not normalized_roots:
        return ""
    quoted_roots = " ".join(shlex.quote(root) for root in normalized_roots)
    completed = remote.run_ssh(
        "\n".join(
            (
                f"remote_python={shlex.quote(remote_python)}",
                f"find {quoted_roots} -type d -name __pycache__ -exec sudo chown -R \"$(id -u):$(id -g)\" {{}} +",
                f"\"$remote_python\" -m compileall -q -f --invalidation-mode checked-hash {quoted_roots}",
            )
        )
    )
    summary = summarize_output(completed)
    if summary:
        return summary
    roots_summary = ", ".join(normalized_roots)
    return f"refreshed checked-hash bytecode for {roots_summary}"


def install_browser_automation_runtime_support(
    *,
    remote: PiRemoteExecutor,
    remote_root: str,
    install_python_requirements: bool,
    install_playwright_browsers: bool,
    progress_callback: ProgressCallback | None = None,
) -> str:
    """Install browser-automation runtime support from the deployed snapshot.

    Args:
        remote: Remote executor targeting the Pi acceptance host.
        remote_root: Mirrored Twinr checkout root on the Pi.
        install_python_requirements: Whether to install Python packages from
            ``browser_automation/runtime_requirements.txt``.
        install_playwright_browsers: Whether to install Playwright browser
            binaries listed in ``browser_automation/playwright_browsers.txt``.

    Returns:
        One compact summary of the remote install output.
    """

    remote_python = f"{remote_root}/.venv/bin/python"
    browsers_path = f"{remote_root}/browser_automation/playwright_browsers.txt"
    summary_parts: list[str] = []
    if install_python_requirements:
        summary_parts.append(
            install_python_requirements_manifest(
                remote=remote,
                remote_root=remote_root,
                manifest_relpath="browser_automation/runtime_requirements.txt",
                label="browser_automation",
                progress_callback=progress_callback,
                progress_phase="browser_automation_runtime",
            )
        )
    if install_playwright_browsers:
        script = "\n".join(
            (
                f"{shlex.quote(remote_python)} - <<'PY'",
                "from __future__ import annotations",
                "import json",
                "import os",
                "from pathlib import Path",
                "import re",
                "import subprocess",
                f"remote_python = {json.dumps(remote_python)}",
                f"browsers_manifest = Path({json.dumps(browsers_path)})",
                "if not browsers_manifest.is_file() or browsers_manifest.stat().st_size <= 0:",
                "    raise RuntimeError(f'Playwright browser manifest missing or empty: {browsers_manifest}')",
                "browser_names = []",
                "allowed_name = re.compile(r'^[A-Za-z0-9._-]+$')",
                "for raw_line in browsers_manifest.read_text(encoding='utf-8').splitlines():",
                "    line = raw_line.split('#', 1)[0].strip()",
                "    if not line:",
                "        continue",
                "    browser_name = line.split()[0].strip()",
                "    if not allowed_name.fullmatch(browser_name):",
                "        raise RuntimeError(f'Invalid Playwright browser name in {browsers_manifest}: {browser_name!r}')",
                "    if browser_name not in browser_names:",
                "        browser_names.append(browser_name)",
                "if not browser_names:",
                "    raise RuntimeError('playwright browser manifest did not contain any browser names')",
                "playwright_browsers_path = os.environ.get('PLAYWRIGHT_BROWSERS_PATH') or str(Path.home() / '.cache' / 'ms-playwright')",
                "env = dict(os.environ)",
                "env['PLAYWRIGHT_BROWSERS_PATH'] = playwright_browsers_path",
                "def summarize(text: str) -> str:",
                "    normalized = '\\n'.join(line.rstrip() for line in str(text or '').splitlines() if line.strip())",
                "    if not normalized:",
                "        return ''",
                "    lines = normalized.splitlines()",
                "    if len(lines) > 12:",
                "        normalized = '\\n'.join(lines[-12:])",
                "    if len(normalized) > 1200:",
                "        normalized = normalized[-1200:]",
                "    return normalized",
                "def run(command: list[str], *, env_override: dict[str, str]) -> str:",
                "    completed = subprocess.run(",
                "        command,",
                "        check=False,",
                "        capture_output=True,",
                "        text=True,",
                "        encoding='utf-8',",
                "        errors='replace',",
                "        env=env_override,",
                "    )",
                "    if completed.returncode != 0:",
                "        detail = summarize(completed.stderr or completed.stdout or '')",
                "        rendered = ' '.join(command)",
                "        raise RuntimeError(f'{rendered} failed: {detail}')",
                "    return summarize(completed.stdout or completed.stderr or '')",
                "deps_summary = run(['sudo', remote_python, '-m', 'playwright', 'install-deps', *browser_names], env_override=env)",
                "install_summary = run([remote_python, '-m', 'playwright', 'install', *browser_names], env_override=env)",
                "print(json.dumps({",
                "    'browser_names': browser_names,",
                "    'playwright_browsers_path': playwright_browsers_path,",
                "    'deps_summary': deps_summary,",
                "    'install_summary': install_summary,",
                "}, ensure_ascii=False))",
                "PY",
            )
        )
        with progress_span(
            progress_callback,
            phase="browser_automation_runtime",
            step="install_playwright_browsers",
            detail="chromium_manifest",
        ):
            completed = remote.run_ssh(script)
        payload = json.loads((completed.stdout or "{}").strip() or "{}")
        browser_names = tuple(
            str(item).strip()
            for item in payload.get("browser_names", ())
            if str(item).strip()
        ) if isinstance(payload.get("browser_names", ()), list) else ()
        browsers_path_summary = str(payload.get("playwright_browsers_path", "")).strip()
        install_details = summarize_text(
            "\n".join(
                part
                for part in (
                    str(payload.get("deps_summary", "")).strip(),
                    str(payload.get("install_summary", "")).strip(),
                )
                if part
            )
        )
        prefix = "installed Playwright browsers"
        if browser_names:
            prefix += f" ({', '.join(browser_names)})"
        if browsers_path_summary:
            prefix += f" into {browsers_path_summary}"
        summary_parts.append(prefix if not install_details else f"{prefix}\n{install_details}")
    return summarize_text("\n".join(part for part in summary_parts if part))


def install_service_units(
    *,
    remote: PiRemoteExecutor,
    remote_root: str,
    services: Sequence[str],
) -> None:
    """Install the productive service units from the mirrored repo onto the Pi."""

    normalized_services = tuple(str(name).strip() for name in services if str(name).strip())
    if not normalized_services:
        return

    lines = [f"remote_root={shlex.quote(remote_root)}", "verify_paths=()"]
    for service_name in normalized_services:
        source_path = shlex.quote(f"{remote_root}/hardware/ops/{service_name}")
        target_path = shlex.quote(f"/etc/systemd/system/{service_name}")
        lines.extend(
            (
                f"test -f {source_path}",
                f"sudo install -m 644 {source_path} {target_path}",
                f"verify_paths+=({target_path})",
            )
        )
    services_arg = " ".join(shlex.quote(name) for name in normalized_services)
    lines.extend(
        (
            "verify_args=()",
            "if systemd-analyze --help 2>&1 | grep -q -- '--recursive-errors'; then verify_args+=(--recursive-errors=yes); fi",
            "sudo systemd-analyze verify \"${verify_args[@]}\" \"${verify_paths[@]}\"",
            "sudo systemctl daemon-reload",
            f"sudo systemctl enable {services_arg}",
        )
    )
    remote.run_ssh("\n".join(lines))


def repair_runtime_state_permissions(
    *,
    remote: PiRemoteExecutor,
    remote_root: str,
    owner_user: str,
) -> str:
    """Repair known shared-state paths so Pi services and operator tools agree."""

    lines = [
        f"remote_root={shlex.quote(remote_root)}",
        f"owner_user={shlex.quote(owner_user)}",
        "state_dir=\"$remote_root/state\"",
        f"sudo install -d -m {_STATE_DIR_MODE} -o \"$owner_user\" -g \"$owner_user\" \"$state_dir\"",
        f"repaired_paths=(\"$state_dir:{_STATE_DIR_MODE}\")",
        "repair_path_if_exists() {",
        "  local path=\"$1\"",
        "  local mode=\"$2\"",
        "  if [ ! -e \"$path\" ]; then",
        "    return 0",
        "  fi",
        "  sudo chown \"$owner_user:$owner_user\" \"$path\"",
        "  sudo chmod \"$mode\" \"$path\"",
        "  repaired_paths+=(\"$path:$mode\")",
        "}",
    ]
    for relative_path, mode in _STATE_PERMISSION_SPECS:
        lines.append(f"repair_path_if_exists \"$state_dir/{relative_path}\" {mode}")
    lines.append("printf '%s\\n' \"${repaired_paths[@]}\"")
    completed = remote.run_ssh("\n".join(lines))
    repaired_paths = tuple(line.strip() for line in (completed.stdout or "").splitlines() if line.strip())
    if not repaired_paths:
        return "verified Pi runtime state permissions; no shared-state paths required repair"
    return "repaired Pi runtime state permissions: " + ", ".join(repaired_paths)


def verify_runtime_state_permissions(
    *,
    remote: PiRemoteExecutor,
    remote_root: str,
    owner_user: str,
) -> str:
    """Verify known shared-state ownership and modes after the deploy restart."""

    lines = [
        f"remote_root={shlex.quote(remote_root)}",
        f"owner_user={shlex.quote(owner_user)}",
        "state_dir=\"$remote_root/state\"",
        "verified_paths=()",
        "if [ -d \"$state_dir\" ]; then",
        "  actual_owner=$(stat -c '%U' \"$state_dir\")",
        "  actual_group=$(stat -c '%G' \"$state_dir\")",
        "  actual_mode=$(stat -c '%a' \"$state_dir\")",
        f"  if [ \"$actual_owner\" != \"$owner_user\" ] || [ \"$actual_group\" != \"$owner_user\" ] || [ \"$actual_mode\" != \"{_STATE_DIR_MODE}\" ]; then",
        f"    echo \"$state_dir expected $owner_user:$owner_user {_STATE_DIR_MODE} but found $actual_owner:$actual_group $actual_mode\" >&2",
        "    exit 1",
        "  fi",
        f"  verified_paths+=(\"$state_dir:{_STATE_DIR_MODE}\")",
        "fi",
        "verify_path_if_exists() {",
        "  local path=\"$1\"",
        "  local mode=\"$2\"",
        "  if [ ! -e \"$path\" ]; then",
        "    return 0",
        "  fi",
        "  local actual_owner actual_group actual_mode",
        "  actual_owner=$(stat -c '%U' \"$path\")",
        "  actual_group=$(stat -c '%G' \"$path\")",
        "  actual_mode=$(stat -c '%a' \"$path\")",
        "  if [ \"$actual_owner\" != \"$owner_user\" ] || [ \"$actual_group\" != \"$owner_user\" ] || [ \"$actual_mode\" != \"$mode\" ]; then",
        "    echo \"$path expected $owner_user:$owner_user $mode but found $actual_owner:$actual_group $actual_mode\" >&2",
        "    exit 1",
        "  fi",
        "  verified_paths+=(\"$path:$mode\")",
        "}",
    ]
    for relative_path, mode in _STATE_PERMISSION_SPECS:
        lines.append(f"verify_path_if_exists \"$state_dir/{relative_path}\" {mode}")
    lines.append("printf '%s\\n' \"${verified_paths[@]}\"")
    completed = remote.run_ssh("\n".join(lines))
    verified_paths = tuple(line.strip() for line in (completed.stdout or "").splitlines() if line.strip())
    if not verified_paths:
        return "verified Pi runtime state permissions; no shared-state paths existed"
    return "verified Pi runtime state permissions: " + ", ".join(verified_paths)


def repair_ops_artifact_permissions(
    *,
    remote: PiRemoteExecutor,
    remote_root: str,
    owner_user: str,
) -> str:
    """Repair the shared ops event-store paths used by services and operator probes."""

    lines = [
        f"remote_root={shlex.quote(remote_root)}",
        f"owner_user={shlex.quote(owner_user)}",
        "ops_dir=\"$remote_root/artifacts/stores/ops\"",
        f"sudo install -d -m {_OPS_DIR_MODE} -o \"$owner_user\" -g \"$owner_user\" \"$ops_dir\"",
        f"repaired_paths=(\"$ops_dir:{_OPS_DIR_MODE}\")",
        "repair_path_if_exists() {",
        "  local path=\"$1\"",
        "  local mode=\"$2\"",
        "  if [ ! -e \"$path\" ]; then",
        "    return 0",
        "  fi",
        "  sudo chown \"$owner_user:$owner_user\" \"$path\"",
        "  sudo chmod \"$mode\" \"$path\"",
        "  repaired_paths+=(\"$path:$mode\")",
        "}",
    ]
    for relative_path, mode in _OPS_ARTIFACT_PERMISSION_SPECS:
        lines.append(f"repair_path_if_exists \"$ops_dir/{relative_path}\" {mode}")
    lines.append("printf '%s\\n' \"${repaired_paths[@]}\"")
    completed = remote.run_ssh("\n".join(lines))
    repaired_paths = tuple(line.strip() for line in (completed.stdout or "").splitlines() if line.strip())
    if not repaired_paths:
        return "verified Pi ops artifact permissions; no shared ops paths required repair"
    return "repaired Pi ops artifact permissions: " + ", ".join(repaired_paths)


def verify_ops_artifact_permissions(
    *,
    remote: PiRemoteExecutor,
    remote_root: str,
    owner_user: str,
) -> str:
    """Verify the shared ops event-store permissions after service restart."""

    lines = [
        f"remote_root={shlex.quote(remote_root)}",
        f"owner_user={shlex.quote(owner_user)}",
        "ops_dir=\"$remote_root/artifacts/stores/ops\"",
        "verified_paths=()",
        "if [ -d \"$ops_dir\" ]; then",
        "  actual_owner=$(stat -c '%U' \"$ops_dir\")",
        "  actual_group=$(stat -c '%G' \"$ops_dir\")",
        "  actual_mode=$(stat -c '%a' \"$ops_dir\")",
        f"  if [ \"$actual_owner\" != \"$owner_user\" ] || [ \"$actual_group\" != \"$owner_user\" ] || [ \"$actual_mode\" != \"{_OPS_DIR_MODE}\" ]; then",
        f"    echo \"$ops_dir expected $owner_user:$owner_user {_OPS_DIR_MODE} but found $actual_owner:$actual_group $actual_mode\" >&2",
        "    exit 1",
        "  fi",
        f"  verified_paths+=(\"$ops_dir:{_OPS_DIR_MODE}\")",
        "fi",
        "verify_mode_if_exists() {",
        "  local path=\"$1\"",
        "  local mode=\"$2\"",
        "  if [ ! -e \"$path\" ]; then",
        "    return 0",
        "  fi",
        "  local actual_mode",
        "  actual_mode=$(stat -c '%a' \"$path\")",
        "  if [ \"$actual_mode\" != \"$mode\" ]; then",
        "    echo \"$path expected mode $mode but found $actual_mode\" >&2",
        "    exit 1",
        "  fi",
        "  verified_paths+=(\"$path:$mode\")",
        "}",
    ]
    for relative_path, mode in _OPS_ARTIFACT_PERMISSION_SPECS:
        lines.append(f"verify_mode_if_exists \"$ops_dir/{relative_path}\" {mode}")
    lines.append("printf '%s\\n' \"${verified_paths[@]}\"")
    completed = remote.run_ssh("\n".join(lines))
    verified_paths = tuple(line.strip() for line in (completed.stdout or "").splitlines() if line.strip())
    if not verified_paths:
        return "verified Pi ops artifact permissions; no shared ops paths existed"
    return "verified Pi ops artifact permissions: " + ", ".join(verified_paths)


def restart_services(*, remote: PiRemoteExecutor, services: Sequence[str]) -> None:
    """Restart the selected productive services on the Pi."""

    normalized_services = tuple(str(name).strip() for name in services if str(name).strip())
    if not normalized_services:
        return
    services_arg = " ".join(shlex.quote(name) for name in normalized_services)
    remote.run_ssh(f"sudo systemctl restart {services_arg}")


def wait_for_services(
    *,
    remote: PiRemoteExecutor,
    services: Sequence[str],
    wait_timeout_s: float,
) -> tuple[PiSystemdServiceState, ...]:
    """Poll the productive services until they report healthy or time out."""

    normalized_services = tuple(str(name).strip() for name in services if str(name).strip())
    if not normalized_services:
        return ()

    deadline = time.monotonic() + wait_timeout_s
    latest_states: tuple[PiSystemdServiceState, ...] = ()
    while True:
        latest_states = load_service_states(remote=remote, services=normalized_services)
        if latest_states and all(state.healthy for state in latest_states):
            return latest_states
        if time.monotonic() >= deadline:
            break
        time.sleep(1.0)
    failing = [state for state in latest_states if not state.healthy]
    failing_names = ", ".join(state.name for state in failing) or ", ".join(normalized_services)
    journal_excerpt = load_journal_excerpt(
        remote=remote,
        service_name=failing[0].name if failing else normalized_services[0],
    )
    raise RuntimeError(
        f"services did not become healthy within {wait_timeout_s:.1f}s: {failing_names}\n{journal_excerpt}"
    )


def load_service_states(
    *,
    remote: PiRemoteExecutor,
    services: Sequence[str],
) -> tuple[PiSystemdServiceState, ...]:
    """Load the current ``systemctl show`` snapshot for the requested services."""

    normalized_services = tuple(str(name).strip() for name in services if str(name).strip())
    if not normalized_services:
        return ()

    script = "\n".join(
        (
            "python3 - <<'PY'",
            "import json",
            "import subprocess",
            f"services = {json.dumps(list(normalized_services), ensure_ascii=False)}",
            "payload = []",
            "for name in services:",
            "    completed = subprocess.run(",
            "        [",
            "            'systemctl',",
            "            'show',",
            "            name,",
            "            '--property=LoadState,ActiveState,SubState,UnitFileState,MainPID,ExecMainStatus,Result,Type',",
            "        ],",
            "        check=False,",
            "        capture_output=True,",
            "        text=True,",
            "        encoding='utf-8',",
            "        errors='replace',",
            "    )",
            "    values = {}",
            "    for raw_line in completed.stdout.splitlines():",
            "        if '=' not in raw_line:",
            "            continue",
            "        key, value = raw_line.split('=', 1)",
            "        values[key] = value",
            "    payload.append(",
            "        {",
            "            'name': name,",
            "            'load_state': values.get('LoadState', ''),",
            "            'active_state': values.get('ActiveState', ''),",
            "            'sub_state': values.get('SubState', ''),",
            "            'unit_file_state': values.get('UnitFileState', ''),",
            "            'main_pid': values.get('MainPID', ''),",
            "            'exec_main_status': values.get('ExecMainStatus', ''),",
            "            'service_result': values.get('Result', ''),",
            "            'service_type': values.get('Type', ''),",
            "        }",
            "    )",
            "print(json.dumps(payload, ensure_ascii=False))",
            "PY",
        )
    )
    completed = remote.run_ssh(script)
    raw_payload = json.loads((completed.stdout or "[]").strip() or "[]")
    states: list[PiSystemdServiceState] = []
    for item in raw_payload:
        load_state = str(item.get("load_state", "")).strip()
        active_state = str(item.get("active_state", "")).strip()
        sub_state = str(item.get("sub_state", "")).strip()
        service_result = str(item.get("service_result", "")).strip()
        service_type = str(item.get("service_type", "")).strip()
        healthy = (
            load_state == "loaded"
            and active_state == "active"
            and sub_state in _SERVICE_HEALTHY_SUBSTATES
            and service_result in {"", "success"}
            and (
                sub_state != "exited"
                or service_type == "oneshot"
            )
        )
        states.append(
            PiSystemdServiceState(
                name=str(item.get("name", "")).strip(),
                active_state=active_state,
                sub_state=sub_state,
                unit_file_state=str(item.get("unit_file_state", "")).strip(),
                main_pid=parse_optional_int(item.get("main_pid")),
                exec_main_status=parse_optional_int(item.get("exec_main_status")),
                healthy=healthy,
                load_state=load_state,
                service_type=service_type,
                service_result=service_result,
            )
        )
    return tuple(states)


def load_journal_excerpt(
    *,
    remote: PiRemoteExecutor,
    service_name: str,
    lines: int = 40,
) -> str:
    """Return a short recent journal excerpt for one failing service."""

    completed = remote.run_ssh(
        f"journalctl -u {shlex.quote(service_name)} -n {int(lines)} --no-pager --output short-iso-precise || true"
    )
    text = (completed.stdout or completed.stderr or "").strip()
    return summarize_text(text)


def run_env_contract_probe(
    *,
    remote: PiRemoteExecutor,
    remote_root: str,
    env_path: str,
    live_text: str | None,
    live_search: str | None,
) -> dict[str, object]:
    """Run the bounded Pi env-contract probe and parse its JSON result."""

    remote_python = f"{remote_root}/.venv/bin/python"
    args = [
        shlex.quote(remote_python),
        shlex.quote(f"{remote_root}/hardware/ops/check_pi_openai_env_contract.py"),
        "--env-file",
        shlex.quote(env_path),
    ]
    if live_text is not None:
        args.extend(("--live-text", shlex.quote(live_text)))
    if live_search is not None:
        args.extend(("--live-search", shlex.quote(live_search)))
    completed = remote.run_ssh(" ".join(args))
    return json.loads((completed.stdout or "").strip() or "{}")


def wait_for_remote_watchdog_ready(
    *,
    remote: PiRemoteExecutor,
    remote_root: str,
    env_path: str,
    min_sample_captured_at: str | None = None,
    wait_timeout_s: float = 180.0,
    poll_interval_s: float = 5.0,
) -> dict[str, object]:
    """Wait until the Pi watchdog publishes one fresh ready sample.

    The retention-canary retry after a backend repair must not start from a
    stale pre-repair watchdog artifact. This helper evaluates the same local
    watchdog artifact contract the runtime supervisor uses and optionally
    requires that the ready sample was captured after a specific UTC boundary.
    """

    effective_wait_timeout_s = float(wait_timeout_s)
    effective_poll_interval_s = float(poll_interval_s)
    if effective_wait_timeout_s <= 0:
        raise ValueError("wait_timeout_s must be greater than zero")
    if effective_poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be greater than zero")
    remote_python = f"{remote_root}/.venv/bin/python"
    remote_timeout_s = effective_wait_timeout_s + max(30.0, effective_poll_interval_s * 2.0)
    script = "\n".join(
        (
            "set -euo pipefail",
            "export PYTHONPATH="
            + shlex.quote(f"{remote_root}/src")
            + ':${PYTHONPATH:-""}',
            shlex.quote(remote_python) + " - <<'PY'",
            "from datetime import datetime, timezone",
            "import json",
            "import time",
            "from twinr.agent.base_agent.config import TwinrConfig",
            "from twinr.agent.workflows.required_remote_snapshot import assess_required_remote_watchdog_snapshot",
            "from twinr.ops.remote_memory_watchdog_state import RemoteMemoryWatchdogStore",
            f"env_path = {env_path!r}",
            f"min_sample_captured_at = {min_sample_captured_at!r}",
            f"wait_timeout_s = {effective_wait_timeout_s!r}",
            f"poll_interval_s = {effective_poll_interval_s!r}",
            "",
            "def _parse_utc(value: str | None) -> datetime | None:",
            "    text = str(value or '').strip()",
            "    if not text:",
            "        return None",
            "    try:",
            "        return datetime.fromisoformat(text.replace('Z', '+00:00'))",
            "    except ValueError:",
            "        return None",
            "",
            "config = TwinrConfig.from_env(env_path)",
            "store = RemoteMemoryWatchdogStore.from_config(config)",
            "gate_timestamp = _parse_utc(min_sample_captured_at)",
            "deadline = time.monotonic() + wait_timeout_s",
            "last_payload: dict[str, object] | None = None",
            "while True:",
            "    assessment = assess_required_remote_watchdog_snapshot(config, store=store)",
            "    snapshot = store.load()",
            "    sample_captured_at = None",
            "    if snapshot is not None and getattr(snapshot, 'current', None) is not None:",
            "        sample_captured_at = getattr(snapshot.current, 'captured_at', None)",
            "    parsed_sample_captured_at = _parse_utc(sample_captured_at)",
            "    sample_fresh_after_gate = gate_timestamp is None or (",
            "        parsed_sample_captured_at is not None and parsed_sample_captured_at >= gate_timestamp",
            "    )",
            "    last_payload = {",
            "        'ready': bool(assessment.ready and sample_fresh_after_gate),",
            "        'assessment_ready': bool(assessment.ready),",
            "        'detail': assessment.detail,",
            "        'artifact_path': assessment.artifact_path,",
            "        'sample_status': assessment.sample_status,",
            "        'sample_ready': assessment.sample_ready,",
            "        'sample_age_s': assessment.sample_age_s,",
            "        'sample_latency_ms': assessment.sample_latency_ms,",
            "        'sample_captured_at': sample_captured_at,",
            "        'min_sample_captured_at': min_sample_captured_at,",
            "        'sample_fresh_after_gate': sample_fresh_after_gate,",
            "        'snapshot_updated_at': assessment.snapshot_updated_at,",
            "        'snapshot_stale': assessment.snapshot_stale,",
            "        'heartbeat_age_s': assessment.heartbeat_age_s,",
            "        'heartbeat_updated_at': assessment.heartbeat_updated_at,",
            "        'probe_inflight': assessment.probe_inflight,",
            "        'probe_age_s': assessment.probe_age_s,",
            "    }",
            "    if bool(last_payload['ready']):",
            "        print(json.dumps(last_payload, ensure_ascii=False))",
            "        break",
            "    if time.monotonic() >= deadline:",
            "        print(json.dumps(last_payload, ensure_ascii=False))",
            "        break",
            "    time.sleep(max(0.1, poll_interval_s))",
            "PY",
        )
    )
    completed = remote.run_ssh(script, timeout_s=remote_timeout_s)
    payload = json.loads((completed.stdout or "").strip() or "{}")
    return payload if isinstance(payload, dict) else {}


def run_retention_canary_probe(
    *,
    remote: PiRemoteExecutor,
    remote_root: str,
    env_path: str,
    probe_id: str,
    command_timeout_s: float | None = None,
) -> dict[str, object]:
    """Run the bounded live retention canary and parse its JSON result."""

    def _load_report_payload_after_inline_stdout() -> dict[str, object] | None:
        return _wait_for_remote_retention_canary_report(
            remote=remote,
            remote_root=remote_root,
            probe_id=probe_id,
            wait_timeout_s=5.0,
            poll_interval_s=1.0,
        )

    remote_python = f"{remote_root}/.venv/bin/python"
    effective_command_timeout_s = remote.timeout_s if command_timeout_s is None else float(command_timeout_s)
    if effective_command_timeout_s <= 0:
        raise ValueError("command_timeout_s must be greater than zero")
    command = " ".join(
        (
            shlex.quote(remote_python),
            "-m",
            "twinr.memory.longterm.evaluation.live_retention_canary",
            "--env-file",
            shlex.quote(env_path),
            "--probe-id",
            shlex.quote(probe_id),
        )
    )
    try:
        completed = remote.run_ssh(
            "\n".join(
                (
                    "set -euo pipefail",
                    "cd " + shlex.quote(remote_root),
                    "export PYTHONPATH="
                    + shlex.quote(f"{remote_root}/src")
                    + ':${PYTHONPATH:-""}',
                    "set +e",
                    "retention_output=$(" + command + " 2>&1)",
                    "retention_status=$?",
                    "printf '__TWINR_RETENTION_EXIT_STATUS__=%s\\n' \"$retention_status\"",
                    "printf '%s\\n' \"$retention_output\"",
                )
            ),
            timeout_s=effective_command_timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        payload = _wait_for_remote_retention_canary_report(
            remote=remote,
            remote_root=remote_root,
            probe_id=probe_id,
        )
        if isinstance(payload, dict) and payload:
            if not bool(payload.get("ready")):
                raise RetentionCanaryProbeError(
                    "retention canary timed out waiting for stdout but completed on the Pi: "
                    + _summarize_retention_canary_failure_payload(payload),
                    payload=payload,
                ) from exc
            return payload
        raise RuntimeError(
            "retention canary timed out waiting for stdout and no final Pi report was available: "
            f"probe_id={probe_id} timeout_s={exc.timeout}"
        ) from exc
    exit_status, raw_output = _parse_remote_retention_canary_stdout(completed.stdout or "")
    try:
        payload = _parse_retention_canary_payload(raw_output)
    except json.JSONDecodeError as exc:
        report_payload = _load_report_payload_after_inline_stdout()
        if isinstance(report_payload, dict) and report_payload:
            if not bool(report_payload.get("ready")):
                raise RetentionCanaryProbeError(
                    "retention canary emitted non-JSON inline output but produced a Pi report: "
                    + _summarize_retention_canary_failure_payload(report_payload),
                    payload=report_payload,
                ) from exc
            return report_payload
        raw_detail = summarize_text(raw_output) if raw_output.strip() else "no stdout/stderr payload"
        raise RuntimeError(
            "retention canary emitted malformed inline JSON and no final Pi report was available: "
            f"probe_id={probe_id} exit_status={exit_status} detail={raw_detail} parse_error={exc}"
        ) from exc
    if payload is None:
        report_payload = _load_report_payload_after_inline_stdout()
        if isinstance(report_payload, dict) and report_payload:
            if not bool(report_payload.get("ready")):
                raise RetentionCanaryProbeError(
                    "retention canary exited without inline JSON but produced a Pi report: "
                    + _summarize_retention_canary_failure_payload(report_payload),
                    payload=report_payload,
                )
            return report_payload
        raw_detail = summarize_text(raw_output) if raw_output.strip() else "no stdout/stderr payload"
        raise RuntimeError(
            "retention canary exited without JSON output: "
            f"probe_id={probe_id} exit_status={exit_status} detail={raw_detail}"
        )
    if not bool(payload.get("ready")):
        raise RetentionCanaryProbeError(
            "retention canary failed: " + _summarize_retention_canary_failure_payload(payload),
            payload=payload,
        )
    return payload


def _parse_remote_retention_canary_stdout(stdout_text: str) -> tuple[int | None, str]:
    """Split the retention-canary wrapper marker from the raw command output."""

    marker_prefix = "__TWINR_RETENTION_EXIT_STATUS__="
    lines = stdout_text.splitlines()
    if not lines:
        return None, ""
    first_line = lines[0].strip()
    exit_status: int | None = None
    if first_line.startswith(marker_prefix):
        raw_status = first_line.removeprefix(marker_prefix).strip()
        try:
            exit_status = int(raw_status)
        except ValueError:
            exit_status = None
        return exit_status, "\n".join(lines[1:]).strip()
    return None, stdout_text.strip()


def _parse_retention_canary_payload(raw_output: str) -> dict[str, object] | None:
    """Return the structured canary payload when the remote command emitted JSON."""

    stripped = raw_output.strip()
    if not stripped:
        return None
    payload = json.loads(stripped)
    return payload if isinstance(payload, dict) else None


def _wait_for_remote_retention_canary_report(
    *,
    remote: PiRemoteExecutor,
    remote_root: str,
    probe_id: str,
    wait_timeout_s: float = 120.0,
    poll_interval_s: float = 5.0,
) -> dict[str, object] | None:
    """Poll the Pi for one completed per-probe canary report after SSH timeout."""

    deadline = time.monotonic() + max(0.0, wait_timeout_s)
    report_path = (
        Path(remote_root)
        / "artifacts"
        / "reports"
        / "retention_live_canary"
        / f"{str(probe_id).strip()}.json"
    )
    while True:
        payload = load_remote_json_artifact(remote=remote, remote_path=str(report_path))
        if isinstance(payload, dict) and payload.get("probe_id") == probe_id:
            status = str(payload.get("status") or "").strip().lower()
            if status in {"ok", "failed"}:
                return payload
        if time.monotonic() >= deadline:
            return payload if isinstance(payload, dict) else None
        time.sleep(max(0.1, poll_interval_s))


def load_remote_json_artifact(
    *,
    remote: PiRemoteExecutor,
    remote_path: str,
) -> dict[str, object] | None:
    """Return one remote JSON artifact when present."""

    script = "\n".join(
        (
            f"artifact_path={shlex.quote(remote_path)}",
            "if [ ! -f \"$artifact_path\" ]; then",
            "  printf '{}\\n'",
            "  exit 0",
            "fi",
            "cat \"$artifact_path\"",
        )
    )
    completed = remote.run_ssh(script)
    raw_text = (completed.stdout or "").strip()
    if not raw_text or raw_text == "{}":
        return None
    payload = json.loads(raw_text)
    return payload if isinstance(payload, dict) else None


def _summarize_retention_canary_failure_payload(payload: dict[str, object]) -> str:
    """Return one compact operator summary for a failed retention canary."""

    if not isinstance(payload, dict):
        return summarize_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    parts: list[str] = []
    status = str(payload.get("status") or "").strip()
    if status:
        parts.append(f"status={status}")
    failure_stage = str(payload.get("failure_stage") or "").strip()
    if failure_stage:
        parts.append(f"failure_stage={failure_stage}")
    consistency = payload.get("consistency_assessment")
    if isinstance(consistency, dict):
        relation = str(consistency.get("relation") or "").strip()
        if relation:
            parts.append(f"relation={relation}")
        summary = str(consistency.get("summary") or "").strip()
        if summary:
            parts.append(summary)
    watchdog_observations = payload.get("watchdog_observations")
    if isinstance(watchdog_observations, list) and watchdog_observations:
        latest = watchdog_observations[-1]
        if isinstance(latest, dict):
            sample_status = str(latest.get("sample_status") or "").strip()
            if sample_status:
                parts.append(f"watchdog_status={sample_status}")
            sample_ready = latest.get("sample_ready")
            if isinstance(sample_ready, bool):
                parts.append(f"watchdog_ready={'true' if sample_ready else 'false'}")
            sample_detail = str(latest.get("sample_detail") or "").strip()
            if sample_detail:
                parts.append(f"watchdog_detail={sample_detail}")
    error_message = str(payload.get("error_message") or "").strip()
    if error_message:
        parts.append(f"error={error_message}")
    if parts:
        return summarize_text(" | ".join(parts))
    return summarize_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def parse_optional_int(value: object) -> int | None:
    """Normalize systemd numeric fields to ``int | None``."""

    text = str(value or "").strip()
    if not text:
        return None
    try:
        number = int(text)
    except ValueError:
        return None
    return number if number > 0 else None


def summarize_output(completed: subprocess.CompletedProcess[str]) -> str:
    """Return one compact summary string for a completed subprocess."""

    text = "\n".join(
        part.strip()
        for part in (completed.stdout or "", completed.stderr or "")
        if part and part.strip()
    )
    return summarize_text(text)


def summarize_text(text: str, *, max_lines: int = 12, max_chars: int = 1200) -> str:
    """Collapse multi-line command output into a bounded summary string."""

    normalized = "\n".join(line.rstrip() for line in str(text or "").splitlines() if line.strip())
    if not normalized:
        return ""
    lines = normalized.splitlines()
    if len(lines) > max_lines:
        normalized = "\n".join(lines[-max_lines:])
    if len(normalized) > max_chars:
        normalized = normalized[-max_chars:]
    return normalized


def sshpass_env(password: str) -> dict[str, str]:
    """Return the local environment with ``SSHPASS`` injected."""

    env = dict(os.environ)
    env["SSHPASS"] = password
    return env


__all__ = [
    "PiPythonImportContractResult",
    "PiRemoteRepoAttestationResult",
    "PiRemoteExecutor",
    "PiSyncedFileResult",
    "PiSystemdServiceState",
    "PiVenvSystemSiteBridgeResult",
    "PiVenvScriptRepairResult",
    "RetentionCanaryProbeError",
    "attest_remote_repo_entries",
    "install_browser_automation_runtime_support",
    "install_editable_package",
    "install_python_requirements_manifest",
    "install_service_units",
    "load_journal_excerpt",
    "load_remote_json_artifact",
    "load_service_states",
    "parse_optional_int",
    "read_remote_sha256",
    "refresh_python_bytecode",
    "repair_ops_artifact_permissions",
    "repair_runtime_state_permissions",
    "restart_services",
    "run_env_contract_probe",
    "run_retention_canary_probe",
    "wait_for_remote_watchdog_ready",
    "repair_venv_python_shebangs",
    "sshpass_env",
    "summarize_output",
    "summarize_text",
    "sync_authoritative_file",
    "verify_ops_artifact_permissions",
    "verify_runtime_state_permissions",
    "verify_python_import_contract",
    "wait_for_services",
]
