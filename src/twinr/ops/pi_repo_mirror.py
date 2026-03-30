# CHANGELOG: 2026-03-30
# BUG-1: Serialized mirror cycles with local and remote locks to stop concurrent rsync --delete
#        runs from clobbering each other and causing non-deterministic remote state.
# BUG-2: Replaced checksum-on-every-watchdog-cycle defaults in run() with a hybrid metadata-first
#        loop plus periodic checksum audits, eliminating a real Pi CPU / disk-I/O bottleneck.
# BUG-3: Added remote-root readiness checks plus automatic destination creation so first-run / missing
#        target failures do not break healing cycles.
# SEC-1: Replaced StrictHostKeyChecking=no with host-key persistence and accept-new by default; changed
#        keys now fail closed instead of being silently accepted.
# SEC-2: Stopped exporting SSH passwords via SSHPASS; sshpass now receives the password via stdin (-d 0),
#        and key-based auth is preferred whenever configured.
# SEC-3: Refuse dangerous remote roots by default and reject symlinked remote_root targets to avoid
#        catastrophic delete-on-wrong-target scenarios.
# IMP-1: Added SSH connection reuse (ControlMaster/ControlPersist), UpdateHostKeys, HashKnownHosts, and
#        optional known_hosts / identity file wiring for 2026-grade noninteractive SSH.
# IMP-2: Added rsync capability detection and modern flags when supported: --delete-delay,
#        --delay-updates, --mkpath, --compress-choice=zstd, and --checksum-choice=xxh128.
# IMP-3: Added optional watchfiles-backed local event triggering with periodic drift audits to achieve a
#        lower-latency, lower-overhead frontier loop on real deployments.

"""Mirror the leading Twinr repo into the Pi acceptance checkout.

This module owns the operator workflow that keeps the authoritative
development tree in sync with the Raspberry Pi runtime checkout without
clobbering Pi-local runtime state. It treats the local repo as the only source
of truth, uses ``rsync --delete`` on the source-managed scope, and can run as a
long-lived watchdog that repeatedly heals drift.

# BREAKING:
# - Noninteractive SSH now defaults to ``StrictHostKeyChecking=accept-new``
#   instead of ``no``. New hosts are still accepted automatically, but changed
#   host keys now fail closed.
# - ``run()`` now defaults to ``checksum_always=False``. It performs
#   metadata-first probes on the fast path and periodic checksum audits via
#   ``checksum_every_s``. ``probe_once()`` still defaults to checksum=True.
# - Dangerous remote roots such as ``/`` are rejected unless
#   ``allow_risky_remote_root=True`` is explicitly set.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import importlib.util
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Iterator, Protocol, Sequence

try:
    from watchfiles import watch as _watchfiles_watch  # type: ignore[import-not-found]  # pylint: disable=import-error
except Exception:  # pragma: no cover - optional dependency
    _watchfiles_watch = None

_SubprocessRunner = Any
_SleepFn = Callable[[float], None]
_MonotonicFn = Callable[[], float]
_CycleCallback = Callable[["PiRepoMirrorCycleResult"], None]
_ErrorCallback = Callable[[Exception, int], None]
_SELF_CODING_PI_PATH = Path(__file__).resolve().with_name("self_coding_pi.py")
_SELF_CODING_PI_SPEC = importlib.util.spec_from_file_location(
    "twinr_ops_self_coding_pi_for_repo_mirror",
    _SELF_CODING_PI_PATH,
)
if _SELF_CODING_PI_SPEC is None or _SELF_CODING_PI_SPEC.loader is None:
    raise RuntimeError(f"Could not load Pi connection helper from {_SELF_CODING_PI_PATH}")
_SELF_CODING_PI_MODULE = importlib.util.module_from_spec(_SELF_CODING_PI_SPEC)
sys.modules[_SELF_CODING_PI_SPEC.name] = _SELF_CODING_PI_MODULE
_SELF_CODING_PI_SPEC.loader.exec_module(_SELF_CODING_PI_MODULE)
load_pi_connection_settings = _SELF_CODING_PI_MODULE.load_pi_connection_settings

DEFAULT_PROTECTED_PATTERNS: tuple[str, ...] = (
    "/.git/",
    "/.venv/",
    "/.env",
    "/.env.pi",
    "/.cache/",
    "/.codex/",
    "/.pytest_cache/",
    "/.mypy_cache/",
    "/.ruff_cache/",
    "/node_modules/",
    "/artifacts/",
    "/state/",
    "/src/twinr/channels/whatsapp/worker/state/",
    "/__legacy__/",
)

DEFAULT_IGNORED_PATTERNS: tuple[str, ...] = (
    "**/__pycache__/",
    "**/*.pyc",
    "**/*.pyo",
    "**/*.egg-info/",
    "**/node_modules/",
    "**/browser_automation/artifacts/",
    "/hardware/bitcraze/twinr_on_device_failsafe/build/",
)

DEFAULT_REMOTE_LOCK_WAIT_S = 30.0
_DEFAULT_CONNECT_TIMEOUT_S = 10
_DEFAULT_SERVER_ALIVE_INTERVAL_S = 15
_DEFAULT_SERVER_ALIVE_COUNT_MAX = 3
_DEFAULT_SSH_CONTROL_PERSIST_S = 60.0
_DEFAULT_NONPORTABLE_REFRESH_INTERVAL_S = 30.0
_DEFAULT_WATCH_RUST_TIMEOUT_MS = 1000
_DEFAULT_WATCH_DEBOUNCE_MS = 400
_DEFAULT_WATCH_POLL_DELAY_MS = 250

_DANGEROUS_REMOTE_ROOTS = frozenset(
    {
        "/",
        "/bin",
        "/boot",
        "/dev",
        "/etc",
        "/home",
        "/home/pi",
        "/lib",
        "/lib64",
        "/media",
        "/mnt",
        "/opt",
        "/proc",
        "/root",
        "/run",
        "/sbin",
        "/srv",
        "/sys",
        "/tmp",
        "/usr",
        "/var",
    }
)


@dataclass(frozen=True, slots=True)
class PiRepoMirrorCycleResult:
    """Summarize one mirror/watchdog cycle."""

    host: str
    remote_root: str
    drift_detected: bool
    sync_applied: bool
    checksum_used: bool
    verified_clean: bool | None
    change_count: int
    sampled_change_lines: tuple[str, ...]
    duration_s: float


@dataclass(frozen=True, slots=True)
class PiRepoMirrorEntryDigest:
    """Describe one authoritative repo entry that must match on the Pi."""

    relative_path: str
    kind: str
    sha256: str | None = None
    link_target: str | None = None


@dataclass(frozen=True, slots=True)
class PiRepoMirrorRunResult:
    """Summarize a bounded watchdog run."""

    cycles: int
    syncs_applied: int
    failures: int
    last_cycle: PiRepoMirrorCycleResult | None


@dataclass(frozen=True, slots=True)
class PiRepoMirrorCapabilities:
    """Describe the negotiated local/remote rsync feature surface."""

    local_rsync_version: str | None
    remote_rsync_version: str | None
    supports_delete_delay: bool
    supports_delay_updates: bool
    supports_mkpath: bool
    supports_checksum_choice: bool
    supports_compress_choice: bool
    supports_xxh128: bool
    supports_zstd: bool
    supports_fsync: bool


@dataclass(frozen=True, slots=True)
class _ScopeContext:
    protected_patterns: tuple[str, ...]
    protected_prefixes: tuple[str, ...]
    nonportable_paths: frozenset[str]
    nonportable_prefixes: tuple[str, ...]


class _PiConnectionSettingsLike(Protocol):
    """Describe the SSH settings shape required by the mirror watchdog."""

    @property
    def host(self) -> str: ...

    @property
    def user(self) -> str: ...

    @property
    def password(self) -> str: ...


class PiRepoMirrorWatchdog:
    """Keep the Pi runtime checkout aligned with the leading repo."""

    def __init__(
        self,
        *,
        project_root: str | Path,
        connection_settings: _PiConnectionSettingsLike,
        remote_root: str = "/twinr",
        protected_patterns: Sequence[str] = DEFAULT_PROTECTED_PATTERNS,
        timeout_s: float = 120.0,
        verify_with_checksum_on_sync: bool = True,
        known_hosts_path: str | Path | None = None,
        ssh_identity_file: str | Path | None = None,
        ssh_port: int | None = None,
        host_key_checking: str = "accept-new",
        ssh_control_persist_s: float = _DEFAULT_SSH_CONTROL_PERSIST_S,
        allow_risky_remote_root: bool = False,
        atomic_updates: bool = True,
        rsync_fsync: bool = False,
        nonportable_refresh_interval_s: float = _DEFAULT_NONPORTABLE_REFRESH_INTERVAL_S,
        subprocess_runner: _SubprocessRunner = subprocess.run,
        sleep_fn: _SleepFn = time.sleep,
        monotonic_fn: _MonotonicFn = time.monotonic,
    ) -> None:
        """Initialize one Pi mirror watchdog.

        Args:
            project_root: Leading-repo root on the development machine.
            connection_settings: SSH credentials for the Pi acceptance host.
            remote_root: Runtime checkout root on the Raspberry Pi.
            protected_patterns: Relative paths/patterns that must stay local to
                the Pi and therefore stay outside the mirrored scope.
            timeout_s: Per-subprocess timeout in seconds.
            verify_with_checksum_on_sync: When true, run a checksum-based
                dry-run verify after a healing sync changed files.
            known_hosts_path: Optional dedicated known_hosts file used by this
                watchdog. Defaults to a private cache file under the user's home.
            ssh_identity_file: Optional SSH private key. When supplied, key auth
                is preferred over password auth.
            ssh_port: Optional SSH TCP port.
            host_key_checking: SSH StrictHostKeyChecking policy. ``accept-new``
                is the safe noninteractive default.
            ssh_control_persist_s: SSH ControlPersist duration used for
                connection reuse across watchdog cycles.
            allow_risky_remote_root: Allow remote roots that are otherwise
                considered too dangerous for ``rsync --delete``.
            atomic_updates: Use safer delete/update ordering when the local and
                remote rsync support it.
            rsync_fsync: Fsync each received file when supported.
            nonportable_refresh_interval_s: How often to rescan the source tree
                for sockets/FIFOs/devices that must stay outside the mirror.
            subprocess_runner: Injectable subprocess runner for tests.
            sleep_fn: Injectable sleep function for tests.
            monotonic_fn: Injectable monotonic clock for tests.
        """

        resolved_root = Path(project_root).resolve()
        if not resolved_root.exists() or not resolved_root.is_dir():
            raise ValueError(f"project root does not exist: {resolved_root}")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be greater than zero")
        if ssh_control_persist_s < 0:
            raise ValueError("ssh_control_persist_s must be zero or greater")
        if nonportable_refresh_interval_s <= 0:
            raise ValueError("nonportable_refresh_interval_s must be greater than zero")
        self.project_root = resolved_root
        self.connection_settings = connection_settings
        self.remote_root = _normalize_remote_root(
            remote_root,
            allow_risky_remote_root=allow_risky_remote_root,
        )
        self.protected_patterns = _normalize_patterns(protected_patterns)
        self.timeout_s = timeout_s
        self.verify_with_checksum_on_sync = verify_with_checksum_on_sync
        self.atomic_updates = atomic_updates
        self.rsync_fsync = rsync_fsync
        self.nonportable_refresh_interval_s = nonportable_refresh_interval_s
        self._subprocess_runner = subprocess_runner
        self._sleep = sleep_fn
        self._monotonic = monotonic_fn

        self._state_dir = _ensure_private_dir(_default_local_state_dir())
        remote_identity = _remote_identity_string(
            user=str(self.connection_settings.user),
            host=str(self.connection_settings.host),
            port=_coerce_optional_int(
                ssh_port,
                fallback=_first_connection_setting(
                    self.connection_settings,
                    ("port", "ssh_port"),
                ),
            ),
            remote_root=self.remote_root,
        )
        self._remote_identity_hash = hashlib.sha256(remote_identity.encode("utf-8")).hexdigest()[:20]
        self._known_hosts_path = self._resolve_known_hosts_path(known_hosts_path)
        self._control_path = str(self._state_dir / f"cm-{self._remote_identity_hash}")
        self._local_lock_path = self._state_dir / f"lock-{self._remote_identity_hash}.lck"

        configured_identity = _coerce_optional_path(
            ssh_identity_file,
            fallback=_first_connection_setting(
                self.connection_settings,
                ("ssh_identity_file", "identity_file", "private_key_path", "ssh_private_key_path"),
            ),
        )
        self._ssh_identity_file = str(configured_identity) if configured_identity is not None else None
        self._ssh_port = _coerce_optional_int(
            ssh_port,
            fallback=_first_connection_setting(
                self.connection_settings,
                ("port", "ssh_port"),
            ),
        )
        configured_host_key_checking = _first_connection_setting(
            self.connection_settings,
            ("strict_host_key_checking", "host_key_checking"),
        )
        self._host_key_checking = str(
            configured_host_key_checking if configured_host_key_checking is not None else host_key_checking
        ).strip() or "accept-new"
        if self._host_key_checking not in {"yes", "ask", "accept-new", "no", "off"}:
            raise ValueError(
                "host_key_checking must be one of: yes, ask, accept-new, no, off"
            )
        self._ssh_control_persist_s = ssh_control_persist_s
        self._password = _optional_nonempty(getattr(self.connection_settings, "password", None))
        self._ssh_proxy_jump = _optional_nonempty(
            _first_connection_setting(self.connection_settings, ("proxy_jump", "jump_host"))
        )
        self._ssh_known_hosts_command = _optional_nonempty(
            _first_connection_setting(self.connection_settings, ("known_hosts_command",))
        )

        self._capabilities: PiRepoMirrorCapabilities | None = None
        self._nonportable_paths_cache: tuple[str, ...] | None = None
        self._nonportable_paths_cache_expires_at = 0.0

    @classmethod
    def from_env(
        cls,
        *,
        project_root: str | Path,
        pi_env_path: str | Path,
        remote_root: str = "/twinr",
        protected_patterns: Sequence[str] = DEFAULT_PROTECTED_PATTERNS,
        timeout_s: float = 120.0,
        verify_with_checksum_on_sync: bool = True,
        known_hosts_path: str | Path | None = None,
        ssh_identity_file: str | Path | None = None,
        ssh_port: int | None = None,
        host_key_checking: str = "accept-new",
        ssh_control_persist_s: float = _DEFAULT_SSH_CONTROL_PERSIST_S,
        allow_risky_remote_root: bool = False,
        atomic_updates: bool = True,
        rsync_fsync: bool = False,
        nonportable_refresh_interval_s: float = _DEFAULT_NONPORTABLE_REFRESH_INTERVAL_S,
        subprocess_runner: _SubprocessRunner = subprocess.run,
        sleep_fn: _SleepFn = time.sleep,
        monotonic_fn: _MonotonicFn = time.monotonic,
    ) -> "PiRepoMirrorWatchdog":
        """Build a watchdog from the shared `.env.pi` connection settings."""

        return cls(
            project_root=project_root,
            connection_settings=load_pi_connection_settings(pi_env_path),
            remote_root=remote_root,
            protected_patterns=protected_patterns,
            timeout_s=timeout_s,
            verify_with_checksum_on_sync=verify_with_checksum_on_sync,
            known_hosts_path=known_hosts_path,
            ssh_identity_file=ssh_identity_file,
            ssh_port=ssh_port,
            host_key_checking=host_key_checking,
            ssh_control_persist_s=ssh_control_persist_s,
            allow_risky_remote_root=allow_risky_remote_root,
            atomic_updates=atomic_updates,
            rsync_fsync=rsync_fsync,
            nonportable_refresh_interval_s=nonportable_refresh_interval_s,
            subprocess_runner=subprocess_runner,
            sleep_fn=sleep_fn,
            monotonic_fn=monotonic_fn,
        )

    def probe_once(
        self,
        *,
        apply_sync: bool = True,
        checksum: bool = True,
        max_change_lines: int = 40,
    ) -> PiRepoMirrorCycleResult:
        """Run one drift probe and optionally heal the Pi checkout.

        By default this compares file contents via ``rsync --checksum`` so a
        same-size, same-mtime drift cannot be misclassified as clean. When the
        source tree changes during a healing sync, the watchdog retries one
        extra sync+verify pass before reporting non-convergence.
        """

        if max_change_lines <= 0:
            raise ValueError("max_change_lines must be greater than zero")
        started = self._monotonic()
        with self._local_mirror_lock():
            change_lines = self._run_rsync(dry_run=not apply_sync, checksum=checksum)
            drift_detected = bool(change_lines)
            verified_clean: bool | None
            if not apply_sync:
                verified_clean = None
            elif not drift_detected:
                verified_clean = True
            elif self.verify_with_checksum_on_sync:
                verification_changes = self._run_rsync(dry_run=True, checksum=True)
                if verification_changes:
                    verification_changes = self._recover_from_post_sync_drift(
                        verification_changes,
                        checksum=checksum,
                    )
                if verification_changes:
                    preview = ", ".join(verification_changes[:3])
                    raise RuntimeError(
                        "Pi repo mirror did not converge after sync; remaining drift: "
                        f"{preview}"
                    )
                verified_clean = True
            else:
                verified_clean = True
        duration_s = self._monotonic() - started
        return PiRepoMirrorCycleResult(
            host=self.connection_settings.host,
            remote_root=self.remote_root,
            drift_detected=drift_detected,
            sync_applied=apply_sync and drift_detected,
            checksum_used=checksum,
            verified_clean=verified_clean,
            change_count=len(change_lines),
            sampled_change_lines=tuple(change_lines[:max_change_lines]),
            duration_s=duration_s,
        )

    def run(
        self,
        *,
        interval_s: float = 5.0,
        duration_s: float | None = None,
        checksum_every_s: float | None = 300.0,
        checksum_always: bool = False,
        apply_sync: bool = True,
        max_change_lines: int = 40,
        watch_local_changes: bool | None = None,
        watch_debounce_ms: int = _DEFAULT_WATCH_DEBOUNCE_MS,
        watch_poll_delay_ms: int = _DEFAULT_WATCH_POLL_DELAY_MS,
        on_cycle: _CycleCallback | None = None,
        on_error: _ErrorCallback | None = None,
        max_cycles: int | None = None,
    ) -> PiRepoMirrorRunResult:
        """Run the mirror loop for a bounded duration or until interrupted.

        The 2026 fast path is hybrid: local file-system changes can trigger
        immediate metadata-first probes, while checksum-based exact-content
        audits continue on the slower ``checksum_every_s`` cadence. Callers can
        restore the older behavior with ``checksum_always=True`` or disable the
        event-driven path with ``watch_local_changes=False``.
        """

        if interval_s <= 0:
            raise ValueError("interval_s must be greater than zero")
        if duration_s is not None and duration_s <= 0:
            raise ValueError("duration_s must be greater than zero when set")
        if checksum_every_s is not None and checksum_every_s <= 0:
            raise ValueError("checksum_every_s must be greater than zero when set")
        if max_cycles is not None and max_cycles <= 0:
            raise ValueError("max_cycles must be greater than zero when set")
        if watch_debounce_ms <= 0:
            raise ValueError("watch_debounce_ms must be greater than zero")
        if watch_poll_delay_ms <= 0:
            raise ValueError("watch_poll_delay_ms must be greater than zero")
        cycles = 0
        syncs_applied = 0
        failures = 0
        last_cycle: PiRepoMirrorCycleResult | None = None
        now = self._monotonic()
        deadline = None if duration_s is None else now + duration_s
        next_checksum_at = None if checksum_every_s is None else now
        use_watch = self._should_use_watch_backend(watch_local_changes)

        def execute_cycle(*, checksum: bool) -> None:
            nonlocal cycles, syncs_applied, failures, last_cycle, next_checksum_at
            try:
                cycle = self.probe_once(
                    apply_sync=apply_sync,
                    checksum=checksum,
                    max_change_lines=max_change_lines,
                )
            except Exception as exc:
                failures += 1
                if on_error is not None:
                    on_error(exc, failures)
            else:
                cycles += 1
                syncs_applied += int(cycle.sync_applied)
                last_cycle = cycle
                if on_cycle is not None:
                    on_cycle(cycle)
            if checksum and not checksum_always and checksum_every_s is not None:
                next_checksum_at = self._monotonic() + checksum_every_s

        execute_cycle(checksum=self._next_cycle_uses_checksum(checksum_always, next_checksum_at))
        if max_cycles is not None and (cycles + failures) >= max_cycles:
            return PiRepoMirrorRunResult(
                cycles=cycles,
                syncs_applied=syncs_applied,
                failures=failures,
                last_cycle=last_cycle,
            )

        if use_watch:
            last_attempt_at = self._monotonic()
            for local_event in self._iter_local_watch_events(
                interval_s=interval_s,
                deadline=deadline,
                debounce_ms=watch_debounce_ms,
                poll_delay_ms=watch_poll_delay_ms,
            ):
                if deadline is not None and self._monotonic() >= deadline:
                    break
                now = self._monotonic()
                due = local_event or now >= (last_attempt_at + interval_s)
                if not due:
                    continue
                execute_cycle(checksum=self._next_cycle_uses_checksum(checksum_always, next_checksum_at))
                last_attempt_at = self._monotonic()
                if max_cycles is not None and (cycles + failures) >= max_cycles:
                    break
        else:
            while True:
                if deadline is not None and self._monotonic() >= deadline:
                    break
                self._sleep(interval_s)
                execute_cycle(checksum=self._next_cycle_uses_checksum(checksum_always, next_checksum_at))
                if max_cycles is not None and (cycles + failures) >= max_cycles:
                    break

        return PiRepoMirrorRunResult(
            cycles=cycles,
            syncs_applied=syncs_applied,
            failures=failures,
            last_cycle=last_cycle,
        )

    def _recover_from_post_sync_drift(
        self,
        verification_changes: list[str],
        *,
        checksum: bool,
    ) -> list[str]:
        """Try bounded recovery before failing a post-sync verification.

        First handle the known cache-only delete case. If drift still remains,
        run one extra sync+verify pass so a shared-worktree source update that
        landed between the first sync and the checksum audit can converge
        without making operators rerun the whole deploy manually.
        """

        remaining_changes = self._recover_from_cache_only_directory_drift(
            verification_changes,
            checksum=checksum,
        )
        if not remaining_changes:
            return remaining_changes
        self._run_rsync(dry_run=False, checksum=checksum)
        return self._run_rsync(dry_run=True, checksum=True)

    def _recover_from_cache_only_directory_drift(
        self,
        verification_changes: list[str],
        *,
        checksum: bool,
    ) -> list[str]:
        """Retry once when remaining delete drift is blocked by ignored caches."""

        stale_directories = _stale_directory_deletion_targets(verification_changes)
        if not stale_directories:
            return verification_changes
        self._prune_remote_cache_only_directories(stale_directories)
        self._run_rsync(dry_run=False, checksum=checksum)
        return self._run_rsync(dry_run=True, checksum=True)

    def _run_rsync(self, *, dry_run: bool, checksum: bool) -> list[str]:
        args = self._build_rsync_args(dry_run=dry_run, checksum=checksum)
        completed = self._run_process(args, use_sshpass=self._uses_sshpass())
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "").strip()
            if not message:
                message = "rsync mirror command failed"
            raise RuntimeError(message)
        return _parse_rsync_change_lines(completed.stdout)

    def _build_rsync_args(self, *, dry_run: bool, checksum: bool) -> list[str]:
        capabilities = self._get_capabilities()
        args = [
            "rsync",
            "-az",
            "--no-specials",
            "--no-devices",
            "--delete",
            "--itemize-changes",
            "--out-format=%i %n%L",
            "-e",
            shlex.join(self._ssh_base_command()),
            "--rsync-path",
            self._build_rsync_path_wrapper(),
        ]
        if self.atomic_updates and capabilities.supports_delete_delay:
            args.append("--delete-delay")
        if self.atomic_updates and capabilities.supports_delay_updates:
            args.append("--delay-updates")
        if capabilities.supports_mkpath:
            args.append("--mkpath")
        if capabilities.supports_checksum_choice and capabilities.supports_xxh128:
            args.append("--checksum-choice=xxh128,xxh128")
        if capabilities.supports_compress_choice and capabilities.supports_zstd:
            args.append("--compress-choice=zstd")
        if self.rsync_fsync and capabilities.supports_fsync:
            args.append("--fsync")
        # Python bytecode, nested node dependency trees, and browser-automation
        # captures are local runtime or build artefacts. Keeping them outside
        # the authoritative mirror avoids sync failures when productive
        # processes or local SDK installs mutate those trees independently of
        # the source-managed repo content.
        for pattern in DEFAULT_IGNORED_PATTERNS:
            args.append(f"--exclude={pattern}")
        # Repo mirroring should only carry ordinary files, symlinks, and
        # directories. Transient local FIFOs/devices from tools or root-owned
        # helpers must not enter the Pi checkout or block sync cycles.
        for path in self._get_nonportable_paths():
            args.append(f"--exclude=/{path}")
        if dry_run:
            args.append("--dry-run")
        if checksum:
            args.append("--checksum")
        # Perishable filters keep Pi-local runtime paths at the remote root
        # without pinning unrelated deleted directory trees in place.
        for pattern in self.protected_patterns:
            args.append(f"--filter=-p {pattern}")
        args.append(f"{self.project_root}/")
        args.append(f"{self._remote_destination()}:{self.remote_root}/")
        return args

    def _build_rsync_path_wrapper(self) -> str:
        """Return a remote rsync wrapper that serializes access and validates the target."""

        quoted_root = shlex.quote(self.remote_root)
        quoted_lock_key = shlex.quote(self._remote_identity_hash)
        max_attempts = max(1, int(DEFAULT_REMOTE_LOCK_WAIT_S / 0.2))
        script = (
            "set -eu; "
            f'root={quoted_root}; '
            'case "$root" in ""|"/") echo "unsafe remote_root" >&2; exit 97 ;; esac; '
            'if [ -e "$root" ] && [ -L "$root" ]; then echo "remote_root must not be a symlink" >&2; exit 98; fi; '
            'mkdir -p -- "$root"; '
            'if [ ! -d "$root" ]; then echo "remote_root is not a directory" >&2; exit 99; fi; '
            'lock_base="$HOME/.cache/twinr_repo_mirror/locks"; '
            f'lock_dir="$lock_base/{quoted_lock_key}.lock"; '
            'mkdir -p -- "$lock_base"; '
            f'max_attempts="{max_attempts}"; '
            'attempt=0; '
            'while ! mkdir "$lock_dir" 2>/dev/null; do '
            'attempt=$((attempt + 1)); '
            'if [ "$attempt" -ge "$max_attempts" ]; then '
            'echo "timeout waiting for remote mirror lock: $lock_dir" >&2; exit 75; '
            'fi; '
            'sleep 0.2; '
            'done; '
            'cleanup(){ rmdir "$lock_dir" >/dev/null 2>&1 || true; }; '
            'trap cleanup EXIT INT TERM; '
            'rsync "$@"; '
            'status=$?; '
            'exit "$status"'
        )
        return "sh -c " + shlex.quote(script) + " sh"

    def _get_capabilities(self) -> PiRepoMirrorCapabilities:
        if self._capabilities is None:
            self._capabilities = self._discover_capabilities()
        return self._capabilities

    def _discover_capabilities(self) -> PiRepoMirrorCapabilities:
        local_version_text = ""
        local_help_text = ""
        remote_version_text = ""
        remote_help_text = ""
        local_version = None
        remote_version = None
        try:
            local_version_completed = self._run_process(["rsync", "--version"], use_sshpass=False)
            if local_version_completed.returncode == 0:
                local_version_text = local_version_completed.stdout
                local_version = _parse_rsync_version(local_version_text)
            local_help_completed = self._run_process(["rsync", "--help"], use_sshpass=False)
            if local_help_completed.returncode == 0:
                local_help_text = local_help_completed.stdout
        except Exception:
            pass
        try:
            remote_version_completed = self._run_remote_command(["sh", "-lc", "rsync --version"])
            if remote_version_completed.returncode == 0:
                remote_version_text = remote_version_completed.stdout
                remote_version = _parse_rsync_version(remote_version_text)
            remote_help_completed = self._run_remote_command(["sh", "-lc", "rsync --help"])
            if remote_help_completed.returncode == 0:
                remote_help_text = remote_help_completed.stdout
        except Exception:
            pass
        local_blob = (local_version_text + "\n" + local_help_text).lower()
        remote_blob = (remote_version_text + "\n" + remote_help_text).lower()
        return PiRepoMirrorCapabilities(
            local_rsync_version=local_version,
            remote_rsync_version=remote_version,
            supports_delete_delay="--delete-delay" in local_help_text and "--delete-delay" in remote_help_text,
            supports_delay_updates="--delay-updates" in local_help_text and "--delay-updates" in remote_help_text,
            supports_mkpath="--mkpath" in local_help_text and "--mkpath" in remote_help_text,
            supports_checksum_choice="--checksum-choice" in local_help_text and "--checksum-choice" in remote_help_text,
            supports_compress_choice="--compress-choice" in local_help_text and "--compress-choice" in remote_help_text,
            supports_xxh128="xxh128" in local_blob and "xxh128" in remote_blob,
            supports_zstd="zstd" in local_blob and "zstd" in remote_blob,
            supports_fsync="--fsync" in local_help_text and "--fsync" in remote_help_text,
        )

    def _prune_remote_cache_only_directories(self, relative_paths: Sequence[str]) -> None:
        """Remove remote stale directories when they contain only Python caches."""

        normalized_paths = tuple(
            dict.fromkeys(path.strip("/").strip() for path in relative_paths if path.strip("/"))
        )
        if not normalized_paths:
            return
        remote_targets = " ".join(
            shlex.quote(posixpath.join(self.remote_root, path))
            for path in normalized_paths
        )
        script = f"""
set -eu
for target in {remote_targets}; do
  [ -e "$target" ] || continue
  if [ -L "$target" ]; then
    continue
  fi
  if find "$target" -mindepth 1 ! \\( -type d -name '__pycache__' -o -type f \\( -name '*.pyc' -o -name '*.pyo' \\) \\) -print -quit | grep -q .; then
    continue
  fi
  rm -rf -- "$target"
done
"""
        completed = self._run_remote_locked_shell(script)
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "").strip()
            if not message:
                message = "remote stale-directory prune failed"
            raise RuntimeError(message)

    def _run_remote_locked_shell(self, script: str) -> subprocess.CompletedProcess[str]:
        quoted_lock_key = shlex.quote(self._remote_identity_hash)
        lock_script = (
            "set -eu; "
            'lock_base="$HOME/.cache/twinr_repo_mirror/locks"; '
            f'lock_dir="$lock_base/{quoted_lock_key}.lock"; '
            'mkdir -p -- "$lock_base"; '
            'attempt=0; '
            'max_attempts=150; '
            'while ! mkdir "$lock_dir" 2>/dev/null; do '
            'attempt=$((attempt + 1)); '
            'if [ "$attempt" -ge "$max_attempts" ]; then '
            'echo "timeout waiting for remote mirror lock: $lock_dir" >&2; exit 75; '
            'fi; '
            'sleep 0.2; '
            'done; '
            'cleanup(){ rmdir "$lock_dir" >/dev/null 2>&1 || true; }; '
            'trap cleanup EXIT INT TERM; '
            + str(script).strip()
        )
        return self._run_remote_command(["sh", "-lc", lock_script])

    def _run_remote_command(self, remote_args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        args = [*self._ssh_base_command(), self._remote_destination(), *remote_args]
        return self._run_process(args, use_sshpass=self._uses_sshpass())

    def _run_process(
        self,
        args: Sequence[str],
        *,
        use_sshpass: bool,
    ) -> subprocess.CompletedProcess[str]:
        command = list(args)
        input_text: str | None = None
        if use_sshpass:
            if shutil.which("sshpass") is None:
                raise RuntimeError("sshpass is required for password-based Pi mirroring")
            if self._password is None:
                raise RuntimeError("password-based Pi mirroring requested without a password")
            command = ["sshpass", "-d", "0", *command]
            input_text = self._password + "\n"
        return self._subprocess_runner(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            input=input_text,
            timeout=self.timeout_s,
            cwd=str(self.project_root),
        )

    def _ssh_base_command(self) -> list[str]:
        args = [
            "ssh",
            "-o",
            f"StrictHostKeyChecking={self._host_key_checking}",
            "-o",
            f"UserKnownHostsFile={self._known_hosts_path}",
            "-o",
            "UpdateHostKeys=yes",
            "-o",
            "HashKnownHosts=yes",
            "-o",
            f"ConnectTimeout={_DEFAULT_CONNECT_TIMEOUT_S}",
            "-o",
            f"ServerAliveInterval={_DEFAULT_SERVER_ALIVE_INTERVAL_S}",
            "-o",
            f"ServerAliveCountMax={_DEFAULT_SERVER_ALIVE_COUNT_MAX}",
            "-o",
            f"BatchMode={'no' if self._uses_sshpass() else 'yes'}",
            "-o",
            "ControlMaster=auto",
            "-o",
            f"ControlPath={self._control_path}",
            "-o",
            f"ControlPersist={'no' if self._ssh_control_persist_s <= 0 else f'{int(self._ssh_control_persist_s)}s'}",
            "-o",
            "LogLevel=ERROR",
        ]
        if self._ssh_port is not None:
            args.extend(["-p", str(self._ssh_port)])
        if self._ssh_identity_file is not None:
            args.extend(["-i", self._ssh_identity_file])
        if self._ssh_proxy_jump is not None:
            args.extend(["-J", self._ssh_proxy_jump])
        if self._ssh_known_hosts_command is not None:
            args.extend(["-o", f"KnownHostsCommand={self._ssh_known_hosts_command}"])
        return args

    def _remote_destination(self) -> str:
        return f"{self.connection_settings.user}@{self.connection_settings.host}"

    def _resolve_known_hosts_path(self, known_hosts_path: str | Path | None) -> str:
        configured = _coerce_optional_path(
            known_hosts_path,
            fallback=_first_connection_setting(
                self.connection_settings,
                ("known_hosts_path", "ssh_known_hosts_path"),
            ),
        )
        if configured is None:
            configured = self._state_dir / "known_hosts"
        configured.parent.mkdir(parents=True, exist_ok=True)
        return str(configured)

    def _uses_sshpass(self) -> bool:
        return self._ssh_identity_file is None and self._password is not None

    def _get_nonportable_paths(self) -> tuple[str, ...]:
        now = self._monotonic()
        if (
            self._nonportable_paths_cache is None
            or now >= self._nonportable_paths_cache_expires_at
        ):
            self._nonportable_paths_cache = _discover_nonportable_paths(self.project_root)
            self._nonportable_paths_cache_expires_at = now + self.nonportable_refresh_interval_s
        return self._nonportable_paths_cache

    def _invalidate_nonportable_paths_cache(self) -> None:
        self._nonportable_paths_cache = None
        self._nonportable_paths_cache_expires_at = 0.0

    @contextmanager
    def _local_mirror_lock(self) -> Iterator[None]:
        self._local_lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._local_lock_path.open("a+", encoding="utf-8") as handle:
            start = self._monotonic()
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if (self._monotonic() - start) >= self.timeout_s:
                        raise TimeoutError(
                            f"timed out waiting for local mirror lock: {self._local_lock_path}"
                        )
                    self._sleep(0.2)
            handle.seek(0)
            handle.truncate()
            handle.write(f"{os.getpid()}\n")
            handle.flush()
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _should_use_watch_backend(self, watch_local_changes: bool | None) -> bool:
        if watch_local_changes is False:
            return False
        if _watchfiles_watch is None:
            return False
        return True

    def _iter_local_watch_events(
        self,
        *,
        interval_s: float,
        deadline: float | None,
        debounce_ms: int,
        poll_delay_ms: int,
    ) -> Iterator[bool]:
        if _watchfiles_watch is None:
            while True:
                if deadline is not None and self._monotonic() >= deadline:
                    return
                self._sleep(interval_s)
                yield False
        stop_event = None
        timeout_ms = min(_DEFAULT_WATCH_RUST_TIMEOUT_MS, max(50, int(interval_s * 1000)))
        for changes in _watchfiles_watch(
            self.project_root,
            watch_filter=self._watch_filter,
            debounce=debounce_ms,
            rust_timeout=timeout_ms,
            yield_on_timeout=True,
            force_polling=None,
            poll_delay_ms=poll_delay_ms,
            recursive=True,
            stop_event=stop_event,
            ignore_permission_denied=True,
            raise_interrupt=True,
        ):
            if deadline is not None and self._monotonic() >= deadline:
                return
            local_event = bool(changes)
            if local_event:
                self._invalidate_nonportable_paths_cache()
            yield local_event

    def _watch_filter(self, _change: Any, path: str) -> bool:
        try:
            relative_path = Path(path).resolve().relative_to(self.project_root).as_posix()
        except Exception:
            return False
        scope = _build_scope_context(self.protected_patterns, self._get_nonportable_paths())
        return not _path_is_excluded_from_authoritative_scope(
            relative_path=relative_path,
            is_dir=False,
            scope=scope,
        )

    def _next_cycle_uses_checksum(
        self,
        checksum_always: bool,
        next_checksum_at: float | None,
    ) -> bool:
        if checksum_always:
            return True
        if next_checksum_at is None:
            return False
        return self._monotonic() >= next_checksum_at


def _normalize_patterns(patterns: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_pattern in patterns:
        pattern = str(raw_pattern).strip()
        if not pattern or pattern in seen:
            continue
        normalized.append(pattern)
        seen.add(pattern)
    if not normalized:
        raise ValueError("at least one protected pattern is required")
    return tuple(normalized)


def build_authoritative_repo_entry_digests(
    project_root: str | Path,
    *,
    protected_patterns: Sequence[str] = DEFAULT_PROTECTED_PATTERNS,
) -> tuple[PiRepoMirrorEntryDigest, ...]:
    """Return the authoritative mirrored repo entries with stable digests.

    The returned entries follow the same high-level authority contract as the
    Pi repo mirror: protected Pi-local runtime paths stay out, transient local
    artefacts stay out, and regular files plus symlinks in the mirrored scope
    are attested by content digest or symlink target.
    """

    resolved_root = Path(project_root).resolve()
    if not resolved_root.exists() or not resolved_root.is_dir():
        raise ValueError(f"project root does not exist: {resolved_root}")
    scope = _build_scope_context(
        _normalize_patterns(protected_patterns),
        _discover_nonportable_paths(resolved_root),
    )
    entries: list[PiRepoMirrorEntryDigest] = []
    for current_root, dirnames, filenames in os.walk(resolved_root, topdown=True, followlinks=False):
        current_root_path = Path(current_root)
        retained_dirnames: list[str] = []
        for dirname in dirnames:
            candidate = current_root_path / dirname
            relative_path = candidate.relative_to(resolved_root).as_posix()
            if _path_is_excluded_from_authoritative_scope(
                relative_path=relative_path,
                is_dir=True,
                scope=scope,
            ):
                continue
            if candidate.is_symlink():
                entries.append(_build_repo_entry_digest(candidate, resolved_root))
                continue
            retained_dirnames.append(dirname)
        dirnames[:] = retained_dirnames
        for filename in filenames:
            candidate = current_root_path / filename
            relative_path = candidate.relative_to(resolved_root).as_posix()
            if _path_is_excluded_from_authoritative_scope(
                relative_path=relative_path,
                is_dir=False,
                scope=scope,
            ):
                continue
            entries.append(_build_repo_entry_digest(candidate, resolved_root))
    entries.sort(key=lambda item: item.relative_path)
    return tuple(entries)


def _build_repo_entry_digest(path: Path, project_root: Path) -> PiRepoMirrorEntryDigest:
    """Build one attestation digest for a mirrored file or symlink."""

    relative_path = path.relative_to(project_root).as_posix()
    if path.is_symlink():
        return PiRepoMirrorEntryDigest(
            relative_path=relative_path,
            kind="symlink",
            link_target=os.readlink(path),
        )
    return PiRepoMirrorEntryDigest(
        relative_path=relative_path,
        kind="file",
        sha256=_sha256_for_file(path),
    )


def _sha256_for_file(path: Path) -> str:
    """Return the SHA256 digest for one local file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _build_scope_context(
    protected_patterns: Sequence[str],
    nonportable_paths: Sequence[str],
) -> _ScopeContext:
    return _ScopeContext(
        protected_patterns=tuple(protected_patterns),
        protected_prefixes=_protected_root_prefixes(protected_patterns),
        nonportable_paths=frozenset(
            str(path or "").strip().strip("/")
            for path in nonportable_paths
            if str(path or "").strip().strip("/")
        ),
        nonportable_prefixes=_directory_prefixes(nonportable_paths),
    )


def _path_is_excluded_from_authoritative_scope(
    *,
    relative_path: str,
    is_dir: bool,
    scope: _ScopeContext,
) -> bool:
    """Return whether one repo-relative path stays outside Pi source authority."""

    normalized_path = str(relative_path or "").strip().strip("/")
    if not normalized_path:
        return False
    if normalized_path in scope.nonportable_paths or any(
        normalized_path.startswith(prefix)
        for prefix in scope.nonportable_prefixes
    ):
        return True
    if _path_matches_root_prefixes(normalized_path, scope.protected_prefixes):
        return True
    parts = tuple(part for part in normalized_path.split("/") if part)
    if "__pycache__" in parts:
        return True
    if any(part == "node_modules" for part in parts):
        return True
    if any(part.endswith(".egg-info") for part in parts):
        return True
    if _contains_segment_pair(parts, "browser_automation", "artifacts"):
        return True
    if _path_matches_root_prefixes(
        normalized_path,
        ("hardware/bitcraze/twinr_on_device_failsafe/build",),
    ):
        return True
    if not is_dir and (normalized_path.endswith(".pyc") or normalized_path.endswith(".pyo")):
        return True
    return False


def _directory_prefixes(paths: Sequence[str]) -> tuple[str, ...]:
    """Return directory-style prefixes for exact nonportable-path exclusions."""

    prefixes = []
    for raw_path in paths:
        normalized = str(raw_path or "").strip().strip("/")
        if not normalized:
            continue
        prefixes.append(f"{normalized}/")
    return tuple(prefixes)


def _protected_root_prefixes(patterns: Sequence[str]) -> tuple[str, ...]:
    """Normalize root-anchored preserve rules into relative path prefixes."""

    prefixes: list[str] = []
    for raw_pattern in patterns:
        normalized = str(raw_pattern or "").strip()
        if not normalized.startswith("/"):
            continue
        candidate = normalized.lstrip("/").rstrip("/")
        if not candidate:
            continue
        prefixes.append(candidate)
    return tuple(dict.fromkeys(prefixes))


def _path_matches_root_prefixes(path: str, prefixes: Sequence[str]) -> bool:
    """Return whether one relative path falls under any root-anchored prefix."""

    normalized_path = str(path or "").strip().strip("/")
    return any(
        normalized_path == prefix or normalized_path.startswith(f"{prefix}/")
        for prefix in prefixes
    )


def _contains_segment_pair(parts: Sequence[str], first: str, second: str) -> bool:
    """Return whether one path contains two adjacent directory segments."""

    if len(parts) < 2:
        return False
    for index in range(len(parts) - 1):
        if parts[index] == first and parts[index + 1] == second:
            return True
    return False


def _discover_nonportable_paths(project_root: Path) -> tuple[str, ...]:
    """Return repo-relative paths that rsync must ignore completely.

    Rsync's ``--no-specials``/``--no-devices`` is not sufficient when local
    helper processes leave behind FIFOs or sockets inside the leading repo.
    Excluding them explicitly keeps the mirror from touching matching paths on
    the Pi at all, which avoids receiver-side metadata failures.
    """

    excluded: list[str] = []
    for root, dirnames, filenames in os.walk(project_root):
        root_path = Path(root)
        dirnames[:] = [
            name
            for name in dirnames
            if not _is_nonportable_path(root_path / name)
        ]
        for name in filenames:
            candidate = root_path / name
            if _is_nonportable_path(candidate):
                excluded.append(candidate.relative_to(project_root).as_posix())
    return tuple(sorted(excluded))


def _is_nonportable_path(path: Path) -> bool:
    """Return true for repo entries that should never be mirrored."""

    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return False
    return stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode) or stat.S_ISBLK(mode) or stat.S_ISCHR(mode)


def _parse_rsync_change_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("*deleting "):
            lines.append(line)
            continue
        if len(line) > 11 and line[11] == " ":
            lines.append(line)
    return lines


def _stale_directory_deletion_targets(change_lines: Sequence[str]) -> tuple[str, ...]:
    """Return delete-only directory drift candidates from rsync itemized lines."""

    targets: list[str] = []
    for raw_line in change_lines:
        line = str(raw_line or "").strip()
        if not line.startswith("*deleting"):
            return ()
        path = line.removeprefix("*deleting").strip()
        if not path.endswith("/"):
            return ()
        targets.append(path.rstrip("/"))
    return tuple(targets)


def _parse_rsync_version(text: str) -> str | None:
    match = re.search(r"rsync\s+version\s+([0-9]+\.[0-9]+\.[0-9]+)", str(text or ""), flags=re.IGNORECASE)
    if match is None:
        return None
    return match.group(1)


def _normalize_remote_root(remote_root: str, *, allow_risky_remote_root: bool) -> str:
    raw = str(remote_root or "").strip()
    if not raw:
        raise ValueError("remote_root must not be empty")
    pure = PurePosixPath(raw)
    if not pure.is_absolute():
        raise ValueError("remote_root must be an absolute POSIX path")
    normalized = posixpath.normpath(str(pure))
    if normalized == "/":
        raise ValueError("remote_root must not be '/'")
    if not allow_risky_remote_root and normalized in _DANGEROUS_REMOTE_ROOTS:
        raise ValueError(
            f"remote_root is too dangerous for rsync --delete: {normalized!r}; "
            "set allow_risky_remote_root=True to override"
        )
    if any(part == ".." for part in pure.parts):
        raise ValueError("remote_root must not contain '..'")
    return normalized


def _default_local_state_dir() -> Path:
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()
    return cache_home / "twinr_repo_mirror"


def _ensure_private_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    return path


def _remote_identity_string(*, user: str, host: str, port: int | None, remote_root: str) -> str:
    port_part = "" if port is None else f":{port}"
    return f"{user}@{host}{port_part}:{remote_root}"


def _coerce_optional_path(value: object, *, fallback: object | None = None) -> Path | None:
    candidate = value if value not in (None, "") else fallback
    if candidate in (None, ""):
        return None
    return Path(str(candidate)).expanduser()


def _coerce_optional_int(value: object, *, fallback: object | None = None) -> int | None:
    candidate = value if value not in (None, "") else fallback
    if candidate in (None, ""):
        return None
    return int(candidate)


def _first_connection_setting(connection_settings: object, names: Sequence[str]) -> object | None:
    for name in names:
        if not hasattr(connection_settings, name):
            continue
        value = getattr(connection_settings, name)
        if value not in (None, ""):
            return value
    return None


def _optional_nonempty(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "DEFAULT_PROTECTED_PATTERNS",
    "PiRepoMirrorCapabilities",
    "PiRepoMirrorEntryDigest",
    "PiRepoMirrorCycleResult",
    "PiRepoMirrorRunResult",
    "PiRepoMirrorWatchdog",
    "build_authoritative_repo_entry_digests",
]
