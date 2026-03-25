"""Mirror the leading Twinr repo into the Pi acceptance checkout.

This module owns the operator workflow that keeps the authoritative
development tree in sync with the Raspberry Pi runtime checkout without
clobbering Pi-local runtime state. It treats the local repo as the only source
of truth, uses ``rsync --delete`` on the source-managed scope, and can run as a
long-lived watchdog that repeatedly heals drift.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Protocol, Sequence

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
    "**/node_modules/",
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
class PiRepoMirrorRunResult:
    """Summarize a bounded watchdog run."""

    cycles: int
    syncs_applied: int
    failures: int
    last_cycle: PiRepoMirrorCycleResult | None


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
            subprocess_runner: Injectable subprocess runner for tests.
            sleep_fn: Injectable sleep function for tests.
            monotonic_fn: Injectable monotonic clock for tests.
        """

        resolved_root = Path(project_root).resolve()
        if not resolved_root.exists() or not resolved_root.is_dir():
            raise ValueError(f"project root does not exist: {resolved_root}")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be greater than zero")
        self.project_root = resolved_root
        self.connection_settings = connection_settings
        self.remote_root = remote_root.rstrip("/") or "/"
        self.protected_patterns = _normalize_patterns(protected_patterns)
        self.timeout_s = timeout_s
        self.verify_with_checksum_on_sync = verify_with_checksum_on_sync
        self._subprocess_runner = subprocess_runner
        self._sleep = sleep_fn
        self._monotonic = monotonic_fn

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
        same-size, same-mtime drift cannot be misclassified as clean.
        """

        if max_change_lines <= 0:
            raise ValueError("max_change_lines must be greater than zero")
        started = self._monotonic()
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
                verification_changes = self._recover_from_cache_only_directory_drift(
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
        checksum_always: bool = True,
        apply_sync: bool = True,
        max_change_lines: int = 40,
        on_cycle: _CycleCallback | None = None,
        on_error: _ErrorCallback | None = None,
        max_cycles: int | None = None,
    ) -> PiRepoMirrorRunResult:
        """Run the mirror loop for a bounded duration or until interrupted.

        Exact-content checks are the default on every cycle. ``checksum_every_s``
        only matters when callers explicitly opt into metadata-only probes.
        """

        if interval_s <= 0:
            raise ValueError("interval_s must be greater than zero")
        if duration_s is not None and duration_s <= 0:
            raise ValueError("duration_s must be greater than zero when set")
        if checksum_every_s is not None and checksum_every_s <= 0:
            raise ValueError("checksum_every_s must be greater than zero when set")
        if max_cycles is not None and max_cycles <= 0:
            raise ValueError("max_cycles must be greater than zero when set")
        cycles = 0
        syncs_applied = 0
        failures = 0
        last_cycle: PiRepoMirrorCycleResult | None = None
        deadline = None if duration_s is None else self._monotonic() + duration_s
        next_checksum_at = None if checksum_every_s is None else self._monotonic() + checksum_every_s
        while True:
            checksum = checksum_always or (
                next_checksum_at is not None and self._monotonic() >= next_checksum_at
            )
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
            if max_cycles is not None and (cycles + failures) >= max_cycles:
                break
            if deadline is not None and self._monotonic() >= deadline:
                break
            self._sleep(interval_s)
        return PiRepoMirrorRunResult(
            cycles=cycles,
            syncs_applied=syncs_applied,
            failures=failures,
            last_cycle=last_cycle,
        )

    def _run_rsync(self, *, dry_run: bool, checksum: bool) -> list[str]:
        completed = self._subprocess_runner(
            self._build_rsync_args(dry_run=dry_run, checksum=checksum),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            env=_sshpass_env(self.connection_settings.password),
            timeout=self.timeout_s,
            cwd=str(self.project_root),
        )
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "").strip()
            if not message:
                message = "rsync mirror command failed"
            raise RuntimeError(message)
        return _parse_rsync_change_lines(completed.stdout)

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

    def _prune_remote_cache_only_directories(self, relative_paths: Sequence[str]) -> None:
        """Remove remote stale directories when they contain only Python caches."""

        normalized_paths = tuple(
            dict.fromkeys(path.strip("/").strip() for path in relative_paths if path.strip("/"))
        )
        if not normalized_paths:
            return
        remote_targets = " ".join(
            shlex.quote(f"{self.remote_root}/{path}")
            for path in normalized_paths
        )
        script = f"""
for target in {remote_targets}; do
  [ -e "$target" ] || continue
  if find "$target" -mindepth 1 ! \\( -type d -name '__pycache__' -o -type f \\( -name '*.pyc' -o -name '*.pyo' \\) \\) -print -quit | grep -q .; then
    continue
  fi
  rm -rf "$target" || true
done
"""
        completed = self._subprocess_runner(
            [
                "sshpass",
                "-e",
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "ConnectTimeout=10",
                f"{self.connection_settings.user}@{self.connection_settings.host}",
                "bash -lc " + shlex.quote("set -euo pipefail; " + script),
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            env=_sshpass_env(self.connection_settings.password),
            timeout=self.timeout_s,
            cwd=str(self.project_root),
        )
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "").strip()
            if not message:
                message = "remote stale-directory prune failed"
            raise RuntimeError(message)

    def _build_rsync_args(self, *, dry_run: bool, checksum: bool) -> list[str]:
        args = [
            "sshpass",
            "-e",
            "rsync",
            "-az",
            "--no-specials",
            "--no-devices",
            "--delete",
            "--itemize-changes",
            "--out-format=%i %n%L",
            "-e",
            "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10",
        ]
        # Python bytecode and nested node dependency trees are local runtime or
        # build artefacts. Keeping them outside the authoritative mirror avoids
        # sync failures when productive processes or local SDK installs mutate
        # those trees independently of the source-managed repo content.
        for pattern in DEFAULT_IGNORED_PATTERNS:
            args.append(f"--exclude={pattern}")
        # Repo mirroring should only carry ordinary files, symlinks, and
        # directories. Transient local FIFOs/devices from tools or root-owned
        # helpers must not enter the Pi checkout or block sync cycles.
        for path in _discover_nonportable_paths(self.project_root):
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
        args.append(
            f"{self.connection_settings.user}@{self.connection_settings.host}:{self.remote_root}/"
        )
        return args


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


def _sshpass_env(password: str) -> dict[str, str]:
    env = dict(os.environ)
    env["SSHPASS"] = password
    return env


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


__all__ = [
    "DEFAULT_PROTECTED_PATTERNS",
    "PiRepoMirrorCycleResult",
    "PiRepoMirrorRunResult",
    "PiRepoMirrorWatchdog",
]
