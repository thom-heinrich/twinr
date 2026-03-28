#!/usr/bin/env python3
# CHANGELOG: 2026-03-27
# BUG-1: Bound the post-flash probe with a hard timeout. SyncCrazyflie.open_link() in current cflib waits on Event.wait() without a timeout, so the old script could hang forever during probe.
# BUG-2: Handle cfloader spawn failures (missing execute bit / bad shebang / permission errors) as bounded report failures instead of uncaught tracebacks.
# SEC-1: Serialize access to the Crazyradio/target with an advisory lock to prevent concurrent flash/probe races on shared Raspberry Pi deployments.
# SEC-2: Require an explicit acknowledgement for cold-boot flashing because Bitcraze documents untargeted cold boot as unpredictable if multiple Crazyflies are in bootloader range.
# IMP-1: Support manifest-aware artifacts (.bin, Bitcraze release .zip, or manifest.json) and record SHA-256 + manifest metadata in the report.
# IMP-2: Add transient-aware retry policy, sanitized cfloader subprocess environment, per-URI TOC cache isolation, and optional link-statistics capture during verification.

"""Flash and verify the Twinr on-device Crazyflie failsafe app.

Purpose
-------
Keep the firmware rollout for the on-device STM32 failsafe deterministic and
operator-friendly. This script flashes the already-built OOT firmware image
through ``cfloader`` and then proves that the new ``twinrFs`` app-layer param
surface is visible over the normal radio link.

Usage
-----
Command-line examples::

    /twinr/bitcraze/.venv/bin/python hardware/bitcraze/flash_on_device_failsafe.py
    /twinr/bitcraze/.venv/bin/python hardware/bitcraze/flash_on_device_failsafe.py --skip-probe
    /twinr/bitcraze/.venv/bin/python hardware/bitcraze/flash_on_device_failsafe.py --boot-mode cold --allow-unsafe-cold-boot --json
    /twinr/bitcraze/.venv/bin/python hardware/bitcraze/flash_on_device_failsafe.py --binary release-2025.09.zip --expected-sha256 <sha256>

Inputs
------
- A reachable Crazyflie on ``radio://0/80/2M`` by default
- A raw STM32 image, or a Bitcraze release ZIP / ``manifest.json`` that contains
  one main-board ``cf2``/``stm32`` firmware entry
- A working Bitcraze workspace under ``/twinr/bitcraze``

Outputs
-------
- A bounded flash/probe report on stdout
- Exit code ``0`` when flashing succeeded and the ``twinrFs`` app is visible
- Exit code ``1`` when flashing or probing failed
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import shlex
import signal
import stat
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Iterator, Mapping, Sequence, cast
import zipfile

try:
    import fcntl  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - Windows only
    fcntl = None  # type: ignore[assignment]


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from on_device_failsafe import OnDeviceFailsafeAvailability, probe_on_device_failsafe  # noqa: E402


DEFAULT_URI = "radio://0/80/2M"
DEFAULT_WORKSPACE = Path("/twinr/bitcraze")
DEFAULT_BINARY = SCRIPT_DIR / "twinr_on_device_failsafe" / "build" / "twinr_on_device_failsafe.bin"
DEFAULT_CFLOADER = DEFAULT_WORKSPACE / ".venv" / "bin" / "cfloader"

DEFAULT_FLASH_TIMEOUT_S = 120.0
DEFAULT_FLASH_ATTEMPTS_WARM = 2
DEFAULT_FLASH_ATTEMPTS_COLD = 1
DEFAULT_FLASH_RETRY_BACKOFF_S = 1.5

DEFAULT_PROBE_SETTLE_S = 3.0
DEFAULT_PROBE_TIMEOUT_S = 20.0
DEFAULT_PROBE_ATTEMPTS = 3
DEFAULT_PROBE_RETRY_BACKOFF_S = 1.0
DEFAULT_PROBE_LINK_STATS_SAMPLE_S = 0.35

DEFAULT_LOCK_TIMEOUT_S = 15.0
REPORT_VERSION = 2


@dataclass(frozen=True)
class ResolvedFlashArtifact:
    requested_path: str
    binary_path: str
    source_kind: str
    manifest_path: str | None
    manifest_member: str | None
    release: str | None
    repository: str | None
    platform: str | None
    target: str | None
    firmware_type: str | None
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class FlashAttemptReport:
    attempt: int
    command: tuple[str, ...]
    elapsed_s: float
    returncode: int | None
    timed_out: bool
    stdout_tail: tuple[str, ...]
    stderr_tail: tuple[str, ...]
    error: str | None


@dataclass(frozen=True)
class OnDeviceFailsafeFlashReport:
    """Persist the bounded outcome of one firmware flash/probe run."""

    report_version: int
    uri: str
    workspace: str
    requested_binary_path: str
    binary_path: str
    cfloader_executable: str
    boot_mode: str
    flash_command: tuple[str, ...]
    flashed: bool
    probe_attempted: bool
    availability: OnDeviceFailsafeAvailability | None
    availability_snapshot: dict[str, Any] | None
    link_stats: dict[str, Any]
    flash_stdout_tail: tuple[str, ...]
    flash_stderr_tail: tuple[str, ...]
    failures: tuple[str, ...]
    warnings: tuple[str, ...]
    artifact_source_kind: str | None
    artifact_sha256: str | None
    artifact_size_bytes: int | None
    artifact_release: str | None
    artifact_repository: str | None
    artifact_platform: str | None
    artifact_target: str | None
    artifact_firmware_type: str | None
    flash_attempt_reports: tuple[FlashAttemptReport, ...]
    flash_attempts_requested: int
    probe_attempts_requested: int
    probe_attempts_used: int
    flash_elapsed_s: float
    probe_elapsed_s: float | None
    elapsed_s: float
    lock_path: str | None
    started_at_epoch_s: float
    finished_at_epoch_s: float


def _tail_lines(text: str, *, limit: int = 8) -> tuple[str, ...]:
    """Return only the last few non-empty output lines for durable reports."""

    lines = [line.rstrip() for line in str(text).splitlines() if line.strip()]
    if len(lines) <= limit:
        return tuple(lines)
    return tuple(lines[-limit:])


def _normalize_process_output(text: bytes | str | None) -> str:
    """Normalize subprocess output into text for durable tail capture."""

    if text is None:
        return ""
    if isinstance(text, bytes):
        return text.decode("utf-8", errors="replace")
    return text


def _jsonable(value: Any) -> Any:
    """Convert nested report values into JSON-safe primitives."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return {field_name: _jsonable(field_value) for field_name, field_value in asdict(cast(Any, value)).items()}
    if hasattr(value, "__dict__"):
        return {str(key): _jsonable(item) for key, item in vars(value).items() if not str(key).startswith("_")}
    return repr(value)


def _snapshot_availability(availability: Any) -> dict[str, Any] | None:
    """Extract the user-visible availability fields for durable reporting."""

    if availability is None:
        return None

    preferred_fields = ("loaded", "protocol_version", "state_name", "reason_name", "failures")
    snapshot: dict[str, Any] = {}
    for name in preferred_fields:
        if hasattr(availability, name):
            snapshot[name] = _jsonable(getattr(availability, name))

    if snapshot:
        return snapshot

    converted = _jsonable(availability)
    if isinstance(converted, dict):
        return converted

    return {"value": converted}


def _reconstruct_availability(snapshot: dict[str, Any] | None) -> OnDeviceFailsafeAvailability | None:
    """Best-effort reconstruction for programmatic callers."""

    if not snapshot:
        return None
    try:
        return OnDeviceFailsafeAvailability(**snapshot)
    except Exception:
        return None


def _availability_value(
    availability: OnDeviceFailsafeAvailability | None,
    snapshot: dict[str, Any] | None,
    key: str,
    default: Any = None,
) -> Any:
    if availability is not None and hasattr(availability, key):
        return getattr(availability, key)
    if snapshot is not None:
        return snapshot.get(key, default)
    return default


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_sha256(text: str | None) -> str | None:
    if text is None:
        return None
    value = str(text).strip().lower()
    if not value:
        return None
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError("expected SHA-256 must be exactly 64 hexadecimal characters")
    return value


def _safe_uri_token(uri: str) -> str:
    return hashlib.sha256(str(uri).encode("utf-8")).hexdigest()[:16]


def _ensure_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)


def _write_private_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(data)
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def _select_bitcraze_manifest_entry(manifest: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("Bitcraze manifest does not contain a valid `files` mapping")

    candidates: list[tuple[str, Mapping[str, Any]]] = []
    for filename, metadata in files.items():
        if not isinstance(filename, str) or not isinstance(metadata, Mapping):
            continue
        platform = str(metadata.get("platform", "")).strip().lower()
        target = str(metadata.get("target", "")).strip().lower()
        fw_type = str(metadata.get("type", "")).strip().lower()
        if platform == "cf2" and target == "stm32" and fw_type == "fw":
            candidates.append((filename, metadata))

    if not candidates:
        raise ValueError("no `cf2`/`stm32` firmware entry found in Bitcraze manifest")

    if len(candidates) == 1:
        return candidates[0]

    preferred = [item for item in candidates if str(item[1].get("repository", "")).strip() == "crazyflie-firmware"]
    if len(preferred) == 1:
        return preferred[0]

    names = ", ".join(sorted(name for name, _ in candidates))
    raise ValueError(f"manifest contains multiple STM32 firmware entries: {names}")


def _resolve_flash_artifact(
    *,
    requested_path: Path,
    workspace: Path,
) -> ResolvedFlashArtifact:
    """Resolve a raw binary, Bitcraze release ZIP, or manifest.json into one concrete .bin file."""

    path = Path(requested_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"artifact not found: {path}")

    resolved_path = path.resolve()
    suffix = resolved_path.suffix.lower()

    if suffix == ".bin":
        if not resolved_path.is_file():
            raise FileNotFoundError(f"binary not found: {resolved_path}")
        return ResolvedFlashArtifact(
            requested_path=str(path),
            binary_path=str(resolved_path),
            source_kind="bin",
            manifest_path=None,
            manifest_member=None,
            release=None,
            repository=None,
            platform=None,
            target=None,
            firmware_type=None,
            sha256=_sha256_file(resolved_path),
            size_bytes=resolved_path.stat().st_size,
        )

    if suffix == ".json":
        manifest = json.loads(resolved_path.read_text(encoding="utf-8"))
        entry_name, metadata = _select_bitcraze_manifest_entry(manifest)
        binary_path = (resolved_path.parent / entry_name).resolve()
        if not binary_path.is_file():
            raise FileNotFoundError(f"manifest entry not found on disk: {binary_path}")
        return ResolvedFlashArtifact(
            requested_path=str(path),
            binary_path=str(binary_path),
            source_kind="manifest",
            manifest_path=str(resolved_path),
            manifest_member=entry_name,
            release=str(metadata.get("release")) if metadata.get("release") is not None else None,
            repository=str(metadata.get("repository")) if metadata.get("repository") is not None else None,
            platform=str(metadata.get("platform")) if metadata.get("platform") is not None else None,
            target=str(metadata.get("target")) if metadata.get("target") is not None else None,
            firmware_type=str(metadata.get("type")) if metadata.get("type") is not None else None,
            sha256=_sha256_file(binary_path),
            size_bytes=binary_path.stat().st_size,
        )

    if suffix == ".zip":
        with zipfile.ZipFile(resolved_path, "r") as archive:
            try:
                manifest_bytes = archive.read("manifest.json")
            except KeyError as exc:
                raise ValueError(f"Bitcraze ZIP is missing manifest.json: {resolved_path}") from exc
            manifest = json.loads(manifest_bytes.decode("utf-8"))
            entry_name, metadata = _select_bitcraze_manifest_entry(manifest)
            try:
                binary_bytes = archive.read(entry_name)
            except KeyError as exc:
                raise FileNotFoundError(f"manifest entry {entry_name!r} is missing from ZIP {resolved_path}") from exc

        sha256 = _sha256_bytes(binary_bytes)
        materialized_dir = workspace / ".cache" / "flash_artifacts" / sha256[:16]
        materialized_path = materialized_dir / Path(entry_name).name
        if not materialized_path.exists() or _sha256_file(materialized_path) != sha256:
            _write_private_file(materialized_path, binary_bytes)

        return ResolvedFlashArtifact(
            requested_path=str(path),
            binary_path=str(materialized_path.resolve()),
            source_kind="zip",
            manifest_path=str(resolved_path),
            manifest_member=entry_name,
            release=str(metadata.get("release")) if metadata.get("release") is not None else None,
            repository=str(metadata.get("repository")) if metadata.get("repository") is not None else None,
            platform=str(metadata.get("platform")) if metadata.get("platform") is not None else None,
            target=str(metadata.get("target")) if metadata.get("target") is not None else None,
            firmware_type=str(metadata.get("type")) if metadata.get("type") is not None else None,
            sha256=sha256,
            size_bytes=len(binary_bytes),
        )

    raise ValueError(f"unsupported artifact type for {resolved_path}; expected .bin, .json or .zip")


def build_cfloader_command(
    *,
    cfloader_executable: Path,
    binary_path: Path,
    uri: str,
    boot_mode: str,
) -> tuple[str, ...]:
    """Build the exact ``cfloader`` command for one firmware flash attempt."""

    command: list[str] = [str(cfloader_executable)]
    normalized_boot_mode = str(boot_mode).strip().lower()
    if normalized_boot_mode == "warm":
        command.extend(["-w", str(uri).strip() or DEFAULT_URI])
    elif normalized_boot_mode != "cold":
        raise ValueError(f"unsupported boot mode: {boot_mode}")
    command.extend(["flash", str(binary_path), "stm32-fw"])
    return tuple(command)


def _import_cflib() -> tuple[Any, Any, Any]:
    """Import the small cflib slice needed for post-flash probing."""

    import cflib.crtp as crtp  # pylint: disable=import-error
    from cflib.crazyflie import Crazyflie  # pylint: disable=import-error
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie  # pylint: disable=import-error

    return crtp, Crazyflie, SyncCrazyflie


def _build_subprocess_env() -> dict[str, str]:
    """Build a small, deterministic environment for cfloader."""

    allowed_prefixes = ("LC_",)
    allowed_keys = {
        "HOME",
        "LANG",
        "LOGNAME",
        "PATH",
        "SYSTEMROOT",
        "TEMP",
        "TERM",
        "TMP",
        "TMPDIR",
        "USER",
        "WINDIR",
    }

    env: dict[str, str] = {}
    for key, value in os.environ.items():
        if key in allowed_keys or any(key.startswith(prefix) for prefix in allowed_prefixes):
            env[key] = value

    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.pop("LD_PRELOAD", None)
    env.pop("DYLD_INSERT_LIBRARIES", None)
    env.setdefault("LANG", "C.UTF-8")
    env.setdefault("LC_ALL", env["LANG"])
    return env


def _execute_process(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout_s: float,
    env: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    """Execute a subprocess with process-group cleanup on timeout."""

    timeout_s = max(1.0, float(timeout_s))

    if os.name == "posix":
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(cwd),
            env=dict(env),
            close_fds=True,
            start_new_session=True,
        )
    else:  # pragma: no cover - Windows only
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(cwd),
            env=dict(env),
            close_fds=True,
        )

    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:  # pragma: no cover - Windows only
            process.kill()
        stdout, stderr = process.communicate()
        timeout_output = _normalize_process_output(exc.output) + _normalize_process_output(stdout)
        timeout_stderr = _normalize_process_output(exc.stderr) + _normalize_process_output(stderr)
        raise subprocess.TimeoutExpired(
            cmd=list(command),
            timeout=timeout_s,
            output=timeout_output,
            stderr=timeout_stderr,
        ) from exc

    return subprocess.CompletedProcess(
        args=list(command),
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _run_cfloader_attempt(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout_s: float,
    env: Mapping[str, str],
    attempt_index: int,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> tuple[subprocess.CompletedProcess[str] | None, FlashAttemptReport]:
    """Execute one bounded cfloader attempt and normalize failure modes."""

    started = time.monotonic()
    try:
        if runner is None or runner is subprocess.run:
            completed = _execute_process(command, cwd=cwd, timeout_s=timeout_s, env=env)
        else:
            completed = runner(
                list(command),
                capture_output=True,
                text=True,
                timeout=max(1.0, float(timeout_s)),
                check=False,
                cwd=str(cwd),
                env=dict(env),
            )
    except subprocess.TimeoutExpired as exc:
        elapsed_s = time.monotonic() - started
        attempt_report = FlashAttemptReport(
            attempt=attempt_index,
            command=tuple(command),
            elapsed_s=elapsed_s,
            returncode=None,
            timed_out=True,
            stdout_tail=_tail_lines(_normalize_process_output(getattr(exc, "stdout", None) or getattr(exc, "output", None))),
            stderr_tail=_tail_lines(_normalize_process_output(getattr(exc, "stderr", None))),
            error=f"cfloader timeout after {float(exc.timeout):.1f}s",
        )
        return None, attempt_report
    except OSError as exc:
        elapsed_s = time.monotonic() - started
        attempt_report = FlashAttemptReport(
            attempt=attempt_index,
            command=tuple(command),
            elapsed_s=elapsed_s,
            returncode=None,
            timed_out=False,
            stdout_tail=(),
            stderr_tail=(),
            error=f"cfloader spawn failed: {exc.__class__.__name__}: {exc}",
        )
        return None, attempt_report

    elapsed_s = time.monotonic() - started
    attempt_report = FlashAttemptReport(
        attempt=attempt_index,
        command=tuple(command),
        elapsed_s=elapsed_s,
        returncode=completed.returncode,
        timed_out=False,
        stdout_tail=_tail_lines(completed.stdout),
        stderr_tail=_tail_lines(completed.stderr),
        error=None if completed.returncode == 0 else f"cfloader exited with {completed.returncode}",
    )
    return completed, attempt_report


def _remove_callback(callback_container: Any, callback: Callable[..., Any]) -> None:
    try:
        callback_container.remove_callback(callback)
    except Exception:
        pass


@contextmanager
def _radio_lock(
    *,
    workspace: Path,
    uri: str,
    timeout_s: float,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[str]:
    """Serialize radio access on one host to avoid concurrent flash/probe races."""

    lock_dir = workspace / ".locks"
    _ensure_directory(lock_dir)
    lock_path = lock_dir / f"flash_on_device_failsafe.{_safe_uri_token(uri)}.lock"

    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    acquired = False
    deadline = time.monotonic() + max(0.1, float(timeout_s))

    try:
        while True:
            try:
                if fcntl is None:  # pragma: no cover - Windows only
                    acquired = True
                    break
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"radio lock timed out after {float(timeout_s):.1f}s: {lock_path}")
                sleep(0.2)

        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n{uri}\n".encode("utf-8", errors="replace"))
        yield str(lock_path)
    finally:
        if acquired and fcntl is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)


def _link_stats_callbacks() -> tuple[tuple[str, str], ...]:
    return (
        ("latency_updated", "latency"),
        ("uplink_rate_updated", "uplink_rate"),
        ("downlink_rate_updated", "downlink_rate"),
        ("uplink_rssi_updated", "uplink_rssi"),
        ("downlink_rssi_updated", "downlink_rssi"),
        ("link_quality_updated", "link_quality"),
        ("uplink_congestion_updated", "uplink_congestion"),
        ("downlink_congestion_updated", "downlink_congestion"),
        ("congestion_updated", "congestion"),
    )


def _probe_loaded_failsafe(
    *,
    uri: str,
    workspace: Path,
    settle_s: float,
    link_stats_sample_s: float,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[OnDeviceFailsafeAvailability, dict[str, Any]]:
    """Reconnect over the normal radio link and prove that ``twinrFs`` is live."""

    if settle_s > 0.0:
        sleep(float(settle_s))

    crtp, crazyflie_cls, sync_crazyflie_cls = _import_cflib()
    crtp.init_drivers(enable_debug_driver=False)

    cache_dir = workspace / ".cache" / "cflib_toc" / _safe_uri_token(uri)
    _ensure_directory(cache_dir)

    captured_link_stats: dict[str, Any] = {}
    registered_callbacks: list[tuple[Any, Callable[..., Any]]] = []

    sync_context = sync_crazyflie_cls(str(uri).strip() or DEFAULT_URI, cf=crazyflie_cls(rw_cache=str(cache_dir)))
    with sync_context as sync_cf:
        link_statistics = getattr(sync_cf.cf, "link_statistics", None)

        if link_statistics is not None:
            for attr_name, key in _link_stats_callbacks():
                callback_container = getattr(link_statistics, attr_name, None)
                if callback_container is None or not hasattr(callback_container, "add_callback"):
                    continue

                def _callback(value: Any, *, _key: str = key) -> None:
                    captured_link_stats[_key] = _jsonable(value)

                callback_container.add_callback(_callback)
                registered_callbacks.append((callback_container, _callback))

            if hasattr(link_statistics, "start"):
                try:
                    link_statistics.start()
                except Exception:
                    pass

        if hasattr(sync_cf, "wait_for_params"):
            sync_cf.wait_for_params()

        availability = probe_on_device_failsafe(sync_cf)

        if link_stats_sample_s > 0.0:
            sleep(float(link_stats_sample_s))

        if link_statistics is not None and hasattr(link_statistics, "stop"):
            try:
                link_statistics.stop()
            except Exception:
                pass

        for callback_container, callback in registered_callbacks:
            _remove_callback(callback_container, callback)

        return availability, captured_link_stats


def _probe_worker(
    result_queue: "mp.Queue[dict[str, Any]]",
    *,
    uri: str,
    workspace: str,
    settle_s: float,
    link_stats_sample_s: float,
) -> None:
    """Child-process worker to keep the parent probe fully bounded."""

    try:
        availability, link_stats = _probe_loaded_failsafe(
            uri=uri,
            workspace=Path(workspace),
            settle_s=settle_s,
            link_stats_sample_s=link_stats_sample_s,
        )
        result_queue.put(
            {
                "ok": True,
                "availability_snapshot": _snapshot_availability(availability),
                "link_stats": _jsonable(link_stats),
            }
        )
    except Exception as exc:
        result_queue.put(
            {
                "ok": False,
                "error": f"probe failed: {exc.__class__.__name__}: {exc}",
                "traceback_tail": _tail_lines(traceback.format_exc(), limit=12),
            }
        )


def _run_probe_with_timeout(
    *,
    uri: str,
    workspace: Path,
    settle_s: float,
    timeout_s: float,
    link_stats_sample_s: float,
) -> tuple[dict[str, Any] | None, dict[str, Any], str | None, tuple[str, ...]]:
    """Run one probe attempt in a child process and bound it with a hard timeout."""

    ctx = mp.get_context("spawn")
    result_queue: "mp.Queue[dict[str, Any]]" = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_probe_worker,
        kwargs={
            "result_queue": result_queue,
            "uri": str(uri),
            "workspace": str(workspace),
            "settle_s": float(settle_s),
            "link_stats_sample_s": float(link_stats_sample_s),
        },
    )
    process.daemon = True
    process.start()
    process.join(max(1.0, float(timeout_s)))

    if process.is_alive():
        process.terminate()
        process.join(2.0)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()  # pragma: no cover - hard hang fallback
            process.join(2.0)
        try:
            result_queue.close()
        except Exception:
            pass
        try:
            result_queue.join_thread()
        except Exception:
            pass
        return None, {}, f"probe timed out after {float(timeout_s):.1f}s", ()

    try:
        payload = result_queue.get(timeout=0.5)
    except queue.Empty:
        return None, {}, f"probe subprocess exited with code {process.exitcode} without a result", ()
    finally:
        try:
            result_queue.close()
        except Exception:
            pass
        try:
            result_queue.join_thread()
        except Exception:
            pass

    if not payload.get("ok", False):
        error = str(payload.get("error", f"probe subprocess exited with code {process.exitcode}"))
        traceback_tail = tuple(str(line) for line in payload.get("traceback_tail", ()))
        return None, {}, error, traceback_tail

    snapshot = payload.get("availability_snapshot")
    if not isinstance(snapshot, dict):
        snapshot = None

    link_stats = payload.get("link_stats")
    if not isinstance(link_stats, dict):
        link_stats = {}

    return snapshot, link_stats, None, ()


def _default_flash_attempts(boot_mode: str) -> int:
    return DEFAULT_FLASH_ATTEMPTS_WARM if str(boot_mode).strip().lower() == "warm" else DEFAULT_FLASH_ATTEMPTS_COLD


def flash_on_device_failsafe(
    *,
    uri: str = DEFAULT_URI,
    workspace: Path = DEFAULT_WORKSPACE,
    binary_path: Path = DEFAULT_BINARY,
    cfloader_executable: Path = DEFAULT_CFLOADER,
    boot_mode: str = "warm",
    flash_timeout_s: float = DEFAULT_FLASH_TIMEOUT_S,
    flash_attempts: int | None = None,
    flash_retry_backoff_s: float = DEFAULT_FLASH_RETRY_BACKOFF_S,
    probe_settle_s: float = DEFAULT_PROBE_SETTLE_S,
    probe_timeout_s: float = DEFAULT_PROBE_TIMEOUT_S,
    probe_attempts: int = DEFAULT_PROBE_ATTEMPTS,
    probe_retry_backoff_s: float = DEFAULT_PROBE_RETRY_BACKOFF_S,
    probe_link_stats_sample_s: float = DEFAULT_PROBE_LINK_STATS_SAMPLE_S,
    lock_timeout_s: float = DEFAULT_LOCK_TIMEOUT_S,
    expected_sha256: str | None = None,
    allow_unsafe_cold_boot: bool = False,
    skip_probe: bool = False,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
) -> OnDeviceFailsafeFlashReport:
    """Flash the firmware app and optionally verify the ``twinrFs`` param surface."""

    started_at_epoch_s = time.time()
    overall_started = time.monotonic()

    normalized_uri = str(uri).strip() or DEFAULT_URI
    normalized_boot_mode = str(boot_mode).strip().lower() or "warm"
    normalized_workspace = Path(workspace).expanduser()
    normalized_cfloader = Path(cfloader_executable).expanduser()

    failures: list[str] = []
    warnings: list[str] = []

    flash_attempt_reports: list[FlashAttemptReport] = []
    stdout_tail: tuple[str, ...] = ()
    stderr_tail: tuple[str, ...] = ()

    availability: OnDeviceFailsafeAvailability | None = None
    availability_snapshot: dict[str, Any] | None = None
    link_stats: dict[str, Any] = {}

    flashed = False
    probe_attempted = False
    probe_attempts_used = 0
    probe_elapsed_s: float | None = None
    flash_elapsed_s = 0.0
    lock_path: str | None = None
    flash_command: tuple[str, ...] = ()
    resolved_artifact: ResolvedFlashArtifact | None = None

    try:
        normalized_expected_sha256 = _normalize_sha256(expected_sha256)
    except ValueError as exc:
        normalized_expected_sha256 = None
        failures.append(str(exc))

    if not normalized_workspace.exists() or not normalized_workspace.is_dir():
        failures.append(f"workspace not found: {normalized_workspace}")

    if normalized_boot_mode not in {"warm", "cold"}:
        failures.append(f"unsupported boot mode: {boot_mode}")

    if normalized_boot_mode == "cold" and not bool(allow_unsafe_cold_boot):
        failures.append("cold boot requires --allow-unsafe-cold-boot")
        # BREAKING: cold boot now requires an explicit acknowledgement because Bitcraze documents untargeted
        # cold-boot flashing as unpredictable if multiple Crazyflies are in bootloader range.

    requested_flash_attempts = int(flash_attempts) if flash_attempts is not None else _default_flash_attempts(normalized_boot_mode)
    requested_flash_attempts = max(1, requested_flash_attempts)

    if normalized_boot_mode == "cold" and requested_flash_attempts > 1:
        failures.append("cold boot does not support automatic multi-attempt flashing; use --flash-attempts 1")
        # BREAKING: cold boot retries remain disabled because retries cannot target one specific device in untargeted bootloader discovery.

    normalized_cfloader_resolved: Path | None = None
    if not failures:
        try:
            normalized_cfloader_resolved = normalized_cfloader.resolve(strict=True)
        except FileNotFoundError:
            failures.append(f"cfloader not found: {normalized_cfloader}")
        else:
            if not normalized_cfloader_resolved.is_file():
                failures.append(f"cfloader not found: {normalized_cfloader_resolved}")
            elif not os.access(str(normalized_cfloader_resolved), os.X_OK):
                failures.append(f"cfloader is not executable: {normalized_cfloader_resolved}")

    if not failures:
        try:
            resolved_artifact = _resolve_flash_artifact(
                requested_path=Path(binary_path),
                workspace=normalized_workspace,
            )
        except Exception as exc:
            failures.append(f"artifact resolution failed: {exc.__class__.__name__}: {exc}")

    if not failures and resolved_artifact is not None and normalized_expected_sha256 is not None:
        if resolved_artifact.sha256 != normalized_expected_sha256:
            failures.append(
                f"artifact SHA-256 mismatch: expected {normalized_expected_sha256}, got {resolved_artifact.sha256}"
            )

    if not failures and normalized_cfloader_resolved is not None and resolved_artifact is not None:
        try:
            flash_command = build_cfloader_command(
                cfloader_executable=normalized_cfloader_resolved,
                binary_path=Path(resolved_artifact.binary_path),
                uri=normalized_uri,
                boot_mode=normalized_boot_mode,
            )
        except ValueError as exc:
            failures.append(str(exc))

    if failures:
        finished_at_epoch_s = time.time()
        return OnDeviceFailsafeFlashReport(
            report_version=REPORT_VERSION,
            uri=normalized_uri,
            workspace=str(normalized_workspace),
            requested_binary_path=str(binary_path),
            binary_path=str(resolved_artifact.binary_path) if resolved_artifact is not None else str(Path(binary_path).expanduser()),
            cfloader_executable=str(normalized_cfloader),
            boot_mode=normalized_boot_mode,
            flash_command=flash_command,
            flashed=False,
            probe_attempted=False,
            availability=None,
            availability_snapshot=None,
            link_stats={},
            flash_stdout_tail=stdout_tail,
            flash_stderr_tail=stderr_tail,
            failures=tuple(failures),
            warnings=tuple(warnings),
            artifact_source_kind=resolved_artifact.source_kind if resolved_artifact is not None else None,
            artifact_sha256=resolved_artifact.sha256 if resolved_artifact is not None else None,
            artifact_size_bytes=resolved_artifact.size_bytes if resolved_artifact is not None else None,
            artifact_release=resolved_artifact.release if resolved_artifact is not None else None,
            artifact_repository=resolved_artifact.repository if resolved_artifact is not None else None,
            artifact_platform=resolved_artifact.platform if resolved_artifact is not None else None,
            artifact_target=resolved_artifact.target if resolved_artifact is not None else None,
            artifact_firmware_type=resolved_artifact.firmware_type if resolved_artifact is not None else None,
            flash_attempt_reports=(),
            flash_attempts_requested=requested_flash_attempts,
            probe_attempts_requested=0 if skip_probe else max(1, int(probe_attempts)),
            probe_attempts_used=0,
            flash_elapsed_s=0.0,
            probe_elapsed_s=None,
            elapsed_s=time.monotonic() - overall_started,
            lock_path=None,
            started_at_epoch_s=started_at_epoch_s,
            finished_at_epoch_s=finished_at_epoch_s,
        )

    assert resolved_artifact is not None
    assert normalized_cfloader_resolved is not None

    env = _build_subprocess_env()
    flash_started = time.monotonic()

    try:
        with _radio_lock(
            workspace=normalized_workspace,
            uri=normalized_uri,
            timeout_s=lock_timeout_s,
            sleep=sleep,
        ) as acquired_lock_path:
            lock_path = acquired_lock_path

            for attempt_index in range(1, requested_flash_attempts + 1):
                completed, attempt_report = _run_cfloader_attempt(
                    flash_command,
                    cwd=normalized_workspace,
                    timeout_s=flash_timeout_s,
                    env=env,
                    attempt_index=attempt_index,
                    runner=runner,
                )
                flash_attempt_reports.append(attempt_report)
                stdout_tail = attempt_report.stdout_tail
                stderr_tail = attempt_report.stderr_tail

                if completed is not None and completed.returncode == 0:
                    flashed = True
                    break

                if attempt_index < requested_flash_attempts:
                    warnings.append(
                        f"flash attempt {attempt_index}/{requested_flash_attempts} failed: {attempt_report.error or 'unknown error'}"
                    )
                    sleep(max(0.0, float(flash_retry_backoff_s)) * float(attempt_index))

            flash_elapsed_s = time.monotonic() - flash_started

            if not flashed:
                last_error = flash_attempt_reports[-1].error if flash_attempt_reports else "unknown flash failure"
                failures.append(f"cfloader failed after {requested_flash_attempts} attempt(s): {last_error}")

            if flashed and not skip_probe:
                probe_attempted = True
                probe_started = time.monotonic()
                requested_probe_attempts = max(1, int(probe_attempts))
                for attempt_index in range(1, requested_probe_attempts + 1):
                    probe_attempts_used = attempt_index
                    snapshot, captured_link_stats, probe_error, traceback_tail = _run_probe_with_timeout(
                        uri=normalized_uri,
                        workspace=normalized_workspace,
                        settle_s=probe_settle_s if attempt_index == 1 else 0.0,
                        timeout_s=probe_timeout_s,
                        link_stats_sample_s=probe_link_stats_sample_s,
                    )
                    if probe_error is not None:
                        if attempt_index < requested_probe_attempts:
                            warnings.append(f"probe attempt {attempt_index}/{requested_probe_attempts} failed: {probe_error}")
                            if traceback_tail:
                                warnings.extend(f"probe traceback={line}" for line in traceback_tail)
                            sleep(max(0.0, float(probe_retry_backoff_s)) * float(attempt_index))
                            continue
                        failures.append(probe_error)
                        if traceback_tail:
                            failures.extend(f"probe traceback={line}" for line in traceback_tail)
                        break

                    availability_snapshot = snapshot
                    availability = _reconstruct_availability(snapshot)
                    link_stats = dict(captured_link_stats)

                    probe_failures = tuple(str(item) for item in _availability_value(availability, snapshot, "failures", ()) or ())
                    loaded = bool(_availability_value(availability, snapshot, "loaded", False))
                    if probe_failures:
                        if attempt_index < requested_probe_attempts:
                            warnings.extend(probe_failures)
                            warnings.append(
                                f"probe attempt {attempt_index}/{requested_probe_attempts} did not pass validation; retrying"
                            )
                            sleep(max(0.0, float(probe_retry_backoff_s)) * float(attempt_index))
                            continue
                        failures.extend(probe_failures)

                    if not loaded:
                        error_text = "post-flash probe did not see the `twinrFs` firmware app"
                        if attempt_index < requested_probe_attempts:
                            warnings.append(f"{error_text}; retrying")
                            sleep(max(0.0, float(probe_retry_backoff_s)) * float(attempt_index))
                            continue
                        failures.append(error_text)
                        break

                    break

                probe_elapsed_s = time.monotonic() - probe_started

    except TimeoutError as exc:
        flash_elapsed_s = time.monotonic() - flash_started
        failures.append(str(exc))

    finished_at_epoch_s = time.time()
    return OnDeviceFailsafeFlashReport(
        report_version=REPORT_VERSION,
        uri=normalized_uri,
        workspace=str(normalized_workspace),
        requested_binary_path=str(binary_path),
        binary_path=str(resolved_artifact.binary_path),
        cfloader_executable=str(normalized_cfloader_resolved),
        boot_mode=normalized_boot_mode,
        flash_command=flash_command,
        flashed=flashed,
        probe_attempted=probe_attempted,
        availability=availability,
        availability_snapshot=availability_snapshot,
        link_stats=link_stats,
        flash_stdout_tail=stdout_tail,
        flash_stderr_tail=stderr_tail,
        failures=tuple(failures),
        warnings=tuple(warnings),
        artifact_source_kind=resolved_artifact.source_kind,
        artifact_sha256=resolved_artifact.sha256,
        artifact_size_bytes=resolved_artifact.size_bytes,
        artifact_release=resolved_artifact.release,
        artifact_repository=resolved_artifact.repository,
        artifact_platform=resolved_artifact.platform,
        artifact_target=resolved_artifact.target,
        artifact_firmware_type=resolved_artifact.firmware_type,
        flash_attempt_reports=tuple(flash_attempt_reports),
        flash_attempts_requested=requested_flash_attempts,
        probe_attempts_requested=0 if skip_probe else max(1, int(probe_attempts)),
        probe_attempts_used=probe_attempts_used,
        flash_elapsed_s=flash_elapsed_s,
        probe_elapsed_s=probe_elapsed_s,
        elapsed_s=time.monotonic() - overall_started,
        lock_path=lock_path,
        started_at_epoch_s=started_at_epoch_s,
        finished_at_epoch_s=finished_at_epoch_s,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Flash and verify the Twinr on-device Crazyflie failsafe app.")
    parser.add_argument("--uri", default=DEFAULT_URI, help="Crazyflie radio URI for warm-boot flashing and probing.")
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE), help="Bitcraze workspace root (default: /twinr/bitcraze).")
    parser.add_argument(
        "--binary",
        "--artifact",
        dest="binary",
        default=str(DEFAULT_BINARY),
        help=(
            "Firmware artifact to flash: raw .bin, Bitcraze release .zip, or Bitcraze manifest.json "
            f"(default: {DEFAULT_BINARY})."
        ),
    )
    parser.add_argument(
        "--cfloader",
        default=str(DEFAULT_CFLOADER),
        help="cfloader executable to use for flashing (default: /twinr/bitcraze/.venv/bin/cfloader).",
    )
    parser.add_argument(
        "--boot-mode",
        choices=("warm", "cold"),
        default="warm",
        help="Warm boot reboots the current URI into bootloader; cold boot waits for a manual restart (default: warm).",
    )
    parser.add_argument(
        "--allow-unsafe-cold-boot",
        action="store_true",
        help="Acknowledge the documented risk of untargeted cold-boot flashing when multiple devices are in range.",
    )
    parser.add_argument(
        "--expected-sha256",
        default=None,
        help="Fail before flashing unless the resolved STM32 firmware binary matches this SHA-256.",
    )
    parser.add_argument(
        "--flash-timeout-s",
        type=float,
        default=DEFAULT_FLASH_TIMEOUT_S,
        help="Maximum seconds to wait for each cfloader attempt before failing (default: 120).",
    )
    parser.add_argument(
        "--flash-attempts",
        type=int,
        default=None,
        help="Flash attempts before giving up (default: 2 for warm boot, 1 for cold boot).",
    )
    parser.add_argument(
        "--flash-retry-backoff-s",
        type=float,
        default=DEFAULT_FLASH_RETRY_BACKOFF_S,
        help="Seconds to wait between flash retries (default: 1.5).",
    )
    parser.add_argument(
        "--probe-settle-s",
        type=float,
        default=DEFAULT_PROBE_SETTLE_S,
        help="Seconds to wait after flash/reset before starting the first post-flash probe (default: 3).",
    )
    parser.add_argument(
        "--probe-timeout-s",
        type=float,
        default=DEFAULT_PROBE_TIMEOUT_S,
        help="Hard timeout for each post-flash probe subprocess (default: 20).",
    )
    parser.add_argument(
        "--probe-attempts",
        type=int,
        default=DEFAULT_PROBE_ATTEMPTS,
        help="Number of bounded probe attempts before failing (default: 3).",
    )
    parser.add_argument(
        "--probe-retry-backoff-s",
        type=float,
        default=DEFAULT_PROBE_RETRY_BACKOFF_S,
        help="Seconds to wait between probe retries (default: 1.0).",
    )
    parser.add_argument(
        "--probe-link-stats-sample-s",
        type=float,
        default=DEFAULT_PROBE_LINK_STATS_SAMPLE_S,
        help="Extra seconds to keep the link open after probing so current cflib link statistics can update (default: 0.35).",
    )
    parser.add_argument(
        "--lock-timeout-s",
        type=float,
        default=DEFAULT_LOCK_TIMEOUT_S,
        help="Maximum seconds to wait for the local radio lock (default: 15).",
    )
    parser.add_argument("--skip-probe", action="store_true", help="Only flash the binary and skip the post-flash twinrFs probe.")
    parser.add_argument("--json", action="store_true", help="Emit the full bounded report as JSON.")
    return parser


def _format_command(command: Sequence[str]) -> str:
    try:
        return shlex.join(list(command))
    except Exception:
        return " ".join(command)


def _print_human_report(report: OnDeviceFailsafeFlashReport) -> None:
    """Print one compact operator-facing flash/probe report."""

    print(f"flashed={str(report.flashed).lower()}")
    print(f"uri={report.uri}")
    print(f"workspace={report.workspace}")
    print(f"requested_binary_path={report.requested_binary_path}")
    print(f"binary_path={report.binary_path}")
    print(f"artifact.source_kind={report.artifact_source_kind}")
    if report.artifact_sha256 is not None:
        print(f"artifact.sha256={report.artifact_sha256}")
    if report.artifact_size_bytes is not None:
        print(f"artifact.size_bytes={report.artifact_size_bytes}")
    if report.artifact_release is not None:
        print(f"artifact.release={report.artifact_release}")
    if report.artifact_repository is not None:
        print(f"artifact.repository={report.artifact_repository}")
    if report.artifact_platform is not None:
        print(f"artifact.platform={report.artifact_platform}")
    if report.artifact_target is not None:
        print(f"artifact.target={report.artifact_target}")
    if report.artifact_firmware_type is not None:
        print(f"artifact.firmware_type={report.artifact_firmware_type}")
    print(f"boot_mode={report.boot_mode}")
    print(f"cfloader={report.cfloader_executable}")
    print(f"flash_command={_format_command(report.flash_command)}")
    print(f"flash_attempts_requested={report.flash_attempts_requested}")
    print(f"flash_elapsed_s={report.flash_elapsed_s:.3f}")
    print(f"probe_attempted={str(report.probe_attempted).lower()}")
    print(f"probe_attempts_used={report.probe_attempts_used}")
    if report.probe_elapsed_s is not None:
        print(f"probe_elapsed_s={report.probe_elapsed_s:.3f}")
    print(f"elapsed_s={report.elapsed_s:.3f}")
    if report.lock_path is not None:
        print(f"lock_path={report.lock_path}")

    snapshot = report.availability_snapshot
    loaded = _availability_value(report.availability, snapshot, "loaded", None)
    protocol_version = _availability_value(report.availability, snapshot, "protocol_version", None)
    state_name = _availability_value(report.availability, snapshot, "state_name", None)
    reason_name = _availability_value(report.availability, snapshot, "reason_name", None)

    if loaded is not None:
        print(f"twinrFs.loaded={str(bool(loaded)).lower()}")
    if protocol_version is not None:
        print(f"twinrFs.protocol_version={protocol_version}")
    if state_name is not None:
        print(f"twinrFs.state={state_name}")
    if reason_name is not None:
        print(f"twinrFs.reason={reason_name}")

    for key in sorted(report.link_stats):
        print(f"link_stats.{key}={report.link_stats[key]}")

    for attempt_report in report.flash_attempt_reports:
        print(f"flash_attempt.{attempt_report.attempt}.elapsed_s={attempt_report.elapsed_s:.3f}")
        print(f"flash_attempt.{attempt_report.attempt}.timed_out={str(attempt_report.timed_out).lower()}")
        print(f"flash_attempt.{attempt_report.attempt}.returncode={attempt_report.returncode}")
        for line in attempt_report.stdout_tail:
            print(f"flash_attempt.{attempt_report.attempt}.stdout={line}")
        for line in attempt_report.stderr_tail:
            print(f"flash_attempt.{attempt_report.attempt}.stderr={line}")
        if attempt_report.error:
            print(f"flash_attempt.{attempt_report.attempt}.error={attempt_report.error}")

    for line in report.flash_stdout_tail:
        print(f"cfloader.stdout={line}")
    for line in report.flash_stderr_tail:
        print(f"cfloader.stderr={line}")
    for warning in report.warnings:
        print(f"warning={warning}")
    for failure in report.failures:
        print(f"failure={failure}")


def main() -> int:
    """Parse args, flash one firmware image, and emit the bounded report."""

    args = _build_parser().parse_args()
    report = flash_on_device_failsafe(
        uri=str(args.uri).strip() or DEFAULT_URI,
        workspace=Path(str(args.workspace).strip() or str(DEFAULT_WORKSPACE)),
        binary_path=Path(str(args.binary).strip() or str(DEFAULT_BINARY)),
        cfloader_executable=Path(str(args.cfloader).strip() or str(DEFAULT_CFLOADER)),
        boot_mode=str(args.boot_mode).strip() or "warm",
        flash_timeout_s=max(5.0, float(args.flash_timeout_s)),
        flash_attempts=None if args.flash_attempts is None else max(1, int(args.flash_attempts)),
        flash_retry_backoff_s=max(0.0, float(args.flash_retry_backoff_s)),
        probe_settle_s=max(0.0, float(args.probe_settle_s)),
        probe_timeout_s=max(1.0, float(args.probe_timeout_s)),
        probe_attempts=max(1, int(args.probe_attempts)),
        probe_retry_backoff_s=max(0.0, float(args.probe_retry_backoff_s)),
        probe_link_stats_sample_s=max(0.0, float(args.probe_link_stats_sample_s)),
        lock_timeout_s=max(0.1, float(args.lock_timeout_s)),
        expected_sha256=args.expected_sha256,
        allow_unsafe_cold_boot=bool(args.allow_unsafe_cold_boot),
        skip_probe=bool(args.skip_probe),
    )
    if args.json:
        print(json.dumps(_jsonable(report), indent=2, sort_keys=True))
    else:
        _print_human_report(report)
    return 0 if report.flashed and not report.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
