#!/usr/bin/env python3
# CHANGELOG: 2026-03-27
# BUG-1: Replaced unsafe per-request mission threads with one bounded mission worker queue so only one mission can execute at a time; this also fixes active_mission_id corruption, duplicate manual-arm starts, and mission-id collisions.
# BUG-2: Fixed arm/cancel races, enforced bounded shutdown/cancellation for hover and inspection workers, and stopped leaving helper subprocesses orphaned on SIGINT/SIGTERM.
# SEC-1: Replaced stdlib http.server with FastAPI+Uvicorn, added loopback-by-default network policy plus API-key auth for any non-loopback access, and stopped trusting proxy headers unless explicitly enabled.
# IMP-1: Added strict request/response validation, JSON-only error envelopes, stale-while-refresh probe caching, freshness/confidence safety gates, and just-in-time preflight before mission execution.
# IMP-2: Added bounded outstanding mission depth, atomic artifact writes with tighter permissions, subprocess isolation for stationary inspection, and PEP-723 inline dependencies for reproducible single-file deployment.
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "fastapi>=0.110.0",
#   "uvicorn[standard]>=0.38.0",
#   "pydantic>=2.12.0",
# ]
# ///

"""Serve bounded drone-mission control for Twinr.

Purpose
-------
Expose a strict HTTP boundary between Twinr's slow semantic layer and the
faster drone-control layer. This daemon keeps safety first: it supports
preflight checks, manual arm gating, mission queuing, and bounded stationary
inspection evidence capture by default. When the operator explicitly starts it
in ``bounded_hover_test_only`` mode, it can also run one minimal
takeoff-hover-land test primitive without enabling free navigation.

Usage
-----
Run the daemon in the foreground for local checks::

    python3 hardware/ops/drone_daemon.py --repo-root /home/thh/twinr --env-file /home/thh/twinr/.env
    python3 hardware/ops/drone_daemon.py --pose-provider stub_ok --bind 127.0.0.1 --port 8791
    python3 hardware/ops/drone_daemon.py --pose-provider stub_ok --skill-layer-mode bounded_hover_test_only

    # Remote access now requires an API key and explicit opt-in.
    python3 hardware/ops/drone_daemon.py \
        --bind 0.0.0.0 \
        --allow-remote-reads \
        --allow-remote-ops \
        --api-key-file /run/secrets/twinr_drone_api_key

Inputs
------
- ``GET /healthz`` for a compact health payload
- ``GET /state`` for the full bounded drone state view
- ``GET /pose`` for the current pose-provider sample
- ``POST /missions`` with one bounded mission request
- ``GET /missions/<id>`` to inspect mission state
- ``POST /missions/<id>/cancel`` to abort or clear a queued/running mission
- ``POST /ops/missions/<id>/arm`` for operator-only manual arm approval

Outputs
-------
- JSON payloads only
- bounded stationary inspection artifacts under ``artifacts/ops/drone_missions/``
- bounded hover-test artifacts under ``artifacts/ops/drone_missions/`` when hover mode is enabled
- partial artifacts with worker trace/stdout/stderr when a worker times out or is cancelled
- exit code ``0`` on clean shutdown

Notes
-----
This daemon intentionally rejects direct low-level flight commands. Twinr must
stay above the mission boundary. The default mode remains
``stationary_observe_only``. The optional ``bounded_hover_test_only`` mode is
an explicit operator path for the first takeoff-hover-land test primitive and
is not meant to stand in for a real navigation stack.
"""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hmac
import ipaddress
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import IO, Any, Callable, Generic, Literal, Protocol, TypeVar
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import uuid

from fastapi import Depends, FastAPI, HTTPException, Request as FastAPIRequest, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict
import uvicorn


LOGGER = logging.getLogger(__name__)

_DEFAULT_BIND_HOST = "127.0.0.1"
_DEFAULT_PORT = 8791
_DEFAULT_POSE_TIMEOUT_S = 1.0
_DEFAULT_MISSION_TIMEOUT_S = 45.0
_DEFAULT_MULTIRANGER_TIMEOUT_S = 15.0
_DEFAULT_HOVER_TEST_HEIGHT_M = 0.25
_DEFAULT_HOVER_TEST_DURATION_S = 3.0
_DEFAULT_HOVER_TEST_VELOCITY_MPS = 0.2
_DEFAULT_HOVER_TEST_MIN_VBAT_V = 3.8
_DEFAULT_HOVER_TEST_MIN_BATTERY_LEVEL = 20
_DEFAULT_POSE_CACHE_TTL_S = 0.5
_DEFAULT_RADIO_CACHE_TTL_S = 2.0
_DEFAULT_MAX_POSE_AGE_S = 1.5
_DEFAULT_MIN_POSE_CONFIDENCE = 0.5
_DEFAULT_MAX_OUTSTANDING_MISSIONS = 8
_DEFAULT_LIMIT_CONCURRENCY = 16
_DEFAULT_BACKLOG = 128
_DEFAULT_KEEPALIVE_TIMEOUT_S = 5
_DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_S = 20
_HOVER_TEST_ENABLED_SKILL_MODES = frozenset({"bounded_hover_test_only", "bounded_test_primitives_only"})
_MAX_BODY_BYTES = 16384
_FINAL_MISSION_STATES = frozenset({"completed", "failed", "cancelled"})
_HOVER_WORKER_STDIO_TAIL_CHARS = 4000
_API_KEY_HEADER_NAME = "X-Twinr-Key"
_ALLOWED_TRACKING_STATES = frozenset({"tracking"})

T = TypeVar("T")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(
    value: object,
    *,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            parsed = default
    else:
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _safe_int(
    value: object,
    *,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        parsed = int(value)
    elif isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError:
            parsed = default
    else:
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _coerce_bool(value: object, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _coerce_payload_float(value: object, *, default: float | None = None) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return default
        try:
            return float(normalized)
        except ValueError:
            return default
    return default


def _payload_dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _payload_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [normalized for item in value if (normalized := str(item).strip())]


def _remote_error_code(exc: BaseException) -> str:
    reason = getattr(exc, "reason", None)
    if isinstance(reason, BaseException):
        return reason.__class__.__name__
    if reason is not None:
        normalized = str(reason).strip()
        if normalized:
            return normalized
    return exc.__class__.__name__


def _tail_text(text: str, *, max_chars: int = _HOVER_WORKER_STDIO_TAIL_CHARS) -> str:
    normalized = str(text or "")
    if len(normalized) <= max_chars:
        return normalized
    return normalized[-max_chars:]


def _ensure_src_path(repo_root: Path) -> None:
    src_root = repo_root.resolve(strict=False) / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


def _structured_log(event: str, **fields: object) -> None:
    payload = {"event": event, "ts": _utc_now_iso(), **fields}
    LOGGER.info(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def _new_mission_id() -> str:
    uuid7_fn = getattr(uuid, "uuid7", None)
    generated = uuid7_fn() if callable(uuid7_fn) else uuid.uuid4()
    return f"DRN-{str(generated).upper()}"


def _compare_secret(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def _safe_file_permissions(path: Path, *, mode: int = 0o600) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        return


def _safe_dir_permissions(path: Path, *, mode: int = 0o750) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        return


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _safe_dir_permissions(path.parent)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp_path.write_bytes(data)
    _safe_file_permissions(temp_path)
    temp_path.replace(path)
    _safe_file_permissions(path)


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    _atomic_write_bytes(path, text.encode(encoding))


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _normalize_executable_path(value: Path) -> Path:
    expanded = Path(value).expanduser()
    if expanded.is_absolute():
        return expanded
    return Path.cwd() / expanded


def _extract_last_json_object(text: str) -> dict[str, object] | None:
    normalized = (text or "").strip()
    if not normalized:
        return None
    for candidate in reversed(normalized.splitlines()):
        line = candidate.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _is_loopback_client(host: str | None) -> bool:
    normalized = str(host or "").strip()
    if not normalized:
        return False
    if normalized == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    if ip.is_loopback:
        return True
    mapped = getattr(ip, "ipv4_mapped", None)
    return bool(mapped and mapped.is_loopback)


def _start_process_stream_collector(
    stream: IO[str] | None,
    *,
    name: str,
) -> tuple[list[str], threading.Thread | None]:
    chunks: list[str] = []
    if stream is None:
        return chunks, None

    def _reader() -> None:
        try:
            while True:
                chunk = stream.read(4096)
                if not chunk:
                    break
                chunks.append(str(chunk))
        finally:
            try:
                stream.close()
            except Exception:
                pass

    thread = threading.Thread(target=_reader, name=name, daemon=True)
    thread.start()
    return chunks, thread


def _finish_process_stream_collector(
    chunks: list[str],
    thread: threading.Thread | None,
    *,
    timeout_s: float = 2.0,
) -> str:
    if thread is not None:
        thread.join(timeout=max(0.1, timeout_s))
    return "".join(chunks)


def _interrupt_subprocess(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    except PermissionError:
        try:
            process.send_signal(signal.SIGINT)
        except Exception:
            pass
    try:
        process.wait(timeout=8.0)
        return
    except subprocess.TimeoutExpired:
        pass
    process.terminate()
    try:
        process.wait(timeout=5.0)
        return
    except subprocess.TimeoutExpired:
        pass
    process.kill()
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        return


@dataclass(frozen=True, slots=True)
class PoseSnapshot:
    healthy: bool
    tracking_state: str
    confidence: float
    source_timestamp: float | None = None
    x_m: float | None = None
    y_m: float | None = None
    z_m: float | None = None
    yaw_deg: float | None = None

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


class PoseProvider(Protocol):
    def snapshot(self) -> PoseSnapshot:
        ...


class StubPoseProvider:
    def __init__(self, *, healthy: bool) -> None:
        self._healthy = healthy

    def snapshot(self) -> PoseSnapshot:
        if not self._healthy:
            return PoseSnapshot(
                healthy=False,
                tracking_state="unavailable",
                confidence=0.0,
                source_timestamp=time.time(),
            )
        return PoseSnapshot(
            healthy=True,
            tracking_state="tracking",
            confidence=0.95,
            source_timestamp=time.time(),
            x_m=0.0,
            y_m=0.0,
            z_m=0.0,
            yaw_deg=0.0,
        )


class ExternalHttpPoseProvider:
    def __init__(self, *, base_url: str, timeout_s: float = _DEFAULT_POSE_TIMEOUT_S) -> None:
        normalized = str(base_url or "").strip().rstrip("/")
        if not normalized:
            raise ValueError("external pose provider requires --pose-base-url")
        parsed = urlsplit(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("external pose provider base URL must be http(s)://host[:port]")
        self._base_url = normalized
        self._timeout_s = _safe_float(timeout_s, default=_DEFAULT_POSE_TIMEOUT_S, minimum=0.1)

    def snapshot(self) -> PoseSnapshot:
        request = Request(
            f"{self._base_url}/pose",
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            with urlopen(request, timeout=self._timeout_s) as response:
                charset = response.headers.get_content_charset("utf-8")
                payload = json.loads(response.read().decode(charset, errors="replace"))
        except (HTTPError, URLError, TimeoutError) as exc:
            return PoseSnapshot(
                healthy=False,
                tracking_state=f"transport_error:{_remote_error_code(exc)}",
                confidence=0.0,
                source_timestamp=time.time(),
            )
        except Exception as exc:  # pragma: no cover - defensive path
            return PoseSnapshot(
                healthy=False,
                tracking_state=f"parse_error:{exc.__class__.__name__}",
                confidence=0.0,
                source_timestamp=time.time(),
            )
        if not isinstance(payload, dict):
            return PoseSnapshot(
                healthy=False,
                tracking_state="invalid_payload",
                confidence=0.0,
                source_timestamp=time.time(),
            )
        confidence = _coerce_payload_float(payload.get("confidence"), default=0.0) or 0.0
        source_timestamp = _coerce_payload_float(payload.get("source_timestamp"), default=time.time()) or time.time()
        return PoseSnapshot(
            healthy=_coerce_bool(payload.get("healthy")),
            tracking_state=str(payload.get("tracking_state") or "unavailable"),
            confidence=confidence,
            source_timestamp=source_timestamp,
            x_m=_coerce_payload_float(payload.get("x_m")),
            y_m=_coerce_payload_float(payload.get("y_m")),
            z_m=_coerce_payload_float(payload.get("z_m")),
            yaw_deg=_coerce_payload_float(payload.get("yaw_deg")),
        )


class BitcrazeRadioStatusProvider:
    def __init__(
        self,
        *,
        repo_root: Path,
        workspace: Path,
        bitcraze_python: Path,
    ) -> None:
        self.repo_root = repo_root
        self.workspace = workspace
        self.bitcraze_python = bitcraze_python

    def snapshot(self) -> dict[str, object]:
        script_path = self.repo_root / "hardware" / "bitcraze" / "probe_crazyradio.py"
        probed_at = _utc_now_iso()
        if not script_path.exists():
            return {"radio_ready": False, "error": f"missing_probe:{script_path}", "probed_at": probed_at}
        if not self.bitcraze_python.exists():
            return {
                "radio_ready": False,
                "error": f"missing_bitcraze_python:{self.bitcraze_python}",
                "probed_at": probed_at,
            }
        try:
            result = subprocess.run(
                [
                    str(self.bitcraze_python),
                    str(script_path),
                    "--workspace",
                    str(self.workspace),
                    "--json",
                ],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=12.0,
            )
        except Exception as exc:
            return {
                "radio_ready": False,
                "error": f"probe_exec_failed:{exc.__class__.__name__}",
                "probed_at": probed_at,
            }
        if result.returncode != 0:
            stderr = (result.stderr or "").strip() or "probe_failed"
            return {"radio_ready": False, "error": stderr, "probed_at": probed_at}
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return {"radio_ready": False, "error": "probe_invalid_json", "probed_at": probed_at}
        workspace_payload = payload.get("workspace") if isinstance(payload, dict) else None
        radio_ready = bool(isinstance(workspace_payload, dict) and workspace_payload.get("radio_access_ok"))
        error_text = None if radio_ready else "radio_access_unavailable"
        if isinstance(workspace_payload, dict):
            error_text = workspace_payload.get("radio_access_error") or error_text
        return {
            "radio_ready": radio_ready,
            "radio_version": workspace_payload.get("radio_version") if isinstance(workspace_payload, dict) else None,
            "error": error_text,
            "probed_at": probed_at,
        }


@dataclass(frozen=True, slots=True)
class MissionRecord:
    mission_id: str
    mission_type: str
    target_hint: str
    capture_intent: str
    max_duration_s: float
    return_policy: str
    requires_manual_arm: bool
    state: str
    summary: str
    created_at: str
    updated_at: str
    artifact_name: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    queue_position: int | None = None

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ValidatedMissionRequest:
    mission_type: str
    target_hint: str
    capture_intent: str
    max_duration_s: float
    return_policy: str


@dataclass(frozen=True, slots=True)
class HoverTestSkillConfig:
    height_m: float = _DEFAULT_HOVER_TEST_HEIGHT_M
    hover_duration_s: float = _DEFAULT_HOVER_TEST_DURATION_S
    takeoff_velocity_mps: float = _DEFAULT_HOVER_TEST_VELOCITY_MPS
    land_velocity_mps: float = _DEFAULT_HOVER_TEST_VELOCITY_MPS
    min_vbat_v: float = _DEFAULT_HOVER_TEST_MIN_VBAT_V
    min_battery_level: int = _DEFAULT_HOVER_TEST_MIN_BATTERY_LEVEL


@dataclass(frozen=True, slots=True)
class HoverWorkerRunResult:
    report: dict[str, object]
    trace_file: str | None
    trace_events: tuple[dict[str, object], ...]
    stdout: str
    stderr: str
    return_code: int | None


@dataclass(frozen=True, slots=True)
class SafetyEvaluation:
    can_arm: bool
    manual_arm_required: bool
    radio_ready: bool
    pose_ready: bool
    motion_mode: str
    reasons: tuple[str, ...]
    evaluated_at: str
    pose_age_s: float | None
    pose_confidence: float
    pose_cache_age_s: float
    radio_cache_age_s: float

    def to_payload(self) -> dict[str, object]:
        return {
            "can_arm": self.can_arm,
            "manual_arm_required": self.manual_arm_required,
            "radio_ready": self.radio_ready,
            "pose_ready": self.pose_ready,
            "motion_mode": self.motion_mode,
            "reasons": list(self.reasons),
            "evaluated_at": self.evaluated_at,
            "pose_age_s": self.pose_age_s,
            "pose_confidence": self.pose_confidence,
            "pose_cache_age_s": self.pose_cache_age_s,
            "radio_cache_age_s": self.radio_cache_age_s,
        }


@dataclass(frozen=True, slots=True)
class TimedSnapshot(Generic[T]):
    value: T
    collected_at: str
    collected_monotonic: float


class SnapshotCache(Generic[T]):
    def __init__(self, *, name: str, loader: Callable[[], T], ttl_s: float) -> None:
        self._name = name
        self._loader = loader
        self._ttl_s = max(0.05, ttl_s)
        self._lock = threading.Lock()
        self._value: TimedSnapshot[T] | None = None
        self._refreshing = False

    def get(self, *, force_refresh: bool = False) -> TimedSnapshot[T]:
        now = time.monotonic()
        if force_refresh:
            return self._refresh_sync()
        with self._lock:
            current = self._value
            if current is not None and (now - current.collected_monotonic) < self._ttl_s:
                return current
            if current is None:
                pass
            elif not self._refreshing:
                self._refreshing = True
                thread = threading.Thread(
                    target=self._refresh_background,
                    name=f"cache-refresh-{self._name}",
                    daemon=True,
                )
                thread.start()
                return current
            else:
                return current
        return self._refresh_sync()

    def _refresh_background(self) -> None:
        try:
            self._refresh_sync()
        except Exception:
            LOGGER.exception("Background refresh failed for cache %s.", self._name)
            with self._lock:
                self._refreshing = False

    def _refresh_sync(self) -> TimedSnapshot[T]:
        value = self._loader()
        snapshot = TimedSnapshot(value=value, collected_at=_utc_now_iso(), collected_monotonic=time.monotonic())
        with self._lock:
            self._value = snapshot
            self._refreshing = False
        return snapshot


def _mission_started_summary(mission_type: str) -> str:
    if mission_type == "hover_test":
        return "Bounded hover test started."
    return "Stationary inspection started."


def _mission_completed_summary(mission_type: str) -> str:
    if mission_type == "hover_test":
        return "Bounded hover test completed and landed."
    return "Stationary inspection evidence captured."


def _mission_failure_prefix(mission_type: str) -> str:
    if mission_type == "hover_test":
        return "Hover test failed"
    return "Inspection failed"


def _manual_arm_summary(mission_type: str) -> str:
    if mission_type == "hover_test":
        return "Mission armed locally and queued for bounded hover test."
    return "Mission armed locally and queued for bounded inspection."


class ApiError(RuntimeError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "internal_error"

    def __init__(
        self,
        message: str,
        *,
        error_code: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        if error_code is not None:
            self.error_code = error_code
        if status_code is not None:
            self.status_code = status_code


class InputError(ApiError):
    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "invalid_request"


class ConflictError(ApiError):
    status_code = status.HTTP_409_CONFLICT
    error_code = "conflict"


class NotFoundError(ApiError):
    status_code = status.HTTP_404_NOT_FOUND
    error_code = "not_found"


class ServiceBusyError(ApiError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_code = "service_busy"


class MissionCancelled(RuntimeError):
    def __init__(self, message: str, *, artifact_name: str | None = None) -> None:
        super().__init__(message)
        self.artifact_name = artifact_name


class MissionExecutionError(RuntimeError):
    def __init__(self, message: str, *, artifact_name: str | None = None) -> None:
        super().__init__(message)
        self.artifact_name = artifact_name


class BodyTooLargeError(RuntimeError):
    pass


class DroneDaemonService:
    def __init__(
        self,
        *,
        repo_root: Path,
        env_file: Path | None,
        pose_provider: PoseProvider,
        radio_provider: BitcrazeRadioStatusProvider,
        skill_layer_mode: str = "stationary_observe_only",
        manual_arm_required: bool = True,
        allow_remote_ops: bool = False,
        allow_remote_reads: bool = False,
        api_key: str | None = None,
        multiranger_timeout_s: float = _DEFAULT_MULTIRANGER_TIMEOUT_S,
        hover_test_config: HoverTestSkillConfig | None = None,
        pose_cache_ttl_s: float = _DEFAULT_POSE_CACHE_TTL_S,
        radio_cache_ttl_s: float = _DEFAULT_RADIO_CACHE_TTL_S,
        max_pose_age_s: float = _DEFAULT_MAX_POSE_AGE_S,
        min_pose_confidence: float = _DEFAULT_MIN_POSE_CONFIDENCE,
        max_outstanding_missions: int = _DEFAULT_MAX_OUTSTANDING_MISSIONS,
    ) -> None:
        resolved_repo_root = repo_root.resolve(strict=False)
        _ensure_src_path(resolved_repo_root)
        from twinr.agent.base_agent.config import TwinrConfig

        self.repo_root = resolved_repo_root
        self.env_file = env_file
        self.pose_provider = pose_provider
        self.radio_provider = radio_provider
        self.skill_layer_mode = str(skill_layer_mode or "stationary_observe_only").strip() or "stationary_observe_only"
        self.manual_arm_required = bool(manual_arm_required)
        self.allow_remote_ops = bool(allow_remote_ops)
        self.allow_remote_reads = bool(allow_remote_reads)
        self.api_key = str(api_key or "").strip() or None
        self.multiranger_timeout_s = _safe_float(
            multiranger_timeout_s,
            default=_DEFAULT_MULTIRANGER_TIMEOUT_S,
            minimum=1.0,
        )
        self.hover_test_config = hover_test_config or HoverTestSkillConfig()
        self.max_pose_age_s = _safe_float(max_pose_age_s, default=_DEFAULT_MAX_POSE_AGE_S, minimum=0.1)
        self.min_pose_confidence = _safe_float(
            min_pose_confidence,
            default=_DEFAULT_MIN_POSE_CONFIDENCE,
            minimum=0.0,
            maximum=1.0,
        )
        self.max_outstanding_missions = _safe_int(
            max_outstanding_missions,
            default=_DEFAULT_MAX_OUTSTANDING_MISSIONS,
            minimum=1,
            maximum=256,
        )
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._missions: dict[str, MissionRecord] = {}
        self._queue: deque[str] = deque()
        self._cancel_requested: set[str] = set()
        self._active_mission_id: str | None = None
        self._active_process: subprocess.Popen[str] | None = None
        self._active_process_kind: str | None = None
        self._accepting_new_missions = True
        self._shutdown_requested = False
        self._lifecycle_state = "inactive"
        self._worker_thread: threading.Thread | None = None
        if env_file is None:
            self.config = TwinrConfig(project_root=str(resolved_repo_root))
        else:
            self.config = TwinrConfig.from_env(env_file)
        self._artifact_root = Path(self.config.project_root) / "artifacts" / "ops" / "drone_missions"
        self._self_script_path = Path(__file__).resolve(strict=False)
        self._pose_cache: SnapshotCache[PoseSnapshot] = SnapshotCache(
            name="pose",
            loader=self.pose_provider.snapshot,
            ttl_s=pose_cache_ttl_s,
        )
        self._radio_cache: SnapshotCache[dict[str, object]] = SnapshotCache(
            name="radio",
            loader=self.radio_provider.snapshot,
            ttl_s=radio_cache_ttl_s,
        )

    def start(self) -> None:
        with self._cond:
            if self._worker_thread is not None:
                return
            self._lifecycle_state = "active"
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="drone-mission-worker",
                daemon=False,
            )
            self._worker_thread.start()
        _structured_log(
            "daemon_started",
            skill_layer_mode=self.skill_layer_mode,
            manual_arm_required=self.manual_arm_required,
        )

    def shutdown(self) -> None:
        worker: threading.Thread | None
        active_process: subprocess.Popen[str] | None = None
        with self._cond:
            if self._shutdown_requested:
                worker = self._worker_thread
            else:
                self._shutdown_requested = True
                self._accepting_new_missions = False
                self._lifecycle_state = "finalizing"
                now = _utc_now_iso()

                queued_ids = list(self._queue)
                self._queue.clear()
                for mission_id in queued_ids:
                    mission = self._missions.get(mission_id)
                    if mission is None or mission.state != "queued":
                        continue
                    self._store_mission_locked(
                        replace(
                            mission,
                            state="cancelled",
                            summary="Mission cancelled during daemon shutdown before execution.",
                            updated_at=now,
                            finished_at=now,
                        )
                    )

                for mission in list(self._missions.values()):
                    if mission.state == "pending_manual_arm":
                        self._store_mission_locked(
                            replace(
                                mission,
                                state="cancelled",
                                summary="Mission cancelled during daemon shutdown before arming.",
                                updated_at=now,
                                finished_at=now,
                            )
                        )

                if self._active_mission_id is not None:
                    self._cancel_requested.add(self._active_mission_id)
                    active_process = self._active_process

                self._cond.notify_all()
                worker = self._worker_thread
        if active_process is not None:
            _interrupt_subprocess(active_process)
        if worker is not None and worker.is_alive():
            worker.join(timeout=_DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_S)
        with self._cond:
            self._lifecycle_state = "finalized"
        _structured_log("daemon_stopped")

    def _hover_test_enabled(self) -> bool:
        return self.skill_layer_mode in _HOVER_TEST_ENABLED_SKILL_MODES

    def _sync_pose_snapshot(self, *, force_refresh: bool) -> TimedSnapshot[PoseSnapshot]:
        return self._pose_cache.get(force_refresh=force_refresh)

    def _sync_radio_snapshot(self, *, force_refresh: bool) -> TimedSnapshot[dict[str, object]]:
        return self._radio_cache.get(force_refresh=force_refresh)

    def _evaluate_safety(
        self,
        *,
        force_refresh: bool,
    ) -> tuple[TimedSnapshot[dict[str, object]], TimedSnapshot[PoseSnapshot], SafetyEvaluation]:
        radio_snapshot = self._sync_radio_snapshot(force_refresh=force_refresh)
        pose_snapshot = self._sync_pose_snapshot(force_refresh=force_refresh)
        radio = radio_snapshot.value
        pose = pose_snapshot.value
        reasons: list[str] = []

        if self._lifecycle_state != "active":
            reasons.append(f"service_{self._lifecycle_state}")

        radio_ready = bool(radio.get("radio_ready"))
        if not radio_ready:
            reasons.append(str(radio.get("error") or "radio_unavailable"))

        pose_age_s = None
        if pose.source_timestamp is None:
            reasons.append("pose_missing_timestamp")
        else:
            pose_age_s = max(0.0, time.time() - pose.source_timestamp)
            if pose_age_s > self.max_pose_age_s:
                reasons.append(f"pose_stale:{pose_age_s:.2f}s")

        if not pose.healthy:
            reasons.append(str(pose.tracking_state or "pose_unavailable"))
        if pose.tracking_state not in _ALLOWED_TRACKING_STATES:
            reasons.append(f"pose_not_tracking:{pose.tracking_state}")
        if pose.confidence < self.min_pose_confidence:
            reasons.append(f"pose_confidence_low:{pose.confidence:.2f}")

        pose_ready = (
            pose.healthy
            and pose.tracking_state in _ALLOWED_TRACKING_STATES
            and pose.confidence >= self.min_pose_confidence
            and pose_age_s is not None
            and pose_age_s <= self.max_pose_age_s
        )
        evaluation = SafetyEvaluation(
            can_arm=radio_ready and pose_ready and self._lifecycle_state == "active",
            manual_arm_required=self.manual_arm_required,
            radio_ready=radio_ready,
            pose_ready=pose_ready,
            motion_mode=self.skill_layer_mode,
            reasons=tuple(dict.fromkeys(reasons)),
            evaluated_at=_utc_now_iso(),
            pose_age_s=pose_age_s,
            pose_confidence=pose.confidence,
            pose_cache_age_s=max(0.0, time.monotonic() - pose_snapshot.collected_monotonic),
            radio_cache_age_s=max(0.0, time.monotonic() - radio_snapshot.collected_monotonic),
        )
        return radio_snapshot, pose_snapshot, evaluation

    def health_payload(self) -> dict[str, object]:
        state = self.state_payload()
        safety = _payload_dict(state.get("safety"))
        return {
            "ok": self._lifecycle_state == "active",
            "service": "drone_daemon",
            "manual_arm_required": self.manual_arm_required,
            "skill_layer_mode": self.skill_layer_mode,
            "lifecycle_state": self._lifecycle_state,
            "radio_ready": bool(safety.get("radio_ready")),
            "pose_ready": bool(safety.get("pose_ready")),
            "can_arm": bool(safety.get("can_arm")),
            "active_mission_id": state.get("active_mission_id"),
            "queue_depth": state.get("queue_depth"),
            "accepting_new_missions": state.get("accepting_new_missions"),
        }

    def pose_payload(self) -> dict[str, object]:
        pose_snapshot = self._sync_pose_snapshot(force_refresh=False)
        return pose_snapshot.value.to_payload()

    def state_payload(self) -> dict[str, object]:
        _radio_snapshot, pose_snapshot, safety = self._evaluate_safety(force_refresh=False)
        with self._lock:
            active_mission_id = self._active_mission_id
            queue_depth = sum(
                1
                for mid in self._queue
                if self._missions.get(mid) and self._missions[mid].state == "queued"
            )
            pending_manual_arm = sum(
                1
                for mission in self._missions.values()
                if mission.state == "pending_manual_arm"
            )
            accepting_new_missions = self._accepting_new_missions and not self._shutdown_requested
        return {
            "service_status": "ready" if self._lifecycle_state == "active" else self._lifecycle_state,
            "lifecycle_state": self._lifecycle_state,
            "active_mission_id": active_mission_id,
            "manual_arm_required": self.manual_arm_required,
            "skill_layer_mode": self.skill_layer_mode,
            "radio_ready": safety.radio_ready,
            "queue_depth": queue_depth,
            "pending_manual_arm_count": pending_manual_arm,
            "accepting_new_missions": accepting_new_missions,
            "pose": pose_snapshot.value.to_payload(),
            "safety": safety.to_payload(),
        }

    def _assert_accepting_new_missions(self) -> None:
        if self._shutdown_requested or not self._accepting_new_missions:
            raise ServiceBusyError("Mission service is shutting down and not accepting new missions.")

    def _ensure_outstanding_capacity_locked(self) -> None:
        outstanding = sum(1 for mission in self._missions.values() if mission.state not in _FINAL_MISSION_STATES)
        if outstanding >= self.max_outstanding_missions:
            raise ServiceBusyError("Mission queue is full.")

    def create_mission(self, payload: dict[str, object]) -> MissionRecord:
        request = self._validate_mission_payload(payload)
        _radio_snapshot, _pose_snapshot, safety = self._evaluate_safety(force_refresh=True)
        if not safety.can_arm:
            reasons = ", ".join(safety.reasons) or "preflight_failed"
            raise ConflictError(f"Mission preflight failed: {reasons}")
        with self._cond:
            self._assert_accepting_new_missions()
            self._ensure_outstanding_capacity_locked()
            now = _utc_now_iso()
            mission = MissionRecord(
                mission_id=_new_mission_id(),
                mission_type=request.mission_type,
                target_hint=request.target_hint,
                capture_intent=request.capture_intent,
                max_duration_s=request.max_duration_s,
                return_policy=request.return_policy,
                requires_manual_arm=self.manual_arm_required,
                state="pending_manual_arm" if self.manual_arm_required else "queued",
                summary=(
                    "Mission queued and waiting for local manual arm approval."
                    if self.manual_arm_required
                    else "Mission accepted and queued for execution."
                ),
                created_at=now,
                updated_at=now,
            )
            self._missions[mission.mission_id] = mission
            if not self.manual_arm_required:
                self._queue.append(mission.mission_id)
                self._cond.notify_all()
            mission = self._with_queue_position_locked(mission)
        _structured_log(
            "mission_created",
            mission_id=mission.mission_id,
            mission_type=mission.mission_type,
            state=mission.state,
        )
        return mission

    def mission_payload(self, mission_id: str) -> dict[str, object]:
        with self._lock:
            return self._with_queue_position_locked(self._require_mission_locked(mission_id)).to_payload()

    def cancel_mission(self, mission_id: str) -> MissionRecord:
        process_to_interrupt: subprocess.Popen[str] | None = None
        with self._cond:
            mission = self._require_mission_locked(mission_id)
            if mission.state in _FINAL_MISSION_STATES:
                return self._with_queue_position_locked(mission)
            now = _utc_now_iso()
            if mission.state == "pending_manual_arm":
                mission = self._store_mission_locked(
                    replace(
                        mission,
                        state="cancelled",
                        summary="Mission cancelled before arming.",
                        updated_at=now,
                        finished_at=now,
                    )
                )
                self._cond.notify_all()
                return self._with_queue_position_locked(mission)
            if mission.state == "queued":
                mission = self._store_mission_locked(
                    replace(
                        mission,
                        state="cancelled",
                        summary="Mission cancelled before execution.",
                        updated_at=now,
                        finished_at=now,
                    )
                )
                self._cancel_requested.discard(mission_id)
                self._cond.notify_all()
                return self._with_queue_position_locked(mission)
            if mission.state == "cancel_requested":
                return self._with_queue_position_locked(mission)
            self._cancel_requested.add(mission_id)
            mission = self._store_mission_locked(
                replace(
                    mission,
                    state="cancel_requested",
                    summary="Mission cancellation requested.",
                    updated_at=now,
                )
            )
            if self._active_mission_id == mission_id and self._active_process is not None:
                process_to_interrupt = self._active_process
            self._cond.notify_all()
        if process_to_interrupt is not None:
            _interrupt_subprocess(process_to_interrupt)
        _structured_log("mission_cancel_requested", mission_id=mission_id)
        return mission

    def manual_arm(self, mission_id: str) -> MissionRecord:
        _radio_snapshot, _pose_snapshot, safety = self._evaluate_safety(force_refresh=True)
        if not safety.can_arm:
            reasons = ", ".join(safety.reasons) or "preflight_failed"
            raise ConflictError(f"Mission cannot arm: {reasons}")
        with self._cond:
            mission = self._require_mission_locked(mission_id)
            if mission.state != "pending_manual_arm":
                raise ConflictError(f"Mission {mission_id} is not waiting for manual arm.")
            self._assert_accepting_new_missions()
            mission = self._store_mission_locked(
                replace(
                    mission,
                    # BREAKING: armed missions now enter a serialized queue instead of spawning immediately.
                    state="queued",
                    summary=_manual_arm_summary(mission.mission_type),
                    updated_at=_utc_now_iso(),
                )
            )
            self._queue.append(mission.mission_id)
            self._cond.notify_all()
            mission = self._with_queue_position_locked(mission)
        _structured_log("mission_armed", mission_id=mission_id, mission_type=mission.mission_type)
        return mission

    def _validate_mission_payload(self, payload: dict[str, object]) -> ValidatedMissionRequest:
        mission_type = str(payload.get("mission_type") or "").strip().lower()
        if mission_type == "inspect":
            target_hint = str(payload.get("target_hint") or "").strip()
            if not target_hint:
                raise InputError("target_hint must not be empty.")
            capture_intent = str(payload.get("capture_intent") or "scene").strip().lower() or "scene"
            if capture_intent not in {"scene", "object_check", "look_closer"}:
                raise InputError("capture_intent must be one of: scene, object_check, look_closer.")
            return_policy = str(payload.get("return_policy") or "return_and_land").strip().lower() or "return_and_land"
            if return_policy not in {"return_and_land"}:
                raise InputError("return_policy must currently be `return_and_land`.")
            max_duration_s = _safe_float(
                payload.get("max_duration_s"),
                default=_DEFAULT_MISSION_TIMEOUT_S,
                minimum=5.0,
            )
            return ValidatedMissionRequest(
                mission_type=mission_type,
                target_hint=target_hint,
                capture_intent=capture_intent,
                max_duration_s=min(max_duration_s, 120.0),
                return_policy=return_policy,
            )
        if mission_type == "hover_test":
            if not self._hover_test_enabled():
                raise ConflictError(
                    "hover_test missions require the daemon to run in `bounded_hover_test_only` skill mode."
                )
            target_hint = str(payload.get("target_hint") or "bounded hover test").strip() or "bounded hover test"
            return_policy = str(payload.get("return_policy") or "return_and_land").strip().lower() or "return_and_land"
            if return_policy not in {"return_and_land"}:
                raise InputError("return_policy must currently be `return_and_land`.")
            max_duration_s = _safe_float(payload.get("max_duration_s"), default=20.0, minimum=10.0)
            return ValidatedMissionRequest(
                mission_type=mission_type,
                target_hint=target_hint,
                capture_intent="hover_test",
                max_duration_s=min(max_duration_s, 60.0),
                return_policy=return_policy,
            )
        raise InputError("mission_type must currently be `inspect` or `hover_test`.")

    def _with_queue_position_locked(self, mission: MissionRecord) -> MissionRecord:
        if mission.state != "queued":
            return replace(mission, queue_position=None)
        position = 1
        for queued_mission_id in self._queue:
            queued_mission = self._missions.get(queued_mission_id)
            if queued_mission is None or queued_mission.state != "queued":
                continue
            if queued_mission_id == mission.mission_id:
                return replace(mission, queue_position=position)
            position += 1
        return replace(mission, queue_position=None)

    def _get_mission(self, mission_id: str) -> MissionRecord:
        with self._lock:
            return self._with_queue_position_locked(self._require_mission_locked(mission_id))

    def _require_mission_locked(self, mission_id: str) -> MissionRecord:
        normalized_id = str(mission_id or "").strip()
        mission = self._missions.get(normalized_id)
        if mission is None:
            raise NotFoundError(f"Unknown mission: {normalized_id}")
        return mission

    def _store_mission_locked(self, mission: MissionRecord) -> MissionRecord:
        self._missions[mission.mission_id] = mission
        return mission

    def _is_cancel_requested(self, mission_id: str) -> bool:
        with self._lock:
            return mission_id in self._cancel_requested or self._shutdown_requested

    def _worker_loop(self) -> None:
        while True:
            with self._cond:
                mission: MissionRecord | None = None
                while mission is None:
                    if self._shutdown_requested and not self._queue and self._active_mission_id is None:
                        return
                    while self._queue:
                        mission_id = self._queue.popleft()
                        candidate = self._missions.get(mission_id)
                        if candidate is None or candidate.state in _FINAL_MISSION_STATES:
                            continue
                        if candidate.state != "queued":
                            continue
                        if mission_id in self._cancel_requested:
                            now = _utc_now_iso()
                            self._store_mission_locked(
                                replace(
                                    candidate,
                                    state="cancelled",
                                    summary="Mission cancelled before execution.",
                                    updated_at=now,
                                    finished_at=now,
                                )
                            )
                            self._cancel_requested.discard(mission_id)
                            continue
                        self._active_mission_id = mission_id
                        mission = self._store_mission_locked(
                            replace(
                                candidate,
                                state="starting",
                                summary="Mission reserved for execution.",
                                updated_at=_utc_now_iso(),
                                queue_position=None,
                            )
                        )
                        break
                    if mission is not None:
                        break
                    self._cond.wait(timeout=0.5)
            try:
                self._execute_mission(mission)
            finally:
                with self._cond:
                    self._active_mission_id = None
                    self._active_process = None
                    self._active_process_kind = None
                    self._cancel_requested.discard(mission.mission_id)
                    self._cond.notify_all()

    def _execute_mission(self, mission: MissionRecord) -> None:
        if self._is_cancel_requested(mission.mission_id):
            now = _utc_now_iso()
            with self._lock:
                self._store_mission_locked(
                    replace(
                        mission,
                        state="cancelled",
                        summary="Mission cancelled before capture.",
                        updated_at=now,
                        finished_at=now,
                    )
                )
            return

        _radio_snapshot, _pose_snapshot, safety = self._evaluate_safety(force_refresh=True)
        if not safety.can_arm:
            reasons = ", ".join(safety.reasons) or "preflight_failed"
            now = _utc_now_iso()
            with self._lock:
                self._store_mission_locked(
                    replace(
                        mission,
                        state="failed",
                        summary=f"{_mission_failure_prefix(mission.mission_type)}: preflight failed before execution: {reasons}",
                        updated_at=now,
                        finished_at=now,
                    )
                )
            return

        with self._lock:
            mission = self._store_mission_locked(
                replace(
                    mission,
                    state="running",
                    summary=_mission_started_summary(mission.mission_type),
                    updated_at=_utc_now_iso(),
                    started_at=_utc_now_iso(),
                )
            )
        _structured_log("mission_started", mission_id=mission.mission_id, mission_type=mission.mission_type)

        try:
            if mission.mission_type == "hover_test":
                artifact_name = self._run_hover_test_mission(mission)
                if self._is_cancel_requested(mission.mission_id):
                    raise MissionCancelled("Mission cancelled during bounded hover test.", artifact_name=artifact_name)
            else:
                artifact_name = self._run_stationary_inspection_mission(mission)
                if self._is_cancel_requested(mission.mission_id):
                    raise MissionCancelled("Mission cancelled during bounded inspection.", artifact_name=artifact_name)

            now = _utc_now_iso()
            with self._lock:
                self._store_mission_locked(
                    replace(
                        mission,
                        state="completed",
                        summary=_mission_completed_summary(mission.mission_type),
                        updated_at=now,
                        finished_at=now,
                        artifact_name=artifact_name,
                    )
                )
            _structured_log("mission_completed", mission_id=mission.mission_id, artifact_name=artifact_name)

        except MissionCancelled as exc:
            now = _utc_now_iso()
            with self._lock:
                self._store_mission_locked(
                    replace(
                        mission,
                        state="cancelled",
                        summary=str(exc),
                        updated_at=now,
                        finished_at=now,
                        artifact_name=exc.artifact_name or mission.artifact_name,
                    )
                )
            _structured_log("mission_cancelled", mission_id=mission.mission_id, artifact_name=exc.artifact_name)

        except Exception as exc:  # pragma: no cover - live hardware path
            LOGGER.exception("Drone mission %s failed.", mission.mission_id)
            now = _utc_now_iso()
            with self._lock:
                self._store_mission_locked(
                    replace(
                        mission,
                        state="failed",
                        summary=f"{_mission_failure_prefix(mission.mission_type)}: {exc}",
                        updated_at=now,
                        finished_at=now,
                        artifact_name=getattr(exc, "artifact_name", None) or mission.artifact_name,
                    )
                )
            _structured_log("mission_failed", mission_id=mission.mission_id, error=str(exc))

    def _run_stationary_inspection_mission(self, mission: MissionRecord) -> str:
        self._artifact_root.mkdir(parents=True, exist_ok=True)
        _safe_dir_permissions(self._artifact_root)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        report_name = f"{mission.mission_id.lower()}-{stamp}.json"
        image_name = f"{mission.mission_id.lower()}-{stamp}.png"
        return self._run_inspection_worker(
            mission,
            stamp=stamp,
            report_name=report_name,
            image_name=image_name,
        )

    def _run_inspection_worker(
        self,
        mission: MissionRecord,
        *,
        stamp: str,
        report_name: str,
        image_name: str,
    ) -> str:
        command = [
            str(sys.executable),
            "-u",
            str(self._self_script_path),
            "--internal-run-inspection-worker",
            "--repo-root",
            str(self.repo_root),
            "--artifact-root",
            str(self._artifact_root),
            "--mission-id",
            mission.mission_id,
            "--target-hint",
            mission.target_hint,
            "--capture-intent",
            mission.capture_intent,
            "--max-duration-s",
            f"{mission.max_duration_s:.3f}",
            "--return-policy",
            mission.return_policy,
            "--requires-manual-arm",
            "1" if mission.requires_manual_arm else "0",
            "--created-at",
            mission.created_at,
            "--updated-at",
            mission.updated_at,
            "--report-name",
            report_name,
            "--image-name",
            image_name,
            "--bitcraze-workspace",
            str(self.radio_provider.workspace),
            "--bitcraze-python",
            str(self.radio_provider.bitcraze_python),
            "--multiranger-timeout-s",
            f"{self.multiranger_timeout_s:.3f}",
        ]
        if self.env_file is not None:
            command.extend(["--env-file", str(self.env_file)])

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
        with self._lock:
            self._active_process = process
            self._active_process_kind = "inspection"

        stdout_chunks, stdout_collector = _start_process_stream_collector(
            process.stdout,
            name="inspection-worker-stdout",
        )
        stderr_chunks, stderr_collector = _start_process_stream_collector(
            process.stderr,
            name="inspection-worker-stderr",
        )

        cancellation_requested = False
        deadline_exceeded = False
        return_code: int | None = None
        deadline = time.monotonic() + max(5.0, mission.max_duration_s)

        while True:
            try:
                return_code = process.wait(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                if self._is_cancel_requested(mission.mission_id):
                    cancellation_requested = True
                    _interrupt_subprocess(process)
                    break
                if time.monotonic() > deadline:
                    cancellation_requested = True
                    deadline_exceeded = True
                    _interrupt_subprocess(process)
                    break

        stdout = _finish_process_stream_collector(stdout_chunks, stdout_collector)
        stderr = _finish_process_stream_collector(stderr_chunks, stderr_collector)
        return_code = int(process.returncode) if process.returncode is not None else return_code
        parsed_payload = _extract_last_json_object(stdout) or {}

        if cancellation_requested:
            summary = (
                "Inspection exceeded its bounded runtime. Capture worker interrupted."
                if deadline_exceeded
                else "Inspection cancelled. Capture worker interrupted."
            )
            artifact_name = self._persist_inspection_partial_artifact(
                mission,
                stamp=stamp,
                summary=summary,
                command=command,
                stdout=stdout,
                stderr=stderr,
                return_code=return_code,
                cancellation_requested=True,
                deadline_exceeded=deadline_exceeded,
                image_name=image_name,
                parsed_payload=parsed_payload,
            )
            raise MissionCancelled(summary, artifact_name=artifact_name)

        if return_code == 130:
            summary = "Inspection interrupted. Capture worker stopped."
            artifact_name = self._persist_inspection_partial_artifact(
                mission,
                stamp=stamp,
                summary=summary,
                command=command,
                stdout=stdout,
                stderr=stderr,
                return_code=return_code,
                cancellation_requested=True,
                deadline_exceeded=False,
                image_name=image_name,
                parsed_payload=parsed_payload,
            )
            raise MissionCancelled(summary, artifact_name=artifact_name)

        if return_code != 0:
            detail = str(
                parsed_payload.get("error")
                or (stderr or stdout).strip()
                or f"inspection_worker_failed:{return_code}"
            )
            artifact_name = self._persist_inspection_partial_artifact(
                mission,
                stamp=stamp,
                summary=detail,
                command=command,
                stdout=stdout,
                stderr=stderr,
                return_code=return_code,
                cancellation_requested=False,
                deadline_exceeded=False,
                image_name=image_name,
                parsed_payload=parsed_payload,
            )
            raise MissionExecutionError(detail, artifact_name=artifact_name)

        worker_report_name = str(parsed_payload.get("report_name") or "").strip()
        if worker_report_name != report_name:
            summary = "inspection_worker_missing_report"
            artifact_name = self._persist_inspection_partial_artifact(
                mission,
                stamp=stamp,
                summary=summary,
                command=command,
                stdout=stdout,
                stderr=stderr,
                return_code=return_code,
                cancellation_requested=False,
                deadline_exceeded=False,
                image_name=image_name,
                parsed_payload=parsed_payload,
            )
            raise MissionExecutionError(summary, artifact_name=artifact_name)

        return report_name

    def _persist_inspection_partial_artifact(
        self,
        mission: MissionRecord,
        *,
        stamp: str,
        summary: str,
        command: list[str],
        stdout: str,
        stderr: str,
        return_code: int | None,
        cancellation_requested: bool,
        deadline_exceeded: bool,
        image_name: str,
        parsed_payload: dict[str, object],
    ) -> str:
        artifact_payload = {
            "mission": mission.to_payload(),
            "completed_at": _utc_now_iso(),
            "summary": summary,
            "partial": True,
            "camera": {"image_file": image_name},
            "inspection_worker_diagnostics": {
                "command": command,
                "return_code": return_code,
                "cancellation_requested": cancellation_requested,
                "deadline_exceeded": deadline_exceeded,
                "stdout_tail": _tail_text(stdout),
                "stderr_tail": _tail_text(stderr),
                "worker_payload": parsed_payload,
            },
        }
        report_name = f"{mission.mission_id.lower()}-{stamp}.partial.json"
        _atomic_write_json(self._artifact_root / report_name, artifact_payload)
        return report_name

    def _run_hover_test_mission(self, mission: MissionRecord) -> str:
        self._artifact_root.mkdir(parents=True, exist_ok=True)
        _safe_dir_permissions(self._artifact_root)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        worker_result = self._run_hover_test_worker(mission, stamp=stamp)
        artifact_payload = {
            "mission": mission.to_payload(),
            "completed_at": _utc_now_iso(),
            "summary": "Bounded hover test completed and landed.",
            "hover_test": worker_result.report,
            "hover_worker_diagnostics": {
                "trace_file": worker_result.trace_file,
                "trace_events": list(worker_result.trace_events),
                "trace_event_count": len(worker_result.trace_events),
                "last_trace_phase": (
                    str(worker_result.trace_events[-1].get("phase")) if worker_result.trace_events else None
                ),
                "last_trace_status": (
                    str(worker_result.trace_events[-1].get("status")) if worker_result.trace_events else None
                ),
                "return_code": worker_result.return_code,
                "stdout_tail": _tail_text(worker_result.stdout),
                "stderr_tail": _tail_text(worker_result.stderr),
            },
        }
        report_name = f"{mission.mission_id.lower()}-{stamp}.json"
        _atomic_write_json(self._artifact_root / report_name, artifact_payload)
        return report_name

    @staticmethod
    def _read_hover_worker_trace(trace_path: Path) -> tuple[dict[str, object], ...]:
        if not trace_path.exists():
            return ()
        events: list[dict[str, object]] = []
        for raw_line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                events.append(payload)
        return tuple(events)

    def _persist_hover_worker_partial_artifact(
        self,
        mission: MissionRecord,
        *,
        stamp: str,
        summary: str,
        command: list[str],
        trace_path: Path,
        stdout: str,
        stderr: str,
        return_code: int | None,
        parsed_payload: dict[str, object],
        cancellation_requested: bool,
        deadline_exceeded: bool,
    ) -> str:
        trace_events = self._read_hover_worker_trace(trace_path)
        artifact_payload = {
            "mission": mission.to_payload(),
            "completed_at": _utc_now_iso(),
            "summary": summary,
            "partial": True,
            "hover_test": parsed_payload.get("report") if isinstance(parsed_payload.get("report"), dict) else None,
            "hover_worker_diagnostics": {
                "command": command,
                "trace_file": trace_path.name,
                "trace_events": list(trace_events),
                "trace_event_count": len(trace_events),
                "last_trace_phase": str(trace_events[-1].get("phase")) if trace_events else None,
                "last_trace_status": str(trace_events[-1].get("status")) if trace_events else None,
                "return_code": return_code,
                "cancellation_requested": cancellation_requested,
                "deadline_exceeded": deadline_exceeded,
                "stdout_tail": _tail_text(stdout),
                "stderr_tail": _tail_text(stderr),
                "failures": parsed_payload.get("failures") if isinstance(parsed_payload.get("failures"), list) else [],
            },
        }
        report_name = f"{mission.mission_id.lower()}-{stamp}.partial.json"
        _atomic_write_json(self._artifact_root / report_name, artifact_payload)
        return report_name

    def _run_hover_test_worker(self, mission: MissionRecord, *, stamp: str) -> HoverWorkerRunResult:
        bitcraze_python = self.radio_provider.bitcraze_python
        script_path = self.repo_root / "hardware" / "bitcraze" / "run_hover_test.py"
        if not bitcraze_python.exists():
            raise RuntimeError(f"hover_test_worker_missing_python:{bitcraze_python}")
        if not script_path.exists():
            raise RuntimeError(f"hover_test_worker_missing_script:{script_path}")

        trace_path = self._artifact_root / f"{mission.mission_id.lower()}-{stamp}.hover-trace.jsonl"
        command = [
            str(bitcraze_python),
            str(script_path),
            "--workspace",
            str(self.radio_provider.workspace),
            "--height-m",
            f"{self.hover_test_config.height_m:.3f}",
            "--hover-duration-s",
            f"{self.hover_test_config.hover_duration_s:.3f}",
            "--takeoff-velocity-mps",
            f"{self.hover_test_config.takeoff_velocity_mps:.3f}",
            "--land-velocity-mps",
            f"{self.hover_test_config.land_velocity_mps:.3f}",
            "--min-vbat-v",
            f"{self.hover_test_config.min_vbat_v:.3f}",
            "--min-battery-level",
            str(self.hover_test_config.min_battery_level),
            "--trace-file",
            str(trace_path),
            "--json",
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
        with self._lock:
            self._active_process = process
            self._active_process_kind = "hover_test"

        stdout_chunks, stdout_collector = _start_process_stream_collector(
            process.stdout,
            name="hover-worker-stdout",
        )
        stderr_chunks, stderr_collector = _start_process_stream_collector(
            process.stderr,
            name="hover-worker-stderr",
        )

        cancellation_requested = False
        deadline_exceeded = False
        cancel_summary: str | None = None
        return_code: int | None = None
        deadline = time.monotonic() + max(5.0, mission.max_duration_s)

        while True:
            try:
                return_code = process.wait(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                if self._is_cancel_requested(mission.mission_id):
                    cancellation_requested = True
                    _interrupt_subprocess(process)
                    cancel_summary = "Hover test cancelled. Landing requested."
                    break
                if time.monotonic() > deadline:
                    cancellation_requested = True
                    deadline_exceeded = True
                    _interrupt_subprocess(process)
                    cancel_summary = "Hover test exceeded its bounded runtime. Landing requested."
                    break

        stdout = _finish_process_stream_collector(stdout_chunks, stdout_collector)
        stderr = _finish_process_stream_collector(stderr_chunks, stderr_collector)
        return_code = int(process.returncode) if process.returncode is not None else return_code
        payload = self._parse_hover_worker_payload(stdout=stdout, stderr=stderr)
        trace_events = self._read_hover_worker_trace(trace_path)
        report = payload.get("report")

        if cancel_summary is not None:
            artifact_name = self._persist_hover_worker_partial_artifact(
                mission,
                stamp=stamp,
                summary=cancel_summary,
                command=command,
                trace_path=trace_path,
                stdout=stdout,
                stderr=stderr,
                return_code=return_code,
                parsed_payload=payload,
                cancellation_requested=cancellation_requested,
                deadline_exceeded=deadline_exceeded,
            )
            raise MissionCancelled(cancel_summary, artifact_name=artifact_name)

        if return_code == 130:
            summary = "Hover test interrupted. Landing requested."
            artifact_name = self._persist_hover_worker_partial_artifact(
                mission,
                stamp=stamp,
                summary=summary,
                command=command,
                trace_path=trace_path,
                stdout=stdout,
                stderr=stderr,
                return_code=return_code,
                parsed_payload=payload,
                cancellation_requested=True,
                deadline_exceeded=False,
            )
            raise MissionCancelled(summary, artifact_name=artifact_name)

        if return_code != 0:
            failures = payload.get("failures")
            if isinstance(failures, list) and failures:
                summary = "; ".join(str(item) for item in failures if str(item).strip())
            else:
                summary = (stderr or stdout).strip() or f"hover_test_worker_failed:{return_code}"
            artifact_name = self._persist_hover_worker_partial_artifact(
                mission,
                stamp=stamp,
                summary=summary,
                command=command,
                trace_path=trace_path,
                stdout=stdout,
                stderr=stderr,
                return_code=return_code,
                parsed_payload=payload,
                cancellation_requested=False,
                deadline_exceeded=False,
            )
            raise MissionExecutionError(summary, artifact_name=artifact_name)

        if not isinstance(report, dict):
            summary = "hover_test_worker_missing_report"
            artifact_name = self._persist_hover_worker_partial_artifact(
                mission,
                stamp=stamp,
                summary=summary,
                command=command,
                trace_path=trace_path,
                stdout=stdout,
                stderr=stderr,
                return_code=return_code,
                parsed_payload=payload,
                cancellation_requested=False,
                deadline_exceeded=False,
            )
            raise MissionExecutionError(summary, artifact_name=artifact_name)

        return HoverWorkerRunResult(
            report=report,
            trace_file=trace_path.name,
            trace_events=trace_events,
            stdout=stdout,
            stderr=stderr,
            return_code=return_code,
        )

    @staticmethod
    def _parse_hover_worker_payload(*, stdout: str, stderr: str) -> dict[str, object]:
        payload = _extract_last_json_object(stdout)
        if payload is not None:
            return payload
        normalized_error = (stderr or "").strip() or (stdout or "").strip() or "hover_test_worker_no_output"
        return {"report": None, "failures": [normalized_error]}

    def _capture_multiranger_payload(self) -> dict[str, object] | None:
        return _capture_multiranger_payload_static(
            repo_root=self.repo_root,
            workspace=self.radio_provider.workspace,
            bitcraze_python=self.radio_provider.bitcraze_python,
            timeout_s=self.multiranger_timeout_s,
        )


class MissionCreateRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    mission_type: Literal["inspect", "hover_test"]
    target_hint: str | None = None
    capture_intent: Literal["scene", "object_check", "look_closer"] | None = None
    max_duration_s: float | None = None
    return_policy: Literal["return_and_land"] | None = None


class PosePayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    healthy: bool
    tracking_state: str
    confidence: float
    source_timestamp: float | None = None
    x_m: float | None = None
    y_m: float | None = None
    z_m: float | None = None
    yaw_deg: float | None = None


class MissionPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mission_id: str
    mission_type: str
    target_hint: str
    capture_intent: str
    max_duration_s: float
    return_policy: str
    requires_manual_arm: bool
    state: str
    summary: str
    created_at: str
    updated_at: str
    artifact_name: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    queue_position: int | None = None


class SafetyPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    can_arm: bool
    manual_arm_required: bool
    radio_ready: bool
    pose_ready: bool
    motion_mode: str
    reasons: list[str]
    evaluated_at: str
    pose_age_s: float | None = None
    pose_confidence: float
    pose_cache_age_s: float
    radio_cache_age_s: float


class StatePayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service_status: str
    lifecycle_state: str
    active_mission_id: str | None = None
    manual_arm_required: bool
    skill_layer_mode: str
    radio_ready: bool
    queue_depth: int
    pending_manual_arm_count: int
    accepting_new_missions: bool
    pose: PosePayloadModel
    safety: SafetyPayloadModel


class HealthPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    service: str
    manual_arm_required: bool
    skill_layer_mode: str
    lifecycle_state: str
    radio_ready: bool
    pose_ready: bool
    can_arm: bool
    active_mission_id: str | None = None
    queue_depth: int
    accepting_new_missions: bool


class ErrorDetailModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str


class ErrorEnvelopeModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: ErrorDetailModel


class MissionEnvelopeModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mission: MissionPayloadModel


class StateEnvelopeModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: StatePayloadModel


def _error_envelope(*, code: str, message: str) -> dict[str, object]:
    return ErrorEnvelopeModel(error=ErrorDetailModel(code=code, message=message)).model_dump()


class BodySizeLimitMiddleware:
    def __init__(self, app: Any, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(
        self,
        scope: dict[str, object],
        receive: Callable[[], Any],
        send: Callable[[dict[str, object]], Any],
    ) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        method = str(scope.get("method") or "").upper()
        if method not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return

        headers = {bytes(k).lower(): bytes(v) for k, v in scope.get("headers", [])}
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            try:
                content_length = int(raw_length.decode("ascii", errors="strict"))
            except ValueError:
                response = JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content=_error_envelope(
                        code="invalid_request",
                        message="Invalid Content-Length header.",
                    ),
                )
                await response(scope, receive, send)
                return
            if content_length > self.max_bytes:
                response = JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content=_error_envelope(
                        code="request_too_large",
                        message="Request body is too large.",
                    ),
                )
                await response(scope, receive, send)
                return

        seen = 0

        async def limited_receive() -> dict[str, object]:
            nonlocal seen
            message = await receive()
            if message.get("type") == "http.request":
                body = message.get("body", b"") or b""
                seen += len(body)
                if seen > self.max_bytes:
                    raise BodyTooLargeError
            return message

        try:
            await self.app(scope, limited_receive, send)
        except BodyTooLargeError:
            response = JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content=_error_envelope(
                    code="request_too_large",
                    message="Request body is too large.",
                ),
            )
            await response(scope, receive, send)


def build_app(service: DroneDaemonService) -> FastAPI:
    api_key_header = APIKeyHeader(name=_API_KEY_HEADER_NAME, auto_error=False)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service.start()
        try:
            yield
        finally:
            service.shutdown()

    app = FastAPI(
        title="Twinr Drone Daemon",
        version="2.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=_MAX_BODY_BYTES)

    def _authorize(request: FastAPIRequest, presented_key: str | None, *, write: bool) -> None:
        client_host = request.client.host if request.client is not None else None
        if _is_loopback_client(client_host):
            return
        # BREAKING: all non-loopback access is deny-by-default unless explicitly enabled.
        if write and not service.allow_remote_ops:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="remote write access is disabled",
            )
        if not write and not service.allow_remote_reads:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="remote read access is disabled",
            )
        if not service.api_key:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="remote access requires a daemon API key",
            )
        if not _compare_secret(service.api_key, presented_key):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="invalid daemon API key",
            )

    def _read_guard(
        request: FastAPIRequest,
        api_key: str | None = Depends(api_key_header),
    ) -> None:
        _authorize(request, api_key, write=False)

    def _write_guard(
        request: FastAPIRequest,
        api_key: str | None = Depends(api_key_header),
    ) -> None:
        _authorize(request, api_key, write=True)

    @app.exception_handler(ApiError)
    async def _handle_api_error(_request: FastAPIRequest, exc: ApiError) -> JSONResponse:
        # BREAKING: all error responses are JSON-only now to keep one machine-readable contract.
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_envelope(code=exc.error_code, message=str(exc)),
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        _request: FastAPIRequest,
        exc: RequestValidationError,
    ) -> JSONResponse:
        message = "Request body must be a valid JSON object."
        errors = exc.errors()
        if errors:
            first = errors[0]
            first_message = str(first.get("msg") or "").strip()
            if first_message:
                message = f"Invalid request: {first_message}"
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=_error_envelope(code="invalid_request", message=message),
        )

    @app.exception_handler(HTTPException)
    async def _handle_http_exception(
        _request: FastAPIRequest,
        exc: HTTPException,
    ) -> JSONResponse:
        detail = str(exc.detail) if exc.detail is not None else "Request failed."
        code = "forbidden" if exc.status_code == status.HTTP_403_FORBIDDEN else "http_error"
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_envelope(code=code, message=detail),
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected_exception(
        _request: FastAPIRequest,
        exc: Exception,
    ) -> JSONResponse:
        LOGGER.exception("Unhandled exception in drone daemon request path.")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_envelope(
                code="internal_error",
                message=f"internal server error: {exc.__class__.__name__}",
            ),
        )

    @app.get("/healthz", response_model=HealthPayloadModel, dependencies=[Depends(_read_guard)])
    def get_healthz() -> dict[str, object]:
        return service.health_payload()

    @app.get("/state", response_model=StateEnvelopeModel, dependencies=[Depends(_read_guard)])
    def get_state() -> dict[str, object]:
        return {"state": service.state_payload()}

    @app.get("/pose", response_model=PosePayloadModel, dependencies=[Depends(_read_guard)])
    def get_pose() -> dict[str, object]:
        return service.pose_payload()

    @app.post(
        "/missions",
        response_model=MissionEnvelopeModel,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(_write_guard)],
    )
    def create_mission(body: MissionCreateRequestModel) -> dict[str, object]:
        mission = service.create_mission(body.model_dump(exclude_none=True))
        return {"mission": mission.to_payload()}

    @app.get("/missions/{mission_id}", response_model=MissionEnvelopeModel, dependencies=[Depends(_read_guard)])
    def get_mission(mission_id: str) -> dict[str, object]:
        return {"mission": service._get_mission(mission_id).to_payload()}

    @app.post(
        "/missions/{mission_id}/cancel",
        response_model=MissionEnvelopeModel,
        dependencies=[Depends(_write_guard)],
    )
    def cancel_mission(mission_id: str) -> dict[str, object]:
        mission = service.cancel_mission(mission_id)
        return {"mission": mission.to_payload()}

    @app.post(
        "/ops/missions/{mission_id}/arm",
        response_model=MissionEnvelopeModel,
        dependencies=[Depends(_write_guard)],
    )
    def manual_arm(mission_id: str) -> dict[str, object]:
        mission = service.manual_arm(mission_id)
        return {"mission": mission.to_payload()}

    return app


def _build_pose_provider(args: argparse.Namespace) -> PoseProvider:
    mode = str(args.pose_provider or "external_http").strip().lower()
    if mode == "stub_ok":
        return StubPoseProvider(healthy=True)
    if mode == "stub_unhealthy":
        return StubPoseProvider(healthy=False)
    if mode == "external_http":
        return ExternalHttpPoseProvider(
            base_url=str(args.pose_base_url or ""),
            timeout_s=args.pose_timeout_s,
        )
    raise ValueError("pose provider must be one of: external_http, stub_ok, stub_unhealthy")


def _capture_multiranger_payload_static(
    *,
    repo_root: Path,
    workspace: Path,
    bitcraze_python: Path,
    timeout_s: float,
) -> dict[str, object] | None:
    script_path = repo_root / "hardware" / "bitcraze" / "probe_multiranger.py"
    if not bitcraze_python.exists() or not script_path.exists():
        return None
    try:
        result = subprocess.run(
            [
                str(bitcraze_python),
                str(script_path),
                "--workspace",
                str(workspace),
                "--duration-s",
                "1.5",
                "--sample-period-s",
                "0.1",
                "--json",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
    except Exception as exc:
        return {"status": "error", "error": f"probe_exec_failed:{exc.__class__.__name__}"}
    if result.returncode != 0:
        return {"status": "error", "error": (result.stderr or "").strip() or "probe_failed"}
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"status": "error", "error": "probe_invalid_json"}
    return payload if isinstance(payload, dict) else {"status": "error", "error": "probe_invalid_json"}


def _build_twinr_config(*, repo_root: Path, env_file: Path | None) -> Any:
    _ensure_src_path(repo_root)
    from twinr.agent.base_agent.config import TwinrConfig

    if env_file is None:
        return TwinrConfig(project_root=str(repo_root))
    return TwinrConfig.from_env(env_file)


def _run_internal_inspection_worker(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve(strict=False)
    env_file = None if args.env_file is None else Path(args.env_file).resolve(strict=False)
    artifact_root = Path(args.artifact_root).resolve(strict=False)
    report_name = str(args.report_name)
    image_name = str(args.image_name)

    try:
        config = _build_twinr_config(repo_root=repo_root, env_file=env_file)
        from twinr.hardware.camera import V4L2StillCamera

        artifact_root.mkdir(parents=True, exist_ok=True)
        _safe_dir_permissions(artifact_root)

        camera = V4L2StillCamera.from_config(config)
        capture = camera.capture_photo(filename=image_name)
        capture_data = getattr(capture, "data", b"")
        if isinstance(capture_data, bytearray):
            capture_bytes = bytes(capture_data)
        elif isinstance(capture_data, bytes):
            capture_bytes = capture_data
        elif capture_data is None:
            capture_bytes = b""
        else:
            capture_bytes = bytes(capture_data)

        image_path = artifact_root / image_name
        _atomic_write_bytes(image_path, capture_bytes)

        mission_payload = {
            "mission_id": str(args.mission_id),
            "mission_type": "inspect",
            "target_hint": str(args.target_hint),
            "capture_intent": str(args.capture_intent),
            "max_duration_s": float(args.max_duration_s),
            "return_policy": str(args.return_policy),
            "requires_manual_arm": _coerce_bool(args.requires_manual_arm),
            "state": "running",
            "summary": "Stationary inspection started.",
            "created_at": str(args.created_at),
            "updated_at": str(args.updated_at),
            "artifact_name": report_name,
            "started_at": _utc_now_iso(),
            "finished_at": None,
            "queue_position": None,
        }

        artifact_payload: dict[str, object] = {
            "mission": mission_payload,
            "captured_at": _utc_now_iso(),
            "summary": "Stationary inspection evidence captured without flight primitives.",
            "camera": {
                "source_device": getattr(capture, "source_device", ""),
                "input_format": getattr(capture, "input_format", None),
                "content_type": getattr(capture, "content_type", ""),
                "image_file": image_name,
                "bytes": len(capture_bytes),
            },
        }

        multiranger_payload = _capture_multiranger_payload_static(
            repo_root=repo_root,
            workspace=Path(args.bitcraze_workspace).resolve(strict=False),
            bitcraze_python=_normalize_executable_path(Path(args.bitcraze_python)),
            timeout_s=_safe_float(
                args.multiranger_timeout_s,
                default=_DEFAULT_MULTIRANGER_TIMEOUT_S,
                minimum=1.0,
            ),
        )
        if multiranger_payload is not None:
            artifact_payload["multiranger"] = multiranger_payload

        _atomic_write_json(artifact_root / report_name, artifact_payload)
        print(json.dumps({"ok": True, "report_name": report_name, "image_name": image_name}), flush=True)
        return 0

    except KeyboardInterrupt:
        print(json.dumps({"ok": False, "error": "inspection_worker_interrupted"}), flush=True)
        return 130
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"inspection_worker_failed:{exc.__class__.__name__}",
                    "detail": str(exc),
                }
            ),
            flush=True,
        )
        return 1


def _load_api_key(args: argparse.Namespace) -> str | None:
    if args.api_key and args.api_key_file:
        raise ValueError("Use only one of --api-key or --api-key-file.")
    if args.api_key:
        return str(args.api_key).strip() or None
    if args.api_key_file:
        key_path = Path(args.api_key_file).expanduser().resolve(strict=False)
        if not key_path.exists():
            raise ValueError(f"API key file does not exist: {key_path}")
        return key_path.read_text(encoding="utf-8", errors="replace").strip() or None
    env_key = os.getenv("TWINR_DRONE_DAEMON_API_KEY", "").strip()
    return env_key or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Twinr bounded drone daemon")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Leading repo root.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Optional Twinr env file used for camera capture and artifacts.",
    )
    parser.add_argument("--bind", default=_DEFAULT_BIND_HOST, help="Bind host for the daemon.")
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT, help="TCP port for the daemon.")
    parser.add_argument(
        "--pose-provider",
        default="external_http",
        help="Pose-provider mode: external_http, stub_ok, or stub_unhealthy.",
    )
    parser.add_argument("--pose-base-url", help="Base URL for the external pose provider.")
    parser.add_argument(
        "--pose-timeout-s",
        type=float,
        default=_DEFAULT_POSE_TIMEOUT_S,
        help="Pose-provider timeout.",
    )
    parser.add_argument(
        "--pose-cache-ttl-s",
        type=float,
        default=_DEFAULT_POSE_CACHE_TTL_S,
        help="Pose snapshot cache TTL for read paths.",
    )
    parser.add_argument(
        "--radio-cache-ttl-s",
        type=float,
        default=_DEFAULT_RADIO_CACHE_TTL_S,
        help="Radio probe cache TTL for read paths.",
    )
    parser.add_argument(
        "--max-pose-age-s",
        type=float,
        default=_DEFAULT_MAX_POSE_AGE_S,
        help="Maximum acceptable age of the pose sample for arming.",
    )
    parser.add_argument(
        "--min-pose-confidence",
        type=float,
        default=_DEFAULT_MIN_POSE_CONFIDENCE,
        help="Minimum acceptable pose confidence for arming.",
    )
    parser.add_argument(
        "--bitcraze-workspace",
        type=Path,
        default=Path("/twinr/bitcraze"),
        help="Bitcraze workspace root used by the existing radio/multiranger probes.",
    )
    parser.add_argument(
        "--bitcraze-python",
        type=Path,
        default=Path("/twinr/bitcraze/.venv/bin/python"),
        help="Python interpreter for Bitcraze probe helpers.",
    )
    parser.add_argument(
        "--skill-layer-mode",
        default="stationary_observe_only",
        help="Reported skill-layer mode; use `stationary_observe_only` by default or `bounded_hover_test_only` to allow hover-test missions.",
    )
    parser.add_argument(
        "--allow-remote-reads",
        action="store_true",
        help="Allow non-loopback clients to call read endpoints. Requires --api-key or --api-key-file.",
    )
    parser.add_argument(
        "--allow-remote-ops",
        action="store_true",
        help="Allow non-loopback clients to call write endpoints. Requires --api-key or --api-key-file.",
    )
    parser.add_argument(
        "--api-key",
        help="Daemon API key for any non-loopback access. Prefer --api-key-file in production.",
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        help="Read the daemon API key from a file for any non-loopback access.",
    )
    parser.add_argument(
        "--trust-proxy-headers",
        action="store_true",
        help="Trust forwarded headers from explicitly allowed proxy IPs.",
    )
    parser.add_argument(
        "--forwarded-allow-ips",
        default="127.0.0.1,::1",
        help="Comma-separated proxy IPs or CIDRs trusted for forwarded headers when --trust-proxy-headers is enabled.",
    )
    parser.add_argument(
        "--multiranger-timeout-s",
        type=float,
        default=_DEFAULT_MULTIRANGER_TIMEOUT_S,
        help="Timeout for the best-effort Multi-ranger snapshot helper.",
    )
    parser.add_argument(
        "--hover-test-height-m",
        type=float,
        default=_DEFAULT_HOVER_TEST_HEIGHT_M,
        help="Hover-test takeoff height in meters when hover mode is enabled.",
    )
    parser.add_argument(
        "--hover-test-duration-s",
        type=float,
        default=_DEFAULT_HOVER_TEST_DURATION_S,
        help="Hover-test hold duration in seconds when hover mode is enabled.",
    )
    parser.add_argument(
        "--hover-test-takeoff-velocity-mps",
        type=float,
        default=_DEFAULT_HOVER_TEST_VELOCITY_MPS,
        help="Hover-test takeoff velocity in meters per second.",
    )
    parser.add_argument(
        "--hover-test-land-velocity-mps",
        type=float,
        default=_DEFAULT_HOVER_TEST_VELOCITY_MPS,
        help="Hover-test landing velocity in meters per second.",
    )
    parser.add_argument(
        "--hover-test-min-vbat-v",
        type=float,
        default=_DEFAULT_HOVER_TEST_MIN_VBAT_V,
        help="Minimum battery voltage gate for hover-test missions.",
    )
    parser.add_argument(
        "--hover-test-min-battery-level",
        type=int,
        default=_DEFAULT_HOVER_TEST_MIN_BATTERY_LEVEL,
        help="Minimum battery percentage gate for hover-test missions.",
    )
    parser.add_argument(
        "--max-outstanding-missions",
        type=int,
        default=_DEFAULT_MAX_OUTSTANDING_MISSIONS,
        help="Maximum queued + pending + running missions.",
    )
    parser.add_argument(
        "--limit-concurrency",
        type=int,
        default=_DEFAULT_LIMIT_CONCURRENCY,
        help="Maximum concurrent Uvicorn connections/tasks before 503.",
    )
    parser.add_argument(
        "--backlog",
        type=int,
        default=_DEFAULT_BACKLOG,
        help="Maximum Uvicorn socket backlog.",
    )
    parser.add_argument(
        "--timeout-keep-alive-s",
        type=int,
        default=_DEFAULT_KEEPALIVE_TIMEOUT_S,
        help="Uvicorn keep-alive timeout.",
    )
    parser.add_argument(
        "--timeout-graceful-shutdown-s",
        type=int,
        default=_DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_S,
        help="Uvicorn graceful shutdown timeout.",
    )
    parser.add_argument("--ssl-keyfile", type=Path, help="Optional TLS private key for direct HTTPS.")
    parser.add_argument("--ssl-certfile", type=Path, help="Optional TLS certificate for direct HTTPS.")
    parser.add_argument("--ssl-keyfile-password", help="Optional password for the TLS private key.")
    parser.add_argument("--ssl-ca-certs", type=Path, help="Optional CA bundle for client-certificate validation.")
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=0,
        help="TLS client certificate policy. 0=none, 1=optional, 2=required.",
    )

    parser.add_argument("--internal-run-inspection-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--artifact-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--mission-id", help=argparse.SUPPRESS)
    parser.add_argument("--target-hint", help=argparse.SUPPRESS)
    parser.add_argument("--capture-intent", help=argparse.SUPPRESS)
    parser.add_argument("--max-duration-s", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--return-policy", help=argparse.SUPPRESS)
    parser.add_argument("--requires-manual-arm", help=argparse.SUPPRESS)
    parser.add_argument("--created-at", help=argparse.SUPPRESS)
    parser.add_argument("--updated-at", help=argparse.SUPPRESS)
    parser.add_argument("--report-name", help=argparse.SUPPRESS)
    parser.add_argument("--image-name", help=argparse.SUPPRESS)

    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()

    if args.internal_run_inspection_worker:
        return _run_internal_inspection_worker(args)

    try:
        api_key = _load_api_key(args)
        pose_provider = _build_pose_provider(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if (args.allow_remote_reads or args.allow_remote_ops) and not api_key:
        raise SystemExit("Remote access now requires --api-key, --api-key-file, or TWINR_DRONE_DAEMON_API_KEY.")
    if (args.ssl_keyfile is None) != (args.ssl_certfile is None):
        raise SystemExit("Direct TLS requires both --ssl-keyfile and --ssl-certfile.")

    repo_root = Path(args.repo_root).resolve(strict=False)
    radio_provider = BitcrazeRadioStatusProvider(
        repo_root=repo_root,
        workspace=Path(args.bitcraze_workspace).resolve(strict=False),
        bitcraze_python=_normalize_executable_path(Path(args.bitcraze_python)),
    )
    service = DroneDaemonService(
        repo_root=repo_root,
        env_file=None if args.env_file is None else Path(args.env_file).resolve(strict=False),
        pose_provider=pose_provider,
        radio_provider=radio_provider,
        skill_layer_mode=args.skill_layer_mode,
        manual_arm_required=True,
        allow_remote_ops=args.allow_remote_ops,
        allow_remote_reads=args.allow_remote_reads,
        api_key=api_key,
        multiranger_timeout_s=args.multiranger_timeout_s,
        hover_test_config=HoverTestSkillConfig(
            height_m=_safe_float(
                args.hover_test_height_m,
                default=_DEFAULT_HOVER_TEST_HEIGHT_M,
                minimum=0.1,
            ),
            hover_duration_s=_safe_float(
                args.hover_test_duration_s,
                default=_DEFAULT_HOVER_TEST_DURATION_S,
                minimum=1.0,
            ),
            takeoff_velocity_mps=_safe_float(
                args.hover_test_takeoff_velocity_mps,
                default=_DEFAULT_HOVER_TEST_VELOCITY_MPS,
                minimum=0.05,
            ),
            land_velocity_mps=_safe_float(
                args.hover_test_land_velocity_mps,
                default=_DEFAULT_HOVER_TEST_VELOCITY_MPS,
                minimum=0.05,
            ),
            min_vbat_v=_safe_float(
                args.hover_test_min_vbat_v,
                default=_DEFAULT_HOVER_TEST_MIN_VBAT_V,
                minimum=0.0,
            ),
            min_battery_level=_safe_int(
                args.hover_test_min_battery_level,
                default=_DEFAULT_HOVER_TEST_MIN_BATTERY_LEVEL,
                minimum=0,
                maximum=100,
            ),
        ),
        pose_cache_ttl_s=_safe_float(
            args.pose_cache_ttl_s,
            default=_DEFAULT_POSE_CACHE_TTL_S,
            minimum=0.05,
        ),
        radio_cache_ttl_s=_safe_float(
            args.radio_cache_ttl_s,
            default=_DEFAULT_RADIO_CACHE_TTL_S,
            minimum=0.1,
        ),
        max_pose_age_s=_safe_float(
            args.max_pose_age_s,
            default=_DEFAULT_MAX_POSE_AGE_S,
            minimum=0.1,
        ),
        min_pose_confidence=_safe_float(
            args.min_pose_confidence,
            default=_DEFAULT_MIN_POSE_CONFIDENCE,
            minimum=0.0,
            maximum=1.0,
        ),
        max_outstanding_missions=_safe_int(
            args.max_outstanding_missions,
            default=_DEFAULT_MAX_OUTSTANDING_MISSIONS,
            minimum=1,
            maximum=256,
        ),
    )
    app = build_app(service)

    forwarded_allow_ips = args.forwarded_allow_ips if args.trust_proxy_headers else ""
    config = uvicorn.Config(
        app=app,
        host=str(args.bind),
        port=int(args.port),
        log_level="info",
        access_log=False,
        proxy_headers=bool(args.trust_proxy_headers),
        forwarded_allow_ips=forwarded_allow_ips,
        limit_concurrency=_safe_int(
            args.limit_concurrency,
            default=_DEFAULT_LIMIT_CONCURRENCY,
            minimum=1,
            maximum=4096,
        ),
        backlog=_safe_int(args.backlog, default=_DEFAULT_BACKLOG, minimum=16, maximum=8192),
        timeout_keep_alive=_safe_int(
            args.timeout_keep_alive_s,
            default=_DEFAULT_KEEPALIVE_TIMEOUT_S,
            minimum=1,
            maximum=300,
        ),
        timeout_graceful_shutdown=_safe_int(
            args.timeout_graceful_shutdown_s,
            default=_DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_S,
            minimum=5,
            maximum=300,
        ),
        ssl_keyfile=None if args.ssl_keyfile is None else str(Path(args.ssl_keyfile).resolve(strict=False)),
        ssl_certfile=None if args.ssl_certfile is None else str(Path(args.ssl_certfile).resolve(strict=False)),
        ssl_keyfile_password=args.ssl_keyfile_password,
        ssl_ca_certs=None if args.ssl_ca_certs is None else str(Path(args.ssl_ca_certs).resolve(strict=False)),
        ssl_cert_reqs=_safe_int(args.ssl_cert_reqs, default=0, minimum=0, maximum=2),
        server_header=False,
        date_header=True,
        lifespan="on",
    )
    server = uvicorn.Server(config)
    _structured_log(
        "server_listening",
        bind=str(args.bind),
        port=int(args.port),
        skill_layer_mode=service.skill_layer_mode,
        manual_arm_required=service.manual_arm_required,
        allow_remote_reads=service.allow_remote_reads,
        allow_remote_ops=service.allow_remote_ops,
        tls_enabled=bool(args.ssl_keyfile and args.ssl_certfile),
    )
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())