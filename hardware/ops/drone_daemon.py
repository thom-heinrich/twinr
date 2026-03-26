#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Serve bounded drone-mission control for Twinr.

Purpose
-------
Expose a strict HTTP boundary between Twinr's slow semantic layer and the
faster drone-control layer. The current vertical slice keeps safety first: it
supports preflight checks, manual arm gating, mission queuing, and bounded
stationary inspection evidence capture, but it does not yet enable free flight.

Usage
-----
Run the daemon in the foreground for local checks::

    python3 hardware/ops/drone_daemon.py --repo-root /home/thh/twinr --env-file /home/thh/twinr/.env
    python3 hardware/ops/drone_daemon.py --pose-provider stub_ok --bind 127.0.0.1 --port 8791

Inputs
------
- ``GET /healthz`` for a small service-health payload
- ``GET /state`` for the full bounded drone state view
- ``GET /pose`` for the current pose-provider sample
- ``POST /missions`` with one bounded mission request
- ``GET /missions/<id>`` to inspect mission state
- ``POST /missions/<id>/cancel`` to abort or clear a pending mission
- ``POST /ops/missions/<id>/arm`` for operator-only manual arm approval

Outputs
-------
- JSON payloads only
- bounded stationary inspection artifacts under ``artifacts/ops/drone_missions/``
- exit code ``0`` on clean shutdown

Notes
-----
This daemon intentionally rejects direct low-level flight commands. Twinr must
stay above the mission boundary. Until a real skill/primitive executor is
wired in, the daemon reports ``stationary_observe_only`` and produces safe
inspection evidence without moving the aircraft.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


LOGGER = logging.getLogger(__name__)

_DEFAULT_BIND_HOST = "127.0.0.1"
_DEFAULT_PORT = 8791
_DEFAULT_POSE_TIMEOUT_S = 1.0
_DEFAULT_MISSION_TIMEOUT_S = 45.0
_DEFAULT_MULTIRANGER_TIMEOUT_S = 15.0
_MAX_BODY_BYTES = 16384
_FINAL_MISSION_STATES = frozenset({"completed", "failed", "cancelled"})


def _utc_now_iso() -> str:
    """Return one ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: object, *, default: float, minimum: float | None = None) -> float:
    """Return one bounded float from CLI or config input."""

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
    return parsed


def _safe_int(value: object, *, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    """Return one bounded integer from CLI or config input."""

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
    """Return one permissive boolean from JSON-like payloads."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _coerce_payload_float(value: object, *, default: float | None = None) -> float | None:
    """Return one optional float from a JSON-like payload."""

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
    """Return one payload dict or an empty fallback."""

    return value if isinstance(value, dict) else {}


def _payload_string_list(value: object) -> list[str]:
    """Return one list of normalized non-empty strings from a JSON-like payload."""

    if not isinstance(value, list):
        return []
    return [normalized for item in value if (normalized := str(item).strip())]


def _remote_error_code(exc: BaseException) -> str:
    """Render one short transport error token."""

    reason = getattr(exc, "reason", None)
    if isinstance(reason, BaseException):
        return reason.__class__.__name__
    if reason is not None:
        normalized = str(reason).strip()
        if normalized:
            return normalized
    return exc.__class__.__name__


@dataclass(frozen=True, slots=True)
class PoseSnapshot:
    """Represent one current pose-provider sample."""

    healthy: bool
    tracking_state: str
    confidence: float
    source_timestamp: float | None = None
    x_m: float | None = None
    y_m: float | None = None
    z_m: float | None = None
    yaw_deg: float | None = None

    def to_payload(self) -> dict[str, object]:
        """Serialize the pose for the wire contract."""

        return asdict(self)


class PoseProvider(Protocol):
    """Return one current pose snapshot for safety gating."""

    def snapshot(self) -> PoseSnapshot:
        """Return the newest bounded pose sample."""


class StubPoseProvider:
    """Return a deterministic healthy or unhealthy pose sample."""

    def __init__(self, *, healthy: bool) -> None:
        self._healthy = healthy

    def snapshot(self) -> PoseSnapshot:
        """Return the current stub pose state."""

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
    """Fetch pose samples from one external VIO/SLAM helper service."""

    def __init__(self, *, base_url: str, timeout_s: float = _DEFAULT_POSE_TIMEOUT_S) -> None:
        normalized = str(base_url or "").strip().rstrip("/")
        if not normalized:
            raise ValueError("external pose provider requires --pose-base-url")
        self._base_url = normalized
        self._timeout_s = _safe_float(timeout_s, default=_DEFAULT_POSE_TIMEOUT_S, minimum=0.1)

    def snapshot(self) -> PoseSnapshot:
        """Return the current external pose sample or an unhealthy fallback."""

        request = Request(
            f"{self._base_url}/pose",
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            with urlopen(request, timeout=self._timeout_s) as response:
                charset = response.headers.get_content_charset("utf-8")
                payload = json.loads(response.read().decode(charset))
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
    """Reuse the existing Bitcraze probe helper for radio-health checks."""

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
        """Return the bounded Crazyradio health payload."""

        script_path = self.repo_root / "hardware" / "bitcraze" / "probe_crazyradio.py"
        if not script_path.exists():
            return {"radio_ready": False, "error": f"missing_probe:{script_path}"}
        if not self.bitcraze_python.exists():
            return {"radio_ready": False, "error": f"missing_bitcraze_python:{self.bitcraze_python}"}
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
                timeout=12.0,
            )
        except Exception as exc:
            return {"radio_ready": False, "error": f"probe_exec_failed:{exc.__class__.__name__}"}
        if result.returncode != 0:
            stderr = (result.stderr or "").strip() or "probe_failed"
            return {"radio_ready": False, "error": stderr}
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return {"radio_ready": False, "error": "probe_invalid_json"}
        workspace_payload = payload.get("workspace") if isinstance(payload, dict) else None
        radio_ready = bool(isinstance(workspace_payload, dict) and workspace_payload.get("radio_access_ok"))
        error_text = None if radio_ready else "radio_access_unavailable"
        if isinstance(workspace_payload, dict):
            error_text = workspace_payload.get("radio_access_error") or error_text
        return {
            "radio_ready": radio_ready,
            "radio_version": workspace_payload.get("radio_version") if isinstance(workspace_payload, dict) else None,
            "error": error_text,
        }


@dataclass(frozen=True, slots=True)
class MissionRecord:
    """Store one daemon mission in a JSON-friendly shape."""

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

    def to_payload(self) -> dict[str, object]:
        """Serialize the mission for the wire contract."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class ValidatedMissionRequest:
    """Store one validated bounded mission request."""

    mission_type: str
    target_hint: str
    capture_intent: str
    max_duration_s: float
    return_policy: str


class DroneDaemonService:
    """Own one bounded drone-mission surface for Twinr."""

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
        multiranger_timeout_s: float = _DEFAULT_MULTIRANGER_TIMEOUT_S,
    ) -> None:
        resolved_repo_root = repo_root.resolve(strict=False)
        src_root = resolved_repo_root / "src"
        if str(src_root) not in sys.path:
            sys.path.insert(0, str(src_root))
        from twinr.agent.base_agent.config import TwinrConfig

        self.repo_root = resolved_repo_root
        self.env_file = env_file
        self.pose_provider = pose_provider
        self.radio_provider = radio_provider
        self.skill_layer_mode = str(skill_layer_mode or "stationary_observe_only").strip() or "stationary_observe_only"
        self.manual_arm_required = bool(manual_arm_required)
        self.allow_remote_ops = bool(allow_remote_ops)
        self.multiranger_timeout_s = _safe_float(
            multiranger_timeout_s,
            default=_DEFAULT_MULTIRANGER_TIMEOUT_S,
            minimum=1.0,
        )
        self._lock = threading.RLock()
        self._missions: dict[str, MissionRecord] = {}
        self._cancel_requested: set[str] = set()
        self._active_mission_id: str | None = None
        if env_file is None:
            self.config = TwinrConfig(project_root=str(resolved_repo_root))
        else:
            self.config = TwinrConfig.from_env(env_file)
        self._artifact_root = Path(self.config.project_root) / "artifacts" / "ops" / "drone_missions"

    def health_payload(self) -> dict[str, object]:
        """Return a compact service-health payload."""

        state = self.state_payload()
        safety = _payload_dict(state.get("safety"))
        return {
            "ok": True,
            "service": "drone_daemon",
            "manual_arm_required": self.manual_arm_required,
            "skill_layer_mode": self.skill_layer_mode,
            "radio_ready": bool(safety.get("radio_ready")),
            "pose_ready": bool(safety.get("pose_ready")),
            "can_arm": bool(safety.get("can_arm")),
            "active_mission_id": state.get("active_mission_id"),
        }

    def pose_payload(self) -> dict[str, object]:
        """Return the current pose payload."""

        return self.pose_provider.snapshot().to_payload()

    def state_payload(self) -> dict[str, object]:
        """Return the current daemon state payload."""

        radio = self.radio_provider.snapshot()
        pose = self.pose_provider.snapshot()
        reasons: list[str] = []
        radio_ready = bool(radio.get("radio_ready"))
        pose_ready = bool(pose.healthy)
        if not radio_ready:
            reasons.append(str(radio.get("error") or "radio_unavailable"))
        if not pose_ready:
            reasons.append(str(pose.tracking_state or "pose_unavailable"))
        safety = {
            "can_arm": radio_ready and pose_ready,
            "manual_arm_required": self.manual_arm_required,
            "radio_ready": radio_ready,
            "pose_ready": pose_ready,
            "motion_mode": self.skill_layer_mode,
            "reasons": reasons,
        }
        with self._lock:
            active_mission_id = self._active_mission_id
        return {
            "service_status": "ready",
            "active_mission_id": active_mission_id,
            "manual_arm_required": self.manual_arm_required,
            "skill_layer_mode": self.skill_layer_mode,
            "radio_ready": radio_ready,
            "pose": pose.to_payload(),
            "safety": safety,
        }

    def create_mission(self, payload: dict[str, object]) -> MissionRecord:
        """Create one bounded mission after preflight validation."""

        request = self._validate_mission_payload(payload)
        state = self.state_payload()
        safety = _payload_dict(state.get("safety"))
        if not bool(safety.get("can_arm")):
            reasons = ", ".join(_payload_string_list(safety.get("reasons"))) or "preflight_failed"
            raise ConflictError(f"Mission preflight failed: {reasons}")
        mission = MissionRecord(
            mission_id=f"DRN-{int(time.time() * 1000)}",
            mission_type=request.mission_type,
            target_hint=request.target_hint,
            capture_intent=request.capture_intent,
            max_duration_s=request.max_duration_s,
            return_policy=request.return_policy,
            requires_manual_arm=self.manual_arm_required,
            state="pending_manual_arm" if self.manual_arm_required else "running",
            summary=(
                "Mission queued and waiting for local manual arm approval."
                if self.manual_arm_required
                else "Mission started."
            ),
            created_at=_utc_now_iso(),
            updated_at=_utc_now_iso(),
        )
        with self._lock:
            self._missions[mission.mission_id] = mission
        if not self.manual_arm_required:
            self._start_mission_thread(mission.mission_id)
        return mission

    def mission_payload(self, mission_id: str) -> dict[str, object]:
        """Return one serialized mission payload."""

        return self._get_mission(mission_id).to_payload()

    def cancel_mission(self, mission_id: str) -> MissionRecord:
        """Cancel one mission or request cancellation for an active run."""

        with self._lock:
            mission = self._require_mission_locked(mission_id)
            if mission.state in _FINAL_MISSION_STATES:
                return mission
            if mission.state == "pending_manual_arm":
                mission = self._store_mission_locked(
                    replace(
                        mission,
                        state="cancelled",
                        summary="Mission cancelled before arming.",
                        updated_at=_utc_now_iso(),
                    )
                )
                return mission
            self._cancel_requested.add(mission_id)
            mission = self._store_mission_locked(
                replace(
                    mission,
                    state="cancel_requested",
                    summary="Mission cancellation requested.",
                    updated_at=_utc_now_iso(),
                )
            )
            return mission

    def manual_arm(self, mission_id: str) -> MissionRecord:
        """Approve manual arming for one pending mission."""

        with self._lock:
            mission = self._require_mission_locked(mission_id)
            if mission.state != "pending_manual_arm":
                raise ConflictError(f"Mission {mission_id} is not waiting for manual arm.")
        state = self.state_payload()
        safety = _payload_dict(state.get("safety"))
        if not bool(safety.get("can_arm")):
            reasons = ", ".join(_payload_string_list(safety.get("reasons"))) or "preflight_failed"
            raise ConflictError(f"Mission cannot arm: {reasons}")
        with self._lock:
            mission = self._store_mission_locked(
                replace(
                    mission,
                    state="armed",
                    summary="Mission armed locally. Starting bounded inspection.",
                    updated_at=_utc_now_iso(),
                )
            )
        self._start_mission_thread(mission.mission_id)
        return mission

    def _validate_mission_payload(self, payload: dict[str, object]) -> ValidatedMissionRequest:
        """Validate the bounded mission wire contract."""

        mission_type = str(payload.get("mission_type") or "").strip().lower()
        if mission_type != "inspect":
            raise InputError("mission_type must currently be `inspect`.")
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

    def _start_mission_thread(self, mission_id: str) -> None:
        """Run one mission in a background thread."""

        thread = threading.Thread(
            target=self._run_mission,
            args=(mission_id,),
            name=f"drone-mission-{mission_id}",
            daemon=True,
        )
        thread.start()

    def _run_mission(self, mission_id: str) -> None:
        """Execute one bounded stationary inspection mission."""

        with self._lock:
            mission = self._require_mission_locked(mission_id)
            self._active_mission_id = mission_id
            mission = self._store_mission_locked(
                replace(
                    mission,
                    state="running",
                    summary="Stationary inspection started.",
                    updated_at=_utc_now_iso(),
                )
            )
        try:
            if self._is_cancel_requested(mission_id):
                raise MissionCancelled("Mission cancelled before capture.")
            artifact_name = self._capture_stationary_inspection(mission)
            if self._is_cancel_requested(mission_id):
                raise MissionCancelled("Mission cancelled during bounded inspection.")
            with self._lock:
                self._store_mission_locked(
                    replace(
                        mission,
                        state="completed",
                        summary="Stationary inspection evidence captured.",
                        updated_at=_utc_now_iso(),
                        artifact_name=artifact_name,
                    )
                )
        except MissionCancelled as exc:
            with self._lock:
                self._store_mission_locked(
                    replace(
                        mission,
                        state="cancelled",
                        summary=str(exc),
                        updated_at=_utc_now_iso(),
                    )
                )
        except Exception as exc:  # pragma: no cover - live hardware path
            LOGGER.exception("Drone mission %s failed.", mission_id)
            with self._lock:
                self._store_mission_locked(
                    replace(
                        mission,
                        state="failed",
                        summary=f"Inspection failed: {exc}",
                        updated_at=_utc_now_iso(),
                    )
                )
        finally:
            with self._lock:
                self._active_mission_id = None
                self._cancel_requested.discard(mission_id)

    def _capture_stationary_inspection(self, mission: MissionRecord) -> str:
        """Capture bounded mission evidence without moving the aircraft."""

        from twinr.hardware.camera import V4L2StillCamera

        self._artifact_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        image_name = f"{mission.mission_id.lower()}-{stamp}.png"
        image_path = self._artifact_root / image_name
        camera = V4L2StillCamera.from_config(self.config)
        capture = camera.capture_photo(output_path=image_path, filename=image_name)
        artifact_payload: dict[str, object] = {
            "mission": mission.to_payload(),
            "captured_at": _utc_now_iso(),
            "summary": "Stationary inspection evidence captured without flight primitives.",
            "camera": {
                "source_device": getattr(capture, "source_device", ""),
                "input_format": getattr(capture, "input_format", None),
                "content_type": getattr(capture, "content_type", ""),
                "image_file": image_name,
                "bytes": len(getattr(capture, "data", b"")),
            },
        }
        multiranger_payload = self._capture_multiranger_payload()
        if multiranger_payload is not None:
            artifact_payload["multiranger"] = multiranger_payload
        report_name = f"{mission.mission_id.lower()}-{stamp}.json"
        (self._artifact_root / report_name).write_text(
            json.dumps(artifact_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return report_name

    def _capture_multiranger_payload(self) -> dict[str, object] | None:
        """Capture one optional bounded Multi-ranger snapshot."""

        bitcraze_python = self.radio_provider.bitcraze_python
        script_path = self.repo_root / "hardware" / "bitcraze" / "probe_multiranger.py"
        if not bitcraze_python.exists() or not script_path.exists():
            return None
        try:
            result = subprocess.run(
                [
                    str(bitcraze_python),
                    str(script_path),
                    "--workspace",
                    str(self.radio_provider.workspace),
                    "--duration-s",
                    "1.5",
                    "--sample-period-s",
                    "0.1",
                    "--json",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.multiranger_timeout_s,
            )
        except Exception as exc:
            return {"status": "error", "error": f"probe_exec_failed:{exc.__class__.__name__}"}
        if result.returncode != 0:
            return {"status": "error", "error": (result.stderr or "").strip() or "probe_failed"}
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return {"status": "error", "error": "probe_invalid_json"}

    def _get_mission(self, mission_id: str) -> MissionRecord:
        """Return one mission outside the shared lock."""

        with self._lock:
            return self._require_mission_locked(mission_id)

    def _require_mission_locked(self, mission_id: str) -> MissionRecord:
        """Return one mission from the locked mission store."""

        normalized_id = str(mission_id or "").strip()
        mission = self._missions.get(normalized_id)
        if mission is None:
            raise NotFoundError(f"Unknown mission: {normalized_id}")
        return mission

    def _store_mission_locked(self, mission: MissionRecord) -> MissionRecord:
        """Persist one mission back into the locked mission store."""

        self._missions[mission.mission_id] = mission
        return mission

    def _is_cancel_requested(self, mission_id: str) -> bool:
        """Return whether the given mission has a pending cancel request."""

        with self._lock:
            return mission_id in self._cancel_requested


def build_handler(service: DroneDaemonService) -> type[BaseHTTPRequestHandler]:
    """Return one request handler class bound to the shared daemon service."""

    class DroneDaemonHandler(BaseHTTPRequestHandler):
        server_version = "TwinrDroneDaemon/1.0"
        sys_version = ""

        def do_GET(self) -> None:  # noqa: N802 - stdlib API
            parsed = urlsplit(self.path)
            if parsed.path == "/healthz":
                self._send_json(HTTPStatus.OK, service.health_payload())
                return
            if parsed.path == "/state":
                self._send_json(HTTPStatus.OK, {"state": service.state_payload()})
                return
            if parsed.path == "/pose":
                self._send_json(HTTPStatus.OK, service.pose_payload())
                return
            mission_id = _parse_mission_path(parsed.path)
            if mission_id is not None:
                try:
                    self._send_json(HTTPStatus.OK, {"mission": service.mission_payload(mission_id)})
                except NotFoundError as exc:
                    self._send_error_text(HTTPStatus.NOT_FOUND, str(exc))
                return
            self._send_error_text(HTTPStatus.NOT_FOUND, "not found")

        def do_POST(self) -> None:  # noqa: N802 - stdlib API
            parsed = urlsplit(self.path)
            try:
                if parsed.path == "/missions":
                    payload = _read_json_payload(self)
                    mission = service.create_mission(payload)
                    self._send_json(HTTPStatus.CREATED, {"mission": mission.to_payload()})
                    return
                mission_id = _parse_cancel_path(parsed.path)
                if mission_id is not None:
                    mission = service.cancel_mission(mission_id)
                    self._send_json(HTTPStatus.OK, {"mission": mission.to_payload()})
                    return
                mission_id = _parse_manual_arm_path(parsed.path)
                if mission_id is not None:
                    if not service.allow_remote_ops and not _is_loopback_client(self.client_address[0]):
                        self._send_error_text(HTTPStatus.FORBIDDEN, "manual arm is local-host only")
                        return
                    mission = service.manual_arm(mission_id)
                    self._send_json(HTTPStatus.OK, {"mission": mission.to_payload()})
                    return
            except InputError as exc:
                self._send_error_text(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except NotFoundError as exc:
                self._send_error_text(HTTPStatus.NOT_FOUND, str(exc))
                return
            except ConflictError as exc:
                self._send_error_text(HTTPStatus.CONFLICT, str(exc))
                return
            self._send_error_text(HTTPStatus.NOT_FOUND, "not found")

        def log_message(self, fmt: str, *args: object) -> None:
            message = fmt % args
            print(f"drone_daemon {self.address_string()} {message}", flush=True)

        def _send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_error_text(self, status: HTTPStatus, message: str) -> None:
            body = message.encode("utf-8", errors="replace")
            self.send_response(int(status))
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return DroneDaemonHandler


class InputError(ValueError):
    """Raised when one client request is structurally invalid."""


class ConflictError(RuntimeError):
    """Raised when one request conflicts with current daemon state."""


class NotFoundError(RuntimeError):
    """Raised when one mission or route target does not exist."""


class MissionCancelled(RuntimeError):
    """Raised when a mission is cancelled during execution."""


def _parse_mission_path(path: str) -> str | None:
    """Return the mission id from ``/missions/<id>`` routes."""

    parts = [part for part in path.split("/") if part]
    if len(parts) == 2 and parts[0] == "missions":
        return parts[1]
    return None


def _parse_cancel_path(path: str) -> str | None:
    """Return the mission id from ``/missions/<id>/cancel`` routes."""

    parts = [part for part in path.split("/") if part]
    if len(parts) == 3 and parts[0] == "missions" and parts[2] == "cancel":
        return parts[1]
    return None


def _parse_manual_arm_path(path: str) -> str | None:
    """Return the mission id from ``/ops/missions/<id>/arm`` routes."""

    parts = [part for part in path.split("/") if part]
    if len(parts) == 4 and parts[0] == "ops" and parts[1] == "missions" and parts[3] == "arm":
        return parts[2]
    return None


def _is_loopback_client(host: str) -> bool:
    """Return whether the HTTP client is local to the daemon host."""

    return host in {"127.0.0.1", "::1", "localhost"}


def _read_json_payload(handler: BaseHTTPRequestHandler) -> dict[str, object]:
    """Read and validate one JSON request body."""

    try:
        content_length = int(handler.headers.get("Content-Length", "0"))
    except ValueError as exc:
        raise InputError("Invalid Content-Length header.") from exc
    if content_length <= 0:
        return {}
    if content_length > _MAX_BODY_BYTES:
        raise InputError("Request body is too large.")
    raw = handler.rfile.read(content_length)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise InputError("Request body must be valid JSON.") from exc
    if not isinstance(payload, dict):
        raise InputError("Request body must be a JSON object.")
    return payload


def _build_pose_provider(args: argparse.Namespace) -> PoseProvider:
    """Build the configured pose provider implementation."""

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


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the drone daemon."""

    parser = argparse.ArgumentParser(description="Twinr bounded drone daemon")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2], help="Leading repo root.")
    parser.add_argument("--env-file", type=Path, help="Optional Twinr env file used for camera capture and artifacts.")
    parser.add_argument("--bind", default=_DEFAULT_BIND_HOST, help="Bind host for the daemon.")
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT, help="TCP port for the daemon.")
    parser.add_argument(
        "--pose-provider",
        default="external_http",
        help="Pose-provider mode: external_http, stub_ok, or stub_unhealthy.",
    )
    parser.add_argument("--pose-base-url", help="Base URL for the external pose provider.")
    parser.add_argument("--pose-timeout-s", type=float, default=_DEFAULT_POSE_TIMEOUT_S, help="Pose-provider timeout.")
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
        help="Reported skill-layer mode; v1 default keeps motion disabled.",
    )
    parser.add_argument(
        "--allow-remote-ops",
        action="store_true",
        help="Allow non-loopback clients to call operator-only endpoints such as manual arm.",
    )
    parser.add_argument(
        "--multiranger-timeout-s",
        type=float,
        default=_DEFAULT_MULTIRANGER_TIMEOUT_S,
        help="Timeout for the best-effort Multi-ranger snapshot helper.",
    )
    return parser


def main() -> int:
    """Parse CLI arguments, start the daemon, and block until shutdown."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    repo_root = Path(args.repo_root).resolve(strict=False)
    pose_provider = _build_pose_provider(args)
    radio_provider = BitcrazeRadioStatusProvider(
        repo_root=repo_root,
        workspace=Path(args.bitcraze_workspace).resolve(strict=False),
        bitcraze_python=Path(args.bitcraze_python).resolve(strict=False),
    )
    service = DroneDaemonService(
        repo_root=repo_root,
        env_file=None if args.env_file is None else Path(args.env_file).resolve(strict=False),
        pose_provider=pose_provider,
        radio_provider=radio_provider,
        skill_layer_mode=args.skill_layer_mode,
        manual_arm_required=True,
        allow_remote_ops=args.allow_remote_ops,
        multiranger_timeout_s=args.multiranger_timeout_s,
    )
    server = ThreadingHTTPServer((args.bind, int(args.port)), build_handler(service))
    print(f"drone_daemon listening on http://{args.bind}:{int(args.port)}", flush=True)
    print(f"drone_daemon skill_layer_mode={service.skill_layer_mode}", flush=True)
    print(f"drone_daemon manual_arm_required={str(service.manual_arm_required).lower()}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - manual stop path
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
