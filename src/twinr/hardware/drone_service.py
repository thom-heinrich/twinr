"""Reach a bounded drone-mission daemon over HTTP.

This module gives Twinr one strict client boundary for drone work. Twinr stays
above the flight layer and only submits bounded mission requests, reads status,
or cancels work. Direct attitude or thrust commands are intentionally absent.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import socket
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from twinr.agent.base_agent.config import TwinrConfig

_DEFAULT_DRONE_TIMEOUT_S = 2.0
_DEFAULT_DRONE_MISSION_TIMEOUT_S = 45.0


def _normalize_base_url(value: object) -> str:
    """Normalize one optional HTTP base URL or return an empty string."""

    return str(value or "").strip().rstrip("/")


def _bounded_timeout_s(value: object, *, default: float) -> float:
    """Return one bounded request timeout in seconds."""

    if isinstance(value, (int, float, str)):
        try:
            parsed = float(value)
        except ValueError:
            parsed = default
    else:
        parsed = default
    return max(0.1, parsed)


def _coerce_bool(value: object, *, default: bool = False) -> bool:
    """Return one permissive boolean value from JSON-ish payloads."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _coerce_float(value: object) -> float | None:
    """Return one optional float from a JSON payload."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        try:
            return float(normalized)
        except ValueError:
            return None
    return None


@dataclass(frozen=True, slots=True)
class DroneServiceConfig:
    """Store Twinr's remote drone-daemon wiring."""

    enabled: bool
    base_url: str | None
    require_manual_arm: bool = True
    mission_timeout_s: float = _DEFAULT_DRONE_MISSION_TIMEOUT_S
    request_timeout_s: float = _DEFAULT_DRONE_TIMEOUT_S

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DroneServiceConfig":
        """Build one normalized drone-service config from ``TwinrConfig``."""

        base_url = _normalize_base_url(getattr(config, "drone_base_url", None)) or None
        return cls(
            enabled=bool(getattr(config, "drone_enabled", False)),
            base_url=base_url,
            require_manual_arm=bool(getattr(config, "drone_require_manual_arm", True)),
            mission_timeout_s=_bounded_timeout_s(
                getattr(config, "drone_mission_timeout_s", _DEFAULT_DRONE_MISSION_TIMEOUT_S),
                default=_DEFAULT_DRONE_MISSION_TIMEOUT_S,
            ),
            request_timeout_s=_bounded_timeout_s(
                getattr(config, "drone_request_timeout_s", _DEFAULT_DRONE_TIMEOUT_S),
                default=_DEFAULT_DRONE_TIMEOUT_S,
            ),
        )


@dataclass(frozen=True, slots=True)
class DroneMissionRequest:
    """Describe one bounded high-level drone mission request."""

    mission_type: str
    target_hint: str
    capture_intent: str = "scene"
    max_duration_s: float = _DEFAULT_DRONE_MISSION_TIMEOUT_S
    return_policy: str = "return_and_land"

    def to_payload(self) -> dict[str, object]:
        """Serialize the request into the daemon wire contract."""

        return {
            "mission_type": self.mission_type,
            "target_hint": self.target_hint,
            "capture_intent": self.capture_intent,
            "max_duration_s": float(self.max_duration_s),
            "return_policy": self.return_policy,
        }


@dataclass(frozen=True, slots=True)
class DronePoseSnapshot:
    """Represent one normalized pose sample from the drone daemon."""

    healthy: bool
    tracking_state: str
    confidence: float
    source_timestamp: float | None = None
    x_m: float | None = None
    y_m: float | None = None
    z_m: float | None = None
    yaw_deg: float | None = None

    @classmethod
    def from_payload(cls, payload: object) -> "DronePoseSnapshot":
        """Build one pose snapshot from a daemon JSON object."""

        raw = payload if isinstance(payload, dict) else {}
        return cls(
            healthy=_coerce_bool(raw.get("healthy")),
            tracking_state=str(raw.get("tracking_state") or "unavailable"),
            confidence=float(_coerce_float(raw.get("confidence")) or 0.0),
            source_timestamp=_coerce_float(raw.get("source_timestamp")),
            x_m=_coerce_float(raw.get("x_m")),
            y_m=_coerce_float(raw.get("y_m")),
            z_m=_coerce_float(raw.get("z_m")),
            yaw_deg=_coerce_float(raw.get("yaw_deg")),
        )


@dataclass(frozen=True, slots=True)
class DroneSafetyStatus:
    """Represent the daemon's current arm-/mission-safety gate."""

    can_arm: bool
    manual_arm_required: bool
    radio_ready: bool
    pose_ready: bool
    motion_mode: str
    reasons: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, payload: object) -> "DroneSafetyStatus":
        """Build one safety snapshot from a daemon JSON object."""

        raw = payload if isinstance(payload, dict) else {}
        reasons = raw.get("reasons")
        if not isinstance(reasons, list):
            reasons = []
        return cls(
            can_arm=_coerce_bool(raw.get("can_arm")),
            manual_arm_required=_coerce_bool(raw.get("manual_arm_required"), default=True),
            radio_ready=_coerce_bool(raw.get("radio_ready")),
            pose_ready=_coerce_bool(raw.get("pose_ready")),
            motion_mode=str(raw.get("motion_mode") or "unknown"),
            reasons=tuple(str(item).strip() for item in reasons if str(item).strip()),
        )


@dataclass(frozen=True, slots=True)
class DroneStateSnapshot:
    """Represent one top-level drone-daemon state snapshot."""

    service_status: str
    active_mission_id: str | None
    manual_arm_required: bool
    skill_layer_mode: str
    radio_ready: bool
    pose: DronePoseSnapshot
    safety: DroneSafetyStatus

    @classmethod
    def from_payload(cls, payload: object) -> "DroneStateSnapshot":
        """Build one state snapshot from a daemon JSON object."""

        raw = payload if isinstance(payload, dict) else {}
        return cls(
            service_status=str(raw.get("service_status") or "unknown"),
            active_mission_id=str(raw.get("active_mission_id") or "").strip() or None,
            manual_arm_required=_coerce_bool(raw.get("manual_arm_required"), default=True),
            skill_layer_mode=str(raw.get("skill_layer_mode") or "unknown"),
            radio_ready=_coerce_bool(raw.get("radio_ready")),
            pose=DronePoseSnapshot.from_payload(raw.get("pose")),
            safety=DroneSafetyStatus.from_payload(raw.get("safety")),
        )


@dataclass(frozen=True, slots=True)
class DroneMissionStatus:
    """Represent one daemon-managed mission record."""

    mission_id: str
    mission_type: str
    state: str
    summary: str
    target_hint: str
    capture_intent: str
    max_duration_s: float
    return_policy: str
    requires_manual_arm: bool
    created_at: str
    updated_at: str
    artifact_name: str | None = None

    @classmethod
    def from_payload(cls, payload: object) -> "DroneMissionStatus":
        """Build one mission-status view from a daemon JSON object."""

        raw = payload if isinstance(payload, dict) else {}
        return cls(
            mission_id=str(raw.get("mission_id") or "").strip(),
            mission_type=str(raw.get("mission_type") or "inspect"),
            state=str(raw.get("state") or "unknown"),
            summary=str(raw.get("summary") or ""),
            target_hint=str(raw.get("target_hint") or ""),
            capture_intent=str(raw.get("capture_intent") or "scene"),
            max_duration_s=float(_coerce_float(raw.get("max_duration_s")) or _DEFAULT_DRONE_MISSION_TIMEOUT_S),
            return_policy=str(raw.get("return_policy") or "return_and_land"),
            requires_manual_arm=_coerce_bool(raw.get("requires_manual_arm"), default=True),
            created_at=str(raw.get("created_at") or ""),
            updated_at=str(raw.get("updated_at") or ""),
            artifact_name=str(raw.get("artifact_name") or "").strip() or None,
        )


class RemoteDroneServiceClient:
    """Speak Twinr's bounded drone-daemon HTTP contract."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_s: float = _DEFAULT_DRONE_TIMEOUT_S,
    ) -> None:
        normalized_base_url = _normalize_base_url(base_url)
        if not normalized_base_url:
            raise RuntimeError("Twinr drone support requires TWINR_DRONE_BASE_URL")
        self._base_url = normalized_base_url
        self._timeout_s = _bounded_timeout_s(timeout_s, default=_DEFAULT_DRONE_TIMEOUT_S)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "RemoteDroneServiceClient":
        """Build one remote drone client from ``TwinrConfig``."""

        drone_config = DroneServiceConfig.from_config(config)
        return cls(
            base_url=drone_config.base_url or "",
            timeout_s=drone_config.request_timeout_s,
        )

    def health_payload(self) -> dict[str, object]:
        """Return the daemon's health payload."""

        return self._request_json("GET", "/healthz")

    def state(self) -> DroneStateSnapshot:
        """Return the daemon's normalized state snapshot."""

        payload = self._request_json("GET", "/state")
        return DroneStateSnapshot.from_payload(payload.get("state"))

    def create_mission(self, request: DroneMissionRequest) -> DroneMissionStatus:
        """Create one bounded mission and return its initial status."""

        payload = self._request_json("POST", "/missions", payload=request.to_payload())
        return DroneMissionStatus.from_payload(payload.get("mission"))

    def create_inspect_mission(
        self,
        *,
        target_hint: str,
        capture_intent: str = "scene",
        max_duration_s: float = _DEFAULT_DRONE_MISSION_TIMEOUT_S,
        return_policy: str = "return_and_land",
    ) -> DroneMissionStatus:
        """Create one inspection mission through the bounded wire contract."""

        return self.create_mission(
            DroneMissionRequest(
                mission_type="inspect",
                target_hint=target_hint,
                capture_intent=capture_intent,
                max_duration_s=max_duration_s,
                return_policy=return_policy,
            )
        )

    def create_hover_test_mission(
        self,
        *,
        target_hint: str = "bounded hover test",
        max_duration_s: float = _DEFAULT_DRONE_MISSION_TIMEOUT_S,
    ) -> DroneMissionStatus:
        """Create one bounded takeoff-hover-land test mission."""

        return self.create_mission(
            DroneMissionRequest(
                mission_type="hover_test",
                target_hint=target_hint,
                capture_intent="hover_test",
                max_duration_s=max_duration_s,
                return_policy="return_and_land",
            )
        )

    def mission_status(self, mission_id: str) -> DroneMissionStatus:
        """Return one existing mission record."""

        payload = self._request_json("GET", f"/missions/{mission_id}")
        return DroneMissionStatus.from_payload(payload.get("mission"))

    def cancel_mission(self, mission_id: str) -> DroneMissionStatus:
        """Cancel one existing mission."""

        payload = self._request_json("POST", f"/missions/{mission_id}/cancel")
        return DroneMissionStatus.from_payload(payload.get("mission"))

    def manual_arm(self, mission_id: str) -> DroneMissionStatus:
        """Arm one mission through the daemon's operator-only endpoint."""

        payload = self._request_json("POST", f"/ops/missions/{mission_id}/arm")
        return DroneMissionStatus.from_payload(payload.get("mission"))

    def _request_json(
        self,
        method: str,
        route: str,
        *,
        payload: dict[str, object] | None = None,
        query: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Perform one bounded JSON request against the drone daemon."""

        normalized_route = route.strip().lstrip("/")
        if query:
            encoded_query = urlencode({key: str(value) for key, value in query.items() if value is not None})
            if encoded_query:
                normalized_route = f"{normalized_route}?{encoded_query}"
        body: bytes | None = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            f"{self._base_url}/{normalized_route}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self._timeout_s) as response:
                charset = response.headers.get_content_charset("utf-8")
                response_payload = json.loads(response.read().decode(charset))
        except HTTPError as exc:
            raise RuntimeError(_render_http_error(exc)) from exc
        except (URLError, TimeoutError, socket.timeout) as exc:
            raise RuntimeError(f"Twinr drone request failed: {_remote_error_code(exc)}") from exc
        if not isinstance(response_payload, dict):
            raise RuntimeError("Twinr drone daemon returned a non-object JSON payload")
        return response_payload


def _render_http_error(exc: HTTPError) -> str:
    """Normalize one HTTP error from the drone daemon."""

    try:
        body = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        body = ""
    if body:
        return f"Twinr drone request failed: HTTP {exc.code} {body}"
    return f"Twinr drone request failed: HTTP {exc.code}"


def _remote_error_code(exc: BaseException) -> str:
    """Render one short transport error token."""

    reason = getattr(exc, "reason", None)
    if isinstance(reason, BaseException):
        return reason.__class__.__name__
    if reason is not None:
        normalized_reason = str(reason).strip()
        if normalized_reason:
            return normalized_reason
    return exc.__class__.__name__
