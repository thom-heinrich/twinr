# CHANGELOG: 2026-03-28
# BUG-1: Config booleans were parsed with bool(...), so string values like "false" or "0" could enable drone behavior unexpectedly.
# BUG-2: drone_enabled and drone_mission_timeout_s existed in config but were not actually enforced; disabled configs could still build a live client and missions were not truly bounded.
# BUG-3: Response parsing silently coerced malformed daemon payloads into empty/unknown objects, hiding contract drift and producing incorrect mission/state data.
# BUG-4: mission_id path segments were interpolated without URL encoding, so malformed IDs could hit the wrong endpoint.
# SEC-1: urllib inherited proxy settings from the environment and could follow redirects, which could leak mission traffic or bypass the intended daemon boundary.
# SEC-2: Cleartext HTTP to non-loopback hosts was accepted by default, which is unsafe on real Raspberry Pi deployments unless explicitly opted into or replaced with HTTPS/UDS.
# IMP-1: Upgraded transport to a pooled persistent HTTPX client with split timeouts, bounded response sizes, connect retries, and optional Unix-domain-socket support.
# IMP-2: Added strict wire-contract validation, optional bearer/operator auth, best-effort Idempotency-Key support, structured exceptions, and legacy + enveloped response compatibility.

"""Reach a bounded drone-mission daemon over HTTP or a local Unix socket.

This module gives Twinr one strict client boundary for drone work. Twinr stays
above the flight layer and only submits bounded mission requests, reads status,
or cancels work. Direct attitude or thrust commands are intentionally absent.

Transport/security defaults are stricter than the legacy implementation:
- a persistent pooled client is used instead of one TCP connection per request
- environment proxies are ignored
- HTTP redirects are refused
- cleartext HTTP is only allowed to loopback by default
- local on-device deployments can use a Unix domain socket to avoid exposing
  a TCP port
"""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import json
import math
import re
import threading
from types import TracebackType
from typing import Any, Final
from urllib.parse import quote, urlsplit, urlunsplit
import uuid
import weakref

from twinr.agent.base_agent.config import TwinrConfig

_DEFAULT_DRONE_TIMEOUT_S: Final[float] = 2.0
_DEFAULT_DRONE_MISSION_TIMEOUT_S: Final[float] = 45.0
_DEFAULT_DRONE_CONNECT_TIMEOUT_S: Final[float] = 1.0
_DEFAULT_DRONE_POOL_TIMEOUT_S: Final[float] = 0.25
_DEFAULT_DRONE_CONNECT_RETRIES: Final[int] = 1
_DEFAULT_DRONE_MAX_RESPONSE_BYTES: Final[int] = 1_048_576
_DEFAULT_USER_AGENT: Final[str] = "TwinrDroneClient/2026.03"
_DEFAULT_ALLOWED_MISSION_TYPES: Final[tuple[str, ...]] = (
    "inspect",
    "hover_test",
    "inspect_local_zone",
)
_LOCAL_HOSTS: Final[frozenset[str]] = frozenset({"localhost", "127.0.0.1", "::1"})
_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_MISSING: Final[object] = object()


class DroneClientError(RuntimeError):
    """Base class for bounded drone-client failures."""


class DroneDisabledError(DroneClientError):
    """Raised when drone support is disabled in configuration."""


class DroneSecurityError(DroneClientError):
    """Raised when endpoint or transport settings violate safety policy."""


class DroneTransportError(DroneClientError):
    """Raised for network-transport failures."""


class DroneHTTPError(DroneClientError):
    """Raised for non-2xx HTTP responses."""


class DroneProtocolError(DroneClientError):
    """Raised when the daemon violates the expected JSON contract."""


def _load_httpx() -> Any:
    """Import HTTPX lazily so module import stays lightweight."""

    try:
        import httpx  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise DroneClientError(
            "Twinr drone client requires the 'httpx' package. Install httpx>=0.27 on the Raspberry Pi."
        ) from exc
    return httpx


def _normalize_base_url(value: object) -> str:
    """Normalize one optional HTTP base URL or return an empty string."""

    return str(value or "").strip().rstrip("/")


def _normalize_optional_text(value: object) -> str | None:
    """Normalize one optional text value."""

    normalized = str(value or "").strip()
    return normalized or None


def _normalize_uds_path(value: object) -> str | None:
    """Normalize one optional Unix-domain-socket path."""

    normalized = _normalize_optional_text(value)
    if not normalized:
        return None
    normalized_text = str(normalized)
    if "\x00" in normalized_text:
        raise DroneSecurityError("Twinr drone socket path must not contain NUL bytes")
    return normalized_text


def _bounded_timeout_s(value: object, *, default: float) -> float:
    """Return one bounded timeout in seconds."""

    try:
        parsed = float(value) if isinstance(value, (int, float, str)) else float(default)
    except (TypeError, ValueError):
        parsed = float(default)
    if not math.isfinite(parsed):
        parsed = float(default)
    return max(0.1, parsed)


def _bounded_positive_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    """Return one bounded positive integer."""

    try:
        parsed = int(value) if isinstance(value, (int, float, str)) else int(default)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(minimum, min(maximum, parsed))


def _coerce_bool(value: object, *, default: bool = False) -> bool:
    """Return one permissive boolean value for configuration parsing."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _coerce_token_tuple(value: object, *, default: tuple[str, ...]) -> tuple[str, ...]:
    """Parse one config value into a normalized token tuple.

    A single string value of "*" / "all" / "any" disables the allowlist and
    returns an empty tuple.
    """

    if value is None:
        return default
    items: list[str]
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return default
        if normalized.lower() in {"*", "all", "any"}:
            return ()
        items = re.split(r"[,\s]+", normalized)
    elif isinstance(value, (list, tuple, set, frozenset)):
        items = [str(item).strip() for item in value]
    else:
        return default
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not item:
            continue
        token = _normalize_token(item, field="config token")
        if token not in seen:
            seen.add(token)
            result.append(token)
    return tuple(result) if result else default


def _normalize_token(value: object, *, field: str) -> str:
    """Normalize one outbound token-like value."""

    if not isinstance(value, str):
        raise DroneProtocolError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise DroneProtocolError(f"{field} must not be empty")
    if not _TOKEN_RE.fullmatch(normalized):
        raise DroneProtocolError(
            f"{field} must match {_TOKEN_RE.pattern} and stay within 128 characters"
        )
    return normalized


def _normalize_text(value: object, *, field: str, max_len: int) -> str:
    """Normalize one outbound free-text value."""

    if not isinstance(value, str):
        raise DroneProtocolError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise DroneProtocolError(f"{field} must not be empty")
    if len(normalized) > max_len:
        raise DroneProtocolError(f"{field} must be <= {max_len} characters")
    return normalized


def _is_loopback_host(hostname: str | None) -> bool:
    """Return whether one hostname is clearly loopback/local-only."""

    if not hostname:
        return False
    normalized = hostname.strip().lower()
    if normalized in _LOCAL_HOSTS or normalized.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _validate_http_base_url(base_url: str, *, allow_insecure_http: bool, uds_path: str | None) -> str:
    """Validate and normalize one HTTP(S) base URL."""

    normalized = _normalize_base_url(base_url)
    if not normalized:
        if uds_path:
            return "http://localhost"
        raise DroneClientError("Twinr drone support requires drone_base_url or drone_socket_path")
    parts = urlsplit(normalized)
    if parts.scheme not in {"http", "https"}:
        raise DroneSecurityError("Twinr drone base URL must use http:// or https://")
    if not parts.netloc:
        raise DroneSecurityError("Twinr drone base URL must include a host")
    if parts.username or parts.password:
        raise DroneSecurityError("Twinr drone base URL must not embed credentials")
    if parts.query or parts.fragment:
        raise DroneSecurityError("Twinr drone base URL must not include query or fragment components")
    # BREAKING: non-loopback cleartext HTTP now requires explicit opt-in via drone_allow_insecure_http.
    if parts.scheme == "http" and not uds_path and not allow_insecure_http and not _is_loopback_host(parts.hostname):
        raise DroneSecurityError(
            "Refusing cleartext HTTP to a non-loopback drone daemon. Use HTTPS, a Unix domain socket, "
            "or set drone_allow_insecure_http=True intentionally."
        )
    normalized_path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme, parts.netloc, normalized_path, "", ""))


def _require_mapping(value: object, *, field: str) -> dict[str, object]:
    """Return one JSON object or raise."""

    if not isinstance(value, dict):
        raise DroneProtocolError(f"Twinr drone daemon returned invalid {field}: expected JSON object")
    return value


def _wire_string(
    raw: dict[str, object],
    key: str,
    *,
    default: str | None = _MISSING,  # type: ignore[assignment]
    allow_empty: bool = False,
    max_len: int = 256,
) -> str:
    """Read one string field from a daemon payload."""

    value = raw.get(key, _MISSING)
    if value is _MISSING:
        if default is _MISSING:
            raise DroneProtocolError(f"Twinr drone daemon payload is missing required field '{key}'")
        return default  # type: ignore[return-value]
    if not isinstance(value, str):
        raise DroneProtocolError(f"Twinr drone daemon field '{key}' must be a string")
    normalized = value.strip()
    if not normalized and not allow_empty:
        if default is not _MISSING:
            return default  # type: ignore[return-value]
        raise DroneProtocolError(f"Twinr drone daemon field '{key}' must not be empty")
    if len(normalized) > max_len:
        raise DroneProtocolError(f"Twinr drone daemon field '{key}' exceeds {max_len} characters")
    return normalized


def _wire_optional_string(
    raw: dict[str, object],
    key: str,
    *,
    max_len: int = 256,
) -> str | None:
    """Read one optional string field from a daemon payload."""

    value = raw.get(key, None)
    if value is None:
        return None
    if not isinstance(value, str):
        raise DroneProtocolError(f"Twinr drone daemon field '{key}' must be a string or null")
    normalized = value.strip()
    if not normalized:
        return None
    if len(normalized) > max_len:
        raise DroneProtocolError(f"Twinr drone daemon field '{key}' exceeds {max_len} characters")
    return normalized


def _wire_token(
    raw: dict[str, object],
    key: str,
    *,
    default: str | None = _MISSING,  # type: ignore[assignment]
) -> str:
    """Read one token-like field from a daemon payload."""

    value = _wire_string(raw, key, default=default, allow_empty=False, max_len=128)
    if not _TOKEN_RE.fullmatch(value):
        raise DroneProtocolError(f"Twinr drone daemon field '{key}' has invalid token syntax")
    return value


def _wire_bool(
    raw: dict[str, object],
    key: str,
    *,
    default: bool | object = _MISSING,
) -> bool:
    """Read one strict JSON boolean from a daemon payload."""

    value = raw.get(key, _MISSING)
    if value is _MISSING:
        if default is _MISSING:
            raise DroneProtocolError(f"Twinr drone daemon payload is missing required field '{key}'")
        return bool(default)
    # BREAKING: the wire contract now rejects stringly-typed booleans such as "true".
    if not isinstance(value, bool):
        raise DroneProtocolError(f"Twinr drone daemon field '{key}' must be a JSON boolean")
    return value


def _wire_float(
    raw: dict[str, object],
    key: str,
    *,
    default: float | None | object = _MISSING,
) -> float | None:
    """Read one strict JSON number from a daemon payload."""

    value = raw.get(key, _MISSING)
    if value is _MISSING or value is None:
        if default is _MISSING:
            raise DroneProtocolError(f"Twinr drone daemon payload is missing required field '{key}'")
        return default if default is not _MISSING else None  # type: ignore[return-value]
    # BREAKING: the wire contract now rejects stringly-typed numbers such as "45.0".
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DroneProtocolError(f"Twinr drone daemon field '{key}' must be a JSON number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise DroneProtocolError(f"Twinr drone daemon field '{key}' must be finite")
    return parsed


def _wire_string_tuple(raw: dict[str, object], key: str) -> tuple[str, ...]:
    """Read one list[str] field from a daemon payload."""

    value = raw.get(key, [])
    if value is None:
        return ()
    if not isinstance(value, list):
        raise DroneProtocolError(f"Twinr drone daemon field '{key}' must be a JSON array")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise DroneProtocolError(
                f"Twinr drone daemon field '{key}[{index}]' must be a string"
            )
        normalized = item.strip()
        if normalized:
            result.append(normalized)
    return tuple(result)


def _wire_optional_mapping(raw: dict[str, object], key: str) -> dict[str, object]:
    """Read one optional nested JSON object from a daemon payload."""

    value = raw.get(key, None)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise DroneProtocolError(f"Twinr drone daemon field '{key}' must be a JSON object")
    return value


def _wire_float_mapping(raw: dict[str, object], key: str) -> dict[str, float | None]:
    """Read one string-keyed float mapping from a daemon payload."""

    nested = _wire_optional_mapping(raw, key)
    result: dict[str, float | None] = {}
    for nested_key, nested_value in nested.items():
        if not isinstance(nested_key, str):
            raise DroneProtocolError(f"Twinr drone daemon field '{key}' must use string keys")
        if nested_value is None:
            result[nested_key] = None
            continue
        if isinstance(nested_value, bool) or not isinstance(nested_value, (int, float)):
            raise DroneProtocolError(
                f"Twinr drone daemon field '{key}.{nested_key}' must be a JSON number or null"
            )
        parsed = float(nested_value)
        if not math.isfinite(parsed):
            raise DroneProtocolError(
                f"Twinr drone daemon field '{key}.{nested_key}' must be finite"
            )
        result[nested_key] = parsed
    return result


def _wire_int_mapping(raw: dict[str, object], key: str) -> dict[str, int | None]:
    """Read one string-keyed int mapping from a daemon payload."""

    nested = _wire_optional_mapping(raw, key)
    result: dict[str, int | None] = {}
    for nested_key, nested_value in nested.items():
        if not isinstance(nested_key, str):
            raise DroneProtocolError(f"Twinr drone daemon field '{key}' must use string keys")
        if nested_value is None:
            result[nested_key] = None
            continue
        if isinstance(nested_value, bool) or not isinstance(nested_value, (int, float)):
            raise DroneProtocolError(
                f"Twinr drone daemon field '{key}.{nested_key}' must be a JSON number or null"
            )
        parsed = int(nested_value)
        result[nested_key] = parsed
    return result


def _new_request_id() -> str:
    """Create one unique request correlation identifier."""

    uuid_factory = getattr(uuid, "uuid7", uuid.uuid4)
    generated = uuid_factory()
    return str(generated)


def _quote_path_segment(value: str, *, field: str) -> str:
    """Validate and quote one path segment."""

    normalized = value.strip()
    if not normalized:
        raise DroneProtocolError(f"{field} must not be empty")
    if len(normalized) > 256:
        raise DroneProtocolError(f"{field} exceeds 256 characters")
    return quote(normalized, safe="")


def _query_params(query: dict[str, object] | None) -> dict[str, str] | None:
    """Normalize one query payload into string parameters."""

    if not query:
        return None
    params: dict[str, str] = {}
    for key, value in query.items():
        if value is None:
            continue
        params[str(key)] = str(value)
    return params or None


def _decode_json_object(raw_bytes: bytes, *, content_type: str) -> dict[str, object]:
    """Decode one bounded response body into a JSON object."""

    if not raw_bytes:
        raise DroneProtocolError("Twinr drone daemon returned an empty response body")
    if content_type and "json" not in content_type.lower():
        raise DroneProtocolError(
            f"Twinr drone daemon returned unexpected content-type {content_type!r}; expected JSON"
        )
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DroneProtocolError("Twinr drone daemon returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise DroneProtocolError("Twinr drone daemon returned a non-object JSON payload")
    return payload


def _error_body_excerpt(raw_bytes: bytes, *, limit: int = 256) -> str:
    """Render one short human-readable response excerpt."""

    if not raw_bytes:
        return ""
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
        if isinstance(payload, dict):
            for key in ("detail", "title", "error", "message"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    text = value.strip()
                    return text[:limit]
    except Exception:
        pass
    text = raw_bytes.decode("utf-8", errors="replace").strip()
    text = " ".join(text.split())
    return text[:limit]


def _unwrap_response_envelope(payload: dict[str, object]) -> dict[str, object]:
    """Support both legacy payloads and modern {success,data,error,meta} envelopes."""

    success = payload.get("success", _MISSING)
    if isinstance(success, bool) and ("data" in payload or "error" in payload or "meta" in payload):
        if not success:
            error_value = payload.get("error")
            if isinstance(error_value, dict):
                detail = (
                    _wire_optional_string(error_value, "detail", max_len=512)
                    or _wire_optional_string(error_value, "message", max_len=512)
                    or _wire_optional_string(error_value, "code", max_len=128)
                    or "unknown daemon error"
                )
            else:
                detail = str(error_value or "unknown daemon error").strip() or "unknown daemon error"
            raise DroneProtocolError(f"Twinr drone daemon returned success=false: {detail}")
        data = payload.get("data")
        return _require_mapping(data, field="data")
    return payload


def _render_http_error(status_code: int, raw_bytes: bytes) -> str:
    """Normalize one non-2xx HTTP response."""

    excerpt = _error_body_excerpt(raw_bytes)
    if excerpt:
        return f"Twinr drone request failed: HTTP {status_code} {excerpt}"
    return f"Twinr drone request failed: HTTP {status_code}"


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


@dataclass(frozen=True, slots=True)
class DroneServiceConfig:
    """Store Twinr's remote drone-daemon wiring."""

    enabled: bool
    base_url: str | None
    uds_path: str | None = None
    auth_token: str | None = None
    ops_auth_token: str | None = None
    allow_insecure_http: bool = False
    require_manual_arm: bool = True
    mission_timeout_s: float = _DEFAULT_DRONE_MISSION_TIMEOUT_S
    request_timeout_s: float = _DEFAULT_DRONE_TIMEOUT_S
    connect_timeout_s: float = _DEFAULT_DRONE_CONNECT_TIMEOUT_S
    read_timeout_s: float = _DEFAULT_DRONE_TIMEOUT_S
    write_timeout_s: float = _DEFAULT_DRONE_TIMEOUT_S
    pool_timeout_s: float = _DEFAULT_DRONE_POOL_TIMEOUT_S
    connect_retries: int = _DEFAULT_DRONE_CONNECT_RETRIES
    max_response_bytes: int = _DEFAULT_DRONE_MAX_RESPONSE_BYTES
    allowed_mission_types: tuple[str, ...] = _DEFAULT_ALLOWED_MISSION_TYPES
    allowed_return_policies: tuple[str, ...] = ()
    http2_enabled: bool = False

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DroneServiceConfig":
        """Build one normalized drone-service config from ``TwinrConfig``."""

        request_timeout_s = _bounded_timeout_s(
            getattr(config, "drone_request_timeout_s", _DEFAULT_DRONE_TIMEOUT_S),
            default=_DEFAULT_DRONE_TIMEOUT_S,
        )
        connect_timeout_default = min(_DEFAULT_DRONE_CONNECT_TIMEOUT_S, request_timeout_s)
        pool_timeout_default = min(_DEFAULT_DRONE_POOL_TIMEOUT_S, request_timeout_s)
        return cls(
            enabled=_coerce_bool(getattr(config, "drone_enabled", False)),
            base_url=_normalize_base_url(getattr(config, "drone_base_url", None)) or None,
            uds_path=_normalize_uds_path(
                getattr(config, "drone_socket_path", getattr(config, "drone_uds_path", None))
            ),
            auth_token=_normalize_optional_text(getattr(config, "drone_auth_token", None)),
            ops_auth_token=_normalize_optional_text(
                getattr(
                    config,
                    "drone_ops_auth_token",
                    getattr(config, "drone_operator_auth_token", None),
                )
            ),
            allow_insecure_http=_coerce_bool(getattr(config, "drone_allow_insecure_http", False)),
            require_manual_arm=_coerce_bool(getattr(config, "drone_require_manual_arm", True), default=True),
            mission_timeout_s=_bounded_timeout_s(
                getattr(config, "drone_mission_timeout_s", _DEFAULT_DRONE_MISSION_TIMEOUT_S),
                default=_DEFAULT_DRONE_MISSION_TIMEOUT_S,
            ),
            request_timeout_s=request_timeout_s,
            connect_timeout_s=_bounded_timeout_s(
                getattr(config, "drone_connect_timeout_s", connect_timeout_default),
                default=connect_timeout_default,
            ),
            read_timeout_s=_bounded_timeout_s(
                getattr(config, "drone_read_timeout_s", request_timeout_s),
                default=request_timeout_s,
            ),
            write_timeout_s=_bounded_timeout_s(
                getattr(config, "drone_write_timeout_s", request_timeout_s),
                default=request_timeout_s,
            ),
            pool_timeout_s=_bounded_timeout_s(
                getattr(config, "drone_pool_timeout_s", pool_timeout_default),
                default=pool_timeout_default,
            ),
            connect_retries=_bounded_positive_int(
                getattr(config, "drone_connect_retries", _DEFAULT_DRONE_CONNECT_RETRIES),
                default=_DEFAULT_DRONE_CONNECT_RETRIES,
                minimum=0,
                maximum=3,
            ),
            max_response_bytes=_bounded_positive_int(
                getattr(config, "drone_max_response_bytes", _DEFAULT_DRONE_MAX_RESPONSE_BYTES),
                default=_DEFAULT_DRONE_MAX_RESPONSE_BYTES,
                minimum=4_096,
                maximum=8_388_608,
            ),
            # BREAKING: the bounded client boundary now allow-lists mission types by default.
            # Set drone_allowed_mission_types="*" only if you intentionally want to disable this.
            allowed_mission_types=_coerce_token_tuple(
                getattr(config, "drone_allowed_mission_types", None),
                default=_DEFAULT_ALLOWED_MISSION_TYPES,
            ),
            allowed_return_policies=_coerce_token_tuple(
                getattr(config, "drone_allowed_return_policies", None),
                default=(),
            ),
            http2_enabled=_coerce_bool(getattr(config, "drone_http2_enabled", False)),
        )


@dataclass(frozen=True, slots=True)
class DroneMissionRequest:
    """Describe one bounded high-level drone mission request."""

    mission_type: str
    target_hint: str
    capture_intent: str = "scene"
    max_duration_s: float = _DEFAULT_DRONE_MISSION_TIMEOUT_S
    return_policy: str = "return_and_land"

    def normalized(
        self,
        *,
        mission_timeout_limit_s: float,
        allowed_mission_types: tuple[str, ...],
        allowed_return_policies: tuple[str, ...],
    ) -> "DroneMissionRequest":
        """Return one validated and bounded mission request."""

        mission_type = _normalize_token(self.mission_type, field="mission_type")
        if allowed_mission_types and mission_type not in allowed_mission_types:
            raise DroneProtocolError(
                f"mission_type={mission_type!r} is outside the bounded client allowlist {allowed_mission_types!r}"
            )
        capture_intent = _normalize_token(self.capture_intent, field="capture_intent")
        return_policy = _normalize_token(self.return_policy, field="return_policy")
        if allowed_return_policies and return_policy not in allowed_return_policies:
            raise DroneProtocolError(
                f"return_policy={return_policy!r} is outside the configured allowlist {allowed_return_policies!r}"
            )
        target_hint = _normalize_text(self.target_hint, field="target_hint", max_len=512)
        bounded_duration = min(
            _bounded_timeout_s(self.max_duration_s, default=mission_timeout_limit_s),
            mission_timeout_limit_s,
        )
        return DroneMissionRequest(
            mission_type=mission_type,
            target_hint=target_hint,
            capture_intent=capture_intent,
            max_duration_s=bounded_duration,
            return_policy=return_policy,
        )

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

        raw = _require_mapping(payload, field="pose")
        confidence = _wire_float(raw, "confidence", default=0.0)
        return cls(
            healthy=_wire_bool(raw, "healthy", default=False),
            tracking_state=_wire_token(raw, "tracking_state", default="unavailable"),
            confidence=float(confidence if confidence is not None else 0.0),
            source_timestamp=_wire_float(raw, "source_timestamp", default=None),
            x_m=_wire_float(raw, "x_m", default=None),
            y_m=_wire_float(raw, "y_m", default=None),
            z_m=_wire_float(raw, "z_m", default=None),
            yaw_deg=_wire_float(raw, "yaw_deg", default=None),
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

        raw = _require_mapping(payload, field="safety")
        return cls(
            can_arm=_wire_bool(raw, "can_arm", default=False),
            manual_arm_required=_wire_bool(raw, "manual_arm_required", default=True),
            radio_ready=_wire_bool(raw, "radio_ready", default=False),
            pose_ready=_wire_bool(raw, "pose_ready", default=False),
            motion_mode=_wire_token(raw, "motion_mode", default="unknown"),
            reasons=_wire_string_tuple(raw, "reasons"),
        )


@dataclass(frozen=True, slots=True)
class DroneTelemetryCatalog:
    """Represent bounded telemetry catalog counts from the daemon state."""

    log_group_count: int
    log_variable_count: int
    param_group_count: int
    param_count: int

    @classmethod
    def from_payload(cls, payload: object) -> "DroneTelemetryCatalog":
        raw = _require_mapping(payload, field="telemetry.catalog")
        return cls(
            log_group_count=int(_wire_float(raw, "log_group_count", default=0.0) or 0.0),
            log_variable_count=int(_wire_float(raw, "log_variable_count", default=0.0) or 0.0),
            param_group_count=int(_wire_float(raw, "param_group_count", default=0.0) or 0.0),
            param_count=int(_wire_float(raw, "param_count", default=0.0) or 0.0),
        )


@dataclass(frozen=True, slots=True)
class DroneDeckTelemetrySnapshot:
    """Represent the daemon's current deck-flag snapshot."""

    flags: dict[str, int | None]
    refreshed_age_s: float | None

    @classmethod
    def from_payload(cls, payload: object) -> "DroneDeckTelemetrySnapshot":
        raw = _require_mapping(payload, field="telemetry.deck")
        return cls(
            flags=_wire_int_mapping(raw, "flags"),
            refreshed_age_s=_wire_float(raw, "refreshed_age_s", default=None),
        )


@dataclass(frozen=True, slots=True)
class DronePowerTelemetrySnapshot:
    """Represent the daemon's current power telemetry snapshot."""

    vbat_v: float | None
    battery_level: int | None
    state: int | None
    state_name: str

    @classmethod
    def from_payload(cls, payload: object) -> "DronePowerTelemetrySnapshot":
        raw = _require_mapping(payload, field="telemetry.power")
        battery_level = _wire_float(raw, "battery_level", default=None)
        state = _wire_float(raw, "state", default=None)
        return cls(
            vbat_v=_wire_float(raw, "vbat_v", default=None),
            battery_level=None if battery_level is None else int(battery_level),
            state=None if state is None else int(state),
            state_name=_wire_token(raw, "state_name", default="unknown"),
        )


@dataclass(frozen=True, slots=True)
class DroneRangeTelemetrySnapshot:
    """Represent the daemon's current range telemetry snapshot."""

    zrange_m: float | None
    front_m: float | None
    back_m: float | None
    left_m: float | None
    right_m: float | None
    up_m: float | None
    downward_observed: bool

    @classmethod
    def from_payload(cls, payload: object) -> "DroneRangeTelemetrySnapshot":
        raw = _require_mapping(payload, field="telemetry.range")
        return cls(
            zrange_m=_wire_float(raw, "zrange_m", default=None),
            front_m=_wire_float(raw, "front_m", default=None),
            back_m=_wire_float(raw, "back_m", default=None),
            left_m=_wire_float(raw, "left_m", default=None),
            right_m=_wire_float(raw, "right_m", default=None),
            up_m=_wire_float(raw, "up_m", default=None),
            downward_observed=_wire_bool(raw, "downward_observed", default=False),
        )


@dataclass(frozen=True, slots=True)
class DroneFlightTelemetrySnapshot:
    """Represent the daemon's current flight telemetry snapshot."""

    roll_deg: float | None
    pitch_deg: float | None
    yaw_deg: float | None
    x_m: float | None
    y_m: float | None
    z_m: float | None
    vx_mps: float | None
    vy_mps: float | None
    vz_mps: float | None
    thrust: float | None
    motion_squal: int | None
    supervisor_info: int | None
    can_fly: bool | None
    is_flying: bool | None
    unsafe_supervisor_flags: tuple[str, ...]

    @classmethod
    def from_payload(cls, payload: object) -> "DroneFlightTelemetrySnapshot":
        raw = _require_mapping(payload, field="telemetry.flight")
        motion_squal = _wire_float(raw, "motion_squal", default=None)
        supervisor_info = _wire_float(raw, "supervisor_info", default=None)
        can_fly = raw.get("can_fly", None)
        is_flying = raw.get("is_flying", None)
        if can_fly is not None and not isinstance(can_fly, bool):
            raise DroneProtocolError("Twinr drone daemon field 'telemetry.flight.can_fly' must be a JSON boolean or null")
        if is_flying is not None and not isinstance(is_flying, bool):
            raise DroneProtocolError("Twinr drone daemon field 'telemetry.flight.is_flying' must be a JSON boolean or null")
        return cls(
            roll_deg=_wire_float(raw, "roll_deg", default=None),
            pitch_deg=_wire_float(raw, "pitch_deg", default=None),
            yaw_deg=_wire_float(raw, "yaw_deg", default=None),
            x_m=_wire_float(raw, "x_m", default=None),
            y_m=_wire_float(raw, "y_m", default=None),
            z_m=_wire_float(raw, "z_m", default=None),
            vx_mps=_wire_float(raw, "vx_mps", default=None),
            vy_mps=_wire_float(raw, "vy_mps", default=None),
            vz_mps=_wire_float(raw, "vz_mps", default=None),
            thrust=_wire_float(raw, "thrust", default=None),
            motion_squal=None if motion_squal is None else int(motion_squal),
            supervisor_info=None if supervisor_info is None else int(supervisor_info),
            can_fly=can_fly,
            is_flying=is_flying,
            unsafe_supervisor_flags=_wire_string_tuple(raw, "unsafe_supervisor_flags"),
        )


@dataclass(frozen=True, slots=True)
class DroneLinkTelemetrySnapshot:
    """Represent the daemon's current radio/link telemetry snapshot."""

    radio_connected: bool | None
    rssi_dbm: float | None
    observation_age_s: float | None
    latency_ms: float | None
    link_quality: float | None
    uplink_rssi: float | None
    uplink_rate_hz: float | None
    downlink_rate_hz: float | None
    uplink_congestion: float | None
    downlink_congestion: float | None
    monitor_available: bool
    monitor_failure: str | None

    @classmethod
    def from_payload(cls, payload: object) -> "DroneLinkTelemetrySnapshot":
        raw = _require_mapping(payload, field="telemetry.link")
        radio_connected = raw.get("radio_connected", None)
        if radio_connected is not None and not isinstance(radio_connected, bool):
            raise DroneProtocolError("Twinr drone daemon field 'telemetry.link.radio_connected' must be a JSON boolean or null")
        return cls(
            radio_connected=radio_connected,
            rssi_dbm=_wire_float(raw, "rssi_dbm", default=None),
            observation_age_s=_wire_float(raw, "observation_age_s", default=None),
            latency_ms=_wire_float(raw, "latency_ms", default=None),
            link_quality=_wire_float(raw, "link_quality", default=None),
            uplink_rssi=_wire_float(raw, "uplink_rssi", default=None),
            uplink_rate_hz=_wire_float(raw, "uplink_rate_hz", default=None),
            downlink_rate_hz=_wire_float(raw, "downlink_rate_hz", default=None),
            uplink_congestion=_wire_float(raw, "uplink_congestion", default=None),
            downlink_congestion=_wire_float(raw, "downlink_congestion", default=None),
            monitor_available=_wire_bool(raw, "monitor_available", default=False),
            monitor_failure=_wire_optional_string(raw, "monitor_failure", max_len=256),
        )


@dataclass(frozen=True, slots=True)
class DroneFailsafeTelemetrySnapshot:
    """Represent the daemon's current `twinrFs` telemetry snapshot."""

    loaded: bool
    protocol_version: int | None
    enabled: bool | None
    state_code: int | None
    state_name: str | None
    reason_code: int | None
    reason_name: str | None
    heartbeat_age_ms: int | None
    last_status_received_age_s: float | None
    session_id: int | None
    rejected_packets: int | None
    last_reject_code: int | None

    @classmethod
    def from_payload(cls, payload: object) -> "DroneFailsafeTelemetrySnapshot":
        raw = _require_mapping(payload, field="telemetry.failsafe")
        enabled = raw.get("enabled", None)
        if enabled is not None and not isinstance(enabled, bool):
            raise DroneProtocolError("Twinr drone daemon field 'telemetry.failsafe.enabled' must be a JSON boolean or null")
        return cls(
            loaded=_wire_bool(raw, "loaded", default=False),
            protocol_version=_wire_optional_int(raw, "protocol_version"),
            enabled=enabled,
            state_code=_wire_optional_int(raw, "state_code"),
            state_name=_wire_optional_string(raw, "state_name", max_len=128),
            reason_code=_wire_optional_int(raw, "reason_code"),
            reason_name=_wire_optional_string(raw, "reason_name", max_len=128),
            heartbeat_age_ms=_wire_optional_int(raw, "heartbeat_age_ms"),
            last_status_received_age_s=_wire_float(raw, "last_status_received_age_s", default=None),
            session_id=_wire_optional_int(raw, "session_id"),
            rejected_packets=_wire_optional_int(raw, "rejected_packets"),
            last_reject_code=_wire_optional_int(raw, "last_reject_code"),
        )


@dataclass(frozen=True, slots=True)
class DroneCommandTelemetrySnapshot:
    """Represent the daemon's current command-state snapshot."""

    mission_name: str | None
    phase: str
    phase_status: str | None
    age_s: float | None
    target_height_m: float | None
    hover_duration_s: float | None
    forward_m: float | None
    left_m: float | None
    translation_velocity_mps: float | None
    takeoff_confirmed: bool
    aborted: bool
    abort_reason: str | None
    last_message: str | None

    @classmethod
    def from_payload(cls, payload: object) -> "DroneCommandTelemetrySnapshot":
        raw = _require_mapping(payload, field="telemetry.command")
        return cls(
            mission_name=_wire_optional_string(raw, "mission_name", max_len=128),
            phase=_wire_token(raw, "phase", default="idle"),
            phase_status=_wire_optional_string(raw, "phase_status", max_len=128),
            age_s=_wire_float(raw, "age_s", default=None),
            target_height_m=_wire_float(raw, "target_height_m", default=None),
            hover_duration_s=_wire_float(raw, "hover_duration_s", default=None),
            forward_m=_wire_float(raw, "forward_m", default=None),
            left_m=_wire_float(raw, "left_m", default=None),
            translation_velocity_mps=_wire_float(raw, "translation_velocity_mps", default=None),
            takeoff_confirmed=_wire_bool(raw, "takeoff_confirmed", default=False),
            aborted=_wire_bool(raw, "aborted", default=False),
            abort_reason=_wire_optional_string(raw, "abort_reason", max_len=256),
            last_message=_wire_optional_string(raw, "last_message", max_len=256),
        )


@dataclass(frozen=True, slots=True)
class DroneTelemetryDivergence:
    """Represent one bounded command-vs-observed divergence event."""

    code: str
    severity: str
    message: str

    @classmethod
    def from_payload(cls, payload: object) -> "DroneTelemetryDivergence":
        raw = _require_mapping(payload, field="telemetry.divergence")
        return cls(
            code=_wire_token(raw, "code"),
            severity=_wire_token(raw, "severity", default="warning"),
            message=_wire_string(raw, "message", max_len=512),
        )


@dataclass(frozen=True, slots=True)
class DroneTelemetrySnapshot:
    """Represent one bounded runtime telemetry snapshot from the daemon."""

    profile: str
    collected_at: str
    healthy: bool
    failures: tuple[str, ...]
    freshness_by_signal: dict[str, float | None]
    catalog: DroneTelemetryCatalog
    deck: DroneDeckTelemetrySnapshot
    power: DronePowerTelemetrySnapshot
    range: DroneRangeTelemetrySnapshot
    flight: DroneFlightTelemetrySnapshot
    link: DroneLinkTelemetrySnapshot
    failsafe: DroneFailsafeTelemetrySnapshot
    command: DroneCommandTelemetrySnapshot
    divergences: tuple[DroneTelemetryDivergence, ...]
    available_blocks: tuple[str, ...]
    skipped_blocks: tuple[str, ...]

    @classmethod
    def from_payload(cls, payload: object) -> "DroneTelemetrySnapshot":
        raw = _require_mapping(payload, field="telemetry")
        divergences_raw = raw.get("divergences", [])
        if divergences_raw is None:
            divergences_raw = []
        if not isinstance(divergences_raw, list):
            raise DroneProtocolError("Twinr drone daemon field 'telemetry.divergences' must be a JSON array")
        return cls(
            profile=_wire_token(raw, "profile", default="operator"),
            collected_at=_wire_string(raw, "collected_at", default="", allow_empty=True, max_len=128),
            healthy=_wire_bool(raw, "healthy", default=False),
            failures=_wire_string_tuple(raw, "failures"),
            freshness_by_signal=_wire_float_mapping(raw, "freshness_by_signal"),
            catalog=DroneTelemetryCatalog.from_payload(raw.get("catalog")),
            deck=DroneDeckTelemetrySnapshot.from_payload(raw.get("deck")),
            power=DronePowerTelemetrySnapshot.from_payload(raw.get("power")),
            range=DroneRangeTelemetrySnapshot.from_payload(raw.get("range")),
            flight=DroneFlightTelemetrySnapshot.from_payload(raw.get("flight")),
            link=DroneLinkTelemetrySnapshot.from_payload(raw.get("link")),
            failsafe=DroneFailsafeTelemetrySnapshot.from_payload(raw.get("failsafe")),
            command=DroneCommandTelemetrySnapshot.from_payload(raw.get("command")),
            divergences=tuple(DroneTelemetryDivergence.from_payload(item) for item in divergences_raw),
            available_blocks=_wire_string_tuple(raw, "available_blocks"),
            skipped_blocks=_wire_string_tuple(raw, "skipped_blocks"),
        )


def _wire_optional_int(raw: dict[str, object], key: str) -> int | None:
    """Read one optional integer field from a daemon payload."""

    value = raw.get(key, None)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DroneProtocolError(f"Twinr drone daemon field '{key}' must be a JSON number or null")
    return int(value)


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
    telemetry: DroneTelemetrySnapshot | None = None

    @classmethod
    def from_payload(cls, payload: object) -> "DroneStateSnapshot":
        """Build one state snapshot from a daemon JSON object."""

        raw = _require_mapping(payload, field="state")
        return cls(
            service_status=_wire_token(raw, "service_status", default="unknown"),
            active_mission_id=_wire_optional_string(raw, "active_mission_id", max_len=256),
            manual_arm_required=_wire_bool(raw, "manual_arm_required", default=True),
            skill_layer_mode=_wire_token(raw, "skill_layer_mode", default="unknown"),
            radio_ready=_wire_bool(raw, "radio_ready", default=False),
            pose=DronePoseSnapshot.from_payload(raw.get("pose")),
            safety=DroneSafetyStatus.from_payload(raw.get("safety")),
            telemetry=(
                None
                if raw.get("telemetry", None) is None
                else DroneTelemetrySnapshot.from_payload(raw.get("telemetry"))
            ),
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

        raw = _require_mapping(payload, field="mission")
        max_duration = _wire_float(raw, "max_duration_s", default=_DEFAULT_DRONE_MISSION_TIMEOUT_S)
        return cls(
            mission_id=_wire_string(raw, "mission_id", max_len=256),
            mission_type=_wire_token(raw, "mission_type", default="inspect"),
            state=_wire_token(raw, "state"),
            summary=_wire_string(raw, "summary", default="", allow_empty=True, max_len=2_048),
            target_hint=_wire_string(raw, "target_hint", default="", allow_empty=True, max_len=1_024),
            capture_intent=_wire_token(raw, "capture_intent", default="scene"),
            max_duration_s=float(max_duration if max_duration is not None else _DEFAULT_DRONE_MISSION_TIMEOUT_S),
            return_policy=_wire_token(raw, "return_policy", default="return_and_land"),
            requires_manual_arm=_wire_bool(raw, "requires_manual_arm", default=True),
            created_at=_wire_string(raw, "created_at", default="", allow_empty=True, max_len=128),
            updated_at=_wire_string(raw, "updated_at", default="", allow_empty=True, max_len=128),
            artifact_name=_wire_optional_string(raw, "artifact_name", max_len=256),
        )


class RemoteDroneServiceClient:
    """Speak Twinr's bounded drone-daemon HTTP contract."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        uds_path: str | None = None,
        timeout_s: float = _DEFAULT_DRONE_TIMEOUT_S,
        connect_timeout_s: float | None = None,
        read_timeout_s: float | None = None,
        write_timeout_s: float | None = None,
        pool_timeout_s: float | None = None,
        auth_token: str | None = None,
        ops_auth_token: str | None = None,
        allow_insecure_http: bool = False,
        mission_timeout_limit_s: float = _DEFAULT_DRONE_MISSION_TIMEOUT_S,
        max_response_bytes: int = _DEFAULT_DRONE_MAX_RESPONSE_BYTES,
        connect_retries: int = _DEFAULT_DRONE_CONNECT_RETRIES,
        allowed_mission_types: tuple[str, ...] = _DEFAULT_ALLOWED_MISSION_TYPES,
        allowed_return_policies: tuple[str, ...] = (),
        require_manual_arm: bool = True,
        http2_enabled: bool = False,
    ) -> None:
        self._httpx = _load_httpx()
        self._uds_path = _normalize_uds_path(uds_path)
        self._base_url = _validate_http_base_url(
            base_url or "",
            allow_insecure_http=allow_insecure_http,
            uds_path=self._uds_path,
        )
        self._base_parts = urlsplit(self._base_url)
        self._auth_token = _normalize_optional_text(auth_token)
        self._ops_auth_token = _normalize_optional_text(ops_auth_token)
        self._mission_timeout_limit_s = _bounded_timeout_s(
            mission_timeout_limit_s,
            default=_DEFAULT_DRONE_MISSION_TIMEOUT_S,
        )
        self._max_response_bytes = _bounded_positive_int(
            max_response_bytes,
            default=_DEFAULT_DRONE_MAX_RESPONSE_BYTES,
            minimum=4_096,
            maximum=8_388_608,
        )
        self._connect_retries = _bounded_positive_int(
            connect_retries,
            default=_DEFAULT_DRONE_CONNECT_RETRIES,
            minimum=0,
            maximum=3,
        )
        self._allowed_mission_types = tuple(allowed_mission_types)
        self._allowed_return_policies = tuple(allowed_return_policies)
        self._require_manual_arm = bool(require_manual_arm)
        effective_timeout_s = _bounded_timeout_s(timeout_s, default=_DEFAULT_DRONE_TIMEOUT_S)
        effective_connect_timeout_s = _bounded_timeout_s(
            connect_timeout_s if connect_timeout_s is not None else min(_DEFAULT_DRONE_CONNECT_TIMEOUT_S, effective_timeout_s),
            default=min(_DEFAULT_DRONE_CONNECT_TIMEOUT_S, effective_timeout_s),
        )
        effective_pool_timeout_s = _bounded_timeout_s(
            pool_timeout_s if pool_timeout_s is not None else min(_DEFAULT_DRONE_POOL_TIMEOUT_S, effective_timeout_s),
            default=min(_DEFAULT_DRONE_POOL_TIMEOUT_S, effective_timeout_s),
        )
        effective_read_timeout_s = _bounded_timeout_s(
            read_timeout_s if read_timeout_s is not None else effective_timeout_s,
            default=effective_timeout_s,
        )
        effective_write_timeout_s = _bounded_timeout_s(
            write_timeout_s if write_timeout_s is not None else effective_timeout_s,
            default=effective_timeout_s,
        )
        transport = self._httpx.HTTPTransport(
            retries=self._connect_retries,
            uds=self._uds_path,
        )
        timeout = self._httpx.Timeout(
            connect=effective_connect_timeout_s,
            read=effective_read_timeout_s,
            write=effective_write_timeout_s,
            pool=effective_pool_timeout_s,
        )
        limits = self._httpx.Limits(
            max_connections=4,
            max_keepalive_connections=2,
            keepalive_expiry=15.0,
        )
        self._client = self._httpx.Client(
            transport=transport,
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
            limits=limits,
            http2=http2_enabled,
            headers={
                "Accept": "application/json",
                "User-Agent": _DEFAULT_USER_AGENT,
            },
        )
        self._close_lock = threading.Lock()
        self._closed = False
        self._finalizer = weakref.finalize(self, self._close_client, self._client)

    @staticmethod
    def _close_client(client: Any) -> None:
        """Close one underlying HTTP client quietly."""

        try:
            client.close()
        except Exception:
            return

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "RemoteDroneServiceClient":
        """Build one remote drone client from ``TwinrConfig``."""

        drone_config = DroneServiceConfig.from_config(config)
        # BREAKING: disabled drone configs now fail fast instead of constructing a live-capable client.
        if not drone_config.enabled:
            raise DroneDisabledError("Twinr drone support is disabled")
        return cls(
            base_url=drone_config.base_url,
            uds_path=drone_config.uds_path,
            timeout_s=drone_config.request_timeout_s,
            connect_timeout_s=drone_config.connect_timeout_s,
            read_timeout_s=drone_config.read_timeout_s,
            write_timeout_s=drone_config.write_timeout_s,
            pool_timeout_s=drone_config.pool_timeout_s,
            auth_token=drone_config.auth_token,
            ops_auth_token=drone_config.ops_auth_token,
            allow_insecure_http=drone_config.allow_insecure_http,
            mission_timeout_limit_s=drone_config.mission_timeout_s,
            max_response_bytes=drone_config.max_response_bytes,
            connect_retries=drone_config.connect_retries,
            allowed_mission_types=drone_config.allowed_mission_types,
            allowed_return_policies=drone_config.allowed_return_policies,
            require_manual_arm=drone_config.require_manual_arm,
            http2_enabled=drone_config.http2_enabled,
        )

    def close(self) -> None:
        """Close the pooled HTTP client."""

        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._finalizer()

    def __enter__(self) -> "RemoteDroneServiceClient":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def health_payload(self) -> dict[str, object]:
        """Return the daemon's health payload."""

        return self._request_json("GET", "/healthz")

    def state(self) -> DroneStateSnapshot:
        """Return the daemon's normalized state snapshot."""

        payload = self._request_json("GET", "/state")
        return DroneStateSnapshot.from_payload(payload.get("state", payload))

    def create_mission(self, request: DroneMissionRequest) -> DroneMissionStatus:
        """Create one bounded mission and return its initial status."""

        normalized_request = request.normalized(
            mission_timeout_limit_s=self._mission_timeout_limit_s,
            allowed_mission_types=self._allowed_mission_types,
            allowed_return_policies=self._allowed_return_policies,
        )
        payload = self._request_json(
            "POST",
            "/missions",
            payload=normalized_request.to_payload(),
            extra_headers={
                "Idempotency-Key": _new_request_id(),
                "X-Twinr-Manual-Arm-Required": "1" if self._require_manual_arm else "0",
            },
        )
        return DroneMissionStatus.from_payload(payload.get("mission", payload))

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

    def create_inspect_local_zone_mission(
        self,
        *,
        target_hint: str = "local inspect zone",
        capture_intent: str = "scene",
        max_duration_s: float = _DEFAULT_DRONE_MISSION_TIMEOUT_S,
        return_policy: str = "return_and_land",
    ) -> DroneMissionStatus:
        """Create one bounded local inspect mission."""

        return self.create_mission(
            DroneMissionRequest(
                mission_type="inspect_local_zone",
                target_hint=target_hint,
                capture_intent=capture_intent,
                max_duration_s=max_duration_s,
                return_policy=return_policy,
            )
        )

    def mission_status(self, mission_id: str) -> DroneMissionStatus:
        """Return one existing mission record."""

        safe_mission_id = _quote_path_segment(mission_id, field="mission_id")
        payload = self._request_json("GET", f"/missions/{safe_mission_id}")
        return DroneMissionStatus.from_payload(payload.get("mission", payload))

    def cancel_mission(self, mission_id: str) -> DroneMissionStatus:
        """Cancel one existing mission."""

        safe_mission_id = _quote_path_segment(mission_id, field="mission_id")
        payload = self._request_json(
            "POST",
            f"/missions/{safe_mission_id}/cancel",
            extra_headers={"Idempotency-Key": _new_request_id()},
        )
        return DroneMissionStatus.from_payload(payload.get("mission", payload))

    def manual_arm(self, mission_id: str) -> DroneMissionStatus:
        """Arm one mission through the daemon's operator-only endpoint."""

        safe_mission_id = _quote_path_segment(mission_id, field="mission_id")
        extra_headers: dict[str, str] = {"Idempotency-Key": _new_request_id()}
        if self._ops_auth_token:
            extra_headers["Authorization"] = f"Bearer {self._ops_auth_token}"
        payload = self._request_json(
            "POST",
            f"/ops/missions/{safe_mission_id}/arm",
            extra_headers=extra_headers,
        )
        return DroneMissionStatus.from_payload(payload.get("mission", payload))

    def _build_url(self, route: str) -> str:
        """Build one absolute request URL while preserving any base-path prefix."""

        normalized_route = route.strip().lstrip("/")
        if not normalized_route:
            raise DroneProtocolError("route must not be empty")
        base_path = self._base_parts.path.rstrip("/")
        full_path = f"{base_path}/{normalized_route}" if base_path else f"/{normalized_route}"
        return urlunsplit((self._base_parts.scheme, self._base_parts.netloc, full_path, "", ""))

    def _ensure_open(self) -> None:
        """Raise if the client has already been closed."""

        if self._closed:
            raise DroneClientError("Twinr drone client is closed")

    def _request_json(
        self,
        method: str,
        route: str,
        *,
        payload: dict[str, object] | None = None,
        query: dict[str, object] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, object]:
        """Perform one bounded JSON request against the drone daemon."""

        self._ensure_open()
        normalized_method = method.strip().upper()
        if normalized_method not in {"GET", "POST"}:
            raise DroneProtocolError(f"unsupported HTTP method {normalized_method!r}")
        url = self._build_url(route)
        request_headers: dict[str, str] = {"X-Request-ID": _new_request_id()}
        if self._auth_token:
            request_headers["Authorization"] = f"Bearer {self._auth_token}"
        if extra_headers:
            request_headers.update({key: value for key, value in extra_headers.items() if value})
        body: bytes | None = None
        if payload is not None:
            try:
                body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            except (TypeError, ValueError) as exc:
                raise DroneProtocolError("Twinr drone request payload is not JSON-serializable") from exc
            request_headers["Content-Type"] = "application/json"
        params = _query_params(query)
        request = self._client.build_request(
            normalized_method,
            url,
            params=params,
            content=body,
            headers=request_headers,
        )
        response = None
        try:
            response = self._client.send(request, stream=True, follow_redirects=False)
            if 300 <= response.status_code < 400:
                location = (response.headers.get("location") or "").strip()
                if location:
                    raise DroneSecurityError(
                        f"Twinr drone daemon attempted redirect to {location!r}; redirects are refused"
                    )
                raise DroneSecurityError("Twinr drone daemon attempted redirect; redirects are refused")
            raw_bytes = self._read_bounded_body(response)
            status_code = int(response.status_code)
            if status_code >= 400:
                raise DroneHTTPError(_render_http_error(status_code, raw_bytes))
            content_type = response.headers.get("content-type", "")
        except DroneClientError:
            raise
        except self._httpx.TimeoutException as exc:
            raise DroneTransportError(f"Twinr drone request failed: {_remote_error_code(exc)}") from exc
        except self._httpx.TransportError as exc:
            raise DroneTransportError(f"Twinr drone request failed: {_remote_error_code(exc)}") from exc
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass
        payload_obj = _decode_json_object(raw_bytes, content_type=content_type)
        return _unwrap_response_envelope(payload_obj)

    def _read_bounded_body(self, response: Any) -> bytes:
        """Read one response body without exceeding the configured byte budget."""

        content_length = response.headers.get("content-length")
        if content_length:
            try:
                announced_length = int(content_length)
            except ValueError:
                announced_length = -1
            if announced_length > self._max_response_bytes:
                raise DroneProtocolError(
                    f"Twinr drone daemon response exceeds configured limit of {self._max_response_bytes} bytes"
                )
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_bytes():
            total += len(chunk)
            if total > self._max_response_bytes:
                raise DroneProtocolError(
                    f"Twinr drone daemon response exceeds configured limit of {self._max_response_bytes} bytes"
                )
            chunks.append(chunk)
        return b"".join(chunks)
