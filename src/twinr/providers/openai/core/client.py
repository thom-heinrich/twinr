"""Build validated OpenAI SDK clients for Twinr.

This module centralizes credential normalization, endpoint selection, and
network-policy defaults that every synchronous OpenAI caller in Twinr shares.

Optional TwinrConfig attributes supported by the 2026 transport hardening pass:
- openai_base_url
- openai_websocket_base_url
- openai_organization_id / openai_organization
- openai_webhook_secret
- openai_connect_timeout_s
- openai_read_timeout_s
- openai_write_timeout_s
- openai_pool_timeout_s
- openai_max_connections
- openai_max_keepalive_connections
- openai_keepalive_expiry_s
- openai_http2
- openai_proxy_url
- openai_trust_env
- openai_allow_insecure_http
- openai_ssl_ca_file
- openai_ssl_ca_dir
- openai_ssl_use_system_store
"""

# CHANGELOG: 2026-03-30
# BUG-1: Retry-count parsing is now strict; fractional values no longer silently truncate via int(...).
# BUG-2: Project and organization identifiers now reject embedded whitespace so broken auth headers fail fast.
# SEC-1: The client now pins an explicit base_url instead of silently inheriting OPENAI_BASE_URL from the process environment.
# SEC-2: The HTTP transport now defaults to trust_env=False, blocking proxy and CA-bundle env poisoning unless explicitly opted in.
# IMP-1: Added a 2026-style explicit HTTP transport with granular connect/read/write/pool timeouts, connection limits, keep-alive tuning, optional HTTP/2, and validated proxies.
# IMP-2: Added validated support for organization, webhook secret, websocket base URL, custom CA bundles, and optional system trust stores for private gateways.
# IMP-3: Reused client instances by normalized settings so Pi 4 deployments keep connection pooling warm instead of recreating TLS sessions on every factory call.

from __future__ import annotations

import math
import os
import ssl
from dataclasses import dataclass
from threading import Lock
from typing import Any
from urllib.parse import SplitResult, urlsplit, urlunsplit

from twinr.agent.base_agent.config import TwinrConfig

_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_OPENAI_TIMEOUT_S = 45.0
_DEFAULT_OPENAI_CONNECT_TIMEOUT_S = 5.0
_DEFAULT_OPENAI_POOL_TIMEOUT_S = 5.0
_DEFAULT_OPENAI_MAX_RETRIES = 1
_DEFAULT_OPENAI_MAX_CONNECTIONS = 8
_DEFAULT_OPENAI_MAX_KEEPALIVE_CONNECTIONS = 4
_DEFAULT_OPENAI_KEEPALIVE_EXPIRY_S = 30.0

_PROJECT_SCOPED_KEY_PREFIX = "sk-proj-"
_TRUTHY_STRINGS = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSY_STRINGS = frozenset({"0", "false", "f", "no", "n", "off"})
_PROXY_SCHEMES = frozenset({"http", "https", "socks5", "socks5h"})
_HTTP_BASE_URL_SCHEMES = frozenset({"http", "https"})
_WS_BASE_URL_SCHEMES = frozenset({"ws", "wss"})

_CLIENT_CACHE: dict["_ClientSettings", Any] = {}
_CLIENT_CACHE_LOCK = Lock()


@dataclass(frozen=True, slots=True)
class _ClientSettings:
    api_key: str
    base_url: str
    websocket_base_url: str | None
    organization: str | None
    project: str | None
    webhook_secret: str | None
    timeout_s: float
    connect_timeout_s: float
    read_timeout_s: float
    write_timeout_s: float
    pool_timeout_s: float
    max_retries: int
    max_connections: int
    max_keepalive_connections: int
    keepalive_expiry_s: float
    http2: bool
    proxy_url: str | None
    trust_env: bool
    ssl_ca_file: str | None
    ssl_ca_dir: str | None
    ssl_use_system_store: bool


def _default_client_factory(config: TwinrConfig) -> Any:
    """Build or reuse the default synchronous OpenAI client from ``TwinrConfig``."""

    settings = _build_client_settings(config)

    cached_client = _CLIENT_CACHE.get(settings)
    if cached_client is not None:
        return cached_client

    with _CLIENT_CACHE_LOCK:
        cached_client = _CLIENT_CACHE.get(settings)
        if cached_client is not None:
            return cached_client

        client = _create_openai_client(settings)
        _CLIENT_CACHE[settings] = client
        return client


def close_cached_openai_clients() -> None:
    """Close and discard cached OpenAI clients."""

    with _CLIENT_CACHE_LOCK:
        clients = list(_CLIENT_CACHE.values())
        _CLIENT_CACHE.clear()

    for client in clients:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def _create_openai_client(settings: _ClientSettings) -> Any:
    """Instantiate a synchronous OpenAI client from normalized settings."""

    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - exercised when dependency is missing at runtime
        raise RuntimeError(
            "The HTTPX dependency is not installed in this environment. Install the `httpx` package for the Twinr runtime."
        ) from exc

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - exercised when dependency is missing at runtime
        raise RuntimeError(
            "The OpenAI SDK is not installed in this environment. Install the `openai` package for the Twinr runtime."
        ) from exc

    try:
        from openai import DefaultHttpxClient  # type: ignore[attr-defined]
    except ImportError:  # pragma: no cover - older SDKs
        DefaultHttpxClient = None  # type: ignore[assignment]

    timeout = httpx.Timeout(
        settings.timeout_s,
        connect=settings.connect_timeout_s,
        read=settings.read_timeout_s,
        write=settings.write_timeout_s,
        pool=settings.pool_timeout_s,
    )
    limits = httpx.Limits(
        max_connections=settings.max_connections,
        max_keepalive_connections=settings.max_keepalive_connections,
        keepalive_expiry=settings.keepalive_expiry_s,
    )
    verify = _build_ssl_verify(settings)

    http_client_kwargs: dict[str, Any] = {
        "timeout": timeout,
        "limits": limits,
        "follow_redirects": True,
        "verify": verify,
        "trust_env": settings.trust_env,
        "http2": settings.http2,
    }
    if settings.proxy_url is not None:
        http_client_kwargs["proxy"] = settings.proxy_url

    if DefaultHttpxClient is not None:
        http_client = DefaultHttpxClient(**http_client_kwargs)
    else:  # pragma: no cover - compatibility fallback for outdated SDKs
        http_client = httpx.Client(**http_client_kwargs)

    kwargs: dict[str, Any] = {
        "api_key": settings.api_key,
        "base_url": settings.base_url,
        "timeout": timeout,
        "max_retries": settings.max_retries,
        "http_client": http_client,
    }
    if settings.websocket_base_url is not None:
        kwargs["websocket_base_url"] = settings.websocket_base_url
    if settings.organization is not None:
        kwargs["organization"] = settings.organization
    if settings.project is not None:
        kwargs["project"] = settings.project
    if settings.webhook_secret is not None:
        kwargs["webhook_secret"] = settings.webhook_secret

    try:
        client = OpenAI(**kwargs)
    except (TypeError, ValueError) as exc:
        try:
            http_client.close()
        except Exception:
            pass
        raise RuntimeError(
            "Failed to initialize the OpenAI client from TwinrConfig. Check the OpenAI-related configuration values."
        ) from exc

    # Force the returned client to match TwinrConfig exactly, even if the SDK would
    # otherwise infer optional values from process-level environment variables.
    client.organization = settings.organization
    client.project = settings.project
    client.webhook_secret = settings.webhook_secret
    client.websocket_base_url = settings.websocket_base_url

    return client


def _build_client_settings(config: TwinrConfig) -> _ClientSettings:
    """Normalize TwinrConfig into an immutable OpenAI transport configuration."""

    api_key = _normalize_identifier(
        "OPENAI_API_KEY",
        getattr(config, "openai_api_key", None),
        required=True,
        reject_whitespace=True,
    )
    project_id = _normalize_identifier(
        "OPENAI_PROJECT_ID",
        getattr(config, "openai_project_id", None),
        required=False,
        reject_whitespace=True,
    )
    organization = _normalize_identifier(
        "OPENAI_ORG_ID",
        _get_config_value(config, "openai_organization_id", "openai_organization"),
        required=False,
        reject_whitespace=True,
    )
    webhook_secret = _normalize_identifier(
        "OPENAI_WEBHOOK_SECRET",
        getattr(config, "openai_webhook_secret", None),
        required=False,
        reject_whitespace=True,
    )

    # BREAKING: The client no longer honors OPENAI_BASE_URL implicitly.
    # Configure `openai_base_url` in TwinrConfig when you need a non-default endpoint.
    allow_insecure_http = _coerce_optional_bool(
        "openai_allow_insecure_http",
        getattr(config, "openai_allow_insecure_http", None),
    )
    base_url = _normalize_url(
        "openai_base_url",
        getattr(config, "openai_base_url", None),
        default=_DEFAULT_OPENAI_BASE_URL,
        allowed_schemes=_HTTP_BASE_URL_SCHEMES,
        allow_insecure=bool(allow_insecure_http),
        allow_userinfo=False,
    )
    websocket_base_url = _normalize_url(
        "openai_websocket_base_url",
        getattr(config, "openai_websocket_base_url", None),
        default=None,
        allowed_schemes=_WS_BASE_URL_SCHEMES,
        allow_insecure=bool(allow_insecure_http),
        allow_userinfo=False,
    )

    send_project_header = _should_send_project_header_value(
        api_key=api_key,
        project_id=project_id,
        configured_value=getattr(config, "openai_send_project_header", None),
    )

    timeout_s = _coerce_positive_float(
        "openai_timeout_s",
        getattr(config, "openai_timeout_s", None),
        default=_DEFAULT_OPENAI_TIMEOUT_S,
        minimum=0.1,
    )
    connect_timeout_s = _coerce_positive_float(
        "openai_connect_timeout_s",
        getattr(config, "openai_connect_timeout_s", None),
        default=min(_DEFAULT_OPENAI_CONNECT_TIMEOUT_S, timeout_s),
        minimum=0.1,
    )
    read_timeout_s = _coerce_positive_float(
        "openai_read_timeout_s",
        getattr(config, "openai_read_timeout_s", None),
        default=timeout_s,
        minimum=0.1,
    )
    write_timeout_s = _coerce_positive_float(
        "openai_write_timeout_s",
        getattr(config, "openai_write_timeout_s", None),
        default=timeout_s,
        minimum=0.1,
    )
    pool_timeout_s = _coerce_positive_float(
        "openai_pool_timeout_s",
        getattr(config, "openai_pool_timeout_s", None),
        default=min(_DEFAULT_OPENAI_POOL_TIMEOUT_S, timeout_s),
        minimum=0.0,
    )
    max_retries = _coerce_bounded_int(
        "openai_max_retries",
        getattr(config, "openai_max_retries", None),
        default=_DEFAULT_OPENAI_MAX_RETRIES,
        minimum=0,
        maximum=5,
    )
    max_connections = _coerce_bounded_int(
        "openai_max_connections",
        getattr(config, "openai_max_connections", None),
        default=_DEFAULT_OPENAI_MAX_CONNECTIONS,
        minimum=1,
        maximum=64,
    )
    max_keepalive_connections = _coerce_bounded_int(
        "openai_max_keepalive_connections",
        getattr(config, "openai_max_keepalive_connections", None),
        default=min(_DEFAULT_OPENAI_MAX_KEEPALIVE_CONNECTIONS, max_connections),
        minimum=0,
        maximum=max_connections,
    )
    keepalive_expiry_s = _coerce_positive_float(
        "openai_keepalive_expiry_s",
        getattr(config, "openai_keepalive_expiry_s", None),
        default=_DEFAULT_OPENAI_KEEPALIVE_EXPIRY_S,
        minimum=0.0,
    )

    http2 = _coerce_optional_bool("openai_http2", getattr(config, "openai_http2", None))
    proxy_url = _normalize_url(
        "openai_proxy_url",
        getattr(config, "openai_proxy_url", None),
        default=None,
        allowed_schemes=_PROXY_SCHEMES,
        allow_insecure=True,
        allow_userinfo=True,
    )

    ssl_ca_file = _normalize_existing_file("openai_ssl_ca_file", getattr(config, "openai_ssl_ca_file", None))
    ssl_ca_dir = _normalize_existing_dir("openai_ssl_ca_dir", getattr(config, "openai_ssl_ca_dir", None))
    ssl_use_system_store = _coerce_optional_bool(
        "openai_ssl_use_system_store",
        getattr(config, "openai_ssl_use_system_store", None),
    )

    # BREAKING: The HTTP transport no longer trusts HTTP(S)_PROXY / ALL_PROXY /
    # SSL_CERT_FILE / SSL_CERT_DIR implicitly. Set `openai_trust_env=True` if you
    # intentionally rely on process-level environment variables instead of TwinrConfig.
    trust_env = _coerce_optional_bool("openai_trust_env", getattr(config, "openai_trust_env", None))

    return _ClientSettings(
        api_key=api_key,
        base_url=base_url,
        websocket_base_url=websocket_base_url,
        organization=organization,
        project=project_id if send_project_header else None,
        webhook_secret=webhook_secret,
        timeout_s=timeout_s,
        connect_timeout_s=connect_timeout_s,
        read_timeout_s=read_timeout_s,
        write_timeout_s=write_timeout_s,
        pool_timeout_s=pool_timeout_s,
        max_retries=max_retries,
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive_connections,
        keepalive_expiry_s=keepalive_expiry_s,
        http2=bool(http2) if http2 is not None else False,
        proxy_url=proxy_url,
        trust_env=bool(trust_env) if trust_env is not None else False,
        ssl_ca_file=ssl_ca_file,
        ssl_ca_dir=ssl_ca_dir,
        ssl_use_system_store=bool(ssl_use_system_store) if ssl_use_system_store is not None else False,
    )


def _build_ssl_verify(settings: _ClientSettings) -> ssl.SSLContext | bool:
    """Build the SSL verification policy for the OpenAI transport."""

    if settings.ssl_use_system_store:
        try:
            import truststore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "openai_ssl_use_system_store=True requires the optional `truststore` package to be installed."
            ) from exc
        context: ssl.SSLContext = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        if settings.ssl_ca_file is not None or settings.ssl_ca_dir is not None:
            context.load_verify_locations(cafile=settings.ssl_ca_file, capath=settings.ssl_ca_dir)
        return context

    if settings.ssl_ca_file is None and settings.ssl_ca_dir is None:
        return True

    return ssl.create_default_context(cafile=settings.ssl_ca_file, capath=settings.ssl_ca_dir)


def _should_send_project_header(config: TwinrConfig) -> bool:
    """Decide whether OpenAI requests should include the project header."""

    api_key = _normalize_identifier(
        "OPENAI_API_KEY",
        getattr(config, "openai_api_key", None),
        required=False,
        reject_whitespace=True,
    ) or ""
    project_id = _normalize_identifier(
        "OPENAI_PROJECT_ID",
        getattr(config, "openai_project_id", None),
        required=False,
        reject_whitespace=True,
    )
    return _should_send_project_header_value(
        api_key=api_key,
        project_id=project_id,
        configured_value=getattr(config, "openai_send_project_header", None),
    )


def _should_send_project_header_value(*, api_key: str, project_id: str | None, configured_value: Any) -> bool:
    """Resolve project-header behavior from normalized credential inputs."""

    if project_id is None:
        return False

    parsed_flag = _coerce_optional_bool("openai_send_project_header", configured_value)
    if parsed_flag is not None:
        return parsed_flag

    return not api_key.startswith(_PROJECT_SCOPED_KEY_PREFIX)


def _get_config_value(config: TwinrConfig, *names: str) -> Any:
    """Return the first non-None config attribute from a list of candidate names."""

    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return None


def _normalize_identifier(name: str, value: Any, *, required: bool, reject_whitespace: bool) -> str | None:
    """Validate and normalize a config-backed identifier value."""

    if value is None:
        if required:
            raise RuntimeError(f"{name} is required to use the OpenAI backend")
        return None

    if not isinstance(value, str):
        raise RuntimeError(f"{name} must be a string")

    normalized = value.strip()
    if not normalized:
        if required:
            raise RuntimeError(f"{name} is required to use the OpenAI backend")
        return None

    if reject_whitespace and any(character.isspace() for character in normalized):
        raise RuntimeError(f"{name} must not contain whitespace characters")
    if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
        raise RuntimeError(f"{name} contains invalid control characters")

    return normalized


def _normalize_url(
    name: str,
    value: Any,
    *,
    default: str | None,
    allowed_schemes: frozenset[str],
    allow_insecure: bool,
    allow_userinfo: bool,
) -> str | None:
    """Validate, normalize, and canonicalize a URL-like config value."""

    if value is None:
        return default

    if not isinstance(value, str):
        raise RuntimeError(f"{name} must be a URL string")

    normalized = value.strip()
    if not normalized:
        return default

    if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
        raise RuntimeError(f"{name} contains invalid control characters")

    parsed = urlsplit(normalized)
    scheme = parsed.scheme.lower()
    if scheme not in allowed_schemes:
        allowed = ", ".join(sorted(allowed_schemes))
        raise RuntimeError(f"{name} must use one of the following URL schemes: {allowed}")
    if not parsed.netloc:
        raise RuntimeError(f"{name} must include a hostname")
    if parsed.query or parsed.fragment:
        raise RuntimeError(f"{name} must not include query parameters or fragments")
    if not allow_userinfo and (parsed.username is not None or parsed.password is not None):
        raise RuntimeError(f"{name} must not embed credentials in the URL")
    if scheme in {"http", "ws"} and not allow_insecure:
        raise RuntimeError(
            f"{name} must use TLS (`https://` or `wss://`) unless openai_allow_insecure_http=True is set explicitly"
        )

    cleaned = SplitResult(
        scheme=scheme,
        netloc=parsed.netloc,
        path=(parsed.path.rstrip("/") if parsed.path else ""),
        query="",
        fragment="",
    )
    return urlunsplit(cleaned)


def _normalize_existing_file(name: str, value: Any) -> str | None:
    """Normalize an optional file path and ensure it exists."""

    normalized = _normalize_optional_path(name, value)
    if normalized is None:
        return None
    if not os.path.isfile(normalized):
        raise RuntimeError(f"{name} must point to an existing file")
    return normalized


def _normalize_existing_dir(name: str, value: Any) -> str | None:
    """Normalize an optional directory path and ensure it exists."""

    normalized = _normalize_optional_path(name, value)
    if normalized is None:
        return None
    if not os.path.isdir(normalized):
        raise RuntimeError(f"{name} must point to an existing directory")
    return normalized


def _normalize_optional_path(name: str, value: Any) -> str | None:
    """Normalize an optional filesystem path value."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise RuntimeError(f"{name} must be a string path")

    normalized = value.strip()
    if not normalized:
        return None
    if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
        raise RuntimeError(f"{name} contains invalid control characters")
    return normalized


def _coerce_optional_bool(name: str, value: Any) -> bool | None:
    """Parse an optional bool-like config value into ``True``/``False``/``None``."""

    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, int) and value in (0, 1):
        return bool(value)

    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return None
        if normalized in _TRUTHY_STRINGS:
            return True
        if normalized in _FALSY_STRINGS:
            return False

    allowed_values = ", ".join(sorted(_TRUTHY_STRINGS | _FALSY_STRINGS))
    raise RuntimeError(f"{name} must be a boolean or one of: {allowed_values}")


def _coerce_positive_float(name: str, value: Any, *, default: float, minimum: float) -> float:
    """Parse a finite float config value that respects a minimum bound."""

    if value is None:
        return default

    if isinstance(value, bool):
        raise RuntimeError(f"{name} must be a finite number greater than or equal to {minimum}")

    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be a finite number greater than or equal to {minimum}") from exc

    if not math.isfinite(parsed) or parsed < minimum:
        raise RuntimeError(f"{name} must be a finite number greater than or equal to {minimum}")

    return parsed


def _coerce_bounded_int(name: str, value: Any, *, default: int, minimum: int, maximum: int) -> int:
    """Parse an integer config value that must stay within fixed bounds."""

    if value is None:
        return default

    # BREAKING: values such as 1.5 are now rejected instead of being silently truncated to 1.
    if isinstance(value, bool):
        raise RuntimeError(f"{name} must be an integer between {minimum} and {maximum}")

    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise RuntimeError(f"{name} must be an integer between {minimum} and {maximum}")
        parsed = int(value)
    elif isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise RuntimeError(f"{name} must be an integer between {minimum} and {maximum}")
        signless = normalized[1:] if normalized[0] in {"+", "-"} else normalized
        if not signless.isdigit():
            raise RuntimeError(f"{name} must be an integer between {minimum} and {maximum}")
        parsed = int(normalized, 10)
    else:
        raise RuntimeError(f"{name} must be an integer between {minimum} and {maximum}")

    if parsed < minimum or parsed > maximum:
        raise RuntimeError(f"{name} must be an integer between {minimum} and {maximum}")

    return parsed