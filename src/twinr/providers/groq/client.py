"""Build validated Groq SDK clients for Twinr provider adapters.

This module owns the configuration-to-client boundary for Groq. It validates
API keys, timeout values, retry policy, and base URL overrides before
constructing the SDK client that the higher-level adapters use.

The module now prefers the official `groq` SDK because Groq recommends it for
current platform features, but it can fall back to the OpenAI-compatible client
surface when required for legacy adapters.
"""

# CHANGELOG: 2026-03-30
# BUG-1: Fixed a real endpoint-selection bug: host-only overrides such as
#   https://localhost:8080 were accepted even though the legacy OpenAI-compatible
#   client requires an /openai/v1 path prefix, which could route requests to the
#   wrong endpoint or produce hard-to-diagnose 404s.
# SEC-1: Disabled implicit HTTPX environment inheritance by default
#   (HTTP_PROXY/HTTPS_PROXY/ALL_PROXY/SSL_CERT_FILE/SSL_CERT_DIR) so a poisoned
#   service environment cannot silently redirect Twinr traffic or TLS trust and
#   exfiltrate the Groq API key.
# IMP-1: Prefer the official `groq` SDK, with validated OpenAI-compatible
#   fallback, so Twinr can track Groq's current typed surface area more closely
#   (including the modern Responses/tooling ecosystem) without abandoning legacy
#   adapters.
# IMP-2: Added validated retries, granular timeouts, deterministic redirect
#   policy, Pi-friendly connection-pool sizing, legacy/official base URL
#   normalization, and an async client factory.

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from twinr.agent.base_agent.config import TwinrConfig

_DEFAULT_GROQ_ORIGIN = "https://api.groq.com"
_DEFAULT_GROQ_REQUEST_TIMEOUT_S = 30.0
_DEFAULT_GROQ_CONNECT_TIMEOUT_S = 5.0
_DEFAULT_GROQ_WRITE_TIMEOUT_S = 10.0
_DEFAULT_GROQ_POOL_TIMEOUT_S = 5.0
_DEFAULT_GROQ_MAX_RETRIES = 2
_DEFAULT_GROQ_MAX_CONNECTIONS = 20
_DEFAULT_GROQ_MAX_KEEPALIVE_CONNECTIONS = 10
_DEFAULT_GROQ_KEEPALIVE_EXPIRY_S = 30.0
_DEFAULT_GROQ_HTTP2 = False
_DEFAULT_GROQ_TRUST_ENV = False
_DEFAULT_GROQ_FOLLOW_REDIRECTS = False
_DEFAULT_GROQ_SDK_BACKEND = "auto"
_VALID_GROQ_SDK_BACKENDS = frozenset({"auto", "groq", "openai_compat"})
_OPENAI_COMPAT_SUFFIX = "/openai/v1"

_DEFAULT_ALLOWED_GROQ_BASE_URL_HOSTS = frozenset(
    {"api.groq.com", "localhost", "127.0.0.1", "::1"}
)
_LOCALHOST_GROQ_BASE_URL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_ALLOWED_GROQ_BASE_URL_HOSTS_ENV = "TWINR_GROQ_ALLOWED_BASE_URL_HOSTS"
_GROQ_SDK_BACKEND_ENV = "TWINR_GROQ_SDK_BACKEND"
_GROQ_TRUST_ENV_ENV = "TWINR_GROQ_TRUST_ENV"


@dataclass(frozen=True)
class _ValidatedGroqEndpoint:
    """Validated base URL components for Groq or Groq-compatible providers."""

    scheme: str
    netloc: str
    hostname: str
    path: str

    def official_sdk_base_url(self) -> str:
        """Return a base URL suitable for the official `groq` SDK."""
        path = self.path
        if path.endswith(_OPENAI_COMPAT_SUFFIX):
            path = path[: -len(_OPENAI_COMPAT_SUFFIX)]
        return urlunsplit((self.scheme, self.netloc, path, "", ""))

    def openai_compat_base_url(self) -> str:
        """Return a base URL suitable for OpenAI-compatible clients."""
        path = self.path
        if not path:
            path = _OPENAI_COMPAT_SUFFIX
        elif not path.endswith(_OPENAI_COMPAT_SUFFIX):
            path = f"{path}{_OPENAI_COMPAT_SUFFIX}"
        return urlunsplit((self.scheme, self.netloc, path, "", ""))


@dataclass(frozen=True)
class _ValidatedGroqClientSettings:
    api_key: str
    endpoint: _ValidatedGroqEndpoint
    timeout: httpx.Timeout
    max_retries: int
    limits: httpx.Limits
    http2: bool
    trust_env: bool
    follow_redirects: bool
    backend: str


def default_groq_client(config: TwinrConfig) -> Any:
    """Build a validated synchronous Groq SDK client."""
    settings = _validated_groq_client_settings(config)
    return _build_sync_groq_client(settings)


def default_async_groq_client(config: TwinrConfig) -> Any:
    """Build a validated asynchronous Groq SDK client."""
    settings = _validated_groq_client_settings(config)
    return _build_async_groq_client(settings)


def _build_sync_groq_client(settings: _ValidatedGroqClientSettings) -> Any:
    if settings.backend in {"auto", "groq"}:
        try:
            from groq import Groq

            # BREAKING: The module now prefers the official `groq` SDK when it is
            # available. Force the legacy surface with
            # `groq_sdk_backend="openai_compat"` or
            # `TWINR_GROQ_SDK_BACKEND=openai_compat`.
            return Groq(
                api_key=settings.api_key,
                base_url=settings.endpoint.official_sdk_base_url(),
                timeout=settings.timeout,
                max_retries=settings.max_retries,
                http_client=_new_sync_http_client(settings),
            )
        except ImportError:
            if settings.backend == "groq":
                raise RuntimeError(
                    "The official Groq SDK is not installed. Install it with `pip install groq`."
                ) from None

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "No compatible Groq client SDK is installed. Install `groq` "
            "(preferred) or `openai` for the legacy OpenAI-compatible path."
        ) from exc

    return OpenAI(
        api_key=settings.api_key,
        base_url=settings.endpoint.openai_compat_base_url(),
        timeout=settings.timeout,
        max_retries=settings.max_retries,
        http_client=_new_sync_http_client(settings),
    )


def _build_async_groq_client(settings: _ValidatedGroqClientSettings) -> Any:
    if settings.backend in {"auto", "groq"}:
        try:
            from groq import AsyncGroq

            return AsyncGroq(
                api_key=settings.api_key,
                base_url=settings.endpoint.official_sdk_base_url(),
                timeout=settings.timeout,
                max_retries=settings.max_retries,
                http_client=_new_async_http_client(settings),
            )
        except ImportError:
            if settings.backend == "groq":
                raise RuntimeError(
                    "The official Groq SDK is not installed. Install it with `pip install groq`."
                ) from None

    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "No compatible Groq client SDK is installed. Install `groq` "
            "(preferred) or `openai` for the legacy OpenAI-compatible path."
        ) from exc

    return AsyncOpenAI(
        api_key=settings.api_key,
        base_url=settings.endpoint.openai_compat_base_url(),
        timeout=settings.timeout,
        max_retries=settings.max_retries,
        http_client=_new_async_http_client(settings),
    )


def _new_sync_http_client(settings: _ValidatedGroqClientSettings) -> httpx.Client:
    # BREAKING: HTTPX environment inheritance is now disabled by default, so
    # HTTP_PROXY / HTTPS_PROXY / ALL_PROXY / SSL_CERT_FILE / SSL_CERT_DIR no
    # longer affect Groq traffic unless you opt in with `groq_trust_env=True`
    # or `TWINR_GROQ_TRUST_ENV=1`.
    return httpx.Client(
        timeout=settings.timeout,
        limits=settings.limits,
        http2=settings.http2,
        trust_env=settings.trust_env,
        follow_redirects=settings.follow_redirects,
    )


def _new_async_http_client(
    settings: _ValidatedGroqClientSettings,
) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=settings.timeout,
        limits=settings.limits,
        http2=settings.http2,
        trust_env=settings.trust_env,
        follow_redirects=settings.follow_redirects,
    )


def _validated_groq_client_settings(
    config: TwinrConfig,
) -> _ValidatedGroqClientSettings:
    if config is None:
        raise TypeError("config must not be None")

    request_timeout_s = _validated_positive_finite_float(
        "groq_timeout_s",
        _get_config_value(config, "groq_timeout_s"),
        default=_DEFAULT_GROQ_REQUEST_TIMEOUT_S,
    )
    connect_timeout_s = _validated_positive_finite_float(
        "groq_connect_timeout_s",
        _get_config_value(config, "groq_connect_timeout_s"),
        default=min(_DEFAULT_GROQ_CONNECT_TIMEOUT_S, request_timeout_s),
    )
    write_timeout_s = _validated_positive_finite_float(
        "groq_write_timeout_s",
        _get_config_value(config, "groq_write_timeout_s"),
        default=min(_DEFAULT_GROQ_WRITE_TIMEOUT_S, request_timeout_s),
    )
    pool_timeout_s = _validated_positive_finite_float(
        "groq_pool_timeout_s",
        _get_config_value(config, "groq_pool_timeout_s"),
        default=min(_DEFAULT_GROQ_POOL_TIMEOUT_S, request_timeout_s),
    )

    timeout = httpx.Timeout(
        timeout=request_timeout_s,
        connect=min(connect_timeout_s, request_timeout_s),
        read=request_timeout_s,
        write=min(write_timeout_s, request_timeout_s),
        pool=min(pool_timeout_s, request_timeout_s),
    )

    max_connections = _validated_positive_int(
        "groq_max_connections",
        _get_config_value(config, "groq_max_connections"),
        default=_DEFAULT_GROQ_MAX_CONNECTIONS,
    )
    max_keepalive_connections = _validated_non_negative_int(
        "groq_max_keepalive_connections",
        _get_config_value(config, "groq_max_keepalive_connections"),
        default=min(_DEFAULT_GROQ_MAX_KEEPALIVE_CONNECTIONS, max_connections),
    )
    if max_keepalive_connections > max_connections:
        raise RuntimeError(
            "groq_max_keepalive_connections must be <= groq_max_connections"
        )

    limits = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive_connections,
        keepalive_expiry=_validated_positive_finite_float(
            "groq_keepalive_expiry_s",
            _get_config_value(config, "groq_keepalive_expiry_s"),
            default=_DEFAULT_GROQ_KEEPALIVE_EXPIRY_S,
        ),
    )

    return _ValidatedGroqClientSettings(
        api_key=_validated_groq_api_key(config),
        endpoint=_validated_groq_endpoint(config),
        timeout=timeout,
        max_retries=_validated_non_negative_int(
            "groq_max_retries",
            _get_config_value(config, "groq_max_retries"),
            default=_DEFAULT_GROQ_MAX_RETRIES,
        ),
        limits=limits,
        http2=_validated_boolean(
            "groq_http2",
            _get_config_value(config, "groq_http2"),
            default=_DEFAULT_GROQ_HTTP2,
        ),
        trust_env=_validated_boolean(
            "groq_trust_env",
            _get_config_value(config, "groq_trust_env", env_name=_GROQ_TRUST_ENV_ENV),
            default=_DEFAULT_GROQ_TRUST_ENV,
        ),
        follow_redirects=_validated_boolean(
            "groq_follow_redirects",
            _get_config_value(config, "groq_follow_redirects"),
            default=_DEFAULT_GROQ_FOLLOW_REDIRECTS,
        ),
        backend=_validated_groq_sdk_backend(config),
    )


def _validated_groq_api_key(config: TwinrConfig) -> str:
    raw_api_key = _get_config_value(config, "groq_api_key")
    if raw_api_key is None:
        api_key = ""
    elif isinstance(raw_api_key, str):
        api_key = raw_api_key.strip()
    else:
        raise RuntimeError("groq_api_key must be a string")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required to use the Groq provider")
    return api_key


def _validated_groq_sdk_backend(config: TwinrConfig) -> str:
    raw_backend = _get_config_value(
        config,
        "groq_sdk_backend",
        env_name=_GROQ_SDK_BACKEND_ENV,
    )
    if raw_backend is None or raw_backend == "":
        return _DEFAULT_GROQ_SDK_BACKEND
    if not isinstance(raw_backend, str):
        raise RuntimeError(
            f"groq_sdk_backend must be one of {sorted(_VALID_GROQ_SDK_BACKENDS)}"
        )

    backend = raw_backend.strip().lower()
    if backend not in _VALID_GROQ_SDK_BACKENDS:
        raise RuntimeError(
            f"groq_sdk_backend must be one of {sorted(_VALID_GROQ_SDK_BACKENDS)}"
        )
    return backend


def _allowed_groq_base_url_hosts() -> set[str]:
    raw_hosts = os.getenv(_ALLOWED_GROQ_BASE_URL_HOSTS_ENV, "")
    extra_hosts = {
        host.strip().rstrip(".").lower()
        for host in raw_hosts.split(",")
        if host.strip()
    }
    return set(_DEFAULT_ALLOWED_GROQ_BASE_URL_HOSTS | extra_hosts)


def _validated_groq_endpoint(config: TwinrConfig) -> _ValidatedGroqEndpoint:
    raw_base_url = _get_config_value(config, "groq_base_url")
    if raw_base_url is None:
        base_url = _DEFAULT_GROQ_ORIGIN
    elif isinstance(raw_base_url, str):
        base_url = raw_base_url.strip() or _DEFAULT_GROQ_ORIGIN
    else:
        raise RuntimeError("groq_base_url must be a string")

    if any(character.isspace() for character in base_url):
        raise RuntimeError("groq_base_url must not contain whitespace")

    parsed = urlsplit(base_url)
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").rstrip(".").lower()
    if scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("groq_base_url must be a valid http(s) URL")

    if hostname not in _allowed_groq_base_url_hosts():
        raise RuntimeError(
            f"groq_base_url host '{hostname or '<empty>'}' is not allowed; "
            f"extend {_ALLOWED_GROQ_BASE_URL_HOSTS_ENV} only for audited proxy hosts"
        )

    if scheme != "https" and hostname not in _LOCALHOST_GROQ_BASE_URL_HOSTS:
        raise RuntimeError("groq_base_url must use https for non-local hosts")

    if parsed.username is not None or parsed.password is not None:
        raise RuntimeError("groq_base_url must not contain embedded credentials")

    if parsed.query or parsed.fragment:
        raise RuntimeError(
            "groq_base_url must not contain query parameters or fragments"
        )

    path = parsed.path.rstrip("/")
    if path in {".", ".."} or "/./" in path or "/../" in path:
        raise RuntimeError("groq_base_url must not contain dot-segments in its path")

    return _ValidatedGroqEndpoint(
        scheme=scheme,
        netloc=parsed.netloc,
        hostname=hostname,
        path=path,
    )


def _get_config_value(
    config: TwinrConfig,
    attribute_name: str,
    *,
    env_name: str | None = None,
) -> Any:
    value = getattr(config, attribute_name, None)
    if value is not None and value != "":
        return value
    if env_name:
        env_value = os.getenv(env_name)
        if env_value not in {None, ""}:
            return env_value
    return value


def _validated_boolean(name: str, raw_value: Any, *, default: bool) -> bool:
    if raw_value is None or raw_value == "":
        return default
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise RuntimeError(f"{name} must be a boolean")


def _validated_positive_finite_float(
    name: str,
    raw_value: Any,
    *,
    default: float,
) -> float:
    if raw_value is None or raw_value == "":
        return default
    if isinstance(raw_value, bool):
        raise RuntimeError(f"{name} must be a finite positive number of seconds")
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{name} must be a finite positive number of seconds"
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"{name} must be a finite positive number of seconds")
    return value


def _validated_non_negative_int(name: str, raw_value: Any, *, default: int) -> int:
    if raw_value is None or raw_value == "":
        return default
    if isinstance(raw_value, bool):
        raise RuntimeError(f"{name} must be a non-negative integer")

    if isinstance(raw_value, int):
        value = raw_value
    elif isinstance(raw_value, str):
        raw_str = raw_value.strip()
        if not raw_str:
            return default
        try:
            value = int(raw_str)
        except ValueError as exc:
            raise RuntimeError(f"{name} must be a non-negative integer") from exc
    else:
        raise RuntimeError(f"{name} must be a non-negative integer")

    if value < 0:
        raise RuntimeError(f"{name} must be a non-negative integer")
    return value


def _validated_positive_int(name: str, raw_value: Any, *, default: int) -> int:
    value = _validated_non_negative_int(name, raw_value, default=default)
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return value