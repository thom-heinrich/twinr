"""Build validated Groq SDK clients for Twinr provider adapters.

This module owns the configuration-to-client boundary for Groq. It validates
API keys, timeout values, and base URL overrides before constructing the SDK
client that the higher-level adapters use.
"""

from __future__ import annotations

import math  # AUDIT-FIX(#2): Needed for finite timeout validation.
import os  # AUDIT-FIX(#1): Needed for backward-compatible host allowlist overrides.
from typing import Any
from urllib.parse import urlsplit, urlunsplit  # AUDIT-FIX(#1): Needed for safe base URL parsing and normalization.

from twinr.agent.base_agent.config import TwinrConfig

_DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"  # AUDIT-FIX(#1): Single safe default keeps endpoint selection deterministic.
_DEFAULT_GROQ_TIMEOUT_S = 30.0  # AUDIT-FIX(#2): Safe fallback prevents unbounded or SDK-defined timeout behaviour.
_DEFAULT_ALLOWED_GROQ_BASE_URL_HOSTS = frozenset(
    {"api.groq.com", "localhost", "127.0.0.1", "::1"}
)  # AUDIT-FIX(#1): Secure-by-default host allowlist blocks silent key exfiltration to arbitrary domains.
_LOCALHOST_GROQ_BASE_URL_HOSTS = frozenset(
    {"localhost", "127.0.0.1", "::1"}
)  # AUDIT-FIX(#1): Local proxies may use plain HTTP during development or audited on-device routing.
_ALLOWED_GROQ_BASE_URL_HOSTS_ENV = (
    "TWINR_GROQ_ALLOWED_BASE_URL_HOSTS"
)  # AUDIT-FIX(#1): Backward-compatible escape hatch for explicitly audited proxy hosts.


def default_groq_client(config: TwinrConfig) -> Any:
    """Build a Groq-compatible OpenAI SDK client from validated config values."""
    if config is None:
        raise TypeError("config must not be None")  # AUDIT-FIX(#4): Fail fast with a clear error instead of AttributeError later.

    api_key = _validated_groq_api_key(config)  # AUDIT-FIX(#4): Validate config-derived values before SDK construction.
    base_url = _default_groq_base_url(config)  # AUDIT-FIX(#1): Reject malformed or unsafe override URLs early.
    timeout_s = _validated_groq_timeout_s(config)  # AUDIT-FIX(#2): Enforce a finite positive request timeout.

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError(
            "The OpenAI SDK is not installed. Run `pip install -e .` in /twinr first."
        ) from exc

    try:
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_s,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to create the Groq client from the current Twinr configuration."
        ) from exc  # AUDIT-FIX(#3): Convert opaque SDK constructor failures into a deterministic configuration error.


def _validated_groq_api_key(config: TwinrConfig) -> str:
    """Return the configured Groq API key or raise a deterministic config error."""
    raw_api_key = getattr(config, "groq_api_key", None)  # AUDIT-FIX(#4): Tolerate incomplete config objects and validate explicitly.
    if raw_api_key is None:
        api_key = ""
    elif isinstance(raw_api_key, str):
        api_key = raw_api_key.strip()
    else:
        raise RuntimeError("groq_api_key must be a string")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required to use the Groq provider")
    return api_key


def _validated_groq_timeout_s(config: TwinrConfig) -> float:
    """Normalize the Groq request timeout to a finite positive float."""
    raw_timeout = getattr(config, "groq_timeout_s", None)  # AUDIT-FIX(#2): Read timeout defensively; older configs may omit it.
    if raw_timeout is None or raw_timeout == "":
        return _DEFAULT_GROQ_TIMEOUT_S

    if isinstance(raw_timeout, bool):
        raise RuntimeError("groq_timeout_s must be a finite positive number of seconds")  # AUDIT-FIX(#2): Reject bool because bool is a subclass of int.

    try:
        timeout_s = float(raw_timeout)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "groq_timeout_s must be a finite positive number of seconds"
        ) from exc  # AUDIT-FIX(#2): Make config errors explicit.

    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise RuntimeError(
            "groq_timeout_s must be a finite positive number of seconds"
        )  # AUDIT-FIX(#2): Prevent hangs, instant expiry, and NaN/inf bugs.

    return timeout_s


def _allowed_groq_base_url_hosts() -> set[str]:
    """Return the audited host allow-list for Groq base URL overrides."""
    raw_hosts = os.getenv(_ALLOWED_GROQ_BASE_URL_HOSTS_ENV, "")  # AUDIT-FIX(#1): Optional env extension preserves backward compatibility for audited proxies.
    extra_hosts = {
        host.strip().lower()
        for host in raw_hosts.split(",")
        if host.strip()
    }
    return set(_DEFAULT_ALLOWED_GROQ_BASE_URL_HOSTS | extra_hosts)


def _default_groq_base_url(config: TwinrConfig) -> str:
    """Validate and normalize the configured Groq base URL override."""
    raw_base_url = getattr(config, "groq_base_url", None)  # AUDIT-FIX(#1): Read base URL defensively instead of assuming a perfect config object.
    if raw_base_url is None:
        base_url = _DEFAULT_GROQ_BASE_URL
    elif isinstance(raw_base_url, str):
        base_url = raw_base_url.strip() or _DEFAULT_GROQ_BASE_URL
    else:
        raise RuntimeError("groq_base_url must be a string")

    if any(character.isspace() for character in base_url):
        raise RuntimeError(
            "groq_base_url must not contain whitespace"
        )  # AUDIT-FIX(#1): Reject copy-paste artifacts before URL parsing and secret transmission.

    parsed = urlsplit(base_url)
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    if scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError(
            "groq_base_url must be a valid http(s) URL"
        )  # AUDIT-FIX(#1): Reject malformed or non-network schemes before secrets are sent.

    if hostname not in _allowed_groq_base_url_hosts():
        raise RuntimeError(
            f"groq_base_url host '{hostname or '<empty>'}' is not allowed; "
            f"extend {_ALLOWED_GROQ_BASE_URL_HOSTS_ENV} only for audited proxy hosts"
        )  # AUDIT-FIX(#1): Block arbitrary remote endpoints from receiving the Groq API key.

    if scheme != "https" and hostname not in _LOCALHOST_GROQ_BASE_URL_HOSTS:
        raise RuntimeError(
            "groq_base_url must use https for non-local hosts"
        )  # AUDIT-FIX(#1): Prevent plaintext key exposure on remote networks.

    if parsed.username is not None or parsed.password is not None:
        raise RuntimeError(
            "groq_base_url must not contain embedded credentials"
        )  # AUDIT-FIX(#1): Prevent accidental secret leakage via URLs.

    if parsed.query or parsed.fragment:
        raise RuntimeError(
            "groq_base_url must not contain query parameters or fragments"
        )  # AUDIT-FIX(#1): Keep endpoint construction deterministic and auditable.

    normalized_path = parsed.path.rstrip("/")
    return urlunsplit(
        (scheme, parsed.netloc, normalized_path, "", "")
    )  # AUDIT-FIX(#1): Preserve valid host-only overrides while removing ambiguous trailing slashes.
