"""Shared helpers extracted from the legacy `twinr.web.app` module."""

from __future__ import annotations

import base64
import binascii
import html
import ipaddress
import logging
import secrets
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urlsplit

from fastapi import HTTPException, Request, status
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from starlette.concurrency import run_in_threadpool

from twinr.web.presenters import _resolve_named_file
from twinr.web.support.auth import (
    WebAuthState,
    build_web_auth_session_cookie,
    web_auth_session_cookie_name,
    web_auth_session_max_age_seconds,
)

logger = logging.getLogger("twinr.web.app")

_DEFAULT_ALLOWED_HOSTS = "localhost,127.0.0.1,[::1]"
_DEFAULT_MAX_FORM_BYTES = 64 * 1024
_MAX_FORM_BYTES_CAP = 1024 * 1024


def _env_bool(raw_value: Any, *, default: bool) -> bool:
    """Parse a boolean-like env value with a fallback."""

    if raw_value is None:
        return default
    value = str(raw_value).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _env_int(raw_value: Any, *, default: int, minimum: int, maximum: int) -> int:
    """Parse and clamp an integer-like env value."""

    try:
        value = int(str(raw_value).strip())
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


def _normalize_host_header(raw_host: str) -> str:
    """Normalize a host header or allowlist entry for comparison."""

    host = raw_host.strip()
    if not host:
        return ""
    if host.startswith("["):
        end = host.find("]")
        if end != -1:
            return host[1:end].strip().lower()
    if host.count(":") == 1:
        return host.split(":", 1)[0].strip().lower()
    return host.lower()


def _parse_allowed_hosts(raw_value: str | None) -> tuple[str, ...]:
    """Split and normalize the configured host allowlist."""

    configured = raw_value or _DEFAULT_ALLOWED_HOSTS
    hosts = []
    for chunk in configured.split(","):
        normalized = _normalize_host_header(chunk)
        if normalized:
            hosts.append(normalized)
    return tuple(dict.fromkeys(hosts))


def _is_allowed_host(host: str, allowed_hosts: tuple[str, ...]) -> bool:
    """Return whether a normalized host is allowed."""

    if not host:
        return False
    if "*" in allowed_hosts:
        return True
    return host in allowed_hosts


def _is_loopback_host(host: str) -> bool:
    """Return whether a client host resolves to loopback."""

    normalized = host.strip().lower()
    if normalized in {"", "localhost"}:
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _has_valid_basic_auth(request: Request, username: str, password: str) -> bool:
    """Validate HTTP Basic credentials against configured values."""

    header = request.headers.get("authorization", "")
    if not header.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(header[6:].strip(), validate=True).decode("utf-8")
    except (ValueError, binascii.Error, UnicodeDecodeError):
        return False
    provided_username, separator, provided_password = decoded.partition(":")
    if not separator:
        return False
    return secrets.compare_digest(provided_username, username) and secrets.compare_digest(provided_password, password)


def _request_target_path(request: Request) -> str:
    """Return the current path plus query string for safe local redirects."""

    path = request.url.path or "/"
    query = str(request.url.query or "").strip()
    if query:
        return f"{path}?{query}"
    return path


def _safe_next_path(raw_value: object | None) -> str:
    """Normalize one untrusted local redirect target to a safe in-app path."""

    if raw_value is None:
        return "/"
    candidate = str(raw_value).strip()
    if not candidate.startswith("/") or candidate.startswith("//"):
        return "/"
    if candidate.startswith("/auth/login"):
        return "/"
    return candidate


def _auth_login_location(request: Request) -> str:
    """Build the managed-login redirect path for one unauthenticated request."""

    next_path = _safe_next_path(_request_target_path(request))
    if next_path == "/":
        return "/auth/login"
    return f"/auth/login?next={quote_plus(next_path)}"


def _set_managed_auth_cookie(
    response: Response,
    *,
    state: WebAuthState,
    username: str,
    request: Request,
) -> None:
    """Attach one signed managed-auth cookie to the response."""

    response.set_cookie(
        key=web_auth_session_cookie_name(),
        value=build_web_auth_session_cookie(state, username=username),
        max_age=web_auth_session_max_age_seconds(),
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
        path="/",
    )


def _clear_managed_auth_cookie(response: Response) -> None:
    """Remove the managed-auth cookie from the response."""

    response.delete_cookie(web_auth_session_cookie_name(), path="/")


def _apply_auth_context(
    request: Request,
    *,
    mode: str,
    username: str | None = None,
    must_change_password: bool = False,
) -> None:
    """Expose lightweight auth UI state to templates via `request.state`."""

    request.state.web_auth_context = {
        "mode": mode,
        "logged_in": bool(username),
        "username": username or "",
        "must_change_password": bool(must_change_password),
        "password_path": "/auth/password",
        "logout_path": "/auth/logout",
    }


def _request_origin(request: Request) -> str:
    """Build the request origin from direct or proxy-forwarded request headers."""

    forwarded_proto = _forwarded_header_value(request.headers.get("x-forwarded-proto"))
    forwarded_host = _forwarded_header_value(request.headers.get("x-forwarded-host"))
    forwarded_port = _forwarded_header_value(request.headers.get("x-forwarded-port"))
    scheme = forwarded_proto if forwarded_proto in {"http", "https"} else request.url.scheme
    host = forwarded_host or request.headers.get("host", "").strip()
    if forwarded_port and host and ":" not in host and not _is_default_origin_port(scheme, forwarded_port):
        host = f"{host}:{forwarded_port}"
    return f"{scheme}://{host}".rstrip("/")


def _forwarded_header_value(raw_value: str | None) -> str:
    """Return the first comma-separated proxy-forwarded header token."""

    return str(raw_value or "").split(",", 1)[0].strip().lower()


def _is_default_origin_port(scheme: str, port: str) -> bool:
    """Return whether one port is the default for the given origin scheme."""

    return (scheme == "http" and port == "80") or (scheme == "https" and port == "443")


def _is_same_origin_url(candidate: str, expected_origin: str) -> bool:
    """Return whether a URL matches the expected origin."""

    parsed = urlsplit(candidate)
    if not parsed.scheme or not parsed.netloc:
        return False
    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/") == expected_origin.rstrip("/")


def _has_trusted_same_origin(request: Request) -> bool:
    """Check Origin, Referer, or Fetch metadata for same-origin requests."""

    expected_origin = _request_origin(request)
    origin = request.headers.get("origin", "").strip()
    if origin:
        return _is_same_origin_url(origin, expected_origin)
    referer = request.headers.get("referer", "").strip()
    if referer:
        return _is_same_origin_url(referer, expected_origin)
    sec_fetch_site = request.headers.get("sec-fetch-site", "").strip().lower()
    if sec_fetch_site:
        return sec_fetch_site in {"same-origin", "none"}
    client_host = request.client.host if request.client else ""
    return _is_loopback_host(client_host)


def _secure_response(response: Response) -> Response:
    """Apply no-store and basic hardening headers to a response."""

    response.headers.setdefault("Cache-Control", "no-store")
    response.headers.setdefault("Pragma", "no-cache")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    return response


def _error_response(request: Request, *, status_code: int, message: str) -> Response:
    """Render a plain-language error response in HTML or text."""

    accept = request.headers.get("accept", "")
    if "text/html" in accept or "*/*" in accept:
        body = (
            "<!doctype html>"
            "<html lang=\"en\">"
            "<head><meta charset=\"utf-8\"><title>Twinr Control</title></head>"
            f"<body><h1>Twinr Control</h1><p>{html.escape(message)}</p></body></html>"
        )
        return _secure_response(HTMLResponse(body, status_code=status_code))
    return _secure_response(PlainTextResponse(message, status_code=status_code))


def _public_error_message(exc: Exception, *, fallback: str) -> str:
    """Map internal exceptions to a safe operator-facing message."""

    if isinstance(exc, HTTPException):
        detail = exc.detail if isinstance(exc.detail, str) else ""
        clean_detail = " ".join(detail.split()).strip()
        if clean_detail:
            return clean_detail
        return fallback
    if isinstance(exc, ValueError):
        clean_value = " ".join(str(exc).split()).strip()
        if clean_value and len(clean_value) <= 160 and not any(token in clean_value for token in ("/", "\\", "Traceback", "ALSA", "errno")):
            return clean_value
    if isinstance(exc, PermissionError):
        return "Twinr could not access the needed local file or device."
    if isinstance(exc, FileNotFoundError):
        return "Twinr could not find the needed local file."
    return fallback


def _redirect_location(path: str, *, saved: bool = False, error_message: str | None = None, step: str | None = None) -> str:
    """Build one redirect location with optional saved/error/step query state."""

    query_parts: list[str] = []
    if saved:
        query_parts.append("saved=1")
    if error_message:
        query_parts.append(f"error={quote_plus(error_message)}")
    if step:
        query_parts.append(f"step={quote_plus(step)}")
    if not query_parts:
        return path
    return f"{path}?{'&'.join(query_parts)}"


def _redirect_with_error(path: str, message: str, *, step: str | None = None) -> RedirectResponse:
    """Redirect to a page with a flash-style error message."""

    return RedirectResponse(_redirect_location(path, error_message=message, step=step), status_code=status.HTTP_303_SEE_OTHER)


def _redirect_saved(path: str, *, step: str | None = None) -> RedirectResponse:
    """Redirect to a page with the saved flag set."""

    return RedirectResponse(_redirect_location(path, saved=True, step=step), status_code=status.HTTP_303_SEE_OTHER)


def _conversation_lab_href(session_id: str | None = None) -> str:
    """Return the `/ops/debug` conversation-lab location for one optional session."""

    base = "/ops/debug?tab=conversation_lab"
    normalized = str(session_id or "").strip()
    if not normalized:
        return base
    return f"{base}&lab_session={quote_plus(normalized)}"


def _require_non_empty(value: str, *, message: str) -> str:
    """Return a stripped non-empty string or raise `ValueError`."""

    normalized = value.strip()
    if not normalized:
        raise ValueError(message)
    return normalized


def _require_positive_int(value: str, *, message: str) -> int:
    """Return one positive integer parsed from form input."""

    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if parsed < 1:
        raise ValueError(message)
    return parsed


def _safe_project_subpath(project_root: Path, configured_path: str | Path, *, label: str) -> Path:
    """Resolve a configured project-relative path and keep it rooted."""

    resolved_root = project_root.resolve()
    candidate = (resolved_root / Path(configured_path)).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the Twinr project folder.") from exc
    return candidate


def _safe_file_in_dir(parent_dir: Path, filename: str, *, label: str) -> Path:
    """Validate one filename under a trusted parent directory."""

    leaf = Path(filename)
    if leaf.name != filename:
        raise ValueError(f"{label} must be a single file name.")
    candidate = parent_dir / leaf
    if candidate.is_symlink():
        raise ValueError(f"{label} cannot be a symlink.")
    if candidate.exists():
        resolved_candidate = candidate.resolve()
        try:
            resolved_candidate.relative_to(parent_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"{label} must stay inside its parent folder.") from exc
    return candidate


def _resolve_downloadable_file(root: Path, requested_name: str) -> Path:
    """Resolve and validate one downloadable artifact path."""

    try:
        candidate = _resolve_named_file(root, requested_name)
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="The requested file was not found.") from exc

    root_resolved = root.resolve()
    if candidate.is_symlink():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="The requested file was not found.")

    try:
        resolved_candidate = candidate.resolve(strict=True)
        resolved_candidate.relative_to(root_resolved)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="The requested file was not found.") from exc

    if not resolved_candidate.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="The requested file was not found.")
    return resolved_candidate


def _reminder_sort_key(entry: Any) -> tuple[int, str]:
    """Sort reminders with dated entries before undated ones."""

    due_at = getattr(entry, "due_at", None)
    if due_at is None:
        return (1, "")
    return (0, str(due_at))


async def _call_sync(func: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Run a blocking callable in the threadpool and await the result."""

    return await run_in_threadpool(partial(func, *args, **kwargs))


async def _parse_bounded_form(request: Request, *, max_form_bytes: int) -> dict[str, str]:
    """Read one bounded URL-encoded form submission."""

    content_type = request.headers.get("content-type", "")
    if content_type and not content_type.startswith("application/x-www-form-urlencoded"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="This page accepts only standard form posts.",
        )

    raw_length = request.headers.get("content-length", "").strip()
    if raw_length:
        try:
            declared_length = int(raw_length)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The form size header was invalid.") from exc
        if declared_length < 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The form size header was invalid.")
        if declared_length > max_form_bytes:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="That form submission is too large.")

    body = await request.body()
    if len(body) > max_form_bytes:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="That form submission is too large.")
    from twinr.web.support.store import parse_urlencoded_form

    return parse_urlencoded_form(body)


__all__ = [
    "logger",
    "_DEFAULT_ALLOWED_HOSTS",
    "_DEFAULT_MAX_FORM_BYTES",
    "_MAX_FORM_BYTES_CAP",
    "_env_bool",
    "_env_int",
    "_normalize_host_header",
    "_parse_allowed_hosts",
    "_is_allowed_host",
    "_is_loopback_host",
    "_has_valid_basic_auth",
    "_request_target_path",
    "_safe_next_path",
    "_auth_login_location",
    "_set_managed_auth_cookie",
    "_clear_managed_auth_cookie",
    "_apply_auth_context",
    "_request_origin",
    "_forwarded_header_value",
    "_is_default_origin_port",
    "_is_same_origin_url",
    "_has_trusted_same_origin",
    "_secure_response",
    "_error_response",
    "_public_error_message",
    "_redirect_location",
    "_redirect_with_error",
    "_redirect_saved",
    "_conversation_lab_href",
    "_require_non_empty",
    "_require_positive_int",
    "_safe_project_subpath",
    "_safe_file_in_dir",
    "_resolve_downloadable_file",
    "_reminder_sort_key",
    "_call_sync",
    "_parse_bounded_form",
]
