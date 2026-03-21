"""Serve Twinr's local FastAPI control surface.

This module assembles the web app, applies control-plane security guards, and
wires route handlers to presenter, support, ops, and automation helpers.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import html
import ipaddress
import logging
import os
import secrets
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urlsplit

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from twinr.agent.base_agent import AdaptiveTimingStore, TwinrConfig
from twinr.agent.workflows.required_remote_snapshot import assess_required_remote_watchdog_snapshot
from twinr.agent.self_coding import SelfCodingActivationService
from twinr.agent.self_coding.retest import run_self_coding_skill_retest
from twinr.agent.self_coding.runtime import SelfCodingSkillRuntimeStore
from twinr.agent.self_coding.operator_status import build_self_coding_operator_status
from twinr.agent.self_coding.watchdog import cleanup_stale_compile_status, cleanup_stale_execution_run
from twinr.hardware.voice_profile import VoiceProfileMonitor
from twinr.integrations import build_managed_integrations, integration_automation_family_providers
from twinr.memory.reminders import format_due_label
from twinr.memory.longterm.retrieval.operator_search import run_long_term_operator_search
from twinr.ops import (
    TwinrSelfTestRunner,
    build_support_bundle,
    check_summary,
    collect_device_overview,
    collect_system_health,
    redact_env_values,
    resolve_ops_paths,
    run_config_checks,
)
from twinr.web.automations import (
    build_automation_page_context,
    delete_automation,
    save_sensor_automation,
    save_time_automation,
    toggle_automation_enabled,
)
from twinr.web.conversation_lab import (
    create_conversation_lab_session,
    load_conversation_lab_state,
    run_conversation_lab_turn,
)
from twinr.web.context import WebAppContext
from twinr.web.support.channel_onboarding import InProcessChannelPairingRegistry
from twinr.web.support.contracts import DashboardCard
from twinr.web.support.forms import _collect_standard_updates
from twinr.web.support.auth import (
    WebAuthState,
    build_web_auth_session_cookie,
    default_web_auth_username,
    load_authenticated_web_session,
    verify_web_auth_password,
    web_auth_password_min_length,
    web_auth_session_cookie_name,
    web_auth_session_max_age_seconds,
)
from twinr.web.support.store import FileBackedSetting, parse_urlencoded_form, read_text_file, write_env_updates, write_text_file
from twinr.web.support.whatsapp import (
    WhatsAppPairingCoordinator,
    canonicalize_whatsapp_allow_from,
    normalize_project_relative_directory,
    probe_whatsapp_runtime,
)
from twinr.web.presenters import (
    _adaptive_timing_view,
    _build_calendar_integration_record,
    _build_email_integration_record,
    _build_smart_home_integration_record,
    _calendar_integration_sections,
    build_ops_debug_page_context,
    _capture_voice_profile_sample,
    _connect_sections,
    coerce_ops_debug_tab,
    _default_reminder_due_at,
    _email_integration_sections,
    _format_log_rows,
    _format_usage_rows,
    _health_card_detail,
    _integration_overview_rows,
    _memory_sections,
    _provider_status,
    _recent_named_files,
    _reminder_rows,
    _resolve_named_file,
    _smart_home_integration_sections,
    _settings_sections,
    _whatsapp_integration_context,
    build_self_coding_ops_page_context,
    build_whatsapp_wizard_page_context,
    _voice_action_result,
    _voice_profile_page_context,
    _voice_snapshot_label,
)


logger = logging.getLogger(__name__)

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


# AUDIT-FIX(#5): Push blocking disk, zip, and hardware helpers off the event loop to keep the single-process UI responsive.
async def _call_sync(func: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Run a blocking callable in the threadpool and await the result."""

    return await run_in_threadpool(partial(func, *args, **kwargs))


# AUDIT-FIX(#3): Accept only bounded standard form posts so a bad client cannot memory-DoS the Raspberry Pi.
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
    return parse_urlencoded_form(body)


def create_app(env_file: str | Path = ".env") -> FastAPI:
    """Create the FastAPI app for Twinr's local control surface.

    Args:
        env_file: Path to the Twinr `.env` file that defines this install.

    Returns:
        Configured FastAPI application instance.
    """

    env_path = Path(env_file).resolve()
    if env_path.is_dir():
        raise RuntimeError(f"Twinr Control expected an env file path, got directory: {env_path}")
    project_root = env_path.parent.resolve()
    ops_paths = resolve_ops_paths(project_root)

    templates_dir = Path(__file__).resolve().parent / "templates"
    static_dir = Path(__file__).resolve().parent / "static"
    if not templates_dir.is_dir():
        raise RuntimeError(f"Twinr Control templates directory is missing: {templates_dir}")
    if not static_dir.is_dir():
        raise RuntimeError(f"Twinr Control static directory is missing: {static_dir}")

    templates = Jinja2Templates(directory=str(templates_dir))

    app = FastAPI(title="Twinr Control", version="0.1.0")
    app.mount(
        "/static",
        StaticFiles(directory=str(static_dir)),
        name="static",
    )

    ctx = WebAppContext(
        env_path=env_path,
        project_root=project_root,
        ops_paths=ops_paths,
        templates=templates,
    )

    try:
        _startup_config, startup_env_values = ctx.load_state()
    except Exception:
        startup_env_values = {}

    security_env_values = {
        **startup_env_values,
        **{key: value for key, value in os.environ.items() if key.startswith("TWINR_WEB_")},
    }

    # AUDIT-FIX(#1,#3): Lock down the control plane defaults and bound form-body size to protect the local device.
    max_form_bytes = _env_int(
        security_env_values.get("TWINR_WEB_MAX_FORM_BYTES"),
        default=_DEFAULT_MAX_FORM_BYTES,
        minimum=4 * 1024,
        maximum=_MAX_FORM_BYTES_CAP,
    )
    allowed_hosts = _parse_allowed_hosts(security_env_values.get("TWINR_WEB_ALLOWED_HOSTS"))
    allow_remote = _env_bool(security_env_values.get("TWINR_WEB_ALLOW_REMOTE"), default=False)
    require_auth = _env_bool(
        security_env_values.get("TWINR_WEB_REQUIRE_AUTH"),
        default=bool(security_env_values.get("TWINR_WEB_USERNAME") or security_env_values.get("TWINR_WEB_PASSWORD")),
    )
    auth_username = str(security_env_values.get("TWINR_WEB_USERNAME", "")).strip()
    auth_password = str(security_env_values.get("TWINR_WEB_PASSWORD", ""))
    static_auth_partially_configured = bool(auth_username or auth_password)
    if static_auth_partially_configured and not (auth_username and auth_password):
        raise RuntimeError("Twinr Control static web auth requires both TWINR_WEB_USERNAME and TWINR_WEB_PASSWORD.")
    managed_auth_enabled = bool(require_auth and not static_auth_partially_configured)
    managed_auth_store = ctx.web_auth_store() if managed_auth_enabled else None
    if allow_remote and not require_auth:
        raise RuntimeError("Twinr Control remote access requires web sign-in. Enable TWINR_WEB_REQUIRE_AUTH first.")

    # AUDIT-FIX(#7): Serialize file-backed control-plane writes inside the single-process app to avoid state corruption.
    state_write_lock = asyncio.Lock()
    ops_job_lock = asyncio.Lock()
    conversation_lab_lock = asyncio.Lock()
    voice_profile_lock = asyncio.Lock()
    managed_auth_write_lock = asyncio.Lock()
    channel_pairing_registry = InProcessChannelPairingRegistry()
    whatsapp_pairing = WhatsAppPairingCoordinator(
        store=ctx.channel_onboarding_store("whatsapp"),
        registry=channel_pairing_registry,
    )

    @app.middleware("http")
    async def guard_control_plane(request: Request, call_next: Any) -> Response:
        """Enforce host, remote-access, auth, and same-origin policies."""

        request_path = request.url.path.rstrip("/") or "/"
        public_auth_path = request_path.startswith("/static/") or request_path == "/auth/login"
        normalized_host = _normalize_host_header(request.headers.get("host", ""))
        if not _is_allowed_host(normalized_host, allowed_hosts):
            return _error_response(
                request,
                status_code=status.HTTP_400_BAD_REQUEST,
                message="Twinr Control rejected this host name. Check TWINR_WEB_ALLOWED_HOSTS.",
            )

        client_host = request.client.host if request.client else ""
        client_is_loopback = _is_loopback_host(client_host)
        managed_auth_state: WebAuthState | None = None
        managed_session_username: str | None = None
        if managed_auth_enabled:
            try:
                managed_auth_state = await _call_sync(managed_auth_store.load_or_bootstrap)
            except Exception as exc:
                logger.exception("Failed to load managed web auth state", exc_info=exc)
                return _error_response(
                    request,
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    message="Twinr Control could not load the local web sign-in state.",
                )
            managed_session_username = load_authenticated_web_session(
                managed_auth_state,
                request.cookies.get(web_auth_session_cookie_name()),
            )
            _apply_auth_context(
                request,
                mode="managed",
                username=managed_session_username,
                must_change_password=managed_auth_state.must_change_password,
            )

        if not allow_remote and not client_is_loopback:
            return _error_response(
                request,
                status_code=status.HTTP_403_FORBIDDEN,
                message=(
                    "Twinr Control only accepts browser access from this device right now. "
                    "To allow remote access, set TWINR_WEB_ALLOW_REMOTE=1, TWINR_WEB_ALLOWED_HOSTS, and web sign-in."
                ),
            )
        # AUDIT-FIX(#1,#2): Require auth when configured and block cross-site state-changing requests.
        if require_auth and not managed_auth_enabled and not _has_valid_basic_auth(request, auth_username, auth_password):
            response = _error_response(
                request,
                status_code=status.HTTP_401_UNAUTHORIZED,
                message="Sign-in is required for Twinr Control.",
            )
            response.headers["WWW-Authenticate"] = 'Basic realm="Twinr Control", charset="UTF-8"'
            return response
        if managed_auth_enabled and not public_auth_path:
            if not managed_session_username:
                return _secure_response(RedirectResponse(_auth_login_location(request), status_code=status.HTTP_303_SEE_OTHER))
            if managed_auth_state is not None and managed_auth_state.must_change_password and request_path not in {"/auth/password", "/auth/logout"}:
                return _secure_response(RedirectResponse("/auth/password", status_code=status.HTTP_303_SEE_OTHER))

        if request.method in {"POST", "PUT", "PATCH", "DELETE"} and not _has_trusted_same_origin(request):
            return _error_response(
                request,
                status_code=status.HTTP_403_FORBIDDEN,
                message="Twinr blocked that form because it did not look like a trusted same-browser request.",
            )

        response = await call_next(request)
        return _secure_response(response)

    @app.exception_handler(HTTPException)
    async def handle_http_exception(request: Request, exc: HTTPException) -> Response:
        """Convert `HTTPException` instances into safe operator responses."""

        return _error_response(
            request,
            status_code=exc.status_code,
            message=_public_error_message(exc, fallback="Twinr Control could not finish that request."),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_exception(request: Request, exc: Exception) -> Response:
        """Log unexpected failures and return a plain-language 500 page."""

        # AUDIT-FIX(#6): Keep internal errors in logs and show plain, non-technical fallback text to the operator.
        logger.exception("Unhandled Twinr Control error on %s %s", request.method, request.url.path, exc_info=exc)
        return _error_response(
            request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="Twinr Control hit a local problem and could not finish that page. Please try again.",
        )

    @app.get("/auth/login", response_class=HTMLResponse)
    async def auth_login(request: Request) -> HTMLResponse:
        """Render the managed-login page used by the permanent Pi web service."""

        if not managed_auth_enabled or managed_auth_store is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="This page is not available.")

        auth_state = await _call_sync(managed_auth_store.load_or_bootstrap)
        authenticated_username = load_authenticated_web_session(
            auth_state,
            request.cookies.get(web_auth_session_cookie_name()),
        )
        if authenticated_username:
            if auth_state.must_change_password:
                return RedirectResponse("/auth/password", status_code=status.HTTP_303_SEE_OTHER)
            return RedirectResponse(
                _safe_next_path(request.query_params.get("next")),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        _apply_auth_context(request, mode="managed")
        return ctx.render(
            request,
            "auth_login.html",
            page_title="Sign in",
            active_page="auth_login",
            restart_notice=None,
            intro=(
                "Sign in to Twinr Control. On the first sign-in use admin / admin. "
                "Twinr will ask for a new password right away."
            ),
            shell_mode="auth",
            next_path=_safe_next_path(request.query_params.get("next")),
            suggested_username=default_web_auth_username(),
        )

    @app.post("/auth/login")
    async def auth_login_post(request: Request) -> Response:
        """Authenticate one managed web login and start a signed session."""

        if not managed_auth_enabled or managed_auth_store is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="This page is not available.")

        auth_state = await _call_sync(managed_auth_store.load_or_bootstrap)
        form_values = await _parse_bounded_form(request, max_form_bytes=max_form_bytes)
        username = str(form_values.get("username", "") or "").strip()
        password = str(form_values.get("password", "") or "")
        next_path = _safe_next_path(form_values.get("next"))
        if not verify_web_auth_password(auth_state, username=username, password=password):
            _apply_auth_context(request, mode="managed")
            return ctx.render(
                request,
                "auth_login.html",
                page_title="Sign in",
                active_page="auth_login",
                restart_notice=None,
                intro=(
                    "Sign in to Twinr Control. On the first sign-in use admin / admin. "
                    "Twinr will ask for a new password right away."
                ),
                shell_mode="auth",
                next_path=next_path,
                suggested_username=username or default_web_auth_username(),
                error_message="Username or password was not correct.",
            )

        response = RedirectResponse(
            "/auth/password" if auth_state.must_change_password else next_path,
            status_code=status.HTTP_303_SEE_OTHER,
        )
        _set_managed_auth_cookie(response, state=auth_state, username=auth_state.username, request=request)
        return response

    @app.get("/auth/password", response_class=HTMLResponse)
    async def auth_password(request: Request) -> HTMLResponse:
        """Render the password-change page for managed web sign-in."""

        if not managed_auth_enabled or managed_auth_store is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="This page is not available.")

        auth_state = await _call_sync(managed_auth_store.load_or_bootstrap)
        authenticated_username = load_authenticated_web_session(
            auth_state,
            request.cookies.get(web_auth_session_cookie_name()),
        )
        if not authenticated_username:
            return RedirectResponse(_auth_login_location(request), status_code=status.HTTP_303_SEE_OTHER)
        _apply_auth_context(
            request,
            mode="managed",
            username=authenticated_username,
            must_change_password=auth_state.must_change_password,
        )
        return ctx.render(
            request,
            "auth_password.html",
            page_title="Change password",
            active_page="auth_password",
            restart_notice=(
                "Finish this step once. After that, Twinr Control stays unlocked for normal LAN sign-in."
                if auth_state.must_change_password
                else "Change the local Twinr Control password used for sign-in."
            ),
            intro=(
                "Choose a new password before the full Twinr Control portal is unlocked."
                if auth_state.must_change_password
                else "Update the password used to sign in to Twinr Control."
            ),
            shell_mode="auth",
            password_min_length=web_auth_password_min_length(),
        )

    @app.post("/auth/password")
    async def auth_password_post(request: Request) -> Response:
        """Persist one managed web password change."""

        if not managed_auth_enabled or managed_auth_store is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="This page is not available.")

        auth_state = await _call_sync(managed_auth_store.load_or_bootstrap)
        authenticated_username = load_authenticated_web_session(
            auth_state,
            request.cookies.get(web_auth_session_cookie_name()),
        )
        if not authenticated_username:
            return RedirectResponse(_auth_login_location(request), status_code=status.HTTP_303_SEE_OTHER)

        form_values = await _parse_bounded_form(request, max_form_bytes=max_form_bytes)
        current_password = str(form_values.get("current_password", "") or "")
        new_password = str(form_values.get("new_password", "") or "")
        confirm_password = str(form_values.get("confirm_password", "") or "")
        try:
            async with managed_auth_write_lock:
                updated_auth_state = await _call_sync(
                    managed_auth_store.update_password,
                    current_password=current_password,
                    new_password=new_password,
                    confirm_password=confirm_password,
                )
        except ValueError as exc:
            _apply_auth_context(
                request,
                mode="managed",
                username=authenticated_username,
                must_change_password=auth_state.must_change_password,
            )
            return ctx.render(
                request,
                "auth_password.html",
                page_title="Change password",
                active_page="auth_password",
                restart_notice=(
                    "Finish this step once. After that, Twinr Control stays unlocked for normal LAN sign-in."
                    if auth_state.must_change_password
                    else "Change the local Twinr Control password used for sign-in."
                ),
                intro=(
                    "Choose a new password before the full Twinr Control portal is unlocked."
                    if auth_state.must_change_password
                    else "Update the password used to sign in to Twinr Control."
                ),
                shell_mode="auth",
                password_min_length=web_auth_password_min_length(),
                error_message=_public_error_message(exc, fallback="Twinr could not save the new password."),
            )

        response = RedirectResponse("/?saved=1", status_code=status.HTTP_303_SEE_OTHER)
        _set_managed_auth_cookie(
            response,
            state=updated_auth_state,
            username=updated_auth_state.username,
            request=request,
        )
        return response

    @app.post("/auth/logout")
    async def auth_logout(request: Request) -> Response:
        """End the current managed web session."""

        if not managed_auth_enabled:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="This page is not available.")
        response = RedirectResponse("/auth/login", status_code=status.HTTP_303_SEE_OTHER)
        _clear_managed_auth_cookie(response)
        return response

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request) -> HTMLResponse:
        """Render the dashboard overview page."""

        config, env_values = await _call_sync(ctx.load_state)
        snapshot = await _call_sync(ctx.load_snapshot, config)
        reminders = await _call_sync(ctx.reminder_store(config).load_entries)
        # AUDIT-FIX(#10): Sort pending reminders before selecting the next one so the dashboard reflects the true earliest due item.
        pending_reminders = tuple(sorted((entry for entry in reminders if not entry.delivered), key=_reminder_sort_key))
        delivered_reminders = tuple(entry for entry in reminders if entry.delivered)
        next_due_entry = pending_reminders[0] if pending_reminders else None
        ops_event_store = ctx.event_store()
        checks = await _call_sync(run_config_checks, config)
        checks_summary = check_summary(checks)
        usage_store = ctx.usage_store()
        usage_summary = await _call_sync(usage_store.summary, within_hours=24)
        recent_event_rows = await _call_sync(ops_event_store.tail, limit=25)
        health_snapshot = await _call_sync(collect_system_health, config, snapshot=snapshot, event_store=ops_event_store)
        self_coding_status = await _call_sync(build_self_coding_operator_status, ctx.self_coding_store())
        recent_errors = [
            entry
            for entry in recent_event_rows
            if str(entry.get("level", "")).lower() == "error"
        ][-3:]
        cards = (
            DashboardCard(
                title="Conversation",
                value=snapshot.status.title(),
                detail=f"{env_values.get('TWINR_PROVIDER_REALTIME', 'openai').upper()} realtime · LLM {config.default_model}",
                href="/memory",
            ),
            DashboardCard(
                title="Personality",
                value="Loaded",
                detail=f"SYSTEM/PERSONALITY/USER from {config.personality_dir}/",
                href="/personality",
            ),
            DashboardCard(
                title="Memory",
                value=f"{snapshot.memory_count} live turns",
                detail=f"Keep recent {config.memory_keep_recent} · cap {config.memory_max_turns}",
                href="/memory",
            ),
            DashboardCard(
                title="Reminders",
                value=f"{len(pending_reminders)} pending",
                detail=(
                    f"Next due {format_due_label(next_due_entry.due_at, timezone_name=config.local_timezone_name)}"
                    if next_due_entry is not None
                    else f"{len(delivered_reminders)} delivered"
                ),
                href="/memory",
            ),
            DashboardCard(
                title="Printer",
                value=config.printer_queue,
                detail=f"Header {config.printer_header_text} · width {config.printer_line_width}",
                href="/settings",
            ),
            DashboardCard(
                title="Ops",
                value=f"{checks_summary.get('fail', 0)} fail · {checks_summary.get('warn', 0)} warn",
                detail="Self-tests, config checks, usage, health, support",
                href="/ops/config",
            ),
            DashboardCard(
                title="LLM usage",
                value=f"{usage_summary.requests_total} req · {usage_summary.total_tokens} tok",
                detail=usage_summary.latest_model or "No usage records yet",
                href="/ops/usage",
            ),
            DashboardCard(
                title="System",
                value=health_snapshot.status.upper(),
                detail=_health_card_detail(health_snapshot),
                href="/ops/health",
            ),
        )
        if self_coding_status.has_activity:
            cards = (
                *cards,
                DashboardCard(
                    title="Self-coding",
                    value=self_coding_status.card_value(),
                    detail=self_coding_status.card_detail(),
                    href="/ops/self-coding",
                ),
            )
        return ctx.render(
            request,
            "dashboard.html",
            page_title="Dashboard",
            active_page="dashboard",
            config=config,
            env_values=env_values,
            cards=cards,
            provider_status=_provider_status(env_values),
            snapshot=snapshot,
            check_summary=checks_summary,
            recent_errors=_format_log_rows(recent_errors),
            usage_summary=usage_summary,
            health_snapshot=health_snapshot,
            voice_snapshot_label=_voice_snapshot_label(snapshot),
        )

    @app.get("/ops/self-test", response_class=HTMLResponse)
    async def ops_self_test(request: Request) -> HTMLResponse:
        """Render the self-test selection page."""

        config, _env_values = await _call_sync(ctx.load_state)
        tests = await _call_sync(TwinrSelfTestRunner.available_tests)
        return ctx.render(
            request,
            "ops_self_test.html",
            page_title="Hardware Self-Test",
            active_page="ops_self_test",
            restart_notice="These self-tests run against the local device and may access real hardware.",
            config=config,
            tests=tests,
            result=None,
            artifact_href=None,
        )

    @app.post("/ops/self-test", response_class=HTMLResponse)
    async def run_ops_self_test(request: Request) -> Response:
        """Run one hardware self-test and render the result."""

        try:
            config, _env_values = await _call_sync(ctx.load_state)
            form = await _parse_bounded_form(request, max_form_bytes=max_form_bytes)
            test_name = _require_non_empty(form.get("test_name", ""), message="Please choose a self-test.")
            async with ops_job_lock:
                # AUDIT-FIX(#5,#6): Offload the blocking self-test runner and turn failures into safe operator feedback.
                result = await _call_sync(TwinrSelfTestRunner(config).run, test_name)
            tests = await _call_sync(TwinrSelfTestRunner.available_tests)
        except Exception as exc:
            logger.exception("Twinr self-test failed", exc_info=exc)
            return _redirect_with_error(
                "/ops/self-test",
                _public_error_message(exc, fallback="Twinr could not run that self-test. Please try again."),
            )

        artifact_href = None
        if result.artifact_name:
            artifact_href = f"/ops/self-test/artifacts/{result.artifact_name}"
        return ctx.render(
            request,
            "ops_self_test.html",
            page_title="Hardware Self-Test",
            active_page="ops_self_test",
            restart_notice="These self-tests run against the local device and may access real hardware.",
            config=config,
            tests=tests,
            result=result,
            artifact_href=artifact_href,
        )

    @app.get("/ops/self-coding", response_class=HTMLResponse)
    async def ops_self_coding(request: Request) -> HTMLResponse:
        """Render the self-coding operator page."""

        page_context = await _call_sync(build_self_coding_ops_page_context, ctx.self_coding_store())
        return ctx.render(
            request,
            "ops_self_coding.html",
            page_title="Self-coding operations",
            active_page="ops_self_coding",
            restart_notice="This page shows learned-skill compile telemetry, health, and explicit operator controls.",
            **page_context,
        )

    @app.post("/ops/self-coding/pause")
    async def pause_self_coding_activation(request: Request) -> Response:
        """Pause one learned self-coding skill version from the web UI."""

        try:
            config, _env_values = await _call_sync(ctx.load_state)
            form = await _parse_bounded_form(request, max_form_bytes=max_form_bytes)
            skill_id = _require_non_empty(form.get("skill_id", ""), message="Please choose a learned skill.")
            version = _require_positive_int(form.get("version", ""), message="Please choose a valid learned skill version.")
            reason = form.get("reason", "").strip() or "operator_pause"
            activation_service = SelfCodingActivationService(
                store=ctx.self_coding_store(),
                automation_store=ctx.automation_store(config),
            )
            async with state_write_lock:
                await _call_sync(
                    activation_service.pause_activation,
                    skill_id=skill_id,
                    version=version,
                    reason=reason,
                )
        except Exception as exc:
            return _redirect_with_error(
                "/ops/self-coding",
                _public_error_message(exc, fallback="Twinr could not pause that learned skill."),
            )
        return _redirect_saved("/ops/self-coding")

    @app.post("/ops/self-coding/reactivate")
    async def reactivate_self_coding_activation(request: Request) -> Response:
        """Re-enable one paused learned self-coding skill version from the web UI."""

        try:
            config, _env_values = await _call_sync(ctx.load_state)
            form = await _parse_bounded_form(request, max_form_bytes=max_form_bytes)
            skill_id = _require_non_empty(form.get("skill_id", ""), message="Please choose a learned skill.")
            version = _require_positive_int(form.get("version", ""), message="Please choose a valid learned skill version.")
            activation_service = SelfCodingActivationService(
                store=ctx.self_coding_store(),
                automation_store=ctx.automation_store(config),
            )
            async with state_write_lock:
                await _call_sync(
                    activation_service.reactivate_activation,
                    skill_id=skill_id,
                    version=version,
                )
        except Exception as exc:
            return _redirect_with_error(
                "/ops/self-coding",
                _public_error_message(exc, fallback="Twinr could not reactivate that learned skill."),
            )
        return _redirect_saved("/ops/self-coding")

    @app.post("/ops/self-coding/rollback")
    async def rollback_self_coding_activation(request: Request) -> Response:
        """Restore the previous learned-skill version from the web UI."""

        try:
            config, _env_values = await _call_sync(ctx.load_state)
            form = await _parse_bounded_form(request, max_form_bytes=max_form_bytes)
            skill_id = _require_non_empty(form.get("skill_id", ""), message="Please choose a learned skill.")
            raw_target_version = form.get("target_version", "").strip()
            target_version = None if not raw_target_version else _require_positive_int(
                raw_target_version,
                message="Please choose a valid rollback target version.",
            )
            activation_service = SelfCodingActivationService(
                store=ctx.self_coding_store(),
                automation_store=ctx.automation_store(config),
            )
            async with state_write_lock:
                await _call_sync(
                    activation_service.rollback_activation,
                    skill_id=skill_id,
                    target_version=target_version,
                )
        except Exception as exc:
            return _redirect_with_error(
                "/ops/self-coding",
                _public_error_message(exc, fallback="Twinr could not roll back that learned skill."),
            )
        return _redirect_saved("/ops/self-coding")

    @app.post("/ops/self-coding/retest")
    async def retest_self_coding_activation(request: Request) -> Response:
        """Run one capture-only retest for an active learned skill version."""

        try:
            form = await _parse_bounded_form(request, max_form_bytes=max_form_bytes)
            skill_id = _require_non_empty(form.get("skill_id", ""), message="Please choose a learned skill.")
            version = _require_positive_int(form.get("version", ""), message="Please choose a valid learned skill version.")
            async with state_write_lock:
                await _call_sync(
                    run_self_coding_skill_retest,
                    project_root=ctx.project_root,
                    env_file=ctx.env_path,
                    skill_id=skill_id,
                    version=version,
                    environment="web",
                )
        except Exception as exc:
            return _redirect_with_error(
                "/ops/self-coding",
                _public_error_message(exc, fallback="Twinr could not retest that learned skill."),
            )
        return _redirect_saved("/ops/self-coding")

    @app.post("/ops/self-coding/cleanup")
    async def cleanup_self_coding_activation(request: Request) -> Response:
        """Retire one inactive learned skill version and remove its runtime artifacts."""

        try:
            config, _env_values = await _call_sync(ctx.load_state)
            form = await _parse_bounded_form(request, max_form_bytes=max_form_bytes)
            skill_id = _require_non_empty(form.get("skill_id", ""), message="Please choose a learned skill.")
            version = _require_positive_int(form.get("version", ""), message="Please choose a valid learned skill version.")
            activation_service = SelfCodingActivationService(
                store=ctx.self_coding_store(),
                automation_store=ctx.automation_store(config),
            )
            runtime_store = SelfCodingSkillRuntimeStore(ctx.self_coding_store().root)
            async with state_write_lock:
                await _call_sync(
                    activation_service.cleanup_activation,
                    skill_id=skill_id,
                    version=version,
                    runtime_store=runtime_store,
                )
        except Exception as exc:
            return _redirect_with_error(
                "/ops/self-coding",
                _public_error_message(exc, fallback="Twinr could not clean up that learned skill version."),
            )
        return _redirect_saved("/ops/self-coding")

    @app.post("/ops/self-coding/cleanup-run")
    async def cleanup_self_coding_run(request: Request) -> Response:
        """Mark one stale sandbox or retest run as cleaned from the operator UI."""

        try:
            form = await _parse_bounded_form(request, max_form_bytes=max_form_bytes)
            run_id = _require_non_empty(form.get("run_id", ""), message="Please choose a valid self-coding run.")
            async with state_write_lock:
                await _call_sync(
                    cleanup_stale_execution_run,
                    store=ctx.self_coding_store(),
                    run_id=run_id,
                    reason="operator_cleanup",
                )
        except Exception as exc:
            return _redirect_with_error(
                "/ops/self-coding",
                _public_error_message(exc, fallback="Twinr could not clean up that stale run."),
            )
        return _redirect_saved("/ops/self-coding")

    @app.post("/ops/self-coding/cleanup-compile")
    async def cleanup_self_coding_compile(request: Request) -> Response:
        """Abort one stale compile run from the operator UI."""

        try:
            form = await _parse_bounded_form(request, max_form_bytes=max_form_bytes)
            job_id = _require_non_empty(form.get("job_id", ""), message="Please choose a valid compile job.")
            async with state_write_lock:
                await _call_sync(
                    cleanup_stale_compile_status,
                    store=ctx.self_coding_store(),
                    job_id=job_id,
                    reason="operator_cleanup",
                )
        except Exception as exc:
            return _redirect_with_error(
                "/ops/self-coding",
                _public_error_message(exc, fallback="Twinr could not clean up that stale compile."),
            )
        return _redirect_saved("/ops/self-coding")

    @app.get("/ops/self-test/artifacts/{artifact_name}")
    async def download_self_test_artifact(artifact_name: str) -> FileResponse:
        """Return one self-test artifact after path validation."""

        # AUDIT-FIX(#8): Re-validate downloadable artifact paths here so missing or unsafe names fail closed with a 404.
        artifact_path = _resolve_downloadable_file(ctx.ops_paths.self_tests_root, artifact_name)
        return FileResponse(artifact_path, filename=artifact_path.name)

    @app.get("/ops/logs", response_class=HTMLResponse)
    async def ops_logs(request: Request) -> HTMLResponse:
        """Render recent structured ops events."""

        logs = await _call_sync(ctx.event_store().tail, limit=100)
        return ctx.render(
            request,
            "ops_logs.html",
            page_title="Ops Logs",
            active_page="ops_logs",
            restart_notice="This view shows the latest 100 structured local events.",
            logs=_format_log_rows(logs),
        )

    @app.get("/ops/usage", response_class=HTMLResponse)
    async def ops_usage(request: Request) -> HTMLResponse:
        """Render LLM usage summaries and recent usage rows."""

        store = ctx.usage_store()
        summary_all = await _call_sync(store.summary)
        summary_24h = await _call_sync(store.summary, within_hours=24)
        usage_rows = await _call_sync(store.tail, limit=100)
        return ctx.render(
            request,
            "ops_usage.html",
            page_title="LLM Usage",
            active_page="ops_usage",
            restart_notice="Usage records are written locally whenever Twinr completes a tracked OpenAI response call.",
            summary_all=summary_all,
            summary_24h=summary_24h,
            usage_rows=_format_usage_rows(usage_rows),
            usage_path=str(ctx.ops_paths.usage_path),
        )

    @app.get("/ops/health", response_class=HTMLResponse)
    async def ops_health(request: Request) -> HTMLResponse:
        """Render live system-health details."""

        config, _env_values = await _call_sync(ctx.load_state)
        snapshot = await _call_sync(ctx.load_snapshot, config)
        remote_memory_watchdog = None
        remote_memory_watchdog_error = None
        try:
            remote_memory_watchdog = await _call_sync(ctx.load_remote_memory_watchdog, config)
        except Exception as exc:
            remote_memory_watchdog_error = _public_error_message(
                exc,
                fallback="Twinr could not read the remote memory watchdog state.",
            )
        ops_event_store = ctx.event_store()
        recent_event_rows = await _call_sync(ops_event_store.tail, limit=25)
        recent_errors = [
            entry
            for entry in recent_event_rows
            if str(entry.get("level", "")).lower() == "error"
        ][-5:]
        health = await _call_sync(collect_system_health, config, snapshot=snapshot, event_store=ops_event_store)
        return ctx.render(
            request,
            "ops_health.html",
            page_title="System Health",
            active_page="ops_health",
            restart_notice="This page reads live Raspberry Pi and Twinr process state from the local machine.",
            health=health,
            snapshot=snapshot,
            remote_memory_watchdog=remote_memory_watchdog,
            remote_memory_watchdog_error=remote_memory_watchdog_error,
            recent_errors=_format_log_rows(recent_errors),
        )

    @app.get("/ops/debug", response_class=HTMLResponse)
    async def ops_debug(request: Request) -> HTMLResponse:
        """Render the full tabbed operator debug view."""

        active_tab = coerce_ops_debug_tab(request.query_params.get("tab"))
        config, env_values = await _call_sync(ctx.load_state)
        snapshot = await _call_sync(ctx.load_snapshot, config)
        event_store = ctx.event_store()
        usage_store = ctx.usage_store()
        recent_events = await _call_sync(event_store.tail, limit=120)
        recent_usage = await _call_sync(usage_store.tail, limit=80)
        summary_all = await _call_sync(usage_store.summary)
        summary_24h = await _call_sync(usage_store.summary, within_hours=24)
        health = await _call_sync(collect_system_health, config, snapshot=snapshot, event_store=event_store)

        remote_memory_watchdog = None
        remote_memory_watchdog_assessment = None
        remote_memory_watchdog_error = None
        try:
            remote_memory_watchdog = await _call_sync(ctx.load_remote_memory_watchdog, config)
            remote_memory_watchdog_assessment = await _call_sync(
                assess_required_remote_watchdog_snapshot,
                config,
            )
        except Exception as exc:
            remote_memory_watchdog_error = _public_error_message(
                exc,
                fallback="Twinr could not read the remote memory watchdog state.",
            )

        device_overview = None
        if active_tab == "hardware":
            device_overview = await _call_sync(collect_device_overview, config, event_store=event_store)

        config_checks: tuple[Any, ...] = ()
        config_check_summary = None
        if active_tab == "raw":
            config_checks = await _call_sync(run_config_checks, config)
            config_check_summary = await _call_sync(check_summary, config_checks)

        memory_search_query = str(request.query_params.get("memory_query", "")).strip()
        memory_search_result = None
        memory_search_error = None
        if active_tab == "memory_search" and memory_search_query:
            try:
                memory_search_result = await _call_sync(
                    run_long_term_operator_search,
                    config,
                    memory_search_query,
                )
            except Exception as exc:
                memory_search_error = _public_error_message(
                    exc,
                    fallback="Twinr could not search long-term memory right now.",
                )

        conversation_lab_state = None
        if active_tab == "conversation_lab":
            conversation_lab_state = await _call_sync(
                load_conversation_lab_state,
                ctx.ops_paths,
                session_id=str(request.query_params.get("lab_session", "")).strip() or None,
            )

        page_context = build_ops_debug_page_context(
            active_tab=active_tab,
            env_path=ctx.env_path,
            config=config,
            ops_paths=ctx.ops_paths,
            snapshot=snapshot,
            health=health,
            remote_memory_watchdog=remote_memory_watchdog,
            remote_memory_watchdog_assessment=remote_memory_watchdog_assessment,
            remote_memory_watchdog_error=remote_memory_watchdog_error,
            recent_events=tuple(recent_events),
            recent_usage=tuple(recent_usage),
            summary_all=summary_all,
            summary_24h=summary_24h,
            device_overview=device_overview,
            redacted_env_values=redact_env_values(env_values),
            config_checks=tuple(config_checks),
            config_check_summary=config_check_summary,
            memory_search_query=memory_search_query,
            memory_search_result=memory_search_result,
            memory_search_error=memory_search_error,
            conversation_lab_state=conversation_lab_state,
        )
        return ctx.render(
            request,
            "ops_debug.html",
            page_title="Debug View",
            active_page="ops_debug",
            intro=(
                "Interactive operator view of Twinr runtime evidence, grouped into runtime, ChonkyDB, memory, LLM, events, hardware, and raw artifacts."
                if active_tab == "conversation_lab"
                else "Read-only operator view of Twinr runtime evidence, grouped into runtime, ChonkyDB, LLM, events, hardware, and raw artifacts."
            ),
            restart_notice=(
                "Conversation Lab runs real provider, tool, reminder, automation, settings, and remote-memory paths against this Twinr instance."
                if active_tab == "conversation_lab"
                else "This page is read-only and reflects the latest local Twinr artifacts."
            ),
            **page_context,
        )

    @app.post("/ops/debug/conversation-lab/new")
    async def create_ops_debug_conversation_lab_session(request: Request) -> Response:
        """Create one empty portal conversation-lab session."""

        try:
            await _parse_bounded_form(request, max_form_bytes=max_form_bytes)
            async with conversation_lab_lock:
                session_id = await _call_sync(create_conversation_lab_session, ctx.ops_paths)
        except Exception as exc:
            return _redirect_with_error(
                _conversation_lab_href(),
                _public_error_message(exc, fallback="Twinr could not start a new portal conversation."),
            )
        return RedirectResponse(_conversation_lab_href(session_id), status_code=status.HTTP_303_SEE_OTHER)

    @app.post("/ops/debug/conversation-lab/send")
    async def send_ops_debug_conversation_lab_turn(request: Request) -> Response:
        """Run one portal conversation-lab turn against the real Twinr text path."""

        current_session_id = None
        try:
            config, _env_values = await _call_sync(ctx.load_state)
            form = await _parse_bounded_form(request, max_form_bytes=max_form_bytes)
            current_session_id = str(form.get("session_id", "")).strip() or None
            prompt = _require_non_empty(
                form.get("prompt", ""),
                message="Please enter a prompt for the portal conversation.",
            )
            async with conversation_lab_lock:
                async with state_write_lock:
                    resolved_session_id = await _call_sync(
                        run_conversation_lab_turn,
                        config,
                        ctx.env_path,
                        ctx.ops_paths,
                        session_id=current_session_id,
                        prompt=prompt,
                    )
        except Exception as exc:
            return _redirect_with_error(
                _conversation_lab_href(current_session_id),
                _public_error_message(exc, fallback="Twinr could not run that portal conversation turn."),
            )
        return RedirectResponse(_conversation_lab_href(resolved_session_id), status_code=status.HTTP_303_SEE_OTHER)

    @app.get("/ops/devices", response_class=HTMLResponse)
    async def ops_devices(request: Request) -> HTMLResponse:
        """Render the detected device overview."""

        config, _env_values = await _call_sync(ctx.load_state)
        overview = await _call_sync(collect_device_overview, config, event_store=ctx.event_store())
        return ctx.render(
            request,
            "ops_devices.html",
            page_title="Devices",
            active_page="ops_devices",
            restart_notice=(
                "This page shows only signals Twinr can confirm locally. "
                "Unknown means the current device path does not expose that signal."
            ),
            overview=overview,
        )

    @app.get("/ops/config", response_class=HTMLResponse)
    async def ops_config(request: Request) -> HTMLResponse:
        """Render config checks plus redacted env values."""

        config, env_values = await _call_sync(ctx.load_state)
        checks = await _call_sync(run_config_checks, config)
        return ctx.render(
            request,
            "ops_config.html",
            page_title="Config Checks",
            active_page="ops_config",
            restart_notice="This page checks plausibility, not full end-to-end hardware behavior.",
            config=config,
            checks=checks,
            summary=check_summary(checks),
            artifacts_root=str(ctx.ops_paths.artifacts_root),
            redacted_env=redact_env_values(env_values),
        )

    @app.get("/ops/support", response_class=HTMLResponse)
    async def ops_support(request: Request) -> HTMLResponse:
        """Render support-bundle status and recent bundles."""

        config, _env_values = await _call_sync(ctx.load_state)
        bundles = await _call_sync(_recent_named_files, ctx.ops_paths.bundles_root, suffix=".zip")
        return ctx.render(
            request,
            "ops_support.html",
            page_title="Support Bundle",
            active_page="ops_support",
            restart_notice="Bundles are written under artifacts/ops/support_bundles and contain only redacted environment data.",
            config=config,
            bundle=None,
            bundles=bundles,
        )

    @app.post("/ops/support", response_class=HTMLResponse)
    async def create_support_bundle(request: Request) -> Response:
        """Build a support bundle and render the result."""

        try:
            config, _env_values = await _call_sync(ctx.load_state)
            await _parse_bounded_form(request, max_form_bytes=max_form_bytes)
            async with ops_job_lock:
                bundle = await _call_sync(build_support_bundle, config, env_path=ctx.env_path)
            bundles = await _call_sync(_recent_named_files, ctx.ops_paths.bundles_root, suffix=".zip")
        except Exception as exc:
            logger.exception("Twinr support bundle creation failed", exc_info=exc)
            return _redirect_with_error(
                "/ops/support",
                _public_error_message(exc, fallback="Twinr could not build the support bundle. Please try again."),
            )

        return ctx.render(
            request,
            "ops_support.html",
            page_title="Support Bundle",
            active_page="ops_support",
            restart_notice="Bundles are written under artifacts/ops/support_bundles and contain only redacted environment data.",
            config=config,
            bundle=bundle,
            bundles=bundles,
        )

    @app.get("/ops/support/download/{bundle_name}")
    async def download_support_bundle(bundle_name: str) -> FileResponse:
        """Return one support bundle after path validation."""

        artifact_path = _resolve_downloadable_file(ctx.ops_paths.bundles_root, bundle_name)
        return FileResponse(artifact_path, filename=artifact_path.name)

    @app.get("/integrations", response_class=HTMLResponse)
    async def integrations(request: Request) -> HTMLResponse:
        """Render the integrations configuration page."""

        config, env_values = await _call_sync(ctx.load_state)
        store = ctx.integration_store()
        email_record = await _call_sync(store.get, "email_mailbox")
        calendar_record = await _call_sync(store.get, "calendar_agenda")
        smart_home_record = await _call_sync(store.get, "smart_home_hub")
        runtime = await _call_sync(build_managed_integrations, ctx.project_root, env_path=ctx.env_path)
        pairing_snapshot = await _call_sync(whatsapp_pairing.load_snapshot)
        return ctx.render(
            request,
            "integrations_page.html",
            page_title="Integrations",
            active_page="integrations",
            restart_notice=(
                "These settings stay local to this Twinr install. "
                "Email secrets remain only in .env, while WhatsApp keeps its linked-device session in its own auth folder."
            ),
            intro=(
                "Set up safe external connections for Twinr. Email and calendar live here as managed integrations, "
                "and WhatsApp uses a guided self-chat wizard so pairing stays bounded and visible."
            ),
            integration_store_path=str(store.path),
            overview_rows=_integration_overview_rows(runtime.readiness),
            whatsapp_summary=_whatsapp_integration_context(
                config,
                env_values,
                env_path=ctx.env_path,
                pairing_snapshot=pairing_snapshot,
            ),
            email_sections=_email_integration_sections(email_record, env_values),
            calendar_sections=_calendar_integration_sections(calendar_record),
            smart_home_sections=_smart_home_integration_sections(smart_home_record, env_values),
        )

    @app.post("/integrations")
    async def save_integrations(request: Request) -> RedirectResponse:
        """Persist one integration form submission."""

        try:
            _config, env_values = await _call_sync(ctx.load_state)
            form = await _parse_bounded_form(request, max_form_bytes=max_form_bytes)
            integration_id = _require_non_empty(
                form.get("_integration_id", ""),
                message="Please choose a valid integration form.",
            )
            store = ctx.integration_store()

            if integration_id == "email_mailbox":
                record, env_updates = await _call_sync(_build_email_integration_record, form, env_values)
            elif integration_id == "calendar_agenda":
                record, env_updates = await _call_sync(_build_calendar_integration_record, form)
            elif integration_id == "smart_home_hub":
                record, env_updates = await _call_sync(_build_smart_home_integration_record, form, env_values)
            else:
                raise ValueError("Please choose a valid integration form.")

            # AUDIT-FIX(#7): Keep integration store writes and matching .env updates serialized.
            async with state_write_lock:
                await _call_sync(store.save, record)
                if env_updates:
                    await _call_sync(write_env_updates, ctx.env_path, env_updates)
        except Exception as exc:
            logger.exception("Twinr integration save failed", exc_info=exc)
            return _redirect_with_error(
                "/integrations",
                _public_error_message(exc, fallback="Twinr could not save that integration. Please check the fields and try again."),
            )
        return _redirect_saved("/integrations")

    @app.get("/voice-profile", response_class=HTMLResponse)
    async def voice_profile_page(request: Request) -> HTMLResponse:
        """Render the voice-profile page."""

        config, _env_values = await _call_sync(ctx.load_state)
        snapshot = await _call_sync(ctx.load_snapshot, config)
        return ctx.render(
            request,
            "voice_profile_page.html",
            page_title="Voice Profile",
            active_page="voice_profile",
            restart_notice=(
                "Voice profiling is local-only and soft-gated. It does not replace explicit confirmation for sensitive actions."
            ),
            intro=(
                "Phase 1 uses the normal conversation microphone only. No raw enrollment audio is stored, "
                "and support bundles omit live voice-assessment fields."
            ),
            **_voice_profile_page_context(config, snapshot),
        )

    @app.post("/voice-profile", response_class=HTMLResponse)
    async def voice_profile_action(request: Request) -> Response:
        """Run one bounded voice-profile action and re-render the page."""

        try:
            config, _env_values = await _call_sync(ctx.load_state)
            form = await _parse_bounded_form(request, max_form_bytes=max_form_bytes)
            action = _require_non_empty(form.get("_action", ""), message="Please choose a voice profile action.")

            async with voice_profile_lock:
                # AUDIT-FIX(#5,#6,#11): Run microphone/profile work off the event loop, sanitize failures, and reload snapshot after changes.
                if action == "enroll":
                    def _enroll() -> dict[str, str]:
                        """Capture one sample and append it to the local voice profile."""

                        monitor = VoiceProfileMonitor.from_config(config)
                        sample = _capture_voice_profile_sample(config)
                        template = monitor.enroll_wav_bytes(sample)
                        return {
                            "status": "ok",
                            "title": "Profile updated",
                            "detail": (
                                f"Stored local template sample {template.sample_count}/{config.voice_profile_max_samples}. "
                                "No raw audio was kept."
                            ),
                        }

                    action_result = await _call_sync(_enroll)
                elif action == "verify":
                    def _verify() -> dict[str, str]:
                        """Capture one sample and assess it against the local voice profile."""

                        monitor = VoiceProfileMonitor.from_config(config)
                        sample = _capture_voice_profile_sample(config)
                        assessment = monitor.assess_wav_bytes(sample)
                        return _voice_action_result(assessment)

                    action_result = await _call_sync(_verify)
                elif action == "reset":
                    def _reset() -> dict[str, str]:
                        """Delete the stored local voice profile."""

                        monitor = VoiceProfileMonitor.from_config(config)
                        monitor.reset()
                        return {
                            "status": "ok",
                            "title": "Profile reset",
                            "detail": "The local voice profile template was deleted.",
                        }

                    action_result = await _call_sync(_reset)
                else:
                    raise ValueError("Please choose a valid voice profile action.")

            snapshot = await _call_sync(ctx.load_snapshot, config)
        except Exception as exc:
            logger.exception("Twinr voice profile action failed", exc_info=exc)
            config, _env_values = await _call_sync(ctx.load_state)
            snapshot = await _call_sync(ctx.load_snapshot, config)
            return ctx.render(
                request,
                "voice_profile_page.html",
                page_title="Voice Profile",
                active_page="voice_profile",
                restart_notice=(
                    "Voice profiling is local-only and soft-gated. It does not replace explicit confirmation for sensitive actions."
                ),
                intro=(
                    "Phase 1 uses the normal conversation microphone only. No raw enrollment audio is stored, "
                    "and support bundles omit live voice-assessment fields."
                ),
                **_voice_profile_page_context(
                    config,
                    snapshot,
                    action_error=_public_error_message(
                        exc,
                        fallback="Twinr could not finish that voice profile step. Please try again.",
                    ),
                ),
            )

        return ctx.render(
            request,
            "voice_profile_page.html",
            page_title="Voice Profile",
            active_page="voice_profile",
            restart_notice=(
                "Voice profiling is local-only and soft-gated. It does not replace explicit confirmation for sensitive actions."
            ),
            intro=(
                "Phase 1 uses the normal conversation microphone only. No raw enrollment audio is stored, "
                "and support bundles omit live voice-assessment fields."
            ),
            **_voice_profile_page_context(config, snapshot, action_result=action_result),
        )

    @app.get("/automations", response_class=HTMLResponse)
    async def automations(request: Request) -> HTMLResponse:
        """Render the automations page."""

        config, _env_values = await _call_sync(ctx.load_state)
        store = ctx.automation_store(config)
        integration_family_providers = tuple(await _call_sync(integration_automation_family_providers, ctx.project_root))
        edit_ref = request.query_params.get("edit", "").strip() or None
        page_context = await _call_sync(
            build_automation_page_context,
            store,
            timezone_name=config.local_timezone_name,
            edit_ref=edit_ref,
            integration_blocks=tuple(provider.block() for provider in integration_family_providers),
        )
        return ctx.render(
            request,
            "automations_page.html",
            page_title="Automations",
            active_page="automations",
            restart_notice=(
                "Automations run locally while Twinr is idle. Keep them short, bounded, and easy to understand."
            ),
            intro=(
                "Automations let Twinr do scheduled or sensor-driven work without adding new user-facing modes. "
                "This page stays family-based so future integrations can add their own automation blocks."
            ),
            **page_context,
        )

    @app.post("/automations")
    async def save_automations(request: Request) -> RedirectResponse:
        """Persist one automation change requested by the web UI."""

        try:
            config = await _call_sync(TwinrConfig.from_env, ctx.env_path)
            form = await _parse_bounded_form(request, max_form_bytes=max_form_bytes)
            action = _require_non_empty(form.get("_action", ""), message="Please choose an automation action.")
            store = ctx.automation_store(config)
            integration_family_providers = tuple(await _call_sync(integration_automation_family_providers, ctx.project_root))

            # AUDIT-FIX(#7,#9): Serialize automation writes and fail fast on invalid action/input combinations.
            async with state_write_lock:
                if action == "save_time_automation":
                    await _call_sync(save_time_automation, store, form, timezone_name=config.local_timezone_name)
                elif action == "save_sensor_automation":
                    await _call_sync(save_sensor_automation, store, form)
                elif action == "toggle_automation":
                    automation_id = _require_non_empty(
                        form.get("automation_id", ""),
                        message="Please choose an automation to change.",
                    )
                    await _call_sync(toggle_automation_enabled, store, automation_id)
                elif action == "delete_automation":
                    automation_id = _require_non_empty(
                        form.get("automation_id", ""),
                        message="Please choose an automation to delete.",
                    )
                    await _call_sync(delete_automation, store, automation_id)
                elif any(provider.handles_action(action) for provider in integration_family_providers):
                    handled = False
                    for provider in integration_family_providers:
                        if not provider.handles_action(action):
                            continue
                        handled = await _call_sync(provider.handle_action, action, form=form, automation_store=store)
                        if handled:
                            break
                    if not handled:
                        raise ValueError("That integration automation cannot be changed yet.")
                else:
                    raise ValueError("Please choose a valid automation action.")
        except Exception as exc:
            logger.exception("Twinr automation save failed", exc_info=exc)
            return _redirect_with_error(
                "/automations",
                _public_error_message(exc, fallback="Twinr could not save that automation. Please check the fields and try again."),
            )
        return _redirect_saved("/automations")

    @app.get("/connect", response_class=HTMLResponse)
    async def connect(request: Request) -> HTMLResponse:
        """Render provider and credential settings."""

        _config, env_values = await _call_sync(ctx.load_state)
        sections = _connect_sections(env_values)
        return ctx.render(
            request,
            "connect_page.html",
            page_title="Connect",
            active_page="connect",
            intro="Choose providers and manage credentials. Hover the small (?) labels for short explanations. Today only OpenAI is wired in the runtime; the other providers are stored for the next integration pass.",
            form_action="/connect",
            sections=sections,
        )

    @app.post("/connect")
    async def save_connect(request: Request) -> RedirectResponse:
        """Persist provider selection and credential changes."""

        try:
            env_values = (await _call_sync(ctx.load_state))[1]
            form = await _parse_bounded_form(request, max_form_bytes=max_form_bytes)
            updates = _collect_standard_updates(form, exclude={"OPENAI_API_KEY", "DEEPINFRA_API_KEY", "OPENROUTER_API_KEY"})
            for secret_key in ("OPENAI_API_KEY", "DEEPINFRA_API_KEY", "OPENROUTER_API_KEY"):
                secret_value = form.get(secret_key, "").strip()
                if secret_value:
                    updates[secret_key] = secret_value
                elif secret_key not in env_values and secret_key == "OPENAI_API_KEY":
                    updates.setdefault(secret_key, "")
            # AUDIT-FIX(#7): Serialize provider credential writes to avoid partial .env updates.
            async with state_write_lock:
                await _call_sync(write_env_updates, ctx.env_path, updates)
        except Exception as exc:
            logger.exception("Twinr connect settings save failed", exc_info=exc)
            return _redirect_with_error(
                "/connect",
                _public_error_message(exc, fallback="Twinr could not save those provider settings. Please try again."),
            )
        return _redirect_saved("/connect")

    @app.get("/connect/whatsapp", response_class=HTMLResponse)
    async def connect_whatsapp(request: Request) -> HTMLResponse:
        """Render the WhatsApp self-chat setup wizard."""

        config, env_values = await _call_sync(ctx.load_state)
        pairing_snapshot = await _call_sync(whatsapp_pairing.load_snapshot)
        page_context = await _call_sync(
            build_whatsapp_wizard_page_context,
            config,
            env_values,
            env_path=ctx.env_path,
            pairing_snapshot=pairing_snapshot,
            requested_step=request.query_params.get("step"),
        )
        return ctx.render(
            request,
            "whatsapp_wizard.html",
            page_title="WhatsApp Setup",
            active_page="connect",
            restart_notice=(
                "The dashboard pairing window is temporary and bounded. "
                "The linked-device session itself stays in its own auth folder."
            ),
            intro=(
                "This wizard keeps WhatsApp in one internal self-chat with your own number. "
                "It forces self-chat mode on, blocks groups, and can open the pairing window directly from the UI."
            ),
            **page_context,
        )

    @app.post("/connect/whatsapp")
    async def save_connect_whatsapp(request: Request) -> RedirectResponse:
        """Persist one WhatsApp wizard step."""

        current_step = "chat"
        try:
            form = await _parse_bounded_form(request, max_form_bytes=max_form_bytes)
            action = _require_non_empty(
                form.get("_action", ""),
                message="Please choose a valid WhatsApp setup step.",
            )

            if action == "save_chat":
                allow_from = canonicalize_whatsapp_allow_from(
                    _require_non_empty(
                        form.get("TWINR_WHATSAPP_ALLOW_FROM", ""),
                        message="Please enter the WhatsApp number that Twinr should allow.",
                    )
                )
                updates = {
                    "TWINR_WHATSAPP_ALLOW_FROM": allow_from,
                    "TWINR_WHATSAPP_SELF_CHAT_MODE": "true",
                    "TWINR_WHATSAPP_GROUPS_ENABLED": "false",
                }
                next_step = "runtime"
                async with state_write_lock:
                    await _call_sync(write_env_updates, ctx.env_path, updates)
                return _redirect_saved("/connect/whatsapp", step=next_step)

            if action == "save_runtime":
                current_step = "runtime"
                node_binary = _require_non_empty(
                    form.get("TWINR_WHATSAPP_NODE_BINARY", ""),
                    message="Please enter the Node.js binary that should start the WhatsApp worker.",
                )
                auth_dir = normalize_project_relative_directory(
                    ctx.project_root,
                    form.get("TWINR_WHATSAPP_AUTH_DIR", ""),
                    label="the WhatsApp auth folder",
                )
                updates = {
                    "TWINR_WHATSAPP_NODE_BINARY": node_binary,
                    "TWINR_WHATSAPP_AUTH_DIR": auth_dir,
                }
                next_step = "pairing"
                async with state_write_lock:
                    await _call_sync(write_env_updates, ctx.env_path, updates)
                return _redirect_saved("/connect/whatsapp", step=next_step)

            if action == "start_pairing":
                current_step = "pairing"
                config, _env_values = await _call_sync(ctx.load_state)
                if not str(getattr(config, "whatsapp_allow_from", "") or "").strip():
                    raise ValueError("Please save your own WhatsApp number before you start pairing.")
                if not bool(config.whatsapp_self_chat_mode) or bool(config.whatsapp_groups_enabled):
                    raise ValueError("Twinr must keep self-chat mode on and group chats off before pairing starts.")
                runtime_probe = await _call_sync(probe_whatsapp_runtime, config, env_path=ctx.env_path)
                if not runtime_probe.node_ready:
                    raise ValueError(runtime_probe.node_detail)
                if not runtime_probe.worker_ready:
                    raise ValueError(runtime_probe.worker_detail)
                await _call_sync(whatsapp_pairing.load_snapshot)
                await _call_sync(whatsapp_pairing.start_pairing, config)
                return RedirectResponse(url="/connect/whatsapp?step=pairing", status_code=status.HTTP_303_SEE_OTHER)

            raise ValueError("Please choose a valid WhatsApp setup step.")
        except Exception as exc:
            logger.exception("Twinr WhatsApp setup save failed", exc_info=exc)
            return _redirect_with_error(
                "/connect/whatsapp",
                _public_error_message(exc, fallback="Twinr could not save that WhatsApp setup step. Please check the fields and try again."),
                step=current_step,
            )

    @app.get("/settings", response_class=HTMLResponse)
    async def settings(request: Request) -> HTMLResponse:
        """Render the main settings page."""

        config, env_values = await _call_sync(ctx.load_state)
        sections = _settings_sections(config, env_values)
        return ctx.render(
            request,
            "settings_page.html",
            page_title="Settings",
            active_page="settings",
            intro="All main device and runtime tuning lives here. Hover the small (?) labels for a quick explanation of each field.",
            form_action="/settings",
            sections=sections,
            adaptive_timing=_adaptive_timing_view(config),
        )

    @app.post("/settings")
    async def save_settings(request: Request) -> RedirectResponse:
        """Persist settings changes or reset adaptive timing."""

        try:
            config = await _call_sync(TwinrConfig.from_env, ctx.env_path)
            form = await _parse_bounded_form(request, max_form_bytes=max_form_bytes)
            action = form.get("_action", "save_settings").strip() or "save_settings"
            if action not in {"save_settings", "reset_adaptive_timing"}:
                raise ValueError("Please choose a valid settings action.")
            # AUDIT-FIX(#7,#9): Serialize settings writes and refuse unknown actions instead of silently falling through.
            async with state_write_lock:
                if action == "reset_adaptive_timing":
                    await _call_sync(AdaptiveTimingStore(config.adaptive_timing_store_path, config=config).reset)
                else:
                    await _call_sync(write_env_updates, ctx.env_path, _collect_standard_updates(form, exclude={"_action"}))
        except Exception as exc:
            logger.exception("Twinr settings save failed", exc_info=exc)
            return _redirect_with_error(
                "/settings",
                _public_error_message(exc, fallback="Twinr could not save those settings. Please try again."),
            )
        return _redirect_saved("/settings")

    @app.get("/memory", response_class=HTMLResponse)
    async def memory(request: Request) -> HTMLResponse:
        """Render memory, reminder, and print-bound settings."""

        config, env_values = await _call_sync(ctx.load_state)
        sections = _memory_sections(config, env_values)
        snapshot = await _call_sync(ctx.load_snapshot, config)
        durable_store = ctx.memory_store(config)
        durable_entries = await _call_sync(durable_store.load_entries)
        reminder_entries = await _call_sync(ctx.reminder_store(config).load_entries)
        reminder_rows = _reminder_rows(reminder_entries, timezone_name=config.local_timezone_name)
        return ctx.render(
            request,
            "memory_page.html",
            page_title="Memory",
            active_page="memory",
            intro="Tune on-device memory and print summarization bounds. Twinr also uses the configured long-term memory path for graph recall and background episodic persistence.",
            form_action="/memory",
            sections=sections,
            snapshot=snapshot,
            durable_memory_entries=durable_entries,
            durable_memory_path=str(Path(config.memory_markdown_path)),
            reminder_entries=tuple(row for row in reminder_rows if not row["delivered"]),
            delivered_reminder_entries=tuple(row for row in reminder_rows if row["delivered"]),
            reminder_path=str(Path(config.reminder_store_path)),
            reminder_default_due_at=_default_reminder_due_at(config),
            timezone_name=config.local_timezone_name,
        )

    @app.post("/memory")
    async def save_memory(request: Request) -> RedirectResponse:
        """Persist memory settings or reminder operations."""

        try:
            config = await _call_sync(TwinrConfig.from_env, ctx.env_path)
            form = await _parse_bounded_form(request, max_form_bytes=max_form_bytes)
            action = form.get("_action", "save_settings").strip() or "save_settings"
            if action not in {
                "save_settings",
                "add_memory",
                "add_reminder",
                "mark_reminder_delivered",
                "delete_reminder",
            }:
                raise ValueError("Please choose a valid memory action.")

            async with state_write_lock:
                # AUDIT-FIX(#9): Reject unknown actions and require the core fields that keep memory/reminder records coherent.
                if action == "add_memory":
                    summary = _require_non_empty(form.get("memory_summary", ""), message="Please enter a short memory summary.")
                    await _call_sync(
                        ctx.memory_store(config).remember,
                        kind=form.get("memory_kind", "") or "memory",
                        summary=summary,
                        details=form.get("memory_details", "") or None,
                    )
                elif action == "add_reminder":
                    due_at = _require_non_empty(form.get("reminder_due_at", ""), message="Please choose when the reminder is due.")
                    summary = _require_non_empty(form.get("reminder_summary", ""), message="Please enter a reminder summary.")
                    await _call_sync(
                        ctx.reminder_store(config).schedule,
                        due_at=due_at,
                        summary=summary,
                        details=form.get("reminder_details", "") or None,
                        kind=form.get("reminder_kind", "") or "reminder",
                        source="web_ui",
                        original_request=form.get("reminder_original_request", "") or None,
                    )
                elif action == "mark_reminder_delivered":
                    reminder_id = _require_non_empty(
                        form.get("reminder_id", ""),
                        message="Please choose a reminder first.",
                    )
                    await _call_sync(ctx.reminder_store(config).mark_delivered, reminder_id)
                elif action == "delete_reminder":
                    reminder_id = _require_non_empty(
                        form.get("reminder_id", ""),
                        message="Please choose a reminder first.",
                    )
                    await _call_sync(ctx.reminder_store(config).delete, reminder_id)
                else:
                    await _call_sync(write_env_updates, ctx.env_path, _collect_standard_updates(form, exclude={"_action"}))
        except KeyError as exc:
            logger.exception("Twinr reminder lookup failed", exc_info=exc)
            return _redirect_with_error("/memory", "That reminder was not found anymore.")
        except Exception as exc:
            logger.exception("Twinr memory save failed", exc_info=exc)
            return _redirect_with_error(
                "/memory",
                _public_error_message(exc, fallback="Twinr could not save that memory change. Please try again."),
            )
        return _redirect_saved("/memory")

    @app.get("/personality", response_class=HTMLResponse)
    async def personality(request: Request) -> HTMLResponse:
        """Render the hidden personality context editor."""

        config, _env_values = await _call_sync(ctx.load_state)
        # AUDIT-FIX(#4): Resolve the configured personality directory under the project root before any file access.
        personality_dir = _safe_project_subpath(ctx.project_root, config.personality_dir, label="Personality directory")
        system_file = _safe_file_in_dir(personality_dir, "SYSTEM.md", label="Personality system file")
        store = ctx.personality_context_store(config)
        system_text = await _call_sync(read_text_file, system_file)
        base_text = await _call_sync(store.load_base_text)
        managed_entries = await _call_sync(store.load_entries)
        return ctx.render(
            request,
            "context_page.html",
            page_title="Personality",
            active_page="personality",
            intro="Hidden system context for Twinr. Keep it short and stable so the assistant uses it silently instead of talking about it.",
            base_form_action="/personality",
            managed_form_action="/personality",
            raw_section_title="Base files",
            raw_section_description="Edit the stable base behavior here. Managed updates from tool calls stay separate below.",
            raw_fields=(
                FileBackedSetting(
                    key="SYSTEM",
                    label="SYSTEM.md",
                    value=system_text,
                    help_text="Core product behavior and permanent operating rules.",
                    input_type="textarea",
                ),
                FileBackedSetting(
                    key="PERSONALITY_BASE",
                    label="PERSONALITY.md base text",
                    value=base_text,
                    help_text="The stable hand-written part of the personality file. Managed tool updates are shown separately below.",
                    input_type="textarea",
                ),
            ),
            managed_section_title="Managed personality updates",
            managed_section_description="These entries were added by explicit user requests such as “speak more slowly” or “be less funny”.",
            managed_entries=managed_entries,
            managed_form_title="Add or update a managed personality rule",
            managed_form_description="Use a short category so future updates replace the right rule instead of creating duplicates.",
            managed_category_placeholder="response_style",
            managed_category_help="Examples: response_style, humor, pacing, confirmation_style.",
            managed_instruction_placeholder="Keep answers short, calm, and practical.",
            managed_instruction_help="Short, stable future behavior instruction.",
        )

    @app.post("/personality")
    async def save_personality(request: Request) -> RedirectResponse:
        """Persist personality base text or managed rules."""

        try:
            config = await _call_sync(TwinrConfig.from_env, ctx.env_path)
            # AUDIT-FIX(#4): Re-check the configured personality path on write so edits cannot escape the project tree.
            personality_dir = _safe_project_subpath(ctx.project_root, config.personality_dir, label="Personality directory")
            system_file = _safe_file_in_dir(personality_dir, "SYSTEM.md", label="Personality system file")
            store = ctx.personality_context_store(config)
            form = await _parse_bounded_form(request, max_form_bytes=max_form_bytes)
            action = form.get("_action", "save_base").strip() or "save_base"
            if action not in {"save_base", "upsert_managed"}:
                raise ValueError("Please choose a valid personality action.")
            # AUDIT-FIX(#7,#9): Serialize personality writes and require non-empty managed rule fields.
            async with state_write_lock:
                if action == "upsert_managed":
                    await _call_sync(
                        store.upsert,
                        category=_require_non_empty(form.get("category", ""), message="Please enter a short category."),
                        instruction=_require_non_empty(form.get("instruction", ""), message="Please enter a short instruction."),
                    )
                else:
                    await _call_sync(write_text_file, system_file, form.get("SYSTEM", ""))
                    await _call_sync(store.replace_base_text, form.get("PERSONALITY_BASE", ""))
        except Exception as exc:
            logger.exception("Twinr personality save failed", exc_info=exc)
            return _redirect_with_error(
                "/personality",
                _public_error_message(exc, fallback="Twinr could not save that personality change. Please try again."),
            )
        return _redirect_saved("/personality")

    @app.get("/user", response_class=HTMLResponse)
    async def user(request: Request) -> HTMLResponse:
        """Render the user profile context editor."""

        config, _env_values = await _call_sync(ctx.load_state)
        store = ctx.user_context_store(config)
        base_text = await _call_sync(store.load_base_text)
        managed_entries = await _call_sync(store.load_entries)
        return ctx.render(
            request,
            "context_page.html",
            page_title="User",
            active_page="user",
            intro="Compact user profile facts. These should remain factual and short so Twinr can use them quietly as context.",
            base_form_action="/user",
            managed_form_action="/user",
            raw_section_title="Base user profile",
            raw_section_description="Edit the hand-written stable profile here. Managed updates from explicit “remember this about me” requests stay separate below.",
            raw_fields=(
                FileBackedSetting(
                    key="USER_BASE",
                    label="USER.md base text",
                    value=base_text,
                    help_text="Short profile facts about the current Twinr user.",
                    input_type="textarea",
                ),
            ),
            managed_section_title="Managed user profile updates",
            managed_section_description="These entries were added by explicit user requests such as “remember that I have two dogs”.",
            managed_entries=managed_entries,
            managed_form_title="Add or update a managed user fact",
            managed_form_description="Use a short category so future edits replace the right fact instead of duplicating it.",
            managed_category_placeholder="pets",
            managed_category_help="Examples: pets, mobility, medication, preferences, family.",
            managed_instruction_placeholder="Thom has two dogs.",
            managed_instruction_help="Short factual profile entry.",
        )

    @app.post("/user")
    async def save_user(request: Request) -> RedirectResponse:
        """Persist user profile base text or managed facts."""

        try:
            config = await _call_sync(TwinrConfig.from_env, ctx.env_path)
            store = ctx.user_context_store(config)
            form = await _parse_bounded_form(request, max_form_bytes=max_form_bytes)
            action = form.get("_action", "save_base").strip() or "save_base"
            if action not in {"save_base", "upsert_managed"}:
                raise ValueError("Please choose a valid user action.")
            # AUDIT-FIX(#7,#9): Serialize user-profile writes and require non-empty managed fact fields.
            async with state_write_lock:
                if action == "upsert_managed":
                    await _call_sync(
                        store.upsert,
                        category=_require_non_empty(form.get("category", ""), message="Please enter a short category."),
                        instruction=_require_non_empty(form.get("instruction", ""), message="Please enter a short fact."),
                    )
                else:
                    await _call_sync(store.replace_base_text, form.get("USER_BASE", ""))
        except Exception as exc:
            logger.exception("Twinr user save failed", exc_info=exc)
            return _redirect_with_error(
                "/user",
                _public_error_message(exc, fallback="Twinr could not save that user profile change. Please try again."),
            )
        return _redirect_saved("/user")

    return app
