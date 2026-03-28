"""Auth, middleware, and exception wiring for the Twinr web control plane."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from twinr.web.support.auth import (
    default_web_auth_username,
    load_authenticated_web_session,
    verify_web_auth_password,
    web_auth_password_min_length,
    web_auth_session_cookie_name,
)

from .compat import (
    _apply_auth_context,
    _auth_login_location,
    _call_sync,
    _clear_managed_auth_cookie,
    _error_response,
    _has_trusted_same_origin,
    _has_valid_basic_auth,
    _is_allowed_host,
    _is_loopback_host,
    _normalize_host_header,
    _parse_bounded_form,
    _public_error_message,
    _safe_next_path,
    _secure_response,
    _set_managed_auth_cookie,
    logger,
)
from .runtime import AppRuntime


def register_auth_routes(app: FastAPI, runtime: AppRuntime) -> None:
    """Register control-plane guards plus the managed auth pages."""

    ctx = runtime.ctx
    security = runtime.security

    @app.middleware("http")
    async def guard_control_plane(request: Request, call_next: Any) -> Response:
        """Enforce host, remote-access, auth, and same-origin policies."""

        request_path = request.url.path.rstrip("/") or "/"
        public_auth_path = request_path.startswith("/static/") or request_path == "/auth/login"
        normalized_host = _normalize_host_header(request.headers.get("host", ""))
        if not _is_allowed_host(normalized_host, security.allowed_hosts):
            return _error_response(
                request,
                status_code=status.HTTP_400_BAD_REQUEST,
                message="Twinr Control rejected this host name. Check TWINR_WEB_ALLOWED_HOSTS.",
            )

        client_host = request.client.host if request.client else ""
        client_is_loopback = _is_loopback_host(client_host)
        managed_auth_state = None
        managed_session_username: str | None = None
        if security.managed_auth_enabled:
            assert security.managed_auth_store is not None
            try:
                managed_auth_state = await _call_sync(
                    security.managed_auth_store.load_or_bootstrap
                )
            except Exception as exc:
                logger.exception(
                    "Failed to load managed web auth state",
                    exc_info=exc,
                )
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

        if not security.allow_remote and not client_is_loopback:
            return _error_response(
                request,
                status_code=status.HTTP_403_FORBIDDEN,
                message=(
                    "Twinr Control only accepts browser access from this device right now. "
                    "To allow remote access, set TWINR_WEB_ALLOW_REMOTE=1, TWINR_WEB_ALLOWED_HOSTS, and web sign-in."
                ),
            )
        if (
            security.require_auth
            and not security.managed_auth_enabled
            and not _has_valid_basic_auth(
                request,
                security.auth_username,
                security.auth_password_value,
            )
        ):
            response = _error_response(
                request,
                status_code=status.HTTP_401_UNAUTHORIZED,
                message="Sign-in is required for Twinr Control.",
            )
            response.headers["WWW-Authenticate"] = (
                'Basic realm="Twinr Control", charset="UTF-8"'
            )
            return response
        if security.managed_auth_enabled and not public_auth_path:
            if not managed_session_username:
                return _secure_response(
                    RedirectResponse(
                        _auth_login_location(request),
                        status_code=status.HTTP_303_SEE_OTHER,
                    )
                )
            if (
                managed_auth_state is not None
                and managed_auth_state.must_change_password
                and request_path not in {"/auth/password", "/auth/logout"}
            ):
                return _secure_response(
                    RedirectResponse(
                        "/auth/password",
                        status_code=status.HTTP_303_SEE_OTHER,
                    )
                )

        if (
            request.method in {"POST", "PUT", "PATCH", "DELETE"}
            and not _has_trusted_same_origin(request)
        ):
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
            message=_public_error_message(
                exc,
                fallback="Twinr Control could not finish that request.",
            ),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_exception(request: Request, exc: Exception) -> Response:
        """Log unexpected failures and return a plain-language 500 page."""

        logger.exception(
            "Unhandled Twinr Control error on %s %s",
            request.method,
            request.url.path,
            exc_info=exc,
        )
        return _error_response(
            request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="Twinr Control hit a local problem and could not finish that page. Please try again.",
        )

    @app.get("/auth/login", response_class=HTMLResponse)
    async def auth_login(request: Request) -> Response:
        """Render the managed-login page used by the permanent Pi web service."""

        if not security.managed_auth_enabled or security.managed_auth_store is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="This page is not available.",
            )

        auth_state = await _call_sync(security.managed_auth_store.load_or_bootstrap)
        authenticated_username = load_authenticated_web_session(
            auth_state,
            request.cookies.get(web_auth_session_cookie_name()),
        )
        if authenticated_username:
            if auth_state.must_change_password:
                return RedirectResponse(
                    "/auth/password",
                    status_code=status.HTTP_303_SEE_OTHER,
                )
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

        if not security.managed_auth_enabled or security.managed_auth_store is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="This page is not available.",
            )

        auth_state = await _call_sync(security.managed_auth_store.load_or_bootstrap)
        form_values = await _parse_bounded_form(
            request,
            max_form_bytes=runtime.max_form_bytes,
        )
        username = str(form_values.get("username", "") or "").strip()
        password = str(form_values.get("password", "") or "")
        next_path = _safe_next_path(form_values.get("next"))
        if not verify_web_auth_password(
            auth_state,
            username=username,
            password=password,
        ):
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
        _set_managed_auth_cookie(
            response,
            state=auth_state,
            username=auth_state.username,
            request=request,
        )
        return response

    @app.get("/auth/password", response_class=HTMLResponse)
    async def auth_password(request: Request) -> Response:
        """Render the password-change page for managed web sign-in."""

        if not security.managed_auth_enabled or security.managed_auth_store is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="This page is not available.",
            )

        auth_state = await _call_sync(security.managed_auth_store.load_or_bootstrap)
        authenticated_username = load_authenticated_web_session(
            auth_state,
            request.cookies.get(web_auth_session_cookie_name()),
        )
        if not authenticated_username:
            return RedirectResponse(
                _auth_login_location(request),
                status_code=status.HTTP_303_SEE_OTHER,
            )
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

        if not security.managed_auth_enabled or security.managed_auth_store is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="This page is not available.",
            )

        auth_state = await _call_sync(security.managed_auth_store.load_or_bootstrap)
        authenticated_username = load_authenticated_web_session(
            auth_state,
            request.cookies.get(web_auth_session_cookie_name()),
        )
        if not authenticated_username:
            return RedirectResponse(
                _auth_login_location(request),
                status_code=status.HTTP_303_SEE_OTHER,
            )

        form_values = await _parse_bounded_form(
            request,
            max_form_bytes=runtime.max_form_bytes,
        )
        current_password = str(form_values.get("current_password", "") or "")
        new_password = str(form_values.get("new_password", "") or "")
        confirm_password = str(form_values.get("confirm_password", "") or "")
        try:
            async with runtime.locks.managed_auth_write_lock:
                updated_auth_state = await _call_sync(
                    security.managed_auth_store.update_password,
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
                error_message=_public_error_message(
                    exc,
                    fallback="Twinr could not save the new password.",
                ),
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

        del request
        if not security.managed_auth_enabled:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="This page is not available.",
            )
        response = RedirectResponse(
            "/auth/login",
            status_code=status.HTTP_303_SEE_OTHER,
        )
        _clear_managed_auth_cookie(response)
        return response
