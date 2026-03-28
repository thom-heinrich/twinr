"""App-factory orchestration for the refactored Twinr web package."""

from __future__ import annotations

import asyncio
import os
from importlib import import_module
from pathlib import Path
from types import ModuleType

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from twinr.ops import resolve_ops_paths
from twinr.web.context import WebAppContext
from twinr.web.support.channel_onboarding import InProcessChannelPairingRegistry

from .auth import register_auth_routes
from .automation import register_automation_routes
from .compat import (
    _DEFAULT_MAX_FORM_BYTES,
    _MAX_FORM_BYTES_CAP,
    _env_bool,
    _env_int,
    _parse_allowed_hosts,
)
from .integrations import register_integrations_routes
from .ops import register_ops_routes
from .preferences import register_preference_routes
from .runtime import AppRuntime, LockSet, SecurityConfig, SurfaceProxy
from .shell import register_shell_routes


def create_app(
    env_file: str | Path = ".env",
    *,
    surface_module: ModuleType | None = None,
) -> FastAPI:
    """Create the FastAPI app for Twinr's local control surface."""

    env_path = Path(env_file).resolve()
    if env_path.is_dir():
        raise RuntimeError(f"Twinr Control expected an env file path, got directory: {env_path}")
    project_root = env_path.parent.resolve()
    ops_paths = resolve_ops_paths(project_root)

    web_root = Path(__file__).resolve().parents[1]
    templates_dir = web_root / "templates"
    static_dir = web_root / "static"
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
    auth_password_value = str(security_env_values.get("TWINR_WEB_PASSWORD", ""))
    static_auth_partially_configured = bool(auth_username or auth_password_value)
    if static_auth_partially_configured and not (auth_username and auth_password_value):
        raise RuntimeError("Twinr Control static web auth requires both TWINR_WEB_USERNAME and TWINR_WEB_PASSWORD.")
    managed_auth_enabled = bool(require_auth and not static_auth_partially_configured)
    managed_auth_store = ctx.web_auth_store() if managed_auth_enabled else None
    if allow_remote and not require_auth:
        raise RuntimeError("Twinr Control remote access requires web sign-in. Enable TWINR_WEB_REQUIRE_AUTH first.")

    surface = SurfaceProxy(surface_module or import_module("twinr.web.app"))
    locks = LockSet(
        state_write_lock=asyncio.Lock(),
        ops_job_lock=asyncio.Lock(),
        conversation_lab_lock=asyncio.Lock(),
        voice_profile_lock=asyncio.Lock(),
        managed_auth_write_lock=asyncio.Lock(),
    )
    channel_pairing_registry = InProcessChannelPairingRegistry()
    whatsapp_pairing = surface.WhatsAppPairingCoordinator(
        store=ctx.channel_onboarding_store("whatsapp"),
        registry=channel_pairing_registry,
    )
    runtime = AppRuntime(
        ctx=ctx,
        surface=surface,
        max_form_bytes=max_form_bytes,
        security=SecurityConfig(
            allowed_hosts=allowed_hosts,
            allow_remote=allow_remote,
            require_auth=require_auth,
            auth_username=auth_username,
            auth_password_value=auth_password_value,
            managed_auth_enabled=managed_auth_enabled,
            managed_auth_store=managed_auth_store,
        ),
        locks=locks,
        channel_pairing_registry=channel_pairing_registry,
        whatsapp_pairing=whatsapp_pairing,
    )

    register_auth_routes(app, runtime)
    register_shell_routes(app, runtime)
    register_ops_routes(app, runtime)
    register_integrations_routes(app, runtime)
    register_automation_routes(app, runtime)
    register_preference_routes(app, runtime)
    return app
