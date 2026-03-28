"""Automation routes for the refactored Twinr web control surface."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from twinr.agent.base_agent.config import TwinrConfig
from twinr.integrations import integration_automation_family_providers
from twinr.web.automations import (
    build_automation_page_context,
    delete_automation,
    save_sensor_automation,
    save_time_automation,
    toggle_automation_enabled,
)

from .compat import (
    _call_sync,
    _parse_bounded_form,
    _public_error_message,
    _redirect_saved,
    _redirect_with_error,
    _require_non_empty,
    logger,
)
from .runtime import AppRuntime


def register_automation_routes(app: FastAPI, runtime: AppRuntime) -> None:
    """Register the `/automations` page and mutation routes."""

    ctx = runtime.ctx

    @app.get("/automations", response_class=HTMLResponse)
    async def automations(request: Request) -> HTMLResponse:
        """Render the automations page."""

        config, _env_values = await _call_sync(ctx.load_state)
        store = ctx.automation_store(config)
        integration_family_providers = tuple(
            await _call_sync(
                integration_automation_family_providers,
                ctx.project_root,
            )
        )
        edit_ref = request.query_params.get("edit", "").strip() or None
        page_context = await _call_sync(
            build_automation_page_context,
            store,
            timezone_name=config.local_timezone_name,
            edit_ref=edit_ref,
            integration_blocks=tuple(
                provider.block() for provider in integration_family_providers
            ),
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
            form = await _parse_bounded_form(
                request,
                max_form_bytes=runtime.max_form_bytes,
            )
            action = _require_non_empty(
                form.get("_action", ""),
                message="Please choose an automation action.",
            )
            store = ctx.automation_store(config)
            integration_family_providers = tuple(
                await _call_sync(
                    integration_automation_family_providers,
                    ctx.project_root,
                )
            )

            async with runtime.locks.state_write_lock:
                if action == "save_time_automation":
                    await _call_sync(
                        save_time_automation,
                        store,
                        form,
                        timezone_name=config.local_timezone_name,
                    )
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
                elif any(
                    provider.handles_action(action)
                    for provider in integration_family_providers
                ):
                    handled = False
                    for provider in integration_family_providers:
                        if not provider.handles_action(action):
                            continue
                        handled = await _call_sync(
                            provider.handle_action,
                            action,
                            form=form,
                            automation_store=store,
                        )
                        if handled:
                            break
                    if not handled:
                        raise ValueError(
                            "That integration automation cannot be changed yet."
                        )
                else:
                    raise ValueError("Please choose a valid automation action.")
        except Exception as exc:
            logger.exception("Twinr automation save failed", exc_info=exc)
            return _redirect_with_error(
                "/automations",
                _public_error_message(
                    exc,
                    fallback="Twinr could not save that automation. Please check the fields and try again.",
                ),
            )
        return _redirect_saved("/automations")
