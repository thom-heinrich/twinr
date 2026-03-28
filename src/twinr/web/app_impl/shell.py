"""Home and operator-hub routes for the Twinr web control surface."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from twinr.agent.self_coding.operator_status import build_self_coding_operator_status
from twinr.memory.reminders import format_due_label
from twinr.ops import check_summary, collect_system_health, run_config_checks
from twinr.web.presenters import (
    _format_log_rows,
    _voice_snapshot_label,
    build_advanced_hub_page_context,
    build_home_destination_cards,
)

from .compat import _call_sync, _reminder_sort_key
from .runtime import AppRuntime


def register_shell_routes(app: FastAPI, runtime: AppRuntime) -> None:
    """Register the top-level dashboard and advanced hub pages."""

    ctx = runtime.ctx

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request) -> HTMLResponse:
        """Render the dashboard overview page."""

        config, _env_values = await _call_sync(ctx.load_state)
        snapshot = await _call_sync(ctx.load_snapshot, config)
        reminders = await _call_sync(ctx.reminder_store(config).load_entries)
        pending_reminders = tuple(
            sorted(
                (entry for entry in reminders if not entry.delivered),
                key=_reminder_sort_key,
            )
        )
        delivered_reminders = tuple(entry for entry in reminders if entry.delivered)
        next_due_entry = pending_reminders[0] if pending_reminders else None
        ops_event_store = ctx.event_store()
        checks = await _call_sync(run_config_checks, config)
        checks_summary = check_summary(checks)
        usage_store = ctx.usage_store()
        usage_summary = await _call_sync(usage_store.summary, within_hours=24)
        recent_event_rows = await _call_sync(ops_event_store.tail, limit=25)
        health_snapshot = await _call_sync(
            collect_system_health,
            config,
            snapshot=snapshot,
            event_store=ops_event_store,
        )
        self_coding_status = await _call_sync(
            build_self_coding_operator_status,
            ctx.self_coding_store(),
        )
        recent_errors = [
            entry
            for entry in recent_event_rows
            if str(entry.get("level", "")).lower() == "error"
        ][-3:]
        next_reminder_label = (
            f"Next due {format_due_label(next_due_entry.due_at, timezone_name=config.local_timezone_name)}"
            if next_due_entry is not None
            else None
        )
        cards = build_home_destination_cards(
            snapshot=snapshot,
            pending_reminders_count=len(pending_reminders),
            delivered_reminders_count=len(delivered_reminders),
            next_reminder_label=next_reminder_label,
            health_snapshot=health_snapshot,
            checks_summary=checks_summary,
        )
        return ctx.render(
            request,
            "dashboard.html",
            page_title="Home",
            active_page="dashboard",
            config=config,
            cards=cards,
            snapshot=snapshot,
            check_summary=checks_summary,
            recent_errors=_format_log_rows(recent_errors),
            usage_summary=usage_summary,
            health_snapshot=health_snapshot,
            voice_snapshot_label=_voice_snapshot_label(snapshot),
            next_reminder_label=next_reminder_label,
            self_coding_status=self_coding_status,
        )

    @app.get("/advanced", response_class=HTMLResponse)
    async def advanced(request: Request) -> HTMLResponse:
        """Render the grouped operator hub page."""

        config, _env_values = await _call_sync(ctx.load_state)
        snapshot = await _call_sync(ctx.load_snapshot, config)
        ops_event_store = ctx.event_store()
        checks = await _call_sync(run_config_checks, config)
        checks_summary = check_summary(checks)
        usage_summary = await _call_sync(ctx.usage_store().summary, within_hours=24)
        health_snapshot = await _call_sync(
            collect_system_health,
            config,
            snapshot=snapshot,
            event_store=ops_event_store,
        )
        self_coding_status = await _call_sync(
            build_self_coding_operator_status,
            ctx.self_coding_store(),
        )
        page_context = build_advanced_hub_page_context(
            checks_summary=checks_summary,
            usage_summary=usage_summary,
            health_snapshot=health_snapshot,
            self_coding_status=self_coding_status,
        )
        return ctx.render(
            request,
            "advanced_page.html",
            page_title="Advanced",
            active_page="advanced",
            intro="Technical checks, operator tools, and support pages live here so the main shell stays simpler.",
            **page_context,
        )
