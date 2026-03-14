from __future__ import annotations

from pathlib import Path
from urllib.parse import quote_plus

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from twinr.agent.base_agent import AdaptiveTimingStore, TwinrConfig
from twinr.hardware.voice_profile import VoiceProfileMonitor
from twinr.integrations import build_managed_integrations, integration_automation_family_providers
from twinr.memory.reminders import format_due_label
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
from twinr.web.context import WebAppContext
from twinr.web.contracts import DashboardCard
from twinr.web.forms import _collect_standard_updates
from twinr.web.store import FileBackedSetting, parse_urlencoded_form, read_text_file, write_env_updates, write_text_file
from twinr.web.viewmodels import (
    _adaptive_timing_view,
    _build_calendar_integration_record,
    _build_email_integration_record,
    _calendar_integration_sections,
    _capture_voice_profile_sample,
    _connect_sections,
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
    _settings_sections,
    _voice_action_result,
    _voice_profile_page_context,
    _voice_snapshot_label,
)


def create_app(env_file: str | Path = ".env") -> FastAPI:
    env_path = Path(env_file).resolve()
    project_root = env_path.parent
    ops_paths = resolve_ops_paths(project_root)
    templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

    app = FastAPI(title="Twinr Control", version="0.1.0")
    app.mount(
        "/static",
        StaticFiles(directory=str(Path(__file__).resolve().parent / "static")),
        name="static",
    )

    ctx = WebAppContext(
        env_path=env_path,
        project_root=project_root,
        ops_paths=ops_paths,
        templates=templates,
    )

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request) -> HTMLResponse:
        config, env_values = ctx.load_state()
        snapshot = ctx.load_snapshot(config)
        reminders = ctx.reminder_store(config).load_entries()
        pending_reminders = tuple(entry for entry in reminders if not entry.delivered)
        delivered_reminders = tuple(entry for entry in reminders if entry.delivered)
        next_due_entry = pending_reminders[0] if pending_reminders else None
        ops_event_store = ctx.event_store()
        checks = run_config_checks(config)
        checks_summary = check_summary(checks)
        usage_summary = ctx.usage_store().summary(within_hours=24)
        health_snapshot = collect_system_health(config, snapshot=snapshot, event_store=ops_event_store)
        recent_errors = [
            entry
            for entry in ops_event_store.tail(limit=25)
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
        config, _env_values = ctx.load_state()
        return ctx.render(
            request,
            "ops_self_test.html",
            page_title="Hardware Self-Test",
            active_page="ops_self_test",
            restart_notice="These self-tests run against the local device and may access real hardware.",
            config=config,
            tests=TwinrSelfTestRunner.available_tests(),
            result=None,
            artifact_href=None,
        )

    @app.post("/ops/self-test", response_class=HTMLResponse)
    async def run_ops_self_test(request: Request) -> HTMLResponse:
        config, _env_values = ctx.load_state()
        form = parse_urlencoded_form(await request.body())
        result = TwinrSelfTestRunner(config).run(form.get("test_name", ""))
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
            tests=TwinrSelfTestRunner.available_tests(),
            result=result,
            artifact_href=artifact_href,
        )

    @app.get("/ops/self-test/artifacts/{artifact_name}")
    async def download_self_test_artifact(artifact_name: str) -> FileResponse:
        artifact_path = _resolve_named_file(ctx.ops_paths.self_tests_root, artifact_name)
        return FileResponse(artifact_path, filename=artifact_path.name)

    @app.get("/ops/logs", response_class=HTMLResponse)
    async def ops_logs(request: Request) -> HTMLResponse:
        return ctx.render(
            request,
            "ops_logs.html",
            page_title="Ops Logs",
            active_page="ops_logs",
            restart_notice="This view shows the latest 100 structured local events.",
            logs=_format_log_rows(ctx.event_store().tail(limit=100)),
        )

    @app.get("/ops/usage", response_class=HTMLResponse)
    async def ops_usage(request: Request) -> HTMLResponse:
        store = ctx.usage_store()
        return ctx.render(
            request,
            "ops_usage.html",
            page_title="LLM Usage",
            active_page="ops_usage",
            restart_notice="Usage records are written locally whenever Twinr completes a tracked OpenAI response call.",
            summary_all=store.summary(),
            summary_24h=store.summary(within_hours=24),
            usage_rows=_format_usage_rows(store.tail(limit=100)),
            usage_path=str(ctx.ops_paths.usage_path),
        )

    @app.get("/ops/health", response_class=HTMLResponse)
    async def ops_health(request: Request) -> HTMLResponse:
        config, _env_values = ctx.load_state()
        snapshot = ctx.load_snapshot(config)
        ops_event_store = ctx.event_store()
        recent_errors = [
            entry
            for entry in ops_event_store.tail(limit=25)
            if str(entry.get("level", "")).lower() == "error"
        ][-5:]
        return ctx.render(
            request,
            "ops_health.html",
            page_title="System Health",
            active_page="ops_health",
            restart_notice="This page reads live Raspberry Pi and Twinr process state from the local machine.",
            health=collect_system_health(config, snapshot=snapshot, event_store=ops_event_store),
            snapshot=snapshot,
            recent_errors=_format_log_rows(recent_errors),
        )

    @app.get("/ops/devices", response_class=HTMLResponse)
    async def ops_devices(request: Request) -> HTMLResponse:
        config, _env_values = ctx.load_state()
        return ctx.render(
            request,
            "ops_devices.html",
            page_title="Devices",
            active_page="ops_devices",
            restart_notice=(
                "This page shows only signals Twinr can confirm locally. "
                "Unknown means the current device path does not expose that signal."
            ),
            overview=collect_device_overview(config, event_store=ctx.event_store()),
        )

    @app.get("/ops/config", response_class=HTMLResponse)
    async def ops_config(request: Request) -> HTMLResponse:
        config, env_values = ctx.load_state()
        checks = run_config_checks(config)
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
        config, _env_values = ctx.load_state()
        return ctx.render(
            request,
            "ops_support.html",
            page_title="Support Bundle",
            active_page="ops_support",
            restart_notice="Bundles are written under artifacts/ops/support_bundles and contain only redacted environment data.",
            config=config,
            bundle=None,
            bundles=_recent_named_files(ctx.ops_paths.bundles_root, suffix=".zip"),
        )

    @app.post("/ops/support", response_class=HTMLResponse)
    async def create_support_bundle(request: Request) -> HTMLResponse:
        config, _env_values = ctx.load_state()
        bundle = build_support_bundle(config, env_path=ctx.env_path)
        return ctx.render(
            request,
            "ops_support.html",
            page_title="Support Bundle",
            active_page="ops_support",
            restart_notice="Bundles are written under artifacts/ops/support_bundles and contain only redacted environment data.",
            config=config,
            bundle=bundle,
            bundles=_recent_named_files(ctx.ops_paths.bundles_root, suffix=".zip"),
        )

    @app.get("/ops/support/download/{bundle_name}")
    async def download_support_bundle(bundle_name: str) -> FileResponse:
        bundle_path = _resolve_named_file(ctx.ops_paths.bundles_root, bundle_name)
        return FileResponse(bundle_path, filename=bundle_path.name)

    @app.get("/integrations", response_class=HTMLResponse)
    async def integrations(request: Request) -> HTMLResponse:
        _config, env_values = ctx.load_state()
        store = ctx.integration_store()
        email_record = store.get("email_mailbox")
        calendar_record = store.get("calendar_agenda")
        runtime = build_managed_integrations(ctx.project_root, env_path=ctx.env_path)
        return ctx.render(
            request,
            "integrations_page.html",
            page_title="Integrations",
            active_page="integrations",
            restart_notice=(
                "These settings stay local to this Twinr install. "
                "Email secrets remain only in .env, while non-secret integration config lives under artifacts/stores/integrations."
            ),
            intro="Set up safe external connections for Twinr. Calendar and mail are prepared first because they are useful and comparatively low-risk.",
            integration_store_path=str(store.path),
            overview_rows=_integration_overview_rows(runtime.readiness),
            email_sections=_email_integration_sections(email_record, env_values),
            calendar_sections=_calendar_integration_sections(calendar_record),
        )

    @app.post("/integrations")
    async def save_integrations(request: Request) -> RedirectResponse:
        _config, env_values = ctx.load_state()
        form = parse_urlencoded_form(await request.body())
        integration_id = form.get("_integration_id", "").strip()
        store = ctx.integration_store()

        try:
            if integration_id == "email_mailbox":
                record, env_updates = _build_email_integration_record(form, env_values)
            elif integration_id == "calendar_agenda":
                record, env_updates = _build_calendar_integration_record(form)
            else:
                raise ValueError("Unknown integration form submission.")
        except ValueError as exc:
            return RedirectResponse(f"/integrations?error={quote_plus(str(exc))}", status_code=303)

        store.save(record)
        if env_updates:
            write_env_updates(ctx.env_path, env_updates)
        return RedirectResponse("/integrations?saved=1", status_code=303)

    @app.get("/voice-profile", response_class=HTMLResponse)
    async def voice_profile_page(request: Request) -> HTMLResponse:
        config, _env_values = ctx.load_state()
        snapshot = ctx.load_snapshot(config)
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
    async def voice_profile_action(request: Request) -> HTMLResponse:
        config, _env_values = ctx.load_state()
        snapshot = ctx.load_snapshot(config)
        monitor = VoiceProfileMonitor.from_config(config)
        form = parse_urlencoded_form(await request.body())
        action = form.get("_action", "").strip()

        try:
            if action == "enroll":
                sample = _capture_voice_profile_sample(config)
                template = monitor.enroll_wav_bytes(sample)
                action_result = {
                    "status": "ok",
                    "title": "Profile updated",
                    "detail": (
                        f"Stored local template sample {template.sample_count}/{config.voice_profile_max_samples}. "
                        "No raw audio was kept."
                    ),
                }
            elif action == "verify":
                sample = _capture_voice_profile_sample(config)
                assessment = monitor.assess_wav_bytes(sample)
                action_result = _voice_action_result(assessment)
            elif action == "reset":
                monitor.reset()
                action_result = {
                    "status": "ok",
                    "title": "Profile reset",
                    "detail": "The local voice profile template was deleted.",
                }
            else:
                raise ValueError("Unknown voice profile action.")
        except Exception as exc:
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
                **_voice_profile_page_context(config, snapshot, action_error=str(exc)),
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
        config, _env_values = ctx.load_state()
        store = ctx.automation_store(config)
        integration_family_providers = integration_automation_family_providers(ctx.project_root)
        edit_ref = request.query_params.get("edit", "").strip() or None
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
            **build_automation_page_context(
                store,
                timezone_name=config.local_timezone_name,
                edit_ref=edit_ref,
                integration_blocks=tuple(provider.block() for provider in integration_family_providers),
            ),
        )

    @app.post("/automations")
    async def save_automations(request: Request) -> RedirectResponse:
        config = TwinrConfig.from_env(ctx.env_path)
        form = parse_urlencoded_form(await request.body())
        action = form.get("_action", "").strip()
        store = ctx.automation_store(config)
        integration_family_providers = integration_automation_family_providers(ctx.project_root)
        try:
            if action == "save_time_automation":
                save_time_automation(store, form, timezone_name=config.local_timezone_name)
            elif action == "save_sensor_automation":
                save_sensor_automation(store, form)
            elif action == "toggle_automation":
                automation_id = form.get("automation_id", "").strip()
                if not automation_id:
                    raise ValueError("Missing automation id.")
                toggle_automation_enabled(store, automation_id)
            elif action == "delete_automation":
                automation_id = form.get("automation_id", "").strip()
                if not automation_id:
                    raise ValueError("Missing automation id.")
                delete_automation(store, automation_id)
            elif any(provider.handles_action(action) for provider in integration_family_providers):
                handled = False
                for provider in integration_family_providers:
                    if not provider.handles_action(action):
                        continue
                    handled = provider.handle_action(action, form=form, automation_store=store)
                    if handled:
                        break
                if not handled:
                    raise ValueError("This integration automation family is not writable yet.")
            else:
                raise ValueError("Unknown automation form submission.")
        except ValueError as exc:
            return RedirectResponse(f"/automations?error={quote_plus(str(exc))}", status_code=303)
        return RedirectResponse("/automations?saved=1", status_code=303)

    @app.get("/connect", response_class=HTMLResponse)
    async def connect(request: Request) -> HTMLResponse:
        _config, env_values = ctx.load_state()
        sections = _connect_sections(env_values)
        return ctx.render(
            request,
            "form_page.html",
            page_title="Connect",
            active_page="connect",
            intro="Choose providers and manage credentials. Hover the small (?) labels for short explanations. Today only OpenAI is wired in the runtime; the other providers are stored for the next integration pass.",
            form_action="/connect",
            sections=sections,
        )

    @app.post("/connect")
    async def save_connect(request: Request) -> RedirectResponse:
        env_values = ctx.load_state()[1]
        form = parse_urlencoded_form(await request.body())
        updates = _collect_standard_updates(form, exclude={"OPENAI_API_KEY", "DEEPINFRA_API_KEY", "OPENROUTER_API_KEY"})
        for secret_key in ("OPENAI_API_KEY", "DEEPINFRA_API_KEY", "OPENROUTER_API_KEY"):
            secret_value = form.get(secret_key, "").strip()
            if secret_value:
                updates[secret_key] = secret_value
            elif secret_key not in env_values and secret_key == "OPENAI_API_KEY":
                updates.setdefault(secret_key, "")
        write_env_updates(ctx.env_path, updates)
        return RedirectResponse("/connect?saved=1", status_code=303)

    @app.get("/settings", response_class=HTMLResponse)
    async def settings(request: Request) -> HTMLResponse:
        config, env_values = ctx.load_state()
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
        config = TwinrConfig.from_env(ctx.env_path)
        form = parse_urlencoded_form(await request.body())
        action = form.get("_action", "save_settings")
        if action == "reset_adaptive_timing":
            AdaptiveTimingStore(config.adaptive_timing_store_path, config=config).reset()
            return RedirectResponse("/settings?saved=1", status_code=303)
        write_env_updates(ctx.env_path, _collect_standard_updates(form, exclude={"_action"}))
        return RedirectResponse("/settings?saved=1", status_code=303)

    @app.get("/memory", response_class=HTMLResponse)
    async def memory(request: Request) -> HTMLResponse:
        config, env_values = ctx.load_state()
        sections = _memory_sections(config, env_values)
        snapshot = ctx.load_snapshot(config)
        durable_store = ctx.memory_store(config)
        reminder_rows = _reminder_rows(ctx.reminder_store(config).load_entries(), timezone_name=config.local_timezone_name)
        return ctx.render(
            request,
            "memory_page.html",
            page_title="Memory",
            active_page="memory",
            intro="Tune on-device memory and print summarization bounds. Twinr also uses the configured long-term memory path for graph recall and background episodic persistence.",
            form_action="/memory",
            sections=sections,
            snapshot=snapshot,
            durable_memory_entries=durable_store.load_entries(),
            durable_memory_path=str(Path(config.memory_markdown_path)),
            reminder_entries=tuple(row for row in reminder_rows if not row["delivered"]),
            delivered_reminder_entries=tuple(row for row in reminder_rows if row["delivered"]),
            reminder_path=str(Path(config.reminder_store_path)),
            reminder_default_due_at=_default_reminder_due_at(config),
            timezone_name=config.local_timezone_name,
        )

    @app.post("/memory")
    async def save_memory(request: Request) -> RedirectResponse:
        config = TwinrConfig.from_env(ctx.env_path)
        form = parse_urlencoded_form(await request.body())
        action = form.get("_action", "save_settings")
        if action == "add_memory":
            ctx.memory_store(config).remember(
                kind=form.get("memory_kind", "") or "memory",
                summary=form.get("memory_summary", ""),
                details=form.get("memory_details", "") or None,
            )
            return RedirectResponse("/memory?saved=1", status_code=303)
        if action == "add_reminder":
            try:
                ctx.reminder_store(config).schedule(
                    due_at=form.get("reminder_due_at", ""),
                    summary=form.get("reminder_summary", ""),
                    details=form.get("reminder_details", "") or None,
                    kind=form.get("reminder_kind", "") or "reminder",
                    source="web_ui",
                    original_request=form.get("reminder_original_request", "") or None,
                )
            except ValueError as exc:
                return RedirectResponse(f"/memory?error={quote_plus(str(exc))}", status_code=303)
            return RedirectResponse("/memory?saved=1", status_code=303)
        if action == "mark_reminder_delivered":
            reminder_id = form.get("reminder_id", "").strip()
            if not reminder_id:
                return RedirectResponse("/memory?error=Missing+reminder+id", status_code=303)
            try:
                ctx.reminder_store(config).mark_delivered(reminder_id)
            except KeyError:
                return RedirectResponse("/memory?error=Reminder+not+found", status_code=303)
            return RedirectResponse("/memory?saved=1", status_code=303)
        if action == "delete_reminder":
            reminder_id = form.get("reminder_id", "").strip()
            if not reminder_id:
                return RedirectResponse("/memory?error=Missing+reminder+id", status_code=303)
            try:
                ctx.reminder_store(config).delete(reminder_id)
            except KeyError:
                return RedirectResponse("/memory?error=Reminder+not+found", status_code=303)
            return RedirectResponse("/memory?saved=1", status_code=303)
        write_env_updates(ctx.env_path, _collect_standard_updates(form, exclude={"_action"}))
        return RedirectResponse("/memory?saved=1", status_code=303)

    @app.get("/personality", response_class=HTMLResponse)
    async def personality(request: Request) -> HTMLResponse:
        config, _env_values = ctx.load_state()
        personality_dir = ctx.project_root / config.personality_dir
        store = ctx.personality_context_store(config)
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
                    value=read_text_file(personality_dir / "SYSTEM.md"),
                    help_text="Core product behavior and permanent operating rules.",
                    input_type="textarea",
                ),
                FileBackedSetting(
                    key="PERSONALITY_BASE",
                    label="PERSONALITY.md base text",
                    value=store.load_base_text(),
                    help_text="The stable hand-written part of the personality file. Managed tool updates are shown separately below.",
                    input_type="textarea",
                ),
            ),
            managed_section_title="Managed personality updates",
            managed_section_description="These entries were added by explicit user requests such as “speak more slowly” or “be less funny”.",
            managed_entries=store.load_entries(),
            managed_form_title="Add or update a managed personality rule",
            managed_form_description="Use a short category so future updates replace the right rule instead of creating duplicates.",
            managed_category_placeholder="response_style",
            managed_category_help="Examples: response_style, humor, pacing, confirmation_style.",
            managed_instruction_placeholder="Keep answers short, calm, and practical.",
            managed_instruction_help="Short, stable future behavior instruction.",
        )

    @app.post("/personality")
    async def save_personality(request: Request) -> RedirectResponse:
        config = TwinrConfig.from_env(ctx.env_path)
        personality_dir = ctx.project_root / config.personality_dir
        store = ctx.personality_context_store(config)
        form = parse_urlencoded_form(await request.body())
        action = form.get("_action", "save_base")
        if action == "upsert_managed":
            store.upsert(
                category=form.get("category", ""),
                instruction=form.get("instruction", ""),
            )
            return RedirectResponse("/personality?saved=1", status_code=303)
        write_text_file(personality_dir / "SYSTEM.md", form.get("SYSTEM", ""))
        store.replace_base_text(form.get("PERSONALITY_BASE", ""))
        return RedirectResponse("/personality?saved=1", status_code=303)

    @app.get("/user", response_class=HTMLResponse)
    async def user(request: Request) -> HTMLResponse:
        config, _env_values = ctx.load_state()
        store = ctx.user_context_store(config)
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
                    value=store.load_base_text(),
                    help_text="Short profile facts about the current Twinr user.",
                    input_type="textarea",
                ),
            ),
            managed_section_title="Managed user profile updates",
            managed_section_description="These entries were added by explicit user requests such as “remember that I have two dogs”.",
            managed_entries=store.load_entries(),
            managed_form_title="Add or update a managed user fact",
            managed_form_description="Use a short category so future edits replace the right fact instead of duplicating it.",
            managed_category_placeholder="pets",
            managed_category_help="Examples: pets, mobility, medication, preferences, family.",
            managed_instruction_placeholder="Thom has two dogs.",
            managed_instruction_help="Short factual profile entry.",
        )

    @app.post("/user")
    async def save_user(request: Request) -> RedirectResponse:
        config = TwinrConfig.from_env(ctx.env_path)
        store = ctx.user_context_store(config)
        form = parse_urlencoded_form(await request.body())
        action = form.get("_action", "save_base")
        if action == "upsert_managed":
            store.upsert(
                category=form.get("category", ""),
                instruction=form.get("instruction", ""),
            )
            return RedirectResponse("/user?saved=1", status_code=303)
        store.replace_base_text(form.get("USER_BASE", ""))
        return RedirectResponse("/user?saved=1", status_code=303)

    return app
