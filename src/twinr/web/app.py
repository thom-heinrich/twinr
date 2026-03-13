from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
from urllib.parse import quote_plus

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from twinr.agent.base_agent import RuntimeSnapshot, RuntimeSnapshotStore, TwinrConfig
from twinr.automations import AutomationStore
from twinr.hardware.audio import SilenceDetectedRecorder
from twinr.hardware.voice_profile import VoiceAssessment, VoiceProfileMonitor
from twinr.integrations import (
    EMAIL_APP_PASSWORD_ENV_KEY,
    ManagedIntegrationConfig,
    TwinrIntegrationStore,
    build_managed_integrations,
    integration_automation_family_providers,
    validate_calendar_source,
)
from twinr.memory.context_store import ManagedContextFileStore, PersistentMemoryMarkdownStore
from twinr.memory.reminders import ReminderStore, format_due_label, now_in_timezone
from twinr.ops import (
    TwinrOpsEventStore,
    TwinrSelfTestRunner,
    TwinrUsageStore,
    build_support_bundle,
    check_summary,
    collect_device_overview,
    collect_system_health,
    redact_env_values,
    loop_lock_owner,
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
from twinr.web.store import (
    FileBackedSetting,
    mask_secret,
    parse_urlencoded_form,
    read_env_values,
    read_text_file,
    write_env_updates,
    write_text_file,
)

_DEFAULT_RESTART_NOTICE = "Changes to providers, models, or network settings need a Twinr process restart."
_PROVIDER_OPTIONS = (
    ("openai", "OpenAI"),
    ("deepinfra", "DeepInfra"),
    ("openrouter", "OpenRouter"),
)
_BOOL_OPTIONS = (("true", "Enabled"), ("false", "Disabled"))
_TRISTATE_BOOL_OPTIONS = (("", "Auto"), ("true", "Always send"), ("false", "Never send"))
_YES_NO_OPTIONS = (("true", "Yes"), ("false", "No"))
_REASONING_EFFORT_OPTIONS = (("low", "Low"), ("medium", "Medium"), ("high", "High"))
_SEARCH_CONTEXT_OPTIONS = (("low", "Low"), ("medium", "Medium"), ("high", "High"))
_CONVERSATION_WEB_SEARCH_OPTIONS = (("auto", "Auto"), ("always", "Always"), ("never", "Never"))
_VISION_DETAIL_OPTIONS = (("auto", "Auto"), ("low", "Low"), ("high", "High"))
_GPIO_BIAS_OPTIONS = (("pull-up", "Pull-up"), ("pull-down", "Pull-down"), ("disabled", "Off"))
_EMAIL_PROFILE_OPTIONS = (("gmail", "Gmail"), ("generic_imap_smtp", "Generic IMAP/SMTP"))
_CALENDAR_SOURCE_OPTIONS = (("ics_file", "ICS file"), ("ics_url", "ICS URL"))
_EMAIL_SECRET_KEY = EMAIL_APP_PASSWORD_ENV_KEY


@dataclass(frozen=True, slots=True)
class DashboardCard:
    title: str
    value: str
    detail: str
    href: str


@dataclass(frozen=True, slots=True)
class SettingsSection:
    title: str
    description: str
    fields: tuple[FileBackedSetting, ...]


@dataclass(frozen=True, slots=True)
class IntegrationOverviewRow:
    label: str
    status: str
    summary: str
    detail: str


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

    def load_state() -> tuple[TwinrConfig, dict[str, str]]:
        env_values = read_env_values(env_path)
        return TwinrConfig.from_env(env_path), env_values

    def load_snapshot(config: TwinrConfig) -> RuntimeSnapshot:
        return RuntimeSnapshotStore(config.runtime_state_path).load()

    def memory_store(config: TwinrConfig) -> PersistentMemoryMarkdownStore:
        return PersistentMemoryMarkdownStore(config.memory_markdown_path)

    def reminder_store(config: TwinrConfig) -> ReminderStore:
        return ReminderStore(
            config.reminder_store_path,
            timezone_name=config.local_timezone_name,
            retry_delay_s=config.reminder_retry_delay_s,
            max_entries=config.reminder_max_entries,
        )

    def automation_store(config: TwinrConfig) -> AutomationStore:
        return AutomationStore(
            config.automation_store_path,
            timezone_name=config.local_timezone_name,
            max_entries=config.automation_max_entries,
        )

    def user_context_store(config: TwinrConfig) -> ManagedContextFileStore:
        return ManagedContextFileStore(
            project_root / config.personality_dir / "USER.md",
            section_title="Twinr managed user updates",
        )

    def personality_context_store(config: TwinrConfig) -> ManagedContextFileStore:
        return ManagedContextFileStore(
            project_root / config.personality_dir / "PERSONALITY.md",
            section_title="Twinr managed personality updates",
        )

    def event_store() -> TwinrOpsEventStore:
        return TwinrOpsEventStore.from_project_root(project_root)

    def usage_store() -> TwinrUsageStore:
        return TwinrUsageStore.from_project_root(project_root)

    def integration_store() -> TwinrIntegrationStore:
        return TwinrIntegrationStore.from_project_root(project_root)

    def render(
        request: Request,
        template_name: str,
        *,
        page_title: str,
        active_page: str,
        restart_notice: str | None = _DEFAULT_RESTART_NOTICE,
        **context: Any,
    ) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            template_name,
            {
                "request": request,
                "page_title": page_title,
                "active_page": active_page,
                "nav_items": _nav_items(),
                "saved": request.query_params.get("saved") == "1",
                "error_message": request.query_params.get("error"),
                "restart_notice": restart_notice,
                "env_path": str(env_path),
                **context,
            },
        )

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request) -> HTMLResponse:
        config, env_values = load_state()
        snapshot = load_snapshot(config)
        reminders = reminder_store(config).load_entries()
        pending_reminders = tuple(entry for entry in reminders if not entry.delivered)
        delivered_reminders = tuple(entry for entry in reminders if entry.delivered)
        next_due_entry = pending_reminders[0] if pending_reminders else None
        ops_event_store = event_store()
        checks = run_config_checks(config)
        checks_summary = check_summary(checks)
        usage_summary = usage_store().summary(within_hours=24)
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
        return render(
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
        config, _env_values = load_state()
        return render(
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
        config, _env_values = load_state()
        form = parse_urlencoded_form(await request.body())
        result = TwinrSelfTestRunner(config).run(form.get("test_name", ""))
        artifact_href = None
        if result.artifact_name:
            artifact_href = f"/ops/self-test/artifacts/{result.artifact_name}"
        return render(
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
        artifact_path = _resolve_named_file(ops_paths.self_tests_root, artifact_name)
        return FileResponse(artifact_path, filename=artifact_path.name)

    @app.get("/ops/logs", response_class=HTMLResponse)
    async def ops_logs(request: Request) -> HTMLResponse:
        return render(
            request,
            "ops_logs.html",
            page_title="Ops Logs",
            active_page="ops_logs",
            restart_notice="This view shows the latest 100 structured local events.",
            logs=_format_log_rows(event_store().tail(limit=100)),
        )

    @app.get("/ops/usage", response_class=HTMLResponse)
    async def ops_usage(request: Request) -> HTMLResponse:
        store = usage_store()
        return render(
            request,
            "ops_usage.html",
            page_title="LLM Usage",
            active_page="ops_usage",
            restart_notice="Usage records are written locally whenever Twinr completes a tracked OpenAI response call.",
            summary_all=store.summary(),
            summary_24h=store.summary(within_hours=24),
            usage_rows=_format_usage_rows(store.tail(limit=100)),
            usage_path=str(ops_paths.usage_path),
        )

    @app.get("/ops/health", response_class=HTMLResponse)
    async def ops_health(request: Request) -> HTMLResponse:
        config, _env_values = load_state()
        snapshot = load_snapshot(config)
        ops_event_store = event_store()
        recent_errors = [
            entry
            for entry in ops_event_store.tail(limit=25)
            if str(entry.get("level", "")).lower() == "error"
        ][-5:]
        return render(
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
        config, _env_values = load_state()
        return render(
            request,
            "ops_devices.html",
            page_title="Devices",
            active_page="ops_devices",
            restart_notice=(
                "This page shows only signals Twinr can confirm locally. "
                "Unknown means the current device path does not expose that signal."
            ),
            overview=collect_device_overview(config, event_store=event_store()),
        )

    @app.get("/ops/config", response_class=HTMLResponse)
    async def ops_config(request: Request) -> HTMLResponse:
        config, env_values = load_state()
        checks = run_config_checks(config)
        return render(
            request,
            "ops_config.html",
            page_title="Config Checks",
            active_page="ops_config",
            restart_notice="This page checks plausibility, not full end-to-end hardware behavior.",
            config=config,
            checks=checks,
            summary=check_summary(checks),
            artifacts_root=str(ops_paths.artifacts_root),
            redacted_env=redact_env_values(env_values),
        )

    @app.get("/ops/support", response_class=HTMLResponse)
    async def ops_support(request: Request) -> HTMLResponse:
        config, _env_values = load_state()
        return render(
            request,
            "ops_support.html",
            page_title="Support Bundle",
            active_page="ops_support",
            restart_notice="Bundles are written under artifacts/ops/support_bundles and contain only redacted environment data.",
            config=config,
            bundle=None,
            bundles=_recent_named_files(ops_paths.bundles_root, suffix=".zip"),
        )

    @app.post("/ops/support", response_class=HTMLResponse)
    async def create_support_bundle(request: Request) -> HTMLResponse:
        config, _env_values = load_state()
        bundle = build_support_bundle(config, env_path=env_path)
        return render(
            request,
            "ops_support.html",
            page_title="Support Bundle",
            active_page="ops_support",
            restart_notice="Bundles are written under artifacts/ops/support_bundles and contain only redacted environment data.",
            config=config,
            bundle=bundle,
            bundles=_recent_named_files(ops_paths.bundles_root, suffix=".zip"),
        )

    @app.get("/ops/support/download/{bundle_name}")
    async def download_support_bundle(bundle_name: str) -> FileResponse:
        bundle_path = _resolve_named_file(ops_paths.bundles_root, bundle_name)
        return FileResponse(bundle_path, filename=bundle_path.name)

    @app.get("/integrations", response_class=HTMLResponse)
    async def integrations(request: Request) -> HTMLResponse:
        _config, env_values = load_state()
        store = integration_store()
        email_record = store.get("email_mailbox")
        calendar_record = store.get("calendar_agenda")
        runtime = build_managed_integrations(project_root, env_path=env_path)
        return render(
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
        _config, env_values = load_state()
        form = parse_urlencoded_form(await request.body())
        integration_id = form.get("_integration_id", "").strip()
        store = integration_store()

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
            write_env_updates(env_path, env_updates)
        return RedirectResponse("/integrations?saved=1", status_code=303)

    @app.get("/voice-profile", response_class=HTMLResponse)
    async def voice_profile_page(request: Request) -> HTMLResponse:
        config, _env_values = load_state()
        snapshot = load_snapshot(config)
        return render(
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
        config, _env_values = load_state()
        snapshot = load_snapshot(config)
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
            return render(
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

        return render(
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
        config, _env_values = load_state()
        store = automation_store(config)
        integration_family_providers = integration_automation_family_providers(project_root)
        edit_ref = request.query_params.get("edit", "").strip() or None
        return render(
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
        config = TwinrConfig.from_env(env_path)
        form = parse_urlencoded_form(await request.body())
        action = form.get("_action", "").strip()
        store = automation_store(config)
        integration_family_providers = integration_automation_family_providers(project_root)
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
        _config, env_values = load_state()
        sections = _connect_sections(env_values)
        return render(
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
        env_values = read_env_values(env_path)
        form = parse_urlencoded_form(await request.body())
        updates = _collect_standard_updates(form, exclude={"OPENAI_API_KEY", "DEEPINFRA_API_KEY", "OPENROUTER_API_KEY"})
        for secret_key in ("OPENAI_API_KEY", "DEEPINFRA_API_KEY", "OPENROUTER_API_KEY"):
            secret_value = form.get(secret_key, "").strip()
            if secret_value:
                updates[secret_key] = secret_value
            elif secret_key not in env_values and secret_key == "OPENAI_API_KEY":
                updates.setdefault(secret_key, "")
        write_env_updates(env_path, updates)
        return RedirectResponse("/connect?saved=1", status_code=303)

    @app.get("/settings", response_class=HTMLResponse)
    async def settings(request: Request) -> HTMLResponse:
        config, env_values = load_state()
        sections = _settings_sections(config, env_values)
        return render(
            request,
            "form_page.html",
            page_title="Settings",
            active_page="settings",
            intro="All main device and runtime tuning lives here. Hover the small (?) labels for a quick explanation of each field.",
            form_action="/settings",
            sections=sections,
        )

    @app.post("/settings")
    async def save_settings(request: Request) -> RedirectResponse:
        form = parse_urlencoded_form(await request.body())
        write_env_updates(env_path, _collect_standard_updates(form))
        return RedirectResponse("/settings?saved=1", status_code=303)

    @app.get("/memory", response_class=HTMLResponse)
    async def memory(request: Request) -> HTMLResponse:
        config, env_values = load_state()
        sections = _memory_sections(config, env_values)
        snapshot = load_snapshot(config)
        durable_store = memory_store(config)
        reminder_rows = _reminder_rows(reminder_store(config).load_entries(), timezone_name=config.local_timezone_name)
        return render(
            request,
            "memory_page.html",
            page_title="Memory",
            active_page="memory",
            intro="Tune on-device memory and print summarization bounds. Long-term memory is not wired yet, so this page focuses on the active local memory path.",
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
        config = TwinrConfig.from_env(env_path)
        form = parse_urlencoded_form(await request.body())
        action = form.get("_action", "save_settings")
        if action == "add_memory":
            memory_store(config).remember(
                kind=form.get("memory_kind", "") or "memory",
                summary=form.get("memory_summary", ""),
                details=form.get("memory_details", "") or None,
            )
            return RedirectResponse("/memory?saved=1", status_code=303)
        if action == "add_reminder":
            try:
                reminder_store(config).schedule(
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
                reminder_store(config).mark_delivered(reminder_id)
            except KeyError:
                return RedirectResponse("/memory?error=Reminder+not+found", status_code=303)
            return RedirectResponse("/memory?saved=1", status_code=303)
        if action == "delete_reminder":
            reminder_id = form.get("reminder_id", "").strip()
            if not reminder_id:
                return RedirectResponse("/memory?error=Missing+reminder+id", status_code=303)
            try:
                reminder_store(config).delete(reminder_id)
            except KeyError:
                return RedirectResponse("/memory?error=Reminder+not+found", status_code=303)
            return RedirectResponse("/memory?saved=1", status_code=303)
        write_env_updates(env_path, _collect_standard_updates(form, exclude={"_action"}))
        return RedirectResponse("/memory?saved=1", status_code=303)

    @app.get("/personality", response_class=HTMLResponse)
    async def personality(request: Request) -> HTMLResponse:
        config, _env_values = load_state()
        personality_dir = project_root / config.personality_dir
        store = personality_context_store(config)
        return render(
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
        config = TwinrConfig.from_env(env_path)
        personality_dir = project_root / config.personality_dir
        store = personality_context_store(config)
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
        config, _env_values = load_state()
        store = user_context_store(config)
        return render(
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
        config = TwinrConfig.from_env(env_path)
        store = user_context_store(config)
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


def _nav_items() -> tuple[tuple[str, str, str], ...]:
    return (
        ("dashboard", "Dashboard", "/"),
        ("ops_self_test", "Self-Test", "/ops/self-test"),
        ("ops_devices", "Devices", "/ops/devices"),
        ("ops_usage", "LLM Usage", "/ops/usage"),
        ("ops_health", "System Health", "/ops/health"),
        ("ops_logs", "Ops Logs", "/ops/logs"),
        ("ops_config", "Config Checks", "/ops/config"),
        ("ops_support", "Support", "/ops/support"),
        ("integrations", "Integrations", "/integrations"),
        ("voice_profile", "Voice Profile", "/voice-profile"),
        ("automations", "Automations", "/automations"),
        ("personality", "Personality", "/personality"),
        ("memory", "Memory", "/memory"),
        ("connect", "Connect", "/connect"),
        ("settings", "Settings", "/settings"),
        ("user", "User", "/user"),
    )


def _provider_status(env_values: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return (
        ("OpenAI key", mask_secret(env_values.get("OPENAI_API_KEY"))),
        ("OpenAI project", env_values.get("OPENAI_PROJ_ID", "Not configured") or "Not configured"),
        ("DeepInfra key", mask_secret(env_values.get("DEEPINFRA_API_KEY"))),
        ("OpenRouter key", mask_secret(env_values.get("OPENROUTER_API_KEY"))),
    )


def _default_reminder_due_at(config: TwinrConfig) -> str:
    return now_in_timezone(config.local_timezone_name).replace(second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M")


def _reminder_rows(entries: tuple[Any, ...], *, timezone_name: str) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        status_key = "delivered" if entry.delivered else ("retry" if entry.last_error else "pending")
        if status_key == "delivered":
            status_label = "Delivered"
        elif status_key == "retry":
            status_label = "Retrying"
        else:
            status_label = "Pending"
        rows.append(
            {
                "reminder_id": entry.reminder_id,
                "kind": entry.kind,
                "summary": entry.summary,
                "details": entry.details,
                "source": entry.source,
                "original_request": entry.original_request,
                "delivery_attempts": entry.delivery_attempts,
                "last_error": entry.last_error,
                "created_at": entry.created_at.isoformat(),
                "updated_at": entry.updated_at.isoformat(),
                "due_label": format_due_label(entry.due_at, timezone_name=timezone_name),
                "next_attempt_label": (
                    format_due_label(entry.next_attempt_at, timezone_name=timezone_name)
                    if entry.next_attempt_at is not None
                    else None
                ),
                "delivered_at_label": (
                    format_due_label(entry.delivered_at, timezone_name=timezone_name)
                    if entry.delivered_at is not None
                    else None
                ),
                "delivered": entry.delivered,
                "status_key": status_key,
                "status_label": status_label,
            }
        )
    return tuple(rows)


def _integration_overview_rows(
    readiness_items,
) -> tuple[IntegrationOverviewRow, ...]:
    return tuple(
        IntegrationOverviewRow(
            label=item.label,
            status=item.status,
            summary=item.summary,
            detail=item.detail,
        )
        for item in readiness_items
    )


def _email_integration_sections(
    record: ManagedIntegrationConfig,
    env_values: dict[str, str],
) -> tuple[SettingsSection, ...]:
    values = dict(record.settings)
    profile = values.get("profile", "gmail")
    account_email = values.get("account_email", "")
    gmail_default = profile == "gmail"
    return (
        SettingsSection(
            title="Email",
            description="Set up a mailbox connection for reading mail summaries and preparing approved replies.",
            fields=(
                _select_field(
                    "enabled",
                    "Email enabled",
                    values,
                    _BOOL_OPTIONS,
                    "true" if record.enabled else "false",
                    tooltip_text="Enable only after the account and password are configured.",
                ),
                _select_field(
                    "profile",
                    "Profile",
                    values,
                    _EMAIL_PROFILE_OPTIONS,
                    profile,
                    tooltip_text="Gmail pre-fills the standard IMAP/SMTP defaults. Generic keeps everything manual.",
                ),
                _text_field(
                    "account_email",
                    "Account email",
                    values,
                    account_email,
                    placeholder="name@gmail.com",
                    tooltip_text="The mailbox address Twinr will read from and usually also send from.",
                ),
                _text_field(
                    "from_address",
                    "From address",
                    values,
                    values.get("from_address", account_email),
                    placeholder="name@gmail.com",
                    tooltip_text="Outgoing sender address. Leave it equal to the account unless you know you need a different sender.",
                ),
                FileBackedSetting(
                    key=_EMAIL_SECRET_KEY,
                    label="App password",
                    value="",
                    help_text=(
                        f"Credential state: {_credential_state_label(env_values.get(_EMAIL_SECRET_KEY))}. "
                        "Leave blank to keep it unchanged."
                    ),
                    tooltip_text="For Gmail use a Google app password, not the normal Google account password.",
                    input_type="password",
                    placeholder="16-character app password",
                    secret=True,
                ),
            ),
        ),
        SettingsSection(
            title="Mailbox transport",
            description="These values are stored locally for the future live adapter wiring. Gmail works with the defaults shown here.",
            fields=(
                _text_field(
                    "imap_host",
                    "IMAP host",
                    values,
                    values.get("imap_host", "imap.gmail.com" if gmail_default else ""),
                    placeholder="imap.gmail.com",
                    tooltip_text="Incoming mail server hostname.",
                ),
                _text_field(
                    "imap_port",
                    "IMAP port",
                    values,
                    values.get("imap_port", "993" if gmail_default else ""),
                    tooltip_text="Incoming mail server port. Gmail uses 993.",
                ),
                _text_field(
                    "imap_mailbox",
                    "Mailbox",
                    values,
                    values.get("imap_mailbox", "INBOX"),
                    tooltip_text="Mailbox folder Twinr should read from.",
                ),
                _text_field(
                    "smtp_host",
                    "SMTP host",
                    values,
                    values.get("smtp_host", "smtp.gmail.com" if gmail_default else ""),
                    placeholder="smtp.gmail.com",
                    tooltip_text="Outgoing mail server hostname.",
                ),
                _text_field(
                    "smtp_port",
                    "SMTP port",
                    values,
                    values.get("smtp_port", "587" if gmail_default else ""),
                    tooltip_text="Outgoing mail server port. Gmail uses 587 with STARTTLS.",
                ),
            ),
        ),
        SettingsSection(
            title="Guardrails",
            description="Reads are open by default. Drafts and sends still need explicit approval in Twinr. The two strict toggles below are optional extra fences.",
            fields=(
                _select_field(
                    "unread_only_default",
                    "Read unread only",
                    values,
                    _BOOL_OPTIONS,
                    values.get("unread_only_default", "true"),
                    tooltip_text="When enabled, Twinr prefers unread mail for summaries by default.",
                ),
                _select_field(
                    "restrict_reads_to_known_senders",
                    "Restrict reads to known senders",
                    values,
                    _BOOL_OPTIONS,
                    values.get("restrict_reads_to_known_senders", "false"),
                    tooltip_text="Optional extra fence. Leave off if Twinr should summarize any mailbox sender.",
                ),
                _select_field(
                    "restrict_recipients_to_known_contacts",
                    "Restrict send to known contacts",
                    values,
                    _BOOL_OPTIONS,
                    values.get("restrict_recipients_to_known_contacts", "false"),
                    tooltip_text="Optional extra fence. Leave off if explicit approval should be enough for sending.",
                ),
                _textarea_field(
                    "known_contacts_text",
                    "Known contacts",
                    values,
                    values.get("known_contacts_text", ""),
                    placeholder="Anna <anna@example.com>\nDoctor <doctor@example.com>",
                    tooltip_text="Optional future contact hints. One contact per line.",
                    rows=5,
                ),
            ),
        ),
    )


def _calendar_integration_sections(record: ManagedIntegrationConfig) -> tuple[SettingsSection, ...]:
    values = dict(record.settings)
    return (
        SettingsSection(
            title="Calendar",
            description="Read-only agenda setup for day plans, appointment summaries, and later reminder synchronization.",
            fields=(
                _select_field(
                    "enabled",
                    "Calendar enabled",
                    values,
                    _BOOL_OPTIONS,
                    "true" if record.enabled else "false",
                    tooltip_text="Enable only when a local ICS file or feed URL is configured.",
                ),
                _select_field(
                    "source_kind",
                    "Source type",
                    values,
                    _CALENDAR_SOURCE_OPTIONS,
                    values.get("source_kind", "ics_file"),
                    tooltip_text="Phase 1 uses a simple ICS file or ICS feed only.",
                ),
                _text_field(
                    "source_value",
                    "ICS path or URL",
                    values,
                    values.get("source_value", ""),
                    placeholder="state/calendar.ics or https://...",
                    tooltip_text="Relative file paths are resolved from the Twinr project root.",
                ),
                _text_field(
                    "timezone",
                    "Timezone",
                    values,
                    values.get("timezone", "Europe/Berlin"),
                    placeholder="Europe/Berlin",
                    tooltip_text="Used for all-day events and local agenda rendering.",
                ),
                _text_field(
                    "default_upcoming_days",
                    "Upcoming days",
                    values,
                    values.get("default_upcoming_days", "7"),
                    tooltip_text="Default look-ahead window for upcoming agenda summaries.",
                ),
                _text_field(
                    "max_events",
                    "Max events",
                    values,
                    values.get("max_events", "12"),
                    tooltip_text="Upper bound for one agenda readout so the device stays short and readable.",
                ),
            ),
        ),
    )


def _build_email_integration_record(
    form: dict[str, str],
    env_values: dict[str, str],
) -> tuple[ManagedIntegrationConfig, dict[str, str]]:
    enabled = form.get("enabled", "false") == "true"
    profile = (form.get("profile", "gmail") or "gmail").strip()
    account_email = form.get("account_email", "").strip()
    from_address = form.get("from_address", "").strip() or account_email
    imap_host = form.get("imap_host", "").strip() or ("imap.gmail.com" if profile == "gmail" else "")
    imap_port = form.get("imap_port", "").strip() or ("993" if profile == "gmail" else "")
    imap_mailbox = form.get("imap_mailbox", "").strip() or "INBOX"
    smtp_host = form.get("smtp_host", "").strip() or ("smtp.gmail.com" if profile == "gmail" else "")
    smtp_port = form.get("smtp_port", "").strip() or ("587" if profile == "gmail" else "")
    if enabled and not account_email:
        raise ValueError("Email account address is required when email is enabled.")
    if enabled and not (form.get(_EMAIL_SECRET_KEY, "").strip() or env_values.get(_EMAIL_SECRET_KEY)):
        raise ValueError("Email app password is required when email is enabled.")

    env_updates: dict[str, str] = {}
    secret_value = form.get(_EMAIL_SECRET_KEY, "").strip()
    if secret_value:
        env_updates[_EMAIL_SECRET_KEY] = secret_value

    record = ManagedIntegrationConfig(
        integration_id="email_mailbox",
        enabled=enabled,
        settings={
            "profile": profile,
            "account_email": account_email,
            "from_address": from_address,
            "imap_host": imap_host,
            "imap_port": imap_port,
            "imap_mailbox": imap_mailbox,
            "smtp_host": smtp_host,
            "smtp_port": smtp_port,
            "unread_only_default": "true" if form.get("unread_only_default", "true") == "true" else "false",
            "restrict_reads_to_known_senders": (
                "true" if form.get("restrict_reads_to_known_senders", "false") == "true" else "false"
            ),
            "restrict_recipients_to_known_contacts": (
                "true" if form.get("restrict_recipients_to_known_contacts", "false") == "true" else "false"
            ),
            "known_contacts_text": form.get("known_contacts_text", "").strip(),
        },
    )
    return record, env_updates


def _build_calendar_integration_record(form: dict[str, str]) -> tuple[ManagedIntegrationConfig, dict[str, str]]:
    enabled = form.get("enabled", "false") == "true"
    source_kind = (form.get("source_kind", "ics_file") or "ics_file").strip()
    source_value = form.get("source_value", "").strip()
    if enabled and not source_value:
        raise ValueError("Calendar source path or URL is required when calendar is enabled.")
    if enabled:
        validate_calendar_source(source_kind=source_kind, source_value=source_value)

    record = ManagedIntegrationConfig(
        integration_id="calendar_agenda",
        enabled=enabled,
        settings={
            "source_kind": source_kind,
            "source_value": source_value,
            "timezone": form.get("timezone", "").strip() or "Europe/Berlin",
            "default_upcoming_days": form.get("default_upcoming_days", "").strip() or "7",
            "max_events": form.get("max_events", "").strip() or "12",
        },
    )
    return record, {}


def _connect_sections(env_values: dict[str, str]) -> tuple[SettingsSection, ...]:
    return (
        SettingsSection(
            title="Provider routing",
            description="Choose which backend handles each pipeline stage.",
            fields=(
                _select_field(
                    "TWINR_PROVIDER_LLM",
                    "LLM provider",
                    env_values,
                    _PROVIDER_OPTIONS,
                    "openai",
                    tooltip_text="Controls which backend answers normal text questions.",
                ),
                _select_field(
                    "TWINR_PROVIDER_STT",
                    "STT provider",
                    env_values,
                    _PROVIDER_OPTIONS,
                    "openai",
                    tooltip_text="Controls which backend turns speech into text.",
                ),
                _select_field(
                    "TWINR_PROVIDER_TTS",
                    "TTS provider",
                    env_values,
                    _PROVIDER_OPTIONS,
                    "openai",
                    tooltip_text="Controls which backend turns replies into spoken audio.",
                ),
                _select_field(
                    "TWINR_PROVIDER_REALTIME",
                    "Realtime provider",
                    env_values,
                    _PROVIDER_OPTIONS,
                    "openai",
                    tooltip_text="Controls the low-latency voice session backend.",
                ),
            ),
        ),
        SettingsSection(
            title="OpenAI",
            description="Main account and auth settings for the currently active runtime.",
            fields=(
                FileBackedSetting(
                    key="OPENAI_API_KEY",
                    label="API key",
                    value="",
                    help_text=f"Current value: {mask_secret(env_values.get('OPENAI_API_KEY'))}. Leave blank to keep it unchanged.",
                    tooltip_text="The main OpenAI secret used for chat, speech, vision, and realtime requests.",
                    input_type="password",
                    placeholder="sk-...",
                    secret=True,
                ),
                _text_field(
                    "OPENAI_PROJ_ID",
                    "Project ID",
                    env_values,
                    "",
                    placeholder="proj_...",
                    tooltip_text="Optional project id. Only set this when your account setup requires an explicit OpenAI project.",
                ),
                _select_field(
                    "OPENAI_SEND_PROJECT_HEADER",
                    "Project header",
                    env_values,
                    _TRISTATE_BOOL_OPTIONS,
                    "",
                    help_text="Use auto unless you explicitly need to force the header on or off.",
                    tooltip_text="Auto is usually correct. Force this only if your OpenAI key/project setup needs it.",
                ),
            ),
        ),
        SettingsSection(
            title="Other providers",
            description="Stored here now so later provider adapters can use them without editing files by hand.",
            fields=(
                FileBackedSetting(
                    key="DEEPINFRA_API_KEY",
                    label="DeepInfra API key",
                    value="",
                    help_text=f"Current value: {mask_secret(env_values.get('DEEPINFRA_API_KEY'))}. Leave blank to keep it unchanged.",
                    tooltip_text="Credential for a future DeepInfra provider integration.",
                    input_type="password",
                    placeholder="DeepInfra key",
                    secret=True,
                ),
                FileBackedSetting(
                    key="OPENROUTER_API_KEY",
                    label="OpenRouter API key",
                    value="",
                    help_text=f"Current value: {mask_secret(env_values.get('OPENROUTER_API_KEY'))}. Leave blank to keep it unchanged.",
                    tooltip_text="Credential for a future OpenRouter provider integration.",
                    input_type="password",
                    placeholder="OpenRouter key",
                    secret=True,
                ),
            ),
        ),
    )


def _credential_state_label(value: str | None) -> str:
    return "Configured" if value else "Not configured"


def _settings_sections(config: TwinrConfig, env_values: dict[str, str]) -> tuple[SettingsSection, ...]:
    green_button_gpio = "" if config.green_button_gpio is None else str(config.green_button_gpio)
    yellow_button_gpio = "" if config.yellow_button_gpio is None else str(config.yellow_button_gpio)
    pir_motion_gpio = "" if config.pir_motion_gpio is None else str(config.pir_motion_gpio)
    button_probe_lines = ",".join(str(line) for line in config.button_probe_lines)
    return (
        SettingsSection(
            title="Models and voices",
            description="Main OpenAI model choices for chat, speech, and realtime audio.",
            fields=(
                _text_field(
                    "OPENAI_MODEL",
                    "LLM model",
                    env_values,
                    config.default_model,
                    tooltip_text="Main chat and reasoning model for the standard Twinr flow.",
                ),
                _select_field(
                    "OPENAI_REASONING_EFFORT",
                    "Reasoning effort",
                    env_values,
                    _REASONING_EFFORT_OPTIONS,
                    config.openai_reasoning_effort,
                    tooltip_text="Higher effort can improve harder answers but usually costs more time and tokens.",
                ),
                _text_field(
                    "OPENAI_STT_MODEL",
                    "STT model",
                    env_values,
                    config.openai_stt_model,
                    tooltip_text="Model used when Twinr converts recorded speech into text.",
                ),
                _text_field(
                    "OPENAI_TTS_MODEL",
                    "TTS model",
                    env_values,
                    config.openai_tts_model,
                    tooltip_text="Model used to synthesize spoken replies.",
                ),
                _text_field(
                    "OPENAI_TTS_VOICE",
                    "TTS voice",
                    env_values,
                    config.openai_tts_voice,
                    tooltip_text="Voice name used for normal spoken replies.",
                ),
                _text_field(
                    "OPENAI_TTS_FORMAT",
                    "TTS format",
                    env_values,
                    config.openai_tts_format,
                    tooltip_text="Audio format for generated speech, for example wav or mp3.",
                ),
                _textarea_field(
                    "OPENAI_TTS_INSTRUCTIONS",
                    "TTS instructions",
                    env_values,
                    config.openai_tts_instructions or "",
                    placeholder="Speak in clear, warm, natural standard German...",
                    tooltip_text="Optional speaking instructions sent with text-to-speech requests.",
                    rows=4,
                ),
                _text_field(
                    "OPENAI_REALTIME_MODEL",
                    "Realtime model",
                    env_values,
                    config.openai_realtime_model,
                    tooltip_text="Model used by the low-latency live voice session.",
                ),
                _text_field(
                    "OPENAI_REALTIME_VOICE",
                    "Realtime voice",
                    env_values,
                    config.openai_realtime_voice,
                    tooltip_text="Voice name used inside the realtime session.",
                ),
                _text_field(
                    "OPENAI_REALTIME_TRANSCRIPTION_MODEL",
                    "Realtime transcription model",
                    env_values,
                    config.openai_realtime_transcription_model,
                    tooltip_text="Speech-to-text model used inside the realtime session.",
                ),
                _text_field(
                    "OPENAI_REALTIME_LANGUAGE",
                    "Realtime language",
                    env_values,
                    config.openai_realtime_language or "",
                    placeholder="de",
                    tooltip_text="Hint for the main spoken language during realtime audio.",
                ),
                _text_field(
                    "OPENAI_REALTIME_INPUT_SAMPLE_RATE",
                    "Realtime input sample rate",
                    env_values,
                    str(config.openai_realtime_input_sample_rate),
                    tooltip_text="Sample rate sent to the realtime backend after local resampling.",
                ),
                _textarea_field(
                    "OPENAI_REALTIME_INSTRUCTIONS",
                    "Realtime instructions",
                    env_values,
                    config.openai_realtime_instructions or "",
                    placeholder="Speak in clear, warm, natural standard German...",
                    tooltip_text="Optional system-style instructions used only in realtime mode.",
                    rows=4,
                ),
            ),
        ),
        SettingsSection(
            title="Search",
            description="Controls when Twinr is allowed to browse and how broad that search context should be.",
            fields=(
                _select_field(
                    "TWINR_OPENAI_ENABLE_WEB_SEARCH",
                    "OpenAI web search",
                    env_values,
                    _BOOL_OPTIONS,
                    "true" if config.openai_enable_web_search else "false",
                    tooltip_text="Master switch for OpenAI-backed web search in the standard conversation loop.",
                ),
                _select_field(
                    "TWINR_CONVERSATION_WEB_SEARCH",
                    "Conversation web search",
                    env_values,
                    _CONVERSATION_WEB_SEARCH_OPTIONS,
                    config.conversation_web_search,
                    tooltip_text="Auto searches only freshness-sensitive questions. Always forces search. Never disables it for normal chats.",
                ),
                _text_field(
                    "OPENAI_SEARCH_MODEL",
                    "Search model",
                    env_values,
                    config.openai_search_model,
                    tooltip_text="Model used when Twinr performs a web-backed response.",
                ),
                _select_field(
                    "TWINR_OPENAI_WEB_SEARCH_CONTEXT_SIZE",
                    "Search context size",
                    env_values,
                    _SEARCH_CONTEXT_OPTIONS,
                    config.openai_web_search_context_size,
                    tooltip_text="How much retrieved web context to include with the answer.",
                ),
                _text_field(
                    "TWINR_OPENAI_WEB_SEARCH_COUNTRY",
                    "Search country",
                    env_values,
                    config.openai_web_search_country or "",
                    placeholder="DE",
                    tooltip_text="Optional country hint for localized search results.",
                ),
                _text_field(
                    "TWINR_OPENAI_WEB_SEARCH_REGION",
                    "Search region",
                    env_values,
                    config.openai_web_search_region or "",
                    placeholder="HH",
                    tooltip_text="Optional region hint for localized search results.",
                ),
                _text_field(
                    "TWINR_OPENAI_WEB_SEARCH_CITY",
                    "Search city",
                    env_values,
                    config.openai_web_search_city or "",
                    placeholder="Hamburg",
                    tooltip_text="Optional city hint for localized search results.",
                ),
                _text_field(
                    "TWINR_OPENAI_WEB_SEARCH_TIMEZONE",
                    "Search timezone",
                    env_values,
                    config.openai_web_search_timezone or "",
                    placeholder="Europe/Berlin",
                    tooltip_text="Timezone hint used when time-sensitive search answers need local context.",
                ),
            ),
        ),
        SettingsSection(
            title="Conversation flow",
            description="Short-turn timing around button conversations and automatic follow-up listening.",
            fields=(
                _text_field(
                    "TWINR_SPEECH_PAUSE_MS",
                    "Speech pause (ms)",
                    env_values,
                    str(config.speech_pause_ms),
                    tooltip_text="How long Twinr waits after silence before it stops recording a turn.",
                ),
                _select_field(
                    "TWINR_CONVERSATION_FOLLOW_UP_ENABLED",
                    "Follow-up listening",
                    env_values,
                    _BOOL_OPTIONS,
                    "true" if config.conversation_follow_up_enabled else "false",
                    tooltip_text="If enabled, Twinr briefly listens again after each spoken reply.",
                ),
                _text_field(
                    "TWINR_CONVERSATION_FOLLOW_UP_TIMEOUT_S",
                    "Follow-up timeout (s)",
                    env_values,
                    str(config.conversation_follow_up_timeout_s),
                    tooltip_text="Length of the extra listening window after Twinr has answered.",
                ),
                _text_field(
                    "TWINR_AUDIO_FOLLOW_UP_SPEECH_START_CHUNKS",
                    "Follow-up speech start chunks",
                    env_values,
                    str(config.audio_follow_up_speech_start_chunks),
                    tooltip_text="How many active audio chunks are needed before follow-up recording starts.",
                ),
                _text_field(
                    "TWINR_AUDIO_FOLLOW_UP_IGNORE_MS",
                    "Follow-up ignore (ms)",
                    env_values,
                    str(config.audio_follow_up_ignore_ms),
                    tooltip_text="Short ignore window right after playback so Twinr does not immediately hear itself.",
                ),
            ),
        ),
        SettingsSection(
            title="Audio capture",
            description="Microphone device selection and thresholds for recorded speech turns.",
            fields=(
                _text_field(
                    "TWINR_AUDIO_INPUT_DEVICE",
                    "Input device",
                    env_values,
                    config.audio_input_device,
                    tooltip_text="ALSA input device used for normal button-based recordings.",
                ),
                _text_field(
                    "TWINR_AUDIO_OUTPUT_DEVICE",
                    "Output device",
                    env_values,
                    config.audio_output_device,
                    tooltip_text="ALSA output device used for Twinr speech playback.",
                ),
                _text_field(
                    "TWINR_AUDIO_SAMPLE_RATE",
                    "Sample rate",
                    env_values,
                    str(config.audio_sample_rate),
                    tooltip_text="Capture sample rate for the normal audio recording path.",
                ),
                _text_field(
                    "TWINR_AUDIO_CHANNELS",
                    "Channels",
                    env_values,
                    str(config.audio_channels),
                    tooltip_text="Number of microphone channels used for normal recording.",
                ),
                _text_field(
                    "TWINR_AUDIO_CHUNK_MS",
                    "Chunk size (ms)",
                    env_values,
                    str(config.audio_chunk_ms),
                    tooltip_text="Length of each captured audio chunk while listening.",
                ),
                _text_field(
                    "TWINR_AUDIO_PREROLL_MS",
                    "Preroll (ms)",
                    env_values,
                    str(config.audio_preroll_ms),
                    tooltip_text="Keeps a small amount of audio from just before speech starts.",
                ),
                _text_field(
                    "TWINR_AUDIO_SPEECH_THRESHOLD",
                    "Speech threshold",
                    env_values,
                    str(config.audio_speech_threshold),
                    tooltip_text="Volume threshold used to decide when speech has started.",
                ),
                _text_field(
                    "TWINR_AUDIO_SPEECH_START_CHUNKS",
                    "Speech start chunks",
                    env_values,
                    str(config.audio_speech_start_chunks),
                    tooltip_text="How many active chunks are needed before a new recording officially starts.",
                ),
                _text_field(
                    "TWINR_AUDIO_START_TIMEOUT_S",
                    "Start timeout (s)",
                    env_values,
                    str(config.audio_start_timeout_s),
                    tooltip_text="How long Twinr waits for speech before abandoning a listen attempt.",
                ),
                _text_field(
                    "TWINR_AUDIO_MAX_RECORD_SECONDS",
                    "Max record seconds",
                    env_values,
                    str(config.audio_max_record_seconds),
                    tooltip_text="Upper limit for one captured turn so recordings do not run forever.",
                ),
            ),
        ),
        SettingsSection(
            title="Audio feedback",
            description="Confirmation beeps and optional search progress tones.",
            fields=(
                _text_field(
                    "TWINR_AUDIO_BEEP_FREQUENCY_HZ",
                    "Beep frequency (Hz)",
                    env_values,
                    str(config.audio_beep_frequency_hz),
                    tooltip_text="Pitch of the short confirmation beep.",
                ),
                _text_field(
                    "TWINR_AUDIO_BEEP_DURATION_MS",
                    "Beep duration (ms)",
                    env_values,
                    str(config.audio_beep_duration_ms),
                    tooltip_text="Length of the normal confirmation beep.",
                ),
                _text_field(
                    "TWINR_AUDIO_BEEP_VOLUME",
                    "Beep volume",
                    env_values,
                    str(config.audio_beep_volume),
                    tooltip_text="Volume multiplier for local beeps.",
                ),
                _text_field(
                    "TWINR_AUDIO_BEEP_SETTLE_MS",
                    "Beep settle (ms)",
                    env_values,
                    str(config.audio_beep_settle_ms),
                    tooltip_text="Small pause after a beep so the next step does not clip.",
                ),
                _select_field(
                    "TWINR_SEARCH_FEEDBACK_TONES_ENABLED",
                    "Search feedback tones",
                    env_values,
                    _BOOL_OPTIONS,
                    "true" if config.search_feedback_tones_enabled else "false",
                    tooltip_text="If enabled, Twinr plays quiet tones while a web search answer is still in progress.",
                ),
                _text_field(
                    "TWINR_SEARCH_FEEDBACK_DELAY_MS",
                    "Search tone delay (ms)",
                    env_values,
                    str(config.search_feedback_delay_ms),
                    tooltip_text="How long Twinr waits before starting search progress tones.",
                ),
                _text_field(
                    "TWINR_SEARCH_FEEDBACK_PAUSE_MS",
                    "Search tone pause (ms)",
                    env_values,
                    str(config.search_feedback_pause_ms),
                    tooltip_text="Pause between search progress tone bursts.",
                ),
                _text_field(
                    "TWINR_SEARCH_FEEDBACK_VOLUME",
                    "Search tone volume",
                    env_values,
                    str(config.search_feedback_volume),
                    tooltip_text="Volume multiplier for the quieter search progress tones.",
                ),
            ),
        ),
        SettingsSection(
            title="Camera and vision",
            description="Camera capture and image understanding settings.",
            fields=(
                _text_field(
                    "TWINR_CAMERA_DEVICE",
                    "Camera device",
                    env_values,
                    config.camera_device,
                    tooltip_text="Video device Twinr uses for still captures and proactive observation.",
                ),
                _text_field(
                    "TWINR_CAMERA_WIDTH",
                    "Camera width",
                    env_values,
                    str(config.camera_width),
                    tooltip_text="Capture width for still images.",
                ),
                _text_field(
                    "TWINR_CAMERA_HEIGHT",
                    "Camera height",
                    env_values,
                    str(config.camera_height),
                    tooltip_text="Capture height for still images.",
                ),
                _text_field(
                    "TWINR_CAMERA_FRAMERATE",
                    "Camera framerate",
                    env_values,
                    str(config.camera_framerate),
                    tooltip_text="Requested frame rate used when ffmpeg grabs from the camera.",
                ),
                _text_field(
                    "TWINR_CAMERA_INPUT_FORMAT",
                    "Camera input format",
                    env_values,
                    config.camera_input_format or "",
                    placeholder="bayer_grbg8",
                    tooltip_text="Optional raw camera pixel format. Leave blank unless the camera needs a specific format.",
                ),
                _text_field(
                    "TWINR_CAMERA_FFMPEG_PATH",
                    "ffmpeg path",
                    env_values,
                    config.camera_ffmpeg_path,
                    tooltip_text="Command path for ffmpeg, which Twinr uses to capture still images.",
                ),
                _select_field(
                    "OPENAI_VISION_DETAIL",
                    "Vision detail",
                    env_values,
                    _VISION_DETAIL_OPTIONS,
                    config.openai_vision_detail,
                    tooltip_text="How much image detail Twinr asks OpenAI to inspect.",
                ),
                _text_field(
                    "TWINR_VISION_REFERENCE_IMAGE",
                    "Reference image path",
                    env_values,
                    config.vision_reference_image_path or "",
                    placeholder="/home/thh/reference-user.jpg",
                    tooltip_text="Optional stored portrait that Twinr can send alongside a live camera frame.",
                ),
            ),
        ),
        SettingsSection(
            title="Proactive behavior",
            description="Bounded idle-time prompts based on PIR, camera, and optional background audio.",
            fields=(
                _text_field(
                    "TWINR_USER_DISPLAY_NAME",
                    "Display name",
                    env_values,
                    config.user_display_name or "",
                    placeholder="Thom",
                    tooltip_text="Name Twinr may use in gentle proactive prompts.",
                ),
                _select_field(
                    "TWINR_PROACTIVE_ENABLED",
                    "Proactive mode",
                    env_values,
                    _BOOL_OPTIONS,
                    "true" if config.proactive_enabled else "false",
                    tooltip_text="Turns the proactive observation service on while Twinr is idle.",
                ),
                _select_field(
                    "TWINR_PROACTIVE_AUDIO_ENABLED",
                    "Background audio",
                    env_values,
                    _BOOL_OPTIONS,
                    "true" if config.proactive_audio_enabled else "false",
                    tooltip_text="Lets the proactive watcher sample ambient audio while Twinr is waiting.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_AUDIO_DEVICE",
                    "Background audio device",
                    env_values,
                    config.proactive_audio_input_device or "",
                    placeholder="plughw:CARD=CameraB409241,DEV=0",
                    tooltip_text="Optional dedicated ALSA input for the background watcher, for example the PS-Eye microphone.",
                ),
                _select_field(
                    "TWINR_PROACTIVE_AUDIO_DISTRESS_ENABLED",
                    "Distress heuristic",
                    env_values,
                    _BOOL_OPTIONS,
                    "true" if config.proactive_audio_distress_enabled else "false",
                    tooltip_text="Experimental ambient audio heuristic for stronger distress-like sounds.",
                ),
            ),
        ),
        SettingsSection(
            title="Proactive timing",
            description="Idle timing, capture spacing, and per-trigger hold durations. Lower values react faster; higher values reduce chatter.",
            fields=(
                _text_field(
                    "TWINR_PROACTIVE_POLL_INTERVAL_S",
                    "Wake pause (s)",
                    env_values,
                    str(config.proactive_poll_interval_s),
                    tooltip_text="How long Twinr sleeps between proactive monitor wakeups while idle.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_CAPTURE_INTERVAL_S",
                    "Camera pause (s)",
                    env_values,
                    str(config.proactive_capture_interval_s),
                    tooltip_text="Minimum pause between proactive camera inspections.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_MOTION_WINDOW_S",
                    "Motion window (s)",
                    env_values,
                    str(config.proactive_motion_window_s),
                    tooltip_text="How long recent PIR motion keeps proactive inspection active before Twinr goes quiet again.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_LOW_MOTION_AFTER_S",
                    "Idle after (s)",
                    env_values,
                    str(config.proactive_low_motion_after_s),
                    tooltip_text="After this many quiet seconds without motion, the scene is treated as idle / low-motion.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_AUDIO_SAMPLE_MS",
                    "Background sample (ms)",
                    env_values,
                    str(config.proactive_audio_sample_ms),
                    tooltip_text="Length of each ambient audio sample window.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_PERSON_RETURNED_ABSENCE_S",
                    "Return absence (s)",
                    env_values,
                    str(config.proactive_person_returned_absence_s),
                    tooltip_text="How long someone must be absent before Twinr may greet them as returned.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_PERSON_RETURNED_RECENT_MOTION_S",
                    "Return motion recency (s)",
                    env_values,
                    str(config.proactive_person_returned_recent_motion_s),
                    tooltip_text="How recent PIR motion must be to count as a real return instead of stale presence.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_ATTENTION_WINDOW_S",
                    "Attention hold (s)",
                    env_values,
                    str(config.proactive_attention_window_s),
                    tooltip_text="How long someone must look toward Twinr while staying quiet before it offers help.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_SLUMPED_QUIET_S",
                    "Slumped quiet hold (s)",
                    env_values,
                    str(config.proactive_slumped_quiet_s),
                    tooltip_text="How long a slumped, quiet, low-motion posture must persist before Twinr checks in.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_POSSIBLE_FALL_STILLNESS_S",
                    "Fall stillness hold (s)",
                    env_values,
                    str(config.proactive_possible_fall_stillness_s),
                    tooltip_text="How long the post-fall floor state must stay still before Twinr asks about help.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_FLOOR_STILLNESS_S",
                    "Floor stillness hold (s)",
                    env_values,
                    str(config.proactive_floor_stillness_s),
                    tooltip_text="How long someone must remain low, quiet, and still before Twinr asks for a response.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_SHOWING_INTENT_HOLD_S",
                    "Showing hold (s)",
                    env_values,
                    str(config.proactive_showing_intent_hold_s),
                    tooltip_text="How long a hand or object near the camera should persist before Twinr asks whether you want to show something.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_POSITIVE_CONTACT_HOLD_S",
                    "Smile hold (s)",
                    env_values,
                    str(config.proactive_positive_contact_hold_s),
                    tooltip_text="How long a visible smile toward the device should persist before Twinr opens a positive greeting.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_DISTRESS_HOLD_S",
                    "Distress hold (s)",
                    env_values,
                    str(config.proactive_distress_hold_s),
                    tooltip_text="How long the distress-like audio pattern must last before Twinr gently checks in.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_FALL_TRANSITION_WINDOW_S",
                    "Fall transition window (s)",
                    env_values,
                    str(config.proactive_fall_transition_window_s),
                    tooltip_text="Maximum time between an upright posture and a floor posture for the event to count as a possible fall.",
                ),
            ),
        ),
        SettingsSection(
            title="Proactive sensitivity",
            description="Minimum normalized evidence score per proactive trigger. Range 0.0 to 1.0. Lower values trigger more easily.",
            fields=(
                _text_field(
                    "TWINR_PROACTIVE_PERSON_RETURNED_SCORE_THRESHOLD",
                    "Person returned min score",
                    env_values,
                    str(config.proactive_person_returned_score_threshold),
                    tooltip_text="Minimum combined evidence score for the `person_returned` trigger.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_ATTENTION_WINDOW_SCORE_THRESHOLD",
                    "Attention min score",
                    env_values,
                    str(config.proactive_attention_window_score_threshold),
                    tooltip_text="Minimum combined evidence score for the `attention_window` trigger.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_SLUMPED_QUIET_SCORE_THRESHOLD",
                    "Slumped quiet min score",
                    env_values,
                    str(config.proactive_slumped_quiet_score_threshold),
                    tooltip_text="Minimum combined evidence score for the `slumped_quiet` trigger.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_POSSIBLE_FALL_SCORE_THRESHOLD",
                    "Possible fall min score",
                    env_values,
                    str(config.proactive_possible_fall_score_threshold),
                    tooltip_text="Minimum combined evidence score for the `possible_fall` trigger.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_FLOOR_STILLNESS_SCORE_THRESHOLD",
                    "Floor stillness min score",
                    env_values,
                    str(config.proactive_floor_stillness_score_threshold),
                    tooltip_text="Minimum combined evidence score for the `floor_stillness` trigger.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_SHOWING_INTENT_SCORE_THRESHOLD",
                    "Showing intent min score",
                    env_values,
                    str(config.proactive_showing_intent_score_threshold),
                    tooltip_text="Minimum combined evidence score for the `showing_intent` trigger.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_POSITIVE_CONTACT_SCORE_THRESHOLD",
                    "Positive contact min score",
                    env_values,
                    str(config.proactive_positive_contact_score_threshold),
                    tooltip_text="Minimum combined evidence score for the `positive_contact` trigger.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_DISTRESS_POSSIBLE_SCORE_THRESHOLD",
                    "Distress min score",
                    env_values,
                    str(config.proactive_distress_possible_score_threshold),
                    tooltip_text="Minimum combined evidence score for the `distress_possible` trigger.",
                ),
            ),
        ),
        SettingsSection(
            title="Web UI and files",
            description="Operator UI binding plus the main local file paths Twinr writes to.",
            fields=(
                _text_field(
                    "TWINR_WEB_HOST",
                    "Web host",
                    env_values,
                    config.web_host,
                    tooltip_text="Host interface for the local settings dashboard.",
                ),
                _text_field(
                    "TWINR_WEB_PORT",
                    "Web port",
                    env_values,
                    str(config.web_port),
                    tooltip_text="Port for the local settings dashboard.",
                ),
                _text_field(
                    "TWINR_PERSONALITY_DIR",
                    "Personality directory",
                    env_values,
                    config.personality_dir,
                    tooltip_text="Folder containing SYSTEM.md, PERSONALITY.md, and USER.md.",
                ),
                _text_field(
                    "TWINR_RUNTIME_STATE_PATH",
                    "Runtime state path",
                    env_values,
                    config.runtime_state_path,
                    tooltip_text="JSON snapshot file the running Twinr process updates for the dashboard.",
                ),
                _text_field(
                    "TWINR_MEMORY_MARKDOWN_PATH",
                    "Memory markdown path",
                    env_values,
                    config.memory_markdown_path,
                    tooltip_text="Markdown file where durable memories are stored.",
                ),
                _text_field(
                    "TWINR_REMINDER_STORE_PATH",
                    "Reminder store path",
                    env_values,
                    config.reminder_store_path,
                    tooltip_text="JSON file used for saved reminders.",
                ),
                _select_field(
                    "TWINR_RESTORE_RUNTIME_STATE_ON_STARTUP",
                    "Restore runtime state",
                    env_values,
                    _BOOL_OPTIONS,
                    "true" if config.restore_runtime_state_on_startup else "false",
                    tooltip_text="If enabled, Twinr restores the last saved runtime snapshot at startup.",
                ),
                _text_field(
                    "TWINR_REMINDER_POLL_INTERVAL_S",
                    "Reminder poll interval (s)",
                    env_values,
                    str(config.reminder_poll_interval_s),
                    tooltip_text="How often Twinr checks whether a reminder is due.",
                ),
                _text_field(
                    "TWINR_REMINDER_RETRY_DELAY_S",
                    "Reminder retry delay (s)",
                    env_values,
                    str(config.reminder_retry_delay_s),
                    tooltip_text="Delay before a reminder is retried after a failed attempt.",
                ),
                _text_field(
                    "TWINR_REMINDER_MAX_ENTRIES",
                    "Reminder max entries",
                    env_values,
                    str(config.reminder_max_entries),
                    tooltip_text="Upper bound for stored reminder entries.",
                ),
            ),
        ),
        SettingsSection(
            title="Buttons and motion sensor",
            description="GPIO wiring and debounce settings for buttons and the PIR sensor.",
            fields=(
                _text_field(
                    "TWINR_GPIO_CHIP",
                    "GPIO chip",
                    env_values,
                    config.gpio_chip,
                    tooltip_text="GPIO chip device name used for all button, PIR, and display lines.",
                ),
                _text_field(
                    "TWINR_GREEN_BUTTON_GPIO",
                    "Green button GPIO",
                    env_values,
                    green_button_gpio,
                    tooltip_text="GPIO line number for the green conversation button.",
                ),
                _text_field(
                    "TWINR_YELLOW_BUTTON_GPIO",
                    "Yellow button GPIO",
                    env_values,
                    yellow_button_gpio,
                    tooltip_text="GPIO line number for the yellow print button.",
                ),
                _select_field(
                    "TWINR_BUTTON_ACTIVE_LOW",
                    "Buttons active low",
                    env_values,
                    _YES_NO_OPTIONS,
                    "true" if config.button_active_low else "false",
                    tooltip_text="Set this to Yes when the button pulls the line low when pressed.",
                ),
                _select_field(
                    "TWINR_BUTTON_BIAS",
                    "Button bias",
                    env_values,
                    _GPIO_BIAS_OPTIONS,
                    config.button_bias,
                    tooltip_text="Internal pull configuration for the button lines.",
                ),
                _text_field(
                    "TWINR_BUTTON_DEBOUNCE_MS",
                    "Button debounce (ms)",
                    env_values,
                    str(config.button_debounce_ms),
                    tooltip_text="Debounce time applied to button presses.",
                ),
                _text_field(
                    "TWINR_BUTTON_PROBE_LINES",
                    "Button probe lines",
                    env_values,
                    button_probe_lines,
                    help_text="Comma-separated GPIO line numbers.",
                    tooltip_text="Fallback GPIO lines the button setup tools probe during hardware discovery.",
                ),
                _text_field(
                    "TWINR_PIR_MOTION_GPIO",
                    "PIR motion GPIO",
                    env_values,
                    pir_motion_gpio,
                    tooltip_text="GPIO line number connected to the PIR motion sensor.",
                ),
                _select_field(
                    "TWINR_PIR_ACTIVE_HIGH",
                    "PIR active high",
                    env_values,
                    _YES_NO_OPTIONS,
                    "true" if config.pir_active_high else "false",
                    tooltip_text="Set this to Yes when the PIR sensor drives the line high on motion.",
                ),
                _select_field(
                    "TWINR_PIR_BIAS",
                    "PIR bias",
                    env_values,
                    _GPIO_BIAS_OPTIONS,
                    config.pir_bias,
                    tooltip_text="Internal pull configuration for the PIR input line.",
                ),
                _text_field(
                    "TWINR_PIR_DEBOUNCE_MS",
                    "PIR debounce (ms)",
                    env_values,
                    str(config.pir_debounce_ms),
                    tooltip_text="Debounce time applied to PIR motion edges.",
                ),
            ),
        ),
        SettingsSection(
            title="Display and printer",
            description="E-paper wiring plus printer queue and receipt layout.",
            fields=(
                _text_field(
                    "TWINR_DISPLAY_DRIVER",
                    "Display driver",
                    env_values,
                    config.display_driver,
                    tooltip_text="Driver id for the configured e-paper display.",
                ),
                _text_field(
                    "TWINR_DISPLAY_VENDOR_DIR",
                    "Display vendor dir",
                    env_values,
                    config.display_vendor_dir,
                    tooltip_text="Path to the vendor display driver files.",
                ),
                _text_field(
                    "TWINR_DISPLAY_SPI_BUS",
                    "Display SPI bus",
                    env_values,
                    str(config.display_spi_bus),
                    tooltip_text="SPI bus number used by the e-paper display.",
                ),
                _text_field(
                    "TWINR_DISPLAY_SPI_DEVICE",
                    "Display SPI device",
                    env_values,
                    str(config.display_spi_device),
                    tooltip_text="SPI device number used by the e-paper display.",
                ),
                _text_field(
                    "TWINR_DISPLAY_CS_GPIO",
                    "Display CS GPIO",
                    env_values,
                    str(config.display_cs_gpio),
                    tooltip_text="GPIO line used for display chip select.",
                ),
                _text_field(
                    "TWINR_DISPLAY_DC_GPIO",
                    "Display DC GPIO",
                    env_values,
                    str(config.display_dc_gpio),
                    tooltip_text="GPIO line used for display data/command switching.",
                ),
                _text_field(
                    "TWINR_DISPLAY_RESET_GPIO",
                    "Display reset GPIO",
                    env_values,
                    str(config.display_reset_gpio),
                    tooltip_text="GPIO line used to reset the display panel.",
                ),
                _text_field(
                    "TWINR_DISPLAY_BUSY_GPIO",
                    "Display busy GPIO",
                    env_values,
                    str(config.display_busy_gpio),
                    tooltip_text="GPIO line that reports when the display is still busy refreshing.",
                ),
                _text_field(
                    "TWINR_DISPLAY_WIDTH",
                    "Display width",
                    env_values,
                    str(config.display_width),
                    tooltip_text="Logical display width in pixels.",
                ),
                _text_field(
                    "TWINR_DISPLAY_HEIGHT",
                    "Display height",
                    env_values,
                    str(config.display_height),
                    tooltip_text="Logical display height in pixels.",
                ),
                _text_field(
                    "TWINR_DISPLAY_ROTATION_DEGREES",
                    "Display rotation",
                    env_values,
                    str(config.display_rotation_degrees),
                    tooltip_text="Rotation applied before content is drawn to the display.",
                ),
                _text_field(
                    "TWINR_DISPLAY_FULL_REFRESH_INTERVAL",
                    "Full refresh interval",
                    env_values,
                    str(config.display_full_refresh_interval),
                    tooltip_text="How many partial updates happen before forcing a full display refresh.",
                ),
                _text_field(
                    "TWINR_DISPLAY_POLL_INTERVAL_S",
                    "Display poll interval (s)",
                    env_values,
                    str(config.display_poll_interval_s),
                    tooltip_text="How often the display service checks for new runtime state.",
                ),
                _text_field(
                    "TWINR_PRINTER_QUEUE",
                    "Printer queue",
                    env_values,
                    config.printer_queue,
                    tooltip_text="CUPS queue name for the receipt printer.",
                ),
                _text_field(
                    "TWINR_PRINTER_DEVICE_URI",
                    "Printer device URI",
                    env_values,
                    config.printer_device_uri or "",
                    placeholder="usb://...",
                    tooltip_text="Optional direct CUPS device URI for the printer.",
                ),
                _text_field(
                    "TWINR_PRINTER_HEADER_TEXT",
                    "Printer header",
                    env_values,
                    config.printer_header_text,
                    tooltip_text="Header text printed at the top of each receipt.",
                ),
                _text_field(
                    "TWINR_PRINTER_LINE_WIDTH",
                    "Printer line width",
                    env_values,
                    str(config.printer_line_width),
                    help_text="Lower values make the receipt narrower.",
                    tooltip_text="Maximum characters per printed line.",
                ),
                _text_field(
                    "TWINR_PRINTER_FEED_LINES",
                    "Printer feed lines",
                    env_values,
                    str(config.printer_feed_lines),
                    tooltip_text="Blank lines fed after a print job completes.",
                ),
                _text_field(
                    "TWINR_PRINT_BUTTON_COOLDOWN_S",
                    "Print button cooldown (s)",
                    env_values,
                    str(config.print_button_cooldown_s),
                    tooltip_text="Minimum delay between yellow-button print actions.",
                ),
            ),
        ),
    )


def _memory_sections(config: TwinrConfig, env_values: dict[str, str]) -> tuple[SettingsSection, ...]:
    return (
        SettingsSection(
            title="On-device memory",
            description="Controls how much rolling conversation state Twinr keeps locally before compacting it.",
            fields=(
                _text_field(
                    "TWINR_MEMORY_MAX_TURNS",
                    "Max turns",
                    env_values,
                    str(config.memory_max_turns),
                    tooltip_text="Upper bound for rolling conversation turns kept in active memory before compaction.",
                ),
                _text_field(
                    "TWINR_MEMORY_KEEP_RECENT",
                    "Keep recent turns",
                    env_values,
                    str(config.memory_keep_recent),
                    tooltip_text="Number of newest turns kept verbatim when older turns are compacted.",
                ),
            ),
        ),
        SettingsSection(
            title="Print memory",
            description="Bounds for the print composer so button and tool-based prints stay short and safe.",
            fields=(
                _text_field(
                    "TWINR_PRINT_CONTEXT_TURNS",
                    "Print context turns",
                    env_values,
                    str(config.print_context_turns),
                    tooltip_text="How many recent turns the print composer may inspect before creating a receipt.",
                ),
                _text_field(
                    "TWINR_PRINT_MAX_LINES",
                    "Print max lines",
                    env_values,
                    str(config.print_max_lines),
                    tooltip_text="Maximum printed line count for one answer receipt.",
                ),
                _text_field(
                    "TWINR_PRINT_MAX_CHARS",
                    "Print max chars",
                    env_values,
                    str(config.print_max_chars),
                    tooltip_text="Hard upper bound for receipt text length.",
                ),
            ),
        ),
        SettingsSection(
            title="Long-term memory",
            description="Reserved for the later ChonkyDB integration. These values are stored now but not used by the runtime yet.",
            fields=(
                _select_field(
                    "TWINR_LONG_TERM_MEMORY_ENABLED",
                    "Long-term memory",
                    env_values,
                    _BOOL_OPTIONS,
                    "false",
                    tooltip_text="Prepares long-term memory settings, but the runtime does not actively use them yet.",
                ),
                _text_field(
                    "TWINR_LONG_TERM_MEMORY_BACKEND",
                    "Backend",
                    env_values,
                    "chonkydb",
                    tooltip_text="Identifier for the future long-term memory backend.",
                ),
                _text_field(
                    "TWINR_LONG_TERM_MEMORY_PATH",
                    "Storage path",
                    env_values,
                    "/twinr/data/chonkydb",
                    tooltip_text="Where the future long-term memory database will live on disk.",
                ),
            ),
        ),
    )


def _text_field(
    key: str,
    label: str,
    env_values: dict[str, str],
    default: str,
    *,
    help_text: str = "",
    tooltip_text: str = "",
    placeholder: str = "",
) -> FileBackedSetting:
    return FileBackedSetting(
        key=key,
        label=label,
        value=env_values.get(key, default),
        help_text=help_text,
        tooltip_text=tooltip_text,
        placeholder=placeholder,
    )


def _select_field(
    key: str,
    label: str,
    env_values: dict[str, str],
    options: tuple[tuple[str, str], ...],
    default: str,
    *,
    help_text: str = "",
    tooltip_text: str = "",
) -> FileBackedSetting:
    return FileBackedSetting(
        key=key,
        label=label,
        value=env_values.get(key, default),
        help_text=help_text,
        tooltip_text=tooltip_text,
        input_type="select",
        options=options,
    )


def _textarea_field(
    key: str,
    label: str,
    env_values: dict[str, str],
    default: str,
    *,
    help_text: str = "",
    tooltip_text: str = "",
    placeholder: str = "",
    rows: int = 4,
) -> FileBackedSetting:
    return FileBackedSetting(
        key=key,
        label=label,
        value=env_values.get(key, default),
        help_text=help_text,
        tooltip_text=tooltip_text,
        input_type="textarea",
        placeholder=placeholder,
        rows=rows,
        wide=True,
    )


def _collect_standard_updates(form: dict[str, str], *, exclude: set[str] | None = None) -> dict[str, str]:
    blocked = exclude or set()
    return {key: value.strip() for key, value in form.items() if key and key.isupper() and key not in blocked}


def _format_log_rows(entries: list[dict[str, object]]) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for entry in reversed(entries):
        data = entry.get("data")
        rows.append(
            {
                "created_at": entry.get("created_at", "—"),
                "level": entry.get("level", "info"),
                "event": entry.get("event", "unknown"),
                "message": entry.get("message", ""),
                "data_pretty": (
                    json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True)
                    if isinstance(data, dict) and data
                    else ""
                ),
            }
        )
    return tuple(rows)


def _format_usage_rows(records) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for record in reversed(records):
        token_usage = record.token_usage.to_dict() if record.token_usage is not None else {}
        rows.append(
            {
                "created_at": record.created_at,
                "source": record.source,
                "request_kind": record.request_kind,
                "model": record.model or "unknown",
                "response_id": record.response_id or "—",
                "request_id": record.request_id or "—",
                "used_web_search": (
                    "yes"
                    if record.used_web_search is True
                    else ("no" if record.used_web_search is False else "—")
                ),
                "total_tokens": record.total_tokens if record.total_tokens is not None else "—",
                "input_tokens": token_usage.get("input_tokens", "—"),
                "output_tokens": token_usage.get("output_tokens", "—"),
                "cached_input_tokens": token_usage.get("cached_input_tokens", "—"),
                "reasoning_tokens": token_usage.get("reasoning_tokens", "—"),
                "metadata_pretty": (
                    json.dumps(record.metadata, indent=2, ensure_ascii=False, sort_keys=True)
                    if record.metadata
                    else ""
                ),
            }
        )
    return tuple(rows)


def _health_card_detail(health) -> str:
    parts: list[str] = []
    if health.cpu_temperature_c is not None:
        parts.append(f"{health.cpu_temperature_c:.1f}C")
    if health.memory_used_percent is not None:
        parts.append(f"mem {health.memory_used_percent:.0f}%")
    if health.disk_used_percent is not None:
        parts.append(f"disk {health.disk_used_percent:.0f}%")
    return " · ".join(parts) if parts else "Live Pi snapshot"


def _voice_profile_page_context(
    config: TwinrConfig,
    snapshot: RuntimeSnapshot,
    *,
    action_result: dict[str, str] | None = None,
    action_error: str | None = None,
) -> dict[str, object]:
    monitor = VoiceProfileMonitor.from_config(config)
    return {
        "profile_summary": monitor.summary(),
        "snapshot": snapshot,
        "voice_snapshot_label": _voice_snapshot_label(snapshot),
        "capture_block_reason": _voice_profile_capture_block_reason(config),
        "action_result": action_result,
        "action_error": action_error,
    }


def _voice_snapshot_label(snapshot: RuntimeSnapshot) -> str:
    status = (snapshot.user_voice_status or "").strip()
    if not status:
        return "No recent live voice check."
    label = status.replace("_", " ")
    if snapshot.user_voice_confidence is None:
        return label
    return f"{label} ({snapshot.user_voice_confidence * 100:.0f}%)"


def _voice_profile_capture_block_reason(config: TwinrConfig) -> str | None:
    busy: list[str] = []
    for loop_name, label in (("hardware-loop", "hardware loop"), ("realtime-loop", "realtime loop")):
        owner = loop_lock_owner(config, loop_name)
        if owner is not None:
            busy.append(f"{label} pid {owner}")
    if not busy:
        return None
    joined = ", ".join(busy)
    return f"Stop the running {joined} before capturing a voice profile sample."


def _capture_voice_profile_sample(config: TwinrConfig) -> bytes:
    blocked_reason = _voice_profile_capture_block_reason(config)
    if blocked_reason:
        raise RuntimeError(blocked_reason)
    recorder = SilenceDetectedRecorder.from_config(config)
    return recorder.record_until_pause(pause_ms=config.speech_pause_ms)


def _voice_action_result(assessment: VoiceAssessment) -> dict[str, str]:
    status = "warn"
    if assessment.status == "likely_user":
        status = "ok"
    elif assessment.status in {"disabled", "not_enrolled"}:
        status = "muted"
    detail = assessment.detail
    if assessment.confidence is not None:
        detail = f"{detail} Confidence {assessment.confidence_percent()}."
    return {
        "status": status,
        "title": assessment.label,
        "detail": detail,
    }


def _recent_named_files(directory: Path, *, suffix: str) -> tuple[dict[str, str], ...]:
    if not directory.exists():
        return ()
    files = sorted(
        [path for path in directory.iterdir() if path.is_file() and path.name.endswith(suffix)],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return tuple(
        {
            "name": path.name,
            "path": str(path),
            "download_href": (
                f"/ops/support/download/{path.name}" if suffix == ".zip" else f"/ops/self-test/artifacts/{path.name}"
            ),
        }
        for path in files[:8]
    )


def _resolve_named_file(root: Path, name: str) -> Path:
    safe_name = Path(name).name
    if safe_name != name:
        raise HTTPException(status_code=404, detail="File not found")
    candidate = (root / safe_name).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="File not found") from exc
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return candidate
