from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from twinr.agent.base_agent import RuntimeSnapshot, RuntimeSnapshotStore, TwinrConfig
from twinr.agent.base_agent.context_store import ManagedContextFileStore, PersistentMemoryMarkdownStore
from twinr.ops import (
    TwinrOpsEventStore,
    TwinrSelfTestRunner,
    TwinrUsageStore,
    build_support_bundle,
    check_summary,
    collect_system_health,
    redact_env_values,
    resolve_ops_paths,
    run_config_checks,
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
                "restart_notice": restart_notice,
                "env_path": str(env_path),
                **context,
            },
        )

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request) -> HTMLResponse:
        config, env_values = load_state()
        snapshot = load_snapshot(config)
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

    @app.get("/connect", response_class=HTMLResponse)
    async def connect(request: Request) -> HTMLResponse:
        _config, env_values = load_state()
        sections = _connect_sections(env_values)
        return render(
            request,
            "form_page.html",
            page_title="Connect",
            active_page="connect",
            intro="Choose providers and manage credentials. Today only OpenAI is wired in the runtime; the other providers are stored for the next integration pass.",
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
            intro="Model selection and device/runtime tuning. Printer width and header live here because they directly affect the output layout.",
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
        ("ops_usage", "LLM Usage", "/ops/usage"),
        ("ops_health", "System Health", "/ops/health"),
        ("ops_logs", "Ops Logs", "/ops/logs"),
        ("ops_config", "Config Checks", "/ops/config"),
        ("ops_support", "Support", "/ops/support"),
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


def _connect_sections(env_values: dict[str, str]) -> tuple[SettingsSection, ...]:
    return (
        SettingsSection(
            title="Provider routing",
            description="These switches decide which provider each pipeline stage should use once multiple backends are wired.",
            fields=(
                _select_field("TWINR_PROVIDER_LLM", "LLM provider", env_values, _PROVIDER_OPTIONS, "openai"),
                _select_field("TWINR_PROVIDER_STT", "STT provider", env_values, _PROVIDER_OPTIONS, "openai"),
                _select_field("TWINR_PROVIDER_TTS", "TTS provider", env_values, _PROVIDER_OPTIONS, "openai"),
                _select_field("TWINR_PROVIDER_REALTIME", "Realtime provider", env_values, _PROVIDER_OPTIONS, "openai"),
            ),
        ),
        SettingsSection(
            title="OpenAI",
            description="Main provider currently used by the runtime.",
            fields=(
                FileBackedSetting(
                    key="OPENAI_API_KEY",
                    label="API key",
                    value="",
                    help_text=f"Current value: {mask_secret(env_values.get('OPENAI_API_KEY'))}. Leave blank to keep it unchanged.",
                    input_type="password",
                    placeholder="sk-...",
                    secret=True,
                ),
                _text_field("OPENAI_PROJ_ID", "Project ID", env_values, "proj_..."),
                _select_field(
                    "OPENAI_SEND_PROJECT_HEADER",
                    "Project header",
                    env_values,
                    _TRISTATE_BOOL_OPTIONS,
                    "",
                    help_text="Use auto unless you explicitly need to force the header on or off.",
                ),
            ),
        ),
        SettingsSection(
            title="Other providers",
            description="Stored now, ready for later provider adapters.",
            fields=(
                FileBackedSetting(
                    key="DEEPINFRA_API_KEY",
                    label="DeepInfra API key",
                    value="",
                    help_text=f"Current value: {mask_secret(env_values.get('DEEPINFRA_API_KEY'))}. Leave blank to keep it unchanged.",
                    input_type="password",
                    placeholder="DeepInfra key",
                    secret=True,
                ),
                FileBackedSetting(
                    key="OPENROUTER_API_KEY",
                    label="OpenRouter API key",
                    value="",
                    help_text=f"Current value: {mask_secret(env_values.get('OPENROUTER_API_KEY'))}. Leave blank to keep it unchanged.",
                    input_type="password",
                    placeholder="OpenRouter key",
                    secret=True,
                ),
            ),
        ),
    )


def _settings_sections(config: TwinrConfig, env_values: dict[str, str]) -> tuple[SettingsSection, ...]:
    return (
        SettingsSection(
            title="Models",
            description="One model per job type. OpenAI is the only runtime-backed provider today.",
            fields=(
                _text_field("OPENAI_MODEL", "LLM model", env_values, config.default_model),
                _text_field("OPENAI_STT_MODEL", "STT model", env_values, config.openai_stt_model),
                _text_field("OPENAI_TTS_MODEL", "TTS model", env_values, config.openai_tts_model),
                _text_field("OPENAI_TTS_VOICE", "TTS voice", env_values, config.openai_tts_voice),
                _text_field("OPENAI_REALTIME_MODEL", "Realtime model", env_values, config.openai_realtime_model),
                _text_field("OPENAI_REALTIME_VOICE", "Realtime voice", env_values, config.openai_realtime_voice),
            ),
        ),
        SettingsSection(
            title="Runtime",
            description="Latency and network settings for the live Twinr process.",
            fields=(
                _text_field("TWINR_WEB_HOST", "Web host", env_values, config.web_host),
                _text_field("TWINR_WEB_PORT", "Web port", env_values, str(config.web_port)),
                _text_field("TWINR_SPEECH_PAUSE_MS", "Speech pause (ms)", env_values, str(config.speech_pause_ms)),
                _text_field(
                    "TWINR_CONVERSATION_FOLLOW_UP_TIMEOUT_S",
                    "Follow-up timeout (s)",
                    env_values,
                    str(config.conversation_follow_up_timeout_s),
                ),
                _select_field(
                    "TWINR_CONVERSATION_FOLLOW_UP_ENABLED",
                    "Follow-up listening",
                    env_values,
                    _BOOL_OPTIONS,
                    "true" if config.conversation_follow_up_enabled else "false",
                ),
                _text_field(
                    "TWINR_AUDIO_SPEECH_THRESHOLD",
                    "Speech threshold",
                    env_values,
                    str(config.audio_speech_threshold),
                ),
            ),
        ),
        SettingsSection(
            title="Audio and printer",
            description="Feedback tone and thermal receipt layout.",
            fields=(
                _text_field(
                    "TWINR_AUDIO_BEEP_FREQUENCY_HZ",
                    "Beep frequency (Hz)",
                    env_values,
                    str(config.audio_beep_frequency_hz),
                ),
                _text_field(
                    "TWINR_AUDIO_BEEP_DURATION_MS",
                    "Beep duration (ms)",
                    env_values,
                    str(config.audio_beep_duration_ms),
                ),
                _text_field(
                    "TWINR_PRINTER_HEADER_TEXT",
                    "Printer header",
                    env_values,
                    config.printer_header_text,
                ),
                _text_field(
                    "TWINR_PRINTER_LINE_WIDTH",
                    "Printer line width",
                    env_values,
                    str(config.printer_line_width),
                    help_text="Lower values make the receipt narrower.",
                ),
                _text_field(
                    "TWINR_PRINTER_FEED_LINES",
                    "Printer feed lines",
                    env_values,
                    str(config.printer_feed_lines),
                ),
                _text_field("TWINR_PRINTER_QUEUE", "Printer queue", env_values, config.printer_queue),
            ),
        ),
    )


def _memory_sections(config: TwinrConfig, env_values: dict[str, str]) -> tuple[SettingsSection, ...]:
    return (
        SettingsSection(
            title="On-device memory",
            description="Controls how much rolling conversation state Twinr keeps locally before compacting it.",
            fields=(
                _text_field("TWINR_MEMORY_MAX_TURNS", "Max turns", env_values, str(config.memory_max_turns)),
                _text_field("TWINR_MEMORY_KEEP_RECENT", "Keep recent turns", env_values, str(config.memory_keep_recent)),
            ),
        ),
        SettingsSection(
            title="Print memory",
            description="Bounds for the print composer so button and tool-based prints stay short and safe.",
            fields=(
                _text_field("TWINR_PRINT_CONTEXT_TURNS", "Print context turns", env_values, str(config.print_context_turns)),
                _text_field("TWINR_PRINT_MAX_LINES", "Print max lines", env_values, str(config.print_max_lines)),
                _text_field("TWINR_PRINT_MAX_CHARS", "Print max chars", env_values, str(config.print_max_chars)),
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
                ),
                _text_field("TWINR_LONG_TERM_MEMORY_BACKEND", "Backend", env_values, "chonkydb"),
                _text_field("TWINR_LONG_TERM_MEMORY_PATH", "Storage path", env_values, "/twinr/data/chonkydb"),
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
) -> FileBackedSetting:
    return FileBackedSetting(
        key=key,
        label=label,
        value=env_values.get(key, default),
        help_text=help_text,
    )


def _select_field(
    key: str,
    label: str,
    env_values: dict[str, str],
    options: tuple[tuple[str, str], ...],
    default: str,
    *,
    help_text: str = "",
) -> FileBackedSetting:
    return FileBackedSetting(
        key=key,
        label=label,
        value=env_values.get(key, default),
        help_text=help_text,
        input_type="select",
        options=options,
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
