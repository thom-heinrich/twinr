from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from twinr.agent.base_agent import RuntimeSnapshot, RuntimeSnapshotStore, TwinrConfig
from twinr.web.store import (
    FileBackedSetting,
    mask_secret,
    parse_urlencoded_form,
    read_env_values,
    read_text_file,
    write_env_updates,
    write_text_file,
)

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

    def render(
        request: Request,
        template_name: str,
        *,
        page_title: str,
        active_page: str,
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
                "restart_notice": "Changes to providers, models, or network settings need a Twinr process restart.",
                "env_path": str(env_path),
                **context,
            },
        )

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request) -> HTMLResponse:
        config, env_values = load_state()
        snapshot = load_snapshot(config)
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
                title="Web UI",
                value=f"{config.web_host}:{config.web_port}",
                detail="LAN-local control surface",
                href="/settings",
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
        )

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
        return render(
            request,
            "memory_page.html",
            page_title="Memory",
            active_page="memory",
            intro="Tune on-device memory and print summarization bounds. Long-term memory is not wired yet, so this page focuses on the active local memory path.",
            form_action="/memory",
            sections=sections,
            snapshot=snapshot,
        )

    @app.post("/memory")
    async def save_memory(request: Request) -> RedirectResponse:
        form = parse_urlencoded_form(await request.body())
        write_env_updates(env_path, _collect_standard_updates(form))
        return RedirectResponse("/memory?saved=1", status_code=303)

    @app.get("/personality", response_class=HTMLResponse)
    async def personality(request: Request) -> HTMLResponse:
        config, _env_values = load_state()
        personality_dir = project_root / config.personality_dir
        return render(
            request,
            "text_page.html",
            page_title="Personality",
            active_page="personality",
            intro="Hidden system context for Twinr. Keep it short and stable so the assistant uses it silently instead of talking about it.",
            form_action="/personality",
            fields=(
                FileBackedSetting(
                    key="SYSTEM",
                    label="SYSTEM.md",
                    value=read_text_file(personality_dir / "SYSTEM.md"),
                    help_text="Core product behavior and permanent operating rules.",
                    input_type="textarea",
                ),
                FileBackedSetting(
                    key="PERSONALITY",
                    label="PERSONALITY.md",
                    value=read_text_file(personality_dir / "PERSONALITY.md"),
                    help_text="Tone, style, humor level, and conversational boundaries.",
                    input_type="textarea",
                ),
            ),
        )

    @app.post("/personality")
    async def save_personality(request: Request) -> RedirectResponse:
        config = TwinrConfig.from_env(env_path)
        personality_dir = project_root / config.personality_dir
        form = parse_urlencoded_form(await request.body())
        write_text_file(personality_dir / "SYSTEM.md", form.get("SYSTEM", ""))
        write_text_file(personality_dir / "PERSONALITY.md", form.get("PERSONALITY", ""))
        return RedirectResponse("/personality?saved=1", status_code=303)

    @app.get("/user", response_class=HTMLResponse)
    async def user(request: Request) -> HTMLResponse:
        config, _env_values = load_state()
        personality_dir = project_root / config.personality_dir
        return render(
            request,
            "text_page.html",
            page_title="User",
            active_page="user",
            intro="Compact user profile facts. These should remain factual and short so Twinr can use them quietly as context.",
            form_action="/user",
            fields=(
                FileBackedSetting(
                    key="USER",
                    label="USER.md",
                    value=read_text_file(personality_dir / "USER.md"),
                    help_text="Short profile facts about the current Twinr user.",
                    input_type="textarea",
                ),
            ),
        )

    @app.post("/user")
    async def save_user(request: Request) -> RedirectResponse:
        config = TwinrConfig.from_env(env_path)
        personality_dir = project_root / config.personality_dir
        form = parse_urlencoded_form(await request.body())
        write_text_file(personality_dir / "USER.md", form.get("USER", ""))
        return RedirectResponse("/user?saved=1", status_code=303)

    return app


def _nav_items() -> tuple[tuple[str, str, str], ...]:
    return (
        ("dashboard", "Dashboard", "/"),
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
