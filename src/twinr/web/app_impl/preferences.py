"""Settings, memory, and profile routes for the refactored Twinr web app."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.conversation.adaptive_timing import AdaptiveTimingStore
from twinr.hardware.voice_profile import VoiceProfileMonitor
from twinr.web.presenters import (
    _adaptive_timing_view,
    _default_reminder_due_at,
    _memory_sections,
    _reminder_rows,
    _settings_sections,
    _voice_action_result,
    _voice_profile_page_context,
    build_activity_advanced_metrics,
    build_activity_overview_cards,
    build_activity_snapshot_metrics,
    build_memory_section_groups,
    build_settings_overview_cards,
    build_settings_section_groups,
    build_settings_shortcut_cards,
)
from twinr.web.support.forms import _collect_standard_updates
from twinr.web.support.store import (
    FileBackedSetting,
    read_text_file,
    write_env_updates,
    write_text_file,
)

from .compat import (
    _call_sync,
    _parse_bounded_form,
    _public_error_message,
    _redirect_saved,
    _redirect_with_error,
    _require_non_empty,
    _safe_file_in_dir,
    _safe_project_subpath,
    logger,
)
from .runtime import AppRuntime


def register_preference_routes(app: FastAPI, runtime: AppRuntime) -> None:
    """Register settings, memory, and context-editor routes."""

    ctx = runtime.ctx
    surface = runtime.surface

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
            form = await _parse_bounded_form(
                request,
                max_form_bytes=runtime.max_form_bytes,
            )
            action = _require_non_empty(
                form.get("_action", ""),
                message="Please choose a voice profile action.",
            )

            async with runtime.locks.voice_profile_lock:
                if action == "enroll":

                    def _enroll() -> dict[str, str]:
                        monitor = VoiceProfileMonitor.from_config(config)
                        sample = surface._capture_voice_profile_sample(config)
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
                        monitor = VoiceProfileMonitor.from_config(config)
                        sample = surface._capture_voice_profile_sample(config)
                        assessment = monitor.assess_wav_bytes(sample)
                        return _voice_action_result(assessment)

                    action_result = await _call_sync(_verify)
                elif action == "reset":

                    def _reset() -> dict[str, str]:
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
            **_voice_profile_page_context(
                config,
                snapshot,
                action_result=action_result,
            ),
        )

    @app.get("/settings", response_class=HTMLResponse)
    async def settings(request: Request) -> HTMLResponse:
        """Render the main settings page."""

        config, env_values = await _call_sync(ctx.load_state)
        sections = _settings_sections(config, env_values)
        section_groups = build_settings_section_groups(sections)
        return ctx.render(
            request,
            "settings_page.html",
            page_title="Settings",
            active_page="settings",
            intro="Start with voice, listening, and everyday behavior. Recovery and file-heavy controls stay grouped at the end.",
            form_action="/settings",
            sections=sections,
            section_groups=section_groups,
            overview_cards=build_settings_overview_cards(section_groups),
            shortcut_cards=build_settings_shortcut_cards(),
            adaptive_timing=_adaptive_timing_view(config),
        )

    @app.post("/settings")
    async def save_settings(request: Request) -> RedirectResponse:
        """Persist settings changes or reset adaptive timing."""

        try:
            config = await _call_sync(TwinrConfig.from_env, ctx.env_path)
            form = await _parse_bounded_form(
                request,
                max_form_bytes=runtime.max_form_bytes,
            )
            action = form.get("_action", "save_settings").strip() or "save_settings"
            if action not in {"save_settings", "reset_adaptive_timing"}:
                raise ValueError("Please choose a valid settings action.")
            async with runtime.locks.state_write_lock:
                if action == "reset_adaptive_timing":
                    await _call_sync(
                        AdaptiveTimingStore(
                            config.adaptive_timing_store_path,
                            config=config,
                        ).reset
                    )
                else:
                    await _call_sync(
                        write_env_updates,
                        ctx.env_path,
                        _collect_standard_updates(form, exclude={"_action"}),
                    )
        except Exception as exc:
            logger.exception("Twinr settings save failed", exc_info=exc)
            return _redirect_with_error(
                "/settings",
                _public_error_message(
                    exc,
                    fallback="Twinr could not save those settings. Please try again.",
                ),
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
        reminder_rows = _reminder_rows(
            reminder_entries,
            timezone_name=config.local_timezone_name,
        )
        pending_reminder_rows = tuple(
            row for row in reminder_rows if not row["delivered"]
        )
        delivered_reminder_rows = tuple(
            row for row in reminder_rows if row["delivered"]
        )
        return ctx.render(
            request,
            "memory_page.html",
            page_title="Activity & Memory",
            active_page="memory",
            intro="Check what Twinr is carrying right now, manage reminders and saved memories, then open the raw traces only when needed.",
            form_action="/memory",
            sections=sections,
            overview_cards=build_activity_overview_cards(
                snapshot,
                pending_reminder_rows,
                delivered_reminder_rows,
                durable_entries,
            ),
            snapshot_metrics=build_activity_snapshot_metrics(snapshot),
            advanced_metrics=build_activity_advanced_metrics(snapshot),
            settings_groups=build_memory_section_groups(sections),
            snapshot=snapshot,
            durable_memory_entries=durable_entries,
            durable_memory_path=str(Path(config.memory_markdown_path)),
            reminder_entries=pending_reminder_rows,
            delivered_reminder_entries=delivered_reminder_rows,
            reminder_path=str(Path(config.reminder_store_path)),
            reminder_default_due_at=_default_reminder_due_at(config),
            timezone_name=config.local_timezone_name,
        )

    @app.post("/memory")
    async def save_memory(request: Request) -> RedirectResponse:
        """Persist memory settings or reminder operations."""

        try:
            config = await _call_sync(TwinrConfig.from_env, ctx.env_path)
            form = await _parse_bounded_form(
                request,
                max_form_bytes=runtime.max_form_bytes,
            )
            action = form.get("_action", "save_settings").strip() or "save_settings"
            if action not in {
                "save_settings",
                "add_memory",
                "add_reminder",
                "mark_reminder_delivered",
                "delete_reminder",
            }:
                raise ValueError("Please choose a valid memory action.")

            async with runtime.locks.state_write_lock:
                if action == "add_memory":
                    summary = _require_non_empty(
                        form.get("memory_summary", ""),
                        message="Please enter a short memory summary.",
                    )
                    await _call_sync(
                        ctx.memory_store(config).remember,
                        kind=form.get("memory_kind", "") or "memory",
                        summary=summary,
                        details=form.get("memory_details", "") or None,
                    )
                elif action == "add_reminder":
                    due_at = _require_non_empty(
                        form.get("reminder_due_at", ""),
                        message="Please choose when the reminder is due.",
                    )
                    summary = _require_non_empty(
                        form.get("reminder_summary", ""),
                        message="Please enter a reminder summary.",
                    )
                    await _call_sync(
                        ctx.reminder_store(config).schedule,
                        due_at=due_at,
                        summary=summary,
                        details=form.get("reminder_details", "") or None,
                        kind=form.get("reminder_kind", "") or "reminder",
                        source="web_ui",
                        original_request=form.get("reminder_original_request", "")
                        or None,
                    )
                elif action == "mark_reminder_delivered":
                    reminder_id = _require_non_empty(
                        form.get("reminder_id", ""),
                        message="Please choose a reminder first.",
                    )
                    await _call_sync(
                        ctx.reminder_store(config).mark_delivered,
                        reminder_id,
                    )
                elif action == "delete_reminder":
                    reminder_id = _require_non_empty(
                        form.get("reminder_id", ""),
                        message="Please choose a reminder first.",
                    )
                    await _call_sync(ctx.reminder_store(config).delete, reminder_id)
                else:
                    await _call_sync(
                        write_env_updates,
                        ctx.env_path,
                        _collect_standard_updates(form, exclude={"_action"}),
                    )
        except KeyError as exc:
            logger.exception("Twinr reminder lookup failed", exc_info=exc)
            return _redirect_with_error("/memory", "That reminder was not found anymore.")
        except Exception as exc:
            logger.exception("Twinr memory save failed", exc_info=exc)
            return _redirect_with_error(
                "/memory",
                _public_error_message(
                    exc,
                    fallback="Twinr could not save that memory change. Please try again.",
                ),
            )
        return _redirect_saved("/memory")

    @app.get("/personality", response_class=HTMLResponse)
    async def personality(request: Request) -> HTMLResponse:
        """Render the hidden personality context editor."""

        config, _env_values = await _call_sync(ctx.load_state)
        personality_dir = _safe_project_subpath(
            ctx.project_root,
            config.personality_dir,
            label="Personality directory",
        )
        system_file = _safe_file_in_dir(
            personality_dir,
            "SYSTEM.md",
            label="Personality system file",
        )
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
            personality_dir = _safe_project_subpath(
                ctx.project_root,
                config.personality_dir,
                label="Personality directory",
            )
            system_file = _safe_file_in_dir(
                personality_dir,
                "SYSTEM.md",
                label="Personality system file",
            )
            store = ctx.personality_context_store(config)
            form = await _parse_bounded_form(
                request,
                max_form_bytes=runtime.max_form_bytes,
            )
            action = form.get("_action", "save_base").strip() or "save_base"
            if action not in {"save_base", "upsert_managed"}:
                raise ValueError("Please choose a valid personality action.")
            async with runtime.locks.state_write_lock:
                if action == "upsert_managed":
                    await _call_sync(
                        store.upsert,
                        category=_require_non_empty(
                            form.get("category", ""),
                            message="Please enter a short category.",
                        ),
                        instruction=_require_non_empty(
                            form.get("instruction", ""),
                            message="Please enter a short instruction.",
                        ),
                    )
                else:
                    await _call_sync(write_text_file, system_file, form.get("SYSTEM", ""))
                    await _call_sync(
                        store.replace_base_text,
                        form.get("PERSONALITY_BASE", ""),
                    )
        except Exception as exc:
            logger.exception("Twinr personality save failed", exc_info=exc)
            return _redirect_with_error(
                "/personality",
                _public_error_message(
                    exc,
                    fallback="Twinr could not save that personality change. Please try again.",
                ),
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
            form = await _parse_bounded_form(
                request,
                max_form_bytes=runtime.max_form_bytes,
            )
            action = form.get("_action", "save_base").strip() or "save_base"
            if action not in {"save_base", "upsert_managed"}:
                raise ValueError("Please choose a valid user action.")
            async with runtime.locks.state_write_lock:
                if action == "upsert_managed":
                    await _call_sync(
                        store.upsert,
                        category=_require_non_empty(
                            form.get("category", ""),
                            message="Please enter a short category.",
                        ),
                        instruction=_require_non_empty(
                            form.get("instruction", ""),
                            message="Please enter a short fact.",
                        ),
                    )
                else:
                    await _call_sync(
                        store.replace_base_text,
                        form.get("USER_BASE", ""),
                    )
        except Exception as exc:
            logger.exception("Twinr user save failed", exc_info=exc)
            return _redirect_with_error(
                "/user",
                _public_error_message(
                    exc,
                    fallback="Twinr could not save that user profile change. Please try again.",
                ),
            )
        return _redirect_saved("/user")
