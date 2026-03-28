"""Ops and debug routes for the refactored Twinr web control surface."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response

from twinr.agent.self_coding import SelfCodingActivationService
from twinr.agent.self_coding.runtime import SelfCodingSkillRuntimeStore
from twinr.agent.self_coding.watchdog import (
    cleanup_stale_compile_status,
    cleanup_stale_execution_run,
)
from twinr.agent.workflows.required_remote_snapshot import (
    assess_required_remote_watchdog_snapshot,
)
from twinr.ops import (
    build_support_bundle,
    check_summary,
    collect_system_health,
    redact_env_values,
    run_config_checks,
)
from twinr.web.presenters import (
    _format_log_rows,
    _format_usage_rows,
    build_ops_debug_page_context,
    build_self_coding_ops_page_context,
    coerce_ops_debug_tab,
)

from .compat import (
    _call_sync,
    _conversation_lab_href,
    _parse_bounded_form,
    _public_error_message,
    _redirect_saved,
    _redirect_with_error,
    _require_non_empty,
    _require_positive_int,
    _resolve_downloadable_file,
    logger,
)
from .runtime import AppRuntime


def register_ops_routes(app: FastAPI, runtime: AppRuntime) -> None:
    """Register `/ops/*` pages plus debug helper routes."""

    ctx = runtime.ctx
    surface = runtime.surface

    @app.get("/ops/self-test", response_class=HTMLResponse)
    async def ops_self_test(request: Request) -> HTMLResponse:
        """Render the self-test selection page."""

        config, _env_values = await _call_sync(ctx.load_state)
        tests = await _call_sync(surface.TwinrSelfTestRunner.available_tests)
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
            form = await _parse_bounded_form(
                request,
                max_form_bytes=runtime.max_form_bytes,
            )
            test_name = _require_non_empty(
                form.get("test_name", ""),
                message="Please choose a self-test.",
            )
            async with runtime.locks.ops_job_lock:
                result = await _call_sync(
                    surface.TwinrSelfTestRunner(config).run,
                    test_name,
                )
            tests = await _call_sync(surface.TwinrSelfTestRunner.available_tests)
        except Exception as exc:
            logger.exception("Twinr self-test failed", exc_info=exc)
            return _redirect_with_error(
                "/ops/self-test",
                _public_error_message(
                    exc,
                    fallback="Twinr could not run that self-test. Please try again.",
                ),
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

        page_context = await _call_sync(
            build_self_coding_ops_page_context,
            ctx.self_coding_store(),
        )
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
            form = await _parse_bounded_form(
                request,
                max_form_bytes=runtime.max_form_bytes,
            )
            skill_id = _require_non_empty(
                form.get("skill_id", ""),
                message="Please choose a learned skill.",
            )
            version = _require_positive_int(
                form.get("version", ""),
                message="Please choose a valid learned skill version.",
            )
            reason = form.get("reason", "").strip() or "operator_pause"
            activation_service = SelfCodingActivationService(
                store=ctx.self_coding_store(),
                automation_store=ctx.automation_store(config),
            )
            async with runtime.locks.state_write_lock:
                await _call_sync(
                    activation_service.pause_activation,
                    skill_id=skill_id,
                    version=version,
                    reason=reason,
                )
        except Exception as exc:
            return _redirect_with_error(
                "/ops/self-coding",
                _public_error_message(
                    exc,
                    fallback="Twinr could not pause that learned skill.",
                ),
            )
        return _redirect_saved("/ops/self-coding")

    @app.post("/ops/self-coding/reactivate")
    async def reactivate_self_coding_activation(request: Request) -> Response:
        """Re-enable one paused learned self-coding skill version from the web UI."""

        try:
            config, _env_values = await _call_sync(ctx.load_state)
            form = await _parse_bounded_form(
                request,
                max_form_bytes=runtime.max_form_bytes,
            )
            skill_id = _require_non_empty(
                form.get("skill_id", ""),
                message="Please choose a learned skill.",
            )
            version = _require_positive_int(
                form.get("version", ""),
                message="Please choose a valid learned skill version.",
            )
            activation_service = SelfCodingActivationService(
                store=ctx.self_coding_store(),
                automation_store=ctx.automation_store(config),
            )
            async with runtime.locks.state_write_lock:
                await _call_sync(
                    activation_service.reactivate_activation,
                    skill_id=skill_id,
                    version=version,
                )
        except Exception as exc:
            return _redirect_with_error(
                "/ops/self-coding",
                _public_error_message(
                    exc,
                    fallback="Twinr could not reactivate that learned skill.",
                ),
            )
        return _redirect_saved("/ops/self-coding")

    @app.post("/ops/self-coding/rollback")
    async def rollback_self_coding_activation(request: Request) -> Response:
        """Restore the previous learned-skill version from the web UI."""

        try:
            config, _env_values = await _call_sync(ctx.load_state)
            form = await _parse_bounded_form(
                request,
                max_form_bytes=runtime.max_form_bytes,
            )
            skill_id = _require_non_empty(
                form.get("skill_id", ""),
                message="Please choose a learned skill.",
            )
            raw_target_version = form.get("target_version", "").strip()
            target_version = (
                None
                if not raw_target_version
                else _require_positive_int(
                    raw_target_version,
                    message="Please choose a valid rollback target version.",
                )
            )
            activation_service = SelfCodingActivationService(
                store=ctx.self_coding_store(),
                automation_store=ctx.automation_store(config),
            )
            async with runtime.locks.state_write_lock:
                await _call_sync(
                    activation_service.rollback_activation,
                    skill_id=skill_id,
                    target_version=target_version,
                )
        except Exception as exc:
            return _redirect_with_error(
                "/ops/self-coding",
                _public_error_message(
                    exc,
                    fallback="Twinr could not roll back that learned skill.",
                ),
            )
        return _redirect_saved("/ops/self-coding")

    @app.post("/ops/self-coding/retest")
    async def retest_self_coding_activation(request: Request) -> Response:
        """Run one capture-only retest for an active learned skill version."""

        try:
            form = await _parse_bounded_form(
                request,
                max_form_bytes=runtime.max_form_bytes,
            )
            skill_id = _require_non_empty(
                form.get("skill_id", ""),
                message="Please choose a learned skill.",
            )
            version = _require_positive_int(
                form.get("version", ""),
                message="Please choose a valid learned skill version.",
            )
            async with runtime.locks.state_write_lock:
                await _call_sync(
                    surface.run_self_coding_skill_retest,
                    project_root=ctx.project_root,
                    env_file=ctx.env_path,
                    skill_id=skill_id,
                    version=version,
                    environment="web",
                )
        except Exception as exc:
            return _redirect_with_error(
                "/ops/self-coding",
                _public_error_message(
                    exc,
                    fallback="Twinr could not retest that learned skill.",
                ),
            )
        return _redirect_saved("/ops/self-coding")

    @app.post("/ops/self-coding/cleanup")
    async def cleanup_self_coding_activation(request: Request) -> Response:
        """Retire one inactive learned skill version and remove its runtime artifacts."""

        try:
            config, _env_values = await _call_sync(ctx.load_state)
            form = await _parse_bounded_form(
                request,
                max_form_bytes=runtime.max_form_bytes,
            )
            skill_id = _require_non_empty(
                form.get("skill_id", ""),
                message="Please choose a learned skill.",
            )
            version = _require_positive_int(
                form.get("version", ""),
                message="Please choose a valid learned skill version.",
            )
            activation_service = SelfCodingActivationService(
                store=ctx.self_coding_store(),
                automation_store=ctx.automation_store(config),
            )
            runtime_store = SelfCodingSkillRuntimeStore(ctx.self_coding_store().root)
            async with runtime.locks.state_write_lock:
                await _call_sync(
                    activation_service.cleanup_activation,
                    skill_id=skill_id,
                    version=version,
                    runtime_store=runtime_store,
                )
        except Exception as exc:
            return _redirect_with_error(
                "/ops/self-coding",
                _public_error_message(
                    exc,
                    fallback="Twinr could not clean up that learned skill version.",
                ),
            )
        return _redirect_saved("/ops/self-coding")

    @app.post("/ops/self-coding/cleanup-run")
    async def cleanup_self_coding_run(request: Request) -> Response:
        """Mark one stale sandbox or retest run as cleaned from the operator UI."""

        try:
            form = await _parse_bounded_form(
                request,
                max_form_bytes=runtime.max_form_bytes,
            )
            run_id = _require_non_empty(
                form.get("run_id", ""),
                message="Please choose a valid self-coding run.",
            )
            async with runtime.locks.state_write_lock:
                await _call_sync(
                    cleanup_stale_execution_run,
                    store=ctx.self_coding_store(),
                    run_id=run_id,
                    reason="operator_cleanup",
                )
        except Exception as exc:
            return _redirect_with_error(
                "/ops/self-coding",
                _public_error_message(
                    exc,
                    fallback="Twinr could not clean up that stale run.",
                ),
            )
        return _redirect_saved("/ops/self-coding")

    @app.post("/ops/self-coding/cleanup-compile")
    async def cleanup_self_coding_compile(request: Request) -> Response:
        """Abort one stale compile run from the operator UI."""

        try:
            form = await _parse_bounded_form(
                request,
                max_form_bytes=runtime.max_form_bytes,
            )
            job_id = _require_non_empty(
                form.get("job_id", ""),
                message="Please choose a valid compile job.",
            )
            async with runtime.locks.state_write_lock:
                await _call_sync(
                    cleanup_stale_compile_status,
                    store=ctx.self_coding_store(),
                    job_id=job_id,
                    reason="operator_cleanup",
                )
        except Exception as exc:
            return _redirect_with_error(
                "/ops/self-coding",
                _public_error_message(
                    exc,
                    fallback="Twinr could not clean up that stale compile.",
                ),
            )
        return _redirect_saved("/ops/self-coding")

    @app.get("/ops/self-test/artifacts/{artifact_name}")
    async def download_self_test_artifact(artifact_name: str) -> FileResponse:
        """Return one self-test artifact after path validation."""

        artifact_path = _resolve_downloadable_file(
            ctx.ops_paths.self_tests_root,
            artifact_name,
        )
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
            remote_memory_watchdog = await _call_sync(
                ctx.load_remote_memory_watchdog,
                config,
            )
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
        health = await _call_sync(
            collect_system_health,
            config,
            snapshot=snapshot,
            event_store=ops_event_store,
        )
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
        health = await _call_sync(
            collect_system_health,
            config,
            snapshot=snapshot,
            event_store=event_store,
        )

        remote_memory_watchdog = None
        remote_memory_watchdog_assessment = None
        remote_memory_watchdog_error = None
        try:
            remote_memory_watchdog = await _call_sync(
                ctx.load_remote_memory_watchdog,
                config,
            )
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
            device_overview = await _call_sync(
                surface.collect_device_overview,
                config,
                event_store=event_store,
            )

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
                    surface.run_long_term_operator_search,
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
                surface.load_conversation_lab_state,
                ctx.ops_paths,
                session_id=str(request.query_params.get("lab_session", "")).strip()
                or None,
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
            await _parse_bounded_form(request, max_form_bytes=runtime.max_form_bytes)
            async with runtime.locks.conversation_lab_lock:
                session_id = await _call_sync(
                    surface.create_conversation_lab_session,
                    ctx.ops_paths,
                )
        except Exception as exc:
            return _redirect_with_error(
                _conversation_lab_href(),
                _public_error_message(
                    exc,
                    fallback="Twinr could not start a new portal conversation.",
                ),
            )
        return RedirectResponse(
            _conversation_lab_href(session_id),
            status_code=303,
        )

    @app.post("/ops/debug/conversation-lab/send")
    async def send_ops_debug_conversation_lab_turn(request: Request) -> Response:
        """Run one portal conversation-lab turn against the real Twinr text path."""

        current_session_id = None
        try:
            config, _env_values = await _call_sync(ctx.load_state)
            form = await _parse_bounded_form(
                request,
                max_form_bytes=runtime.max_form_bytes,
            )
            current_session_id = str(form.get("session_id", "")).strip() or None
            prompt = _require_non_empty(
                form.get("prompt", ""),
                message="Please enter a prompt for the portal conversation.",
            )
            async with runtime.locks.conversation_lab_lock:
                async with runtime.locks.state_write_lock:
                    resolved_session_id = await _call_sync(
                        surface.run_conversation_lab_turn,
                        config,
                        ctx.env_path,
                        ctx.ops_paths,
                        session_id=current_session_id,
                        prompt=prompt,
                    )
        except Exception as exc:
            return _redirect_with_error(
                _conversation_lab_href(current_session_id),
                _public_error_message(
                    exc,
                    fallback="Twinr could not run that portal conversation turn.",
                ),
            )
        return RedirectResponse(
            _conversation_lab_href(resolved_session_id),
            status_code=303,
        )

    @app.get("/ops/devices", response_class=HTMLResponse)
    async def ops_devices(request: Request) -> HTMLResponse:
        """Render the detected device overview."""

        config, _env_values = await _call_sync(ctx.load_state)
        overview = await _call_sync(
            surface.collect_device_overview,
            config,
            event_store=ctx.event_store(),
        )
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
        bundles = await _call_sync(
            surface._recent_named_files,
            ctx.ops_paths.bundles_root,
            suffix=".zip",
        )
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
            await _parse_bounded_form(request, max_form_bytes=runtime.max_form_bytes)
            async with runtime.locks.ops_job_lock:
                bundle = await _call_sync(
                    build_support_bundle,
                    config,
                    env_path=ctx.env_path,
                )
            bundles = await _call_sync(
                surface._recent_named_files,
                ctx.ops_paths.bundles_root,
                suffix=".zip",
            )
        except Exception as exc:
            logger.exception("Twinr support bundle creation failed", exc_info=exc)
            return _redirect_with_error(
                "/ops/support",
                _public_error_message(
                    exc,
                    fallback="Twinr could not build the support bundle. Please try again.",
                ),
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

        artifact_path = _resolve_downloadable_file(
            ctx.ops_paths.bundles_root,
            bundle_name,
        )
        return FileResponse(artifact_path, filename=artifact_path.name)
