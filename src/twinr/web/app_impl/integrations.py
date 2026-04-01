"""Integration and connection routes for the refactored Twinr web app."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from twinr.integrations import (
    SOCIAL_HISTORY_LEARNING_INTEGRATION_ID,
    SocialHistoryLearningConfig,
    build_managed_integrations,
    social_history_record_with_import_state,
)
from twinr.web.presenters import (
    _apply_email_wizard_connection_test_result,
    _build_calendar_integration_record,
    _build_email_integration_record,
    _build_email_wizard_account_record,
    _build_email_wizard_connection_test_configs,
    _build_email_wizard_connection_test_record,
    _build_email_wizard_guardrail_record,
    _build_email_wizard_profile_record,
    _build_email_wizard_transport_record,
    _build_social_history_learning_record,
    _build_smart_home_integration_record,
    _connect_sections,
    _email_integration_context,
    _integration_overview_rows,
    _social_history_learning_panel,
    _social_history_learning_sections,
    _smart_home_integration_sections,
    _whatsapp_integration_context,
    _calendar_integration_sections,
    build_email_wizard_page_context,
    build_whatsapp_wizard_page_context,
)
from twinr.web.support.forms import _collect_standard_updates
from twinr.web.support.store import write_env_updates
from twinr.web.support.whatsapp import (
    canonicalize_whatsapp_allow_from,
    normalize_project_relative_directory,
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


def register_integrations_routes(app: FastAPI, runtime: AppRuntime) -> None:
    """Register integrations, connect, and WhatsApp/email wizard routes."""

    ctx = runtime.ctx
    surface = runtime.surface
    whatsapp_pairing = runtime.whatsapp_pairing

    @app.get("/integrations", response_class=HTMLResponse)
    async def integrations(request: Request) -> HTMLResponse:
        """Render the integrations configuration page."""

        config, env_values = await _call_sync(ctx.load_state)
        store = ctx.integration_store()
        email_record = await _call_sync(store.get, "email_mailbox")
        calendar_record = await _call_sync(store.get, "calendar_agenda")
        smart_home_record = await _call_sync(store.get, "smart_home_hub")
        social_history_record = await _call_sync(
            store.get,
            SOCIAL_HISTORY_LEARNING_INTEGRATION_ID,
        )
        runtime_state = await _call_sync(
            build_managed_integrations,
            ctx.project_root,
            env_path=ctx.env_path,
        )
        pairing_snapshot = await _call_sync(whatsapp_pairing.load_snapshot)
        return ctx.render(
            request,
            "integrations_page.html",
            page_title="Integrations",
            active_page="integrations",
            restart_notice=(
                "These settings stay local to this Twinr install. "
                "Email secrets remain only in .env, while WhatsApp keeps its linked-device session in its own auth folder."
            ),
            intro=(
                "Set up safe external connections for Twinr. Email and calendar live here as managed integrations, "
                "and WhatsApp uses a guided self-chat wizard so pairing stays bounded and visible."
            ),
            integration_store_path=str(store.path),
            overview_rows=_integration_overview_rows(runtime_state.readiness),
            email_summary=_email_integration_context(
                email_record,
                env_values,
                readiness=runtime_state.readiness_for("email_mailbox"),
            ),
            whatsapp_summary=_whatsapp_integration_context(
                config,
                env_values,
                env_path=ctx.env_path,
                pairing_snapshot=pairing_snapshot,
            ),
            social_history_panel=_social_history_learning_panel(social_history_record),
            social_history_sections=_social_history_learning_sections(
                social_history_record
            ),
            calendar_sections=_calendar_integration_sections(calendar_record),
            smart_home_sections=_smart_home_integration_sections(
                smart_home_record,
                env_values,
            ),
        )

    @app.post("/integrations")
    async def save_integrations(request: Request) -> RedirectResponse:
        """Persist one integration form submission."""

        try:
            config, env_values = await _call_sync(ctx.load_state)
            form = await _parse_bounded_form(
                request,
                max_form_bytes=runtime.max_form_bytes,
            )
            integration_id = _require_non_empty(
                form.get("_integration_id", ""),
                message="Please choose a valid integration form.",
            )
            store = ctx.integration_store()

            if integration_id == "email_mailbox":
                record, env_updates = await _call_sync(
                    _build_email_integration_record,
                    form,
                    env_values,
                )
            elif integration_id == "calendar_agenda":
                current_record = await _call_sync(store.get, "calendar_agenda")
                record, env_updates = await _call_sync(
                    _build_calendar_integration_record,
                    form,
                    current_record,
                )
            elif integration_id == "smart_home_hub":
                record, env_updates = await _call_sync(
                    _build_smart_home_integration_record,
                    form,
                    env_values,
                )
            elif integration_id == SOCIAL_HISTORY_LEARNING_INTEGRATION_ID:
                current_record = await _call_sync(
                    store.get,
                    SOCIAL_HISTORY_LEARNING_INTEGRATION_ID,
                )
                record = await _call_sync(
                    _build_social_history_learning_record,
                    form,
                    current_record,
                )
                env_updates = {}
            else:
                raise ValueError("Please choose a valid integration form.")

            async with runtime.locks.state_write_lock:
                await _call_sync(store.save, record)
                if integration_id == SOCIAL_HISTORY_LEARNING_INTEGRATION_ID:
                    action = str(
                        form.get(
                            "_integration_action",
                            "save_social_history",
                        )
                        or "save_social_history"
                    ).strip()
                    social_history = SocialHistoryLearningConfig.from_record(record)
                    if (
                        social_history.enabled
                        and action == "save_and_import_social_history"
                    ):
                        runtime_probe = await _call_sync(
                            surface.probe_whatsapp_runtime,
                            config,
                            env_path=ctx.env_path,
                        )
                        if not runtime_probe.paired:
                            raise ValueError(
                                "WhatsApp must be paired before Twinr can import history."
                            )
                        import_request = await _call_sync(
                            surface.WhatsAppHistoryImportQueue.from_twinr_config(
                                config
                            ).submit_request,
                            source=social_history.source,
                            lookback_key=social_history.lookback_key,
                        )
                        queued_record = social_history_record_with_import_state(
                            await _call_sync(
                                store.get,
                                SOCIAL_HISTORY_LEARNING_INTEGRATION_ID,
                            ),
                            status="queued",
                            request_id=import_request.request_id,
                            detail="Twinr queued one bounded WhatsApp history import.",
                        )
                        await _call_sync(store.save, queued_record)
                if env_updates:
                    await _call_sync(write_env_updates, ctx.env_path, env_updates)
        except Exception as exc:
            logger.exception("Twinr integration save failed", exc_info=exc)
            return _redirect_with_error(
                "/integrations",
                _public_error_message(
                    exc,
                    fallback="Twinr could not save that integration. Please check the fields and try again.",
                ),
            )
        return _redirect_saved("/integrations")

    @app.get("/integrations/email", response_class=HTMLResponse)
    async def integrations_email_wizard(request: Request) -> HTMLResponse:
        """Render the guided email mailbox setup wizard."""

        _config, env_values = await _call_sync(ctx.load_state)
        store = ctx.integration_store()
        email_record = await _call_sync(store.get, "email_mailbox")
        runtime_state = await _call_sync(
            build_managed_integrations,
            ctx.project_root,
            env_path=ctx.env_path,
        )
        page_context = await _call_sync(
            build_email_wizard_page_context,
            email_record,
            env_values,
            readiness=runtime_state.readiness_for("email_mailbox"),
            requested_step=request.query_params.get("step"),
        )
        return ctx.render(
            request,
            "setup_wizard.html",
            page_title="Email Setup",
            active_page="integrations",
            restart_notice="Mailbox credentials stay in .env. The wizard stores only non-secret mail settings in the integration store.",
            intro=(
                "Use this wizard to connect one mailbox for readouts and approval-first replies. "
                "Twinr keeps mail access bounded and operator-visible."
            ),
            wizard_kicker="Integrations / Email",
            wizard_title="Email setup wizard",
            wizard_form_action="/integrations/email",
            wizard_path="/integrations/email",
            back_href="/integrations",
            back_label="Back to Integrations",
            **page_context,
        )

    @app.post("/integrations/email")
    async def save_integrations_email_wizard(
        request: Request,
    ) -> RedirectResponse:
        """Persist one guided email wizard step."""

        current_step = "profile"
        try:
            _config, env_values = await _call_sync(ctx.load_state)
            form = await _parse_bounded_form(
                request,
                max_form_bytes=runtime.max_form_bytes,
            )
            action = _require_non_empty(
                form.get("_action", ""),
                message="Please choose a valid email setup step.",
            )
            store = ctx.integration_store()
            current_record = await _call_sync(store.get, "email_mailbox")

            if action == "save_profile":
                record, env_updates = await _call_sync(
                    _build_email_wizard_profile_record,
                    form,
                    current_record,
                )
                next_step = "account"
            elif action == "save_account":
                current_step = "account"
                record, env_updates = await _call_sync(
                    _build_email_wizard_account_record,
                    form,
                    current_record,
                    env_values,
                )
                next_step = "transport"
            elif action == "save_transport":
                current_step = "transport"
                record, env_updates = await _call_sync(
                    _build_email_wizard_transport_record,
                    form,
                    current_record,
                )
                next_step = "guardrails"
            elif action == "run_connection_test":
                current_step = "guardrails"
                record, env_updates = await _call_sync(
                    _build_email_wizard_connection_test_record,
                    form,
                    current_record,
                    env_values,
                )
                imap_config, smtp_config = await _call_sync(
                    _build_email_wizard_connection_test_configs,
                    record,
                    env_values,
                )
                test_result = await _call_sync(
                    surface.run_email_connectivity_test,
                    imap_config,
                    smtp_config,
                )
                record = await _call_sync(
                    _apply_email_wizard_connection_test_result,
                    record,
                    test_result,
                )
                next_step = "guardrails"
            elif action == "save_guardrails":
                current_step = "guardrails"
                record, env_updates = await _call_sync(
                    _build_email_wizard_guardrail_record,
                    form,
                    current_record,
                    env_values,
                )
                next_step = "guardrails"
            else:
                raise ValueError("Please choose a valid email setup step.")

            async with runtime.locks.state_write_lock:
                await _call_sync(store.save, record)
                if env_updates:
                    await _call_sync(write_env_updates, ctx.env_path, env_updates)
            return _redirect_saved("/integrations/email", step=next_step)
        except Exception as exc:
            logger.exception("Twinr email setup save failed", exc_info=exc)
            return _redirect_with_error(
                "/integrations/email",
                _public_error_message(
                    exc,
                    fallback="Twinr could not save that email setup step. Please check the fields and try again.",
                ),
                step=current_step,
            )

    @app.get("/connect", response_class=HTMLResponse)
    async def connect(request: Request) -> HTMLResponse:
        """Render provider and credential settings."""

        _config, env_values = await _call_sync(ctx.load_state)
        sections = _connect_sections(env_values)
        return ctx.render(
            request,
            "connect_page.html",
            page_title="Connect",
            active_page="connect",
            intro="Choose providers and manage credentials. Hover the small (?) labels for short explanations. Today only OpenAI is wired in the runtime; the other providers are stored for the next integration pass.",
            form_action="/connect",
            sections=sections,
        )

    @app.post("/connect")
    async def save_connect(request: Request) -> RedirectResponse:
        """Persist provider selection and credential changes."""

        try:
            env_values = (await _call_sync(ctx.load_state))[1]
            form = await _parse_bounded_form(
                request,
                max_form_bytes=runtime.max_form_bytes,
            )
            updates = _collect_standard_updates(
                form,
                exclude={
                    "OPENAI_API_KEY",
                    "DEEPINFRA_API_KEY",
                    "OPENROUTER_API_KEY",
                },
            )
            for secret_key in (
                "OPENAI_API_KEY",
                "DEEPINFRA_API_KEY",
                "OPENROUTER_API_KEY",
            ):
                secret_value = form.get(secret_key, "").strip()
                if secret_value:
                    updates[secret_key] = secret_value
                elif secret_key not in env_values and secret_key == "OPENAI_API_KEY":
                    updates.setdefault(secret_key, "")
            async with runtime.locks.state_write_lock:
                await _call_sync(write_env_updates, ctx.env_path, updates)
        except Exception as exc:
            logger.exception("Twinr connect settings save failed", exc_info=exc)
            return _redirect_with_error(
                "/connect",
                _public_error_message(
                    exc,
                    fallback="Twinr could not save those provider settings. Please try again.",
                ),
            )
        return _redirect_saved("/connect")

    @app.get("/connect/whatsapp", response_class=HTMLResponse)
    async def connect_whatsapp(request: Request) -> HTMLResponse:
        """Render the WhatsApp self-chat setup wizard."""

        config, env_values = await _call_sync(ctx.load_state)
        pairing_snapshot = await _call_sync(whatsapp_pairing.load_snapshot)
        page_context = await _call_sync(
            build_whatsapp_wizard_page_context,
            config,
            env_values,
            env_path=ctx.env_path,
            pairing_snapshot=pairing_snapshot,
            requested_step=request.query_params.get("step"),
        )
        return ctx.render(
            request,
            "whatsapp_wizard.html",
            page_title="WhatsApp Setup",
            active_page="connect",
            restart_notice=(
                "The dashboard pairing window is temporary and bounded. "
                "The linked-device session itself stays in its own auth folder."
            ),
            intro=(
                "This wizard keeps WhatsApp in one internal self-chat with your own number. "
                "It forces self-chat mode on, blocks groups, and can open the pairing window directly from the UI."
            ),
            **page_context,
        )

    @app.post("/connect/whatsapp")
    async def save_connect_whatsapp(request: Request) -> RedirectResponse:
        """Persist one WhatsApp wizard step."""

        current_step = "chat"
        try:
            form = await _parse_bounded_form(
                request,
                max_form_bytes=runtime.max_form_bytes,
            )
            action = _require_non_empty(
                form.get("_action", ""),
                message="Please choose a valid WhatsApp setup step.",
            )

            if action == "save_chat":
                allow_from = canonicalize_whatsapp_allow_from(
                    _require_non_empty(
                        form.get("TWINR_WHATSAPP_ALLOW_FROM", ""),
                        message="Please enter the WhatsApp number that Twinr should allow.",
                    )
                )
                updates = {
                    "TWINR_WHATSAPP_ALLOW_FROM": allow_from,
                    "TWINR_WHATSAPP_SELF_CHAT_MODE": "true",
                    "TWINR_WHATSAPP_GROUPS_ENABLED": "false",
                }
                next_step = "runtime"
                async with runtime.locks.state_write_lock:
                    await _call_sync(write_env_updates, ctx.env_path, updates)
                return _redirect_saved("/connect/whatsapp", step=next_step)

            if action == "save_runtime":
                current_step = "runtime"
                node_binary = _require_non_empty(
                    form.get("TWINR_WHATSAPP_NODE_BINARY", ""),
                    message="Please enter the Node.js binary that should start the WhatsApp worker.",
                )
                auth_dir = normalize_project_relative_directory(
                    ctx.project_root,
                    form.get("TWINR_WHATSAPP_AUTH_DIR", ""),
                    label="the WhatsApp auth folder",
                )
                updates = {
                    "TWINR_WHATSAPP_NODE_BINARY": node_binary,
                    "TWINR_WHATSAPP_AUTH_DIR": auth_dir,
                }
                next_step = "pairing"
                async with runtime.locks.state_write_lock:
                    await _call_sync(write_env_updates, ctx.env_path, updates)
                return _redirect_saved("/connect/whatsapp", step=next_step)

            if action == "start_pairing":
                current_step = "pairing"
                config, _env_values = await _call_sync(ctx.load_state)
                if not str(getattr(config, "whatsapp_allow_from", "") or "").strip():
                    raise ValueError(
                        "Please save your own WhatsApp number before you start pairing."
                    )
                if not bool(config.whatsapp_self_chat_mode) or bool(
                    config.whatsapp_groups_enabled
                ):
                    raise ValueError(
                        "Twinr must keep self-chat mode on and group chats off before pairing starts."
                    )
                runtime_probe = await _call_sync(
                    surface.probe_whatsapp_runtime,
                    config,
                    env_path=ctx.env_path,
                )
                if not runtime_probe.node_ready:
                    raise ValueError(runtime_probe.node_detail)
                if not runtime_probe.worker_ready:
                    raise ValueError(runtime_probe.worker_detail)
                await _call_sync(whatsapp_pairing.load_snapshot)
                await _call_sync(whatsapp_pairing.start_pairing, config)
                return RedirectResponse(
                    url="/connect/whatsapp?step=pairing",
                    status_code=303,
                )

            raise ValueError("Please choose a valid WhatsApp setup step.")
        except Exception as exc:
            logger.exception("Twinr WhatsApp setup save failed", exc_info=exc)
            return _redirect_with_error(
                "/connect/whatsapp",
                _public_error_message(
                    exc,
                    fallback="Twinr could not save that WhatsApp setup step. Please check the fields and try again.",
                ),
                step=current_step,
            )
