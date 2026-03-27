"""Build and validate the guided email setup wizard for the local dashboard."""

from __future__ import annotations

from datetime import datetime, timezone
from collections.abc import Mapping
from typing import Any

from twinr.integrations import ManagedIntegrationConfig
from twinr.integrations.email import IMAPMailboxConfig, SMTPMailSenderConfig
from twinr.integrations.email.connectivity import EmailConnectionTestResult
from twinr.integrations.email.profiles import DEFAULT_EMAIL_PROFILE_ID, email_provider_profile
from twinr.web.support.contracts import DetailMetric, WizardCheckRow, WizardStep
from twinr.web.support.forms import _select_field, _text_field, _textarea_field
from twinr.web.support.store import FileBackedSetting
from twinr.web.presenters.common import (
    _BOOL_OPTIONS,
    _EMAIL_PROFILE_OPTIONS,
    _EMAIL_SECRET_KEY,
    _credential_state_label,
)
from twinr.web.presenters.integrations import (
    _build_email_integration_record,
    _coerce_text,
    _normalize_choice,
    _normalize_persisted_secret,
    _normalize_secret_value,
    _safe_bool_string,
    _safe_normalize_choice,
    _string_settings,
    _validate_email_address,
    _validate_host,
    _validate_mailbox,
    _validate_port,
)


_WIZARD_STEP_KEYS = ("profile", "account", "transport", "guardrails")
_CONNECTION_TEST_STATUS_KEY = "connection_test_status"
_CONNECTION_TEST_SUMMARY_KEY = "connection_test_summary"
_CONNECTION_TEST_DETAIL_KEY = "connection_test_detail"
_CONNECTION_TEST_IMAP_STATUS_KEY = "connection_test_imap_status"
_CONNECTION_TEST_IMAP_SUMMARY_KEY = "connection_test_imap_summary"
_CONNECTION_TEST_IMAP_DETAIL_KEY = "connection_test_imap_detail"
_CONNECTION_TEST_SMTP_STATUS_KEY = "connection_test_smtp_status"
_CONNECTION_TEST_SMTP_SUMMARY_KEY = "connection_test_smtp_summary"
_CONNECTION_TEST_SMTP_DETAIL_KEY = "connection_test_smtp_detail"
_CONNECTION_TEST_TESTED_AT_KEY = "connection_test_tested_at"
_CONNECTION_TEST_SETTING_KEYS = (
    _CONNECTION_TEST_STATUS_KEY,
    _CONNECTION_TEST_SUMMARY_KEY,
    _CONNECTION_TEST_DETAIL_KEY,
    _CONNECTION_TEST_IMAP_STATUS_KEY,
    _CONNECTION_TEST_IMAP_SUMMARY_KEY,
    _CONNECTION_TEST_IMAP_DETAIL_KEY,
    _CONNECTION_TEST_SMTP_STATUS_KEY,
    _CONNECTION_TEST_SMTP_SUMMARY_KEY,
    _CONNECTION_TEST_SMTP_DETAIL_KEY,
    _CONNECTION_TEST_TESTED_AT_KEY,
)


def build_email_wizard_page_context(
    record: ManagedIntegrationConfig,
    env_values: Mapping[str, object],
    *,
    readiness: Any | None = None,
    requested_step: str | None = None,
) -> dict[str, object]:
    """Build the full page context for the guided email setup wizard."""

    values = _current_email_values(record)
    provider_profile = email_provider_profile(values["profile"], default=DEFAULT_EMAIL_PROFILE_ID)
    secret_present = bool(_normalize_persisted_secret(env_values.get(_EMAIL_SECRET_KEY)))
    transport_ready = _transport_ready(values)
    contacts_saved = _known_contact_count(values["known_contacts_text"])
    provider_supported = provider_profile.supported
    connection_test = _connection_test_state(values)
    connection_test_ready = connection_test["status"] == "ok"
    connection_test_can_run = provider_supported and bool(values["account_email"] and secret_present and transport_ready)

    default_step = _default_step(
        provider_supported=provider_supported,
        account_email=values["account_email"],
        secret_present=secret_present,
        transport_ready=transport_ready,
    )
    current_step = _sanitize_step(requested_step, fallback=default_step)

    if record.enabled and readiness is not None:
        overall_status = _coerce_text(getattr(readiness, "status", "warn"), default="warn") or "warn"
        overall_status_label = _coerce_text(getattr(readiness, "summary", "Needs setup"), default="Needs setup") or "Needs setup"
        overall_detail = _coerce_text(getattr(readiness, "detail", provider_profile.setup_hint), default=provider_profile.setup_hint) or provider_profile.setup_hint
    elif not provider_supported:
        overall_status = "warn"
        overall_status_label = "Needs OAuth2"
        overall_detail = provider_profile.support_detail
    elif connection_test_ready:
        overall_status = "ok"
        overall_status_label = "Ready to enable"
        overall_detail = "The mailbox login and outgoing mail login both worked. You can enable mail in the final step when ready."
    elif connection_test["status"] == "fail":
        overall_status = "fail"
        overall_status_label = "Connection failed"
        overall_detail = connection_test["detail"]
    elif values["account_email"] or secret_present or transport_ready:
        overall_status = "warn"
        overall_status_label = "In progress"
        overall_detail = "Finish the remaining steps, run the connection test, and then enable mail in the final guardrails step."
    else:
        overall_status = "blocked"
        overall_status_label = "Needs setup"
        overall_detail = "Choose a provider, save the mailbox login, review the transport defaults, run the connection test, and then enable mail when ready."

    summary_metrics = (
        DetailMetric(
            label="Provider",
            value=provider_profile.label,
            detail=provider_profile.setup_hint or "Provider preset used for this mailbox.",
        ),
        DetailMetric(
            label="Mailbox",
            value=values["account_email"] or "Not saved yet",
            detail="Twinr reads from and usually also sends from this mailbox address.",
        ),
        DetailMetric(
            label="Credential",
            value=_credential_state_label(_normalize_persisted_secret(env_values.get(_EMAIL_SECRET_KEY))),
            detail=f"Stored separately in .env as the {provider_profile.secret_label.lower()}.",
        ),
        DetailMetric(
            label="Transport",
            value=_transport_metric_value(values),
            detail=provider_profile.transport_hint or "Incoming and outgoing mail stay on the saved IMAP and SMTP servers.",
        ),
        DetailMetric(
            label="Connection test",
            value=connection_test["summary"],
            detail=connection_test["detail"],
        ),
    )

    steps = (
        WizardStep(
            key="profile",
            index=1,
            title="Choose the mail provider",
            description="Pick a reviewed profile so Twinr can pre-fill the usual secure server settings.",
            status="fail" if record.enabled and not provider_supported else "ok",
            status_label="Needs OAuth2" if not provider_supported else "Chosen",
            detail=provider_profile.support_detail if not provider_supported else (provider_profile.setup_hint or "You can change this later without losing the saved mailbox address."),
            fields=(
                _select_field(
                    "profile",
                    "Mail provider",
                    values,
                    _EMAIL_PROFILE_OPTIONS,
                    values["profile"],
                    tooltip_text="Use a reviewed preset when possible. Generic IMAP/SMTP keeps the server details manual.",
                ),
            ),
            checks=(
                WizardCheckRow(
                    label="Provider preset",
                    summary=provider_profile.label,
                    detail=provider_profile.setup_hint or "Twinr uses this preset for account hints and transport defaults.",
                    status="ok" if provider_supported else "warn",
                ),
                WizardCheckRow(
                    label="Compatibility",
                    summary="Password-based IMAP/SMTP" if provider_supported else "OAuth2 needed",
                    detail=(
                        "Twinr can use this profile with the mailbox credential fields below."
                        if provider_supported
                        else provider_profile.support_detail
                    ),
                    status="ok" if provider_supported else "fail",
                ),
            ),
            action="save_profile",
            action_label="Save provider step",
            action_hint=provider_profile.setup_hint or provider_profile.support_detail,
            current=current_step == "profile",
        ),
        WizardStep(
            key="account",
            index=2,
            title="Save the mailbox login",
            description="Tell Twinr which mailbox to read from and store the mailbox credential separately in .env.",
            status="ok" if values["account_email"] and secret_present else "warn",
            status_label="Saved" if values["account_email"] and secret_present else "Needed",
            detail=provider_profile.secret_help_text,
            fields=(
                _text_field(
                    "account_email",
                    "Account email",
                    values,
                    values["account_email"],
                    placeholder=provider_profile.account_placeholder,
                    tooltip_text="Twinr uses this address as the mailbox login unless the provider says otherwise.",
                ),
                _text_field(
                    "from_address",
                    "From address",
                    values,
                    values["from_address"],
                    placeholder=provider_profile.from_placeholder,
                    tooltip_text="Leave this equal to the mailbox address unless your provider gave you a different sender.",
                ),
                FileBackedSetting(
                    key=_EMAIL_SECRET_KEY,
                    label=provider_profile.secret_label,
                    value="",
                    help_text=(
                        f"Credential state: {_credential_state_label(_normalize_persisted_secret(env_values.get(_EMAIL_SECRET_KEY)))}. "
                        f"Leave blank to keep it unchanged. {provider_profile.secret_help_text}"
                    ),
                    tooltip_text=provider_profile.setup_hint or provider_profile.secret_help_text,
                    input_type="password",
                    placeholder=provider_profile.secret_placeholder,
                    secret=True,
                ),
            ),
            checks=(
                WizardCheckRow(
                    label="Mailbox account",
                    summary=values["account_email"] or "Not saved yet",
                    detail="Twinr reads from and usually also sends from this mailbox address.",
                    status="ok" if values["account_email"] else "warn",
                ),
                WizardCheckRow(
                    label="Credential",
                    summary=_credential_state_label(_normalize_persisted_secret(env_values.get(_EMAIL_SECRET_KEY))),
                    detail=f"Stored separately in .env as the {provider_profile.secret_label.lower()}.",
                    status="ok" if secret_present else "warn",
                ),
            ),
            action="save_account",
            action_label="Save account step",
            action_hint=provider_profile.secret_help_text,
            current=current_step == "account",
        ),
        WizardStep(
            key="transport",
            index=3,
            title="Review the server settings",
            description="Confirm the incoming and outgoing server details before Twinr tries to use the mailbox.",
            status="ok" if transport_ready else "warn",
            status_label="Saved" if transport_ready else "Needed",
            detail=provider_profile.transport_hint or "Use the exact IMAP and SMTP endpoints from your provider.",
            fields=(
                _text_field(
                    "imap_host",
                    "IMAP host",
                    values,
                    values["imap_host"],
                    placeholder=provider_profile.default_imap_host or "imap.example.com",
                    tooltip_text="Incoming mail server hostname.",
                ),
                _text_field(
                    "imap_port",
                    "IMAP port",
                    values,
                    values["imap_port"],
                    tooltip_text="Incoming mail server port.",
                ),
                _text_field(
                    "imap_mailbox",
                    "Mailbox",
                    values,
                    values["imap_mailbox"],
                    tooltip_text="Mailbox folder that Twinr should read from.",
                ),
                _text_field(
                    "smtp_host",
                    "SMTP host",
                    values,
                    values["smtp_host"],
                    placeholder=provider_profile.default_smtp_host or "smtp.example.com",
                    tooltip_text="Outgoing mail server hostname.",
                ),
                _text_field(
                    "smtp_port",
                    "SMTP port",
                    values,
                    values["smtp_port"],
                    tooltip_text="Outgoing mail server port.",
                ),
            ),
            checks=(
                WizardCheckRow(
                    label="Incoming mail",
                    summary=f"{values['imap_host']}:{values['imap_port']}" if values["imap_host"] else "Not saved yet",
                    detail="Twinr reads recent messages through this IMAP endpoint.",
                    status="ok" if values["imap_host"] and values["imap_port"] else "warn",
                ),
                WizardCheckRow(
                    label="Outgoing mail",
                    summary=f"{values['smtp_host']}:{values['smtp_port']}" if values["smtp_host"] else "Not saved yet",
                    detail="Twinr uses this SMTP endpoint when a reply is explicitly approved.",
                    status="ok" if values["smtp_host"] and values["smtp_port"] else "warn",
                ),
                WizardCheckRow(
                    label="Mailbox folder",
                    summary=values["imap_mailbox"],
                    detail="The default folder is usually INBOX.",
                    status="ok",
                ),
            ),
            action="save_transport",
            action_label="Save transport step",
            action_hint=provider_profile.transport_hint,
            current=current_step == "transport",
        ),
        WizardStep(
            key="guardrails",
            index=4,
            title="Enable mail with clear guardrails",
            description="Choose the mailbox safety fences and only enable mail when the steps above are complete.",
            status=_guardrail_step_status(
                record_enabled=record.enabled,
                readiness=readiness,
                provider_supported=provider_supported,
                connection_test_ready=connection_test_ready,
                connection_test_status=connection_test["status"],
            ),
            status_label=_guardrail_step_label(
                record_enabled=record.enabled,
                readiness=readiness,
                provider_supported=provider_supported,
                connection_test_ready=connection_test_ready,
                connection_test_status=connection_test["status"],
            ),
            detail=(
                provider_profile.support_detail
                if not provider_supported
                else (
                    "Reading recent mail is open by default. Drafts and sends still require explicit approval inside Twinr."
                    if connection_test_ready
                    else "Save the fences you want, then run the connection test before enabling mail."
                )
            ),
            fields=(
                _select_field(
                    "enabled",
                    "Email enabled",
                    values,
                    _BOOL_OPTIONS,
                    "true" if record.enabled else "false",
                    tooltip_text="Turn this on only after the mailbox login and servers are saved.",
                ),
                _select_field(
                    "unread_only_default",
                    "Read unread only",
                    values,
                    _BOOL_OPTIONS,
                    values["unread_only_default"],
                    tooltip_text="When enabled, Twinr prefers unread mail for summaries by default.",
                ),
                _select_field(
                    "restrict_reads_to_known_senders",
                    "Restrict reads to known senders",
                    values,
                    _BOOL_OPTIONS,
                    values["restrict_reads_to_known_senders"],
                    tooltip_text="Optional extra fence. Leave off if Twinr should summarize any mailbox sender.",
                ),
                _select_field(
                    "restrict_recipients_to_known_contacts",
                    "Restrict send to known contacts",
                    values,
                    _BOOL_OPTIONS,
                    values["restrict_recipients_to_known_contacts"],
                    tooltip_text="Optional extra fence. Leave off if explicit approval should be enough for sending.",
                ),
                _textarea_field(
                    "known_contacts_text",
                    "Known contacts",
                    values,
                    values["known_contacts_text"],
                    help_text="One contact per line, for example: Anna <anna@example.com>",
                    tooltip_text="Optional contact allowlist used by the stricter sender and recipient fences.",
                    placeholder="Anna <anna@example.com>",
                ),
            ),
            checks=(
                WizardCheckRow(
                    label="Current status",
                    summary=overall_status_label,
                    detail=overall_detail,
                    status=overall_status,
                ),
                WizardCheckRow(
                    label="Connection test",
                    summary=connection_test["summary"],
                    detail=connection_test["detail"],
                    status=connection_test["status"],
                ),
                WizardCheckRow(
                    label="IMAP login",
                    summary=connection_test["imap_summary"],
                    detail=connection_test["imap_detail"],
                    status=connection_test["imap_status"],
                ),
                WizardCheckRow(
                    label="SMTP login",
                    summary=connection_test["smtp_summary"],
                    detail=connection_test["smtp_detail"],
                    status=connection_test["smtp_status"],
                ),
                WizardCheckRow(
                    label="Known contacts",
                    summary=f"{contacts_saved} saved" if contacts_saved else "None saved",
                    detail="Helpful when you want Twinr to fence reads or sends to a known contact list.",
                    status="ok" if contacts_saved else "warn",
                ),
                WizardCheckRow(
                    label="Send safety",
                    summary=(
                        "Known contacts required"
                        if values["restrict_recipients_to_known_contacts"] == "true"
                        else "Explicit approval only"
                    ),
                    detail=(
                        "Twinr will keep outbound mail limited to the saved contact list."
                        if values["restrict_recipients_to_known_contacts"] == "true"
                        else "Twinr can still prepare or send mail only after explicit approval."
                    ),
                    status="ok" if values["restrict_recipients_to_known_contacts"] == "true" and contacts_saved else "warn" if values["restrict_recipients_to_known_contacts"] == "true" else "ok",
                ),
            ),
            action="save_guardrails",
            action_label="Save guardrails and finish" if connection_test_ready else "Save guardrails only",
            secondary_action="run_connection_test",
            secondary_action_label="Run connection test",
            secondary_action_enabled=connection_test_can_run,
            action_hint=(
                provider_profile.support_detail
                if not provider_supported
                else (
                    "Twinr will allow email to be enabled only after the bounded connection test is green."
                    if not connection_test_ready
                    else "When this step is saved with email enabled, the runtime builder will immediately use the same settings."
                )
            ),
            current=current_step == "guardrails",
        ),
    )

    return {
        "overall_status": overall_status,
        "overall_status_label": overall_status_label,
        "overall_detail": overall_detail,
        "summary_metrics": summary_metrics,
        "steps": steps,
    }


def _build_email_wizard_profile_record(
    form: Mapping[str, object],
    current_record: ManagedIntegrationConfig,
) -> tuple[ManagedIntegrationConfig, dict[str, str]]:
    """Persist only the provider-profile step."""

    current_values = _current_email_values(current_record)
    profile = _normalize_choice(
        form.get("profile", DEFAULT_EMAIL_PROFILE_ID),
        "Mail provider",
        default=DEFAULT_EMAIL_PROFILE_ID,
        options=_EMAIL_PROFILE_OPTIONS,
        fallback=("generic",),
    )
    provider_profile = email_provider_profile(profile, default=DEFAULT_EMAIL_PROFILE_ID)
    profile = provider_profile.profile_id
    profile_changed = current_values["profile"] != profile
    settings = _merged_settings(current_record, profile=profile)
    if profile_changed:
        settings = _clear_connection_test_state(settings)
    enabled = current_record.enabled and provider_profile.supported and not profile_changed
    return _replace_email_record(current_record, settings=settings, enabled=enabled), {}


def _build_email_wizard_account_record(
    form: Mapping[str, object],
    current_record: ManagedIntegrationConfig,
    persisted_env_values: Mapping[str, object],
) -> tuple[ManagedIntegrationConfig, dict[str, str]]:
    """Persist only the mailbox-account step."""
    current_values = _current_email_values(current_record)
    account_email = _coerce_text(form.get("account_email", "")).strip()
    if account_email:
        account_email = _validate_email_address(account_email, "Account email")
    from_address = _coerce_text(form.get("from_address", "")).strip() or account_email
    if from_address:
        from_address = _validate_email_address(from_address, "From address")
    secret_value = _normalize_secret_value(form.get(_EMAIL_SECRET_KEY, ""), profile=current_values["profile"])

    env_updates: dict[str, str] = {}
    if secret_value:
        env_updates[_EMAIL_SECRET_KEY] = secret_value

    current_secret_present = bool(_normalize_persisted_secret(persisted_env_values.get(_EMAIL_SECRET_KEY)))
    account_changed = (
        current_values["account_email"] != account_email
        or current_values["from_address"] != from_address
        or bool(secret_value)
        or not current_secret_present
    )
    settings = _merged_settings(
        current_record,
        account_email=account_email,
        from_address=from_address,
    )
    if account_changed:
        settings = _clear_connection_test_state(settings)
    return _replace_email_record(current_record, settings=settings, enabled=False if account_changed else None), env_updates


def _build_email_wizard_transport_record(
    form: Mapping[str, object],
    current_record: ManagedIntegrationConfig,
) -> tuple[ManagedIntegrationConfig, dict[str, str]]:
    """Persist only the transport step."""

    current_values = _current_email_values(current_record)
    provider_profile = email_provider_profile(current_values["profile"], default=DEFAULT_EMAIL_PROFILE_ID)
    imap_host = _coerce_text(form.get("imap_host", "")).strip() or provider_profile.default_imap_host
    smtp_host = _coerce_text(form.get("smtp_host", "")).strip() or provider_profile.default_smtp_host
    imap_port = _coerce_text(form.get("imap_port", "")).strip() or provider_profile.default_imap_port
    smtp_port = _coerce_text(form.get("smtp_port", "")).strip() or provider_profile.default_smtp_port
    imap_mailbox = _coerce_text(form.get("imap_mailbox", "")).strip() or provider_profile.default_mailbox

    if not imap_host:
        raise ValueError("IMAP host is required for this provider.")
    if not smtp_host:
        raise ValueError("SMTP host is required for this provider.")
    if not imap_port:
        raise ValueError("IMAP port is required for this provider.")
    if not smtp_port:
        raise ValueError("SMTP port is required for this provider.")

    validated_settings = {
        "imap_host": _validate_host(imap_host, "IMAP host"),
        "imap_port": _validate_port(imap_port, "IMAP port"),
        "imap_mailbox": _validate_mailbox(imap_mailbox),
        "smtp_host": _validate_host(smtp_host, "SMTP host"),
        "smtp_port": _validate_port(smtp_port, "SMTP port"),
    }
    transport_changed = any(current_values[key] != value for key, value in validated_settings.items())
    settings = _merged_settings(
        current_record,
        **validated_settings,
    )
    if transport_changed:
        settings = _clear_connection_test_state(settings)
    return _replace_email_record(current_record, settings=settings, enabled=False if transport_changed else None), {}


def _build_email_wizard_connection_test_record(
    form: Mapping[str, object],
    current_record: ManagedIntegrationConfig,
    env_values: Mapping[str, object],
) -> tuple[ManagedIntegrationConfig, dict[str, str]]:
    """Persist guardrail fields without enabling mail before a connection test."""

    current_values = _current_email_values(current_record)
    merged_form = {
        "enabled": "true" if current_record.enabled else "false",
        "profile": current_values["profile"],
        "account_email": current_values["account_email"],
        "from_address": current_values["from_address"],
        "imap_host": current_values["imap_host"],
        "imap_port": current_values["imap_port"],
        "imap_mailbox": current_values["imap_mailbox"],
        "smtp_host": current_values["smtp_host"],
        "smtp_port": current_values["smtp_port"],
        "unread_only_default": form.get("unread_only_default", current_values["unread_only_default"]),
        "restrict_reads_to_known_senders": form.get(
            "restrict_reads_to_known_senders",
            current_values["restrict_reads_to_known_senders"],
        ),
        "restrict_recipients_to_known_contacts": form.get(
            "restrict_recipients_to_known_contacts",
            current_values["restrict_recipients_to_known_contacts"],
        ),
        "known_contacts_text": form.get("known_contacts_text", current_values["known_contacts_text"]),
        _EMAIL_SECRET_KEY: "",
    }
    return _build_email_integration_record(merged_form, env_values)


def _build_email_wizard_guardrail_record(
    form: Mapping[str, object],
    current_record: ManagedIntegrationConfig,
    env_values: Mapping[str, object],
) -> tuple[ManagedIntegrationConfig, dict[str, str]]:
    """Persist the final guardrail step through the full email validation path."""

    current_values = _current_email_values(current_record)
    wants_enabled = _safe_bool_string(form.get("enabled", "false"), default=False) == "true"
    if wants_enabled and current_values[_CONNECTION_TEST_STATUS_KEY] != "ok":
        raise ValueError("Run the connection test successfully before enabling email.")
    merged_form = {
        "enabled": form.get("enabled", "false"),
        "profile": current_values["profile"],
        "account_email": current_values["account_email"],
        "from_address": current_values["from_address"],
        "imap_host": current_values["imap_host"],
        "imap_port": current_values["imap_port"],
        "imap_mailbox": current_values["imap_mailbox"],
        "smtp_host": current_values["smtp_host"],
        "smtp_port": current_values["smtp_port"],
        "unread_only_default": form.get("unread_only_default", current_values["unread_only_default"]),
        "restrict_reads_to_known_senders": form.get(
            "restrict_reads_to_known_senders",
            current_values["restrict_reads_to_known_senders"],
        ),
        "restrict_recipients_to_known_contacts": form.get(
            "restrict_recipients_to_known_contacts",
            current_values["restrict_recipients_to_known_contacts"],
        ),
        "known_contacts_text": form.get("known_contacts_text", current_values["known_contacts_text"]),
        _EMAIL_SECRET_KEY: form.get(_EMAIL_SECRET_KEY, ""),
    }
    return _build_email_integration_record(merged_form, env_values)


def _current_email_values(record: ManagedIntegrationConfig) -> dict[str, str]:
    """Normalize the saved email settings for wizard rendering and merging."""

    values = _string_settings(getattr(record, "settings", {}))
    profile = _safe_normalize_choice(
        values.get("profile", DEFAULT_EMAIL_PROFILE_ID),
        default=DEFAULT_EMAIL_PROFILE_ID,
        options=_EMAIL_PROFILE_OPTIONS,
        fallback=("generic",),
    )
    provider_profile = email_provider_profile(profile, default=DEFAULT_EMAIL_PROFILE_ID)
    profile = provider_profile.profile_id
    account_email = values.get("account_email", "").strip()
    return {
        "profile": profile,
        "account_email": account_email,
        "from_address": values.get("from_address", "").strip() or account_email,
        "imap_host": values.get("imap_host", provider_profile.default_imap_host).strip(),
        "imap_port": values.get("imap_port", provider_profile.default_imap_port).strip(),
        "imap_mailbox": values.get("imap_mailbox", provider_profile.default_mailbox).strip() or provider_profile.default_mailbox,
        "smtp_host": values.get("smtp_host", provider_profile.default_smtp_host).strip(),
        "smtp_port": values.get("smtp_port", provider_profile.default_smtp_port).strip(),
        "unread_only_default": _safe_bool_string(values.get("unread_only_default", "true"), default=True),
        "restrict_reads_to_known_senders": _safe_bool_string(
            values.get("restrict_reads_to_known_senders", "false"),
            default=False,
        ),
        "restrict_recipients_to_known_contacts": _safe_bool_string(
            values.get("restrict_recipients_to_known_contacts", "false"),
            default=False,
        ),
        "known_contacts_text": values.get("known_contacts_text", ""),
        _CONNECTION_TEST_STATUS_KEY: values.get(_CONNECTION_TEST_STATUS_KEY, ""),
        _CONNECTION_TEST_SUMMARY_KEY: values.get(_CONNECTION_TEST_SUMMARY_KEY, ""),
        _CONNECTION_TEST_DETAIL_KEY: values.get(_CONNECTION_TEST_DETAIL_KEY, ""),
        _CONNECTION_TEST_IMAP_STATUS_KEY: values.get(_CONNECTION_TEST_IMAP_STATUS_KEY, ""),
        _CONNECTION_TEST_IMAP_SUMMARY_KEY: values.get(_CONNECTION_TEST_IMAP_SUMMARY_KEY, ""),
        _CONNECTION_TEST_IMAP_DETAIL_KEY: values.get(_CONNECTION_TEST_IMAP_DETAIL_KEY, ""),
        _CONNECTION_TEST_SMTP_STATUS_KEY: values.get(_CONNECTION_TEST_SMTP_STATUS_KEY, ""),
        _CONNECTION_TEST_SMTP_SUMMARY_KEY: values.get(_CONNECTION_TEST_SMTP_SUMMARY_KEY, ""),
        _CONNECTION_TEST_SMTP_DETAIL_KEY: values.get(_CONNECTION_TEST_SMTP_DETAIL_KEY, ""),
        _CONNECTION_TEST_TESTED_AT_KEY: values.get(_CONNECTION_TEST_TESTED_AT_KEY, ""),
    }


def _sanitize_step(requested_step: str | None, *, fallback: str) -> str:
    """Return a known wizard step or the provided fallback."""

    normalized = _coerce_text(requested_step).strip().lower()
    return normalized if normalized in _WIZARD_STEP_KEYS else fallback


def _default_step(
    *,
    provider_supported: bool,
    account_email: str,
    secret_present: bool,
    transport_ready: bool,
) -> str:
    """Pick the most useful wizard step for the current email setup state."""

    if not provider_supported:
        return "profile"
    if not account_email or not secret_present:
        return "account"
    if not transport_ready:
        return "transport"
    return "guardrails"


def _transport_ready(values: Mapping[str, str]) -> bool:
    """Return whether all transport fields are currently populated."""

    return bool(
        values["imap_host"]
        and values["imap_port"]
        and values["imap_mailbox"]
        and values["smtp_host"]
        and values["smtp_port"]
    )


def _transport_metric_value(values: Mapping[str, str]) -> str:
    """Render one compact transport summary for the wizard hero."""

    if not _transport_ready(values):
        return "Not saved yet"
    return f"IMAP {values['imap_host']}:{values['imap_port']} / SMTP {values['smtp_host']}:{values['smtp_port']}"


def _known_contact_count(text: str) -> int:
    """Return the number of non-empty contact lines."""

    return sum(1 for line in text.splitlines() if line.strip())


def _guardrail_step_status(
    *,
    record_enabled: bool,
    readiness: Any | None,
    provider_supported: bool,
    connection_test_ready: bool,
    connection_test_status: str,
) -> str:
    """Return the status token for the final guardrail step."""

    if not provider_supported and record_enabled:
        return "fail"
    if not provider_supported:
        return "warn"
    if record_enabled:
        if readiness is None:
            return "warn"
        return _coerce_text(getattr(readiness, "status", "warn"), default="warn") or "warn"
    if connection_test_ready:
        return "ok"
    if connection_test_status == "fail":
        return "fail"
    return "warn"


def _guardrail_step_label(
    *,
    record_enabled: bool,
    readiness: Any | None,
    provider_supported: bool,
    connection_test_ready: bool,
    connection_test_status: str,
) -> str:
    """Return the short label for the final guardrail step."""

    if not provider_supported and record_enabled:
        return "Needs OAuth2"
    if not provider_supported:
        return "Needs OAuth2"
    if record_enabled:
        if readiness is None:
            return "Needs setup"
        return _coerce_text(getattr(readiness, "summary", "Needs setup"), default="Needs setup") or "Needs setup"
    if connection_test_ready:
        return "Ready to enable"
    if connection_test_status == "fail":
        return "Fix connection"
    return "Not enabled"


def _merged_settings(current_record: ManagedIntegrationConfig, **overrides: str) -> dict[str, str]:
    """Merge a partial set of email settings into the current record."""

    settings = _string_settings(getattr(current_record, "settings", {}))
    settings.update({key: value for key, value in overrides.items()})
    return settings


def _clear_connection_test_state(settings: Mapping[str, str]) -> dict[str, str]:
    """Drop any persisted connection-test result from one settings mapping."""

    return {key: value for key, value in settings.items() if key not in _CONNECTION_TEST_SETTING_KEYS}


def _connection_test_state(values: Mapping[str, str]) -> dict[str, str]:
    """Normalize the persisted connection-test state for wizard rendering."""

    tested_at = _format_connection_test_timestamp(values.get(_CONNECTION_TEST_TESTED_AT_KEY, ""))
    tested_suffix = f" Last checked {tested_at}." if tested_at else ""
    status = values.get(_CONNECTION_TEST_STATUS_KEY, "").strip().lower()
    summary = values.get(_CONNECTION_TEST_SUMMARY_KEY, "").strip()
    detail = values.get(_CONNECTION_TEST_DETAIL_KEY, "").strip()
    imap_status = values.get(_CONNECTION_TEST_IMAP_STATUS_KEY, "").strip().lower()
    imap_summary = values.get(_CONNECTION_TEST_IMAP_SUMMARY_KEY, "").strip()
    imap_detail = values.get(_CONNECTION_TEST_IMAP_DETAIL_KEY, "").strip()
    smtp_status = values.get(_CONNECTION_TEST_SMTP_STATUS_KEY, "").strip().lower()
    smtp_summary = values.get(_CONNECTION_TEST_SMTP_SUMMARY_KEY, "").strip()
    smtp_detail = values.get(_CONNECTION_TEST_SMTP_DETAIL_KEY, "").strip()

    if status not in {"ok", "warn", "fail"}:
        status = "warn"
        summary = "Not tested yet"
        detail = "Twinr has not run the bounded mailbox connection test yet."
    if not imap_status:
        imap_status = "muted"
    if not imap_summary:
        imap_summary = "Not tested yet"
    if not imap_detail:
        imap_detail = "Twinr has not tried the mailbox login yet."
    if not smtp_status:
        smtp_status = "muted"
    if not smtp_summary:
        smtp_summary = "Not tested yet"
    if not smtp_detail:
        smtp_detail = "Twinr has not tried the outgoing mail login yet."

    return {
        "status": status,
        "summary": summary,
        "detail": f"{detail}{tested_suffix}".strip(),
        "imap_status": imap_status,
        "imap_summary": imap_summary,
        "imap_detail": f"{imap_detail}{tested_suffix}".strip() if imap_status != "muted" else imap_detail,
        "smtp_status": smtp_status,
        "smtp_summary": smtp_summary,
        "smtp_detail": f"{smtp_detail}{tested_suffix}".strip() if smtp_status != "muted" else smtp_detail,
    }


def _format_connection_test_timestamp(value: str) -> str:
    """Format a persisted ISO timestamp for simple operator display."""

    candidate = _coerce_text(value).strip()
    if not candidate:
        return ""
    normalized = candidate[:-1] + "+00:00" if candidate.endswith("Z") else candidate
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return ""
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _build_email_wizard_connection_test_configs(
    record: ManagedIntegrationConfig,
    env_values: Mapping[str, object],
) -> tuple[IMAPMailboxConfig, SMTPMailSenderConfig]:
    """Build the bounded IMAP and SMTP probe configs for the wizard."""

    values = _current_email_values(record)
    provider_profile = email_provider_profile(values["profile"], default=DEFAULT_EMAIL_PROFILE_ID)
    if not provider_profile.supported:
        raise ValueError(provider_profile.support_detail)

    account_email = values["account_email"].strip()
    from_address = values["from_address"].strip() or account_email
    secret_value = _normalize_persisted_secret(env_values.get(_EMAIL_SECRET_KEY))
    if not account_email:
        raise ValueError("Save the mailbox address before running the connection test.")
    if not secret_value:
        raise ValueError(f"Save the {provider_profile.secret_label.lower()} before running the connection test.")
    if not _transport_ready(values):
        raise ValueError("Save the IMAP and SMTP server settings before running the connection test.")

    return (
        IMAPMailboxConfig(
            host=values["imap_host"],
            port=int(values["imap_port"]),
            username=account_email,
            password=secret_value,
            mailbox=values["imap_mailbox"],
            use_ssl=True,
            connect_timeout_seconds=8.0,
            operation_timeout_seconds=8.0,
            max_retries=0,
            retry_backoff_seconds=0.0,
        ),
        SMTPMailSenderConfig(
            host=values["smtp_host"],
            port=int(values["smtp_port"]),
            username=account_email,
            password=secret_value,
            from_address=from_address,
            use_starttls=True,
            use_ssl=False,
            timeout_s=10.0,
        ),
    )


def _apply_email_wizard_connection_test_result(
    current_record: ManagedIntegrationConfig,
    result: EmailConnectionTestResult,
) -> ManagedIntegrationConfig:
    """Persist one redacted connection-test result on the managed email record."""

    settings = _clear_connection_test_state(_string_settings(getattr(current_record, "settings", {})))
    settings.update(
        {
            _CONNECTION_TEST_STATUS_KEY: result.status,
            _CONNECTION_TEST_SUMMARY_KEY: result.summary,
            _CONNECTION_TEST_DETAIL_KEY: result.detail,
            _CONNECTION_TEST_IMAP_STATUS_KEY: result.imap.status,
            _CONNECTION_TEST_IMAP_SUMMARY_KEY: result.imap.summary,
            _CONNECTION_TEST_IMAP_DETAIL_KEY: result.imap.detail,
            _CONNECTION_TEST_SMTP_STATUS_KEY: result.smtp.status,
            _CONNECTION_TEST_SMTP_SUMMARY_KEY: result.smtp.summary,
            _CONNECTION_TEST_SMTP_DETAIL_KEY: result.smtp.detail,
            _CONNECTION_TEST_TESTED_AT_KEY: result.tested_at,
        }
    )
    return _replace_email_record(current_record, settings=settings)


def _replace_email_record(
    current_record: ManagedIntegrationConfig,
    *,
    settings: Mapping[str, str],
    enabled: bool | None = None,
) -> ManagedIntegrationConfig:
    """Return a new managed email record with merged settings."""

    return ManagedIntegrationConfig(
        integration_id="email_mailbox",
        enabled=current_record.enabled if enabled is None else bool(enabled),
        settings=dict(settings),
    )


__all__ = [
    "_build_email_wizard_account_record",
    "_apply_email_wizard_connection_test_result",
    "_build_email_wizard_connection_test_configs",
    "_build_email_wizard_connection_test_record",
    "_build_email_wizard_guardrail_record",
    "_build_email_wizard_profile_record",
    "_build_email_wizard_transport_record",
    "build_email_wizard_page_context",
]
