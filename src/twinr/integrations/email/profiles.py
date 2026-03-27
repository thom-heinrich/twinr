"""Define reviewed email provider profiles for Twinr's setup UI and runtime.

This module centralizes the small set of provider presets that Twinr can show
to operators. The profiles keep UI labels, transport defaults, and
compatibility notes aligned across the dashboard wizard and the managed email
runtime.
"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_EMAIL_PROFILE_ID = "gmail"


@dataclass(frozen=True, slots=True)
class EmailProviderProfile:
    """Describe one operator-facing mail provider profile."""

    profile_id: str
    label: str
    default_imap_host: str = ""
    default_imap_port: str = ""
    default_smtp_host: str = ""
    default_smtp_port: str = ""
    default_mailbox: str = "INBOX"
    account_placeholder: str = "name@example.com"
    from_placeholder: str = "name@example.com"
    secret_label: str = "Mailbox password"
    secret_placeholder: str = "Mailbox password"
    secret_help_text: str = "Use the password for this mailbox."
    setup_hint: str = ""
    transport_hint: str = ""
    supported: bool = True
    support_detail: str = ""
    strip_secret_spaces: bool = False


_EMAIL_PROVIDER_PROFILES: dict[str, EmailProviderProfile] = {
    "gmail": EmailProviderProfile(
        profile_id="gmail",
        label="Gmail",
        default_imap_host="imap.gmail.com",
        default_imap_port="993",
        default_smtp_host="smtp.gmail.com",
        default_smtp_port="587",
        account_placeholder="name@gmail.com",
        from_placeholder="name@gmail.com",
        secret_label="App password",
        secret_placeholder="16-character app password",
        secret_help_text=(
            "Twinr uses password-based IMAP/SMTP here. For personal Gmail, an app password is the usual choice "
            "when 2-Step Verification is on."
        ),
        setup_hint=(
            "Good default for personal Gmail accounts. Managed Google Workspace accounts often require OAuth instead "
            "of a password or app password."
        ),
        transport_hint="Gmail uses IMAP 993 with SSL/TLS and SMTP 587 with STARTTLS.",
        strip_secret_spaces=True,
    ),
    "united_domains": EmailProviderProfile(
        profile_id="united_domains",
        label="United Domains",
        default_imap_host="imaps.udag.de",
        default_imap_port="993",
        default_smtp_host="smtps.udag.de",
        default_smtp_port="587",
        secret_label="Mailbox password",
        secret_placeholder="Mailbox password",
        secret_help_text=(
            "Use the password for the mailbox itself. United Domains also accepts the mailbox name as username, "
            "but Twinr uses the email address by default."
        ),
        setup_hint="Recommended for United Domains mailboxes.",
        transport_hint="United Domains uses IMAP 993 with SSL/TLS and SMTP 587 with STARTTLS.",
    ),
    "icloud_mail": EmailProviderProfile(
        profile_id="icloud_mail",
        label="iCloud Mail",
        default_imap_host="imap.mail.me.com",
        default_imap_port="993",
        default_smtp_host="smtp.mail.me.com",
        default_smtp_port="587",
        account_placeholder="name@icloud.com",
        from_placeholder="name@icloud.com",
        secret_label="App-specific password",
        secret_placeholder="App-specific password",
        secret_help_text=(
            "Generate an app-specific password in your Apple Account security settings. Twinr uses the full email "
            "address as the login name."
        ),
        setup_hint="Use this when the mailbox lives in iCloud Mail.",
        transport_hint="iCloud Mail uses IMAP 993 with SSL/TLS and SMTP 587 with STARTTLS.",
    ),
    "generic_imap_smtp": EmailProviderProfile(
        profile_id="generic_imap_smtp",
        label="Generic IMAP/SMTP",
        default_imap_port="993",
        default_smtp_port="587",
        secret_label="Mailbox password",
        secret_placeholder="Mailbox password",
        secret_help_text="Use the password or mailbox-specific secret that your provider gave you.",
        setup_hint="Use this when your provider offers standard IMAP and SMTP but is not listed above.",
        transport_hint="Common secure defaults are IMAP 993 with SSL/TLS and SMTP 587 with STARTTLS.",
    ),
    "outlook_oauth": EmailProviderProfile(
        profile_id="outlook_oauth",
        label="Outlook.com / Microsoft mail",
        default_imap_host="outlook.office365.com",
        default_imap_port="993",
        default_smtp_host="smtp-mail.outlook.com",
        default_smtp_port="587",
        account_placeholder="name@outlook.com",
        from_placeholder="name@outlook.com",
        secret_label="Modern Auth sign-in",
        secret_placeholder="OAuth2 required",
        secret_help_text=(
            "Microsoft usually expects OAuth2 / Modern Auth for this profile. Twinr's mail wizard does not support "
            "that sign-in flow yet."
        ),
        setup_hint="Common stress case: Microsoft mail usually needs OAuth2 instead of a raw IMAP/SMTP password.",
        transport_hint="The server defaults are shown for reference, but login still needs Modern Auth.",
        supported=False,
        support_detail=(
            "This provider currently needs OAuth2 / Modern Auth. Twinr's mail setup only supports password-based "
            "IMAP/SMTP right now."
        ),
    ),
}


def email_provider_profile(
    raw_profile: object,
    *,
    default: str = DEFAULT_EMAIL_PROFILE_ID,
) -> EmailProviderProfile:
    """Return the canonical provider profile for one stored or submitted value."""

    normalized = str(raw_profile or default).strip().lower() or default
    if normalized == "generic":
        normalized = "generic_imap_smtp"
    return _EMAIL_PROVIDER_PROFILES.get(normalized, _EMAIL_PROVIDER_PROFILES[default])


def email_profile_choice_options() -> tuple[tuple[str, str], ...]:
    """Return template-ready ``(value, label)`` options for the provider select."""

    return tuple((profile.profile_id, profile.label) for profile in _EMAIL_PROVIDER_PROFILES.values())


def email_profile_ids() -> tuple[str, ...]:
    """Return the known stable provider identifiers."""

    return tuple(_EMAIL_PROVIDER_PROFILES.keys())
