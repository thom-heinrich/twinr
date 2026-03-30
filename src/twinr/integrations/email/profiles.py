"""Define reviewed email provider profiles for Twinr's setup UI and runtime.

# CHANGELOG: 2026-03-30
# BUG-1: Unknown or misspelled profile IDs no longer silently resolve to Gmail; they now resolve to
#        Generic IMAP/SMTP (or raise in strict mode), preventing wrong server presets and accidental
#        credential submission to the wrong provider.
# BUG-2: Provider metadata now models protocol-specific username candidates and IMAP path prefixes,
#        fixing real-world setup failures such as iCloud username quirks and united-domains INBOX roots.
# SEC-1: Hardened profile normalization and lookup so corrupted stored profile IDs cannot redirect
#        credentials to Gmail by default.
# IMP-1: Added 2026-ready auth/security metadata (OAuth2/XOAUTH2, app-password, TLS mode, alt ports,
#        provider domain aliases, account toggles, and OAuth scope hints).
# IMP-2: Added first-class Google Workspace and Microsoft 365 / Outlook OAuth profiles plus
#        domain-based auto-detection helpers for constrained-device setup flows.

This module centralizes the provider presets that Twinr can show to operators.
The profiles now carry structured transport, authentication, alias, and
auto-detection metadata so the dashboard wizard and the managed email runtime can
make consistent decisions without scraping user-facing hint strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, Mapping

AuthMode = Literal["password", "app_password", "oauth2_xoauth2"]
SecurityMode = Literal["ssl_tls", "starttls", "plain"]
LoginIdentity = Literal["full_email", "localpart"]
UnknownProfileStrategy = Literal["generic", "default", "raise"]

DEFAULT_EMAIL_PROFILE_ID: Final[str] = "gmail"
GENERIC_EMAIL_PROFILE_ID: Final[str] = "generic_imap_smtp"


def _coerce_text(raw: str | bytes | None) -> str:
    """Convert text-like input to ``str`` without throwing on bytes."""

    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="ignore")
    if isinstance(raw, str):
        return raw
    return str(raw)


def _normalize_lookup_token(raw: str | bytes | None) -> str:
    """Normalize profile IDs, aliases, and free-form provider selections."""

    value = _coerce_text(raw).strip().lower()
    if not value:
        return ""
    normalized = []
    last_was_separator = False
    for char in value:
        if char.isalnum():
            normalized.append(char)
            last_was_separator = False
            continue
        if char in {"@", "."}:
            normalized.append(char)
            last_was_separator = False
            continue
        if not last_was_separator:
            normalized.append("_")
            last_was_separator = True
    return "".join(normalized).strip("_")


def _extract_domain(raw: str | bytes | None) -> str:
    """Return a normalized mailbox domain or raw domain fragment."""

    value = _coerce_text(raw).strip().lower()
    if not value:
        return ""
    if "@" in value:
        _, _, value = value.rpartition("@")
    return value.strip(" .")


def _dedupe(items: tuple[str, ...]) -> tuple[str, ...]:
    """Preserve order while removing empty or duplicate values."""

    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return tuple(result)


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

    # Structured 2026-era capability metadata.
    aliases: tuple[str, ...] = ()
    domain_aliases: tuple[str, ...] = ()
    auth_modes: tuple[AuthMode, ...] = ("password",)
    recommended_auth_mode: AuthMode = "password"
    default_imap_security: SecurityMode = "ssl_tls"
    default_smtp_security: SecurityMode = "starttls"
    smtp_alt_ports: tuple[str, ...] = ()
    imap_login_candidates: tuple[LoginIdentity, ...] = ("full_email",)
    smtp_login_candidates: tuple[LoginIdentity, ...] = ("full_email",)
    imap_path_prefix: str = ""
    accepts_primary_password: bool = True
    accepts_provider_mailbox_name: bool = False
    supports_oauth2: bool = False
    requires_oauth_sign_in: bool = False
    requires_imap_enablement: bool = False
    oauth_provider: str = ""
    oauth_scopes: tuple[str, ...] = ()
    oauth_flow_hints: tuple[str, ...] = ()
    auto_config_sources: tuple[str, ...] = ("domain_autoconfig", "thunderbird_ispdb", "rfc6186")
    notes: tuple[str, ...] = ()

    def canonical_secret(self, raw_secret: str | bytes | None) -> str:
        """Normalize a secret exactly as this provider expects it.

        For normal mailbox passwords we preserve the secret verbatim.
        For app-password formats that are commonly shown with spaces, we drop all
        whitespace so copied secrets become paste-safe.
        """

        secret = _coerce_text(raw_secret)
        if self.strip_secret_spaces:
            return "".join(secret.split())
        return secret

    def username_candidates(self, account: str | bytes | None, *, protocol: str = "imap") -> tuple[str, ...]:
        """Return provider-specific username candidates for IMAP or SMTP.

        The runtime can try these values in order when authentication fails.
        """

        address = _coerce_text(account).strip()
        if not address:
            return ()

        localpart = address.partition("@")[0] or address
        candidate_kinds = (
            self.imap_login_candidates if protocol.lower() == "imap" else self.smtp_login_candidates
        )
        resolved: list[str] = []
        for kind in candidate_kinds:
            if kind == "full_email":
                resolved.append(address)
            elif kind == "localpart":
                resolved.append(localpart)
        return _dedupe(tuple(resolved))

    def matches_domain(self, raw_address_or_domain: str | bytes | None) -> bool:
        """Return whether this profile is a known match for a mailbox domain."""

        domain = _extract_domain(raw_address_or_domain)
        if not domain:
            return False
        return domain in self.domain_aliases


_EMAIL_PROVIDER_PROFILES: Mapping[str, EmailProviderProfile] = MappingProxyType(
    {
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
                "Prefer OAuth2/XOAUTH2 where available. If Twinr uses password-style IMAP/SMTP auth, "
                "personal Gmail typically needs a 16-character app password with 2-Step Verification enabled."
            ),
            setup_hint=(
                "Good default for personal Gmail. Managed Google Workspace accounts should use the dedicated "
                "Google Workspace OAuth2 profile instead of a raw mailbox password."
            ),
            transport_hint="IMAP 993 over SSL/TLS. SMTP 587 with STARTTLS or 465 with implicit TLS.",
            strip_secret_spaces=True,
            aliases=("gmail_personal", "google_mail", "googlemail"),
            domain_aliases=("gmail.com", "googlemail.com"),
            auth_modes=("app_password", "oauth2_xoauth2"),
            recommended_auth_mode="oauth2_xoauth2",
            default_imap_security="ssl_tls",
            default_smtp_security="starttls",
            smtp_alt_ports=("465",),
            accepts_primary_password=False,
            supports_oauth2=True,
            oauth_provider="google",
            oauth_scopes=("https://mail.google.com/",),
            oauth_flow_hints=("browser_pkce",),
            notes=(
                "Public apps using the full Gmail IMAP/SMTP scope may require Google app verification.",
            ),
        ),
        "google_workspace_oauth": EmailProviderProfile(
            profile_id="google_workspace_oauth",
            label="Google Workspace / Gmail (OAuth2)",
            default_imap_host="imap.gmail.com",
            default_imap_port="993",
            default_smtp_host="smtp.gmail.com",
            default_smtp_port="587",
            account_placeholder="name@company.com",
            from_placeholder="name@company.com",
            secret_label="Google sign-in",
            secret_placeholder="OAuth2 / Sign in with Google",
            secret_help_text=(
                "Use OAuth2/XOAUTH2 for managed Google mailboxes. This avoids storing a primary mailbox password "
                "on the Raspberry Pi and matches Google's current third-party access guidance."
            ),
            setup_hint=(
                "Use this for Google Workspace or any Gmail deployment where Twinr should perform a modern Google "
                "sign-in instead of asking for a mailbox password."
            ),
            transport_hint="IMAP 993 over SSL/TLS. SMTP 587 with STARTTLS or 465 with implicit TLS.",
            aliases=("google_workspace", "workspace", "gmail_oauth"),
            auth_modes=("oauth2_xoauth2",),
            recommended_auth_mode="oauth2_xoauth2",
            default_imap_security="ssl_tls",
            default_smtp_security="starttls",
            smtp_alt_ports=("465",),
            accepts_primary_password=False,
            supports_oauth2=True,
            requires_oauth_sign_in=True,
            oauth_provider="google",
            oauth_scopes=("https://mail.google.com/",),
            oauth_flow_hints=("browser_pkce",),
            notes=(
                "The Gmail IMAP/SMTP OAuth scope is the full https://mail.google.com/ scope.",
                "Public apps that request full Gmail mail access may need Google verification.",
            ),
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
                "Use the password for the mailbox itself. united-domains accepts the email address and, in some "
                "setups, the provider mailbox name as the username."
            ),
            setup_hint="Recommended preset for united-domains mailboxes.",
            transport_hint=(
                "IMAP 993 over SSL/TLS (143 with STARTTLS also exists). SMTP 587 with STARTTLS or 465 with SSL/TLS."
            ),
            aliases=("uniteddomains", "ud", "udag"),
            auth_modes=("password",),
            recommended_auth_mode="password",
            default_imap_security="ssl_tls",
            default_smtp_security="starttls",
            smtp_alt_ports=("465",),
            imap_path_prefix="INBOX",
            accepts_primary_password=True,
            accepts_provider_mailbox_name=True,
            notes=(
                "Some mail clients need IMAP path prefix INBOX for folder mapping.",
            ),
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
                "Generate an app-specific password in Apple Account security settings. For IMAP, Apple says the "
                "username is usually the local part of the address; if that fails, retry with the full address. "
                "SMTP typically uses the full email address."
            ),
            setup_hint="Use this when the mailbox lives in iCloud Mail.",
            transport_hint="IMAP 993 over SSL/TLS. SMTP 587 with STARTTLS/TLS.",
            aliases=("icloud", "apple_mail", "me.com", "mac.com"),
            domain_aliases=("icloud.com", "me.com", "mac.com"),
            auth_modes=("app_password",),
            recommended_auth_mode="app_password",
            default_imap_security="ssl_tls",
            default_smtp_security="starttls",
            imap_login_candidates=("localpart", "full_email"),
            smtp_login_candidates=("full_email",),
            accepts_primary_password=False,
        ),
        "outlook_oauth": EmailProviderProfile(
            profile_id="outlook_oauth",
            label="Outlook.com",
            default_imap_host="outlook.office365.com",
            default_imap_port="993",
            default_smtp_host="smtp-mail.outlook.com",
            default_smtp_port="587",
            account_placeholder="name@outlook.com",
            from_placeholder="name@outlook.com",
            secret_label="Microsoft sign-in",
            secret_placeholder="OAuth2 / Modern Auth",
            secret_help_text=(
                "Outlook.com requires Modern Auth / OAuth2. IMAP and POP access are disabled by default and must be "
                "enabled in Outlook.com settings before first use."
            ),
            setup_hint="Use this for @outlook.com, @hotmail.com, @live.com, and @msn.com accounts.",
            transport_hint="IMAP 993 over SSL/TLS. SMTP 587 with STARTTLS.",
            aliases=("outlook", "hotmail", "live", "msn"),
            domain_aliases=("outlook.com", "hotmail.com", "live.com", "msn.com"),
            auth_modes=("oauth2_xoauth2",),
            recommended_auth_mode="oauth2_xoauth2",
            default_imap_security="ssl_tls",
            default_smtp_security="starttls",
            accepts_primary_password=False,
            supports_oauth2=True,
            requires_oauth_sign_in=True,
            requires_imap_enablement=True,
            oauth_provider="microsoft",
            oauth_scopes=(
                "https://outlook.office.com/IMAP.AccessAsUser.All",
                "https://outlook.office.com/SMTP.Send",
                "offline_access",
            ),
            oauth_flow_hints=("device_code", "browser_auth_code"),
        ),
        "microsoft_365_oauth": EmailProviderProfile(
            profile_id="microsoft_365_oauth",
            label="Microsoft 365 / Exchange Online",
            default_imap_host="outlook.office365.com",
            default_imap_port="993",
            default_smtp_host="smtp.office365.com",
            default_smtp_port="587",
            account_placeholder="name@company.com",
            from_placeholder="name@company.com",
            secret_label="Microsoft sign-in",
            secret_placeholder="OAuth2 / Modern Auth",
            secret_help_text=(
                "Use OAuth2/XOAUTH2 with Microsoft Entra. Exchange Online supports OAuth for IMAP and SMTP, while "
                "basic auth is deprecated and SMTP basic auth removal has reached the 2026 cutoff."
            ),
            setup_hint=(
                "Use this for Microsoft 365 work or school mailboxes. If sending fails, check whether SMTP AUTH is "
                "disabled tenant-wide or for the mailbox."
            ),
            transport_hint="IMAP 993 over SSL/TLS. SMTP 587 with STARTTLS.",
            aliases=("office365", "microsoft365", "m365", "exchange_online", "exchange"),
            auth_modes=("oauth2_xoauth2",),
            recommended_auth_mode="oauth2_xoauth2",
            default_imap_security="ssl_tls",
            default_smtp_security="starttls",
            accepts_primary_password=False,
            supports_oauth2=True,
            requires_oauth_sign_in=True,
            oauth_provider="microsoft",
            oauth_scopes=(
                "https://outlook.office.com/IMAP.AccessAsUser.All",
                "https://outlook.office.com/SMTP.Send",
                "offline_access",
            ),
            oauth_flow_hints=("device_code", "browser_auth_code", "client_credentials"),
            notes=(
                "SMTP AUTH can be disabled globally or per-mailbox in Exchange Online.",
            ),
        ),
        "generic_imap_smtp": EmailProviderProfile(
            profile_id="generic_imap_smtp",
            label="Generic IMAP/SMTP",
            default_imap_port="993",
            default_smtp_port="587",
            secret_label="Mailbox password / provider secret",
            secret_placeholder="Mailbox password or app password",
            secret_help_text=(
                "Use the mailbox password, an app-specific password, or another provider-issued secret. Prefer "
                "provider auto-configuration or provider docs over guesswork."
            ),
            setup_hint=(
                "Use this when the provider is not listed. A 2026-grade client should try domain auto-config, "
                "Thunderbird ISPDB data, or RFC 6186 discovery before falling back to manual hosts."
            ),
            transport_hint="Common secure defaults: IMAP 993 over SSL/TLS; SMTP 587 with STARTTLS or 465 with TLS.",
            aliases=("generic", "manual", "custom"),
            auth_modes=("password", "app_password", "oauth2_xoauth2"),
            recommended_auth_mode="password",
            default_imap_security="ssl_tls",
            default_smtp_security="starttls",
            smtp_alt_ports=("465",),
            accepts_primary_password=True,
            supports_oauth2=True,
        ),
    }
)

_PROFILE_ID_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        alias: profile_id
        for profile_id, profile in _EMAIL_PROVIDER_PROFILES.items()
        for alias in (
            _normalize_lookup_token(profile_id),
            *(_normalize_lookup_token(item) for item in profile.aliases),
            *(_normalize_lookup_token(item) for item in profile.domain_aliases),
        )
    }
)

_DOMAIN_TO_PROFILE_ID: Mapping[str, str] = MappingProxyType(
    {
        domain: profile_id
        for profile_id, profile in _EMAIL_PROVIDER_PROFILES.items()
        for domain in profile.domain_aliases
    }
)


def email_provider_profile_or_none(raw_profile: str | bytes | None) -> EmailProviderProfile | None:
    """Return the canonical provider profile for a known stored or submitted value."""

    normalized = _normalize_lookup_token(raw_profile)
    if not normalized:
        return None
    resolved_id = _PROFILE_ID_ALIASES.get(normalized)
    if not resolved_id:
        return None
    return _EMAIL_PROVIDER_PROFILES[resolved_id]


def email_provider_profile(
    raw_profile: str | bytes | None,
    *,
    default: str = DEFAULT_EMAIL_PROFILE_ID,
    on_unknown: UnknownProfileStrategy = "generic",
) -> EmailProviderProfile:
    """Return the canonical provider profile for one stored or submitted value.

    Empty input still uses ``default``. Unknown input no longer silently becomes
    Gmail because that can route credentials to the wrong provider.
    """

    default_profile = email_provider_profile_or_none(default) or _EMAIL_PROVIDER_PROFILES[DEFAULT_EMAIL_PROFILE_ID]
    resolved = email_provider_profile_or_none(raw_profile)
    if resolved is not None:
        return resolved

    if not _normalize_lookup_token(raw_profile):
        return default_profile

    if on_unknown == "raise":
        raise KeyError(f"Unknown email provider profile: {raw_profile!r}")
    if on_unknown == "default":
        return default_profile

    # BREAKING: unknown profile IDs now resolve to Generic IMAP/SMTP instead of Gmail/default.
    return _EMAIL_PROVIDER_PROFILES[GENERIC_EMAIL_PROFILE_ID]


def email_provider_profile_for_address(
    raw_address_or_domain: str | bytes | None,
    *,
    fallback_profile: str = GENERIC_EMAIL_PROFILE_ID,
) -> EmailProviderProfile:
    """Best-effort provider auto-detection from an email address or plain domain."""

    domain = _extract_domain(raw_address_or_domain)
    if domain:
        resolved_id = _DOMAIN_TO_PROFILE_ID.get(domain)
        if resolved_id:
            return _EMAIL_PROVIDER_PROFILES[resolved_id]
    return email_provider_profile(fallback_profile, default=fallback_profile, on_unknown="default")


def email_provider_profiles(*, include_unsupported: bool = True) -> tuple[EmailProviderProfile, ...]:
    """Return the known provider profiles in stable UI order."""

    profiles = tuple(_EMAIL_PROVIDER_PROFILES.values())
    if include_unsupported:
        return profiles
    return tuple(profile for profile in profiles if profile.supported)


def email_profile_choice_options(*, include_unsupported: bool = True) -> tuple[tuple[str, str], ...]:
    """Return template-ready ``(value, label)`` options for the provider select."""

    return tuple(
        (profile.profile_id, profile.label)
        for profile in email_provider_profiles(include_unsupported=include_unsupported)
    )


def email_profile_ids(*, include_unsupported: bool = True) -> tuple[str, ...]:
    """Return the known stable provider identifiers."""

    return tuple(profile.profile_id for profile in email_provider_profiles(include_unsupported=include_unsupported))


__all__ = [
    "DEFAULT_EMAIL_PROFILE_ID",
    "GENERIC_EMAIL_PROFILE_ID",
    "EmailProviderProfile",
    "email_provider_profile",
    "email_provider_profile_for_address",
    "email_provider_profile_or_none",
    "email_provider_profiles",
    "email_profile_choice_options",
    "email_profile_ids",
]