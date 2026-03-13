# Twinr Integrations

## Goal

`src/twinr/integrations/` is the generic foundation for connecting Twinr to external systems in a conservative way.

The module is intentionally narrow:

- only bounded MVP adapters for mail and calendar so far
- runtime builders exist, but there is still no direct conversation-flow wiring yet
- no raw credential storage
- deny-by-default policy checks before any adapter execution

This keeps future integrations for email, messenger, smart home, security, and health explicit and reviewable.

## Package Layout

- `models.py`
  - request/result types, manifests, operations, safety profiles, secret references
- `catalog.py`
  - built-in conservative domain manifests for the target integration areas
- `policy.py`
  - deny-by-default policy engine with confirmation and origin checks
- `adapter.py`
  - adapter protocol and callable adapter helper
- `registry.py`
  - thread-safe adapter registration and guarded dispatch
- `email/`
  - allowlist-driven mailbox adapter plus stdlib IMAP/SMTP helpers
- `calendar/`
  - read-only agenda adapter plus simple ICS parsing/source helpers
- `runtime.py`
  - builds live mail/calendar adapter instances from local portal config plus `.env` secrets
  - exposes only redacted readiness summaries for the portal

## Safety Model

The generic layer assumes that external systems can expose personal, security-sensitive, or health-sensitive data.

Because of that, the defaults are intentionally strict:

- integrations must be explicitly enabled in `SafeIntegrationPolicy`
- remote-triggered requests are denied by default
- background polling is denied unless both policy and operation allow it
- oversized payloads are rejected
- critical operations are denied by default
- send/write/control actions require at least user confirmation
- security and health actions use stricter confirmation baselines

The layer also separates secrets from code. Adapters only declare `SecretReference` metadata such as env var names or vault handles.

## Built-In Catalog

The built-in catalog is only a starting point. It covers:

- `calendar_agenda`
  - read today's agenda, upcoming events, next event
- `email_mailbox`
  - read recent messages, draft reply, send message
- `messenger_bridge`
  - read recent thread, send message, send caregiver check-in
- `smart_home_hub`
  - read device state, run safe scene
- `security_monitor`
  - read status, read camera snapshot, send help alert
- `health_records`
  - read daily summary, read medication schedule, send caregiver update

Dangerous controls are intentionally absent from the generic catalog. Examples:

- door unlock
- alarm disarm
- medication edits
- diagnosis changes

Those need a dedicated reviewed adapter and a stronger product decision.

## Current MVP Adapters

Twinr now has two concrete reviewed phase-1 adapters:

- `calendar_agenda`
  - use `ReadOnlyCalendarAdapter` with `ICSCalendarSource`
  - read-only only
  - suitable for printed day plans, spoken summaries, and reminders derived from a trusted agenda feed
- `email_mailbox`
  - use `EmailMailboxAdapter` with `ApprovedEmailContacts`
  - read recent messages normally; known contacts are optional context, not a hard default gate
  - draft or send mail only after explicit approval
  - optional stricter mode can still restrict recipients or senders to known contacts
  - optional stdlib helpers: `IMAPMailboxReader` and `SMTPMailSender`

The local portal now also has a simple setup page at `/integrations` for storing mail and calendar configuration per device. Email secrets stay in `.env`; non-secret integration settings are written to `artifacts/stores/integrations/integrations.json`.

Readiness is derived from the same runtime builder that future Twinr flows will use. The portal only shows redacted state such as `credential stored separately in .env`; it does not echo secret characters back.

Phase-1 limits are intentional:

- no calendar writes or attendee changes
- no mail attachments
- no send/draft without explicit approval
- no mailbox credentials in request payloads or logs
- no calendar URLs with embedded credentials, query tokens, or fragments in the local store

## Minimal Usage Pattern

```python
from twinr.integrations import (
    CallableIntegrationAdapter,
    IntegrationRegistry,
    IntegrationRequest,
    IntegrationResult,
    SafeIntegrationPolicy,
    manifest_for_id,
)

manifest = manifest_for_id("email_mailbox")
assert manifest is not None

adapter = CallableIntegrationAdapter(
    manifest=manifest,
    handler=lambda request: IntegrationResult(ok=True, summary="Email sent."),
)
registry = IntegrationRegistry((adapter,))
policy = SafeIntegrationPolicy(enabled_integrations=frozenset({"email_mailbox"}))

request = IntegrationRequest(
    integration_id="email_mailbox",
    operation_id="send_message",
    parameters={"to": "caregiver@example.com", "body": "Ich bin zuhause."},
    explicit_user_confirmation=True,
)
result = registry.dispatch(request, policy=policy)
```

## Next Steps

Concrete adapters should stay separate from the generic layer and be added only when the product flow is clear.

Recommended order:

1. wire calendar readouts into local reminder/day-plan flows
2. wire mail summaries and caregiver replies into explicit button-confirmed flows
3. log redacted audit events through Twinr ops
4. add one reviewed live provider path at a time beyond IMAP/SMTP and ICS
