# AGENTS.md — /src/twinr/integrations/email

## Scope

This directory owns Twinr's email normalization models, integration adapter,
IMAP mailbox reader, and SMTP sender. Structural ownership and exports live in
[component.yaml](./component.yaml).

Out of scope:
- integration settings loading and readiness assembly in `../runtime.py`
- integration storage and web form handling under `src/twinr/web`
- non-email integration manifests, policy gating, or calendar logic elsewhere in `src/twinr/integrations`

## Key files

- `models.py` — canonical email normalization, drafts, summaries, and approved contacts
- `adapter.py` — integration request validation, contact policy enforcement, and result shaping
- `imap.py` — bounded IMAP mailbox reader and preview extraction
- `smtp.py` — bounded SMTP sender and transport validation
- `__init__.py` — public export surface; treat export changes as API-impacting

## Invariants

- `models.py` is the source of truth for email normalization and approved-contact resolution; do not duplicate address or alias parsing rules elsewhere in this package.
- `adapter.py` must require explicit approval before draft or send operations and must return structured `IntegrationResult` failures instead of leaking provider exceptions.
- Recipient restrictions flow through `ApprovedEmailContacts` and `EmailAdapterSettings`; transport modules must not bypass those policy decisions.
- `imap.py` and `smtp.py` must keep network work bounded with validated timeouts and encrypted transport defaults.
- Public exports are defined in `__init__.py` and re-exported from `src/twinr/integrations/__init__.py`; changing them is a compatibility change.

## Verification

After any edit in this directory, run:

```bash
PYTHONPATH=src ./.venv/bin/pytest test/test_integrations_email.py test/test_integration_runtime.py -q
```

If `__init__.py`, `adapter.py`, or [component.yaml](./component.yaml) changed, also run:

```bash
PYTHONPATH=src ./.venv/bin/pytest test/test_integrations.py -q
```

For runtime-facing behavior changes, deploy to `/twinr` and validate on the Pi with:

```bash
cd /twinr && PYTHONPATH=src ./.venv/bin/pytest test/test_integrations_email.py test/test_integration_runtime.py -q
```

## Coupling

`models.py` changes → also check:
- `adapter.py`
- `imap.py`
- `smtp.py`
- `src/twinr/integrations/runtime.py`
- `test/test_integrations_email.py`
- `test/test_integration_runtime.py`

`adapter.py` changes → also check:
- `src/twinr/integrations/catalog.py`
- `src/twinr/integrations/runtime.py`
- `test/test_integrations.py`
- `test/test_integrations_email.py`

`imap.py` changes → also check:
- `src/twinr/integrations/runtime.py`
- `test/test_integrations_email.py`
- `test/test_integration_runtime.py`

`smtp.py` changes → also check:
- `src/twinr/integrations/runtime.py`
- `test/test_integrations_email.py`
- `test/test_integration_runtime.py`

`__init__.py` changes → also check:
- `src/twinr/integrations/__init__.py`
- `src/twinr/integrations/runtime.py`
- `test/test_integrations.py`
- `test/test_integrations_email.py`

## Security

- Never log mailbox passwords, full email bodies, or raw provider payloads.
- Keep recipient, header, and preview normalization centralized in `models.py` and adapter validation; do not reintroduce unchecked parsing in transport modules.
- Maintain encrypted transports and bounded timeouts for both IMAP and SMTP; insecure auth must remain explicit and exceptional.

## Output expectations

- Preserve strict separation of concerns: normalization in `models.py`, integration orchestration in `adapter.py`, IMAP transport in `imap.py`, SMTP transport in `smtp.py`.
- Update [component.yaml](./component.yaml), [README.md](./README.md), and in-script docstrings when exports, invariants, or ownership change.
- Keep user-facing error summaries plain, short, and safe for voice output.
