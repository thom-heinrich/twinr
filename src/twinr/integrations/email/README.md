# email

`email` owns Twinr's email integration package. It provides canonical contact,
draft, and mailbox-summary models plus the adapter, IMAP reader, and SMTP
sender used by the managed integration runtime.

## Responsibility

`email` owns:
- define canonical email normalization, approved contacts, drafts, and summaries
- translate integration requests into bounded read, draft, and send actions
- read recent mailbox summaries from IMAP providers
- send validated drafts through SMTP with explicit transport checks

`email` does **not** own:
- integration settings loading or readiness assembly in [runtime.py](../runtime.py)
- integration storage or web form normalization under `src/twinr/web`
- catalog metadata or non-email integration behavior elsewhere in `src/twinr/integrations`

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Public email exports |
| [models.py](./models.py) | Canonical email models |
| [adapter.py](./adapter.py) | Integration request adapter |
| [imap.py](./imap.py) | IMAP mailbox reader |
| [smtp.py](./smtp.py) | SMTP sender |
| [component.yaml](./component.yaml) | Structured package metadata |

## Usage

```python
from twinr.integrations.email import ApprovedEmailContacts, EmailContact, normalize_email

address = normalize_email("Anna@Example.com")
contacts = ApprovedEmailContacts(
    (EmailContact(email=address, display_name="Anna"),)
)
```

```python
from twinr.integrations.email import EmailDraft

draft = EmailDraft(
    to=("anna@example.com",),
    subject="Check-in",
    body="Ich bin zuhause.",
)
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [runtime.py](../runtime.py)
- [test_integrations_email.py](../../../../test/test_integrations_email.py)
