# integrations

`integrations` owns Twinr's shared integration layer. It defines the common
contracts, builtin manifests, safety policy, runtime/store wiring, and
dashboard metadata that provider packages plug into.

## Responsibility

`integrations` owns:
- define canonical manifests, requests, results, and safety metadata
- publish builtin manifest lookups for supported integration domains
- evaluate policy and dispatch requests to registered adapters
- assemble managed email/calendar adapters from store and environment state
- persist integration settings and expose integration-family blocks to `src/twinr/web`

`integrations` does **not** own:
- provider-specific email logic in [email/](./email)
- provider-specific calendar parsing and agenda logic in [calendar/](./calendar)
- web routes, forms, or templates outside integration-family data providers
- agent orchestration outside the shared integration boundary

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Stable package exports |
| [models.py](./models.py) | Canonical contracts and value objects |
| [catalog.py](./catalog.py) | Builtin manifest registry |
| [policy.py](./policy.py) | Safety gate decisions |
| [registry.py](./registry.py) | Adapter registration and dispatch |
| [runtime.py](./runtime.py) | Managed email/calendar wiring |
| [store.py](./store.py) | File-backed integration config store |
| [web_automation_families.py](./web_automation_families.py) | Dashboard family blocks |
| [calendar/](./calendar) | Calendar provider package |
| [email/](./email) | Email provider package |

## Usage

```python
from twinr.integrations import IntegrationRegistry, IntegrationRequest, SafeIntegrationPolicy

registry = IntegrationRegistry(adapters=(adapter,))
policy = SafeIntegrationPolicy(enabled_integrations={"calendar_agenda"})
result = registry.dispatch(
    IntegrationRequest(integration_id="calendar_agenda", operation_id="read_today"),
    policy=policy,
)
```

```python
from twinr.integrations import build_managed_integrations

runtime = build_managed_integrations(project_root=".")
readiness = runtime.readiness_for("email_mailbox")
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [calendar/README.md](./calendar/README.md)
- [email/README.md](./email/README.md)
