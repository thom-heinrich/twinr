# presenters

Presenter builders for Twinr's local web control surface. This package turns
config, runtime, ops, reminder, and integration state into template-ready data
for [`app.py`](../app.py) and [`context.py`](../context.py).

## Responsibility

`presenters` owns:
- shape route data into `SettingsSection`, `AdaptiveTimingView`, and template dictionaries
- normalize malformed persisted values into stable operator-safe display values
- validate managed integration form submissions before route handlers persist them
- expose the presenter import surface through [`__init__.py`](./__init__.py)

`presenters` does **not** own:
- FastAPI routes, redirects, or request parsing
- template markup, CSS, or client-side behavior
- store implementations for env, integrations, reminders, or ops logs
- hardware/runtime orchestration outside narrow guarded helpers

## Key files

| File | Purpose |
|---|---|
| [`__init__.py`](./__init__.py) | Curated presenter export surface |
| [`common.py`](./common.py) | Shared nav and reminder helpers |
| [`settings.py`](./settings.py) | Global settings page builders |
| [`integrations.py`](./integrations.py) | Integration sections and validators |
| [`voice.py`](./voice.py) | Voice profile page helpers |
| [`ops.py`](./ops.py) | Ops row formatters and redaction |
| [`connect.py`](./connect.py) | Provider-routing section builders |
| [`memory.py`](./memory.py) | Memory-related section builders |

## Usage

```python
from twinr.web.presenters import _settings_sections, _voice_profile_page_context

sections = _settings_sections(config, env_values)
page_context = _voice_profile_page_context(config, snapshot)
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [`app.py`](../app.py)
- [`context.py`](../context.py)
