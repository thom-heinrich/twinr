# support

Shared support primitives for Twinr's local web control surface. This package
defines immutable web contracts, env-backed field helpers, and hardened
file-backed read/write utilities used by routes and presenters.

## Responsibility

`support` owns:
- define reusable dashboard and settings contract objects
- build declared env-backed form fields and collect fail-closed updates
- parse form bodies and atomically read/write `.env` and text-backed settings
- expose the curated support import surface through [`__init__.py`](./__init__.py)

`support` does **not** own:
- FastAPI route definitions, redirects, or page composition
- page-specific presenter defaults, normalization, or integration validation
- template markup, CSS, or client-side behavior
- runtime hardware, provider, or memory orchestration

## Key files

| File | Purpose |
|---|---|
| [`__init__.py`](./__init__.py) | Curated support export surface |
| [`contracts.py`](./contracts.py) | Immutable dashboard and settings models |
| [`forms.py`](./forms.py) | Env-backed field builders and update filtering |
| [`store.py`](./store.py) | Atomic text and `.env` persistence helpers |

## Usage

```python
from twinr.web.support import parse_urlencoded_form, read_env_values
from twinr.web.support.contracts import SettingsSection
from twinr.web.support.forms import _collect_standard_updates

form = parse_urlencoded_form(request_body)
updates = _collect_standard_updates(form)
env_values = read_env_values(env_path)
section = SettingsSection(title="General", description="...", fields=())
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [`app.py`](../app.py)
- [`presenters`](../presenters/README.md)
