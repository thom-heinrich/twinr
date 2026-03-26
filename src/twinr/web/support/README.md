# support

Shared support primitives for Twinr's local web control surface. This package
defines immutable web contracts, declared settings-field helpers, and hardened
file-backed read/write utilities used by routes and presenters.

## Responsibility

`support` owns:
- define reusable dashboard and settings contract objects
- build declared form fields for env-backed and managed-store settings and collect fail-closed updates
- parse form bodies and atomically read/write `.env` and text-backed settings
- persist the managed web sign-in state, bootstrap the local `admin/admin` login, and sign or validate portal session cookies
- keep compatibility wrappers for the generic channel-onboarding and WhatsApp pairing helpers that now live under `src/twinr/channels`
- validate WhatsApp wizard inputs and reuse the shared runtime-neutral pairing helpers from the channels package instead of duplicating onboarding logic inside the web package
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
| [`contracts.py`](./contracts.py) | Immutable dashboard, wizard, and settings models |
| [`forms.py`](./forms.py) | Declared settings-field builders and update filtering |
| [`store.py`](./store.py) | Atomic text and `.env` persistence helpers |
| [`auth.py`](./auth.py) | Managed web login state, password hashing, and signed session cookies |
| [`channel_onboarding.py`](./channel_onboarding.py) | Compatibility re-export for the runtime-neutral channel onboarding helpers in `twinr.channels.onboarding` |
| [`whatsapp.py`](./whatsapp.py) | Compatibility re-export for the shared WhatsApp pairing helpers in `twinr.channels.whatsapp.pairing` |

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
