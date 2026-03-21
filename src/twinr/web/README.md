# web

FastAPI package for Twinr's local control surface. This directory owns the app
factory, request-time web context, automation-page helpers, and the package-local
templates and CSS that power the operator dashboard.

## Responsibility

`web` owns:
- assemble the local FastAPI app, middleware, and page/form routes
- load config, runtime snapshot, and store handles through [`context.py`](./context.py)
- enforce local or managed sign-in policy for the portal, including first-login password change for the permanent Pi web service over LAN
- keep state-changing form requests fail-closed while still accepting trusted same-origin browser posts behind an HTTPS reverse proxy
- render the ops health page with the persisted remote-memory watchdog state
- render the tabbed debug page that groups runtime, ChonkyDB, live memory attestation, LLM, events, hardware, and raw local artifacts
- render a read-only long-term memory search tab in `/ops/debug` that queries the real retrieval stack and groups hits by durable, midterm, episodic, and conflict memory
- run the interactive `/ops/debug` Conversation Lab tab that sends a real text turn through Twinr's provider/tool/memory path and persists human-readable routing, tool, retrieval, and memory-write traces
- support camera-backed Conversation Lab tool turns for portrait enrollment and `inspect_camera` without pushing camera orchestration into the route layer
- keep the `/ops/debug` operator surface high-contrast and readable on external monitors, including dense log-like sections
- render the self-coding operator page with compile telemetry, health, live-e2e state, stale compile/run watchdog visibility, and learned-skill lifecycle controls
- render the `/connect/whatsapp` wizard that guides one internal WhatsApp self-chat setup through allowlist, runtime, a bounded pairing window, and the final self-chat test
- persist per-channel onboarding snapshots so web-driven channel setup can show live QR-needed, paired, reconnect, and repair-needed status without moving runtime state into templates
- compose presenters, support helpers, templates, and static assets into operator pages
- persist safe web-driven changes for settings, reminders, automations, integrations, personality, and user context, including Hue-backed smart-home settings on `/integrations`

`web` does **not** own:
- Twinr runtime, provider, memory, or hardware business logic outside page orchestration
- presenter-specific section shaping or integration normalization in [`presenters`](./presenters/README.md)
- generic file-backed contracts, form declarations, or persistence helpers in [`support`](./support/README.md)
- Raspberry Pi bootstrap/setup scripts or non-web operator workflows

## Key files

| File | Purpose |
|---|---|
| [`__init__.py`](./__init__.py) | Export `create_app` |
| [`app.py`](./app.py) | FastAPI factory and route handlers |
| [`conversation_lab.py`](./conversation_lab.py) | Portal conversation-lab session store and real text-turn execution helper |
| [`conversation_lab_vision.py`](./conversation_lab_vision.py) | Camera and vision helpers used by Conversation Lab tool owners |
| [`context.py`](./context.py) | Shared loaders and template rendering |
| [`automations.py`](./automations.py) | Automation page/form helpers |
| [`support/channel_onboarding.py`](./support/channel_onboarding.py) | Generic file-backed onboarding snapshots and bounded in-process pairing registry |
| [`support/whatsapp.py`](./support/whatsapp.py) | WhatsApp wizard validation, runtime probes, and bounded pairing coordination |
| [`presenters`](./presenters/README.md) | Template-ready section builders |
| [`support`](./support/README.md) | Shared contracts and file helpers |
| [`templates`](./templates/) | Jinja page templates |
| [`static`](./static/) | Dashboard CSS assets |

## Usage

```python
from pathlib import Path

from twinr.web import create_app

app = create_app(Path(".env"))
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [`src/twinr/__main__.py`](../__main__.py)
- [`presenters`](./presenters/README.md)
- [`support`](./support/README.md)
