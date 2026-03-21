# presenters

Presenter builders for Twinr's local web control surface. This package turns
config, runtime, ops, reminder, and integration state into template-ready data
for [`app.py`](../app.py) and [`context.py`](../context.py).

## Responsibility

`presenters` owns:
- shape route data into `SettingsSection`, `AdaptiveTimingView`, and template dictionaries
- normalize malformed persisted values into stable operator-safe display values
- validate managed integration form submissions before route handlers persist them
- validate managed smart-home Hue integration form submissions, including multi-bridge host lists and per-bridge secrets, before route handlers persist them
- build self-coding operator telemetry rows from persisted compile, activation, health, live-e2e, and lifecycle-control state
- build the tabbed `/ops/debug` operator page context from runtime, watchdog, memory-attestation, usage, hardware, and raw artifact state
- shape the `/ops/debug` memory-search tab, including grouped durable/midterm/episodic/conflict hits from the real long-term retrieval path
- shape the `/ops/debug` Conversation Lab tab from stored portal session traces without moving turn execution into templates
- build the `/connect/whatsapp` wizard context, including step status, guarded runtime probes, persisted channel-onboarding snapshots, live portal QR rendering, and operator-facing pairing/repair guidance
- build the compact WhatsApp setup summary for `/integrations`, while keeping the stateful QR pairing flow in the dedicated wizard
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
| [`integrations.py`](./integrations.py) | Email/calendar/smart-home sections, including Hue multi-bridge host/secret validation, plus the compact WhatsApp setup summary for `/integrations` |
| [`voice.py`](./voice.py) | Voice profile page helpers |
| [`ops.py`](./ops.py) | Ops row formatters and redaction |
| [`debug.py`](./debug.py) | Tabbed operator debug page context builder |
| [`conversation_lab.py`](./conversation_lab.py) | Presenter shaping for the interactive `/ops/debug` Conversation Lab tab |
| [`memory_search.py`](./memory_search.py) | Presenter shaping for the read-only `/ops/debug` memory-search tab |
| [`connect.py`](./connect.py) | Provider-routing section builders |
| [`whatsapp_wizard.py`](./whatsapp_wizard.py) | WhatsApp self-chat wizard page context with bounded pairing, live QR rendering, and live status rows |
| [`memory.py`](./memory.py) | Memory-related section builders |
| [`self_coding.py`](./self_coding.py) | Self-coding operator page context builder |

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
