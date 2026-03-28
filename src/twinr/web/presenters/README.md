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
- build the `/integrations/email` wizard context plus stepwise validated record updates for provider choice, mailbox login, transport, bounded connection-test state, and final guardrails
- build the compact email and WhatsApp setup summaries for `/integrations`, while keeping the stateful WhatsApp QR pairing flow in the dedicated wizard
- build the social-history learning consent/status panel for `/integrations`, including source selection, bounded lookback windows, and import-status copy
- own the flatter primary shell grouping, home destination cards, and grouped `/advanced` operator hub context so top-level navigation stays short and consistent
- own the calmer in-page grouping and overview-card composition for `/settings` and `/memory`, keeping everyday operator work ahead of deeper runtime detail
- feed templates that keep `/automations` and the specialized setup surfaces focused on overview first, guided work second, and advanced detail last
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
| [`common.py`](./common.py) | Shared reminder, provider-status, and safe file-link helpers |
| [`shell.py`](./shell.py) | Grouped primary shell navigation plus flatter Home, Settings shortcut, and Advanced hub card builders |
| [`settings.py`](./settings.py) | Grouped settings page builders |
| [`integrations.py`](./integrations.py) | Integration overview rows, compact email/WhatsApp setup summaries, and validated email/calendar/smart-home form builders |
| [`email_wizard.py`](./email_wizard.py) | Guided email mailbox setup wizard page context, bounded connection-test state, and stepwise record validation |
| [`social_history.py`](./social_history.py) | Presenter shaping and form validation for the `/integrations` social-history learning panel |
| [`voice.py`](./voice.py) | Voice profile page helpers |
| [`ops.py`](./ops.py) | Ops row formatters and redaction |
| [`debug.py`](./debug.py) | Tabbed operator debug page context builder |
| [`conversation_lab.py`](./conversation_lab.py) | Presenter shaping for the interactive `/ops/debug` Conversation Lab tab |
| [`memory_search.py`](./memory_search.py) | Presenter shaping for the read-only `/ops/debug` memory-search tab |
| [`connect.py`](./connect.py) | Provider-routing section builders |
| [`whatsapp_wizard.py`](./whatsapp_wizard.py) | WhatsApp self-chat wizard page context with bounded pairing, live QR rendering, and live status rows |
| [`memory.py`](./memory.py) | Activity & Memory overview cards, runtime metrics, and grouped memory section builders |
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
