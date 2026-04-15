# handlers

`handlers` owns the concrete realtime tool handlers used during live turns. It
translates tool payloads into bounded runtime reads or mutations for automations,
smart-home, memory, reminders, browser automation, external service-connect
flows, guided user-discovery, output, self-coding, settings, voice-profile,
portrait-identity, shared household-identity flows, and
RSS/world-intelligence source configuration, and keeps shared
voice/argument guards close to that boundary. It also carries the focused
helper modules that multiple handlers reuse for best-effort telemetry and
automation parsing.

## Responsibility

`handlers` owns:
- implement per-tool handler functions called by `RealtimeToolExecutor`
- validate and normalize tool arguments before they reach runtime methods
- bridge the optional versioned browser-automation boundary into the realtime tool surface without importing concrete browser stacks into `src/`
- expose one bounded service-connect tool surface so a spoken request such as "verbinde mich mit WhatsApp" can start pairing and surface the QR/state on Twinr's right info panel
- expose one bounded WhatsApp-send tool surface that resolves remembered contacts, requires explicit final confirmation, and hands delivery off to the running channel service
- expose one smart-home tool surface for discovery, filtered state queries, low-risk control, and bounded stream inspection
- expose one explicit durable-memory read surface so exact recall of `remember_memory` facts goes through a bounded runtime read instead of hidden prompt text
- expose one temporary voice-quiet tool surface that suppresses transcript-first wake and automatic follow-up for a bounded runtime window without touching persistent settings
- keep handler-local telemetry and audit side effects best-effort, including search-turn journaling for requested model, actual model, fallback cause, and the bounded output-budget trace actually used by the provider
- keep handler-local telemetry and audit side effects best-effort, including machine-readable search verification and site-follow-up metadata so the specialist can route unresolved web answers into an explicit browser-permission step
- share sensitive-action confirmation and live-audio guard helpers
- translate local camera-driven portrait-enrollment results into structured guidance for the model
- expose one shared local household-identity tool surface for face, voice, status, and explicit confirm or deny feedback
- expose one bounded guided user-discovery tool surface for onboarding, lifelong profile learning, learned-fact review, and explicit correction or deletion
- bridge the self-coding front-stage flow plus learned-skill control/runtime hooks into deterministic ASE modules
- expose one explicit tool surface for RSS/world-intelligence subscription setup and occasional recalibration

`handlers` does **not** own:
- tool registry, binding, schemas, or prompt instruction assembly
- runtime, store, provider, or backend implementations
- workflow-loop orchestration or background delivery control
- web dashboard parsing or operator-facing UI behavior

## Key files

| File | Purpose |
|---|---|
| [automation_support.py](./automation_support.py) | Shared automation delivery, timezone, weekday, and tag parsing |
| [automations.py](./automations.py) | Automation CRUD handlers |
| [browser.py](./browser.py) | Optional browser-automation tool handler that validates payloads and calls the local browser boundary |
| [handler_telemetry.py](./handler_telemetry.py) | Shared best-effort telemetry, event, and usage helpers |
| [service_connect.py](./service_connect.py) | Bounded external service-connect handler that starts supported pairing flows such as WhatsApp |
| [whatsapp.py](./whatsapp.py) | Bounded WhatsApp send handler for remembered contacts with explicit confirmation and channel-owned delivery |
| [smarthome.py](./smarthome.py) | Smart-home discovery, filtered state-read, control, and stream handlers |
| [intelligence.py](./intelligence.py) | RSS/world-intelligence subscription and refresh-config handler |
| [memory.py](./memory.py) | Durable-memory tool handlers |
| [user_discovery.py](./user_discovery.py) | Guided get-to-know-you, review, correction, and deletion handler |
| [output.py](./output.py) | Print, search, and camera handlers |
| [household_identity.py](./household_identity.py) | Shared local household identity manager tool handler |
| [portrait_identity.py](./portrait_identity.py) | Local portrait-identity enrollment, status, and reset handlers |
| [reminders.py](./reminders.py) | Reminder scheduling handler |
| [self_coding.py](./self_coding.py) | Self-coding learning, activation, pause/reactivate, and hidden runtime handlers |
| [self_coding_support.py](./self_coding_support.py) | Shared self-coding runtime/cache/signature support for handler entrypoints |
| [settings.py](./settings.py) | Simple setting mutation handler |
| [support.py](./support.py) | Shared voice and argument guards |
| [voice_quiet.py](./voice_quiet.py) | Temporary runtime-only voice quiet-window handler |
| [voice_profile.py](./voice_profile.py) | Voice-profile tool handlers |
| [component.yaml](./component.yaml) | Structured package metadata |

## Usage

```python
from twinr.agent.tools.runtime.executor import RealtimeToolExecutor

executor = RealtimeToolExecutor(owner)
result = executor.handle_schedule_reminder(
    {"summary": "Tabletten nehmen", "due_at": "2026-03-17T08:00:00+01:00"}
)
```

```python
from twinr.agent.tools.handlers.self_coding import handle_propose_skill_learning

result = handle_propose_skill_learning(
    owner,
    {"name": "Daily Check-In", "action": "Ask how the user feels", "capabilities": ["speaker", "rules"]},
)
```

## See also

- [component.yaml](./component.yaml)
- [runtime](../runtime/README.md)
- [base-agent runtime](../../base_agent/runtime/README.md)
