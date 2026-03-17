# handlers

`handlers` owns the concrete realtime tool handlers used during live turns. It
translates tool payloads into bounded runtime reads or mutations for automations,
memory, reminders, output, self-coding, settings, and voice-profile flows, and
keeps shared voice/argument guards close to that boundary.

## Responsibility

`handlers` owns:
- implement per-tool handler functions called by `RealtimeToolExecutor`
- validate and normalize tool arguments before they reach runtime methods
- keep handler-local telemetry and audit side effects best-effort
- share sensitive-action confirmation and live-audio guard helpers
- bridge the self-coding front-stage flow plus learned-skill control/runtime hooks into deterministic ASE modules

`handlers` does **not** own:
- tool registry, binding, schemas, or prompt instruction assembly
- runtime, store, provider, or backend implementations
- workflow-loop orchestration or background delivery control
- web dashboard parsing or operator-facing UI behavior

## Key files

| File | Purpose |
|---|---|
| [automations.py](./automations.py) | Automation CRUD handlers |
| [memory.py](./memory.py) | Durable-memory tool handlers |
| [output.py](./output.py) | Print, search, and camera handlers |
| [reminders.py](./reminders.py) | Reminder scheduling handler |
| [self_coding.py](./self_coding.py) | Self-coding learning, activation, pause/reactivate, and hidden runtime handlers |
| [settings.py](./settings.py) | Simple setting mutation handler |
| [support.py](./support.py) | Shared voice and argument guards |
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
