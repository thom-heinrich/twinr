# runtime

`runtime` owns the canonical `TwinrRuntime` object for the base agent. It
assembles focused mixins for lifecycle setup, state transitions, provider
context, structured memory, reminders and automations, and runtime snapshot
durability.

## Responsibility

`runtime` owns:
- compose `TwinrRuntime` from focused mixins
- bootstrap and shut down runtime-owned stores and services
- drive state transitions for listening, answering, printing, and failures
- assemble provider-facing context, adaptive timing, and voice guidance
- mediate on-device memory, durable memory, reminders, automations, and snapshots

`runtime` does **not** own:
- workflow loops or hardware/audio orchestration
- provider transport or instruction-text assembly
- reminder, automation, graph-memory, or snapshot store implementations
- state-machine definitions outside the runtime-facing methods

## Key files

| File | Purpose |
|---|---|
| [runtime.py](./runtime.py) | Compose `TwinrRuntime` |
| [base.py](./base.py) | Bootstrap services and shutdown |
| [flow.py](./flow.py) | Drive turn and print flow |
| [context.py](./context.py) | Build provider and timing context |
| [memory.py](./memory.py) | Mutate structured and durable memory |
| [automation.py](./automation.py) | Manage reminders and automations |
| [snapshot.py](./snapshot.py) | Persist and restore runtime state |
| [component.yaml](./component.yaml) | Structured package metadata |

## Usage

```python
from twinr.agent.base_agent import TwinrConfig, TwinrRuntime

runtime = TwinrRuntime(config=TwinrConfig())
runtime.begin_listening(request_source="button", button="green")
runtime.submit_transcript("Bitte erinnere mich morgen an meine Tabletten.")
runtime.complete_agent_turn("Ich erinnere dich morgen daran.")
runtime.shutdown(timeout_s=1.0)
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [conversation](../conversation/README.md)
- [state](../state/README.md)
- [runner.py](../../workflows/runner.py)
- [realtime_runner.py](../../workflows/realtime_runner.py)
