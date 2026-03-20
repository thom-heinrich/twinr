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
- rearm follow-up listening directly from `answering` when a conversation stays open after a spoken reply
- assemble provider-facing context, adaptive timing, and voice guidance
- keep the first-word lane on bounded local context so direct replies never block on remote long-term retrieval
- keep the supervisor fast lane on the same remote-free path while still surfacing one local on-device memory summary
- mediate on-device memory, durable memory, reminders, automations, and snapshots
- forward completed tool history into long-term-backed personality learning after a turn is finalized
- expose runtime-side RSS/world-intelligence configuration so persistent feed subscriptions can shape Twinr's calm place/world awareness

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
runtime.rearm_follow_up()
runtime.shutdown(timeout_s=1.0)
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [conversation](../conversation/README.md)
- [state](../state/README.md)
- [runner.py](../../workflows/runner.py)
- [realtime_runner.py](../../workflows/realtime_runner.py)
