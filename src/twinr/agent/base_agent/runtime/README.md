# runtime

`runtime` owns the canonical `TwinrRuntime` object for the base agent. It
assembles focused mixins for lifecycle setup, state transitions, provider
context, structured memory, guided user-discovery, reminders and automations,
and runtime snapshot durability.

## Responsibility

`runtime` owns:
- compose `TwinrRuntime` from focused mixins
- bootstrap and shut down runtime-owned stores and services
- drive state transitions for listening, answering, printing, and failures
- rearm follow-up listening directly from `answering` when a conversation stays open after a spoken reply
- assemble provider-facing context, adaptive timing, voice guidance, and active guided-discovery state hints for tool-capable turns
- normalize legacy voice-status labels before provider guidance so low-risk discovery turns do not accidentally fall back to conservative unknown-speaker wording
- surface the currently visible reserve-lane card as bounded grounding context for supervisor/search lanes when the user is clearly reacting to what Twinr is showing, including a stronger per-turn authoritative overlay for fast supervisor decisions
- keep the first-word lane on bounded local context so direct replies never block on remote long-term retrieval
- keep the supervisor fast lane on the same remote-free path while still surfacing one local on-device memory summary
- mediate on-device memory, durable memory, reminders, automations, and snapshots
- expose reminder-reservation release semantics so workflow races can unwind a reserved reminder before speech starts without forcing a fake failure retry delay
- expose the runtime-side guided user-discovery flow so Twinr can persist, review, correct, and delete high-value get-to-know-you facts across managed context, graph memory, and durable memory without bloating orchestration code
- expose active self-coding dialogue and compile-job guidance to the tool lane so follow-up answers and activation requests can reuse runtime-owned `session_id` / `job_id` state instead of asking the user for internal identifiers
- forward completed tool history into long-term-backed personality learning after a turn is finalized
- keep the explicit best-effort `flush_long_term_memory()` API bounded to the caller's timeout while reserving the stricter remote-primary minimum only for long-term object/graph/midterm durability paths, not already-attested prompt-context writes
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
| [discovery.py](./discovery.py) | Runtime bridge for the guided user-discovery service |
| [self_coding.py](./self_coding.py) | Runtime bridge for active self-coding dialogue and compile-job guidance |
| [display_grounding.py](./display_grounding.py) | Read the active reserve-lane cue and turn it into bounded provider-grounding messages plus an authoritative per-turn supervisor overlay |
| [memory.py](./memory.py) | Mutate structured and durable memory |
| [automation.py](./automation.py) | Manage reminders and automations, including releasing reserved reminders when delivery is aborted before output starts |
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
- [realtime_runner.py](../../workflows/realtime_runner.py)
