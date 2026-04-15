# state

Internal package for the base agent's runtime state primitives: canonical
status/event definitions, deterministic transitions, and durable snapshot
persistence for runtime, display, web, and ops consumers.

## Responsibility

`state` owns:
- Define the canonical runtime statuses and events
- Validate deterministic state transitions and recovery paths
- Persist and restore runtime snapshots as normalized JSON
- Normalize snapshot timestamps, memory fields, voice-status fields, and the orthogonal `printing_active` flag
- Reapply persisted `status` plus orthogonal `printing_active` through `TwinrStateMachine.restore_snapshot_state()` so runtime bootstrap restores state without bypassing state-machine invariants, while fresh startup can deliberately refuse stale non-idle or `error` statuses
- accept both `answering -> listening` and the delayed `waiting -> listening` follow-up reopen so visible speech completion can clear immediately after playback without losing the later gated reopen
- Preserve local household voice-match metadata alongside the current voice-status snapshot
- Keep runtime snapshot files cross-service readable (`0644`) so the external remote-memory watchdog can inspect them without weakening runtime ownership
- Keep the runtime snapshot lock file on one canonical hidden path and make it cross-service writable (`0666`) so root-owned services and `thh`-run validation tools coordinate on the same snapshot guard

`state` does **not** own:
- Runtime orchestration or business flow decisions
- Memory summarization or long-term memory logic
- Display rendering or web view-model formatting
- Generic config/env persistence outside runtime snapshots

## Key files

| File | Purpose |
|---|---|
| `machine.py` | Define statuses, events, and transitions |
| `snapshot.py` | Persist and restore runtime snapshots |
| `__init__.py` | Mark package only |

## Usage

```python
from twinr.agent.base_agent.state.machine import TwinrEvent, TwinrStateMachine
from twinr.agent.base_agent.state.snapshot import RuntimeSnapshotStore

state_machine = TwinrStateMachine()
state_machine.transition(TwinrEvent.GREEN_BUTTON_PRESSED)

store = RuntimeSnapshotStore(config.runtime_state_path)
snapshot = store.save(
    status=state_machine.status.value,
    memory_turns=(),
    last_transcript=None,
    last_response=None,
)
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [runtime](../runtime/README.md)
- [display service](../../../display/service.py)
- [web context](../../../web/context.py)
