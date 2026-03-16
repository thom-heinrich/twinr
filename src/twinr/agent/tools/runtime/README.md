# runtime

`runtime` owns the runtime-side tool orchestration used during Twinr live turns.
It adapts workflow owners into handler call surfaces, validates the canonical
tool binding map, and runs the bounded tool loops that talk to provider
streaming APIs.

## Responsibility

`runtime` owns:
- adapt workflow owners into `handle_*` methods for realtime tools
- validate the canonical tool-name to handler binding registry
- run the generic streaming tool loop and the supervisor/specialist handoff loop
- emit speech-lane deltas and serialize tool results safely
- enforce the focused broker policy for background automation `tool_call` execution

`runtime` does **not** own:
- concrete tool business logic or payload normalization in `../handlers`
- tool schema definitions or prompt instruction text
- higher-level workflow/session orchestration above the tool-loop boundary
- provider backend implementations beyond the tool-calling contracts

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Runtime export surface |
| [executor.py](./executor.py) | Owner-to-handler dispatcher |
| [registry.py](./registry.py) | Tool binding validation |
| [broker_policy.py](./broker_policy.py) | Background automation tool-call policy |
| [streaming_loop.py](./streaming_loop.py) | Generic tool round loop |
| [dual_lane_loop.py](./dual_lane_loop.py) | Supervisor/specialist handoff loop |
| [component.yaml](./component.yaml) | Structured package metadata |

## Usage

```python
from twinr.agent.tools.runtime import (
    RealtimeToolExecutor,
    bind_realtime_tool_handlers,
)

executor = RealtimeToolExecutor(owner)
tool_handlers = bind_realtime_tool_handlers(executor)
```

```python
from twinr.agent.tools.runtime import DualLaneToolLoop

loop = DualLaneToolLoop(
    supervisor_provider=supervisor_provider,
    specialist_provider=specialist_provider,
    tool_handlers=tool_handlers,
    tool_schemas=tool_schemas,
    supervisor_instructions=supervisor_instructions,
    specialist_instructions=specialist_instructions,
)
```

## See also

- [component.yaml](./component.yaml)
- [handlers](../handlers/README.md)
- [schemas](../schemas/README.md)
- [streaming runner](../../workflows/streaming_runner.py)
