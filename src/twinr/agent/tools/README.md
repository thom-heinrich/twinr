# tools

`tools` owns the root package surface for Twinr's tool-capable agent stack. It
provides one import point for tool-lane prompting, tool schemas, and runtime
tool orchestration while leaving concrete handler modules internal.

## Responsibility

`tools` owns:
- expose the stable `twinr.agent.tools` import surface
- route callers toward the correct child package for prompting, schemas, runtime, or handlers
- preserve the package boundary that keeps concrete handlers off the public API
- carry the canonical schema/instruction surface for reminder, automation, memory, and self-coding tools, including learned-skill activation and control
- expose the route-aware first-word overlay builder used when the local semantic router short-circuits into `web`, `memory`, or `tool` handoffs

`tools` does **not** own:
- concrete tool-side business logic or guard helpers
- prompt-text authoring details
- schema rule definitions
- runtime loop implementation details

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Root export surface |
| [prompting/README.md](./prompting/README.md) | Tool instruction package |
| [schemas/README.md](./schemas/README.md) | Tool schema package |
| [runtime/README.md](./runtime/README.md) | Runtime orchestration package |
| [handlers/README.md](./handlers/README.md) | Internal handler package |
| [component.yaml](./component.yaml) | Structured package metadata |

## Usage

```python
from twinr.agent.tools import (
    RealtimeToolExecutor,
    bind_realtime_tool_handlers,
    build_realtime_tool_schemas,
    build_tool_agent_instructions,
)

instructions = build_tool_agent_instructions(config)
tool_schemas = build_realtime_tool_schemas(None)
executor = RealtimeToolExecutor(owner)
tool_handlers = bind_realtime_tool_handlers(executor)
```

## See also

- [component.yaml](./component.yaml)
- [prompting](./prompting/README.md)
- [schemas](./schemas/README.md)
- [runtime](./runtime/README.md)
- [handlers](./handlers/README.md)
