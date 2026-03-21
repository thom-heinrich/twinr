# prompting

`prompting` owns the hidden instruction text and builder helpers for Twinr's
tool-capable agent lanes. It is the canonical package for assembling the
default, compact, supervisor, first-word, and specialist instruction bundles.

## Responsibility

`prompting` owns:
- define the canonical tool-lane instruction constants and fast ack phrases
- encode when the fast supervisor must carry structured `location_hint` / `date_context` fields and when it must escalate memory-dependent turns to fuller context
- merge time and simple-setting context into tool-lane instruction bundles
- encode the learned-skill tool policy for self-coding activation, rollback, pause, and reactivation
- encode when the model should discover, filter, aggregate, control, or inspect smart-home state through the dedicated tool surface
- expose public builders for default, compact, supervisor, first-word, and specialist lanes

`prompting` does **not** own:
- personality file loading or section ordering in `base_agent/prompting`
- tool schema definitions or handler binding
- workflow-loop orchestration or provider transport behavior

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Prompting export surface |
| [instructions.py](./instructions.py) | Canonical tool-lane instruction builders |
| [component.yaml](./component.yaml) | Structured package metadata |

## Usage

```python
from twinr.agent.tools.prompting import build_tool_agent_instructions

instructions = build_tool_agent_instructions(config)
```

## See also

- [component.yaml](./component.yaml)
- [base_agent/prompting](../../base_agent/prompting/README.md)
- [schemas](../schemas/README.md)
