# prompting

`prompting` owns the hidden instruction text and builder helpers for Twinr's
tool-capable agent lanes. It is the canonical package for assembling the
default, compact, supervisor, first-word, and specialist instruction bundles.

## Responsibility

`prompting` owns:
- define the canonical tool-lane instruction constants and fast ack phrases
- merge time and simple-setting context into tool-lane instruction bundles
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
