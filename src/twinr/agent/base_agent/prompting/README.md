# prompting

`prompting` assembles the base agent's hidden instruction context from personality
files, structured agent-personality layers, and selected runtime state. It exposes the canonical loaders that provider,
workflow, and conversation callers use when they need ordered instruction strings.

## Responsibility

`prompting` owns:
- load `SYSTEM`, `PERSONALITY`, and `USER` sections from the configured personality directory
- merge those legacy sections with structured layers from [`../../personality`](../../personality/README.md) when a typed personality snapshot is available, including contextual `MINDSHARE` when Twinr has ongoing themes it may naturally speak from
- merge memory, reminder, and automation context into loop-specific section sets
- cache rendered instruction bundles against local source signatures so steady-state Pi turns do not re-fetch unchanged prompt context from remote storage on every turn
- turn ordered sections into model-facing instruction strings
- keep turn-controller lane instructions separate from provider-owned tool-loop base instructions so that lane-specific evaluators do not accidentally prepend the same base bundle twice
- keep the conversation-closure loader lean by reading only the dedicated closure-controller instruction file instead of the full tool-loop bundle
- provide small instruction-composition helpers shared across callers

`prompting` does **not** own:
- provider-specific prompt text or phrasing
- storage logic for memory, reminders, automations, or remote snapshots
- conversation policy beyond selecting which instruction bundle to load

## Key files

| File | Purpose |
|---|---|
| [personality.py](./personality.py) | Canonical context loaders |
| [component.yaml](./component.yaml) | Structured package metadata |
| [AGENTS.md](./AGENTS.md) | Local invariants and checks |

## Usage

```python
from twinr.agent.base_agent.prompting.personality import (
    load_personality_instructions,
    load_tool_loop_instructions,
)

base_instructions = load_personality_instructions(config)
tool_instructions = load_tool_loop_instructions(config)
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [conversation](../conversation/README.md)
- [../../personality](../../personality/README.md)
- [tools/prompting](../../tools/prompting/README.md)
