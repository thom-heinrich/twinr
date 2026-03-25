# prompting

`prompting` owns the hidden instruction text and builder helpers for Twinr's
tool-capable agent lanes. It is the canonical package for assembling the
default, compact, supervisor, first-word, and specialist instruction bundles.

## Responsibility

`prompting` owns:
- define the canonical tool-lane instruction constants and optional model-authored progress-line policy
- encode when the fast supervisor must carry structured `location_hint` / `date_context` fields and when it must escalate memory-dependent turns to fuller context
- merge time and simple-setting context into tool-lane instruction bundles
- build route-aware first-word overlays for authoritative local semantic-router handoffs so the instant spoken bridge stays LLM-generated and provisional
- encode the learned-skill tool policy for self-coding activation, rollback, pause, and reactivation
- encode that active self-coding follow-up answers and activation requests must reuse runtime-known `session_id` / `job_id` state instead of asking the user for internal ids
- encode when the model should discover, filter, aggregate, control, or inspect smart-home state through the dedicated tool surface
- encode that exact routed smart-home entity IDs spoken directly by the user may go straight to `read_smart_home_state` without a redundant relist step
- encode that the user's own smart-home inventory, room/device state, and recent in-home smart-home events are local tool work, not web-search work
- encode when the model should start discovery, continue discovery from free self-disclosure or wish-form self-statements about preferred name/addressing, treat the active discovery speaker as the current profile subject, prioritize explicit correction/deletion of already learned details over normal next-answer handling, review learned profile facts, split distinct learned details into separate discovery routes, and use replace/delete actions with exact reviewed fact ids instead of inventing ad-hoc profile narration
- encode the generic multi-query planning policy for broad smart-home and house-status questions so agents compose live status from multiple tool reads instead of a special-case summary path
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
