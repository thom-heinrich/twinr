# personality

`personality` owns the typed evolving companion-state layer for `twinr.agent`.
It sits between long-term storage and base-agent prompting: the package defines
structured personality models, translates them into ordered prompt layers, and
now also owns the policy-gated background evolution path that persists signals
and snapshot commits through remote-primary state.

## Responsibility

`personality` owns:
- define typed models for stable character, learned conversation style, humor, relationship context, continuity, place/world awareness, and reflection deltas
- build ordered prompt layers that preserve Twinr's legacy `SYSTEM` / `PERSONALITY` / `USER` contract
- provide storage seams for remote snapshot loading and saving without pushing persistence into base-agent prompting
- extract structured interaction, continuity, place, and world signals from long-term consolidation output and tool history
- gate background learning so repeated interaction, place, and world signals can evolve the durable personality snapshot without allowing one-off drift
- manage RSS/Atom subscriptions plus calm refresh/discovery state for place/world intelligence
- learn slow-changing topic/region source-calibration signals from structured conversation and tool evidence
- condense repeated feed updates into situational-awareness threads so world context can mature over days/weeks instead of staying headline-shaped
- decay stale world and relationship context so Twinr stays situationally aware without becoming news-stuck or overfitting old preferences

`personality` does **not** own:
- provider-specific prompt copy or wording
- memory, reminder, automation, or graph persistence implementations
- foreground runtime loop decisions about which dynamic sections belong to tool, supervisor, or main turns

## Key files

| File | Purpose |
|---|---|
| [models.py](./models.py) | Typed evolving-personality state and prompt-layer models |
| [context_builder.py](./context_builder.py) | Ordered prompt-layer assembly |
| [store.py](./store.py) | Remote snapshot persistence seam for prompt state, signals, and deltas |
| [evolution.py](./evolution.py) | Policy-gated background learning loop, contradiction/sensitivity gates, and decay maintenance |
| [signals.py](./signals.py) | Structured signal taxonomy plus extraction for style, humor, topic, continuity, place, and world evidence |
| [learning.py](./learning.py) | Runtime-facing bridge from extracted signals into background evolution |
| [service.py](./service.py) | Prompting-facing integration service |
| [intelligence/](./intelligence) | RSS/world-intelligence subscriptions, calibration signals, reflection-phase recalibration, and refresh-to-world-signal conversion |
| [component.yaml](./component.yaml) | Structured package metadata |
| [AGENTS.md](./AGENTS.md) | Local invariants and verification |

## Usage

```python
from twinr.agent.personality import PersonalityContextService

service = PersonalityContextService()
sections = service.build_static_sections(
    legacy_sections=(("SYSTEM", "Base system"), ("PERSONALITY", "Base style")),
    config=config,
    remote_state=remote_state,
)
```

```python
from twinr.agent.personality import (
    PersonalityLearningService,
)

learning = PersonalityLearningService.from_config(config, remote_state=remote_state)
result = learning.record_tool_history(
    tool_calls=(tool_call,),
    tool_results=(tool_result,),
)
```

## Learning model

The current signal taxonomy is intentionally structured and conservative:

- `topic_affinity` and `topic_aversion` shape relationship salience only after repeated support
- `verbosity_preference` and `initiative_preference` update a separate conversation-style profile instead of rewriting core traits
- `humor_feedback` may move humor intensity slowly, but sensitive-context turns never reinforce humor upward implicitly
- `thread` summaries refresh continuity directly and expire over time
- `world` signals are freshness-bound and decay out of the prompt once stale
- RSS-backed world intelligence is separate from ad-hoc live search: discovery is occasional and explicit, calibration learns slowly from structured conversation/tool evidence, refresh is cadence-bound, and feed items become contextual `WORLD`/`CONTINUITY` updates plus slower situational-awareness threads

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [../base_agent/prompting/README.md](../base_agent/prompting/README.md)
- [../../../../docs/persistent_personality_architecture.md](../../../../docs/persistent_personality_architecture.md)
