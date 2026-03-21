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
- render a bounded conversational self-expression and `MINDSHARE` layer so Twinr can naturally speak from ongoing themes without pretending to have a human inner life
- derive authoritative per-turn steering cues from shared-thread and appetite state so Twinr can decide more consistently when to briefly update, when one calm follow-up is acceptable, and when it should simply observe
- derive explicit per-topic positive-engagement actions such as `silent`, `hint`, `brief_update`, `ask_one`, and `invite_follow_up` so Twinr can foster welcomed interaction without becoming pushy
- serialize those steering cues for runtime evaluators with compact semantic match summaries and region-qualified local titles where available, then resolve matched topics back into concrete follow-up keep-open versus release-after-answer behavior
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
| [self_expression.py](./self_expression.py) | Conversational self-expression policy plus prompt-facing current-mindshare rendering |
| [positive_engagement.py](./positive_engagement.py) | Explicit positive-engagement actions derived generically from appetite, ongoing interest, and co-attention |
| [steering.py](./steering.py) | Authoritative turn-steering cues plus runtime follow-up resolution derived from co-attention and conversation appetite |
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
- Twinr's conversational self-expression stays bounded: it may speak from `MINDSHARE` when the user invites open conversation or asks what Twinr has been following, but it must not invent secret emotions, private off-screen experiences, or human-like inner life
- `MINDSHARE` surfacing stays generic and data-driven: selection may vary slightly via bounded stochasticity, but it must be based on source type and salience rather than hardcoded entity names or place-first special cases
- repeated user engagement with one topic may boost how often matching mindshare items surface and how strongly the RSS intelligence layer continues to watch that topic, but this must remain bounded and evidence-driven
- explicit structured reactions like follow-up requests may raise topic engagement faster than passive affinity, but the signal must stay typed and auditable
- boosted topic engagement must decay over time and let RSS tuning fall back to baseline when the user stops leaning into that topic
- Twinr's engagement model distinguishes `resonant`, `warm`, `uncertain`, `cooling`, and `avoid`; absence alone must not count as dislike unless repeated prior exposure and non-reengagement are structurally evidenced
- `MINDSHARE` now also derives a topic-specific `conversation appetite` from those engagement states plus the learned global style profile, so each surfaced theme carries bounded cues for `depth`, `follow-up`, and `proactivity` instead of relying on hidden prompt guesswork
- `positive_engagement.py` lifts that implicit guidance into explicit turn actions: each surfaced topic resolves generically to `silent`, `hint`, `brief_update`, `ask_one`, or `invite_follow_up`, and the policy must stay topic-generic instead of accreting hardcoded examples
- repeated user returns to a topic may also raise Twinr's persisted `ongoing interest` in it; this is the companion analogue of `stop/linger/return` signals in ranking systems, but bounded for dignity and calm rather than optimized for addiction or time-on-device
- when sustained user interest and Twinr's own feed coverage line up, the system now also tracks bounded `co-attention` so some themes can become genuine shared running threads rather than isolated prompt hints
- `steering.py` turns that same state into authoritative turn guidance inside `PERSONALITY`, serializes compact machine-readable cues for the closure evaluator, and resolves matched topics back into concrete runtime follow-up behavior without turning `MINDSHARE` itself into instruction authority
- steering cues should stay semantically narrow: if a durable signal already says a topic is local/regional and region-scoped, the cue title may carry that regional anchor so closure matching does not confuse nearby community topics with public-policy threads

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [../base_agent/prompting/README.md](../base_agent/prompting/README.md)
- [../../../../docs/persistent_personality_architecture.md](../../../../docs/persistent_personality_architecture.md)
