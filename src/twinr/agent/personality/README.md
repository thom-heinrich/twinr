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
- derive calm ambient display-impulse candidates from that same structured state so Twinr's evolving tone, topics, co-attention, and memory-follow-up needs can surface visually outside spoken turns
- correlate shown reserve-card exposures with later structured turn evidence so the right-hand lane can feed back into the same bounded engagement model as spoken conversation
- classify shown-card reactions more precisely, including immediate pickup versus delayed pickup or visible pushback, so Twinr can remember not just whether a card landed but how directly it landed
- serialize those steering cues for runtime evaluators with compact semantic match summaries and region-qualified local titles where available, then resolve matched topics back into concrete follow-up keep-open versus release-after-answer behavior
- provide storage seams for remote snapshot loading and saving without pushing persistence into base-agent prompting
- extract structured interaction, continuity, place, and world signals from long-term consolidation output and tool history
- gate background learning so repeated interaction, place, and world signals can evolve the durable personality snapshot without allowing one-off drift
- expose a narrow public engagement-signal read seam so non-prompt runtime consumers such as the ambient reserve bus can adapt from the same persisted state instead of rebuilding their own engagement view
- expose both synchronous commit paths and foreground-safe queueing paths so runtime turn finalization can hand off tool-history learning without blocking on remote-primary personality writes
- manage RSS/Atom subscriptions plus calm refresh/discovery state for place/world intelligence
- learn slow-changing topic/region source-calibration signals from structured conversation and tool evidence
- condense repeated feed updates into situational-awareness threads so world context can mature over days/weeks instead of staying headline-shaped
- decay stale world and relationship context so Twinr stays situationally aware without becoming news-stuck or overfitting old preferences

`personality` does **not** own:
- provider-specific prompt copy or wording
- memory, reminder, automation, or graph persistence implementations
- foreground runtime loop decisions about which dynamic sections belong to tool, supervisor, or main turns
- the fast supervisor lane's final prompt selection and routing policy

## Key files

| File | Purpose |
|---|---|
| [models.py](./models.py) | Typed evolving-personality state and prompt-layer models |
| [context_builder.py](./context_builder.py) | Ordered prompt-layer assembly |
| [self_expression.py](./self_expression.py) | Conversational self-expression policy plus prompt-facing current-mindshare rendering |
| [display_impulse_copy.py](./display_impulse_copy.py) | Deterministic question-first fallback copy for ambient display impulses when the bounded reserve-lane LLM rewrite is unavailable |
| [display_impulses.py](./display_impulses.py) | Silent ambient display-impulse candidates derived generically from mindshare and positive-engagement state, including structured generation context plus a generic candidate-family label for reserve-bus planning mix |
| [ambient_feedback.py](./ambient_feedback.py) | Correlate shown reserve-card exposures with later structured turn evidence and turn those reactions into generic engagement signals |
| [positive_engagement.py](./positive_engagement.py) | Explicit positive-engagement actions derived generically from appetite, ongoing interest, and co-attention |
| [steering.py](./steering.py) | Authoritative turn-steering cues plus runtime follow-up resolution derived from co-attention and conversation appetite |
| [store.py](./store.py) | Remote snapshot persistence seam for prompt state, signals, and deltas |
| [evolution.py](./evolution.py) | Policy-gated background learning loop, contradiction/sensitivity gates, and decay maintenance |
| [signals.py](./signals.py) | Structured signal taxonomy plus extraction for style, humor, topic, continuity, place, and world evidence |
| [learning.py](./learning.py) | Runtime-facing bridge from extracted signals into background evolution |
| [service.py](./service.py) | Prompting-facing integration service plus the public read seam for persisted engagement signals used by the ambient reserve bus |
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

```python
learning.enqueue_tool_history(
    tool_calls=(tool_call,),
    tool_results=(tool_result,),
)
# later, one background commit path flushes the queued signals
learning.flush_pending()
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
- `display_impulses.py` and `display_impulse_copy.py` reuses the same generic state for non-voice ambient display moments: reserve-card candidates must stay positive, question-first, and generic instead of hardcoding specific places, topics, or benchmark phrases
- reserve-card candidates should stay companion-like rather than search-box-like: raw `live_search` residue may still shape durable engagement state, but it should not surface directly as a visible right-lane opener
- `display_impulses.py` now also carries one generic `candidate_family` per reserve-card candidate so the right-hand bus can mix world/place/memory/social families without ad-hoc named-topic spacing rules
- ambient reserve-card copy should prefer engaging questions and short natural continuations over topic-label chrome, but it still must carry one clear topical anchor so the user immediately knows which thread Twinr wants to open
- ambient reserve-card copy may also surface calm memory-improving prompts such as open-thread follow-ups, clarifying questions, or consistency checks, but only when those come from structured durable state rather than ad-hoc string hacks
- reserve-lane generation should receive generic structured context for those memory or shared-thread prompts, so the visible wording can stay natural and personality-shaped instead of echoing raw topic labels
- `self_expression.py` and `display_impulses.py` should hand reserve generation the real thread/topic summary itself, not internal English scaffolding like durable-attention wrappers, so visible German copy can stay grounded in the actual topic instead of paraphrasing implementation noise
- shown reserve cards now participate in the same generic engagement loop: later structured turn evidence may resolve one card into pickup, cooling, aversion, or repeated non-pickup, and those outcomes flow back through the normal personality/world-intelligence evolution path
- immediate shown-card pickup should count as stronger positive evidence than a much later return, and the structured feedback path may also emit a short-lived reserve-bus feedback hint so the same-day display plan can adapt faster without hardcoding topics in the runtime loop
- repeated user returns to a topic may also raise Twinr's persisted `ongoing interest` in it; this is the companion analogue of `stop/linger/return` signals in ranking systems, but bounded for dignity and calm rather than optimized for addiction or time-on-device
- when sustained user interest and Twinr's own feed coverage line up, the system now also tracks bounded `co-attention` so some themes can become genuine shared running threads rather than isolated prompt hints
- `steering.py` turns that same state into authoritative turn guidance inside `PERSONALITY`, serializes compact machine-readable cues for the closure evaluator, and resolves matched topics back into concrete runtime follow-up behavior without turning `MINDSHARE` itself into instruction authority
- steering cues should stay semantically narrow: if a durable signal already says a topic is local/regional and region-scoped, the cue title may carry that regional anchor so closure matching does not confuse nearby community topics with public-policy threads
- the fast supervisor lane should consume only a lean subset of this package: stable character/style may flow there, but volatile layers such as `MINDSHARE`, `CONTINUITY`, `PLACE`, `WORLD`, and `REFLECTION` must stay out so noisy search routing does not get semantically dragged by ambient state

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [../base_agent/prompting/README.md](../base_agent/prompting/README.md)
- [../../../../docs/persistent_personality_architecture.md](../../../../docs/persistent_personality_architecture.md)
