# retrieval

Assemble prompt-ready long-term memory recall context and silent
personalization cues for a single user turn.

## Responsibility

`retrieval` owns:
- assemble durable, episodic, mid-term, graph, and conflict context
- keep conflict-context supporting object loads on the same bounded selection-hydration contract as the upstream conflict selector, so one queue render does not rehydrate per-item documents after the relevant ids are already known
- merge original-language and canonical-English query variants during store-backed recall so optional rewrites improve recall without suppressing same-language memories
- keep that cross-variant merge bounded but wide enough that a noisy first-language hit set does not starve better canonical-query episodic or durable matches before reranking
- rerank durable recall across merged query variants so confirmed/current facts surface ahead of generic siblings for meta-memory questions
- render durable-memory prompt blocks with explicit `status`/`confirmed_by_user` semantics so live meta-memory questions reliably distinguish active user-confirmed facts from generic or unconfirmed siblings
- rerank episodic recall across merged query variants as well, so recent distractors from the first query wording do not crowd out the actually relevant episode carried by a later canonical rewrite
- compile adaptive mid-term policy hints from confirmed memory, recurring routines, and proactive success/skip history
- compile explicit confirmed response-channel policy packets so confirmed ReSpeaker-derived channel preferences can guide later delivery behavior without promoting mere observations
- compile persistent restart-recall policy packets from stable durable memories so fresh runtime roots retain a small provenance-rich continuity layer
- surface room-agnostic smart-home environment reflections and recent environment packets as ambient behavior context while preserving their uncertainty markers, quality states, and transition/new-normal classifications
- compile and render silent personalization cues from graph and episodic memory, including graph-only turns where no episodic match is available
- opportunistically reuse or background-build expensive silent subtext packets so live foreground turns do not stall on cold personalization compilation
- build one tiny fast-topic hint block from current-scope ChonkyDB objects for latency-sensitive answer lanes without running the full retriever
- build one internal unified retrieval plan across episodic entries, durable objects, conflict queues, selected midterm packets, adaptive policy packets, and graph candidates before rendering the stable public context sections
- keep that unified plan on a minimal kept-evidence contract: reorder candidates, prune non-focus components, drop non-dominant source families when a query clearly resolves to continuity vs practical recall, and expose dropped-candidate reasons in the internal plan for debugging
- rescue strong anchorless practical multimodal patterns when the only explainable focus anchor comes from continuity-only thread/episode evidence, so routine patterns are not pruned as `out_of_focus_component` just because a conversation turn carried the first explainable anchor
- rescue strong practical multimodal patterns when provisional focus anchors come only from graph/plan/place continuity noise and no practical candidate contributed to those anchors, so selected routine evidence is not pruned just because graph focus hardened first
- promote graph/plan-driven multimodal queries from `continuity` to at least `mixed` once the planner already rescued strong practical routine candidates outside those anchors, so combined weather/support questions keep both the episodic half and the durable routine half
- keep continuity support fail-closed on explainable anchor alignment once graph or practical focus is known, so phone/address queries cannot keep anchorless episodic distractors just because generic words overlap
- keep one support-aligned episodic continuity item for practical or mixed multimodal queries when graph-dominant focus anchors would otherwise starve the practical lane, so mixed weather/button or camera/support questions can keep the right episode and durable pattern together
- keep one lexically grounded episodic continuity item for practical or mixed routine queries that have no explainable focus anchors at all, so button/print/weather routines can still carry the right episode instead of dropping every episodic candidate as `missing_focus_anchor`
- treat direct non-user graph hits as real focus anchors for graph-heavy entity queries, and only rescue anchorless continuity candidates when focus is still generic or absent, so graph-only contact questions do not drag in unrelated midterm packets
- derive exact graph contact-attribute intent from both direct graph hits and normalized query terms, and fail closed to graph-only when no same-entity support exists, so `Anna Becker email` cannot leak unrelated durable, episodic, or conflict sections just because the graph matched only the owning person node
- treat broad semantic domains such as `memory_domain:*` as explainability hints only, not as candidate-connectivity edges, so unrelated memories in the same domain cannot collapse into one giant retrieval component
- reuse the graph selected by that unified plan for subtext compilation so one turn does not issue a second graph-selection pass just to render silent personalization
- suppress raw graph prompt blocks in the normal provider path when unified subtext planning found no relevant graph payload and no other recall sections survived, so graph-only control queries do not over-personalize replies with irrelevant person/preference context
- derive tool-facing redacted context from that same already-loaded retrieval input set instead of repeating conflict or durable remote reads after the main selection pass
- surface retrieval/build failures back to runtime callers instead of returning synthetic empty/partial long-term context
- keep tool-facing recall bounded by dropping silent subtext entirely and only fetching graph context as a fallback when visible structured recall would otherwise be empty
- sanitize recalled memory payloads before prompt serialization or optional LLM use
- reject compiled personalization payloads that leak schema/JSON/markup structure and fall back to the static subtext path instead of injecting corrupted hidden guidance
- expose a read-only operator search view over durable, episodic, midterm, graph, and conflict recall without constructing the full runtime service

`retrieval` does **not** own:
- persist or mutate long-term memory state
- extract, consolidate, or resolve memory objects
- runtime/service orchestration outside retrieval-specific helpers

Smart-home environment entries are ambient behavior signals, not diagnoses.
Retrieval must preserve quality flags and blockers from
`smart_home_environment` memories and must not expand them into raw sensor
dumps.

## Key files

| File | Purpose |
|---|---|
| `__init__.py` | Package marker |
| `retriever.py` | Context assembly, including environment-aware durable/midterm rendering, unified-plan attachment of adaptive/subtext metadata, and fail-closed graph-context suppression when no relevant graph payload survived |
| `unified_plan.py` | Internal candidate normalization, focus-component pruning, semantic join-anchor selection, and explainable mixed graph+structured+episodic+midterm query plans |
| `fast_topic.py` | One-shot current-topic hint builder for low-latency answer paths |
| `adaptive_policy.py` | Adaptive prompt-policy compiler from stored long-term signals, including confirmed response-channel preference packets |
| `restart_recall_policy.py` | Persistent restart-recall packet compiler from stable durable memory |
| `operator_search.py` | Read-only operator search over the real long-term retrieval stack |
| `subtext.py` | Subtext compiler and builder |
| `component.yaml` | Structured ownership metadata |
| `AGENTS.md` | Local editing rules |

## Usage

```python
from twinr.memory.longterm.retrieval.retriever import LongTermRetriever
from twinr.memory.longterm.retrieval.adaptive_policy import LongTermAdaptivePolicyBuilder
from twinr.memory.longterm.retrieval.subtext import LongTermSubtextBuilder

subtext_builder = LongTermSubtextBuilder(config=config, graph_store=graph_store)
retriever = LongTermRetriever(
    config=config,
    prompt_context_store=prompt_context_store,
    graph_store=graph_store,
    object_store=object_store,
    midterm_store=midterm_store,
    conflict_resolver=conflict_resolver,
    subtext_builder=subtext_builder,
    adaptive_policy_builder=LongTermAdaptivePolicyBuilder(
        proactive_state_store=proactive_state_store,
    ),
)
context = retriever.build_context(query=query_profile, original_query_text=user_text)
```

```python
from twinr.memory.longterm.retrieval.fast_topic import LongTermFastTopicContextBuilder

fast_topic = LongTermFastTopicContextBuilder(
    config=config,
    object_store=object_store,
)
topic_hint = fast_topic.build(query_profile=query_profile)
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [../MEMORY_ARCHITECTURE.md](../MEMORY_ARCHITECTURE.md)
- [../runtime/README.md](../runtime/README.md)
