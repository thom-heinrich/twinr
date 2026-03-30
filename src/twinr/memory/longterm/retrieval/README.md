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
- build one internal unified retrieval plan across durable objects, conflict queues, graph candidates, and selected midterm packets before rendering the stable public context sections
- reuse the graph selected by that unified plan for subtext compilation so one turn does not issue a second graph-selection pass just to render silent personalization
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
| `retriever.py` | Context assembly, including environment-aware durable and midterm rendering |
| `unified_plan.py` | Internal candidate normalization, explicit-anchor joins, and explainable mixed graph+structured+midterm query plans |
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
