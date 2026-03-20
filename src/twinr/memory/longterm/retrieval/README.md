# retrieval

Assemble prompt-ready long-term memory recall context and silent
personalization cues for a single user turn.

## Responsibility

`retrieval` owns:
- assemble durable, episodic, mid-term, graph, and conflict context
- merge original-language and canonical-English query variants during store-backed recall so optional rewrites improve recall without suppressing same-language memories
- rerank durable recall across merged query variants so confirmed/current facts surface ahead of generic siblings for meta-memory questions
- compile adaptive mid-term policy hints from confirmed memory, recurring routines, and proactive success/skip history
- compile explicit confirmed response-channel policy packets so confirmed ReSpeaker-derived channel preferences can guide later delivery behavior without promoting mere observations
- compile persistent restart-recall policy packets from stable durable memories so fresh runtime roots retain a small provenance-rich continuity layer
- compile and render silent personalization cues from graph and episodic memory
- opportunistically reuse or background-build expensive silent subtext packets so live foreground turns do not stall on cold personalization compilation
- sanitize recalled memory payloads before prompt serialization or optional LLM use
- reject compiled personalization payloads that leak schema/JSON/markup structure and fall back to the static subtext path instead of injecting corrupted hidden guidance
- expose a read-only operator search view over durable, episodic, midterm, graph, and conflict recall without constructing the full runtime service

`retrieval` does **not** own:
- persist or mutate long-term memory state
- extract, consolidate, or resolve memory objects
- runtime/service orchestration outside retrieval-specific helpers

## Key files

| File | Purpose |
|---|---|
| `__init__.py` | Package marker |
| `retriever.py` | Context assembly |
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

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [../MEMORY_ARCHITECTURE.md](../MEMORY_ARCHITECTURE.md)
- [../runtime/README.md](../runtime/README.md)
