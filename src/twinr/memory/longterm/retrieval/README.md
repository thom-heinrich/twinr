# retrieval

Assemble prompt-ready long-term memory recall context and silent
personalization cues for a single user turn.

## Responsibility

`retrieval` owns:
- assemble durable, episodic, mid-term, graph, and conflict context
- compile and render silent personalization cues from graph and episodic memory
- sanitize recalled memory payloads before prompt serialization or optional LLM use

`retrieval` does **not** own:
- persist or mutate long-term memory state
- extract, consolidate, or resolve memory objects
- runtime/service orchestration outside retrieval-specific helpers

## Key files

| File | Purpose |
|---|---|
| `__init__.py` | Package marker |
| `retriever.py` | Context assembly |
| `subtext.py` | Subtext compiler and builder |
| `component.yaml` | Structured ownership metadata |
| `AGENTS.md` | Local editing rules |

## Usage

```python
from twinr.memory.longterm.retrieval.retriever import LongTermRetriever
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
)
context = retriever.build_context(query=query_profile, original_query_text=user_text)
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [../MEMORY_ARCHITECTURE.md](../MEMORY_ARCHITECTURE.md)
- [../runtime/README.md](../runtime/README.md)
