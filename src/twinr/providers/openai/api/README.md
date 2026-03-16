# api

`api` owns Twinr's public OpenAI API-layer surfaces. It exports the canonical
backend class, wraps that backend in contract-specific adapters, and assembles
the provider bundle consumed by runtime loops and factories.

## Responsibility

`api` owns:
- compose the canonical `OpenAIBackend` from shared mixins
- bridge Twinr speech, text, tool-calling, supervisor, and first-word contracts onto the backend
- assemble the OpenAI provider bundle used by runtime callers

`api` does **not** own:
- low-level OpenAI client construction or shared request helpers in [`../core`](../core/README.md)
- capability-specific search, speech, print, phrasing, or response logic in [`../capabilities`](../capabilities/README.md)
- realtime websocket session orchestration in [`../realtime`](../realtime/README.md)

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Public package exports |
| [backend.py](./backend.py) | Canonical backend composition |
| [adapters.py](./adapters.py) | Contract adapters and bundle assembly |
| [component.yaml](./component.yaml) | Structured package metadata |
| [AGENTS.md](./AGENTS.md) | Local editing invariants |

## Usage

```python
from twinr.providers.openai import OpenAIBackend

backend = OpenAIBackend(config=config)
response = backend.respond_with_metadata("Sag hallo")
```

```python
from twinr.providers.openai import OpenAIProviderBundle

bundle = OpenAIProviderBundle.from_config(config)
turn = bundle.tool_agent.start_turn_streaming("Bitte drucke das", tool_schemas=schemas)
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [core](../core/README.md)
- [capabilities](../capabilities/README.md)
- [realtime](../realtime/README.md)
