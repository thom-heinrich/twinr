# api

`api` owns Twinr's public OpenAI API-layer surfaces. Its package root exports
the canonical backend class and adapter types lazily so import probes do not
pull the full adapter/backend stack until a caller needs a concrete export.
The package wraps that backend in contract-specific adapters and assembles the
provider bundle consumed by runtime loops and factories.

## Responsibility

`api` owns:
- compose the canonical `OpenAIBackend` from shared mixins
- bridge Twinr speech, text, tool-calling, supervisor, closure-decision, and first-word contracts onto the backend
- validate structured supervisor decisions that carry explicit `prompt`, `location_hint`, `date_context`, and `context_scope` fields for safer fast-lane routing, while allowing handoff `spoken_ack` to stay null when no immediate bridge speech should be spoken, flooring GPT-5-family supervisor budgets high enough to finish the strict JSON contract on live Pi calls, retrying bounded GPT-5 supervisor ladders when live structured output still ends in `status=incomplete`, and decoding structured Responses payloads from SDK parsed data or balanced JSON text instead of trusting concatenated free-text output alone
- assemble the OpenAI provider bundle used by runtime callers

`api` does **not** own:
- low-level OpenAI client construction or shared request helpers in [`../core`](../core/README.md)
- capability-specific search, speech, print, phrasing, or response logic in [`../capabilities`](../capabilities/README.md)
- realtime websocket session orchestration in [`../realtime`](../realtime/README.md)

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Lazy public package exports |
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

```python
from twinr.providers.openai import OpenAIBackend, OpenAIConversationClosureDecisionProvider

backend = OpenAIBackend(config=config)
closure = OpenAIConversationClosureDecisionProvider(backend)
decision = closure.decide("closure prompt", instructions="Return one structured closure decision.")
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [core](../core/README.md)
- [capabilities](../capabilities/README.md)
- [realtime](../realtime/README.md)
