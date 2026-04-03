# core

`core` owns the shared OpenAI provider primitives reused across Twinr's API,
capability, and realtime layers. Its package root re-exports shared client
helpers and value types lazily so probes can inspect the package without
eagerly importing SDK wiring or dataclass helpers. It centralizes client
construction, request assembly helpers, validated instruction constants, and
the canonical response and image dataclasses consumed by higher OpenAI
packages.

## Responsibility

`core` owns:
- build validated default OpenAI SDK clients from `TwinrConfig`
- preserve shared HTTP-transport ownership when Twinr clones SDK clients with `with_options(...)`
- normalize shared Responses API request payloads, including role-aware conversation replay, and tool settings
- define reusable instruction constants, fallback model lists, and typed response/image objects, including structured search-attempt metadata for runtime journaling
- define reusable instruction constants, fallback model lists, and typed response/image objects, including structured search verification/follow-up fields for runtime search routing

`core` does **not** own:
- public adapter surfaces or backend composition in [`../api`](../api/README.md)
- capability-specific speech, search, print, phrasing, or response flows in [`../capabilities`](../capabilities/README.md)
- realtime websocket session lifecycle in [`../realtime`](../realtime/README.md)

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Lazy shared package exports |
| [base.py](./base.py) | Request-building backend base |
| [client.py](./client.py) | SDK client factory and transport lifecycle helpers |
| [instructions.py](./instructions.py) | Prompt constants and fallbacks |
| [types.py](./types.py) | Response and image dataclasses |
| [component.yaml](./component.yaml) | Structured package metadata |
| [AGENTS.md](./AGENTS.md) | Local editing invariants |

## Usage

```python
from twinr.providers.openai.core.base import OpenAIBackendBase
from twinr.providers.openai.core.types import OpenAIImageInput

request = backend._build_response_request(
    "Wie wird das Wetter?",
    model="gpt-5",
    reasoning_effort="minimal",
)
image = OpenAIImageInput.from_path("/tmp/camera.jpg", label="Kameraansicht")
```

```python
from twinr.providers.openai.core.client import _default_client_factory
from twinr.providers.openai.core.instructions import SEARCH_AGENT_INSTRUCTIONS

client = _default_client_factory(config)
instructions = SEARCH_AGENT_INSTRUCTIONS
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [api](../api/README.md)
- [capabilities](../capabilities/README.md)
- [realtime](../realtime/README.md)
