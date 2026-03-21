# openai

`openai` is the stable package-root import surface for Twinr's OpenAI provider
stack. It re-exports the backend, provider adapters including the fast
conversation-closure decision adapter, realtime session surface, and shared
value objects while leaving implementation in focused
subpackages.

## Responsibility

`openai` owns:
- expose the stable import surface used by Twinr runtime code
- group the package into `api`, `core`, `capabilities`, and `realtime`
- re-export only the runtime-facing backend, adapter, realtime, and value types

`openai` does **not** own:
- backend implementation details outside [`api`](./api/README.md)
- shared client, request, and type implementation outside [`core`](./core/README.md)
- capability-specific mixins outside [`capabilities`](./capabilities/README.md)
- realtime session internals outside [`realtime`](./realtime/README.md)

## Key files

| File | Purpose |
|---|---|
| [`__init__.py`](./__init__.py) | Stable package-root re-exports |
| [`api/`](./api/README.md) | Backend and adapter package |
| [`capabilities/`](./capabilities/README.md) | Capability mixin package |
| [`core/`](./core/README.md) | Shared client and types |
| [`realtime/`](./realtime/README.md) | Realtime session package |
| [`component.yaml`](./component.yaml) | Structured package metadata |
| [`AGENTS.md`](./AGENTS.md) | Local editing rules |

## Usage

```python
from twinr.providers.openai import OpenAIBackend, OpenAIProviderBundle

backend = OpenAIBackend(config=config)
bundle = OpenAIProviderBundle.from_backend(backend)
```

```python
from twinr.providers.openai import OpenAIImageInput, OpenAIRealtimeSession

image = OpenAIImageInput.from_path("/tmp/camera.jpg", label="Kamera")
with OpenAIRealtimeSession(config, tool_handlers=handlers) as session:
    turn = session.run_text_turn("Wie ist das Wetter?")
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [api](./api/README.md)
- [capabilities](./capabilities/README.md)
- [core](./core/README.md)
- [realtime](./realtime/README.md)
