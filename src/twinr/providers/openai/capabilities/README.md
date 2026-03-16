# capabilities

`capabilities` owns the reusable behavior mixins that make up Twinr's OpenAI
backend. Each module keeps one capability-specific slice of runtime behavior:
responses, live search, speech, reminder phrasing, or printing.

## Responsibility

`capabilities` owns:
- response, vision, and streaming helpers built on the OpenAI Responses API
- live-search prompt shaping, source extraction, and search-model fallback logic
- speech-to-text and text-to-speech request handling for OpenAI audio endpoints
- reminder, proactive, and automation phrasing with deterministic local fallbacks
- printer-safe receipt composition and print-text sanitization

`capabilities` does **not** own:
- low-level client construction, shared request builders, or base types in [`../core`](../core/README.md)
- public backend composition and contract adapters in [`../api`](../api/README.md)
- realtime websocket session orchestration in [`../realtime`](../realtime/README.md)

## Key files

| File | Purpose |
|---|---|
| [`__init__.py`](./__init__.py) | Public mixin exports |
| [`responses.py`](./responses.py) | Text, vision, and streaming helpers |
| [`search.py`](./search.py) | Live-search orchestration |
| [`speech.py`](./speech.py) | STT and TTS helpers |
| [`phrasing.py`](./phrasing.py) | Reminder and automation phrasing |
| [`printing.py`](./printing.py) | Printer-safe composition |
| [`component.yaml`](./component.yaml) | Structured package metadata |
| [`AGENTS.md`](./AGENTS.md) | Local editing rules |

## Usage

```python
from twinr.providers.openai.api.backend import OpenAIBackend

backend = OpenAIBackend(config=config)
reply = backend.respond_with_metadata("Sag hallo")
search = backend.search_live_info_with_metadata("Wie wird das Wetter morgen?")
```

```python
print_job = backend.compose_print_job_with_metadata(conversation=conversation)
audio = backend.synthesize("Guten Morgen")
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [api](../api/README.md)
- [core](../core/README.md)
- [realtime](../realtime/README.md)
