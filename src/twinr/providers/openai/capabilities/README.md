# capabilities

`capabilities` owns the reusable behavior mixins that make up Twinr's OpenAI
backend. Each module keeps one capability-specific slice of runtime behavior:
responses, live search, speech, reminder phrasing, or printing.

## Responsibility

`capabilities` owns:
- response, vision, and streaming helpers built on the OpenAI Responses API
- live-search prompt shaping, source extraction, search-model fallback logic, and structured spoken-answer output for voice playback
- speech-to-text and text-to-speech request handling for OpenAI audio endpoints
- reminder, proactive, and automation phrasing with deterministic local fallbacks
- printer-safe receipt composition and print-text sanitization

Search prompt shaping keeps approximate default geography in structured
`user_location` metadata, preserves only bounded recent user/assistant search
context, and now injects explicit caller-supplied place/date hints into the
final search prompt as structured context so ASR-noisy or relative-date turns
can still ground correctly without overriding a clearly different explicit
topic. Responses-based web search also retries incomplete
`max_output_tokens` stops through a bounded budget ladder so newer GPT-5 search
models can finish cleanly instead of degrading into generic blank-answer
errors. Successful search results now also retain the requested model,
structured per-attempt metadata, and any concrete fallback reason so runtime
callers can persist what actually happened when a search left its primary
model. The search voice-rewrite step also returns structured verification and
follow-up metadata so Twinr can distinguish verified answers from partial or
unverified results and decide whether to ask for a deeper website check.

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
