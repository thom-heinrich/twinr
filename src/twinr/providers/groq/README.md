# groq

`groq` owns Twinr's Groq-specific text and tool-calling adapters. Its package
root re-exports the public providers lazily so import probes do not pull the
adapter stack until a caller requests it. The package validates Groq client
configuration, translates Twinr conversation state into Groq chat completion
requests, and preserves continuation state across tool turns.

## Responsibility

`groq` owns:
- validate Groq client config and base URL overrides
- run streaming and non-streaming text turns against Groq
- manage continuation state for Groq tool-calling turns

When `allow_web_search=True`, Groq uses native server-side search first.
Cross-provider fallback is opt-in via `groq_allow_search_fallback=True`, and
vision fallback is opt-in via `groq_allow_vision_fallback=True`. Native search
also enforces the configured `groq_request_timeout_seconds` as a wall-clock
budget so Pi turns cannot wait indefinitely on search responses that keep the
socket alive.

`groq` does **not** own:
- OpenAI support-provider behavior or live-search execution
- provider bundle selection in `src/twinr/providers/factory.py`
- speech-to-text or text-to-speech behavior

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Lazy public package exports |
| [adapters.py](./adapters.py) | Text and tool provider logic |
| [client.py](./client.py) | Validated SDK client builder |
| [types.py](./types.py) | Response value object |
| [component.yaml](./component.yaml) | Structured package metadata |
| [AGENTS.md](./AGENTS.md) | Local editing invariants |

## Usage

```python
from twinr.providers.groq import GroqAgentTextProvider

provider = GroqAgentTextProvider(config, support_provider=openai_agent)
response = provider.respond_streaming("Sag hallo")
```

```python
from twinr.providers.groq import GroqToolCallingAgentProvider

tool_provider = GroqToolCallingAgentProvider(config)
turn = tool_provider.start_turn_streaming("Bitte drucke das", tool_schemas=schemas)
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [provider factory](../factory.py)
- [OpenAI support provider](../openai)
