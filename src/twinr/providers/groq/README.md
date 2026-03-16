# groq

`groq` owns Twinr's Groq-specific text and tool-calling adapters. It validates
Groq client configuration, translates Twinr conversation state into Groq chat
completion requests, and preserves continuation state across tool turns.

## Responsibility

`groq` owns:
- validate Groq client config and base URL overrides
- run streaming and non-streaming text turns against Groq
- manage continuation state for Groq tool-calling turns

`groq` does **not** own:
- OpenAI support-provider behavior or live-search execution
- provider bundle selection in `src/twinr/providers/factory.py`
- speech-to-text or text-to-speech behavior

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Public package exports |
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
