# realtime

`realtime` owns Twinr's OpenAI realtime session wrapper and the completed-turn
value object returned to the runtime. Its package root re-exports that surface
lazily so probes can inspect the package without opening the full realtime
implementation stack. The package wraps websocket session setup, event
consumption, tool-call bridging, and bounded timeout handling behind a small
synchronous surface.

## Responsibility

`realtime` owns:
- open and close the OpenAI realtime websocket session for Twinr
- refresh realtime instructions and tool schemas before each turn, reusing the canonical tool-agent instruction builder instead of a divergent realtime-only prompt copy
- normalize streamed provider events into `OpenAIRealtimeTurn` results

`realtime` does **not** own:
- shared client construction and request helpers in [`../core`](../core/README.md)
- non-realtime backend composition in [`../api`](../api/README.md)
- runtime button, audio, and state orchestration in [`../../../agent/workflows/realtime_runner.py`](../../../agent/workflows/realtime_runner.py)

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Lazy public package exports |
| [session.py](./session.py) | Session lifecycle and turn handling |
| [component.yaml](./component.yaml) | Structured package metadata |
| [AGENTS.md](./AGENTS.md) | Local editing invariants |

## Usage

```python
from twinr.providers.openai.realtime import OpenAIRealtimeSession

with OpenAIRealtimeSession(config, tool_handlers=handlers) as session:
    turn = session.run_text_turn("Wie wird das Wetter?")
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [core](../core/README.md)
- [api](../api/README.md)
- [realtime_runner.py](../../../agent/workflows/realtime_runner.py)
