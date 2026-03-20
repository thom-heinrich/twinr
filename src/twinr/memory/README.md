# memory

`memory` owns Twinr's memory subsystem root. It exposes the package-level
import surface for short-term memory, prompt-context stores, reminders, shared
retrieval helpers, and the `on_device`, `chonkydb`, and `longterm`
subpackages.

## Responsibility

`memory` owns:
- expose the canonical `twinr.memory` import surface
- persist durable prompt memory, managed user/personality context, and reminders
- overlap independent prompt/user/personality remote snapshot bootstrap reads so required-remote readiness is bounded by the slowest prompt-context snapshot instead of the sum of all three
- provide shared full-text and query-normalization helpers reused by memory stores
- keep live long-term recall from blocking on cold query-rewrite misses by returning an immediate fallback profile and filling the rewrite cache asynchronously in the background
- host the `on_device`, `chonkydb`, and `longterm` subpackages

`memory` does **not** own:
- top-level runtime loops or prompt-assembly policy outside memory store boundaries
- web routes, forms, or dashboard rendering in `src/twinr/web`
- hardware capture, playback, display, or printer behavior
- provider transport implementations outside memory-specific rewrite helpers

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Public memory exports |
| [context_store.py](./context_store.py) | Prompt and managed-context stores |
| [reminders.py](./reminders.py) | Reminder persistence and rendering |
| [query_normalization.py](./query_normalization.py) | Retrieval query rewrite cache |
| [fulltext.py](./fulltext.py) | In-memory FTS selector |
| [component.yaml](./component.yaml) | Structured package metadata |

## Usage

```python
from twinr.memory import OnDeviceMemory, PromptContextStore, ReminderStore

short_term = OnDeviceMemory(max_turns=6, keep_recent=3)
prompt_store = PromptContextStore.from_config(config)
reminders = ReminderStore(config.reminder_store_path, timezone_name=config.timezone)
```

```python
from twinr.memory.query_normalization import LongTermQueryRewriter

profile = LongTermQueryRewriter.from_config(config).profile("Wo ist meine Brille?")
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [on_device](./on_device/README.md)
- [chonkydb](./chonkydb/README.md)
- [longterm/runtime](./longterm/runtime/README.md)
