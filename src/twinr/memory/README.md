# memory

`memory` owns Twinr's memory subsystem root. It exposes the package-level
import surface for short-term memory, prompt-context stores, reminders, shared
retrieval helpers, and the `on_device`, `chonkydb`, and `longterm`
subpackages.

## Responsibility

`memory` owns:
- expose the canonical `twinr.memory` import surface
- resolve that public surface lazily so narrow imports do not eagerly pull the
  full long-term or graph stack into processes that only need prompt-context or
  on-device memory types
- persist durable prompt memory, managed user/personality context, and reminders
- keep reminder reservation state reversible when a background delivery loses its idle window before speech starts, so Twinr does not invent retry backoff for work that never began
- persist the guided user-discovery state that powers Twinr's resumable onboarding and lifelong profile-learning flow
- review, correct, and delete learned discovery facts while keeping managed-context and structured-memory commits synchronized
- keep runtime-context guided-discovery status local to persisted discovery state, so live prompt assembly never opens authoritative remote profile reads just to render a hint
- write local prompt-context snapshots world-readable (`0644`) so the dedicated remote-memory watchdog can validate them even when the productive runtime itself runs as `root`
- persist prompt memory plus managed user/personality context remotely as typed current-head/item records rather than treating ChonkyDB as a generic prompt-context blob store
- let prompt-memory and managed-context readiness reuse a current head or a synthetic empty head in the live prompt path, while keeping legacy-head probes only for explicit seed/migration checks, so the Pi watchdog stays read-only and pointer/blob fallback stays out of first-turn rendering
- keep prompt-memory and managed-context seed helpers read-only when a current-head probe returns `unavailable`; only truly missing or invalid heads may be re-seeded from local bootstrap content, so required-remote recovery does not devolve into slow local re-publish loops
- overlap independent prompt/user/personality remote current-head bootstrap reads so required-remote readiness is bounded by the slowest prompt-context check instead of the sum of all three
- provide shared full-text and query-normalization helpers reused by memory stores
- keep live long-term recall from blocking on cold query-rewrite misses by returning an immediate fallback profile and filling the rewrite cache asynchronously in the background
- let later wait-capable callers keep waiting on an in-flight canonical rewrite even after a fallback profile was cached, so one cold miss does not lock the whole runtime onto the untranslated query until a later cache refresh
- let synchronous provider-context and live-front materialization callers wait a bounded first-turn window sized for real rewrite latency, so multilingual graph/subtext recall does not miss just because canonical English arrives around one to three seconds later
- host the `on_device`, `chonkydb`, and `longterm` subpackages

`memory` does **not** own:
- top-level runtime loops or prompt-assembly policy outside memory store boundaries
- web routes, forms, or dashboard rendering in `src/twinr/web`
- hardware capture, playback, display, or printer behavior
- provider transport implementations outside memory-specific rewrite helpers

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Public memory exports resolved lazily to avoid import-time coupling between prompt-context stores and heavier long-term/graph modules |
| [context_store.py](./context_store.py) | Prompt and managed-context stores that persist remote current heads plus typed entry records for prompt memory, user context, and personality context; live reads stay current-head-only while seed/migration probes can still inspect legacy heads |
| [reminders.py](./reminders.py) | Reminder persistence, reservation release, and rendering |
| [user_discovery.py](./user_discovery.py) | Thin compatibility wrapper that preserves the stable `twinr.memory.user_discovery` import surface while delegating the guided-discovery implementation into focused runtime modules |
| [user_discovery_impl](./user_discovery_impl/) | Internal package split by concern across discovery catalog/helpers, data models, state persistence, commit routing, selection/presentation logic, and the public service orchestration |
| [user_discovery_authoritative_profile.py](./user_discovery_authoritative_profile.py) | Narrow adapter that projects authoritative graph facts and the shared runtime query-first structured-memory selector layer into discovery-topic coverage without hydrating full remote object snapshots |
| [user_discovery_policy.py](./user_discovery_policy.py) | Adaptive discovery topic scoring from reserve-lane engagement |
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
