# storage

Persist long-term object, conflict, archive, midterm, and remote catalog state.

## Responsibility

`storage` owns:
- persist durable object, conflict, and archive snapshots
- persist midterm packets used for short-horizon recall
- keep local structured-memory and midterm snapshot files cross-service readable (`0644`) so the dedicated `thh` remote-memory watchdog can validate root-written runtime state without changing the authoritative runtime owner
- bridge validated local snapshots to fine-grained remote ChonkyDB documents plus compact catalog heads and bounded segment documents
- write each remote memory item as a real ChonkyDB document whose public metadata/content contract is sufficient for Twinr readback
- reuse unchanged remote item documents across snapshot rewrites so stable memories are not pointlessly re-upserted and older readback-safe documents stay intact
- reuse unchanged remote item documents directly from payload fingerprints plus the existing catalog selectors, so transient per-item readback failures do not amplify a no-op snapshot rewrite into a large `records/bulk` storm
- validate and repair remote catalog/pointer consistency before long-term memory is exposed to runtime callers
- direct-assemble catalog-backed remote object snapshots from rich segment entries when possible, and use bounded-parallel `retrieve` plus `allowed_doc_ids` only for sparse legacy catalogs
- load compact remote catalog segment documents through bounded-parallel reads so required startup does not serialize every segment fetch
- consume the configured remote retry/backoff budget for catalog segment reads across exact-document plus same-URI fetches, so one transient `/v1/external/documents/full` spike does not abort required provider-context assembly
- keep required-readiness probes read-only even for sparse legacy object catalogs, so watchdog startup never blocks on catalog rewrites
- reuse successful remote snapshot probes within one readiness cycle so store bootstrap and health attestation do not refetch the same snapshot twice
- let ordinary warm readiness probes reuse already learned exact snapshot document ids before re-walking the current pointer history, so required health checks stay fail-closed without paying repeated multi-second pointer lookups
- persist write-learned exact snapshot document ids across restarts, so a fresh Twinr process can reuse the last attested remote snapshot head instead of re-walking pointer history on every boot
- let runtime orchestration clear stale local cooldown and negative probe-cache state after an external watchdog attestation has already proved the required remote namespace readable
- prefer explicitly remembered remote document ids on ordinary snapshot reads right after local writes, while avoiding sticky read-learned hints that could keep a fresh reader on stale snapshot documents after another runtime updates the same namespace
- keep cold `origin_uri` resolution on a separate bounded bootstrap timeout, so startup can still recover through large historical pointer chains without stretching the hot-path exact-document timeout
- keep fail-closed remote health/status probes on their own bounded timeout, so startup and fresh-reader attestation do not misclassify a loaded-but-live backend as down just because `/v1/external/instance` is slower than the hot-path snapshot reads
- keep TTL-bounded in-process read-through caches for remote snapshot heads, catalog segments, search hits, and item payloads so repeated foreground recall can reuse the last proven remote state instead of rehydrating the same objects on every turn
- build local catalog search selectors from those cached remote entries so cache-enabled foreground recall can answer the first query-specific turn without waiting on another remote full-text roundtrip
- expose explicit payload-cache prewarm hooks so text-channel startup can front-load the first-hit object/conflict document fetches instead of making the user's first real message pay that network cost
- clear or refresh those read caches on local writes so remote-only truth stays authoritative even while warmed reads remain fast
- prefer ChonkyDB `topk_records(scope_ref=...)` for current-scope object/conflict/archive retrieval even when a local catalog projection is already cached, so Twinr keeps exercising the authoritative remote one-shot retrieval path and only falls back to the cached selector when that path is temporarily unavailable
- keep required fast-topic reads fail-closed on real remote loss, but allow one bounded rescue through the current remote catalog projection when only the narrower `topk_records(scope_ref=...)` endpoint flakes; the local fast-topic selector remains valid only in explicit non-required local-first mode
- preserve same-process read-after-write consistency for required remote snapshots and object-query selectors by reusing the just-written local snapshot only until remote reads catch up to the same `written_at`, so `confirm_memory()` does not immediately reread stale remote state or stale current-scope object hits
- treat false-empty durable-object current-scope hits as recoverable and continue into catalog-backed selection, so inflected queries like `früher ... Thermoskanne` or `aktuell gespeichert` do not disappear just because the remote one-shot path under-matches one wording
- keep catalog-backed rescue alive when current-scope object payloads collapse away during status/kind partitioning, so newly confirmed facts are not hidden just because a stale scope hit now deserializes as `superseded`
- stop shared context-object selection early when active current-scope payloads are still semantically off-topic after partitioning, so no-hit semantic questions do not pay redundant catalog rescue and hydration work only to return an empty context
- treat direct semantic query matches as topic anchors and keep only topic-related sibling facts around them, so meta-memory questions keep the confirmed current fact without leaking unrelated `confirmed/gespeichert` memories such as an old thermos location
- keep store-level object/conflict fallback ownership single-layered, so `search_current_item_payloads()` does not first do its own scope rescue and then make the caller repeat a second remote catalog fallback for the same miss
- fall back from a false-empty current-scope conflict `topk_records` hit set to the current remote catalog instead of double-filtering the same conflict away, so open conflicts remain visible when the remote one-shot search under-matches inflected wording
- fall back from false-empty durable-object catalog top-k results to the already loaded local catalog selector, but keep empty episodic scope misses authoritative so off-topic episode queries do not rehydrate the whole episodic catalog into the hot path
- top up underfilled durable sections with one bounded durable-only rescue when the shared episodic/durable query pool is dominated by episodes, so multimodal print/button facts are not starved by recent distractor turns
- attest every required remote snapshot write and subsequent pointer update with immediate parseable readbacks before Twinr trusts the saved snapshot or updates its fresh-reader document-id hints
- route exact-read remote snapshot and pointer writes through ChonkyDB's fast async content path, then prove visibility via readback attestation instead of waiting on the timeout-prone synchronous `records/bulk` content-processing path
- treat `long_term_memory_remote_write_timeout_s` as the HTTP transport bound only, and use the broader remote flush budget for accepted async bulk jobs plus their readback window, so Twinr does not kill its own accepted writes with an artificial 15s server-side timeout
- keep async snapshot and pointer attestation retries on the readback side when ChonkyDB acknowledges before either the fresh exact-document body or the new same-URI head is visible, so Twinr waits within one bounded visibility window instead of blindly duplicating accepted writes
- route fine-grained remote catalog/segment writes through async `records/bulk` plus bounded same-URI readback attestation, so live device writes stop timing out on ChonkyDB's synchronous bulk ingestion path while still failing closed if the accepted payload never becomes readable
- skip `retrieve_search` entirely when the current remote catalog candidate set already fits inside the caller's requested limit, so small conflict/object lookups do not burn an extra timeout-prone backend roundtrip
- overlap the independent `objects` / `conflicts` / `archive` required-startup reads so object-store readiness is bounded by the slowest current snapshot instead of the sum of all three
- emit structured ops diagnostics for remote ChonkyDB read and write failures plus recoverable retrieve-batch fallbacks so DNS issues, timeouts, backend flakes, client-contract payload problems, exact endpoints, and request-payload types can be separated after live incidents
- keep bounded ChonkyDB problem-detail fields such as `detail`, `error`, and `error_type` in remote-write diagnostics, so live `records/bulk` 503s can be traced to the exact server-side busy/init branch instead of only showing opaque status codes
- reject ChonkyDB bulk responses that return item-level store failures inside an HTTP-success envelope, so Twinr never treats a fast backend rejection as a durable snapshot write
- stamp each remote `records/bulk` failure with a bounded request correlation id, batch coordinates, and request bytes, then propagate that same context into watchdog/runtime failure surfaces for Pi incident tracing
- persist structured retrieve-search/retrieve-batch latency histograms plus explicit timeout/slow-read alert events so `/twinr` operators can see ChonkyDB read spikes without replaying raw logs
- preserve separate semantics for "required remote backend unavailable" versus "specific required remote read failed", and stamp fast `topk_records(scope_ref=...)` incidents with status/detail/timeout/retry evidence for Pi forensics
- preserve restart-recall midterm packets while reflection refreshes the rest of the midterm snapshot, so confirmed/stable durable facts can survive fresh runtime roots as explicit policy context
- preserve a small bounded set of the newest immediate turn-continuity packets across reflection refreshes, so nightly/next-day continuity prompts do not disappear just because a slower reflection pass rewrote the midterm snapshot
- match midterm packet queries on content-bearing terms, including compound-word containment, and fail closed when no real topic overlap exists so restart-recall packets do not leak into control questions
- project structured memory-state semantics such as `confirmed`, `aktuell`, `gespeichert`, and `superseded` into durable-object search text so meta-memory queries can retrieve the right fact instead of a generic sibling
- rank selected durable objects by combined query overlap, confirmation state, and recency before returning them to retrieval/runtime callers
- gate durable-object and conflict recall on content-bearing query terms so control questions do not pull in off-topic memory just because they share auxiliary words like `ist`
- gate quick-memory fast-topic hints on the same content-bearing overlap checks after the remote one-shot hit set is loaded, so neutral control questions like `Was ist ein Regenbogen?` do not leak unrelated favorites just because one weak current-scope hit happened to rank first
- keep raw storage identifiers such as `slot_key` and `value_key` out of retrieval text so dates and internal IDs do not make unrelated math/control questions match durable memory
- keep fine-grained remote bulk writes bounded by item count and request bytes before they hit ChonkyDB
- bootstrap fresh required remote namespaces with empty structured snapshots instead of failing before the first live write
- treat missing local/remote midterm baselines as empty bootstrap state, not as malformed payload warnings
- keep local snapshot path-validation warnings explicitly framed as caller/probe path misuse rather than memory corruption, and avoid emitting them during remote-only preflight sizing
- expose store-level mutation and review helpers for runtime code

`storage` does **not** own:
- define long-term memory schemas or ontology normalization
- extract, consolidate, reflect, or retain memories
- assemble prompt context or runtime service orchestration

## Key files

| File | Purpose |
|---|---|
| `__init__.py` | Package marker and summary |
| `store.py` | Object/conflict/archive store |
| `midterm_store.py` | Midterm packet store |
| `remote_catalog.py` | Public compatibility wrapper for the fine-grained remote object/conflict/archive catalog adapter |
| `_remote_catalog/` | Internal split implementation modules for remote catalog caching, search, hydration, and write attestation |
| `remote_read_diagnostics.py` | Structured ops-event diagnostics for remote long-term I/O failures and fallbacks |
| `remote_read_observability.py` | Persisted retrieve/top-k histogram + alert helper for remote read spikes |
| `remote_state.py` | Small remote snapshot/catalog adapter plus shared remote-read exception types |
| `component.yaml` | Structured ownership metadata |
| `AGENTS.md` | Local editing rules |

## Usage

```python
from twinr.memory.longterm import LongTermMidtermStore, LongTermStructuredStore
from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore

object_store = LongTermStructuredStore.from_config(config)
midterm_store = LongTermMidtermStore.from_config(config)
remote_state = LongTermRemoteStateStore.from_config(config)
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [../MEMORY_ARCHITECTURE.md](../MEMORY_ARCHITECTURE.md)
- [../runtime/README.md](../runtime/README.md)
- [../retrieval/README.md](../retrieval/README.md)
