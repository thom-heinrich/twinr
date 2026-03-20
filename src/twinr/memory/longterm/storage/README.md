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
- keep required-readiness probes read-only even for sparse legacy object catalogs, so watchdog startup never blocks on catalog rewrites
- reuse successful remote snapshot probes within one readiness cycle so store bootstrap and health attestation do not refetch the same snapshot twice
- let ordinary warm readiness probes reuse already learned exact snapshot document ids before re-walking the current pointer history, so required health checks stay fail-closed without paying repeated multi-second pointer lookups
- persist write-learned exact snapshot document ids across restarts, so a fresh Twinr process can reuse the last attested remote snapshot head instead of re-walking pointer history on every boot
- prefer explicitly remembered remote document ids on ordinary snapshot reads right after local writes, while avoiding sticky read-learned hints that could keep a fresh reader on stale snapshot documents after another runtime updates the same namespace
- keep cold `origin_uri` resolution on a separate bounded bootstrap timeout, so startup can still recover through large historical pointer chains without stretching the hot-path exact-document timeout
- keep TTL-bounded in-process read-through caches for remote snapshot heads, catalog segments, search hits, and item payloads so repeated foreground recall can reuse the last proven remote state instead of rehydrating the same objects on every turn
- build local catalog search selectors from those cached remote entries so cache-enabled foreground recall can answer the first query-specific turn without waiting on another remote full-text roundtrip
- expose explicit payload-cache prewarm hooks so text-channel startup can front-load the first-hit object/conflict document fetches instead of making the user's first real message pay that network cost
- clear or refresh those read caches on local writes so remote-only truth stays authoritative even while warmed reads remain fast
- prefer ChonkyDB `topk_records(scope_ref=...)` for current-scope object/conflict/archive retrieval even when a local catalog projection is already cached, so Twinr keeps exercising the authoritative remote one-shot retrieval path and only falls back to the cached selector when that path is temporarily unavailable
- attest every required remote snapshot write and subsequent pointer update with immediate parseable readbacks before Twinr trusts the saved snapshot or updates its fresh-reader document-id hints
- route exact-read remote snapshot and pointer writes through ChonkyDB's fast async content path, then prove visibility via readback attestation instead of waiting on the timeout-prone synchronous `records/bulk` content-processing path
- skip `retrieve_search` entirely when the current remote catalog candidate set already fits inside the caller's requested limit, so small conflict/object lookups do not burn an extra timeout-prone backend roundtrip
- overlap the independent `objects` / `conflicts` / `archive` required-startup reads so object-store readiness is bounded by the slowest current snapshot instead of the sum of all three
- emit structured ops diagnostics for remote ChonkyDB read and write failures plus recoverable retrieve-batch fallbacks so DNS issues, timeouts, backend flakes, and client-contract payload problems can be separated after live incidents
- reject ChonkyDB bulk responses that return item-level store failures inside an HTTP-success envelope, so Twinr never treats a fast backend rejection as a durable snapshot write
- stamp each remote `records/bulk` failure with a bounded request correlation id, batch coordinates, and request bytes, then propagate that same context into watchdog/runtime failure surfaces for Pi incident tracing
- persist structured retrieve-search/retrieve-batch latency histograms plus explicit timeout/slow-read alert events so `/twinr` operators can see ChonkyDB read spikes without replaying raw logs
- preserve restart-recall midterm packets while reflection refreshes the rest of the midterm snapshot, so confirmed/stable durable facts can survive fresh runtime roots as explicit policy context
- match midterm packet queries on content-bearing terms, including compound-word containment, and fail closed when no real topic overlap exists so restart-recall packets do not leak into control questions
- project structured memory-state semantics such as `confirmed`, `aktuell`, `gespeichert`, and `superseded` into durable-object search text so meta-memory queries can retrieve the right fact instead of a generic sibling
- rank selected durable objects by combined query overlap, confirmation state, and recency before returning them to retrieval/runtime callers
- gate durable-object and conflict recall on content-bearing query terms so control questions do not pull in off-topic memory just because they share auxiliary words like `ist`
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
| `remote_catalog.py` | Fine-grained remote object/conflict/archive catalog adapter |
| `remote_read_diagnostics.py` | Structured ops-event diagnostics for remote long-term I/O failures and fallbacks |
| `remote_read_observability.py` | Persisted retrieve histogram + alert helper for remote read spikes |
| `remote_state.py` | Small remote snapshot/catalog adapter |
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
