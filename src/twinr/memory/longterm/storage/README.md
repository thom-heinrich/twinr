# storage

Persist long-term object, conflict, archive, midterm, and remote catalog state.

## Responsibility

`storage` owns:
- persist durable object, conflict, and archive snapshots
- persist midterm packets used for short-horizon recall
- persist materialized live-provider answer fronts as bounded current-head collections so runtime can consume prompt-ready long-term blocks without rebuilding the broad retriever inline
- keep local structured-memory and midterm snapshot files cross-service readable (`0644`) so the dedicated `thh` remote-memory watchdog can validate root-written runtime state without changing the authoritative runtime owner
- bridge validated local snapshots to fine-grained remote ChonkyDB documents plus compact catalog heads and bounded segment documents
- persist those current catalog heads as fixed-URI remote-catalog documents (`.../catalog/current`) instead of generic `remote_state` snapshot heads, so current object/conflict/archive authority no longer depends on pointer-walked blob snapshots
- treat a missing fixed `.../catalog/current` head as a legitimate empty bootstrap state for structured-memory writes, while keeping 5xx/timeout current-head reads fail-closed; do not revive legacy snapshot compat reads just to seed a fresh namespace
- treat metadata-only current-head probes as real backend contract calls: keep them on the fixed current-head URI, use backend-valid request parameters, and never downgrade a real backend failure into an empty-head seed
- when a metadata-only fixed current-head probe gets a backend `400` validation reject, retry the same fixed URI once with a content-bearing `documents/full` read before falling back, so readable graph/object/prompt heads do not degrade into legacy snapshot recovery just because the lighter request shape is not accepted
- when a fixed current-head URI resolves to multiple historical documents, always pick the newest valid catalog payload by `written_at`; stale same-URI heads must not suppress the latest current state
- once a collection is on the fixed current-head contract, do not mirror the same head back into generic `remote_state.save_snapshot(...)`; that legacy snapshot write path is compatibility-only and must not be revived by current-head persistence, except for the deliberately tiny midterm compatibility head that fresh readers still use read-only when the fixed current head lags cross-process visibility
- write each remote memory item as a real ChonkyDB document whose public payload/metadata/content contract is sufficient for Twinr readback
- carry the authoritative Twinr object/conflict payload inside the public record payload envelope, while keeping metadata-embedded payload copies only as a bounded compatibility bridge for older live document shapes
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
- treat mutable structured-memory current heads (`objects/conflicts/archive`) as same-URI moving targets on fallback snapshot reads; those paths must re-resolve through current-head/pointer authority instead of pinning rescue reads to a remembered exact document id from an older head
- keep cold `origin_uri` resolution on a separate bounded bootstrap timeout, so startup can still recover through large historical pointer chains without stretching the hot-path exact-document timeout
- keep fixed `catalog/current` head reads on that same bounded bootstrap timeout, so metadata-only `400` retries can still complete their follow-up full-document read on a fresh reader instead of failing closed at the shorter hot-path timeout
- keep fail-closed remote health/status probes on their own bounded timeout, so startup and fresh-reader attestation do not misclassify a loaded-but-live backend as down just because `/v1/external/instance` is slower than the hot-path snapshot reads
- when `/v1/external/instance` stalls, allow one bounded synthetic `documents/full` control probe to prove transport/auth/routing liveness before the deeper readiness pass validates the real Twinr snapshot namespace
- let readiness/bootstrap probes try metadata-only snapshot envelopes before requesting large `content` bodies, so required-remote recovery can validate existing snapshot heads without always paying the full content-read path
- keep readiness/bootstrap snapshot probes strictly read-only even when they recover through `origin_uri`, so the watchdog does not block startup on synchronous snapshot-pointer repair writes
- let structured-store bootstrap trust those light remote probes when they already prove a valid current-head document or legacy catalog manifest exists, so watchdog recovery does not hydrate every `objects/conflicts/archive` segment just to decide nothing needs seeding
- keep structured-store and midterm `*_for_readiness` bootstrap paths read-only for fresh empty required namespaces; they may repair missing remote heads from real local state, but they must not manufacture empty remote state during startup
- let prompt-memory and managed-context bootstrap reuse a readable legacy catalog head, or synthesize the canonical empty head locally when both remote heads are absent and the local store is empty, so startup never stalls on an empty `records/bulk` seed write
- publish empty prompt-memory and managed-context current heads directly when the caller is intentionally writing an empty collection, so deleting the last item or repairing an already-known invalid blank head does not first re-read the broken old head
- let ordinary non-bootstrap readers still promote legacy catalog snapshot heads into the fixed current-head document contract when they explicitly choose the full load path, but keep readiness/bootstrap probes themselves strictly read-only
- keep TTL-bounded in-process read-through caches for remote snapshot heads, catalog segments, search hits, and item payloads so repeated foreground recall can reuse the last proven remote state instead of rehydrating the same objects on every turn
- build local catalog search selectors from those cached remote entries so cache-enabled foreground recall can answer the first query-specific turn without waiting on another remote full-text roundtrip
- expose explicit payload-cache prewarm hooks so text-channel startup can front-load the first-hit object/conflict document fetches instead of making the user's first real message pay that network cost
- clear or refresh those read caches on local writes so remote-only truth stays authoritative even while warmed reads remain fast
- prefer ChonkyDB `topk_records(scope_ref=...)` for current-scope object/conflict/archive retrieval even when a local catalog projection is already cached, so Twinr keeps exercising the authoritative remote one-shot retrieval path and only falls back to the cached selector when that path is temporarily unavailable
- keep query-time current-scope and allowlist retrieval on explicit lightweight `allowed_indexes`, so backend default widening cannot silently re-enable cold HNSW candidate generation on the live remote-memory path
- keep required fast-topic reads fail-closed on real remote loss, but allow one bounded rescue only through a same-process already-proven current remote catalog projection when the narrower `topk_records(scope_ref=...)` endpoint flakes; do not trigger a cold remote catalog load inside that rescue path, and keep the local fast-topic selector valid only in explicit non-required local-first mode
- preserve same-process read-after-write consistency for required remote snapshots and object-query selectors by reusing the just-written local snapshot until remote reads are strictly newer than that `written_at`, so equal-timestamp remote visibility does not prematurely discard the writer's bridge before hot-path reads like fast-topic have stabilized
- treat false-empty durable-object current-scope hits as recoverable and continue into catalog-backed selection, so inflected queries like `früher ... Thermoskanne` or `aktuell gespeichert` do not disappear just because the remote one-shot path under-matches one wording
- keep catalog-backed rescue alive when current-scope object payloads collapse away during status/kind partitioning, so newly confirmed facts are not hidden just because a stale scope hit now deserializes as `superseded`
- stop shared context-object selection early when active current-scope payloads are still semantically off-topic after partitioning, so no-hit semantic questions do not pay redundant catalog rescue and hydration work only to return an empty context
- treat direct semantic query matches as topic anchors and keep only topic-related sibling facts around them, so meta-memory questions keep the confirmed current fact without leaking unrelated `confirmed/gespeichert` memories such as an old thermos location
- keep store-level object/conflict fallback ownership single-layered, so `search_current_item_payloads()` does not first do its own scope rescue and then make the caller repeat a second remote catalog fallback for the same miss
- keep a bounded `selection_projection` on object/conflict catalog entries and use dedicated selection-only hydrators for object/conflict context reads, so query-time rescue can rebuild prompt/ranking payloads without falling back to per-item `documents/full`
- once a fixed current head exposes segment `document_id`s, resolve those segment and item records through `topk_records(allowed_doc_ids=...)` / `retrieve(allowed_doc_ids=...)` before considering any `documents/full` read, and request segment batches with `include_content=true` so live `service.payload_blob` responses still carry the authoritative segment JSON in `content`
- fall back from a false-empty current-scope conflict `topk_records` hit set to the current remote catalog instead of double-filtering the same conflict away, so open conflicts remain visible when the remote one-shot search under-matches inflected wording
- fall back from false-empty durable-object catalog top-k results to the already loaded local catalog selector, but keep empty episodic scope misses authoritative so off-topic episode queries do not rehydrate the whole episodic catalog into the hot path
- top up underfilled durable sections with one bounded durable-only rescue when the shared episodic/durable query pool is dominated by episodes, so multimodal print/button facts are not starved by recent distractor turns
- attest every required remote snapshot write and subsequent pointer update with immediate parseable readbacks before Twinr trusts the saved snapshot or updates its fresh-reader document-id hints
- route exact-read remote snapshot and pointer writes through ChonkyDB's fast async content path, then prove visibility via readback attestation instead of waiting on the timeout-prone synchronous `records/bulk` content-processing path
- treat `long_term_memory_remote_write_timeout_s` as the HTTP transport bound only, and use the broader remote flush budget for accepted async bulk jobs plus their readback window, so Twinr does not kill its own accepted writes with an artificial 15s server-side timeout
- give `graph_nodes` / `graph_edges` current-view async jobs a higher bounded server-side execution floor than generic writes, because graph payload processing can legitimately outlive the normal flush budget while the HTTP transport contract must stay short and fail-closed
- treat transient remote write overload signals such as HTTP `429` / `queue_saturated` as bounded retryable conditions, and honor `Retry-After` when ChonkyDB exposes it, so required startup does not fail closed on short queue-pressure spikes
- keep async snapshot and pointer attestation retries on the readback side when ChonkyDB acknowledges before either the fresh exact-document body or the new same-URI head is visible, so Twinr waits within one bounded visibility window instead of blindly duplicating accepted writes
- route fine-grained remote catalog/segment writes through async `records/bulk`, prefer jobs-endpoint exact `document_id` attestation for immutable item/segment documents when available, and keep same-URI readback as the bounded fallback for mutable heads or job-status flakeouts, so live device writes stop timing out on ChonkyDB's synchronous bulk ingestion path without reintroducing stale same-URI attestation on versioned records
- send explicit Twinr `target_indexes` on remote writes so searchable item records populate only the runtime's real non-vector search lanes and fixed current-head/segment docs stay off the default vector/hnsw ingest path
- treat accepted async jobs endpoint reads as best-effort exact-id hints, cap each poll to the remaining visibility budget, and fall back to authoritative same-URI readback attestation when `/v1/external/jobs/...` times out or flakes, so accepted writes do not false-fail just because the job-status endpoint lags the actual record visibility
- skip `retrieve_search` entirely when the current remote catalog candidate set already fits inside the caller's requested limit, so small conflict/object lookups do not burn an extra timeout-prone backend roundtrip
- overlap the independent `objects` / `conflicts` / `archive` required-startup reads so object-store readiness is bounded by the slowest current snapshot instead of the sum of all three
- emit structured ops diagnostics for remote ChonkyDB read and write failures plus recoverable retrieve-batch fallbacks so DNS issues, timeouts, backend flakes, client-contract payload problems, exact endpoints, and request-payload types can be separated after live incidents
- keep bounded ChonkyDB problem-detail fields such as `detail`, `error`, and `error_type` in remote-write diagnostics, so live `records/bulk` 503s can be traced to the exact server-side busy/init branch instead of only showing opaque status codes
- reject ChonkyDB bulk responses that return item-level store failures inside an HTTP-success envelope, so Twinr never treats a fast backend rejection as a durable snapshot write
- stamp each remote `records/bulk` failure with a bounded request correlation id, batch coordinates, and request bytes, then propagate that same context into watchdog/runtime failure surfaces for Pi incident tracing
- persist structured retrieve-search/retrieve-batch latency histograms plus explicit timeout/slow-read alert events so `/twinr` operators can see ChonkyDB read spikes without replaying raw logs
- classify every remote ChonkyDB request into coarse live-path buckets such as `catalog_current_head`, `topk_scope_query`, `retrieve_batch`, `documents_full`, and `legacy_snapshot_compat`, and persist those counts for both read and write traffic so blob regressions stay visible in bounded artifacts
- keep query-time exact object loads on the same selection-hydration contract as the upstream retrieval that discovered them, so conflict/context assembly does not silently escalate into per-item `documents/full` reads after a bounded remote search already narrowed the candidate set
- allow query-time exact object loads for conflict/context assembly to cold-load the current catalog when necessary, but keep that cold path on current-head plus batch-retrieve semantics instead of reintroducing item-document `documents/full`
- expose exact-id, slot-key, and reference-filtered object/conflict selectors for review and mutation flows, so operator-facing confirm/invalidate/delete/resolve paths stop hydrating whole object or conflict snapshots just to touch one bounded subset
- commit operator-facing confirm/invalidate/delete/resolve writes through collection deltas and touched slot-key deletes, so mutation paths stop rewriting full current object/conflict/archive state just to retire one conflict or mutate one memory
- build bounded active persistence/backfill working sets from touched memory ids, slot competitors, source event ids, and reference-linked objects, so conversation/multimodal writes and ops backfill stop hydrating the full current object/conflict/archive state before consolidation
- commit those active-path structured-memory changes as collection deltas against the fixed `catalog/current` heads, so required remote writes stop routing active persistence/backfill through whole-state `write_snapshot(...)` rewrites
- prefilter online retention candidates from object `selection_projection` metadata and hydrate only the records whose projections imply an archive/prune action, so remote-primary retention no longer starts from a global fine-grained object sweep
- serialize persisted remote-read histograms behind a dedicated cross-process lock file and unique temp writes, so runtime services and live validation prompts can update observability without `.tmp` collisions or lost increments
- preserve separate semantics for "required remote backend unavailable" versus "specific required remote read failed", and stamp fast `topk_records(scope_ref=...)` incidents with status/detail/timeout/retry evidence for Pi forensics
- surface server-reported remote catalog query plans as retrieval workflow events so current-scope search decisions are explainable without hydrating full catalogs during debugging
- persist remote midterm state as fine-grained packet records plus a fixed `midterm/.../catalog/current` head instead of a pointer-walked `save_snapshot/load_snapshot("midterm")` blob
- mirror that small midterm catalog head onto the legacy snapshot URI as a read-only compatibility bridge so fresh readers can prove the already-written current state when the fixed current head lags; the compatibility head is not the authority and must not grow into a packet blob again
- carry complete `selection_projection` payloads on midterm catalog entries and hydrate packet reads from those projections before exact item documents, so a visible current head does not collapse into empty midterm recall just because item-document visibility lags the head
- expose store-level remote midterm head probes and current-packet-id loaders so runtime health and live acceptance can attest the real authoritative remote contract instead of a legacy snapshot shadow
- let graph and midterm bootstrap/load paths reuse those small compatibility current-head payloads when the direct fixed-URI heads lag cross-process, so fresh readers do not reseed or fail on the old blob-snapshot path
- preserve restart-recall midterm packets while reflection refreshes the rest of the midterm snapshot, so confirmed/stable durable facts can survive fresh runtime roots as explicit policy context
- preserve a small bounded set of the newest immediate turn-continuity packets across reflection refreshes, so nightly/next-day continuity prompts do not disappear just because a slower reflection pass rewrote the midterm snapshot
- match midterm packet queries on content-bearing terms, including compound-word containment, and fail closed when no real topic overlap exists so restart-recall packets do not leak into control questions
- project structured memory-state semantics such as `confirmed`, `aktuell`, `gespeichert`, and `superseded` into durable-object search text so meta-memory queries can retrieve the right fact instead of a generic sibling
- rank selected durable objects by combined query overlap, confirmation state, and recency before returning them to retrieval/runtime callers
- gate durable-object and conflict recall on content-bearing query terms so control questions do not pull in off-topic memory just because they share auxiliary words like `ist`
- gate quick-memory fast-topic hints on the same content-bearing overlap checks after the remote one-shot hit set is loaded, so neutral control questions like `Was ist ein Regenbogen?` do not leak unrelated favorites just because one weak current-scope hit happened to rank first
- keep raw storage identifiers such as `slot_key` and `value_key` out of retrieval text so dates and internal IDs do not make unrelated math/control questions match durable memory
- keep fine-grained remote bulk writes bounded by item count and request bytes before they hit ChonkyDB
- expose fine-grained current-state loaders for objects/conflicts/archive plus one shared bundled current-state contract so runtime persistence and maintenance paths can enumerate remote state itemwise instead of hydrating snapshot blobs; the only allowed raw-snapshot exception is the legitimate empty archive snapshot
- keep whole-state materialization out of active review and mutation paths unless the caller is explicitly writing a full current-state snapshot, and prefer fine-grained current-state loads over legacy snapshot-shaped helpers even for local fallback
- bootstrap fresh required remote namespaces with empty structured snapshots instead of failing before the first live write
- treat missing local/remote midterm baselines as empty bootstrap state, not as malformed payload warnings
- keep local snapshot path-validation warnings explicitly framed as caller/probe path misuse rather than memory corruption, and avoid emitting them during remote-only preflight sizing
- treat ops-event append failures from `remote_read_diagnostics.py` / `remote_read_observability.py` as best-effort only: a foreign-owned Pi event store may suppress those artifacts, but it must not dump traceback noise into manual CLI smokes or change required-remote behavior
- expose store-level mutation and review helpers for runtime code

`storage` does **not** own:
- define long-term memory schemas or ontology normalization
- extract, consolidate, reflect, or retain memories
- assemble prompt context or runtime service orchestration

## Key files

| File | Purpose |
|---|---|
| `__init__.py` | Package marker and summary |
| `store.py` | Public compatibility wrapper for `LongTermStructuredStore` |
| `_structured_store/` | Internal split implementation modules for structured snapshot IO, mutations, ranking, and retrieval |
| `_structured_store/active_delta.py` | Active-path selective working-set loading plus delta commits for conversation/multimodal persistence and backfill |
| `midterm_store.py` | Midterm packet store |
| `provider_answer_front_store.py` | Remote current-head store for materialized live-provider answer-front blocks |
| `remote_catalog.py` | Public compatibility wrapper for the fine-grained remote object/conflict/archive catalog adapter |
| `_remote_catalog/` | Internal split implementation modules for remote catalog caching, search, hydration, and write attestation |
| `_remote_current_records.py` | Shared current-head helper for prompt-memory, managed-context, and other typed remote collections that need read-only legacy-head fallback plus synthetic empty-head bootstrap |
| `_remote_state/` | Internal split implementation modules for remote snapshot reads, writes, cache state, and shared value types |
| `remote_read_diagnostics.py` | Structured diagnostics for remote read/write failures and fallback paths |
| `remote_read_observability.py` | Persisted read/write histograms, alerts, and access classification |
| `remote_state.py` | Public compatibility wrapper exporting the split remote snapshot state store and its shared read-status/failure types |
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
