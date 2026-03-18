# storage

Persist long-term object, conflict, archive, midterm, and remote catalog state.

## Responsibility

`storage` owns:
- persist durable object, conflict, and archive snapshots
- persist midterm packets used for short-horizon recall
- bridge validated local snapshots to fine-grained remote ChonkyDB documents plus compact catalog heads and bounded segment documents
- write each remote memory item as a real ChonkyDB document whose public metadata/content contract is sufficient for Twinr readback
- reuse unchanged remote item documents across snapshot rewrites so stable memories are not pointlessly re-upserted and older readback-safe documents stay intact
- validate and repair remote catalog/pointer consistency before long-term memory is exposed to runtime callers
- direct-assemble catalog-backed remote object snapshots from rich segment entries when possible, and use bounded-parallel `retrieve` plus `allowed_doc_ids` only for sparse legacy catalogs
- load compact remote catalog segment documents through bounded-parallel reads so required startup does not serialize every segment fetch
- keep required-readiness probes read-only even for sparse legacy object catalogs, so watchdog startup never blocks on catalog rewrites
- reuse successful remote snapshot probes within one readiness cycle so store bootstrap and health attestation do not refetch the same snapshot twice
- prefer the last successful remote document id on ordinary snapshot reads, so steady-state loads skip repeat pointer resolution while still falling back safely when the hint goes stale
- skip `retrieve_search` entirely when the current remote catalog candidate set already fits inside the caller's requested limit, so small conflict/object lookups do not burn an extra timeout-prone backend roundtrip
- overlap the independent `objects` / `conflicts` / `archive` required-startup reads so object-store readiness is bounded by the slowest current snapshot instead of the sum of all three
- emit structured ops diagnostics for remote ChonkyDB read and write failures plus recoverable retrieve-batch fallbacks so DNS issues, timeouts, backend flakes, and client-contract payload problems can be separated after live incidents
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
