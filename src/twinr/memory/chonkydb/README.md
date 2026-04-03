# chonkydb

`chonkydb` owns Twinr's ChonkyDB integration package. It provides the
validated HTTP client, typed API payload models, and the personal-graph store
used by Twinr's long-term memory flows.

## Responsibility

`chonkydb` owns:
- call the ChonkyDB external API through a validated client boundary
- keep that client on a pooled, redirect-fail-closed transport so repeated Pi memory reads/writes do not pay a fresh DNS/TLS setup every call
- preserve HTTP error metadata such as `Retry-After` on `ChonkyDBError`, so higher storage layers can treat transient `429 queue_saturated` write pressure as bounded retryable backend overload instead of a generic opaque failure
- keep `ChonkyDBError` compatible with normal Python exception chaining and traceback reassignment, so readiness/probe context managers surface the real backend failure instead of masking it with a secondary runtime error
- define typed request and response models for record, retrieve, and graph endpoints
- expose typed `graph/store_many` requests so Twinr can persist versioned graph generations without falling back to whole-graph snapshot blobs
- expose a one-shot `topk_records` retrieval contract that returns structured payloads from ChonkyDB in a single roundtrip
- expose async job-status reads so higher-level remote-memory writers can turn accepted async writes into exact `document_id` attestation instead of over-relying on eventually-consistent same-URI heads
- carry Twinr's `namespace + scope_ref` contract for current-snapshot retrieval so server-side scope resolution can replace oversized client allowlists
- define the canonical Twinr personal-graph schema and its remote-authoritative store
- keep the personal-graph inter-process lock on a runtime-shared lock path so
  watchdog and supervisor processes can coordinate even when the ChonkyDB data
  directory itself is root-owned
- persist the personal graph as graph-native topology generations plus fine-grained `graph_nodes` / `graph_edges` current-view heads instead of mirroring one whole-graph snapshot blob
- validate that a claimed remote graph current view is actually hydratable before treating it as healthy, and when a local graph cache exists repair broken `graph_nodes` / `graph_edges` current-view documents by rewriting the generation instead of silently trusting stale same-URI records
- carry a bounded `selection_projection` on `graph_nodes` / `graph_edges` catalog entries and hydrate current-view reads from that projection before exact item-document fetches, because live `documents/full` responses can resolve the origin lookup without returning a usable payload body
- prefer query-first remote subgraph selection for prompt-context and subtext building by combining current-view search, graph-path expansion, and graph-neighbor expansion before hydrating any node or edge payloads
- keep query-first remote subgraph selection fresh-reader-safe by falling back to current-head catalog entries plus their `selection_projection` payloads when exact graph item documents lag behind the already visible current head
- consume a small bounded retry budget around remote `graph_path` and `graph_neighbors` expansion calls so one transient backend `503` does not eject the first turn out of query-first subgraph mode
- surface explainable graph-selection query plans in prompt/subtext payloads and workflow events so graph retrieval decisions can be inspected without dumping the full graph
- expose graph selection and graph rendering as separate steps so the long-term retriever can reuse one bounded query-first subgraph inside a larger mixed-source query plan instead of reselecting the graph during rendering
- support adding and deleting graph-backed contacts, preferences, and plans so discovery corrections can keep structured memory consistent
- build prompt-context and subtext payloads from persisted graph memory
- keep final prompt/subtext overlap checks permissive enough for exact short domain tokens such as `tea` or `sms`, so query-first graph selection is not nullified by an over-strict length filter on the last ranking step
- probe graph readiness through the current-view heads so required-remote watchdog recovery does not depend on legacy graph snapshot blobs
- cap readiness-only direct graph current-head probes to a small fail-closed timeout instead of the colder origin-resolution budget, so unhealthy remote memory surfaces quickly and does not drag service shutdown behind a long same-URI head read
- let fresh-reader bootstrap fall back from a lagging fixed-URI graph head to the already-written small compatibility current-head payload in a strictly read-only way, so cross-process readiness does not reseed or re-promote the whole graph just because direct head visibility lags
- treat a real fixed-URI graph-head `404` on a fresh empty namespace as a legitimate read-only bootstrap state, and synthesize the canonical empty current-view summary instead of forcing an empty remote seed write during required readiness
- keep compatible graph current-head fallback on the mutable snapshot URI instead of pinned exact document IDs, so fresh readers do not stick to superseded heads or drag startup through slow full-document recovery
- treat generic `graph_nodes` / `graph_edges` `catalog/current` payloads as insufficient for current-view readiness unless they also carry the graph generation metadata, and let graph repair writes proceed from a valid local cache even when the advertised current view is broken
- treat the direct fixed-URI `graph_nodes` / `graph_edges` head envelope itself as authoritative during repair/readiness checks; do not let a hidden `metadata.twinr_payload` shadow make an already generic/incomplete current-head record look healthy
- keep graph current-view `probe` and `load` semantics distinct: metadata-only probes may return `None` for incomplete `graph_edges` heads, but authoritative load paths must still perform the full fixed-URI current-head read before failing closed

`chonkydb` does **not** own:
- the ChonkyDB server implementation or deployment
- long-term extraction/retrieval orchestration outside the graph-store boundary
- top-level memory policy, provider selection, or web-dashboard behavior

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Public package exports |
| [client.py](./client.py) | HTTP client, one-shot top-k API boundary, and data path |
| [models.py](./models.py) | API request/response models, including structured top-k retrieval |
| [schema.py](./schema.py) | Graph schema and types |
| [_remote_graph_state.py](./_remote_graph_state.py) | Remote current-view sync and query-first subgraph selection, including read-only compatibility fallback when direct graph heads lag |
| [personal_graph.py](./personal_graph.py) | Graph store and prompt extraction |
| [component.yaml](./component.yaml) | Structured package metadata |

## Usage

```python
from twinr.memory.chonkydb import (
    ChonkyDBClient,
    ChonkyDBConnectionConfig,
    TwinrPersonalGraphStore,
)

client = ChonkyDBClient(ChonkyDBConnectionConfig(base_url="https://memory.example"))
graph_store = TwinrPersonalGraphStore.from_config(config)
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [longterm/storage](../longterm/storage/README.md)
- [longterm/retrieval](../longterm/retrieval/README.md)
- [../../../../docs/CHONKYDB_ONE_SHOT_RETRIEVAL.md](../../../../docs/CHONKYDB_ONE_SHOT_RETRIEVAL.md)
