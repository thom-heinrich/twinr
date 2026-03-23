# chonkydb

`chonkydb` owns Twinr's ChonkyDB integration package. It provides the
validated HTTP client, typed API payload models, and the personal-graph store
used by Twinr's long-term memory flows.

## Responsibility

`chonkydb` owns:
- call the ChonkyDB external API through a validated client boundary
- keep that client on a pooled, redirect-fail-closed transport so repeated Pi memory reads/writes do not pay a fresh DNS/TLS setup every call
- define typed request and response models for record, retrieve, and graph endpoints
- expose a one-shot `topk_records` retrieval contract that returns structured payloads from ChonkyDB in a single roundtrip
- expose async job-status reads so higher-level remote-memory writers can turn accepted async writes into exact `document_id` attestation instead of over-relying on eventually-consistent same-URI heads
- carry Twinr's `namespace + scope_ref` contract for current-snapshot retrieval so server-side scope resolution can replace oversized client allowlists
- define the canonical Twinr personal-graph schema and its local store
- keep the personal-graph inter-process lock on a runtime-shared lock path so
  watchdog and supervisor processes can coordinate even when the ChonkyDB data
  directory itself is root-owned
- support adding and deleting graph-backed contacts, preferences, and plans so discovery corrections can keep structured memory consistent
- build prompt-context and subtext payloads from persisted graph memory
- reuse the last successful remote graph document id during steady-state readiness reads so graph bootstrap avoids repeated pointer resolution when the remote snapshot has not moved

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
