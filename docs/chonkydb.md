# ChonkyDB Basics

Twinr now includes a small external ChonkyDB client under `src/twinr/memory/chonkydb/`.

The current scope is intentionally narrow:

- inspect instance readiness
- inspect auth contract
- list records
- fetch one record
- fetch one full document
- retrieve search results
- submit single or bulk record writes
- define a versioned Twinr graph schema
- call graph edge, neighbor, path, and pattern endpoints

The external client is still intentionally narrow. Twinr's normal live runtime now routes long-term retrieval and non-blocking episodic persistence through `src/twinr/memory/longterm/`, while remote ChonkyDB network writes remain an explicit future integration step.

## Twinr graph schema v1

Twinr now carries a small versioned graph format under `src/twinr/memory/chonkydb/schema.py`.

The graph document format is JSON-serializable and intentionally small:

```json
{
  "schema": {"name": "twinr_graph", "version": 1},
  "subject_node_id": "user:main",
  "nodes": [
    {"id": "user:main", "type": "user", "label": "Erika"},
    {"id": "person:corinna_maier", "type": "person", "label": "Corinna Maier"}
  ],
  "edges": [
    {
      "source": "person:corinna_maier",
      "type": "social_supports_user_as",
      "target": "user:main",
      "status": "active",
      "attributes": {"role": "physiotherapist"},
      "confirmed_by_user": true
    }
  ]
}
```

The current v1 edge namespaces are:

- `social_*`
- `general_*`
- `temporal_*`
- `spatial_*`
- `user_*`

The initial allowed edge catalog is intentionally narrow and additive. New edge types can be added later without breaking older records as long as existing edge meanings are not silently changed.

### First live Twinr graph flows

Twinr now has a first local personal-graph layer on top of the schema:

- remember contacts with phone, email, relation, and role
- look up contacts with clarification when multiple matches exist
- remember stable preferences such as liked brands or favored shops
- remember short user plans such as wanting to go for a walk today
- feed a compact graph-based personalization hint into the provider context

The current implementation is local-first and writes to `twinr_graph_v1.json` under the configured `TWINR_LONG_TERM_MEMORY_PATH`. It does not yet push real graph writes into the shared external ChonkyDB service during normal Twinr runtime.

### Canonical language contract

Twinr now treats memory language and user-output language as separate concerns:

- persistent memory and profile payloads should use canonical English for semantic text
- names, phone numbers, email addresses, IDs, codes, and direct quotes stay verbatim
- the provider context gets a structured English memory graph
- the user still hears and sees replies in the configured Twinr spoken language

This avoids coupling memory quality to one user language and keeps the memory layer stable even when the user speaks German or switches languages.

### Current v1 edge types

- `social_family_of`
- `social_friend_of`
- `social_supports_user_as`
- `general_alias_of`
- `general_carries_brand`
- `general_has_contact_method`
- `general_related_to`
- `general_sells`
- `temporal_occurs_on`
- `temporal_usually_happens_in`
- `temporal_valid_from`
- `temporal_valid_to`
- `spatial_located_in`
- `spatial_near`
- `user_dislikes`
- `user_likes`
- `user_plans`
- `user_prefers_brand`
- `user_usually_buys_at`

### Why v1 looks like this

- stable IDs keep people and places disambiguated
- typed edges give Twinr structured recall for relationships and preferences
- document-level schema versioning allows additive growth later
- edge status and confidence support clarification instead of silent overwrite

## Canonical Twinr env names

Use Twinr naming in `.env`:

```bash
TWINR_LONG_TERM_MEMORY_ENABLED=true
TWINR_LONG_TERM_MEMORY_BACKEND=chonkydb
TWINR_LONG_TERM_MEMORY_PATH=state/chonkydb

TWINR_CHONKYDB_BASE_URL=https://tessairact.com:2149
TWINR_CHONKYDB_API_KEY=replace-me
TWINR_CHONKYDB_API_KEY_HEADER=x-api-key
TWINR_CHONKYDB_ALLOW_BEARER_AUTH=false
TWINR_CHONKYDB_TIMEOUT_S=20
```

Compatibility fallback is present for legacy `CCODEX_MEMORY_BASE_URL` and `CCODEX_MEMORY_API_KEY`, but the Twinr code and docs should call the backend `chonkydb`.

## Minimal usage

```python
from twinr.config import TwinrConfig
from twinr.memory.chonkydb import ChonkyDBClient, ChonkyDBRetrieveRequest

config = TwinrConfig.from_env(".env")
client = ChonkyDBClient.from_twinr_config(config)

instance = client.instance()
auth = client.auth_info()
search = client.retrieve(
    ChonkyDBRetrieveRequest(
        query_text="medication reminder",
        result_limit=3,
        include_metadata=True,
    )
)
```

Graph helpers are also available:

```python
from twinr.memory.chonkydb import (
    ChonkyDBGraphAddEdgeSmartRequest,
    ChonkyDBGraphNeighborsRequest,
    TwinrGraphDocumentV1,
    TwinrGraphEdgeV1,
    TwinrGraphNodeV1,
)

document = TwinrGraphDocumentV1(
    subject_node_id="user:main",
    nodes=(
        TwinrGraphNodeV1(node_id="user:main", node_type="user", label="Erika"),
        TwinrGraphNodeV1(node_id="brand:melitta", node_type="brand", label="Melitta"),
    ),
    edges=(
        TwinrGraphEdgeV1(
            source_node_id="user:main",
            edge_type="user_prefers_brand",
            target_node_id="brand:melitta",
            confidence=0.86,
        ),
    ),
)

client.add_graph_edge_smart(
    ChonkyDBGraphAddEdgeSmartRequest(
        from_ref="user:main",
        to_ref="brand:melitta",
        edge_type="user_prefers_brand",
    )
)

neighbors = client.graph_neighbors(
    ChonkyDBGraphNeighborsRequest(
        label_or_id="user:main",
        edge_types=("user_prefers_brand",),
        with_edges=True,
    )
)
```

## Notes

- Both `x-api-key` and `Authorization: Bearer ...` are supported by the live service.
- The current shared service still identifies itself as `ccodex_memory` in some response fields. Twinr treats that as remote service metadata, not as the local product name.
- The public `https://tessairact.com:2149` endpoint is currently reachable from the Twinr server and the Pi.
