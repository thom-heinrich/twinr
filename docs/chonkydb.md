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

The external client now also backs Twinr's remote-primary long-term memory mode through `src/twinr/memory/longterm/remote_state.py`. In that mode, ChonkyDB becomes the primary long-term source, while local `state/chonkydb` files remain cache and migration artifacts.

## Twinr graph schema v2

Twinr now carries a small versioned graph format under `src/twinr/memory/chonkydb/schema.py`.

The graph document format is JSON-serializable and intentionally small:

```json
{
  "schema": {"name": "twinr_graph", "version": 2},
  "subject_node_id": "user:main",
  "nodes": [
    {"id": "user:main", "type": "user", "label": "Erika"},
    {"id": "person:corinna_maier", "type": "person", "label": "Corinna Maier"}
  ],
  "edges": [
    {
      "source": "person:corinna_maier",
      "type": "social_related_to_user",
      "target": "user:main",
      "status": "active",
      "attributes": {"role": "physiotherapist", "relation": "trusted_helper"},
      "confirmed_by_user": true
    }
  ]
}
```

The current v2 edge namespaces are:

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
- remember stable preferences such as favored products, places, brands, services, or recurring activities
- remember short user plans such as wanting to go for a walk today
- feed a compact graph-based personalization hint into the provider context

The current implementation still writes a local graph cache under the configured `TWINR_LONG_TERM_MEMORY_PATH`. When `TWINR_LONG_TERM_MEMORY_MODE=remote_primary`, that graph cache is synced through ChonkyDB-backed remote snapshots and is no longer treated as the source of truth.

### Canonical language contract

Twinr now treats memory language and user-output language as separate concerns:

- persistent memory and profile payloads should use canonical English for semantic text
- names, phone numbers, email addresses, IDs, codes, and direct quotes stay verbatim
- the provider context gets a structured English memory graph
- the user still hears and sees replies in the configured Twinr spoken language

This avoids coupling memory quality to one user language and keeps the memory layer stable even when the user speaks German or switches languages.

### Current v2 edge types

- `social_related_to`
- `social_related_to_user`
- `general_alias_of`
- `general_has_contact_method`
- `general_related_to`
- `temporal_occurs_on`
- `temporal_usually_happens_in`
- `temporal_valid_from`
- `temporal_valid_to`
- `spatial_located_in`
- `spatial_near`
- `user_avoids`
- `user_engages_with`
- `user_plans`
- `user_prefers`

### Why v2 looks like this

- stable IDs keep people and places disambiguated
- typed edges give Twinr structured recall for relationships and preferences
- document-level schema versioning allows additive growth later
- edge status and confidence support clarification instead of silent overwrite

## Canonical Twinr env names

Use Twinr naming in `.env`:

```bash
TWINR_LONG_TERM_MEMORY_ENABLED=true
TWINR_LONG_TERM_MEMORY_BACKEND=chonkydb
TWINR_LONG_TERM_MEMORY_MODE=remote_primary
TWINR_LONG_TERM_MEMORY_REMOTE_REQUIRED=true
TWINR_LONG_TERM_MEMORY_PATH=state/chonkydb
TWINR_LONG_TERM_MEMORY_REMOTE_READ_TIMEOUT_S=8
TWINR_LONG_TERM_MEMORY_REMOTE_WRITE_TIMEOUT_S=15
TWINR_LONG_TERM_MEMORY_MIGRATION_ENABLED=true

TWINR_CHONKYDB_BASE_URL=https://tessairact.com:2149
TWINR_CHONKYDB_API_KEY=replace-me
TWINR_CHONKYDB_API_KEY_HEADER=x-api-key
TWINR_CHONKYDB_ALLOW_BEARER_AUTH=false
TWINR_CHONKYDB_TIMEOUT_S=20
```

Compatibility fallback is present for legacy `CCODEX_MEMORY_BASE_URL` and `CCODEX_MEMORY_API_KEY`, but the Twinr code and docs should call the backend `chonkydb`.

## Current live topology

Twinr's live runtime endpoint and the backend operator endpoint are not the same
thing:

- `.env` carries the live Twinr runtime URL: `https://tessairact.com:2149`
- `.env.chonkydb` carries operator/backend provenance for the dedicated
  ChonkyDB instance on `thh1986`
- the current live routing contract is:
  `https://tessairact.com:2149` -> `thh1986.ddns.net` local `127.0.0.1:3044`

Do not repoint Twinr directly to `thh1986.ddns.net:3044` unless the public
proxy/routing contract has deliberately changed and the Pi/runtime config has
been updated together.

For P0 operator recovery on the development host, Twinr now ships a dedicated
repair helper:

```bash
python3 hardware/ops/repair_remote_chonkydb.py --no-restart
python3 hardware/ops/repair_remote_chonkydb.py
```

It probes the public `https://tessairact.com:2149` URL, the dedicated backend
systemd unit on `thh1986`, and the backend loopback
`http://127.0.0.1:3044/v1/external/instance` separately. The helper only
restarts the backend when the backend itself is unhealthy; if the backend
loopback is already healthy and only the public URL is down, it reports that
as a public proxy/routing outage instead of causing extra downtime with a blind
restart.

In `remote_primary` mode:

- ChonkyDB is the primary long-term source of truth.
- local `state/chonkydb/*.json` files are still updated as cache and migration artifacts.
- if ChonkyDB is unavailable and `TWINR_LONG_TERM_MEMORY_REMOTE_REQUIRED=true`, Twinr withholds long-term graph/fact/midterm memory instead of silently replaying stale local long-term state.

## Continuous watchdog

Twinr now ships a dedicated rolling watchdog for required remote memory:

```bash
PYTHONPATH=src python3 -m twinr --env-file .env --watch-remote-memory
```

The watchdog:

- runs the same fail-closed `LongTermMemoryService.ensure_remote_ready()` probe Twinr uses at runtime
- prints one JSON sample line per probe for journald/systemd capture
- writes a rolling artifact to `artifacts/stores/ops/remote_memory_watchdog.json`
- defaults to a `1.0s` cadence via `TWINR_LONG_TERM_MEMORY_REMOTE_WATCHDOG_INTERVAL_S`
- accepts `TWINR_LONG_TERM_MEMORY_REMOTE_WATCHDOG_PROBE_TIMEOUT_S` as the steady-state base timeout budget
- accepts `TWINR_LONG_TERM_MEMORY_REMOTE_WATCHDOG_STARTUP_PROBE_TIMEOUT_S` as the pre-first-success startup budget; when unset, Twinr keeps the 15s steady-state base timeout, grants the first deep probe up to 45s, and then pads the effective in-flight cutoff from recent successful probe latencies so slow-but-healthy archive checks do not synthesize false fail snapshots

For Pi deployment, install the provided systemd unit under [`hardware/ops`](../hardware/ops/README.md).

## Minimal usage

```python
from twinr.agent.base_agent import TwinrConfig
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
            edge_type="user_prefers",
            target_node_id="brand:melitta",
            confidence=0.86,
            attributes={"category": "brand", "for_product": "coffee"},
        ),
    ),
)

client.add_graph_edge_smart(
    ChonkyDBGraphAddEdgeSmartRequest(
        from_ref="user:main",
        to_ref="brand:melitta",
        edge_type="user_prefers",
    )
)

neighbors = client.graph_neighbors(
    ChonkyDBGraphNeighborsRequest(
        label_or_id="user:main",
        edge_types=("user_prefers",),
        with_edges=True,
    )
)
```

## Notes

- Both `x-api-key` and `Authorization: Bearer ...` are supported by the live service.
- The current shared service still identifies itself as `ccodex_memory` in some response fields. Twinr treats that as remote service metadata, not as the local product name.
- The public `https://tessairact.com:2149` endpoint is currently reachable from the Twinr server and the Pi and proxies to the dedicated `thh1986` backend on local `127.0.0.1:3044`.

## Retrieval target

Twinr's current remote-primary path is operationally fast now, but it still
uses a hybrid of remote authority plus locally warmed projection caches for
sub-second Pi recall. The target ChonkyDB contract for getting back to a true
one-shot remote retrieval engine is documented in
[CHONKYDB_ONE_SHOT_RETRIEVAL.md](./CHONKYDB_ONE_SHOT_RETRIEVAL.md).
