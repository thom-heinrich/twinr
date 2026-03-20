# ChonkyDB One-Shot Retrieval Target

This document describes the target ChonkyDB retrieval contract Twinr needs to
use ChonkyDB as a real low-latency multi-index retrieval engine instead of a
remote authority plus locally warmed projection cache.

## Problem

Twinr's current remote-primary long-term path still pays too much per-turn
coordination cost when it tries to use ChonkyDB as a query-time retrieval
engine:

- the client first resolves the current logical snapshot into a catalog
- the client sends large `allowed_doc_ids` sets to `/v1/external/retrieve`
- the response returns ranked candidates, but not the final structured Twinr
  payloads
- Twinr then performs one or more follow-up item fetches or retrieve batches to
  hydrate the selected records

That means the live path is effectively:

1. resolve current catalog
2. search
3. hydrate top-k
4. deserialize
5. rank/render in Twinr

This is acceptable for storage correctness, but it is not the right shape for a
sub-second Pi turn path.

## Target outcome

The target path is a single remote request that:

- resolves the current logical Twinr scope server-side
- runs lexical/vector/metadata retrieval server-side
- reranks server-side
- returns the final top-k structured record payloads directly
- includes score breakdown and query-plan diagnostics in the same response

The live turn path should become:

1. send one request
2. deserialize top-k payloads
3. render Twinr context

No second network roundtrip should be required for ordinary top-k recall.

## Non-goals

- returning the entire current working set on every query
- moving Twinr-specific prompt rendering into ChonkyDB
- encoding Twinr user-specific heuristics into the database engine
- replacing ChonkyDB as the authoritative remote store

## Current bottlenecks

The current contract has three structural latency costs:

### 1. Candidate scoping is client-heavy

Twinr currently computes the current candidate set and sends it as
`allowed_doc_ids`. That scales badly in request bytes, parsing, and repeated
client work.

### 2. Retrieval and hydration are separate

`/v1/external/retrieve` ranks candidates, but Twinr still needs follow-up reads
to obtain the structured payloads it actually uses for memory rendering.

### 3. The server does not expose a logical "current snapshot scope"

Twinr already has the concept of:

- namespace
- snapshot kind: `objects`, `conflicts`, `archive`, `midterm`, `graph`
- current logical head

But the search endpoint does not accept that as a first-class scope reference.

## Target API

Recommended new endpoint:

`POST /v1/external/retrieve/topk_records`

The name is less important than the contract. The essential change is that the
endpoint returns final top-k records, not just ranked candidate handles.

### Request

```json
{
  "namespace": "twinr-prod",
  "scope_ref": "longterm:objects:current",
  "query_text": "Wo stand früher meine rote Thermoskanne?",
  "result_limit": 6,
  "response_mode": "structured_records",
  "filters": {
    "status": ["active", "candidate", "uncertain"]
  },
  "ranking": {
    "lexical_weight": 1.0,
    "vector_weight": 1.0,
    "freshness_weight": 0.15,
    "metadata_weight": 0.25
  },
  "include": {
    "payload": true,
    "metadata": true,
    "content": false,
    "score_breakdown": true,
    "query_plan": true
  },
  "timeout_seconds": 0.8,
  "client_request_id": "..."
}
```

### Response

```json
{
  "success": true,
  "scope_ref": "longterm:objects:current",
  "results": [
    {
      "rank": 1,
      "document_id": "doc-123",
      "item_id": "fact:thermos_location_old",
      "payload_schema": "twinr_memory_object_record_v2",
      "payload": {
        "memory_id": "fact:thermos_location_old",
        "kind": "fact",
        "summary": "Früher stand deine rote Thermoskanne im Flurschrank.",
        "status": "active",
        "slot_key": "location:thermos",
        "value_key": "flurschrank"
      },
      "metadata": {
        "twinr_snapshot_kind": "objects",
        "updated_at": "2026-03-19T12:00:00Z",
        "payload_sha256": "..."
      },
      "score": 0.98,
      "score_breakdown": {
        "lexical": 0.91,
        "vector": 0.88,
        "freshness": 0.31,
        "metadata": 0.67
      }
    }
  ],
  "query_plan": {
    "candidate_scope_mode": "scope_ref",
    "candidate_count": 418,
    "indexes_used": ["bm25", "vector", "metadata"],
    "reranked_count": 24,
    "latency_ms": {
      "scope_resolve": 3.1,
      "candidate_fetch": 8.4,
      "index_search": 29.7,
      "rerank": 4.5,
      "payload_materialize": 6.2,
      "total": 52.6
    }
  }
}
```

## Required semantics

### 1. First-class scope references

The request must accept a server-side scope reference such as:

- `longterm:objects:current`
- `longterm:conflicts:current`
- `longterm:archive:current`
- `graph:current`
- `midterm:current`

ChonkyDB should resolve that scope internally to the current active candidate
set. Twinr should not ship thousands of `allowed_doc_ids` on every query.

### 2. Structured payload return

The endpoint must return the normalized application payload that Twinr actually
uses, not only document ids plus free-text content.

This is what removes the follow-up hydration roundtrip.

### 3. Multi-index retrieval stays server-side

The server should combine and rerank:

- lexical full-text
- vector similarity
- metadata/facet filters
- optional temporal constraints

Twinr should receive the already merged top-k set.

### 4. Explainability

Every response should optionally include:

- score breakdown
- indexes used
- candidate count
- latency partitioning

This is critical for debugging live Pi latency regressions without packet
captures or guesswork.

### 5. Backward compatibility

The new endpoint should not replace `/v1/external/retrieve` immediately.

Recommended rollout:

- keep `/retrieve` for generic search clients
- add `/retrieve/topk_records` for low-latency structured retrieval
- migrate Twinr long-term object/conflict paths first
- only later consider reusing it for graph and midterm retrieval

## Server-side execution model

The target execution model inside ChonkyDB should look like this:

1. resolve `scope_ref` to a current logical candidate set
2. collect the scoped postings/vector candidates server-side
3. merge and rerank candidates
4. materialize the final top-k structured payloads directly from the hot
   document store
5. return payloads and diagnostics in one response

Important: payload materialization should happen after reranking, not for the
entire scoped candidate set.

## Storage implications

To make this fast, ChonkyDB likely needs one of these internal shapes:

### Option A: payload-aware retrieve pipeline

Keep the current record store, but let the retrieve pipeline fetch the final
top-k payloads directly from the authoritative record/document storage layer.

This is the simplest migration path.

### Option B: payload sidecar in retrieval index

Store a compact structured payload sidecar with the retrieval document so the
server can return final payloads without another storage hop.

This is faster, but it increases index write complexity and invalidation cost.

### Recommendation

Start with **Option A**. It is operationally safer and preserves one
authoritative document source. Only add sidecars if profiling proves the final
payload materialization hop is still too expensive.

## Twinr integration target

Once the endpoint exists, Twinr should replace this current split path:

- `search_catalog_entries(...)`
- `load_item_payloads(...)`
- local cache-based selector fallback

with one storage-level call per scope:

- `topk_object_payloads(query, limit, filters)`
- `topk_conflict_payloads(query, limit, filters)`

The Twinr retriever should then only:

1. deserialize the returned payloads into typed models
2. apply Twinr-specific relevance/confirmation policy
3. render prompt context

The local working-set prewarm can then shrink back to:

- remote readiness
- tiny metadata hints
- optional very small safety cache

and should no longer need to hydrate the full current object/conflict payload
set.

## Performance targets

For Twinr Pi use, the target budgets should be:

- ChonkyDB server p95 for `topk_records`: `<150 ms`
- Pi to server end-to-end p95: `<350 ms`
- Twinr memory context build after response parse: `<250 ms`
- total first query recall on Pi: `<700 ms`

Warm-path total should target `<300 ms`.

## Migration phases

### Phase 1: API contract

- add `scope_ref` request support
- add structured top-k response support
- add query-plan latency partitioning

### Phase 2: Twinr object/conflict adoption

- teach the Twinr ChonkyDB client the new endpoint
- switch object/conflict retrieval to one-shot top-k calls
- keep current cache-based path as rollback fallback

### Phase 3: remove oversized client scoping

- delete the large `allowed_doc_ids` query path for normal object/conflict
  recall
- keep `allowed_doc_ids` only for explicit debugging or special bounded calls

### Phase 4: reduce local warm projection

- stop full payload prewarm for ordinary startup
- keep only minimal readiness and tiny hot caches

## Why this is the right target

This keeps the architecture clean:

- ChonkyDB remains the authoritative remote memory system
- ChonkyDB also becomes the real multi-index retrieval engine
- Twinr stops reconstructing a search engine client-side
- the Pi hot path goes back to one remote read plus local rendering

That is the shape Twinr originally wanted, and it is the right long-term target
if ChonkyDB is meant to be more than a remote document store.
