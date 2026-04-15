# Twinr Live ChonkyDB Incident Report

Date: 2026-04-12

Audience: ChonkyDB operators and backend admins

Status: open

This report is intentionally self-contained. It is written so that ChonkyDB admins
can understand the Twinr-side production symptoms without needing access to Twinr's
internal artifact stores, local chat history, or fixreports.

## Executive summary

Twinr repeatedly observes the same contract failure pattern against the live
ChonkyDB lane:

1. `GET /v1/external/instance` can report `ready=true`.
2. In some runs it also reports `contract_scope=full`,
   `ready_scopes.full=true`, and `validation.full_ready=true`.
3. A real Twinr evaluation run then immediately fails on production endpoints such
   as:
   - `POST /v1/graph/store_many`
   - `GET /v1/external/documents/full`
   - `POST /v1/external/retrieve`
   - `POST /v1/external/retrieve/topk_records`
4. The failure modes are not benign:
   - `503 Service warmup in progress`
   - `503 Upstream unavailable or restarting`
   - `404 document_not_found` on data that was just accepted or should be visible
   - hard read timeouts on production requests

The main operational conclusion is:

`/v1/external/instance` readiness is currently not a reliable indicator that the
actual Twinr production query and write surface is stable under load.

Twinr therefore cannot safely use `instance ready/full_ready` as its only gate.

## Scope of what Twinr is doing

Twinr is not calling obscure or private ChonkyDB internals. The live runtime is
using the documented external and graph-facing production routes:

- readiness and contract probe:
  - `GET /v1/external/instance`
- graph topology persist:
  - `POST /v1/graph/store_many`
- long-term structured write:
  - `POST /v1/external/records/bulk`
- authoritative readback / current-head load:
  - `GET /v1/external/documents/full`
- long-term retrieval:
  - `POST /v1/external/retrieve`
  - `POST /v1/external/retrieve/topk_records`
- graph traversal support:
  - `POST /v1/external/graph/path`

Twinr's live routing topology is:

- public runtime URL:
  - `https://tessairact.com:2149`
- backend provenance:
  - public `2149` proxies to backend host `thh1986`
  - local backend loopback there is `http://127.0.0.1:3044`

Twinr is configured to fail closed when remote memory is unhealthy. There is no
allowed local mirror or fallback for long-term remote memory. In practice:

- remote memory healthy -> Twinr runs with normal long-term memory
- remote memory unhealthy -> Twinr runtime is blocked

That fail-closed behavior is intentional and correct.

## What Twinr repeatedly experiences

The failures fall into four externally visible buckets.

### 1. Ready checks can be green shortly before real writes fail

Example from 2026-04-12:

- direct public probe against `GET /v1/external/instance` returned:
  - `status=200`
  - `ready=true`
  - `contract_scope=full`
  - `ready_scopes.full=true`
  - `validation.full_ready=true`
  - `graph_ready=true`
  - `temporal_ready=true`
  - `vector_ready=true`

Immediately afterwards, a real Twinr evaluation run failed before the first test
case executed:

- request:
  - `POST /v1/graph/store_many`
- symptom:
  - hard read timeout at `15.0s`
- Twinr-side top error:
  - `LongTermRemoteUnavailableError: Failed to persist graph topology generation ...`

This directly demonstrates that `full_ready=true` is not currently sufficient as an
acceptance signal for real Twinr use.

### 2. Readiness can be green before the lane collapses into warmup or restart

On other runs, Twinr started from a green readiness state and the live lane then
degraded during the same evaluation into:

- `503 Service warmup in progress`
- `503 Upstream unavailable or restarting`
- post-run `backend_restart_required`

This happened on production read paths, not on optional diagnostics.

### 3. Snapshot and readback paths are unstable under real load

Twinr repeatedly saw failures on current-head and readback routes:

- `GET /v1/external/documents/full`
  - `503 Service warmup in progress`
  - `503 Upstream unavailable or restarting`
  - `404 document_not_found`
  - severe latency spikes, including multi-second and double-digit-second waits

These failures hit live snapshot heads such as:

- `__pointer__:graph_nodes`
- `graph_nodes`
- `midterm`
- `objects`

This matters because Twinr's long-term memory contract requires accepted writes to
become visible again on the authoritative read path. If the readback path is not
stable, Twinr must treat the memory backend as unavailable.

### 4. Retrieval surfaces degrade under live reader load

Twinr repeatedly saw degradation on:

- `POST /v1/external/retrieve`
- `POST /v1/external/retrieve/topk_records`
- `POST /v1/external/graph/path`

Observed symptoms included:

- `503 Service warmup in progress`
- `503 Upstream unavailable or restarting`
- multi-second slow reads
- hard read timeouts around `8s` and `15s`

This is especially damaging during Twinr's fresh-reader phase, where the whole point
is to verify that persisted long-term memory is actually retrievable after restart
and without writer-local context.

## Short chronology of the relevant live runs

The exact runs below were executed from `/home/thh/twinr` against the public live
lane at `https://tessairact.com:2149`.

### 2026-04-09: Retry 29

Probe ID:

- `20260409_unified_residual_fix_host_retry29`

Observed outcome:

- `status=ok`
- the run completed end-to-end
- this was the first strong proof that Twinr-side fixes for the `Anna Becker` and
  `Corinna` unified retrieval issues were working live

Important caveat:

- the run still carried ChonkyDB instability signs during execution, including
  remote read/query errors and degraded query surfaces

Why this matters:

- a good outcome was possible
- but it did not prove the lane was stable, only that it was sometimes good enough

### 2026-04-10: Retry 30

Probe ID:

- `20260410_unified_residual_fix_host_retry30`

Observed outcome:

- `status=ok`
- `ready=false`
- unified quality was effectively green:
  - all `anna_email_graph_only*` live cases passed
  - all `corinna_phone_full_stack*` live cases passed
  - all `corinna_recent_call_continuity*` live cases passed

Important caveat:

- non-unified categories were still not fully green
- at least one fresh-reader case already showed:
  - `LongTermRemoteUnavailableError: Failed to query remote long-term 'objects' catalog segments.`

Why this matters:

- even on one of the best runs, the lane was still producing real remote-memory
  failures on production read paths

### 2026-04-10: Retry 31

Probe ID:

- `20260410_unified_residual_fix_host_retry31`

Observed outcome:

- `status=ok`
- `ready=false`
- writer side still looked much better than fresh-reader side
- fresh-reader collapsed hard compared with earlier stronger runs

Twinr-side observed pattern during the run:

- live reads degraded into:
  - `GET /v1/external/documents/full -> 503`
  - `POST /v1/external/retrieve -> 503`
  - `POST /v1/external/retrieve/topk_records -> 503`
- response detail on several of those failures:
  - `Service warmup in progress`

Post-run gate:

- the same lane then reported `backend_restart_required`
- public `2149` and backend `3044` both returned warmup / restart signals

Why this matters:

- the lane did not just run slowly
- it visibly drifted from "usable" into "warmup / restart" during or immediately
  around live Twinr reader traffic

### 2026-04-10: Retry 32

Probe ID:

- `20260410_unified_residual_fix_host_retry32`

Observed outcome:

- `status=failed`
- `executed=false`

Immediate failure:

- `POST /v1/graph/store_many`
- `503`
- `detail="Upstream unavailable or restarting"`

This is a direct contradiction to the idea that a green `instance` response means
the graph write surface is ready for Twinr.

### 2026-04-12: Retry 33

Probe ID:

- `20260412_unified_residual_fix_host_retry33`

Observed outcome:

- `status=failed`
- `executed=false`

Immediate failure:

- `POST /v1/graph/store_many`
- hard read timeout
- `read timeout=15.0`

Why retry33 is especially important:

- this happened after a direct public readiness probe that returned:
  - `ready=true`
  - `contract_scope=full`
  - `ready_scopes.full=true`
  - `validation.full_ready=true`

This is the clearest recent reproduction of the contract mismatch:

- readiness says "full"
- first real graph topology write still times out

### Current spot check on 2026-04-12

The most recent direct public check on `GET /v1/external/instance` returned:

- `503`
- `detail="Service warmup in progress"`

So the lane is not only intermittently bad under load; at this point it is again
publicly advertising an active warmup state on the top-level external instance route.

## Host-side findings from the backend machine

Twinr also inspected the backend host repository and service state directly to answer
the question "did something change on the host side?"

The short answer is:

- yes, there are strong signs that the operational readiness and query-surface
  behavior on the host has been actively modified
- no, we have not yet isolated a single clean commit that alone explains the full
  live failure pattern

### The backend host has its own bug and fix memory

On `/home/thh/tessairact` there is an independent repo and artifact setup with
stores such as:

- `artifacts/stores/chat`
- `artifacts/stores/findings`
- `artifacts/stores/fixreport`
- `artifacts/stores/hypothesis`
- `artifacts/stores/healthstream`
- `artifacts/stores/tasks`

This matters because the ChonkyDB side already has its own durable debugging memory.
Twinr is not the only place where these incidents can be tracked.

### The ChonkyDB service is running with many active override files

From the live systemd status on the backend host, the active ChonkyDB unit has these
drop-ins:

- `10-refuse-manual-restart.conf`
- `10-startup-readiness.conf`
- `20-startup-hash-perf.conf`
- `30-fulltext-rebuild-on-open.conf`
- `40-twinr-query-surface-readiness.conf`
- `45-twinr-disable-payload-read-gate.conf`
- `46-twinr-disable-sync-bulk-api-ready-gate.conf`
- `47-twinr-bound-sync-write-timeout.conf`
- `48-twinr-bound-origin-lookup-timeout.conf`
- `49-twinr-sync-default-targets.conf`
- `50-commit-durability-full.conf`
- `51-twinr-vector-read-surface-rootcause.conf`
- `52-twinr-vector-warmup-budget.conf`
- `53-twinr-disable-graph-startup.conf`
- `99-docid-correctness.conf`
- `override.conf`

This is not a pristine or default service shape. It is an actively modified service
with explicit Twinr-facing readiness, timeout, gating, vector, and graph-startup
behavior changes.

The most important operational implication is:

- the meaning of "ready" is now partly defined by this override layer
- any mismatch between `instance ready` and actual surface stability may live there,
  not only in Twinr

### Host-side service logs show heavy read-path contention

During the same time windows as the live Twinr incidents, the backend service logs
contained repeated warnings such as:

- `ChonkFileCore.read_block.telemetry`
- `ChonkService.payload_read_telemetry slow_read`
- `external_hit_validation ... /v1/external/retrieve/topk_records ... latency_ms=2801.162`
- payload reads in the `0.25s` to `1.4s` range
- non-zero lock waits, snapshot lock waits, and cache-commit lock waits

The most striking write-path indicators around the retry33 window included:

- `_extend_physical_file slow`
  - `file_type=DOCID_MAPPING`
  - extend operation around `1504 ms`
- `DocIDMap logseg persist perf`
  - one `external_payload_sync_bulk` persist around `226 ms`
  - another around `1197 ms`

These log lines do not by themselves prove why `graph/store_many` timed out, but they
do show that the backend was not in a low-latency steady state while Twinr was using
it.

### Host-side git history is non-trivial and partially damaged

A focused git log on backend source files relevant to API service implementation and
graph APIs showed recent changes in March 2026, including commits:

- `58127962` on `2026-03-11`
- `b75238f0` on `2026-03-06`
- `22fe1470` on `2026-03-05`
- `7f2c5489` on `2026-03-05`
- `f48a5099` on `2026-03-03`

Twinr could not cleanly walk the full relevant history because the host repository
reported tree-object read failures while traversing some paths, including:

- `fatal: unable to read tree ...`
- `fatal: cannot simplify commit ...`

This does not prove that git corruption is causing the live API issue, but it does
mean the host codebase is not in a clean, easily auditable state right now.

## Why this is not a Twinr-only bug

This report is not claiming that every bad benchmark score is ChonkyDB's fault.
Twinr did have real retrieval bugs earlier in this investigation, and those were
fixed in Twinr.

However, the repeated failures described here are independent of those Twinr-side
quality fixes. The relevant reasons are:

1. The failures often happen before the first case executes.
   - example:
     - immediate `POST /v1/graph/store_many` timeout or `503`
2. The failures hit authoritative ChonkyDB paths directly.
   - example:
     - `GET /v1/external/documents/full`
     - `POST /v1/external/retrieve`
3. The same lane can move from `ready/full_ready` to warmup or restart symptoms.
4. The host service itself has Twinr-specific readiness and query-surface overrides.

Twinr therefore treats this as a backend contract problem, not as a benchmark-only
problem.

## What Twinr needs from ChonkyDB

Twinr needs a stronger operational guarantee than the current `instance ready` signal.

The minimum acceptable contract is:

1. if `GET /v1/external/instance` says `ready=true` and `full_ready=true`,
   then the following operations must also be stable immediately afterwards:
   - `POST /v1/graph/store_many`
   - `POST /v1/external/records/bulk`
   - `GET /v1/external/documents/full`
   - `POST /v1/external/retrieve`
   - `POST /v1/external/retrieve/topk_records`
2. accepted writes must become visible again on the authoritative read path
3. the lane must not fall back into `warmup in progress` during a normal Twinr
   writer -> restart -> fresh-reader cycle

In other words:

- "instance ready" must mean "the real production write and read surfaces are stable"
- not merely "one narrow health endpoint is currently answering"

## Concrete operator questions and suspected areas

These are the operator-side questions Twinr thinks are worth answering next.

### 1. What exactly is `full_ready` gating today?

Given the active drop-ins, verify the true meaning of:

- `contract_scope`
- `ready_scopes.full`
- `validation.full_ready`

Specifically:

- does `full_ready=true` require `graph/store_many` to succeed under real write load?
- does it require `documents/full` current-head reads to be stable?
- does it require `retrieve` and `retrieve/topk_records` to be stable?

If the answer is "no", then Twinr needs a different canary or the readiness contract
must be tightened.

### 2. Why does the service re-enter warmup or restart states during live use?

Investigate the exact transition path from:

- public ready state

to:

- `Service warmup in progress`
- `Upstream unavailable or restarting`
- `backend_restart_required`

Possible focus areas:

- startup-readiness vs query-surface-readiness interaction
- graph-startup gating behavior
- vector and temporal warmup coupling
- origin lookup and current-head resolution
- payload-read slow paths under load
- docid mapping growth / extension stalls

### 3. Why can graph topology persist fail first?

`POST /v1/graph/store_many` is repeatedly the first production operation to fail in
some runs. That makes it a strong candidate for a real preflight canary.

Questions to answer:

- is graph topology persist blocked by a hidden dependency that readiness does not
  wait for?
- does `53-twinr-disable-graph-startup.conf` change the relationship between graph
  readiness and actual graph write readiness?
- is the timeout path proxy-side, router-side, or lower-level storage-side?

### 4. Are the service overrides still coherent as a set?

The active drop-in stack is now large. It is plausible that the current behavior is
emerging from override interaction rather than one single bug.

At minimum, audit the combined effect of:

- `40-twinr-query-surface-readiness.conf`
- `45-twinr-disable-payload-read-gate.conf`
- `46-twinr-disable-sync-bulk-api-ready-gate.conf`
- `47-twinr-bound-sync-write-timeout.conf`
- `51-twinr-vector-read-surface-rootcause.conf`
- `52-twinr-vector-warmup-budget.conf`
- `53-twinr-disable-graph-startup.conf`

## Reproduction pattern for admins

The simplest high-value reproduction is:

1. verify public `GET /v1/external/instance`
2. do not stop there
3. immediately run a real Twinr-style sequence:
   - `POST /v1/graph/store_many`
   - `POST /v1/external/records/bulk`
   - `GET /v1/external/documents/full`
   - `POST /v1/external/retrieve`
   - `POST /v1/external/retrieve/topk_records`
4. keep the lane under a realistic writer -> reader cycle, not just one isolated ping
5. confirm the service does not degrade into warmup or restart signals during that cycle

If ChonkyDB wants a dedicated Twinr-facing readiness probe, it should look more like
that sequence and less like a single instance ping.

## What Twinr is not asking for

Twinr is not asking for:

- a local fallback
- a local mirror
- a degraded mode that hides backend failures
- relaxed consistency on accepted writes

Twinr's requirement is simpler:

- if remote memory is the primary source of truth, the production read and write
  surfaces must actually be stable when readiness says they are stable

## Final conclusion

The current production issue is not best described as "Twinr hit a random timeout."
The repeated evidence supports a stronger statement:

- the live ChonkyDB lane currently exposes a readiness contract that is too weak for
  Twinr's real production use
- host-side service overrides and live logs strongly suggest that readiness,
  warmup, graph startup, query-surface gating, and storage contention are all part
  of the same operational problem space
- until `instance/full_ready` reliably predicts success on `graph/store_many`,
  `documents/full`, `retrieve`, and `retrieve/topk_records`, Twinr cannot treat the
  lane as production-stable

## Recommended next actions for ChonkyDB admins

1. Reproduce the mismatch explicitly:
   - verify `full_ready=true`
   - immediately run `graph/store_many` and the external read/query surface
   - confirm where the first failure appears
2. Audit the active systemd drop-ins as one combined readiness/gating system, not as
   isolated tweaks.
3. Decide whether:
   - the readiness contract must become stricter, or
   - the underlying warmup / graph / retrieval surfaces must be stabilized so that the
     current readiness contract becomes true in practice.
