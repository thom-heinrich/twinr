---
name: instrument
description: >
  Forensic-grade instrumentation for Python systems (ingest pipelines, LLM runs, agents, tools, retrieval, orchestration),
  producing first-run evidence that makes every decision, branch, cost/KPI driver, and failure mode transparent and
  replayable. Output: correlated structured logs + traces + metrics + runpack. Redaction-by-default.
---

## Objective
Instrument a Python codebase so that **a single run** yields enough **evidence** to answer, for any observed behavior:

- **What** happened (event + payload summary)
- **When** it happened (monotonic + wall clock)
- **Where** it happened (file/line/function + thread/task/process)
- **In which context** (run_id, request/job_id, trace_id/span_id, agent_id, turn_id)
- **Why** it happened (explicit decision record: options, scores, constraints, selected action, rationale)
- **What it cost** (latency, tokens, $-cost estimate, CPU/RAM/IO, queueing, retries)
- **What it affected** (state deltas, cache hits/misses, downstream calls, produced artifacts)

Maxim: “Complete” means **complete decision accountability + cost attribution**, not indiscriminate dumping.

## When to use
- Ingest + indexing pipelines (ETL, chunking, embedding, upserts)
- LLM inference + tool use + agent loops (planner/executor/reflection)
- Any system where “why did it do X” or “where did performance go” matters

## When NOT to use
- If the user requests logging secrets/PII verbatim.
- If production constraints forbid overhead and you cannot apply sampling/scoping.

## Hard Safety Constraints (non-negotiable)
Redaction-by-default:
- Never log: API keys, tokens, passwords, private keys, cookies, Authorization headers, session identifiers.
- Never log full raw documents or full prompts/responses by default.
- Instead log: hashes, lengths, MIME/types, schema summaries, selected whitelisted fields, and **repro pointers**.
- All log fields must be size-bounded and escaped.

## Definition of Done (first-run evidence pack)
A single run produces:
1) `run.jsonl` — strict-schema structured events (JSON Lines)
2) `run.trace` — OpenTelemetry traces export (OTLP to collector or file depending on env)
3) `run.metrics.json` — KPI time series and aggregates (latency/token/cost/resource)
4) `run.summary.json` — top decisions, slow spans, failure clusters, retry stats, cache stats, bottlenecks
5) `run.repro/` — sanitized config snapshot, args, seeds, git commit, version info, environment whitelist
6) Optional: `run.profile.txt` — profiler output for hotspots (mode dependent)

## Instrumentation Modes (must be explicit)
- **MODE=forensic (default for debugging)**: maximum evidence, narrow scope + tight redaction
- **MODE=prod**: sampling + rate limits + minimal payloads, still decision-accountable
- **MODE=deep-exec**: runtime event monitoring for branch/jump/line events (use only when needed)

## Core Design: Three Signals + One “Decision Ledger”
You must implement all of these:

### A) Structured Logs (schema-stable)
Every event is a JSON object with these required fields:
- `ts_wall_utc` (ISO8601), `ts_mono_ns` (int)
- `level` (DEBUG/INFO/WARN/ERROR)
- `run_id` (uuid4), `event_id` (ulid/uuid), `parent_event_id` (optional)
- `trace_id`, `span_id` (optional but expected when tracing is on)
- `proc_id`, `thread_id`, `task_id` (asyncio), `host`, `service`, `version`
- `loc`: `{module, func, file, line}`
- `kind` (ENUM): `run_start`, `run_end`, `span_start`, `span_end`, `decision`, `llm_call`, `tool_call`,
  `retrieval`, `ingest_step`, `mutation`, `cache`, `io`, `db`, `http`, `queue`, `retry`, `exception`,
  `metric`, `branch`, `invariant`, `warning`
- `msg` (short, stable)
- `details` (object; strictly bounded)
- `kpi` (object; latency/tokens/cost/resources if relevant)
- `reason` (object; REQUIRED for `decision` and `branch`)

### B) Traces (OpenTelemetry)
- Every major operation is a span: ingest stage, chunking, embed, upsert, retrieval, rerank, planner step, tool call.
- Logs MUST include `trace_id` and `span_id` for correlation. OpenTelemetry explicitly supports correlating logs with traces by including TraceId/SpanId in LogRecords. :contentReference[oaicite:0]{index=0}
- Spans must carry attributes needed for KPI attribution (model, tokens, doc counts, cache hit, etc.). :contentReference[oaicite:1]{index=1}

### C) Metrics (KPI accounting)
At minimum:
- latency: end-to-end + per stage + queue wait + retries
- tokens: prompt_tokens, completion_tokens, total_tokens per call + per agent turn
- cost estimate: by model pricing table (configurable; do not hardcode without versioning)
- resources: cpu_time, rss_mb, io_bytes, net_bytes
- throughput: items/sec, docs/sec, chunks/sec
- cache: hit/miss rates, eviction counts
- errors: type counts, retry/backoff histograms

### D) Decision Ledger (the point of “why”)
Every non-trivial choice must emit a `decision` event with:
- `decision_id` (stable within run)
- `question`: “What are we choosing?”
- `context`: minimal state summary used for decision
- `options`: list of `{id, summary, score_components, constraints_violated}`
- `selected`: `{id, justification, expected_outcome}`
- `counterfactuals`: top-2 alternatives with why-not
- `confidence`: numeric or categorical
- `guardrails`: thresholds/rules that applied
- `kpi_impact_estimate`: expected latency/token/cost delta

This is mandatory for:
- agent policy choices (tool vs think vs retrieve vs stop)
- routing (which model, which prompt template, which system role)
- retrieval choices (which index, top-k, filters)
- reranking thresholds / accept-reject
- chunking strategy choice
- caching decisions and TTL
- retry decisions and backoff selection
- any branch that can change output quality/cost materially

## Agent/LLM-Specific Requirements (must implement)
### 1) LLM Call Wrapper (universal)
Every LLM invocation must generate:
- `llm_call` event + span
- Inputs: **hashed** prompt parts + token counts + template ids + config ids (temperature, top_p, seed if supported)
- Outputs: **hashed** completion + token counts + finish_reason + tool_calls summary
- Latency breakdown: queue wait, network, provider, client-side parsing
- Determinism: record seed/config; record sampling params
- Safety: store raw prompts/responses only behind explicit flag `ALLOW_RAW_TEXT=1` and with redaction.

### 2) Agent Turn Structure
For each turn:
- `turn_start` span: includes `agent_id`, `turn_id`, `goal_id`
- `observation` summary event (inputs to the agent)
- `decision` event: action selection (including “do nothing/stop”)
- `tool_call` span(s) when tools invoked
- `reflection`/`postmortem` event (short) with: what changed, what was learned, what will be tried next
- `turn_end` with deltas: state hash before/after, memory writes, cache effects

### 3) Retrieval / RAG Transparency
For each retrieval step:
- query fingerprint (hash), embedding model id, index id
- filters applied, k, recall budget, time spent per stage
- top results: only IDs + scores + short safe metadata (no raw doc text by default)
- reranker used, threshold decision emitted as `decision`
- grounding: what evidence was used for final output (IDs + offsets) without dumping full text unless allowed

### 4) Ingest Transparency
For ingest pipelines:
- stage spans: parse → normalize → chunk → embed → dedupe → upsert
- per stage: counts, sizes, duration, error buckets
- chunking: decision record for chunk params; per-doc chunk count distribution
- dedupe: algorithm id, threshold, collisions summary
- embedding: model id, batching strategy decision, cache hits
- upsert: id mapping, retries, conflict resolution decisions

## “Turn left/right” Execution Transparency (branch evidence)
You must produce **branch/decision** evidence without drowning the system.

### Preferred: sys.monitoring (Python 3.12+; PEP 669)
- Use `sys.monitoring` selectively for **calls/returns/exceptions/lines/jumps** in target modules only.
- Record branch/jump events as compact `branch` events with:
  - location, condition fingerprint (where possible), and which edge taken
  - parent span linkage
PEP 669 defines low-impact monitoring with a range of events including calls/returns/lines/exceptions/jumps, enabling “pay only for what you use.” :contentReference[oaicite:2]{index=2}

### Fallback: sys.setprofile / sys.settrace
- Use `sys.setprofile` for call/return coverage when sys.monitoring unavailable.
- Use `sys.settrace` only in deep-exec mode for short, scoped captures.

### Rule: Scope first, then depth
- Never enable line/jump tracing for the whole world by default.
- Target only: your project modules, agent framework, ingest pipeline, tool adapters.

## Exceptions and Failure Forensics (mandatory)
- Install:
  - global `sys.excepthook`
  - `threading.excepthook`
  - `asyncio` exception handler
- Emit `exception` events with:
  - type, message hash, stack summary, causal chain
  - local context summary (bounded): relevant IDs/configs
  - retry decision record if applicable
- Include a failure taxonomy in `run.summary.json`:
  - transient vs deterministic vs data-dependent vs concurrency

## State Mutation Forensics (mandatory for “why changed?” bugs)
For critical state objects:
- Create mutation sentinels:
  - wrapper types, explicit setter hooks, or scoped monitoring hooks
- On mutation:
  - emit `mutation` event with object_id, before/after fingerprints (hash/size), and callsite stack summary
- Do not dump entire structures; emit diff summaries and sampled keys only.

## Performance Guardrails (mandatory)
- Logging must be non-blocking in hot paths (queue-based handler).
- Rate limit high-frequency events (branch/line) and aggregate where possible.
- Hard caps: max event size, max field length, max stack depth.
- Sampling knobs:
  - `INSTRUMENT=0/1`
  - `INSTRUMENT_MODE=forensic|prod|deep-exec`
  - `INSTRUMENT_SCOPE=module_prefixes`
  - `INSTRUMENT_SAMPLE_RATE`
  - `ALLOW_RAW_TEXT=0/1`

## Reproducibility Pack (mandatory)
- Record:
  - sanitized env whitelist + config snapshot + CLI args
  - random seeds (random/numpy/torch if present)
  - python version + OS + git commit
- Ensure `run_id` ties all artifacts together.

## “Perfect Skill” execution protocol (Codex must follow strictly)
1) Identify entry points and critical flows: ingest stages, agent loop, tool adapters, retrieval.
2) Implement the four pillars: logs + traces + metrics + decision ledger.
3) Add LLM/agent-specific wrappers and ensure every action selection emits a `decision`.
4) Add ingest/retrieval spans and their decision events (thresholds, batching, caching, rerank).
5) Add exception hooks and mutation sentinels for critical state.
6) Add optional deep execution monitoring (sys.monitoring) only for selected modules and only in deep-exec mode.
7) Produce artifacts and a summary that answers: top costs, top slow spans, top failure causes, top decision pivots.

## Acceptance Tests (must pass)
1) Single run produces all required artifacts and consistent correlation IDs.
2) Every agent turn contains at least one `decision` event.
3) Every LLM call logs tokens, latency, model id, and redacted prompt/response fingerprints.
4) Ingest pipeline stages each emit spans with counts/durations and at least one decision record (chunk/batch/dedupe).
5) Branch evidence exists in deep-exec mode for target modules (not global).
6) Redaction test: seeded secret strings never appear in logs.

## Notes
- OpenTelemetry traces define spans and trace context; you must use that context to tie together everything across modules and tools. :contentReference[oaicite:3]{index=3}
- OpenTelemetry logs are most useful when schema-stable and correlated with trace/span identifiers. :contentReference[oaicite:4]{index=4}
- For deep runtime events, prefer PEP 669 monitoring via sys.monitoring on Python 3.12+. :contentReference[oaicite:5]{index=5}
---
