# Twinr External Browser Benchmark Plan

This document defines the first external benchmark integration path for Twinr's
browser agent. The goal is to measure Twinr against an existing benchmark stack
instead of building a new internal benchmark website or a large custom harness.

## Decision

Use:

- `AgentLab` as the experiment/evaluation runner
- `BrowserGym` as the benchmark environment layer
- `WebArena-Verified` as the first external target benchmark

This is the smallest practical path to a real external score because:

- BrowserGym already exposes maintained benchmark packages and task execution
  APIs.
- AgentLab is the official experiment layer above BrowserGym and already
  supports WebArena-family benchmarks.
- WebArena-Verified is self-hosted and reproducible, which makes it a better
  first external score than noisier live-web benchmarks.

## What "No New Harness" Means Here

Twinr should not build a second benchmark website, a second scoring system, or a
bespoke task runner that duplicates AgentLab/BrowserGym.

The only Twinr-specific code we should add is a thin adapter that lets AgentLab
or BrowserGym call the current Twinr browser agent.

Allowed Twinr-owned code:

- one adapter from BrowserGym observations/actions to Twinr browser calls
- one small launch/config wrapper for repeatable local runs
- a small set of contract tests for the adapter

Not allowed in this phase:

- a Twinr-owned clone of WebArena
- a Twinr-owned benchmark scorer
- a large parallel experiment framework duplicating AgentLab
- moving benchmark websites into tracked repo paths

## Repo Safety And Gitignore Invariant

Tracked code must stay separate from local browser-workspace state.

Tracked paths:

- `agentic_tools/` for benchmark launch helpers and reproducibility wrappers
- `test/` for narrow adapter tests
- `docs/` for the integration plan and later benchmark runbooks

Ignored local workspaces:

- `/browser_automation/**`
- `/web_automation/**`

That means:

- benchmark sites, docker state, cloned benchmark repos, screenshots, traces,
  and local experiment outputs stay outside tracked source code
- the current Twinr browser workspace continues to stay ignored
- any future alternate web-automation workspace also stays ignored

## Target Architecture

### 1. Twinr benchmark adapter

Add a small tracked adapter layer under `agentic_tools/` that owns:

- reading benchmark task instructions
- invoking Twinr's current browser tool/runtime path
- translating Twinr results into the benchmark runner's expected outcome shape
- capturing run metadata for reproducibility

This adapter should not contain browser policy logic. It only bridges external
benchmark APIs to the existing Twinr browser runtime.

### 2. External benchmark workspace

Keep the actual benchmark stack outside tracked code, inside ignored workspaces.

Recommended local layout:

- `/browser_automation/benchmarks/agentlab/`
- `/browser_automation/benchmarks/browsergym/`
- `/browser_automation/benchmarks/webarena_verified/`

Optional future alternative:

- `/web_automation/benchmarks/...`

Twinr should treat these as disposable local environments, not as repo source of
truth.

### 3. First benchmark target

Start with `WebArena-Verified`.

Reason:

- self-hosted and reproducible
- broad enough to be meaningful
- less noisy than open live-web suites
- already sits on the BrowserGym/AgentLab path we want anyway

## Implementation Phases

### Phase 0: Environment spike

Objective:

- prove that the benchmark stack boots locally without touching Twinr runtime
  logic

Steps:

1. Create the ignored local benchmark workspace under
   `/browser_automation/benchmarks/`.
2. Install AgentLab, BrowserGym, Playwright, and the WebArena-Verified
   benchmark prerequisites there.
3. Bring up one official benchmark environment and run one reference agent from
   AgentLab unchanged.

Exit criteria:

- AgentLab can launch one benchmark task successfully
- BrowserGym environment resets correctly
- no tracked repo files are required for benchmark site hosting

### Phase 1: Thin Twinr adapter

Objective:

- let AgentLab run Twinr as just another agent

Steps:

1. Add a small tracked adapter module under `agentic_tools/`.
2. Feed the benchmark task instruction into Twinr's browser-entry path.
3. Map benchmark step budget, timeout budget, and final result fields into the
   Twinr request.
4. Return Twinr's completion/failure result in the format expected by the
   benchmark runner.

Exit criteria:

- one BrowserGym task can be executed through Twinr end to end
- adapter code stays thin and contains no duplicated browser policy
- adapter tests prove request/result translation

### Phase 2: WebArena-Verified smoke

Objective:

- obtain the first honest external benchmark result

Steps:

1. Run a very small WebArena-Verified subset first.
2. Capture exact package versions, benchmark version, commit hash, model id, and
   Twinr config.
3. Store only the summarized benchmark result and report artifacts in the repo's
   normal artifact stores; keep raw benchmark environment state in ignored
   workspace paths.

Exit criteria:

- one reproducible Twinr study exists against WebArena-Verified
- result can be re-run from one documented command sequence

### Phase 3: Repeatable Twinr command

Objective:

- turn the spike into one stable operator/developer workflow

Steps:

1. Add a small `agentic_tools/` launcher or wrapper script.
2. Add a doc/runbook with prerequisites, benchmark workspace location, and exact
   commands.
3. Keep all heavy benchmark assets external to tracked code.

Exit criteria:

- one developer can reproduce the benchmark run without inventing setup steps

## Verification Strategy

### Local verification

- contract test for the adapter request/result mapping
- one single-task BrowserGym smoke through Twinr
- one tiny WebArena-Verified subset run

### Pi verification

Do not start with full benchmark hosting on the Pi.

Pi acceptance is only required if the adapter changes runtime-facing Twinr code.
If that happens:

- deploy the changed Twinr code to `/twinr`
- prove that the normal Twinr browser runtime still passes the existing local Pi
  acceptance gates
- optionally run one narrow adapter smoke on the Pi if the runtime path itself
  changed

### Reproducibility metadata

For every benchmark run capture:

- Twinr git commit
- AgentLab version
- BrowserGym version
- benchmark package/version
- model id
- timeout and step-budget settings
- command line used

## Why Not Start With Live-Web Benchmarks

Not first.

Benchmarks like AssistantBench or other live-web suites are useful later, but
they introduce:

- region variance
- language variance
- silent site changes
- harder result comparison over time

That makes them a bad first external score for Twinr.

## Recommended Tracked File Layout

Keep the tracked integration minimal:

- `agentic_tools/browser_benchmarks/`
  - environment checks
  - launcher wrapper
  - reproducibility capture
- `test/test_browsergym_adapter.py`
  - adapter contract tests
- `docs/BROWSERGYM_AGENTLAB_WEB_ARENA_VERIFIED_PLAN.md`
  - this plan

Keep all benchmark environments and cloned benchmark repos ignored under the root
workspace paths.

## Non-Goals For This First Pass

- no WorkArena integration yet
- no VisualWebArena integration yet
- no live-web benchmark first
- no leaderboard optimization
- no benchmark-shaped Twinr logic

## Success Criteria

This phase is complete when:

1. Twinr can be launched by AgentLab through a thin adapter.
2. WebArena-Verified can run against Twinr without a Twinr-owned benchmark
   website or scorer.
3. All heavy browser/web automation workspaces remain ignored by git.
4. Twinr still passes its existing internal browser acceptance gates after the
   adapter lands.

## Primary Sources

- BrowserGym docs: <https://browsergym.readthedocs.io/latest/>
- BrowserGym repo: <https://github.com/ServiceNow/BrowserGym>
- AgentLab repo: <https://github.com/ServiceNow/AgentLab>
- WebArena-x overview: <https://webarena.dev/>
- WebArena paper: <https://arxiv.org/abs/2307.13854>
