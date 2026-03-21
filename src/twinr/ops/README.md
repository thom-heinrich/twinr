# ops

`ops` owns Twinr's local operational support layer. It provides config audits,
device and host snapshots, singleton loop locks, file-backed ops stores,
the rolling remote-memory watchdog, bounded soak evidence for that watchdog,
bounded self-tests, detached-runtime audio-session env priming, leading-repo to
Pi repo mirroring, and redacted support exports for the web UI and operator
tools.

## Responsibility

`ops` owns:
- resolve canonical paths for ops artifacts and stores
- persist sanitized ops events and usage telemetry
- collect config, device, and system-health snapshots
- run a dedicated rolling ChonkyDB remote-memory watchdog
- persist structured remote-readiness probe evidence and supervisor-seeded watchdog bootstrap snapshots so restart phases do not look like dead/stale watchdog failures
- run strict bootstrap/recovery remote probes once, then reuse a cheaper steady-state keepalive that proves current remote readability without reseeding every snapshot on every tick
- keep fresh watchdog heartbeats authoritative during bounded steady-state idle gaps so the supervisor does not false-fail a healthy remote watchdog between deep probes
- persist only compact recent-sample summaries in watchdog artifacts so Pi heartbeats stay cheap instead of fsyncing multi-megabyte historical probe payloads every tick
- keep the heavy `archive` snapshot out of the steady-state watchdog hot path while retaining archive-inclusive bootstrap and recovery proofs
- persist structured long-term remote-read diagnostics when ChonkyDB retrieve/fetch paths fail or degrade to bounded fallback, so operators can separate backend HTTP flakes, timeouts, and client-contract issues
- ensure the dedicated remote-memory watchdog process is running for live Pi runtimes
- allow the productive runtime supervisor to consume that external watchdog as the long-lived owner, so restarting the supervisor does not cold-reset the watchdog's warm remote state
- seed detached Pi runtime processes with the user-session audio env they need for Pulse/ALSA default playback
- supervise the productive Pi streaming loop and, when configured, consume the external remote watchdog artifact instead of always recycling a fresh watchdog child
- adopt an already-running streaming-loop owner after supervisor restarts instead of thrashing new children into singleton-lock failures
- consume the shared display heartbeat contract so ops health and the runtime supervisor read the same companion-progress semantics the display loop writes
- keep display-companion degradation visible in ops health without letting a display fault tear down the speech path
- recycle failed watchdog service instances so transient remote-state poison does not stick forever
- run bounded soak observations that prove the watchdog stays healthy over time
- infer companion-loop health from loop locks plus authoritative forward-progress heartbeats when no standalone process exists
- tolerate bounded display-render inflight windows when evaluating companion health, so long Waveshare refreshes are not misclassified as dead threads
- coordinate per-loop singleton locks
- run bounded self-tests and build support bundles
- mirror the authoritative leading repo into `/twinr` while preserving Pi-local runtime-only paths, healing acceptance drift, and using exact-content checks by default so false-clean metadata matches do not slip through
- bootstrap the Pi-side self-coding Codex runtime prerequisites from the leading repo
- expose config checks that fail clearly when the self-coding Codex bridge, CLI, or auth is not ready

`ops` does **not** own:
- runtime orchestration in `src/twinr/agent/workflows`
- hardware adapter implementations in `src/twinr/hardware`
- web routing and template rendering in `src/twinr/web`
- provider request execution beyond normalized usage extraction

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Public ops exports |
| [paths.py](./paths.py) | Canonical ops paths |
| [locks.py](./locks.py) | Loop singleton locks |
| [events.py](./events.py) | Ops event JSONL store |
| [usage.py](./usage.py) | Usage telemetry store |
| [checks.py](./checks.py) | Config audit checks |
| [health.py](./health.py) | Host and service health, including display-companion assessment via the shared display heartbeat contract |
| [remote_memory_watchdog.py](./remote_memory_watchdog.py) | Continuous fail-closed ChonkyDB readiness watchdog plus structured probe/bootstrap artifacts |
| [remote_memory_watchdog_companion.py](./remote_memory_watchdog_companion.py) | Start the external watchdog process for live Pi loops when needed |
| [runtime_env.py](./runtime_env.py) | Seed detached Pi runtimes with the minimal user audio-session environment |
| [runtime_scope.py](./runtime_scope.py) | Build scoped runtime configs so auxiliary loops do not overwrite the primary display/runtime snapshot |
| [runtime_supervisor.py](./runtime_supervisor.py) | Authoritative Pi runtime supervisor for the streaming loop that can either own or consume the dedicated remote watchdog while leaving display degradation to ops health instead of recycling the speech path |
| [pi_repo_mirror.py](./pi_repo_mirror.py) | One-way repo mirror watchdog that keeps `/twinr` aligned with the authoritative leading repo without deleting Pi-local runtime state |
| [self_coding_pi.py](./self_coding_pi.py) | Pi bootstrap for pinned self-coding Codex bridge, CLI, auth sync, and remote self-test |
| [remote_memory_watchdog_soak.py](./remote_memory_watchdog_soak.py) | Bounded soak recorder for watchdog stability proof |
| [devices.py](./devices.py) | Device overview probes |
| [self_test.py](./self_test.py) | Bounded hardware self-tests |
| [support.py](./support.py) | Support bundle export |
| [component.yaml](./component.yaml) | Structured package metadata |

The device overview and config checks now also surface ReSpeaker XVF3800
runtime and host-control state separately from generic audio-device listings,
so operators can see the difference between `USB-visible`, `capture-ready`,
host-control degradation, and `not detected`. The ReSpeaker device surface also
includes conservative direction-confidence, overlap, and barge-in facts when
the direct XVF3800 primitives support them.

## Usage

```python
from twinr.ops import TwinrOpsEventStore, resolve_ops_paths_for_config

ops_paths = resolve_ops_paths_for_config(config)
TwinrOpsEventStore.from_config(config).append(
    event="assistant_started",
    message="Twinr runtime booted.",
    data={"events_path": str(ops_paths.events_path)},
)
```

```python
from twinr.ops import TwinrSelfTestRunner, collect_system_health

health = collect_system_health(config)
result = TwinrSelfTestRunner(config).run("printer")
```

```python
from twinr.ops import RemoteMemoryWatchdog

watchdog = RemoteMemoryWatchdog.from_config(config)
watchdog.run(duration_s=5.0)
```

```python
from twinr.ops.runtime_supervisor import TwinrRuntimeSupervisor

supervisor = TwinrRuntimeSupervisor(config=config, env_file="/twinr/.env")
supervisor.run(duration_s=5.0)
```

```bash
PYTHONPATH=src python3 -m twinr.ops.remote_memory_watchdog_soak --project-root /twinr --duration-s 14400 --interval-s 30
```

```bash
python3 hardware/ops/watch_pi_repo_mirror.py --once
python3 hardware/ops/watch_pi_repo_mirror.py --interval-s 5
python3 hardware/ops/watch_pi_repo_mirror.py --interval-s 5 --metadata-only
python3 hardware/ops/bootstrap_self_coding_pi.py
PYTHONPATH=src python3 -m twinr --env-file .env --self-coding-codex-self-test --self-coding-live-auth-check
PYTHONPATH=src python3 -m twinr --env-file .env --long-term-memory-live-acceptance
```

The repo mirror uses `rsync --checksum` on every cycle by default. Only opt
into `--metadata-only` when you explicitly accept the weaker quick-check plus
periodic checksum-audit model.
Its Pi-local preserve rules are perishable, which keeps root runtime state
protected while still allowing accidental nested repo copies to be deleted.

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [web app](../web/app.py)
- [display health consumer](../display/service.py)
