# ops

`ops` owns Twinr's local operational support layer. It provides config audits,
device and host snapshots, singleton loop locks, file-backed ops stores,
the rolling remote-memory watchdog, bounded soak evidence for that watchdog,
bounded self-tests, detached-runtime audio-session env priming, and redacted
support exports for the web UI and operator tools.

## Responsibility

`ops` owns:
- resolve canonical paths for ops artifacts and stores
- persist sanitized ops events and usage telemetry
- collect config, device, and system-health snapshots
- run a dedicated rolling ChonkyDB remote-memory watchdog
- persist structured remote-readiness probe evidence and supervisor-seeded watchdog bootstrap snapshots so restart phases do not look like dead/stale watchdog failures
- persist structured long-term remote-read diagnostics when ChonkyDB retrieve/fetch paths fail or degrade to bounded fallback, so operators can separate backend HTTP flakes, timeouts, and client-contract issues
- ensure the dedicated remote-memory watchdog process is running for live Pi runtimes
- seed detached Pi runtime processes with the user-session audio env they need for Pulse/ALSA default playback
- supervise the productive Pi streaming loop plus remote watchdog under one authoritative owner and prime child audio-session env before each spawn
- consume the shared display heartbeat contract so ops health and the runtime supervisor read the same companion-progress semantics the display loop writes
- keep display-companion degradation visible in ops health without letting a display fault tear down the speech path
- recycle failed watchdog service instances so transient remote-state poison does not stick forever
- run bounded soak observations that prove the watchdog stays healthy over time
- infer companion-loop health from loop locks plus authoritative forward-progress heartbeats when no standalone process exists
- tolerate bounded display-render inflight windows when evaluating companion health, so long Waveshare refreshes are not misclassified as dead threads
- coordinate per-loop singleton locks
- run bounded self-tests and build support bundles
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
| [runtime_supervisor.py](./runtime_supervisor.py) | Authoritative Pi runtime supervisor for the streaming loop and remote watchdog while leaving display degradation to ops health instead of recycling the speech path |
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
python3 hardware/ops/bootstrap_self_coding_pi.py
PYTHONPATH=src python3 -m twinr --env-file .env --self-coding-codex-self-test --self-coding-live-auth-check
PYTHONPATH=src python3 -m twinr --env-file .env --long-term-memory-live-acceptance
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [web app](../web/app.py)
- [display health consumer](../display/service.py)
