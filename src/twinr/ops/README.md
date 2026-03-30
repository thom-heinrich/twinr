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
- validate the Pi-side OpenAI env contract for acceptance scripts so `/twinr/.env` can be trusted directly for provider probes without one-off shell injection
- run a dedicated rolling ChonkyDB remote-memory watchdog
- persist structured remote-readiness probe evidence and supervisor-seeded watchdog bootstrap snapshots so restart phases do not look like dead/stale watchdog failures
- run strict bootstrap/recovery remote probes once, then reuse a no-bootstrap steady-state keepalive that still proves archive-safe remote readability without reseeding every snapshot on every tick
- keep fresh watchdog heartbeats authoritative during bounded steady-state idle gaps so the supervisor does not false-fail a healthy remote watchdog between deep probes
- persist only compact recent-sample summaries in watchdog artifacts so Pi heartbeats stay cheap instead of fsyncing multi-megabyte historical probe payloads every tick
- keep watchdog attestation tiers explicit so current-only probes stay degraded and only archive-safe probes become green/ready
- persist structured long-term remote-read diagnostics when ChonkyDB retrieve/fetch paths fail or degrade to bounded fallback, including exact endpoint and request-payload type, so operators can separate backend HTTP flakes, timeouts, and client-contract issues
- ensure the dedicated remote-memory watchdog process is running for live Pi runtimes
- allow the productive runtime supervisor to consume that external watchdog as the long-lived owner, so restarting the supervisor does not cold-reset the watchdog's warm remote state
- let the productive runtime supervisor self-heal a dead externally managed watchdog owner by re-spawning the detached companion when the watchdog PID/artifact proves the owner is gone
- reseed the persisted watchdog bootstrap snapshot when the detached companion adopts or spawns a new external owner PID, so handoff windows do not keep advertising a dead previous process
- seed detached Pi runtime processes with the user-session audio env they need for Pulse/ALSA default playback, including the configured desktop runtime dir on productive root-owned Pi services where the live audio session belongs to the logged-in user
- supervise the productive Pi streaming loop and, when configured, consume the external remote watchdog artifact instead of always recycling a fresh watchdog child
- launch supervisor-owned runtime children in dedicated process groups so restart/stop paths also tear down helper descendants that still own GPIO or other runtime-critical resources
- adopt an already-running streaming-loop owner after supervisor restarts instead of thrashing new children into singleton-lock failures
- consume the shared display heartbeat contract so ops health and the runtime supervisor read the same companion-progress semantics the display loop writes
- keep display-companion degradation visible in ops health without letting a display fault tear down the speech path
- recycle failed watchdog service instances so transient remote-state poison does not stick forever
- run bounded soak observations that prove the watchdog stays healthy over time
- infer companion-loop health from loop locks plus authoritative forward-progress heartbeats when no standalone process exists
- tolerate bounded display-render inflight windows when evaluating companion health, so long Waveshare refreshes are not misclassified as dead threads
- coordinate per-loop singleton locks
- run bounded self-tests and build support bundles
- mirror the authoritative leading repo into `/twinr` while preserving Pi-local runtime-only paths such as `.env`, `.venv`, `state/`, `artifacts/`, and `.cache/`, healing acceptance drift, using exact-content checks by default so false-clean metadata matches do not slip through, and ignoring transient local devices/FIFOs/special files that do not belong in the Pi checkout
- deploy the authoritative leading repo plus runtime `.env` onto the Pi acceptance host, refresh the editable install, heal direct-dependency duplicates where a stale venv copy shadows a bridged Pi system package such as `PyQt5`, install optional mirrored browser-automation runtime manifests when present, replace the old manual mirror-as-deploy workflow, independently attest the mirrored repo contents by SHA256/link-target before restart, restart the productive Pi service set, and verify post-restart health
- repair known shared `/twinr/state` ownership and mode contracts during Pi deploys so cross-user runtime files such as `automations.json`, `automations.json.lock`, and `user_discovery.json` stay usable after rollout
- verify those shared `/twinr/state` ownership and mode contracts again after the productive services restarted, so post-restart acceptance fails closed if a service recreated one of the shared files with the wrong owner or mode
- hand root-owned mirrored `__pycache__/` trees back to the runtime user, rebuild checked-hash Python bytecode for mirrored source trees during Pi deploys, and attest critical runtime APIs such as the local camera stream surface plus memory bootstrap imports before services restart
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
| [openai_env_contract.py](./openai_env_contract.py) | Fail-closed validation for the Pi-side OpenAI `.env` contract used by acceptance probes |
| [health.py](./health.py) | Host and service health, including display-companion assessment via the shared display heartbeat contract, supervisor-aware degradation when the streaming child is being restarted, argv-exact service detection that ignores shell debug script text, and memory-pressure classification that prefers `MemAvailable` headroom over raw used-percent alone |
| [remote_memory_watchdog.py](./remote_memory_watchdog.py) | Continuous fail-closed ChonkyDB readiness watchdog plus structured probe/bootstrap artifacts |
| [remote_memory_watchdog_state.py](./remote_memory_watchdog_state.py) | Internal sample/snapshot/store helpers for persisted watchdog state and bootstrap artifacts |
| [remote_memory_watchdog_companion.py](./remote_memory_watchdog_companion.py) | Start or adopt the external watchdog process for live Pi loops and reseed bootstrap attestation when the owner PID changes |
| [runtime_env.py](./runtime_env.py) | Seed detached Pi runtimes with the minimal user audio-session environment, preferring the configured desktop runtime dir for productive root-owned Pi services when needed |
| [runtime_scope.py](./runtime_scope.py) | Build scoped runtime configs so auxiliary loops do not overwrite the primary display/runtime snapshot |
| [runtime_supervisor.py](./runtime_supervisor.py) | Authoritative Pi runtime supervisor for the streaming loop that can either own or consume the dedicated remote watchdog, self-heal a dead external watchdog owner, and leave display degradation to ops health instead of recycling the speech path |
| [runtime_supervisor_process.py](./runtime_supervisor_process.py) | Internal child-process, timestamp, and runtime-env helpers kept separate from supervisor orchestration, including dedicated child process-group handling for restart-safe teardown |
| [pi_repo_mirror.py](./pi_repo_mirror.py) | One-way repo mirror watchdog that keeps `/twinr` aligned with the authoritative leading repo without deleting Pi-local runtime state |
| [pi_runtime_deploy.py](./pi_runtime_deploy.py) | Operator-facing Pi deploy orchestration: mirror code, sync the authoritative runtime `.env`, independently attest the mirrored repo contents before restart, refresh the editable install, heal stale venv duplicates of bridged Pi system packages before dependency verification, verify critical direct-import runtime modules including the memory bootstrap path, repair and then post-restart verify shared `/twinr/state` permissions, restart the base services plus any repo-backed Pi runtime units already enabled on the host, support first rollout of disabled optional Pi units, and verify restart health |
| [pi_runtime_deploy_remote.py](./pi_runtime_deploy_remote.py) | Internal SSH/SCP/service-state helper layer kept separate from deploy phase orchestration, including remote repo-content attestation, bridged-system duplicate cleanup, and shared-state permission repair/verification helpers |
| [venv_bridged_system_cleanup.py](./venv_bridged_system_cleanup.py) | Detect direct dependencies where a stale venv dist shadows an acceptable bridged Pi system dist |
| [self_coding_pi.py](./self_coding_pi.py) | Pi bootstrap for pinned self-coding Codex bridge, CLI, auth sync, and remote self-test |
| [remote_memory_watchdog_soak.py](./remote_memory_watchdog_soak.py) | Bounded soak recorder for watchdog stability proof |
| [devices.py](./devices.py) | Device overview probes |
| [self_test.py](./self_test.py) | Bounded hardware self-tests, including AI-Deck WiFi handover, frame capture, image-sanity checks, and proactive-mic ownership guards that only block while an active Twinr voice runtime already owns the same capture device |
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
from twinr.ops import check_openai_env_contract

status = check_openai_env_contract("/twinr/.env")
assert status.ok, status.detail
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
python3 hardware/ops/deploy_pi_runtime.py
python3 hardware/ops/deploy_pi_runtime.py --rollout-service twinr-whatsapp-channel
python3 hardware/ops/check_pi_openai_env_contract.py --env-file /twinr/.env
python3 hardware/ops/bootstrap_self_coding_pi.py
PYTHONPATH=src python3 -m twinr --env-file .env --self-coding-codex-self-test --self-coding-live-auth-check
PYTHONPATH=src python3 -m twinr --env-file .env --long-term-memory-live-acceptance
```

The repo mirror uses `rsync --checksum` on every cycle by default. Only opt
into `--metadata-only` when you explicitly accept the weaker quick-check plus
periodic checksum-audit model.
After a healing sync, the watchdog also retries one extra sync+verify pass if
the checksum audit still sees source-managed drift, which absorbs brief
shared-worktree churn without hiding persistent Pi-side divergence.
Its Pi-local preserve rules are perishable, which keeps root runtime state
protected while still allowing accidental nested repo copies to be deleted.
That preserved scope intentionally includes `/twinr/.cache/`, because runtime
helpers such as the startup boot-sound renderer can create root-owned cache
artifacts there and those must not block later repo deploys.
Python bytecode caches are intentionally excluded from the mirror because they
are runtime artefacts and productive root-owned imports on the Pi can otherwise
Installer-generated `*.egg-info/` trees are excluded for the same reason: the
Pi-side editable install rewrites them locally and they are not authoritative
repo content.
block later sync cycles.
Nested `node_modules/` trees are intentionally excluded from the mirror for the
same reason: SDK and channel dependency installs are local build artefacts, not
authoritative source content for `/twinr`.
Captured `browser_automation/artifacts/` output is intentionally excluded from
the mirror as well, so gitignored screenshots/HTML traces on either host do not
block official Pi deploy convergence while the small mirrored browser-runtime
manifests remain available for install.
The generated Crazyflie STM32 failsafe build tree under
`hardware/bitcraze/twinr_on_device_failsafe/build/` is intentionally excluded
from the mirror as well because it is a large local firmware build workspace,
not authoritative Twinr runtime source, and it can mutate mid-transfer.
The Pi deploy also attests `/twinr/.venv/bin` after the editable refresh,
normalizes stale copied Python-wrapper shebangs plus activation-script
`VIRTUAL_ENV` roots back to `/twinr/.venv`, hands any existing root-owned mirrored
`__pycache__/` trees back to the runtime user, rebuilds mirrored source
bytecode with `compileall --invalidation-mode checked-hash`, and verifies a
critical runtime API contract for camera-stream entrypoints and the
memory/bootstrap import path before productive services restart. That keeps
preserved Pi bytecode caches from shadowing fresh
source files after mirror syncs or historical checkout moves.
When a deleted remote tree survives only because it still contains ignored
Python caches, the mirror prunes those cache-only stale directories and reruns
the sync once so the acceptance checkout can still converge cleanly.
The deploy helper builds on that mirror contract and adds the explicit pieces
the mirror intentionally does not own: authoritative `.env` sync, a no-deps
editable refresh by default, early activation of the bridged Pi
`dist-packages` view inside the preserved venv, targeted removal of stale venv
copies when the bridged system distribution already satisfies the direct repo
requirement, selective backfill of mirrored project runtime dependencies that
are still missing or out of spec on the Pi, optional
Pi-only runtime supplement installs via `hardware/ops/pi_runtime_requirements.txt`,
which mirrors `project.optional-dependencies.pi-runtime` in `pyproject.toml`,
optional mirrored browser-automation runtime installs via
`browser_automation/runtime_requirements.txt` plus
`browser_automation/playwright_browsers.txt`, with those ignored workspace
manifests synced explicitly before the remote install step, productive unit
installation/restart, a fixed `/twinr/.venv/bin/python` import contract for
critical direct-import modules, and bounded post-restart verification. By default it
always manages the base runtime units and also picks up any repo-backed Pi unit
that is already enabled on the acceptance host, which keeps optional services
such as WhatsApp inside the same restart proof once they are live there. The
same default path also repairs an optional Pi unit whose install symlink still
exists but whose installed unit file became masked or otherwise corrupted on the
host. Before any install or restart, the deploy now also uploads an
authoritative local manifest and independently attests every mirrored regular
file and symlink on `/twinr` by SHA256 or link target, so the deploy cannot
report a green rollout solely because the rsync phase believed it converged.
It also repairs the known cross-user state paths under `/twinr/state/` before
services restart, so deploys do not leave stale permissions on shared files
such as `automations.json`, `automations.json.lock`, or `user_discovery.json`.
Opt into a dependency-refreshing install only when you deliberately want
the Pi to re-resolve the full runtime package graph. For first rollout of an optional Pi unit
that is present in the repo but not enabled on the host yet, add
`--rollout-service <unit>` so the deploy includes, enables, restarts, and
verifies that unit without replacing the default base service set.

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [web app](../web/app.py)
- [display health consumer](../display/service.py)
