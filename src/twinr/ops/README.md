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
- keep the sanitized ops-event JSONL store readable across Pi runtime/operator users so deploy and acceptance diagnostics do not go blind when services and probes run under different accounts
- keep shared ops JSON artifacts such as `remote_memory_watchdog.json`, `display_ambient_impulse.json`, `display_heartbeat.json`, and `streaming_memory_segments.json` readable across Pi runtime/operator users while still confining them inside the `0700` ops directory, so fail-closed runtime probes can consume fresh watchdog/display state without needing `sudo`
- keep `display_render_state.json` under that same shared readable contract and
  treat it as the only authoritative source for claims about what the visible
  panel currently shows; heartbeat remains liveness only
- collect config, device, and system-health snapshots, including bounded memory-pressure hysteresis so near-threshold Pi reclaim jitter does not flap operator-facing `warn`/`error` state and swap-saturated Pi hosts stay degraded only when Twinr's live runtime actually owns swapped-out pages instead of lingering in misleading `warn` from cold non-Twinr desktop swap alone
- validate the Pi-side OpenAI env contract for acceptance scripts so `/twinr/.env` can be trusted directly for provider probes without one-off shell injection
- run a dedicated rolling ChonkyDB remote-memory watchdog
- persist structured remote-readiness probe evidence and supervisor-seeded watchdog bootstrap snapshots so restart phases do not look like dead/stale watchdog failures
- run strict bootstrap/recovery remote probes once, then reuse a no-bootstrap steady-state keepalive that proves the configured required-remote contract without reseeding every snapshot on every tick; live Pi `watchdog_artifact` mode defaults that contract to the backend's current-only/`token_fast` serving lane while explicit archive-inclusive proof remains available when needed
- keep fresh watchdog heartbeats authoritative during bounded steady-state idle gaps so the supervisor does not false-fail a healthy remote watchdog between deep probes
- persist only compact recent-sample summaries in watchdog artifacts so Pi heartbeats stay cheap instead of fsyncing multi-megabyte historical probe payloads every tick
- keep watchdog attestation tiers and proof contracts explicit so persisted samples can distinguish current-only versus archive-inclusive readiness and only turn green when they satisfy the configured required-remote contract
- persist watchdog `pid_starttime_ticks` as the primary identity attestation and keep epoch `pid_create_time_s` as legacy compatibility evidence only, because live Pi wall-clock corrections can shift `/proc/stat btime` without meaning the watchdog PID changed
- keep the watchdog proof contract explicit inside the persisted probe payload so `ready=true` is documented as the configured-namespace warm-read contract, not as a blanket proof for isolated-namespace write/retention/fresh-reader transactions
- recover the structured per-probe retention-canary report when the remote canary outlives the SSH stdout timeout, so deploy output still surfaces `failure_stage` plus the watchdog/canary relation instead of collapsing the mismatch into a generic timeout
- persist structured long-term remote-read diagnostics when ChonkyDB retrieve/fetch paths fail or degrade to bounded fallback, including exact endpoint and request-payload type, so operators can separate backend HTTP flakes, timeouts, and client-contract issues
- diagnose the public Twinr-facing ChonkyDB URL against the dedicated backend service, its loopback `127.0.0.1:3044` surface, and any foreign non-Twinr consumers still pointed at that dedicated backend so P0 incidents can be repaired without blind backend restarts
- fail closed when the remote backend's own status surface still says `ready=false` unless the stronger deep probe already satisfies the configured required-remote proof contract, so the watchdog only advertises green readiness when the query/read surface it actually depends on is proven
- preserve explicit transient backend details such as `Service warmup in progress`, `Upstream unavailable or restarting`, and backend `ready=false` / `instance responded but is not ready` surfaces all the way into watchdog samples, so restart transit stays on the short reprobe lane instead of decaying into a stale generic `ChonkyDBError` block; local remote-state cooldown samples also stay on the base 1s cadence so the supervisor sees fresh fail-closed evidence instead of a stale artifact while recovery is still in progress
- let the authoritative watchdog run one explicit external-attestation reprobe through a local remote-state cooldown gate, so the Pi can recover automatically once the backend is healthy again instead of waiting forever on a process-local circuit breaker that only an external proof can clear
- stabilize the dedicated ChonkyDB host itself when shared-host systemd system units, user-session units, or non-Twinr workers directly wired to `127.0.0.1:3044` reclaim CPU/I/O from Twinr's required backend
- force-repair unreadable prompt-memory and managed-context `catalog/current` heads on an explicit remote namespace when a broken blank head times out before the normal probe-first repair path can publish the canonical empty head
- ensure the dedicated remote-memory watchdog process is running for live Pi runtimes
- allow the productive runtime supervisor to consume that external watchdog as the long-lived owner, so restarting the supervisor does not cold-reset the watchdog's warm remote state
- let the productive runtime supervisor self-heal a dead externally managed watchdog owner by requesting the dedicated systemd watchdog unit instead of spawning a second raw watchdog process behind systemd's back
- keep the Pi watchdog on one authoritative owner lane: when a dedicated systemd watchdog unit is configured, the companion may request that unit but must never raw-spawn a second detached watchdog outside systemd
- reseed the persisted watchdog bootstrap snapshot when the companion adopts or starts a new authoritative owner PID, so handoff windows do not keep advertising a dead previous process
- keep the productive runtime supervisor fail-closed on startup so the streaming loop never starts before the required-remote watchdog reports ready; the watchdog startup grace is only for watchdog-stall recovery, not for bypassing required-remote gating
- seed detached Pi runtime processes with the user-session audio env they need for Pulse/ALSA default playback, including the configured desktop runtime dir on productive root-owned Pi services where the live audio session belongs to the logged-in user
- supervise the productive Pi display loop and streaming loop and, when configured, consume the external remote watchdog artifact instead of always recycling a fresh watchdog child
- launch supervisor-owned runtime children in dedicated process groups so restart/stop paths also tear down helper descendants that still own GPIO or other runtime-critical resources
- keep the authoritative visible Twinr screen alive even while required-remote gating legitimately blocks the speech runtime, so backend outages show a real Twinr error/blocked surface instead of falling back to the desktop
- adopt an already-running streaming-loop owner after supervisor restarts instead of thrashing new children into singleton-lock failures
- consume the shared display heartbeat contract so ops health and the runtime supervisor read the same companion-progress semantics the display loop writes
- keep display-companion degradation visible in ops health without letting a display fault tear down the speech path
- persist process-local streaming-loop memory attribution so memory-pressure warnings can name the concrete heavy owner path, current RSS/anonymous footprint, and largest startup/render delta instead of only surfacing host-level symptoms
- throttle subsystem-specific streaming-memory checkpoints from hot workers such as voice orchestrator, proactive monitor, and realtime housekeeping so Pi leak hunts can distinguish a real growing lane from whichever loop merely happened to render last
- recycle failed watchdog service instances so transient remote-state poison does not stick forever
- run bounded soak observations that prove the watchdog stays healthy over time
- infer companion-loop health from loop locks plus authoritative forward-progress heartbeats when no standalone process exists
- tolerate bounded display-render inflight windows when evaluating companion health, so long Waveshare refreshes are not misclassified as dead threads
- coordinate per-loop singleton locks
- run bounded self-tests and build support bundles
- mirror the authoritative leading repo into `/twinr` while preserving Pi-local runtime-only paths such as `.env`, `.venv`, `state/`, `artifacts/`, and `.cache/`, healing acceptance drift, using exact-content checks by default so false-clean metadata matches do not slip through, ignoring transient local devices/FIFOs/special files that do not belong in the Pi checkout, and mirroring only a deterministic tracked-file snapshot so ignored or untracked workspace debris never becomes Pi runtime code
- deploy the authoritative leading repo plus runtime `.env` onto the Pi acceptance host, refresh the editable install, heal direct-dependency duplicates where a stale venv copy shadows a bridged Pi system package such as `PyQt5`, install browser-automation runtime support only when its allowlisted manifests are already part of the same authoritative release snapshot, replace the old manual mirror-as-deploy workflow, independently attest the mirrored repo contents by SHA256/link-target before restart, persist the last successful tracked-file release manifest under `artifacts/stores/ops/current_release_manifest.json`, restart the productive Pi service set, run the bounded live retention canary, diagnose dedicated-backend host contention on canary failure, apply one bounded remote-host stabilization pass, re-diagnose after failed stabilization so overload that worsens during the stabilization window is not missed, escalate once into the guarded dedicated-backend repair flow when the backend itself stays unhealthy afterwards, re-diagnose again after a successful repair so reactivated conflict units or unhealthy query surfaces cannot slip into the retry window, apply one extra bounded post-repair host stabilization when that quiet-host hold already broke again, wait for a fresh Pi watchdog-ready sample after the repair path, then retry the canary once, and verify post-restart health
- produce one compact read-only Pi release audit that joins the current local authoritative release, the persisted deployed `current_release_manifest.json`, and a live checksum drift probe into a single operator-facing status view
- snapshot the authoritative repo mirror scope before Pi deploy sync so shared-worktree edits cannot self-abort a rollout or mix multiple source states into one acceptance checkout
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
| [process_memory.py](./process_memory.py) | Procfs-backed streaming-loop memory attribution snapshot keyed to the live owner PID/start-time so ops health can report the concrete heavy subsystem behind memory pressure |
| [openai_env_contract.py](./openai_env_contract.py) | Fail-closed validation for the Pi-side OpenAI `.env` contract used by acceptance probes |
| [health.py](./health.py) | Host and service health, including display-companion assessment via the shared display heartbeat contract, supervisor-aware degradation when the streaming child is being restarted, argv-exact service detection that ignores shell debug script text, procfs-backed memory-owner attribution for the live streaming PID, and memory-pressure classification that prefers `MemAvailable` headroom over raw used-percent alone while only treating saturated swap as degraded when Twinr's live streaming process itself is swapped out |
| [remote_memory_watchdog.py](./remote_memory_watchdog.py) | Continuous fail-closed ChonkyDB readiness watchdog plus structured probe/bootstrap artifacts |
| [remote_chonkydb_repair.py](./remote_chonkydb_repair.py) | Operator-facing diagnosis and bounded repair planner for the dedicated remote ChonkyDB backend, including detection of active foreign services still pointed at `127.0.0.1:3044`, startup-contract repair for unsafe token-fast fulltext gating plus a disabled payload-read startup lane, and automatic widening of an undersized vector warmup timeout when full-scope readiness is configured to wait longer |
| [remote_chonkydb_host_stabilizer.py](./remote_chonkydb_host_stabilizer.py) | Operator-facing host-contention stabilizer that quiesces the proven non-Twinr shared-host conflict set plus user-session units, installs a reboot-persistent boot-pacing policy so the same CAIA services/timers/paths come back in controlled waves instead of a single boot burst, keeps `caia-external-site.service`, `caia-consumer-portal-demo.service`, `caia-ollama-gpu-proxy.service`, and user-session `caia-molt.service` out of that boot release lane, still suppresses external CAIA live-restart guard service/timer/path lanes plus `codex-portal-live-override.service` when they were proven to re-enable `caia-consumer-portal.service` and `caia-ccodex-memory-api.service` mid-pass, blocks the remote `caia-twinr-host-control-guard.service` while the quiet-host lane is active so it cannot immediately reopen the very services Twinr just suppressed, stages runtime `/run/systemd/system/<unit>.d/91-twinr-private-host-stabilizer-block.conf` guards for the live quiet-host window, adds persistent `/etc` boot-pacer drop-ins plus a dedicated `caia-twinr-host-boot-pacer.service` for reboot recovery, explicitly cleans up the legacy shared `90-twinr-host-stabilizer-block.conf` guard filename, inserts a short per-unit quiesce pause plus a longer post-kill cooldown so live reclaim happens incrementally instead of one burst, bounded-kills proven stale user-session code-graph benchmark runners plus long-running `chonkycode.cli artifact-ingest` and `ccodex_memory_locomo_mc10_eval.py` workloads, unmanaged loopback `chonkydb.api.server` listeners outside the allowed systemd cgroups, plus direct writers against the dedicated `twinr_dedicated_<port>/data` ChonkyDB path that bypass systemd unit control, raises backend CPU/IO priority, still surfaces the remote host-control guard `required_units` for operator forensics, and validates the public empty-scope-safe current-scope query surface instead of a weaker `/instance` liveness-only check |
| [remote_prompt_current_head_repair.py](./remote_prompt_current_head_repair.py) | Operator-facing forced empty-head publisher for prompt-memory, user-context, and personality-context `catalog/current` repair on one explicit remote namespace |
| [remote_memory_watchdog_state.py](./remote_memory_watchdog_state.py) | Internal sample/snapshot/store helpers for persisted watchdog state and bootstrap artifacts |
| [remote_memory_watchdog_companion.py](./remote_memory_watchdog_companion.py) | Start or adopt the external watchdog owner for live Pi loops, preferring the dedicated systemd unit on `/twinr`, forbidding raw detached watchdog spawns when that unit is configured, and reseeding bootstrap attestation when the owner PID changes |
| [runtime_env.py](./runtime_env.py) | Seed detached Pi runtimes with the minimal user audio-session environment, preferring the configured desktop runtime dir for productive root-owned Pi services when needed |
| [runtime_scope.py](./runtime_scope.py) | Build scoped runtime configs so auxiliary loops do not overwrite the primary display/runtime snapshot |
| [runtime_supervisor.py](./runtime_supervisor.py) | Authoritative Pi runtime supervisor for the display loop and streaming loop that can either own or consume the dedicated remote watchdog, self-heal a dead external watchdog owner, keep the visible Twinr screen alive across required-remote gate blocks, and avoid recycling the speech path for display faults |
| [runtime_supervisor_process.py](./runtime_supervisor_process.py) | Internal child-process, timestamp, and runtime-env helpers kept separate from supervisor orchestration, including dedicated child process-group handling for restart-safe teardown |
| [pi_repo_mirror.py](./pi_repo_mirror.py) | One-way repo mirror watchdog that keeps `/twinr` aligned with the authoritative leading repo without deleting Pi-local runtime state, but now mirrors from a deterministic tracked-file snapshot instead of the raw workspace |
| [pi_runtime_deploy.py](./pi_runtime_deploy.py) | Operator-facing Pi deploy orchestration: build a deterministic tracked-file snapshot and release manifest, mirror that stable code image, sync the authoritative runtime `.env`, independently attest the mirrored repo contents before restart, refresh the editable install, heal stale venv duplicates of bridged Pi system packages before dependency verification, install browser-automation runtime support only when its allowlisted manifests are already inside the authoritative release snapshot, verify critical direct-import runtime modules including the memory bootstrap path, repair and then post-restart verify shared `/twinr/state` permissions, persist the last successful release manifest under `artifacts/stores/ops/current_release_manifest.json`, restart the base services plus any repo-backed Pi runtime units already enabled on the host, run the bounded live retention canary, diagnose dedicated-backend host contention on canary failure, apply one bounded host-stabilization pass, re-diagnose when failed stabilization still leaves the public surface unhealthy, escalate once into the guarded backend-repair flow when the service remains unhealthy afterwards, re-diagnose the host again after a successful repair, restabilize once more when conflict units or unhealthy query surfaces already returned, wait for a fresh post-repair watchdog-ready sample before the retry, keep the recovery SSH budget aligned with the stabilizer/repair path instead of truncating it, support first rollout of disabled optional Pi units, and verify restart health |
| [pi_release_audit.py](./pi_release_audit.py) | Compact read-only operator audit that joins the current local release summary, the Pi's persisted `current_release_manifest.json`, and a live checksum drift probe into one status report |
| [pi_runtime_deploy_remote.py](./pi_runtime_deploy_remote.py) | Internal SSH/SCP/service-state helper layer kept separate from deploy phase orchestration, including remote repo-content attestation, bridged-system duplicate cleanup, shared-state permission repair/verification helpers, and the remote retention-canary probe |
| [retention_canary_host_recovery.py](./retention_canary_host_recovery.py) | Diagnose dedicated ChonkyDB host contention for failed retention canaries, run one bounded host-stabilization recovery pass first, re-diagnose after failed stabilization, escalate once into guarded backend repair when the service still remains unhealthy afterwards, and fail closed when the post-repair host already lost its quiet-hold again until one extra bounded restabilization proves that conflicts stayed away |
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

The live current ops event stream (`events.jsonl` plus its lock file) is kept
on a shared-writer permission contract so Pi services and operator-triggered
runtime probes append to the same sanitized diagnostics stream without
cross-user `PermissionError` drift. Shared read-mostly ops artifacts such as
`remote_memory_watchdog.json`, `current_release_manifest.json`,
`display_ambient_impulse.json`, `display_heartbeat.json`, and
`display_render_state.json`, and `streaming_memory_segments.json` likewise stay
operator-readable inside the otherwise `0700` ops directory so non-root
acceptance probes can fail closed on the same fresh artifacts the productive
root-owned services write. `current_release_manifest.json` is the last
successful tracked-file Pi deploy manifest, so operators can see exactly which
release payload the acceptance host is supposed to be running.
The private usage SQLite sidecar (`usage.jsonl.sqlite3` plus its lock) stays on
an explicit owner-only deploy-repair contract as well, so root-owned service
restarts cannot strand later `thh` acceptance probes behind stale lock-file
ownership drift.

Health payloads now expose separate visible-display fields
(`display_visible_state`, `display_visible_operator_status`,
`display_visible_state_verdict`, `display_visible_state_reason`,
`display_visible_state_source`, `display_visible_rendered_at`) so callers do
not infer panel truth from `display_heartbeat.json`. The verdict is only
`proved` when the rendered-state artifact itself supports the claim; drift
between heartbeat render evidence and that artifact is surfaced explicitly via
`display_visible_state_verdict="drift"`.

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
python3 hardware/ops/repair_remote_chonkydb.py --no-restart
python3 hardware/ops/stabilize_remote_chonkydb_host.py
python3 hardware/ops/repair_remote_prompt_current_heads.py --namespace twinr_longterm_v1:twinr:a7f1ed265838 --base-url http://127.0.0.1:43044 --force
python3 hardware/ops/bootstrap_self_coding_pi.py
PYTHONPATH=src python3 -m twinr --env-file .env --self-coding-codex-self-test --self-coding-live-auth-check
PYTHONPATH=src python3 -m twinr --env-file .env --long-term-memory-live-acceptance
```

The repo mirror uses `rsync --checksum` on every cycle by default. Only opt
into `--metadata-only` when you explicitly accept the weaker quick-check plus
periodic checksum-audit model. The mirror now snapshots git-tracked files
first, so ignored or untracked workspace clutter no longer enters the Pi
authority contract.
When the live remote-memory endpoint is unhealthy on the development host, use
`hardware/ops/repair_remote_chonkydb.py` before restarting anything by hand.
That command proves whether the public `https://tessairact.com:2149` URL, the
dedicated backend service, or the backend loopback `127.0.0.1:3044` is the
actual failing layer, surfaces active foreign consumers still pointed at the
dedicated backend, repairs dedicated data-dir owner/group drift against the
service account before any guarded restart so root-owned WAL/index files do not
survive as latent startup poison, now also detects the proven Twinr outage
shape where the dedicated startup contract drifted away from the sanctioned
payload-read lane or where the `token_fast` serving contract was incorrectly
re-blocked on fulltext warmup, rewrites those dedicated startup drop-ins back
to the canonical Twinr contract before restart, now also disables the dedicated
`payloads_sync_bulk` API-ready gate
when the service is still pinned to full-scope startup readiness even though
Twinr only needs the current-scope query/write surface, hardens the remote
systemd env when `FT_REBUILD_ON_OPEN` was left active without the matching
query-surface warmup gate, and only restarts the backend when the backend
itself is the proven failing layer instead of when shared-host contention is
the root cause. When the dedicated unit still reports `active/running` but the
loopback probe times out with `status_code=0`, the diagnose path now classifies
that state as `backend_active_but_unresponsive` so operators do not confuse a
hung-but-running backend with a healthy one. The guarded restart path now creates
the short-lived `/run/caia/maintenance/twinr_host_control.permit` contract
required by the host control guard, temporarily removes the persistent
`10-refuse-manual-restart.conf` drop-in because live-host systemd does not load
fresh `[Unit]` `RefuseManual*=no` drop-ins into `DropInPaths` during the same
repair window, verifies that `RefuseManualStart/Stop` actually opened, performs
the bounded stop/kill/start bounce, and then restores and re-verifies the
protected drop-in immediately afterwards. The Pi-side remote-memory watchdog now
also preserves explicit backend warmup details such as `Service warmup in
progress` in its rolling artifact and keeps a short re-probe cadence while that
exact server-reported startup state is active, so Twinr recovers promptly once
the dedicated backend finishes warming instead of waiting through the generic
multi-minute failure backoff.
If the backend is up but one prompt `catalog/current` head itself is a broken
blank document that times out on every read, use
`hardware/ops/repair_remote_prompt_current_heads.py` with an explicit namespace
and, when needed, a direct loopback SSH tunnel to `127.0.0.1:3044`. That path
publishes the canonical empty prompt head directly instead of blocking on the
already-broken old head.
If the public endpoint stays up but becomes slow or freeze-prone because the
dedicated backend host is reclaiming CPU or I/O for unrelated CAIA work, use
`hardware/ops/stabilize_remote_chonkydb_host.py`. That command touches the
host-side kill-switches for the worst reoffenders, disables the full proven
conflict-unit set across both systemd system scope and the active `thh`
user-session scope, still exposes the remote
`caia-twinr-host-control-guard.service` `required_units` only as operator
forensics, temporarily blocks that same guard service so it cannot reopen the
quiet-host lane mid-pass, and now preempts heavyweight workers such as
`caia-consumer-portal.service`, `caia-consumer-portal-demo.service`,
`caia-ops-chonky-search-guardrail.service`, and `ollama-gpu.service` when they
reclaim CPU from Twinr's
dedicated backend, bounded-kills proven stale long-running code-graph
benchmark runners, long-running user-session `chonkycode.cli artifact-ingest`
and `ccodex_memory_locomo_mc10_eval.py` workloads, and direct non-systemd
writers against the dedicated `twinr_dedicated_<port>/data` store path when
they bypass those unit lists through interactive user sessions, raises `caia-twinr-chonkydb-alt.service`
CPU/IO weights, keeps the short-lived
`/run/caia/maintenance/twinr_host_control.permit` contract open for the full
quiet-host window while the remote guard SSOT is being consulted, stages
system-scope runtime block drop-ins under `/run/systemd/system/<unit>.d/`
instead of relying on `mask` against `/etc/systemd/system` unit files that
cannot be replaced in place,
does the same for user-scope conflicts under
`$XDG_RUNTIME_DIR/systemd/user/<unit>.d/` plus an explicit user-manager
`daemon-reload` instead of trying to `mask` real `~/.config/systemd/user/*.service`
files in place,
and now also syncs reboot-persistent boot-release drop-ins under `/etc` plus a
dedicated `caia-twinr-host-boot-pacer.service` that recreates those same
conflict lanes in ordered post-boot waves rather than letting dependency,
timer, path, or guard activation stampede the host all at once,
keeps those runtime blockers tied to a Twinr-private maintenance token instead
of the historical shared `twinr_host_stabilizer_unblock` path that the
external CAIA host-control guard can also write,
and now stages them under the private
`91-twinr-private-host-stabilizer-block.conf` filename while deleting the
legacy shared `90-twinr-host-stabilizer-block.conf` file so foreign or stale
writers cannot silently switch Twinr back onto the shared unblock contract,
while leaving `caia-external-site.service`,
`caia-consumer-portal-demo.service`, `caia-ollama-gpu-proxy.service`, and
user-session `caia-molt.service` out of the boot-release plan entirely,
stages its remote sudo Python helper plus JSON payload through root-owned temp
files so large stabilization payloads cannot break on nested shell quoting, and
excludes that helper's own SSH/sudo process tree from the stale-process killer
so the stabilization pass cannot terminate itself while matching dedicated
backend data-path writers, and now bounds every remote `systemctl stop` with a
short timeout plus `kill --kill-who=all --signal=SIGKILL` fallback so one
stuck or `RefuseManualStop=yes` shared-host service cannot hang the whole
stabilization run or be silently treated as stopped, and only issues
`reset-failed` when a unit still reports a loaded failed/non-success state
instead of unconditionally tripping over inactive units that systemd no longer
considers resettable,
to the highest priority, and then re-probes the live public current-scope query
surface with the same empty-scope-safe `404 document_not_found` semantics used
by the repair helper. It now also verifies that the quiesced conflict units
really ended `inactive` and not still `enabled`; if systemd reactivates a held
unit during the first reload storm, the stabilizer first runs one short bounded
quiet-hold poll window so transient `active`/`enabled` samples can settle back
to `inactive` or `not-found`, then runs one bounded recovery pass for the still
violating units, and otherwise fails closed with
`conflict_units_reactivated_after_host_stabilization` instead of pretending the
host is quiet.
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
the sync once so the acceptance checkout can still converge cleanly. If the
stale cache tree is still owned by `root`, that prune path now prefers
passwordless `sudo rm -rf` before falling back to a normal user-owned delete.
The deploy helper builds on that mirror contract and adds the explicit pieces
the mirror intentionally does not own: authoritative `.env` sync, a no-deps
editable refresh by default, early activation of the bridged Pi
`dist-packages` view inside the preserved venv, targeted removal of stale venv
copies when the bridged system distribution already satisfies the direct repo
requirement, selective backfill of mirrored project runtime dependencies that
are still missing or out of spec on the Pi, optional
Pi-only runtime supplement installs via `hardware/ops/pi_runtime_requirements.txt`,
which mirrors `project.optional-dependencies.pi-runtime` in `pyproject.toml`,
optional browser-automation runtime installs via
`browser_automation/runtime_requirements.txt` plus
`browser_automation/playwright_browsers.txt` when those allowlisted manifests
are already part of the authoritative release snapshot, productive unit
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
Before that sync starts, the deploy snapshots the authoritative mirror scope
into a temporary local tree and mirrors from that immutable image, so parallel
shared-worktree edits cannot trip the deploy midway or produce a mixed-source
Pi checkout.
After the normal restart and env/import/permissions checks, the deploy also
runs the bounded live retention canary against a fresh remote-memory namespace
by default; use the low-level API or the operator `--skip-retention-canary`
flag only when you intentionally want to bypass that extra remote-memory
proof. The canary now has its own dedicated timeout budget instead of sharing
the generic per-SSH deploy timeout, because a real Pi-side retention pass can
legitimately outlive ordinary sync/install/restart substeps while still being
healthy.
The operator-facing CLI keeps stdout reserved for the final JSON payload and
emits live structured phase/substep progress on stderr so long Pi installs no
longer look indistinguishable from a hang.
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
