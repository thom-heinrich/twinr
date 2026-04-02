# hardware/ops

Pi-side operating-system service definitions plus leading-repo mirror helpers,
plus the development-host units that expose the remote-only voice gateway
contract to the Pi.

Legacy note: the helper-Pi camera and servo proxies are no longer kept inside
the active `hardware/ops` tree. Historical recovery artifacts now live under
`__legacy__/hardware/ops/`. Current Twinr runtime wiring is single-Pi; the
canonical config loader rejects the retired helper-Pi camera and servo envs
instead of treating them as supported operating modes.

## Responsibility

`hardware/ops` owns:
- the authoritative systemd unit that keeps the productive Twinr runtime running on the Pi
- the dedicated systemd unit that keeps the remote-memory watchdog alive across runtime-supervisor restarts
- OS-level launch wiring for background checks and runtime processes that must survive shell logout or crashes
- the development-host bounded drone daemon that exposes Twinr's mission-level inspection contract behind manual-arm and preflight gates
- the development-host orchestrator server unit that keeps the host-side `remote_asr` websocket endpoint and embedded `/v1/transcribe` surface alive
- the development-host LAN bridge that exposes a stable `:8797` websocket port to the Pi while forwarding byte-for-byte into that host-side orchestrator endpoint
- the Pi-side bootstrap entrypoint for self-coding Codex prerequisites
- the Pi-side operator script that fail-closes the OpenAI env contract before isolated provider probes
- the development-host operator script that diagnoses the public ChonkyDB endpoint against the dedicated backend service and only restarts the backend when that backend is the proven failing layer
- the development-host operator script that stabilizes the dedicated Twinr ChonkyDB host when shared-host system units, user-session units, or non-Twinr workers directly pointed at the dedicated backend reclaim CPU/I/O from it
- the development-host operator script that force-repairs unreadable prompt-memory and managed-context `catalog/current` heads on one explicit remote namespace when those heads themselves have become unreadable blank documents
- the development-machine watchdog that mirrors the authoritative repo into `/twinr` without deleting Pi-local runtime state
- the leading-repo deploy command that snapshots the authoritative repo scope, mirrors code, syncs the authoritative runtime `.env`, reinstalls the editable package, restarts the productive Pi unit set, runs the bounded live retention canary, and verifies the Pi acceptance runtime

`hardware/ops` does **not** own:
- the watchdog implementation itself; that lives in `src/twinr/ops`
- Twinr runtime orchestration or product logic
- voice activation logic, transcript matching, or websocket semantics; transport-only
  bridges here must stay byte-for-byte

## Files

| File | Purpose |
|---|---|
| [twinr-remote-memory-watchdog.service](./twinr-remote-memory-watchdog.service) | Dedicated unit: keep the fail-closed remote-memory watchdog warm and continuously refreshing its artifact |
| [twinr-runtime-supervisor.service](./twinr-runtime-supervisor.service) | Productive unit: authoritatively supervise the streaming loop while consuming the external remote-memory-watchdog artifact |
| [twinr-web.service](./twinr-web.service) | Productive unit: keep the Twinr web control portal running with managed sign-in |
| [drone_daemon.py](./drone_daemon.py) | Development-host bounded drone mission daemon: preflight, manual-arm gate, stationary-observe evidence capture by default, plus an explicit hover-test mode for the first takeoff-hover-land primitive |
| [twinr-drone-daemon.service](./twinr-drone-daemon.service) | Development-host unit: keep the bounded drone daemon alive on a stable local HTTP endpoint for Twinr mission planning |
| [twinr-orchestrator-server.service](./twinr-orchestrator-server.service) | Development-host unit: keep the host-side orchestrator websocket endpoint plus embedded `/v1/transcribe` remote-ASR surface alive on `127.0.0.1:8798` |
| [twinr-voice-gateway-bridge.service](./twinr-voice-gateway-bridge.service) | Development-host unit: expose `0.0.0.0:8797` to the Pi and forward it byte-for-byte into the host-side `127.0.0.1:8798` orchestrator endpoint |
| [bootstrap_self_coding_pi.py](./bootstrap_self_coding_pi.py) | Reproducibly sync the pinned self-coding Codex bridge/auth and run the remote self-test |
| [install_whatsapp_node_runtime.py](./install_whatsapp_node_runtime.py) | Download, verify, and stage the pinned local Node.js runtime under `state/tools/` for the WhatsApp Baileys worker |
| [check_pi_openai_env_contract.py](./check_pi_openai_env_contract.py) | Validate `/twinr/.env` for direct OpenAI-backed acceptance probes and optionally run one real provider request without manual key injection |
| [repair_remote_chonkydb.py](./repair_remote_chonkydb.py) | Diagnose the public ChonkyDB URL against the dedicated backend host and optionally repair the backend service without blind restarts |
| [stabilize_remote_chonkydb_host.py](./stabilize_remote_chonkydb_host.py) | Quiesce known shared-host conflict units on `thh1986` across both system and active user-session scope, stop non-Twinr workers that were pointed at Twinr's dedicated backend, and raise the dedicated Twinr backend CPU/IO priority before re-probing public `/instance` availability |
| [repair_remote_prompt_current_heads.py](./repair_remote_prompt_current_heads.py) | Force-publish canonical empty prompt-memory / managed-context current heads on one explicit remote namespace when the old heads are unreadable |
| [deploy_pi_runtime.py](./deploy_pi_runtime.py) | Operator-facing Pi deploy command: snapshot the authoritative mirror scope, mirror that stable repo image onto the Pi, sync the authoritative runtime `.env`, independently attest mirrored repo contents on `/twinr`, reinstall Twinr into the Pi venv, repair stale venv entrypoints, restart the base services plus any already-enabled repo-backed Pi runtime units, run the bounded live retention canary by default, optionally first-rollout a disabled Pi unit, and verify post-restart health |
| [voice_gateway_tcp_proxy.py](./voice_gateway_tcp_proxy.py) | Transport-only TCP bridge that exposes a LAN-visible port and forwards it to an already-established loopback tunnel for the real thh1986 voice gateway |
| [watch_pi_repo_mirror.py](./watch_pi_repo_mirror.py) | Continuously mirror the leading repo into `/twinr`, detect drift, and preserve Pi-local runtime-only paths such as `.env`, `.venv`, `state/`, and `artifacts/` |

The runtime supervisor intentionally runs as `root` so the productive
streaming loop keeps access to GPIO devices on deployed hosts. The dedicated
remote-memory watchdog also runs as `root` now, so both units share one
runtime-state ownership model and do not flap on root-vs-user lock files
inside `/twinr/state/` or `/twinr/state/chonkydb/`.
Because those root-owned services also refresh shared operator diagnostics
under `/twinr/artifacts/stores/ops/`, the deploy permission-repair step now
explicitly keeps `remote_memory_watchdog.json` and
`display_ambient_impulse.json` operator-readable inside that otherwise `0700`
ops directory, so non-root acceptance probes can read the same live artifacts
without weakening directory-level confinement.
That dedicated watchdog unit now also treats exit code `75` as
restart-preventing, so an already-owned singleton does not thrash in an
immediate restart loop while the existing owner is still healthy.

The development-host voice bridge is intentionally transport-only. It must not
run voice activation logic, STT logic, or websocket-aware product behavior itself.
Those stay in the dedicated host-side orchestrator endpoint on `127.0.0.1:8798`;
the bridge only exposes that endpoint on a stable LAN-visible `:8797` socket
for the Pi. The bridge unit is ordered after `twinr-orchestrator-server.service`
and waits briefly for `:8798` to listen so simultaneous host restarts do not
flap the LAN listener before the orchestrator socket exists.
Do not reintroduce NAT port-redirect rules in front of this service;
`PREROUTING` redirects can intercept external Pi traffic before the LAN listener
sees it and cause immediate `Connection refused` failures even though the bridge
is up.

The drone daemon is intentionally mission-bounded as well. Twinr may only queue
high-level inspect missions, read state, cancel work, or request local manual
arm approval. Direct roll/pitch/yaw/thrust commands do not belong in this
surface. The current `stationary_observe_only` mode is the accepted first
runtime slice: it proves the API, preflight gates, and artifact path while
keeping motion disabled until the future primitive executor and external
pose-provider stack are ready. A second explicit mode,
`bounded_hover_test_only`, is now available for the first live
`takeoff -> hover -> land` primitive, but it must be enabled intentionally by
the operator and is not the default service mode.

Retired standalone break-glass units are no longer tracked here. The dedicated
remote-memory watchdog service is not break-glass; it is the productive owner
for the watchdog so warm remote state survives runtime-supervisor restarts. If
an operator wants to keep local copies of older units around during cleanup,
they belong under the ignored top-level `__legacy__/hardware/ops/` folder.

## Install

The Pi runtime units below are Pi-only. Do not install them on development
laptops or other non-Pi hosts, even if those machines also have a `/twinr`
checkout.

```bash
sudo systemctl disable --now twinr-streaming-loop.service twinr-display-loop.service || true
sudo cp hardware/ops/twinr-remote-memory-watchdog.service /etc/systemd/system/
sudo cp hardware/ops/twinr-runtime-supervisor.service /etc/systemd/system/
sudo cp hardware/ops/twinr-web.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now twinr-remote-memory-watchdog.service twinr-runtime-supervisor.service twinr-web.service
sudo systemctl status twinr-remote-memory-watchdog.service
sudo systemctl status twinr-runtime-supervisor.service
sudo systemctl status twinr-web.service
python3 hardware/ops/install_whatsapp_node_runtime.py
python3 hardware/ops/bootstrap_self_coding_pi.py
python3 hardware/ops/check_pi_openai_env_contract.py --env-file /twinr/.env
python3 hardware/ops/repair_remote_chonkydb.py --no-restart
python3 hardware/ops/stabilize_remote_chonkydb_host.py
python3 hardware/ops/repair_remote_prompt_current_heads.py --namespace twinr_longterm_v1:twinr:a7f1ed265838 --base-url http://127.0.0.1:43044 --force
python3 hardware/ops/deploy_pi_runtime.py --live-text "Antworte nur mit: ok."
```

The `twinr-web.service` unit keeps the portal alive, but remote browser access
still stays fail-closed until `/twinr/.env` enables `TWINR_WEB_ALLOW_REMOTE=1`,
`TWINR_WEB_REQUIRE_AUTH=1`, and a matching `TWINR_WEB_ALLOWED_HOSTS` allowlist.

For a text-only operator proof that exercises the direct answer path, a real
web/tool turn, and the live remote-memory acceptance matrix in one run, use:

```bash
PYTHONPATH=src python3 -m twinr --env-file .env --non-voice-e2e-acceptance
```

That command persists the rolling result to
`artifacts/stores/ops/non_voice_e2e_acceptance.json` and a per-run report under
`artifacts/reports/non_voice_e2e_acceptance/`, so deploy-time checks no longer
depend on manually comparing separate direct/tool/memory probes.

When the development-host remote-memory endpoint itself is unhealthy, use the
dedicated repair helper before issuing any manual backend restart:

```bash
python3 hardware/ops/repair_remote_chonkydb.py --no-restart
python3 hardware/ops/repair_remote_chonkydb.py
python3 hardware/ops/stabilize_remote_chonkydb_host.py
```

The first command is diagnose-only. It proves the public
`https://tessairact.com:2149` URL, the dedicated backend systemd unit on
`thh1986`, and the backend loopback `127.0.0.1:3044` separately. The second
command keeps the same diagnosis but is allowed to restart the backend service
when the backend itself is the failing layer. If the backend loopback is
already healthy and only the public URL is down, the helper refuses a blind
backend restart and reports a public-proxy/routing outage instead.
The third command is for the opposite failure shape: the host is up and the
public endpoint may still answer, but shared-host boot catch-up or background
CAIA work has made the dedicated Twinr backend slow or freeze-prone. The
stabilizer touches the worst kill-switches, disables the curated conflict-unit
set across both system and active user-session scope, stops proven non-Twinr
workers that were directly wired to Twinr's dedicated `127.0.0.1:3044`
backend, raises `caia-twinr-chonkydb-alt.service` CPU/IO priority, and then
re-probes the public `/instance` endpoint so operators can see whether the host
slowdown actually cleared. That is intentionally a host-availability proof; if
you need the stricter current-scope query contract as well, follow with
`repair_remote_chonkydb.py --no-restart`.
If the backend stays healthy but prompt-context reads time out on one broken
blank `catalog/current` document, open an SSH tunnel to the backend loopback
and run `repair_remote_prompt_current_heads.py` against that explicit namespace.
That helper publishes the canonical empty prompt head directly instead of
waiting on the unreadable old head.

Install the development-host orchestrator endpoint plus LAN bridge only on
the machine that owns the leading repo checkout:

```bash
sudo cp hardware/ops/twinr-orchestrator-server.service /etc/systemd/system/
sudo cp hardware/ops/twinr-voice-gateway-bridge.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now twinr-orchestrator-server.service
sudo systemctl enable --now twinr-voice-gateway-bridge.service
sudo systemctl status twinr-orchestrator-server.service
sudo systemctl status twinr-voice-gateway-bridge.service
```

Install the development-host bounded drone daemon only on the machine that owns
the Crazyradio/Bitcraze workspace and future pose-provider sidecars:

```bash
sudo cp hardware/ops/twinr-drone-daemon.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now twinr-drone-daemon.service
sudo systemctl status twinr-drone-daemon.service
curl --fail http://127.0.0.1:8791/healthz
python3 -m twinr --env-file .env --drone-status
```

For bench-safe bring-up before a real pose-provider is available, run the daemon
in the foreground with the healthy stub pose source:

```bash
python3 hardware/ops/drone_daemon.py --repo-root /home/thh/twinr --env-file /home/thh/twinr/.env --pose-provider stub_ok --bind 127.0.0.1 --port 8791
python3 -m twinr --env-file .env --drone-inspect "self test"
python3 -m twinr --env-file .env --drone-manual-arm DRN-...
```

For the first live hover test, start the daemon in the explicit hover-test
mode and queue the new bounded mission type:

```bash
python3 hardware/ops/drone_daemon.py --repo-root /home/thh/twinr --env-file /home/thh/twinr/.env --pose-provider stub_ok --skill-layer-mode bounded_hover_test_only --bind 127.0.0.1 --port 8791
python3 -m twinr --env-file .env --drone-hover-test
python3 -m twinr --env-file .env --drone-manual-arm DRN-...
```

The bounded operator proof is:
- `POST /missions` returns `pending_manual_arm`
- `POST /ops/missions/<id>/arm` is local-host only by default
- mission execution captures stationary evidence by default instead of moving the aircraft
- hover motion is only available when the daemon was explicitly started in `bounded_hover_test_only`
- before any live takeoff setpoint, the hover worker now explicitly applies and verifies `stabilizer.estimator=2`, `stabilizer.controller=1`, `motion.disable=0`, and a bounded `kalman.resetEstimation` pulse; those final values are persisted into the mission artifact
- the hover worker now blocks takeoff until a bounded estimator-settle gate sees stable `kalman.varPX/PY/PZ`, quiet roll/pitch, adequate `motion.squal`, and a valid downward `range.zrange`
- hover execution now runs through an explicit stateful hover-setpoint primitive with its own abort/landing path instead of depending on the more implicit `MotionCommander` context-manager flow
- the landing path no longer permits blind motor cutoff: after the staged descent it waits for deterministic landing-complete signals before issuing `send_stop_setpoint()`, first preferring fresh downward `range.zrange` confirmation at or below the current `5 cm` touchdown-cut gate for three consecutive samples and otherwise accepting the firmware supervisor reporting `is flying = false`
- once the aircraft is airborne, the hover primitive no longer aborts the active landing path by touchdown-timeout exception; instead it keeps driving the bounded zero-height landing sequence until one of the completion signals arrives
- successful hover missions now persist the worker's bounded stability telemetry alongside the mission summary so the operator can inspect flow, z-range, directional clearance, velocity, gyro, thrust, radio RSSI, and attitude evidence after each run; the hover summary is computed from the inferred airborne window so ground-settle transients do not inflate drift metrics
- when the Multi-ranger deck is present, the hover worker now takes one short clearance snapshot before takeoff and blocks the mission if nearby front/back/left/right/up obstacles are already inside the configured hover envelope
- completed hover runs are now fail-closed on the recorded telemetry: large altitude overshoot above the requested hover height or excessive under-load battery sag downgrade the run to `unstable` instead of reporting `completed`
- if the hover worker times out or is cancelled after a real flight, the daemon now also persists a partial hover artifact with the worker trace file, last trace phase/status, and stdout/stderr tails so teardown hangs can be debugged without losing the run
- the first bounded hover path now also expects the Twinr STM32 app-layer failsafe (`twinrFs`) to be flashed on the Crazyflie; the worker sends Appchannel heartbeats while the host is healthy, but heartbeat-loss, low-battery, and clearance-triggered safe-land logic then continues locally on the aircraft without the daemon
- `GET /state` keeps `manual_arm_required=true` and exposes preflight reasons when radio or pose is unhealthy

Before relying on live hover missions, build and flash the on-device failsafe:

```bash
bash hardware/bitcraze/build_on_device_failsafe.sh
/twinr/bitcraze/.venv/bin/python hardware/bitcraze/flash_on_device_failsafe.py
```

The post-flash probe is important: it reconnects over the normal radio URI and
only passes when the Crazyflie exposes the `twinrFs.*` param surface that
proves the firmware app is live.

## Legacy helper-Pi proxies

The old helper-Pi camera and servo proxy scripts plus their units were moved
out of the active tree and now live only under `__legacy__/hardware/ops/`.
They are retained solely for historical recovery, migration archaeology, or
manual reference. They are not part of the supported productive runtime and are
intentionally excluded from active `hardware/component.yaml` metadata and the
normal hardware validation slice.

## Mirror watchdog

Run the repo mirror watchdog on the development machine, not on the Pi:

```bash
python3 hardware/ops/watch_pi_repo_mirror.py --once
python3 hardware/ops/watch_pi_repo_mirror.py --interval-s 5
python3 hardware/ops/watch_pi_repo_mirror.py --interval-s 5 --metadata-only
```

The mirror is intentionally one-way: `/home/thh/twinr` is authoritative, `/twinr`
is healed toward it, and runtime-only Pi paths stay protected from deletion.
Exact-content drift detection is the default on every cycle; `--metadata-only`
is an explicit throughput tradeoff that falls back to periodic checksum audits.
Protected Pi-local paths use perishable rsync filters so stale nested repo
copies can still be deleted instead of being pinned by an inner `.env` or
similar runtime-only file. The mirror also excludes local devices, FIFOs, and
other special files so transient tooling artefacts in the leading repo cannot
break Pi sync cycles. Python bytecode caches (`__pycache__/`, `*.pyc`, `*.pyo`)
also stay outside the mirror because they are runtime-local artefacts and can
become root-owned on the Pi. Nested `node_modules/` trees are also excluded
from the mirror because SDK and channel dependency installs are local build
artefacts, not authoritative repo content. The generated Crazyflie failsafe
firmware build tree under `hardware/bitcraze/twinr_on_device_failsafe/build/`
is excluded for the same reason.

## Deploy command

Run the full acceptance deploy from the leading repo when code or productive
Pi settings changed:

```bash
python3 hardware/ops/deploy_pi_runtime.py
python3 hardware/ops/deploy_pi_runtime.py --live-text "Antworte nur mit: ok."
python3 hardware/ops/deploy_pi_runtime.py --rollout-service twinr-whatsapp-channel
python3 hardware/ops/deploy_pi_runtime.py --skip-env-sync --service twinr-runtime-supervisor
```

By default the deploy command:

- mirrors `/home/thh/twinr` into `/twinr`
- treats the local `.env` as authoritative and overwrites `/twinr/.env` with a backup
- refreshes `/twinr/.venv` via `pip install --no-deps -e /twinr`
- activates the bridged Pi `dist-packages` view inside `/twinr/.venv` before dependency checks and drops stale venv duplicates when the bridged system package already satisfies the direct repo requirement, which prevents preserved overlays such as `PyQt5` from poisoning `pip check`
- compares the mirrored `pyproject.toml` runtime dependencies against the Pi venv and installs only missing or out-of-spec packages, so new lightweight dependencies can roll out without a full resolver pass
- installs mirrored Pi-only runtime supplement packages from `hardware/ops/pi_runtime_requirements.txt` when that manifest exists locally, so optional direct-import extras such as `RapidFuzz`, `wcwidth`, `onnx`, `msgspec`, `orjson`, `portalocker`, `zstandard`, `h2`, and `opentelemetry-api` stay present on the acceptance Pi without forcing every local environment to carry them
- treats `hardware/ops/pi_runtime_requirements.txt` as the Pi copy of `project.optional-dependencies.pi-runtime` from `pyproject.toml`; the repo tests fail if those two lists drift apart
- attests `/twinr/.venv/bin` after that refresh and normalizes stale copied Python-wrapper shebangs back to `/twinr/.venv/bin/python` so direct console scripts such as `pytest` keep working on the Pi
- hands any root-owned mirrored `__pycache__/` trees back to the runtime user before rebuilding checked-hash bytecode, so old productive imports cannot block later deploys
- runs a fixed `/twinr/.venv/bin/python` import contract before restart, so critical direct-import modules must still import successfully after the deploy instead of failing only later in live runtime
- explicitly syncs local browser-automation runtime manifests from `browser_automation/runtime_requirements.txt` and `browser_automation/playwright_browsers.txt`, then installs those requirements and Playwright browsers on the Pi when they exist locally
- snapshots the authoritative repo mirror scope into a temporary local tree before the rsync phase, so unrelated shared-worktree edits cannot self-abort the rollout or produce a mixed-source Pi checkout mid-deploy
- uploads an authoritative local manifest and independently attests the mirrored `/twinr` repo contents by SHA256/link target before any productive restart, so stale source files cannot still yield a green deploy
- installs the mirrored productive systemd unit files into `/etc/systemd/system/`
- restarts `twinr-remote-memory-watchdog.service`, `twinr-runtime-supervisor.service`, and `twinr-web.service`
- also picks up any additional repo-backed Pi runtime unit that is already enabled on the Pi, such as `twinr-whatsapp-channel.service`
- verifies that those services are active again, runs the bounded Pi env-contract probe, and then runs the bounded live retention canary unless `--skip-retention-canary` was requested
- re-bases repo-owned workflow-trace env paths such as `TWINR_WORKFLOW_TRACE_DIR=/home/thh/twinr/...` onto the active Pi checkout root during env sync, so `/twinr` does not inherit stale leading-repo absolute paths
- emits nested `retention_canary` progress events while the fresh-namespace probe is still running, so long remote-memory canary runs do not look like a silent deploy hang

`deploy_pi_runtime.py` is the operator-facing replacement for the old manual
"run the mirror watchdog as the deploy step" workflow. The mirror still exists
as the internal code-sync mechanism and drift-diagnostic tool, but normal Pi
runtime rollout should go through the deploy command.

Use `--skip-env-sync` only when the Pi env must intentionally stay divergent
from the leading repo. Use `--live-text` or `--live-search` when you want the
post-deploy verification to include one real OpenAI-backed proof, not just the
fail-closed env-contract plus service-health checks.
During long-running installs the operator command now writes structured deploy
progress JSON lines to stderr while keeping stdout reserved for the final
success or error payload.
Use `--skip-retention-canary` only when you intentionally want to bypass the
fresh-namespace remote-memory retention proof after restart.
If an optional Pi runtime unit such as `twinr-whatsapp-channel.service` was
already enabled before but its installed unit file later became masked or
corrupted, the default deploy path now repairs it automatically as long as the
original enable symlink still exists on the host.
Use `--rollout-service ...` when you are rolling out a new optional Pi unit for
the first time and it is not enabled on the target host yet. Use explicit
`--service ...` flags only when you intentionally want to replace the automatic
deploy target set with a narrower or different one.
Use `--install-with-deps` only when you intentionally want the Pi deploy to
re-resolve the full runtime dependency graph; the default no-deps editable
refresh now still backfills only missing or out-of-spec mirrored project
dependencies, heals stale venv copies that shadow bridged Pi system packages,
and avoids rebuilding Pi-host packages such as `PyQt5` on every deploy.

## Current camera flow

The productive Twinr runtime now expects the camera on the main Pi. Use the
local camera path and the main runtime services for bounded still capture,
proactive observation, HDMI attention, and gesture behavior. The older helper-
Pi proxy flow above is legacy-only and should not be reintroduced into active
`.env` files.
