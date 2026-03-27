# hardware/ops

Pi-side operating-system service definitions plus leading-repo mirror helpers,
plus the development-host units that expose the remote-only voice gateway
contract to the Pi.

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
- the development-machine watchdog that mirrors the authoritative repo into `/twinr` without deleting Pi-local runtime state
- the leading-repo deploy command that mirrors code, syncs the authoritative runtime `.env`, reinstalls the editable package, restarts the productive Pi unit set, and verifies the Pi acceptance runtime

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
| [deploy_pi_runtime.py](./deploy_pi_runtime.py) | Operator-facing Pi deploy command: mirror the repo, sync the authoritative runtime `.env`, reinstall Twinr into the Pi venv, restart the base services plus any already-enabled repo-backed Pi runtime units, optionally first-rollout a disabled Pi unit, and verify post-restart health |
| [peer_ai_camera_observation_proxy.py](./peer_ai_camera_observation_proxy.py) | Transport-only peer-Pi HTTP service that exposes bounded live IMX500 observation payloads plus a coherent detection-plus-frame bundle from the dedicated AI-camera proxy Pi on `10.42.0.2:8767` |
| [peer_camera_snapshot_proxy.py](./peer_camera_snapshot_proxy.py) | Transport-only peer-Pi HTTP snapshot service that exposes bounded `rpicam-still` captures from the dedicated AI-camera proxy Pi on `10.42.0.2:8766` |
| [peer_servo_proxy.py](./peer_servo_proxy.py) | Transport-only peer-Pi HTTP service that exposes one local helper-Pi servo writer on `10.42.0.2:8768`, either a Pololu Maestro command port or a direct GPIO servo output such as GPIO18 via `lgpio_pwm` |
| [twinr-peer-ai-camera-proxy.service](./twinr-peer-ai-camera-proxy.service) | Proxy-Pi unit: keep the peer AI-camera observation service alive on the dedicated direct-link address |
| [twinr-peer-camera-proxy.service](./twinr-peer-camera-proxy.service) | Proxy-Pi unit: keep the peer camera snapshot service alive on the dedicated direct-link address |
| [twinr-peer-servo-proxy.service](./twinr-peer-servo-proxy.service) | Proxy-Pi unit: keep the peer Pololu Maestro service alive on the dedicated direct-link address |
| [voice_gateway_tcp_proxy.py](./voice_gateway_tcp_proxy.py) | Transport-only TCP bridge that exposes a LAN-visible port and forwards it to an already-established loopback tunnel for the real thh1986 voice gateway |
| [watch_pi_repo_mirror.py](./watch_pi_repo_mirror.py) | Continuously mirror the leading repo into `/twinr`, detect drift, and preserve Pi-local runtime-only paths such as `.env`, `.venv`, `state/`, and `artifacts/` |

The runtime supervisor intentionally runs as `root` so the productive
streaming loop keeps access to GPIO devices on deployed hosts. The dedicated
remote-memory watchdog also runs as `root` now, so both units share one
runtime-state ownership model and do not flap on root-vs-user lock files
inside `/twinr/state/` or `/twinr/state/chonkydb/`.

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

The peer camera snapshot proxy is also intentionally transport-only. It serves
bounded still PNGs from the dedicated proxy Pi and must not grow Twinr runtime
policy, camera interpretation, or ad-hoc SSH orchestration into the service.
The peer AI-camera observation proxy is equally transport-only. It may expose
bounded IMX500 observation payloads and debug facts from the helper Pi, but it
must not grow main-runtime orchestration, HDMI policy, or gesture/UI decisions
into the helper service. Its busy-cache path is intentionally short-lived only:
the helper may reuse one just-captured payload to bridge immediate lock
overlap, but stale helper frames must time out explicitly instead of being
replayed indefinitely as if they were fresh.

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
python3 hardware/ops/deploy_pi_runtime.py --live-text "Antworte nur mit: ok."
```

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

Install the peer camera snapshot proxy only on the dedicated camera proxy Pi:

```bash
sudo install -d -m 0755 /opt/twinr-peer-camera-proxy
sudo install -m 0755 hardware/ops/peer_camera_snapshot_proxy.py /opt/twinr-peer-camera-proxy/peer_camera_snapshot_proxy.py
sudo install -m 0644 hardware/ops/twinr-peer-camera-proxy.service /etc/systemd/system/twinr-peer-camera-proxy.service
sudo systemctl daemon-reload
sudo systemctl enable --now twinr-peer-camera-proxy.service
curl --fail http://10.42.0.2:8766/healthz
```

Install the peer AI-camera observation proxy on the same dedicated helper Pi
when the main Twinr Pi needs live HDMI attention and gesture behavior through
the direct-link camera proxy:

```bash
sudo install -d -m 0755 /opt/twinr-peer-ai-camera/repo
sudo install -m 0755 hardware/ops/peer_ai_camera_observation_proxy.py /opt/twinr-peer-ai-camera/repo/hardware/ops/peer_ai_camera_observation_proxy.py
sudo install -m 0644 hardware/ops/twinr-peer-ai-camera-proxy.service /etc/systemd/system/twinr-peer-ai-camera-proxy.service
sudo systemctl daemon-reload
sudo systemctl enable --now twinr-peer-ai-camera-proxy.service
curl --fail http://10.42.0.2:8767/healthz
```

Install the peer servo proxy on the helper Pi when the main Twinr Pi should
keep the high-level attention logic but the physical servo output lives on the
helper Pi instead of the main host. The checked-in service unit now targets the
current direct helper-Pi GPIO wiring on `GPIO18` and exposes that as logical
channel `1`. For an older local Pololu Maestro transport, run the same proxy
without the GPIO-specific flags.

```bash
sudo install -d -m 0755 /opt/twinr-peer-servo/repo/hardware/ops
sudo install -d -m 0755 /opt/twinr-peer-servo/repo/src/twinr/hardware
sudo install -m 0755 hardware/ops/peer_servo_proxy.py /opt/twinr-peer-servo/repo/hardware/ops/peer_servo_proxy.py
sudo install -m 0644 src/twinr/hardware/servo_maestro.py /opt/twinr-peer-servo/repo/src/twinr/hardware/servo_maestro.py
sudo install -m 0644 hardware/ops/twinr-peer-servo-proxy.service /etc/systemd/system/twinr-peer-servo-proxy.service
sudo systemctl daemon-reload
sudo systemctl enable --now twinr-peer-servo-proxy.service
curl --fail http://10.42.0.2:8768/healthz
```

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
artefacts, not authoritative repo content.

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
- installs mirrored browser-automation runtime requirements from `browser_automation/runtime_requirements.txt` and Playwright browsers listed in `browser_automation/playwright_browsers.txt` when those manifests exist locally
- installs the mirrored productive systemd unit files into `/etc/systemd/system/`
- restarts `twinr-remote-memory-watchdog.service`, `twinr-runtime-supervisor.service`, and `twinr-web.service`
- also picks up any additional repo-backed Pi runtime unit that is already enabled on the Pi, such as `twinr-whatsapp-channel.service`
- verifies that those services are active again and runs the bounded Pi env-contract probe

`deploy_pi_runtime.py` is the operator-facing replacement for the old manual
"run the mirror watchdog as the deploy step" workflow. The mirror still exists
as the internal code-sync mechanism and drift-diagnostic tool, but normal Pi
runtime rollout should go through the deploy command.

Use `--skip-env-sync` only when the Pi env must intentionally stay divergent
from the leading repo. Use `--live-text` or `--live-search` when you want the
post-deploy verification to include one real OpenAI-backed proof, not just the
fail-closed env-contract plus service-health checks.
If an optional Pi runtime unit such as `twinr-whatsapp-channel.service` was
already enabled before but its installed unit file later became masked or
corrupted, the default deploy path now repairs it automatically as long as the
original enable symlink still exists on the host.
Use `--rollout-service ...` when you are rolling out a new optional Pi unit for
the first time and it is not enabled on the target host yet. Use explicit
`--service ...` flags only when you intentionally want to replace the automatic
deploy target set with a narrower or different one.
Use `--install-with-deps` only when you intentionally want the Pi deploy to
re-resolve runtime dependencies; the default no-deps editable refresh avoids
rebuilding Pi-host packages such as `PyQt5` on every deploy.

## Peer camera flow

Use the dedicated proxy Pi when the main Twinr Pi has no local still camera but
needs the usual still-photo contract. When live HDMI attention/gesture is also
enabled, use the combined AI-camera proxy service so one process owns the
IMX500:

```bash
export TWINR_CAMERA_PROXY_SNAPSHOT_URL=http://10.42.0.2:8767/snapshot.png
PYTHONPATH=src ./.venv/bin/python -m twinr --env-file .env --camera-capture-output /tmp/peer-proxy.png
```

`src/twinr/hardware/camera.py` keeps the upstream `CapturedPhoto` contract
unchanged, but fetches the PNG bytes over the direct Ethernet link instead of
opening a local `/dev/video*` node.

For live HDMI attention and gesture behavior, point the main Pi's proactive
vision provider at the helper Pi:

```bash
export TWINR_PROACTIVE_VISION_PROVIDER=remote_proxy
export TWINR_PROACTIVE_REMOTE_CAMERA_BASE_URL=http://10.42.0.2:8767
```

For open-source-friendly wiring, prefer the high-level topology env instead of
remembering the lower-level camera flags:

```bash
export TWINR_CAMERA_HOST_MODE=second_pi
export TWINR_CAMERA_SECOND_PI_BASE_URL=http://10.42.0.2:8767
```

That high-level mode keeps still capture on the helper Pi via
`/snapshot.png` and defaults the proactive camera path to `remote_frame`, so
the helper Pi holds the physical camera while the main Pi can do the hot
gesture/attention lifting locally from one coherent helper detection-plus-frame
bundle instead of separate live detection and snapshot fetches.
