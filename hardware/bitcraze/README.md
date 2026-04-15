# bitcraze

Provision a clean Crazyradio/Crazyflie workspace for Twinr-side drone work.

## Responsibility

`hardware/bitcraze` owns:
- install the Crazyradio USB udev rules needed for non-root access
- stage pinned Crazyradio 2.0 firmware assets for compatibility and recovery
- create an isolated Python workspace under `/twinr/bitcraze`
- install the pinned Bitcraze Python packages into that workspace
- probe the connected Bitcraze USB device and confirm whether it is ready for `cflib`

`hardware/bitcraze` does **not** own:
- Twinr runtime orchestration
- drone behavior policy
- direct source-of-truth code changes in `/twinr` without first updating `/home/thh/twinr`

## Compatibility note

The current official `cflib` and `cfclient` releases still expect the classic
Crazyradio PA USB identity (`1915:7777`). If a Crazyradio 2.0 is attached in
its UF2/native state (`35f0:bad2`), this setup path stages both firmware
variants but defaults to the PA-emulation UF2 for immediate Python-library
compatibility.

## Key files

| File | Purpose |
|---|---|
| [setup_bitcraze.sh](./setup_bitcraze.sh) | Install USB access rules, stage firmware assets, create `/twinr/bitcraze`, optionally flash a compatible UF2, and install the pinned Bitcraze Python workspace |
| [probe_crazyradio.py](./probe_crazyradio.py) | Inspect connected Bitcraze USB devices, classify the current mode, and validate the local workspace/`cflib` access path |
| [prepare_olimex_jtag.sh](./prepare_olimex_jtag.sh) | Check or prepare host prerequisites for AI-Deck GAP8 JTAG recovery with the Olimex ARM-USB-TINY-H bundle, including optional `openocd`/`docker` installation and runtime-user group prep |
| [CRAZYFLIE_RECOVERY_RUNBOOK.md](./CRAZYFLIE_RECOVERY_RUNBOOK.md) | Authoritative Crazyflie 2.x recovery path, decision order, and incident learnings from the proven brick/recovery case |
| [FIRMWARE_RELEASE_LANES.md](./FIRMWARE_RELEASE_LANES.md) | Fail-closed dev/bench/operator/recovery firmware release process for the on-device failsafe |
| [probe_multiranger.py](./probe_multiranger.py) | Connect to the Crazyflie, read the current deck flags, and sample Multi-ranger directions plus supporting Flow/Z-ranger presence for immediate post-install acceptance |
| [probe_deck_bus.py](./probe_deck_bus.py) | Run one bounded non-rotor deck-bus probe that power-cycles the STM32/deck rail by default, then captures firmware `deck.*`, refreshed memory TOC evidence, and startup console lines for the proven `I2C1` deck-bus failure mode |
| [on_device_failsafe.py](./on_device_failsafe.py) | Host-side Appchannel adapter that proves the firmware app is loaded, sends heartbeat/config packets plus one bounded hover intent, and records the v4 returned mission/failsafe status packets including bounded debug truth |
| [hover_primitive.py](./hover_primitive.py) | Internal helper module for deterministic hover pre-arm params, SITL/replay hover primitives, and shared takeoff/landing contracts; live hardware no longer uses it as the takeoff/hover control owner |
| [local_navigation.py](./local_navigation.py) | Pure host-side local-navigation policy that converts one bounded Multi-ranger clearance snapshot into one deterministic short-range translation or hover-anchor-only decision |
| [capture_runtime_telemetry.py](./capture_runtime_telemetry.py) | Thin wrapper that starts the shared runtime telemetry module on one Crazyflie session and emits one bounded JSON snapshot for the daemon/operator path |
| [replay_hover_trace.py](./replay_hover_trace.py) | Replay one stored hover report and optional phase trace through the real bounded hover primitive so unsafe live runs become deterministic regressions before the next flight |
| [run_hover_test.py](./run_hover_test.py) | Run one bounded takeoff-hover-land flight test; on real hardware it now sends one bounded hover intent to `twinrFs` and only observes/reports mission status, while SITL still uses the host primitive lane |
| [run_local_inspect_mission.py](./run_local_inspect_mission.py) | Run one bounded `takeoff -> optional local translation -> still capture -> land` inspect mission; live hardware is currently blocked until the mission is migrated onto the same on-device `twinrFs` control lane |
| [build_on_device_failsafe.sh](./build_on_device_failsafe.sh) | Build the Crazyflie STM32 on-device failsafe app from a pinned Bitcraze firmware checkout and emit a build attestation |
| [build_patched_crazyflie_firmware.sh](./build_patched_crazyflie_firmware.sh) | Build one patched official Crazyflie STM32 firmware image from a pinned clean checkout plus explicit repo-local patch files, then emit `cf2.bin`, `cf2.elf`, and a build manifest outside the temporary worktree |
| [patches/crazyflie_firmware_zranger2_invalid_status_fail_closed.patch](./patches/crazyflie_firmware_zranger2_invalid_status_fail_closed.patch) | Repo-local Crazyflie firmware fix for the `zranger2` driver so invalid VL53L1 readings surface as explicit invalid range instead of collapsing to `0 mm` |
| [firmware_release_policy.py](./firmware_release_policy.py) | Shared lane/attestation policy for build, promotion, flash, and recovery interlocks |
| [promote_on_device_failsafe_release.py](./promote_on_device_failsafe_release.py) | Promote one attested custom firmware artifact into the bench or operator release lane |
| [flash_on_device_failsafe.py](./flash_on_device_failsafe.py) | Flash a lane-authorized STM32 artifact through `cfloader` and verify the new `twinrFs` firmware app over the normal radio link |
| [twinr_on_device_failsafe/](./twinr_on_device_failsafe) | Crazyflie app-layer hover/failsafe controller in C that owns bounded takeoff/hover/landing plus local safe-land and obstacle-repel behavior without host tick control |

## Usage

```bash
sudo ./hardware/bitcraze/setup_bitcraze.sh --workspace /twinr/bitcraze --runtime-user thh
python3 hardware/bitcraze/probe_crazyradio.py --workspace /twinr/bitcraze
python3 hardware/bitcraze/probe_crazyradio.py --workspace /twinr/bitcraze --json
sudo ./hardware/bitcraze/prepare_olimex_jtag.sh --workspace /twinr/bitcraze --install-apt --ensure-user-groups
/twinr/bitcraze/.venv/bin/python hardware/bitcraze/probe_multiranger.py --workspace /twinr/bitcraze --require-deck multiranger --json
bash hardware/bitcraze/build_patched_crazyflie_firmware.sh --firmware-root /tmp/crazyflie-firmware --expected-firmware-revision 2025.12.1
/twinr/bitcraze/.venv/bin/python hardware/bitcraze/probe_deck_bus.py --workspace /twinr/bitcraze --json
bash hardware/bitcraze/build_on_device_failsafe.sh --firmware-root /tmp/crazyflie-firmware --expected-firmware-revision 2025.12.1
bash hardware/bitcraze/build_on_device_failsafe.sh --firmware-root /tmp/crazyflie-firmware --expected-firmware-revision 2025.12.1 --patch hardware/bitcraze/patches/crazyflie_firmware_zranger2_invalid_status_fail_closed.patch
python3 hardware/bitcraze/promote_on_device_failsafe_release.py --lane bench --release-id twinr-failsafe-bench-2026-04-11-r1 --approved-by thh --reason "bench validation passed" --validation-evidence test:test_bitcraze_flash_on_device_failsafe.py
/twinr/bitcraze/.venv/bin/python hardware/bitcraze/flash_on_device_failsafe.py --lane dev --device-role dev --attestation hardware/bitcraze/twinr_on_device_failsafe/build/twinr_on_device_failsafe.build-attestation.json --json
/twinr/bitcraze/.venv/bin/python hardware/bitcraze/run_hover_test.py --workspace /twinr/bitcraze --json
python3 hardware/bitcraze/run_local_inspect_mission.py --repo-root /home/thh/twinr --workspace /twinr/bitcraze --bitcraze-python /twinr/bitcraze/.venv/bin/python --artifact-root /tmp/drone-artifacts --image-name inspect.png --json
python3 hardware/bitcraze/capture_runtime_telemetry.py --workspace /twinr/bitcraze --profile operator --json
python3 hardware/bitcraze/replay_hover_trace.py --report-json /tmp/hover-report.json --trace-file /tmp/hover-trace.jsonl --json
```

For operator recovery after a bad Crazyflie flash, use
[CRAZYFLIE_RECOVERY_RUNBOOK.md](./CRAZYFLIE_RECOVERY_RUNBOOK.md). The important
default is now explicit: if the `nRF` bootloader is alive, prefer the official
coordinated Bitcraze release ZIP over `Crazyradio` before escalating to direct
`SWD`/`JTAG` work.

There is now one proven narrower `nRF` exception to keep in mind. On the
current `Crazyflie 2.1 + AI deck + Flow v2 + Multi-ranger` stack, we proved one
state where:
- the `STM32` side was already correct
- the `Crazyradio` dongle was already fixed
- the `nRF` received valid packets
- but the radio path still returned no usable ACKs

For that exact state, the recovery that restored the normal radio path was:
- keep the official `nRF` bootloader/MBS/UICR intact
- rebuild the official `crazyflie2-nrf-firmware` tag `2025.12` with `BLE=0`
- flash only the `nRF` app image at `0x0001B000` over `SWD`

That exact sequence is documented in
[CRAZYFLIE_RECOVERY_RUNBOOK.md](./CRAZYFLIE_RECOVERY_RUNBOOK.md) under
`Verified nRF Radio ACK Failure Mode`.

There is now also one proven `STM32`-side exception that can look deceptively
similar from the host:

- raw `Crazyradio` ACKs are healthy
- `scan_interfaces()` still returns `radio://0/80/2M`
- but `protocol_version` stays `-1` and the platform handshake never completes

In that case the remaining blocker may be the `STM32` app hanging on the
`I2C1` deck bus before `commInit()`, not the radio path. The verified
reproduction in `2025.12.1` was an unbounded deck-bus unlock loop in
`i2cdrvdevUnlockBus()` while SDA stayed low. The repo-local bounded fail-closed
patch is:

- `hardware/bitcraze/patches/crazyflie_firmware_i2c1_boot_hang_fail_closed.patch`

The full recovery sequence and proof path are recorded in
`hardware/bitcraze/CRAZYFLIE_RECOVERY_RUNBOOK.md` under
`Verified STM32 Deck-Bus Boot Hang Failure Mode`.

The new non-rotor follow-up probe for that exact failure mode is
[probe_deck_bus.py](./probe_deck_bus.py). It does not spin motors. It captures:
- firmware `deck.*` flags after a clean `fully_connected` state
- explicit `cf.mem.refresh()` evidence from the memory TOC
- bounded firmware console output after boot

By default it also performs one bounded `STM32`/deck power-cycle first. That
is intentional: Bitcraze deck discovery is startup-only, so a hot-reconfigured
deck stack can otherwise look falsely missing even when the software path is
healthy.

That is the correct software-side truth surface before any physical deck-stack
isolation. If those three sources all show the deck stack missing, the
remaining narrowing step is physical: remove or reseat one deck or interconnect
at a time and rerun the same bounded probe until the missing evidence comes
back.

For deck bring-up and post-recovery acceptance, the runbook now also captures a
second important rule: do not treat deck LEDs or deck power as proof of deck
detection. The authoritative order is:
- firmware `deck.*` params
- firmware memory TOC after `cf.mem.refresh()`
- one clean bounded probe such as [probe_multiranger.py](./probe_multiranger.py)

If those disagree, trust the firmware-side evidence over visual cues and follow
the deck-enumeration diagnostic order in
[CRAZYFLIE_RECOVERY_RUNBOOK.md](./CRAZYFLIE_RECOVERY_RUNBOOK.md) before
concluding that the deck stack is electrically missing.

Related operator rule: after any physical deck-stack change, either fully
power-cycle the Crazyflie or use the default `probe_deck_bus.py` path that
power-cycles the `STM32` and decks before probing. Do not trust `deck.*` from
the already-running session after hot reattachment.

There is now one narrower Z-ranger software failure mode to keep in mind during
hover acceptance. The upstream `zranger2` driver used to ignore the VL53L1
`RangeStatus` field and wrote invalid readings straight into `range.zrange` as
`0 mm`. Under real takeoff vibration and bus activity that can look like a hard
floor lock even though the correct interpretation is "invalid reading". The
repo-local patch above converts those invalid readings into the normal invalid
sentinel path instead of silent zero-height truth.

Once the AI-Deck WiFi streamer is flashed, Twinr can treat it as a normal
still camera by setting `TWINR_CAMERA_DEVICE=aideck://192.168.4.1:5000`. On
hosts that already sit on the AI-Deck AP, the standard Twinr camera capture
and vision-request entrypoints read one bounded frame directly from the deck
stream. On single-WiFi hosts with `nmcli`, Twinr now performs one bounded
handover to the AI-Deck AP for the capture and then restores the previous WiFi
connection before the upstream OpenAI/runtime path continues.

When the Pi receives a DHCP lease on `192.168.4.x` and TCP connect to
`192.168.4.1:5000` succeeds but no bytes arrive, the remaining blocker is the
AI-Deck streamer itself rather than Twinr's network handover. In that state,
power-cycle the drone and prefer the Bitcraze station-mode setup over AP mode
for longer-running Twinr vision experiments.

For the incoming Olimex ARM-USB-TINY-H bundle, Twinr now has a dedicated host
prep script that can stage the JTAG recovery workspace, install missing
`openocd`/`docker` prerequisites, and ensure the runtime user is in the
expected serial/USB groups before the hardware is attached.

For the incoming Multi-ranger deck, Twinr now has a dedicated bounded probe
that checks `deck.bcMultiranger`, `deck.bcFlow2`, `deck.bcZRanger2`, and
samples `front/back/left/right/up/down` range data over the normal radio URI.
That gives us an immediate acceptance test as soon as the deck is mounted. Use
that probe together with the raw firmware `deck.*` values when diagnosing
strange deck states; one stale or host-disturbed read is not enough evidence to
declare the whole stack dead.

For the first live flight primitive, Twinr now also ships one bounded hover
worker that only does `takeoff -> hover -> land` after explicit deck and
battery preflight. On real hardware that worker is now mission-intent only: it
submits one bounded hover mission over Appchannel to `twinrFs`, then observes
and reports the resulting on-device state machine. The host no longer owns
per-tick takeoff/hover corrections over radio on the real aircraft. When the
Multi-ranger deck is present, the same preflight samples one short clearance
snapshot and blocks takeoff if front/back/left/right/up obstacles are already
too close for a safe hover envelope. Before the mission starts, the worker now
also applies and verifies a deterministic hover-time firmware setup:
- `stabilizer.estimator=2`
- `stabilizer.controller=1`
- `motion.disable=0`
- `kalman.resetEstimation` pulse and verification back to `0`

Takeoff is then gated behind a bounded Kalman-settling check that waits for
stable `kalman.varPX/PY/PZ`, quiet roll/pitch, usable `motion.squal`, and a
valid downward `range.zrange` reading. On the real aircraft, the subsequent
staged mission now runs on-device inside `twinrFs`. On the host, the old
stateful hover primitive remains only as the SITL/replay/shared-contract lane.

That split is deliberate. The real aircraft now proves its staged
`takeoff -> hover -> landing` contract next to the Crazyflie EKF/controller
path on the STM32. The host no longer tries to infer or correct those phases
over the radio link in real time.

The April 2026 on-device refactor also made the STM32 lane structurally
explicit instead of keeping one growing C file. `twinrFs` is now split into
protocol, telemetry, mission-control, failsafe-control, and state-machine
modules. The next SOC slice then pulled the remaining control arithmetic out of
mission orchestration as well: `twinrFs` now has dedicated pure-C
`vertical_control` and `disturbance_control` helpers for bounded hover-thrust
estimation, takeoff progress classification, and lateral disturbance
classification. The host stays limited to mission intent plus observation.

The April 2026 control-ownership change now makes one stricter split between
simulator and real aircraft. CrazySim still starts its bounded hover lane with
the host-side primitive because the simulator-owned height lane is the thing
under test there. Real hardware no longer does that. The bounded takeoff /
hover / landing mission state machine now lives in `twinrFs` on the STM32 and
receives only one bounded hover intent from the host. That keeps the critical
decisions next to the firmware EKF/controller path instead of on a slower
host-radio loop.

For live-flight forensics, the worker now records the on-device `twinrFs`
status surface rather than pretending to own the control loop. That gives each
hardware run one explicit state/reason trail from the firmware mission owner
instead of a second competing host-side bootstrap story.

The STM32 takeoff truth is now stricter as well. `twinrFs` no longer treats one
single good `range.zrange`/`motion.squal` sample as enough to hand off into
hover. It keeps short consecutive counters for range freshness, range rise,
flow liveness, and attitude quiet, exposes those counters in the `twinrFs`
log group, and times out with an explicit on-device reason when one of those
proofs never converges. The same log group now also exports the vertical
progress class plus filtered-battery and disturbance severity/recoverability
signals so a failed live hover can be classified from the firmware owner
without rebuilding controller state in Python.

That on-device truth lane now also classifies the next two real post-qualify
failure states instead of leaving them implicit in the trace:

- `truth_stale`: hover had already qualified, then range/flow truth went stale
  for the bounded on-device window
- `state_flapping`: takeoff-truth inputs regressed repeatedly before handoff
  and never converged into one stable qualified-hover state
- `ceiling_without_progress`: the vertical helper hit its bounded takeoff
  ceiling long enough without proving real range rise
- `disturbance_nonrecoverable`: the lateral disturbance helper kept observing a
  non-recoverable XY bias long enough that local hover recovery was no longer
  accepted
- `takeoff_overshoot`: the vertical helper saw strong bounded overshoot above
  the micro-liftoff target instead of a controlled hover handoff

For local regression, Twinr now also has one firmware-near harness for those
pure on-device helpers under
[../../test/fixtures/twinrfs_control_harness.c](../../test/fixtures/twinrfs_control_harness.c)
plus [../../test/test_bitcraze_twinrfs_control_harness.py](../../test/test_bitcraze_twinrfs_control_harness.py).
That harness compiles the extracted C modules directly and proves the vertical
handoff/ceiling-abort and lateral recover/abort contracts before the next
hardware flash.

The host preflight no longer trusts one arbitrary `pm.state` / supervisor
packet either. It now captures one bounded preflight status window, treats the
latest sample as the authoritative start state, and records any `power.state`,
`can_arm`, `can_fly`, or `is_armed` flapping explicitly as a blocked-preflight
reason. That removes the old single-sample randomness from repeated host-side
hover reruns.

There is one important grounded-start exception now documented in code as well
as in the worker behavior. Bitcraze documents the PMW3901 optical-flow sensor
as only tracking motion from about `80 mm`, and Bitcraze engineering guidance
for takeoff issues suggests ignoring flow below roughly `0.1 m`. The hover
worker therefore does not hard-block takeoff on `motion.squal` or lateral
Kalman variance while the downward range still says the craft is sitting in
that blind near-ground band. In that state it only requires the vertical range,
vertical variance, and bench-stable roll/pitch gates before allowing the
bounded takeoff primitive to lift into the flow sensor's usable range.

The worker also writes bounded in-flight telemetry for attitude, position and
velocity estimates, gyro motion, optical-flow quality, downward and directional
range sensing, battery sag, radio RSSI, thrust, and supervisor safety flags so
a successful run includes real stability evidence rather than only a pass/fail
outcome. The persisted artifact now also includes the verified pre-arm param
snapshot, the estimator-settle summary, and the primitive outcome in addition
to the in-air telemetry summary. That telemetry now comes from the canonical
shared runtime lane in `src/twinr/hardware/crazyflie_telemetry.py` rather than
from per-worker collectors, so hover, local inspect, and daemon state all read
the same bounded profile-driven runtime surface. `run_local_inspect_mission.py`
now also consumes the same explicit `runtime_mode` contract as
`run_hover_test.py` instead of carrying a separate implicit hardware-only path.
The Bitcraze scripts still run through `/twinr/bitcraze/.venv`, not the
repo-local `.venv`.

The landing side of the primitive is now staged as well instead of dropping
straight from hover to stop-setpoints: it first descends to a low near-ground
floor, briefly settles there, then switches to a slower touchdown stage,
briefly holds zero-height setpoints, and only then allows motor cutoff after a
deterministic landing-complete signal. Landing completion is now intentionally
joint: fresh downward `range.zrange` at or below the current `5 cm` cutoff and
a fresh supervisor `is flying = false` report must both be present for the
required confirmation window. This is deliberate: Bitcraze's
`send_stop_setpoint()` is a hard motor stop that can otherwise make the drone
fall, while range-only ground contact can still lag the firmware's internal
flight state enough to produce a short re-hop. Once airborne, the hover worker
therefore keeps driving the landing path until that joint ground truth arrives
or the bounded timeout forces a loud `touchdown_not_confirmed` outcome.
The explicit CrazySim `sitl` lane now keeps that hardware contract intact for
real flights while using one equally explicit simulator contract: touchdown may
complete from the simulator-owned height lane without waiting for
hardware-style supervisor grounded telemetry that stays high under repeated
`hover(z=0)` setpoints in SITL.

For timeout/debug forensics it also accepts `--trace-file` and streams phase
breadcrumbs such as `sync_connect`, `pre_arm_params`, `estimator_settle`,
`telemetry_stop`, `sync_disconnect`, `report_build`, and `json_emit` to a
JSONL file so teardown hangs still leave durable evidence. The persisted
telemetry summary uses the inferred airborne window instead of mixing in
pre-takeoff estimator-settle samples, and it drops invalid range sentinels
like `0` and `32766` so drift and clearance numbers better reflect the actual
hover. It now also distinguishes the structured cases `no_samples` versus
`raw_samples_missing_airborne_window` so a live run with captured telemetry but
missing airborne qualification no longer looks like a total telemetry outage.
That summary now also keeps two separate truth surfaces for optical flow:
airborne-window `flow_observed` and raw-run `raw_flow_observed`. This matters
when a failed takeoff never enters the airborne window even though `motion.squal`
was already nonzero in the raw telemetry. In the same report, the telemetry
summary now classifies the takeoff lateral lane separately through
`takeoff_lateral_classification` so we can distinguish:

- nonzero on-device lateral commands during `mission_takeoff`
- unexpected lateral command sources during `mission_takeoff`
- estimator-bias evidence without any commanded takeoff laterals
- fully raw takeoff drift with zero on-device lateral command and zero
  disturbance estimate

Hover acceptance is fail-closed as well: runs are marked unstable when
the measured altitude rises far above the requested hover height or when
battery sag under load drops below the bounded safety floor, even if the
telemetry stream itself stayed alive.

For host-loss safety, the hover path now has a second layer below Python: a
Crazyflie app-layer failsafe in [twinr_on_device_failsafe/](./twinr_on_device_failsafe)
runs directly on the STM32 in C with a `20 ms` local control loop. The host
only sends bounded Appchannel heartbeats plus thresholds through
[on_device_failsafe.py](./on_device_failsafe.py). If that heartbeat stops,
the on-device app can still locally brake, repel away from nearby obstacles,
descend, and only disarm after touchdown confirmation. This is the correct
language/runtime split for performance and determinism: the real emergency path
stays on the aircraft, Python only configures and observes it.

The important April 2026 takeoff lesson is that the on-device app must not arm
lateral `front/back/left/right` clearance just because the supervisor or
estimator momentarily claims flight while the craft is still on the floor.
`twinrFs` now treats valid downward `range.zrange` as the authoritative liftoff
proof and only arms lateral clearance once that measured height really exceeds
the near-ground floor. Pre-liftoff it still honors `up` clearance, but a failed
takeoff must now end in one quiet landed state instead of looping through
`failsafe_brake -> failsafe_descend -> touchdown_confirm -> monitoring` on the
ground.

There is now one host-side rule on top of that firmware change:
`run_hover_test.py` and `run_local_inspect_mission.py` start `twinrFs` with
lateral clearance explicitly disarmed and only arm it after the host primitive
has confirmed a real takeoff. The host preflight now follows the same contract:
while the craft is still below the shared active-takeoff height, it requires
`up` clearance but does not hard-block on side-facing `front/back/left/right`
ToF readings that can still see floor geometry near the ground. Once the shared
active height is reached, the normal lateral clearance gate becomes strict.

Hover acceptance is stricter now as well. A run is marked unstable not only on
overshoot or battery sag, but also when it never reaches the commanded
target-height gate or when the in-air telemetry shows excessive drift,
excessive roll/pitch, or excessive horizontal speed. The primitive now has an
explicit `stabilize -> hover_hold` split: it first waits for a bounded stable
window after takeoff, and if that live guard keeps seeing too much height
error, drift, speed, or attitude, it aborts early into landing instead of
continuing an obviously bad hover for seconds. A brief hop or eiernder ascent
no longer counts as a successful bounded hover.

That current hover lane should also be named precisely: it is
`flow_relative_hover`, not true absolute `position_hold`. The host sends
`send_hover_setpoint(vx=0, vy=0, yawrate=0, z=...)`, and the Crazyflie holds
height plus body-frame zero velocity using relative flow/dead-reckoning. That
is enough for bounded hover acceptance and short local inspect maneuvers, but
it is not the end architecture for "stand still at one exact point". Real
stationary position hold remains a separate positioning track that needs an
absolute pose source or a future visual-anchor lane.

Twinr now adds one bounded host-side outer-loop on top of that `flow_relative_hover`
lane: [crazyflie_flow_anchor.py](../../src/twinr/hardware/crazyflie_flow_anchor.py)
turns fresh `stateEstimate.{x,y,z,vx,vy}`, `stabilizer.yaw`, and downward range
into small corrective body-frame velocity targets plus a trusted-height signal.
This does not replace the Crazyflie inner controller. It only gives the host a
single, bounded way to resist lateral drift and to reject raw `range.zrange`
surface switches or ToF outliers as fake hover altitude.

Build and flash that firmware path before relying on bounded hover acceptance:

```bash
bash hardware/bitcraze/build_on_device_failsafe.sh --firmware-root /tmp/crazyflie-firmware --expected-firmware-revision 2025.12.1
/twinr/bitcraze/.venv/bin/python hardware/bitcraze/flash_on_device_failsafe.py --lane dev --device-role dev --attestation hardware/bitcraze/twinr_on_device_failsafe/build/twinr_on_device_failsafe.build-attestation.json
```

The generated build output stays under
`hardware/bitcraze/twinr_on_device_failsafe/build/`, but that tree is a local
firmware workspace only. It is not treated as authoritative source for Pi repo
mirroring and is intentionally excluded from the `/home/thh/twinr -> /twinr`
sync contract.

When extending the `twinrFs` param surface, keep Bitcraze's `CMD_GET_ITEM_V2`
TOC size limit in mind. For params and logs, `group + name + 2` must stay at
or below `26` bytes. Exceeding that budget can assert in `param_logic.c`
during boot and later fail the normal-start check through
`cfAssertNormalStartTest()`.

The post-flash helper reconnects over the normal radio URI and proves the app
through `twinrFs.protocolVersion`, `twinrFs.enable`, `twinrFs.state`, and
`twinrFs.reason`. The hover worker's default `--on-device-failsafe-mode required`
now blocks takeoff unless that proof is present.

If the `STM32` app hash is already correct but the post-flash reconnect still
dies with missing ACKs or `Too many packets lost`, debug the `nRF` radio path
before reflashing the same `STM32` image again. In the proven April 2026
incident, the remaining blocker after the `STM32` fix was the `nRF` ACK-return
path, and the decisive recovery was an official `2025.12` `nRF` rebuild with
`BLE=0`.

On top of that hover/failsafe lane, Twinr now has a first bounded local-inspect
mission worker in [run_local_inspect_mission.py](./run_local_inspect_mission.py).
It deliberately does not create a second flight stack. Instead it reuses the
same pre-arm params, estimator-settle gate, shared runtime telemetry, and
on-device failsafe proof as the hover worker, then adds only three
mission-specific layers:
- host-side bounded local-navigation planning in [local_navigation.py](./local_navigation.py)
- one short body-frame translation via the explicit stateful primitive in [hover_primitive.py](./hover_primitive.py)
- one bounded still capture through Twinr's normal camera abstraction before deterministic landing

`run_local_inspect_mission.py` now reuses the same live stabilize/hover guard
as `run_hover_test.py`, so local inspect can no longer skip directly from
takeoff to translation/capture on a visibly unstable hover state.

That same shared runtime telemetry lane also has one thin operator/daemon entry
point in [capture_runtime_telemetry.py](./capture_runtime_telemetry.py). It
starts one explicit profile such as `operator`, `hover_acceptance`,
`inspect_local_zone`, or `forensics`, then emits one bounded JSON snapshot that
the daemon can expose through `GET /state` without reimplementing cflib log
ownership itself.

Hover debugging now also has a sim-first lane. First capture the unsafe live
hover report and optional phase trace, then replay it through
[replay_hover_trace.py](./replay_hover_trace.py) so the same bounded hover
primitive, guard logic, and landing contract run deterministically without
touching live hardware. Only after that replay path proves the fix should the
same bug return to live flight.

`replay_hover_trace.py` now replays the real hardware start contract too:
- `--runtime-mode hardware` exercises the bounded raw-thrust/manual `micro_liftoff` and requires fresh `range.zrange` plus nonzero flow before hover-mode handoff
- `--runtime-mode sitl` keeps using the simulator-owned `stateEstimate.z` ground-distance contract and does not require flow quality

For closed-loop SITL beyond raw replay, Twinr now owns a strict CrazySim
adapter in `src/twinr/hardware/crazysim_adapter.py`. Twinr does not vendor
CrazySim or guess its workspace layout; the adapter accepts only an
operator-managed checkout with the expected CrazySim single-agent launch script
and fails closed otherwise.

On top of that baseline SITL lane, Twinr now also has a deterministic CrazySim
scenario runner in [run_hover_sim_scenarios.py](./run_hover_sim_scenarios.py).
It does not invent a second hover stack. Instead it:
- captures one real CrazySim hover baseline through [run_hover_sim.py](./run_hover_sim.py) or reuses one stored hover report
- mutates that baseline into bounded scenario artifacts such as `drift_bias`, `flow_dropout`, `zrange_outlier`, `attitude_spike`, `wall_proximity`, and transient recovery disturbances
- replays each scenario through the same bounded hover primitive and shared stability evaluation used by the live worker

That gives Twinr one repeatable sim-first regression lane for exactly the
classes of failures that were dangerous on real hardware: uncontrolled lateral
motion, untrusted flow, fake downward-range altitude jumps, close obstacle
approach, and bounded recovery after a transient shove. The recovery lane does
not pretend to prove full plant-physics restabilization. It proves the current
software contract instead: Twinr must issue the expected opposing correction
and return to bounded hover output within the declared recovery window. Use it
before the next live hover iteration:

```bash
./.venv/bin/python hardware/bitcraze/run_hover_sim_scenarios.py \
  --crazysim-root /tmp/crazysim \
  --backend mujoco \
  --python-bin /tmp/crazysim-cflib-venv/bin/python \
  --json \
  -- --height-m 0.25 --hover-duration-s 1.0
```

Or reuse a stored hover report without relaunching CrazySim:

```bash
./.venv/bin/python hardware/bitcraze/run_hover_sim_scenarios.py \
  --baseline-report-json /tmp/hover-report.json \
  --scenario transient_forward_drift_recovery \
  --scenario persistent_forward_drift_abort \
  --scenario transient_left_drift_recovery \
  --scenario transient_height_drop_recovery \
  --json
```

The transient recovery scenarios currently cover:
- `transient_forward_drift_recovery`: a bounded forward drift impulse must trigger a bounded backward command and then settle
- `persistent_forward_drift_abort`: the same forward axis, but sustained long enough that the hover guard must block on bounded instability instead of recovering
- `transient_left_drift_recovery`: a bounded lateral drift impulse must trigger the opposing lateral command and then settle
- `transient_height_drop_recovery`: a bounded height underestimate must raise the commanded hover height briefly and then settle

The scenario lane now separates two contracts explicitly:
- `recover`: the disturbance must produce the expected opposing command and must not drive the stability guard into `blocked`
- `abort`: the disturbance must trip the stability guard on declared failure codes instead of silently drifting into a bad post-flight grade

To make this a single acceptance lane instead of two separate manual steps,
Twinr now also has [run_hover_acceptance_gate.py](./run_hover_acceptance_gate.py).
It gates hover/guard work on both:
- one or more stored JSONL/report replay cases through the shared replay lane
- one deterministic CrazySim scenario suite against either a fresh or stored baseline
- one CrazySim physical disturbance suite, either fresh from MuJoCo or loaded from a stored suite JSON
- and, for fresh MuJoCo runs, repeated nominal baselines that must all pass the same touchdown contract before the gate can mark the lane `live_flight_eligible=true`

The gate fails closed if either side fails:

```bash
./.venv/bin/python hardware/bitcraze/run_hover_acceptance_gate.py \
  --replay-case "/tmp/unsafe-live-hover.json||hardware" \
  --crazysim-root /tmp/crazysim \
  --backend mujoco \
  --python-bin /tmp/crazysim-cflib-venv/bin/python \
  --json \
  -- --height-m 0.25 --hover-duration-s 1.0
```

For stored-only validation without relaunching CrazySim:

```bash
./.venv/bin/python hardware/bitcraze/run_hover_acceptance_gate.py \
  --replay-case "/tmp/unsafe-live-hover.json||hardware" \
  --baseline-report-json /tmp/crazysim-baseline-hover.json \
  --physical-suite-json /tmp/crazysim-physical-suite.json \
  --scenario baseline_nominal \
  --scenario persistent_forward_drift_abort \
  --scenario drift_bias \
  --scenario flow_dropout \
  --scenario attitude_spike \
  --scenario wall_proximity \
  --json
```

The acceptance gate is now explicit about its policy:
- replay proves that real unsafe flights stay reproducible as regressions
- the adversarial SITL suite proves guard behavior against telemetry/state mutations
- the physical disturbance suite proves recover-vs-abort behavior against live MuJoCo force/torque disturbances
- fresh nominal repeatability proves that CrazySim readiness and touchdown semantics are deterministic enough to trust the rest of the fresh gate

MuJoCo acceptance now runs headless by default. That is deliberate: the viewer
lane is useful for manual operator inspection, but repeated fresh regression
and disturbance suites proved less stable there because the GUI/GLFW startup
path can fail before the same hover stack even starts. Use
`--visualize --display :0` only for manual debugging, not as the default proof
lane.

If any of those three lanes fail, the gate stays closed and the next live hover should not happen yet.
Stored-only gate runs remain useful for regression work, but they intentionally do not mark the stack `live_flight_eligible=true`.

For recovery-specific acceptance on top of one fresh CrazySim baseline:

```bash
./.venv/bin/python hardware/bitcraze/run_hover_sim_scenarios.py \
  --crazysim-root /tmp/crazysim \
  --backend mujoco \
  --python-bin /tmp/crazysim-cflib-venv/bin/python \
  --scenario baseline_nominal \
  --scenario transient_forward_drift_recovery \
  --scenario persistent_forward_drift_abort \
  --scenario transient_left_drift_recovery \
  --scenario transient_height_drop_recovery \
  --json
```

For real MuJoCo plant disturbances instead of replay-only mutations, Twinr also
has [run_hover_sim_disturbances.py](./run_hover_sim_disturbances.py). That lane
injects explicit force/torque pulses into the live CrazySim plant and then
replays the resulting hover artifact through the same recover-vs-abort
contracts. Use it when the question is no longer "would the replay logic ask
for the right correction?" but "does the current hover stack still recover or
abort correctly when the simulated aircraft is actually shoved off balance?"

```bash
./.venv/bin/python hardware/bitcraze/run_hover_sim_disturbances.py \
  --crazysim-root /tmp/crazysim \
  --backend mujoco \
  --python-bin /tmp/crazysim-cflib-venv/bin/python \
  --json \
  -- --height-m 0.25 --hover-duration-s 1.0
```

The physical suite currently separates two categories on the same live plant:
- `physical_*_recovery`: bounded impulses that must trigger the expected opposing command and then resettle without a blocked hover guard
- `physical_*_abort`: stronger or sustained disturbances that must fail closed through the declared hover guard or takeoff abort path instead of drifting into an ambiguous outcome

Current default physical cases:
- `physical_forward_impulse_recovery`
- `physical_left_impulse_recovery`
- `physical_height_drop_recovery`
- `physical_persistent_forward_abort`
- `physical_roll_torque_abort`

The current local-navigation policy is intentionally conservative:
- it reads one short-range Multi-ranger clearance envelope
- it either chooses one small bounded `forward/left/right/back` translation or stays at the current hover anchor
- it prefers the lane with the largest bounded travel budget after enforcing the post-move clearance contract
- it never attempts open-ended navigation, mapping, or free flight

The custom firmware path itself is now split out into
[FIRMWARE_RELEASE_LANES.md](./FIRMWARE_RELEASE_LANES.md). That document is the
authoritative path for:
- pinned source builds
- bench promotion
- operator promotion
- strict separation between custom firmware and official recovery

Run the Bitcraze probes with `/twinr/bitcraze/.venv` rather than the repo-local
`.venv`. The Bitcraze stack currently wants a newer `numpy` line than Twinr, so
the isolated Bitcraze workspace remains the supported runtime path.

## Generated workspace

The setup script creates a reproducible runtime workspace under
`/twinr/bitcraze` containing:
- `.venv/` with pinned `cflib` and `cfclient`
- `firmware/` with staged UF2 assets and any captured `CURRENT.UF2` backups
- `README.md` with the exact pinned versions and the local probe commands

## See also

- [hardware/README.md](../README.md)
- [hardware/AGENTS.md](../AGENTS.md)
