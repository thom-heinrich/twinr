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
| [probe_multiranger.py](./probe_multiranger.py) | Connect to the Crazyflie, read the current deck flags, and sample Multi-ranger directions plus supporting Flow/Z-ranger presence for immediate post-install acceptance |
| [on_device_failsafe.py](./on_device_failsafe.py) | Minimal host-side Appchannel helper that proves the firmware app is loaded, sends heartbeat/config packets, and records returned failsafe status packets |
| [hover_primitive.py](./hover_primitive.py) | Internal helper module that owns deterministic hover pre-arm params, Kalman-settling gating, and the explicit bounded hover-setpoint primitive used by the operator-facing hover worker |
| [run_hover_test.py](./run_hover_test.py) | Run one bounded takeoff-hover-land flight test with deck, battery, and in-flight stability telemetry gates through the isolated Bitcraze workspace |
| [build_on_device_failsafe.sh](./build_on_device_failsafe.sh) | Build the Crazyflie STM32 on-device failsafe app as an out-of-tree firmware image against a Crazyflie firmware checkout |
| [flash_on_device_failsafe.py](./flash_on_device_failsafe.py) | Flash the built STM32 image through `cfloader` and verify the new `twinrFs` firmware app over the normal radio link |
| [twinr_on_device_failsafe/](./twinr_on_device_failsafe) | Crazyflie app-layer failsafe in C that keeps local safe-land and obstacle-repel behavior alive without the host |

## Usage

```bash
sudo ./hardware/bitcraze/setup_bitcraze.sh --workspace /twinr/bitcraze --runtime-user thh
python3 hardware/bitcraze/probe_crazyradio.py --workspace /twinr/bitcraze
python3 hardware/bitcraze/probe_crazyradio.py --workspace /twinr/bitcraze --json
sudo ./hardware/bitcraze/prepare_olimex_jtag.sh --workspace /twinr/bitcraze --install-apt --ensure-user-groups
/twinr/bitcraze/.venv/bin/python hardware/bitcraze/probe_multiranger.py --workspace /twinr/bitcraze --require-deck multiranger --json
/twinr/bitcraze/.venv/bin/python hardware/bitcraze/flash_on_device_failsafe.py --json
/twinr/bitcraze/.venv/bin/python hardware/bitcraze/run_hover_test.py --workspace /twinr/bitcraze --json
```

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
That gives us an immediate acceptance test as soon as the deck is mounted.

For the first live flight primitive, Twinr now also ships one bounded hover
worker that only does `takeoff -> hover -> land` after explicit deck and
battery preflight. When the Multi-ranger deck is present, the same preflight
samples one short clearance snapshot and blocks takeoff if front/back/left/
right/up obstacles are already too close for a safe hover envelope. Before the
first takeoff setpoint, the worker now also applies and verifies a
deterministic hover-time firmware setup:
- `stabilizer.estimator=2`
- `stabilizer.controller=1`
- `motion.disable=0`
- `kalman.resetEstimation` pulse and verification back to `0`

Takeoff is then gated behind a bounded Kalman-settling check that waits for
stable `kalman.varPX/PY/PZ`, quiet roll/pitch, usable `motion.squal`, and a
valid downward `range.zrange` reading. Only after that does the worker run the
explicit stateful hover-setpoint primitive from [hover_primitive.py](./hover_primitive.py),
which owns the `takeoff -> hold -> land` state machine and its abort/landing
path instead of relying on the more implicit `MotionCommander` context-manager
behavior.

The worker also writes bounded in-flight telemetry for attitude, position and
velocity estimates, gyro motion, optical-flow quality, downward and directional
range sensing, battery sag, radio RSSI, thrust, and supervisor safety flags so
a successful run includes real stability evidence rather than only a pass/fail
outcome. The persisted artifact now also includes the verified pre-arm param
snapshot, the estimator-settle summary, and the primitive outcome in addition
to the in-air telemetry summary. It lives beside the other Bitcraze scripts and
also runs through `/twinr/bitcraze/.venv`, not the repo-local `.venv`.

The landing side of the primitive is now staged as well instead of dropping
straight from hover to stop-setpoints: it first descends to a low near-ground
floor, briefly settles there, then switches to a slower touchdown stage,
briefly holds zero-height setpoints, and only then allows motor cutoff after a
deterministic landing-complete signal. The primary completion signal is fresh
downward `range.zrange` confirmation at or below the current `5 cm` cutoff for
three consecutive samples; the secondary completion signal is the firmware
supervisor no longer reporting `is flying`. This is deliberate: Bitcraze's
`send_stop_setpoint()` is a hard motor stop that can otherwise make the drone
fall. Once airborne, the hover worker therefore keeps driving the landing path
until one of those completion signals arrives instead of aborting the active
landing by exception.

For timeout/debug forensics it also accepts `--trace-file` and streams phase
breadcrumbs such as `sync_connect`, `pre_arm_params`, `estimator_settle`,
`telemetry_stop`, `sync_disconnect`, `report_build`, and `json_emit` to a
JSONL file so teardown hangs still leave durable evidence. The persisted
telemetry summary uses the inferred airborne window instead of mixing in
pre-takeoff estimator-settle samples, and it drops invalid range sentinels
like `0` and `32766` so drift and clearance numbers better reflect the actual
hover. Hover acceptance is fail-closed as well: runs are marked unstable when
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

Build and flash that firmware path before relying on bounded hover acceptance:

```bash
bash hardware/bitcraze/build_on_device_failsafe.sh
/twinr/bitcraze/.venv/bin/python hardware/bitcraze/flash_on_device_failsafe.py
```

The post-flash helper reconnects over the normal radio URI and proves the app
through `twinrFs.protocolVersion`, `twinrFs.enable`, `twinrFs.state`, and
`twinrFs.reason`. The hover worker's default `--on-device-failsafe-mode required`
now blocks takeoff unless that proof is present.

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
