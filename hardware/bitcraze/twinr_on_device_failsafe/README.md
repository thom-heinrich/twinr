# Twinr On-Device Failsafe

This folder contains the Crazyflie STM32 app-layer hover/failsafe controller
that Twinr uses for real bounded hover missions and for fail-closed recovery
when the host link disappears or the aircraft enters a bad local state.

## Why this lives in C on the Crazyflie

The safety-critical loop has to keep running even when the host process, WiFi,
or radio link is gone. That is why the actual safety controller lives on the
Crazyflie itself as a Crazyflie app-layer module in C:

- heartbeat-loss detection is local to the aircraft
- low-battery and clearance triggers are evaluated on the aircraft
- bounded `takeoff -> hover -> landing` mission ownership is local to the aircraft
- the safe-land path keeps running without Python or the daemon
- the control loop runs every `20 ms` on the STM32 instead of waiting on host I/O

Twinr's Python side only mirrors thresholds, sends bounded Appchannel
heartbeats, submits one bounded hover intent, and observes returned mission
status while the host is healthy.

There is one grounded-start rule now proven by live telemetry and enforced in
the firmware: valid downward `range.zrange` is the authoritative liftoff proof.
The app does not arm lateral `front/back/left/right` clearance until the craft
has actually left the floor, because supervisor/estimator state alone can jump
early during a takeoff attempt. If a pre-liftoff abort still happens, the app
must settle into one quiet landed state and stay there until a fresh session or
explicit re-enable instead of retriggering a ground-level failsafe loop.
An active bounded-hover mission also stays inside its own takeoff state machine
while the craft is still grounded; it must not fall back into the generic
ground-monitoring branch before `runMissionControl()` has either proven lift or
rejected the mission locally.

The host now participates in that same contract explicitly. It starts the
heartbeat session, submits one bounded hover intent, and only performs the one
allowed in-flight config change: arm lateral clearance for the same session
after takeoff is proven. `twinrFs` still rejects all other in-flight
reconfiguration and any in-flight disable request.

## Control ownership

There is now one live hardware control lane:

- host Python sends mission intent and records status
- `twinrFs` on the STM32 owns the bounded mission state machine
- the host no longer sends per-tick takeoff/hover corrections over radio

That split is deliberate. Simulator/replay helpers may still exercise shared
contracts off-device, but the real aircraft's critical takeoff/hover/landing
path lives here in C.

## Source layout

The app is no longer a single monolith. The STM32 lane is split by concern:

- [src/twinr_on_device_failsafe.c](./src/twinr_on_device_failsafe.c): app entrypoint, sensor/log binding, log/param surface
- [src/twinr_on_device_failsafe_state_machine.c](./src/twinr_on_device_failsafe_state_machine.c): one bounded on-device mission/failsafe state machine step
- [src/twinr_on_device_failsafe_protocol.c](./src/twinr_on_device_failsafe_protocol.c): Appchannel packet parsing, session binding, mission start/abort/land commands
- [src/twinr_on_device_failsafe_mission_control.c](./src/twinr_on_device_failsafe_mission_control.c): bounded phase orchestration that consumes the extracted vertical/disturbance decisions
- [src/twinr_on_device_failsafe_vertical_control.c](./src/twinr_on_device_failsafe_vertical_control.c): pure C vertical bootstrap and hover-thrust-estimate helper
- [src/twinr_on_device_failsafe_disturbance_control.c](./src/twinr_on_device_failsafe_disturbance_control.c): pure C bounded lateral disturbance observer
- [src/twinr_on_device_failsafe_failsafe_control.c](./src/twinr_on_device_failsafe_failsafe_control.c): local brake/descend/touchdown failsafe lane
- [src/twinr_on_device_failsafe_telemetry.c](./src/twinr_on_device_failsafe_telemetry.c): status packets, trigger/landed events, shared range/battery helpers
- [src/twinr_on_device_failsafe_internal.h](./src/twinr_on_device_failsafe_internal.h): one shared internal contract for the module set

## Files

| File | Purpose |
|---|---|
| [src/twinr_on_device_failsafe.c](./src/twinr_on_device_failsafe.c) | Crazyflie app-layer entrypoint and exported log/param surface |
| [Makefile](./Makefile) | Out-of-tree Crazyflie firmware build entrypoint |
| [app-config](./app-config) | Enables the app layer for the OOT build |
| [Kbuild](./Kbuild) | Top-level OOT build wiring |
| [src/Kbuild](./src/Kbuild) | App source registration for the Crazyflie firmware build |

## Protocol surface

`twinrFs` now advertises protocol version `4` on the Crazyflie param surface.
The host heartbeat packet remains intentionally small and backward-compatible,
but hover mission commands and status packets use the v4 contract. The status
packet now carries enough bounded debug truth to explain live mission failures
without guessing:

- mission flags
- takeoff/landing debug flags
- target and commanded height
- EKF `stateEstimate.z`
- upward clearance
- `motion.squal`
- mission touchdown confirmation count
- current hover-thrust estimate in permille

That richer status contract exists so the host can observe the on-device
controller without re-implementing it. The stricter takeoff-truth debounce
counters stay out of the fixed 30-byte Appchannel packet on purpose; they are
exported through the `twinrFs` log group as `tkRfCnt`, `tkRrCnt`, `tkFlCnt`,
`tkAtCnt`, `tkStCnt`, `tkFpCnt`, `tkPgCls`, `tkBatMv`, `cmdVx`, `cmdVy`,
`cmdSrc`, `distSev`, and `distRec` so live forensics can still see exactly how
close the STM32 was to proving range freshness, range rise, flow liveness, and
attitude quiet, what vertical progress class the bootstrap is in, what filtered
battery value the vertical helper is using, and whether any lateral drift came
from an explicit on-device command lane or from motion outside the commanded
takeoff contract.

Takeoff qualification is also no longer one-sample permissive. The STM32 now
requires short consecutive windows for:

- valid downward range freshness
- downward-range rise above the grounded baseline
- usable flow/squal above the flow gate
- quiet roll/pitch before hover handoff

After hover is qualified, the STM32 stays fail-closed on truth loss:

- `truth_stale` means qualified hover lost downward-range or flow truth for the
  bounded stale window
- `state_flapping` means the takeoff-truth inputs regressed repeatedly before
  handoff and never converged into one stable qualified-hover state
- `ceiling_without_progress` means the vertical helper hit its bounded thrust
  ceiling for the configured window without proving enough range rise
- `disturbance_nonrecoverable` means the lateral disturbance helper observed a
  persistent XY bias outside the recoverable envelope for too many consecutive
  samples
- `takeoff_overshoot` means the bootstrap overshot well above the micro-liftoff
  target instead of handing off in a controlled envelope

## Build

Build against a pinned Crazyflie firmware checkout:

```bash
bash hardware/bitcraze/build_on_device_failsafe.sh \
  --firmware-root /tmp/crazyflie-firmware \
  --expected-firmware-revision 2025.12.1
```

The resulting STM32 image and build attestation are written to:

```text
hardware/bitcraze/twinr_on_device_failsafe/build/twinr_on_device_failsafe.bin
hardware/bitcraze/twinr_on_device_failsafe/build/twinr_on_device_failsafe.build-attestation.json
```

## Local control-harness proof

The extracted pure helpers also have one firmware-near harness in
[../../test/fixtures/twinrfs_control_harness.c](../../test/fixtures/twinrfs_control_harness.c).
That harness compiles the same C modules the STM32 build uses and drives four
stepwise scenarios without starting the full Crazyflie firmware image:

- `vertical_handoff`
- `vertical_ceiling_abort`
- `disturbance_recoverable`
- `disturbance_nonrecoverable`

Run the corresponding regression with:

```bash
PYTHONPATH=src ./.venv/bin/pytest test/test_bitcraze_twinrfs_control_harness.py
```

This does not replace flash or live proof. It exists so control-law regressions
in the extracted pure modules fail locally before the next hardware run.

## Param/Log naming guardrail

Bitcraze `2025.12.1` enforces the `CMD_GET_ITEM_V2` TOC size limit during boot.
For `PARAM_*` and `LOG_*` entries this effectively caps:

```text
strlen(group) + strlen(name) + 2 <= 26
```

If an out-of-tree app exceeds that budget, the firmware can assert in
`param_logic.c` during startup and the next boot will fail the normal-start
check through `cfAssertNormalStartTest()`. Keep `twinrFs` group entries within
that bound when adding new params or logs.

## Flash and verify

Flash the image over the normal radio link and prove that the new `twinrFs`
param surface is present:

```bash
/twinr/bitcraze/.venv/bin/python hardware/bitcraze/flash_on_device_failsafe.py \
  --lane dev \
  --device-role dev \
  --attestation hardware/bitcraze/twinr_on_device_failsafe/build/twinr_on_device_failsafe.build-attestation.json
```

That helper now performs three checks:

1. verify the requested firmware lane and device role
2. flash the STM32 firmware image through `cfloader`
3. reconnect over the standard radio URI and read `twinrFs.*`

That reconnect proof depends on a healthy `nRF` radio path. In the April 2026
incident, the `STM32` image was already correct and the remaining blocker was an
`nRF` radio ACK failure. The decisive recovery there was not another `STM32`
reflash but an official `crazyflie2-nrf-firmware` `2025.12` rebuild with
`BLE=0`, flashed only into the `nRF` app region at `0x0001B000`. If a future
post-flash verification fails while SWD hash checks already prove the `STM32`
image is correct, switch to the `nRF` radio-path diagnosis in
[../CRAZYFLIE_RECOVERY_RUNBOOK.md](../CRAZYFLIE_RECOVERY_RUNBOOK.md) instead of
looping the same `STM32` flash again.

Operator and bench promotion are documented in
[../FIRMWARE_RELEASE_LANES.md](../FIRMWARE_RELEASE_LANES.md). The first bounded
hover path now requires that proof by default.
