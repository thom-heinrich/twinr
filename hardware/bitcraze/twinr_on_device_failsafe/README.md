# Twinr On-Device Failsafe

This folder contains the Crazyflie STM32 app-layer failsafe that Twinr uses
for bounded hover safety when the host link disappears or the aircraft enters a
bad local state.

## Why this lives in C on the Crazyflie

The safety-critical loop has to keep running even when the host process, WiFi,
or radio link is gone. That is why the actual fallback controller lives on the
Crazyflie itself as a Crazyflie app-layer module in C:

- heartbeat-loss detection is local to the aircraft
- low-battery and clearance triggers are evaluated on the aircraft
- the safe-land path keeps running without Python or the daemon
- the control loop runs every `20 ms` on the STM32 instead of waiting on host I/O

Twinr's Python side only mirrors thresholds and sends bounded Appchannel
heartbeats while the host is healthy.

## Files

| File | Purpose |
|---|---|
| [src/twinr_on_device_failsafe.c](./src/twinr_on_device_failsafe.c) | Crazyflie app-layer failsafe controller and Appchannel contract |
| [Makefile](./Makefile) | Out-of-tree Crazyflie firmware build entrypoint |
| [app-config](./app-config) | Enables the app layer for the OOT build |
| [Kbuild](./Kbuild) | Top-level OOT build wiring |
| [src/Kbuild](./src/Kbuild) | App source registration for the Crazyflie firmware build |

## Build

Build against a Crazyflie firmware checkout:

```bash
bash hardware/bitcraze/build_on_device_failsafe.sh
```

The resulting STM32 image is written to:

```text
hardware/bitcraze/twinr_on_device_failsafe/build/twinr_on_device_failsafe.bin
```

## Flash and verify

Flash the image over the normal radio link and prove that the new `twinrFs`
param surface is present:

```bash
/twinr/bitcraze/.venv/bin/python hardware/bitcraze/flash_on_device_failsafe.py
```

That helper performs two steps:

1. flash the STM32 firmware image through `cfloader`
2. reconnect over the standard radio URI and read `twinrFs.*`

The first bounded hover path now requires that proof by default.
