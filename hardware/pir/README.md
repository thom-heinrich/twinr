# Twinr PIR Motion Sensor

Twinr now supports a dedicated PIR motion input as a first-class hardware path.

## Current prototype mapping

- Motion sensor output: `GPIO26`
- GPIO chip: `gpiochip0`
- Default polarity: `active_high=true`
- Default bias: `pull-down`

On a Raspberry Pi 4 header this is typically:

- `GPIO26` -> physical pin `37`

Some wiring guides or HAT silkscreens call this `IO26`. In Twinr that means BCM `GPIO26`.

## Persist or update the mapping

```bash
cd /twinr
hardware/pir/setup_pir.sh --motion 26 --probe
```

This writes the PIR settings into `/twinr/.env` and can run a short motion probe immediately after saving.

## Probe the PIR input

```bash
cd /twinr
python3 hardware/pir/probe_pir.py --env-file /twinr/.env --duration 30
```

Move in front of the PIR while the probe is running. The script prints the current level and every detected motion edge.

## Runtime integration

The Python helper in `src/twinr/hardware/pir.py` exposes:

- `build_pir_binding()` for config-to-GPIO binding
- `configured_pir_monitor()` for normal Twinr configuration
- `GpioPirMonitor` for direct GPIO sampling or motion waits

## Wiring guardrails

- PIR `OUT` -> `GPIO26` / physical pin `37`
- PIR `GND` -> any Pi `GND`
- PIR `VCC` -> use the supply voltage your PIR module expects
- The signal presented to the Pi GPIO must never exceed `3.3V`

If your PIR board is not explicitly GPIO-safe at `3.3V`, use a level shifter or a tested resistor divider before connecting the output to the Pi.
