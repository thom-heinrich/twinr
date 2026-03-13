# Twinr Buttons

Current Raspberry Pi inspection shows that the physical Twinr buttons are not exposed as dedicated Linux input devices. They need to be handled as GPIO lines on `gpiochip0`.

## Current state

- No dedicated button devices are present under `/proc/bus/input/devices`
- The accessible GPIO controller is `gpiochip0`
- The current mapping is `green=GPIO23` and `yellow=GPIO22`

## Persist or update the mapping

```bash
cd /twinr
hardware/buttons/setup_buttons.sh --green 23 --yellow 22
```

This script writes the button mapping into `/twinr/.env` and can run a short configured probe when `--probe` is added.

## Probe the buttons

```bash
cd /twinr
python3 hardware/buttons/probe_buttons.py --env-file /twinr/.env --configured --duration 15
```

## Runtime integration

The Python helper in `src/twinr/hardware/buttons.py` exposes:

- `build_button_bindings()` for mapping config to green/yellow buttons
- `configured_button_monitor()` for the main Twinr event loop
- `GpioButtonMonitor` for direct GPIO polling and edge handling
