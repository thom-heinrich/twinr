# display

Pi-side setup and smoke-test scripts for Twinr's display backends.

## Responsibility

`display` owns:
- persist HDMI Wayland/framebuffer or Waveshare env settings
- install and patch the Waveshare vendor driver when the legacy e-paper path is selected
- run bounded display smoke commands outside the main runtime

`display` does **not** own:
- runtime rendering logic in `src/twinr/display`
- main Twinr CLI orchestration
- unrelated GPIO, audio, or printer setup

## Key files

| File | Purpose |
|---|---|
| [setup_display.sh](./setup_display.sh) | Configure the active display backend and env wiring |
| [vendor_patch.py](./vendor_patch.py) | Download and harden the generated Waveshare vendor package |
| [display_test.py](./display_test.py) | Render one-shot test pattern on the configured backend |
| [run_display_loop.py](./run_display_loop.py) | Run the standalone status loop |
| [probe_busy_path.py](./probe_busy_path.py) | Instrument one bounded BUSY/RST/SPI full-refresh probe on the Pi for the Waveshare path |

## Usage

```bash
sudo ./hardware/display/setup_display.sh --env-file .env
sudo ./hardware/display/setup_display.sh --env-file .env --driver hdmi_wayland --layout debug_log --runtime-trace true
sudo ./hardware/display/setup_display.sh --env-file .env --driver hdmi_fbdev
python3 hardware/display/display_test.py --env-file .env
python3 hardware/display/run_display_loop.py --env-file .env --duration 30
python3 hardware/display/probe_busy_path.py --env-file .env --steady-renders 6
```

`setup_display.sh` now defaults to `hdmi_wayland`. It probes `/dev/fb0` for the
active mode, auto-detects the active Wayland socket when possible, writes the
Wayland/runtime env needed for the visible fullscreen Twinr surface, and for
HDMI also sets the senior-facing `default` layout with runtime tracing
disabled. Pass `--layout debug_log --runtime-trace true` when operators
explicitly want the richer live-diagnostics surface on the panel. HDMI setup
also removes the old Waveshare GPIO/SPI env keys so the runtime stops touching
the retired e-paper path.

`display_test.py` forwards bounded telemetry to stdout. On `hdmi_wayland` it
prints the resolved Wayland socket/runtime dir and mode; on `hdmi_fbdev` it
prints the resolved framebuffer path and mode; on Waveshare it still prints the
legacy driver and BUSY GPIO.

For manual HDMI face-trigger validation, write one short-lived cue artifact
into the configured cue path and let the running display loop pick it up on the
next cycle. Example:

```bash
PYTHONPATH=src python3 - <<'PY'
from twinr.config import TwinrConfig
from twinr.display import (
    DisplayFaceBrowStyle,
    DisplayFaceExpressionController,
    DisplayFaceGazeDirection,
    DisplayFaceMouthStyle,
)

config = TwinrConfig.from_env(".env")
controller = DisplayFaceExpressionController.from_config(config, default_source="camera_surface")
cue = controller.show(
    gaze=DisplayFaceGazeDirection.UP_RIGHT,
    brows=DisplayFaceBrowStyle.RAISED,
    mouth=DisplayFaceMouthStyle.SMILE,
    hold_seconds=5.0,
)
print(controller.store.path)
print(cue)
PY
```

`probe_busy_path.py` remains Waveshare-only: it instruments vendor `ReadBusy()`,
GPIO writes for `RST`/`PWR`, SPI command/bulk-write phases, and bounded supply
snapshots so Pi debugging can prove whether a hang occurred during `init()`,
`display()/TurnOnDisplay()`, or the recovery `Clear()` path.

For live runtime debugging, enable:

```dotenv
TWINR_DISPLAY_RUNTIME_TRACE_ENABLED=true
```

That keeps Twinr on the supervisor path but emits bounded `display_trace=...`,
`display_trace_gpio=...`, and `display_trace_supply=...` lines into the
service journal with phase, caller, last command, previous status, GPIO, and
supply context. Use the standalone probe when you need an exclusive panel
session; use the runtime trace when you need to catch the first stall inside
the normal supervisor-managed Twinr loop.

## See also

- [Top-level hardware README](../README.md)
- [Runtime display package](../../src/twinr/display/README.md)
