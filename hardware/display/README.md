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
| [run_visual_qc.py](./run_visual_qc.py) | Capture the visible HDMI scene set, diff it, and persist a report-backed artifact bundle |
| [probe_busy_path.py](./probe_busy_path.py) | Instrument one bounded BUSY/RST/SPI full-refresh probe on the Pi for the Waveshare path |

## Usage

```bash
sudo ./hardware/display/setup_display.sh --env-file .env
sudo ./hardware/display/setup_display.sh --env-file .env --driver hdmi_wayland --layout debug_log --runtime-trace true
sudo ./hardware/display/setup_display.sh --env-file .env --driver hdmi_fbdev
python3 hardware/display/display_test.py --env-file .env
python3 hardware/display/run_display_loop.py --env-file .env --duration 30
python3 hardware/display/run_visual_qc.py --env-file .env
python3 hardware/display/probe_busy_path.py --env-file .env --steady-renders 6
```

`setup_display.sh` now defaults to `hdmi_wayland`. It probes `/dev/fb0` for the
active mode, auto-detects the active Wayland socket when possible, writes the
Wayland/runtime env needed for the visible fullscreen Twinr surface, and for
HDMI also sets the senior-facing `default` layout with runtime tracing
disabled. For HDMI it also installs `fonts-noto-color-emoji`, because the
right-hand Unicode emoji reserve depends on the system
`NotoColorEmoji.ttf` font. Pass `--layout debug_log --runtime-trace true` when operators
explicitly want the richer live-diagnostics surface on the panel. HDMI setup
also removes the old Waveshare GPIO/SPI env keys so the runtime stops touching
the retired e-paper path.

`display_test.py` forwards bounded telemetry to stdout. On `hdmi_wayland` it
prints the resolved Wayland socket/runtime dir and mode; on `hdmi_fbdev` it
prints the resolved framebuffer path and mode; on Waveshare it still prints the
legacy driver and BUSY GPIO.

`run_visual_qc.py` is the screenshot-backed HDMI acceptance path. It:
- drives the public face-expression and presentation controllers through a bounded scene set
- captures the visible Wayland surface with `grim`
- computes transition diff heatmaps and changed-pixel metrics
- writes a durable report under `artifacts/reports/report/<RPT...>/` when the
  repo report tool is usable, otherwise under
  `artifacts/reports/display_visual_qc/<RUN_ID>/`

The current scene set covers `idle_home`, `face_react`, `presentation_mid`,
`presentation_focused`, and `restored_home`, so Twinr's senior-facing screen
can be validated with concrete before/mid/focused/after evidence instead of
manual screenshots.

For the optional HDMI news ticker, enable it in `.env`. The runtime display
loop then reads from Twinr's active world-intelligence subscriptions and
fetches/cache headlines asynchronously:

```dotenv
TWINR_DISPLAY_NEWS_TICKER_ENABLED=true
TWINR_DISPLAY_NEWS_TICKER_STORE_PATH=artifacts/stores/ops/display_news_ticker.json
TWINR_DISPLAY_NEWS_TICKER_REFRESH_INTERVAL_S=900
TWINR_DISPLAY_NEWS_TICKER_ROTATION_INTERVAL_S=12
TWINR_DISPLAY_NEWS_TICKER_MAX_ITEMS=12
TWINR_DISPLAY_NEWS_TICKER_TIMEOUT_S=4
```

The ticker bar only appears on HDMI default surfaces, hides during fullscreen
presentations, and uses cached fallback text when a fresh feed download is not
yet available.

Older Pi `.env` files may still contain `TWINR_DISPLAY_NEWS_TICKER_FEED_URLS`.
Twinr imports that legacy list one-way into the shared world-intelligence
subscription pool when the pool is still empty, then keeps reading from the
shared pool instead of from a separate static ticker list.

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

For the full screenshot-backed QC bundle:

```bash
python3 hardware/display/run_visual_qc.py --env-file .env --keep-workdir
```

That command prints the generated `report_id`, the report markdown path, the
imported report assets, and per-transition diff metrics.

For manual HDMI presentation validation, use the producer-facing presentation
controller instead of hand-writing raw JSON. Example:

```bash
PYTHONPATH=src python3 - <<'PY'
from twinr.config import TwinrConfig
from twinr.display import DisplayPresentationController

config = TwinrConfig.from_env(".env")
controller = DisplayPresentationController.from_config(config, default_source="operator")
cue = controller.show_rich_card(
    title="Family call",
    subtitle="Marta is waiting",
    body_lines=("Press the green button to answer.",),
    accent="warm",
    hold_seconds=15.0,
)
print(controller.store.path)
print(cue)
PY
```

For fullscreen image validation:

```bash
PYTHONPATH=src python3 - <<'PY'
from twinr.config import TwinrConfig
from twinr.display import DisplayPresentationController

config = TwinrConfig.from_env(".env")
controller = DisplayPresentationController.from_config(config, default_source="operator")
cue = controller.show_image(
    image_path="/tmp/family_photo.png",
    title="New photo",
    subtitle="Delivered just now",
    body_lines=("The card should morph to fullscreen.",),
    accent="success",
    hold_seconds=20.0,
)
print(controller.store.path)
print(cue)
PY
```

For prioritized multi-card scene validation:

```bash
PYTHONPATH=src python3 - <<'PY'
from twinr.config import TwinrConfig
from twinr.display import DisplayPresentationCardCue, DisplayPresentationController

config = TwinrConfig.from_env(".env")
controller = DisplayPresentationController.from_config(config, default_source="operator")
cue = controller.show_scene(
    cards=(
        DisplayPresentationCardCue(key="summary", title="Daily summary", priority=20, accent="info"),
        DisplayPresentationCardCue(
            key="family_photo",
            kind="image",
            title="Family photo",
            image_path="/tmp/family_photo.png",
            priority=90,
            accent="warm",
            face_emotion="happy",
        ),
    ),
    hold_seconds=20.0,
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
