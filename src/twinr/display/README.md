# display

`display` owns Twinr's e-paper runtime surface. It turns runtime snapshots and
health signals into short status cards and provides the Waveshare 4.2 V2 panel
adapter used by Twinr's Pi runtime.

## Responsibility

`display` owns:
- translate runtime snapshots into bounded e-paper status frames
- compose panel-bounded layout variants such as the operator-facing `debug_log`
  view with grouped `System Log`, `LLM Log`, and `Hardware Log` sections
- derive the debug-log sections from persisted ops events, usage telemetry,
  host health, and the remote-memory watchdog artifact
- persist an authoritative display heartbeat so ops health and the runtime
  supervisor can detect a hung companion thread instead of trusting lock
  ownership alone
- expose one shared heartbeat contract for display writes, ops companion-health
  assessment, and runtime-supervisor progress checks so those three paths
  evaluate the same signal instead of drifting apart
- tolerate bounded in-flight panel renders as healthy progress so supervision
  does not kill a live companion just because one Waveshare refresh runs longer
  than the idle heartbeat budget
- render and upload Waveshare 4.2 V2 images with validated driver settings
- prefer stable two-plane refresh paths on the Waveshare 4.2 V2 panel so live
  status animations do not invert panel polarity
- keep debug-log lines stable between real state changes so operator screens do
  not churn the e-paper panel every poll cycle
- keep debug-log hardware and ChonkyDB summary lines semantically current but
  bucketed/stable enough for e-paper refresh budgets
- discard cached vendor imports after hardware faults so the next render starts cleanly
- run the optional companion display loop beside hardware/runtime loops

`display` does **not** own:
- runtime snapshot storage or health sampling implementations
- GPIO/SPI bootstrap and vendor installation scripts in `hardware/display`
- top-level CLI parsing or loop-lock policy outside display-specific behavior

The generated Waveshare vendor files should live under `state/display/vendor/`
on the Pi so runtime deploy syncs do not wipe the driver package. The
authoritative display heartbeat lives separately under
`artifacts/stores/ops/display_heartbeat.json` so the unprivileged runtime can
always refresh it even when the vendor directory is root-owned.

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Public display exports |
| [debug_log.py](./debug_log.py) | Build grouped operator log sections from ops/usage stores |
| [heartbeat.py](./heartbeat.py) | Persist display forward-progress heartbeats and expose the shared companion-health contract for ops/supervision |
| [service.py](./service.py) | Snapshot-driven status loop |
| [layouts.py](./layouts.py) | Status-card layout composition |
| [waveshare_v2.py](./waveshare_v2.py) | Panel adapter and rendering |
| [companion.py](./companion.py) | Optional sidecar loop |
| [component.yaml](./component.yaml) | Structured package metadata |

## Usage

```python
from twinr.display import TwinrStatusDisplayLoop, WaveshareEPD4In2V2

display = WaveshareEPD4In2V2.from_config(config)
display.show_test_pattern()

loop = TwinrStatusDisplayLoop.from_config(config)
loop.run(duration_s=15)
```

```python
from twinr.display.companion import optional_display_companion

with optional_display_companion(config, enabled=True):
    realtime_loop.run(duration_s=15)
```

To activate the debug layout on the Pi/operator runtime, set:

```dotenv
TWINR_DISPLAY_LAYOUT=debug_log
```

`debug_log` removes the face entirely and uses the full screen for grouped
`System Log`, `LLM Log`, and `Hardware Log` sections sourced from Twinr's
runtime snapshot, ops event store, usage store, host health snapshot, and
remote ChonkyDB watchdog artifact. The legacy env value `debug_face` is kept
as a compatibility alias and normalizes to `debug_log`.

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [hardware/display](../../../hardware/display/README.md)
