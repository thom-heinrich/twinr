# display

`display` owns Twinr's e-paper runtime surface. It turns runtime snapshots and
health signals into short status cards and provides the Waveshare 4.2 V2 panel
adapter used by Twinr's Pi runtime.

## Responsibility

`display` owns:
- translate runtime snapshots into bounded e-paper status frames
- render and upload Waveshare 4.2 V2 images with validated driver settings
- discard cached vendor imports after hardware faults so the next render starts cleanly
- run the optional companion display loop beside hardware/runtime loops

`display` does **not** own:
- runtime snapshot storage or health sampling implementations
- GPIO/SPI bootstrap and vendor installation scripts in `hardware/display`
- top-level CLI parsing or loop-lock policy outside display-specific behavior

The generated Waveshare vendor files should live under `state/display/vendor/`
on the Pi so runtime deploy syncs do not wipe the driver package.

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Public display exports |
| [service.py](./service.py) | Snapshot-driven status loop |
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

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [hardware/display](../../../hardware/display/README.md)
