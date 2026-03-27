# service_impl

Internal implementation package behind [`service.py`](../service.py).

No migration is required. The public import path stays
`twinr.proactive.runtime.service`; the wrapper now delegates into this package.

## Responsibilities

- `compat.py`: legacy helper functions, constants, and telemetry-safe emit helpers
- `coordinator_core.py`: dependency wiring plus the main proactive tick/orchestration path
- `coordinator_display.py`: dedicated HDMI attention and gesture refresh workflows
- `coordinator_observation.py`: stable observation-mixin facade that composes the focused observation helper modules
- `coordinator_observation_display.py`: automation-dispatch, HDMI display, gesture, and camera-surface helper bridges
- `coordinator_observation_facts.py`: automation fact payload assembly and rising-edge sensor-event derivation
- `coordinator.py`: final `ProactiveCoordinator` composition
- `monitor.py`: background worker lifecycle wrapper
- `builder.py`: default production assembly with wrapper-supplied dependencies for patch compatibility

## Compatibility

- `service.py` remains the stable import surface and compatibility shim.
- `build_default_proactive_monitor()` still accepts the same arguments and keeps wrapper-level monkeypatch points working.
- `ProactiveCoordinator`, `ProactiveMonitorService`, and `ProactiveTickResult` remain available from the legacy module path.
