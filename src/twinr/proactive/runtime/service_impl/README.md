# service_impl

Internal implementation package behind [`service.py`](../service.py).

No migration is required. The public import path stays
`twinr.proactive.runtime.service`; the wrapper now delegates into this package.

## Responsibilities

- `../perception_orchestrator.py`: shared runtime perception contract that resolves the single authoritative attention/gesture snapshot consumed by the service implementation
- `compat.py`: legacy helper functions, constants, and telemetry-safe emit helpers
- `coordinator_core.py`: dependency wiring plus the main proactive tick/orchestration path
- `coordinator_display.py`: dedicated HDMI attention and gesture refresh workflows that consume the shared runtime perception orchestrator instead of direct lane-local temporal policy
- `coordinator_perception.py`: shared display-refresh cycle state that reuses one combined `perception_stream` snapshot when display lanes can safely share it, while dedicated gesture-fast refresh can still bypass the shared capture to preserve low-latency HDMI acknowledgement
- `coordinator_observation.py`: stable observation-mixin facade that composes the focused observation helper modules
- `coordinator_observation_display.py`: automation-dispatch, HDMI display, gesture, and camera-surface helper bridges
- `coordinator_observation_facts.py`: automation fact payload assembly and rising-edge sensor-event derivation, reusing the same runtime attention orchestrator that drives HDMI and servo
- `coordinator.py`: final `ProactiveCoordinator` composition
- `monitor.py`: background worker lifecycle wrapper that now keeps shared attention/gesture perception deadlines anchored to the same loop start under Pi overrun so the productive combined capture path does not silently dephase away
- `builder.py`: default production assembly with wrapper-supplied dependencies for patch compatibility

## Compatibility

- `service.py` remains the stable import surface and compatibility shim.
- `build_default_proactive_monitor()` still accepts the same arguments and keeps wrapper-level monkeypatch points working.
- `ProactiveCoordinator`, `ProactiveMonitorService`, and `ProactiveTickResult` remain available from the legacy module path.
