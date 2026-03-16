# runtime

Proactive monitor orchestration and wakeword presence-session control for Twinr.

This package wires the runtime-facing monitor loop, degraded sensor handling,
and presence-session state used to arm wakeword listening and proactive checks.

## Responsibility

`runtime` owns:
- Coordinate proactive monitor ticks across PIR, camera, ambient audio, and wakeword policy paths
- Maintain wakeword presence-session state from recent sensor activity
- Export normalized observation facts and ops telemetry from proactive monitoring
- Open and close the bounded proactive monitor worker and its resources

`runtime` does **not** own:
- Score social triggers or define proactive prompt content
- Implement wakeword matching, calibration, or evaluation algorithms
- Implement raw camera, PIR, or microphone adapters
- Enforce delivery cooldown policy after a trigger becomes a candidate

## Key files

| File | Purpose |
|---|---|
| `__init__.py` | Package export surface |
| `presence.py` | Presence-session state machine |
| `service.py` | Monitor orchestration and lifecycle |
| `component.yaml` | Structured package metadata |
| `AGENTS.md` | Local editing rules |

## Usage

```python
from twinr.proactive.runtime import build_default_proactive_monitor

monitor = build_default_proactive_monitor(
    config=config,
    runtime=runtime,
    backend=backend,
    camera=camera,
    camera_lock=camera_lock,
    audio_lock=audio_lock,
    trigger_handler=trigger_handler,
)
if monitor is not None:
    with monitor:
        ...
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [../governance/README.md](../governance/README.md)
- [../social/README.md](../social/README.md)
- [../wakeword/README.md](../wakeword/README.md)
