# social

Social-trigger models, sensing adapters, prompt routing, and visual review for
Twinr's proactive monitor.

This package turns normalized camera, ambient-audio, and PIR-derived state into
scored proactive trigger candidates and optional visual second opinions.

## Responsibility

`social` owns:
- Define the social-trigger domain models and score thresholds
- Evaluate stateful trigger candidates from normalized observation ticks
- Wrap ambient-audio and camera observations into bounded conservative snapshots
- Stabilize automation-facing camera snapshots, person-count/zone anchors, and event surfaces
- Route safety prompts and render bounded evidence facts for proactive prompting
- Buffer recent camera frames and request conservative visual second opinions

`social` does **not** own:
- Proactive monitor orchestration or worker lifecycle
- Delivery-governance cooldown policy after a candidate is emitted
- Raw hardware driver implementations outside runtime-facing wrappers
- Speech delivery, printing, or long-term proactive planning

## Key files

| File | Purpose |
|---|---|
| `__init__.py` | Package export surface |
| `camera_surface.py` | Debounced camera snapshot, person-count/zone anchor, and rising-edge event surface |
| `engine.py` | Stateful social-trigger scoring engine |
| `observers.py` | Audio and vision observation providers |
| `prompting.py` | Prompt routing and evidence rendering |
| `scoring.py` | Weighted scoring primitives |
| `vision_review.py` | Buffered visual second-opinion reviewer |
| `component.yaml` | Structured package metadata |
| `AGENTS.md` | Local editing rules |

## Usage

```python
from twinr.proactive.social import SocialObservation, SocialTriggerEngine

engine = SocialTriggerEngine.from_config(config)
decision = engine.observe(
    SocialObservation(
        observed_at=now,
        inspected=True,
    )
)
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [../runtime/README.md](../runtime/README.md)
- [../governance/README.md](../governance/README.md)
