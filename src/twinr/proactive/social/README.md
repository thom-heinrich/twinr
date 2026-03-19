# social

Social-trigger models, sensing adapters, prompt routing, and visual review for
Twinr's proactive monitor.

This package turns normalized camera, ambient-audio, and PIR-derived state into
scored proactive trigger candidates and optional visual second opinions.

## Responsibility

`social` owns:
- Define the social-trigger domain models and score thresholds
- Evaluate stateful trigger candidates from normalized observation ticks
- Wrap ambient-audio, ReSpeaker XVF3800, legacy OpenAI vision, and local-first AI-camera observations into bounded conservative snapshots
- Preserve conservative ReSpeaker facts such as `assistant_output_active`, `direction_confidence`, `speech_overlap_likely`, and `barge_in_detected`
- Stabilize automation-facing camera snapshots, including person-count/zone anchors, coarse motion, and coarse/fine gesture event surfaces
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
| `camera_surface.py` | Debounced camera snapshot, person-count/zone anchor, coarse motion, and rising-edge coarse/fine gesture event surface |
| `engine.py` | Stateful social-trigger scoring engine and normalized vision contract |
| `local_camera_provider.py` | Maps the local IMX500 + MediaPipe adapter onto the social vision contract, including motion plus coarse/fine gesture output |
| `observers.py` | Audio, ReSpeaker overlay, and vision observation providers |
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
