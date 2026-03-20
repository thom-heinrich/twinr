# event_fusion

Short-window multimodal event fusion for Twinr's proactive sensing stack.

## Responsibility

`event_fusion` owns:
- keep RAM-only rolling buffers for recent audio and vision history
- derive bounded audio micro-events from normalized audio observations
- derive short temporal vision sequences from recent camera observations
- assemble conservative fused event claims with explicit delivery gates
- plan compact keyframe-review candidates from fused evidence windows

`event_fusion` does **not** own:
- proactive monitor orchestration or worker lifecycle
- raw camera, microphone, or PIR adapter implementations
- social-trigger prompt wording or delivery cooldown policy
- emotion, wellbeing, or diagnosis claims

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Package export surface |
| [buffers.py](./buffers.py) | RAM-only rolling buffer primitives |
| [audio_events.py](./audio_events.py) | Audio micro-event derivation |
| [vision_sequences.py](./vision_sequences.py) | Temporal vision-sequence derivation |
| [claims.py](./claims.py) | Fused claim contract and V1 gates |
| [review.py](./review.py) | Relevance-plus-coverage keyframe review plans |
| [fusion.py](./fusion.py) | Stateful multimodal fusion tracker |
| [component.yaml](./component.yaml) | Structured package metadata |
| [AGENTS.md](./AGENTS.md) | Local editing rules |

## Usage

```python
from twinr.proactive.event_fusion import MultimodalEventFusionTracker
from twinr.proactive.social import SocialObservation

tracker = MultimodalEventFusionTracker()
claims = tracker.observe(SocialObservation(observed_at=now, inspected=True))
```

## See also

- [AGENTS.md](./AGENTS.md)
- [component.yaml](./component.yaml)
- [../../../../docs/MULTIMODAL_EVENT_FUSION_V1.md](../../../../docs/MULTIMODAL_EVENT_FUSION_V1.md)
- [../social/README.md](../social/README.md)
- [../runtime/README.md](../runtime/README.md)
