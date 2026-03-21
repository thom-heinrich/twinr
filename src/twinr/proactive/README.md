# proactive

`proactive` is the top-level package boundary for Twinr's proactive stack. It
keeps the broad `twinr.proactive` compatibility surface alongside the focused
child packages that own runtime orchestration, social-trigger scoring,
short-window event fusion, delivery governance, and wakeword behavior.

## Responsibility

`proactive` owns:
- expose the `twinr.proactive` compatibility import surface across proactive child packages
- define the package boundary between `runtime`, `social`, `event_fusion`, `governance`, and `wakeword`
- direct new work into the correct child package instead of growing the package root
- keep optional wakeword backend/tooling dependencies from breaking the root import surface when those paths are not selected at runtime

`proactive` does **not** own:
- proactive monitor orchestration details, ReSpeaker policy-hook derivation, conservative speaker-association/multimodal-initiative fusion, runtime alerts, or structured observation export in [`runtime`](./runtime/README.md)
- social-trigger scoring, sensing adapters, or review behavior in [`social`](./social/README.md)
- short-window multimodal event buffering, sequence derivation, or fused event claims in [`event_fusion`](./event_fusion/README.md)
- delivery reservation and cooldown policy details in [`governance`](./governance/README.md)
- wakeword matching, calibration, streaming, or evaluation details in [`wakeword`](./wakeword/README.md)

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Root compatibility export surface |
| [component.yaml](./component.yaml) | Structured package metadata |
| [runtime/README.md](./runtime/README.md) | Runtime monitor package |
| [social/README.md](./social/README.md) | Social-trigger package |
| [event_fusion/README.md](./event_fusion/README.md) | Short-window event-fusion package |
| [governance/README.md](./governance/README.md) | Delivery-governance package |
| [wakeword/README.md](./wakeword/README.md) | Wakeword package |

## Usage

```python
from twinr.proactive import (
    ProactiveGovernor,
    SocialFineHandGesture,
    SocialTriggerEngine,
    WakewordDecisionPolicy,
    build_default_proactive_monitor,
)
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [runtime](./runtime/README.md)
- [social](./social/README.md)
- [event_fusion](./event_fusion/README.md)
- [governance](./governance/README.md)
- [wakeword](./wakeword/README.md)
