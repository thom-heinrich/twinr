# proactive

`proactive` is the top-level package boundary for Twinr's proactive stack. It
keeps the broad `twinr.proactive` compatibility surface alongside the focused
child packages that own runtime orchestration, social-trigger scoring, delivery
governance, and wakeword behavior.

## Responsibility

`proactive` owns:
- expose the `twinr.proactive` compatibility import surface across proactive child packages
- define the package boundary between `runtime`, `social`, `governance`, and `wakeword`
- direct new work into the correct child package instead of growing the package root

`proactive` does **not** own:
- proactive monitor orchestration details in [`runtime`](./runtime/README.md)
- social-trigger scoring, sensing adapters, or review behavior in [`social`](./social/README.md)
- delivery reservation and cooldown policy details in [`governance`](./governance/README.md)
- wakeword matching, calibration, streaming, or evaluation details in [`wakeword`](./wakeword/README.md)

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Root compatibility export surface |
| [component.yaml](./component.yaml) | Structured package metadata |
| [runtime/README.md](./runtime/README.md) | Runtime monitor package |
| [social/README.md](./social/README.md) | Social-trigger package |
| [governance/README.md](./governance/README.md) | Delivery-governance package |
| [wakeword/README.md](./wakeword/README.md) | Wakeword package |

## Usage

```python
from twinr.proactive import (
    ProactiveGovernor,
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
- [governance](./governance/README.md)
- [wakeword](./wakeword/README.md)
