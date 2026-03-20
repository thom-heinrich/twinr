# intelligence

`intelligence` owns Twinr's RSS-backed place/world-awareness source layer for
the personality system. It persists which feeds Twinr follows, learns which
topics and regions deserve source coverage, discovers new RSS or Atom sources
from explicit web-search source pages, and refreshes due feeds into bounded
`WORLD` and `CONTINUITY` context updates plus slower situational-awareness
threads.

## Responsibility

`intelligence` owns:
- define typed subscription, calibration-state, situational-awareness, and refresh-result models for RSS/world intelligence
- persist subscriptions and refresh/discovery timing through remote-primary snapshots
- derive slow-changing topic/region interest signals from structured conversation or tool evidence
- discover RSS or Atom feeds from source pages returned by the live web backend
- recalibrate feed coverage during reflection-style maintenance windows instead of treating discovery as a daily poll
- refresh due feeds on a calm cadence and translate fresh items into world signals, continuity threads, and condensed awareness threads

`intelligence` does **not** own:
- prompt-layer rendering
- core personality evolution policy
- ordinary one-off live search answers
- workflow or tool orchestration above its explicit service boundary

## Key files

| File | Purpose |
|---|---|
| [models.py](./models.py) | Typed feed subscriptions, timing state, and refresh/config result models |
| [calibration.py](./calibration.py) | Convert structured personality/tool evidence into world-intelligence calibration signals |
| [store.py](./store.py) | Remote-primary snapshot seam for subscriptions and refresh/discovery timing |
| [service.py](./service.py) | Feed discovery, reflection-phase recalibration, due refresh, and conversion into world/continuity/awareness context |
| [component.yaml](./component.yaml) | Structured package metadata |
| [AGENTS.md](./AGENTS.md) | Local invariants and verification |

## See also

- [../README.md](../README.md)
- [../../tools/handlers/intelligence.py](../../tools/handlers/intelligence.py)
- [../../../../../../docs/persistent_personality_architecture.md](../../../../../../docs/persistent_personality_architecture.md)
