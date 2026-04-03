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
- persist subscriptions and refresh/discovery timing through remote-primary current-head/state records instead of monolithic snapshots
- dual-write subscriptions into a bounded embedded fallback snapshot so freshly saved feed sets remain readable when current-head item hydration lags behind the visible head across processes
- keep prompt-time state reads read-only so first-turn world-intelligence loads do not stall on legacy current-head promotion
- treat a partially hydrated subscription current head as an inconsistent remote-state view, not as proof that Twinr follows zero feeds; in that case recover from the same remote state's embedded legacy subscription snapshot until item hydration is coherent again
- reuse the parent package's shared payload-normalization and remote-state adapter helpers so RSS state decoding stays aligned with the rest of the personality package
- derive slow-changing topic/region interest signals from structured conversation or tool evidence
- keep live-search tool interests as situational evidence only; only conversation-derived or explicit world-intelligence interests may seed durable feed discovery during recalibration
- derive stronger engagement evidence from explicit structured follow-up reactions so topics the user asks to revisit are weighted above passive affinity
- translate explicit topic aversion into durable `avoid` state so Twinr backs off both conversationally and in source calibration
- track bounded user engagement with those interests so Twinr can prioritize the topics that repeatedly draw the user back in
- classify each learned topic into `resonant`, `warm`, `uncertain`, `cooling`, or `avoid`, separating healthy interest from exposure-aware non-uptake
- persist a slower `ongoing interest` layer so feed discovery and refresh can follow the topics that stay alive over repeated returns instead of re-deriving that judgment ad hoc
- persist a bounded `co-attention` layer so Twinr can distinguish a merely interesting topic from a genuinely shared running thread that both the user and the RSS/world layer keep returning to
- derive mild cross-session cooling from repeated topic switches or non-reengagement, but only when there was prior positive exposure evidence
- discover RSS or Atom feeds from source pages returned by the live web backend
- skip individual broken or oversized source pages during discovery instead of aborting the entire discovery/recalibration pass
- default outbound fetches to HTTP/1.1; only enable HTTP/2 explicitly when the optional `h2` transport dependency is installed
- recalibrate feed coverage during reflection-style maintenance windows instead of treating discovery as a daily poll
- refresh due feeds on a calm cadence and translate fresh items into world signals, continuity threads, and condensed awareness threads
- only let `co-attention` grow on genuinely new shared evidence such as fresh unseen feed items or new awareness-thread updates, not on every successful stale refresh
- decay stale engagement, ongoing interest, and co-attention so boosted feed priority/cadence falls back to baseline when the topic stops pulling the user in
- provide the shared persisted RSS source pool that Twinr's HDMI bottom ticker may read so display headlines stay aligned with the same feeds Twinr actually follows

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
| [store.py](./store.py) | Remote-primary current-head/item seam for subscriptions plus refresh/discovery timing state; prompt-time state loads stay read-only while compatibility promotion remains on write/migration paths |
| [service.py](./service.py) | Feed discovery, reflection-phase recalibration, due refresh, and conversion into world/continuity/awareness context |
| [../_payload_utils.py](../_payload_utils.py) | Shared payload normalization helpers reused by both personality and intelligence model decoding |
| [../_remote_state_utils.py](../_remote_state_utils.py) | Shared remote-primary adapter resolution reused by both snapshot-store seams |
| [component.yaml](./component.yaml) | Structured package metadata |
| [AGENTS.md](./AGENTS.md) | Local invariants and verification |

## See also

- [../README.md](../README.md)
- [../../tools/handlers/intelligence.py](../../tools/handlers/intelligence.py)
- [../../../../../../docs/persistent_personality_architecture.md](../../../../../../docs/persistent_personality_architecture.md)
