# base_agent

`base_agent` is the canonical package surface for Twinr's core runtime
primitives. It keeps config loading, provider/runtime contracts, and stable
root imports together while pushing behavior into focused subpackages.

## Responsibility

`base_agent` owns:
- expose stable imports for config, runtime, state, prompting, and conversation helpers
- define `TwinrConfig` and the shared env-loading path, including the central `OPENAI_MODEL` default that non-coding LLM paths inherit unless an explicit override is set, smart-home background-worker tuning, calibrated attention-servo driver selection up to the custom Twinr kernel-servo path and the local Pololu Mini Maestro command-port path, the Pi-only ReSpeaker LED companion toggle for calm ring feedback, the single-Pi camera topology wiring that now fail-closes retired helper-Pi `second_pi` / proxy envs, calm-motion safety knobs such as visible-target latching, optional exit-only servo follow with a configurable off-center degree clamp, periodic visible-user recenter dwell/tolerance knobs so long-lived off-axis users can be corrected without continuous jitter, explicit exit-activation delay, a configurable visible side-departure threshold plus an explicit visible-box edge threshold so monotone pursuit only starts when the authoritative person geometry is actually near the image boundary, a short exit settle-hold for loaded hardware, centered visible-reacquire cooldown, monotone exit pursuit with cooldown and long-absence rest return, a dedicated slower rest-motion profile for startup and calm recentering, exact-center snap plus release semantics so the current servo writer's neutral state is not left skewed after calm returns, exit-trajectory extrapolation when a tracked user leaves the frame, an opt-in forensic trace flag for per-tick servo decision ledgers, release-after-settle to avoid loaded servo buzz, secure-by-default realtime sensitive-tool gating with explicit override knobs, and bounded workflow timing budgets such as the dedicated wider timeout envelope for live search final lanes plus the GPT-5-capable fast-lane supervisor model/budget defaults
- define provider protocols, bundles, and composite provider glue, including the structured supervisor decision contract used by the fast voice lane

`base_agent` does **not** own:
- workflow loops in `src/twinr/agent/workflows`
- provider transport implementations in `src/twinr/providers`
- hardware/audio adapters
- reminder, automation, and long-term-memory implementations

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Lazy root export surface |
| [config.py](./config.py) | Stable import facade for the canonical runtime config surface |
| [_config/schema.py](./_config/schema.py) | Immutable `TwinrConfig` dataclass shape, helper properties, and delegated `from_env()` entry |
| [_config/loading.py](./_config/loading.py) | Canonical `.env` + process-env loading flow that composes domain loaders |
| [_config/load_*.py](./_config/loading.py) | Domain-separated env loaders for providers, streaming, channels, vision, memory, and hardware |
| [_config/normalization.py](./_config/normalization.py) | Post-init normalization and validation orchestration for bounded config values |
| [_config/normalization_updates.py](./_config/normalization_updates.py) | Grouped frozen-dataclass update helpers for normalized config fields |
| [_config/parsing.py](./_config/parsing.py) | Primitive env parsing helpers kept separate from the dataclass and loaders |
| [_config/constants.py](./_config/constants.py) | Shared runtime-config constants and default value anchors |
| [contracts.py](./contracts.py) | Provider/runtime contracts |
| [conversation/](./conversation/README.md) | Conversation micro-policies |
| [prompting/](./prompting/README.md) | Hidden instruction assembly |
| [runtime/](./runtime/README.md) | `TwinrRuntime` composition |
| [settings/](./settings/README.md) | Bounded runtime settings |
| [state/](./state/README.md) | State machine and snapshots |

## Usage

```python
from twinr.agent.base_agent import TwinrConfig, TwinrRuntime

config = TwinrConfig.from_env(".env")
runtime = TwinrRuntime(config=config)
```

No migration is required: callers still import `TwinrConfig` and helper symbols
from [`config.py`](./config.py), while the implementation now lives in the
focused [`_config/`](./_config/schema.py) package with domain-specific loaders
and grouped normalization helpers.

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [conversation](./conversation/README.md)
- [runtime](./runtime/README.md)
- [state](./state/README.md)
