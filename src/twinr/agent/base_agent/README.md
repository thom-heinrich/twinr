# base_agent

`base_agent` is the canonical package surface for Twinr's core runtime
primitives. It keeps config loading, provider/runtime contracts, and stable
root imports together while pushing behavior into focused subpackages.

## Responsibility

`base_agent` owns:
- expose stable imports for config, runtime, state, prompting, and conversation helpers
- define `TwinrConfig` and the shared env-loading path, including the central `OPENAI_MODEL` default that non-coding LLM paths inherit unless an explicit override is set, smart-home background-worker tuning, calibrated attention-servo driver selection up to the custom Twinr kernel-servo path, the Pi-only ReSpeaker LED companion toggle for calm ring feedback, calm-motion safety knobs such as visible-target latching, optional exit-only servo follow with a configurable off-center degree clamp, explicit exit-activation delay, a configurable visible-edge departure threshold for monotone pursuit when a tracked person sticks at the frame edge, a short exit settle-hold for loaded hardware, centered visible-reacquire cooldown, monotone exit pursuit with cooldown and long-absence rest return, a dedicated slower rest-motion profile for startup and calm recentering, exact-center snap plus release semantics so the kernel-servo neutral state is not left skewed after calm returns, exit-trajectory extrapolation when a tracked user leaves the frame, release-after-settle to avoid loaded servo buzz, and bounded workflow timing budgets such as the dedicated wider timeout envelope for live search final lanes plus the GPT-5-capable fast-lane supervisor model/budget defaults
- define provider protocols, bundles, and composite provider glue, including the structured supervisor decision contract used by the fast voice lane
- preserve thin compatibility wrappers during the package split

`base_agent` does **not** own:
- workflow loops in `src/twinr/agent/workflows`
- provider transport implementations in `src/twinr/providers`
- hardware/audio adapters
- reminder, automation, and long-term-memory implementations

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Lazy root export surface |
| [config.py](./config.py) | Canonical runtime config, including bounded HDMI attention-refresh cadence, session-focus hold tuning for gaze-follow, calibrated attention-servo driver selection up to the Twinr kernel-servo writer and calm-motion safety knobs such as visible-target latching, optional exit-only physical follow with a configurable degree clamp plus loss-confirmation delay and visible-edge departure threshold, a short exit settle-hold, centered visible-reacquire cooldown, cooldown and long-absence rest-return timing, a dedicated slower rest-motion profile for startup/recenter moves, exact-center snap/release semantics for kernel-servo neutral alignment, exit-trajectory extrapolation, release-after-settle for quiet loaded holds, and smart-home background-worker tuning |
| [contracts.py](./contracts.py) | Provider/runtime contracts |
| [conversation/](./conversation/README.md) | Conversation micro-policies |
| [prompting/](./prompting/README.md) | Hidden instruction assembly |
| [runtime/](./runtime/README.md) | `TwinrRuntime` composition |
| [settings/](./settings/README.md) | Bounded runtime settings |
| [state/](./state/README.md) | State machine and snapshots |
| [adaptive_timing.py](./adaptive_timing.py), [language.py](./language.py), [turn_controller.py](./turn_controller.py), [conversation_closure.py](./conversation_closure.py), [personality.py](./personality.py) | Compatibility shims |

## Usage

```python
from twinr.agent.base_agent import TwinrConfig, TwinrRuntime

config = TwinrConfig.from_env(".env")
runtime = TwinrRuntime(config=config)
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [conversation](./conversation/README.md)
- [runtime](./runtime/README.md)
- [state](./state/README.md)
