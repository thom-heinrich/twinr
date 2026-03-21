# base_agent

`base_agent` is the canonical package surface for Twinr's core runtime
primitives. It keeps config loading, provider/runtime contracts, and stable
root imports together while pushing behavior into focused subpackages.

## Responsibility

`base_agent` owns:
- expose stable imports for config, runtime, state, prompting, and conversation helpers
- define `TwinrConfig` and the shared env-loading path, including smart-home background-worker tuning
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
| [config.py](./config.py) | Canonical runtime config, including bounded HDMI attention-refresh cadence, session-focus hold tuning for gaze-follow, and smart-home background-worker tuning |
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
