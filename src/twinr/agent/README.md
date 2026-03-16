# agent

`agent` is the top-level package boundary for Twinr's agent stack. It keeps the
small `twinr.agent` compatibility surface alongside the focused child packages
that own core runtime, tool, and workflow behavior.

## Responsibility

`agent` owns:
- expose the thin `twinr.agent` import surface for `TwinrConfig`, `TwinrRuntime`, and `TwinrStatus`
- define the package boundary between `base_agent`, `tools`, and `workflows`
- direct changes toward the correct child package for runtime, tool, or workflow work

`agent` does **not** own:
- base-agent implementation details in [`base_agent`](./base_agent/README.md)
- tool prompt, schema, handler, or tool-runtime logic in [`tools`](./tools/README.md)
- workflow-loop orchestration details in [`workflows`](./workflows/README.md)
- concrete self-coding runtime behavior at the package-root level; compile contracts, compiler validation, local Codex driver status, and related work belong in the focused `self_coding/` subpackage

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Lazy root export surface |
| [component.yaml](./component.yaml) | Structured package metadata |
| [base_agent/README.md](./base_agent/README.md) | Core runtime package |
| [tools/README.md](./tools/README.md) | Tool stack package |
| [workflows/README.md](./workflows/README.md) | Workflow package |
| [self_coding/README.md](./self_coding/README.md) | Adaptive Skill Engine core package |
| [self_coding/REQUIREMENTS.md](./self_coding/REQUIREMENTS.md) | Adaptive Skill Engine requirements |
| [self_coding/IMPLEMENTATION_PLAN.md](./self_coding/IMPLEMENTATION_PLAN.md) | Concrete phased work packages and rollout plan |

## Usage

```python
from twinr.agent import TwinrConfig, TwinrRuntime

config = TwinrConfig.from_env(".env")
runtime = TwinrRuntime(config=config)
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [base_agent](./base_agent/README.md)
- [tools](./tools/README.md)
- [workflows](./workflows/README.md)
- [self_coding](./self_coding/README.md)
