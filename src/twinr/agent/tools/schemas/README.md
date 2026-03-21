# schemas

`schemas` owns the canonical JSON-schema builders for Twinr's tool-capable
agent lanes. It is the single source of truth for full, compact, and
realtime-safe schema variants.

## Responsibility

`schemas` owns:
- define canonical tool JSON schemas and shared property helpers
- derive compact schema variants for token-constrained callers
- derive realtime-safe schema variants and attach compatibility notes
- describe the self-coding learned-skill control tools alongside the learning/compile tools
- describe the smart-home discovery, filtered-state, aggregation, control, and sensor-stream tools alongside the rest of the canonical tool surface

`schemas` does **not** own:
- tool handler execution or runtime-side validation effects
- prompt instruction text or tool-calling policy
- workflow orchestration or provider session logic beyond schema shape

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Schema export surface |
| [contracts.py](./contracts.py) | Canonical schema builders |
| [component.yaml](./component.yaml) | Structured package metadata |

## Usage

```python
from twinr.agent.tools.schemas import build_agent_tool_schemas

schemas = build_agent_tool_schemas(tool_names)
```

## See also

- [component.yaml](./component.yaml)
- [handlers](../handlers/README.md)
- [prompting](../prompting/README.md)
