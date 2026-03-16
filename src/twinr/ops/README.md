# ops

`ops` owns Twinr's local operational support layer. It provides config audits,
device and host snapshots, singleton loop locks, file-backed ops stores,
bounded self-tests, and redacted support exports for the web UI and operator
tools.

## Responsibility

`ops` owns:
- resolve canonical paths for ops artifacts and stores
- persist sanitized ops events and usage telemetry
- collect config, device, and system-health snapshots
- coordinate per-loop singleton locks
- run bounded self-tests and build support bundles

`ops` does **not** own:
- runtime orchestration in `src/twinr/agent/workflows`
- hardware adapter implementations in `src/twinr/hardware`
- web routing and template rendering in `src/twinr/web`
- provider request execution beyond normalized usage extraction

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Public ops exports |
| [paths.py](./paths.py) | Canonical ops paths |
| [locks.py](./locks.py) | Loop singleton locks |
| [events.py](./events.py) | Ops event JSONL store |
| [usage.py](./usage.py) | Usage telemetry store |
| [checks.py](./checks.py) | Config audit checks |
| [health.py](./health.py) | Host and service health |
| [devices.py](./devices.py) | Device overview probes |
| [self_test.py](./self_test.py) | Bounded hardware self-tests |
| [support.py](./support.py) | Support bundle export |
| [component.yaml](./component.yaml) | Structured package metadata |

## Usage

```python
from twinr.ops import TwinrOpsEventStore, resolve_ops_paths_for_config

ops_paths = resolve_ops_paths_for_config(config)
TwinrOpsEventStore.from_config(config).append(
    event="assistant_started",
    message="Twinr runtime booted.",
    data={"events_path": str(ops_paths.events_path)},
)
```

```python
from twinr.ops import TwinrSelfTestRunner, collect_system_health

health = collect_system_health(config)
result = TwinrSelfTestRunner(config).run("printer")
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [web app](../web/app.py)
- [display health consumer](../display/service.py)
