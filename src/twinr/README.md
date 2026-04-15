# twinr

`twinr` is the root Python package for the device runtime. It owns the package
boundary, the CLI bootstrap, the lazy package-root import surface, and a small
set of shared helpers that multiple subsystems reuse.

## Responsibility

`twinr` owns:
- expose the canonical `twinr` import surface
- bootstrap `python -m twinr` and the installed `twinr` script
- gate `/twinr`-only runtime sidecars such as the authoritative display surface and remote-memory watchdog so the acceptance instance stays authoritative, while allowing explicit per-env display-companion overrides
- fail closed when host-side service commands such as `run-orchestrator-server` are launched from anything other than the leading repo root and import path, so stale stage snapshots cannot masquerade as the active runtime
- define the versioned package boundary around optional browser automation contracts and its local unversioned workspace hook
- keep shared root helpers for text normalization, local-date parsing, and structured JSON response handling
- keep shared authoritative-host policy helpers such as `/twinr` root detection, Raspberry Pi host detection, and visible-display launch policy so the CLI and runtime supervisor do not drift
- define the boundary around the documented child packages

`twinr` does **not** own:
- workflow-loop behavior in [`agent/workflows`](./agent/workflows/README.md)
- browser-agent implementation details in [`browser_automation`](./browser_automation/README.md)
- long-lived external messaging transports in [`channels`](./channels/README.md)
- provider transport logic in [`providers`](./providers/README.md)
- memory internals in [`memory`](./memory/README.md)
- hardware adapter behavior in [`hardware`](./hardware/README.md)
- dashboard routes or presenters in [`web`](./web/README.md)

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Lazy root exports |
| [__main__.py](./__main__.py) | CLI bootstrap and dispatch |
| [runtime_host.py](./runtime_host.py) | Authoritative Pi host/root and visible-display launch policy helpers |
| [browser_automation/README.md](./browser_automation/README.md) | Optional browser-automation API boundary |
| [channels/README.md](./channels/README.md) | External text-channel package |
| [component.yaml](./component.yaml) | Structured package metadata |
| [llm_json.py](./llm_json.py) | Structured-response helper |
| [runtime_paths.py](./runtime_paths.py) | Pi-only system-site-path bootstrap helper for OS-managed modules |
| [temporal.py](./temporal.py) | Local-date parsing helper |
| [text_utils.py](./text_utils.py) | Text and identifier helper |
| [agent/README.md](./agent/README.md) | Core runtime package |
| [memory/README.md](./memory/README.md) | Memory subsystem |
| [providers/README.md](./providers/README.md) | Provider subsystem |
| [web/README.md](./web/README.md) | Operator dashboard |

## Usage

```python
from twinr import TwinrConfig, TwinrRuntime

config = TwinrConfig.from_env(".env")
runtime = TwinrRuntime(config=config)
```

```bash
PYTHONPATH=src python3 -m twinr --env-file .env --run-web
PYTHONPATH=src python3 -m twinr --env-file .env --run-streaming-loop --loop-duration 15
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [browser_automation](./browser_automation/README.md)
- [channels](./channels/README.md)
- [agent](./agent/README.md)
- [memory](./memory/README.md)
- [providers](./providers/README.md)
- [hardware](./hardware/README.md)
- [web](./web/README.md)
