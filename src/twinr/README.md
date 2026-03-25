# twinr

`twinr` is the root Python package for the device runtime. It owns the package
boundary, the CLI bootstrap, the lazy package-root import surface, and a small
set of shared helpers that multiple subsystems reuse.

## Responsibility

`twinr` owns:
- expose the canonical `twinr` import surface
- bootstrap `python -m twinr` and the installed `twinr` script
- gate `/twinr`-only runtime sidecars such as the display companion and remote-memory watchdog so the acceptance instance stays authoritative, while allowing explicit per-env display-companion overrides
- keep shared root helpers for text normalization, local-date parsing, and structured JSON response handling
- define the boundary around the documented child packages

`twinr` does **not** own:
- workflow-loop behavior in [`agent/workflows`](./agent/workflows/README.md)
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
| [channels/README.md](./channels/README.md) | External text-channel package |
| [component.yaml](./component.yaml) | Structured package metadata |
| [llm_json.py](./llm_json.py) | Structured-response helper |
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
- [channels](./channels/README.md)
- [agent](./agent/README.md)
- [memory](./memory/README.md)
- [providers](./providers/README.md)
- [hardware](./hardware/README.md)
- [web](./web/README.md)
