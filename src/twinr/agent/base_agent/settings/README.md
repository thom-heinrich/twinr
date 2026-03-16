# settings

Internal package for the base agent's bounded runtime-adjustable settings:
memory-capacity presets, supported spoken voices, speed/timing controls, and
safe `.env` persistence for tool-driven changes.

## Responsibility

`settings` owns:
- Define the supported simple-setting catalog and its bounds
- Normalize user-requested setting changes into stable env updates
- Resolve descriptive voice requests to one supported Twinr voice
- Persist tool-driven `.env` changes without dropping comments

`settings` does **not** own:
- Full config parsing or default resolution
- Web dashboard form handling or generic env editing
- Tool confirmation flow or runtime-side value application
- Audio/provider behavior beyond exposing supported setting values

## Key files

| File | Purpose |
|---|---|
| `simple_settings.py` | Normalize simple settings and env writes |
| `__init__.py` | Mark package only |

## Usage

```python
from pathlib import Path

from twinr.agent.base_agent.settings.simple_settings import (
    update_simple_setting,
    write_env_updates,
)

result = update_simple_setting(config, setting="speech_speed", action="decrease")
write_env_updates(Path(".env"), result.env_updates)
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [settings handler](../../tools/handlers/settings.py)
- [tool prompting](../../tools/prompting/instructions.py)
