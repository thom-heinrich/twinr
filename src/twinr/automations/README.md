# automations

`automations` owns Twinr's canonical automation package. It defines the stored
automation model, evaluates time and sensor triggers, and exposes the helpers
other subsystems use to create and inspect supported automations.

## Responsibility

`automations` owns:
- define automation triggers, conditions, actions, and stored entries
- evaluate scheduled and fact-based triggers
- persist automations safely to the JSON store and backup file
- keep the automation JSON store, backup, and lock file owner-only (`0600`) so the Pi runtime and operator tools coordinate on the same state without reopening world-readable/writable state files
- map supported sensor triggers, including smart-home motion, button, alarm, and device-health events, to canonical if/then trigger shapes

`automations` does **not** own:
- web form parsing or automation page layout
- action execution after an automation matches
- provider prompt wording or tool schema definitions

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Package export surface |
| [store.py](./store.py) | Models, engine, and JSON store |
| [sensors.py](./sensors.py) | Sensor trigger mapping helpers |
| [component.yaml](./component.yaml) | Structured package metadata |
| [AGENTS.md](./AGENTS.md) | Local editing rules |

## Usage

```python
from twinr.automations import AutomationAction, AutomationStore

store = AutomationStore("state/automations.json", timezone_name="Europe/Berlin")
entry = store.create_time_automation(
    name="Morning greeting",
    schedule="daily",
    time_of_day="08:00",
    actions=[AutomationAction(kind="say", text="Good morning.")],
)
```

```python
from twinr.automations import build_sensor_trigger

trigger = build_sensor_trigger("pir_no_motion", hold_seconds=300, cooldown_seconds=60)
```

```python
from twinr.automations import build_sensor_trigger

trigger = build_sensor_trigger("smart_home_motion_detected", cooldown_seconds=30)
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [web automations UI](../web/automations.py)
- [tool schemas](../agent/tools/schemas/README.md)
