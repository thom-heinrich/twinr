# smarthome

`smarthome` owns Twinr's vendor-neutral smart-home integration surface.

It exists so future providers such as Hue, Matter, Zigbee coordinators,
Z-Wave controllers, or cloud hubs can share one bounded Twinr contract for:

- user-driven state reads
- explicit device control
- background sensor/event stream reads for automations and proactive logic

## Responsibility

`smarthome` owns:
- generic entity, command, and event models
- the shared smart-home adapter surface that maps Twinr integration requests to provider code
- bounded read/control/stream limits that all smart-home providers must respect

`smarthome` does **not** own:
- Hue bridge transport details
- provider-specific resource normalization
- automation trigger wiring in `src/twinr/automations`
- runtime loop orchestration outside the integration boundary

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Package export surface |
| [models.py](./models.py) | Generic smart-home entities, commands, and events |
| [adapter.py](./adapter.py) | Shared read/control/stream adapter and provider protocols |
| [component.yaml](./component.yaml) | Structured package metadata |
| [hue/](./hue) | First provider package built on the generic contracts |

## Design notes

- `list_entities` is the generic read path for LLM-driven user commands.
- `control_entities` is the generic control path for explicit device actions.
- `read_sensor_stream` is the bounded background path for automation and sensor streaming.
- Provider packages must keep unsafe or high-risk device classes out of the generic low-risk control surface unless a stricter reviewed policy is added later.
