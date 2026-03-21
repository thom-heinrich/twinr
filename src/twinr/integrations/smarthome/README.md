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
- generic selector, exact scalar state-filter, pagination, and aggregation logic for entity and event queries
- route-aware aggregation that combines multiple bridges or hubs behind one generic surface
- bounded read/control/stream limits that all smart-home providers must respect
- stream-to-observation helpers that normalize provider events into Twinr facts and event names

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
| [query.py](./query.py) | Generic entity/event selectors, scalar state filters, pagination, and aggregation helpers |
| [aggregate.py](./aggregate.py) | Route-aware aggregation for multi-bridge or multi-hub smart-home providers |
| [runtime.py](./runtime.py) | Smart-home event normalization and bounded sensor worker |
| [component.yaml](./component.yaml) | Structured package metadata |
| [hue/](./hue) | First provider package built on the generic contracts |

## Design notes

- `list_entities` is the generic read path for LLM-driven user commands. It now supports provider-neutral selectors, exact scalar state filters, pagination, and simple aggregations instead of a special-case house-summary tool.
- `control_entities` is the generic control path for explicit device actions.
- `read_sensor_stream` is the bounded background path for automation and sensor streaming, and it can also apply generic event selectors plus simple aggregations for user-visible inspection turns.
- `AggregatedSmartHomeProvider` rewrites child entity IDs into route-qualified IDs when Twinr reads more than one bridge at once, so read/control/stream calls stay unambiguous.
- `SmartHomeObservationBuilder` keeps a compact `facts["smart_home"]` snapshot so runtime automations can react to motion, button, alarm, and device-health changes without parsing vendor payloads.
- `SmartHomeSensorWorker` polls one configured provider or aggregate stream and emits normalized observations into the realtime loop.
- Longitudinal behavior logic should stay room-agnostic by default. Provider labels such as Hue room names are metadata only, not stable behavioral ground truth. The current environment-profile design for that path lives in [`../../../../docs/SMART_HOME_ENVIRONMENT_PROFILE_V1.md`](../../../../docs/SMART_HOME_ENVIRONMENT_PROFILE_V1.md).
- Provider packages must keep unsafe or high-risk device classes out of the generic low-risk control surface unless a stricter reviewed policy is added later.
