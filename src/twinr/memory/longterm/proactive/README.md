# proactive

Plan, gate, and persist long-term proactive suggestions for the memory runtime.

## Responsibility

`proactive` owns:
- Rank bounded proactive candidates from active long-term memory objects
- Combine durable memory cues with live sensor facts for routine offers
- Persist reservation, delivery, and skip history for proactive candidates
- Apply confidence, sensitivity, and cooldown policy before reservation

`proactive` does **not** own:
- Extract or consolidate long-term memory objects
- Orchestrate runtime polling or spoken delivery loops
- Decide cross-feature proactive governor policy outside long-term memory
- Persist general long-term memory objects or archives

## Key files

| File | Purpose |
|---|---|
| `__init__.py` | Package export surface |
| `planner.py` | Candidate ranking logic |
| `state.py` | History store and policy |
| `component.yaml` | Package metadata map |

## Usage

```python
from twinr.memory.longterm.proactive import (
    LongTermProactivePlanner,
    LongTermProactivePolicy,
    LongTermProactiveStateStore,
)

planner = LongTermProactivePlanner(timezone_name=config.local_timezone_name)
state_store = LongTermProactiveStateStore.from_config(config)
policy = LongTermProactivePolicy(config=config, state_store=state_store)
plan = planner.plan(objects=memory_objects, live_facts=live_facts)
reservation = policy.reserve_candidate(plan=plan)
```

## See also

- [../MEMORY_ARCHITECTURE.md](../MEMORY_ARCHITECTURE.md)
- [../runtime/README.md](../runtime/README.md)
- [AGENTS.md](./AGENTS.md)
- [component.yaml](./component.yaml)
