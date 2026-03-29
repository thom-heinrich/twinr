# governance

In-memory proactive delivery-governance policy for Twinr runtime workflows.

This package owns reservation gating, cooldown checks, and bounded history for
proactive candidates before they are delivered.

## Responsibility

`governance` owns:
- Normalize proactive candidates and reservation timestamps
- Gate speech deliveries behind global, per-source, rolling-window, and presence-session limits
- Track one active reservation at a time
- Record delivered and skipped outcomes for later policy checks

`governance` does **not** own:
- Score social triggers or detect wakewords
- Poll sensors or run background worker loops
- Deliver prompts, speech, or print output
- Plan long-term proactive candidates

## Key files

| File | Purpose |
|---|---|
| `__init__.py` | Package export surface |
| `governor.py` | Cooldown and reservation policy |
| `component.yaml` | Structured package metadata |
| `AGENTS.md` | Local editing rules |

## Usage

```python
from twinr.proactive.governance import ProactiveGovernor, ProactiveGovernorCandidate

governor = ProactiveGovernor.from_config(config)
candidate = ProactiveGovernorCandidate(
    source_kind="social",
    source_id="possible_fall",
    summary="Possible fall check-in",
)
decision = governor.try_reserve(candidate)
if decision.allowed and decision.reservation is not None:
    governor.mark_delivered(decision.reservation)
```

## Config Notes

- Presence-session prompt budgets can be bounded explicitly with
  `TWINR_PROACTIVE_GOVERNOR_PRESENCE_SESSION_WINDOW_S`.
- `TWINR_PROACTIVE_GOVERNOR_PRESENCE_GRACE_S` remains available as a legacy
  alias for the same budget window.
- If neither field is set, the governor now falls back conservatively to the
  largest of the default grace window, the rolling governor window, and the
  voice follow-up timeout instead of collapsing to the short follow-up timeout.

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [../runtime/README.md](../runtime/README.md)
- [../social/README.md](../social/README.md)
