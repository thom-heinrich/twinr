# AGENTS.md — /src/twinr/proactive/governance

## Scope

This directory owns the in-memory proactive governor: candidate normalization,
reservation issuance, cooldown checks, and bounded finalized history. Structural
metadata lives in [component.yaml](./component.yaml).

Out of scope:
- trigger scoring, wakeword logic, or presence-session detection
- sensor polling or background worker lifecycle code
- actual prompt delivery, printing, or spoken output handling
- durable storage beyond this in-memory policy object

## Key files

- `governor.py` — reservation, cooldown, and history policy implementation
- `__init__.py` — package export surface; treat changes as API-impacting
- `component.yaml` — structured metadata, callers, and tests

## Invariants

- At most one active reservation may exist at a time.
- Only the issued `reservation_token` may cancel or finalize a reservation; stale or synthetic reservations must be rejected.
- History trimming must retain every entry still needed for global, source-repeat, rolling-window, and presence-session enforcement.
- All externally supplied datetimes must remain timezone-aware before policy math.
- `safety_exempt` deliveries must bypass cooldown budgets without weakening non-exempt policy checks.

## Verification

After any edit in this directory, run:

```bash
python3 -m compileall src/twinr/proactive/governance
PYTHONPATH=src pytest test/test_proactive_governor.py -q
```

If config field coercion or package exports changed, also run:

```bash
PYTHONPATH=src pytest test/test_config.py -q
```

## Coupling

`governor.py` changes -> also check:
- `src/twinr/agent/base_agent/runtime/base.py`
- `src/twinr/agent/base_agent/runtime/context.py`
- `src/twinr/agent/workflows/runner.py`
- `src/twinr/agent/workflows/realtime_runner_background.py`
- `test/test_proactive_governor.py`

`__init__.py` changes -> also check:
- `src/twinr/proactive/__init__.py`
- import sites in runner and realtime background workflows

## Security

- Do not relax explicit bool, int, or aware-datetime validation into truthy/falsy coercion.
- Keep reservation tokens opaque and unguessable.
- Do not silently accept unknown finalized reservations; they indicate stale or synthetic callers.

## Output expectations

- Update docstrings when reservation semantics or cooldown behavior change.
- Keep [README.md](./README.md), [AGENTS.md](./AGENTS.md), and [component.yaml](./component.yaml) aligned when file roles or verification commands change.
- Treat export changes in `src/twinr/proactive/__init__.py` as follow-up API work.
