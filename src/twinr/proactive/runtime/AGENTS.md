# AGENTS.md — /src/twinr/proactive/runtime

## Scope

This directory owns proactive runtime orchestration: presence-session control,
monitor tick coordination, wakeword arming, degraded-mode sensor handling, and
bounded monitor lifecycle. Structural metadata lives in
[component.yaml](./component.yaml).

Out of scope:
- social trigger scoring, review prompt content, or wakeword matching internals
- raw hardware adapters and low-level audio/camera/PIR implementations
- proactive delivery cooldown policy in `../governance/`
- top-level agent runtime loops above the monitor boundary

## Key files

- `presence.py` — source of truth for wakeword presence-session arming
- `service.py` — monitor orchestration, degraded-mode handling, and lifecycle
- `__init__.py` — package export surface; treat changes as API-impacting
- `component.yaml` — structured metadata, callers, and tests

## Invariants

- `PresenceSessionController.observe()` must preserve a monotonic internal timeline even when callers pass regressing or malformed timestamps.
- Wakeword arming must fail closed on malformed sensor flags or unavailable dependencies.
- `service.py` stays orchestration-focused; trigger scoring and wakeword matching belong in sibling packages.
- `ProactiveMonitorService` lifecycle must remain idempotent, bounded, and safe under partial startup or shutdown failure.
- `build_default_proactive_monitor()` must not start an inert monitor when no operational sensor path exists.

## Verification

After any edit in this directory, run:

```bash
python3 -m compileall src/twinr/proactive/runtime
PYTHONPATH=src pytest test/test_proactive_monitor.py -q
```

If workflow wiring or export surface changed, also run:

```bash
PYTHONPATH=src pytest test/test_runner.py test/test_realtime_runner.py -q
```

## Coupling

`presence.py` changes -> also check:
- `service.py`
- `src/twinr/proactive/wakeword/stream.py`
- `test/test_proactive_monitor.py`

`service.py` changes -> also check:
- `src/twinr/proactive/social/`
- `src/twinr/proactive/wakeword/`
- `src/twinr/agent/workflows/runner.py`
- `src/twinr/agent/workflows/realtime_runner.py`
- `test/test_proactive_monitor.py`

`__init__.py` changes -> also check:
- `src/twinr/proactive/__init__.py`
- monitor construction call sites in runtime workflows

## Security

- Keep wakeword capture writes under validated artifacts-root subpaths; do not widen path or symlink handling.
- Do not bypass the privacy gate that disables camera-triggered proactive behavior when `proactive_enabled` is false.
- Keep emit and ops-event paths non-throwing so sensor and worker faults remain recoverable.

## Output expectations

- Update docstrings when coordinator entry points, lifecycle semantics, or exports change.
- Keep [README.md](./README.md), [AGENTS.md](./AGENTS.md), and [component.yaml](./component.yaml) aligned when file roles or verification commands change.
- Treat export changes in `src/twinr/proactive/__init__.py` as follow-up API work.
