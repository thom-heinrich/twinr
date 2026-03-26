# AGENTS.md — /src/twinr/browser_automation

## Scope

This directory owns Twinr's optional browser-automation boundary: versioned
request/response contracts plus the fail-closed loader for the local
repo-root workspace.

Out of scope:
- actual browser-agent implementation details in the gitignored repo-root `browser_automation/`
- prompting or tool-routing decisions in `src/twinr/agent`
- senior-facing confirmation or voice UX policy

## Key files

- `contracts.py` — canonical request/result/availability dataclasses plus the driver protocol
- `loader.py` — safe local-workspace resolution and dynamic import bridge
- `__init__.py` — stable package export surface
- `README.md` — package boundary and local workspace contract
- `component.yaml` — structured ownership metadata

## Invariants

- Keep this package small and versioned; concrete browser stacks must stay outside git in the repo-root `browser_automation/` workspace.
- `loader.py` must fail closed on disabled, missing, malformed, or escaping workspace paths. No silent fallback to other folders is allowed.
- The contract surface must stay stable and JSON-friendly so future tool/runtime callers can depend on it without importing a concrete backend.
- Do not add Playwright- or browser-use-specific business logic here; that belongs in the local implementation until Twinr intentionally adopts one.

## Verification

After edits in this directory, run:

```bash
PYTHONPATH=src ./.venv/bin/pytest test/test_browser_automation.py test/test_config.py -q
```

## Coupling

`contracts.py` changes → also check:
- `loader.py`
- `test/test_browser_automation.py`

`loader.py` changes → also check:
- `src/twinr/agent/base_agent/config.py`
- `test/test_browser_automation.py`
- `test/test_config.py`

## Output expectations

- Preserve strict separation between versioned API and local implementation.
- Update `README.md`, `AGENTS.md`, and `component.yaml` whenever the workspace contract or exports change.
