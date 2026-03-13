---
name: refactor
description: Refactor Twinr code without changing intended behavior, physical interaction semantics, or operator-facing contracts.
---

# Twinr Refactor

Use this skill for structural cleanup, decomposition, and package/module reshaping when the expected behavior should stay the same.

## Mission

Refactor code so it becomes easier to maintain while preserving:

- button semantics
- runtime-state behavior
- config/env behavior
- operator-facing CLI/web behavior
- printer/display/user-visible behavior

This is a structural change skill, not a stealth feature-change skill.

## Hard Rules

- Keep behavior stable unless the user explicitly asked for behavior changes.
- Preserve public entrypoints unless there is a strong reason not to.
- Preserve `.env` and `TwinrConfig` expectations unless intentionally changing config.
- Keep the green/yellow interaction model intact.
- Do not bundle product changes into a pure refactor.
- Add tests before or alongside the refactor so parity is provable.

## What Counts As Compatibility Surface In Twinr

- `twinr` / `python -m twinr` CLI behavior
- `TwinrConfig` env names and defaults
- state-machine transitions
- hardware-loop and realtime-loop entrypoints
- web routes and template expectations
- print formatting constraints
- display-state signaling

## Recommended Workflow

1. Inspect the target path and context:
   - `./agentic_tools/scriptinfo <path>`
2. Create or claim a task for non-trivial refactors:
   - `./agentic_tools/tasks ...`
3. Identify the behavior that must remain stable.
4. Add or update targeted tests first when possible.
5. Perform the structural refactor in small steps.
6. Re-run the tests and relevant smokes.
7. Record material script/workflow changes:
   - `./agentic_tools/scripthistory ...`

## Good Refactor Outcomes

- responsibilities are clearer
- modules are smaller and easier to reason about
- tests become easier to write
- config and runtime boundaries are easier to follow
- no new user confusion is introduced

## Bad Refactor Smells

- changed behavior without explicit intent
- mixed structural cleanup plus hidden product changes
- moved code into the wrong repo area
- no parity proof
- broken imports, routes, CLI options, or env expectations
