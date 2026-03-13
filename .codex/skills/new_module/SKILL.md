---
name: new_module
description: Build a new Twinr module in the correct repo location with strong accessibility, stability, tests, and operator clarity.
---

# Twinr New Module

Use this skill when adding a substantial new capability or subsystem.

## Mission

Create a new module or subpackage that fits Twinr’s product goals:

- easy to reason about
- reliable on Raspberry Pi hardware
- clear for future maintenance
- consistent with the existing repo layout
- tested and documented enough to operate safely

## Placement Rules

Choose the existing home that matches the job:

- `src/twinr/agent/base_agent/`
  - state, config, runtime core
- `src/twinr/agent/workflows/`
  - orchestration loops and turn flow
- `src/twinr/provider/openai/`
  - provider-facing integrations
- `src/twinr/hardware/`
  - runtime adapters for audio/buttons/printer/camera
- `src/twinr/display/`
  - e-paper rendering or display service logic
- `src/twinr/memory/`
  - on-device memory
- `src/twinr/web/`
  - local dashboard/backend UI
- `hardware/...`
  - setup/probe/bootstrap shell or standalone hardware scripts
- `agentic_tools/...`
  - workflow/provenance/debugging tools only

Do not create new top-level product areas unless there is a clear architectural need.

## Hard Rules

- Preserve the green-button / yellow-button product model.
- Keep config centralized through `TwinrConfig` where relevant.
- Keep user-facing language simple and non-technical.
- Avoid hidden control flow.
- Avoid broad silent exception handling.
- Add tests for all non-trivial behavior.
- Update docs when the module changes how Twinr is run, configured, or operated.

## Required Workflow

1. Decide the correct placement before writing code.
2. Create or claim a task for non-trivial work:
   - `./agentic_tools/tasks ...`
3. If the work depends on new external behavior or research, use:
   - `$deep-research`
4. Implement the module with narrow responsibilities.
5. Add targeted tests under `test/`.
6. Add or update docs under `docs/` if operators/developers need them.
7. Record material workflow/script changes in:
   - `./agentic_tools/scripthistory ...`

## Definition Of Done

Before closing:

- the module lives in the correct place
- its API/responsibility is clear
- tests cover the new behavior
- config/docs were updated where needed
- the change keeps Twinr simpler or more reliable, not more confusing
