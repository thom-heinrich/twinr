---
name: bugfixing
description: Unified Twinr bugfix entrypoint. Route through systematic root-cause work, evidence gathering, and local findings/fixreport closure.
---

# Twinr Bugfixing

Use this as the default bug/regression entrypoint in `twinr`.

## Route

- Use `$systematic-debugging` when the root cause is not yet proven.
- Use `$instrument` when the behavior is flaky, timing-sensitive, hardware-sensitive, or otherwise weakly evidenced.
- Use `$bugfinding` for durable intake/closure via Findings, Tasks, FixReports, and Scripthistory.

Default flow:

1. Reproduce and understand the problem.
2. Prove root cause.
3. Apply the smallest correct fix.
4. Validate with targeted tests or bounded smokes.
5. Write the closure artifacts.

## Mandatory Workflow

### 1) Start with evidence

- Do not patch symptoms first.
- Capture the exact failing behavior:
  - failing test
  - clear repro steps
  - runtime error
  - hardware/log evidence
- If the issue is timing-sensitive or hard to isolate, activate `$instrument`.

### 2) Track the work

For multi-step or non-trivial bugfixes:

- create or claim a task:
  - `./agentic_tools/tasks ...`

### 3) Create a Finding when the bug is durable/actionable

Use:

- `./agentic_tools/findings_cli ...`

Do this when the bug is:

- user-visible
- operator-visible
- recurring
- hardware/runtime relevant
- likely to matter after context compression

### 4) Fix only confirmed root causes

- Keep fixes minimal and local.
- Do not bundle speculative cleanups into the same bugfix.
- If you discover a broader structural issue, split it into follow-up work.

### 5) Validate

Run the narrowest proof that closes the bug:

- targeted `pytest`
- runtime smoke
- bounded hardware loop
- display/printer/button probe
- web UI smoke

The validation should prove the pre-fix failure is gone and the changed surface still works.

### 6) Close the loop

After every real bugfix:

- write a FixReport:
  - `./agentic_tools/fixreport_cli ...`
- add a Scripthistory entry for material script/workflow changes:
  - `./agentic_tools/scripthistory ...`
- resolve or update the related Finding when applicable

### 7) Optional hypothesis

If the bug reveals a reusable thesis about failure causality or architecture:

- add/update a hypothesis:
  - `./agentic_tools/hypothesis ...`

## Output Contract

Before handing back to the user, make sure you can state:

1. the root cause
2. the concrete fix
3. the validation performed
4. any remaining risk
5. the related task/finding/fixreport ids if you created them
