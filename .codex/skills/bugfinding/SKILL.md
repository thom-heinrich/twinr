---
name: bugfinding
description: Track Twinr bugs as durable findings and close them cleanly through tasks, fixreports, and validation.
---

# Twinr Bugfinding

Use this skill when a bug should become a durable artifact instead of a temporary note.

Preferred entrypoint:

- default: `$bugfixing`
- use `$bugfinding` directly when the root cause is already understood and you are mainly doing intake/closure

## When to Create a Finding

Create a Finding when the issue is:

- user-visible
- operator-visible
- recurring or likely to recur
- hardware/runtime relevant
- likely to matter beyond the current session

Examples:

- button handling regressions
- empty or broken voice responses
- printer formatting/output failures
- confusing or misleading web UI settings
- display-state bugs
- audio capture/playback failures

## Mandatory Flow

### 1) Create or update the Finding

Use the local executable:

- `./agentic_tools/findings_cli`

Do not hand-edit the findings store directly.

### 2) Track the work

If the bug is more than a tiny single-step fix:

- create or claim a task:
  - `./agentic_tools/tasks ...`

Link the finding and the changed script/path where possible.

### 3) Fix and validate

Use a targeted proof:

- relevant `pytest`
- bounded hardware or web smoke
- direct repro check

### 4) Close with a FixReport

After the bug is fixed:

- create a fixreport:
  - `./agentic_tools/fixreport_cli ...`

### 5) Record material script/workflow evolution

If the bugfix changed a script, setup flow, or workflow meaningfully:

- add a Scripthistory entry:
  - `./agentic_tools/scripthistory ...`

### 6) Resolve the Finding

Update the finding status and attach the related fixreport/task context if applicable.

## Good Practice

- Use repo-relative paths.
- Keep summaries short and concrete.
- Never put secrets into findings or fixreports.
- Prefer one finding per distinct bug, not one mega-finding for everything.
- If the bug reveals a reusable causal theory, also update:
  - `./agentic_tools/hypothesis ...`
