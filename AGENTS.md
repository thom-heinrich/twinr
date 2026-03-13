# TWINR AGENTS.md

## Mission

You are working in `/home/thh/twinr`.

Twinr is a voice-first physical assistant for senior citizens. The product goal is:

- maximum accessibility
- extreme ease of use
- clear physical interaction
- calm and trustworthy behavior
- strong operational stability on a Raspberry Pi

Primary product invariant:

- green button: start listening / ask
- yellow button: print the latest answer

Everything you build must preserve or improve that simplicity.

## Quality Standards

Treat these as hard requirements:

- Keep the interaction model obvious and low-cognitive-load.
- Preserve deterministic runtime-state behavior.
- Bound waiting, recording, playback, and print behavior.
- Fail clearly; do not hide failures behind silent fallbacks.
- Keep user-facing language plain, warm, and non-technical.
- Keep caregiver/operator maintenance simple.
- Prefer small, explicit, testable components over clever hidden automation.
- Do not add complexity unless it directly improves accessibility, clarity, or stability.

Bugfix quality bar:

- Root cause first, not symptom patching.
- Add or update a Finding for durable actionable bugs.
- Write a FixReport after every real bugfix.
- Record material script/workflow changes in Scripthistory.
- Run targeted tests or bounded smoke checks before closing work.

Hardware quality bar:

- Button semantics must stay stable.
- Audio capture and playback must stay bounded and recoverable.
- Printer output must stay short, readable, and bounded.
- Display behavior must communicate state clearly at a glance.
- A broken peripheral must not push the runtime into undefined behavior.

## Repo Layout

Use the existing structure. Do not invent new top-level areas without a strong reason.

Runtime application code:

- `src/twinr/agent/base_agent/`
  - config, runtime, state machine, runtime snapshots
- `src/twinr/agent/workflows/`
  - hardware loop orchestration and realtime loop orchestration
- `src/twinr/provider/openai/`
  - active provider implementation path
- `src/twinr/providers/`
  - compatibility wrappers for older imports
- `src/twinr/hardware/`
  - runtime hardware adapters used by the assistant
- `src/twinr/display/`
  - e-paper rendering and display service
- `src/twinr/memory/`
  - on-device memory logic
- `src/twinr/web/`
  - local FastAPI/Jinja configuration dashboard

Repo support areas:

- `hardware/buttons/`
  - GPIO setup and probe scripts
- `hardware/mic/`
  - OS/audio-device setup scripts
- `hardware/printer/`
  - printer/CUPS setup scripts
- `hardware/display/`
  - display setup scripts and smoke commands
- `personality/`
  - hidden system/personality/user context files only
- `docs/`
  - documentation, specs, provider notes
- `test/`
  - tests
- `agentic_tools/`
  - repo workflow/provenance/debugging tools only, not runtime assistant logic
- `common/`
  - shared support code for copied local tools; do not move Twinr runtime logic here unless it is truly cross-tool infrastructure

Placement rules:

- New runtime capability belongs under `src/twinr/...`.
- New Raspberry-Pi/bootstrap shell scripts belong under `hardware/...`.
- New user/operator docs belong under `docs/...`.
- New tests belong under `test/...`.
- New workflow/provenance helpers belong under `agentic_tools/...`, not inside `src/twinr/...`.
- Do not put product runtime code into `agentic_tools/`.
- Do not put OS/bootstrap logic into `src/twinr/hardware/`.

## Core Entrypoints

- `src/twinr/__main__.py`
  - main CLI entrypoint
- installed script:
  - `twinr`

Useful runtime commands:

- `PYTHONPATH=src python3 -m twinr --env-file .env --run-web`
- `PYTHONPATH=src python3 -m twinr --env-file .env --run-hardware-loop --loop-duration 15`
- `PYTHONPATH=src python3 -m twinr --env-file .env --run-realtime-loop --loop-duration 15`
- `PYTHONPATH=src python3 -m twinr --env-file .env --display-test`
- `python3 hardware/buttons/probe_buttons.py --env-file .env --configured --duration 15`
- `python3 hardware/display/display_test.py --env-file .env`

Only run hardware-affecting commands when the machine/peripherals fit the task.

## Local Tools

This repo has a local `agentic_tools` subset. From repo root, use the executables directly:

- `./agentic_tools/tools`
  - tool discovery
- `./agentic_tools/tasks`
  - task board for multi-step work
- `./agentic_tools/chat`
  - durable repo-local coordination/history
- `./agentic_tools/hypothesis`
  - structured root-cause or design hypotheses
- `./agentic_tools/scripthistory`
  - material script/workflow history
- `./agentic_tools/scriptinfo`
  - cross-store context for one path
- `./agentic_tools/findings_cli`
  - issue/smell intake
- `./agentic_tools/fixreport_cli`
  - post-fix memory and closure
- `./agentic_tools/healthstream_cli`
  - structured health signals
- `./agentic_tools/report`
  - durable experiment/benchmark/project reports

Default tool stores:

- `artifacts/stores/tasks/board.json`
- `artifacts/stores/chat/chatlog.json`
- `artifacts/stores/hypothesis/hypotheses.yml`
- `artifacts/stores/scripthistory/scripthistory.json`
- `artifacts/stores/findings/`
- `artifacts/stores/fixreport/`
- `artifacts/stores/healthstream/healthstream.json`
- `artifacts/stores/report/`
- `artifacts/reports/report/<report_id>/`

Recommended usage:

- Start with `./agentic_tools/tools list` if you need discovery.
- Use `tasks` for larger multi-step work.
- Use `scriptinfo` before changing an important path.
- Use `hypothesis` for non-trivial debugging or design reasoning.
- Use `findings_cli` plus `fixreport_cli` for durable bug workflows.
- Use `scripthistory` when a script, workflow, or setup path materially changes.

## Local Skills

Project-local skills live under `.codex/skills/`.

Use these by default:

- `$bugfixing`
  - default bug/regression entrypoint
- `$systematic-debugging`
  - strict root-cause investigation when behavior is unclear
- `$bugfinding`
  - findings/task/fixreport closure flow
- `$instrument`
  - add evidence for timing-sensitive, flaky, or opaque behavior
- `$new_module`
  - add a substantial new capability in the correct repo location
- `$refactor`
  - structural changes without intended behavior changes
- `$design`
  - accessibility, local dashboard UX, printer output UX, display-state clarity
- `$deep-research`
  - current provider, hardware, accessibility, or implementation research

## Validation

Use targeted validation, not random broad runs.

Typical tests:

- `PYTHONPATH=src pytest test/test_config.py`
- `PYTHONPATH=src pytest test/test_runner.py`
- `PYTHONPATH=src pytest test/test_realtime_runner.py`
- `PYTHONPATH=src pytest test/test_openai_backend.py`
- `PYTHONPATH=src pytest test/test_openai_realtime.py`
- `PYTHONPATH=src pytest test/test_display.py`
- `PYTHONPATH=src pytest test/test_display_service.py`
- `PYTHONPATH=src pytest test/test_printer.py`
- `PYTHONPATH=src pytest test/test_web_app.py test/test_web_store.py`
- `PYTHONPATH=src pytest test/test_state_machine.py`

Choose the narrowest test set that proves the change.

## GitHub And Workspace Safety

You may be operating via `sudo codex --yolo` in `/home/thh/twinr`.

Rules:

- GitHub usage is allowed here.
- Web research, GitHub documentation checks, and cloning OSS references are allowed when useful.
- Do not use destructive cleanup commands in a shared worktree unless the user explicitly requests the exact command.
- Do not delete or overwrite unrelated changes.
- Do not use broad reset patterns like `git restore`, `git checkout --`, `git clean`, or `rm -rf` on repo paths unless explicitly instructed.

## Working Style

- Read existing code before adding new code.
- Reuse the current patterns unless they are actively wrong for Twinr.
- Keep code understandable to the next engineer.
- Keep user-visible behavior simple.
- Optimize for reliability and clarity over novelty.

When in doubt, ask:

1. Does this make the device easier to use?
2. Does this make the device more reliable?
3. Does this keep maintenance simple?

If the answer is no, change direction.
