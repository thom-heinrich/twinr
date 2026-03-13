# Agentic Tools For `twinr`

This folder contains the subset of the Tessairact agentic tooling that is directly useful for work on the `twinr` voice assistant and portable enough to run in this repo.

Included tools:
- `tools`: local discovery/help for the copied toolset.
- `tasks`: shared task board for larger multi-step work.
- `chat`: durable repo-local discussion log.
- `hypothesis`: structured debugging and validation hypotheses.
- `scripthistory`: history/knowledge log for setup and runtime scripts.
- `scriptinfo`: cross-store timeline for one script/path.
- `findings_cli`: bug/smell intake for issues worth tracking.
- `fixreport_cli`: post-fix memory for recurring failures.
- `report`: durable experiment/benchmark/project reports.
- `healthstream_cli`: structured health/ops event stream.

Support modules copied as dependencies:
- `agentic_tools/_store_layout.py`
- `agentic_tools/_governance_locking.py`
- `agentic_tools/codexctx.py`
- `agentic_tools/hint_engine.py`
- `agentic_tools/process_hints.py`
- `agentic_tools/hints/`
- `common/links/tessairact_meta.py`

Intentionally not copied:
- `service_health_check`: tied to Tessairact analytics/infra modules.
- `qwen3tts`: GPU-host specific and not a clean fit for the Raspberry Pi repo right now.
- heavier repo-integration tools like `code_graph`, `rules`, `ssot`, `research`, which would pull in significantly more repo internals than needed here.

Notes:
- Package-backed tools keep their module directories (`findings/`, `fixreport/`, `healthstream/`) plus executable wrappers (`findings_cli`, `fixreport_cli`, `healthstream_cli`).
- Running commands from the repo root works best because the wrappers resolve imports relative to the current repo checkout.
