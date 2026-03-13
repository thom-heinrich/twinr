# FixReport (CAIA) — agentic bugfix memory

`fixreport` is a small, agent-facing tool to **persist bugfix learnings** in a
structured, searchable way so we stop re-debugging the same classes of bugs
from scratch.

It is intentionally **analogous to `hypothesis`**:

- **YAML** for storage under `artifacts/stores/fixreport/` (human diffable, non‑JSON storage; legacy: `state/fixreport/`).
- **JSON-only stdout** for agent/tool consumption.
- **Fail-fast** validation against a controlled vocabulary.
- **File locking** to avoid concurrent corruption.

## Data model

The controlled vocabulary lives in `agentic_tools/fixreport/vocab.yaml`.

Storage layout:

- `artifacts/stores/fixreport/reports/BF000001.yml` — one YAML file per fixreport.
- `artifacts/stores/fixreport/index.yml` — lightweight index for listing/search.
- `artifacts/stores/fixreport/events.yml` — append-only event log (created, updated).

Each report file contains:

- `fixreport`: the structured fields (tokens/enums) defined in `vocab.yaml`.
- `failure_modes` (**strongly recommended**): a standardized checklist of recurring failure-modes derived from the `$failure_modes` skill. The catalog lives in `agentic_tools/fixreport/failure_modes_catalog.yaml`.
- `narrative`: optional human text (what happened, root cause detail, fix, validation).
- `evidence`: optional list of file/log/command references.

## Mandatory: Meta anchoring via `links`

For the Portal Meta graph to render **truth edges** across Tasks/Hypotheses/FixReports/Audits/etc,
FixReports must include at least one explicit repo anchor as a `script:<repo_relative_path>` token
in `fixreport.links` (see `architecture/LINKS_CONTRACT.md`).

Ways to satisfy this:

- Provide explicit anchors:
  - `--script services/.../my_file.py` (repeatable)
  - or `--link script:services/.../my_file.py`
- Or rely on inference: when no script links are provided, `fixreport new` will try to infer anchors
  from `--target-path` / `--paths-touched` / `--systemd-unit` if they resolve to repo files.

## CLI

The recommended CLI entrypoint is the shell command `fixreport` (a thin wrapper
around `agentic_tools/fixreport_cli`).

Examples:

- `fixreport schema --json-pretty`
- `fixreport template --json-pretty`
- `fixreport fm --json-pretty` (failure-modes checklist catalog)
- `fixreport init --json-pretty`
- `fixreport new --repo-area services --target-kind service --target-path services/x.py --script services/x.py --scope analytics --mode live --bug-type logic --symptom wrong_output --root-cause wrong_units --failure-mode corrupt_output --fix-type unit_fix --impact-area correctness --severity p1 --verification unit_test --title "unit mismatch in cost bps" --fix "convert bps→fraction" --validation "pytest -q ..."`
- `fixreport new ... --fm FM_HORIZON_DECLARED=pass --fm FM_TIME_SEMANTICS_EVENT_VS_INGEST=pass --fm-notes "validated on replay window 2025-12-31T..."`
- `fixreport new ... --fm-policy strict --fm FM_TIME_SEMANTICS_EVENT_VS_INGEST=pass --fm FM_SCHEMA_CONTRACT_DRIFT_GUARDED=pass --fm FM_COST_SEMANTICS_DECLARED=pass --fm FM_HEARTBEATS_EXCLUDED=pass`
- `fixreport search --root-cause wrong_units --json-pretty`
- `fixreport search --fm FM_TIME_SEMANTICS_EVENT_VS_INGEST=fail --json-pretty`
- `fixreport fm-backfill --apply --json-pretty` (one-time: extend historical FixReports with new FM ids as unknown and infer a few obvious fails from root_cause)
- `fixreport search --since 2025-12-24T00:00:00Z --until 2025-12-31T23:59:59Z --json-pretty`
- `fixreport normalize --apply --json-pretty` (one-time cleanup: split comma-joined list fields like `tags`, `topics`, `paths_touched`)
- `fixreport update --bf-id BF000123 --set-failure-modes-json '{"checks":{"FM_DEFAULT_ZERO_DOMINANCE":"fail"},"notes":"found schema drift"}' --json-pretty`

Isolated testing (does not write into `artifacts/stores/fixreport/`):

- `fixreport --state-dir /tmp/fixreport_test init --json-pretty`
- `fixreport --state-dir /tmp/fixreport_test new ... --commit deadbee --json-pretty`
