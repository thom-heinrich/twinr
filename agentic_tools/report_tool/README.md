# `report` (agentic tool)

Purpose: write durable Abschlussberichte / research reports as hybrid artefacts:

- machine: `artifacts/stores/report/` (YAML reports + `index.yml` + `events.yml`)
- human: `artifacts/reports/report/<RPT...>/report.md` (+ `assets/`)

All cross-tool references are stored as truth edges in `tessairact_meta.links`
(`architecture/LINKS_CONTRACT.md`).

Quickstart:

1) `report init`
2) `report template --format yaml > /tmp/report.yml`
3) edit `/tmp/report.yml`
4) `report create --data-file /tmp/report.yml`
5) `report finalize --id <report_id>`

Quality gates:

- `report create` and `report finalize` reject minimal reports.
- Required sections include rich `summary`, `insights`, `context`, `results` (with KPI explanations), `evidence`, `limitations`, `risks`, `next_actions`, and `repro`.
- For benchmark/test-like reports, KPI and test coverage is stricter (multiple KPIs + explicit benchmark/test entries).
- Use `report schema` to inspect the current machine-readable policy (`quality_policy`).
