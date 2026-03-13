# healthstream (agentic tool)

Central, file-backed, JSON-only event stream for **automated health / DQ / ops** signals.

Goals:
- Replace automated `chat send ...` messages (chat is for humans/agents only).
- Provide a single query surface for agents: `healthstream list ...`.
- Keep producers lightweight and non-invasive (no ClickHouse/Kafka dependencies here).

## Store

- Default store file: `artifacts/stores/healthstream/healthstream.json`
- Override: `HEALTHSTREAM_FILE=/path/to/store.json` (repo-relative allowed)
- Locking: `<store>.lock` (atomic create/unlink)

## CLI

```bash
healthstream schema
healthstream init
healthstream emit --kind dq --status ok --source primary_dq_watchdog --artifact artifacts/stores/health/primary_dq_watchdog_last.json
healthstream list --kind dq --limit 50
```

All stdout is JSON; logs go to stderr.

## Links

Use explicit `links: ["kind:id"]` tokens to make Meta graph edges truthy:
- `script:<repo_relative_path>`
- `service_unit:<unit>`
- `topic:<kafka_topic>`
- `table:<db.table>`

Contract: `architecture/LINKS_CONTRACT.md`.
