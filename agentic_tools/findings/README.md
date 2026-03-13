# `findings` — actionable issue findings (pre-fix)

This tool persists **Findings** under `artifacts/stores/findings/` (legacy: `state/findings/`):

- `artifacts/stores/findings/reports/FND*.yml` — 1 Finding per YAML file
- `artifacts/stores/findings/index.yml` — lightweight index for list/search
- `artifacts/stores/findings/events.yml` — append-only event log (created/updated)

Findings are meant to be created by skills like `bugfinding`, `intersect`, and
`quantaudit` **before** a fix exists. When a fix is implemented, the Finding
should be linked to the corresponding FixReport (`fixreport:BFxxxxxx`) and/or
Tasks.

## Findings vs Hypotheses (trennscharf)

- **Finding** = *observed* Fehler/Smell/Risiko, das adressiert werden soll (pre-fix Intake + Evidence + Target).
  - Root cause darf UNKNOWN sein; Finding ist **kein** “Wissens-Claim”, sondern eine actionable Liste.
- **Hypothesis** = *Theorie / falsifizierbare Aussage* über Ursache/Wirkung (mit Evidence, die supporting/contradicting/inconclusive sein kann).
  - Hypothesen werden supported/refuted; Findings werden taskified/behoben.

**Linking pattern (empfohlen):**
- Wenn eine Hypothese ein Finding erklärt: `links: ["finding:FND…", "hypothesis:H…"]` (beidseitig).
- Wenn ein Finding taskified ist: Task bekommt `--link finding:<FND…>` und das Finding bekommt `task:<id>` als Link.
- Wenn ein Finding gefixt ist: FixReport bekommt `--link finding:<FND…>` und das Finding bekommt `fixreport:<BF…>` + `finding.status=resolved`.

## Addressed (✓) vs Resolved

- `finding.addressed=true` = Finding wurde bereits in Task/UserTask/Work überführt (checkmark) → **nicht** “behoben”.
- “Behoben” = `finding.status=resolved` + `resolution.*` + FixReport-Link.

## Links contract

Findings support the cross-tool `links` graph:

```yaml
links:
  - "script:services/analytics/foo.py"
  - "task:qa:bundle:main:T0001"
  - "fixreport:BF000372"
  - "hypothesis:H00462"
```

See: `architecture/LINKS_CONTRACT.md`.

## CLI

Examples:

```bash
findings init
findings create --repo-area services --scope analytics --mode dev --severity p2 \
  --target-kind script --target-path services/example.py \
  --title "Wrong units" --summary "bps treated as fraction" \
  --link script:services/example.py
findings lint
findings rebuild-index
findings list --severity p1
findings get --id FND000001
```
