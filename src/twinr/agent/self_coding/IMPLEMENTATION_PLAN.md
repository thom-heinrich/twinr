# Adaptive Skill Engine Implementation Plan

Status: Draft v0.1  
Updated: 2026-03-16  
Source: [REQUIREMENTS.md](./REQUIREMENTS.md)

## Ziel

Dieses Dokument übersetzt die ASE-Anforderungen in konkrete Arbeitspakete, die
direkt als Tickets oder Milestones abgearbeitet werden können.

Die Umsetzung folgt zwei klar getrennten Ausbaustufen:

1. **MVP: automation-first**
   Neue Fähigkeiten werden, wo möglich, in bestehende Twinr-Automationsartefakte
   kompiliert und über bestehende Runtime-Pfade aktiviert.
2. **V2: sandboxed skills**
   Für komplexere Cross-Module-Logik kommt ein eigener, hart eingegrenzter
   Skill-Runner mit Proxy-Capabilities und Brokered Effects dazu.

## Festgezogene Architekturentscheidungen

- **Lokaler Coding-Driver:** Primär `codex app-server` via `stdio`, sekundär
  `codex exec --json` als Fallback-Adapter.
- **Kein generierter Produktcode unter `src/`:** Generierte Nutzerartefakte
  liegen in einem dedizierten Runtime-Store, nicht im führenden Source-Tree.
- **Strict SOC:** `self_coding` wird als eigenes Subsystem unter
  `src/twinr/agent/self_coding/` aufgebaut; bestehende Runner bleiben dünne
  Orchestrierer.
- **MVP-Capabilities nur auf vorhandener Runtime-Basis:** Start mit
  `camera`, `pir`, `speaker`, `llm_call`, `memory`, `scheduler`, `rules`,
  `safety`, `email`, `calendar`. WhatsApp und Smart Home sind nicht MVP.
- **Pi-first Acceptance:** Jeder runtime-wirksame Meilenstein gilt erst nach
  Deployment nach `/twinr` und einem schmalen Pi-Beweis als abgeschlossen.

## Bekannte Ist-Lage im Repo

- Twinr hat bereits eine stabile Tool-Registry und Tool-Handler-Struktur.
- Twinr hat ein bestehendes Automationsmodell mit `say`, `print`,
  `llm_prompt` und modelliertem `tool_call`.
- Der Runtime-Pfad führt `tool_call` aktuell noch nicht aus; das ist eine
  explizite Umsetzungslücke für ASE.
- Email- und Kalender-Integrationen existieren bereits; WhatsApp derzeit nicht.
- Strukturierte OpenAI-Responses mit JSON-Schema werden bereits in mehreren
  Pfaden eingesetzt und sollten wiederverwendet werden.
- `codex-cli` ist lokal verfügbar und damit als lokaler Driver realistisch.

## Zielbild nach Releases

### Release 1: Lernangebot bis Aktivierung

- Twinr erkennt lernbare Fähigkeitswünsche.
- Twinr stellt maximal drei Fragen.
- Twinr startet einen lokalen Codex-Hintergrundjob.
- Einfache Fälle werden in validierte Automationsartefakte kompiliert.
- Der Nutzer macht einen Soft-Launch-Test und aktiviert die Fähigkeit.

### Release 2: Operator- und Stabilitätspfad

- Dashboard zeigt Jobstatus, Fehler, Versionen, Pause/Rollback und Health.
- Skills/Aktivierungen haben Versionierung, Rollback und Auto-Pause.

### Release 3: Komplexe sandboxed Skills

- Skills laufen in separatem Prozess mit Trusted Loader, Proxy-Capabilities,
  Broker und Watchdog.
- ASE kann auch Fähigkeiten kompilieren, die nicht auf bestehende Automations-
  primitiven reduzierbar sind.

## Arbeitspakete

### AP1: Self-Coding-Paket und Kern-Contracts anlegen

**Ziel**

Die konzeptionelle ASE-Idee in ein klares Runtime-Subsystem mit stabilen
Contracts, Zuständen und Artefaktformaten übersetzen.

**Umfang**

- Paketstruktur unter `src/twinr/agent/self_coding/` anlegen.
- Zentrale Datamodelle definieren:
  `SkillIntentCandidate`, `SkillSpec`, `FeasibilityResult`, `CompileJob`,
  `CompileArtifact`, `ActivationRecord`, `SkillHealthSnapshot`.
- Statusmodell für Jobs und Skills festlegen:
  `draft`, `questioning`, `queued`, `compiling`, `validating`,
  `soft_launch_ready`, `active`, `paused`, `disabled`, `failed`, `retired`.
- Dedizierten Artefakt-Store unter dem Projekt-Root definieren.

**Geplante Dateien**

- `src/twinr/agent/self_coding/__init__.py`
- `src/twinr/agent/self_coding/contracts.py`
- `src/twinr/agent/self_coding/store.py`
- `src/twinr/agent/self_coding/status.py`

**Abhängigkeiten**

- Keine.

**Abnahme**

- Alle Kernobjekte sind versioniert und JSON-serialisierbar.
- Artefakt- und Jobstatus können ohne freie Textkonventionen persistiert werden.
- Der Package-Scope ist klar und mischt keine Workflow- oder Providerlogik ein.

### AP2: Capability Registry und Module Library MVP

**Ziel**

Eine deterministische Capability-Sicht schaffen, auf der Feasibility-Checks und
der Coding-Agent verlässlich arbeiten können.

**Umfang**

- Capability Registry mit Modulstatus, Konfigurationsstatus, Risk-Class und
  erlaubten Kombinationen bauen.
- Trusted Module Library für das MVP bereitstellen.
- Jedes Modul bekommt einen klaren Docstring-Header als Codex-lesbare API.
- Bestehende Twinr-Funktionen nur kapseln, nicht duplizieren.

**MVP-Module**

- `camera.py`
- `pir.py`
- `speaker.py`
- `llm_call.py`
- `memory.py`
- `scheduler.py`
- `rules.py`
- `safety.py`
- `email.py`
- `calendar.py`

**Geplante Dateien**

- `src/twinr/agent/self_coding/capability_registry.py`
- `src/twinr/agent/self_coding/modules/*.py`

**Abhängigkeiten**

- AP1.

**Abnahme**

- Registry kann `available`, `configured`, `missing`, `blocked` unterscheiden.
- Module liefern nur semantische Primitive, keine Rohhardware oder freien I/O.
- Email und Kalender spiegeln die bestehende Integrations-Readiness korrekt.

### AP3: Phase 0-2 umsetzen: Trigger Detection, Feasibility, Requirements Dialogue

**Ziel**

Den Front-Stage-Flow implementieren, mit dem Twinr Lernwünsche erkennt,
vorsichtig anbietet und in einen strukturierten `SkillSpec` überführt.

**Umfang**

- Trigger-Erkennung für explizite Requests, implizite Frustration und beobachtete
  Routinen.
- Suppression-Logik und Cooldowns.
- Deterministischen Feasibility-Checker gegen Capability Registry bauen.
- State-Machine für maximal drei Fragen implementieren.
- Strukturierte Constraint-Extraktion per JSON-Schema-LLM-Calls.

**Geplante Dateien**

- `src/twinr/agent/self_coding/triggers.py`
- `src/twinr/agent/self_coding/feasibility.py`
- `src/twinr/agent/self_coding/dialogue.py`
- `src/twinr/agent/self_coding/spec_builder.py`

**Abhängigkeiten**

- AP1
- AP2

**Abnahme**

- Feasibility liefert nur `green`, `yellow`, `red` plus strukturierte Gründe.
- Dialog endet deterministisch nach höchstens drei Fragen.
- `SkillSpec` ist vollständig genug für Compilation oder klare Ablehnung.

### AP4: Realtime- und Workflow-Integration für den Lernpfad

**Ziel**

Den normalen Twinr-Agent so anbinden, dass ASE als eigenes Subsystem genutzt
wird und nicht als ungeordnete Zusatzlogik in bestehende Runner hineinwächst.

**Umfang**

- Neue Tool-Handler für Self-Coding anlegen, nicht bestehende Handler aufblasen.
- Tool-Registry und Executor um Self-Coding-Operationen erweitern.
- Runtime-/Workflow-Hooks für:
  Lernangebot, Antwort auf Lernfragen, Aktivierungsbestätigung, Pause, Status.
- Nutzertexte seniorengerecht und knapp halten.

**Geplante Dateien**

- `src/twinr/agent/tools/handlers/self_coding.py`
- `src/twinr/agent/tools/runtime/executor.py`
- `src/twinr/agent/tools/runtime/registry.py`
- gezielte Hooks in `src/twinr/agent/workflows/...`

**Abhängigkeiten**

- AP3

**Abnahme**

- Der Lernpfad ist als eigene Tool-/Runtime-Schicht adressierbar.
- Runner enthalten nur Orchestrierung und delegieren ASE-Logik an `self_coding`.
- Bestehende Gesprächs- und Tool-Flows regressieren nicht.

### AP5: Lokalen Codex-Driver und Compile-Worker bauen

**Ziel**

Einen stabilen lokalen Background-Worker schaffen, der Codex für
Codegenerierung, Tests und Review nutzen kann.

**Umfang**

- Primären Driver für `codex app-server` via `stdio` implementieren.
- Fallback-Driver für `codex exec --json` implementieren.
- Workspace-Builder für temporäre Compile-Workspaces.
- Progress-, Cancel-, Timeout- und Artefakt-Sammlung.
- Klare Fehlerklassen: Driver unavailable, compile failed, validation failed,
  timeout, unsafe artifact.

**Geplante Dateien**

- `src/twinr/agent/self_coding/codex_driver/app_server.py`
- `src/twinr/agent/self_coding/codex_driver/exec_fallback.py`
- `src/twinr/agent/self_coding/codex_driver/workspace.py`
- `src/twinr/agent/self_coding/worker.py`

**Abhängigkeiten**

- AP1

**Abnahme**

- Ein Testjob kann lokal an Codex übergeben werden und liefert strukturierte
  Events und Artefakte zurück.
- Driver-Ausfälle werden klar gemeldet und führen nicht zu hängenden Jobs.
- Compile-Workspaces sind bounded und getrennt vom führenden Source-Tree.

### AP6: Compiler-Pipeline und Compile-Targets

**Ziel**

Den Coding-Agenten nicht direkt “frei coden” lassen, sondern in klare
Compile-Ziele, Eingaben und Validierungsschritte zwingen.

**Umfang**

- Prompt-Paket aus `SkillSpec`, Modul-Docstrings und bis zu drei Beispielen
  zusammensetzen.
- Compile-Target `automation_manifest` für MVP bauen.
- Compile-Target `skill_package` als V2-Platzhalter definieren.
- Simple/complex routing: einfacher Fall -> Automation, komplexer Fall ->
  Skill-Package oder klare Ablehnung.
- Statische Artefaktprüfung für generierte Outputs.

**Wichtige MVP-Entscheidung**

- Wenn ein Wunsch sauber in bestehende Automationsprimitive passt, wird
  **kein** freies Skill-Skript generiert.
- Nur wenn der Wunsch nicht auf `say` / `print` / `llm_prompt` /
  klar definierte Tool-Aufrufe abbildbar ist, wird V2 relevant.

**Geplante Dateien**

- `src/twinr/agent/self_coding/compiler/prompts.py`
- `src/twinr/agent/self_coding/compiler/examples.py`
- `src/twinr/agent/self_coding/compiler/automation_target.py`
- `src/twinr/agent/self_coding/compiler/skill_target.py`
- `src/twinr/agent/self_coding/compiler/validator.py`

**Abhängigkeiten**

- AP2
- AP5

**Abnahme**

- Ein einfacher `SkillSpec` wird deterministisch in ein validierbares
  Automationsartefakt übersetzt.
- Nicht unterstützte Fälle werden sauber als `unsupported_for_mvp` markiert.
- Es gibt keine direkte Generierung von Runtime-Code unter `src/`.

### AP7: Aktivierungspfad, Soft Launch, Rollback und Health

**Ziel**

Aus einem erfolgreichen Compile-Job eine sichere, nachvollziehbare Aktivierung
mit Nutzerbestätigung machen.

**Umfang**

- Soft-Launch-Status und Dry-Run-Pfad.
- Aktivierung erst nach expliziter Bestätigung.
- Versionierte Artefakte und Rollback auf vorherige Version.
- Health-Counter für Fehler, Trigger, Interrupts, Denials.
- Auto-Pause bei schlechter Skill-Gesundheit.

**Wichtige technische Lücke**

- Prüfen und entscheiden, ob `AutomationAction.kind == "tool_call"` für ASE
  im MVP wirklich benötigt wird.
- Falls ja: Runtime-Ausführung dafür sauber implementieren.
- Falls nein: MVP explizit auf `say`, `print`, `llm_prompt` begrenzen.

**Geplante Dateien**

- `src/twinr/agent/self_coding/activation.py`
- `src/twinr/agent/self_coding/rollback.py`
- `src/twinr/agent/self_coding/health.py`
- gezielte Runtime-Anbindung in Automations-/Background-Pfaden

**Abhängigkeiten**

- AP6

**Abnahme**

- Ohne Nutzerbestätigung wird nichts produktiv geschaltet.
- Vorherige Versionen bleiben für Rollback erhalten.
- Health-Signale können Pause/Disable auslösen, ohne die Runtime zu destabilisieren.

### AP8: Web- und Operator-Oberfläche

**Ziel**

ASE für Betreiber und Vertrauenspersonen sichtbar und wartbar machen.

**Umfang**

- Statusseite oder Dashboard-Sektion für Self-Coding-Jobs.
- Sicht auf:
  Jobstatus, letzter Fehler, aktive Fähigkeiten, Soft-Launch-Warteschlange,
  Versionen, Health, letzte Broker-Denials.
- Operator-Aktionen:
  pausieren, reaktivieren, rollback, löschen, guided test erneut starten.

**Geplante Dateien**

- `src/twinr/web/app.py`
- `src/twinr/web/templates/...`
- `src/twinr/web/viewmodels_*.py`

**Abhängigkeiten**

- AP7

**Abnahme**

- Operator kann den kompletten Lebenszyklus einer gelernten Fähigkeit sehen.
- Fehler sind sichtbar und nicht nur in Logs versteckt.
- Die UI bleibt für nichttechnische Betreiber einfach und ruhig.

### AP9: V2 Sandbox-Runner für komplexe Skills

**Ziel**

Das in den Anforderungen beschriebene Modell für komplexe Fähigkeiten mit
prozessisolierten, capability-basierten Skills umsetzen.

**Umfang**

- AST-Validator für das erlaubte Python-Subset.
- Trusted Loader mit Allowlist-Imports und Proxy-Stubs.
- Capability Broker für alle externen Effekte.
- Separater Skill-Prozess mit Timeout-/Memory-Watchdog.
- Schrittweise Härtung:
  1. Prozessgrenze und Proxy-Stubs
  2. Restricted builtins
  3. harte OS-Grenzen wie seccomp / Landlock

**Geplante Dateien**

- `src/twinr/agent/self_coding/sandbox/ast_validator.py`
- `src/twinr/agent/self_coding/sandbox/trusted_loader.py`
- `src/twinr/agent/self_coding/sandbox/broker.py`
- `src/twinr/agent/self_coding/sandbox/skill_runner.py`
- `src/twinr/agent/self_coding/sandbox/watchdog.py`

**Abhängigkeiten**

- AP6
- AP7

**Abnahme**

- Ein Skill kann nur Proxy-Capabilities aufrufen, keine freien Imports und
  keinen direkten I/O durchführen.
- Broker-Denials, Timeouts und Memory-Limits sind testbar und auditierbar.
- Ein fehlerhafter Skill kann die Twinr-Hauptlaufzeit nicht korrupt hinterlassen.

### AP10: Teststrategie, Pi-Deployment und Akzeptanzbeweise

**Ziel**

Die ASE-Umsetzung vom ersten Tag an so absichern, dass sie auf der Pi-Hardware
und im realen Runtime-Kontext belastbar bleibt.

**Umfang**

- Unit-Tests für Contracts, Registry, Feasibility, Dialogue, Driver-Adapter.
- Integrations-Tests mit Fake-Codex-Driver.
- Opt-in Smoke-Test mit echtem lokalem `codex-cli`.
- Pi-Akzeptanzfälle für:
  Lernangebot, Fragefluss, Compile-Job, Soft Launch, Aktivierung, Pause.
- Deployment-Schritte nach `/twinr` dokumentieren.

**Geplante Tests**

- `test/test_self_coding_contracts.py`
- `test/test_self_coding_registry.py`
- `test/test_self_coding_feasibility.py`
- `test/test_self_coding_dialogue.py`
- `test/test_self_coding_worker.py`
- `test/test_self_coding_activation.py`
- `test/test_self_coding_web.py`

**Abhängigkeiten**

- AP1 bis AP8 für MVP
- AP9 für V2-Sandbox-Abnahme

**Abnahme**

- Es gibt einen schmalen, wiederholbaren Pi-Beweis für jeden MVP-Meilenstein.
- Lokale Tests und Pi-Tests sind getrennt dokumentiert.
- Kein Arbeitspaket mit Runtime-Effekt wird ohne Pi-Nachweis als fertig markiert.

## Empfohlene Umsetzungsreihenfolge

1. AP1
2. AP2
3. AP3
4. AP4
5. AP5
6. AP6
7. AP7
8. AP10 für MVP-Abnahme
9. AP8
10. AP9

## Empfohlene MVP-Grenze

Der erste produktionsnahe Schnitt sollte **nicht** versuchen, das komplette
Sandbox-Modell aus Section 7 der Anforderungen sofort zu liefern.

**MVP umfasst:**

- Lernpfad
- Feasibility
- lokaler Codex-Compile-Worker
- Automation-first Compile-Target
- Soft Launch
- Aktivierung/Rollback/Health
- Operator-Sicht

**V2 umfasst:**

- freie komplexe Skill-Skripte
- AST-Subset
- Trusted Loader
- Brokered Effects
- seccomp / Landlock

## Offene Architekturentscheidungen

- Braucht der MVP `tool_call`-Automationsausführung, oder reicht eine harte
  Begrenzung auf `say` / `print` / `llm_prompt`?
- Wo genau liegt der Artefakt-Store im Projekt-Root, damit Runtime und Web
  sauber darauf zugreifen können?
- Welche bestehenden Beispiele eignen sich als Compile-Few-Shots für Codex?
- Wie aggressiv darf Phase-0-Proaktivität bei Senioren standardmäßig sein?
- Wann wird ein `yellow`-Feasibility-Fall dem Nutzer angeboten und wann intern
  zurückgestellt?

## Definition of Done für den ersten ASE-Release

- Twinr kann einen lernbaren Wunsch erkennen und sauber anbieten.
- Twinr kann maximal drei Fragen stellen und daraus einen `SkillSpec` bauen.
- Ein lokaler Codex-Job kann daraus ein validiertes Automationsartefakt bauen.
- Der Nutzer kann die Fähigkeit im Soft Launch testen und explizit aktivieren.
- Die Fähigkeit ist versioniert, pausierbar und rollback-fähig.
- Der komplette Flow ist lokal und auf `/twinr` mit einem schmalen Beweis gelaufen.
