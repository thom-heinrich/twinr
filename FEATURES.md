# Twinr Features

Diese Datei beschreibt, was Twinr aktuell bereits kann.

## Kerninteraktion

- Sprachinteraktion per gruenem Hardware-Button
- Druck der letzten Antwort per gelbem Hardware-Button
- Aufnahme bis kurze Sprechpause statt manuellem Stoppen
- Gesprochene Antwort ueber Lautsprecher
- Kurze, druckfreundliche Antwortfassung fuer den Thermodrucker

## Senior-friendly Geraeteverhalten

- Sehr einfache Zwei-Button-Interaktion
- E-Paper-Gesicht mit klaren Zustaenden: warten, zuhoeren, verarbeiten, sprechen, drucken, Fehler
- Kleine Health-Zeile unter dem Gesicht mit `Internet`, `AI`, `System` und Uhrzeit
- Ruhige, haptische Interaktion statt App-first-Bedienung

## AI- und Assistenzfunktionen

- Speech-to-Text, LLM-Antworten und Text-to-Speech
- Realtime-Sprachmodus mit direkter Audio-Ein-/Ausgabe
- Optionaler Web-Search fuer aktuelle Fragen
- Kamerabild-Analyse fuer visuelle Anfragen
- Tool-gestuetzte Druckauftraege aus dem Realtime-Flow
- Tool-gestuetztes Speichern expliziter Erinnerungen
- Tool-gestuetztes Anlegen von Remindern und Timern

## Memory

- Rolling On-Device Conversation Memory
- Strukturierte Kurzzeit-Memory mit:
  - aktuelle Turns
  - kompakter Ledger-Zusammenfassung
  - Search-Ergebnisse
  - Runtime-State wie Thema, Nutzerziel, pending printable
- Explizit gespeicherte Langzeit-Merkpunkte in `state/MEMORY.md`
- Getrennter Reminder-/Timer-Store in `state/reminders.json`
- Wiederherstellung des Runtime-State aus Snapshot nach Neustart

## Lokales Voice-Profil

- Lokales Stimmprofil auf Basis des Hauptmikrofons
- Softes Vertrauenssignal statt harter Authentifizierung
- Status wie `likely user`, `uncertain` oder `unknown voice`
- Enrollment, Verifikation und Reset ueber das lokale Portal
- Enrollment, Statusabfrage und Reset auch per gesprochenem Realtime-Tool
- Sprecher-Signal wird redigiert in den LLM-Kontext eingespeist
- Unsichere Sprecherlage fuehrt bei persistenten Aenderungen zu extra Bestätigung
- Keine Roh-Audio-Speicherung im Profil-Store
- Voice-Status wird aus Support-Bundles redigiert

## Proaktive Funktionen

- PIR-Motion-Sensor als Presence-/Wake-Signal
- Vorsichtige proactive Social Triggers bei Idle-Zeit
- Begrenzte proaktive Kamera-Beobachtung
- Begrenztes proaktives Ambient-Audio-Sampling
- Cooldowns und bounded behavior, damit Twinr nicht spammt

## Hardware-Unterstuetzung

- GPIO-Buttons
- PIR-Bewegungssensor
- USB-Mikrofon / USB-Lautsprecher
- Thermal-Printer
- Waveshare 4.2" E-Paper-Display
- Kamera fuer Still-Image-Captures

## Lokales Portal / Caregiver UI

- Dashboard auf dem Geraet
- Connect-Seite fuer Provider-/Netzwerk-Settings
- Settings-Seite fuer Runtime-, Hardware-, Print-, Display- und Memory-Parameter
- Personality-Seite
- User-Seite
- Memory-Seite fuer Live-Memory und durable Memory

## Ops / Stabilitaet / Wartung

- `/ops/self-test` fuer Mic, proactive Mic, Speaker, Printer, Kamera, Buttons, PIR
- `/ops/logs` mit lokalen Twinr-Events
- `/ops/usage` mit LLM-Nutzung, Modell, Request-IDs und Tokenzahlen wenn verfuegbar
- `/ops/health` mit Systemzustand, Services, CPU-Temperatur, RAM, Disk, Uptime
- `/ops/config` fuer Plausibilitaetschecks der Konfiguration
- `/ops/support` fuer redigierte Support-Bundles
- Persistentes Event-Log unter `artifacts/stores/ops/events.jsonl`

## Produktgrenzen, die bewusst eingehalten werden

- Green Button bleibt immer: sprechen / fragen
- Yellow Button bleibt immer: letzte Antwort drucken
- Druck bleibt kurz und bounded
- Hardware-Fehler sollen klar sichtbar sein statt stiller Fallbacks
- Twinr bleibt lokal betreibbar mit einfacher Web-Wartung
