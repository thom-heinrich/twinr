# Twinr

Twinr is a minimal digital, maximal haptic, senior-citizen-optimized physical AI agent.

The product is designed around simple physical interaction instead of app-first complexity. A user presses a green button, speaks naturally, and Twinr listens until a short pause is detected. Twinr then transcribes the utterance with a speech-to-text provider, sends the result to a foundation model, performs agentic tasks when needed, and answers via text-to-speech. A yellow button prints the latest answer as a short, easy-to-read thermal receipt.

Twinr also uses a small status display with animated eyes to communicate its current state: waiting, listening, processing, answering, printing, and error. A small footer line under the face shows a calm device-health summary such as `Betrieb ok`, `warm`, or `Achtung`.

## Core principles

- Minimal digital overhead
- Maximum tactile clarity
- Senior-friendly interaction model
- Fast feedback through buttons, voice, display, and paper
- Configurable AI behavior without exposing technical complexity

## Interaction model

### Voice interaction

1. User presses the green button
2. Twinr starts listening
3. Twinr stops listening after a short speech pause
4. Twinr sends audio to an STT provider
5. Twinr processes the request with a foundation model
6. Twinr optionally executes agentic workflows and tool-enabled tasks
7. Twinr speaks the answer via a TTS provider

### Print interaction

1. User presses the yellow button
2. A meta-LLM reformats the most recent answer into a short print-friendly version
3. Twinr prints the result on the thermal printer

## Status display

Twinr uses a display with two eyes to express state in a simple, low-cognitive-load way.

Supported states:

- Waiting
- Listening
- Processing
- Answering
- Printing
- Error

## Memory model

### On-device memory

Twinr keeps a short rolling conversation memory on-device by default. The active short-term schema is structured, not just a flat transcript:

- `raw_tail` keeps the most recent verbatim user/assistant turns
- `ledger` keeps compacted summaries and typed durable notes
- `search_results` stores recent web lookups as first-class entries
- `state` tracks the active topic, last user goal, pending printable, and open loops

When the tail grows beyond the configured cap, Twinr compacts older turns into the ledger and keeps only the newest conversational tail for replay.

By default, the live runtime snapshot is persisted to `state/runtime-state.json` instead of `/tmp`, so the local dashboard and restart flow can restore structured state after bounded restarts.

Durable remembered facts that the user explicitly asks Twinr to keep are written to `state/MEMORY.md`. This file is separate from the rolling runtime snapshot and is only meant for explicit “remember this” items, not every turn.

Scheduled reminders and timers are stored separately in `state/reminders.json`. The voice agent writes to this store only via the reminder tool, and the idle loops deliver due reminders later with bounded retry handling if speech output fails.

Basic local automations are stored in `state/automations.json`. This store is meant for explicit operator-created rules and covers two trigger families today: time-based schedules and simple if-then rules over named facts/events.

The active voice agent now exposes a dedicated automation toolset for:

- listing existing automations
- creating time-based automations
- creating sensor-triggered automations
- updating time-based automations
- updating sensor-triggered automations
- deleting automations

Twinr also exposes a bounded self-settings tool for a few operator-safe runtime changes. Today that includes:

- recent-memory capacity
- speaking pause detection
- follow-up listening window
- spoken voice
- spoken speed

Voice and speed changes are persisted back into `.env`, applied live to the running backend configuration, and kept within a narrow range so the device stays calm and understandable.

Twinr keeps a small curated spoken-voice catalog for this path instead of exposing every possible provider voice. That lets the live voice agent answer questions like “Which voices do you have?” and map descriptive requests such as “use a male voice” or “sound warmer” onto a bounded `spoken_voice` setting without opening up the whole provider surface.

Twinr also keeps a small bounded adaptive timing profile in `state/adaptive_timing.json`. This profile learns slowly from real button turns:

- no-speech timeouts widen the next start-of-speech window
- resumed speech during a pause grace window increases pause tolerance
- repeated clean, fast starts slowly tighten the timing again

The adaptive layer is intentionally conservative. It never removes the hard timing bounds, and it stays anchored to the configured base values so the device remains predictable.

Time-based automations can either:

- speak static text
- print static text
- generate spoken content from an LLM prompt
- generate printed content from an LLM prompt

For scheduled live-information jobs such as daily weather, daily news, or printed headlines, Twinr stores the automation as an `llm_prompt` action with web search enabled and executes it later from the idle loop.

Sensor-triggered automations currently support the following operator-facing trigger kinds:

- `pir_motion_detected`
- `pir_no_motion` with a required hold duration
- `vad_speech_detected`
- `vad_quiet` with a required hold duration
- `camera_person_visible`
- `camera_hand_or_object_near_camera`

The proactive monitor now feeds stable PIR, camera, and VAD facts into the automation engine while Twinr is idle. This lets voice-created if-then automations react to sensor facts without exposing raw camera frames or raw audio chunks as tool-level inputs.

Explicit requests to change future user-profile context or future speaking/behavior rules are written into managed sections inside `personality/USER.md` and `personality/PERSONALITY.md`. Twinr reloads those files on the next provider request instead of baking them permanently into code.

### Long-term memory (optional)

Twinr can optionally be extended with a ChonkyDB-based long-term memory layer. This provides durable storage of information in a combined index spanning:

- Vector search
- Graph relationships
- Full-text search
- Hash-based lookup
- Temporal access patterns

This long-term layer is intended for persistent user preferences, recalled facts, and historical conversations.

Twinr now includes a small external client baseline under `src/twinr/memory/chonkydb/`. The current basics cover instance/auth inspection, record listing, single-record fetches, full-document fetches, unified retrieval queries, single/bulk record submissions, a versioned `twinr_graph` schema v1 with namespaced typed edges for structured relationships, and a first local personal-graph layer for contacts, preferences, plans, and contact clarification.

The runtime orchestration for this path now lives separately under `src/twinr/memory/longterm/`. That layer builds a small structured long-term provider context before each turn and can persist episodic conversation memories in the background after a turn finishes, so recall can improve without blocking the response path.

The canonical Twinr env names for that path are:

- `TWINR_LONG_TERM_MEMORY_ENABLED`
- `TWINR_LONG_TERM_MEMORY_BACKEND`
- `TWINR_LONG_TERM_MEMORY_PATH`
- `TWINR_CHONKYDB_BASE_URL`
- `TWINR_CHONKYDB_API_KEY`
- `TWINR_CHONKYDB_API_KEY_HEADER`
- `TWINR_CHONKYDB_ALLOW_BEARER_AUTH`
- `TWINR_CHONKYDB_TIMEOUT_S`

For migration convenience, the config also accepts legacy `CCODEX_MEMORY_BASE_URL` and `CCODEX_MEMORY_API_KEY` as fallbacks, but Twinr docs and code should refer to the backend as `chonkydb`.

## Web interface

Twinr exposes a web interface on port `1337` for local configuration.

Example configuration areas:

- Personality and tone
- User preferences
- Automations
- Provider selection
- Device settings
- Memory settings
- Print behavior
- Display settings

The web interface should stay simple enough for caregivers, family members, or operators to maintain without developer tooling.

The current implementation uses a server-rendered local dashboard with these sections:

- Dashboard
- Devices
- LLM Usage
- System Health
- Automations
- Personality
- Memory
- Connect
- Settings
- User

The `Memory` page reads a live runtime snapshot from the active Twinr process, so operators can inspect the current on-device conversation memory without attaching to the running loop.
The `Memory` page also exposes the durable `state/MEMORY.md` store, so operators can inspect and add explicit long-lived remembered items without touching the rolling runtime snapshot.
The same `Memory` page now exposes scheduled reminders from `state/reminders.json`, including pending and delivered reminders plus simple operator controls to add, complete, or delete reminder entries.
The `Settings` page now also shows the live bounded adaptive listening profile from `state/adaptive_timing.json`, including the learned start-of-speech and pause windows plus a manual reset action.
The `Automations` page reads `state/automations.json`, groups stored rules by automation family, and exposes explicit create/edit/delete controls for scheduled and sensor-triggered rules. Integration modules now register their own reserved family blocks through an integration-side registry, so later mail/calendar/integration automations can land in their own UI sections without rewriting the core page.
The `Personality` and `User` pages now separate the hand-written base text from Twinr-managed updates, so tool-written behavior/profile changes stay visible and editable without clobbering the base files.
The `Devices` page shows the current local hardware view for printer, camera, audio, PIR, and buttons, plus the newest self-test evidence Twinr can confirm. Signals that the current path cannot expose, such as paper status on the raw USB printer path, are shown as unknown instead of guessed.
The `LLM Usage` page reads the local usage store for tracked OpenAI calls, including model names, request ids, and token counts when the provider returns them.
The `System Health` page reads live Raspberry Pi metrics and Twinr worker presence so caregivers can see whether the box itself looks healthy.
The `Ops Logs` page reads the persistent local event log under `artifacts/stores/ops/events.jsonl`, including proactive observation changes, proactive trigger detections, and spoken proactive prompts.
The `Voice Profile` page manages a local-only speaker template. Phase 1 uses the normal conversation microphone only, stores no raw enrollment audio, and exposes only a soft confidence signal such as `likely user`, `uncertain`, or `unknown voice`. Support bundles omit the live voice-assessment fields.
In realtime voice mode, Twinr can now also manage that profile by spoken request: it can enroll the current spoken turn into the local voice profile, read the current profile status, or reset the profile when the user explicitly asks for it.
The live speaker signal is injected into the LLM context as a short redacted system hint, not as raw audio or a biometric template. When the signal is `uncertain` or `unknown voice`, Twinr now asks for extra confirmation before persistent changes such as saved memory/profile/personality updates or automation changes.

## High-level architecture

### Inputs

- Green hardware button
- Yellow hardware button
- PIR motion sensor input
- Microphone audio
- Web configuration UI

### Outputs

- Speaker audio
- Thermal printer output
- Eye display status
- Local web interface

### Service layers

- Speech-to-text provider layer
- Foundation model / agent layer
- Text-to-speech provider layer
- Memory layer
- Hardware abstraction layer
- Web configuration layer

## Repository layout

- `docs/` — documentation and structured specifications
- `docs/providers/` — provider-specific notes and setup guides
- `hardware/` — Raspberry Pi setup scripts for buttons, PIR, audio, printer, and display hardware
- `personality/` — prompt-context files for system, user, and assistant style
- `state/` — runtime-generated persistent snapshot and durable remembered facts (`gitignored`)
- `src/twinr/agent/` — runtime orchestration, state machine, and hardware workflows
- `src/twinr/display/` — Waveshare display wrapper and image rendering helpers
- `src/twinr/providers/` — provider implementations such as OpenAI
- `src/twinr/memory/` — on-device memory, durable memory stores, and reminders
- `src/twinr/memory/chonkydb/` — external ChonkyDB client basics plus versioned Twinr graph schema for future long-term memory integration
- `src/twinr/memory/longterm/` — long-term retrieval and non-blocking persistence orchestration
- `src/twinr/automations/` — canonical automation definitions, evaluation, and persistent CRUD store
- `src/twinr/hardware/` — audio, button, PIR, camera, and printer adapters
- `state/voice_profile.json` — local speaker-template store for the optional voice-confidence layer
- `test/` — tests and validation assets

## Intended user experience

Twinr should feel calm, obvious, and trustworthy. A user should not need to understand models, APIs, sessions, prompts, or configuration details. The interaction should feel closer to using a helpful household object than using a computer.

## Current scope

This repository defines the foundation for:

- Button-driven voice capture
- Passive PIR motion sensing for local hardware bring-up and future presence-aware behavior
- A stateful social-trigger engine for cautious presence, attention, and safety nudges
- Provider-based STT / LLM / TTS orchestration
- Agentic task execution
- Paper-based answer printing
- Visual state feedback through eyes
- On-device memory with optional long-term expansion
- Local web-based configuration on `:1337`

## Hardware bootstrap

### Buttons

The current Raspberry Pi setup does not expose the physical Twinr buttons as dedicated Linux input devices. They should be handled as GPIO lines on `gpiochip0`.

Persist the current mapping with:

```bash
cd /twinr
hardware/buttons/setup_buttons.sh --green 23 --yellow 22
```

Probe the configured buttons with:

```bash
cd /twinr
./.venv/bin/python hardware/buttons/probe_buttons.py --env-file /twinr/.env --configured --duration 15
```

The runtime-facing GPIO helpers live in `src/twinr/hardware/buttons.py`.

### PIR motion sensor

Twinr now supports a dedicated PIR motion input for hardware validation and future presence-aware behavior.

Persist the current prototype mapping with:

```bash
cd /twinr
hardware/pir/setup_pir.sh --motion 26 --probe
```

Probe the configured PIR input with:

```bash
cd /twinr
./.venv/bin/python hardware/pir/probe_pir.py --env-file /twinr/.env --duration 30
```

The current prototype wiring is:

- `OUT -> GPIO26` on the Pi (`physical pin 37`, often labeled `IO26`)
- `GND -> GND`
- `VCC ->` the voltage required by the PIR module

Software defaults assume `active_high=true` and `bias=pull-down`. The GPIO signal presented to the Pi must stay at or below `3.3V`.

The runtime-facing PIR helpers live in `src/twinr/hardware/pir.py`.

### Social trigger engine

Twinr now has a dedicated proactive subsystem in `src/twinr/proactive/`.
It does not try to "diagnose emotions". Instead it turns explicit sensor observations into cautious prompts such as:

- `person_returned`
- `attention_window`
- `slumped_quiet`
- `possible_fall`
- `floor_stillness`
- `showing_intent`
- `distress_possible`
- `positive_contact`

The core trigger engine is stateful and enforces per-trigger cooldowns. It consumes bounded observations from PIR, audio, and camera paths, then returns at most one suggested spoken nudge at a time.
The proactive monitor now runs as its own modular package and can be enabled behind config while the normal green/yellow button behavior stays unchanged.
Current implementation notes:

- PIR is used as the cheap wake-up signal.
- The proactive path currently uses a dedicated vision classifier prompt over still frames.
- Ambient audio can now sample the configured microphone in short bounded windows, typically the PS-Eye microphone when `TWINR_PROACTIVE_AUDIO_DEVICE` points at it.
- Background audio sampling uses a shared lock with active recording so Twinr does not open the same ALSA input twice at once.
- The hardware and realtime loops expose `handle_social_trigger(...)` so the proactive monitor can speak without pretending the user pressed the green button.

### Audio

Twinr currently uses the `Jabra SPEAK 510 USB` attached to the Raspberry Pi as its default microphone and speaker.
For proactive background listening, Twinr can additionally use the `PlayStation Eye` USB microphone as a separate capture path.

Configure the audio defaults with:

```bash
cd /twinr
sudo hardware/mic/setup_audio.sh --device-match Jabra
```

To keep Jabra as the main default device while also wiring the PS-Eye microphone into the proactive path, run:

```bash
cd /twinr
sudo hardware/mic/setup_audio.sh \
  --device-match Jabra \
  --proactive-device-match Camera-B4.09.24.1 \
  --proactive-sample-ms 900 \
  --test
```

That writes the `TWINR_PROACTIVE_AUDIO_*` env keys and also runs a short direct capture smoke test against the proactive mic when configured.

### Printer

The current thermal printer is the DFRobot `DFR0503-EN`, exposed to CUPS as a Gprinter `GP-58` compatible raw printer.
This printer needs its own `9-24V` power supply for real paper output; a USB connection alone can still enumerate the device without being able to heat/feed paper.

Configure or re-create the printer queue with:

```bash
cd /twinr
sudo hardware/printer/setup_printer.sh --default --test
```

The setup script defaults to the `Thermal_GP58` queue and accepts an explicit `--device-uri` when auto-detection is not sufficient.
Both the setup smoke and the portal printer self-test only prove that a bounded print job was handed to the printer path; the operator still has to confirm the physical paper output on the device.

### Display

Twinr currently uses the `Waveshare 4.2inch e-Paper Module` with the `V2` Python driver and the standard Raspberry Pi SPI wiring.

Install the vendor driver files, persist the GPIO mapping, and run a smoke test with:

```bash
cd /twinr
sudo hardware/display/setup_display.sh
```

The current working mapping is:

- `DIN/MOSI -> GPIO10`
- `CLK -> GPIO11`
- `CS -> GPIO8`
- `DC -> GPIO25`
- `RST -> GPIO17`
- `BUSY -> GPIO24`

Run a manual test card at any time with:

```bash
cd /twinr
./.venv/bin/python hardware/display/display_test.py --env-file /twinr/.env
```

## Web dashboard

Run the local settings UI with:

```bash
cd /twinr
source .venv/bin/activate
twinr --env-file /twinr/.env --run-web
```

By default, the UI binds to `0.0.0.0:1337`, so it is reachable from the local network.

## OpenAI backend

Twinr now includes an OpenAI backend slice for speech-to-text, `gpt-5.2` reasoning, text-to-speech, and optional web search.

Recommended `.env` additions:

```dotenv
OPENAI_REASONING_EFFORT=medium
OPENAI_STT_MODEL=whisper-1
OPENAI_TTS_MODEL=gpt-4o-mini-tts
OPENAI_TTS_VOICE=marin
OPENAI_TTS_FORMAT=wav
OPENAI_TTS_INSTRUCTIONS="Speak in clear, warm, natural standard German with native German pronunciation. No English accent."
OPENAI_REALTIME_MODEL=gpt-4o-realtime-preview
OPENAI_REALTIME_VOICE=sage
OPENAI_REALTIME_TRANSCRIPTION_MODEL=whisper-1
OPENAI_REALTIME_LANGUAGE=de
OPENAI_REALTIME_INPUT_SAMPLE_RATE=24000
OPENAI_REALTIME_INSTRUCTIONS="Speak in clear, warm, natural standard German with native German pronunciation. No English accent."
TWINR_CONVERSATION_FOLLOW_UP_ENABLED=true
TWINR_CONVERSATION_FOLLOW_UP_TIMEOUT_S=4.0
TWINR_AUDIO_BEEP_FREQUENCY_HZ=1175
TWINR_AUDIO_BEEP_DURATION_MS=180
TWINR_AUDIO_BEEP_VOLUME=0.9
TWINR_AUDIO_BEEP_SETTLE_MS=140
OPENAI_SEND_PROJECT_HEADER=false
TWINR_OPENAI_ENABLE_WEB_SEARCH=true
TWINR_OPENAI_WEB_SEARCH_CONTEXT_SIZE=medium
TWINR_USER_DISPLAY_NAME=Thom
OPENAI_VISION_DETAIL=auto
TWINR_CAMERA_DEVICE=/dev/video0
TWINR_CAMERA_WIDTH=640
TWINR_CAMERA_HEIGHT=480
TWINR_CAMERA_FRAMERATE=30
# PS Eye often works reliably with:
TWINR_CAMERA_INPUT_FORMAT=bayer_grbg8
# Optional: a stored portrait/reference image of the main user
TWINR_VISION_REFERENCE_IMAGE=/home/thh/reference-user.jpg
TWINR_PROACTIVE_ENABLED=true
TWINR_PROACTIVE_POLL_INTERVAL_S=4.0
TWINR_PROACTIVE_CAPTURE_INTERVAL_S=6.0
TWINR_PROACTIVE_MOTION_WINDOW_S=20.0
TWINR_PROACTIVE_LOW_MOTION_AFTER_S=12.0
TWINR_PROACTIVE_AUDIO_ENABLED=true
# Optional: dedicate proactive background listening to the PS-Eye mic instead of the normal input device
TWINR_PROACTIVE_AUDIO_DEVICE=plughw:CARD=CameraB409241,DEV=0
TWINR_PROACTIVE_AUDIO_SAMPLE_MS=1000
TWINR_PROACTIVE_AUDIO_DISTRESS_ENABLED=false
TWINR_WAKEWORD_ENABLED=true
TWINR_WAKEWORD_BACKEND=stt
TWINR_WAKEWORD_PHRASES="hey twinr, hey twinna, twinr"
TWINR_WAKEWORD_SAMPLE_MS=1700
TWINR_WAKEWORD_PRESENCE_GRACE_S=900
TWINR_WAKEWORD_MOTION_GRACE_S=300
TWINR_WAKEWORD_SPEECH_GRACE_S=90
TWINR_WAKEWORD_ATTEMPT_COOLDOWN_S=4.0
```

When the OpenAI secret is already project-scoped (`sk-proj-...`), keep `OPENAI_SEND_PROJECT_HEADER=false`. That avoids a redundant project header that can block STT/TTS while `gpt-5.2` still works.

Smoke examples:

```bash
PYTHONPATH=src ./.venv/bin/python -m twinr --env-file /twinr/.env --openai-prompt "Reply with exactly OK."
PYTHONPATH=src ./.venv/bin/python -m twinr --env-file /twinr/.env --openai-prompt "What happened in the world today?" --openai-web-search
PYTHONPATH=src ./.venv/bin/python -m twinr --env-file /twinr/.env --tts-text "Hello from Twinr" --tts-output /tmp/twinr.wav
PYTHONPATH=src ./.venv/bin/python -m twinr --env-file /twinr/.env --camera-capture-output /tmp/twinr-camera.png
PYTHONPATH=src ./.venv/bin/python -m twinr --env-file /twinr/.env --vision-prompt "Bild 1 ist das Live-Kamerabild. Bild 2 ist das Referenzfoto des Nutzers. Ist es dieselbe Person? Antworte nur mit JA oder NEIN." --vision-camera-capture --vision-save-capture /tmp/twinr-camera.png --vision-image /home/thh/reference-user.jpg
PYTHONPATH=src ./.venv/bin/python -m twinr --env-file /twinr/.env --proactive-observe-once
PYTHONPATH=src ./.venv/bin/python -m twinr --env-file /twinr/.env --proactive-audio-observe-once
```

The vision path requires `ffmpeg` on the device because Twinr captures still images from V4L2 and then sends one or more images to the OpenAI Responses API in a single request.
The live hardware loop can also trigger the camera automatically for typical visual requests such as "Schau mich mal an", "Was zeige ich dir?", or "Wie sehe ich heute aus?".
With `TWINR_PROACTIVE_ENABLED=true`, the hardware and realtime loops also start the proactive monitor and let it issue bounded conversation starters while Twinr is idle.
With `TWINR_WAKEWORD_ENABLED=true`, the same idle monitor keeps a presence-gated wakeword session armed after recent PIR motion or visible presence. If ambient speech starts with one of the configured aliases, Twinr either answers the rest of the spoken request directly or opens the normal hands-free listening window.
For lower latency, set `TWINR_WAKEWORD_BACKEND=openwakeword` and point `TWINR_WAKEWORD_OPENWAKEWORD_MODELS` at one or more local `.tflite` or `.onnx` wakeword models. `openWakeWord` runs fully on-device, but it still needs a real model for the chosen phrase; for `Twinr/Twinna/Twinner` that normally means a custom-trained model. Until such a model is configured, Twinr can stay on the slower STT backend.

Run the live hardware loop with:

```bash
source .venv/bin/activate
twinr --env-file /twinr/.env --run-hardware-loop
```

The live loop watches the configured GPIO buttons, records from the default microphone until short silence, sends the transcript through OpenAI, plays the spoken answer, and lets the yellow button print the last answer.

For a lower-latency speech test, run the Realtime loop instead:

```bash
source .venv/bin/activate
twinr --env-file /twinr/.env --run-realtime-loop
```

The Realtime loop keeps the same button UX but uses OpenAI Realtime for direct audio input/output. It now also exposes bounded tools for web research, durable memory writes, reminder/timer scheduling, print requests, camera inspection, and ending the follow-up loop.
With `TWINR_CONVERSATION_FOLLOW_UP_ENABLED=true`, Twinr emits a short beep before listening, answers, then automatically beeps and listens again for a short follow-up window so a short back-and-forth conversation works without pressing the button every turn.
The Realtime loop can now also inspect the camera view for typical visual requests such as "Schau mich mal an", "Was zeige ich dir?", or "Wie sehe ich heute aus?".
If the user says things like `Erinnere mich morgen um 12 Uhr an den Arzttermin`, the realtime agent resolves the time into an absolute due time, stores the reminder in `state/reminders.json`, and later speaks a fresh reminder generated from the stored reminder facts.

For the migration path away from OpenAI Realtime, Twinr now also exposes a provider-neutral streaming loop:

```bash
source .venv/bin/activate
twinr --env-file /twinr/.env --run-streaming-loop
```

The streaming loop keeps the same green/yellow button semantics and background reminder/automation behavior, but runs the spoken turn as `STT -> text/tool agent -> TTS`. Today it is wired through the OpenAI Responses API adapter, so the existing Twinr tool surface still works while the provider stack is being decoupled.

Latency notes:

- `TWINR_SPEECH_PAUSE_MS` controls how quickly Twinr stops listening after you stop speaking.
- `TWINR_CONVERSATION_WEB_SEARCH=auto` keeps web search available for freshness-sensitive questions without forcing search on every turn.
- The live loop now streams LLM text internally and starts speaking sentence-by-sentence instead of waiting for the full answer before playback.

## Developer bootstrap

A minimal Python package skeleton now lives in `src/twinr`.

Twinr requires Python `3.11+`. Use the repo-local `.venv` instead of relying on a machine-global `python3`.

Typical local development flow:

- Create a virtual environment
- Install the project in editable mode
- Run the unit tests
- Run the bootstrap runtime demo
- Probe the GPIO buttons on the Raspberry Pi when hardware is attached

Example commands:

```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install -e . pytest
PYTHONPATH=src ./.venv/bin/python -m unittest discover -s test -v
PYTHONPATH=src ./.venv/bin/python -m twinr --demo-transcript "Hello Twinr"
hardware/buttons/setup_buttons.sh --green 23 --yellow 22
sudo hardware/mic/setup_audio.sh --device-match Jabra
./.venv/bin/python hardware/buttons/probe_buttons.py --env-file /twinr/.env --configured --duration 15
```
