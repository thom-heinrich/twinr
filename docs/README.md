# Twinr

Twinr is a minimal digital, maximal haptic, senior-citizen-optimized physical AI agent.

The product is designed around simple physical interaction instead of app-first complexity. A user presses a green button, speaks naturally, and Twinr listens until a short pause is detected. Twinr then transcribes the utterance with a speech-to-text provider, sends the result to a foundation model, performs agentic tasks when needed, and answers via text-to-speech. A yellow button prints the latest answer as a short, easy-to-read thermal receipt.

Twinr also uses a small status display with animated eyes to communicate its current state: waiting, listening, processing, answering, printing, and error.

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

Twinr keeps a short rolling conversation memory on-device by default. This memory stores recent exchanges and compacts them when needed to stay lightweight and efficient.

### Long-term memory (optional)

Twinr can optionally be extended with a ChonkyDB-based long-term memory layer. This provides durable storage of information in a combined index spanning:

- Vector search
- Graph relationships
- Full-text search
- Hash-based lookup
- Temporal access patterns

This long-term layer is intended for persistent user preferences, recalled facts, and historical conversations.

## Web interface

Twinr exposes a web interface on port `1337` for local configuration.

Example configuration areas:

- Personality and tone
- User preferences
- Provider selection
- Device settings
- Memory settings
- Print behavior
- Display settings

The web interface should stay simple enough for caregivers, family members, or operators to maintain without developer tooling.

## High-level architecture

### Inputs

- Green hardware button
- Yellow hardware button
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
- `hardware/` — hardware-specific integrations for display, printer, buttons, mic, and miscellaneous devices
- `print/` — print formatting and thermal printer logic
- `provider/` — provider adapters for OpenAI, DeepInfra, and OpenRouter
- `display/` — visual display logic such as status eyes
- `agent/` — base agent logic and workflows
- `memory/` — on-device, print-oriented, and long-term memory layers
- `src/twinr/` — Python runtime, config, state machine, memory, and hardware adapters
- `test/` — tests and validation assets

## Intended user experience

Twinr should feel calm, obvious, and trustworthy. A user should not need to understand models, APIs, sessions, prompts, or configuration details. The interaction should feel closer to using a helpful household object than using a computer.

## Current scope

This repository defines the foundation for:

- Button-driven voice capture
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
hardware/buttons/setup_buttons.sh --green 23 --yellow 24
```

Probe the configured buttons with:

```bash
cd /twinr
python3 hardware/buttons/probe_buttons.py --env-file /twinr/.env --configured --duration 15
```

The runtime-facing GPIO helpers live in `src/twinr/hardware/buttons.py`.

### Audio

Twinr currently uses the `Jabra SPEAK 510 USB` attached to the Raspberry Pi as its default microphone and speaker.

Configure the audio defaults with:

```bash
cd /twinr
sudo hardware/mic/setup_audio.sh --device-match Jabra
```

### Printer

The current thermal printer is the DFRobot `DFR0503-EN`, exposed to CUPS as a Gprinter `GP-58` compatible raw printer.

Configure or re-create the printer queue with:

```bash
cd /twinr
sudo hardware/printer/setup_printer.sh --default --test
```

The setup script defaults to the `Thermal_GP58` queue and accepts an explicit `--device-uri` when auto-detection is not sufficient.

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
```

When the OpenAI secret is already project-scoped (`sk-proj-...`), keep `OPENAI_SEND_PROJECT_HEADER=false`. That avoids a redundant project header that can block STT/TTS while `gpt-5.2` still works.

Smoke examples:

```bash
PYTHONPATH=src python3 -m twinr --env-file /twinr/.env --openai-prompt "Reply with exactly OK."
PYTHONPATH=src python3 -m twinr --env-file /twinr/.env --openai-prompt "What happened in the world today?" --openai-web-search
PYTHONPATH=src python3 -m twinr --env-file /twinr/.env --tts-text "Hello from Twinr" --tts-output /tmp/twinr.wav
```

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

The Realtime loop keeps the same button UX but uses OpenAI Realtime for direct audio input/output. The current Realtime path does not invoke web search; keep the existing `--run-hardware-loop` path for search-enabled answers.
With `TWINR_CONVERSATION_FOLLOW_UP_ENABLED=true`, Twinr emits a short beep before listening, answers, then automatically beeps and listens again for a short follow-up window so a short back-and-forth conversation works without pressing the button every turn.

Latency notes:

- `TWINR_SPEECH_PAUSE_MS` controls how quickly Twinr stops listening after you stop speaking.
- `TWINR_CONVERSATION_WEB_SEARCH=auto` keeps web search available for freshness-sensitive questions without forcing search on every turn.
- The live loop now streams LLM text internally and starts speaking sentence-by-sentence instead of waiting for the full answer before playback.

## Developer bootstrap

A minimal Python package skeleton now lives in `src/twinr`.

Typical local development flow:

- Create a virtual environment
- Install the project in editable mode
- Run the unit tests
- Run the bootstrap runtime demo
- Probe the GPIO buttons on the Raspberry Pi when hardware is attached

Example commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
PYTHONPATH=src python3 -m unittest discover -s test -v
PYTHONPATH=src python3 -m twinr --demo-transcript "Hello Twinr"
hardware/buttons/setup_buttons.sh --green 23 --yellow 24
sudo hardware/mic/setup_audio.sh --device-match Jabra
python3 hardware/buttons/probe_buttons.py --env-file /twinr/.env --configured --duration 15
```
