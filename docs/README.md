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
The target-state map for what Twinr should try to derive from PIR, audio, and camera lives in [`SENSOR_FUSION.md`](SENSOR_FUSION.md).
The runtime-layer model for separating local `near_device_presence` from optional smart-home `room_context` and `home_context` lives in [`SMART_HOME_RUNTIME_CONTEXT_MODEL_V1.md`](SMART_HOME_RUNTIME_CONTEXT_MODEL_V1.md).
The short-window rolling-buffer and multimodal event-fusion design for deriving stable `2-8s` event claims from audio plus camera history lives in [`MULTIMODAL_EVENT_FUSION_V1.md`](MULTIMODAL_EVENT_FUSION_V1.md).
The room-agnostic long-horizon smart-home profiling design for motion sensors now lives in [`SMART_HOME_ENVIRONMENT_PROFILE_V1.md`](SMART_HOME_ENVIRONMENT_PROFILE_V1.md). That document defines how Twinr should learn behavior patterns from environment nodes, transitions, fragmentation, and timing without assuming that any sensor belongs to a known room.
The ReSpeaker-specific requirements and build checklist for turning the XVF3800 into a first-class Twinr sensor live in [`RESPEAKER_REQUIREMENTS.md`](RESPEAKER_REQUIREMENTS.md).
The camera-side output contract for later runtime integration lives in [`AI_CAMERA_INTEGRATION_CONTRACT.md`](AI_CAMERA_INTEGRATION_CONTRACT.md). That contract now separates `body_pose`, `motion`, `coarse_arm_gesture`, and `fine_hand_gesture` explicitly so Twinr can keep realistic V1 camera requirements and stricter V2 hand-gesture ambitions apart.
The current fine-hand-gesture evaluation and recommendation path lives in [`AI_CAMERA_FINE_HAND_GESTURE_PATH.md`](AI_CAMERA_FINE_HAND_GESTURE_PATH.md).
The currently accepted Raspberry Pi gesture runtime freeze point lives in [`PI_GESTURE_BASELINE.md`](PI_GESTURE_BASELINE.md). That document records the exact low-light, MediaPipe, ROI, and HDMI ack values that are currently accepted on `/twinr`.

Explicit requests to change future user-profile context or future speaking/behavior rules are written into managed sections inside `personality/USER.md` and `personality/PERSONALITY.md`. Twinr reloads those files on the next provider request instead of baking them permanently into code.
The target-state design for a richer persistent Twinr self now lives in [`persistent_personality_architecture.md`](persistent_personality_architecture.md). That document separates stable core character, relational user model, place intelligence, world intelligence, and bounded reflective evolution instead of treating personality as one growing prompt blob.

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
The planned multimodal replacement for that voice-only confirmation path is documented in `docs/PRIVACY_AUTH_POLICY_V1.md`.
The external browser benchmark integration plan for AgentLab, BrowserGym, and WebArena-Verified is documented in `docs/BROWSERGYM_AGENTLAB_WEB_ARENA_VERIFIED_PLAN.md`.

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
- Optional bounded drone mission layer for inspect-only planning against a separate drone daemon
- Text-to-speech provider layer
- Memory layer
- Hardware abstraction layer
- Web configuration layer

## Repository layout

- `docs/` — documentation and structured specifications
- `docs/providers/` — provider-specific notes and setup guides
- `hardware/` — Raspberry Pi setup scripts for buttons, PIR, audio, printer, and display hardware
- `hardware/ops/drone_daemon.py` — development-host bounded drone daemon that owns mission preflight, manual-arm, and the first stationary-observe inspection slice
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
- Optional body-orientation follow through a bounded Pi servo output fed by the existing multimodal attention target
- A stateful social-trigger engine for cautious presence, attention, and safety nudges
- Provider-based STT / LLM / TTS orchestration
- Optional bounded drone inspection planning through a separate mission daemon with manual-arm-only safety defaults
- Agentic task execution
- Paper-based answer printing
- Visual state feedback through eyes
- On-device memory with optional long-term expansion
- Local web-based configuration on `:1337`

## Hardware bootstrap

For one compact current-build overview, see [WIRING_DIAGRAM.md](./WIRING_DIAGRAM.md).

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

### Attention servo

Twinr can optionally drive one body-orientation servo from the same conservative
multimodal `attention_target` that already powers HDMI eye-follow. The servo
path does not choose its own target; it only turns a normalized left/right
target into bounded servo pulse widths.

The current prototype wiring uses:

- `SIG -> GPIO18` on the Pi (`physical pin 12`, often labeled `IO18`)
- `GND -> GND`
- `V+ ->` the servo supply recommended by the servo module vendor

Recommended runtime env knobs:

- `TWINR_ATTENTION_SERVO_ENABLED=true`
- `TWINR_ATTENTION_SERVO_DRIVER=auto`
- `TWINR_ATTENTION_SERVO_GPIO=18`
- `TWINR_ATTENTION_SERVO_INVERT_DIRECTION=false`

When the physical Pololu Maestro lives on a helper Pi instead of the main Pi,
configure `TWINR_ATTENTION_SERVO_DRIVER=peer_pololu_maestro` and point
`TWINR_ATTENTION_SERVO_PEER_BASE_URL` at the helper's direct-link proxy, for
example `http://10.42.0.2:8768`.

Optional tuning knobs:

- `TWINR_ATTENTION_SERVO_TARGET_HOLD_S`
- `TWINR_ATTENTION_SERVO_LOSS_EXTRAPOLATION_S`
- `TWINR_ATTENTION_SERVO_LOSS_EXTRAPOLATION_GAIN`
- `TWINR_ATTENTION_SERVO_MIN_CONFIDENCE`
- `TWINR_ATTENTION_SERVO_HOLD_MIN_CONFIDENCE`
- `TWINR_ATTENTION_SERVO_DEADBAND`
- `TWINR_ATTENTION_SERVO_MIN_PULSE_WIDTH_US`
- `TWINR_ATTENTION_SERVO_CENTER_PULSE_WIDTH_US`
- `TWINR_ATTENTION_SERVO_MAX_PULSE_WIDTH_US`
- `TWINR_ATTENTION_SERVO_MAX_STEP_US`
- `TWINR_ATTENTION_SERVO_TARGET_SMOOTHING_S`
- `TWINR_ATTENTION_SERVO_MAX_VELOCITY_US_PER_S`
- `TWINR_ATTENTION_SERVO_MAX_ACCELERATION_US_PER_S2`
- `TWINR_ATTENTION_SERVO_MAX_JERK_US_PER_S3`
- `TWINR_ATTENTION_SERVO_MIN_COMMAND_DELTA_US`
- `TWINR_ATTENTION_SERVO_VISIBLE_RETARGET_TOLERANCE_US`
- `TWINR_ATTENTION_SERVO_IDLE_RELEASE_S`
- `TWINR_ATTENTION_SERVO_SETTLED_RELEASE_S`
- `TWINR_ATTENTION_SERVO_FOLLOW_EXIT_ONLY`
- `TWINR_ATTENTION_SERVO_VISIBLE_RECENTER_INTERVAL_S`
- `TWINR_ATTENTION_SERVO_VISIBLE_RECENTER_CENTER_TOLERANCE`
- `TWINR_ATTENTION_SERVO_MECHANICAL_RANGE_DEGREES`
- `TWINR_ATTENTION_SERVO_EXIT_FOLLOW_MAX_DEGREES`
- `TWINR_ATTENTION_SERVO_EXIT_ACTIVATION_DELAY_S`
- `TWINR_ATTENTION_SERVO_CONTINUOUS_MAX_SPEED_DEGREES_PER_S`
- `TWINR_ATTENTION_SERVO_CONTINUOUS_SLOW_ZONE_DEGREES`
- `TWINR_ATTENTION_SERVO_CONTINUOUS_STOP_TOLERANCE_DEGREES`
- `TWINR_ATTENTION_SERVO_CONTINUOUS_MIN_SPEED_PULSE_DELTA_US`
- `TWINR_ATTENTION_SERVO_CONTINUOUS_MAX_SPEED_PULSE_DELTA_US`
- `TWINR_ATTENTION_SERVO_STATE_PATH`
- `TWINR_ATTENTION_SERVO_ESTIMATED_ZERO_MAX_UNCERTAINTY_DEGREES`
- `TWINR_ATTENTION_SERVO_ESTIMATED_ZERO_SETTLE_TOLERANCE_DEGREES`
- `TWINR_ATTENTION_SERVO_ESTIMATED_ZERO_SPEED_SCALE`
- `TWINR_ATTENTION_SERVO_ESTIMATED_ZERO_MOVE_PULSE_DELTA_US`
- `TWINR_ATTENTION_SERVO_ESTIMATED_ZERO_MOVE_PERIOD_S`
- `TWINR_ATTENTION_SERVO_ESTIMATED_ZERO_MOVE_DUTY_CYCLE`
- `TWINR_ATTENTION_SERVO_CONTINUOUS_RETURN_TO_ZERO_AFTER_S`
- `TWINR_ATTENTION_SERVO_PEER_BASE_URL`
- `TWINR_ATTENTION_SERVO_PEER_TIMEOUT_S`

Supported driver values are `auto`, `twinr_kernel`, `sysfs_pwm`, `pigpio`,
`lgpio_pwm`, `lgpio`, `pololu_maestro`, and `peer_pololu_maestro`.

When `TWINR_ATTENTION_SERVO_DRIVER=auto`, Twinr first probes for the custom
`/sys/class/twinr_servo/servo0` kernel-module contract, then a usable
kernel-PWM sysfs path, then a usable hardware-timed `pigpio` path, and only
falls back to `lgpio_pwm` before the older `lgpio` servo helper. The
`twinr_kernel` path is the preferred senior-facing option when the out-of-tree
Twinr kernel servo module is built and loaded on the Pi.

`pololu_maestro` expects the Maestro command port to exist locally on the same
host as the Twinr runtime. `peer_pololu_maestro` keeps the follow logic on the
main Pi but sends bounded channel commands over HTTP to a helper Pi that owns
the physical Maestro USB connection.

For calm physical behavior, do not think only in terms of "smaller pulse
steps". Many hobby servos have a pulse dead band, so tiny frequent pulse
changes can buzz or twitch instead of looking softer. Twinr therefore combines:

- attention-target smoothing
- short exit-trajectory extrapolation, so the body can keep following briefly in the user's leaving direction instead of freezing or snapping back the moment the camera loses them
- an exit-activation delay, so brief camera dropouts do not immediately trigger a body turn
- hard velocity limits
- acceleration and jerk limits in pulse space
- a small command threshold so sub-dead-band plan noise is not emitted as GPIO churn
- release-after-settle on stable off-center targets, so a loaded servo can move into place and then relax instead of audibly holding torque forever
- optional exit-only follow, so Twinr stays physically still while a person remains in frame, cancels exit-follow immediately when any person is visible again, and only turns gently along the departure trajectory up to one configured off-center degree limit before releasing the servo again

Keep the servo signal at `3.3V` logic level on the Pi side, and power the
servo from a supply that matches the specific servo module instead of feeding
it from a weak GPIO pin.

In continuous-rotation mode, Twinr cannot read an absolute shaft angle back
from the servo. The controller therefore treats the physically current pose at
startup as virtual `0°`. If an operator hand-sets a forward-facing neutral
pose, start or restart the runtime only after that pose is in place.

Twinr now persists that virtual heading in `TWINR_ATTENTION_SERVO_STATE_PATH`.
When the persisted state says `hold_until_armed`, startup keeps the continuous
servo output released and returns `manual_hold` instead of immediately
leaving the operator-set `0°` pose. Operators can flip that persisted
continuous-servo state explicitly with `python3 hardware/servo/attention_servo_state.py`
using `hold-current-zero`, `hold`, `arm`, and `return-to-estimated-zero`.

`return-to-estimated-zero` is still open-loop, but it no longer starts from a
generic recenter pulse when a real movement journal exists. Twinr now persists
the bounded outbound movement path from the operator-defined `0°` and, on
return, replays those logged segments in reverse with inverse pulse widths
before falling back to the older estimated-zero planner. Those replay segments
now run through an exact bounded segment player that disables each pulse at
its recorded deadline instead of holding it until the next runtime update
tick. That makes return-to-zero track the actual commanded path instead of
quantizing short segments into longer physical moves. Twinr still keeps a
persisted `heading_uncertainty_degrees`
alongside that journal and rejects the return request once the saved
uncertainty exceeds
`TWINR_ATTENTION_SERVO_ESTIMATED_ZERO_MAX_UNCERTAINTY_DEGREES`. When no
journal is available, the fallback planner still uses its own tighter zero-
settle threshold instead of the broader live-tracking tolerance:
`TWINR_ATTENTION_SERVO_ESTIMATED_ZERO_SETTLE_TOLERANCE_DEGREES`. Because the
gentle fallback return pulses can move the real hardware more slowly than the
normal follow model expects, `TWINR_ATTENTION_SERVO_ESTIMATED_ZERO_SPEED_SCALE`
lets the virtual heading decay more conservatively during return-to-zero.
That fallback slow return is gated through move/release phases:
`TWINR_ATTENTION_SERVO_ESTIMATED_ZERO_MOVE_PULSE_DELTA_US`
caps each movement pulse near neutral, while
`TWINR_ATTENTION_SERVO_ESTIMATED_ZERO_MOVE_PERIOD_S` plus
`TWINR_ATTENTION_SERVO_ESTIMATED_ZERO_MOVE_DUTY_CYCLE` determine how long each
slow nudge should stay active before Twinr releases the output again on the
next servo update.
For ordinary live tracking on a continuous servo, keep
`TWINR_ATTENTION_SERVO_TARGET_HOLD_S` short so only brief tracker dropouts are
projected. If Twinr should wait before drifting back to the persisted `0°`,
use `TWINR_ATTENTION_SERVO_CONTINUOUS_RETURN_TO_ZERO_AFTER_S` instead. That
separate no-user timer keeps the output released during absence and only starts
the slow return-to-zero path after the configured delay has elapsed.
If the visible target stays stable but its confidence bounces around the main
acquisition threshold, `TWINR_ATTENTION_SERVO_HOLD_MIN_CONFIDENCE` lets Twinr
keep following the already-visible track at a lower release threshold instead
of snapping into `recentering` on every short confidence dip.
For active visible follow on a continuous servo, Twinr now closes the loop on
the live image offset itself and maps that error only into the configured
continuous min/max speed pulse window. The persisted virtual heading remains
for journaling and return-to-zero, but active follow no longer routes through
the virtual-heading target calculation that can oscillate at sparse Pi refresh
cadence.
When the continuous-servo path is already at the estimated `0°`, Twinr now
keeps the output released in idle instead of sending a nominal center pulse
that could still drift a real 360-degree servo off the operator-set neutral.

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

Twinr currently uses the `reSpeaker XVF3800 4-Mic Array` as its default playback and microphone device.
The external speaker attached to the ReSpeaker is driven through that same sink.
For proactive background listening, Twinr can additionally use the `PlayStation Eye` USB microphone as a separate capture path.

Configure the audio defaults with:

```bash
cd /twinr
sudo hardware/mic/setup_audio.sh
```

The setup script also normalizes the selected playback sink and ALSA playback
controls so the ReSpeaker-attached speaker does not inherit a near-muted output
level from a previously selected sink.

To keep ReSpeaker as the main default device while also wiring the PS-Eye microphone into the proactive path, run:

```bash
cd /twinr
sudo hardware/mic/setup_audio.sh \
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

Twinr now includes an OpenAI backend slice for speech-to-text, `gpt-5.4-mini` reasoning, text-to-speech, and optional web search.

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
# For the Bitcraze AI-Deck WiFi streamer instead:
# TWINR_CAMERA_DEVICE=aideck://192.168.4.1:5000
TWINR_CAMERA_WIDTH=640
TWINR_CAMERA_HEIGHT=480
TWINR_CAMERA_FRAMERATE=30
# PS Eye often works reliably with:
TWINR_CAMERA_INPUT_FORMAT=bayer_grbg8
# Optional: a stored portrait/reference image of the main user
TWINR_VISION_REFERENCE_IMAGE=/home/thh/reference-user.jpg
TWINR_PORTRAIT_MATCH_STORE_PATH=state/portrait_identities.json
TWINR_PORTRAIT_MATCH_REFERENCE_IMAGE_DIR=state/portrait_identities
TWINR_PORTRAIT_MATCH_PRIMARY_USER_ID=main_user
TWINR_PORTRAIT_MATCH_MAX_REFERENCE_IMAGES_PER_USER=6
TWINR_PORTRAIT_MATCH_IDENTITY_MARGIN=0.05
TWINR_PORTRAIT_MATCH_TEMPORAL_WINDOW_S=300.0
TWINR_PORTRAIT_MATCH_TEMPORAL_MIN_OBSERVATIONS=2
TWINR_PORTRAIT_MATCH_TEMPORAL_MAX_OBSERVATIONS=12
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
TWINR_PROACTIVE_VISION_REVIEW_ENABLED=true
TWINR_PROACTIVE_VISION_REVIEW_BUFFER_FRAMES=8
TWINR_PROACTIVE_VISION_REVIEW_MAX_FRAMES=4
TWINR_PROACTIVE_VISION_REVIEW_MAX_AGE_S=12.0
TWINR_PROACTIVE_VISION_REVIEW_MIN_SPACING_S=1.2
TWINR_VOICE_ORCHESTRATOR_ENABLED=true
TWINR_VOICE_ORCHESTRATOR_WS_URL=ws://192.168.1.154:8797/ws/orchestrator/voice
TWINR_VOICE_ORCHESTRATOR_REMOTE_ASR_URL=http://192.168.1.154:8797
TWINR_VOICE_ORCHESTRATOR_SHARED_SECRET=twinr-voice-gateway-20260322
```

`OPENAI_MODEL` is the central non-coding text model. Search, supervisor routing, first-word replies, conversation closure, long-term-memory text compilation, and display reserve generation inherit it unless a dedicated override is configured. Audio-specialized paths stay separate on purpose.

When the OpenAI secret is already project-scoped (`sk-proj-...`), keep `OPENAI_SEND_PROJECT_HEADER=false`. That avoids a redundant project header that can block STT/TTS while `gpt-5.4-mini` still works.

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

The vision path requires `ffmpeg` on the device when Twinr captures still images from V4L2. When `TWINR_CAMERA_DEVICE` is set to `aideck://192.168.4.1:5000`, Twinr instead reads one bounded AI-Deck WiFi frame directly, debayers raw streamer output into a normal PNG on demand, and then sends one or more images to the OpenAI Responses API in a single request. In that mode, the Twinr host itself must already be connected to the AI-Deck access point.
The active realtime and streaming loops can also trigger the camera automatically for typical visual requests such as "Schau mich mal an", "Was zeige ich dir?", or "Wie sehe ich heute aus?".
With `TWINR_PROACTIVE_ENABLED=true`, the active runtime loops also start the proactive monitor and let it issue bounded conversation starters while Twinr is idle.
With `TWINR_PROACTIVE_VISION_REVIEW_ENABLED=true`, image-driven proactive prompts are reviewed against a short buffered frame sequence before Twinr speaks. That second opinion is conservative: if the recent frames look empty or ambiguous, Twinr skips the proactive prompt instead of speaking.
With `TWINR_VOICE_ORCHESTRATOR_ENABLED=true`, the Pi keeps one live websocket stream open to the transcript-first voice gateway and leaves wake detection there. The same remote stream stays authoritative for wake, transcript commit, continuation, and follow-up closure. Twinr has no separate local wake or STT path in the live product flow.
Remove any retired local voice-selector env from current deployments. The live runtime contract is remote-only and fails closed on legacy local-selector config.

The active runtime loops keep the same green/yellow button UX. They can inspect the camera for typical visual requests such as "Schau mich mal an", "Was zeige ich dir?", or "Wie sehe ich heute aus?".
With `TWINR_CONVERSATION_FOLLOW_UP_ENABLED=true`, Twinr emits a short beep before listening, answers, then automatically beeps and listens again for a short follow-up window so a short back-and-forth conversation works without pressing the button every turn.

Twinr exposes a provider-neutral streaming loop:

```bash
source .venv/bin/activate
twinr --env-file /twinr/.env --run-streaming-loop
```

The streaming loop keeps the same green/yellow button semantics and background reminder/automation behavior, but runs the spoken turn as `STT -> text/tool agent -> TTS`. Today it is wired through the OpenAI Responses API adapter, so the existing Twinr tool surface still works while the provider stack is being decoupled.

The streaming loop now also supports a mixed provider stack through `.env`:

- `TWINR_STT_PROVIDER=openai|deepgram`
- `TWINR_LLM_PROVIDER=openai|groq`
- `TWINR_TTS_PROVIDER=openai`

For the cheaper migration target, set:

```bash
TWINR_STT_PROVIDER=deepgram
TWINR_LLM_PROVIDER=groq
TWINR_TTS_PROVIDER=openai
DEEPGRAM_API_KEY=...
GROQ_API_KEY=...
OPENAI_API_KEY=...
```

`OpenAI` remains the support backend for search, print composition, reminder phrasing, proactive phrasing, camera inspection, and spoken TTS while `Deepgram` handles STT and `Groq` handles the main text/tool loop.

Twinr now also exposes the first websocket orchestrator slice for the planned edge/cloud architecture:

```bash
source .venv/bin/activate
twinr --env-file /twinr/.env --run-orchestrator-server
```

And a bounded text probe against that websocket service:

```bash
source .venv/bin/activate
twinr --env-file /twinr/.env --orchestrator-probe-turn "Wie wird das Wetter morgen in Schwarzenbek?"
```

The current orchestrator slice is intentionally not the default runtime yet. It provides:

- a websocket server at `TWINR_ORCHESTRATOR_WS_URL`
- a structured `ack_id` path for pre-rendered short acknowledgements
- a remote tool bridge so the specialist worker can request Pi-local tool execution over the websocket connection

New orchestrator config keys:

- `TWINR_ORCHESTRATOR_HOST`
- `TWINR_ORCHESTRATOR_PORT`
- `TWINR_ORCHESTRATOR_WS_URL`
- `TWINR_ORCHESTRATOR_SHARED_SECRET` (optional)

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
sudo hardware/mic/setup_audio.sh
./.venv/bin/python hardware/buttons/probe_buttons.py --env-file /twinr/.env --configured --duration 15
```
