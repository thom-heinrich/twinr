# ReSpeaker XVF3800 Requirements

## Purpose

This document defines what Twinr must integrate from the ReSpeaker XVF3800 so
later product work can safely implement the prioritized roadmap around:

- presence-aware behavior
- conservative proactive policy
- resume and turn-taking quality
- audio-side friction detection
- confidence-bearing speaker-track association
- calm, readable device state

This is not a generic microphone checklist. It is the concrete requirements and
todo list for turning the XVF3800 into a first-class Twinr sensor.

## Product Rule

The ReSpeaker should be treated as:

- an `audio presence sensor`
- a `speech activity and interruption sensor`
- a `direction / azimuth hint source`
- a `mute and device-state signal source`

The ReSpeaker should not be treated as:

- an identity sensor
- an emotion detector
- an object detector
- a gesture sensor
- a standalone basis for sensitive personalization

## Capability Boundary

The XVF3800 is useful for Twinr because it can plausibly support or expose:

- bounded microphone capture
- speech activity / VAD-like signals
- direction-of-arrival / azimuth-style hints
- beamformed audio paths
- mute state and mute button semantics
- LED ring status output

The XVF3800 does not by itself provide:

- object detection
- gesture recognition
- head pose or dwell time
- person identity
- safe private-content routing between multiple nearby people

## What Twinr Already Has

The current Twinr stack already has useful foundations that the ReSpeaker path
should plug into instead of bypassing:

- presence-session arming and grace windows
- proactive cooldown and reservation governance
- bounded ambient audio observation
- long-term sensor routine ingestion
- wakeword monitoring and proactive audio-device selection

Relevant current paths:

- `src/twinr/proactive/runtime/`
- `src/twinr/proactive/governance/`
- `src/twinr/proactive/social/`
- `src/twinr/memory/longterm/ingestion/`
- `hardware/mic/`

## Required Integration Layers

### 1. Device And Runtime Contract

Twinr must have a stable ReSpeaker runtime contract on the Raspberry Pi.

Requirements:

- [x] Detect XVF3800 runtime mode versus DFU / safe mode during startup.
- [x] Fail clearly when the board is visible only in DFU mode.
- [x] Keep a stable input-device selection path for both conversation audio and proactive audio.
- [x] Expose one operator-visible health view with device mode, mute state, and capture readiness.
- [x] Ensure hotplug / reboot recovery does not leave Twinr on a dead audio device.
- [x] Bound all audio waits and lock contention so the runtime cannot wedge.
- [x] Keep playback and capture routing explicit when ReSpeaker remains the output sink or operators override it.

Acceptance:

- [x] Pi reboot proof
- [x] unplug / replug proof
- [x] supervisor restart proof
- [x] health page or self-test evidence

Current proof note:

- Live on the Pi, host-side USB deauthorize / re-authorize exercised the
  ReSpeaker runtime path `ready -> dfu_mode/blocker -> ready`, with operator
  HCI changing to `ReSpeaker=DFU` during the disconnected capture phase and
  returning to normal after re-enumeration.

### 2. ReSpeaker Signal Provider

Twinr needs one focused adapter that converts XVF3800-specific state into
structured facts. This should not live as scattered shell calls or ad-hoc code
inside orchestration loops.

Required outputs:

- [x] `speech_detected`
- [x] `room_quiet`
- [x] `recent_speech_age_s`
- [x] `azimuth_deg` or equivalent directional bucket
- [x] `direction_confidence`
- [x] `beam_activity` or equivalent beam-energy summary
- [x] `barge_in_detected`
- [x] `speech_overlap_likely`
- [x] `mute_active`
- [x] `device_runtime_mode`
- [x] `non_speech_audio_likely`
- [x] `background_media_likely`

Current conservative interpretation note:

- `speech_detected` alone is not safe enough for barge-in during playback.
- Single fixed-beam speech energy is also not safe enough on its own during playback.
- V1 therefore treats busy-state `barge_in_detected` as a conservative overlap fact, not as any single speech bit.

Signal-quality rules:

- [x] Every field must carry `captured_at`.
- [x] Every inferred field must carry `confidence`.
- [x] The provider must degrade to explicit `unknown`, not guessed defaults.
- [x] The provider must never emit raw PCM into long-term memory objects.

### 3. Confidence And Source Contract

Every ReSpeaker-derived claim must remain inspectable.

Required metadata:

- [x] `source = respeaker_xvf3800`
- [x] `source_type = observed`
- [x] `confidence`
- [x] `sensor_window_ms`
- [x] `session_id` when tied to an active conversation or presence session
- [x] `requires_confirmation` for anything user-facing beyond direct local state

Required memory classes:

- [x] `ephemeral_state`
- [x] `session_memory`
- [x] `observed_preference`
- [x] `confirmed_preference`

Rules:

- [x] `observed_preference` must never be treated as equivalent to `confirmed_preference`.
- [x] Multi-person or low-confidence audio evidence must not drive sensitive behavior.

Current implementation note:

- The runtime now exports a nested `respeaker.claim_contract` map with
  per-field `captured_at`, `source`, `source_type`, `confidence`,
  `sensor_window_ms`, optional `session_id`, and memory-class semantics.
- ReSpeaker long-term raw pattern seeds keep only the claim subset that backed
  the derived memory object, and the sensor-memory compiler can now emit
  confirm-first `observed_preference` summaries for repeated voice-channel
  behavior.
- Sensitive long-term proactive delivery now fails closed on ambiguous room
  context: explicit multi-person camera context or active low-confidence audio
  direction blocks private/sensitive proactive speech instead of letting weak
  ReSpeaker evidence steer it.

### 4. Policy Hooks

The ReSpeaker must feed the policy layer, not replace it.

Required policy inputs:

- [x] `presence_audio_active`
- [x] `recent_follow_up_speech`
- [x] `room_busy_or_overlapping`
- [x] `quiet_window_open`
- [x] `barge_in_recent`
- [x] `speaker_direction_stable`
- [x] `mute_blocks_voice_capture`

Required suppression gates:

- [x] Never speak proactively when overlap or multi-speaker audio is likely.
- [x] After ignored or interrupted proactive speech, fall back to display-first.
- [x] Enforce global cooldown and per-source repeat cooldown on audio-triggered initiatives.
- [x] Keep quiet-hour behavior visual-first unless explicitly safety-related.

### 5. HCI And Awareness

The device must make ReSpeaker-driven state legible.

Required operator- and user-visible states:

- [x] `mic_muted`
- [x] `listening`
- [x] `heard_speech`
- [x] `noise_blocked`
- [x] `resume_window_open`
- [x] `direction_hint_available`
- [x] `respeaker_unavailable`
- [x] `respeaker_dfu_mode`

Required HCI hooks:

- [x] Map mute state clearly into display / status wording.
- [x] Decide whether the LED ring mirrors listening state, direction hint, or stays unused.
- [x] Keep status semantics calm and deterministic; never flicker between states on weak audio evidence.

### 6. Memory And Learning

The ReSpeaker should support learning only where the signal is defensible.

Allowed long-term learning targets:

- [x] typical conversation-start dayparts
- [x] typical quiet windows
- [x] typical response-channel preferences when later confirmed
- [x] repeated friction patterns around wakeword or interruptions
- [x] repeated resume behavior after short pauses

Disallowed from ReSpeaker alone:

- [ ] identity memory
- [ ] emotional state memory
- [ ] wellbeing claims
- [ ] private-content routing by assumed person identity

### 7. Runtime Scheduling

The XVF3800 path must stay Pi-friendly.

Requirements:

- [x] Keep cheap audio sensing available continuously only when bounded.
- [x] Keep any heavier direction or host-control polling out of the hot path when possible.
- [x] Prefer event-driven or low-frequency polling over constant expensive inspection.
- [x] Make the ReSpeaker path composable with PIR and camera gating rather than always-on maximal processing.

Current implementation note:

- The runtime now keeps the cheap fallback ambient-audio path available every
  proactive tick while a dedicated scheduled XVF3800 wrapper throttles heavier
  host-control refreshes when the room is idle.
- That scheduler takes current monitor context into account, including PIR
  motion, pending camera inspection, wakeword/presence arming, and whether
  Twinr is already speaking.

## Feature Mapping

This section maps the product roadmap to the ReSpeaker contribution.

| Target feature | ReSpeaker contribution | Additional dependencies |
|---|---|---|
| Presence Ladder | Strong | PIR, camera, policy |
| Initiative Score | Strong as feature source | Governor, suppression logic |
| Resume Engine | Strong | Conversation state and follow-up policy |
| Friction Detector | Strong for audio-side friction | Runtime event aggregation |
| Mikro-Consent | Indirect trigger only | LLM policy and memory typing |
| Awareness State Machine | Partial but important | Display and runtime state machine |
| Speaker-track association | Partial with confidence only | Camera tracks and fusion |
| Engagement adaptivity | Audio subset only | Vision for gaze/head pose |
| Object-based proactivity | None | Camera / vision stack |
| Gestural interface | None | Camera / gesture stack |
| Passive routine learning | Partial for audio routines | Long-term memory planner |
| Confidence Layer | Strong requirement | Shared schema across sensors |
| Cooldown / suppression logic | Strong input provider | Governor and runtime policy |
| Memory typing | Indirect | Memory model and policy |
| Runtime scheduler | Strong constraint | Proactive runtime orchestration |

## Build Order

### V1 Foundation

These items are mandatory before any ambitious proactive behavior.

- [x] Implement a dedicated XVF3800 capability probe and runtime-mode detector.
- [x] Implement one `ReSpeakerSignalProvider` module under `src/twinr/...`.
- [x] Emit structured audio-direction facts with confidence and timestamps.
- [x] Feed those facts into presence sessions and proactive governor inputs.
- [x] Add mute-state and device-mode visibility to operator diagnostics.
- [x] Add Pi acceptance proofs for runtime mode, capture, restart, and hotplug recovery.

### V2 Product Hooks

- [x] Use ReSpeaker facts for initiative scoring and resume decisions.
- [x] Add audio-side friction signals such as barge-in and overlap.
- [x] Add awareness-state transitions such as `noise_blocked` and `resume_window_open`.
- [x] Add bounded memory ingestion for audio interaction routines.

### V3 Sensor Fusion

- [x] Fuse azimuth hints with camera tracks for conservative speaker-track association.
- [x] Use ReSpeaker as one confidence-bearing input to richer multimodal initiative policy.
- [x] Gate later proactive behaviors on explicit confidence and suppression rules.

Current implementation note:

- V3 stays conservative: the runtime only associates current speech to the
  single primary visible person anchor when the room is single-person and both
  camera anchor plus ReSpeaker direction confidence are strong. It does not
  claim identity or private routing.
- Social triggers can now force `display-first` from the multimodal initiative
  gate, and long-term proactive prompts are skipped when that gate says the
  room context is too ambiguous for spoken initiative.

## Explicit Non-Goals For This Path

These are not valid reasons to extend the ReSpeaker path on its own:

- emotion detection
- medical-state inference
- hidden-person identity claims
- broad surveillance summaries
- gesture support without a dedicated non-audio sensor

## Pi Acceptance Checklist

The ReSpeaker path is not done until these pass on `/twinr`.

- [x] Twinr starts with ReSpeaker output and ReSpeaker input without manual fixes.
- [x] Device diagnostics show XVF3800 mode and capture readiness.
- [x] Proactive runtime reads ReSpeaker signals without blocking the main loop.
- [x] Wakeword / presence / proactive audio path recovers after supervisor restart.
- [x] Disconnect / reconnect errors are explicit and operator-readable.
- [x] Long-term memory stores only structured facts, not raw audio.

## Documentation Follow-Up

When implementation starts, update these paths together:

- `hardware/mic/README.md`
- `src/twinr/proactive/README.md`
- `src/twinr/proactive/runtime/README.md`
- `src/twinr/memory/longterm/README.md`

This keeps the hardware contract, runtime contract, and memory contract aligned.

Current implementation note:

- `sensor_observation` now exports structured ReSpeaker audio-policy facts plus
  `presence_session_id` from the proactive runtime into long-term ingestion.
- ReSpeaker long-term extraction is allowlisted to the structured `sensor`,
  `vad`, `respeaker`, and `audio_policy` sections only; it does not consume or
  persist PCM or raw audio bytes.
- V2 memory ingestion now seeds bounded raw patterns for:
  - `conversation_start_audio`
  - `quiet_window`
  - `friction_overlap`
  - `resume_follow_up`
- The sensor-memory compiler can aggregate those raw ReSpeaker pattern seeds
  into longer-horizon routines such as typical voice conversation-start
  dayparts and typical quiet windows.
