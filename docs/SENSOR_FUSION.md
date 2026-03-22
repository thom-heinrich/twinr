# Sensor Fusion

## Purpose

This document defines the practical target field for Twinr's current sensor stack:

- PIR motion sensor
- one microphone path
- one RGB camera path

Here, `IR` means the current PIR motion sensor, not a thermal camera, depth camera, or image-producing IR sensor.

The goal is not to guess hidden inner states. The goal is to decide what Twinr can reasonably observe, how good that signal can get on a Raspberry Pi 4, and what kind of action is justified.

The detailed design for short rolling audio/video buffers and `2-8s`
multimodal event fusion lives in
[`MULTIMODAL_EVENT_FUSION_V1.md`](./MULTIMODAL_EVENT_FUSION_V1.md).
The runtime-layer split between local `near_device_presence`, optional
smart-home-backed `room_context`, and house-wide `home_context` lives in
[`SMART_HOME_RUNTIME_CONTEXT_MODEL_V1.md`](./SMART_HOME_RUNTIME_CONTEXT_MODEL_V1.md).

## Quality Legend

- `High`
  Reliable enough for direct local state transitions, arming/disarming, cooldowns, and bounded operator-visible status.
- `Medium`
  Good enough for a gentle proactive prompt or a follow-up question, but not for a hard claim.
- `Low`
  Useful only as supporting evidence, a weak hint, or input to a second opinion.
- `No`
  Not defensible with the current hardware; Twinr should not claim this.

## Action Legend

- `Direct`
  Twinr may use this locally without LLM review.
- `Prompt`
  Twinr may use this to ask a calm question.
- `Review`
  Twinr should only use this after a second opinion, for example buffered frame review.
- `Never`
  Twinr should not present this as fact.

## Current Hardware Limits

The current stack is good at cheap presence signals and coarse social/safety cues. It is not a rich multimodal perception rig.

Hard limits of the current device:

- The PIR sensor only says that motion happened. It does not know who moved or where in the room.
- The PlayStation Eye gives only RGB, low-cost optics, no depth, no thermal map, and weak low-light performance.
- The microphone can detect speech/activity well enough, but far-field semantic audio classification will always be noisy in a normal room.
- Without depth, thermal, or multiple views, Twinr cannot robustly infer medical states, pain, or identity from vision alone.

## What Twinr Already Has

Today Twinr already carries these raw or fused states in the proactive path:

- `pir_motion_detected`
- `low_motion`
- `person_visible`
- `looking_toward_device`
- `body_pose = upright | slumped | floor | unknown`
- `smiling`
- `hand_or_object_near_camera`
- `speech_detected`
- `distress_detected`
- presence-session arming for wakeword
- weighted trigger candidates such as `person_returned`, `attention_window`, `showing_intent`, `slumped_quiet`, `possible_fall`, and `floor_stillness`
- optional OpenAI frame-buffer review before image-driven proactive prompts

Twinr now also uses explicit runtime contract surfaces around these signals:

- `ambiguous_room_guard`
  - fail-closed gate for person-targeted inference when room context is missing, multi-person, noisy, or direction-ambiguous
- `portrait_match`
  - conservative local face-match hint against the enrolled local portrait store, valid only for calm personalization and always confirm-first
- `identity_fusion`
  - bounded temporal/session fusion over portrait, enrolled household voice candidates, visual-anchor history, and presence-session memory, still confirm-first and never identity proof
- `known_user_hint`
  - conservative voice-plus-portrait-plus-room-context hint with optional temporal identity fusion over visual-anchor history and presence-session memory for calm personalization only, never identity proof
- `household identity manager`
  - shared local tool/runtime layer for face enrollment, voice enrollment, explicit confirm or deny feedback, quality scoring, and bounded local household identity status
- `affect_proxy`
  - prompt-only coarse cue such as `positive_contact`, `low_engagement`, or `concern_cue`, never emotion fact

That is a good baseline. The next step is to separate what is already good enough from what needs better local models.

## Maximal Derivable States With Current Sensors

This table is the target field for the existing hardware only.

| State | Main sensors | Max quality on current hardware | Safe action level | Why / notes |
|---|---|---:|---:|---|
| `room_motion_recent` | PIR | High | Direct | Cheap and robust. Best wake/gate signal. |
| `room_likely_occupied` | PIR + camera + audio | High | Direct | Strong when recent motion, person visible, or room speech exists. |
| `person_visible` | Camera | Medium to High | Direct | Better with a local person detector than with prompt-only vision classification. |
| `person_recently_visible` | Camera + history | High | Direct | Very useful for grace windows and wakeword arming. |
| `person_near_device` | Camera | Medium | Prompt | Can be estimated from face/body size and object proximity. |
| `looking_toward_device` | Camera | Medium | Prompt | Good enough for engagement detection, not precise gaze tracking. |
| `engaged_with_device` | Camera + audio | Medium | Prompt | Useful for "Can I help?" when presence, gaze, and quiet line up. |
| `showing_object` | Camera | Medium | Prompt | Works best when the object is intentionally held near the lens. |
| `person_returned_after_absence` | PIR + camera + history | High | Prompt | One of the strongest proactive conversation starters. |
| `speaking_in_room` | Audio | High | Direct | VAD should be strong locally. |
| `room_quiet` | Audio | High | Direct | Good for timing and cooldown logic. |
| `speech_directed_to_device_likely` | Audio + camera + history | Medium | Prompt | Useful for attentiveness, but still a heuristic. |
| `wakeword_candidate` | Audio + presence gating | Medium to High | Direct | Good if kept short and fully local. |
| `upright / seated / slumped / lying_low` | Camera | Medium | Prompt | Coarse posture is realistic. Fine posture semantics are not. |
| `possible_fall` | Camera + PIR + audio + history | Medium | Prompt or Review | Good enough to ask "Is everything okay?". Not good enough for hard emergency claims. |
| `floor_stillness_after_drop` | Camera + audio + history | Medium | Prompt or Review | Stronger than raw fall suspicion, still not a diagnosis. |
| `inactive_after_motion` | PIR + camera + audio | Medium | Prompt | Useful for concern triggers and automation timing. |
| `distress_sound_likely` | Audio | Low to Medium | Review | Better when backed by posture or stillness; poor as a solo trigger. |
| `cough_like / shout_like / alarm_like / water_like` | Audio | Medium | Prompt or Review | Feasible with a light audio event classifier. |
| `smile_or_positive_contact` | Camera | Low to Medium | Prompt | Acceptable as a light conversation starter, not as an emotional fact. |
| `visibly_unwell_or_withdrawn` | Camera + audio + history | Low | Review | Twinr should phrase this as a question, never as a fact. |
| `likely_known_speaker` | Audio voice profile | Low to Medium | Review | Good only as a soft confidence hint, not identity proof. |
| `multiple_people_present` | Camera + audio | Low to Medium | Review | Possible in some scenes, but current camera and framing make this weak. |
| `document_or_object_presented_for_reading` | Camera | Medium | Prompt | Good when the user intentionally holds something up. |
| `routine_deviation_over_time` | PIR + audio + camera + memory | Medium | Prompt | Needs longitudinal memory, not single-frame analysis. |
| `sad` / `lonely` / `confused` / `in pain` | Camera + audio | No | Never | Not defensible from the current sensor stack. |
| `same enrolled user as local portrait store` | Camera + history | Low to Medium | Review only | Better with multiple local references and temporal fusion, still not safe as identity proof. |
| `medical event` | Camera + audio + PIR | No | Never | Twinr may detect concern, not diagnose. |

## What Is Realistically Worth Building Next

For the current device, the best return comes from adding a few light local models and fusing them conservatively.

### Tier 1: Strong local signals

These are realistic on a Pi 4 and should carry most of the system:

- PIR-based motion and recent-presence windows
- continuous VAD
- local wakeword
- person detection
- coarse pose / body posture
- simple object-near-camera detection

This tier should drive:

- wakeword arming
- presence sessions
- showing-intent prompts
- returned-after-absence prompts
- coarse concern prompts

This tier remains the authoritative source for `near_device_presence`. Optional
smart-home context may support timing and confidence, but it must not replace
local PIR, camera, or microphone truth for near-device interaction.

### Tier 2: Light local classification

These are realistic, but should mostly support prompts rather than hard claims:

- coarse audio event classes such as cough-like, shout-like, alarm-like, water-like
- coarse posture states such as upright, slumped, lying low
- activity transitions such as approach, leave, disappear after visibility, prolonged stillness
- simple speaker-consistency hint

### Tier 3: Buffered second opinion

These should stay bounded and rare:

- frame-buffer review for image-triggered proactive prompts
- second opinion for `possible_fall`, `slumped_quiet`, `showing_intent`, or ambiguous positive-contact triggers

The local detector should propose. The review step should veto unclear cases. The review step should not be the first line of sensing.

## Sensor Memory And Routine Learning

Twinr already has the beginning of `sensor memory`.

The concrete V1 implementation sketch for this path lives in [`SENSOR_MEMORY_V1.md`](SENSOR_MEMORY_V1.md).
The room-agnostic smart-home profiling layer for motion sensors lives in [`SMART_HOME_ENVIRONMENT_PROFILE_V1.md`](SMART_HOME_ENVIRONMENT_PROFILE_V1.md).

Today, the long-term layer already stores low-confidence multimodal evidence such as:

- presence near the device by daypart
- camera-side interaction by daypart
- green/yellow button usage by daypart
- print usage by daypart
- camera usage by daypart

That means the base persistence path exists. What is still only partially built is the bridge from stored sensor patterns into live, human-feeling proactive behavior.

### V1 Sensor Memory Goals

1. `presence routine memory`
   - typical presence by `morning`, `afternoon`, `evening`
   - later also `weekday` vs `weekend`
2. `interaction routine memory`
   - when printing is common
   - when the camera is commonly used
   - when a conversation is commonly started
3. `deviation detection`
   - "today there is clearly less activity than usual"
   - "there has been no presence in a normally active time window"
4. `sensor-memory -> proactive bridge`
   - these patterns should not only be retrievable
   - they should be able to cause gentle proactive prompts
5. `confidence + veto`
   - routine deviation should never speak on its own
   - it should only speak when current sensor signals support the concern

### What Already Exists

- multimodal sensor events are already persisted into long-term memory
- daypart-based pattern objects already exist
- repeated evidence can already strengthen through reflection
- the long-term path already has a proactive reservation/policy layer

### What Is Still Missing

- true `routine memory` objects for expected behavior over time
- explicit `weekday` vs `weekend` baselines
- expected-vs-actual comparisons for the current day
- deviation objects that can be consumed by the live proactive path
- a clean live bridge from `sensor memory` into soft proactive prompts

### Recommended Additional Pattern Objects

The next useful structured objects are:

- `routine:presence:{weekday_class}:{daypart}`
  - example: `routine:presence:weekday:morning`
- `routine:interaction:conversation_start:{weekday_class}:{daypart}`
- `routine:interaction:print:{weekday_class}:{daypart}`
- `routine:interaction:camera_use:{weekday_class}:{daypart}`
- `routine:interaction:camera_showing:{weekday_class}:{daypart}`
- `routine:room_speech:{weekday_class}:{daypart}`

These should not be stored as hard truths from one event. They should emerge from repeated support over time.

### Recommended Missing Aggregations

To make the above useful, Twinr still needs bounded aggregation over history:

- support count per pattern
- days-observed count
- recency weighting
- weekday vs weekend split
- earliest and latest typical activity per daypart
- expected-presence windows for the current day
- low-activity / missing-activity comparisons against a recent baseline

The important rule is: these should stay simple and inspectable, not become a black-box scoring jungle.

### Sensor Memory To Proactive Bridge

The right bridge is not:

- "memory says this is unusual, so speak immediately"

The right bridge is:

- long-term memory proposes a soft routine/deviation candidate
- live sensor state confirms or vetoes it
- the proactive governor still applies cooldowns and budgets

Examples:

- a stored morning-presence routine plus current prolonged absence can justify a gentle check-in candidate
- a stored afternoon print routine plus a fresh recipe/shopping conversation can justify a helpful print offer
- a stored camera-showing routine plus current `showing_object` can justify a softer, more contextual prompt

### Confidence And Veto Rules

These should stay hard product rules:

- routine deviation alone must not speak
- current live sensors must support the interpretation
- safety-style prompts must remain conservative
- weak long-term signals should only shape wording or priority, not force a prompt

### First Three V1 Use Cases

These are the best first uses of sensor memory for a more human Twinr:

1. `presence routine deviation`
   - if the device usually sees presence in a time window and today that window stays unusually empty, Twinr may later use that only as weak support for a gentle check-in
2. `interaction routine continuity`
   - if the user often prints after certain conversations, Twinr can more naturally offer printing at the right time
3. `camera interaction continuity`
   - if the user often shows objects/documents in certain contexts, Twinr can more naturally ask whether something should be looked at

### Practical Bottom Line On Sensor Memory

Yes, the beginning of sensor memory is already present.

What is not fully built yet is the crucial last step:

- turning stored patterns into live, useful, calm, human-feeling proactive behavior

That bridge should be built conservatively and should always be veto-able by current live sensor evidence.

## Recommended Pi-Local Stack

This is the most realistic near-term stack for the existing hardware.

### Audio

- `Silero VAD`
  Strong local speech/not-speech gate.
- `openWakeWord`
  Local wakeword only, with a short primary wakeword such as `hey twinna`.
- `YAMNet` or a small `Edge Impulse` audio classifier
  For a few event classes like cough-like, shout-like, alarm-like, water-like, and TV/noise rejection.

### Vision

- `MediaPipe Pose Landmarker` or `MoveNet Lightning`
  For coarse pose, slumped posture, and lying-low cues.
- `MediaPipe Object Detector` or an equivalent light detector
  For person-visible and coarse object/hand-near-camera cues.

### Fusion

Use a weighted state machine, not one large end-to-end multimodal model:

- cheap raw facts first
- bounded hold times
- per-trigger cooldowns
- presence session memory
- second-opinion review only for ambiguous image-driven prompts

This matches Twinr's accessibility and stability goals better than a large "AI understands everything" path.

## Suggested Target States By Use Case

### Conversation starters

These are realistic and desirable:

- `person_returned_after_absence`
- `engaged_with_device`
- `showing_object`
- `document_or_object_presented_for_reading`
- `smile_or_positive_contact`
- `routine_deviation_over_time`

These should lead to soft prompts such as:

- "Schön dich zu sehen. Wie geht's dir?"
- "Möchtest du mir etwas zeigen?"
- "Kann ich dir bei etwas helfen?"

### Concern and safety prompts

These are realistic only as suspicion, not truth:

- `slumped_quiet`
- `possible_fall`
- `floor_stillness_after_drop`
- `distress_sound_likely`
- `inactive_after_motion`

These should lead to calm checks such as:

- "Ist alles in Ordnung?"
- "Brauchst du Hilfe?"
- "Antworte mir kurz: Ist alles okay?"

### States Twinr should never claim directly

- "Du bist traurig."
- "Du hast Schmerzen."
- "Du bist verwirrt."
- "Du bist gestürzt." as a certainty
- "Das bist sicher du." as identity proof

Twinr should instead ask, confirm, or stay quiet.

## Recommended Engineering Rule Set

For the current sensor stack, each derived state should fit one of these buckets:

- `Deterministic local fact`
  Example: recent PIR motion, room speech, person visible.
- `Soft local inference`
  Example: engaged with device, showing object, slumped posture.
- `Local suspicion with second opinion`
  Example: possible fall, ambiguous concern, uncertain showing intent.
- `Out of scope`
  Example: true emotion, diagnosis, hard identity.

If a state does not clearly fit one of those buckets, Twinr should not build user-facing behavior on it.

## Recommended Evaluation Targets

For each future state, track three things:

- detection quality
  Precision / recall or at least false-trigger rate and miss rate
- user impact
  Was the resulting prompt helpful, neutral, or annoying?
- compute cost
  CPU load, memory, and latency on the Pi

Suggested initial acceptance bar:

- `High` states: okay for direct local gating and status
- `Medium` states: okay for a gentle prompt only
- `Low` states: only okay as supporting evidence or review input
- `No` states: do not ship as a claim

## Practical Bottom Line

With the current hardware, Twinr can realistically become good at:

- presence
- attentiveness
- coarse posture
- object-showing intent
- speech activity
- wakeword readiness
- basic concern prompts

It cannot realistically become good at:

- rich emotion reading
- identity proof
- medical diagnosis
- robust fine-grained scene understanding in poor light

That is enough for a strong, calm, senior-friendly device if the fusion stays conservative.

## References

Internal code paths:

- `src/twinr/proactive/social/engine.py`
- `src/twinr/proactive/social/observers.py`
- `src/twinr/proactive/runtime/presence.py`
- `src/twinr/proactive/runtime/service.py`
- `src/twinr/hardware/audio.py`
- `src/twinr/hardware/pir.py`

External references, accessed March 15, 2026:

- MediaPipe Pose Landmarker: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python
- MediaPipe Object Detector: https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector
- Silero VAD: https://github.com/snakers4/silero-vad
- YAMNet: https://www.tensorflow.org/hub/tutorials/yamnet
- openWakeWord: https://github.com/dscripka/openWakeWord
- Home Assistant custom wakeword training: https://www.home-assistant.io/voice_control/create_wake_word/
- Edge Impulse sound recognition: https://docs.edgeimpulse.com/tutorials/end-to-end/sound-recognition
- TinyM2Net multimodal edge paper: https://arxiv.org/abs/2202.04303

Where this document makes implementation recommendations, they are an engineering inference from the current Twinr code plus the references above.
