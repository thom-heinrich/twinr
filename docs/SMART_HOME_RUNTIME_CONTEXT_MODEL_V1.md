# Smart-Home Runtime Context Model V1

## Purpose

This document defines how configured smart-home sensors should enter Twinr's
live runtime fusion without weakening the local on-device interaction model.

The goal is not to turn every house signal into presence truth. The goal is to
keep Twinr's near-device behavior grounded in local PIR, camera, and audio,
while allowing configured smart-home signals to add room-level and home-level
context.

This document answers three questions:

- which runtime context layers Twinr should maintain
- which signals each layer may trust
- where `smart_home` should enter the existing runtime path

## Non-Goals

This document does **not** define:

- generic smart-home control semantics
- room-label inference from provider metadata
- hard identity from smart-home devices
- direct replacement of local presence sensing with cloud or house-wide signals
- long-horizon environment profiling

The long-horizon room-agnostic profile layer remains defined in
[`SMART_HOME_ENVIRONMENT_PROFILE_V1.md`](./SMART_HOME_ENVIRONMENT_PROFILE_V1.md).

## Why The Layers Must Be Separate

Twinr has two very different jobs:

1. decide whether somebody is here with the device right now
2. decide what broader room or home context should shape proactive timing

Those jobs cannot share one undifferentiated `presence` state.

If a hallway motion sensor or a remote room button can directly arm local
voice-activation behavior, Twinr becomes noisy and surprising. If on-device sensing
ignores configured environment context entirely, Twinr loses useful household
awareness such as `someone is elsewhere in the home`, `alarm active`, or
`smart-home stream unhealthy`.

The correct model is hierarchical:

- `near_device_presence`
  - authoritative local interaction state around Twinr itself
- `room_context`
  - optional configured same-room corroboration and veto context
- `home_context`
  - whole-home occupancy, alarm, health, and routine context

## Layer Model

### 1. `near_device_presence`

This is the live local interaction layer. It answers:

- is someone likely near Twinr now
- is a person visible or recently visible
- is room speech aligned with recent local presence
- should voice activation, gesture wakeup, or display-first engagement be armed

Primary sources:

- local PIR
- local camera
- local microphone / VAD / voice-activation path

Allowed derived states:

- `room_motion_recent`
- `room_likely_occupied`
- `person_visible`
- `person_recently_visible`
- `speaking_in_room`
- `voice_activation_candidate`
- presence sessions and local attention windows

Rules:

- `near_device_presence` is authoritative for local HCI.
- `smart_home` must not create `person_visible`.
- `smart_home` must not directly arm voice activation or speaker-targeted attention.
- `smart_home` may only bias timing or caution through higher layers.

### 2. `room_context`

This is an optional same-room corroboration layer. It answers:

- does configured same-room infrastructure support or weaken the local story
- is there evidence that the room around Twinr is active beyond the local PIR
- is the room context ambiguous or stale

Primary sources:

- explicitly configured same-room smart-home motion/contact/button sensors
- optional explicit operator-mapped room-health devices
- on-device `near_device_presence` as the anchor

Allowed derived states:

- `same_room_motion_recent`
- `same_room_secondary_activity`
- `same_room_button_recent`
- `same_room_sensor_stale`
- `same_room_context_available`
- `same_room_context_ambiguous`

Rules:

- Room membership must come from explicit operator configuration, not provider
  room labels.
- `room_context` may support or veto prompt timing.
- `room_context` may strengthen a local claim such as `person_returned` when
  same-room motion lines up.
- `room_context` must fail closed when mappings are missing, stale, or
  provider health is poor.
- `room_context` must not act as identity evidence.

### 3. `home_context`

This is the home-wide context layer. It answers:

- is there activity elsewhere in the home
- is the smart-home stream healthy enough to trust for soft context
- is an alarm active
- is the broader household behavior unusually quiet or fragmented

Primary sources:

- generic smart-home event stream
- smart-home device online/offline state
- smart-home alarms
- long-horizon smart-home environment memory

Allowed derived states:

- `home_occupied_likely`
- `other_room_motion_recent`
- `smart_home_alarm_active`
- `smart_home_stream_unhealthy`
- `environment_deviation_hint`

Rules:

- `home_context` may shape proactive policy, not local person truth.
- `home_context` may suppress or delay non-essential prompts during alarms or
  stream instability.
- `home_context` may support long-term concern candidates when current local
  signals already justify attention.
- `home_context` must not replace local evidence for display or speech
  engagement.

## Trust, Freshness, And Failure Rules

### Trust order

For live local interaction, Twinr should trust:

1. on-device local signals
2. explicit same-room configured smart-home context
3. whole-home smart-home context
4. long-term smart-home memory

Higher layers may shape confidence or timing for lower layers only when they
are fresh and healthy.

### Freshness

Every smart-home-derived runtime fact should carry bounded freshness semantics.

Recommended V1 defaults:

- same-room motion support window: `15-90s`
- same-room button support window: `5-30s`
- other-room/home occupancy window: `30-300s`
- device-health freshness: provider-dependent, but fail stale after a bounded
  timeout

No stale smart-home fact may silently remain active forever.

### Failure mode

Twinr should fail closed:

- missing smart-home configuration disables `room_context`
- stale or offline streams disable smart-home-derived support
- provider room labels are metadata only
- cloud/provider failure must never manufacture local presence

The local PIR/camera/audio path must continue to function without
`smart_home`.

## Where `smart_home` Enters The Existing Runtime

### 1. Event normalization boundary

[`src/twinr/integrations/smarthome/runtime.py`](../src/twinr/integrations/smarthome/runtime.py)
already turns provider events into a compact `facts["smart_home"]` snapshot via
`SmartHomeObservationBuilder`.

This is the correct ingress boundary for generic vendor-neutral smart-home
signals:

- motion activity
- button presses
- alarm state
- device online/offline state
- recent normalized provider events

This layer should stay generic. It should not invent `near_device_presence`,
`room_context`, or `home_context` itself.

### 2. Realtime loop handoff

[`src/twinr/agent/workflows/realtime_runner.py`](../src/twinr/agent/workflows/realtime_runner.py)
already starts `SmartHomeSensorWorker` and forwards each
`SmartHomeObservation` through `handle_sensor_observation`.

This is the correct place to inject `smart_home` into the shared sensor
observation stream, but not the place to interpret it deeply.

### 3. Shared sensor-observation queue

[`src/twinr/agent/workflows/realtime_runtime/background.py`](../src/twinr/agent/workflows/realtime_runtime/background.py)
already stores the newest sensor facts and runs sensor-triggered automations
from that queue.

This is the cleanest place to add one focused context-builder module later:

- input: merged live facts including `facts["smart_home"]`
- output:
  - `near_device_presence` snapshot
  - `room_context` snapshot
  - `home_context` snapshot

That builder should live in its own proactive/runtime module, not inside
`background.py` or `realtime_runner.py`.

### 4. Proactive runtime consumption

The proactive runtime modules should then consume the layered context, not raw
provider payloads.

Recommended consumption split:

- `presence.py`
  - consume `near_device_presence`
  - optionally consult `room_context` only as corroboration or veto
- social trigger engine
  - consume `near_device_presence` first
  - use `room_context` and `home_context` for timing, safety, ambiguity, and
    confidence shaping
- attention / initiative / identity modules
  - remain local-first
  - treat smart-home context as weak environment support only

### 5. Automation and long-term memory paths

These two paths already exist and should remain separate from live local
presence truth:

- sensor-triggered automations consume normalized `smart_home` events via
  [`src/twinr/automations/sensors.py`](../src/twinr/automations/sensors.py)
- long-horizon smart-home environment markers are extracted in
  [`src/twinr/memory/longterm/ingestion/multimodal.py`](../src/twinr/memory/longterm/ingestion/multimodal.py)

Those paths should feed policy and memory, not overwrite `near_device_presence`.

## Recommended V1 Runtime Contract

The runtime should converge on one explicit layered fact shape:

```json
{
  "near_device_presence": {
    "occupied_likely": true,
    "person_visible": true,
    "speech_recent": false,
    "confidence": 0.91,
    "source_summary": "pir+camera"
  },
  "room_context": {
    "available": true,
    "same_room_motion_recent": true,
    "same_room_button_recent": false,
    "stale": false,
    "confidence": 0.63
  },
  "home_context": {
    "home_occupied_likely": true,
    "other_room_motion_recent": true,
    "alarm_active": false,
    "stream_healthy": true,
    "confidence": 0.58
  }
}
```

Contract rules:

- each layer exposes its own confidence and freshness
- no layer silently aliases another
- local consumers depend on the smallest layer they actually need
- raw `facts["smart_home"]` remains available for automations and diagnostics

## What Smart-Home Should Influence

Safe influence areas:

- proactive prompt timing
- ambiguity guards
- concern corroboration
- home occupancy / away context
- stream-health gating
- long-term routine interpretation

Unsafe influence areas:

- local voice-activation arming by itself
- `person_visible`
- hard user identity
- direct speech targeting of a specific person
- emergency or medical claims

## Implementation Direction

The next code step should be one dedicated runtime module that converts the
current raw sensor observation surface into the three explicit context layers.

That module should:

- read local PIR/camera/audio facts plus optional `facts["smart_home"]`
- apply explicit room mapping only when configured
- compute freshness and stream-health gates
- publish `near_device_presence`, `room_context`, and `home_context`
- leave orchestration files thin

This keeps strict separation of concerns intact:

- `smarthome` owns normalization
- realtime runner owns worker lifecycle and handoff
- background runtime owns queueing/orchestration
- proactive runtime owns context derivation and consumption
