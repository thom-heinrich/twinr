# Sensor Memory V1

## Purpose

This document turns the sensor-fusion target field into an implementation plan for a first useful `sensor memory` layer.

The goal of V1 is not rich ambient intelligence. The goal is:

- remember simple recurring sensor patterns
- detect a few meaningful deviations
- use those patterns to make Twinr calmer, more contextual, and more human
- keep the whole path inspectable, bounded, and Pi-friendly

This should build on the current long-term memory and proactive stack, not replace it.

## Product Outcome

If V1 works, Twinr should become better at:

- noticing that a typical presence window seems unusually quiet
- offering help at the right time because it has seen similar interaction patterns before
- sounding more context-aware without pretending to know hidden emotions

Examples:

- "Es ist heute ungewoehnlich ruhig. Ich wollte nur kurz fragen, ob alles in Ordnung ist."
- "Moechtest du das wieder ausdrucken?"
- "Soll ich mir das anschauen?"

## What Already Exists

Twinr already has the core building blocks:

- live proactive facts from PIR, audio, and camera
- multimodal long-term evidence persistence
- low-confidence daypart pattern objects
- reflection over repeated evidence
- long-term proactive planning/policy
- a proactive governor and vision review veto path

Relevant current paths:

- `src/twinr/proactive/`
- `src/twinr/memory/longterm/multimodal.py`
- `src/twinr/memory/longterm/reflect.py`
- `src/twinr/memory/longterm/planner.py`
- `src/twinr/memory/longterm/proactive.py`

## What V1 Should Add

V1 adds four things:

1. routine aggregates over repeated sensor patterns
2. deviation objects for today's activity versus recent baseline
3. a bounded bridge from sensor-memory candidates into the live proactive path
4. conservative veto rules so memory alone never speaks

## Non-Goals

V1 should explicitly not do these:

- identity recognition
- emotion classification
- medical claims
- black-box anomaly detection
- minute-by-minute surveillance summaries

## Design Rules

- Keep all sensor-memory logic local and inspectable.
- Reuse existing `LongTermMemoryObjectV1` instead of inventing a second store.
- Prefer `pattern` and `summary` objects with structured attributes over opaque blobs.
- Memory may raise or lower priority, but memory alone must not force a spoken prompt.
- Deviation prompts must require live confirmation from current sensors.
- The Pi runtime must stay bounded; no heavy per-frame historical analysis in the hot path.

## Data Model

### Reuse Existing Object Types

V1 should continue using:

- `kind="pattern"`
- `kind="summary"`

That avoids widening the long-term object taxonomy before it is necessary.

### New Pattern Families

These are the recommended new durable objects.

#### 1. Presence routine

Example ids:

- `routine:presence:weekday:morning`
- `routine:presence:weekend:afternoon`

Recommended fields:

- `kind="pattern"`
- `slot_key="routine:presence:{weekday_class}:{daypart}"`
- `value_key="presence_routine"`
- `attributes.memory_domain="sensor_routine"`
- `attributes.routine_type="presence"`
- `attributes.weekday_class="weekday|weekend|all_days"`
- `attributes.daypart="morning|afternoon|evening|night"`
- `attributes.days_observed`
- `attributes.days_with_presence`
- `attributes.recent_support_count`
- `attributes.last_observed_date`
- `attributes.baseline_window_days`
- `attributes.typical_presence_ratio`

#### 2. Interaction routine

Example ids:

- `routine:interaction:conversation_start:weekday:morning`
- `routine:interaction:print:weekday:afternoon`
- `routine:interaction:camera_use:weekday:afternoon`
- `routine:interaction:camera_showing:weekday:afternoon`

Recommended fields:

- `kind="pattern"`
- `attributes.memory_domain="sensor_routine"`
- `attributes.routine_type="interaction"`
- `attributes.interaction_type`
- `attributes.weekday_class`
- `attributes.daypart`
- `attributes.days_observed`
- `attributes.days_with_interaction`
- `attributes.typical_interaction_ratio`
- `attributes.last_observed_date`
- `attributes.support_count`

#### 3. Deviation summaries

Example ids:

- `deviation:presence:weekday:morning:2026-03-15`
- `deviation:interaction:print:weekday:afternoon:2026-03-15`

Recommended fields:

- `kind="summary"`
- `attributes.summary_type="sensor_deviation"`
- `attributes.memory_domain="sensor_routine"`
- `attributes.deviation_type="missing_presence|low_activity|missing_interaction"`
- `attributes.weekday_class`
- `attributes.daypart`
- `attributes.date`
- `attributes.expected_ratio`
- `attributes.current_ratio`
- `attributes.delta_ratio`
- `attributes.baseline_window_days`
- `attributes.requires_live_confirmation=true`

These should usually be `candidate`, not immediately `active`.

## Aggregation Plan

### Current raw sources to reuse

V1 should aggregate from the evidence that already exists:

- `pattern:presence:{daypart}:near_device`
- `pattern:camera_interaction:{daypart}`
- `pattern:button:*:{daypart}`
- `pattern:print:*:{daypart}`
- `pattern:camera_use:*:{daypart}`

Plus current-day live facts from the proactive monitor.

### Minimal aggregation dimensions

V1 should aggregate only across:

- `daypart`
- `weekday_class = weekday | weekend`
- `recent window`, for example 14 or 21 days

Do not start with per-hour histograms. The first useful version should stay small.

### Minimal aggregation metrics

For each routine bucket:

- `days_observed`
- `days_with_signal`
- `ratio = days_with_signal / days_observed`
- `last_observed_date`
- `support_count`

For deviations:

- `today_has_signal`
- `expected_ratio`
- `delta_ratio`

This is enough for a first pass.

## Proposed Module Changes

### 1. New aggregator module

Recommended new module:

- `src/twinr/memory/longterm/sensor_memory.py`

Responsibilities:

- load active multimodal pattern objects
- group them by `weekday_class` and `daypart`
- build routine pattern objects
- build same-day deviation summary candidates

This module should stay deterministic and pure.

### 2. Service wiring

Recommended extension points:

- `src/twinr/memory/longterm/service.py`
- `src/twinr/memory/longterm/reflect.py` or a sibling small compiler path

Pragmatic V1 approach:

- keep `multimodal.py` as raw extractor
- keep `reflect.py` for support-count promotion
- add `sensor_memory.py` as a separate bounded aggregator layer over active objects

### 3. Proactive bridge

Recommended extension points:

- `src/twinr/memory/longterm/planner.py`
- `src/twinr/proactive/runtime/service.py`
- `src/twinr/proactive/governance/governor.py`

Planner should emit candidates such as:

- `routine_check_in`
- `routine_print_offer`
- `routine_camera_offer`

But these candidates should only be deliverable when the live runtime confirms enough current evidence.

## Sensor-Memory -> Proactive Bridge

### Candidate types

V1 should add only three long-term candidate kinds:

- `routine_check_in`
- `routine_print_offer`
- `routine_camera_offer`

### Live confirmation rules

#### `routine_check_in`

Needs both:

- a same-day deviation candidate such as missing expected presence
- current live support, for example one of:
  - recent PIR motion after long quiet
  - person visible and quiet
  - a fresh return after an unusually quiet expected window

Should not speak if:

- the room is still empty
- a recent proactive prompt already fired
- the deviation signal is weak

#### `routine_print_offer`

Needs both:

- a learned pattern that print tends to follow similar interactions
- a current live context such as:
  - recent relevant conversation turn
  - print-worthy answer available
  - no recent print already completed

#### `routine_camera_offer`

Needs both:

- a learned camera/showing interaction pattern
- current live context such as:
  - `showing_object`
  - `looking_toward_device`
  - `person_visible`

### Hard veto rules

These should remain hard:

- no prompt from long-term routine memory alone
- no speaking if runtime is busy
- no speaking if proactive governor blocks it
- no speaking if image-driven review vetoes it
- no concern prompt unless current sensors support it now

## First Three V1 Use Cases

### 1. Presence routine deviation

Intent:

- make Twinr slightly more caring without being invasive

Input:

- expected presence routine for this weekday class and daypart
- today unusually little presence so far
- later a live confirming cue

Output:

- a soft `routine_check_in` candidate

Example prompt style:

- "Ich wollte nur kurz nachfragen, ob alles in Ordnung ist."

### 2. Interaction routine continuity for print

Intent:

- make Twinr feel helpful and familiar

Input:

- repeated pattern that prints often follow certain interactions
- a current relevant conversation or answer

Output:

- `routine_print_offer`

Example prompt style:

- "Soll ich dir das auch ausdrucken?"

### 3. Camera interaction continuity

Intent:

- make showing things feel natural

Input:

- repeated camera-showing routine
- current `showing_object` or `looking_toward_device`

Output:

- `routine_camera_offer`

Example prompt style:

- "Moechtest du mir das zeigen?"

## Rollout Plan

### Phase 1

- add deterministic sensor-memory aggregation
- persist routine objects
- no speaking yet
- expose review visibility in memory/debug pages if useful

### Phase 2

- planner emits `routine_*` candidates
- governor/policy handles cooldowns
- still fail closed unless live confirmation exists

### Phase 3

- enable the three V1 spoken use cases
- measure annoyance rate and usefulness
- only then consider broader deviation prompts

## Recommended Config Surface

V1 will likely need a few settings, but keep them small:

- `TWINR_SENSOR_MEMORY_ENABLED`
- `TWINR_SENSOR_MEMORY_BASELINE_DAYS`
- `TWINR_SENSOR_MEMORY_MIN_DAYS_OBSERVED`
- `TWINR_SENSOR_MEMORY_MIN_ROUTINE_RATIO`
- `TWINR_SENSOR_MEMORY_DEVIATION_MIN_DELTA`

Do not expose many weights in V1.

## Evaluation

V1 should be accepted with bounded offline and Pi checks.

### Deterministic tests

- aggregation from seeded multimodal objects
- weekday/weekend and daypart grouping
- deviation candidate generation
- planner candidate generation
- veto rules

### Pi validation

Because this changes runtime-facing proactive behavior, final acceptance must be on `/twinr`.

Suggested Pi checks:

- no prompt from empty room + memory alone
- no prompt from old deviation object without live confirmation
- print offer only when a print-worthy response exists
- camera offer only when live `showing_object`-style facts exist

## Risks

- too many routine objects with too little evidence
- prompting from stale memory without live confirmation
- accidental overlap with the normal proactive trigger engine
- making the device feel "monitoring" instead of helpful

V1 should bias strongly toward under-triggering.

## Recommended Bottom Line

The right V1 is small:

- learn a few daypart/weekday routines
- detect a few simple deviations
- bridge them into only three gentle proactive behaviors
- always require live sensor confirmation

That is enough to make Twinr feel noticeably more contextual and human without making it creepy or unstable.
