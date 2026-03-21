# Smart-Home Environment Profile V1

## Purpose

This document defines the first Twinr data model for learning long-term
behavior patterns from passive smart-home motion sensors without assuming that
any sensor belongs to a known room such as `kitchen`, `bathroom`, or
`bedroom`.

The goal is not to build a surveillance log or a room-semantic ontology. The
goal is:

- learn a stable profile of one person's overall home activity
- detect meaningful deviations over hours, days, and weeks
- stay robust when room labels are missing, wrong, or change over time
- keep the whole path inspectable, bounded, and Raspberry-Pi-friendly

This document is intentionally motion-first. It targets what Twinr can compute
from:

- local PIR sensors
- Hue motion sensors
- future motion-capable smart-home providers through the generic `smarthome`
  layer

## Why Room Labels Are The Wrong Foundation

Real homes are messy:

- sensor labels are often absent or operator-written
- labels may describe furniture, not rooms
- the same person may move house or rename devices
- merged Hue homes and multi-bridge setups do not imply meaningful room
  semantics

Twinr therefore should not hard-require a map like:

- `sensor A = kitchen`
- `sensor B = bathroom`

The more stable abstraction is:

- `environment`
  - one monitored home or flat
- `node`
  - one motion-sensing point in that environment
- `transition`
  - one observed movement from node A to node B
- `activity profile`
  - repeated timing, spread, fragmentation, and transition patterns of the
    whole environment

Later versions may infer latent functional zones such as `entry-like` or
`night-anchor`, but V1 must not depend on them.

## Design Rules

- Keep raw event retention short and marker retention long.
- Learn per person and per environment, not across households.
- Prefer robust marker aggregates over raw motion-count dumps.
- Separate behavior deviations from sensor-health failures.
- Support multiple timescales:
  - immediate
  - daily
  - weekly
  - slow drift
- Do not emit human-facing concern from one marker alone.
- Keep explanations simple:
  - what changed
  - how much
  - over which time range
  - whether sensor quality is sufficient

## Core Terms

### Environment

One physical monitored living space. This is the stable unit for longitudinal
profiling.

### Node

One motion-sensing point, identified by provider and routed entity id, not by a
human room label.

### Activity Epoch

A fixed-width time bucket used to turn sparse motion events into stable
aggregates.

Recommended V1 epoch width:

- `5 minutes` for daily and weekly activity profiles

### Transition

A directed edge from one node to another when motion is seen on node B shortly
after node A.

Recommended V1 transition window:

- `15-90 seconds`, configurable per provider density

### Baseline

A rolling person-specific expectation for one marker.

Recommended V1 baseline windows:

- `14 days` for short-term expected behavior
- `42 days` for slow drift comparison

### Deviation

A statistically or structurally meaningful departure from baseline. Deviations
are typed and explainable, not just one opaque anomaly score.

## Storage Strategy

### Short-lived raw layer

Keep raw motion events only as long as needed for debugging, replay, and marker
rebuilds.

Recommended V1 retention:

- `7-30 days` raw routed smart-home events

### Long-lived profile layer

Persist only compact derived objects:

- node registry
- epoch summaries
- daily marker sets
- weekly rolling baselines
- typed deviations

This keeps the system privacy-bounded and avoids an ever-growing event dump.

## V1 Schema

The profile should be built from five object families.

### 1. Environment Node Registry

This object tracks which motion nodes exist without inventing room semantics.

```json
{
  "schema_version": "v1",
  "kind": "environment_node",
  "environment_id": "home:main",
  "node_id": "node:motion:route:192.168.178.22:c19fafae-...",
  "provider": "hue",
  "source_entity_ids": [
    "route:192.168.178.22:c19fafae-..."
  ],
  "capabilities": [
    "motion"
  ],
  "first_seen_at": "2026-03-21T10:42:11Z",
  "last_seen_at": "2026-03-21T10:42:11Z",
  "health": {
    "online": true,
    "battery_ok": true,
    "stream_ok": true
  },
  "attributes": {
    "provider_label": "Esszimmer 1",
    "provider_area_label": "Erdgeschoss"
  }
}
```

Rules:

- `provider_label` and `provider_area_label` are optional metadata only.
- Behavior logic must not depend on them.

### 2. Environment Epoch Summary

This is the minimal room-agnostic building block for daily profiling.

```json
{
  "schema_version": "v1",
  "kind": "environment_epoch",
  "environment_id": "home:main",
  "epoch_start": "2026-03-21T08:00:00Z",
  "epoch_width_s": 300,
  "active": true,
  "active_node_count": 3,
  "active_node_ids": [
    "node:motion:route:...:1",
    "node:motion:route:...:2",
    "node:motion:route:...:3"
  ],
  "motion_event_count": 8,
  "transition_count": 2,
  "quality_flags": []
}
```

Rules:

- Epochs are derived, not user-facing.
- `active_node_ids` may be omitted after downstream aggregates are computed if
  storage size becomes relevant.

### 3. Daily Environment Marker Set

This is the primary durable behavior profile object.

```json
{
  "schema_version": "v1",
  "kind": "environment_day_profile",
  "environment_id": "home:main",
  "date": "2026-03-21",
  "weekday_class": "weekday",
  "markers": {
    "active_epoch_count_day": 116,
    "first_activity_minute_local": 392,
    "last_activity_minute_local": 1295,
    "longest_daytime_inactivity_min": 147,
    "night_activity_epoch_count": 11,
    "unique_active_node_count_day": 8,
    "mean_active_node_count_per_active_epoch": 1.9,
    "node_entropy_day": 1.72,
    "dominant_node_share_day": 0.28,
    "transition_count_day": 53,
    "transition_entropy_day": 2.11,
    "fragmentation_index_day": 0.43,
    "motion_burst_count_day": 18,
    "circadian_similarity_14d": 0.81,
    "sensor_coverage_ratio_day": 0.94
  },
  "quality_flags": [],
  "supporting_ranges": {
    "night_start_minute_local": 1320,
    "night_end_minute_local": 360
  }
}
```

Rules:

- Store the full marker vector even when only a subset currently drives
  deviation logic.
- `quality_flags` must explain why a day should be down-weighted:
  - `sensor_outage_suspected`
  - `bridge_unreachable`
  - `reconfiguration_recent`
  - `visitor_suspected`

### 4. Rolling Baseline Object

This object stores what is normal for this environment and resident over a
rolling window.

```json
{
  "schema_version": "v1",
  "kind": "environment_baseline",
  "environment_id": "home:main",
  "baseline_id": "weekday:rolling_14d",
  "window_days": 14,
  "weekday_class": "weekday",
  "sample_count": 12,
  "marker_stats": {
    "active_epoch_count_day": {
      "median": 123.0,
      "iqr": 19.0,
      "ewma": 120.4
    },
    "first_activity_minute_local": {
      "median": 401.0,
      "iqr": 24.0,
      "ewma": 398.2
    },
    "fragmentation_index_day": {
      "median": 0.39,
      "iqr": 0.05,
      "ewma": 0.40
    }
  },
  "quality": {
    "usable": true,
    "minimum_days_met": true
  }
}
```

Rules:

- Use robust summary statistics such as `median` and `IQR`, not only means.
- Maintain separate baselines for:
  - `weekday`
  - `weekend`
  - optional `all_days`

### 5. Typed Deviation Object

Deviations must be interpretable and evidence-backed.

```json
{
  "schema_version": "v1",
  "kind": "environment_deviation",
  "environment_id": "home:main",
  "observed_at": "2026-03-21T19:00:00Z",
  "severity": "moderate",
  "deviation_type": "daily_activity_drop",
  "time_scale": "day",
  "markers": [
    {
      "name": "active_epoch_count_day",
      "observed": 84.0,
      "baseline_median": 123.0,
      "delta_ratio": -0.317
    },
    {
      "name": "fragmentation_index_day",
      "observed": 0.49,
      "baseline_median": 0.39,
      "delta_ratio": 0.256
    }
  ],
  "quality_flags": [],
  "blocked_by": [],
  "explanation": {
    "short_label": "less activity and more fragmentation than usual",
    "human_readable": "Activity is lower than this environment's recent weekday baseline and movement is more fragmented than usual."
  }
}
```

Recommended V1 deviation types:

- `extended_inactivity`
- `daily_activity_drop`
- `daily_activity_increase`
- `night_activity_increase`
- `late_start_of_day`
- `early_end_of_day`
- `transition_pattern_shift`
- `fragmentation_shift`
- `possible_sensor_failure`

## Priority Marker Set For Hue Motion And PIR

These markers are the highest-value V1 features that are realistically
computable from motion-only streams without room labels.

### Tier 1: Must-have markers

| Marker | Window | Why it matters | Motion-only ready |
|---|---|---|---|
| `active_epoch_count_day` | day | Robust total in-home activity proxy | Yes |
| `first_activity_minute_local` | day | Detects later-than-usual start of day | Yes |
| `last_activity_minute_local` | day | Detects earlier-than-usual end of day | Yes |
| `longest_daytime_inactivity_min` | day | Flags unusual long quiet periods | Yes |
| `night_activity_epoch_count` | day | Captures night rest disruption without room semantics | Yes |
| `unique_active_node_count_day` | day | Measures spatial spread of activity across the environment | Yes |
| `transition_count_day` | day | Proxy for movement through the home | Yes |
| `fragmentation_index_day` | day | Captures stop-start, choppy activity patterns | Yes |
| `circadian_similarity_14d` | day vs rolling baseline | Detects schedule drift against the person's own routine | Yes |
| `sensor_coverage_ratio_day` | day | Prevents sensor outages from looking like behavior change | Yes |

### Tier 2: Strong supporting markers

| Marker | Window | Why it matters | Motion-only ready |
|---|---|---|---|
| `mean_active_node_count_per_active_epoch` | day | Distinguishes broad vs narrow activity spread | Yes |
| `node_entropy_day` | day | Quantifies how evenly activity is distributed over nodes | Yes |
| `dominant_node_share_day` | day | Captures environment narrowing around one node or zone | Yes |
| `transition_entropy_day` | day | Captures whether movement paths become repetitive | Yes |
| `motion_burst_count_day` | day | Tracks bouts of activity vs long flat silence | Yes |

### Marker Definitions

#### `active_epoch_count_day`

Number of `5 minute` epochs with at least one motion event on any node.

#### `first_activity_minute_local`

Minute-of-day of the first active epoch after local midnight.

#### `last_activity_minute_local`

Minute-of-day of the last active epoch before local midnight.

#### `longest_daytime_inactivity_min`

Longest consecutive inactive period within a configurable daytime window, for
example `06:00-22:00`.

#### `night_activity_epoch_count`

Count of active epochs in the configured night window, for example
`22:00-06:00`.

#### `unique_active_node_count_day`

Number of distinct motion nodes active at least once during the day.

#### `transition_count_day`

Count of node-to-node transitions where a second node becomes active shortly
after a first node. This is the room-agnostic replacement for "room
transitions."

#### `fragmentation_index_day`

Probability of transitioning from `active` to `inactive` between adjacent
epochs:

- high values suggest choppy, fragmented movement
- low values suggest longer continuous activity runs

#### `circadian_similarity_14d`

Similarity between today's hourly activity vector and the rolling `14 day`
baseline vector for the same weekday class.

#### `sensor_coverage_ratio_day`

Fraction of expected healthy motion nodes that reported at least one valid
signal or health heartbeat. This is required to detect technical confounds.

#### `mean_active_node_count_per_active_epoch`

Average number of simultaneously active nodes in active epochs.

#### `node_entropy_day`

Entropy of daily node usage distribution. This is a room-agnostic diversity
measure of how spread out activity is across the environment.

#### `dominant_node_share_day`

Fraction of all node activations captured by the most active node.

#### `transition_entropy_day`

Entropy of the directed transition distribution. Low values indicate repeated
use of a small set of paths; high values indicate more varied movement paths.

#### `motion_burst_count_day`

Count of continuous active bouts separated by at least one inactive epoch.

## Marker Prioritization For V1

If Twinr only ships a minimal first implementation, build these markers first:

1. `active_epoch_count_day`
2. `first_activity_minute_local`
3. `last_activity_minute_local`
4. `longest_daytime_inactivity_min`
5. `night_activity_epoch_count`
6. `unique_active_node_count_day`
7. `transition_count_day`
8. `fragmentation_index_day`
9. `circadian_similarity_14d`
10. `sensor_coverage_ratio_day`

These ten markers cover:

- overall activity
- start/end timing
- sustained silence
- night disturbance
- spread across the environment
- movement structure
- daily rhythm drift
- technical trustworthiness

That is enough for a first meaningful deviation engine without room labels.

## Deviation Logic

V1 should not use one global anomaly score. It should combine typed marker
deviations into a small set of explicit deviation classes.

### Immediate deviations

- unusually long inactivity during a normally active daytime window
- abrupt loss of all sensor coverage

### Daily deviations

- materially lower activity than rolling weekday or weekend baseline
- unusually high night activity
- later start or earlier end than usual
- much narrower movement spread than usual

### Slow-drift deviations

- multi-week increase in fragmentation
- multi-week decrease in activity spread
- progressive schedule drift

## What V1 Should Not Try To Infer

The following remain out of scope:

- room names
- ADL labels such as `cooking` or `toileting`
- identity
- visitors with high confidence
- medical diagnosis
- emotion or cognition from motion alone

These require more context or different sensors.

## Recommended Implementation Path

### Phase 1

- ingest routed motion events from `smarthome`
- maintain a stable node registry
- compile `5 minute` environment epochs
- compute the ten must-have daily markers
- build rolling weekday/weekend baselines
- emit typed deviations with quality flags

### Phase 2

- add transition entropy and node entropy
- add reconfiguration detection when devices are added or removed
- add optional door and light couplings as extra markers

### Phase 3

- infer latent functional zones without hard room labels
- combine motion profile with other modalities such as door, bed, audio, or
  caregiver-confirmed context

## Research Notes

This design follows the direction most strongly supported by the current
literature:

- digital marker sets are more useful than raw event dumps
- transition-based and timing-based markers generalize better than room-name
  assumptions
- activity fragmentation and circadian regularity are clinically meaningful
  long-horizon markers
- anomaly outputs need human-readable explanations
- LLMs should summarize deviations, not replace the primary detection layer

Relevant sources:

- Sprint et al., 2016, behavior-change detection from smart-home sensors
- Schmitter-Edgecombe et al., 2024, digital markers predicting daily cognition
  and lifestyle behaviors
- Fritz and Cook, 2025, interrupted time-series detection of behavior changes
  during external events
- recent motion-complexity and frailty work using longitudinal ambient sensing

