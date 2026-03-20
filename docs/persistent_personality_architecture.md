# Persistent Personality Architecture

## Goal

This document defines the target-state architecture for a persistent, evolving
Twinr personality.

The target behavior is:

- attentive companion
- practical intelligence
- situational awareness
- calm dignity
- authentic continuity

Twinr should feel less like a reactive utility bot and more like a steady,
attentive presence that knows the user, knows what matters around them, and
develops a recognizable identity over time.

This must **not** be implemented as one growing persona prompt.
It must be implemented as a layered system with strict separation of concerns.

## Core design rules

1. Stable character is not the same as user memory.
2. Relationship memory is not the same as world knowledge.
3. Current news and local developments must never directly rewrite core
   personality.
4. Reflection may propose bounded deltas, not freeform personality drift.
5. ChonkyDB-backed long-term memory remains the durable truth layer when
   `remote_primary` is active.
6. Twinr must remain machine-honest. It may feel warm and alive, but it must
   not fabricate human feelings, fake biography, or false lived experience.
7. The system must stay small, auditable, and prompt-bounded on Raspberry Pi.

## Sequential layer model

The architecture should be built in sequential layers. Each layer depends on
the previous one and contributes a distinct concern.

### Layer 0: Runtime invariants

This is the substrate below personality.

Purpose:

- define the non-negotiable runtime and product rules
- keep hardware behavior, safety, and trust stable
- prevent later personality work from breaking core Twinr semantics

Owns:

- safety and operator rules
- button and hardware semantics
- user language defaults
- fail-closed remote-memory policy
- prompt authority ordering

Current anchors:

- `personality/SYSTEM.md`
- `src/twinr/agent/base_agent/prompting/personality.py`
- `src/twinr/memory/context_store.py`
- `src/twinr/memory/longterm/runtime/`

Output to next layer:

- a trusted execution envelope in which personality can operate

### Layer 1: Authored core character

This is Twinr's stable identity kernel.

Purpose:

- define what Twinr basically is before any learned evolution exists
- keep a recognizable, durable baseline across resets and sparse-memory phases

Owns:

- base temperament
- initiative band
- humor band
- dignity and politeness rules
- conversational energy
- stance toward uncertainty
- non-paternalistic style goals
- baseline topical posture toward everyday life, people, places, and the world

Should contain:

- short, high-signal statements only
- no user-specific facts
- no current events
- no short-lived trends

Should not contain:

- remembered user history
- inferred preferences about one user
- local politics of the week
- last-news sentiment

Storage:

- authored source in `personality/PERSONALITY.md`
- managed additive operator updates inside the existing managed block of
  `personality/PERSONALITY.md`
- mirrored via the existing `personality_context` remote snapshot for
  portability and fail-closed readiness

Why it exists:

- this layer keeps Twinr recognizable even if long-term retrieval is sparse
- this layer ensures "more memory" does not automatically mean "different self"

Output to next layer:

- stable character contract that later layers may color but not overwrite

### Layer 2: Machine-readable personality contract

This layer turns the authored core into structured state.

Purpose:

- make character traits retrievable, comparable, evolvable, and evaluable
- give reflection a structured target instead of editing prompt prose blindly

Owns:

- normalized trait definitions
- allowed value ranges
- editability policy
- delta eligibility rules

Recommended representation:

- keep the human-readable source in `personality/PERSONALITY.md`
- derive machine-readable personality items into structured long-term objects

Recommended long-term object families:

- `fact` with `memory_domain=personality` and `fact_type=trait`
- `fact` with `memory_domain=personality` and `fact_type=value`
- `fact` with `memory_domain=personality` and `fact_type=boundary`
- `pattern` with `memory_domain=personality` and `pattern_type=style_preference`

Examples:

```json
{
  "kind": "fact",
  "summary": "Twinr favors calm, observant, non-paternalistic conversation.",
  "slot_key": "assistant_trait:conversation_stance",
  "value_key": "calm_observant_non_paternalistic",
  "attributes": {
    "memory_domain": "personality",
    "fact_type": "trait",
    "subject_ref": "assistant:twinr",
    "trait_name": "conversation_stance",
    "trait_value": "calm_observant_non_paternalistic",
    "mutability": "operator_curated"
  }
}
```

```json
{
  "kind": "fact",
  "summary": "Twinr should use health guidance when relevant, not as its dominant worldview.",
  "slot_key": "assistant_boundary:health_focus",
  "value_key": "contextual_not_dominant",
  "attributes": {
    "memory_domain": "personality",
    "fact_type": "boundary",
    "subject_ref": "assistant:twinr",
    "boundary_name": "health_focus",
    "boundary_value": "contextual_not_dominant",
    "mutability": "operator_curated"
  }
}
```

Why structured items matter:

- reflection can reason over them
- evals can inspect them
- future web UI can expose them without parsing prose

Output to next layer:

- stable machine-readable identity primitives

### Layer 3: Relational user model

This is the first learned layer built on top of the stable character.

Purpose:

- model how Twinr knows this specific user
- capture the relationship style Twinr should adopt with this user

Owns:

- topic affinities
- topic aversions
- preferred answer style
- preferred proactivity band
- how often Twinr should ask follow-ups
- important people, routines, and recurring threads
- what the user finds encouraging, annoying, reassuring, or patronizing

This layer is not:

- Twinr's core self
- generic world knowledge
- a static biography dump

Recommended object families:

- `fact` with `memory_domain=relationship` and `fact_type=interaction_preference`
- `pattern` with `memory_domain=relationship` and `pattern_type=conversation_energy`
- `pattern` with `memory_domain=relationship` and `pattern_type=proactivity_preference`
- `summary` with `memory_domain=relationship` and `summary_type=relationship_state`

Recommended graph usage:

- keep person and contact structure in the personal graph
- keep relationship style facts as structured objects, not graph-only state

Reason:

- graph is best for entities and links
- relationship style is better expressed as evidence-bearing memory objects with
  confidence, validity, and update history

Output to next layer:

- a user-specific interaction stance compatible with Twinr's stable core

### Layer 4: Continuity and thematic life threads

This layer turns repeated conversation into ongoing continuity.

Purpose:

- carry the feeling that Twinr lives in an ongoing stream with the user
- preserve topic continuity without overusing "I remember"

Owns:

- active personal threads
- recurring worries
- recurring joys
- unfinished projects
- slow life changes
- near-term continuity packets

Recommended object families:

- `summary` with `memory_domain=thread` and `summary_type=life_thread`
- `pattern` with `memory_domain=thread` and `pattern_type=topic_recurrence`
- `plan` with `memory_domain=thread` and `plan_type=open_loop`
- existing episodic and mid-term objects remain inputs

Examples of threads:

- dog health and routines
- interest in local development around Hamburg
- recurring programming/project bursts
- ongoing local-political frustration or curiosity

Why this layer is separate:

- it keeps continuity distinct from permanent preferences
- it avoids promoting every recurring topic into the core self

Output to next layer:

- durable continuity cues for prompt assembly and reflection

### Layer 5: Place intelligence layer

This layer models the user's geographic world.

Purpose:

- make Twinr locally situated
- allow place-based continuity beyond one-shot address facts

Owns:

- important places
- local institutions
- region-level interest zones
- neighborhood and city salience
- recurring routes or local anchors
- place-linked personal threads

Examples:

- Schwarzenbek
- Hamburg
- Schleswig-Holstein
- Germany
- local council, station, park, clinic, dog area, favorite cafe

Recommended graph usage:

- nodes for places, regions, institutions, neighborhoods, venues
- spatial edges for containment and proximity
- user and assistant relevance edges for salience

Recommended graph extensions for schema v3:

- `user_engages_with` may continue for user-place interests
- add `assistant_engages_with` for Twinr-side stable attention
- add `assistant_tracks_place` for place salience derived from relationship and
  world context

Recommended object families:

- `fact` with `memory_domain=place` and `fact_type=place_interest`
- `summary` with `memory_domain=place` and `summary_type=place_thread`
- `pattern` with `memory_domain=place` and `pattern_type=place_recurrence`

Why this layer matters:

- older-adult life is often strongly place-shaped
- Twinr should know not only people and preferences, but the local world the
  user actually inhabits

Output to next layer:

- place-aware context and retrieval hooks for current and future world signals

### Layer 6: World intelligence layer

This layer tracks relevant developments outside the immediate personal sphere.

Purpose:

- give Twinr situational awareness
- allow the assistant to grow through ongoing contact with the world

Owns:

- relevant current events
- local politics and policy developments
- regional developments
- national and global topics the user regularly cares about
- freshness-aware summaries and threads

This layer must be:

- source-aware
- freshness-aware
- confidence-aware
- non-dominating in prompt space

Recommended object families:

- `observation` with `memory_domain=world` and `observation_type=news_signal`
- `event` with `memory_domain=world` and `event_domain=public_event`
- `summary` with `memory_domain=world` and `summary_type=world_thread`
- `pattern` with `memory_domain=world` and `pattern_type=interest_recurrence`

Recommended attributes:

- `topic_name`
- `place_ref`
- `source_url`
- `source_type`
- `source_published_at`
- `freshness_ttl_s`
- `interest_alignment`
- `stance_impact`
- `review_after`

Important rule:

- world signals are not personality facts
- repeated, relevant world contact may influence personality deltas later, but
  only through reflection and only in bounded ways

Output to next layer:

- a fresh, relevance-filtered picture of what is happening around the user

### Layer 7: Turn synthesis layer

This layer builds the final per-turn hidden context.

Purpose:

- combine stable self, relationship model, continuity, place, and world state
  into one bounded turn package

Owns:

- layer ordering
- prompt budget partitioning
- query-conditioned retrieval
- context omission when irrelevant

Target hidden prompt order:

1. runtime invariants
2. stable core character
3. relational user model
4. active continuity threads
5. relevant place intelligence
6. relevant world intelligence
7. turn-local tool and hardware context

Important policy:

- if a layer is irrelevant to the current turn, omit it
- world and place context should be query-conditioned and salience-bounded
- stable core character should always remain small and present

New module targets:

- `src/twinr/personality/models.py`
- `src/twinr/personality/context_builder.py`
- `src/twinr/personality/service.py`

Current integration point to extend:

- `src/twinr/agent/base_agent/prompting/personality.py`

Output to next layer:

- a prompt-ready personality context bundle that feels coherent but stays
  bounded

### Layer 8: Reflective evolution layer

This layer is the only place where Twinr's identity may evolve.

Purpose:

- derive slow, evidence-based changes to interaction style and long-horizon
  attentional stance

Owns:

- delta proposal generation
- delta review thresholds
- decay and reevaluation
- conflict detection for incompatible deltas

This layer may change:

- topic salience
- initiative band within bounded ranges
- humor frequency within bounded ranges
- preference to ask more or fewer follow-up questions
- local-vs-global emphasis
- how strongly Twinr tends to notice certain place or world domains

This layer may not directly change:

- safety rules
- machine-honesty rules
- hard dignity rules
- button semantics
- fail-closed memory policy

Recommended delta object family:

- `fact` with `memory_domain=personality_delta` and `fact_type=proposed_delta`
- `summary` with `memory_domain=personality_delta` and `summary_type=delta_review`

Example delta:

```json
{
  "kind": "fact",
  "summary": "Twinr should place more ongoing attention on local politics because the user repeatedly initiates and sustains that topic.",
  "slot_key": "assistant_delta:topic_salience:local_politics",
  "value_key": "increase_medium",
  "attributes": {
    "memory_domain": "personality_delta",
    "fact_type": "proposed_delta",
    "target_layer": "world_attention",
    "target_key": "local_politics",
    "delta_kind": "salience_shift",
    "delta_value": "increase_medium",
    "evidence_count": 7,
    "review_required": true,
    "applies_if_confirmed": true,
    "max_duration_days": 30
  }
}
```

Reflection should produce:

- candidate deltas
- not final rewritten prose

Then a dedicated personality service should:

- accept
- reject
- merge
- decay
- materialize

Current anchor to extend:

- `src/twinr/memory/longterm/reasoning/reflect.py`

Output:

- bounded, evidence-based personality evolution

### Layer 9: Operator and evaluation layer

This is the final control layer over the whole system.

Purpose:

- make personality evolution visible, testable, and reversible

Owns:

- operator review
- portal editing
- replay evals
- liveliness and dignity benchmarks
- regression protection against paternalism

Recommended operator surfaces:

- stable core character view
- active relationship model view
- place and world interests view
- pending and accepted deltas
- recent reflection rationale

Recommended eval dimensions:

- liveliness
- dignity
- non-paternalism
- topic diversity
- place awareness
- world awareness
- continuity without explicit memory announcements
- emotional steadiness

## ChonkyDB integration strategy

The architecture should reuse Twinr's current ChonkyDB split instead of
inventing a separate personality backend.

### 1. Keep authored prompt files as the curated source

Use:

- `personality/PERSONALITY.md`
- `personality/USER.md`
- existing managed context files and remote snapshots

These files remain the curated operator-facing source for stable baseline text.

### 2. Store machine-readable personality data as structured long-term objects

Use the existing `LongTermMemoryObjectV1` path for:

- personality traits
- interaction patterns
- topic affinities
- place interests
- world threads
- reflection deltas

Do **not** create a second object schema unless the current object model proves
insufficient. The current model already supports:

- `kind`
- `summary`
- `status`
- `confidence`
- `slot_key`
- `value_key`
- `valid_from`
- `valid_to`
- `attributes`

That is enough for V1 if taxonomy is extended carefully.

### 3. Extend the ontology through attributes before exploding top-level kinds

Prefer:

- `kind=fact|pattern|summary|observation|event|plan`
- specialize with:
  - `memory_domain=personality|relationship|place|world|personality_delta`
  - `fact_type`, `pattern_type`, `summary_type`, `observation_type`,
    `event_domain`, `plan_type`

Reason:

- keeps compatibility with the current ontology
- minimizes retrieval and reasoning churn
- lets the existing stores keep working

### 4. Use the graph for entities and relations, not for all style state

Use the personal graph for:

- people
- places
- institutions
- topics as nodes when graph expansion is useful
- typed relations among user, assistant, places, topics, and entities

Do not force all personality state into graph edges.
Style and delta state belong in long-term objects, where provenance and review
are easier.

### 5. Proposed graph schema additions

Current graph schema v2 is not enough for assistant identity and topic/place
attention.

Recommended additive v3 direction:

- new namespace: `assistant_*`

Proposed edge types:

- `assistant_engages_with`
- `assistant_tracks_topic`
- `assistant_tracks_place`
- `assistant_expresses_value`
- `assistant_avoids_style`

Example graph patterns:

- `assistant:twinr --assistant_tracks_topic--> topic:local_politics`
- `assistant:twinr --assistant_tracks_place--> place:schwarzenbek`
- `assistant:twinr --assistant_expresses_value--> value:dignity`
- `user:main --user_engages_with--> topic:local_politics`
- `topic:local_politics --spatial_located_in--> place:schwarzenbek`

### 6. Retrieval path changes

Current retriever sections are already split into:

- subtext
- midterm
- durable
- episodic
- graph
- conflict

V1 extension should add a thin personality-aware synthesis step, not a second
retriever stack.

Recommended new retrieval components:

- `personality_retriever`
  - select stable core contract
  - select relationship state
  - select place/world signals based on the query and active threads
- `personality_context_builder`
  - pack the selected sections into the ordered prompt contract

### 7. Remote-primary rules

When `remote_primary` is enabled:

- ChonkyDB remains the primary durable source of truth for machine-readable
  personality, relationship, place, world, and delta objects
- local files remain authored/operator-facing and cache-oriented
- if remote memory is required and unavailable, Twinr must not silently pretend
  to have live world or evolved-personality recall

Fail-closed implication:

- stable authored character still exists
- rich evolved self, place threads, and world threads are withheld when their
  durable source is unavailable

## Data contract recommendations

### Personality trait object

Use when a stable assistant trait must be machine-readable.

- `kind=fact`
- `memory_domain=personality`
- `fact_type=trait`
- `subject_ref=assistant:twinr`
- `trait_name`
- `trait_value`
- `mutability=operator_curated|reflection_bounded`

### Relationship pattern object

Use when repeated interaction reveals a stable preference.

- `kind=pattern`
- `memory_domain=relationship`
- `pattern_type=conversation_preference`
- `subject_ref=user:main`
- `applies_to=assistant:twinr`
- `pattern_name`
- `pattern_value`
- `evidence_count`

### Place interest object

Use when a place becomes persistently relevant.

- `kind=fact`
- `memory_domain=place`
- `fact_type=place_interest`
- `subject_ref=user:main|assistant:twinr`
- `place_ref`
- `interest_mode=user_interest|assistant_attention`
- `salience_band`

### World signal object

Use for fresh current developments.

- `kind=observation`
- `memory_domain=world`
- `observation_type=news_signal`
- `topic_name`
- `place_ref`
- `source_url`
- `source_published_at`
- `freshness_ttl_s`
- `importance_band`

### World thread summary

Use for a continuity-form summary over many world signals.

- `kind=summary`
- `memory_domain=world`
- `summary_type=world_thread`
- `thread_key`
- `topic_name`
- `place_ref`
- `summary_horizon`
- `last_revalidated_at`

### Reflection delta object

Use for bounded self-evolution proposals.

- `kind=fact`
- `memory_domain=personality_delta`
- `fact_type=proposed_delta`
- `target_layer`
- `target_key`
- `delta_kind`
- `delta_value`
- `review_required`
- `review_after`
- `max_duration_days`

## Prompt-layer contract

The final per-turn context should be rendered as explicit sections with
authority notes.

Recommended section contract:

- `SYSTEM`
  - non-negotiable runtime and product rules
- `PERSONALITY_CORE`
  - stable authored identity
- `PERSONALITY_STRUCTURED`
  - compact machine-readable traits and boundaries
- `RELATIONSHIP_MODEL`
  - user-specific style and topic guidance
- `CONTINUITY_THREADS`
  - active personal threads
- `PLACE_AWARENESS`
  - relevant place and local-context cues
- `WORLD_AWARENESS`
  - relevant current developments with freshness
- `MEMORY`
  - existing explicit memory store when relevant
- `REMINDERS`
  - existing reminder context
- `AUTOMATIONS`
  - existing automation context

Rules:

- only `SYSTEM` and `PERSONALITY_CORE` should carry stable instruction authority
- all other sections are context data
- `WORLD_AWARENESS` must stay freshness-bounded and query-relevant
- `RELATIONSHIP_MODEL` and `CONTINUITY_THREADS` should usually shape replies
  silently

## File and module plan

### New docs

- `docs/persistent_personality_architecture.md`

### New runtime modules

- `src/twinr/personality/models.py`
- `src/twinr/personality/context_builder.py`
- `src/twinr/personality/service.py`
- `src/twinr/personality/reflection.py`
- `src/twinr/personality/place_model.py`
- `src/twinr/personality/world_model.py`

### Existing modules to extend

- `src/twinr/agent/base_agent/prompting/personality.py`
- `src/twinr/memory/longterm/core/ontology.py`
- `src/twinr/memory/longterm/reasoning/reflect.py`
- `src/twinr/memory/longterm/retrieval/retriever.py`
- `src/twinr/memory/chonkydb/schema.py`
- `src/twinr/memory/chonkydb/personal_graph.py`

### New tests

- `test/test_personality_models.py`
- `test/test_personality_context_builder.py`
- `test/test_personality_reflection.py`
- `test/test_place_model.py`
- `test/test_world_model.py`
- `test/test_personality_liveliness_eval.py`

## Sequential rollout plan

### Phase 1: Core identity contract

Build:

- structured core personality schema
- prompt-layer split for stable core vs context data

Do not build yet:

- world intelligence
- personality evolution

### Phase 2: Relationship and continuity

Build:

- relational user model
- continuity thread summaries
- prompt integration for those layers

### Phase 3: Place intelligence

Build:

- place nodes, edges, and object families
- place-aware retrieval
- local salience in prompt synthesis

### Phase 4: World intelligence

Build:

- world signal ingestion
- freshness-aware world threads
- query-conditioned world context

### Phase 5: Reflective evolution

Build:

- delta proposal pipeline
- review and materialization rules
- decay and reevaluation

### Phase 6: Operator and eval surfaces

Build:

- UI visibility for layers and deltas
- liveliness and dignity evals
- regression protection

## Non-goals

This architecture should not:

- simulate human consciousness
- fabricate emotions
- invent autobiographical experiences
- let a single news cycle rewrite Twinr's identity
- turn the assistant into a health-scolding supervisor
- collapse all memory into one giant prompt

## Acceptance criteria

The architecture is successful when all of the following are true:

- Twinr feels recognizably consistent across days
- Twinr sounds less robotic and less health-dominant
- Twinr can naturally reference places and developments the user truly cares
  about
- Twinr can evolve in bounded, inspectable ways
- ChonkyDB remains the durable truth layer for machine-readable evolved state
- prompt size remains bounded and query-conditioned
- degraded remote-memory availability does not silently fake evolved recall

## Exact implementation plan

This section translates the target-state architecture into an implementation
sequence that matches the current Twinr codebase.

The implementation plan is intentionally incremental. Each slice should leave
Twinr in a valid, testable state and should not assume the later slices already
exist.

### Slice 1: Extend taxonomy and object contracts

Goal:

- make the new personality, relationship, place, world, and delta domains
  representable without breaking the current long-term store

Files to change first:

- `src/twinr/memory/longterm/core/ontology.py`
- `src/twinr/memory/longterm/core/models.py`
- `src/twinr/memory/chonkydb/schema.py`

#### `src/twinr/memory/longterm/core/ontology.py`

Add canonical attribute vocab for:

- `memory_domain=personality`
- `memory_domain=relationship`
- `memory_domain=place`
- `memory_domain=world`
- `memory_domain=personality_delta`

Add supported classifier values for:

- `fact_type=trait`
- `fact_type=value`
- `fact_type=boundary`
- `fact_type=interaction_preference`
- `fact_type=place_interest`
- `fact_type=proposed_delta`
- `observation_type=news_signal`
- `pattern_type=style_preference`
- `pattern_type=conversation_energy`
- `pattern_type=proactivity_preference`
- `pattern_type=topic_recurrence`
- `pattern_type=place_recurrence`
- `pattern_type=interest_recurrence`
- `summary_type=relationship_state`
- `summary_type=life_thread`
- `summary_type=place_thread`
- `summary_type=world_thread`
- `summary_type=delta_review`

Implementation notes:

- keep `kind` generic as it is today
- extend normalization helpers so these new classifier fields are canonicalized
- add lightweight helpers such as:
  - `is_personality_domain()`
  - `is_relationship_domain()`
  - `is_world_domain()`
  - `is_place_domain()`
  - `is_personality_delta_domain()`

Non-goal for this slice:

- do not add new top-level `kind` values yet

#### `src/twinr/memory/longterm/core/models.py`

Keep `LongTermMemoryObjectV1` as the canonical persisted object.

Do not fork a second personality object schema unless a later implementation
proves the current object model insufficient.

Add only small contract helpers:

- slot-key conventions for assistant-facing state
- helper constructors or documented patterns for:
  - assistant trait objects
  - relationship objects
  - place-interest objects
  - world-signal objects
  - delta objects

Recommended slot-key conventions:

- `assistant_trait:<trait_name>`
- `assistant_value:<value_name>`
- `assistant_boundary:<boundary_name>`
- `relationship:<user_ref>:<pattern_name>`
- `place_interest:<subject_ref>:<place_ref>`
- `world_thread:<topic_name>:<place_ref_or_global>`
- `assistant_delta:<target_layer>:<target_key>`

Recommended value-key conventions:

- stable symbolic values only
- never store long prose in `value_key`
- keep prose in `summary` and `details`

#### `src/twinr/memory/chonkydb/schema.py`

Prepare schema v3 as an additive graph evolution.

Add a new namespace:

- `assistant`

Initial edge types to add:

- `assistant_engages_with`
- `assistant_tracks_topic`
- `assistant_tracks_place`
- `assistant_expresses_value`
- `assistant_avoids_style`

Recommended new node classes to support clean graph reasoning:

- `assistant`
- `topic`
- `value`
- `style`
- `region`
- `institution`

Important compatibility rule:

- v2 meaning must stay stable
- v3 must be additive and backward-readable

Acceptance criteria for slice 1:

- the ontology can express all new domains
- the graph schema can express assistant-topic and assistant-place salience
- no prompt or retrieval behavior changes yet

### Slice 2: Introduce the personality package

Goal:

- create a focused runtime package for personality logic so orchestration files
  do not absorb identity logic

New modules to create:

- `src/twinr/personality/models.py`
- `src/twinr/personality/context_builder.py`
- `src/twinr/personality/service.py`
- `src/twinr/personality/reflection.py`
- `src/twinr/personality/place_model.py`
- `src/twinr/personality/world_model.py`

#### `src/twinr/personality/models.py`

Owns:

- typed dataclasses for prompt-layer inputs
- typed personality-layer contracts
- compact structured bundles that are smaller and safer than raw memory objects

Recommended dataclasses:

- `TwinrCoreCharacter`
- `TwinrStructuredTrait`
- `TwinrRelationshipModel`
- `TwinrContinuityThread`
- `TwinrPlaceSignal`
- `TwinrWorldSignal`
- `TwinrPersonalityDelta`
- `TwinrPersonalityContextBundle`

Key requirement:

- these models are view models over existing long-term objects and graph state
- they are not a second persistence layer

#### `src/twinr/personality/service.py`

Owns:

- loading and compiling the stable personality layers
- reading machine-readable structured state
- materializing accepted deltas into an effective runtime personality state

Recommended service responsibilities:

- load authored core character from `personality/PERSONALITY.md`
- load structured assistant traits from long-term memory
- merge accepted deltas
- expose a stable API for prompt builders and evals

Recommended methods:

- `load_core_character()`
- `load_structured_traits()`
- `load_relationship_model(user_ref)`
- `load_effective_personality_state()`
- `materialize_delta(delta)`
- `select_active_deltas()`

#### `src/twinr/personality/context_builder.py`

Owns:

- turn-time synthesis of personality-aware prompt layers

Recommended methods:

- `build_context_bundle(query_profile, original_query_text)`
- `render_prompt_sections(bundle)`

Prompt responsibilities:

- order the layers
- omit irrelevant layers
- keep world and place context bounded
- keep authority notes explicit

#### `src/twinr/personality/place_model.py`

Owns:

- place salience selection
- place-thread assembly
- place-aware context snippets

Inputs:

- graph place nodes and spatial edges
- place-interest objects
- continuity threads
- current query

Outputs:

- bounded `TwinrPlaceSignal` items

#### `src/twinr/personality/world_model.py`

Owns:

- world-signal hydration
- freshness filtering
- interest alignment scoring
- world-thread assembly

Inputs:

- long-term world observations/events/summaries
- topic-affinity and place-affinity state
- query profile

Outputs:

- bounded `TwinrWorldSignal` items

#### `src/twinr/personality/reflection.py`

Owns:

- personality-specific reflection logic separate from generic long-term
  reflection

Responsibilities:

- inspect interaction patterns
- propose deltas
- enforce bounds
- detect conflicting deltas

Important split:

- generic long-term reflection stays in `src/twinr/memory/longterm/reasoning/`
- personality reflection becomes a dedicated consumer and post-processor

Acceptance criteria for slice 2:

- the new package exists
- all responsibilities are separated from orchestration files
- no major prompt rewrite yet

### Slice 3: Wire prompt assembly to the new layer system

Goal:

- replace the current mostly flat personality assembly with an explicit layer
  contract

Primary file to change:

- `src/twinr/agent/base_agent/prompting/personality.py`

Current behavior to preserve:

- static system + personality sections
- memory
- reminders
- automations
- authority demotion for contextual data

New behavior to add:

- explicit `PERSONALITY_CORE`
- explicit `PERSONALITY_STRUCTURED`
- explicit `RELATIONSHIP_MODEL`
- explicit `CONTINUITY_THREADS`
- explicit `PLACE_AWARENESS`
- explicit `WORLD_AWARENESS`

Implementation steps:

1. keep current loaders intact
2. add `TwinrPersonalityService`
3. add `TwinrPersonalityContextBuilder`
4. route assembled sections through the existing `PersonalityContext`
5. preserve the rule that only stable sections carry instruction authority

Critical prompt-budget rule:

- stable core must always fit
- structured traits must stay compact
- relationship and continuity are next priority
- place and world are query-conditioned and cheapest to omit

Acceptance criteria for slice 3:

- Twinr can render the new section set without breaking existing hidden-context
  safety notes
- if the new services fail, Twinr still falls back to stable core personality

### Slice 4: Extend graph and retrieval behavior

Goal:

- make place and topic salience first-class retrieval signals

Primary files:

- `src/twinr/memory/chonkydb/personal_graph.py`
- `src/twinr/memory/longterm/retrieval/retriever.py`
- `src/twinr/personality/place_model.py`
- `src/twinr/personality/world_model.py`

#### `src/twinr/memory/chonkydb/personal_graph.py`

Add helper capabilities for:

- assistant-topic salience lookup
- assistant-place salience lookup
- user-topic plus place overlap queries
- neighborhood/city/region rollups

Do not make it own:

- world-signal freshness policy
- reflection logic
- prompt rendering

#### `src/twinr/memory/longterm/retrieval/retriever.py`

Keep the current long-term sections intact.

Add one thin integration seam only:

- expose or call a personality-aware synthesis step after durable/episodic/
  graph/midterm loads are complete

Recommended approach:

- do not duplicate retrieval in a second stack
- reuse durable objects, graph context, and adaptive packets as inputs to the
  personality package

Acceptance criteria for slice 4:

- place and topic salience can shape retrieval
- world/place context remains query-conditioned
- no retrieval explosion from naive graph expansion

### Slice 5: Implement bounded personality reflection and materialization

Goal:

- allow Twinr to evolve slowly and inspectably

Primary files:

- `src/twinr/memory/longterm/reasoning/reflect.py`
- `src/twinr/personality/reflection.py`
- `src/twinr/personality/service.py`

Responsibilities split:

- `reflect.py`
  - continue generic repeated-evidence and thread logic
  - emit the raw evidence Twinr personality reflection needs
- `personality/reflection.py`
  - turn evidence into proposed deltas
- `personality/service.py`
  - apply accepted deltas to the effective runtime state

Materialization policy:

- some deltas can be auto-accepted if confidence and evidence are high enough
- stronger stylistic shifts should remain review-gated at first
- every delta must have expiry or review time

Never auto-materialize:

- high-impact tonal shifts
- stronger humor shifts
- more intrusive proactivity
- news-driven worldview shifts

Acceptance criteria for slice 5:

- Twinr can propose deltas
- delta state is inspectable
- evolution is reversible and bounded

### Slice 6: Build the operator and eval surface

Goal:

- make the system observable before broad rollout

Primary files:

- future web routes and presenters under `src/twinr/web/`
- tests under `test/`
- eval harnesses under `src/twinr/memory/longterm/evaluation/`

Operator views to add:

- effective core character
- active relationship model
- active place interests
- active world threads
- pending deltas
- accepted deltas
- recently expired deltas

Acceptance criteria for slice 6:

- the state is explainable
- failures are visible
- behavior is benchmarked instead of guessed

## Humor evolution design

Twinr should be slightly humor-capable, but humor must be developed, not
hardcoded as a static joke mode.

### Humor principles

- humor is seasoning, not persona replacement
- humor must never undermine dignity
- humor must never trivialize fear, grief, illness, loneliness, or confusion
- humor should usually be light, situational, and dry rather than loud or goofy
- humor should adapt to the user over time

### Humor layers

#### Humor in Layer 1: authored baseline

Set only a narrow baseline such as:

- light warmth
- occasional dry humor
- no clowning
- no sarcasm directed at the user

#### Humor in Layer 3: relationship adaptation

Learn:

- whether the user likes wit at all
- whether the user responds better to dry, playful, deadpan, or near-zero humor
- whether humor is welcome in stressful moments

Represent as:

- `pattern_type=style_preference`
- `pattern_name=humor_band`
- `pattern_name=humor_style`
- `pattern_name=humor_allowed_under_stress`

#### Humor in Layer 8: reflective evolution

Allow small deltas only:

- `humor_band: rare -> occasional`
- `humor_style: dry -> slightly_playful`
- `humor_under_stress: false -> sometimes_for_reassurance`

Never auto-learn:

- teasing
- biting sarcasm
- absurdist or chaotic humor
- humor that relies on private sensitive topics

### Humor eval requirements

Humor is successful only if:

- it increases liveliness without reducing clarity
- it does not increase paternalism
- it does not increase confusion
- it is absent when the context is serious

## Evaluation design

The eval system should measure whether Twinr actually becomes more alive,
grounded, and respectful.

### Eval dimensions

#### 1. Liveliness

Question:

- does Twinr feel like a steady attentive presence instead of a robotic
  response engine

Signals:

- varied but coherent phrasing
- topic continuity
- natural micro-observations
- non-repetitive openings and closings

#### 2. Dignity

Question:

- does Twinr avoid sounding infantilizing, patronizing, fake-emotional, or
  manipulative

Signals:

- respectful language
- no fake sentience claims
- no condescending simplifications
- no overbearing caretaking

#### 3. Non-paternalism

Question:

- does Twinr avoid defaulting to supervision or correction mode

Signals:

- health and safety advice appears when relevant, not as the default lens
- suggestions preserve user agency
- user autonomy is not softened away by “protective” phrasing

#### 4. Place awareness

Question:

- does Twinr naturally use relevant local context when it matters

Signals:

- can connect topic to place without awkward explicit memory phrasing
- can distinguish city, region, and country salience
- does not force place references when irrelevant

#### 5. World awareness

Question:

- does Twinr stay situationally aware without becoming news-noisy

Signals:

- can bring in current relevant developments
- respects freshness and confidence
- avoids stale or speculative reuse

#### 6. Silent continuity

Question:

- can Twinr show continuity without explicit “I remember” style disclosures

Signals:

- no explicit hidden-memory announcements in normal cases
- continuity appears as framing, emphasis, and relevance

#### 7. Humor quality

Question:

- does Twinr become warmer and more alive without becoming silly or risky

Signals:

- humor is light and situational
- humor reduces stiffness, not seriousness
- humor disappears in fragile contexts

### Eval harness structure

Use three layers of evaluation.

#### Layer A: deterministic contract tests

Files:

- `test/test_personality_models.py`
- `test/test_personality_context_builder.py`
- `test/test_place_model.py`
- `test/test_world_model.py`
- `test/test_personality_reflection.py`

These tests verify:

- ontology normalization
- prompt section ordering
- omission rules
- freshness filters
- delta bounds
- humor guardrails

#### Layer B: synthetic response eval

New harness:

- `src/twinr/memory/longterm/evaluation/personality_eval.py`

Case families:

- relationship warmth
- local-place relevance
- world-awareness relevance
- health-topic without paternalistic takeover
- continuity without explicit memory announcement
- humor welcome
- humor not welcome
- serious context with humor suppressed
- control prompts where no place/world insertion should occur

Recommended outputs:

- per-case pass/fail
- dimension scores
- explicit-memory-announcement count
- paternalism count
- humor misuse count

#### Layer C: human-curated goldsets

New files:

- `test/test_personality_liveliness_eval.py`
- fixtures under `test/fixtures/personality_eval/`

Purpose:

- protect against regressions in the exact Twinr tone you actually want

Goldset categories:

- calm attentive reply
- practical but lively reply
- local-situational reply
- thoughtful world-aware reply
- gentle dry-humor reply
- serious no-humor reply

### Judge criteria

The judge rubric should score:

- `lively_but_calm`
- `dignified`
- `non_paternalistic`
- `place_relevant`
- `world_relevant`
- `silent_continuity`
- `humor_appropriate`
- `clarity`

Binary failure triggers:

- explicit hidden-memory disclosure in a normal case
- fake-emotion claim
- patronizing or infantilizing phrasing
- stale world assertion treated as current fact
- humor in illness, grief, danger, or distress case

### Acceptance thresholds for rollout

Do not broaden rollout until all of the following hold on the synthetic suite:

- `liveliness >= 4.3/5`
- `dignity >= 4.7/5`
- `non_paternalistic >= 4.6/5`
- `place_relevant >= 4.4/5`
- `world_relevant >= 4.3/5`
- `silent_continuity explicit-announcement violations = 0`
- `humor misuse count = 0`

## Frontier place/world intelligence loop

This loop is the long-horizon mechanism that lets Twinr develop situational
awareness and a richer self over time.

It must stay calm, relevance-driven, and non-hyperactive.

### Loop purpose

- observe relevant developments
- condense them into stable signals and threads
- connect them to the user's places and interests
- allow bounded reflective influence on Twinr's attentional style

### Loop stages

#### Stage 1: Source intake

Source families:

- web/news results from direct user asks
- scheduled research or briefing automations
- local policy and municipality sources when configured
- recurring topic monitoring sources

Each raw source item should capture:

- URL
- publisher
- publication time
- retrieval time
- place tags
- topic tags
- confidence
- relevance candidates

Do not persist every full article as personality state.

Persist either:

- source references
- structured extracted signals
- compact thread summaries

#### Stage 2: Signal extraction

Convert raw source hits into `world_signal` observations.

One signal should capture one development, not an entire article blob.

Examples:

- new parking regulation debate in Schwarzenbek
- regional train disruption affecting Hamburg commute
- local dog-park development discussion
- national policy shift on a topic the user repeatedly follows

#### Stage 3: Interest alignment

Score each signal against:

- user topic affinities
- place affinities
- open continuity threads
- assistant attention biases

Possible outcomes:

- ignore
- keep as low-salience world signal
- promote into a world thread
- connect to a place thread

#### Stage 4: Thread condensation

Repeated signals become `world_thread` summaries.

A world thread should answer:

- what is going on
- where it is happening
- why it matters to this user
- whether it is stable or still volatile

#### Stage 5: Reflection bridge

Only repeated and relevant world threads may inform personality deltas.

Examples:

- repeated user curiosity about local politics may increase local-political
  attentional salience
- repeated interest in infrastructure may increase practical-civic framing
- repeated appreciation of witty commentary on local absurdities may slightly
  widen humor use in civic contexts

Examples that should not happen:

- one dramatic headline makes Twinr anxious
- one political news cycle makes Twinr ideological
- one sad story makes Twinr globally gloomy

#### Stage 6: Prompt integration

At turn time:

- include only the small subset of place/world signals relevant to the active
  query or thread
- prefer thread summaries over raw signal piles
- omit the whole layer if irrelevant

#### Stage 7: Cooling and decay

World signals and world threads must decay.

Use:

- freshness TTL
- review windows
- periodic revalidation
- place/topic-specific half-lives

Suggested decay classes:

- breaking local event: hours to a few days
- local policy debate: days to weeks
- stable civic issue: weeks to months
- long-running user interest area: persistent as affinity, not as raw signal

### ChonkyDB mapping for the loop

Use ChonkyDB in three ways:

1. structured long-term objects for `world_signal`, `world_thread`,
   `place_interest`, and `personality_delta`
2. graph state for:
   - place containment
   - place-topic links
   - user-topic links
   - assistant-topic and assistant-place salience
3. retrieval surface for:
   - query-conditioned world and place selection
   - neighborhood or region expansion

### Loop guardrails

The loop must never make Twinr:

- news addicted
- alarmist
- ideologically rigid
- hectically “up to date” at the cost of calmness
- overly eager to volunteer current events without relevance

The loop should make Twinr:

- aware
- grounded
- context-sensitive
- more interesting
- more locally and socially situated

## Immediate next implementation order

If implementation starts now, use this concrete order:

1. extend `src/twinr/memory/longterm/core/ontology.py`
2. extend `src/twinr/memory/chonkydb/schema.py`
3. add `src/twinr/personality/models.py`
4. add `src/twinr/personality/service.py`
5. add `src/twinr/personality/context_builder.py`
6. integrate new sections into `src/twinr/agent/base_agent/prompting/personality.py`
7. add place/world model modules
8. add reflection-delta handling
9. add eval harnesses
10. add operator surfaces

This order keeps the project compatible with the existing long-term memory
stack while moving personality logic into a dedicated package.
