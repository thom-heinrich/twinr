# Twinr Long-Term Memory Architecture

## Goal

This document defines the target long-term memory system for Twinr.

The target is not "a bigger note store".
The target is a frontier-grade memory system for a voice-first physical assistant that feels:

- human
- proactive
- exact
- calm
- trustworthy

For Twinr, "SOTA memory" means all of the following at once:

- the assistant remembers the right things over long time spans
- the assistant uses memory naturally as conversational subtext
- the assistant resolves conflicts instead of silently drifting
- the assistant knows when not to use a memory
- the assistant surfaces helpful things proactively without becoming annoying or creepy
- the system stays auditable, bounded, and safe on Raspberry Pi hardware

## Product-level requirements

Twinr memory must support:

1. **Human continuity**
   Replies should feel like they come from an assistant that truly knows the person, their routines, their relationships, and their current situation.

2. **Exactness**
   Memory must distinguish:
   - observed
   - said by user
   - inferred
   - confirmed
   - uncertain
   - outdated

3. **Proactivity**
   Memory should not only answer questions.
   It should help Twinr decide what is worth surfacing now.

4. **Subtlety**
   Memory should often influence phrasing and suggestions without explicit "I remember..." language.

5. **Conflict handling**
   If multiple memories compete, Twinr must ask and then consolidate.

6. **Multimodal grounding**
   Long-term memory should eventually use voice, PIR, camera, button patterns, calendar, email, reminders, and portal actions as evidence sources.

## Core principle

Twinr memory is a **memory operating system**, not a single database.

The database is only one layer.
The actual system needs:

- ingestion
- extraction
- canonicalization
- truth maintenance
- consolidation
- retrieval
- proactive planning
- forgetting
- evaluation

## Canonical language contract

Internal memory is canonical English.
User-facing replies stay in the user's language.

This means:

- extracted memory objects are written in English
- entity labels may preserve names exactly
- quoted text may preserve the original wording when required
- retrieval may rewrite the user query into canonical English
- the final spoken reply is still localized to the user language

This separation keeps memory stable and comparable across languages without forcing the user to speak English.

## Memory model

Twinr should store several different memory forms, not just one.

### 0. Working memory

The active turn window and immediate conversation state.

This is not long-term memory.
It is the short-horizon layer used for the current exchange.

### 1. Raw episode evidence

A bounded record of what happened.

Examples:

- a user utterance
- a generated answer
- a PIR event
- a camera snapshot interpretation
- a print action
- an email summary
- a calendar import

Raw evidence is not yet durable truth.

### 2. Episodic memory

A compact narrative unit tied to a situation in time.

Examples:

- "The user said today is warm and that Janina is at the eye doctor."
- "The user asked for Corinna's phone number and clarified that she is the physiotherapist."

Episodes are useful for continuity and recall, but they should not automatically become permanent facts.

### 2.5. Mid-term continuity packets

A bounded near-term layer between rolling dialogue and durable semantic memory.

Examples:

- "Janina has an eye doctor appointment today."
- "Recent conversation suggests the user may still be deciding about a walk."
- "Current practical context around Corinna is likely to be a phone-number clarification."

These packets should be:

- compact
- query-relevant
- grounded in source memories
- short-lived compared to durable semantic facts

### 3. Semantic facts

Stable or semi-stable facts derived from repeated evidence or explicit statements.

Examples:

- "Janina is the user's wife."
- "The user prefers brand A for coffee."
- "Corinna Maier is the user's physiotherapist."

### 4. Graph memory

Entities and typed relationships.

Examples:

- `user:main --social_related_to_user--> person:janina` with `relation=spouse`
- `person:corinna_maier --social_related_to_user--> user:main` with `role=physiotherapist`
- `user:main --user_prefers--> brand:melitta` with `category=brand` and `for_product=coffee`
- `event:janina_eye_laser_2026_03_14 --temporal_occurs_on--> day:2026-03-14`

### 5. Temporal state

Every memory object that can change should carry validity.

Examples:

- active
- uncertain
- superseded
- invalid
- valid_from
- valid_to

### 6. Subtext cues

Small response-shaping hints.

Examples:

- prefer practical phrasing
- continue a current life thread gently
- bias suggestions toward known preferences
- avoid explicit recall unless the user is actually asking for it

Subtext cues are not truth objects.
They are a retrieval product built from truth objects plus current context.

### 7. Proactive candidates

Potentially helpful surfacing opportunities.

Examples:

- remind about today's appointment
- suggest a walk because the user mentioned it yesterday and weather is now good
- surface that a familiar contact may be relevant

These are proposals, not guaranteed actions.

## Extraction: one turn can become many memories

A single turn must be decomposed into multiple candidate memory objects.

Example user utterance:

> "Today is a beautiful Sunday, it is really warm. My wife Janina is at the eye doctor and is getting eye laser treatment."

This should not become one giant blob.
It should become several candidate objects.

### Candidate episode

- Episode summary:
  - "On 2026-03-14, the user said the day is a warm Sunday and that Janina is at the eye doctor for eye laser treatment."

### Candidate semantic facts

- "Janina is the user's wife."
- "Janina has an eye-related appointment on 2026-03-14."
- "Janina is receiving eye laser treatment on 2026-03-14."

### Candidate temporal facts

- "The day referenced is 2026-03-14."
- "The event is happening today."

### Candidate environment facts

- "The user described the day as warm."
- "The user described the day as Sunday."

### Candidate graph updates

- `user:main --social_related_to_user--> person:janina` with `relation=spouse`
- `event:janina_eye_laser_2026_03_14 --general_related_to--> person:janina`
- `event:janina_eye_laser_2026_03_14 --temporal_occurs_on--> day:2026-03-14`

### Important exactness rule

Twinr should **not** silently upgrade weak inference into hard truth.

For example:

- "Janina has vision problems" is **not** strictly the same as "Janina is getting eye laser treatment."
- "Warm Sunday" may be useful as episode context, but usually should not become durable long-term semantic truth.

So extraction must produce:

- explicit facts
- bounded inferences
- salience scores
- confidence scores

Then later stages decide what gets promoted.

## Pipeline

The target architecture should split memory work into multiple stages.

### Stage A: online hot path

Runs during a live turn.

Responsibilities:

- build retrieval query profile
- retrieve relevant memory
- build:
  - silent personalization context
  - explicit recall context
  - safety/conflict context
- inject a small, bounded context package into the model

Hard rule:

- the hot path should not block on slow memory writes or deep consolidation work

### Stage B: nearline extraction worker

Runs right after the turn in the background.

Responsibilities:

- turn transcript + response into candidate memory objects
- extract entities
- extract relationships
- extract candidate plans
- extract candidate preferences
- extract candidate temporal facts
- extract conflicts
- write bounded raw artifacts

This is the first place where one turn becomes many candidate memories.

### Stage C: consolidation / reflection loop

Runs periodically.

Responsibilities:

- merge duplicate entities
- promote repeated evidence into stable semantic memory
- demote stale or contradicted facts
- create summaries of ongoing life threads
- create graph edges from repeated evidence
- mark open conflicts requiring clarification

This is the difference between "a transcript archive" and "real memory".

### Stage D: proactive planner

Runs on a schedule and on important events.

Responsibilities:

- identify action-worthy situations
- score usefulness vs annoyance
- score urgency vs sensitivity
- decide whether to:
  - say nothing
  - wait
  - ask gently
  - surface now

## Truth maintenance

This is mandatory for exactness.

Every fact-like memory object should carry:

- `memory_id`
- `schema_version`
- `kind`
- `status`
- `confidence`
- `source`
- `source_type`
- `source_event_ids`
- `created_at`
- `updated_at`
- `valid_from`
- `valid_to`
- `confirmed_by_user`
- `supersedes`
- `conflicts_with`
- `sensitivity`

### Status model

- `candidate`
  - extracted but not yet consolidated
- `active`
  - current best truth
- `uncertain`
  - plausible but unresolved
- `superseded`
  - replaced by stronger evidence
- `invalid`
  - known false
- `expired`
  - no longer current due to time

### Conflict rule

Twinr must not silently overwrite competing facts.

Example:

- Corinna has two phone numbers
- two Corinnas exist
- the role changed
- a doctor appointment was moved

The system should:

1. detect slot-level conflict
2. ask for clarification at a natural time
3. keep provenance
4. consolidate once clarified

## Graph system

The graph should remain versioned and additive.

Current namespace idea should stay:

- `social_*`
- `general_*`
- `temporal_*`
- `spatial_*`
- `user_*`

This is the right backbone.

The graph should become the main structure for:

- people
- relationships
- contact methods
- preferences
- plans
- locations
- stores
- routines
- events

### Graph design rules

- small stable edge vocabulary
- additive schema evolution
- typed nodes
- validity on edges
- provenance on edges
- no silent reinterpretation of old edges

## Retrieval architecture

Twinr needs three retrieval modes.

### 1. Silent personalization retrieval

Purpose:

- shape tone, suggestions, prioritization, and continuity

Should retrieve:

- preferences
- routines
- current threads
- social context

Should not:

- dump contact details
- overuse explicit memory language
- surface stale private facts without need

### 2. Explicit recall retrieval

Purpose:

- answer direct memory questions
- support exact lookup
- support conflict clarification

Should retrieve:

- canonical facts
- graph edges
- temporal validity
- source confidence

### 3. Proactive retrieval

Purpose:

- decide what may help now

Should retrieve:

- upcoming events
- open threads
- unfinished tasks
- preference-informed suggestions
- risk or support signals

## Retrieval ranking

Frontier quality requires more than recency plus token overlap.

Target ranking should combine:

- semantic relevance
- graph distance
- temporal fit
- recency
- salience
- user confirmation strength
- contradiction penalty
- sensitivity gate
- current modality fit

The result should be a bounded memory packet, not a giant dump.

## Subtext generation

The subtext layer should stay distinct from explicit memory.

Subtext generation should answer:

- what background familiarity is useful here?
- what preference should softly influence suggestions?
- what ongoing thread is relevant?
- what should remain implicit?

Subtext should often result in:

- choice of suggestion
- phrasing
- order of options
- level of reassurance

It should usually not result in:

- overt "I remember you said..."
- creepy unsolicited private details
- false certainty

## Proactive intelligence

Twinr should become proactive through memory, not just timers.

The planner should reason over:

- time
- situation
- presence
- recent conversation
- current user state
- known preferences
- open commitments

Examples:

- If the user planned to walk today and the weather is now good, Twinr may gently suggest it.
- If there is a same-day appointment and the user is present, Twinr may surface a reminder.
- If the user often asks for a certain shop or product, Twinr may bias recommendations accordingly.

### Anti-annoyance policy

A proactive candidate should be suppressed if:

- it is weakly grounded
- it is too repetitive
- it is highly sensitive without prompt
- the user recently ignored similar prompts
- current context suggests interruption would be bad

## Multimodal memory

A frontier Twinr memory system should eventually support multimodal evidence.

Examples:

- PIR presence patterns
- button usage patterns
- camera-observed contextual cues
- printed summaries or reminders
- voice identity confidence
- calendar and email imports

This should allow memories like:

- "The user is usually active in the kitchen in the morning."
- "The user tends to print appointment details after hearing them."
- "Janina often visits on Tuesdays."

But the system must keep modality-derived memories lower confidence unless confirmed.

## Forgetting and retention

SOTA memory is not maximum retention.

Twinr should explicitly decide:

- what to keep forever
- what to summarize
- what to decay
- what to archive
- what to forget

### Likely retention classes

- durable identity facts
- durable relationships
- preferences with confidence decay
- plans with expiry
- appointments with expiry
- low-level weather/small-talk context with short retention
- raw episodes with aggressive summarization

## Safety and privacy

Human-like memory must not become creepy memory.

Twinr should:

- avoid over-personalization
- not surface sensitive knowledge casually
- keep domain-sensitive inference conservative
- separate explicit recall from silent personalization
- support deletion and correction
- preserve provenance

Especially important:

- health-related statements should remain narrowly grounded
- contact details should not be surfaced unless relevant
- voice profile data should stay separate from ordinary memory

## Evaluation

A true SOTA target needs a much stronger eval stack than simple recall tests.

Twinr should maintain at least these eval families:

### 1. Long-horizon recall

- exact fact lookup
- multihop graph recall
- temporal recall
- recall under distractors

### 2. Human continuity

- personalized but not explicit
- continuity over days
- natural follow-up quality

### 3. Conflict handling

- detect contradiction
- ask clean clarification questions
- consolidate correctly

### 4. Proactive quality

- helpfulness
- timing
- non-annoyance
- suppression quality

### 5. Over-personalization

- avoid creepy responses
- avoid unnecessary sensitive detail
- avoid invented intimacy

### 6. Multilingual robustness

- internal English memory
- localized output
- no quality collapse for German or mixed-language turns

## Package layout

The implementation now lives in focused subpackages under `src/twinr/memory/longterm/`.
The package root now exposes the public package API only; implementation modules live in
the subpackages below.

- `core/`
  - shared schemas, envelopes, and ontology helpers
- `ingestion/`
  - intake for turn events, propositions, sensors, and multimodal evidence
- `reasoning/`
  - truth maintenance, consolidation, conflicts, reflection, and retention logic
- `retrieval/`
  - retrieval context and subtext/personalization builders
- `proactive/`
  - proactive planning plus persisted proactive state
- `storage/`
  - object storage, midterm storage, and remote snapshot boundaries
- `runtime/`
  - top-level long-term service orchestration and bounded background workers
- `evaluation/`
  - synthetic, multimodal, and subtext evaluation harnesses

## Canonical data envelope

Every durable long-term object should be serializable in a versioned envelope.

Example:

```json
{
  "schema": "twinr_memory_object",
  "version": 1,
  "memory_id": "event:janina_eye_laser_2026_03_14",
  "kind": "event",
  "status": "active",
  "summary": "Janina has eye laser treatment on 2026-03-14.",
  "details": "Derived from a user utterance on 2026-03-14.",
  "canonical_language": "en",
  "source": {
    "type": "conversation_turn",
    "event_ids": ["turn:2026-03-14T10:12:44Z"],
    "speaker": "user"
  },
  "confidence": 0.93,
  "confirmed_by_user": false,
  "valid_from": "2026-03-14",
  "valid_to": "2026-03-14",
  "sensitivity": "private",
  "attributes": {
    "person": "person:janina",
    "memory_domain": "appointment",
    "event_domain": "appointment",
    "action": "eye laser treatment",
    "place": "eye doctor"
  },
  "conflicts_with": [],
  "supersedes": []
}
```

## Memory promotion policy

Not every extracted object should become durable long-term memory.

Promotion should depend on:

- salience
- repeat mention
- explicit user confirmation
- future usefulness
- identity relevance
- relationship relevance
- planning relevance
- emotional relevance

Example:

- "Janina is my wife" should promote strongly
- "today is warm" should usually remain ephemeral
- "Janina is getting eye laser treatment today" should be a time-bounded event memory
- "Janina has chronic vision problems" should not be promoted unless explicitly supported

## Final target state

Twinr reaches frontier-grade memory behavior only when it can do all of this together:

- decompose one turn into multiple precise candidate memories
- preserve provenance and confidence
- consolidate repeated evidence into stable knowledge
- keep a typed temporal graph of the user's world
- use memory silently as subtext
- ask when uncertain
- proactively surface the right thing at the right time
- forget what should fade
- stay auditable and non-creepy

That is the target system.
