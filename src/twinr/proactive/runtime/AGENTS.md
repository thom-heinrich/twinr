# AGENTS.md — /src/twinr/proactive/runtime

## Scope

This directory owns proactive runtime orchestration: presence-session control,
monitor tick coordination, voice-activation arming, degraded-mode sensor handling, and
bounded monitor lifecycle. Structural metadata lives in
[component.yaml](./component.yaml).

Out of scope:
- social trigger scoring, review prompt content, or voice-activation matching internals
- raw hardware adapters and low-level audio/camera/PIR implementations
- proactive delivery cooldown policy in `../governance/`
- top-level agent runtime loops above the monitor boundary

## Key files

- `ambiguous_room_guard.py` — fail-closed room-ambiguity guard for person-targeted runtime inference
- `identity_fusion.py` — bounded temporal/session identity fusion over voice, portrait, household-voice candidates, and visual-anchor history
- `portrait_match.py` — conservative runtime claim for local portrait-match observations and temporal identity evidence
- `known_user_hint.py` — conservative known-user hint from voice-profile state plus optional temporal identity-fusion evidence and clear room context
- `affect_proxy.py` — prompt-only affect proxy surface from coarse posture, attention, and quiet cues
- `smart_home_context.py` — bounded layered runtime context tracker for `near_device_presence`, optional same-room `room_context`, and house-wide `home_context`
- `person_state.py` — bounded aggregate schema that projects the existing runtime surfaces into stable axes such as presence, attention, conversation, safety, identity, and home context
- `continuous_attention.py` — bounded short-lived visible-person track matching, motion recency, audio-to-vision speaker targeting, and short-horizon target prediction for HDMI follow
- `attention_targeting.py` — bounded multimodal attention-target prioritization over camera anchor, speaker association, showing intent, and short-lived session focus memory
- `perception_orchestrator.py` — internal single-source runtime perception orchestrator that turns one display perception tick into one shared attention/gesture snapshot for HDMI, servo, and wake consumers
- `attention_debug_stream.py` — bounded continuous JSONL debug stream for HDMI attention refresh ticks, including outcome codes, stage timings, current camera/target/cue state, and fast-path source/state plus gate reasons for `LOOKING`/`HAND_NEAR`/`INTENT_LIKELY`
- `display_attention.py` — conservative proactive producer that steers HDMI eye gaze toward the currently relevant visible person without overwriting foreign face cues, while keeping normal follow behavior horizontal-only, mirroring camera-space left/right into user-facing gaze, adding small near-center head turns before full side gaze commits, and renewing or briefly holding matching cues before they expire or disappear on short camera dropouts
- `display_attention_camera_fusion.py` — display-only fusion helper that warms the fast HDMI attention lane with recent richer pose/gesture semantics and short dropout holds, while leaving adapter-stream attention truth authoritative
- `display_debug_signals.py` — bounded publisher that mirrors current camera facts plus brief event/trigger holds into optional HDMI header debug pills such as `LOOKING_PROXY` / `LOOKING_CONFIRMED`, source-aware `ENGAGED` / `ENGAGED_PROXY` where `ENGAGED` is reserved for confirmed looking, or held `POSITIVE_CONTACT` for Pi-side inspection while suppressing redundant unchanged rewrites until the current snapshot actually nears expiry
- `display_gesture_emoji.py` — conservative producer that mirrors stabilized rising-edge user gestures into bounded HDMI emoji acknowledgements without touching the face channel or overwriting foreign emoji cues
- `gesture_stream_guards.py` — shared guard that keeps user-facing HDMI/wakeup side effects tied to current hand evidence and blocks stale/slow rescue provenance from surfacing as live user intent
- `display_reserve_candidates.py` — compatibility wrapper for the modular ambient companion flow; preserve the historic public loader while delegating candidate sourcing elsewhere
- `display_reserve_flow.py` — shared ambient companion orchestration that blends personality, memory, reflection, latent snapshot-topic backfill, and long-horizon reserve-lane learning into one bounded semantic-topic set before deterministic card-surface expansion
- `display_reserve_diversity.py` — explicit diversity-policy layer that normalizes raw reserve candidates into broader conversational seed families and axes before raw-pool truncation or day-plan spacing
- `display_reserve_expansion.py` — deterministic topic bundling and multi-angle card-surface expansion that keeps concrete card ids separate from grouped semantic topic keys
- `display_reserve_learning.py` — long-horizon reserve-lane learning profile derived from shown-card outcomes so future plans can reward welcome openers and cool repetitive ones generically
- `display_reserve_memory.py` — focused memory-derived reserve-candidate conversion for clarification conflicts and gentle follow-ups, including semantic `card_intent` blocks for calm clarification/update wording
- `display_reserve_snapshot_topics.py` — latent snapshot-topic reserve loader that backfills additional continuity, relationship, place, and concrete world-signal seeds when the stronger explicit loaders stay sparse, while staying generic and display-safe
- `display_reserve_reflection.py` — reflection-derived reserve-candidate loading from summaries and midterm packets without coupling planning to reflection internals, including deterministic display-safe fallback copy plus semantic `card_intent` blocks so raw internal packet text, transcript-only continuity residue, farewell/meta residue, and device-watch packets never reach the HDMI right lane
- `display_reserve_world.py` — direct world-intelligence reserve-candidate loading from active subscriptions and condensed awareness threads so the right lane keeps topic breadth even when reflection or follow-up hooks are thin, while staying topic-led instead of source-label-led and emitting semantic `card_intent` blocks for public-topic observation cards
- `display_reserve_copy_contract.py` — shared writer/judge copy assets for reserve-card generation, including normalized copy families, family-scoped gold examples, and the shared quality rubric
- `display_reserve_prompting.py` — compact prompt-payload builder that distills reserve candidates into explicit topical anchors, hook hints, semantic `card_intent`, `expansion_angle`, normalized `copy_family`, family-scoped gold examples, and the shared quality rubric before they reach the LLM rewrite step
- `display_reserve_generation.py` — bounded LLM writer/judge layer for reserve-card text so the right-hand lane reflects Twinr's current personality without moving generation into the hot monitor tick, using small isolated batches instead of one large all-or-nothing request and exposing bounded trace metadata for eval harnesses
- `display_reserve_day_plan.py` — persistent current-day planner that turns expanded reserve card surfaces into one calm daily rotation for the HDMI reserve area, mixing normalized seed families and coarse public/personal/setup axes so the lane does not clump around one conversation mode, spacing sibling cards by semantic topic, and keeping topics in same-day rotation until real user feedback retires them or the nightly planner replaces the plan
- `display_reserve_companion_planner.py` — explicit overnight reserve-lane recalibration that runs nightly reflection/world refresh, reviews recent shown-card outcomes, and stores the prepared next-day plan for morning adoption
- `display_reserve_runtime.py` — shared reserve-lane publish path for all right-hand text cards so planned and spontaneous prompts reuse one cue/history contract with both concrete card ids and grouped semantic topic keys
- `display_ambient_impulses.py` — conservative live publisher that shows the next planned positive personality-driven HDMI reserve impulse only when quiet hours, presence, runtime state, and surface ownership all allow it, then hands the concrete publish to the shared reserve runtime path
- `display_social_reserve.py` — dedicated publisher that routes display-first proactive social prompts into the same right-hand reserve lane and same shared reserve publish path instead of the fullscreen presentation surface
- `safety_trigger_fusion.py` — runtime-only bridge that prefers short-window fused safety claims for `possible_fall`, `floor_stillness`, and `distress_possible` while keeping the legacy engine as fallback and source of cooldown truth
- `gesture_ack_lane.py` — internal low-latency acknowledgement policy used by `perception_orchestrator.py` for the explicit user-facing gesture symbol set
- `gesture_wakeup_dispatcher.py` — single-flight background dispatcher that runs accepted visual wakeups off the proactive monitor worker
- `gesture_wakeup_lane.py` — internal visual-wakeup policy used by `perception_orchestrator.py` for one configured fine-hand symbol
- `gesture_wakeup_priority.py` — interaction-priority guard that lets button/voice outrank visual wakeups before a gesture can steal shared audio capture
- `gesture_debug_stream.py` — bounded continuous JSONL debug stream for HDMI gesture refresh ticks, including raw observations, ack decisions, publish outcomes, and stage timings
- `service.py` — compatibility shim that preserves the historic import/helper surface while delegating to `service_impl/`
- `service_impl/coordinator_core.py` — dependency wiring, trigger review, and main proactive tick orchestration extracted from the legacy service monolith
- `service_impl/coordinator_display.py` — dedicated HDMI attention and gesture refresh workflows extracted from the legacy service monolith
- `service_impl/coordinator_observation.py` — stable observation-mixin facade that composes the focused observation helper modules behind the historic path
- `service_impl/coordinator_observation_display.py` — automation-dispatch, HDMI display, gesture, and camera-surface helper bridges extracted from the legacy observation mixin
- `service_impl/coordinator_observation_facts.py` — automation fact payload assembly and rising-edge sensor-event derivation extracted from the legacy observation mixin
- `service_impl/monitor.py` — bounded background worker lifecycle wrapper around the coordinator
- `service_impl/builder.py` — default monitor assembly path that keeps wrapper-level monkeypatch points stable
- `audio_perception.py` — runtime-faithful one-shot ReSpeaker perception diagnostics and conservative room/device-directedness guard summaries for operator checks and recovery smokes
- `pir_open_gate.py` — bounded startup gate that retries only exact transient PIR GPIO busy overlaps during runtime handover
- `respeaker_capture_gate.py` — bounded startup gate that requires consecutive readable-frame probes before ReSpeaker capture is considered stable
- `claim_metadata.py` — shared `confidence` / `source` / `requires_confirmation` helpers for runtime claims
- `speaker_association.py` — conservative speaker-to-camera-anchor association for the single-primary-person case
- `multimodal_initiative.py` — confidence-bearing display-first/skip gate for later proactive behavior
- `presence.py` — source of truth for voice-activation presence-session arming
- `service.py` — thin compatibility wrapper only; real runtime logic belongs in `service_impl/` and the focused helper modules
- `__init__.py` — package export surface; treat changes as API-impacting
- `component.yaml` — structured metadata, callers, and tests

## Invariants

- `PresenceSessionController.observe()` must preserve a monotonic internal timeline even when callers pass regressing or malformed timestamps.
- Voice-activation arming must fail closed on malformed sensor flags or unavailable dependencies.
- `service.py` stays orchestration-focused; trigger scoring and voice-activation matching belong in sibling packages.
- `safety_trigger_fusion.py` may prefer fused safety claims over the legacy engine when the short-window evidence is stronger, but it must not bypass engine cooldown state, fall-reset state, vision review, presence-session suppression, or later runtime policy gates.
- `service.py` may fan the same conservative `attention_target` out to HDMI face cues and the optional body-follow servo, but it must not create a parallel servo-specific targeting policy.
- `audio_perception.py` must stay a bounded diagnostic/guard surface that reuses existing normalized observation and policy helpers; it must not grow its own parallel monitor orchestration.
- `service.py` must pause the active streaming voice activation capture before handing an accepted streaming voice activation or visual gesture wakeup to an exclusive conversation-recording path on the same ALSA device, and reopen it after the handler returns; otherwise wakeups degenerate into immediate `Device or resource busy` capture failures.
- `perception_orchestrator.py` is the only place where runtime consumers may combine camera `perception_stream` truth with downstream attention-target, HDMI-ack, or visual-wakeup policy. Do not let `service_attention_helpers.py`, `coordinator_display.py`, or later consumers rebuild a second semantic truth from the same raw frame.
- `service.py` must not open a second proactive PCM `arecord` path or ReSpeaker readable-frame startup probe against the same ALSA device while the voice orchestrator already owns that capture and no shared voice activation buffer exists; in that case the proactive monitor should stay signal-only/fail-closed instead of thrashing ALSA.
- `service.py` must not clear a targeted ReSpeaker startup blocker after a single transient readable-frame probe; startup needs sustained bounded capture success so unstable USB enumeration does not flap the runtime between `ok` and `error`.
- `service.py` must tolerate only short exact `EBUSY` handover races when opening the PIR monitor; permanent GPIO contention or non-busy faults stay fail-closed and must not be downgraded.
- `audio_policy.py` must treat classified non-speech/background-media room activity as incompatible with calm `presence_audio_active`; do not let uncorroborated XVF3800 speech flags alone re-elevate media/noise into device-directed presence speech.
- ReSpeaker ambient classification must not depend solely on the speech-threshold chunk gate for non-speech detection; bounded low-dynamic RMS activity may still be media/noise even when `active_ratio` stays at zero.
- ReSpeaker ambient classification may use local PCM-content evidence to veto strong XVF3800 speech false positives, but ambiguous PCM windows must stay fail-closed and allow upstream host-control to remain decisive.
- ReSpeaker ambient classification must not discard strong PCM non-speech evidence just because the legacy chunk/RMS activity gate returns false on a bounded playback window.
- The operator-facing room-context guard should prefer already-classified non-speech/background-media over weak overlap hints; ambiguous low-confidence XVF3800 overlap alone must not relabel playback noise as human room speech.
- `ambiguous_room_guard`, `identity_fusion`, `portrait_match`, `known_user_hint`, and `affect_proxy` must remain conservative claim surfaces, not direct product decisions.
- `smart_home_context.py` must stay a bounded optional context layer: explicit same-room entity IDs only, stale streams fail closed, and smart-home facts must never manufacture local `person_visible` or voice-activation arming.
- `person_state.py` must stay aggregation-only: reuse the current bounded runtime surfaces, keep axis contracts explicit, and do not smuggle new hidden policy or diagnosis logic into the aggregate layer.
- `known_user_hint` must stay weaker than identity even when temporal portrait evidence is strong, and may only clear calm personalization; sensitive behavior still needs confirmation or stronger gates.
- `identity_fusion` may use presence-session memory, enrolled household-voice candidates, and visual-anchor history, but it must stay confirm-first and fail closed on ambiguity or modality conflict.
- `affect_proxy` must never emit emotion, wellbeing, or diagnosis claims; it may only support prompt-only follow-up cues.
- `continuous_attention.py` may keep only short-lived anonymous visible-person tracks. It may prioritize the active visible speaker when audio direction is stable, may conservatively blend live single-speaker audio direction back into that visible anchor for a more truthful follow target, otherwise the most recently moving visible person, and it must preserve unmatched recent tracks across brief detector dropouts instead of collapsing a real multi-person scene immediately.
- `attention_targeting.py` may prioritize the latest visible speaker, showing-intent person, or short-lived session focus, but it must not let generic showing-intent override speaker or last-motion targets in a real multi-person scene. Ordinary person-following may use bounded vertical eye/head drift from live person height, but should not exaggerate that into theatrical "thinking" poses.
- `attention_targeting.py` and `display_attention.py` must fail closed when camera health is explicitly bad and no visible person exists; stale speaker/session focus may not keep steering the face after camera loss.
- `attention_debug_stream.py` may expose bounded operator forensics for the fast attention lane, but it must stay inspection-only: report raw fallback sources, thresholds, and block reasons instead of introducing new runtime policy or a second truth source for camera facts.
- `display_attention.py` may only publish its own `proactive_attention_follow` face cues and must fail closed when another display producer currently owns the cue store.
- `display_debug_signals.py` must stay inspection-only: publish only bounded header pills from current camera facts and brief event/trigger holds, keep the lane file-backed and optional, and do not widen the generic runtime snapshot schema or invent a second alerting surface.
- `display_gesture_emoji.py` may only acknowledge stabilized rising-edge camera gesture events, must stay bounded and short-lived, and must fail closed when another producer currently owns the emoji surface.
- The dedicated live HDMI gesture acknowledgement path is intentionally limited to `thumbs_up`, `thumbs_down`, and `peace_sign`; do not reintroduce `wave`, `pointing`, `open_palm`, `ok_sign`, or other symbols there without fresh Pi acceptance evidence.
- User-facing HDMI acknowledgements and visual wake triggers for deliberate hand symbols must require a bounded visibility window before they fire; do not regress thumbs or `peace_sign` back to one isolated-frame publish/dispatch behavior without fresh Pi false-positive evidence.
- User-facing HDMI acknowledgements and visual wake triggers must also require current hand evidence from the dedicated fast gesture lane; do not publish or dispatch from stale recent-person, stale recent-hand, or slow full-frame rescue outputs as if they were live user intent.
- `display_reserve_day_plan.py` must keep reserve scheduling persistent and local-day-scoped: plan generation belongs there, not inline in the proactive monitor tick, and repetition control must stay generic rather than topic-hardcoded. If the first same-day plan is empty, it must retry after a short bounded backoff rather than leaving the reserve blank until midnight. A current-day plan must keep unresolved semantic topics in same-day rotation instead of treating one visual exposure as retirement; concrete sibling cards may rotate, but only real user feedback or the next nightly replacement may take a semantic topic bundle out of that day’s loop. Fresh shown-card feedback may trigger a rebuild, but only through bounded generic biasing and answer-based retirement rather than ad-hoc named-topic overrides. The planner may spread source families such as world/place/memory/social/reflection, but that family mixing must stay generic and derived from candidate state rather than named-topic rules.
- `display_reserve_companion_planner.py` owns the explicit overnight reserve-lane review. Nightly reflection/world-refresh triggering, prepared next-day plan storage, recent shown-card outcome summarization, and morning adoption rules belong there rather than in `display_ambient_impulses.py` or the active loop classes.
- `display_reserve_flow.py` owns ambient companion candidate orchestration. Keep memory hooks in `display_reserve_memory.py`, slower reflection sourcing in `display_reserve_reflection.py`, and long-horizon shown-card learning in `display_reserve_learning.py` instead of regrowing one large mixed loader.
- `display_reserve_snapshot_topics.py` owns reserve-side latent-topic backfill from `PersonalitySnapshot`. If additional semantic breadth should come from continuity threads, relationship affinities, place focuses, or concrete snapshot world signals, put that sourcing there instead of regrowing fallback loaders inside `display_reserve_flow.py`.
- `display_reserve_diversity.py` owns raw-pool diversity policy. Normalize implementation-shaped source families such as `world_awareness` or `memory_follow_up` into broader conversational seed families there instead of sprinkling family remaps or ad-hoc spacing penalties across `display_reserve_flow.py` and `display_reserve_day_plan.py`.
- `display_reserve_expansion.py` owns the topic-to-card-surface step. Keep semantic-topic bundling, deterministic angle plans, and concrete card-key generation there instead of mixing those concerns into `display_reserve_flow.py`, `display_reserve_generation.py`, or `display_reserve_day_plan.py`.
- `display_reserve_reflection.py` must never pass internal reasoning scaffolding such as English continuity packet summaries/details straight through to user-facing reserve cards. If richer copy is unavailable, fall back to bounded display-safe prompts derived from structured anchors, not raw packet internals.
- `display_reserve_reflection.py` must suppress closure-only, meta-turn, transcript-only continuity residue, and device-watch packets unless they carry a clear structured user-recognizable thread anchor. Stored memory may keep those packets; the visible right lane must not.
- `display_reserve_flow.py` owns orchestration only. Active RSS subscription topics and condensed awareness threads belong in `display_reserve_world.py`, and the right lane should prefer topic-level world candidates over raw feed/source-label cards.
- `display_reserve_candidates.py` must remain a thin compatibility seam only. New reserve-lane sourcing, ranking, or learning logic belongs in the focused companion-flow modules, not in the wrapper.
- `display_reserve_flow.py` must not truncate the raw ranked pool directly when multiple broad seed families are available. Use `display_reserve_diversity.py` so public-topic clusters do not crowd out discovery, shared-thread, or other personal cards before the day planner even sees them.
- `display_reserve_day_plan.py` must space cards by normalized conversational families and axes, not just raw source tokens. World-awareness and world-subscription cards should still count as the same broad public-topic family for mixing, while a single setup/discovery card may appear but must not take over the whole cycle.
- `display_reserve_copy_contract.py` owns the reusable family examples and shared quality rubric for reserve-card writing. Keep those positive copy assets versioned there instead of regrowing ad-hoc example strings or hidden scoring lists inside `display_reserve_prompting.py` or `display_reserve_generation.py`.
- `display_reserve_prompting.py` owns prompt compaction and anchor shaping for reserve-card rewriting. Keep raw candidate contexts out of the LLM call; the rewrite step should see explicit `topic_anchor` / `hook_hint` / semantic `card_intent` / compact context instead of whole nested reflection or world payloads.
- `display_reserve_prompting.py` must only expose human-facing anchors, hook hints, and semantic `card_intent` blocks to the rewrite step. Do not pass deterministic fallback headlines/bodies or raw internal English scaffolding through the prompt payload, or the model will anchor on that low-quality text.
- `display_reserve_prompting.py` may attach normalized `copy_family`, family-scoped gold examples, and the shared quality rubric to the batch prompt, but those assets must stay small, positive, and family-generic rather than becoming a second hidden named-topic policy layer.
- `display_reserve_memory.py`, `display_reserve_world.py`, and `src/twinr/agent/personality/display_impulses.py` should emit semantic `card_intent` blocks whenever they expect the LLM rewrite step to do more than polish phrasing. Do not rely on phrase-like `display_anchor` / `hook_hint` text alone as the main meaning carrier for public-topic or memory cards.
- `display_reserve_generation.py` must stay bounded and rewrite-only: it may shape tone and wording from Twinr's current personality plus structured reserve context, but it must not reorder topics, improvise new memory facts, or move provider calls into the live publish tick. The visible copy should stay natural rather than label-like, it still needs one clear topical anchor, and malformed or failed batches must fail closed rather than degrading to stale deterministic copy. The rewrite should use a bounded writer/judge structure when better language selection is needed: first generate a small set of distinct variants per topic against the family examples and rubric, then run a second bounded pass that picks the strongest Twinr-style final card without inventing a new topic. Treat empty SDK `output_parsed` as a transport quirk, not an automatic failure, when the model still returned valid JSON text, and retry only the exact `max_output_tokens` truncation case once with a larger budget plus no reasoning before the batch is allowed to fail closed.
- `display_reserve_generation.py` may expose bounded trace metadata for eval harnesses and operator reports, but runtime callers must still depend only on the final rewritten candidates unless they explicitly opt into the trace-returning API.
- `display_reserve_generation.py` must keep Pi-side retry pressure bounded. If operators tighten batch size or variant count through config, the generator should honor that rather than silently using fixed internal constants that make one heavy batch dominate the nightly/day-plan rebuild.
- `display_reserve_generation.py` must keep its local per-batch deadline realistic for the active reserve model on the Pi. The guard should still bound runaway calls, but it may not fail closed just because the normal `gpt-5.4` selection pass occasionally needs more than a mini-model-shaped `12s` budget.
- `display_reserve_generation.py` must respect source-level display goals. `invite_user_discovery` cards may not surface setup labels or UI wording such as `Basisinfos`, `Ansprache`, `Gelerntes`, or `Einrichtung`; they should say the human meaning instead, and the visible headline must be a full natural sentence rather than a clause-like label such as `Wie ich dich ansprechen soll.`. Once a discovery topic already has saved coverage, reserve-side hooks must switch to follow-up wording instead of reusing the stale opener. `call_back_to_earlier_conversation` cards must sound like a calm callback to an earlier thread, not like a fresh diagnosis, fault report, or support ticket.
- Explicit `manage_user_discovery` reactions to visible reserve invites must retire the matching `user_discovery:*` reserve topic directly through reserve feedback/history updates instead of relying only on the generic ambient topic matcher.
- `display_reserve_generation.py` defaults should favor visible-language quality over cheapest possible model choice. If no explicit reserve-generation model is configured, inherit Twinr's main model rather than pinning the lane to a weaker mini-model by default.
- `display_ambient_impulses.py` may only publish planned positive reserve-card nudges while Twinr is waiting, the room appears calmly occupied, quiet hours are closed, and no stronger reserve owner already holds the surface. It may adopt an already prepared next-day plan when morning rollover starts and may restore passive fill after temporary overrides expire, but it must not own overnight review or plan preparation itself.
- `display_ambient_impulses.py` must stay topic-generic and policy-driven: planning may depend on personality state and co-attention, but not on hardcoded named-topic exceptions or one-off display-copy hacks.
- `display_ambient_impulses.py` may persist bounded exposure metadata for later learning, but that path must stay write-only from the monitor loop perspective: no reaction inference, transcript parsing, or ranking decisions belong in the live publisher.
- `display_reserve_runtime.py` owns the actual right-lane text-card publish side effects. New reserve-lane producers must reuse it instead of duplicating cue/history writes or adding another right-lane store.
- `display_social_reserve.py` may reuse the shared reserve runtime publisher for visual-first proactive prompts, but it must stay a thin routing layer: no proactive ranking, no personality planning, and no fullscreen presentation fallback logic belongs there.
- `display_reserve_support.py` owns the shared reserve-lane text, timestamp, and local-time normalization helpers. Reserve planners and publishers must reuse it instead of regrowing tiny duplicate parsing helpers.
- `gesture_wakeup_lane.py` may only derive one bounded wake decision from the configured fine-hand trigger. It must not publish HDMI cues itself, must stay independent from emoji acknowledgement cooldowns, and must never open a conversation turn directly.
- `gesture_wakeup_priority.py` must enforce the interaction precedence `button > voice > gesture`: a visual wakeup may not dispatch while Twinr is already in a non-waiting runtime state or while recent speech evidence says the voice path should win.
- Accepted visual gesture wakeups must not run synchronously on the proactive monitor worker thread. Dispatch them through `gesture_wakeup_dispatcher.py` so HDMI attention refresh keeps running during `listening` and `processing`.
- Matching `display_attention.py` cues must refresh their TTL before expiry so a stable visible-person target does not fall back to center between local attention-refresh ticks, and brief center jitter or short no-person/no-anchor blips must not immediately erase a recent local cue. Near-center responsiveness should come from bounded spatial head/gaze tuning, not from reintroducing jittery up/down or random idle eye motion.
- The fast HDMI attention-refresh path must not block on full ambient PCM sampling windows. When ReSpeaker host-control already provides speech/direction hints, use a signal-only snapshot or recent cached audio instead of serializing gaze/gesture refresh behind slow audio reads.
- The fast HDMI attention-refresh path must keep its default cadence genuinely sub-second on Pi HDMI builds; a half-second-plus default render loop is too slow for gaze-follow or gesture acknowledgement.
- HDMI eye-follow and HDMI gesture acknowledgement must stay decoupled. Eye-follow should prefer a cheap attention-only local camera observation plus the adapter-stream's stable attention truth, while gesture acknowledgement may use the heavier full gesture path without clearing, delaying, or otherwise regressing face-follow behavior.
- HDMI gesture acknowledgement should prefer its own dedicated live-gesture observation path and ack-lane state. Do not route user-facing symbol latency through the heavier social camera-surface path when the dedicated gesture refresh is available.
- When the dedicated gesture refresh is marked stream-authoritative, `gesture_ack_lane.py` and `gesture_wakeup_lane.py` may run only behind `perception_orchestrator.py` and must treat the upstream stable gesture state as the only temporal truth. Keep publish/wakeup edge detection and output cooldown local, but do not add another confirmation or visibility timer on top of the adapter stream.
- When the attention refresh is marked stream-authoritative, the stabilized camera surface may mirror and hold that state for automation/export purposes, but it must not add a second debounce policy for `looking`, `engaged`, `hand_near`, or `showing_intent` on top of the adapter stream.
- `display_attention_camera_fusion.py` must stay display-only. It may reuse recent richer semantics from the full observe and gesture lanes to stabilize pose/hand context and short dropout holds, but it must not overwrite adapter-stream `looking` / `engaged` / `visual_attention` truth, widen the generic automation fact contract, or silently turn the HDMI fast path into another full pose loop.
- `service.py` should keep a bounded changed-only ops trace for HDMI attention-follow decisions so Pi debugging can prove whether failures come from camera health, stabilized camera semantics, target selection, or cue publishing.
- `service.py` should also keep a bounded continuous attention debug stream for each HDMI refresh tick so transient 1-3 minute eye-follow dropouts still leave first-run evidence after the fact.
- `service.py` should also keep a bounded continuous gesture debug stream for each HDMI gesture refresh tick so Pi-side symbol latency and missed acknowledgements can be diagnosed from first-run evidence instead of blind retuning. That stream should include bounded provenance from the dedicated gesture pipeline itself when available, so operators can tell whether a miss came from full-frame live recognition, a recent-person ROI fallback, a recent-hand ROI fallback, or publish/ack policy.
- When hard Pi gesture repros still are not explained by `gesture_debug_stream.py`, `service.py` may bind an opt-in end-to-end workflow run-pack around the gesture refresh, but that path must stay bounded, redacted by default, and debugging-only.
- `service_attention_helpers.py` owns HDMI attention-follow live-context export, changed-only follow traces, servo decision logging, and per-tick attention debug packaging, but attention-target derivation itself now belongs to `perception_orchestrator.py`. `service.py` should orchestrate those helpers rather than regrowing attention logic inline.
- `service_gesture_helpers.py` owns HDMI gesture acknowledgement publish, accepted gesture wakeup dispatch, and gesture debug/trace packaging, but gesture ack/wakeup temporal policy now belongs to `perception_orchestrator.py`. `service.py` should remain the lifecycle/tick coordinator, not the gesture-policy implementation file.
- `service_impl/coordinator_core.py`, `service_impl/coordinator_display.py`, `service_impl/coordinator_observation.py`, `service_impl/coordinator_observation_display.py`, and `service_impl/coordinator_observation_facts.py` are the approved homes for additional proactive-service runtime logic; keep `service.py` as a compatibility shim and keep `service_impl/monitor.py` orchestration-only.
- The fast HDMI attention-refresh path must stay local-camera-only, bounded, and separate from full proactive trigger evaluation; do not turn it into an always-on generic vision loop.
- The fast HDMI attention-refresh path may stay active while the main runtime is in `error`, but only for local face-cue following; it must not be used to re-enable agent turns, remote-memory work, or other degraded runtime behavior.
- Speaker association and multimodal initiative must fail closed on multi-person or low-confidence room context; do not let weak audio hints force spoken proactivity.
- `ProactiveMonitorService` lifecycle must remain idempotent, bounded, and safe under partial startup or shutdown failure.
- `build_default_proactive_monitor()` must not start an inert monitor when no operational sensor path exists.

## Verification

After any edit in this directory, run:

```bash
python3 -m compileall src/twinr/proactive/runtime
PYTHONPATH=src pytest test/test_person_state.py -q
PYTHONPATH=src pytest test/test_perception_orchestrator.py -q
PYTHONPATH=src pytest test/test_proactive_monitor.py -q
PYTHONPATH=src pytest test/test_proactive_runtime_service.py -q
PYTHONPATH=src pytest test/test_display_reserve_support.py test/test_display_reserve_expansion.py test/test_display_reserve_generation.py test/test_display_reserve_prompting.py test/test_display_reserve_learning.py test/test_display_reserve_snapshot_topics.py test/test_display_reserve_reflection.py test/test_display_reserve_flow.py test/test_display_reserve_day_plan.py test/test_display_reserve_companion_planner.py test/test_display_reserve_runtime.py test/test_display_ambient_impulses.py test/test_display_social_reserve.py -q
```

If workflow wiring or export surface changed, also run:

```bash
PYTHONPATH=src pytest test/test_realtime_runner.py test/test_streaming_runner.py -q
```

## Coupling

`presence.py` changes -> also check:
- `service.py`
- `test/test_proactive_monitor.py`

`service.py`, `service_attention_helpers.py`, `service_gesture_helpers.py`, `display_ambient_impulses.py`, `display_social_reserve.py`, `display_reserve_runtime.py`, `display_reserve_day_plan.py`, `display_reserve_companion_planner.py`, `display_reserve_flow.py`, `display_reserve_expansion.py`, `display_reserve_learning.py`, `display_reserve_memory.py`, `display_reserve_snapshot_topics.py`, `display_reserve_reflection.py`, `display_reserve_prompting.py`, `display_reserve_generation.py`, or `display_reserve_support.py` changes -> also check:
- `src/twinr/proactive/social/`
- `src/twinr/agent/workflows/realtime_runner.py`
- `src/twinr/agent/workflows/streaming_runner.py`
- `test/test_proactive_monitor.py`
- `test/test_proactive_runtime_service.py`
- `test/test_display_reserve_flow.py`
- `test/test_display_reserve_snapshot_topics.py`
- `test/test_display_reserve_expansion.py`
- `test/test_display_reserve_learning.py`
- `test/test_display_reserve_prompting.py`
- `test/test_display_reserve_reflection.py`
- `test/test_display_reserve_day_plan.py`
- `test/test_display_reserve_companion_planner.py`
- `test/test_display_reserve_runtime.py`
- `test/test_display_reserve_support.py`

`audio_perception.py` or `audio_policy.py` changes -> also check:
- `src/twinr/__main__.py`
- `hardware/mic/setup_audio.sh`
- `src/twinr/proactive/social/observers.py`
- `test/test_audio_perception.py`
- `test/test_main.py`

`__init__.py` changes -> also check:
- `src/twinr/proactive/__init__.py`
- monitor construction call sites in runtime workflows

## Security

- Keep voice activation capture writes under validated artifacts-root subpaths; do not widen path or symlink handling.
- Do not bypass the privacy gate that disables camera-triggered proactive behavior when `proactive_enabled` is false.
- Keep emit and ops-event paths non-throwing so sensor and worker faults remain recoverable.

## Output expectations

- Update docstrings when coordinator entry points, lifecycle semantics, or exports change.
- Keep [README.md](./README.md), [AGENTS.md](./AGENTS.md), and [component.yaml](./component.yaml) aligned when file roles or verification commands change.
- Treat export changes in `src/twinr/proactive/__init__.py` as follow-up API work.
