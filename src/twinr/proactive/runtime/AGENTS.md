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
- `attention_debug_stream.py` — bounded continuous JSONL debug stream for HDMI attention refresh ticks, including outcome codes, stage timings, current camera/target/cue state, and fast-path source/state plus gate reasons for `LOOKING`/`HAND_NEAR`/`INTENT_LIKELY`
- `display_attention.py` — conservative proactive producer that steers HDMI eye gaze toward the currently relevant visible person without overwriting foreign face cues, while keeping normal follow behavior horizontal-only, mirroring camera-space left/right into user-facing gaze, adding small near-center head turns before full side gaze commits, and renewing or briefly holding matching cues before they expire or disappear on short camera dropouts
- `display_attention_camera_fusion.py` — display-only fusion helper that warms the fast HDMI attention lane with recent richer pose/gesture semantics, including `proxy` vs `confirmed` looking provenance, and short dropout holds without turning the fast path into a second full vision loop
- `display_debug_signals.py` — bounded publisher that mirrors current camera facts plus brief event/trigger holds into optional HDMI header debug pills such as `LOOKING_PROXY` / `LOOKING_CONFIRMED`, source-aware `ENGAGED` / `ENGAGED_PROXY` where `ENGAGED` is reserved for confirmed looking, or held `POSITIVE_CONTACT` for Pi-side inspection while suppressing redundant unchanged rewrites until the current snapshot actually nears expiry
- `display_gesture_emoji.py` — conservative producer that mirrors stabilized rising-edge user gestures into bounded HDMI emoji acknowledgements without touching the face channel or overwriting foreign emoji cues
- `display_reserve_candidates.py` — compatibility wrapper for the modular ambient companion flow; preserve the historic public loader while delegating candidate sourcing elsewhere
- `display_reserve_flow.py` — shared ambient companion orchestration that blends personality, memory, reflection, and long-horizon reserve-lane learning into one bounded candidate set
- `display_reserve_learning.py` — long-horizon reserve-lane learning profile derived from shown-card outcomes so future plans can reward welcome openers and cool repetitive ones generically
- `display_reserve_memory.py` — focused memory-derived reserve-candidate conversion for clarification conflicts and gentle follow-ups
- `display_reserve_reflection.py` — reflection-derived reserve-candidate loading from summaries and midterm packets without coupling planning to reflection internals, including deterministic display-safe fallback copy so raw internal packet text, farewell/meta residue, and device-watch packets never reach the HDMI right lane
- `display_reserve_world.py` — direct world-intelligence reserve-candidate loading from active subscriptions and condensed awareness threads so the right lane keeps topic breadth even when reflection or follow-up hooks are thin, while staying topic-led instead of source-label-led
- `display_reserve_prompting.py` — compact prompt-payload builder that distills reserve candidates into explicit topical anchors and hook hints before they reach the LLM rewrite step
- `display_reserve_generation.py` — bounded LLM rewrite layer for reserve-card text so the right-hand lane reflects Twinr's current personality without moving generation into the hot monitor tick, using small isolated batches instead of one large all-or-nothing request
- `display_reserve_day_plan.py` — persistent current-day planner that turns personality-driven reserve candidates into one calm daily rotation for the HDMI reserve area, mixing generic candidate families so the lane does not clump around one source family and keeping topics in same-day rotation until real user feedback retires them or the nightly planner replaces the plan
- `display_reserve_companion_planner.py` — explicit overnight reserve-lane recalibration that runs nightly reflection/world refresh, reviews recent shown-card outcomes, and stores the prepared next-day plan for morning adoption
- `display_reserve_runtime.py` — shared reserve-lane publish path for all right-hand text cards so planned and spontaneous prompts reuse one cue/history contract
- `display_ambient_impulses.py` — conservative live publisher that shows the next planned positive personality-driven HDMI reserve impulse only when quiet hours, presence, runtime state, and surface ownership all allow it, then hands the concrete publish to the shared reserve runtime path
- `display_social_reserve.py` — dedicated publisher that routes display-first proactive social prompts into the same right-hand reserve lane and same shared reserve publish path instead of the fullscreen presentation surface
- `safety_trigger_fusion.py` — runtime-only bridge that prefers short-window fused safety claims for `possible_fall`, `floor_stillness`, and `distress_possible` while keeping the legacy engine as fallback and source of cooldown truth
- `gesture_ack_lane.py` — dedicated low-latency acknowledgement stabilizer for the explicit user-facing gesture symbol set, separate from general camera-surface state
- `gesture_wakeup_dispatcher.py` — single-flight background dispatcher that runs accepted visual wakeups off the proactive monitor worker
- `gesture_wakeup_lane.py` — dedicated visual wakeup stabilizer for one configured fine-hand symbol, separate from emoji acknowledgement and workflow entry orchestration
- `gesture_wakeup_priority.py` — interaction-priority guard that lets button/voice outrank visual wakeups before a gesture can steal shared audio capture
- `gesture_debug_stream.py` — bounded continuous JSONL debug stream for HDMI gesture refresh ticks, including raw observations, ack decisions, publish outcomes, and stage timings
- `service.py` — runtime monitor orchestration plus the opt-in end-to-end gesture-refresh workflow run-pack binding used for hard Pi repro forensics
- `audio_perception.py` — runtime-faithful one-shot ReSpeaker perception diagnostics and conservative room/device-directedness guard summaries for operator checks and recovery smokes
- `pir_open_gate.py` — bounded startup gate that retries only exact transient PIR GPIO busy overlaps during runtime handover
- `respeaker_capture_gate.py` — bounded startup gate that requires consecutive readable-frame probes before ReSpeaker capture is considered stable
- `claim_metadata.py` — shared `confidence` / `source` / `requires_confirmation` helpers for runtime claims
- `speaker_association.py` — conservative speaker-to-camera-anchor association for the single-primary-person case
- `multimodal_initiative.py` — confidence-bearing display-first/skip gate for later proactive behavior
- `presence.py` — source of truth for voice-activation presence-session arming
- `service.py` — monitor orchestration, degraded-mode handling, lifecycle, and shared attention-surface fanout to HDMI plus optional body-follow servo
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
- `display_reserve_day_plan.py` must keep reserve scheduling persistent and local-day-scoped: plan generation belongs there, not inline in the proactive monitor tick, and repetition control must stay generic rather than topic-hardcoded. If the first same-day plan is empty, it must retry after a short bounded backoff rather than leaving the reserve blank until midnight. A current-day plan must keep unresolved topics in same-day rotation instead of treating one visual exposure as retirement; only real user feedback or the next nightly replacement may take a topic out of that day’s loop. Fresh shown-card feedback may trigger a rebuild, but only through bounded generic biasing and answer-based retirement rather than ad-hoc named-topic overrides. The planner may spread source families such as world/place/memory/social/reflection, but that family mixing must stay generic and derived from candidate state rather than named-topic rules.
- `display_reserve_companion_planner.py` owns the explicit overnight reserve-lane review. Nightly reflection/world-refresh triggering, prepared next-day plan storage, recent shown-card outcome summarization, and morning adoption rules belong there rather than in `display_ambient_impulses.py` or the active loop classes.
- `display_reserve_flow.py` owns ambient companion candidate orchestration. Keep memory hooks in `display_reserve_memory.py`, slower reflection sourcing in `display_reserve_reflection.py`, and long-horizon shown-card learning in `display_reserve_learning.py` instead of regrowing one large mixed loader.
- `display_reserve_reflection.py` must never pass internal reasoning scaffolding such as English continuity packet summaries/details straight through to user-facing reserve cards. If richer copy is unavailable, fall back to bounded display-safe prompts derived from structured anchors, not raw packet internals.
- `display_reserve_reflection.py` must suppress closure-only, meta-turn, and device-watch packets unless they carry a clear user-recognizable thread anchor. Stored memory may keep those packets; the visible right lane must not.
- `display_reserve_flow.py` owns orchestration only. Active RSS subscription topics and condensed awareness threads belong in `display_reserve_world.py`, and the right lane should prefer topic-level world candidates over raw feed/source-label cards.
- `display_reserve_candidates.py` must remain a thin compatibility seam only. New reserve-lane sourcing, ranking, or learning logic belongs in the focused companion-flow modules, not in the wrapper.
- `display_reserve_prompting.py` owns prompt compaction and anchor shaping for reserve-card rewriting. Keep raw candidate contexts out of the LLM call; the rewrite step should see explicit `topic_anchor` / `hook_hint` / compact context instead of whole nested reflection or world payloads.
- `display_reserve_prompting.py` must only expose human-facing anchors and hook hints to the rewrite step. Do not pass deterministic fallback headlines/bodies or raw internal English scaffolding through the prompt payload, or the model will anchor on that low-quality text.
- `display_reserve_generation.py` must stay bounded and rewrite-only: it may shape tone and wording from Twinr's current personality plus structured reserve context, but it must not reorder topics, improvise new memory facts, or move provider calls into the live publish tick. The visible copy should stay natural rather than label-like, it still needs one clear topical anchor, and failures must degrade per batch rather than wiping the whole rewrite pass. Treat empty SDK `output_parsed` as a transport quirk, not an automatic failure, when the model still returned valid JSON text.
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
- HDMI eye-follow and HDMI gesture acknowledgement must stay decoupled. Eye-follow should prefer a cheap attention-only local camera observation and its own stabilized camera-surface state, while gesture acknowledgement may use the heavier full gesture path without clearing, delaying, or otherwise regressing face-follow behavior.
- HDMI gesture acknowledgement should prefer its own dedicated live-gesture observation path and ack-lane state. Do not route user-facing symbol latency through the heavier social camera-surface path when the dedicated gesture refresh is available.
- `display_attention_camera_fusion.py` must stay display-only. It may reuse recent richer semantics from the full observe and gesture lanes to stabilize face-follow/debug pills, but it must not widen the generic automation fact contract or silently turn the HDMI fast path into another full pose loop.
- `service.py` should keep a bounded changed-only ops trace for HDMI attention-follow decisions so Pi debugging can prove whether failures come from camera health, stabilized camera semantics, target selection, or cue publishing.
- `service.py` should also keep a bounded continuous attention debug stream for each HDMI refresh tick so transient 1-3 minute eye-follow dropouts still leave first-run evidence after the fact.
- `service.py` should also keep a bounded continuous gesture debug stream for each HDMI gesture refresh tick so Pi-side symbol latency and missed acknowledgements can be diagnosed from first-run evidence instead of blind retuning. That stream should include bounded provenance from the dedicated gesture pipeline itself when available, so operators can tell whether a miss came from full-frame live recognition, a recent-person ROI fallback, a recent-hand ROI fallback, or publish/ack policy.
- When hard Pi gesture repros still are not explained by `gesture_debug_stream.py`, `service.py` may bind an opt-in end-to-end workflow run-pack around the gesture refresh, but that path must stay bounded, redacted by default, and debugging-only.
- `service_attention_helpers.py` owns HDMI attention-follow live-context export, changed-only follow traces, servo decision logging, and per-tick attention debug packaging. `service.py` should orchestrate those helpers rather than regrowing attention logic inline.
- `service_gesture_helpers.py` owns HDMI gesture acknowledgement publish, accepted gesture wakeup dispatch, and gesture debug/trace packaging. `service.py` should remain the lifecycle/tick coordinator, not the gesture-policy implementation file.
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
PYTHONPATH=src pytest test/test_proactive_monitor.py -q
PYTHONPATH=src pytest test/test_proactive_runtime_service.py -q
PYTHONPATH=src pytest test/test_display_reserve_support.py test/test_display_reserve_generation.py test/test_display_reserve_prompting.py test/test_display_reserve_learning.py test/test_display_reserve_reflection.py test/test_display_reserve_flow.py test/test_display_reserve_day_plan.py test/test_display_reserve_companion_planner.py test/test_display_reserve_runtime.py test/test_display_ambient_impulses.py test/test_display_social_reserve.py -q
```

If workflow wiring or export surface changed, also run:

```bash
PYTHONPATH=src pytest test/test_realtime_runner.py test/test_streaming_runner.py -q
```

## Coupling

`presence.py` changes -> also check:
- `service.py`
- `test/test_proactive_monitor.py`

`service.py`, `service_attention_helpers.py`, `service_gesture_helpers.py`, `display_ambient_impulses.py`, `display_social_reserve.py`, `display_reserve_runtime.py`, `display_reserve_day_plan.py`, `display_reserve_companion_planner.py`, `display_reserve_flow.py`, `display_reserve_learning.py`, `display_reserve_memory.py`, `display_reserve_reflection.py`, `display_reserve_prompting.py`, `display_reserve_generation.py`, or `display_reserve_support.py` changes -> also check:
- `src/twinr/proactive/social/`
- `src/twinr/agent/workflows/realtime_runner.py`
- `src/twinr/agent/workflows/streaming_runner.py`
- `test/test_proactive_monitor.py`
- `test/test_proactive_runtime_service.py`
- `test/test_display_reserve_flow.py`
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
