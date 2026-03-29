# runtime

Proactive monitor orchestration and voice-activation presence-session control for Twinr.

This package wires the runtime-facing monitor loop, degraded sensor handling,
and presence-session state used to arm voice-activation listening and proactive checks.

## Responsibility

`runtime` owns:
- Coordinate proactive monitor ticks across PIR, camera, ambient audio, and voice-activation policy paths
- Route short-window multimodal event-fusion safety claims into the active runtime trigger path while preserving the legacy social-trigger engine as fallback for non-safety and visibility-loss cases
- Maintain voice-activation presence-session state from recent sensor activity
- Derive conservative ReSpeaker audio-policy hooks such as quiet windows, resume windows, and overlap guards
- Expose one runtime-faithful ReSpeaker audio-perception diagnostic path for operator checks and post-recovery sanity validation
- Keep one-shot ReSpeaker diagnostics aligned with productive ALSA ownership by skipping competing PCM probes when the voice orchestrator already owns the same capture device
- Derive conservative ReSpeaker speech-delivery defer reasons such as background media or non-speech room activity
- Treat classified non-speech/background-media room activity as incompatible with calm `presence_audio_active`, even when the raw XVF3800 speech flag overfires without corroborating beam evidence
- Treat low-dynamic sustained RMS activity as non-speech/background-media when the XVF3800 speech flag overfires without corroborating beam evidence, so quiet music/TV does not leak back into calm speech handling just because it stays below the speech-threshold chunk gate
- Use one local PCM speech-likeness discriminator to veto strong XVF3800 speech false positives from captured room audio itself, so loudspeaker playback does not rely only on host-control flags
- Let strong PCM non-speech evidence reopen ambient activity even when the old chunk/RMS gate misses a bounded playback window, so media/noise does not disappear behind a false early `audio_activity_detected=False`
- Let already-classified non-speech/background-media outrank low-confidence overlap hints in the operator-facing room-context guard, so ambiguous speaker-playback probes do not get mislabeled as human room speech
- Enforce the hard XVF3800 startup/runtime contract so DFU/safe mode, a fully disconnected board, and unreadable enumerated capture block proactive voice monitoring clearly instead of degrading silently
- Package current ReSpeaker presence/audio facts into explicit governor-input context for downstream proactive delivery paths
- Fuse the single primary camera anchor with ReSpeaker direction confidence into a conservative speaker-association fact
- Derive a bounded multimodal initiative state that can force display-first or skip later proactive work when room context is ambiguous
- Derive an explicit `ambiguous_room_guard` before person-targeted runtime inference proceeds
- Derive a bounded `identity_fusion` surface from voice, portrait, household-voice user candidates, visual-anchor history, and presence-session memory without promoting it to identity proof
- Derive a conservative `portrait_match` claim from local identity evidence plus clear single-person room context
- Derive a conservative `known_user_hint` from fresh voice-profile state plus optional temporal identity fusion and clear single-person room context
- Derive a prompt-only `affect_proxy` surface from coarse posture, attention, quiet, and motion cues without claiming emotion as fact
- Derive bounded `near_device_presence`, optional explicit same-room `room_context`, and house-wide `home_context` layers from merged live local plus smart-home facts without letting smart-home replace local interaction truth
- Aggregate the existing presence, attention, conversation, safety, identity, room-clarity, and smart-home context surfaces into one explicit bounded `person_state` schema for downstream runtime policy and inspection
- Derive a bounded multimodal `attention_target` surface from the current camera anchor, speaker association, showing-intent cues, and short-lived session focus memory; in multi-person scenes speaker and last-motion targets outrank generic showing-intent
- Resolve one internal `PerceptionRuntimeSnapshot` from each local display-perception tick, so HDMI attention, servo follow, gesture acknowledgement, visual wakeup, and later perception consumers read the same backpressure-safe runtime truth instead of re-deriving semantics in parallel
- Derive a bounded continuous visible-person attention target from multiple camera person anchors, face-aware head retargeting, short-lived local track history, last-motion recency, and conservative audio-direction handoff, so Twinr can follow a person continuously, bridge brief detector misses without collapsing a real two-person scene, look closer to a person's face than torso center, reuse the already stabilized primary anchor in single-person scenes instead of reintroducing raw detection-box jitter, blend single-speaker audio direction back into the visual anchor when speech is live, prefer the visible speaker in simple multi-person scenes without claiming identity, and never let a stale held track override a freshly visible person
- Drive calm HDMI face attention-follow cues from the current relevant person anchor, mirroring camera-space left/right into user-facing screen gaze and deriving bounded up/down eye-follow from live person height, with small head turns/drift instead of exaggerated body-language poses, and renewing or briefly holding cues so stationary people do not cause center drift or blink out on short camera dropouts between refresh ticks
- Publish bounded HDMI header debug pills from current camera facts plus brief event/trigger holds so Pi debugging can see states such as `LOOKING_PROXY`, `LOOKING_CONFIRMED`, `POSE_UPRIGHT`, `MOTION_STILL`, `ATTENTION_WINDOW`, or `POSSIBLE_FALL` without widening the generic runtime snapshot schema
- Mirror that same conservative `attention_target` into an optional bounded body-orientation servo, so Twinr can turn toward the currently relevant visible speaker/person without inventing a second targeting policy beside the HDMI follow path
- Fail the HDMI attention-target and cue path closed when camera health is explicitly bad and no visible person exists, so stale speaker/session focus does not keep steering the face away from the user
- Keep the HDMI attention-follow cadence genuinely sub-second in production defaults, so the local face-follow path is not visually delayed by legacy status-loop timings
- Keep the fast HDMI attention-refresh path non-blocking by preferring local signal-only audio snapshots over full ambient PCM windows, so gaze and gesture acknowledgement are not serialized behind one-second audio sampling
- Keep HDMI eye-follow and HDMI gesture acknowledgements on separate local refresh paths: eye-follow should prefer a cheap attention-only camera observation plus the adapter-stream's stable attention truth, while gesture acknowledgements may use the heavier full gesture path without clearing or delaying face-follow state
- When both HDMI attention and gesture refresh are due in the same monitor cycle, fetch one shared local `observe_perception_stream()` snapshot and let both lanes consume that same `perception_stream` truth instead of recapturing the camera twice; the monitor now anchors both follow-up deadlines to that same loop start so long Pi-side observe times cannot silently dephase the shared path away again
- Keep HDMI gesture acknowledgement on its own dedicated low-latency live-gesture policy path, but run that policy only inside the shared `PerceptionStreamOrchestrator` so the user-facing ack decision is derived once per gesture tick from the authoritative adapter activation token instead of being re-derived by downstream consumers
- Derive the bounded visual gesture-wakeup decision for one configured fine-hand symbol in that same orchestrated gesture snapshot, so wakeup and HDMI acknowledgement share one activation truth and only diverge at the last presentation/dispatch step
- Dispatch accepted visual gesture wakeups through a dedicated single-flight background worker so `listening` sessions opened by Peace-sign wake do not block HDMI attention refresh ticks on the proactive monitor thread
- Prime one fresh compact sensor/person-state export from the accepted gesture snapshot immediately before dispatching a Peace-sign wake, so the runner gates the wake against current gesture-time context instead of a stale slower automation tick
- Preserve the local HDMI camera-follow/gesture path even when ReSpeaker startup capture is unreadable, marking the runtime blocked while still building the bounded on-device attention monitor instead of letting the audio contract failure erase local visual HCI entirely
- Mirror only the Pi-critical stabilized user camera gestures `thumbs_up`, `thumbs_down`, and `peace_sign` into short-lived HDMI emoji acknowledgements without touching the face channel or overwriting foreign emoji cues
- Publish very sparse personality-driven HDMI ambient reserve impulses while the room is calmly occupied, so Twinr can show one short positive non-voice nudge from its current personality and co-attention state without spamming or overriding stronger display owners
- Publish bounded get-to-know-you invitations into that same right-hand reserve lane when Twinr still has onboarding or lifelong profile-learning gaps, including short profile-review invites after corrections accumulate, so the visual action surface can reopen short discovery runs without inventing a second profile UI
- Blend those HDMI reserve impulses from personality state plus durable memory hooks such as gentle follow-ups or clarification conflicts, so the right-hand reserve can help Twinr deepen and clean up its own memory without ad-hoc hacks
- Keep reflection-derived reserve candidates focused on actual shared continuity, preferences, and recent context while suppressing operational-only packets, closure-only/meta turn residue, and device-watch packets, and never leak raw internal packet summary/details text like continuity scaffolding into the user-facing right-hand lane
- Derive direct reserve-lane world candidates from active RSS subscriptions and condensed awareness threads, so seeded world breadth still reaches the right-hand lane even when reflection and follow-up hooks are temporarily sparse, but keep the visible lane topic-led rather than source-label-led
- Rewrite the visible reserve-card text through one bounded LLM pass at plan-build time so the right-hand lane can sound like Twinr's current personality instead of a fixed template bank, using structured reserve context for shared threads, gentle follow-ups, and memory clarifications rather than just topic labels
- Keep that rewrite prompt grounded in human-facing anchors and hook hints only: raw internal packet labels, English scaffolding, or awkward deterministic fallback sentences must not be passed through as prompt anchors for the visible German card copy
- Prebuild one persistent local-day reserve plan for those HDMI ambient impulses, so the right-hand reserve can stay meaningfully populated beside the face without improvising cadence or repeating one topic ad hoc on every monitor tick
- Keep the right-hand reserve rotating the same-day card set until the user actually answers a card topic or the nightly planner replaces the day; only when no active topics remain may the lane fall back to a passive idle fill
- Record every actually shown reserve card into a bounded exposure history so later personality learning can tell which visual prompts Twinr showed before it scores user reactions
- Let very fresh shown-card pickup or pushback bias the remaining day plan immediately through one short-lived reserve-bus feedback hint, so the right-hand lane adapts faster than the slower long-term learning loop alone
- Run one explicit overnight reserve-lane maintenance step that reviews recent shown-card outcomes, triggers long-term reflection/world refresh, and prepares the next local-day reserve plan ahead of morning adoption
- Retry a same-day reserve plan after a short backoff when the first build was empty, so late-arriving personality candidates can repopulate the right-hand reserve instead of leaving it blank until midnight
- Keep HDMI attention-follow available on voice-activation/runtime-monitor builds even when `proactive_enabled` remains off for camera-triggered proactive prompts
- Run a bounded local HDMI attention-refresh cadence that keeps gaze-follow responsive even when full proactive inspection is still PIR-gated, and keep that local cue-only path alive while the main runtime is in `error`
- Bootstrap one bounded local vision inspection from live speech when voice-activation mode is enabled, so a quiet or missing PIR does not leave presence-gated voice activation permanently idle
- Retry exact transient `Device or resource busy` PIR startup overlaps for a short bounded window, including libgpiod CLI busy failures surfaced through `gpioget`, so runtime restarts do not flap into error while a previous Twinr process is still releasing GPIO17
- Pause the active streaming voice activation `arecord` handle while an accepted voice activation or visual gesture wakeup opens the exclusive hands-free conversation capture, then resume streaming after the handler returns so successful wakeups do not immediately fail with ALSA `Device or resource busy`
- Disable proactive PCM fallback and ReSpeaker readable-frame startup probes when a long-lived voice-orchestrator capture already owns the same ALSA device and no shared voice activation buffer exists, keeping ReSpeaker host-control monitoring alive without fighting that capture with a second `arecord`
- Export normalized observation facts and ops telemetry from proactive monitoring, including raw local-camera readiness/count/gesture fields for Pi-side presence debugging
- Export a dedicated changed-only HDMI attention-follow ops trace so Pi-side debugging can correlate camera health, stabilized person anchors, attention-target state, and cue-publish decisions without blind tuning
- Persist a bounded continuous HDMI attention debug stream with per-refresh outcome codes and stage timings so short-lived eye-follow dropouts can be diagnosed after the fact
- Persist a bounded continuous HDMI gesture debug stream with per-refresh outcome codes, raw gesture observations, ack-lane decisions, publish results, stage timings, and bounded pipeline provenance from the dedicated gesture lane, and allow an opt-in end-to-end workflow run-pack around the gesture refresh during hard Pi repros, so symbol latency and dropouts can be diagnosed after the fact without guessing whether the miss came from the live path, a recent-person fallback, a recent-hand fallback, or publish policy
- Export structured ReSpeaker audio-policy facts, per-claim confidence/source metadata, and presence-session IDs for bounded automation and long-term memory ingestion
- Inject ReSpeaker XVF3800 signal facts into runtime audio observations when that device is targeted
- Keep bounded ReSpeaker audio observation alive while Twinr is already speaking so interruption facts do not go blind in `answering`
- Schedule heavy XVF3800 host-control polling more slowly than the cheap ambient-audio path whenever the room is idle
- Emit explicit operator-readable ReSpeaker runtime alerts for DFU, disconnect, unreadable capture, and recovery state changes
- Block sensitive proactive behavior when current ReSpeaker and camera context indicates multi-person ambiguity or low-confidence audio direction
- Open and close the bounded proactive monitor worker and its resources

`runtime` does **not** own:
- Score social triggers or define proactive prompt content
- Implement voice-activation matching, calibration, or evaluation algorithms
- Implement raw camera, PIR, or microphone adapters
- Enforce delivery cooldown policy after a trigger becomes a candidate

## Key files

| File | Purpose |
|---|---|
| `__init__.py` | Package export surface |
| `audio_perception.py` | Runtime-faithful one-shot ReSpeaker perception diagnostics and conservative room/device-directedness guard summaries, including the shared-capture rule that stays signal-only while the voice orchestrator already owns the same ALSA device |
| `pir_open_gate.py` | Bounded PIR startup gate that retries only exact transient GPIO busy overlaps |
| `respeaker_capture_gate.py` | Bounded startup gate that requires sustained readable ReSpeaker capture before the runtime leaves startup error state |
| `audio_policy.py` | Conservative ReSpeaker policy-hook, speech-defer, and runtime-alert derivation |
| `ambiguous_room_guard.py` | Fail-closed room-ambiguity guard for person-targeted runtime inferences |
| `identity_fusion.py` | Bounded temporal/session identity fusion over voice, portrait, enrolled household-voice candidates, and visual-anchor history |
| `portrait_match.py` | Conservative runtime claim surface for local portrait-match observations, including temporal evidence metadata |
| `known_user_hint.py` | Conservative known-user hint from voice-profile state plus optional temporal identity-fusion evidence |
| `affect_proxy.py` | Prompt-only affect proxy surface from coarse posture, attention, and quiet cues |
| `smart_home_context.py` | Bounded layered runtime context tracker for local presence plus optional same-room and home-wide smart-home context |
| `person_state.py` | Bounded aggregate `person_state` schema that maps the existing runtime claim surfaces into stable axes for downstream policy |
| `attention_targeting.py` | Bounded multimodal attention-target prioritization over camera anchor, speaker association, showing intent, and session focus memory; speaker and last-motion targets keep priority in multi-person scenes |
| `perception_orchestrator.py` | Internal single-source runtime perception orchestrator that turns one perception tick into one shared attention/gesture snapshot for HDMI, servo, and wake consumers |
| `attention_debug_stream.py` | Bounded JSONL tick stream for every HDMI attention refresh, including outcome codes, stage timings, current camera/target/cue state, and fast-path `LOOKING` source/state plus `HAND_NEAR`/`INTENT_LIKELY` gate reasons for Pi-side dropout forensics |
| `continuous_attention.py` | Short-lived visible-person track matching, stabilized single-person anchor alignment, stale-track bridging across brief detector misses, last-motion fallback, adaptive audio mirror calibration, and continuous target-center prediction for HDMI gaze follow |
| `display_attention.py` | Conservative proactive producer and local refresh policy for HDMI gaze-follow face cues, mirroring camera-space anchors into user-facing gaze, deriving bounded up/down follow from live person height, and renewing or briefly holding cues before they expire or disappear on short camera dropouts |
| `display_attention_camera_fusion.py` | Display-only fusion helper that reuses recent richer full-observe and gesture-lane semantics for pose/hand/dropout carry-over while keeping adapter-stream `LOOKING`/attention truth authoritative on the fast HDMI lane |
| `display_debug_signals.py` | Bounded publisher for optional HDMI header debug pills derived from current camera facts plus brief event/trigger holds such as `LOOKING_PROXY`, `LOOKING_CONFIRMED`, `ENGAGED`, `ENGAGED_PROXY`, `ATTENTION_WINDOW`, `POSITIVE_CONTACT`, or `POSSIBLE_FALL`, with unchanged-write suppression until the current snapshot actually nears expiry; `ENGAGED` is intentionally reserved for confirmed looking while proxy-only attention stays `ENGAGED_PROXY`, and the current local Pi vision path still does not emit smile evidence so `POSITIVE_CONTACT` remains upstream-limited until a smile-capable provider exists |
| `display_gesture_emoji.py` | Conservative runtime producer that mirrors only the Pi-critical stabilized rising-edge user gestures `thumbs_up`, `thumbs_down`, and `peace_sign` into short-lived HDMI emoji acknowledgements without overwriting foreign emoji cues |
| `display_reserve_candidates.py` | Compatibility wrapper that preserves the historic reserve-candidate loader API while delegating to the modular ambient companion flow |
| `display_reserve_flow.py` | Shared ambient companion orchestration that blends personality, memory, reflection, latent snapshot-topic backfill, and long-horizon reserve-lane learning into one bounded semantic-topic pool, then hands those grounded topics into deterministic card-surface expansion before copy rewrite |
| `display_reserve_diversity.py` | Shared reserve-lane diversity policy that normalizes raw candidate sources into broader conversational seed families and axes, then selects a broader raw pool before copy rewrite and day-plan spacing |
| `display_reserve_expansion.py` | Deterministic semantic-topic bundling and multi-angle card expansion layer that turns grounded reserve topics into up to three concrete right-lane card surfaces while preserving one shared semantic topic key for retirement, feedback, and learning |
| `display_reserve_learning.py` | Long-horizon reserve-lane learning profile derived from durable shown-card outcomes so future plans can reward welcome openers and cool repetitive ones generically |
| `display_reserve_memory.py` | Focused translation of durable long-term memory clarification and gentle follow-up hooks into reserve-lane candidate families, including semantic `card_intent` blocks so follow-up and clarification cards are written from structured meaning instead of only question/reason fragments |
| `display_reserve_snapshot_topics.py` | Latent snapshot-topic reserve loader that backfills additional continuity, relationship, place, and concrete world-signal seeds from the persisted personality snapshot when stronger explicit loaders stay sparse, while keeping the output generic and display-safe |
| `display_reserve_user_discovery.py` | Guided get-to-know-you invitation adapter that turns due onboarding, lifelong-learning, and short profile-review prompts into reserve-lane candidates while exposing human-facing prompt anchors plus structured semantic `card_intent` blocks instead of raw setup labels like `Basisinfos` or clause-like visible prompts such as `Wie ich dich ansprechen soll`; prompt hooks now also switch into a follow-up stage once a topic already has saved coverage so partially answered topics do not reopen with the stale opener wording |
| `display_reserve_user_discovery_feedback.py` | Direct bridge from explicit `manage_user_discovery` reactions on visible discovery invites into reserve exposure history plus short-lived reserve-bus feedback so already answered discovery cards retire from the same-day HDMI plan immediately instead of waiting for generic ambient topic matching |
| `display_reserve_reflection.py` | Reflection-derived reserve-candidate loader that turns summaries plus selected midterm continuity/context packets into calm ambient conversation openers while suppressing operational-only, closure-only, transcript-only continuity residue, and device/meta packets, converting internal reflection text into display-safe fallback copy, and emitting semantic `card_intent` blocks for earlier-thread callbacks or personal follow-ups |
| `display_reserve_world.py` | Direct world-intelligence reserve-candidate loader that turns active RSS subscriptions and condensed awareness threads into calm topic-breadth candidates for the right-hand lane without surfacing raw source-label cards, and emits semantic `card_intent` blocks so public-topic cards can read like concrete Twinr observations instead of generic labels |
| `display_reserve_copy_contract.py` | Shared writer/judge copy assets for reserve-card generation, including normalized copy families, small family-scoped gold examples, and the explicit quality rubric that both passes use as a positive style contract |
| `display_reserve_prompting.py` | Compact prompt-payload and anchoring layer that turns rich reserve candidates into small `topic_anchor`/`hook_hint`/`card_intent`/`context_summary` prompt objects plus a condensed `pickup_signal` derived from real reserve-card outcomes for the rewrite step, and now attaches semantic topic anchors, `expansion_angle`, normalized `copy_family`, family-scoped gold examples, and the shared quality rubric to each bounded batch |
| `display_reserve_generation.py` | Rewrite reserve-card copy through small bounded structured LLM batches so the lane sounds personality-shaped in normal idiomatic German instead of translated prompt-language; the rewrite now runs as a bounded writer/judge flow where the first pass proposes a few distinct card variants per concrete card surface and the second pass selects the strongest Twinr-style final card against the shared quality rubric and family examples, while preserving the semantic topic grouping that drives feedback and retirement |
| `display_reserve_day_plan.py` | Persistent current-day reserve planner that turns expanded reserve card surfaces into one calm unique rotation with cursor persistence, generic topic/source/family spacing, normalized seed-family and public/personal/setup axis spacing, bounded retry for empty same-day plans, semantic-topic spacing between sibling cards, same-day looping until real user feedback retires a semantic topic, passive fallback lookup only when no active topics remain, and short-lived feedback-driven replanning that removes answered topic bundles without named-topic hacks |
| `display_reserve_companion_planner.py` | Explicit overnight reserve-lane planner that runs nightly review plus long-term reflection/world refresh, stores prepared next-day plans, and lets the morning runtime adopt that reviewed plan instead of rebuilding ad hoc |
| `display_reserve_support.py` | Shared side-effect-free text, timestamp, and local-time helpers that keep reserve publishers and planners on one normalization contract |
| `display_reserve_runtime.py` | Shared reserve-lane runtime publisher that all right-hand text cards use so planned companion impulses and real-time social/focus prompts hit the same cue/history path, carrying both concrete card ids and grouped semantic topic keys, plus one visible-only publish mode for passive idle fills that must not create duplicate exposure history |
| `display_ambient_impulses.py` | Conservative live publisher that exposes the next planned reserve item only when quiet-hours, presence, runtime state, and reserve-surface ownership all allow it, adopting prepared next-day plans when morning rollover begins and restoring passive fill after temporary overrides or exhausted plans instead of leaving the right lane blank |
| `display_social_reserve.py` | Route visual-first proactive social prompts into the same right-hand reserve lane and the same shared reserve publish path instead of taking over the fullscreen presentation surface |
| `safety_trigger_fusion.py` | Bridge `event_fusion` claims into runtime safety-trigger selection without bypassing engine cooldowns, review, or suppression policy |
| `gesture_ack_lane.py` | Internal low-latency HDMI acknowledgement policy used by `perception_orchestrator.py`; it consumes only the authoritative perception-stream activation token and keeps just one HDMI-specific repeat hold locally |
| `gesture_wakeup_dispatcher.py` | Single-flight background dispatcher that runs accepted visual wakeups off the proactive monitor thread so eye-follow keeps refreshing during listening |
| `gesture_wakeup_lane.py` | Internal visual-wakeup policy used by `perception_orchestrator.py`; it consumes only the authoritative perception-stream activation token and keeps just dispatch edge detection plus wake cooldown locally |
| `gesture_wakeup_priority.py` | Interaction-priority guard that lets button/voice outrank visual wakeups before a gesture can steal shared audio capture |
| `gesture_debug_stream.py` | Bounded JSONL tick stream for HDMI gesture-refresh diagnostics, including latency stages, raw observations, ack-lane decisions, and publish outcomes |
| `claim_metadata.py` | Shared `confidence` / `source` / `requires_confirmation` helpers for multimodal runtime claims |
| `multimodal_initiative.py` | Conservative multimodal initiative readiness and display-first recommendation from camera + ReSpeaker facts |
| `runtime_contract.py` | Hard startup-blocker contract for XVF3800 DFU/safe-mode and disconnected states |
| `speaker_association.py` | Conservative association of current speech to the single primary visible person anchor |
| `sensitive_behavior_gate.py` | Conservative gate that blocks sensitive proactive behavior on ambiguous multi-person or low-confidence audio context |
| `governor_inputs.py` | Focused governor-facing packaging of current ReSpeaker presence/audio facts |
| `presence.py` | Presence-session state machine |
| `service_impl/README.md` | Internal map of the refactored `service.py` implementation package; wrapper path stays stable and no migration is required |
| `service_impl/coordinator_core.py` | Dependency wiring plus the main proactive tick/orchestration path extracted from the legacy `service.py` monolith |
| `service_impl/coordinator_display.py` | Dedicated HDMI attention and gesture refresh workflows extracted from the legacy `service.py` monolith; they now consume one shared orchestrated runtime perception contract instead of directly running lane-local temporal policy |
| `service_impl/coordinator_perception.py` | Shared display-refresh cycle helper that reuses one combined `perception_stream` snapshot across simultaneous HDMI attention and gesture refreshes, while the monitor keeps those shared cycles phase-locked under Pi overrun |
| `service_impl/coordinator_observation.py` | Stable observation-mixin facade that composes the focused observation helper modules behind the historic path |
| `service_impl/coordinator_observation_display.py` | Automation-dispatch, HDMI display, gesture, and camera-surface helper bridges extracted from the legacy observation mixin |
| `service_impl/coordinator_observation_facts.py` | Automation fact payload assembly and rising-edge sensor-event derivation extracted from the legacy observation mixin |
| `service_impl/monitor.py` | Background worker lifecycle wrapper for the proactive coordinator |
| `service_impl/builder.py` | Default production assembly path, with wrapper-supplied dependencies to preserve legacy monkeypatch points |
| `service_attention_helpers.py` | Focused HDMI attention-follow, live-context export, servo-trace, and attention-debug helper surface used by `service.py`; attention-target derivation now comes from `perception_orchestrator.py` so servo and HDMI reuse the same runtime truth |
| `service_gesture_helpers.py` | Focused HDMI gesture acknowledgement, wakeup dispatch, and gesture-trace helper surface used by `service.py` so gesture policy stays out of the main monitor loop |
| `service.py` | Compatibility shim that preserves the historic import path and helper surface while delegating the real implementation to `service_impl/` |
| `component.yaml` | Structured package metadata |
| `AGENTS.md` | Local editing rules |

No migration is required for callers that already import `twinr.proactive.runtime.service`.

## Usage

```python
from twinr.proactive.runtime import build_default_proactive_monitor

monitor = build_default_proactive_monitor(
    config=config,
    runtime=runtime,
    backend=backend,
    camera=camera,
    camera_lock=camera_lock,
    audio_lock=audio_lock,
    trigger_handler=trigger_handler,
)
if monitor is not None:
    with monitor:
        ...
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [Pi Gesture Baseline](../../../../docs/PI_GESTURE_BASELINE.md)
- [../governance/README.md](../governance/README.md)
- [../social/README.md](../social/README.md)
