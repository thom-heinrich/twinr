# runtime

Proactive monitor orchestration and wakeword presence-session control for Twinr.

This package wires the runtime-facing monitor loop, degraded sensor handling,
and presence-session state used to arm wakeword listening and proactive checks.

## Responsibility

`runtime` owns:
- Coordinate proactive monitor ticks across PIR, camera, ambient audio, and wakeword policy paths
- Maintain wakeword presence-session state from recent sensor activity
- Derive conservative ReSpeaker audio-policy hooks such as quiet windows, resume windows, and overlap guards
- Expose one runtime-faithful ReSpeaker audio-perception diagnostic path for operator checks and post-recovery sanity validation
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
- Derive a bounded continuous visible-person attention target from multiple camera person anchors, face-aware head retargeting, short-lived local track history, last-motion recency, and conservative audio-direction handoff, so Twinr can follow a person continuously, bridge brief detector misses without collapsing a real two-person scene, look closer to a person's face than torso center, and prefer the visible speaker in simple multi-person scenes without claiming identity
- Drive calm HDMI face attention-follow cues from the current relevant person anchor, mirroring camera-space left/right into user-facing screen gaze and deriving bounded up/down eye-follow from live person height, with small head turns/drift instead of exaggerated body-language poses, and renewing or briefly holding cues so stationary people do not cause center drift or blink out on short camera dropouts between refresh ticks
- Fail the HDMI attention-target and cue path closed when camera health is explicitly bad and no visible person exists, so stale speaker/session focus does not keep steering the face away from the user
- Keep the HDMI attention-follow cadence genuinely sub-second in production defaults, so the local face-follow path is not visually delayed by legacy status-loop timings
- Keep the fast HDMI attention-refresh path non-blocking by preferring local signal-only audio snapshots over full ambient PCM windows, so gaze and gesture acknowledgement are not serialized behind one-second audio sampling
- Keep HDMI eye-follow and HDMI gesture acknowledgements on separate local refresh paths: eye-follow should prefer a cheap attention-only camera observation and its own stabilized surface, while gesture acknowledgements may use the heavier full gesture path without clearing or delaying face-follow state
- Keep HDMI gesture acknowledgement on its own dedicated low-latency live-gesture lane plus self-contained ack stabilizer, so user-facing symbols do not depend on the broader social camera-surface cadence or regress eye-follow when gesture policy changes
- Derive a bounded visual gesture-wakeup decision from one configured fine-hand symbol and hand it to the workflow entry path without coupling it to HDMI emoji acknowledgement or wakeword audio policy
- Dispatch accepted visual gesture wakeups through a dedicated single-flight background worker so `listening` sessions opened by Peace-sign wake do not block HDMI attention refresh ticks on the proactive monitor thread
- Preserve the local HDMI camera-follow/gesture path even when ReSpeaker startup capture is unreadable, marking the runtime blocked while still building the bounded on-device attention monitor instead of letting the audio contract failure erase local visual HCI entirely
- Mirror clear stabilized user camera gestures such as thumbs-up or waving into short-lived HDMI emoji acknowledgements without touching the face channel or overwriting foreign emoji cues; motion-bearing coarse gestures like waving must outrank a simultaneous generic open-palm hand shape and briefly suppress conflicting custom-only fine-hand symbols like `👌` so `👋` does not flap into a custom false positive while the user is still waving
- Publish very sparse personality-driven HDMI ambient reserve impulses while the room is calmly occupied, so Twinr can show one short positive non-voice nudge from its current personality and co-attention state without spamming or overriding stronger display owners
- Prebuild one persistent local-day reserve plan for those HDMI ambient impulses, so the right-hand reserve can stay meaningfully populated beside the face without improvising cadence or repeating one topic ad hoc on every monitor tick
- Retry a same-day reserve plan after a short backoff when the first build was empty, so late-arriving personality candidates can repopulate the right-hand reserve instead of leaving it blank until midnight
- Keep HDMI attention-follow available on wakeword/runtime-monitor builds even when `proactive_enabled` remains off for camera-triggered proactive prompts
- Run a bounded local HDMI attention-refresh cadence that keeps gaze-follow responsive even when full proactive inspection is still PIR-gated, and keep that local cue-only path alive while the main runtime is in `error`
- Bootstrap one bounded local vision inspection from live speech when wakeword mode is enabled, so a quiet or missing PIR does not leave presence-gated wakeword permanently idle
- Retry exact transient `Device or resource busy` PIR startup overlaps for a short bounded window, so runtime restarts do not flap into error while a previous Twinr process is still releasing GPIO17
- Pause the active streaming wakeword `arecord` handle while an accepted wakeword or visual gesture wakeup opens the exclusive hands-free conversation capture, then resume streaming after the handler returns so successful wakeups do not immediately fail with ALSA `Device or resource busy`
- Export normalized observation facts and ops telemetry from proactive monitoring, including raw local-camera readiness/count/gesture fields for Pi-side presence debugging
- Export a dedicated changed-only HDMI attention-follow ops trace so Pi-side debugging can correlate camera health, stabilized person anchors, attention-target state, and cue-publish decisions without blind tuning
- Persist a bounded continuous HDMI attention debug stream with per-refresh outcome codes and stage timings so short-lived eye-follow dropouts can be diagnosed after the fact
- Persist a bounded continuous HDMI gesture debug stream with per-refresh outcome codes, raw gesture observations, ack-lane decisions, publish results, stage timings, and bounded pipeline provenance from the dedicated gesture lane, so Pi-side symbol latency and dropouts can be diagnosed after the fact without guessing whether the miss came from the live path, a recent-person fallback, a recent-hand fallback, or publish policy
- Export structured ReSpeaker audio-policy facts, per-claim confidence/source metadata, and presence-session IDs for bounded automation and long-term memory ingestion
- Inject ReSpeaker XVF3800 signal facts into runtime audio observations when that device is targeted
- Keep bounded ReSpeaker audio observation alive while Twinr is already speaking so interruption facts do not go blind in `answering`
- Schedule heavy XVF3800 host-control polling more slowly than the cheap ambient-audio path whenever the room is idle
- Emit explicit operator-readable ReSpeaker runtime alerts for DFU, disconnect, unreadable capture, and recovery state changes
- Block sensitive proactive behavior when current ReSpeaker and camera context indicates multi-person ambiguity or low-confidence audio direction
- Open and close the bounded proactive monitor worker and its resources

`runtime` does **not** own:
- Score social triggers or define proactive prompt content
- Implement wakeword matching, calibration, or evaluation algorithms
- Implement raw camera, PIR, or microphone adapters
- Enforce delivery cooldown policy after a trigger becomes a candidate

## Key files

| File | Purpose |
|---|---|
| `__init__.py` | Package export surface |
| `audio_perception.py` | Runtime-faithful one-shot ReSpeaker perception diagnostics and conservative room/device-directedness guard summaries |
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
| `attention_debug_stream.py` | Bounded JSONL tick stream for every HDMI attention refresh, including outcome codes, stage timings, and current camera/target/cue state for Pi-side dropout forensics |
| `continuous_attention.py` | Short-lived visible-person track matching, stale-track bridging across brief detector misses, last-motion fallback, adaptive audio mirror calibration, and continuous target-center prediction for HDMI gaze follow |
| `display_attention.py` | Conservative proactive producer and local refresh policy for HDMI gaze-follow face cues, mirroring camera-space anchors into user-facing gaze, deriving bounded up/down follow from live person height, and renewing or briefly holding cues before they expire or disappear on short camera dropouts |
| `display_gesture_emoji.py` | Conservative runtime producer that mirrors stabilized rising-edge user gestures into short-lived HDMI emoji acknowledgements without overwriting foreign emoji cues and briefly keeps motion gestures authoritative over conflicting custom-only fine-hand false positives |
| `display_reserve_day_plan.py` | Persistent local-day planner that turns personality-driven reserve candidates into one calm per-day sequence with cursor persistence, topic spacing, and bounded retry for empty same-day plans |
| `display_ambient_impulses.py` | Conservative live publisher that exposes the next planned reserve item only when quiet-hours, presence, runtime state, and reserve-surface ownership all allow it |
| `gesture_ack_lane.py` | Dedicated low-latency gesture acknowledgement stabilizer for the explicit user-facing symbol set, separate from general camera-surface state |
| `gesture_wakeup_dispatcher.py` | Single-flight background dispatcher that runs accepted visual wakeups off the proactive monitor thread so eye-follow keeps refreshing during listening |
| `gesture_wakeup_lane.py` | Dedicated visual wakeup stabilizer for one configured fine-hand symbol such as `peace_sign`, separate from both emoji acknowledgement and workflow orchestration |
| `gesture_debug_stream.py` | Bounded JSONL tick stream for HDMI gesture-refresh diagnostics, including latency stages, raw observations, ack-lane decisions, and publish outcomes |
| `claim_metadata.py` | Shared `confidence` / `source` / `requires_confirmation` helpers for multimodal runtime claims |
| `multimodal_initiative.py` | Conservative multimodal initiative readiness and display-first recommendation from camera + ReSpeaker facts |
| `runtime_contract.py` | Hard startup-blocker contract for XVF3800 DFU/safe-mode and disconnected states |
| `speaker_association.py` | Conservative association of current speech to the single primary visible person anchor |
| `sensitive_behavior_gate.py` | Conservative gate that blocks sensitive proactive behavior on ambiguous multi-person or low-confidence audio context |
| `governor_inputs.py` | Focused governor-facing packaging of current ReSpeaker presence/audio facts |
| `presence.py` | Presence-session state machine |
| `service.py` | Monitor orchestration, unreadable-capture blocking, lifecycle, and streaming-wakeword capture pause/resume around accepted hands-free turns |
| `component.yaml` | Structured package metadata |
| `AGENTS.md` | Local editing rules |

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
- [../governance/README.md](../governance/README.md)
- [../social/README.md](../social/README.md)
- [../wakeword/README.md](../wakeword/README.md)
