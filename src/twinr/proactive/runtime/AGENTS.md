# AGENTS.md — /src/twinr/proactive/runtime

## Scope

This directory owns proactive runtime orchestration: presence-session control,
monitor tick coordination, wakeword arming, degraded-mode sensor handling, and
bounded monitor lifecycle. Structural metadata lives in
[component.yaml](./component.yaml).

Out of scope:
- social trigger scoring, review prompt content, or wakeword matching internals
- raw hardware adapters and low-level audio/camera/PIR implementations
- proactive delivery cooldown policy in `../governance/`
- top-level agent runtime loops above the monitor boundary

## Key files

- `ambiguous_room_guard.py` — fail-closed room-ambiguity guard for person-targeted runtime inference
- `identity_fusion.py` — bounded temporal/session identity fusion over voice, portrait, household-voice candidates, and visual-anchor history
- `portrait_match.py` — conservative runtime claim for local portrait-match observations and temporal identity evidence
- `known_user_hint.py` — conservative known-user hint from voice-profile state plus optional temporal identity-fusion evidence and clear room context
- `affect_proxy.py` — prompt-only affect proxy surface from coarse posture, attention, and quiet cues
- `continuous_attention.py` — bounded short-lived visible-person track matching, motion recency, audio-to-vision speaker targeting, and short-horizon target prediction for HDMI follow
- `attention_targeting.py` — bounded multimodal attention-target prioritization over camera anchor, speaker association, showing intent, and short-lived session focus memory
- `attention_debug_stream.py` — bounded continuous JSONL debug stream for HDMI attention refresh ticks, including outcome codes, stage timings, and current camera/target/cue state
- `display_attention.py` — conservative proactive producer that steers HDMI eye gaze toward the currently relevant visible person without overwriting foreign face cues, while keeping normal follow behavior horizontal-only, mirroring camera-space left/right into user-facing gaze, adding small near-center head turns before full side gaze commits, and renewing or briefly holding matching cues before they expire or disappear on short camera dropouts
- `display_gesture_emoji.py` — conservative producer that mirrors stabilized rising-edge user gestures into bounded HDMI emoji acknowledgements without touching the face channel or overwriting foreign emoji cues
- `gesture_ack_lane.py` — dedicated low-latency acknowledgement stabilizer for the explicit user-facing gesture symbol set, separate from general camera-surface state
- `gesture_debug_stream.py` — bounded continuous JSONL debug stream for HDMI gesture refresh ticks, including raw observations, ack decisions, publish outcomes, and stage timings
- `audio_perception.py` — runtime-faithful one-shot ReSpeaker perception diagnostics and conservative room/device-directedness guard summaries for operator checks and recovery smokes
- `pir_open_gate.py` — bounded startup gate that retries only exact transient PIR GPIO busy overlaps during runtime handover
- `respeaker_capture_gate.py` — bounded startup gate that requires consecutive readable-frame probes before ReSpeaker capture is considered stable
- `claim_metadata.py` — shared `confidence` / `source` / `requires_confirmation` helpers for runtime claims
- `speaker_association.py` — conservative speaker-to-camera-anchor association for the single-primary-person case
- `multimodal_initiative.py` — confidence-bearing display-first/skip gate for later proactive behavior
- `presence.py` — source of truth for wakeword presence-session arming
- `service.py` — monitor orchestration, degraded-mode handling, and lifecycle
- `__init__.py` — package export surface; treat changes as API-impacting
- `component.yaml` — structured metadata, callers, and tests

## Invariants

- `PresenceSessionController.observe()` must preserve a monotonic internal timeline even when callers pass regressing or malformed timestamps.
- Wakeword arming must fail closed on malformed sensor flags or unavailable dependencies.
- `service.py` stays orchestration-focused; trigger scoring and wakeword matching belong in sibling packages.
- `audio_perception.py` must stay a bounded diagnostic/guard surface that reuses existing normalized observation and policy helpers; it must not grow its own parallel monitor orchestration.
- `service.py` must pause the active streaming wakeword capture before handing an accepted streaming wakeword to an exclusive conversation-recording path on the same ALSA device, and reopen it after the handler returns; otherwise wakeword success degenerates into immediate `Device or resource busy` capture failures.
- `service.py` must not clear a targeted ReSpeaker startup blocker after a single transient readable-frame probe; startup needs sustained bounded capture success so unstable USB enumeration does not flap the runtime between `ok` and `error`.
- `service.py` must tolerate only short exact `EBUSY` handover races when opening the PIR monitor; permanent GPIO contention or non-busy faults stay fail-closed and must not be downgraded.
- `audio_policy.py` must treat classified non-speech/background-media room activity as incompatible with calm `presence_audio_active`; do not let uncorroborated XVF3800 speech flags alone re-elevate media/noise into device-directed presence speech.
- ReSpeaker ambient classification must not depend solely on the speech-threshold chunk gate for non-speech detection; bounded low-dynamic RMS activity may still be media/noise even when `active_ratio` stays at zero.
- ReSpeaker ambient classification may use local PCM-content evidence to veto strong XVF3800 speech false positives, but ambiguous PCM windows must stay fail-closed and allow upstream host-control to remain decisive.
- ReSpeaker ambient classification must not discard strong PCM non-speech evidence just because the legacy chunk/RMS activity gate returns false on a bounded playback window.
- The operator-facing room-context guard should prefer already-classified non-speech/background-media over weak overlap hints; ambiguous low-confidence XVF3800 overlap alone must not relabel playback noise as human room speech.
- `ambiguous_room_guard`, `identity_fusion`, `portrait_match`, `known_user_hint`, and `affect_proxy` must remain conservative claim surfaces, not direct product decisions.
- `known_user_hint` must stay weaker than identity even when temporal portrait evidence is strong, and may only clear calm personalization; sensitive behavior still needs confirmation or stronger gates.
- `identity_fusion` may use presence-session memory, enrolled household-voice candidates, and visual-anchor history, but it must stay confirm-first and fail closed on ambiguity or modality conflict.
- `affect_proxy` must never emit emotion, wellbeing, or diagnosis claims; it may only support prompt-only follow-up cues.
- `continuous_attention.py` may keep only short-lived anonymous visible-person tracks. It may prioritize the active visible speaker when audio direction is stable, otherwise the most recently moving visible person, and it must preserve unmatched recent tracks across brief detector dropouts instead of collapsing a real multi-person scene immediately.
- `attention_targeting.py` may prioritize the latest visible speaker, showing-intent person, or short-lived session focus, but it must not let generic showing-intent override speaker or last-motion targets in a real multi-person scene. Ordinary person-following may use bounded vertical eye/head drift from live person height, but should not exaggerate that into theatrical "thinking" poses.
- `attention_targeting.py` and `display_attention.py` must fail closed when camera health is explicitly bad and no visible person exists; stale speaker/session focus may not keep steering the face after camera loss.
- `display_attention.py` may only publish its own `proactive_attention_follow` face cues and must fail closed when another display producer currently owns the cue store.
- `display_gesture_emoji.py` may only acknowledge stabilized rising-edge camera gesture events, must stay bounded and short-lived, must fail closed when another producer currently owns the emoji surface, and must prefer motion-bearing coarse gestures like `wave` over a simultaneous generic `open_palm` hand-shape cue.
- Matching `display_attention.py` cues must refresh their TTL before expiry so a stable visible-person target does not fall back to center between local attention-refresh ticks, and brief center jitter or short no-person/no-anchor blips must not immediately erase a recent local cue. Near-center responsiveness should come from bounded spatial head/gaze tuning, not from reintroducing jittery up/down or random idle eye motion.
- The fast HDMI attention-refresh path must not block on full ambient PCM sampling windows. When ReSpeaker host-control already provides speech/direction hints, use a signal-only snapshot or recent cached audio instead of serializing gaze/gesture refresh behind slow audio reads.
- The fast HDMI attention-refresh path must keep its default cadence genuinely sub-second on Pi HDMI builds; a half-second-plus default render loop is too slow for gaze-follow or gesture acknowledgement.
- HDMI eye-follow and HDMI gesture acknowledgement must stay decoupled. Eye-follow should prefer a cheap attention-only local camera observation and its own stabilized camera-surface state, while gesture acknowledgement may use the heavier full gesture path without clearing, delaying, or otherwise regressing face-follow behavior.
- HDMI gesture acknowledgement should prefer its own dedicated live-gesture observation path and ack-lane state. Do not route user-facing symbol latency through the heavier social camera-surface path when the dedicated gesture refresh is available.
- `service.py` should keep a bounded changed-only ops trace for HDMI attention-follow decisions so Pi debugging can prove whether failures come from camera health, stabilized camera semantics, target selection, or cue publishing.
- `service.py` should also keep a bounded continuous attention debug stream for each HDMI refresh tick so transient 1-3 minute eye-follow dropouts still leave first-run evidence after the fact.
- `service.py` should also keep a bounded continuous gesture debug stream for each HDMI gesture refresh tick so Pi-side symbol latency and missed acknowledgements can be diagnosed from first-run evidence instead of blind retuning.
- The fast HDMI attention-refresh path must stay local-camera-only, bounded, and separate from full proactive trigger evaluation; do not turn it into an always-on generic vision loop.
- The fast HDMI attention-refresh path may stay active while the main runtime is in `error`, but only for local face-cue following; it must not be used to re-enable agent turns, remote-memory work, or other degraded runtime behavior.
- Speaker association and multimodal initiative must fail closed on multi-person or low-confidence room context; do not let weak audio hints force spoken proactivity.
- `ProactiveMonitorService` lifecycle must remain idempotent, bounded, and safe under partial startup or shutdown failure.
- `build_default_proactive_monitor()` must not start an inert monitor when no operational sensor path exists.

## Verification

After any edit in this directory, run:

```bash
python3 -m compileall src/twinr/proactive/runtime
PYTHONPATH=src pytest test/test_proactive_monitor.py -q
```

If workflow wiring or export surface changed, also run:

```bash
PYTHONPATH=src pytest test/test_runner.py test/test_realtime_runner.py -q
```

## Coupling

`presence.py` changes -> also check:
- `service.py`
- `src/twinr/proactive/wakeword/stream.py`
- `test/test_proactive_monitor.py`

`service.py` changes -> also check:
- `src/twinr/proactive/social/`
- `src/twinr/proactive/wakeword/`
- `src/twinr/agent/workflows/runner.py`
- `src/twinr/agent/workflows/realtime_runner.py`
- `test/test_proactive_monitor.py`

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

- Keep wakeword capture writes under validated artifacts-root subpaths; do not widen path or symlink handling.
- Do not bypass the privacy gate that disables camera-triggered proactive behavior when `proactive_enabled` is false.
- Keep emit and ops-event paths non-throwing so sensor and worker faults remain recoverable.

## Output expectations

- Update docstrings when coordinator entry points, lifecycle semantics, or exports change.
- Keep [README.md](./README.md), [AGENTS.md](./AGENTS.md), and [component.yaml](./component.yaml) aligned when file roles or verification commands change.
- Treat export changes in `src/twinr/proactive/__init__.py` as follow-up API work.
