# workflows

`workflows` owns Twinr's active runtime loop orchestration for the realtime and
streaming hardware paths. It also contains the workflow-local helpers that keep
capture, speech output, print delivery, and background work bounded, including
the edge-side audio bridge for the new server-backed voice orchestrator path.

## Responsibility

`workflows` owns:
- orchestrate button, voice activation, configured visual gesture-wakeup, and proactive entry points for live loops
- coordinate conversation turns, print delivery, and streamed speech output
- cache and replay one normalized activation-ack payload so the Pi voice-activation `Ja?` stays audible without changing broader speech orchestration
- start one bounded managed smart-home sensor worker so external motion, button, alarm, and device-health events become normal Twinr sensor observations
- keep the yellow print button latency-safe by queuing prints from local short-term context only instead of synchronously rebuilding remote provider context before the print lane starts
- route beep, feedback, and spoken playback through one priority-aware playback coordinator instead of scattered per-call locks
- play one bounded startup boot clip from `media/boot.mp3` through the same playback coordinator, trimmed from the calmer middle of the asset and faded so loop startup stays gentle and interruptible
- render the first `0.8 s` of `media/dragon-studio-computer-startup-sound-effect-312870.mp3` as the default processing cue, prewarm that clip during loop startup/reload, and let a short tone fallback upgrade into the cached media path once the render finishes so THINKING does not stay on the synthetic hum for a whole turn
- keep fatal Pi loop/bootstrap failures on one stable error screen by holding the runtime in `error` and refreshing its snapshot instead of crashing back to the desktop and letting the supervisor ping-pong
- execute each completed streaming transcript turn under one authoritative coordinator/state-machine owner for deadlines, speech lifecycle, cancellation, and completion
- forward the completed tool-call history of a turn into runtime-side personality learning once the authoritative answer has been finalized, but keep that learning work off the user-facing audio-completion path so a slow background flush cannot leave Twinr visibly `speaking` after playback ends
- keep GPIO polling responsive while long turns are active by dispatching button presses off the poll thread and interrupting active turns on a second green press
- keep required remote-memory checks off the GPIO polling thread by gating live loops from the external remote-memory watchdog artifact while still failing closed when remote memory becomes unavailable
- treat fresh external-watchdog heartbeats plus the observed recent watchdog sample cadence as a bounded bridge between healthy steady-state deep probes so the runtime does not false-fail `required_remote` while the watchdog is intentionally idling or persisting heartbeats on a slower Pi-safe cadence
- resynchronize runtime-local remote-memory cooldown state after a successful external watchdog attestation so foreground turns do not fail against a stale in-process circuit breaker while the Pi watchdog already proved the namespace healthy
- keep streamed TTS abortable even before the first chunk arrives so a stalled provider request does not pin the runtime in `answering`
- start the processing cue immediately when a submitted transcript moves the runtime into `processing`, including dual-lane bridge/direct-reply turns, so visible THINKING phases do not sit silent while the lane planner or fast bridge path is still working
- keep coordinator-backed streamed TTS from preempting the processing cue until the first real speech chunk is ready, so long model/TTS startup gaps do not create dead air after only one or two thinking loops
- stop processing/search feedback owners before waiting on streamed speech-output close, so feedback cannot keep reacquiring the speaker while the final reply is draining
- only surface `answering` once spoken audio has actually started instead of when text is merely queued
- separate "wait for audible playback to drain" from the final worker-thread shutdown budget so post-playback cleanup cannot strand the turn in `answering` after the user already heard the full reply
- keep the runtime snapshot fresh during long spoken playback so the Pi supervisor does not false-kill an active answer as stale
- route non-safety proactive prompts through a dedicated delivery policy that can choose speech or display-first based on quiet hours, media/noise suppression, and recent ignored or interrupted prompts, with display-first prompts landing in the right-hand reserve lane instead of taking over the fullscreen presentation surface
- gate spoken/display background delivery starts behind one shared idle-transition window so reminders, automations, and proactive prompts cannot cross from stale `waiting` snapshots into live conversation output
- let the idle housekeeping worker trigger one bounded overnight orchestration pass that flushes long-term writers, refreshes world intelligence, prepares calm morning digest artifacts, and still prepares the next reserve-lane day plan before the user wakes up
- package the latest ReSpeaker presence/audio policy facts into explicit governor-input context so proactive reservations carry the same channel/session/runtime view that chose display-first or speech
- rearm spoken follow-up turns directly from `answering` back to `listening` so the display and operator cues do not briefly fall through `waiting` between a reply and the reopened microphone window
- reuse that already-open `listening` state when a same-stream remote follow-up transcript is committed as a seeded text turn, instead of reopening listening and surfacing a false runtime `error`
- inject one bounded carryover hint into explicit follow-up turns so short repairs like "I meant the time" continue the just-established thread instead of dropping back to a generic local answer
- emit privacy-safe accountability events for committed follow-up transcripts, carryover-hint build/store decisions, and provider-context carryover injection so live wrong-answer turns can be reconstructed after the fact without logging raw speech
- start the post-response closure guard while streamed speech is still draining so follow-up beeps do not sit behind a second model wait after the audible answer ends
- keep the post-playback join against the closure guard bounded so a stalled steering, remote-memory, or closure-provider path cannot leave the runtime stuck in `answering` after speech has ended
- stamp live voice turns with one shared `wake -> commit -> supervisor -> remote-memory -> tts` latency breakdown so Pi journals show exactly which stage consumed the turn budget
- translate structured personality turn-steering state into real follow-up runtime decisions so shared-thread topics can stay gently open while cooling topics are answered briefly and then released
- build per-turn fast-lane supervisor instructions from stable routing policy plus the currently visible reserve-lane card so display-grounded turns can route on the actual shown topic without bloating the loop classes
- keep turn-controller context selection, label-aware guidance, and transcript-verifier gate policy in dedicated runtime components instead of inlining them into the active loop classes
- recover suspicious or empty streaming transcripts with one bounded full-audio STT retry before surfacing a failed turn
- wire the optional OpenAI streaming-transcript verifier from the provider bundle into the live streaming loop so suspicious short Deepgram turns, including empty results after a late speech start, are rechecked against the real captured audio before Twinr drops the turn
- feed transcript-first interim and endpoint transcripts into long-term-memory materialized provider-answer-front prewarm so the live answer path can consume already-persisted long-term prompt blocks instead of waiting for a fresh synchronous remote rebuild after the user stops speaking
- let a calibrated local semantic transcript router front-run high-confidence `web`, `memory`, and `tool` streaming turns while keeping `parametric` answers on the supervisor lane until offline eval says otherwise, including the optional two-stage user-intent-plus-backend path
- when the local semantic router is enabled on the Pi, prewarm its ONNX path before the first live turn so weather/tool/memory turns do not spend the user's first reply budget on cold-start model load
- when the local semantic router front-runs an authoritative `web`, `memory`, or `tool` turn, generate the instant bridge line through the first-word LLM with a route-aware filler overlay and leave the bridge silent when that fast LLM path does not return usable text
- derive dual-lane bridge speech from the fast supervisor decision as the authoritative first spoken lane whenever a supervisor decision provider is available; use the standalone first-word model only as a fallback when that supervisor lane does not exist, and do not fall back to canned watchdog speech
- when the bridge and final lane both depend on the fast supervisor decision, reuse one shared speculative supervisor-decision worker; do not open parallel duplicate structured-decision calls for the same transcript, and if that shared worker misses its reuse budget, fall straight through to the generic supervisor loop instead of starting another structured-decision roundtrip
- downgrade fast-lane decisions that declare `full_context` needs into a filler-plus-final-lane handoff so conversation-recall turns do not get answered from a memory-blind bridge lane
- do not emit a separate supervisor bridge/filler for one-shot runtime-local handoffs that already carry `runtime_tool_name`; the direct runtime-local tool reply must be the only spoken confirmation
- treat an incomplete one-shot runtime-local shortcut payload as a normal specialist handoff instead of executing it directly; this keeps ambiguous quiet-mode requests on the ask-a-duration path instead of throwing the live turn into `Thinking -> Error`
- route resolved or prefetched dual-lane handoffs straight into the specialist path, using full tool-provider context for memory/general work while keeping pure search handoffs on the bounded search-only context
- route `tiny_recent` runtime-local tool handoffs onto the bounded compact tool context instead of the heavy remote-backed tool-provider context, so status/control turns do not stall the final lane on synchronous long-term retrieval
- keep authoritative local `memory` routes on the fast bridge while preserving the full-context handoff, so recall turns still exercise the real long-term-memory path instead of silently downgrading to a recent-only shortcut
- give tool/search handoffs their own wider final-lane timeout budget so live specialist turns are not cut off by the shorter direct-reply watchdog envelope
- only prefetch first-word speech once a partial transcript has enough shape to be meaningful; one dangling tail word must not trigger a filler line on its own
- keep dual-lane search turns to one bounded final-lane search execution instead of launching a speculative background search worker that can outlive the turn
- wait briefly for active filler playback to drain before replacing it with the final lane so the fast acknowledgement is not cut off mid-sentence
- move the runtime back from `answering` to `processing` once a spoken bridge acknowledgement has finished but the final lane is still running, so status and feedback reflect real waiting/search work instead of stale speech
- fail closed for dual-lane final-lane timeout/error and specialist handoff failure paths instead of speaking a synthetic recovery sentence that was never backed by the real search/tool result
- capture bounded final-lane thread snapshots in workflow forensics when a spoken bridge has already finished but the background final lane is still running or has crossed its watchdog/timeout thresholds, so live `Thinking -> Error` regressions can be reduced to one concrete blocked callsite
- emit bounded pre-speech capture diagnostics on listen timeouts so Pi no-speech failures can be proven from first-run logs instead of guessed
- emit a forensic run pack for live-runtime debugging when `TWINR_WORKFLOW_TRACE_ENABLED=1`
- include redacted transcript, context-selection, and final-lane answer provenance in that forensic run pack so semantic answer failures can be proven from one Pi run
- stream bounded edge audio and runtime-state updates to the remote voice orchestrator while keeping the realtime loop as the single owner of physical runtime state changes
- wait briefly for proven transient XVF3800 re-enumeration on the shared live-capture paths so a short USB drop does not instantly flip a listen turn into `error`
- reconnect the edge voice websocket after transient server or network closures and replay the last known runtime state so hands-free wake detection does not stay dead until the Pi service restarts
- keep the live voice gateway server-only after wake: once the Pi has opened a voice turn on thh1986, the same remote stream must stay authoritative for wake, transcript commit, continuation, and follow-up closure instead of pausing capture and reopening a second Pi-local listen phase
- keep answer-time interruption server-owned whenever the live voice gateway is active; do not reopen a second local Pi STT watcher while the remote thh1986 stream already owns barge-in
- require `wss://` for non-loopback voice-gateway websockets by default, and use `TWINR_VOICE_ORCHESTRATOR_ALLOW_INSECURE_WS=true` only for the explicitly attested current LAN bridge until a TLS terminator exists
- keep the Pi-owned household voice identity source local: sync only bounded read-only speaker embeddings to the live gateway for wake-time familiar-speaker bias, and keep any passive profile updates on the Pi runtime side
- replay the runtime-owned temporary voice-quiet window into the live voice orchestrator and suppress automatic follow-up reopening while that bounded quiet window is active
- fail closed on the Pi as well when a late or stale server-side wake confirmation arrives during that runtime-owned voice-quiet window, and immediately re-attest `waiting` plus the active quiet deadline back into the live gateway
- re-check remote follow-up eligibility after streamed tool side effects such as temporary voice quiet mode, so the streaming voice path never re-arms `follow_up_open` from a stale pre-turn snapshot once the runtime has decided to stay quiet
- treat required remote-memory failures in the streaming final lane as fatal runtime blockers instead of masking them behind dual-lane fallback speech
- share workflow-local helpers for feedback tones, reference images, and safe background delivery
- expose the active workflow loop entry points without eager imports that can create runtime/ops import cycles

`workflows` does **not** own:
- runtime-state, memory, reminder, or automation store implementations
- hardware driver internals for audio, printer, camera, buttons, or display
- provider transport adapters, prompt text, or tool schema definitions
- web dashboard logic or Pi/bootstrap scripts
- websocket transport contracts or server-side wake/follow-up decision policy; those stay in `src/twinr/orchestrator`

## Key files

| File | Purpose |
|---|---|
| [realtime_runner.py](./realtime_runner.py) | Compatibility wrapper that preserves the public realtime-loop import path and legacy module patch points |
| [realtime_runner_impl/bootstrap.py](./realtime_runner_impl/bootstrap.py) | Realtime-loop bootstrap, collaborator wiring, tool-surface refresh, and managed sensor-worker setup |
| [realtime_runner_impl/session.py](./realtime_runner_impl/session.py) | Run-loop orchestration, wake triggers, conversation-session lifecycle, and remote voice-orchestrator coordination |
| [realtime_runner_impl/turn_capture.py](./realtime_runner_impl/turn_capture.py) | Turn-controller streaming capture, transcript recovery, and early-snapshot coercion |
| [realtime_runner_impl/turn_execution.py](./realtime_runner_impl/turn_execution.py) | Single-turn execution, streamed playback/interrupt handling, activation ack caching, and print-lane enqueue helpers |
| [realtime_follow_up.py](./realtime_follow_up.py) | Focused follow-up reopening and closure-decision helper used by the realtime loop |
| [voice_orchestrator.py](./voice_orchestrator.py) | Edge-side voice websocket bridge that streams bounded audio and runtime-state updates to the orchestrator server, including bounded transient XVF3800 capture recovery, first-frame stall healing via host-control reboot, and startup-time reconnect recovery when the gateway appears late |
| [voice_orchestrator_runtime.py](./voice_orchestrator_runtime.py) | Runtime-side voice-orchestrator state replay, same-stream transcript handoff, and remote follow-up helper |
| [voice_turn_latency.py](./voice_turn_latency.py) | Shared wake/commit/supervisor/remote-memory/TTS stage tracker for per-turn Pi journal timing breakdowns |
| [voice_identity_runtime.py](./voice_identity_runtime.py) | Runtime-side household voice profile sync plus conservative passive voice-profile updates for the live gateway path |
| [remote_transcript_commit.py](./remote_transcript_commit.py) | Bounded wait coordinator for same-stream server transcript commits during live remote listening |
| [streaming_runner.py](./streaming_runner.py) | Streaming loop entrypoint and orchestration shell, including transcript-first long-term provider-answer-front prewarm delegation |
| [follow_up_steering.py](./follow_up_steering.py) | Runtime bridge from personality steering cues into follow-up reopening decisions, including storage of the next-turn carryover hint |
| [runtime_error_hold.py](./runtime_error_hold.py) | Stable runtime-error hold that keeps the display companion alive and the snapshot fresh after fatal loop failures |
| [turn_guidance.py](./turn_guidance.py) | Bounded turn-controller context and label-aware conversation guidance |
| [streaming_transcript_verifier.py](./streaming_transcript_verifier.py) | Streaming transcript recovery plus explicit verifier KPI gates |
| [streaming_capture.py](./streaming_capture.py) | Streaming microphone capture, same-stream remote transcript waits, transcript-first long-term provider-answer-front prewarm triggers, timeout handling, and batch-STT fallback |
| [streaming_speculation.py](./streaming_speculation.py) | Speculative first-word and supervisor warmup controller |
| [streaming_lane_planner.py](./streaming_lane_planner.py) | Streaming lane-plan and final-lane path selection |
| [streaming_supervisor_context.py](./streaming_supervisor_context.py) | Per-turn fast-lane supervisor instruction builder, including visible-card grounding overlays |
| [streaming_semantic_router.py](./streaming_semantic_router.py) | Workflow bridge from local semantic-router bundles, including optional two-stage user-intent routing, into supervisor-style handoffs |
| [streaming_turn_coordinator.py](./streaming_turn_coordinator.py) | Authoritative streaming turn state machine and completion coordinator |
| [streaming_turn_orchestrator.py](./streaming_turn_orchestrator.py) | Low-level parallel bridge/final lane watchdog executor used by the coordinator |
| [playback_coordinator.py](./playback_coordinator.py) | Single-owner speaker queue with priority-aware, request-bound preemption for beep, feedback, and TTS |
| [respeaker_duplex_keepalive.py](./respeaker_duplex_keepalive.py) | ReSpeaker XVF3800 duplex-stability helper that keeps one Twinr-owned playback stream alive; on the productive softvol path it uses the direct `/dev/zero` ALSA guard instead of silence piped through `WaveAudioPlayer`, suspends that direct guard around real foreground playback, and waits through a short quiet gap before reclaiming the device so back-to-back processing and TTS playback do not race |
| [rendered_audio_clip.py](./rendered_audio_clip.py) | Bounded media-clip renderer/cache for reusable workflow WAV payloads |
| [startup_boot_sound.py](./startup_boot_sound.py) | Bounded startup earcon helper that renders a faded `media/boot.mp3` clip and queues it through the shared playback coordinator |
| [realtime_runtime/background.py](./realtime_runtime/background.py) | Active background delivery helpers used by the realtime loop, including idle-transition-safe reminder/automation/proactive delivery, routing display-first proactive prompts into the reserve lane, and triggering the explicit overnight orchestration hook |
| [realtime_runtime/nightly.py](./realtime_runtime/nightly.py) | Explicit overnight orchestration for long-term flush/refresh, prepared morning digests, and durable nightly run-state artifacts |
| [realtime_runtime/background_delivery.py](./realtime_runtime/background_delivery.py) | Shared idle-window gate that prevents background delivery from crossing from stale `waiting` state into an active conversation |
| [realtime_runtime/reminder_delivery.py](./realtime_runtime/reminder_delivery.py) | Focused reminder phrasing and speech-delivery helper with deterministic local fallback text |
| [realtime_runtime/proactive_delivery.py](./realtime_runtime/proactive_delivery.py) | Display-first versus speech delivery policy for proactive background prompts |
| [realtime_runtime/support.py](./realtime_runtime/support.py) | Active emit/media/config helpers used by the realtime loop |
| [realtime_runtime/required_remote_support.py](./realtime_runtime/required_remote_support.py) | Focused required-remote fail-closed gating, watchdog attestation, and restoration helper for live loops |
| [realtime_runtime/vision_support.py](./realtime_runtime/vision_support.py) | Safe reference-image loading, live camera image bundling, and bounded vision-prompt construction |
| [realtime_runner_tools.py](./realtime_runner_tools.py) | Tool delegate mixin, including smart-home and RSS/world-intelligence wiring |
| [button_dispatch.py](./button_dispatch.py) | Non-blocking button dispatch and busy-turn interruption |
| [required_remote_watch.py](./required_remote_watch.py) | Background required-remote readiness watch for fail-closed runtimes |
| [required_remote_snapshot.py](./required_remote_snapshot.py) | Cheap external-watchdog snapshot evaluation for live runtime gating, including bounded heartbeat bridging keyed to healthy recent steady-state probe cadence, Pi-safe heartbeat quiet windows, and bounded recovery of a dead external watchdog owner |
| [speech_output.py](./speech_output.py) | Interruptible streamed TTS |
| [forensics.py](./forensics.py) | Queue-based forensic runpack tracing for live workflow bugs, including bounded thread snapshots for stuck workflow helpers |
| [listen_timeout_diagnostics.py](./listen_timeout_diagnostics.py) | Shared bounded no-speech timeout diagnostics emission |
| [print_lane.py](./print_lane.py) | Background print lane |
| [working_feedback.py](./working_feedback.py) | Bounded processing, answering, and printing feedback loops with the dragon MP3 as the default think cue, synthesized fallback tones, asynchronous media prewarm, and coordinator-owned stop semantics |
| [working_feedback_tone.py](./working_feedback_tone.py) | Synthesized processing-tone helper used for fallback/defaultless paths and direct tests |
| [component.yaml](./component.yaml) | Structured package metadata |

Legacy compatibility shims that no longer belong to the active package are
archived under [`__legacy__/workflows/`](./__legacy__/workflows/).

The refactored realtime loop keeps the public class at
[`realtime_runner.py`](./realtime_runner.py), but the implementation now lives
under [`realtime_runner_impl/`](./realtime_runner_impl/) so bootstrap, session
orchestration, capture, and per-turn execution stay reviewable as separate
concerns.

## Forensic tracing

Set `TWINR_WORKFLOW_TRACE_ENABLED=1` to write a bounded workflow run pack under
`state/forensics/workflow/<run_id>/`. The pack contains:

- `run.jsonl` — structured workflow events
- `run.trace` — span-correlated trace records
- `run.metrics.json` — aggregated counts and slow spans
- `run.summary.json` — top messages, failure buckets, and slowest spans
- `run.repro/` — sanitized runtime and environment snapshot

If `TWINR_WORKFLOW_TRACE_DIR` still points at another Twinr checkout's absolute
`state/forensics/...` path, the tracer now re-bases that repo-owned tail onto
the active `project_root` before opening files. That keeps `/twinr` probes from
failing on stale leading-repo absolute paths copied through `.env`.

Before each new conversation session, the active loop also checks whether the
current run pack still has enough event budget left to capture a real spoken
turn. If the service-lifetime pack is exhausted or nearly exhausted, the loop
rotates to a fresh `WorkflowForensics` run and closes the old writer
asynchronously. That keeps late live turns from disappearing behind background
noise in a long-running Pi service.

The follow-up path now emits a focused set of privacy-safe accountability
records in those run packs:

- `voice_orchestrator_transcript_committed_payload`
- `conversation_follow_up_context_hint_built`
- `conversation_follow_up_context_hint_update`
- `provider_context_follow_up_carryover_injected`

Each record is redacted by default and carries only bounded structural details
such as source, presence, counts, and short hashes.

## Usage

```python
from twinr.agent.workflows import TwinrRealtimeHardwareLoop
from twinr.agent.workflows.streaming_runner import TwinrStreamingHardwareLoop

realtime_loop = TwinrRealtimeHardwareLoop(config=config)
streaming_loop = TwinrStreamingHardwareLoop(config=config)
realtime_loop.run(duration_s=15)
streaming_loop.run(duration_s=15)
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [runtime](../base_agent/runtime/README.md)
- [routing](../routing/README.md)
- [conversation](../base_agent/conversation/README.md)
- [agent tools runtime](../tools/runtime/README.md)

The remote/server voice path intentionally keeps a strict split:

- `voice_orchestrator.py` owns edge microphone streaming and server-event dispatch
- `voice_orchestrator_runtime.py` owns runtime-side state replay, intent-context refresh, and same-stream transcript handoff into the realtime loop
- `remote_transcript_commit.py` owns the bounded wait/commit handoff for same-stream remote transcripts
- `realtime_runner.py` and `streaming_capture.py` must not reopen a second Pi-local listen capture once the voice orchestrator path is active; they have to wait on the same remote stream instead
- `realtime_runner.py` must not start the local interrupt STT watcher once the voice orchestrator path is active; barge-in belongs to the same remote stream
- `src/twinr/orchestrator/voice_session.py` owns server-side wake, transcript-commit, follow-up, and barge-in decisions
