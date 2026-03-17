# workflows

`workflows` owns Twinr's active runtime loop orchestration for the realtime and
streaming hardware paths plus the compatibility import surfaces that still
point older callers at legacy code. It also contains the workflow-local
helpers that keep capture, speech output, print delivery, and background work
bounded.

## Responsibility

`workflows` owns:
- orchestrate button, wakeword, and proactive entry points for live loops
- coordinate conversation turns, print delivery, and streamed speech output
- keep the yellow print button latency-safe by queuing prints from local short-term context only instead of synchronously rebuilding remote provider context before the print lane starts
- route beep, feedback, and spoken playback through one priority-aware playback coordinator instead of scattered per-call locks
- execute each completed streaming transcript turn under one authoritative coordinator/state-machine owner for deadlines, speech lifecycle, cancellation, and completion
- keep GPIO polling responsive while long turns are active by dispatching button presses off the poll thread and interrupting active turns on a second green press
- keep required remote-memory checks off the GPIO polling thread by gating live loops from the external remote-memory watchdog artifact while still failing closed when remote memory becomes unavailable
- keep streamed TTS abortable even before the first chunk arrives so a stalled provider request does not pin the runtime in `answering`
- only surface `answering` once spoken audio has actually started instead of when text is merely queued
- rearm spoken follow-up turns directly from `answering` back to `listening` so the display and operator cues do not briefly fall through `waiting` between a reply and the reopened microphone window
- start the post-response closure guard while streamed speech is still draining so follow-up beeps do not sit behind a second model wait after the audible answer ends
- recover suspicious or empty streaming transcripts with one bounded full-audio STT retry before surfacing a failed turn
- wire the optional OpenAI streaming-transcript verifier from the provider bundle into the live streaming loop so suspicious short Deepgram turns, including empty results after a late speech start, are rechecked against the real captured audio before Twinr drops the turn
- derive dual-lane bridge speech from the fast supervisor decision as the authoritative first spoken lane whenever a supervisor decision provider is available; use the standalone first-word model only as a fallback when that supervisor lane does not exist, and do not fall back to canned watchdog speech
- downgrade fast-lane decisions that declare `full_context` needs into a filler-plus-final-lane handoff so conversation-recall turns do not get answered from a memory-blind bridge lane
- only prefetch first-word speech once a partial transcript has enough shape to be meaningful; one dangling tail word must not trigger a filler line on its own
- keep dual-lane search turns to one bounded final-lane search execution instead of launching a speculative background search worker that can outlive the turn
- wait briefly for active filler playback to drain before replacing it with the final lane so the fast acknowledgement is not cut off mid-sentence
- emit bounded pre-speech capture diagnostics on listen timeouts so Pi no-speech failures can be proven from first-run logs instead of guessed
- emit a forensic run pack for live-runtime debugging when `TWINR_WORKFLOW_TRACE_ENABLED=1`
- share workflow-local helpers for feedback tones, reference images, and safe background delivery
- expose compatibility workflow imports for the top-level package without eager runner imports that can create runtime/ops import cycles

`workflows` does **not** own:
- runtime-state, memory, reminder, or automation store implementations
- hardware driver internals for audio, printer, camera, buttons, or display
- provider transport adapters, prompt text, or tool schema definitions
- web dashboard logic or Pi/bootstrap scripts

## Key files

| File | Purpose |
|---|---|
| [runner.py](./runner.py) | Compatibility shim to the legacy classic loop in `src/twinr/agent/legacy/classic_hardware_loop.py` |
| [realtime_runner.py](./realtime_runner.py) | Realtime session loop |
| [streaming_runner.py](./streaming_runner.py) | Streaming loop entrypoint and orchestration shell |
| [streaming_capture.py](./streaming_capture.py) | Streaming microphone capture, timeout handling, and batch-STT fallback |
| [streaming_speculation.py](./streaming_speculation.py) | Speculative first-word and supervisor warmup controller |
| [streaming_lane_planner.py](./streaming_lane_planner.py) | Streaming lane-plan and final-lane path selection |
| [streaming_turn_coordinator.py](./streaming_turn_coordinator.py) | Authoritative streaming turn state machine and completion coordinator |
| [streaming_turn_orchestrator.py](./streaming_turn_orchestrator.py) | Low-level parallel bridge/final lane watchdog executor used by the coordinator |
| [playback_coordinator.py](./playback_coordinator.py) | Single-owner speaker queue with priority-aware, request-bound preemption for beep, feedback, and TTS |
| [realtime_runtime/background.py](./realtime_runtime/background.py) | Active background delivery helpers used by the realtime loop |
| [realtime_runtime/support.py](./realtime_runtime/support.py) | Active emit/media/config helpers used by the realtime loop |
| [realtime_runner_background.py](./realtime_runner_background.py) | Compatibility shim for background helpers |
| [realtime_runner_support.py](./realtime_runner_support.py) | Compatibility shim for support helpers |
| [realtime_runner_tools.py](./realtime_runner_tools.py) | Tool delegate mixin |
| [button_dispatch.py](./button_dispatch.py) | Non-blocking button dispatch and busy-turn interruption |
| [required_remote_watch.py](./required_remote_watch.py) | Background required-remote readiness watch for fail-closed runtimes |
| [required_remote_snapshot.py](./required_remote_snapshot.py) | Cheap external-watchdog snapshot evaluation for live runtime gating |
| [speech_output.py](./speech_output.py) | Interruptible streamed TTS |
| [forensics.py](./forensics.py) | Queue-based forensic runpack tracing for live workflow bugs |
| [listen_timeout_diagnostics.py](./listen_timeout_diagnostics.py) | Shared bounded no-speech timeout diagnostics emission |
| [print_lane.py](./print_lane.py) | Background print lane |
| [working_feedback.py](./working_feedback.py) | Bounded tone feedback with coordinator-owned stop semantics so feedback shutdown cannot kill live TTS |
| [component.yaml](./component.yaml) | Structured package metadata |

## Forensic tracing

Set `TWINR_WORKFLOW_TRACE_ENABLED=1` to write a bounded workflow run pack under
`state/forensics/workflow/<run_id>/`. The pack contains:

- `run.jsonl` — structured workflow events
- `run.trace` — span-correlated trace records
- `run.metrics.json` — aggregated counts and slow spans
- `run.summary.json` — top messages, failure buckets, and slowest spans
- `run.repro/` — sanitized runtime and environment snapshot

## Usage

```python
from twinr.agent.legacy.classic_hardware_loop import TwinrHardwareLoop
from twinr.agent.workflows import TwinrRealtimeHardwareLoop

classic_loop = TwinrHardwareLoop(config=config)
realtime_loop = TwinrRealtimeHardwareLoop(config=config)

classic_loop.run(duration_s=15)
realtime_loop.run(duration_s=15)
```

```python
from twinr.agent.workflows.streaming_runner import TwinrStreamingHardwareLoop

streaming_loop = TwinrStreamingHardwareLoop(config=config)
streaming_loop.run(duration_s=15)
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [runtime](../base_agent/runtime/README.md)
- [conversation](../base_agent/conversation/README.md)
- [agent tools runtime](../tools/runtime/README.md)

The classic press-to-talk implementation itself now lives outside the active
workflow package under `src/twinr/agent/legacy/classic_hardware_loop.py`; the
`workflows.runner` module remains only as a compatibility import shim and
should not be used for new internal references.
