# workflows

`workflows` owns Twinr's runtime loop orchestration for the classic,
realtime, and streaming hardware paths. It also contains the workflow-local
helpers that keep speech output, print delivery, and background work bounded.

## Responsibility

`workflows` owns:
- orchestrate button, wakeword, and proactive entry points for live loops
- coordinate conversation turns, print delivery, and streamed speech output
- keep GPIO polling responsive while long turns are active by dispatching button presses off the poll thread and interrupting active turns on a second green press
- keep required remote-memory checks off the GPIO polling thread while still failing closed when remote memory becomes unavailable
- keep streamed TTS abortable even before the first chunk arrives so a stalled provider request does not pin the runtime in `answering`
- only surface `answering` once spoken audio has actually started instead of when text is merely queued
- recover suspicious or empty streaming transcripts with one bounded full-audio STT retry before surfacing a failed turn
- derive dual-lane bridge speech from the fast supervisor decision or deterministic fallback instead of starting a competing first-word model call during search/tool turns
- emit a forensic run pack for live-runtime debugging when `TWINR_WORKFLOW_TRACE_ENABLED=1`
- share workflow-local helpers for feedback tones, reference images, and safe background delivery
- expose compatibility workflow imports for the top-level package

`workflows` does **not** own:
- runtime-state, memory, reminder, or automation store implementations
- hardware driver internals for audio, printer, camera, buttons, or display
- provider transport adapters, prompt text, or tool schema definitions
- web dashboard logic or Pi/bootstrap scripts

## Key files

| File | Purpose |
|---|---|
| [runner.py](./runner.py) | Classic press-to-talk loop |
| [realtime_runner.py](./realtime_runner.py) | Realtime session loop |
| [streaming_runner.py](./streaming_runner.py) | Streaming dual-lane loop |
| [streaming_turn_orchestrator.py](./streaming_turn_orchestrator.py) | Parallel bridge/final lane watchdog orchestration |
| [realtime_runner_background.py](./realtime_runner_background.py) | Background delivery helpers |
| [realtime_runner_support.py](./realtime_runner_support.py) | Shared emit/media/config helpers |
| [realtime_runner_tools.py](./realtime_runner_tools.py) | Tool delegate mixin |
| [button_dispatch.py](./button_dispatch.py) | Non-blocking button dispatch and busy-turn interruption |
| [required_remote_watch.py](./required_remote_watch.py) | Background required-remote readiness watch for fail-closed runtimes |
| [speech_output.py](./speech_output.py) | Interruptible streamed TTS |
| [forensics.py](./forensics.py) | Queue-based forensic runpack tracing for live workflow bugs |
| [print_lane.py](./print_lane.py) | Background print lane |
| [working_feedback.py](./working_feedback.py) | Bounded tone feedback |
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
from twinr.agent.workflows import TwinrHardwareLoop, TwinrRealtimeHardwareLoop

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
