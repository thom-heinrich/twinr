# workflows

`workflows` owns Twinr's runtime loop orchestration for the classic,
realtime, and streaming hardware paths. It also contains the workflow-local
helpers that keep speech output, print delivery, and background work bounded.

## Responsibility

`workflows` owns:
- orchestrate button, wakeword, and proactive entry points for live loops
- coordinate conversation turns, print delivery, and streamed speech output
- recover suspicious or empty streaming transcripts with one bounded full-audio STT retry before surfacing a failed turn
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
| [realtime_runner_background.py](./realtime_runner_background.py) | Background delivery helpers |
| [realtime_runner_support.py](./realtime_runner_support.py) | Shared emit/media/config helpers |
| [realtime_runner_tools.py](./realtime_runner_tools.py) | Tool delegate mixin |
| [speech_output.py](./speech_output.py) | Interruptible streamed TTS |
| [print_lane.py](./print_lane.py) | Background print lane |
| [working_feedback.py](./working_feedback.py) | Bounded tone feedback |
| [component.yaml](./component.yaml) | Structured package metadata |

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
