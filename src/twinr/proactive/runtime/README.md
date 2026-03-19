# runtime

Proactive monitor orchestration and wakeword presence-session control for Twinr.

This package wires the runtime-facing monitor loop, degraded sensor handling,
and presence-session state used to arm wakeword listening and proactive checks.

## Responsibility

`runtime` owns:
- Coordinate proactive monitor ticks across PIR, camera, ambient audio, and wakeword policy paths
- Maintain wakeword presence-session state from recent sensor activity
- Derive conservative ReSpeaker audio-policy hooks such as quiet windows, resume windows, and overlap guards
- Derive conservative ReSpeaker speech-delivery defer reasons such as background media or non-speech room activity
- Enforce the hard XVF3800 startup/runtime contract so DFU/safe mode and unreadable enumerated capture block proactive voice monitoring clearly instead of degrading silently
- Package current ReSpeaker presence/audio facts into explicit governor-input context for downstream proactive delivery paths
- Fuse the single primary camera anchor with ReSpeaker direction confidence into a conservative speaker-association fact
- Derive a bounded multimodal initiative state that can force display-first or skip later proactive work when room context is ambiguous
- Derive an explicit `ambiguous_room_guard` before person-targeted runtime inference proceeds
- Derive a conservative `known_user_hint` from fresh voice-profile state plus clear single-person room context
- Derive a prompt-only `affect_proxy` surface from coarse posture, attention, quiet, and motion cues without claiming emotion as fact
- Export normalized observation facts and ops telemetry from proactive monitoring
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
| `audio_policy.py` | Conservative ReSpeaker policy-hook, speech-defer, and runtime-alert derivation |
| `ambiguous_room_guard.py` | Fail-closed room-ambiguity guard for person-targeted runtime inferences |
| `known_user_hint.py` | Conservative known-user hint from voice-profile state plus clear room context |
| `affect_proxy.py` | Prompt-only affect proxy surface from coarse posture, attention, and quiet cues |
| `claim_metadata.py` | Shared `confidence` / `source` / `requires_confirmation` helpers for multimodal runtime claims |
| `multimodal_initiative.py` | Conservative multimodal initiative readiness and display-first recommendation from camera + ReSpeaker facts |
| `runtime_contract.py` | Hard startup-blocker contract for XVF3800 DFU/safe-mode states |
| `speaker_association.py` | Conservative association of current speech to the single primary visible person anchor |
| `sensitive_behavior_gate.py` | Conservative gate that blocks sensitive proactive behavior on ambiguous multi-person or low-confidence audio context |
| `governor_inputs.py` | Focused governor-facing packaging of current ReSpeaker presence/audio facts |
| `presence.py` | Presence-session state machine |
| `service.py` | Monitor orchestration, unreadable-capture blocking, and lifecycle |
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
