# wakeword

Wakeword matching, openWakeWord spotting, verification policy, calibration,
streaming, and offline evaluation helpers for Twinr.

This package is the wakeword-specific boundary below proactive runtime
orchestration and above raw audio/hardware adapters.

## Responsibility

`wakeword` owns:
- Normalize wakeword phrases, transcripts, and detector labels
- Run openWakeWord clip and frame spotting plus optional STT confirmation
- Persist and apply per-device wakeword calibration overrides
- Replay labeled captures for evaluation, labeling, and autotune workflows
- Stream bounded PCM frames into wakeword detections for proactive runtime

`wakeword` does **not** own:
- Presence-session arming or monitor orchestration
- Social trigger scoring or proactive governance policy
- Raw microphone capture implementations outside the stream monitor boundary
- Web settings rendering for wakeword configuration

## Key files

| File | Purpose |
|---|---|
| `__init__.py` | Package export surface |
| `matching.py` | Transcript and label matching |
| `spotter.py` | openWakeWord clip and frame spotting |
| `policy.py` | Acceptance, fallback, and verification |
| `calibration.py` | Calibration profile persistence |
| `evaluation.py` | Labeling, eval, and autotune |
| `stream.py` | Bounded live audio monitor |
| `component.yaml` | Structured package metadata |
| `AGENTS.md` | Local editing rules |

## Usage

```python
from twinr.proactive.wakeword import WakewordPhraseSpotter, WakewordDecisionPolicy

spotter = WakewordPhraseSpotter(
    backend=backend,
    phrases=config.wakeword_phrases,
    language=config.openai_realtime_language,
)
decision = WakewordDecisionPolicy(
    primary_backend=config.wakeword_primary_backend,
).decide(match=spotter.detect(capture), capture=capture, source="runtime")
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [../runtime/README.md](../runtime/README.md)
- [../governance/README.md](../governance/README.md)
