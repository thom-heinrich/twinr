# wakeword

Wakeword matching, openWakeWord spotting, optional local verifier assets,
verification policy, calibration, streaming, and offline evaluation helpers
for Twinr.

This package is the wakeword-specific boundary below proactive runtime
orchestration and above raw audio/hardware adapters.

## Responsibility

`wakeword` owns:
- Normalize wakeword phrases, transcripts, and detector labels
- Separate canonical wakeword phrases from STT-specific recognition phrases
- Run openWakeWord clip and frame spotting plus optional local verifier and STT confirmation
- Resolve bundled local openWakeWord model assets when the repo ships them
- Auto-load sibling `.verifier.pkl` assets for bundled local models when available
- Persist and apply per-device wakeword calibration overrides
- Replay labeled captures for evaluation, labeling, verifier training, and autotune workflows
- Train reproducible Twinr base models from generated multivoice datasets and select deployment thresholds against labeled acceptance captures
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
| `spotter.py` | openWakeWord clip and frame spotting plus optional local verifier loading |
| `policy.py` | Acceptance, fallback, and verification |
| `calibration.py` | Calibration profile persistence |
| `evaluation.py` | Labeling, manifest replay, verifier training, eval, and autotune |
| `training.py` | Offline base-model training, ONNX export, and threshold selection |
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

```bash
PYTHONPATH=src python3 -m twinr \
  --env-file .env \
  --wakeword-train-verifier \
  --wakeword-manifest /tmp/twinr_oww_capture_room_20260319a/captured_manifest.json
```

```bash
PYTHONPATH=src python3 -m twinr \
  --env-file .env \
  --wakeword-train-model \
  --wakeword-dataset-root /tmp/twinr_family_v3 \
  --wakeword-model-output src/twinr/proactive/wakeword/models/twinr_v2.onnx \
  --wakeword-manifest /tmp/twinr_oww_capture_room_20260319a/captured_manifest.json
```

```bash
PYTHONPATH=src python3 scripts/generate_multivoice_dataset.py \
  --model-name twinr_strict_fpmined_v1 \
  --phrase-profile strict_twinr \
  --generator-root /path/to/piper-sample-generator \
  --generator-model /path/to/piper-sample-generator/models/en_US-libritts_r-medium.pt \
  --extra-positive-dir /tmp/twinr_oww_capture_room_20260319a/positive \
  --exclude-manifest /tmp/twinr_critical16_v2.json \
  --hard-negative-manifest /tmp/twinr_oww_capture_room_20260319a/captured_manifest.json \
  --hard-negative-model src/twinr/proactive/wakeword/models/twinr_v1.onnx
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [models/README.md](./models/README.md)
- [../runtime/README.md](../runtime/README.md)
- [../governance/README.md](../governance/README.md)
