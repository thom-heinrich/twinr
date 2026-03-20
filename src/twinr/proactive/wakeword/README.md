# wakeword

Wakeword matching, openWakeWord spotting, optional local verifier assets,
sequence-aware cascade verification, policy, calibration, streaming, and
offline plus runtime-faithful evaluation helpers for Twinr.

This package is the wakeword-specific boundary below proactive runtime
orchestration and above raw audio/hardware adapters.

## Responsibility

`wakeword` owns:
- Normalize wakeword phrases, transcripts, and detector labels
- Separate canonical wakeword phrases from STT-specific recognition phrases
- Run openWakeWord clip and frame spotting plus optional local verifier and STT confirmation
- Run a Twinr-specific sequence-aware verifier as a second local stage on localized wakeword captures
- Resolve bundled local openWakeWord model assets when the repo ships them
- Auto-load sibling `.verifier.pkl` assets for bundled local models when available
- Auto-load sibling `.sequence_verifier.pkl` assets for bundled local models when available
- Persist and apply per-device wakeword calibration overrides
- Replay labeled captures for evaluation, labeling, verifier training, and autotune workflows
- Replay labeled captures through the streaming runtime path for promotion suites and ambient false-accepts/hour guards
- Train reproducible Twinr base models from generated multivoice datasets and select deployment thresholds against labeled acceptance captures
- Stream bounded PCM frames into wakeword detections for proactive runtime

`wakeword` does **not** own:
- Presence-session arming or monitor orchestration
- Social trigger scoring or proactive governance policy
- Raw microphone capture implementations outside the stream monitor boundary
- Web settings rendering for wakeword configuration

## Sequence verifier notes

- Sequence-verifier training manifests may omit `family_key`; when they do, `cascade.py` now derives a canonical family from the full spoken phrase instead of the trailing token.
- Positive phonetic families such as `Twinna`, `Twina`, and `Twinner` are weighted explicitly against the main confusion families `Twin`, `Winner`, `Winter`, `Tina`, `Timer`, and `Twitter`.
- This keeps family-weighted retrains stable for room captures like `Twinna wie ist das Wetter heute`, which should stay in the `twinna` family rather than collapsing to the last word.

## Key files

| File | Purpose |
|---|---|
| `__init__.py` | Package export surface |
| `cascade.py` | Twinr-specific second-stage DTW-aligned sequence verifier assets and runtime gate |
| `matching.py` | Transcript and label matching |
| `spotter.py` | openWakeWord clip and frame spotting plus optional local verifier loading |
| `policy.py` | Acceptance, fallback, local cascade gating, and STT verification |
| `promotion.py` | Runtime-faithful stream replay, promotion specs, suite guards, and ambient false-accepts/hour reports |
| `calibration.py` | Calibration profile persistence |
| `evaluation.py` | Labeling, manifest replay, verifier training, eval, and autotune |
| `training.py` | Offline base-model training, ONNX export, and threshold selection |
| `training_plan.py` | Canonical Stage-1 family, hard-negative mining, and Pi acceptance plan rendering |
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
  --wakeword-stream-eval \
  --wakeword-manifest /tmp/twinr_critical16_v2.json
```

```bash
PYTHONPATH=src python3 -m twinr \
  --env-file .env \
  --wakeword-promotion-eval \
  --wakeword-promotion-spec /tmp/twinr_stage1_promotion_spec.json
```

```bash
PYTHONPATH=src python3 -m twinr \
  --env-file .env \
  --wakeword-training-plan \
  --wakeword-training-plan-output /tmp/twinr_wakeword_training_plan.md
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
  --wakeword-train-sequence-verifier \
  --wakeword-manifest /tmp/twinr_oww_capture_room_20260319a/captured_manifest.json \
  --wakeword-sequence-verifier-model src/twinr/proactive/wakeword/models/twinr_v2.onnx \
  --wakeword-sequence-verifier-aux-model src/twinr/proactive/wakeword/models/twinr_v1.onnx
```

```bash
PYTHONPATH=src python3 -m twinr \
  --env-file .env \
  --wakeword-train-model \
  --wakeword-dataset-root /tmp/twinr_family_v3 \
  --wakeword-model-output src/twinr/proactive/wakeword/models/twinr_v2.onnx \
  --wakeword-training-model-type mlp \
  --wakeword-training-layer-dim 256 \
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

## Canonical Training Plan

- The repo now exposes one canonical research-backed training plan via `training_plan.py` and `twinr --wakeword-training-plan`.
- Stage 1 is explicitly the broad phonetic family detector for `Twinr`, `Twinna`, `Twina`, and `Twinner`.
- The default base-model trainer now matches the shipped `twinr_v1` family more closely: `StandardScaler + MLP` exported to ONNX, not only the upstream openWakeWord DNN loop.
- Hard-negative mining is mandatory before another promotion attempt, with priority on real Pi false activations and the confusion family `Twin`, `Winner`, `Winter`, `Tina`, `Timer`, and `Twitter`.
- Promotion is blocked unless the candidate passes the held-out suites plus the long-form Pi ambient false-accepts/hour guard on the real Twinr runtime evaluation path, driven by `promotion.py` and `twinr --wakeword-promotion-eval`.

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [models/README.md](./models/README.md)
- [../runtime/README.md](../runtime/README.md)
- [../governance/README.md](../governance/README.md)
