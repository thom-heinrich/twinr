# wakeword

Wakeword matching, local KWS/openWakeWord spotting, optional local verifier
assets, sequence-aware cascade verification, policy, calibration, streaming,
and offline plus runtime-faithful evaluation helpers for Twinr.

This package is the wakeword-specific boundary below proactive runtime
orchestration and above raw audio/hardware adapters.

## Import contract

- Runtime startup must not fail just because optional WeKws and offline
  training helpers are unavailable on the Pi.
- The package surface therefore keeps normal runtime helpers importable even
  when optional WeKws-specific dependencies are missing.
- When Twinr actually uses WeKws tooling or the WeKws backend, `PyYAML`
  remains a declared dependency and must exist in the active environment.

## Responsibility

`wakeword` owns:
- Normalize wakeword phrases, transcripts, and detector labels
- Extract continuation text from a confirmed wake capture even when the tail STT
  omits the already-confirmed wake phrase and only returns the spoken command
- Separate canonical wakeword phrases from STT-specific recognition phrases
- Run local clip/frame wakeword spotting via openWakeWord or sherpa-onnx KWS plus optional verifier and STT confirmation
- Run a Twinr-specific sequence-aware verifier as a second local stage on localized wakeword captures, regardless of whether stage 1 is `openwakeword`, `kws`, or `wekws`
- Resolve bundled local openWakeWord model assets when the repo ships them
- Auto-load sibling `.verifier.pkl` assets for bundled local models when available
- Auto-load sibling `.sequence_verifier.pkl` assets for bundled local models when available
- Fail closed when the configured sherpa-onnx KWS asset bundle is incomplete
- Persist and apply per-device wakeword calibration overrides
- Keep backend-family selection anchored in deployment config; calibration may tune thresholds and phrases, but it must not silently switch a Pi from `kws` back to `openwakeword`
- Replay labeled captures for evaluation, labeling, verifier training, and autotune workflows
- Replay labeled captures through the streaming runtime path for promotion suites and ambient false-accepts/hour guards
- Train reproducible Twinr base models from generated multivoice datasets and select deployment thresholds against labeled acceptance captures
- Export Twinr labeled manifests into WeKws/Kaldi-style split directories so a real custom conventional KWS detector can be trained outside the generic bundle path
- Prepare reproducible WeKws experiment workspaces, configs, and runner scripts from exported Twinr datasets
- Stream bounded PCM frames into wakeword detections for proactive runtime
- Keep optional evaluation/training/WeKws dependencies from breaking the normal runtime import path when those helpers are not selected

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
| `local_verifier.py` | Backend-agnostic config bridge for attaching the sequence verifier to local stage-1 detectors |
| `matching.py` | Transcript, confirmed-tail, and detector-label matching |
| `kws.py` | sherpa-onnx KWS clip and frame spotting |
| `kws_assets.py` | Official sherpa-onnx bundle provisioning, optional phone-lexicon overlays, and Twinr keyword-file generation |
| `spotter.py` | openWakeWord clip and frame spotting plus optional local verifier loading |
| `policy.py` | Acceptance, fallback, local cascade gating, and STT verification |
| `promotion.py` | Runtime-faithful stream replay, promotion specs, suite guards, and ambient false-accepts/hour reports |
| `calibration.py` | Calibration profile persistence |
| `evaluation.py` | Labeling, manifest replay, verifier training, eval, and autotune |
| `training.py` | Offline base-model training, ONNX export, and threshold selection |
| `training_plan.py` | Canonical Stage-1 family, hard-negative mining, and Pi acceptance plan rendering |
| `wekws_export.py` | Export Twinr manifests into WeKws/Kaldi-style training splits for custom conventional KWS training |
| `wekws_experiment.py` | Build a reproducible WeKws experiment workspace, config, runner script, and MDTC-safe ONNX export helper from exported Twinr data |
| `synthetic_corpus.py` | Plan deterministic Qwen3TTS wakeword corpora with many speakers, style conditions, and lightweight channel degradations |
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
TWINR_WAKEWORD_PRIMARY_BACKEND=kws
TWINR_WAKEWORD_KWS_TOKENS_PATH=src/twinr/proactive/wakeword/models/kws/tokens.txt
TWINR_WAKEWORD_KWS_ENCODER_PATH=src/twinr/proactive/wakeword/models/kws/encoder.onnx
TWINR_WAKEWORD_KWS_DECODER_PATH=src/twinr/proactive/wakeword/models/kws/decoder.onnx
TWINR_WAKEWORD_KWS_JOINER_PATH=src/twinr/proactive/wakeword/models/kws/joiner.onnx
TWINR_WAKEWORD_KWS_KEYWORDS_FILE_PATH=src/twinr/proactive/wakeword/models/kws/keywords.txt
```

Provision the bundle reproducibly from the leading repo before switching runtime:

```bash
PYTHONPATH=src python3 -m twinr \
  --env-file .env \
  --wakeword-kws-provision \
  --wakeword-kws-force
```

Phone-based bundles such as `zh_en_3m_phone_int8` can take explicit
pronunciation overlays during provisioning:

```bash
PYTHONPATH=src python3 -m twinr \
  --env-file .env \
  --wakeword-kws-provision \
  --wakeword-kws-bundle zh_en_3m_phone_int8 \
  --wakeword-kws-keyword Twinna \
  --wakeword-kws-keyword Twina \
  --wakeword-kws-keyword Twinner \
  --wakeword-kws-keyword Twinr \
  --wakeword-kws-lexicon-entry 'Twinna=T W IY1 N AH0' \
  --wakeword-kws-lexicon-entry 'Twina=T W IY1 N AH0' \
  --wakeword-kws-lexicon-entry 'Twinner=T W IY1 N ER0' \
  --wakeword-kws-lexicon-entry 'Twinr=T W IY1 N ER0' \
  --wakeword-kws-force
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
  --wakeword-export-wekws \
  --wakeword-wekws-output-dir /tmp/twinr_wekws \
  --wakeword-wekws-train-manifest /tmp/twinr_train_manifest.json \
  --wakeword-wekws-dev-manifest /tmp/twinr_dev_manifest.json \
  --wakeword-wekws-test-manifest /tmp/twinr_test_manifest.json
```

```bash
PYTHONPATH=src python3 -m twinr \
  --env-file .env \
  --wakeword-prepare-wekws-experiment \
  --wakeword-wekws-dataset-dir /tmp/twinr_wekws \
  --wakeword-wekws-experiment-dir /tmp/twinr_wekws_exp \
  --wakeword-wekws-recipe mdtc_fbank_stream
```

```bash
PYTHONPATH=src python3 scripts/generate_qwen3tts_wakeword_corpus.py \
  --output-root /tmp/twinr_qwen3tts_corpus_v1 \
  --device cuda:0 \
  --speaker-shard-index 0 \
  --speaker-shard-count 2
```

```bash
PYTHONPATH=src python3 scripts/generate_qwen3tts_wakeword_corpus.py \
  --output-root /tmp/twinr_qwen3tts_negatives_v2 \
  --device cuda:1 \
  --label-filter negative \
  --style-profile plain,warm,soft \
  --generation-profile stable,diverse
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
  --wakeword-training-difficulty-model src/twinr/proactive/wakeword/models/twinr_v1.onnx \
  --wakeword-training-difficulty-negative-scale 3.0 \
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
- The MLP trainer can optionally reuse one reference detector to upweight deployment-proximate hard examples, so room captures and mined confusions that still activate get more gradient pressure than easy synthetic negatives.
- Hard-negative mining is mandatory before another promotion attempt, with priority on real Pi false activations and the confusion family `Twin`, `Winner`, `Winter`, `Tina`, `Timer`, and `Twitter`.
- Promotion is blocked unless the candidate passes the held-out suites plus the long-form Pi ambient false-accepts/hour guard on the real Twinr runtime evaluation path, driven by `promotion.py` and `twinr --wakeword-promotion-eval`.
- The new `kws` backend is a professional runtime path built around `sherpa-onnx` streaming keyword spotting. It requires an explicit asset bundle (`tokens.txt`, `encoder.onnx`, `decoder.onnx`, `joiner.onnx`, `keywords.txt`) and stays fail-closed when the bundle is missing or incomplete.
- `kws_assets.py` provisions the official upstream bundle plus Twinr-specific `keywords_raw.txt`, `keywords.txt`, optional `bpe.model`, optional phone lexicon files such as `en.phone`, and `bundle_metadata.json` so `/twinr` can be switched without ad-hoc shell work or silently dropped custom wakewords.
- For a true custom conventional detector, `wekws_export.py` now bridges Twinr's labeled Pi captures into WeKws/Kaldi-style `wav.scp`, `text`, `utt2spk`, `wav.dur`, and `dict/` files so the next retrain can leave the generic open-vocabulary bundle path entirely.
- `wekws_experiment.py` now turns those exported splits into a full WeKws workspace with `data.list`, a built-in Twinr recipe, `conf/*.yaml`, `exp/<recipe>/`, and a reproducible `run_wekws.sh` so the GPU training path stops depending on ad-hoc shell history.
- `wekws.py` now treats Twinr WeKws bundles as embedded-CMVN models by default when a CMVN sidecar exists but the ONNX metadata is still legacy-empty. This prevents the earlier double-CMVN runtime collapse where PyTorch and ONNX matched on the same features, but Twinr flattened both classes into one score band by normalizing twice.
- `promotion.py` now runs a hard raw-score separation guard for `wekws` before the runtime-faithful stream replay. Bundles whose positive P10 score is not above the negative P90 score are blocked before they can look good on clip counts for the wrong reason.
- `synthetic_corpus.py` plus `scripts/generate_qwen3tts_wakeword_corpus.py` now provide the large synthetic data engine for Twinr-specific WeKws retrains: many Qwen3TTS speakers, style instructions, deterministic seeds, lightweight far-field/noise/channel degradations, and asymmetric positive-vs-negative corpus runs before the WeKws export step.

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [models/README.md](./models/README.md)
- [../runtime/README.md](../runtime/README.md)
- [../governance/README.md](../governance/README.md)
