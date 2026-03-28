# piaicam

Bounded smoke-test scripts for the Raspberry Pi AI Camera path used by Twinr.

## Responsibility

`piaicam` owns:
- run manual Pi AI Camera smoke tests outside the main Twinr runtime
- verify camera enumeration, still capture, short video capture, and IMX500 AI startup
- stage the official MediaPipe `.task` bundles needed by the hybrid IMX500 + MediaPipe runtime path
- capture labeled still-image datasets for custom gesture training
- train and package bounded custom MediaPipe gesture models on the leading repo
- write bounded logs and artifacts for operator review

`piaicam` does **not** own:
- Twinr runtime vision integration
- persistent boot or package configuration
- unbounded training loops or hidden background services
- user-facing runtime behavior

## Key files

| File | Purpose |
|---|---|
| [capture_custom_gesture_dataset.py](./capture_custom_gesture_dataset.py) | Capture bounded JPEG datasets for labels such as `none`, `thumbs_up`, `thumbs_down`, and `peace_sign` on the Pi |
| [custom_gesture_workflow.py](./custom_gesture_workflow.py) | Shared dataset-validation, capture-planning, and model-copy helpers for the custom gesture workflow |
| [fetch_mediapipe_models.py](./fetch_mediapipe_models.py) | Download the official MediaPipe pose, hand-landmarker, and gesture task bundles used on the Pi |
| [import_public_seed_dataset.py](./import_public_seed_dataset.py) | Download and import a bounded public HaGRID-derived bootstrap sample for `thumbs_up`, `thumbs_down`, `peace_sign`, and `none` |
| [probe_live_gestures.py](./probe_live_gestures.py) | Capture bounded JSONL samples of raw MediaPipe pose/fine-hand outputs plus the final Twinr gesture observation for live calibration |
| [smoke_piaicam.py](./smoke_piaicam.py) | Run bounded Pi AI camera smoke phases |
| [train_custom_gesture_model.py](./train_custom_gesture_model.py) | Train and export a custom MediaPipe Gesture Recognizer `.task` from a folder-per-label dataset |
| [AGENTS.md](./AGENTS.md) | Local editing and verification rules |

## Usage

```bash
python3 hardware/piaicam/fetch_mediapipe_models.py
python3 hardware/piaicam/import_public_seed_dataset.py --dry-run
python3 hardware/piaicam/import_public_seed_dataset.py --count-per-label 128
PYTHONPATH=src ./.venv/bin/python hardware/piaicam/probe_live_gestures.py --env-file /twinr/.env --duration-s 8 --output /tmp/twinr_gesture_probe.jsonl
python3 hardware/piaicam/capture_custom_gesture_dataset.py --label none --count 24
python3 hardware/piaicam/capture_custom_gesture_dataset.py --label peace_sign --count 40 --interval-s 0.35
python3 hardware/piaicam/smoke_piaicam.py
python3 hardware/piaicam/smoke_piaicam.py --profile quick
python3 hardware/piaicam/smoke_piaicam.py --ai-config /usr/share/rpi-camera-assets/imx500_posenet.json
state/mediapipe/model_maker_venv/bin/python hardware/piaicam/train_custom_gesture_model.py --dataset-root state/mediapipe/custom_gesture_dataset --dry-run
```

Artifacts land under `state/piaicam/runs/<timestamp>/` by default, with a `latest`
symlink when the output directory is auto-created.

`probe_live_gestures.py` imports the Twinr runtime and therefore needs the
repo-local Python 3.11 environment, not an arbitrary system `python3`. The
training helper still belongs in the dedicated Model Maker Python 3.10 venv.

For custom training, use a dedicated local venv on the leading repo. The
current reliable combination is `mediapipe-model-maker` plus `setuptools<81`,
because newer `setuptools` drops `pkg_resources`, which `tensorflow_hub` still
imports in this workflow.

The public-seed importer currently targets one reduced HaGRID derivative that
contains `like`, `dislike`, `peace`, `no_gesture`, and `ok`. Twinr maps only
`like -> thumbs_up`, `dislike -> thumbs_down`, `peace -> peace_sign`, and
`no_gesture -> none`; the extra `ok` class is intentionally ignored.

The built-in MediaPipe recognizer already covers gesture families such as
`thumbs_up`, `thumbs_down`, `pointing_up`, `open_palm`, and `victory`. Twinr's
current Pi live-HCI target is nevertheless a small custom model for
`thumbs_up`, `thumbs_down`, `peace_sign`, and `none`, because the product only
cares about those three symbols and the built-in bundle is still too unstable
for `peace_sign` on the real device setup.

The training helper now stages only the explicitly required labels into a fresh
run-local `training_dataset/` directory before calling MediaPipe Model Maker.
That keeps old exploratory labels such as `ok_sign` or `middle_finger` from
silently widening the classifier head when the shared dataset root contains
mixed experiments.

## See also

- [hardware README](../README.md)
- [hardware AGENTS](../AGENTS.md)
- [AGENTS.md](./AGENTS.md)
