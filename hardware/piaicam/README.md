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
| [capture_custom_gesture_dataset.py](./capture_custom_gesture_dataset.py) | Capture bounded JPEG datasets for labels such as `none`, `ok_sign`, and `middle_finger` on the Pi |
| [custom_gesture_workflow.py](./custom_gesture_workflow.py) | Shared dataset-validation, capture-planning, and model-copy helpers for the custom gesture workflow |
| [fetch_mediapipe_models.py](./fetch_mediapipe_models.py) | Download the official MediaPipe pose, hand-landmarker, and gesture task bundles used on the Pi |
| [smoke_piaicam.py](./smoke_piaicam.py) | Run bounded Pi AI camera smoke phases |
| [train_custom_gesture_model.py](./train_custom_gesture_model.py) | Train and export a custom MediaPipe Gesture Recognizer `.task` from a folder-per-label dataset |
| [AGENTS.md](./AGENTS.md) | Local editing and verification rules |

## Usage

```bash
python3 hardware/piaicam/fetch_mediapipe_models.py
python3 hardware/piaicam/capture_custom_gesture_dataset.py --label none --count 24
python3 hardware/piaicam/smoke_piaicam.py
python3 hardware/piaicam/smoke_piaicam.py --profile quick
python3 hardware/piaicam/smoke_piaicam.py --ai-config /usr/share/rpi-camera-assets/imx500_posenet.json
state/mediapipe/model_maker_venv/bin/python hardware/piaicam/train_custom_gesture_model.py --dataset-root state/mediapipe/custom_gesture_dataset --dry-run
```

Artifacts land under `state/piaicam/runs/<timestamp>/` by default, with a `latest`
symlink when the output directory is auto-created.

For custom training, use a dedicated local venv on the leading repo. The
current reliable combination is `mediapipe-model-maker` plus `setuptools<81`,
because newer `setuptools` drops `pkg_resources`, which `tensorflow_hub` still
imports in this workflow.

## See also

- [hardware README](../README.md)
- [hardware AGENTS](../AGENTS.md)
- [AGENTS.md](./AGENTS.md)
