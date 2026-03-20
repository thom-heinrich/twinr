Bundled Twinr wakeword model assets live here.

- `twinr_v1.onnx` is the local openWakeWord-compatible base detector.
- `twinr_v1.metadata.json` stores the training/eval summary used to select the current threshold candidate.
- Additional promoted detector variants may live here as `<model_stem>.onnx` with matching `<model_stem>.metadata.json`.
- New model metadata files are written by `twinr --wakeword-train-model` or `train_wakeword_base_model_from_dataset_root()`.
- Exported Twinr ONNX assets are normalized to single-file models; `.onnx.data` sidecars are treated as a broken export that must be consolidated before deployment.
- `<model_stem>.verifier.pkl`, when present next to the base model, is auto-loaded as a second-stage local verifier for the same model stem.
- `<model_stem>.sequence_verifier.pkl`, when present next to the base model, is auto-loaded as Twinr's sequence-aware DTW verifier for the same detector label.

`TwinrConfig.from_env()` resolves this directory automatically when no explicit `TWINR_WAKEWORD_OPENWAKEWORD_MODELS` override is set.
