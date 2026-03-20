Bundled Twinr wakeword model assets live here.

- `twinr_v1.onnx` is the local openWakeWord-compatible base detector.
- `twinr_v1.metadata.json` stores the training/eval summary used to select the current threshold candidate.
- New model metadata files are written by `twinr --wakeword-train-model` or `train_wakeword_base_model_from_dataset_root()`.
- Exported Twinr ONNX assets are normalized to single-file models; `.onnx.data` sidecars are treated as a broken export that must be consolidated before deployment.
- `twinr_v1.verifier.pkl`, when present next to the base model, is auto-loaded as a second-stage local verifier for the same model stem.

`TwinrConfig.from_env()` resolves this directory automatically when no explicit `TWINR_WAKEWORD_OPENWAKEWORD_MODELS` override is set.
