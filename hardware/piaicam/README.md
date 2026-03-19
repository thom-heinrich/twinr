# piaicam

Bounded smoke-test scripts for the Raspberry Pi AI Camera path used by Twinr.

## Responsibility

`piaicam` owns:
- run manual Pi AI Camera smoke tests outside the main Twinr runtime
- verify camera enumeration, still capture, short video capture, and IMX500 AI startup
- stage the official MediaPipe `.task` bundles needed by the hybrid IMX500 + MediaPipe runtime path
- write bounded logs and artifacts for operator review

`piaicam` does **not** own:
- Twinr runtime vision integration
- persistent boot or package configuration
- model packaging or custom network authoring
- user-facing runtime behavior

## Key files

| File | Purpose |
|---|---|
| [fetch_mediapipe_models.py](./fetch_mediapipe_models.py) | Download the official MediaPipe pose and gesture task bundles used on the Pi |
| [smoke_piaicam.py](./smoke_piaicam.py) | Run bounded Pi AI camera smoke phases |
| [AGENTS.md](./AGENTS.md) | Local editing and verification rules |

## Usage

```bash
python3 hardware/piaicam/fetch_mediapipe_models.py
python3 hardware/piaicam/smoke_piaicam.py
python3 hardware/piaicam/smoke_piaicam.py --profile quick
python3 hardware/piaicam/smoke_piaicam.py --ai-config /usr/share/rpi-camera-assets/imx500_posenet.json
```

Artifacts land under `state/piaicam/runs/<timestamp>/` by default, with a `latest`
symlink when the output directory is auto-created.

## See also

- [hardware README](../README.md)
- [hardware AGENTS](../AGENTS.md)
- [AGENTS.md](./AGENTS.md)
