# hardware

Pi-side setup, probe, and smoke-test scripts for Twinr peripherals.

## Responsibility

`hardware` owns:
- persist Raspberry Pi hardware settings into Twinr env files and OS config
- install vendor or system dependencies for display, audio, and printer paths
- install or stage Pi-side productive systemd units for authoritative runtime supervision and the web portal
- stage low-level Pi-side kernel or OS integration code when a peripheral needs a real kernel-facing driver path instead of a userspace helper
- run bounded manual probes for buttons, PIR motion, display, and printer setup

`hardware` does **not** own:
- runtime device adapters in `src/twinr/hardware` or `src/twinr/display`
- main Twinr loop orchestration or user-facing behavior policy
- direct runtime-only fixes in `/twinr` without leading-repo changes here first

## Key files

| File | Purpose |
|---|---|
| [buttons/](./buttons) | Button setup and GPIO probe |
| [display/](./display) | Display backend setup and smoke tests |
| [mic/](./mic) | Audio default-device setup, playback loudness normalization/softvol, and XVF3800 USB-access rule |
| [ops/](./ops) | Pi-side productive systemd units plus Pi bootstrap helpers |
| [piaicam/](./piaicam) | Pi AI Camera smoke tests, custom gesture dataset capture, and MediaPipe model staging/training |
| [pir/](./pir) | PIR setup and motion probe |
| [printer/](./printer) | Thermal-printer CUPS setup |
| [servo_kernel/](./servo_kernel) | Out-of-tree Raspberry Pi servo kernel module source and build recipe |
| [component.yaml](./component.yaml) | Structured directory metadata |
| [AGENTS.md](./AGENTS.md) | Local editing rules |

Retired standalone break-glass units now live only under the ignored top-level `__legacy__/` workspace folder when needed locally; they are no longer part of the tracked hardware tree.

## Usage

```bash
./hardware/buttons/setup_buttons.sh --env-file .env --green 23 --yellow 22 --probe
python3 hardware/display/display_test.py --env-file .env
sudo ./hardware/mic/setup_audio.sh --env-file .env --proactive-device-match reSpeaker --test
sudo ./hardware/mic/setup_respeaker_access.sh
python3 hardware/piaicam/fetch_mediapipe_models.py
python3 hardware/piaicam/capture_custom_gesture_dataset.py --label none --count 24
python3 hardware/piaicam/smoke_piaicam.py --profile quick
state/mediapipe/model_maker_venv/bin/python hardware/piaicam/train_custom_gesture_model.py --dataset-root state/mediapipe/custom_gesture_dataset --dry-run
python3 hardware/ops/install_whatsapp_node_runtime.py
python3 hardware/ops/bootstrap_self_coding_pi.py
python3 hardware/ops/watch_pi_repo_mirror.py --once
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [Runtime hardware package](../src/twinr/hardware/README.md)
- [Runtime display package](../src/twinr/display/README.md)
