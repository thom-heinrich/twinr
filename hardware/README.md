# hardware

Pi-side setup, probe, and smoke-test scripts for Twinr peripherals.

## Responsibility

`hardware` owns:
- persist Raspberry Pi hardware settings into Twinr env files and OS config
- install vendor or system dependencies for display, audio, and printer paths
- install and stage vendor prerequisites for optional Bitcraze Crazyradio/Crazyflie experimentation under an isolated runtime workspace
- install or stage Pi-side productive systemd units for authoritative runtime supervision and the web portal
- install or stage the development-host bounded drone daemon that exposes Twinr's mission-level inspection surface behind preflight and manual-arm gates
- keep only the productive hardware/bootstrap surface in the tracked tree; retired helper-Pi proxy artifacts now live under `__legacy__/hardware/ops/`
- stage low-level Pi-side kernel or OS integration code when a peripheral needs a real kernel-facing driver path instead of a userspace helper
- run bounded manual probes for buttons, PIR motion, display, and printer setup
- run the end-to-end acceptance deploy command that mirrors the leading repo to the Pi, syncs productive runtime settings, restarts services, and verifies post-restart health

`hardware` does **not** own:
- runtime device adapters in `src/twinr/hardware` or `src/twinr/display`
- main Twinr loop orchestration or user-facing behavior policy
- direct runtime-only fixes in `/twinr` without leading-repo changes here first

## Key files

| File | Purpose |
|---|---|
| [buttons/](./buttons) | Button setup and GPIO probe |
| [display/](./display) | Display backend setup and smoke tests |
| [bitcraze/](./bitcraze) | Crazyradio USB access, pinned firmware staging, isolated `/twinr/bitcraze` workspace provisioning, and the Crazyflie STM32 on-device failsafe build/flash helpers |
| [mic/](./mic) | Audio default-device setup, playback loudness normalization/softvol, and XVF3800 USB-access rule |
| [servo/](./servo) | Pololu Maestro USB access, `USB_DUAL_PORT` setup, and operator hold/arm state control for Twinr's continuous attention-servo runtime |
| [ops/](./ops) | Pi-side productive systemd units, the development-host bounded drone daemon, and Pi/bootstrap deploy helpers |
| [piaicam/](./piaicam) | Pi AI Camera smoke tests, custom gesture dataset capture, and MediaPipe model staging/training |
| [pir/](./pir) | PIR setup and motion probe |
| [printer/](./printer) | Thermal-printer CUPS setup |
| [servo_kernel/](./servo_kernel) | Out-of-tree Raspberry Pi servo kernel module plus low-level C acceptance tool for calm body-motion probes, including persisted last-target safety between bounded runs |
| [component.yaml](./component.yaml) | Structured directory metadata |
| [AGENTS.md](./AGENTS.md) | Local editing rules |

Retired standalone break-glass units now live only under the ignored top-level `__legacy__/` workspace folder when needed locally; they are no longer part of the tracked hardware tree.

## Usage

```bash
./hardware/buttons/setup_buttons.sh --env-file .env --green 23 --yellow 22 --probe
python3 hardware/display/display_test.py --env-file .env
sudo ./hardware/bitcraze/setup_bitcraze.sh --workspace /twinr/bitcraze --runtime-user thh
python3 hardware/bitcraze/probe_crazyradio.py --workspace /twinr/bitcraze
bash hardware/bitcraze/build_on_device_failsafe.sh
/twinr/bitcraze/.venv/bin/python hardware/bitcraze/flash_on_device_failsafe.py
sudo ./hardware/mic/setup_audio.sh --env-file .env --proactive-device-match reSpeaker --test
sudo ./hardware/mic/setup_respeaker_access.sh
sudo ./hardware/servo/setup_pololu_maestro.sh
python3 hardware/servo/attention_servo_state.py status
python3 hardware/piaicam/fetch_mediapipe_models.py
python3 hardware/piaicam/import_public_seed_dataset.py --count-per-label 128
python3 hardware/piaicam/capture_custom_gesture_dataset.py --label none --count 24
python3 hardware/piaicam/capture_custom_gesture_dataset.py --label peace_sign --count 40 --interval-s 0.35
python3 hardware/piaicam/smoke_piaicam.py --profile quick
state/mediapipe/model_maker_venv/bin/python hardware/piaicam/train_custom_gesture_model.py --dataset-root state/mediapipe/custom_gesture_dataset --dry-run
python3 hardware/ops/install_whatsapp_node_runtime.py
python3 hardware/ops/bootstrap_self_coding_pi.py
python3 hardware/ops/drone_daemon.py --pose-provider stub_ok --bind 127.0.0.1 --port 8791
python3 hardware/ops/watch_pi_repo_mirror.py --once
python3 hardware/ops/deploy_pi_runtime.py --live-text "Antworte nur mit: ok."
```

Legacy helper-Pi proxy install commands are intentionally omitted from the main
usage path. Current Twinr runtime wiring is single-Pi; the historical proxy
artifacts now live only under `__legacy__/hardware/ops/`.

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [Runtime hardware package](../src/twinr/hardware/README.md)
- [Runtime display package](../src/twinr/display/README.md)
