# hardware

Pi-side setup, probe, and smoke-test scripts for Twinr peripherals.

## Responsibility

`hardware` owns:
- persist Raspberry Pi hardware settings into Twinr env files and OS config
- install vendor or system dependencies for display, audio, and printer paths
- install or stage Pi-side systemd units for permanent Twinr watchdogs
- run bounded manual probes for buttons, PIR motion, display, and printer setup

`hardware` does **not** own:
- runtime device adapters in `src/twinr/hardware` or `src/twinr/display`
- main Twinr loop orchestration or user-facing behavior policy
- direct runtime-only fixes in `/twinr` without leading-repo changes here first

## Key files

| File | Purpose |
|---|---|
| [buttons/](./buttons) | Button setup and GPIO probe |
| [display/](./display) | E-paper setup and smoke tests |
| [mic/](./mic) | Audio default-device setup |
| [ops/](./ops) | Pi-side systemd units for permanent watchdogs |
| [pir/](./pir) | PIR setup and motion probe |
| [printer/](./printer) | Thermal-printer CUPS setup |
| [component.yaml](./component.yaml) | Structured directory metadata |
| [AGENTS.md](./AGENTS.md) | Local editing rules |

## Usage

```bash
./hardware/buttons/setup_buttons.sh --env-file .env --green 23 --yellow 22 --probe
python3 hardware/display/display_test.py --env-file .env
sudo ./hardware/mic/setup_audio.sh --env-file .env --device-match Jabra --test
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [Runtime hardware package](../src/twinr/hardware/README.md)
- [Runtime display package](../src/twinr/display/README.md)
