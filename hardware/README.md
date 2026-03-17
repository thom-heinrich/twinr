# hardware

Pi-side setup, probe, and smoke-test scripts for Twinr peripherals.

## Responsibility

`hardware` owns:
- persist Raspberry Pi hardware settings into Twinr env files and OS config
- install vendor or system dependencies for display, audio, and printer paths
- install or stage Pi-side productive systemd units for authoritative runtime supervision and the web portal
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
| [ops/](./ops) | Pi-side productive systemd units plus Pi bootstrap helpers |
| [pir/](./pir) | PIR setup and motion probe |
| [printer/](./printer) | Thermal-printer CUPS setup |
| [component.yaml](./component.yaml) | Structured directory metadata |
| [AGENTS.md](./AGENTS.md) | Local editing rules |

Retired standalone break-glass units now live only under the ignored top-level `__legacy__/` workspace folder when needed locally; they are no longer part of the tracked hardware tree.

## Usage

```bash
./hardware/buttons/setup_buttons.sh --env-file .env --green 23 --yellow 22 --probe
python3 hardware/display/display_test.py --env-file .env
sudo ./hardware/mic/setup_audio.sh --env-file .env --device-match Jabra --test
python3 hardware/ops/bootstrap_self_coding_pi.py
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [Runtime hardware package](../src/twinr/hardware/README.md)
- [Runtime display package](../src/twinr/display/README.md)
