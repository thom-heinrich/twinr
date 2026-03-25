# servo

Pi-side setup helper for the Pololu Mini Maestro attention-servo path.

## Responsibility

`servo` owns:
- install the official Pololu USB udev rule used by native Maestro diagnostics
- switch a connected Maestro into `USB_DUAL_PORT` so Twinr's runtime command-port writer can talk to the real command processor
- verify the resulting Maestro status over Pololu's own `UscCmd` utility

`servo` does **not** own:
- runtime servo policy in `src/twinr/hardware/servo_follow.py`
- runtime Maestro command-port I/O in `src/twinr/hardware/servo_maestro.py`
- Raspberry Pi kernel-servo setup in `hardware/servo_kernel/`

## Key files

| File | Purpose |
|---|---|
| [setup_pololu_maestro.sh](./setup_pololu_maestro.sh) | Install the official Pololu USB rule and switch the attached Maestro to `USB_DUAL_PORT` |

## Usage

```bash
sudo ./hardware/servo/setup_pololu_maestro.sh
sudo ./hardware/servo/setup_pololu_maestro.sh --device 00467371
```

The Mini Maestro ships by default as `UART_DETECT_BAUD_RATE`. In that mode the
USB Command Port is wired to the UART RX path instead of the Maestro command
processor, so Twinr's `/dev/ttyACM*` writes do not move the servo. This setup
script installs the official Pololu USB rule and flips the controller to
`USB_DUAL_PORT`, which is the mode Twinr expects for runtime serial control.

## See also

- [Top-level hardware README](../README.md)
- [Runtime Maestro adapter](../../src/twinr/hardware/servo_maestro.py)
- [Servo kernel setup](../servo_kernel/README.md)
