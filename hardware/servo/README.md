# servo

Pi-side setup helper for the Pololu Mini Maestro attention-servo path.

## Responsibility

`servo` owns:
- install the official Pololu USB udev rule used by native Maestro diagnostics
- switch a connected Maestro into `USB_DUAL_PORT` so Twinr's runtime command-port writer can talk to the real command processor
- verify the resulting Maestro status over Pololu's own `UscCmd` utility
- expose one tiny operator CLI to inspect or flip the persisted startup hold/arm state for Twinr's continuous-rotation attention servo

`servo` does **not** own:
- runtime servo policy in `src/twinr/hardware/servo_follow.py`
- runtime Maestro command-port I/O in `src/twinr/hardware/servo_maestro.py`
- Raspberry Pi kernel-servo setup in `hardware/servo_kernel/`

## Key files

| File | Purpose |
|---|---|
| [setup_pololu_maestro.sh](./setup_pololu_maestro.sh) | Install the official Pololu USB rule and switch the attached Maestro to `USB_DUAL_PORT` |
| [attention_servo_state.py](./attention_servo_state.py) | Inspect or change the persisted startup hold/arm state for the continuous attention-servo path |

## Usage

```bash
sudo ./hardware/servo/setup_pololu_maestro.sh
sudo ./hardware/servo/setup_pololu_maestro.sh --device 00467371
python3 hardware/servo/attention_servo_state.py status
python3 hardware/servo/attention_servo_state.py hold-current-zero
python3 hardware/servo/attention_servo_state.py arm
```

The Mini Maestro ships by default as `UART_DETECT_BAUD_RATE`. In that mode the
USB Command Port is wired to the UART RX path instead of the Maestro command
processor, so Twinr's `/dev/ttyACM*` writes do not move the servo. This setup
script installs the official Pololu USB rule and flips the controller to
`USB_DUAL_PORT`, which is the mode Twinr expects for runtime serial control.

The state helper is for the feedbackless 360-degree servo path on the main Pi.
`hold-current-zero` tells Twinr to treat the current physical pose as the
virtual `0°` reference and keep manual hold enabled. `arm` only disables that
manual hold flag; the running runtime picks the change up live on the next
servo update tick.

## See also

- [Top-level hardware README](../README.md)
- [Runtime Maestro adapter](../../src/twinr/hardware/servo_maestro.py)
- [Servo kernel setup](../servo_kernel/README.md)
