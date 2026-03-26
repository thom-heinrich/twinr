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
python3 hardware/servo/attention_servo_state.py hold
python3 hardware/servo/attention_servo_state.py arm
python3 hardware/servo/attention_servo_state.py return-to-estimated-zero
```

The Mini Maestro ships by default as `UART_DETECT_BAUD_RATE`. In that mode the
USB Command Port is wired to the UART RX path instead of the Maestro command
processor, so Twinr's `/dev/ttyACM*` writes do not move the servo. This setup
script installs the official Pololu USB rule and flips the controller to
`USB_DUAL_PORT`, which is the mode Twinr expects for runtime serial control.

The state helper is for the feedbackless 360-degree servo path on the main Pi.
`hold-current-zero` tells Twinr to treat the current physical pose as the
virtual `0°` reference, clear the persisted movement journal, and keep manual
hold enabled. On this continuous-servo path, manual hold means the runtime
keeps the output released until explicitly armed; it does not keep driving a
center pulse. `arm` only disables that manual hold flag; the running runtime
picks the change up live on the next servo update tick. Once armed, Twinr now
persists the actual outbound movement journal from that operator-defined `0°`.
When `return-to-estimated-zero` is requested, the runtime first replays that
logged motion path in reverse, using the inverse pulse widths and durations,
instead of falling back immediately to a generic recenter pulse. That is the
primary path for returning to the operator-defined `0°` on a feedbackless
360-degree servo. The reverse replay no longer holds each return pulse until
the next runtime attention tick; Twinr now plays each segment through an exact
bounded background timer and disables the output at the recorded deadline. If
no journal is available, Twinr still falls back to the older bounded
estimated-zero planner. The saved heading uncertainty still guards the
request, and the fallback planner still uses its tighter settle tolerance,
dedicated estimated-zero speed scale, and slow move/release windows so the
360-degree servo does not whip back aggressively or creep backward on an
imperfect stop pulse. For live following, Twinr now also supports a separate
visible-track release threshold via `TWINR_ATTENTION_SERVO_HOLD_MIN_CONFIDENCE`:
the controller still requires the normal minimum confidence to acquire a new
target, but it can keep following an already-visible person across short
confidence dips instead of kicking the servo into recenter/release bursts. On
the continuous-servo path, a visible follow target also no longer stays parked
on the last released pulse; once the person is still visible, Twinr resumes
active low-speed tracking instead of repeating release/re-engage bursts around
that stale pulse.
For active live following on a continuous servo, Twinr now uses the visible
image offset itself as the closed-loop control signal and only maps that error
into the configured continuous min/max speed pulse window. The virtual heading
planner stays in the loop for persisted state, journaling, and return-to-zero,
but active visible follow no longer runs through the virtual-heading targeter.

## See also

- [Top-level hardware README](../README.md)
- [Runtime Maestro adapter](../../src/twinr/hardware/servo_maestro.py)
- [Servo kernel setup](../servo_kernel/README.md)
