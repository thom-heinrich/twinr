# servo_kernel

`servo_kernel` owns Twinr's out-of-tree Raspberry Pi servo kernel module.

## Responsibility

This folder exists for the low-level body-orientation servo path when userspace
PWM helpers on the Pi are not calm enough for Twinr's senior-facing motion.

It owns:
- the out-of-tree kernel module source
- the module build recipe for Raspberry Pi kernel headers
- the sysfs control surface contract the runtime can target later

It does **not** own:
- Twinr runtime follow policy
- attention-target derivation
- display or proactive orchestration

## Files

| File | Purpose |
|---|---|
| [twinr_servo.c](./twinr_servo.c) | Single-servo hrtimer-driven kernel module with sysfs controls |
| [Makefile](./Makefile) | Standard out-of-tree kernel module build entrypoint |

## Build on the Pi

```bash
cd /twinr/hardware/servo_kernel
make
sudo insmod twinr_servo.ko
ls /sys/class/twinr_servo/servo0
```

## Runtime sysfs contract

The module exposes:

- `/sys/class/twinr_servo/servo0/gpio`
- `/sys/class/twinr_servo/servo0/period_us`
- `/sys/class/twinr_servo/servo0/pulse_width_us`
- `/sys/class/twinr_servo/servo0/enabled`

Example:

```bash
echo 18 | sudo tee /sys/class/twinr_servo/servo0/gpio
echo 1500 | sudo tee /sys/class/twinr_servo/servo0/pulse_width_us
echo 1 | sudo tee /sys/class/twinr_servo/servo0/enabled
echo 0 | sudo tee /sys/class/twinr_servo/servo0/enabled
```
