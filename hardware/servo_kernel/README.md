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
| [twinr_servo_profile.c](./twinr_servo_profile.c) | Bounded C motion-profile probe with min-jerk easing, update gating, optional breakaway compensation, and persisted last-target safety |

## Build on the Pi

```bash
cd /twinr/hardware/servo_kernel
make
sudo insmod twinr_servo.ko
ls /sys/class/twinr_servo/servo0
./twinr_servo_profile --start-us 1500 --target-us 1352 --duration-ms 7260
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

## Motion probe helper

`twinr_servo_profile` is a low-level acceptance helper, not runtime policy. It
lets Pi validation runs reproduce the same calm move shape while keeping the
kernel module focused on pulse generation only. The helper now persists the
last commanded end pulse in `/var/tmp/twinr_servo_profile_last_us` and refuses
future runs whose explicit `--start-us` disagrees with that state unless
`--override-state` is passed. This prevents new bounded probes from first
snapping back toward a stale assumed start position.

Example:

```bash
sudo insmod twinr_servo.ko
./twinr_servo_profile \
  --start-us 1500 \
  --target-us 1352 \
  --duration-ms 7260 \
  --cadence-ms 52 \
  --gate-us 6 \
  --breakaway-us 18 \
  --breakaway-hold-ms 140
sudo rmmod twinr_servo
```

Follow-up moves can omit `--start-us` and continue from the persisted last
target automatically:

```bash
sudo insmod twinr_servo.ko
./twinr_servo_profile \
  --target-us 1500 \
  --duration-ms 5040 \
  --cadence-ms 36 \
  --gate-us 6 \
  --breakaway-us 18 \
  --breakaway-hold-ms 140
sudo rmmod twinr_servo
```
