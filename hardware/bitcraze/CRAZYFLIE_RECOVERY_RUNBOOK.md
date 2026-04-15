# crazyflie recovery runbook

Authoritative runbook for recovering a `Crazyflie 2.x` after a bad flash or
partially broken boot path. This file captures the proven lessons from the
Twinr brick incident and the successful recovery.

## Scope

Use this runbook for:
- `Crazyflie 2.x` recovery after a bad `STM32` or coordinated firmware flash
- operator recovery on a host with `/twinr/bitcraze`
- deciding between `Crazyradio`, `STM32 DFU`, `nRF SWD`, and `AI-Deck JTAG`

Do **not** use this runbook for:
- `AI-Deck GAP8` recovery only
- general flight testing
- deck-specific OTA issues unless the core `Crazyflie` stack is already healthy

## Evidence Base

This runbook is grounded in:
- [FND000621.yml](../../artifacts/stores/findings/reports/FND000621.yml)
- [BF000968.yml](../../artifacts/stores/fixreport/reports/BF000968.yml)
- `H00112` in [hypotheses.yml](../../artifacts/stores/hypothesis/hypotheses.yml)
- adjacent deck-detection failure evidence in [FND000600.yml](../../artifacts/stores/findings/reports/FND000600.yml)
- adjacent Bitcraze stack-alignment lessons in [BF000546.yml](../../artifacts/stores/fixreport/reports/BF000546.yml)
- adjacent AI-Deck/Olimex separation lessons in [BF000588.yml](../../artifacts/stores/fixreport/reports/BF000588.yml)

The important proven conclusion is:
- the decisive fix was **not** an ad-hoc partial binary flash
- the decisive fix was the **official coordinated Bitcraze release ZIP**
  `firmware-cf2-2025.12.1.zip` flashed through the recovered radio bootloader
- the root mistake was flashing an unpinned custom full `stm32-fw` image built
  from an untagged checkout instead of using an official coordinated release

Additional later proof:
- one separate custom-boot regression was caused by an app-layer param TOC
  assert in `param_logic.c:254`
- the trigger was not the radio stack or the deck stack, but overlong
  `twinrFs` parameter names that exceeded Bitcraze's `CMD_GET_ITEM_V2` name
  budget
- the direct `ST-LINK/SWD` proof path was:
  `official firmware reaches systemStart` vs
  `custom firmware fails cfAssertNormalStartTest()` vs
  `assert snapshot points to param_logic.c:254`

## What We Proved

1. A custom full `STM32` flash can leave the craft outside the normal operator
   recovery path even when the hardware is still recoverable.
2. If the `nRF` bootloader can be reached again, the fastest and safest recovery
   is the official full release ZIP over `Crazyradio`.
3. Direct `SWD`/`JTAG` work is a recovery escalator, not the preferred first
   fix when the radio bootloader is alive.
4. The `Olimex ARM-USB-TINY-H` bundle and the `Bitcraze Debug Adapter Kit` are
   easy to misuse because they apply to different targets:
   - `AI-Deck GAP8`: `Olimex + ARM-JTAG-20-10 + JTAG`
   - `Crazyflie nRF51`: `ST-LINK/J-Link + SWD`
5. `ARM-JTAG-20-10` is only a pin adapter. It does **not** turn the Olimex
   bundle into a valid `nRF51 SWD` path.
6. The meta-lesson is the same as in [BF000546.yml](../../artifacts/stores/fixreport/reports/BF000546.yml):
   restore the **coordinated official stack first**, then do narrower component
   work if something still remains broken.

## LED States That Matter

The useful Bitcraze recovery meanings are:

| State | Meaning | Use |
|---|---|---|
| `M2` still blue | normal firmware path | normal boot |
| `M2` slow blink | `nRF` bootloader | preferred radio recovery state |
| `M2` fast blink after very long press | `MBS` / powers STM32 USB DFU path | STM32 recovery path |
| `M3` active while on USB | charging | not a separate boot mode |

Practical lesson:
- if `M2` is slow blinking, treat the craft as being in the correct radio
  bootloader state even if `M3` is also active because USB charging is present

Official references:
- https://www.bitcraze.io/documentation/system/platform/cf2-components/
- https://www.bitcraze.io/documentation/repository/crazyflie2-nrf-firmware/master/development/architecture/

## Decision Order

Always recover in this order:

1. `Crazyradio` + official release ZIP
2. `STM32 DFU` only if the `STM32` is clearly broken or the craft is stuck in
   the `MBS`/USB DFU path
3. `nRF SWD` only if the radio bootloader cannot be reached
4. `AI-Deck JTAG` only for deck-specific `GAP8` recovery, not for the core
   `Crazyflie` radio stack

This order is deliberate. It is the key operational learning from the incident.

## Custom App-Layer Boot Debugging

If an out-of-tree `STM32` image boots badly after flash, do not guess from
radio symptoms alone. Use this direct proof order:

1. restore the official Bitcraze `STM32` image and verify with `ST-LINK/SWD`
   that it reaches `systemStart`
2. flash the custom image only, leaving the official bootloader in place
3. compare the same boot symbols under `SWD`
4. if the custom image fails `cfAssertNormalStartTest()`, read the persisted
   assert snapshot before changing anything else

One proven failure pattern is Bitcraze's `CMD_GET_ITEM_V2` TOC size check in
`param_logic.c:254`. The practical limit is:

```text
strlen(group) + strlen(name) + 2 <= 26
```

For the `twinrFs` group this means parameter and log names must stay short.
Three overlong parameter names were enough to keep the custom image out of the
normal boot path until they were shortened.

## Preferred Recovery Path: Official Release ZIP Over Crazyradio

### Preconditions

- `Crazyradio` attached to the host
- `/twinr/bitcraze/.venv` exists
- the craft is in `nRF` bootloader mode (`M2` slow blink)
- the official release ZIP is available

Verified artifact:
- `/tmp/cf-recovery/firmware-cf2-2025.12.1.zip`

### 1. Verify the host radio path

```bash
python3 hardware/bitcraze/probe_crazyradio.py --workspace /twinr/bitcraze
```

The host should see a `PA`-compatible radio path, typically `1915:7777`.

### 2. Put the craft into radio bootloader mode

Use the documented battery-button path:

1. remove `USB` from the `Crazyflie`
2. remove the battery
3. hold the power button
4. reconnect the battery
5. release after about `2s`

Target state:
- `M2` slow blink

Do **not** hold too long if you want radio recovery. A very long press enters
the `MBS` path instead.

### 3. Verify bootloader connectivity

```bash
/twinr/bitcraze/.venv/bin/python -m cfloader info
```

Successful output should show both targets:
- `stm32 (0xFF)`
- `nrf51 (0xFE)`

This was the key success checkpoint during the incident.

### 4. Flash the official coordinated release

```bash
/twinr/bitcraze/.venv/bin/python -m cfloader flash /tmp/cf-recovery/firmware-cf2-2025.12.1.zip
```

Why ZIP and not loose binaries:
- the ZIP lets Bitcraze's loader reason about `STM32`, `nRF51`, and
  `bootloader+softdevice` together
- the loader explicitly checks whether the `nRF` softdevice/bootloader block
  needs updating before flashing the remaining artifacts
- this is the coordinated recovery path that actually restored the device

Observed successful behavior during recovery:
- `cfloader` reported the current `nRF` stack and the ZIP-provided stack
- `STM32` firmware flashed first
- `nRF51` firmware flashed second
- the process exited cleanly

### 5. Verify normal post-recovery radio connectivity

Minimal proof:

```bash
/twinr/bitcraze/.venv/bin/python - <<'PY'
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

uri = 'radio://0/80/2M'
cflib.crtp.init_drivers()
cf = Crazyflie(rw_cache='/tmp/cfcache')
with SyncCrazyflie(uri, cf=cf) as scf:
    print('connected', scf.cf.link_uri)
    print('param_toc_len', len(scf.cf.param.toc.toc))
    print('log_toc_len', len(scf.cf.log.toc.toc))
PY
```

The successful incident recovery ended with:
- normal `radio://0/80/2M` reconnection
- `param_toc_len=51`
- `log_toc_len=53`

That is the acceptance proof that the craft is back on the normal firmware path.

## Host Failure Pattern: Crazyradio Resets Mid-Flash

We saw one host-side problem that looked like a target problem:
- `cfloader` failed with `usb.core.USBError: [Errno 19] No such device`
- `dmesg` showed repeated USB resets on the `Crazyradio`

If that happens, check:

```bash
lsusb | rg '1915:7777|Bitcraze'
dmesg | tail -n 80
```

If the host keeps resetting the `Crazyradio`, force the USB device out of
autosuspend before retrying:

```bash
CR_PATH="$(
python3 - <<'PY'
from pathlib import Path
for p in Path('/sys/bus/usb/devices').glob('*'):
    try:
        if (p/'idVendor').read_text().strip() == '1915' and (p/'idProduct').read_text().strip() == '7777':
            print(p)
            break
    except Exception:
        pass
PY
)"
printf on | sudo tee "$CR_PATH/power/control" >/dev/null
if [ -f "$CR_PATH/power/autosuspend_delay_ms" ]; then
  printf -- '-1' | sudo tee "$CR_PATH/power/autosuspend_delay_ms" >/dev/null
fi
```

During the successful recovery, disabling autosuspend on the active
`Crazyradio` sysfs node stabilized the host path enough for `cfloader info`
and the official ZIP flash to complete.

## Deck Detection: What Counts As Truth

The recovery incident and the later deck-enumeration investigation both showed
that physical presence and deck power are **not** enough to conclude that the
firmware sees the deck stack.

Use this truth order:

1. firmware `deck.*` params
2. firmware memory TOC after `cf.mem.refresh()`
3. one successful bounded sensor/deck probe such as
   [probe_multiranger.py](./probe_multiranger.py)

Do **not** use these as proof of successful deck detection:
- a blinking `AI-Deck` LED
- deck power rails being present
- a physically correct-looking stack by itself
- one stale `deck.*` snapshot captured during host USB instability

Why this matters:
- the `AI-Deck`, `Flow`, and `Multi-ranger` can be powered while the firmware
  still reports them missing
- the converse is also true: once the firmware re-enumerates the deck memories,
  the correct post-recovery move is to trust the firmware state instead of an
  earlier bad snapshot

## Deck Detection: Verified Diagnostic Order

If a live craft suddenly reports `deck.* = 0`, do this in order before you call
it a hardware failure:

1. wait for the normal `param.all_updated` state before reading `deck.*`
2. read the raw firmware params directly, not only a wrapper script
3. run `cf.mem.refresh()` and inspect the memory TOC
4. check for `TYPE_1W` deck memories and their board names/revisions
5. run the official bounded probe once through
   [probe_multiranger.py](./probe_multiranger.py)
6. only after those steps treat the craft as genuinely not seeing the deck
   stack

Why this order is now in the runbook:
- we saw an `all-zero` `deck.*` state during recovery-era diagnostics
- later on the same recovered craft, the direct memory refresh and official
  probe both showed the stack again:
  - `deck.bcAI = 1`
  - `deck.bcFlow2 = 1`
  - `deck.bcMultiranger = 1`
  - `deck.bcZRanger2 = 1`
  - one-wire memories for `bcMultiranger rev E`, `bcFlow2 rev A`, and
    `bcAI rev C1`
- that means a single bad snapshot is not enough evidence to conclude a dead
  deck stack

Practical lesson:
- treat `deck.* + memory TOC + bounded probe` as the authoritative combined
  acceptance signal
- do not escalate to physical rework or board-level blame from one bad param
  read alone

## Deck Detection: Separate Deck Presence From Sensor Coverage

Deck presence and readable sensor data are related but not identical checks.

Examples:
- `bcMultiranger = 1` proves the firmware sees the deck
- a successful `probe_multiranger.py` run proves that at least the sampled
  runtime data path is alive
- some directions can still be missing in one probe run without invalidating
  the fact that the deck itself is enumerated

For operator decisions:
- use deck presence to decide whether the stack is electrically enumerated
- use bounded probe data to decide whether the stack is ready for the next
  mission or hover primitive

## Host Failure Pattern: Crazyradio USB Busy During Fast Reconnect Loops

We also saw a second host-only failure mode during deck diagnostics:
- repeated immediate reconnect loops can collide with the host USB stack and
  return `usb.core.USBError: [Errno 16] Resource busy`

Treat this as a host transport error, not as new evidence that the deck stack
vanished again.

Operational rule:
- if the host is throwing `Resource busy` or `No such device` on the
  `Crazyradio`, stop tight reconnect loops
- re-run one clean probe after the radio path is idle
- prefer one successful direct param/memory read plus one successful official
  probe over a noisy burst of reconnect attempts

## Escalation Path 1: STM32 DFU

Use this only when the `STM32` itself is broken or the craft is in the `MBS`
/ USB DFU path.

Verified incident artifacts:
- `/tmp/cf-recovery/cf2loader-1.0.bin`
- `/tmp/cf-recovery/cf2-2025.12.1.bin`

Verified addresses:
- `STM32 bootloader` at `0x08000000`
- `STM32 firmware` at `0x08004000`

Important lesson:
- restoring the `STM32` alone did **not** fully restore normal operation
- it was a necessary narrowing step, but not the final cure
- once the `nRF` bootloader became reachable again, the official radio ZIP path
  was still the decisive step

## Escalation Path 2: Direct nRF51 SWD

Use this only if the radio bootloader is unreachable.

Required hardware:
- a real `SWD` probe such as `ST-LINK/V2`, `STLINK-V3MINIE`, or `J-Link`
- the soldered `nRF` debug connector on the `Crazyflie`
- `Bitcraze Debug Adapter Kit`
- `Reset selector -> SWDIO`

Important negative learning:
- `Olimex ARM-USB-TINY-H + ARM-JTAG-20-10` is **not** the same thing as a
  valid `nRF51 SWD` path
- that bundle is appropriate for other Bitcraze JTAG use cases, especially
  `AI-Deck GAP8`, but not for direct `nRF51` SWD on its own

Official references:
- https://www.bitcraze.io/documentation/repository/crazyflie2-nrf-firmware/master/development/starting_development/
- https://www.bitcraze.io/products/debug-adapter-kit/

What we learned from the failed escalation work:
- direct `SWD` probing is useful for narrowing the state
- it is not the preferred operator path when the radio bootloader can still be
  recovered

## Verified nRF Radio ACK Failure Mode

This incident exposed a second, narrower `nRF` failure mode after the `STM32`
side was already correct:

- the `STM32` bootloader region matched the official `cf2loader-1.0.bin`
- the `STM32` app region matched the fixed Twinr custom build bit-for-bit
- the `Crazyradio 2.0` was updated to the official `1.2.1` firmware
- the `nRF` flash layout, platform data, MBS region, and UICR values all
  matched the official `Crazyflie 2.1` layout
- raw `Crazyradio` packets reached `RADIO_IRQHandler` on the live `nRF` with
  valid CRC and a correct unicast address match
- despite that, the normal `Crazyradio` scan and connect path still saw no
  usable ACKs

That proved the remaining blocker was not:
- a bad `STM32` image
- a dead `Crazyradio`
- a broken `nRF` flash map
- or a dead receive path

The remaining blocker was the `nRF` radio return path after a valid receive.

### Exact Recovery That Restored ACKs

The recovery that fixed this exact state was:

1. keep the official `nRF` bootloader, MBS, and UICR layout intact
2. rebuild the official `crazyflie2-nrf-firmware` tag `2025.12` with
   `BLE=0 PLATFORM=cf2`
3. flash only the `nRF` app image to `0x0001B000` over `SWD`
4. let the craft run for more than `10 s` so the `AI`-deck startup delay and
   radio-enable timeout both expire
5. verify that raw `Crazyradio` packets now return ACKs and that the normal
   radio URI reconnects

The exact upstream build command we used was:

```bash
make -C /tmp/cf2-nrf-src clean
make -C /tmp/cf2-nrf-src BLE=0 PLATFORM=cf2
```

The exact flash command we used was:

```bash
openocd \
  -f interface/stlink.cfg \
  -c 'transport select hla_swd' \
  -f target/nrf51.cfg \
  -c 'init' \
  -c 'program /tmp/cf2-nrf-src/_build/cf2_nrf.bin 0x0001b000 verify reset exit'
```

The first post-fix proof was raw `Crazyradio` ACK recovery:
- `scan_interfaces()` returned `radio://0/80/2M`
- raw packets started returning `ack=True`
- the console surfaced normal startup logs again

The second post-fix proof was a real Crazyflie session:
- `twinrFs.*` became readable over the radio link
- `deck.bcAI=1`
- `deck.bcFlow2=1`
- `deck.bcMultiranger=1`
- `deck.bcZRanger2=1`

### Why This Matters

Do not collapse this state into a generic "radio dead" diagnosis.

If the evidence says:
- `STM32` hashes are correct
- `Crazyradio` firmware is current
- `nRF` receives packets with valid CRC
- but no usable ACK comes back

then the next software-first step is not another `STM32` flash. The next step is
to prove or rule out the `nRF` app itself, starting with the official upstream
tag and the `BLE=0` build that restored the ACK path here.

## Separate Path: AI-Deck GAP8 JTAG

This is a different system.

Use:
- `Olimex ARM-USB-TINY-H`
- `ARM-JTAG-20-10`
- `AI-Deck GAP8` connector

Do **not** use it as the default model for `Crazyflie nRF51` recovery.

This separation is backed by:
- [BF000588.yml](../../artifacts/stores/fixreport/reports/BF000588.yml)
- [BF000546.yml](../../artifacts/stores/fixreport/reports/BF000546.yml)

Why it matters:
- the same vendor ecosystem exposes multiple adapters and targets
- mixing `Crazyflie nRF SWD`, `STM32 DFU`, and `AI-Deck GAP8 JTAG` causes bad
  recovery decisions and unnecessary hardware work

## Sequence That Actually Recovered The Bricked Craft

This is the exact high-value sequence from the incident:

1. prove the bad flash came from an unpinned custom full `STM32` image rather
   than an official coordinated Bitcraze release
2. restore the `STM32` side to official artifacts
3. recover the `nRF` bootloader state
4. stop doing partial component guesses once the bootloader is alive
5. flash the official `firmware-cf2-2025.12.1.zip` over `Crazyradio`

## Verified STM32 Deck-Bus Boot Hang Failure Mode

This incident exposed a second, different failure mode that looks similar from
the host side but is **not** a `Crazyradio` or generic `nRF` failure.

Observed pattern:
- raw `Crazyradio` ACKs and `safelink` were healthy
- `scan_interfaces()` returned `radio://0/80/2M`
- the normal `cflib` setup stalled before platform info with
  `protocol_version=-1` and no TOC loading
- direct `LINKCTRL`/`PLATFORM` requests initially returned only null packets

The decisive proof came from `STM32` SWD against the official `2025.12.1`
firmware:
- the bootloader jumped into the app at `0x08004000`
- `systemTask` started
- execution never reached `commInit()`
- the app hung during the second `i2cdevInit(I2C1_DEV)` call
- live halts landed in `sleepus()` while the `I2C1` deck-bus unlock path in
  `i2cdrvdevUnlockBus()` was cycling SCL and waiting for SDA to release

Root cause:
- the `I2C1` deck bus was held low long enough that the official firmware could
  loop forever in the unbounded deck-bus unlock path before `commInit()`
- because `commInit()` never ran, the host saw a radio that ACKed packets but a
  platform path that stayed silent

Proven remediation:
1. patch the Crazyflie firmware so the `I2C1` unlock path is **bounded** and
   returns failure instead of hanging forever
2. bubble that failure back through `i2cdevInit()` and mark the system test
   failed visibly in `systemTask`
3. reflash the patched `STM32` app
4. verify that direct `LINKCTRL/source` returns `Bitcraze Crazyflie` and that
   direct `PLATFORM/getProtocolVersion` returns a real version packet

Repo-local patch:
- `hardware/bitcraze/patches/crazyflie_firmware_i2c1_boot_hang_fail_closed.patch`

Repo-local reproducible build:

```bash
bash hardware/bitcraze/build_patched_crazyflie_firmware.sh \
  --firmware-root /tmp/crazyflie-firmware-clean \
  --expected-firmware-revision 2025.12.1
```

That workflow now:
- requires an exact clean official firmware checkout
- applies the repo-local patch inside a temporary git worktree
- initializes the required firmware submodules
- emits `cf2.bin`, `cf2.elf`, and one build manifest under
  `hardware/bitcraze/build/`

The corresponding non-rotor deck-bus truth probe is:

```bash
/twinr/bitcraze/.venv/bin/python hardware/bitcraze/probe_deck_bus.py \
  --workspace /twinr/bitcraze \
  --json
```

Use that probe after the patched STM32 flash and before any rotor activity. The
probe captures:
- firmware `deck.*` flags
- explicit `cf.mem.refresh()` memory TOC state
- bounded startup console lines

By default it also performs one bounded `STM32`/deck power-cycle first. This is
required for correct deck-truth after any physical reconfiguration: Crazyflie
deck discovery happens at startup, so reading `deck.*` from an already-running
session after hot reattaching a deck can produce a stale false negative.

If all three still say the deck stack is absent, the software-side proof is
done and the next narrowing step is physical: remove or reseat one deck or
interconnect at a time and rerun the same probe after each change until the
missing deck evidence returns.

Practical interpretation:
- if raw radio ACKs are healthy but the platform path stays silent, do **not**
  keep guessing at the `Crazyradio`
- prove whether the `STM32` app is hanging before `commInit()`
- a stuck deck bus can masquerade as a pure radio/connectivity problem
6. verify the normal radio URI comes back and the real `cflib` connection works

That is the path to prefer next time.

## Hard Lessons

1. Never flash an unpinned custom full `stm32-fw` image onto operator hardware.
2. When Bitcraze ships a coordinated release ZIP, prefer that ZIP over a pile of
   hand-picked binaries.
3. When the `nRF` bootloader is alive, do not escalate to `SWD` first.
4. The first question in a mixed Bitcraze setup is always: which target is this
   adapter path actually for?
5. If the host USB layer is unstable, fix the host USB path before drawing
   conclusions about the craft.
6. Restore the coordinated stack first, then component-specific firmware.
7. Do not mix `AI-Deck` recovery assumptions into `Crazyflie core` recovery.
8. Do not treat `LED on` or `deck has power` as proof that the firmware sees
   the deck.
9. Do not call a deck stack dead from one `all-zero deck.*` snapshot without
   checking memory TOC and one clean bounded probe.
11. After changing the physical deck stack, do not trust the old session. Reboot
    the `STM32`/deck rail before concluding that a deck is missing.
10. Do not arm lateral clearance from supervisor/estimator state alone during
    takeoff; ground-start obstacle protection must wait for measured liftoff,
    and a failed takeoff must end in one quiet landed state instead of a
    repeated ground abort loop.

## Anti-Patterns

Do **not** do these again:
- flash untagged custom full `STM32` firmware to the operator craft
- assume `ARM-JTAG-20-10` implies `SWD`
- assume an `Olimex` AI-Deck bundle is automatically the right tool for direct
  `nRF51` recovery
- chase partial `nRF`/`STM32` binaries before trying the official coordinated
  release ZIP once bootloader connectivity exists
- keep retrying the same failing transport without checking whether the host USB
  layer itself is resetting the radio/probe

## Official References

- https://www.bitcraze.io/documentation/system/platform/cf2-components/
- https://www.bitcraze.io/documentation/repository/crazyflie-clients-python/master/functional-areas/cfloader/
- https://www.bitcraze.io/documentation/repository/crazyflie2-nrf-firmware/master/development/starting_development/
- https://www.bitcraze.io/documentation/repository/crazyflie2-nrf-firmware/master/development/architecture/
- https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/development/openocd_gdb_debugging/

## See Also

- [README.md](./README.md)
- [setup_bitcraze.sh](./setup_bitcraze.sh)
- [probe_crazyradio.py](./probe_crazyradio.py)
- [flash_on_device_failsafe.py](./flash_on_device_failsafe.py)
