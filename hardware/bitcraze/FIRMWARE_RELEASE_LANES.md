# firmware release lanes

Fail-closed release process for Twinr's `Crazyflie` on-device firmware path.

## Scope

Use this document for:
- building the Twinr on-device failsafe from pinned sources
- promoting one tested artifact from `dev` to `bench` to `operator`
- flashing the correct lane onto the correct device role
- separating custom firmware deployment from official Bitcraze recovery

Do **not** use this document for:
- `AI-Deck` `GAP8` recovery
- ad-hoc operator recovery after a bad flash

Use [CRAZYFLIE_RECOVERY_RUNBOOK.md](./CRAZYFLIE_RECOVERY_RUNBOOK.md) for the
recovery path.

## Lanes

There are exactly four firmware lanes:

| Lane | Purpose | Allowed target |
|---|---|---|
| `dev` | Direct custom firmware development | dedicated dev drone |
| `bench` | Validated custom firmware on bench hardware | dedicated bench drone |
| `operator` | Approved custom firmware for operator hardware | operator drone |
| `recovery` | Official Bitcraze coordinated release ZIP | any drone during recovery |

The lane split is hard-enforced in
[flash_on_device_failsafe.py](./flash_on_device_failsafe.py).

## Build

The custom build path now fails closed unless the Bitcraze firmware checkout is
clean and matches one explicit expected revision.

```bash
bash hardware/bitcraze/build_on_device_failsafe.sh \
  --firmware-root /tmp/crazyflie-firmware \
  --expected-firmware-revision 2025.12.1
```

Outputs:
- `hardware/bitcraze/twinr_on_device_failsafe/build/twinr_on_device_failsafe.bin`
- `hardware/bitcraze/twinr_on_device_failsafe/build/twinr_on_device_failsafe.build-attestation.json`

The build attestation records:
- artifact `sha256`
- exact Bitcraze firmware commit/tag
- expected firmware revision
- Twinr app-tree digest
- toolchain identity

## Promotion

Promotion is now explicit and monotonic:

1. `dev build attestation -> bench release attestation`
2. `bench release attestation -> operator release attestation`

Bench promotion:

```bash
python3 hardware/bitcraze/promote_on_device_failsafe_release.py \
  --artifact hardware/bitcraze/twinr_on_device_failsafe/build/twinr_on_device_failsafe.bin \
  --source-attestation hardware/bitcraze/twinr_on_device_failsafe/build/twinr_on_device_failsafe.build-attestation.json \
  --lane bench \
  --release-id twinr-failsafe-bench-2026-04-11-r1 \
  --approved-by thh \
  --reason "bounded bench validation passed" \
  --validation-evidence test:test_bitcraze_flash_on_device_failsafe.py
```

Operator promotion:

```bash
python3 hardware/bitcraze/promote_on_device_failsafe_release.py \
  --artifact hardware/bitcraze/twinr_on_device_failsafe/build/twinr_on_device_failsafe.bin \
  --source-attestation hardware/bitcraze/twinr_on_device_failsafe/build/twinr_on_device_failsafe.bin.bench.twinr-failsafe-bench-2026-04-11-r1.release.json \
  --lane operator \
  --release-id twinr-failsafe-operator-2026-04-11-r1 \
  --approved-by thh \
  --reason "operator promotion after bench proof" \
  --validation-evidence report:BF000968
```

Operator promotion additionally requires that the original build was pinned to an
exact Bitcraze firmware tag.

## Flash

### Dev

```bash
/twinr/bitcraze/.venv/bin/python hardware/bitcraze/flash_on_device_failsafe.py \
  --lane dev \
  --device-role dev \
  --binary hardware/bitcraze/twinr_on_device_failsafe/build/twinr_on_device_failsafe.bin \
  --attestation hardware/bitcraze/twinr_on_device_failsafe/build/twinr_on_device_failsafe.build-attestation.json
```

### Bench

```bash
/twinr/bitcraze/.venv/bin/python hardware/bitcraze/flash_on_device_failsafe.py \
  --lane bench \
  --device-role bench \
  --binary hardware/bitcraze/twinr_on_device_failsafe/build/twinr_on_device_failsafe.bin \
  --attestation /path/to/twinr_on_device_failsafe.bin.bench.<release-id>.release.json
```

### Operator

```bash
/twinr/bitcraze/.venv/bin/python hardware/bitcraze/flash_on_device_failsafe.py \
  --lane operator \
  --device-role operator \
  --binary hardware/bitcraze/twinr_on_device_failsafe/build/twinr_on_device_failsafe.bin \
  --attestation /path/to/twinr_on_device_failsafe.bin.operator.<release-id>.release.json
```

### Recovery

```bash
/twinr/bitcraze/.venv/bin/python hardware/bitcraze/flash_on_device_failsafe.py \
  --lane recovery \
  --device-role operator \
  --binary /tmp/cf-recovery/firmware-cf2-2025.12.1.zip \
  --expected-sha256 <sha256>
```

`recovery` is intentionally separate from custom firmware. It accepts only the
official Bitcraze coordinated release ZIP / manifest path.

## Interlocks

The policy now blocks these failure modes:

1. Raw custom `stm32-fw` flashes onto operator devices.
2. Bench/operator flashes without an attestation chain.
3. Operator promotion directly from a dev build.
4. Recovery runs that accidentally use a raw custom `.bin`.
5. Builds from uncommitted or revision-drifted Bitcraze firmware sources.

## Architecture

The local safety split stays:
- on-device `C` controller on the `STM32`
- host as supervisor, heartbeat sender, and verifier only

This preserves the one useful deep-level custom path while removing the unsafe
operator deployment lane that caused the brick incident.

## See also

- [README.md](./README.md)
- [CRAZYFLIE_RECOVERY_RUNBOOK.md](./CRAZYFLIE_RECOVERY_RUNBOOK.md)
- [twinr_on_device_failsafe/README.md](./twinr_on_device_failsafe/README.md)
