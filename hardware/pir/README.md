# pir

Pi-side setup and probe scripts for Twinr's PIR motion sensor input.

## Responsibility

`pir` owns:
- persist PIR GPIO env settings for the Pi runtime
- probe configured or ad-hoc PIR motion lines outside the main runtime

`pir` does **not** own:
- runtime motion monitoring in `src/twinr/hardware/pir.py`
- button GPIO handling outside shared config dependencies
- higher-level proactive behavior policy

## Key files

| File | Purpose |
|---|---|
| [setup_pir.sh](./setup_pir.sh) | Persist PIR GPIO config |
| [probe_pir.py](./probe_pir.py) | Print live PIR motion events |

## Usage

```bash
./hardware/pir/setup_pir.sh --env-file .env --motion 26 --probe
python3 hardware/pir/probe_pir.py --env-file .env --duration 30
```

## See also

- [Top-level hardware README](../README.md)
- [Runtime PIR adapter](../../src/twinr/hardware/pir.py)
