# buttons

Pi-side setup and probe scripts for Twinr's physical GPIO buttons.

## Responsibility

`buttons` owns:
- persist button GPIO env settings for the Pi runtime
- probe configured or ad-hoc button lines outside the main Twinr loops

`buttons` does **not** own:
- runtime button semantics in `src/twinr/hardware/buttons.py`
- PIR motion handling
- higher-level loop orchestration or user interaction policy

## Key files

| File | Purpose |
|---|---|
| [setup_buttons.sh](./setup_buttons.sh) | Persist button GPIO config |
| [probe_buttons.py](./probe_buttons.py) | Print live button events |

## Usage

```bash
./hardware/buttons/setup_buttons.sh --env-file .env --green 23 --yellow 22 --probe
python3 hardware/buttons/probe_buttons.py --env-file .env --configured --duration 15
```

## See also

- [Top-level hardware README](../README.md)
- [Runtime button adapter](../../src/twinr/hardware/buttons.py)
