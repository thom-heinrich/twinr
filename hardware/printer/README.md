# printer

Pi-side setup script for Twinr's thermal printer queue and raw paper-output smoke checks.

## Responsibility

`printer` owns:
- detect or accept the printer device URI for CUPS setup
- create the raw Twinr printer queue and optional default queue
- run a short raw print smoke test for operator confirmation

`printer` does **not** own:
- receipt formatting in `src/twinr/hardware/printer.py`
- reminder or print-trigger policy
- audio, display, or GPIO setup

## Key files

| File | Purpose |
|---|---|
| [setup_printer.sh](./setup_printer.sh) | Configure raw CUPS queue |

## Usage

```bash
sudo ./hardware/printer/setup_printer.sh --default --test
sudo ./hardware/printer/setup_printer.sh --queue Thermal_GP58 --device-uri 'usb://Gprinter/GP-58?serial=WTTING%20'
```

## See also

- [Top-level hardware README](../README.md)
- [Runtime printer adapter](../../src/twinr/hardware/printer.py)
