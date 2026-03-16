# display

Pi-side setup and smoke-test scripts for Twinr's Waveshare e-paper display path.

## Responsibility

`display` owns:
- install and patch the Waveshare vendor driver for Pi use
- persist display GPIO and SPI env settings
- run bounded display smoke commands outside the main runtime

`display` does **not** own:
- runtime rendering logic in `src/twinr/display`
- main Twinr CLI orchestration
- unrelated GPIO, audio, or printer setup

## Key files

| File | Purpose |
|---|---|
| [setup_display.sh](./setup_display.sh) | Install vendor driver and env wiring |
| [vendor_patch.py](./vendor_patch.py) | Download and harden the generated Waveshare vendor package |
| [display_test.py](./display_test.py) | Render one-shot test pattern |
| [run_display_loop.py](./run_display_loop.py) | Run standalone status loop |

## Usage

```bash
sudo ./hardware/display/setup_display.sh --env-file .env
python3 hardware/display/display_test.py --env-file .env
python3 hardware/display/run_display_loop.py --env-file .env --duration 30
```

## See also

- [Top-level hardware README](../README.md)
- [Runtime display package](../../src/twinr/display/README.md)
