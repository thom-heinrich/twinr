# Display Setup

Twinr uses the `Waveshare 4.2inch e-Paper Module` with the `V2` Python driver.

Standard Raspberry Pi wiring:

- `VCC -> 3.3V`
- `GND -> GND`
- `DIN/MOSI -> GPIO10` (`Pin 19`)
- `CLK -> GPIO11` (`Pin 23`)
- `CS -> GPIO8` (`Pin 24`)
- `DC -> GPIO25` (`Pin 22`)
- `RST -> GPIO17` (`Pin 11`)
- `BUSY -> GPIO24` (`Pin 18`)

Setup and smoke test:

```bash
cd /twinr
sudo ./hardware/display/setup_display.sh
```

Standalone test run:

```bash
cd /twinr
./.venv/bin/python hardware/display/display_test.py --env-file /twinr/.env
```

Run the status-display loop:

```bash
cd /twinr
./.venv/bin/python hardware/display/run_display_loop.py --env-file /twinr/.env
```
