# hardware/ops

Pi-side operating-system service definitions for Twinr background watchdogs.

## Responsibility

`hardware/ops` owns:
- systemd units that keep Twinr operational watchdogs running permanently
- OS-level launch wiring for background checks that are not part of the button/audio runtime loop

`hardware/ops` does **not** own:
- the watchdog implementation itself; that lives in `src/twinr/ops`
- Twinr runtime orchestration or product logic

## Files

| File | Purpose |
|---|---|
| [twinr-remote-memory-watchdog.service](./twinr-remote-memory-watchdog.service) | Run the 1 Hz remote ChonkyDB watchdog permanently on the Pi |

## Install

```bash
sudo cp hardware/ops/twinr-remote-memory-watchdog.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now twinr-remote-memory-watchdog.service
sudo systemctl status twinr-remote-memory-watchdog.service
```
