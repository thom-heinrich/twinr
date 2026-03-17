# hardware/ops

Pi-side operating-system service definitions for Twinr background watchdogs.

## Responsibility

`hardware/ops` owns:
- the authoritative systemd unit that keeps the productive Twinr runtime running on the Pi
- OS-level launch wiring for background checks and runtime processes that must survive shell logout or crashes
- the Pi-side bootstrap entrypoint for self-coding Codex prerequisites

`hardware/ops` does **not** own:
- the watchdog implementation itself; that lives in `src/twinr/ops`
- Twinr runtime orchestration or product logic

## Files

| File | Purpose |
|---|---|
| [twinr-runtime-supervisor.service](./twinr-runtime-supervisor.service) | Productive unit: authoritatively supervise the streaming loop, its display companion, and the remote-memory watchdog together |
| [twinr-web.service](./twinr-web.service) | Productive unit: keep the Twinr web control portal running with managed sign-in |
| [bootstrap_self_coding_pi.py](./bootstrap_self_coding_pi.py) | Reproducibly sync the pinned self-coding Codex bridge/auth and run the remote self-test |

Retired standalone break-glass units are no longer tracked here. If an operator wants to keep local copies around during cleanup, they belong under the ignored top-level `__legacy__/hardware/ops/` folder.

## Install

```bash
sudo systemctl disable --now twinr-streaming-loop.service twinr-remote-memory-watchdog.service twinr-display-loop.service || true
sudo cp hardware/ops/twinr-runtime-supervisor.service /etc/systemd/system/
sudo cp hardware/ops/twinr-web.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now twinr-runtime-supervisor.service twinr-web.service
sudo systemctl status twinr-runtime-supervisor.service
sudo systemctl status twinr-web.service
python3 hardware/ops/bootstrap_self_coding_pi.py
```
