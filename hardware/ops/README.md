# hardware/ops

Pi-side operating-system service definitions plus leading-repo mirror helpers.

## Responsibility

`hardware/ops` owns:
- the authoritative systemd unit that keeps the productive Twinr runtime running on the Pi
- the dedicated systemd unit that keeps the remote-memory watchdog alive across runtime-supervisor restarts
- OS-level launch wiring for background checks and runtime processes that must survive shell logout or crashes
- the Pi-side bootstrap entrypoint for self-coding Codex prerequisites
- the development-machine watchdog that mirrors the authoritative repo into `/twinr` without deleting Pi-local runtime state

`hardware/ops` does **not** own:
- the watchdog implementation itself; that lives in `src/twinr/ops`
- Twinr runtime orchestration or product logic

## Files

| File | Purpose |
|---|---|
| [twinr-remote-memory-watchdog.service](./twinr-remote-memory-watchdog.service) | Dedicated unit: keep the fail-closed remote-memory watchdog warm and continuously refreshing its artifact |
| [twinr-runtime-supervisor.service](./twinr-runtime-supervisor.service) | Productive unit: authoritatively supervise the streaming loop while consuming the external remote-memory-watchdog artifact |
| [twinr-web.service](./twinr-web.service) | Productive unit: keep the Twinr web control portal running with managed sign-in |
| [bootstrap_self_coding_pi.py](./bootstrap_self_coding_pi.py) | Reproducibly sync the pinned self-coding Codex bridge/auth and run the remote self-test |
| [install_whatsapp_node_runtime.py](./install_whatsapp_node_runtime.py) | Download, verify, and stage the pinned local Node.js runtime under `state/tools/` for the WhatsApp Baileys worker |
| [watch_pi_repo_mirror.py](./watch_pi_repo_mirror.py) | Continuously mirror the leading repo into `/twinr`, detect drift, and preserve Pi-local runtime-only paths such as `.env`, `.venv`, `state/`, and `artifacts/` |

The runtime supervisor intentionally runs as `root` so the productive
streaming loop keeps access to GPIO devices on deployed hosts. The dedicated
remote-memory watchdog also runs as `root` now, so both units share one
runtime-state ownership model and do not flap on root-vs-user lock files
inside `/twinr/state/` or `/twinr/state/chonkydb/`.

Retired standalone break-glass units are no longer tracked here. The dedicated
remote-memory watchdog service is not break-glass; it is the productive owner
for the watchdog so warm remote state survives runtime-supervisor restarts. If
an operator wants to keep local copies of older units around during cleanup,
they belong under the ignored top-level `__legacy__/hardware/ops/` folder.

## Install

These units are Pi-only. Do not install them on development laptops or other
non-Pi hosts, even if those machines also have a `/twinr` checkout.

```bash
sudo systemctl disable --now twinr-streaming-loop.service twinr-display-loop.service || true
sudo cp hardware/ops/twinr-remote-memory-watchdog.service /etc/systemd/system/
sudo cp hardware/ops/twinr-runtime-supervisor.service /etc/systemd/system/
sudo cp hardware/ops/twinr-web.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now twinr-remote-memory-watchdog.service twinr-runtime-supervisor.service twinr-web.service
sudo systemctl status twinr-remote-memory-watchdog.service
sudo systemctl status twinr-runtime-supervisor.service
sudo systemctl status twinr-web.service
python3 hardware/ops/install_whatsapp_node_runtime.py
python3 hardware/ops/bootstrap_self_coding_pi.py
```

## Mirror watchdog

Run the repo mirror watchdog on the development machine, not on the Pi:

```bash
python3 hardware/ops/watch_pi_repo_mirror.py --once
python3 hardware/ops/watch_pi_repo_mirror.py --interval-s 5
python3 hardware/ops/watch_pi_repo_mirror.py --interval-s 5 --metadata-only
```

The mirror is intentionally one-way: `/home/thh/twinr` is authoritative, `/twinr`
is healed toward it, and runtime-only Pi paths stay protected from deletion.
Exact-content drift detection is the default on every cycle; `--metadata-only`
is an explicit throughput tradeoff that falls back to periodic checksum audits.
Protected Pi-local paths use perishable rsync filters so stale nested repo
copies can still be deleted instead of being pinned by an inner `.env` or
similar runtime-only file.
