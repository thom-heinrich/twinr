# bitcraze

Provision a clean Crazyradio/Crazyflie workspace for Twinr-side drone work.

## Responsibility

`hardware/bitcraze` owns:
- install the Crazyradio USB udev rules needed for non-root access
- stage pinned Crazyradio 2.0 firmware assets for compatibility and recovery
- create an isolated Python workspace under `/twinr/bitcraze`
- install the pinned Bitcraze Python packages into that workspace
- probe the connected Bitcraze USB device and confirm whether it is ready for `cflib`

`hardware/bitcraze` does **not** own:
- Twinr runtime orchestration
- drone behavior policy
- direct source-of-truth code changes in `/twinr` without first updating `/home/thh/twinr`

## Compatibility note

The current official `cflib` and `cfclient` releases still expect the classic
Crazyradio PA USB identity (`1915:7777`). If a Crazyradio 2.0 is attached in
its UF2/native state (`35f0:bad2`), this setup path stages both firmware
variants but defaults to the PA-emulation UF2 for immediate Python-library
compatibility.

## Key files

| File | Purpose |
|---|---|
| [setup_bitcraze.sh](./setup_bitcraze.sh) | Install USB access rules, stage firmware assets, create `/twinr/bitcraze`, optionally flash a compatible UF2, and install the pinned Bitcraze Python workspace |
| [probe_crazyradio.py](./probe_crazyradio.py) | Inspect connected Bitcraze USB devices, classify the current mode, and validate the local workspace/`cflib` access path |
| [prepare_olimex_jtag.sh](./prepare_olimex_jtag.sh) | Check or prepare host prerequisites for AI-Deck GAP8 JTAG recovery with the Olimex ARM-USB-TINY-H bundle, including optional `openocd`/`docker` installation and runtime-user group prep |
| [probe_multiranger.py](./probe_multiranger.py) | Connect to the Crazyflie, read the current deck flags, and sample Multi-ranger directions plus supporting Flow/Z-ranger presence for immediate post-install acceptance |

## Usage

```bash
sudo ./hardware/bitcraze/setup_bitcraze.sh --workspace /twinr/bitcraze --runtime-user thh
python3 hardware/bitcraze/probe_crazyradio.py --workspace /twinr/bitcraze
python3 hardware/bitcraze/probe_crazyradio.py --workspace /twinr/bitcraze --json
sudo ./hardware/bitcraze/prepare_olimex_jtag.sh --workspace /twinr/bitcraze --install-apt --ensure-user-groups
/twinr/bitcraze/.venv/bin/python hardware/bitcraze/probe_multiranger.py --workspace /twinr/bitcraze --require-deck multiranger --json
```

Once the AI-Deck WiFi streamer is flashed, Twinr can treat it as a normal
still camera by setting `TWINR_CAMERA_DEVICE=aideck://192.168.4.1:5000`. On
hosts that already sit on the AI-Deck AP, the standard Twinr camera capture
and vision-request entrypoints read one bounded frame directly from the deck
stream. On single-WiFi hosts with `nmcli`, Twinr now performs one bounded
handover to the AI-Deck AP for the capture and then restores the previous WiFi
connection before the upstream OpenAI/runtime path continues.

When the Pi receives a DHCP lease on `192.168.4.x` and TCP connect to
`192.168.4.1:5000` succeeds but no bytes arrive, the remaining blocker is the
AI-Deck streamer itself rather than Twinr's network handover. In that state,
power-cycle the drone and prefer the Bitcraze station-mode setup over AP mode
for longer-running Twinr vision experiments.

For the incoming Olimex ARM-USB-TINY-H bundle, Twinr now has a dedicated host
prep script that can stage the JTAG recovery workspace, install missing
`openocd`/`docker` prerequisites, and ensure the runtime user is in the
expected serial/USB groups before the hardware is attached.

For the incoming Multi-ranger deck, Twinr now has a dedicated bounded probe
that checks `deck.bcMultiranger`, `deck.bcFlow2`, `deck.bcZRanger2`, and
samples `front/back/left/right/up/down` range data over the normal radio URI.
That gives us an immediate acceptance test as soon as the deck is mounted.

Run the Bitcraze probes with `/twinr/bitcraze/.venv` rather than the repo-local
`.venv`. The Bitcraze stack currently wants a newer `numpy` line than Twinr, so
the isolated Bitcraze workspace remains the supported runtime path.

## Generated workspace

The setup script creates a reproducible runtime workspace under
`/twinr/bitcraze` containing:
- `.venv/` with pinned `cflib` and `cfclient`
- `firmware/` with staged UF2 assets and any captured `CURRENT.UF2` backups
- `README.md` with the exact pinned versions and the local probe commands

## See also

- [hardware/README.md](../README.md)
- [hardware/AGENTS.md](../AGENTS.md)
