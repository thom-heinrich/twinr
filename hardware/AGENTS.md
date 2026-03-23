# AGENTS.md - /hardware

## Scope

This directory owns Raspberry Pi setup, probe, and smoke-test scripts for Twinr
buttons, display, mic, Pi AI Camera, PIR motion, and printer peripherals.

Out of scope:
- runtime device adapters in `src/twinr/hardware`
- runtime display code in `src/twinr/display`
- main Twinr loop orchestration or user-facing behavior policy
- direct runtime-only fixes in `/twinr` without leading-repo changes here first

## Key files

- `buttons/setup_buttons.sh` - persist button GPIO env settings and optional probe
- `buttons/probe_buttons.py` - print button events for configured or ad-hoc lines
- `display/setup_display.sh` - configure the active display backend and persist its env wiring
- `display/display_test.py` - render a one-shot display test pattern
- `display/run_display_loop.py` - run the standalone display loop
- `mic/setup_audio.sh` - configure ALSA or PipeWire playback/capture defaults and proactive audio env
- `mic/setup_respeaker_access.sh` - install the XVF3800 udev rule so the runtime user can read host-control without sudo
- `piaicam/fetch_mediapipe_models.py` - stage the official MediaPipe pose, hand-landmarker, and gesture task bundles for Pi runtime use
- `piaicam/capture_custom_gesture_dataset.py` - capture bounded labeled stills for custom gesture datasets
- `piaicam/train_custom_gesture_model.py` - train and export a custom MediaPipe gesture task bundle on the leading repo
- `piaicam/smoke_piaicam.py` - run bounded Pi AI Camera enumeration, capture, and IMX500 smoke phases
- `pir/setup_pir.sh` - persist PIR GPIO env settings and optional probe
- `pir/probe_pir.py` - print PIR state and motion events
- `printer/setup_printer.sh` - configure the raw CUPS queue and optional test print
- `ops/voice_gateway_tcp_proxy.py` - transport-only bridge that exposes a LAN-visible socket and forwards it into an existing thh1986 voice-gateway tunnel
- `servo_kernel/twinr_servo.c` - out-of-tree Raspberry Pi servo kernel module for low-level body-orientation PWM
- `servo_kernel/Makefile` - out-of-tree kernel module build entrypoint
- `component.yaml` - structured metadata and manual-check map

## Invariants

- Setup scripts that mutate OS or `.env` state must stay idempotent for repeated runs against the same target config.
- `mic/setup_audio.sh` must keep split playback/capture routing explicit when output and microphone devices differ.
- Probe and smoke-test scripts must stay bounded; every hardware wait needs a duration or short fixed timeout.
- Scripts here may persist only the hardware keys they own and must not silently rewrite unrelated `.env` entries.
- `display/setup_display.sh` must keep generated vendor files outside tracked source trees under `state/display/vendor/`.
- Hardware bootstrap logic stays here; runtime behavior and hardware abstractions belong in `src/twinr/hardware` or `src/twinr/display`.
- Kernel modules or other OS-facing driver code for Pi peripherals also stay here; keep the runtime-facing policy and control logic out of the module itself.
- Transport bridges under `hardware/ops` must stay transport-only. Do not add
  wakeword, transcript, or fallback-routing logic there.

## Verification

After any edit in this directory, run:

```bash
bash -n hardware/buttons/setup_buttons.sh hardware/display/setup_display.sh hardware/mic/setup_audio.sh hardware/mic/setup_respeaker_access.sh hardware/pir/setup_pir.sh hardware/printer/setup_printer.sh
PYTHONPATH=src python3 -m py_compile hardware/buttons/probe_buttons.py hardware/display/display_test.py hardware/display/run_display_loop.py hardware/piaicam/custom_gesture_workflow.py hardware/piaicam/capture_custom_gesture_dataset.py hardware/piaicam/fetch_mediapipe_models.py hardware/piaicam/smoke_piaicam.py hardware/piaicam/train_custom_gesture_model.py hardware/pir/probe_pir.py
```

If button, Pi AI Camera, PIR, or display probes changed and you are on Pi acceptance hardware, also run:

```bash
python3 hardware/buttons/probe_buttons.py --env-file .env --configured --duration 5
python3 hardware/piaicam/smoke_piaicam.py --profile quick
python3 hardware/pir/probe_pir.py --env-file .env --duration 5
python3 hardware/display/display_test.py --env-file .env
```

## Coupling

`buttons/setup_buttons.sh` or `buttons/probe_buttons.py` changes -> also check:
- `src/twinr/config.py`
- `src/twinr/hardware/buttons.py`
- `hardware/buttons/README.md`

`pir/setup_pir.sh` or `pir/probe_pir.py` changes -> also check:
- `src/twinr/config.py`
- `src/twinr/hardware/pir.py`
- `hardware/pir/README.md`

`display/setup_display.sh`, `display/display_test.py`, or `display/run_display_loop.py` changes -> also check:
- `src/twinr/display/`
- `hardware/display/README.md`
- `state/display/vendor/` path assumptions

`mic/setup_audio.sh` or `mic/setup_respeaker_access.sh` changes -> also check:
- `src/twinr/hardware/audio.py`
- `src/twinr/hardware/respeaker/`
- `hardware/mic/README.md`

`piaicam/capture_custom_gesture_dataset.py`, `piaicam/custom_gesture_workflow.py`, `piaicam/fetch_mediapipe_models.py`, `piaicam/smoke_piaicam.py`, or `piaicam/train_custom_gesture_model.py` changes -> also check:
- `hardware/piaicam/README.md`
- `hardware/component.yaml`
- runtime Pi asset paths and installed `rpicam-*`/IMX500 tooling assumptions

`printer/setup_printer.sh` changes -> also check:
- `src/twinr/hardware/printer.py`
- `hardware/printer/README.md`

## Security

- Do not print unrelated `.env` contents or secrets while confirming hardware writes.
- Keep `sudo`, `apt-get`, CUPS, and `/etc/asound.conf` writes narrowly scoped to the intended files.
- Do not add unbounded long-running loops or background daemons in this directory.

## Output expectations

- Update the matching local README when script names, flags, or persisted env keys change.
- Prefer explicit flags and printed resulting state over interactive prompts.
- Keep setup and probe behavior separate; do not bury runtime logic inside these scripts.
