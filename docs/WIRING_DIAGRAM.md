# Wiring Diagram

Compact wiring overview for Twinr's current embodied prototype direction:
HDMI mini-screen, Pi AI Camera, ReSpeaker XVF3800, buttons, PIR, thermal
printer, and optional body-rotation servo.

The older Waveshare e-paper path is not shown here.

```mermaid
flowchart TD
    PSU5["5V / 3A+ Pi Power"] --> PI["Raspberry Pi 4 / 5"]
    HDMI5["5V screen power<br/>if required"] --> SCREEN["4\" HDMI mini-screen"]
    PI -->|"HDMI"| SCREEN
    PI -->|"CSI ribbon"| CAM["Pi AI Camera (IMX500)"]
    PI -->|"USB"| RES["ReSpeaker XVF3800"]
    RES -->|"audio out / compatible playback path"| SPK["Speaker / powered speaker"]
    PI -->|"USB data"| PRN["Thermal printer<br/>DFRobot DFR0503-EN"]
    PRNPSU["9-24V printer PSU"] --> PRN
    PI -->|"GPIO23"| BTN1["Green button"]
    PI -->|"GPIO22"| BTN2["Yellow / Print button"]
    BTN1 -->|"other side"| GND["Shared GND"]
    BTN2 -->|"other side"| GND
    PI -->|"GPIO26"| PIR["PIR OUT"]
    PIRV["PIR VCC"] --> PIR
    PIR --> GND
    PI -->|"USB"| MAE["Pololu Mini Maestro (preferred)"]
    MAE -->|"servo signal"| SERVO["Body rotation servo"]
    SRVPSU["Servo PSU"] --> SERVO
    PI -.->|"optional direct SIG on GPIO18"| SERVO
    GND --> PIR
    GND --> MAE
```

## Current GPIO map

| Function | BCM GPIO | Physical pin |
|---|---:|---:|
| Green / "Hey" button | `GPIO23` | `16` |
| Yellow / "Print" button | `GPIO22` | `15` |
| PIR motion `OUT` | `GPIO26` | `37` |
| Optional direct servo `SIG` | `GPIO18` | `12` |

## Wiring notes

- Raspberry Pi GPIO must stay at or below `3.3V`.
- Use shared ground where the Pi and an external controller/sensor need the same reference.
- Power the thermal printer from its own supply.
- Power a large servo from its own supply.
- Prefer the `Pololu Mini Maestro` path for calm, safer servo control.
- Some HDMI mini-screens need separate USB power in addition to HDMI.
- The ReSpeaker playback side depends on the exact board / speaker setup; use a compatible speaker or powered playback path.

## Bring-up commands

```bash
cd /twinr
sudo hardware/display/setup_display.sh --env-file /twinr/.env --driver hdmi_wayland
sudo hardware/mic/setup_audio.sh --env-file /twinr/.env --test
sudo hardware/mic/setup_respeaker_access.sh
hardware/buttons/setup_buttons.sh --green 23 --yellow 22
hardware/pir/setup_pir.sh --motion 26 --probe
sudo hardware/printer/setup_printer.sh --default --test
python3 hardware/piaicam/smoke_piaicam.py
sudo hardware/servo/setup_pololu_maestro.sh
```
