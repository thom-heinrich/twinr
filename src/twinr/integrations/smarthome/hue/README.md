# smarthome.hue

`smarthome.hue` is the first provider package built on Twinr's generic
smart-home contracts.

It owns:

- local Hue bridge HTTPS transport
- bounded Hue event-stream reads
- Hue-to-generic normalization for lights, scenes, motion sensors, and bridge health resources

It does **not** own:

- the generic smart-home request/response contracts
- automation-engine wiring outside `src/twinr/integrations`
- general smart-home policy or catalog logic

## Notes

- This provider is designed for local Hue bridge access, not the remote cloud API.
- It intentionally treats Hue motion and connectivity data as smart-home signals, not as security monitoring.
- The shared generic adapter above this package remains the place where LLM read/control requests and sensor-stream reads meet Twinr's integration contract.
