# social

Social-trigger models, sensing adapters, prompt routing, and visual review for
Twinr's proactive monitor.

This package turns normalized camera, ambient-audio, and PIR-derived state into
scored proactive trigger candidates and optional visual second opinions.

## Responsibility

`social` owns:
- Define the social-trigger domain models and score thresholds
- Evaluate stateful trigger candidates from normalized observation ticks
- Wrap ambient-audio, ReSpeaker XVF3800, legacy OpenAI vision, and local-first AI-camera observations into bounded conservative snapshots
- Preserve conservative ReSpeaker facts such as `assistant_output_active`, `direction_confidence`, `speech_overlap_likely`, and `barge_in_detected`
- Stabilize automation-facing camera snapshots, including person-count/zone anchors, coarse motion, and coarse/fine gesture event surfaces
- Preserve bounded `looking_signal_state/source` metadata on camera snapshots so downstream HDMI/runtime debug surfaces can distinguish face-confirmed `LOOKING` from the cheap body-box proxy without inventing a second camera truth
- Preserve bounded multi-person camera anchors, not just the single primary person box, so higher runtime layers can track who moved last or which visible person is most relevant without treating that as identity
- Treat local camera health faults as `unknown` camera semantics instead of authoritative "no person" ticks, so short IMX500/runtime problems do not instantly erase the last stable person anchor
- Smooth small primary-person center jitter before it reaches the HDMI gaze path, so box wobble does not read as nervous eye movement when the user is standing still
- Keep that center smoothing short enough for live HDMI HCI, so person-following stays calm without adding second-scale lag
- Keep the gesture acknowledgement surface fast enough for HDMI HCI, so changed user symbols like `🖐️` then `👍` are not blocked for several seconds by the slower proactive inspect cadence
- Preserve short explicit fine-hand symbols such as `👍`, `👎`, `👉`, `✌️`, `👌`, or `🖕` across brief motion/dropout jitter so users do not need to freeze unnaturally for the camera path
- Keep per-symbol fine-hand acceptance configurable and bounded, because `👌` and `🖕` are materially easier to confuse than built-in `👍`, `👎`, `✌️`, or `👉`
- Keep default confirmation counts tight enough for sub-second HDMI acknowledgement on the Pi; prefer stricter confidence floors over extra confirmation frames whenever a target gesture still needs to feel realtime
- Let the staged custom gesture model supplement built-in MediaPipe hand symbols instead of globally overriding them, so custom-only `👌` / `🖕` support does not steal `✌️` or `👉` from the built-in recognizer
- Expose a low-latency ReSpeaker signal-only audio snapshot path for HDMI attention refresh, so local gesture/gaze HCI does not inherit the longer ambient PCM sampling window
- Route safety prompts and render bounded evidence facts for proactive prompting
- Buffer recent camera frames and request conservative visual second opinions

`social` does **not** own:
- Proactive monitor orchestration or worker lifecycle
- Delivery-governance cooldown policy after a candidate is emitted
- Fine-hand model training or dataset capture scripts under `hardware/piaicam/`
- Raw hardware driver implementations outside runtime-facing wrappers
- Speech delivery, printing, or long-term proactive planning

## Key files

| File | Purpose |
|---|---|
| `__init__.py` | Package export surface |
| `camera_surface.py` | Debounced camera snapshot, bounded multi-person anchor surface, coarse motion, and rising-edge coarse/fine gesture event surface |
| `gesture_calibration.py` | Bounded per-symbol fine-hand calibration profile loaded from `state/mediapipe/gesture_calibration.json` |
| `engine.py` | Stateful social-trigger scoring engine and normalized vision contract, including visible-person anchor payloads |
| `local_camera_provider.py` | Maps the local IMX500 + MediaPipe adapter onto the social vision contract, including visible-person anchors, motion, and coarse/fine gesture output |
| `observers.py` | Audio, ReSpeaker overlay, and vision observation providers |
| `prompting.py` | Prompt routing and evidence rendering |
| `scoring.py` | Weighted scoring primitives |
| `vision_review.py` | Buffered visual second-opinion reviewer |
| `component.yaml` | Structured package metadata |
| `AGENTS.md` | Local editing rules |

## Usage

```python
from twinr.proactive.social import (
    SocialFineHandGesture,
    SocialObservation,
    SocialTriggerEngine,
)

engine = SocialTriggerEngine.from_config(config)
decision = engine.observe(
    SocialObservation(
        observed_at=now,
        inspected=True,
    )
)
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [../runtime/README.md](../runtime/README.md)
- [../governance/README.md](../governance/README.md)
