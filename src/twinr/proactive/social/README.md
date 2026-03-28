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
- Keep the live AI-camera contract bounded and portable inside the main-Pi runtime while treating any older helper-Pi camera transport as legacy-only code rather than a supported productive topology
- Preserve conservative ReSpeaker facts such as `assistant_output_active`, `direction_confidence`, `speech_overlap_likely`, and `barge_in_detected`
- Stabilize automation-facing camera snapshots, including person-count/zone anchors, coarse motion, and coarse/fine gesture event surfaces
- Preserve bounded `looking_signal_state/source` metadata on camera snapshots so downstream HDMI/runtime debug surfaces can distinguish face-confirmed `LOOKING` from the cheap body-box proxy without inventing a second camera truth
- Preserve bounded multi-person camera anchors, not just the single primary person box, so higher runtime layers can track who moved last or which visible person is most relevant without treating that as identity
- Treat local camera health faults as `unknown` camera semantics instead of authoritative "no person" ticks, so short IMX500/runtime problems do not instantly erase the last stable person anchor
- Smooth small primary-person center jitter before it reaches the HDMI gaze path, so box wobble does not read as nervous eye movement when the user is standing still
- Keep that center smoothing short enough for live HDMI HCI, so person-following stays calm without adding second-scale lag
- Keep the gesture acknowledgement surface fast enough for HDMI HCI, so changed user symbols like `🖐️` then `👍` are not blocked for several seconds by the slower proactive inspect cadence, preferring current-frame live-hand evidence over slower person-ROI or full-frame rescues in the user-facing runtime lane
- Preserve short explicit fine-hand symbols such as `👍`, `👎`, `👉`, `✌️`, `👌`, or `🖕` across brief motion/dropout jitter so users do not need to freeze unnaturally for the camera path
- Keep per-symbol fine-hand acceptance configurable and bounded, because `👌` and `🖕` are materially easier to confuse than built-in `👍`, `👎`, `✌️`, or `👉`
- Keep per-symbol acceptance bounded for live HDMI HCI: prefer a short explicit visibility window for deliberate symbols such as `👍`, `👎`, and `✌️` over brittle single-frame publishes, but do not let that grow into multi-second frozen-pose requirements
- Let the staged custom gesture model supplement built-in MediaPipe hand symbols instead of globally overriding them, so custom-only `👌` / `🖕` support does not steal `✌️` or `👉` from the built-in recognizer; the precision-first user-facing fast lane may still disable that custom branch when Pi evidence shows it misses the current-frame deadline
- Expose a low-latency ReSpeaker signal-only audio snapshot path for HDMI attention refresh, so local gesture/gaze HCI does not inherit the longer ambient PCM sampling window
- Share low-risk social-contract normalization helpers across modules without regrowing duplicate coercion code in hot runtime paths
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
| `camera_surface.py` | Legacy-compatible public wrapper for the debounced camera snapshot and rising-edge event surface |
| `camera_surface_impl/` | Internal package that splits camera surface config, models, signal trackers, presence/health resolution, and gesture stabilization into reviewable modules |
| `gesture_calibration.py` | Bounded per-symbol fine-hand calibration profile loaded from `state/mediapipe/gesture_calibration.json` |
| `engine.py` | Stateful social-trigger scoring engine and normalized vision contract, including visible-person anchor payloads |
| `aideck_camera_provider.py` | Bounded continuous AI-Deck still-camera provider that feeds the OpenAI proactive vision classifier from `aideck://` captures |
| `local_camera_provider.py` | Maps the local IMX500 + MediaPipe adapter onto the social vision contract, including visible-person anchors, motion, and a current-frame fast gesture snapshot for HDMI ack/wakeup |
| `remote_camera_provider.py` | Fetches bounded IMX500 observations from the helper Pi over the direct-link HTTP proxy and also supports a `remote_frame` mode where the main Pi runs the hot attention/gesture lifting locally from a coherent helper-side detection-plus-frame bundle, keeping the user-facing gesture lane on the same current-frame fast policy |
| `normalization.py` | Shared low-risk enum, box, and integer coercion helpers reused by `engine.py` and `camera_surface.py` |
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

The productive Twinr runtime is now single-Pi only. `TwinrConfig.from_env()`
expects the AI camera on the main Pi and fail-closes the retired helper-Pi
camera envs instead of deriving `second_pi` / proxy behavior implicitly.

For a direct Bitcraze AI-Deck camera, set `TWINR_CAMERA_DEVICE=aideck://192.168.4.1:5000`.
When no explicit proactive provider override is set, `TwinrConfig.from_env()`
now derives `proactive_vision_provider=aideck_openai`, which routes continuous
proactive inspect frames through the bounded OpenAI still-image classifier while
leaving low-latency local IMX500-specific attention/gesture paths disabled.

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [../runtime/README.md](../runtime/README.md)
- [../governance/README.md](../governance/README.md)
