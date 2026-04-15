# hardware

`hardware` owns Twinr's low-level audio, GPIO, bounded servo output, camera,
printer, local voice-profile adapters, and the bounded local
household-identity building blocks that combine portrait, voice, and explicit
feedback on device.

## Responsibility

`hardware` owns:
- capture and play bounded audio for conversation and ambient sensing
- lift under-driven PCM16 activation-ack playback to an audible level before it reaches the Pi speaker path when provider TTS arrives too quiet
- reassert the XVF3800 Linux playback mixer during runtime startup so a drifted ReSpeaker card does not leave spoken replies near-muted even when Twinr's own softvol stage is already at 100 percent
- monitor GPIO buttons and PIR motion across supported Pi backends
- terminate CLI-backed GPIO helper processes with the owning Twinr parent on Linux so aborted SSH/`sudo` debug runs cannot strand button-line locks across later productive starts
- drive one bounded body-orientation servo from an already-derived normalized attention target without owning speaker/person targeting policy, including calibrated center/invert controls, Twinr's custom kernel-module servo sysfs contract when available, kernel-PWM sysfs output when the GPIO exposes hardware PWM, `pigpio` as a secondary hardware-timed option, a calmer `lgpio.tx_pwm` software-PWM fallback ahead of the older jitter-prone `lgpio.tx_servo` helper, or a Pololu Mini Maestro command-port path over USB serial once the controller itself is configured for `USB_DUAL_PORT`/`USB_CHAINED`, plus soft end-stop margins, explicit target smoothing, visible-target latching so in-frame micro-jitter does not trigger constant re-justification, an optional exit-only mode where visible users do not trigger physical motion until either the image-exit trajectory is confirmed or the tracked person reaches a calibrated side-departure threshold long enough to justify one monotone pursuit, plus an explicit continuous-rotation control mode that keeps a bounded virtual heading estimate and maps heading error to calm time/speed pulses instead of pretending a 360-degree servo supports absolute position, a tiny persisted continuous-servo state file so an operator hand-set `0°` reference can survive restarts, an explicit `hold_until_armed` startup mode that keeps the Maestro released instead of blindly leaving that reference pose, an explicit visible-box edge gate so exit-only pursuit only starts when the authoritative person geometry is actually near the frame boundary instead of merely off-center, one monotone exit pursuit with a configurable off-center degree clamp, a short configurable settle-hold so loaded hardware can physically complete the exit move before release, centered visible reacquire cooldown once the user is back near the middle of frame, calm return to a rest position after longer absence with a dedicated slower rest-motion profile, acceleration- and jerk-limited motion shaping, small command hysteresis against dead-band chatter, startup seeding from the currently active writer's remembered pulse width so any non-center remembered pose gets recentered before exit-only waiting resumes, an exact-center snap before idle release so the remembered neutral state is not left skewed, calm idle release near neutral, and release-after-settle only for non-visible or projected target states so live visible follow stays physically engaged instead of silently parking on a released off-center pulse
- in exit-only mode, allow one coarse periodic visible-user recenter against a latched target after a configurable off-center dwell interval, so Twinr can quietly re-center a user who stayed visibly off-axis for a long time without regressing into continuous in-frame jitter
- fail closed when startup sees a proven conflicting GPIO18 servo environment such as a foreign `pwm-pio` overlay/process chain, so the HDMI face/runtime can stay up while the servo disables itself instead of jittering against a stolen line
- capture still photos from V4L2 cameras, from a Bitcraze AI-Deck WiFi stream, or through an explicit `rpicam://` or `libcamera://` Pi-camera backend, or fetch the same still-photo contract from a dedicated peer Pi over a bounded HTTP snapshot proxy
- reach one bounded external drone daemon over HTTP so Twinr can queue high-level inspect missions, read state, cancel work, and enforce manual-arm-only safety gates without ever sending direct flight-control commands
- own one profile-driven Crazyflie runtime-telemetry lane that turns bounded cflib logs, param snapshots, link statistics, and `twinrFs` status into typed drone-runtime state for workers and the daemon
- own one deterministic Crazyflie hover-replay lane plus a strict CrazySim adapter so bounded hover bugs can be reproduced and regression-tested without live flight
- persist local portrait identities with multiple reference images per user
- perform bounded local portrait matching from enrolled identities plus still-camera captures
- persist multi-user local household voice identities, assess current-turn audio against enrolled household members, and expose read-only bounded profile snapshots for the live gateway's familiar-speaker wake bias
- coordinate one bounded local household identity manager across portrait matching, multi-user voice matching, explicit confirm or deny feedback, and short session history
- run bounded hybrid IMX500 + MediaPipe camera inference for proactive sensing, with IMX500 always-on gating, MediaPipe pose enrichment, dedicated hand-landmark ROI work, stable full-frame gesture recognition, coarse motion, coarse-arm gestures, and bounded fine-hand gesture output
- preserve bounded multi-person IMX500 person anchors in the camera contract so downstream runtime layers can track visible people continuously instead of collapsing every frame to a single primary box
- supplement the IMX500 person path with bounded local YuNet face anchors when SSD only sees zero or one person, so table-side multi-person attention can still recover more than one visible target without cloud vision
- let the dedicated HDMI gesture lane use those face anchors to recover a human gesture target when the IMX500 primary body box drifts onto furniture or another non-human false positive
- when a YuNet face falls inside an existing person box, retarget that visible-person anchor to the face box for downstream eye-follow, so Twinr looks toward the head instead of the torso centroid while keeping the primary body box for body-pose and presence semantics
- surface IMX500 detection/runtime failures explicitly instead of collapsing them into healthy-looking empty no-person frames, so downstream gaze and gesture consumers can distinguish camera faults from real absence
- expose bounded Pi-side tuning for live hand-gesture responsiveness, including camera FPS, hand-detection/tracking thresholds, gesture score floors, and hand ROI sizing
- keep local AI-camera metadata waits and pose-refresh reuse windows short enough for interactive face-follow and gesture acknowledgement; multi-second waits or stale 10s+ pose reuse are not acceptable for the Pi HCI path
- auto-enable the staged local custom gesture model at `state/mediapipe/models/custom_gesture.task` when it exists, so the Pi can switch from the generic built-in recognizer to a device-trained `thumbs_up`/`thumbs_down`/`peace_sign` model without another env override
- keep custom gesture score thresholds in Twinr's post-recognition arbitration instead of recognizer-construction options, so staged device-trained labels stay stable across MediaPipe runtime variants
- treat negative custom labels such as `none` as authoritative suppression, so a weak custom symbol does not win when the classifier itself prefers "no gesture"
- normalize MediaPipe-bound image buffers centrally so full frames and ROI crops reach the Pi runtime with supported dtype and contiguous layout
- keep multi-ROI MediaPipe tasks monotonic by advancing and reserving timestamps across candidate inference within a frame
- keep MediaPipe gesture recognition latency bounded by preferring ROI gesture recognition when concrete wrist-localized hand crops already exist, and only falling back to the stable full-frame recognizer when ROI crops did not yield a real symbol
- run the dedicated ROI hand-landmark worker in MediaPipe IMAGE mode, because upper-body and wrist crops are independent images rather than one temporally stable video stream; wrist-localized crops stay first for efficiency, but bounded upper-body/full-body rescue crops remain available so weak wrist hints do not suppress hand recovery entirely
- expose a cheap attention-only AI-camera observation that reuses IMX500 person detection plus optional face-anchor supplementation but skips the heavy MediaPipe gesture stack, so HDMI eye-follow can stay reactive while explicit gesture acknowledgement uses the slower full path separately
- expose a dedicated live-stream gesture observation that reuses the RGB camera session but keeps the hot path close to the official Pi-style pattern: one thin built-in live recognizer for the Pi-critical symbols `thumbs_up`, `thumbs_down`, and `peace_sign`, plus a product-specific custom recognizer for the same three labels when staged, bounded primary-person/hand-ROI fallback, and one final whole-frame hand rescue when that fast path still returns `none`
- route long-lived MediaPipe gesture inference through one shared native-heap trim guard so glibc-retained TensorFlow Lite/MediaPipe pages get returned periodically instead of pushing the Pi into swap-heavy RSS runaway
- keep that fast-path whole-frame rescue narrow: only escalate there when the person-ROI hand-landmark worker already localized a hand but the ROI gesture classifier still stayed at `none`, so Twinr can recover brittle `peace_sign` frames without paying the full-frame cost on every miss
- require a strong current-frame live score or one short bounded repeat before a weak live-stream thumb/peace label becomes authoritative in that HDMI fast path, so random single-frame classifier noise does not immediately surface as user-visible acknowledgement
- persist bounded gesture-candidate QA captures from that live-stream lane, with cooldown and retention, so operator debugging can inspect exactly what the Pi saw when Twinr thought a pose or symbol might be present
- configure the official MediaPipe canned gesture classifier options to denylist the generic `None` label while still enforcing Twinr's score thresholds, so built-in hand symbols are not shadowed by the classifier's own fallback class before runtime arbitration
- allow only short-lived primary-person and hand-box hint reuse inside that gesture lane so brief detector flickers do not make the same gesture randomly disappear, while still keeping user-facing HDMI symbol acknowledgement independent from the broader pose/social observation stack
- keep ROI hand crops context-rich and square enough for gesture recognition, instead of collapsing landmark detections into overly brittle slivers that drop finger symbols on real Pi framing
- derive ROI gesture-recognition crops from the hand landmarks projected back onto the full-resolution frame, so the symbol classifier keeps enough pixels and context instead of re-reading a second low-resolution nested crop
- format and submit bounded receipt print jobs
- probe ReSpeaker XVF3800 runtime state and expose typed host-control signals
- derive conservative XVF3800 runtime facts such as direction confidence and busy-state interruption hints
- schedule heavy XVF3800 host-control polling more slowly than the cheap ambient-audio path so Pi runtime loops stay bounded while idle
- prove bounded readable-frame capture before higher runtime layers trust a targeted microphone path as voice-ready
- keep short-lived XVF3800 ambient/self-test/listen helpers in the required duplex state by opening one bounded silent playback guard when capture runs outside the long-lived voice orchestrator
- wait a short bounded window for transient XVF3800 USB/ALSA re-enumeration to clear before live listen capture fails a turn outright
- resolve one calm XVF3800 indicator contract and drive bounded runtime-owned ring pulses for waiting, listening, thinking, speaking, and error states without twitching on weak audio evidence
- attach a per-claim ReSpeaker confidence/source contract that later runtime and memory layers can inspect without guessing
- derive conservative XVF3800 non-speech and background-media suppression facts from the same capture path
- strip display-only env from ALSA subprocesses and, on the productive Pi root runtime, selectively preserve the logged-in user's audio-session env on the proven XVF3800/ALSA paths that require it
- keep productive root-owned ReSpeaker playback on the Twinr softvol/hardware ALSA paths inside that same borrowed-session contract, because the Pi can otherwise hang `aplay` at drain/finalize even when user-owned playback works
- persist and score local voice profiles on device

`hardware` does **not** own:
- hardware-loop orchestration or user interaction policy
- display rendering or e-paper control
- OS/bootstrap setup scripts under `hardware/`
- provider STT/TTS/model logic

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Narrow package export surface |
| [audio.py](./audio.py) | Bounded audio capture, readable-frame probing, and playback, including the productive Pi borrowed-session env opt-in for root-owned ReSpeaker output subprocesses |
| [audio_env.py](./audio_env.py) | Sanitize child-process env for ALSA helpers while allowing explicit root-owned borrowed-session audio on productive Pi paths that need the logged-in user's audio runtime |
| [respeaker_playback_mixer.py](./respeaker_playback_mixer.py) | Bounded runtime repair for the known XVF3800 Linux playback-mixer drift that can make spoken output too quiet |
| [respeaker_duplex_playback.py](./respeaker_duplex_playback.py) | Temporary silent-playback guard for bounded XVF3800 capture helpers that need the hardware's duplex playback state outside the long-lived voice orchestrator; productive voice guards now inherit a Linux parent-death contract so aborted SSH or local debug runs cannot strand orphan `/dev/zero` owners, and a busy playback device is treated as a hard contract failure instead of an "external owner" success |
| [respeaker_capture_recovery.py](./respeaker_capture_recovery.py) | Bounded XVF3800 capture-recovery helper that first waits for transient ALSA recovery and then, on proven warm-stall states, can issue the official host-control REBOOT workaround before retrying |
| [camera_ai/](./camera_ai/) | Internal SoC package for camera contracts, IMX500 runtime, MediaPipe runtime, pose, motion, gesture modules, and the decomposed local adapter internals |
| [camera_ai/adapter.py](./camera_ai/adapter.py) | Stable compatibility facade for the internal local AI-camera adapter import path while the runtime logic lives under `camera_ai/adapter_impl/` |
| [camera_ai/adapter_impl/](./camera_ai/adapter_impl/) | Internal AI-camera adapter package split across observation entrypoints, pose resolution, gesture targeting, runtime lifecycle, and cache/error helpers |
| [camera_ai/live_gesture_pipeline.py](./camera_ai/live_gesture_pipeline.py) | Dedicated low-latency MediaPipe live-stream gesture lane for HDMI symbol acknowledgement |
| [camera_ai/gesture_candidate_capture.py](./camera_ai/gesture_candidate_capture.py) | Bounded JPEG + JSON capture helper for manual optical QA of suspected gesture frames |
| [ai_camera.py](./ai_camera.py) | Stable compatibility facade for the public local AI-camera adapter surface |
| [ai_camera_diagnostics.py](./ai_camera_diagnostics.py) | Bounded Pi-facing diagnostics for pose candidate selection and keypoint support |
| [aideck_camera.py](./aideck_camera.py) | Bounded Bitcraze AI-Deck CPX-over-TCP still-frame adapter that can temporarily join the deck AP, capture one frame, and restore the previous WiFi link before Twinr resumes upstream work |
| [aideck_wifi.py](./aideck_wifi.py) | Focused `nmcli`-backed WiFi handover helper for one bounded AI-Deck capture on single-WiFi hosts |
| [crazyflie_flow_anchor.py](./crazyflie_flow_anchor.py) | Bounded host-side outer-loop for `flow_relative_hover`: trusted-height filtering plus small body-frame velocity corrections around one local flow anchor |
| [crazyflie_start_contract.py](./crazyflie_start_contract.py) | Single start-envelope contract for Crazyflie missions, including the shared rule that side-facing clearance only becomes hard-blocking once the measured takeoff-active height is reached |
| [crazyflie_hover_replay.py](./crazyflie_hover_replay.py) | Deterministic hover replay lane that loads stored hover reports/traces, replays telemetry against a fake monotonic clock, and records synthetic commander/trace outputs for the real hover primitive |
| [crazyflie_trim_observer.py](./crazyflie_trim_observer.py) | Bounded online trim/disturbance observer that estimates neutral forward/left/height bias during micro-liftoff, hover, and landing-identify phases |
| [crazyflie_telemetry.py](./crazyflie_telemetry.py) | Canonical Crazyflie runtime-telemetry lane: profile-driven log blocks, param snapshots, `twinrFs` status ingestion, link statistics, typed snapshots, and command-vs-observed divergence detection |
| [crazyflie_trusted_state.py](./crazyflie_trusted_state.py) | Shared trusted hover-state projection that fuses height trust, optical-flow confidence, pose/velocity freshness, and supervisor flight state into one outer-loop input |
| [crazysim_adapter.py](./crazysim_adapter.py) | Strict adapter around an operator-managed CrazySim checkout so Twinr can launch a bounded single-agent SITL lane without guessing workspace layout |
| [drone_service.py](./drone_service.py) | Strict mission-level Twinr client for the external bounded drone daemon, including stationary inspect, hover-test, bounded local-inspect mission contracts, and normalized safety/state/telemetry snapshots |
| [hand_landmarks.py](./hand_landmarks.py) | Bounded MediaPipe hand-landmark ROI worker for full-frame-to-hand crop resolution |
| [mediapipe_vision.py](./mediapipe_vision.py) | Stable compatibility facade for the public MediaPipe camera pipeline surface |
| [buttons.py](./buttons.py) | GPIO button monitoring backends, including Linux parent-death guards for long-lived CLI helpers |
| [servo_follow.py](./servo_follow.py) | Stable compatibility facade for the public attention-servo API while the implementation lives in the decomposed `servo_follow_impl/` package |
| [servo_follow_impl/](./servo_follow_impl/) | Internal attention-servo package split across config normalization, GPIO/Pololu writers, motion shaping, target resolution, exit-only behavior, and the top-level controller |
| [servo_continuous.py](./servo_continuous.py) | Focused open-loop planner for continuous-rotation servos that estimates one virtual heading, converts heading error into bounded direction/speed pulses, and keeps that motion model out of the higher follow-state machine |
| [servo_state.py](./servo_state.py) | Tiny JSON-backed runtime-state store for continuous servos, persisting the operator-confirmed virtual heading, bounded heading uncertainty, and explicit hold/arm or return-to-estimated-zero transitions outside the main follow controller |
| [servo_segment_player.py](./servo_segment_player.py) | Exact-duration background segment player for continuous-servo reverse replay, starting one bounded pulse immediately and disabling it at the recorded deadline instead of holding it until the next runtime tick |
| [servo_maestro.py](./servo_maestro.py) | Focused Pololu Mini Maestro command-port adapter that auto-discovers the stable `...-if00` USB serial endpoint when present, falls back to Maestro-tagged `ttyACM*` command ports when by-id links are absent, invalidates stale connections after USB faults, emits compact-protocol target commands, reads back current channel positions for startup alignment, fails clearly when the controller is still in the wrong UART serial mode, and keeps Maestro transport concerns out of the higher servo policy module |
| [servo_peer.py](./servo_peer.py) | Focused legacy peer-Pi Pololu Maestro HTTP client retained for historical helper-Pi artifacts; the active Twinr config now fails closed on that topology |
| [camera.py](./camera.py) | bounded still capture with ffmpeg/V4L2, explicit Pi `rpicam-still` capture through `rpicam://` or `libcamera://` device URIs, plus the legacy snapshot-proxy adapter kept for historical compatibility but no longer accepted by the canonical env config |
| [portrait_identity.py](./portrait_identity.py) | Atomic local portrait-identity store with enrolled reference-image management |
| [portrait_match.py](./portrait_match.py) | Local YuNet/SFace portrait-match adapter with enrollment flow, multi-reference scoring, and temporal fusion |
| [household_voice_identity.py](./household_voice_identity.py) | Multi-user local voice-identity store and matcher for enrolled household members, including bounded read-only snapshot export for the live voice gateway |
| [household_identity.py](./household_identity.py) | Shared local household identity manager for portrait, voice, feedback, and bounded session fusion |
| [pir.py](./pir.py) | PIR motion wrapper on buttons |
| [printer.py](./printer.py) | Receipt formatting and CUPS submission |
| [respeaker/](./respeaker/) | XVF3800 probe, host-control transport, typed primitive snapshots, productive USB voice-capture and mux-contract helpers, calm LED profiles/controller/companion loop, conservative derived signals, indicator semantics, claim-contract builders, scheduled polling wrapper, and runtime signal provider |
| [respeaker/ambient_classification.py](./respeaker/ambient_classification.py) | Conservative non-speech/background-media classification from bounded ambient capture plus XVF3800 speech facts |
| [respeaker/voice_capture.py](./respeaker/voice_capture.py) | Productive XVF3800 USB voice-capture contract that reuses the authoritative ASR-beam lane exposed by the live host-control mux state, translates that attested XMOS output pair into the real 6-channel USB lane, and extracts only that lane from the native stream |
| [respeaker/voice_mux.py](./respeaker/voice_mux.py) | XVF3800 host-control voice-mux contract reader/enforcer that proves `AEC_ASROUTONOFF`, identifies the live ASR output pair, and exposes the authoritative USB-lane mapping for capture |
| [respeaker/pcm_content_classifier.py](./respeaker/pcm_content_classifier.py) | Bounded PCM speech-likeness classifier with automatic bundled openWakeWord Silero model discovery plus support for both legacy Silero state tensors and the current `input/sr/h/c -> output/hn/cn` ONNX signature |
| [respeaker/indicator_policy.py](./respeaker/indicator_policy.py) | Shared calm ring/indicator semantics for listening, mute, and unavailable states |
| [respeaker/led_profiles.py](./respeaker/led_profiles.py) | Calm runtime-state-to-color pulse profiles for the XVF3800 ring |
| [respeaker/led_controller.py](./respeaker/led_controller.py) | Bounded XVF3800 LED writes with deduped effect/color updates |
| [respeaker/led_loop.py](./respeaker/led_loop.py) | Snapshot-driven runtime LED loop that keeps the ring aligned with Twinr status |
| [respeaker/companion.py](./respeaker/companion.py) | Optional Pi-runtime background companion for the XVF3800 LED loop |
| [respeaker/scheduled_provider.py](./respeaker/scheduled_provider.py) | Activity-aware wrapper that keeps cheap audio sensing continuous while throttling expensive XVF3800 host-control refreshes on idle ticks |
| [voice_profile.py](./voice_profile.py) | Local voice profile store and scoring |
| [component.yaml](./component.yaml) | Structured package metadata |
| [AGENTS.md](./AGENTS.md) | Local editing rules |

## Usage

```python
from twinr.hardware.audio import SilenceDetectedRecorder
from twinr.hardware.camera import V4L2StillCamera

recorder = SilenceDetectedRecorder.from_config(config)
camera = V4L2StillCamera.from_config(config)
pcm_bytes = recorder.record_pcm_until_pause(pause_ms=900)
```

The productive Twinr runtime is now single-Pi only. `TwinrConfig.from_env()`
expects the still camera on the main Pi and rejects the retired helper-Pi
camera envs (`TWINR_CAMERA_HOST_MODE=second_pi`,
`TWINR_CAMERA_SECOND_PI_BASE_URL`, `TWINR_PROACTIVE_REMOTE_CAMERA_BASE_URL`,
and `TWINR_CAMERA_PROXY_SNAPSHOT_URL`) instead of silently reviving the old
proxy topology.

For the attention-servo path, `TWINR_ATTENTION_SERVO_DRIVER=auto` now prefers
the Pololu Mini Maestro transport whenever a Maestro channel or Maestro device
is configured. That keeps mixed envs from silently falling back to a Pi GPIO
writer when the operator has already declared the intended Maestro output.

When the current camera is the Bitcraze AI-Deck WiFi streamer, configure
`TWINR_CAMERA_DEVICE=aideck://192.168.4.1:5000`. The same
`V4L2StillCamera.from_config(config)` entrypoint will then read one bounded
AI-Deck frame, convert raw Bayer output into a standard PNG on demand, and
keep the upstream Twinr vision contract unchanged. On single-WiFi hosts with
`nmcli`, Twinr now performs one bounded handover to the AI-Deck AP for the
capture itself and then restores the previous WiFi connection before the
OpenAI/remote-memory path continues.

If Twinr reports that the AI-Deck streamer "accepted the TCP connection but
sent no frame bytes", Twinr has already proven the AP handover, DHCP, and TCP
route to `192.168.4.1:5000`. The remaining blocker is the deck-side WiFi
streamer not emitting image data. Bitcraze tracks this freeze upstream in
`bitcraze/aideck-gap8-examples#150`; station mode on the household WiFi is the
preferred path when stable continuous vision matters.

For Raspberry Pi CSI/libcamera cameras, configure `TWINR_CAMERA_DEVICE` as an
explicit `rpicam://<index>` or `libcamera://<index>` URI. Twinr now fails
closed when a `/dev/video*` node is really the Pi `unicam` lane, because that
node is not the supported still-capture lane for the runtime camera contract.

For the bounded drone path, configure:

```bash
export TWINR_DRONE_ENABLED=true
export TWINR_DRONE_BASE_URL=http://127.0.0.1:8791
export TWINR_DRONE_REQUIRE_MANUAL_ARM=true
export TWINR_DRONE_MISSION_TIMEOUT_S=45
```

Twinr stays above the flight layer:

```python
from twinr.hardware.drone_service import DroneServiceConfig, RemoteDroneServiceClient

drone = RemoteDroneServiceClient.from_config(DroneServiceConfig.from_config(config))
mission = drone.create_inspect_mission(target_hint="regal", capture_intent="object_check")
state = drone.state()
```

The drone state surface is now explicitly richer than pose and manual-arm
status. `state.telemetry` carries the daemon's bounded live runtime summary:
- deck flags and catalog counts
- battery and charging state
- downward and directional range state
- radio and link-health observations
- `twinrFs` status
- `twinrFs` takeoff-debug truth such as debounced range/flow/attitude counters,
  bounded truth-stale counts, and takeoff-truth flap counts
- command-vs-observed divergence events such as failed takeoff confirmation or
  host-vs-`twinrFs` takeoff conflict during ascent
- live hover-guard divergence events such as excessive roll/pitch, drift,
  speed, or missing touchdown confirmation

Twinr does not try to stream the full Crazyflie firmware catalog continuously.
The shared telemetry runtime instead uses explicit bounded profiles such as
`operator`, `hover_acceptance`, `inspect_local_zone`, and `forensics`, all
owned centrally in [crazyflie_telemetry.py](./crazyflie_telemetry.py).

The accepted v1 safety contract is:
- Twinr only submits bounded inspect missions; the single explicit exception to "no direct roll/pitch/yaw/thrust commands" is the hardware-only bounded `hover_test` vertical bootstrap, which uses a short closed-loop raw-thrust/manual lift solely until fresh `zrange`, vertical rise, and flow liveness are proven
- the daemon must expose `manual_arm_required=true` by default
- arming goes through the daemon's local ops route, not through free-form voice
- the default runtime slice remains `stationary_observe_only`, so the daemon proves mission/state/evidence plumbing before motion is enabled
- the first live motion primitive is a separate bounded `hover_test` mission, and it is only accepted when the daemon was explicitly started in `bounded_hover_test_only`
- the first bounded autonomous mission is `inspect_local_zone`, and it is only accepted when the daemon was explicitly started in one of the bounded local-flight modes
- `inspect_local_zone` remains mission-bounded: it may do at most one bounded short-range translation before capture and landing, never arbitrary free-flight routing
- the first hover path now also requires the Twinr on-device Crazyflie failsafe app (`twinrFs`) by default, so heartbeat-loss, low-battery, and clearance-triggered safe-land behavior lives on the aircraft in C instead of depending on the host process
- the current hover lane is `flow_relative_hover`, not absolute `position_hold`; Twinr now adds one bounded host-side anchor-hold outer-loop plus shared trusted-state projection and one continuous trim/disturbance observer, but true stationary position hold still remains a separate positioning track
- the start contract is now staged `prearm -> micro_liftoff -> trim_identify -> qualified_hover -> landing_identify -> touchdown`; on real hardware `micro_liftoff` is a bounded closed-loop vertical bootstrap and the worker must not advance into nominal hover until fresh `zrange` rise plus flow liveness are proven and the trim observer has converged on a bounded neutral hover bias
- `micro_liftoff` no longer uses a separate hardcoded `0.4 s` takeoff confirmation window; it reuses the configured takeoff confirmation timeout so conservative live hops can reach trim identification before being judged unstable
- the bootstrap lane now emits per-tick trace decisions that expose range freshness, flow freshness, reference progress, the explicit progress-drive term, raw vs clamped thrust, remaining thrust headroom, and whether the controller actually hit its bounded ceiling before abort; one failed run should be enough to distinguish under-command from dead truth
- the hardware bootstrap is no longer purely reference-limited; when real `zrange` rise is still missing it now drives thrust deterministically toward the configured ceiling, so a failed takeoff can be classified cleanly as "ceiling reached without lift proof" instead of silently timing out at some lower command
- `run_hover_test.py` no longer trusts one arbitrary `pm.state` or supervisor packet during preflight; it samples one bounded start-state window, uses the newest sample as authoritative, and fail-closes explicitly on start-state flapping such as `power.state`, `can_arm`, `can_fly`, or `is_armed` toggling during that window
- the on-device mission lane is now cleaner internally as well: the STM32 owner keeps mission orchestration in `mission_control`, while pure `vertical_control` and `disturbance_control` helpers own hover-thrust estimation, takeoff progress classification, and lateral recover-vs-abort classification
- the shared telemetry surface now exposes that on-device reasoning directly through `twinrFs` fields such as `tkPgCls`, `tkBatMv`, `distSev`, and `distRec`, so host tooling can observe the firmware owner instead of rebuilding controller state heuristically

Twinr now also has a sim-first debug lane for Crazyflie hover work:
- `crazyflie_hover_replay.py` replays persisted hover reports and optional phase traces through the same bounded hover primitive used in live runs, including the hardware-only vertical bootstrap when replay runs under the hardware runtime contract
- `crazyflie_hover_recovery.py` measures whether transient SITL disturbances actually produce the expected opposing recovery commands and bounded resettling
- `crazyflie_hover_recovery.py` also evaluates the replayed stability-guard contract, so recover-vs-abort scenarios are judged by explicit guard state/codes instead of only by post-flight outcome text
- `crazysim_adapter.py` validates and launches one operator-managed CrazySim checkout without vendoring or guessing its layout
- `run_hover_sim.py` runs the real bounded hover worker end-to-end against CrazySim, including the explicit `sitl` touchdown contract for simulator-owned height confirmation and a bounded real CFLib `SyncCrazyflie` readiness probe before the worker starts; MuJoCo acceptance stays headless by default and only enables the passive viewer on explicit operator request
- `run_hover_sim_scenarios.py` mutates one baseline into adversarial SITL scenarios such as drift bias, optical-flow collapse, trusted-height disagreement, attitude spikes, obstacle proximity, and transient recovery disturbances
- `hover_physical_disturbance_scenarios.py` defines real MuJoCo force/torque disturbance proofs that run against the live CrazySim plant instead of only mutating replayed telemetry
- `run_hover_sim_disturbances.py` executes that physical disturbance suite and checks whether bounded cases recover while stronger cases fail closed through the expected guard/abort path
- `run_hover_acceptance_gate.py` combines stored replay cases, a fresh or stored CrazySim scenario suite, and a fresh or stored physical disturbance suite into one fail-closed hover-acceptance gate; on fresh MuJoCo runs it also requires repeated nominal baseline repeatability before it can mark the stack `live_flight_eligible=true`
- replay comes first: every materially unsafe hover bug should become a deterministic replay regression and pass the combined replay+SITL gate before the next live flight
- the transient recovery contract is intentionally narrow: the current lane proves that Twinr commands the correct opposing bounded recovery and returns to bounded hover output, while sustained paired scenarios must trip declared guard codes and abort; the physical MuJoCo disturbance lane raises the bar to live-plant force/torque impulses, but it is still a sim proof, not live-hardware closure
- before full-firmware or live-hardware proof, the extracted on-device helper modules also have one firmware-near harness in [../../test/fixtures/twinrfs_control_harness.c](../../test/fixtures/twinrfs_control_harness.c) and [../../test/test_bitcraze_twinrfs_control_harness.py](../../test/test_bitcraze_twinrfs_control_harness.py); that lane compiles the same pure C control helpers the STM32 build uses and proves vertical handoff/ceiling-abort plus disturbance recover/abort behavior locally

The same bounded path is available to operators through:

```bash
python3 -m twinr --env-file .env --drone-status
python3 -m twinr --env-file .env --drone-inspect "prüfe das regal"
python3 -m twinr --env-file .env --drone-hover-test
python3 -m twinr --env-file .env --drone-cancel-mission DRN-...
python3 -m twinr --env-file .env --drone-manual-arm DRN-...
python3 -m twinr --env-file .env --self-test drone_stack
```

```python
from twinr.hardware import VoiceProfileMonitor, configured_pir_monitor

voice_monitor = VoiceProfileMonitor.from_config(config)
pir_monitor = configured_pir_monitor(config)
```

```python
from twinr.hardware.household_identity import HouseholdIdentityManager

manager = HouseholdIdentityManager.from_config(config, camera=camera, camera_lock=camera_lock)
status = manager.status(audio_pcm=current_turn_audio, sample_rate=16000, channels=1)
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [camera_ai/](./camera_ai/)
- [workflow orchestration](../agent/workflows/README.md)
- [ops self-test entry points](../ops/self_test.py)
