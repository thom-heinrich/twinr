# hardware

`hardware` owns Twinr's low-level audio, GPIO, bounded servo output, camera,
printer, local voice-profile adapters, and the bounded local
household-identity building blocks that combine portrait, voice, and explicit
feedback on device.

## Responsibility

`hardware` owns:
- capture and play bounded audio for conversation and ambient sensing
- lift under-driven PCM16 activation-ack playback to an audible level before it reaches the Pi speaker path when provider TTS arrives too quiet
- monitor GPIO buttons and PIR motion across supported Pi backends
- drive one bounded body-orientation servo from an already-derived normalized attention target without owning speaker/person targeting policy, including calibrated center/invert controls, Twinr's custom kernel-module servo sysfs contract when available, kernel-PWM sysfs output when the GPIO exposes hardware PWM, `pigpio` as a secondary hardware-timed option, a calmer `lgpio.tx_pwm` software-PWM fallback ahead of the older jitter-prone `lgpio.tx_servo` helper, or a Pololu Mini Maestro command-port path over USB serial once the controller itself is configured for `USB_DUAL_PORT`/`USB_CHAINED`, plus soft end-stop margins, explicit target smoothing, visible-target latching so in-frame micro-jitter does not trigger constant re-justification, an optional exit-only mode where visible users do not trigger physical motion until either the image-exit trajectory is confirmed or the tracked person reaches a calibrated side-departure threshold long enough to justify one monotone pursuit, plus an explicit continuous-rotation control mode that keeps a bounded virtual heading estimate and maps heading error to calm time/speed pulses instead of pretending a 360-degree servo supports absolute position, a tiny persisted continuous-servo state file so an operator hand-set `0°` reference can survive restarts, an explicit `hold_until_armed` startup mode that keeps the Maestro released instead of blindly leaving that reference pose, an explicit visible-box edge gate so exit-only pursuit only starts when the authoritative person geometry is actually near the frame boundary instead of merely off-center, one monotone exit pursuit with a configurable off-center degree clamp, a short configurable settle-hold so loaded hardware can physically complete the exit move before release, centered visible reacquire cooldown once the user is back near the middle of frame, calm return to a rest position after longer absence with a dedicated slower rest-motion profile, acceleration- and jerk-limited motion shaping, small command hysteresis against dead-band chatter, startup seeding from the currently active writer's remembered pulse width so any non-center remembered pose gets recentered before exit-only waiting resumes, an exact-center snap before idle release so the remembered neutral state is not left skewed, calm idle release near neutral, and release-after-settle so loaded servos can move into place and then relax instead of buzzing under constant hold torque
- in exit-only mode, allow one coarse periodic visible-user recenter against a latched target after a configurable off-center dwell interval, so Twinr can quietly re-center a user who stayed visibly off-axis for a long time without regressing into continuous in-frame jitter
- fail closed when startup sees a proven conflicting GPIO18 servo environment such as a foreign `pwm-pio` overlay/process chain, so the HDMI face/runtime can stay up while the servo disables itself instead of jittering against a stolen line
- capture still photos from V4L2 cameras, from a Bitcraze AI-Deck WiFi stream, with a bounded `rpicam-still` fallback for Pi camera stacks when the default V4L2 node is missing and for Pi `unicam-image` devices when the V4L2 node is busy, or fetch the same still-photo contract from a dedicated peer Pi over a bounded HTTP snapshot proxy
- reach one bounded external drone daemon over HTTP so Twinr can queue high-level inspect missions, read state, cancel work, and enforce manual-arm-only safety gates without ever sending direct flight-control commands
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
- treat negative custom labels such as `none` as authoritative suppression, so a weak custom symbol does not win when the classifier itself prefers "no gesture"
- normalize MediaPipe-bound image buffers centrally so full frames and ROI crops reach the Pi runtime with supported dtype and contiguous layout
- keep multi-ROI MediaPipe tasks monotonic by advancing and reserving timestamps across candidate inference within a frame
- keep MediaPipe gesture recognition latency bounded by preferring ROI gesture recognition when concrete wrist-localized hand crops already exist, and only falling back to the stable full-frame recognizer when ROI crops did not yield a real symbol
- run the dedicated ROI hand-landmark worker in MediaPipe IMAGE mode, because upper-body and wrist crops are independent images rather than one temporally stable video stream; wrist-localized crops stay first for efficiency, but bounded upper-body/full-body rescue crops remain available so weak wrist hints do not suppress hand recovery entirely
- expose a cheap attention-only AI-camera observation that reuses IMX500 person detection plus optional face-anchor supplementation but skips the heavy MediaPipe gesture stack, so HDMI eye-follow can stay reactive while explicit gesture acknowledgement uses the slower full path separately
- expose a dedicated live-stream gesture observation that reuses the RGB camera session but keeps the hot path close to the official Pi-style pattern: one thin built-in live recognizer for the Pi-critical symbols `thumbs_up`, `thumbs_down`, and `peace_sign`, plus a product-specific custom recognizer for the same three labels when staged, bounded primary-person/hand-ROI fallback, and one final whole-frame hand rescue when that fast path still returns `none`
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
- wait a short bounded window for transient XVF3800 USB/ALSA re-enumeration to clear before live listen capture fails a turn outright
- resolve one calm XVF3800 indicator contract and drive bounded runtime-owned ring pulses for waiting, listening, thinking, speaking, and error states without twitching on weak audio evidence
- attach a per-claim ReSpeaker confidence/source contract that later runtime and memory layers can inspect without guessing
- derive conservative XVF3800 non-speech and background-media suppression facts from the same capture path
- scrub borrowed Wayland/user-session env from root-owned ALSA subprocesses so Pi audio probes do not inherit non-root Pulse/PipeWire runtime sockets from the HDMI surface
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
| [audio.py](./audio.py) | Bounded audio capture, readable-frame probing, and playback |
| [audio_env.py](./audio_env.py) | Sanitize child-process env for ALSA helpers when the Pi runtime borrows a different user's Wayland session |
| [respeaker_capture_recovery.py](./respeaker_capture_recovery.py) | Bounded transient XVF3800 capture-recovery helper shared by live listen and voice-orchestrator paths |
| [camera_ai/](./camera_ai/) | Internal SoC package for camera contracts, IMX500 runtime, MediaPipe runtime, pose, motion, and gesture modules |
| [camera_ai/live_gesture_pipeline.py](./camera_ai/live_gesture_pipeline.py) | Dedicated low-latency MediaPipe live-stream gesture lane for HDMI symbol acknowledgement |
| [camera_ai/gesture_candidate_capture.py](./camera_ai/gesture_candidate_capture.py) | Bounded JPEG + JSON capture helper for manual optical QA of suspected gesture frames |
| [ai_camera.py](./ai_camera.py) | Stable compatibility facade for the public local AI-camera adapter surface |
| [ai_camera_diagnostics.py](./ai_camera_diagnostics.py) | Bounded Pi-facing diagnostics for pose candidate selection and keypoint support |
| [aideck_camera.py](./aideck_camera.py) | Bounded Bitcraze AI-Deck CPX-over-TCP still-frame adapter that can temporarily join the deck AP, capture one frame, and restore the previous WiFi link before Twinr resumes upstream work |
| [aideck_wifi.py](./aideck_wifi.py) | Focused `nmcli`-backed WiFi handover helper for one bounded AI-Deck capture on single-WiFi hosts |
| [drone_service.py](./drone_service.py) | Strict mission-level Twinr client for the external bounded drone daemon, including inspect-mission request contracts and normalized safety/state snapshots |
| [hand_landmarks.py](./hand_landmarks.py) | Bounded MediaPipe hand-landmark ROI worker for full-frame-to-hand crop resolution |
| [mediapipe_vision.py](./mediapipe_vision.py) | Stable compatibility facade for the public MediaPipe camera pipeline surface |
| [buttons.py](./buttons.py) | GPIO button monitoring backends |
| [servo_follow.py](./servo_follow.py) | Bounded body-orientation adapter for normalized attention targets, selecting among Twinr's custom kernel servo sysfs contract, kernel-PWM sysfs, hardware-timed `pigpio`, calmer `lgpio.tx_pwm`, older `lgpio` servo output, a focused local Pololu Mini Maestro USB command-port writer, or a peer-Pi Pololu Maestro HTTP writer, and layering visible-target latching, optional exit-only physical follow with loss-confirmation delay plus frame-edge-confirmed visible departure, one-way exit pursuit toward a bounded side target, coarse periodic visible-user recentering after a long off-center dwell instead of continuous in-frame tracking, a dedicated continuous-rotation mode that drives a virtual heading through bounded speed pulses for 360-degree servos, a short exit settle-hold before release, centered visible reacquire cooldown, a slower rest-position return profile after longer absence, startup seeding from the active writer's remembered pulse width so any non-center remembered pose is neutralized first, an explicit manual-hold branch that keeps a 360-degree servo electrically released until explicitly armed, live reload of the persisted hold/arm state file, an explicit return-to-estimated-zero branch that first replays the persisted movement journal through exact-duration bounded segments before falling back to the older slow planner, released idle semantics once the estimated heading is already back at `0°`, acceleration/jerk motion shaping, soft end-stop margins, output hysteresis, calm idle release, release-after-settle for quieter loaded holds, and a fail-closed startup guard against proven foreign GPIO servo overlays/processes |
| [servo_continuous.py](./servo_continuous.py) | Focused open-loop planner for continuous-rotation servos that estimates one virtual heading, converts heading error into bounded direction/speed pulses, and keeps that motion model out of the higher follow-state machine |
| [servo_state.py](./servo_state.py) | Tiny JSON-backed runtime-state store for continuous servos, persisting the operator-confirmed virtual heading, bounded heading uncertainty, and explicit hold/arm or return-to-estimated-zero transitions outside the main follow controller |
| [servo_segment_player.py](./servo_segment_player.py) | Exact-duration background segment player for continuous-servo reverse replay, starting one bounded pulse immediately and disabling it at the recorded deadline instead of holding it until the next runtime tick |
| [servo_maestro.py](./servo_maestro.py) | Focused Pololu Mini Maestro command-port adapter that auto-discovers the stable `...-if00` USB serial endpoint when present, falls back to Maestro-tagged `ttyACM*` command ports when by-id links are absent, invalidates stale connections after USB faults, emits compact-protocol target commands, reads back current channel positions for startup alignment, fails clearly when the controller is still in the wrong UART serial mode, and keeps Maestro transport concerns out of the higher servo policy module |
| [servo_peer.py](./servo_peer.py) | Focused peer-Pi Pololu Maestro HTTP client that lets the main Twinr runtime keep follow policy local while the helper Pi owns the actual Maestro USB command port |
| [camera.py](./camera.py) | bounded still capture with ffmpeg/V4L2, Pi `rpicam-still` fallback for missing default `video0` nodes and busy `unicam-image` devices, or peer-Pi HTTP snapshot fetches when `TWINR_CAMERA_PROXY_SNAPSHOT_URL` is configured |
| [portrait_identity.py](./portrait_identity.py) | Atomic local portrait-identity store with enrolled reference-image management |
| [portrait_match.py](./portrait_match.py) | Local YuNet/SFace portrait-match adapter with enrollment flow, multi-reference scoring, and temporal fusion |
| [household_voice_identity.py](./household_voice_identity.py) | Multi-user local voice-identity store and matcher for enrolled household members, including bounded read-only snapshot export for the live voice gateway |
| [household_identity.py](./household_identity.py) | Shared local household identity manager for portrait, voice, feedback, and bounded session fusion |
| [pir.py](./pir.py) | PIR motion wrapper on buttons |
| [printer.py](./printer.py) | Receipt formatting and CUPS submission |
| [respeaker/](./respeaker/) | XVF3800 probe, host-control transport, typed primitive snapshots, calm LED profiles/controller/companion loop, conservative derived signals, indicator semantics, claim-contract builders, scheduled polling wrapper, and runtime signal provider |
| [respeaker/ambient_classification.py](./respeaker/ambient_classification.py) | Conservative non-speech/background-media classification from bounded ambient capture plus XVF3800 speech facts |
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

When the main Pi has no directly attached still camera, configure
`TWINR_CAMERA_PROXY_SNAPSHOT_URL=http://10.42.0.2:8767/snapshot.png` and the
same `V4L2StillCamera.from_config(config)` entrypoint will fetch bounded PNG
captures from the dedicated peer camera proxy Pi instead of opening
`/dev/video0`.

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

The accepted v1 safety contract is:
- Twinr only submits bounded inspect missions and never sends direct roll/pitch/yaw/thrust commands
- the daemon must expose `manual_arm_required=true` by default
- arming goes through the daemon's local ops route, not through free-form voice
- the default runtime slice remains `stationary_observe_only`, so the daemon proves mission/state/evidence plumbing before motion is enabled
- the first live motion primitive is a separate bounded `hover_test` mission, and it is only accepted when the daemon was explicitly started in `bounded_hover_test_only`
- the first hover path now also requires the Twinr on-device Crazyflie failsafe app (`twinrFs`) by default, so heartbeat-loss, low-battery, and clearance-triggered safe-land behavior lives on the aircraft in C instead of depending on the host process

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
