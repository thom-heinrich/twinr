# display

`display` owns Twinr's runtime display surfaces. It turns runtime snapshots and
health signals into short status screens and provides a fullscreen HDMI
Wayland adapter for the visible Pi monitor surface, an HDMI framebuffer
fallback backend, and the legacy Waveshare 4.2 V2 panel adapter.

## Responsibility

`display` owns:
- translate runtime snapshots into bounded status frames for the active backend
- compose panel-bounded layout variants such as the operator-facing `debug_log`
  view with grouped `System Log`, `LLM Log`, and `Hardware Log` sections
- derive calm ReSpeaker HCI states such as `DFU`, `fehlt`, `stumm`, or room-audio blockers from authoritative ops events instead of parsing proactive payloads inline
- derive the debug-log sections from persisted ops events, usage telemetry,
  host health, and the remote-memory watchdog artifact
- surface generic realtime tool-call completion/failure events in the operator
  `LLM Log`, including tool name, status, latency, and compact fallback/error
  clues instead of requiring direct JSONL inspection
- persist an authoritative display heartbeat so ops health and the runtime
  supervisor can detect a hung companion thread instead of trusting lock
  ownership alone
- expose one shared heartbeat contract for display writes, ops companion-health
  assessment, and runtime-supervisor progress checks so those three paths
  evaluate the same signal instead of drifting apart
- tolerate bounded in-flight panel renders as healthy progress so supervision
  does not kill a live companion just because one Waveshare refresh runs longer
  than the idle heartbeat budget
- render and upload Waveshare 4.2 V2 images with validated driver settings
- bound Waveshare BUSY-pin waits so panel/driver stalls raise recoverable
  errors instead of wedging the companion thread forever
- prefer stable two-plane refresh paths on the Waveshare 4.2 V2 panel so live
  status animations do not invert panel polarity
- keep the Waveshare default face static in `waiting` so the Pi panel stays
  visually calm instead of rerendering every few seconds while nothing changed
- allow subtle idle waiting motion on HDMI backends so the fullscreen face does
  not look frozen on the visible monitor surface
- allow optional external face-expression cues on HDMI so other Twinr
  capabilities can steer gaze, brows, mouth, or tiny head drift without
  coupling those semantics into the generic runtime snapshot schema
- persist and load optional HDMI header debug signals so operator-only camera
  and fusion states such as `MOTION_STILL`, `POSE_UPRIGHT`, or
  `POSSIBLE_FALL` can appear in one bounded top-lane without leaking those
  transient internals into the generic runtime snapshot schema
- allow optional external HDMI emoji cues so other capabilities can claim the
  otherwise free right-hand reserve area with one real Unicode symbol such as
  `👍` or `👋` without pushing emoji-only semantics into the generic runtime
  snapshot schema
- allow optional external HDMI service-connect cues so voice-triggered pairing
  flows can temporarily own that same right-hand reserve area with short
  status copy and a scannable QR payload without coupling onboarding state into
  the generic runtime snapshot schema
- allow optional external HDMI ambient-impulse cues so personality-driven
  capabilities can claim that same reserve area with one bounded, readable,
  question-first reserve card that stays positive, glanceable, and clearly
  separate from the generic runtime snapshot schema
- persist a bounded reserve-card exposure history so later learning code can
  correlate which right-lane prompts Twinr actually showed with later
  structured user reactions
- persist one short-lived reserve-bus feedback hint so strong immediate pickup
  or pushback on a shown card can reshape the remaining day plan faster than
  the slower long-term learning loop alone
- arbitrate that right-hand HDMI reserve area through one explicit reserve-bus
  helper so service-connect QR/status cards, emoji acknowledgements, calm
  ambient cards, and future reserve-only cue families do not compete through
  inline renderer conditionals
- let the right-hand reserve mirror clear user camera gestures with the same
  symbol when the proactive runtime publishes a bounded acknowledgement, while
  keeping hostile-gesture reactions separate from face emotion
- allow optional external HDMI presentation cues so other capabilities can
  expand the default right-hand panel into a bounded fullscreen image or rich
  card without teaching the generic runtime snapshot schema about
  presentation-only payloads
- fetch, cache, and rotate optional RSS/Atom headlines for a calm HDMI
  bottom-bar ticker without blocking the display loop on network I/O
- allow optional periodic full-refresh cleanup after a bounded number of fast
  incremental updates instead of running indefinitely on fast refresh alone
- emit bounded display-driver telemetry lines for refresh mode, clear, retry,
  driver-reset, and BUSY-timeout decisions so Pi validation can prove which
  hardware paths actually executed
- keep debug-log lines stable between real state changes so operator screens do
  not churn the e-paper panel every poll cycle
- force `debug_log` through the panel's full-refresh path because the text-rich
  operator layout proved unstable on the Pi's fast incremental path
- coalesce rapid debug-log content churn so operator-facing full refreshes stay
  bounded even when ops events arrive in short bursts
- keep debug-log hardware and ChonkyDB summary lines semantically current but
  bucketed/stable enough for e-paper refresh budgets
- keep debug-log host metrics on operator thresholds rather than narrow raw
  buckets so ordinary Pi temperature drift does not cause visual churn
- keep the senior-facing `System` card operational: routine warm CPU drift may
  still appear in ops/debug metrics, but it must not flap the main status card
  between `OK` and `Warm`
- keep watchdog probe transitions, even during repeated ChonkyDB failures, and
  minor host-metric drift from retriggering debug-log rerenders every few
  seconds on the panel
- discard cached vendor imports after hardware faults so the next render starts cleanly
- drive the visible HDMI Wayland path with a full-screen, high-contrast,
  large-type operator UI that stays readable on the Pi monitor
- keep an HDMI framebuffer fallback backend for Pi setups that do not expose a
  usable Wayland session
- run the optional companion display loop beside hardware/runtime loops

`display` does **not** own:
- runtime snapshot storage or health sampling implementations
- GPIO/SPI bootstrap and vendor installation scripts in `hardware/display`
- top-level CLI parsing or loop-lock policy outside display-specific behavior

The generated Waveshare vendor files should live under `state/display/vendor/`
on the Pi so runtime deploy syncs do not wipe the driver package. The
authoritative display heartbeat lives separately under
`artifacts/stores/ops/display_heartbeat.json` so the unprivileged runtime can
always refresh it even when the vendor directory is root-owned.

For the Waveshare default face layout, Twinr keeps the `waiting` state static
instead of animating it on a timer. That idle motion produced unnecessary
e-paper refreshes on the Pi. HDMI backends expose that behavior as an explicit
adapter capability instead: the senior-facing fullscreen scene may use subtle
idle motion while the shared status loop still keeps static backends calm.
`TWINR_DISPLAY_FULL_REFRESH_INTERVAL` remains opt-in and defaults to `0`; when
operators enable it, the adapter injects bounded full-refresh cleanup into
layouts that still use fast incremental updates.

The `debug_log` layout is treated more conservatively: after the first render
it always uses the full-refresh path. Live Pi evidence showed the panel's
`display_Fast` path timing out and leaving the operator screen inverted, so the
text-heavy debug layout no longer attempts incremental refresh there. The
service also compares `debug_log` frames only on the content that the layout
actually draws and applies a bounded holdoff before replaying another
same-status update, so volatile footer fields or short ops bursts do not
translate straight into panel flicker.

The Waveshare adapter also emits short telemetry lines such as
`display_refresh=full reason=interval`, `display_clear=true`,
`display_retry=true`, and `display_busy_timeout=true`. In productive Pi runs
these land in the loop's stdout/journal output, which gives a direct proof of
which refresh or recovery path actually ran.

For runtime-only stalls there is now a second, opt-in layer:
`TWINR_DISPLAY_RUNTIME_TRACE_ENABLED=true`. When enabled, the adapter adds
bounded journal lines such as `display_trace=phase_start`,
`display_trace=epd_call_start`, `display_trace=busy_wait_timeout`,
`display_trace_gpio=...`, and `display_trace_supply=...`. Those lines carry the
render phase, current and previous runtime status, last vendor command, BUSY
caller, GPIO snapshots, and throttling state so the first live stall can be
proven directly from `journalctl` without stopping Twinr for the standalone
probe script.

For HDMI emoji cues, the runtime expects a real system emoji font. On Ubuntu
Pi images that means `fonts-noto-color-emoji`, which provides
`/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf`. When the font is missing
or unreadable, the framebuffer adapter now emits one bounded telemetry line
such as `display_emoji_font=missing ...` instead of silently drawing nothing.

For HDMI deployments on a Raspberry Pi desktop session, prefer:

```dotenv
TWINR_DISPLAY_DRIVER=hdmi_wayland
TWINR_DISPLAY_WAYLAND_DISPLAY=wayland-0
TWINR_DISPLAY_WAYLAND_RUNTIME_DIR=/run/user/1000
```

That path gives Twinr ownership of the visible fullscreen surface. Keep
`hdmi_fbdev` only as a fallback for framebuffer-only environments where no
usable Wayland session exists.

The productive display companion does not start on every `/twinr` host by
default. Automatic startup remains limited to authoritative `/twinr` launches
on real Raspberry Pi hosts so generic servers do not suddenly lose a monitor to
the Twinr face. When an operator intentionally wants the visible companion on a
different authoritative host, set:

```dotenv
TWINR_DISPLAY_COMPANION_ENABLED=true
```

To suppress the fullscreen companion explicitly even on a Pi host, set:

```dotenv
TWINR_DISPLAY_COMPANION_ENABLED=false
```

The default HDMI surface is intentionally much calmer than the operator
`debug_log` view: solid black background, a three-line top header with
`TWINR` on the left, the live runtime state in the middle, and `TIME + SYSTEM`
on the right in the first row, plus two bounded operator-only perception rows
underneath, and an animated white-on-black face on the left that mirrors the familiar
e-paper eye/mouth language. The header state should mirror the real runtime
state directly in uppercase, for example `WAITING`, `LISTENING`, `THINKING`,
or `SPEAKING`, instead of restating generic readiness. The visible `SYSTEM`
indicator in that header is intentionally binary and prominent: `OK` or
`ERROR`. On 800x480 the face should stay the visually dominant element, and
the area to its right should remain visibly free until a later capability
deliberately claims it. That keeps the senior-facing screen glanceable from a
distance while `debug_log` remains the explicit diagnostics layout for
operators.

For Pi debugging, that same HDMI header now reserves two wrapped signal rows
underneath the main status line as one persistent extended view. Those rows
remain file-backed, short-lived, and operator-only: they may show pills such
as `LOOKING_CONFIRMED`, `LOOKING_PROXY`, `HAND_NEAR`, `PERSON_1`,
`POSE_UPRIGHT`, `MOTION_STILL`, `ATTENTION_WINDOW`, or `POSSIBLE_FALL`, but
they must stay bounded to two rows with overflow compaction instead of
turning the senior-facing header into an operator dashboard.

On larger HDMI outputs such as 1920x1080, the same rule still applies: do not
let that free right-hand reserve turn into random chrome. Keep the content
centered, preserve the face-first hierarchy, and allow the face itself to
scale up so the screen still reads as "face first, calm state header second"
instead of turning into one giant empty status card.

On the real 800x480 HDMI surface, that same hierarchy must still read clearly
with the current extended debugging view: keep the top header large enough to
show the perception pills, suppress the visible bottom news ticker so that
vertical budget goes into face and header clarity, and leave the right-hand
reserve area free of status-card chrome so the live screen does not regress
back toward the earlier cramped composition. When a capability does claim that
reserve area, prefer one real emoji cue with a soft halo over dense text or
extra chrome so the surface stays readable at a glance.
Gesture acknowledgements should mirror clear symbols like `👍`, `👎`, `👋`,
`✌️`, `👌`, or `👉` directly instead of inventing extra text, while still
keeping face emotions in the face channel and not in emoji-only surrogates.
Calm companion impulses may use a bounded reserve card there too, but that
card must stay short, positive, and subordinate to stronger
reserve-surface owners such as gesture acknowledgements or fullscreen
presentations. Service-connect flows are stronger still: while Twinr is
opening or waiting on a bounded external pairing flow such as WhatsApp, that
right-hand area may temporarily turn into a readable service card with the
current pairing summary and, when available, the scannable QR image itself.
presentations. On the 800x480 Pi surface that means readable main text first,
minimal chrome, and a prompt-mode reserve card that reads like one large
conversation opener instead of a small label plus helper text.
The reserve card should also read like Twinr, not like a static template bank:
the visible copy may be generated ahead of time from Twinr's current
personality state plus structured reserve context such as a shared thread, a
gentle follow-up, or a memory clarification need, but it must still stay
bounded, readable, and calm. On the real Pi surface that means no left accent
rail in prompt mode, one still-large but slightly reduced text scale for the
full reserve message so the box keeps more breathing room, and wording that
sounds like a natural conversation opener rather than a tiny topic label with
helper chrome. The large headline should stand on its own as the explanatory
statement, while the smaller second line should act as the CTA. Prompt-mode
cards should also fit their shared prompt-mode type scale dynamically to the
actual visible card box before clipping, and consume the
actual visible card height instead of stopping after a fixed two-line budget,
so longer right-lane prompts do not appear cut off halfway down the box.

That waiting surface may also show very rare ambient moments: tiny sparkles,
hearts, crescent moons, wave marks, curious dot clusters, or even a tiny crown
can briefly appear near the face while the status is idle. Those moments are
deliberately HDMI-only, deterministic, and suppressed whenever an external face
cue or presentation is active, so the screen stays calm instead of turning
into a noisy novelty loop.

For HDMI, eye animation should stay calmer than mouth or whole-face motion:
prefer gaze shifts, subtle blinks, and tiny head drift over large eye-resize
swings or inverse-color eyelid strokes that read like extra eyebrows on the
black background. When a directional external face cue is active, that cue
should dominate the `waiting` face instead of being overpainted by the normal
idle side-to-side churn.
When there is no live attention target, the default `waiting` and `error`
baseline should still read as calm eye contact from the front rather than an
up-left idle stare; no-target motion should come from tiny blink/easing only,
not from an off-axis scan.

That only works if the HDMI loop itself stays fast. The default HDMI display
poll cadence is intentionally sub-second so local face cues and emoji
acknowledgements do not feel delayed by a slow generic status loop.

That senior-facing HDMI surface is now modeled as its own scene module instead
of being inlined into the framebuffer adapter. `hdmi_default_scene.py` owns the
default-scene layout, face animation, header model, and reserved right-hand
capability area so future HDMI capabilities such as expanded cards, morph
surfaces, and reserve-bus content owners can grow without bloating the display
loop or the transport adapters. The reserve area itself is now resolved through
`reserve_bus.py`: gesture emoji cues still outrank calmer ambient reserve
cards, but that priority no longer lives as ad-hoc branching inside the scene
renderer.
transitions, or richer per-capability panels can be added without pushing
presentation logic back into the transport backend.

When proactive monitoring targets the XVF3800, the display service now reads
the latest authoritative ReSpeaker ops facts and surfaces only calm status
transitions on the operator card: degraded hardware states such as `DFU` or
`fehlt`, microphone mute or explicit `hört`, recent `Sprache`, strong
direction hints, and room-audio blockers such as `Medien`, loud overlap, or a
short `Pause`/resume window. The default surface intentionally does not mirror
every weak audio bit; direction hints require strong confidence, and the shared
ring semantics are `listening_mute_only`, which means the XVF3800 ring may
mirror explicit listening or mute but never flickers on weak direction/speech
evidence. Richer ReSpeaker detail stays in the `debug_log` hardware section.

The first presentation-capability slice now resolves through a dedicated HDMI
scene graph instead of directly from one cue to one overlay. The graph builder
selects the highest-priority active card, keeps other cards queued, applies
eased intermediate morph states, and derives a calm face-sync cue while the
card is still expanding. That keeps future capability growth in a dedicated
graph layer instead of embedding priority or morph policy inside the renderer.

For acceptance on the real visible monitor surface, `visual_qc.py` now adds a
bounded HDMI visual-QC runner. It drives the public face and presentation
controllers through a deterministic scene set, captures the visible Wayland
surface with `grim`, computes per-transition diff metrics, and assembles a
report-backed artifact bundle instead of relying on one-off screenshots in
`/tmp`.

The visible Wayland window is also split away from image rendering now.
`hdmi_wayland.py` stays responsible for the `hdmi_wayland` adapter contract and
environment wiring, while `wayland_surface_host.py` owns the actual Qt
fullscreen surface. That keeps Wayland-only presentation capabilities isolated
from the generic framebuffer renderer instead of mixing browser or window-host
logic into `hdmi_fbdev.py`.

External HDMI face triggers flow through `face_cues.py`. The runtime display
loop loads one optional cue artifact and merges it only into the `default`
HDMI face. That keeps `debug_log`, Waveshare, and the generic runtime snapshot
schema stable while still allowing other modules to steer:
- `gaze_x` / `gaze_y`
- `mouth`
- `brows`
- `blink`
- `head_dx` / `head_dy`

`gaze_x` / `gaze_y` are intentionally finer-grained than the producer-facing
emotion presets: low-level cues may use bounded multi-step eye offsets in the
range `-3..3` so physical HCI paths like person-following can look smoother on
the Pi instead of snapping between only a few coarse positions. `head_dx` /
`head_dy` stay tighter and smaller.

Ambient HDMI reserve impulses flow through `ambient_impulse_cues.py`. The
display loop loads one optional cue artifact and renders it only on the
default HDMI scene when neither an emoji acknowledgement nor a richer
presentation already owns the reserve area. That keeps companion-like daily
nudges, shared-thread prompts, and personality-toned small updates outside the
generic runtime snapshot schema while preserving a calm face-first surface.

Producer modules should use `face_expressions.py` instead of writing raw JSON.
That helper keeps the low-level cue artifact stable while exposing a richer
combinable API:
- brows: `straight`, `inward_tilt`, `outward_tilt`, `roof`, `raised`, `soft`
- gaze: eight 45-degree directions plus `center`
- mouth: `neutral`, `smile`, `sad`, `thinking`, `pursed`, `scrunched`
- optional `blink`, `head_dx`, and `head_dy`

Legacy raw labels such as `focus`, `flat`, `line`, or `concern` still
normalize into the canonical vocabulary so older producer paths do not break.
The `error` HDMI scene intentionally keeps its own sad mouth/brow expression;
external cues may still steer gaze, blink, and tiny head drift there, but they
must not overwrite the status-owned error expression.

The default artifact path is `artifacts/stores/ops/display_face_cue.json` and
the default cue TTL is `4.0` seconds. Configure them with:

```dotenv
TWINR_DISPLAY_FACE_CUE_PATH=artifacts/stores/ops/display_face_cue.json
TWINR_DISPLAY_FACE_CUE_TTL_S=4.0
```

One bounded low-level payload still looks like:

```json
{
  "source": "camera_surface",
  "gaze_x": 3,
  "gaze_y": -1,
  "mouth": "smile",
  "brows": "raised",
  "blink": false
}
```

The preferred producer-facing path is:

```python
from twinr.display import (
    DisplayFaceBrowStyle,
    DisplayFaceExpressionController,
    DisplayFaceGazeDirection,
    DisplayFaceMouthStyle,
)

controller = DisplayFaceExpressionController.from_config(config, default_source="camera_surface")
controller.show(
    gaze=DisplayFaceGazeDirection.UP_RIGHT,
    brows=DisplayFaceBrowStyle.RAISED,
    mouth=DisplayFaceMouthStyle.SMILE,
    blink=False,
    hold_seconds=5.0,
)
```

Fullscreen HDMI presentations flow through `presentation_cues.py`. That path is
deliberately separate from face cues because a fullscreen card or image belongs
to the presentation layer, not to the generic runtime snapshot schema. The
default artifact path is `artifacts/stores/ops/display_presentation.json` and
the default cue TTL is `20.0` seconds:

```dotenv
TWINR_DISPLAY_PRESENTATION_PATH=artifacts/stores/ops/display_presentation.json
TWINR_DISPLAY_PRESENTATION_TTL_S=20.0
```

The producer-facing path is:

```python
from twinr.display import DisplayPresentationController

controller = DisplayPresentationController.from_config(config, default_source="camera_surface")
controller.show_rich_card(
    title="Family call",
    subtitle="Marta is waiting",
    body_lines=("Press the green button to answer.",),
    accent="warm",
    hold_seconds=15.0,
)
```

Or for a fullscreen image surface:

```python
from twinr.display import DisplayPresentationController

controller = DisplayPresentationController.from_config(config, default_source="operator")
controller.show_image(
    image_path="/tmp/marta_photo.png",
    title="New photo",
    subtitle="Delivered just now",
    body_lines=("The photo fills the presentation surface.",),
    accent="success",
    hold_seconds=20.0,
)
```

For multiple competing cards, use the scene-capable producer path. Each card
gets its own `key`, `priority`, and optional `face_emotion`; the highest
priority card becomes active while the others stay queued in the graph:

```python
from twinr.display import DisplayPresentationCardCue, DisplayPresentationController

controller = DisplayPresentationController.from_config(config, default_source="operator")
controller.show_scene(
    cards=(
        DisplayPresentationCardCue(key="summary", title="Daily summary", priority=20, accent="info"),
        DisplayPresentationCardCue(
            key="family_photo",
            kind="image",
            title="Family photo",
            image_path="/tmp/family_photo.png",
            priority=90,
            accent="warm",
            face_emotion="happy",
        ),
    ),
    hold_seconds=20.0,
)
```

The optional HDMI news ticker flows through `news_ticker.py`. That path stays
separate from face and presentation cues because headline fetch, cache, and
rotation are display-surface concerns, not generic runtime-snapshot state. It
uses a runtime-writable cache artifact and refreshes asynchronously so the
display loop never blocks on remote feed downloads, but the ticker now draws
its sources from Twinr's persisted world-intelligence subscriptions instead of
from a second static feed list.

The current default HDMI extended view keeps that ticker runtime alive but
suppresses the visible bottom bar. This is intentional while the top-of-screen
perception header runs in its persistent three-line debugging mode.

Configure it with:

```dotenv
TWINR_DISPLAY_NEWS_TICKER_ENABLED=true
TWINR_DISPLAY_NEWS_TICKER_STORE_PATH=artifacts/stores/ops/display_news_ticker.json
TWINR_DISPLAY_NEWS_TICKER_REFRESH_INTERVAL_S=900
TWINR_DISPLAY_NEWS_TICKER_ROTATION_INTERVAL_S=12
TWINR_DISPLAY_NEWS_TICKER_MAX_ITEMS=12
TWINR_DISPLAY_NEWS_TICKER_TIMEOUT_S=4
```

The actual feed pool comes from Twinr's active world-intelligence
subscriptions. That keeps the bottom ticker, the right-hand ambient lane, and
Twinr's own place/world awareness aligned to the same RSS source universe
instead of maintaining separate news lists.

For older Pi environments that still carry the legacy
`TWINR_DISPLAY_NEWS_TICKER_FEED_URLS` variable, the ticker performs a one-way
compatibility import into the world-intelligence subscription snapshot the
first time it sees an otherwise empty source pool. After that, the shared
world-intelligence pool remains the authoritative source.

Behavior contract:
- the ticker runtime remains asynchronous and cache-backed whether or not the visible bar is currently suppressed
- the current default HDMI extended view omits the visible bottom ticker bar even when headline text is available
- when that visible ticker bar is re-enabled later, a cold ticker should show `Loading headlines...`
- when the last refresh failed and no cached items exist, a re-enabled ticker should show `Headlines unavailable.`
- fullscreen HDMI presentations suppress the ticker while the focused surface owns the whole screen

`service.py` also keeps display telemetry semantic now: `display_status=...`
is emitted only when the user-meaningful display state changes, such as a real
runtime status transition or a presentation stage change. Idle HDMI animation
frames no longer spam identical status lines into the supervisor journal.

For screenshot-backed HDMI validation on the Pi, run:

```bash
python3 hardware/display/run_visual_qc.py --env-file .env
```

The current default scene set covers:
- `idle_home`
- `face_react`
- `presentation_mid`
- `presentation_focused`
- `restored_home`

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Public display exports |
| [contracts.py](./contracts.py) | Shared adapter/payload contracts |
| [factory.py](./factory.py) | Config-driven display backend selection |
| [debug_log.py](./debug_log.py) | Build grouped operator log sections from ops/usage stores |
| [emoji_cues.py](./emoji_cues.py) | Optional external HDMI emoji cue contract, store, and producer-facing controller |
| [face_cues.py](./face_cues.py) | Optional external HDMI face-expression cue contract and store |
| [face_expressions.py](./face_expressions.py) | Producer-facing combinable expression API for the HDMI face |
| [heartbeat.py](./heartbeat.py) | Persist display forward-progress heartbeats and expose the shared companion-health contract for ops/supervision |
| [hdmi_ambient_moments.py](./hdmi_ambient_moments.py) | Deterministically schedule rare idle-only HDMI ambient moments such as sparkles or hearts |
| [hdmi_default_scene.py](./hdmi_default_scene.py) | Modular default HDMI scene model, face renderer, and status-card composition |
| [hdmi_presentation_graph.py](./hdmi_presentation_graph.py) | Resolve prioritized HDMI presentation cards into eased morph stages and face-sync reactions |
| [hdmi_wayland.py](./hdmi_wayland.py) | Visible fullscreen HDMI Wayland adapter |
| [hdmi_fbdev.py](./hdmi_fbdev.py) | HDMI framebuffer fallback adapter and scene host/transport layer |
| [news_ticker.py](./news_ticker.py) | Bounded HDMI ticker runtime, cache, and headline rotation driven by Twinr's world-intelligence subscription pool |
| [news_ticker_sources.py](./news_ticker_sources.py) | Resolve the active persisted world-intelligence feed URLs that the HDMI ticker is allowed to read |
| [presentation_cues.py](./presentation_cues.py) | Optional fullscreen HDMI presentation-cue contract, store, and producer-facing controller |
| [wayland_env.py](./wayland_env.py) | Resolve and export Wayland socket/runtime details |
| [wayland_surface_host.py](./wayland_surface_host.py) | Native Wayland/Qt surface host kept separate from rendering and scene composition |
| [service.py](./service.py) | Snapshot-driven status loop |
| [visual_qc.py](./visual_qc.py) | Screenshot-backed HDMI visual-QC runner and report-artefact builder |
| [layouts.py](./layouts.py) | Status-card layout composition |
| [waveshare_v2.py](./waveshare_v2.py) | Panel adapter and rendering |
| [companion.py](./companion.py) | Optional sidecar loop |
| [component.yaml](./component.yaml) | Structured package metadata |

## Usage

```python
from twinr.display import TwinrStatusDisplayLoop, create_display_adapter

display = create_display_adapter(config)
display.show_test_pattern()

loop = TwinrStatusDisplayLoop.from_config(config)
loop.run(duration_s=15)
```

```python
from twinr.display.companion import optional_display_companion

with optional_display_companion(config, enabled=True):
    realtime_loop.run(duration_s=15)
```

To activate the debug layout on the Pi/operator runtime, set:

```dotenv
TWINR_DISPLAY_LAYOUT=debug_log
```

`debug_log` removes the face entirely and uses the full screen for grouped
`System Log`, `LLM Log`, and `Hardware Log` sections sourced from Twinr's
runtime snapshot, ops event store, usage store, host health snapshot, and
remote ChonkyDB watchdog artifact. Search turns surface their actual
requested/actual model and compact output-budget trace there as well, so
operators can see at a glance whether a live search stayed on the primary
budget or escalated to the retry budget. The legacy env value `debug_face` is
kept as a compatibility alias and normalizes to `debug_log`.

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [hardware/display](../../../hardware/display/README.md)
