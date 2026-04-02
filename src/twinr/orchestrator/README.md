# orchestrator

`orchestrator` owns Twinr's provider-neutral websocket transport for an
edge/cloud split. It packages the message contracts, client/server entrypoints,
ack mapping, the remote-tool bridge that lets a remote peer execute local Twinr
tools through one bounded turn protocol, and the streaming voice session
contracts used by the Alexa-like hybrid voice path.

## Responsibility

`orchestrator` owns:
- define websocket contracts for turn, tool, ack, delta, and completion traffic
- expose client/server entrypoints for edge-orchestrator deployments
- bridge remote tool calls and results into Twinr's local dual-lane tool loop
- keep fast ack phrases mapped to stable transport IDs
- define the streaming voice websocket contracts for audio frames, runtime-state
  updates, wake confirmations, same-stream transcript commits, follow-up windows,
  and barge-in interrupts
- host the server-side voice session that turns bounded edge audio into wake,
  continuation, and interruption decisions
- call the thh1986-backed remote ASR service for transcript-first wake
  detection, same-stream continuation, and barge-in decisions

`orchestrator` does **not** own:
- tool prompt authoring or concrete tool business logic
- CLI orchestration or runtime loop selection
- provider-specific model, search, or transport implementations
- general runtime-state behavior outside orchestrated turns
- edge audio capture or realtime loop state transitions; those stay in
  `src/twinr/agent/workflows`

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Package export surface |
| [contracts.py](./contracts.py) | Websocket message contracts |
| [client.py](./client.py) | Sync and async client |
| [server.py](./server.py) | FastAPI websocket server |
| [session.py](./session.py) | Tool-loop bridge |
| [remote_asr.py](./remote_asr.py) | Bounded HTTP client for the thh1986 remote ASR service |
| [remote_asr_service.py](./remote_asr_service.py) | Embedded `/v1/transcribe` HTTP surface when the remote-ASR URL points back to the local orchestrator server |
| [voice_gateway_guardrails.py](./voice_gateway_guardrails.py) | Fail-closed startup checks for the remote-only live voice gateway contract |
| [voice_activation.py](./voice_activation.py) | Transcript-first wake phrase matching and continuation extraction |
| [voice_forensics.py](./voice_forensics.py) | Bounded voice-frame telemetry shared by the edge and host-side voice path |
| [voice_audio_debug_store.py](./voice_audio_debug_store.py) | Opt-in rolling WAV artifact store for forensic-only voice debugging |
| [voice_transcript_debug_stream.py](./voice_transcript_debug_stream.py) | Bounded JSONL store for raw transcript-first gateway decisions |
| [voice_contracts.py](./voice_contracts.py) | Streaming voice websocket contracts |
| [voice_runtime_intent.py](./voice_runtime_intent.py) | Compact person-state projection for voice-gateway bias |
| [voice_client.py](./voice_client.py) | Blocking client for the voice websocket |
| [voice_session.py](./voice_session.py) | Stable compatibility wrapper for the server-side voice-session import surface |
| [voice_session_impl/](./voice_session_impl/) | Internal package split across runtime-state handling, observability, backend requests, and utterance scanning |
| [local_bridge_target.py](./local_bridge_target.py) | Host-side probe target resolver that rewrites stale self-LAN websocket URLs to the local `:8797` loopback bridge when available |
| [remote_tool_timeout.py](./remote_tool_timeout.py) | Shared default/env timeout policy for remote tool execution budgets across the client and server bridge |
| [probe_turn.py](./probe_turn.py) | Lightweight text-probe bootstrap for `--orchestrator-probe-turn` with stage timings and no full hardware-loop startup |
| [non_voice_acceptance.py](./non_voice_acceptance.py) | Deterministic direct/tool/memory text-only E2E acceptance runner with persisted artifacts |
| [acks.py](./acks.py) | Ack phrase ID map |
| [component.yaml](./component.yaml) | Structured package metadata |

## Usage

```python
from twinr.orchestrator import OrchestratorTurnRequest, OrchestratorWebSocketClient

client = OrchestratorWebSocketClient(
    "ws://127.0.0.1:8000/ws/orchestrator",
    require_tls=False,
)
result = client.run_turn(
    OrchestratorTurnRequest(prompt="Wie wird das Wetter morgen?"),
    tool_handlers={"search_live_info": lambda arguments: {"answer": arguments["question"]}},
)
```

```python
from twinr.orchestrator import create_app

app = create_app(".env")
```

Twinr's operator `--orchestrator-probe-turn` path now uses the lightweight
bootstrap in [`probe_turn.py`](./probe_turn.py): it still builds runtime
conversation context and the local realtime tool surface, but it deliberately
skips GPIO/audio/proactive/live-voice startup that belongs to the full hardware
loop instead of a text websocket probe. That keeps the probe bounded and makes
its stage timings actionable when Pi acceptance stalls.

On the leading repo host, that probe path also resolves stale self-targeted
LAN websocket URLs back to the authoritative local `ws://127.0.0.1:8797`
bridge when that loopback bridge is actually reachable. That keeps host-side
non-voice acceptance from depending on a DHCP-stable self-IP while leaving the
Pi acceptance instance on `/twinr` untouched.

The probe client and the server-side remote-tool bridge also share one timeout
policy from [`remote_tool_timeout.py`](./remote_tool_timeout.py). That keeps
host acceptance from failing a real live-web tool call locally while the server
is still legitimately waiting for the same tool result.

For a deterministic text-only acceptance proof across the real direct/tool/
memory paths, operators can now run:

```bash
PYTHONPATH=src python3 -m twinr --env-file .env --non-voice-e2e-acceptance
```

That runner executes one short direct turn, one live web/tool turn, and the
live synthetic-memory acceptance matrix, then persists the rolling ops artifact
to `artifacts/stores/ops/non_voice_e2e_acceptance.json` plus a per-run report
under `artifacts/reports/non_voice_e2e_acceptance/`.

Twinr's live voice gateway now has one supported activation path only:
`TWINR_VOICE_ORCHESTRATOR_REMOTE_ASR_URL` must point at the thh1986 ASR service.
The Pi streams room audio continuously to thh1986, thh1986 keeps a rolling
transcript/ringbuffer over that live stream, and the same stream stays open
until the spoken turn reaches a pause/end so the post-wake tail is committed as
the request input. The server rejects startup if somebody tries to relaunch
retired local voice selectors, reintroduce any selectable stage-one backend, or
silently routes the turn back into a second Pi wake/listen phase. Once that
transcript-first gateway is running, there is no alternate local detector or
fallback voice path behind the same live session.

Twinr also keeps follow-up and barge-in decisions on that same remote
`remote_asr` path; the live gateway must not require an `OpenAIBackend` or
`OPENAI_API_KEY` just to build the websocket session and emit `voice_ready`.

In that transcript-first mode the ASR service owns the primary speech/VAD gate.
Twinr keeps the extra candidate active-ratio gate open by default so quiet
far-field activations still reach transcript-first matching. The live gateway no
longer uses repeated tiny `remote_asr` wake scans as the runtime decision path.
Instead it starts one bounded utterance buffer at the latest speech burst,
keeps the quiet nonzero onset that immediately precedes that burst, and waits
for the same-stream endpoint silence before it performs one transcript-first
wake/follow-up decision over the buffered utterance. That keeps `Twinna, schau
mal ...` on one server-side stream instead of fragmenting the turn into a wake
micro-scan plus a second local listen phase.

The practical rules for that utterance scanner are:
- start buffering from the latest speech burst and preserve a bounded quiet
  lead-in so `Twinner` does not collapse into `Winner`
- keep buffering on the same stream until bounded endpoint silence or the
  utterance budget is reached
- transcribe that utterance once and route it as either `wake_confirmed`,
  `follow_up` transcript commit, or ignore
- accept wake matches only from the utterance head, aside from bounded leading
  generic greeting words such as `okay` or `hey`; do not treat a later
  mid-sentence `... Twinna` mention as a fresh wake
- do not route the turn into any retired local detector or rescue capture path

The default transcript-first wake phrase set now keeps the exact safe alias
family `Twinr`, `Twinna`, `Twina`, and `Twinner`, including their bounded
greeting forms such as `hey twinna`. Earlier broad `twi*` recovery rules and
an explicit `twitter` alias recovered some ASR misses, but live Pi/host
artifacts also showed they could wake Twinr from ordinary TV or room speech.
Twinr therefore keeps only the exact safe alias family by default and blocks
broader recovery like `twitter`, `tynna`, or generic `*winner` head variants.
Operators who intentionally want that broader alias recovery must opt into it
explicitly via `TWINR_VOICE_ACTIVATION_PHRASES` and accept the corresponding
false-positive tradeoff.

There is one narrower runtime-owned exception for the live idle wake path: when
the Pi currently attests a speech-directed local person with strong
speaker-association confidence, no background media, and no overlapping speech,
the gateway may check the wake candidate audio against the Pi-owned enrolled
household voice profiles that were synced over the websocket as a read-only
session snapshot. Only when that wake candidate also sounds like a known
household speaker may the `waiting` wake scanner temporarily expand the exact
safe family to include `Tynna` and the existing `*winner` head recovery. That
stronger alias tier never becomes the global default, never applies outside
`waiting`, and still requires an explicit utterance-head wake alias instead of
manufacturing wake from vision alone.

That keeps the websocket frame loop stable because the gateway performs one
decode per utterance instead of multiple overlapping synchronous ASR calls on
consecutive frames. The remote ASR adapter still keeps a narrow retry budget for
transient `429 stt busy` contention so one brief remote decode spike does
not immediately drop an utterance candidate.

When `TWINR_VOICE_ORCHESTRATOR_REMOTE_ASR_URL` points back to the same
orchestrator host/port, Twinr now mounts the bounded `/v1/transcribe` route in
that same FastAPI process instead of expecting a second hidden localhost-only
ASR daemon on another port. The live gateway still uses the same HTTP contract
through [`remote_asr.py`](./remote_asr.py), but the concrete transcription
surface is colocated with the websocket server and protected by the dedicated
remote-ASR bearer token.

When the Pi explicitly re-enters a fresh same-stream `listening` or
`follow_up_open` window, the server now clears stale wake/beep/answer history
before buffering the next utterance. That keeps the next transcript-first
commit focused on the new user speech instead of poisoning the first ASR window
with older playback audio that still sits in the rolling buffer. There is one
explicit exception for the `waiting -> listening` handoff: if the same stream
already has one still-active waiting-origin utterance in flight when the Pi
opens explicit `listening`, the server carries that utterance into
`listening` instead of deleting it with the stale history reset.

Every real transcript-first decision window is now also persisted as bounded
text-only JSONL evidence under
`artifacts/stores/ops/voice_gateway_transcripts.jsonl`. That file records the
raw STT text, matched wake alias, remaining post-wake text, and window-level
speech metrics for wake-stage scans plus follow-up/barge-in transcriptions so
operators can inspect what thh1986 actually heard after a failed live test
without persisting room audio.

For forensic-only live incidents, operators can also opt in to bounded WAV
artifacts with `TWINR_VOICE_ORCHESTRATOR_AUDIO_DEBUG_ENABLED=1`. When enabled,
the gateway stores only a short rolling set of the same bounded decision
windows under `artifacts/stores/ops/voice_gateway_audio/` and links each file
back from the matching transcript-debug JSONL entry. Keep that disabled outside
targeted debugging because it persists raw room audio by design.

The edge runtime now also projects the latest compact `person_state` summary
into each `voice_runtime_state` update: `attention_state`,
`interaction_intent_state`, `person_visible`, `presence_active`,
`interaction_ready`, `targeted_inference_blocked`, `recommended_channel`, the
compact speaker-association fields `speaker_associated` and
`speaker_association_confidence`, plus the media/overlap suppressors
`background_media_likely` and `speech_overlap_likely`. Separately, the edge may
sync a read-only `voice_identity_profiles` snapshot that contains only the
bounded enrolled household voice embeddings needed for wake-time familiar
speaker scoring. The gateway may use those summaries only as an audio-owned
bias, for example by keeping the transcript-first wake window a bit richer,
lowering the minimum activation duration a little, or temporarily unlocking the
stronger `Tynna`/`*winner` alias tier while the runtime already sees
speech-directed intent from the same nearby person and the current wake audio
sounds like a known household speaker. It must never manufacture a wake from
vision alone, and unfamiliar speakers must still be able to wake Twinr through
the exact safe alias family.

The same runtime-state payload now also carries an optional
`voice_quiet_until_utc` deadline. When that bounded runtime-owned quiet window
is active, the gateway must fail closed for transcript-first idle wake and
close any currently open same-stream follow-up window with an explicit
`voice_quiet_active` reason. This temporary quiet gate is not a second wake
path and it does not remove the explicit manual/button listen path.

The edge runtime now also primes one explicit idle `waiting` runtime-state as
soon as the voice websocket comes up, and `voice_hello` now carries that
attested initial state plus the compact `person_state` projection. That matters
because later sensor-observation refreshes can only replay changed intent
context after the gateway has seen at least one concrete runtime state.
Without that attested initial `waiting` state, the server would keep scanning
room audio with `intent_* = null` until the first real turn transition. The
gateway now fails closed until it receives that attested state. In `waiting`,
the transcript-first scanner still honors explicit no-local-presence blocks,
but it also keeps one short recent-visibility grace so a camera flicker cannot
kill an already buffered explicit wake burst mid-utterance. When the camera
stack itself is temporarily unavailable, the compact voice-intent projection no
longer converts that outage into a hard `person_visible=false` wake block. It
may still elevate attested local `near_device_presence` to a positive hold, but
otherwise it degrades to visibility-unknown so explicit audio wake stays
possible even with the camera offline. The broader room-clarity and channel
hints stay advisory for idle wake scanning, because the explicit wake phrase
itself is what establishes the speech turn and should not require proactive
targeting guards to classify the room first. That same rule now also applies
when the camera is online but misses the person: if the broader person-state
aggregate still attests `presence_active=true`, the gateway treats that as
local presence and does not hard-block explicit wake on `person_visible=false`
alone.

That compact runtime-state snapshot now also rides on each streamed
`voice_audio_frame`. The websocket still accepts standalone
`voice_runtime_state` messages, but the inline copy keeps wake scanning bounded
to the freshest edge context even when the server is busy draining queued audio
frames and a separate control message would arrive too late to prevent a stale
`person_visible=false` cancellation.

When the runtime is already in `follow_up_open`, the gateway stays on the same
utterance buffer path it uses in `waiting` and `listening`: keep buffering the
same remote stream until bounded endpoint silence, then run one transcript-
first decision over that utterance. That matters for repeated turns such as
`Twinna, ...` right after a successful answer because the first few 100 ms
frames are often too short for a stable transcript. Buffering the full
utterance keeps the follow-up lane deterministic: repeated wake phrases still
route as `wake_confirmed`, and ordinary continuation speech commits directly as
`transcript_committed`, without reviving any second local capture or generic
alternate STT stage.

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [../agent/tools/README.md](../agent/tools/README.md)
- [../agent/workflows/README.md](../agent/workflows/README.md)
