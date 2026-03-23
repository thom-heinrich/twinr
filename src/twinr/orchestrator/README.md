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
| [voice_transcript_debug_stream.py](./voice_transcript_debug_stream.py) | Bounded JSONL store for raw transcript-first gateway decisions |
| [voice_contracts.py](./voice_contracts.py) | Streaming voice websocket contracts |
| [voice_runtime_intent.py](./voice_runtime_intent.py) | Compact person-state projection for voice-gateway bias |
| [voice_client.py](./voice_client.py) | Blocking client for the voice websocket |
| [voice_session.py](./voice_session.py) | Server-side wake/follow-up/barge-in session |
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

Twinr's live voice gateway now has one supported activation path only:
`TWINR_VOICE_ORCHESTRATOR_WAKE_STAGE1_MODE=remote_asr` with
`TWINR_VOICE_ORCHESTRATOR_REMOTE_ASR_URL` pointing at the thh1986 ASR service.
The Pi streams room audio continuously to thh1986, thh1986 keeps a rolling
transcript/ringbuffer over that live stream, and the same stream stays open
until the spoken turn reaches a pause/end so the post-wake tail is committed as
the request input. The server rejects startup if somebody tries to relaunch the
old `backend/openwakeword/wekws` rescue path, enables any generic wakeword
backend in the gateway process, or silently routes the turn back into a second
local Pi wake/listen phase. Once that transcript-first gateway is running, it
must also keep stage one on `remote_asr`; no hidden `wekws/openwakeword`
frame-detector fallback is allowed behind the same live session.

Twinr also keeps generic follow-up and barge-in transcription on that same
remote `remote_asr` path; the live gateway must not require an
`OpenAIBackend` or `OPENAI_API_KEY` just to build the websocket session and
emit `voice_ready`.

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
- do not call any deleted local-detector tail extractor path

That keeps the websocket frame loop stable because the gateway performs one
decode per utterance instead of multiple overlapping synchronous ASR calls on
consecutive frames. The remote ASR adapter still keeps a narrow retry budget for
transient `429 stt busy` contention so one brief remote decode spike does
not immediately drop an utterance candidate.

Every real transcript-first decision window is now also persisted as bounded
text-only JSONL evidence under
`artifacts/stores/ops/voice_gateway_transcripts.jsonl`. That file records the
raw STT text, matched wake alias, remaining post-wake text, and window-level
speech metrics for wake-stage scans plus follow-up/barge-in transcriptions so
operators can inspect what thh1986 actually heard after a failed live test
without persisting room audio.

The edge runtime now also projects the latest compact `person_state` summary
into each `voice_runtime_state` update: `attention_state`,
`interaction_intent_state`, `person_visible`, `interaction_ready`,
`targeted_inference_blocked`, and `recommended_channel`. The gateway may use
that summary only as an audio-owned bias, for example by keeping the
transcript-first wake window a bit richer or the follow-up window open a bit
longer when the runtime already sees speech-directed intent. It must never
manufacture a wake from vision alone.

When the runtime is already in `follow_up_open`, Twinr now keeps the same
stage-one wake detector alive before it considers the generic follow-up STT
path. That matters for repeated turns such as `Twinna, ...` right after a
successful answer: the first few 100 ms frames are often too short for a full
transcript match, so Twinr buffers until the configured follow-up window is
actually full instead of transcribing an immature partial window and routing
that partial text into a second Pi-local capture. If the user really is
starting a new wake turn, stage one now gets first claim on the same stream;
otherwise a mature, wake-free follow-up window is committed directly as
`transcript_committed` on that same remote stream.

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [../agent/tools/README.md](../agent/tools/README.md)
- [../agent/workflows/README.md](../agent/workflows/README.md)
