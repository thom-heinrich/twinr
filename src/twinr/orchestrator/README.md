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
  updates, wake confirmations, follow-up windows, and barge-in interrupts
- host the server-side voice session that turns bounded edge audio into wake,
  continuation, and interruption decisions
- call the thh1986-backed local STT service for transcript-first wake
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
| [local_stt.py](./local_stt.py) | Bounded HTTP client for colocated local STT |
| [voice_contracts.py](./voice_contracts.py) | Streaming voice websocket contracts |
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

Twinr's live voice gateway now has one supported wake path only:
`TWINR_VOICE_ORCHESTRATOR_WAKE_STAGE1_MODE=local_stt` with
`TWINR_VOICE_ORCHESTRATOR_LOCAL_STT_URL` pointing at the thh1986 STT service.
The Pi streams room audio continuously to thh1986, thh1986 keeps a rolling
transcript/ringbuffer over that live stream, and the same stream stays open
until the spoken turn reaches a pause/end so the post-wake tail is committed as
the request input. The server rejects startup if somebody tries to relaunch the
old `backend/openwakeword/wekws` rescue path, enables any generic wakeword
backend in the gateway process, or silently routes the turn back into a second
local Pi wake/listen phase.

Twinr also keeps generic follow-up and barge-in transcription on that same
colocated `local_stt` path; the live gateway must not require an
`OpenAIBackend` or `OPENAI_API_KEY` just to build the websocket session and
emit `voice_ready`.

In that transcript-first mode the STT service owns the primary speech/VAD gate.
Twinr keeps the extra candidate active-ratio gate open by default so quiet
far-field wakewords still reach transcript-first matching. Those scans stay
intentionally cheap: Twinr transcribes one short rolling stage-one window for
wake candidacy, but it anchors that short window at the start of the latest
active speech burst instead of always using the newest tail. This matters
because a short colocated STT decode can finish hundreds of milliseconds after
the user started speaking; by then the newest 1000 ms tail can already contain
only `schau mal ...` while the wake token has fallen out of the window.
Anchoring at the burst start preserves the wake prefix without growing
stage-one compute, and Twinr keeps a tiny bounded lead-in from the quiet onset
immediately before the active burst so `Twinner` does not degrade to `Winner`
when the first consonant lands below the active speech threshold. Twinr still
waits until one scan is finished before the next wake scan can run, so the
websocket frame loop stays near real time even when one colocated STT decode
takes a few hundred milliseconds.

After `local_stt` has matched the wake phrase, Twinr stops rescanning the
rolling stream for richer wake matches and instead uses the bounded
post-roll/tail buffer plus the local STT tail extractor once to recover
`remaining_text`. That preserves same-stream `Twinna, ...` handling without
stacking multiple synchronous STT calls on every incoming frame. In this mode
Twinr refuses to commit on the first short silence after the wake unless the
same rolling transcript has already seen some post-wake text; otherwise it
keeps buffering until the longer tail budget so `Twinna, ...` does not get cut
off after a brief pause between wakeword and command. The local STT adapter
also supports a small bounded retry budget for transient `429 stt busy`
contention so one brief server-side decode spike does not immediately drop a
wake candidate.

When the runtime is already in `follow_up_open`, Twinr now keeps the same
stage-one wake detector alive before it considers the generic follow-up STT
path. That matters for repeated turns such as `Twinna, ...` right after a
successful answer: the first few 100 ms frames are often too short for a full
transcript match, so Twinr buffers until the configured follow-up window is
actually full instead of transcribing an immature partial window and routing
that partial text into `follow_up_capture_requested`. If the user really is
starting a new wake turn, stage one now gets first claim on the same stream;
only a mature, wake-free follow-up window may reopen a bounded local follow-up
capture on the Pi.

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [../agent/tools/README.md](../agent/tools/README.md)
- [../agent/workflows/README.md](../agent/workflows/README.md)
