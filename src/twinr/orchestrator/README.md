# orchestrator

`orchestrator` owns Twinr's provider-neutral websocket transport for an
edge/cloud split. It packages the message contracts, client/server entrypoints,
ack mapping, and remote-tool bridge that lets a remote peer execute local Twinr
tools through one bounded turn protocol.

## Responsibility

`orchestrator` owns:
- define websocket contracts for turn, tool, ack, delta, and completion traffic
- expose client/server entrypoints for edge-orchestrator deployments
- bridge remote tool calls and results into Twinr's local dual-lane tool loop
- keep fast ack phrases mapped to stable transport IDs

`orchestrator` does **not** own:
- tool prompt authoring or concrete tool business logic
- CLI orchestration or runtime loop selection
- provider-specific model, search, or transport implementations
- general runtime-state behavior outside orchestrated turns

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Package export surface |
| [contracts.py](./contracts.py) | Websocket message contracts |
| [client.py](./client.py) | Sync and async client |
| [server.py](./server.py) | FastAPI websocket server |
| [session.py](./session.py) | Tool-loop bridge |
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

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [../agent/tools/README.md](../agent/tools/README.md)
- [../agent/workflows/README.md](../agent/workflows/README.md)
