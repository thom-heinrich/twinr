# conversation

Internal package for the base agent's conversation micro-policies: adaptive
listening windows, language contract helpers, turn-boundary evaluation, and
post-response closure checks.

## Responsibility

`conversation` owns:
- Adapt listening timeout, pause, and grace values from observed turns
- Build bounded language instructions for user replies and memory text
- Evaluate streaming turn boundaries through tool-calling providers
- Evaluate whether follow-up listening should close after a response

`conversation` does **not** own:
- Audio capture, playback, or hardware control
- Workflow orchestration in `src/twinr/agent/workflows`
- Provider bundle construction or transport adapters
- Personality instruction authoring

## Key files

| File | Purpose |
|---|---|
| `adaptive_timing.py` | Persist bounded listening profile |
| `language.py` | Build language and memory instructions |
| `turn_controller.py` | Evaluate streaming turn boundaries |
| `closure.py` | Evaluate post-response closure |
| `__init__.py` | Mark package only |

## Usage

```python
from twinr.agent.base_agent.conversation.adaptive_timing import AdaptiveTimingStore
from twinr.agent.base_agent.conversation.closure import (
    ToolCallingConversationClosureEvaluator,
)
from twinr.agent.base_agent.conversation.turn_controller import (
    StreamingTurnController,
    ToolCallingTurnDecisionEvaluator,
)

store = AdaptiveTimingStore(config.adaptive_timing_store_path, config=config)
window = store.listening_window(initial_source="button", follow_up=False)

controller = StreamingTurnController(
    config=config,
    evaluator=ToolCallingTurnDecisionEvaluator(config=config, provider=provider),
    conversation_factory=lambda: conversation,
)
closure = ToolCallingConversationClosureEvaluator(config=config, provider=provider)
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [runtime](../runtime/README.md)
- [runner.py](../../workflows/runner.py)
- [realtime_runner.py](../../workflows/realtime_runner.py)
