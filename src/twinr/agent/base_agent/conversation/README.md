# conversation

Internal package for the base agent's conversation micro-policies: adaptive
listening windows, shared decision helpers, language contract helpers,
turn-boundary evaluation, and post-response closure checks.

## Responsibility

`conversation` owns:
- Adapt listening timeout, pause, and grace values from observed turns
- Build bounded language instructions for user replies and memory text
- Evaluate streaming turn boundaries through tool-calling providers
- Evaluate whether follow-up listening should close after a response, including a fast structured-decision path for OpenAI and a hard wall-clock watchdog when provider calls stall
- Accept machine-readable turn-steering cues, including compact semantic match summaries, so closure decisions can return matched shared-thread or cooling topics back to the runtime without over-matching nearby themes

`conversation` does **not** own:
- Audio capture, playback, or hardware control
- Workflow orchestration in `src/twinr/agent/workflows`
- Provider bundle construction or transport adapters
- Personality instruction authoring

## Key files

| File | Purpose |
|---|---|
| `adaptive_timing.py` | Persist bounded listening profile |
| `decision_core.py` | Shared bounded decision helpers and canonical transcript normalization |
| `language.py` | Build language and memory instructions |
| `turn_controller.py` | Evaluate streaming turn boundaries |
| `closure.py` | Evaluate post-response closure and echo matched steering topics |
| `__init__.py` | Mark package only |

`normalize_turn_text()` lives in `decision_core.py` and is the canonical import
for workflow consumers that need transcript equality checks outside the
controller itself.

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
- [realtime_runner.py](../../workflows/realtime_runner.py)
