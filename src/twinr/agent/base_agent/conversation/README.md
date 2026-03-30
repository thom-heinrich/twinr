# conversation

Internal package for the base agent's conversation micro-policies: adaptive
listening windows, shared decision helpers, language contract helpers,
turn-boundary evaluation, and post-response closure checks.

## Responsibility

`conversation` owns:
- Adapt listening timeout, pause, and grace values from observed turns
- Build bounded language instructions for user replies and memory text
- Evaluate streaming turn boundaries through tool-calling providers
- Evaluate whether follow-up listening should close after a response, including an explicit structured `follow_up_action` contract for continue/end, question-style fast paths, a fast structured-decision path for OpenAI, and a hard wall-clock watchdog when provider calls stall
- Derive and store one bounded immediate follow-up carryover hint from the just-finished exchange so short clarification turns can preserve the newly established anchors from the still-open thread
- Build bounded recent-thread carryover system messages for text turns so short repairs or elliptical continuations can keep the anchors from the immediate thread even outside the live follow-up window
- Expose privacy-safe observability helpers for that carryover hint so workflows can trace whether a hint was built, stored, cleared, or injected without logging raw user text
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
| `follow_up_context.py` | Build and store bounded carryover hints for follow-up and text-turn thread continuity, plus privacy-safe trace payload helpers for follow-up accountability |
| `thread_resolution.py` | Focus text turns on the immediate recent thread and optionally rewrite short repair prompts into standalone requests |
| `__init__.py` | Mark package only |

`normalize_turn_text()` lives in `decision_core.py` and is the canonical import
for workflow consumers that need transcript equality checks outside the
controller itself.

## Follow-up carryover accountability

Immediate follow-up carryover is intentionally observable without leaking raw
speech text. `follow_up_context.py` exports privacy-safe trace helpers that
reduce transcripts and summaries to bounded presence, length, and short hash
metadata. Workflow consumers use those helpers to record:

- whether Twinr built a carryover hint after an answer
- whether the runtime stored or cleared that hint for the next turn
- whether provider-context builders actually injected the pending hint

This split keeps the carryover policy inside `conversation` while letting the
runtime prove, after a live bug, whether the right thread anchor was available
to the next clarification turn.

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
