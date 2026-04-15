# prompting

`prompting` owns the hidden instruction text and builder helpers for Twinr's
tool-capable agent lanes. It is the canonical package for assembling the
default, compact, supervisor, first-word, and specialist instruction bundles.

## Responsibility

`prompting` owns:
- define the canonical tool-lane instruction constants and optional model-authored progress-line policy
- force freshness-sensitive external answers to wait for an explicit verification tool instead of being answered from model memory or estimation
- encode when the fast supervisor must carry structured `location_hint` / `date_context` fields and when it must escalate memory-dependent turns to fuller context
- encode that fresh place/business/event status questions must not stay direct just because the likely answer may be negative or uncertain
- merge time and simple-setting context into tool-lane instruction bundles
- build route-aware first-word overlays for authoritative local semantic-router handoffs so the instant spoken bridge stays LLM-generated and provisional
- encode the learned-skill tool policy for self-coding activation, rollback, pause, and reactivation
- encode that active self-coding follow-up answers and activation requests must reuse runtime-known `session_id` / `job_id` state instead of asking the user for internal ids
- encode when the model should discover, filter, aggregate, control, or inspect smart-home state through the dedicated tool surface
- encode that exact routed smart-home entity IDs spoken directly by the user may go straight to `read_smart_home_state` without a redundant relist step
- encode that the user's own smart-home inventory, room/device state, and recent in-home smart-home events are local tool work, not web-search work
- encode that generic fresh or deeper topical web questions should prefer `search_live_info` first, while `browser_automation` is reserved for explicit site interaction or an explicitly user-approved deeper site check after insufficient web research
- encode that unresolved search results such as "could not verify" or "no current evidence found" must not be framed as the final answer when a specific site check could still clarify the requested detail, and that the model should key this decision off structured `verification_status`, `question_resolved`, and `site_follow_up_recommended` search fields when available
- encode that insufficient ordinary web research should trigger a proactive, model-authored offer to try a slower but stronger alternate method when a deeper site check could materially help, instead of stopping at a generic unresolved answer
- encode that `search_live_info` is not a substitute for stepping through booking flows, forms, checkout state, or social profile/post/story checks on the live site
- encode when the model should use `connect_service_integration` for spoken external-service pairing requests and explicitly mention that the QR or progress state appears on the right info panel
- encode when the model should use `send_whatsapp_message` for remembered-contact messaging, including missing-message follow-ups, contact clarification, and explicit final confirmation before delivery
- encode that exact later recall of free-form facts saved with `remember_memory` must use the explicit `review_saved_memories` tool instead of relying on hidden prompt memory text, and that confirmation-required retries must call it again with `confirmed=true` only after a clear yes
- encode when the model should use `manage_voice_quiet_mode` for bounded stay-quiet requests caused by TV/radio/background speech, that ambiguous no-duration quiet requests must ask for a duration instead of pretending quiet is already active, that quiet-status questions should use the `status` action, that resume/listen-again requests should use the `clear` action, that supervisor lanes must hand those runtime-local control turns off instead of answering directly, that structured supervisor decisions may carry direct `runtime_tool_name` plus `runtime_tool_arguments_json` for one-shot local control handoffs, and that this temporary runtime gate is not the same as `end_conversation`
- encode that a short assent to a previously proposed deeper website check counts as explicit browser authorization for that same topic, while sufficiently answered search results should not grow an extra optional browser offer
- encode that a freshness-sensitive question about a place, business, organization, or event does not by itself count as explicit browser authorization just because an official website likely exists
- encode when the model should start discovery, continue discovery from free self-disclosure or wish-form self-statements about preferred name/addressing, treat the active discovery speaker as the current profile subject, prioritize explicit correction/deletion of already learned details over normal next-answer handling, review learned profile facts, split distinct learned details into separate discovery routes, and use replace/delete actions with exact reviewed fact ids instead of inventing ad-hoc profile narration
- encode the generic multi-query planning policy for broad smart-home and house-status questions so agents compose live status from multiple tool reads instead of a special-case summary path
- expose public builders for default, compact, supervisor, first-word, and specialist lanes

`prompting` does **not** own:
- personality file loading or section ordering in `base_agent/prompting`
- tool schema definitions or handler binding
- workflow-loop orchestration or provider transport behavior

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Prompting export surface |
| [instructions.py](./instructions.py) | Canonical tool-lane instruction builders |
| [component.yaml](./component.yaml) | Structured package metadata |

## Usage

```python
from twinr.agent.tools.prompting import build_tool_agent_instructions

instructions = build_tool_agent_instructions(config)
```

## See also

- [component.yaml](./component.yaml)
- [base_agent/prompting](../../base_agent/prompting/README.md)
- [schemas](../schemas/README.md)
