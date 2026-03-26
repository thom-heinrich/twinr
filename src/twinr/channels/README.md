# channels

`channels` owns Twinr's long-lived external messaging transports. It is the
runtime boundary for text channels that need to listen continuously, normalize
incoming messages into a shared contract, and hand those turns to the Twinr
agent.

## Responsibility

`channels` owns:
- generic inbound/outbound contracts for text-based messaging channels
- transport-facing runtime services that convert a text message into one Twinr turn, including both a plain text-completion bridge and a shared tool-capable turn bridge
- channel-aware hidden instruction helpers so external text transports can reuse one bounded response-style policy while staying on Twinr's shared memory and tool surface
- bounded pending-action tracking for external text turns that need a structured follow-up answer on the next message, including file-backed draft persistence so a channel-service restart does not erase an open WhatsApp send flow
- bounded model-guided follow-up resolution for active WhatsApp send drafts so later turns keep using the structured draft instead of falling back to free-form chat
- channel-owned direct-send hooks for cases where the active messaging loop itself must complete a send tool call without enqueueing work and waiting on its own thread
- provenance handoff so external text turns keep their channel identity and text modality through runtime, long-term memory, and personality learning instead of collapsing into a generic voice turn
- startup warmup hooks for shared text-channel runtime dependencies when a cold remote-only path would otherwise penalize the first live turn
- runtime-neutral channel-onboarding snapshots and bounded pairing coordinators that can be reused by both web and voice-triggered service-connect flows
- bounded service-connect entrypoints that turn a spoken "verbinde mich mit ..." request into one supported pairing flow plus display-facing QR/status cues
- channel-specific listener loops, auth/session persistence, reconnect policy, and low-level worker bridges

`channels` does **not** own:
- the core Twinr runtime state machine in [`agent/base_agent`](../agent/README.md)
- provider SDK implementations in [`providers`](../providers/README.md)
- synchronous external integrations in [`integrations`](../integrations/README.md)
- Raspberry Pi bootstrap or operator workflows in [`hardware`](../hardware/README.md) and [`ops`](../ops/README.md)

## Layout

| Path | Purpose |
|---|---|
| [contracts.py](./contracts.py) | Generic text-channel message contracts |
| [onboarding.py](./onboarding.py) | Runtime-neutral channel pairing snapshot persistence and in-process pairing registry |
| [pending_actions.py](./pending_actions.py) | File-backed pending structured follow-up state for external text channels |
| [pending_whatsapp_followup.py](./pending_whatsapp_followup.py) | Bounded model-guided continuation for open WhatsApp send drafts |
| [runtime.py](./runtime.py) | Shared plain text-turn service for external channels |
| [turn_instructions.py](./turn_instructions.py) | Channel-aware hidden instruction builder for shared external text turns |
| [tool_runtime.py](./tool_runtime.py) | Shared tool-capable text-turn service for external channels |
| [service_connect.py](./service_connect.py) | Voice/runtime entrypoint for bounded service-connect and pairing flows |
| [whatsapp](./whatsapp/README.md) | Baileys-backed WhatsApp transport |
| [component.yaml](./component.yaml) | Structured package ownership metadata |
| [AGENTS.md](./AGENTS.md) | Package-specific engineering rules |

## Design notes

- Keep generic channel contracts provider-agnostic so Signal, Telegram, and telephony can reuse them later.
- Keep low-level socket or worker lifecycle logic inside the concrete channel package.
- Prefer a thin listener loop that delegates policy and turn execution to focused helpers.

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [whatsapp](./whatsapp/README.md)
