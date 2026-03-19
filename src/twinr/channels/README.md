# channels

`channels` owns Twinr's long-lived external messaging transports. It is the
runtime boundary for text channels that need to listen continuously, normalize
incoming messages into a shared contract, and hand those turns to the Twinr
agent.

## Responsibility

`channels` owns:
- generic inbound/outbound contracts for text-based messaging channels
- transport-facing runtime services that convert a text message into one Twinr turn
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
| [runtime.py](./runtime.py) | Shared text-turn service for external channels |
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
