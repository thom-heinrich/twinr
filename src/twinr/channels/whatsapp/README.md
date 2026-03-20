# whatsapp

This package connects Twinr to consumer WhatsApp through a dedicated Baileys
worker process. The Python side owns Twinr turn execution, allowlist policy,
echo filtering, and worker lifecycle. The Node worker owns the WhatsApp Web
socket, auth persistence, QR login, and low-level reconnect handling.

Current Baileys releases require `Node.js 20+`. The Python bridge checks this
up front and fails closed when the configured `TWINR_WHATSAPP_NODE_BINARY`
points to an older runtime. When the config still uses the generic `node`
default, Twinr automatically prefers the pinned project-local runtime under
`state/tools/node-v20.20.1-<platform>-<arch>/bin/node` once it has been staged.
Relative `TWINR_WHATSAPP_*` filesystem paths are resolved against Twinr's
`project_root`, not against the worker subprocess cwd.

## Responsibility

- start and supervise the Baileys worker process
- persist WhatsApp linked-device auth in its own directory
- normalize inbound WhatsApp messages into the generic `channels` contract
- enforce exactly-one-user allowlist, group blocking, and self-chat echo guards
- deliver outbound text replies back to the originating chat

## Runtime contract

- The worker speaks newline-delimited JSON over stdout.
- Human-readable logs stay on stderr, while QR updates are emitted both as terminal output and as a portal-renderable SVG payload for the dashboard wizard.
- Only text-like WhatsApp messages are promoted into Twinr turns.
- Outbound replies are sent back to the same chat JID that produced the accepted inbound message.

## Configuration

The runtime reads the following env-backed fields from `TwinrConfig`:

- `TWINR_WHATSAPP_ALLOW_FROM`
- `TWINR_WHATSAPP_AUTH_DIR`
- `TWINR_WHATSAPP_WORKER_ROOT`
- `TWINR_WHATSAPP_NODE_BINARY`
- `TWINR_WHATSAPP_GROUPS_ENABLED`
- `TWINR_WHATSAPP_SELF_CHAT_MODE`
- `TWINR_WHATSAPP_RECONNECT_BASE_DELAY_S`
- `TWINR_WHATSAPP_RECONNECT_MAX_DELAY_S`
- `TWINR_WHATSAPP_SEND_TIMEOUT_S`
- `TWINR_WHATSAPP_SENT_CACHE_TTL_S`
- `TWINR_WHATSAPP_SENT_CACHE_MAX_ENTRIES`

## Pairing

Run the channel loop once, scan the QR shown in the dashboard wizard or the
terminal fallback with WhatsApp's "Linked Devices" flow, and keep the
persisted auth directory stable between restarts.

## Worker Dependencies

The Baileys worker is pinned by `worker/package-lock.json`. Twinr now treats
that lockfile as authoritative at runtime:

- if the required packages are already installed and match the lockfile marker,
  startup continues immediately
- if the packages are missing or the marker is stale, Twinr runs `npm ci` in
  the worker folder before launching the Node process, using the same effective
  Node runtime that the Baileys worker itself will use
- if that install cannot complete, the WhatsApp channel fails closed with a
  clear startup error instead of repeatedly half-starting

## Local Node runtime

Stage the pinned local Node.js runtime with:

```bash
python3 hardware/ops/install_whatsapp_node_runtime.py
```

That installs the official Node.js archive into `state/tools/` and lets Twinr
use the staged runtime without relying on a system-wide `node` binary.
