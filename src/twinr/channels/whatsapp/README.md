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
- accept bounded outbound send requests from the main Twinr runtime through a shared on-disk queue so the long-lived channel service remains the only socket/session owner
- accept bounded WhatsApp history-import requests from the portal/runtime through a shared on-disk queue so the same long-lived channel service can temporarily reopen the worker in history mode without creating a second linked-device socket

## Runtime contract

- The worker speaks newline-delimited JSON over stdout.
- Human-readable logs stay on stderr, while QR updates are emitted both as terminal output and as portal/runtime-renderable SVG plus data-URL payloads for the dashboard wizard and the HDMI right-lane reserve panel.
- Only text-like WhatsApp messages are promoted into Twinr turns.
- WhatsApp history import stays disabled by default; Twinr does not background-sync old chats into memory.
- When the operator explicitly enables social-history learning in the portal and starts an import, the channel loop claims one bounded history-import request, temporarily restarts the worker with history sync enabled, and imports only the approved lookback window into shared Twinr memory.
- Self-chat turns authored on the linked main account are accepted from both Baileys `notify` and `append` upserts, even when the primary phone surfaces them as account-chat traffic without `fromMe=true`, so WhatsApp's own no-notification self-chat path still reaches Twinr.
- Twinr canonicalizes the account's own WhatsApp `@lid` self-chat identity back onto the stable account JID before Python policy checks, so allowlist and self-chat routing do not silently drop manual primary-phone self-chat messages.
- Accepted inbound WhatsApp message IDs are short-term de-duplicated before turn execution so a `notify` plus `append` pair does not trigger two Twinr replies.
- Outbound replies are sent back to the same chat JID that produced the accepted inbound message.
- Accepted inbound WhatsApp turns run through Twinr's shared tool-capable text-channel agent path, so reminders, shared memory writes, contact lookup, and bounded `send_whatsapp_message` use the same core tool surface instead of a standalone chat-only completion path.
- Inbound WhatsApp send flows keep a bounded pending draft across clarification turns, so "Schreib Anna auf WhatsApp" -> "Was soll drinstehen?" -> "<Text>" -> "Ja" stays attached to the same structured send action instead of depending on free-form history reconstruction.
- When a `send_whatsapp_message` tool call originates from the active WhatsApp channel itself, the loop now sends through its already-open worker transport directly instead of queueing and waiting on the same loop thread; the on-disk outbound queue remains only for cross-process callers such as voice/runtime and portal flows.
- Proactive sends to other remembered contacts go through `outbound/pending -> processing -> results` under `state/channels/whatsapp/`, so voice/runtime turns can wait for delivery without starting a second Baileys worker.
- Portal-triggered history imports go through `history_import/pending -> processing -> results` under `state/channels/whatsapp/`, and imported historical turns are marked to survive normal episode retention because they came from an explicit bounded external-history import.

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

Run the channel loop once, or trigger the bounded pairing flow through Twinr's
voice/runtime service-connect path, scan the QR shown in the dashboard wizard,
right-hand HDMI reserve panel, or terminal fallback with WhatsApp's
"Linked Devices" flow, and keep the persisted auth directory stable between
restarts.

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
