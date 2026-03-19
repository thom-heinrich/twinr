/**
 * Run the Baileys-based WhatsApp worker for Twinr over newline-delimited JSON.
 *
 * Stdout is reserved for machine-readable JSONL events:
 * - worker_ready
 * - status / fatal
 * - incoming_message
 * - send_result
 *
 * Stderr is reserved for operator logs, including QR output for linked-device
 * pairing.
 */

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import readline from "node:readline";

import makeWASocket, {
  Browsers,
  DisconnectReason,
  extractMessageContent,
  fetchLatestBaileysVersion,
  getContentType,
  jidNormalizedUser,
  makeCacheableSignalKeyStore,
  useMultiFileAuthState,
} from "@whiskeysockets/baileys";
import pino from "pino";
import QRCodeSvg from "qrcode";
import QRCodeTerminal from "qrcode-terminal";

const logger = pino({ level: "silent" });
const authDir = path.resolve(process.env.TWINR_WHATSAPP_AUTH_DIR || "./auth");
const reconnectBaseMs = parsePositiveInt(process.env.TWINR_WHATSAPP_RECONNECT_BASE_MS, 2000);
const reconnectMaxMs = Math.max(
  reconnectBaseMs,
  parsePositiveInt(process.env.TWINR_WHATSAPP_RECONNECT_MAX_MS, 30000),
);

/** @type {ReturnType<typeof makeWASocket> | null} */
let socket = null;
let stopRequested = false;
let reconnectAttempt = 0;
let reconnectTimer = null;
let currentAccountJid = null;

function emit(payload) {
  process.stdout.write(`${JSON.stringify(payload)}\n`);
}

function log(message = "") {
  process.stderr.write(message.endsWith("\n") ? message : `${message}\n`);
}

function parsePositiveInt(value, fallback) {
  const parsed = Number.parseInt(String(value || ""), 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return fallback;
  }
  return parsed;
}

function normalizeJid(value) {
  if (!value) {
    return "";
  }
  try {
    return jidNormalizedUser(value);
  } catch {
    return String(value).trim();
  }
}

function nowIso() {
  return new Date().toISOString();
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function renderQr(qr) {
  log("[twinr-whatsapp] Scan the QR below with WhatsApp Linked Devices:");
  QRCodeTerminal.generate(qr, { small: true }, (rendered) => {
    process.stderr.write(`${rendered}\n`);
  });
}

async function buildQrSvg(qr) {
  try {
    return await QRCodeSvg.toString(qr, {
      type: "svg",
      errorCorrectionLevel: "M",
      margin: 1,
      width: 280,
    });
  } catch (error) {
    log(`[twinr-whatsapp] Failed to build QR SVG for portal rendering: ${error}`);
    return null;
  }
}

function extractText(message) {
  const content = extractMessageContent(message?.message);
  const type = getContentType(content);
  if (!content || !type) {
    return "";
  }
  if (type === "conversation") {
    return String(content.conversation || "");
  }
  if (type === "extendedTextMessage") {
    return String(content.extendedTextMessage?.text || "");
  }
  if (type === "imageMessage" || type === "videoMessage" || type === "documentMessage") {
    return String(content[type]?.caption || "");
  }
  return "";
}

function describeDisconnect(statusCode) {
  if (typeof statusCode !== "number") {
    return "unknown";
  }
  return DisconnectReason[statusCode] || `status_${statusCode}`;
}

function scheduleReconnect(reason, statusCode) {
  if (stopRequested) {
    return;
  }
  reconnectAttempt += 1;
  const reconnectInMs = Math.min(reconnectBaseMs * 2 ** (reconnectAttempt - 1), reconnectMaxMs);
  emit({
    type: "status",
    connection: "reconnecting",
    detail: reason,
    status_code: statusCode ?? null,
    reconnect_in_ms: reconnectInMs,
    account_jid: currentAccountJid,
    at: nowIso(),
  });
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
  }
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    void startSocket();
  }, reconnectInMs);
  reconnectTimer.unref?.();
}

async function startSocket() {
  if (stopRequested) {
    return;
  }
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  fs.mkdirSync(authDir, { recursive: true });
  emit({
    type: "status",
    connection: "connecting",
    detail: "starting_socket",
    account_jid: currentAccountJid,
    at: nowIso(),
  });
  const { state, saveCreds } = await useMultiFileAuthState(authDir);
  let version;
  try {
    ({ version } = await fetchLatestBaileysVersion());
  } catch (error) {
    log(`[twinr-whatsapp] Failed to fetch latest Baileys version, continuing with bundled defaults: ${error}`);
  }
  const nextSocket = makeWASocket({
    auth: {
      creds: state.creds,
      keys: makeCacheableSignalKeyStore(state.keys, logger),
    },
    browser: Browsers.macOS("Twinr"),
    fireInitQueries: true,
    logger,
    markOnlineOnConnect: false,
    printQRInTerminal: false,
    shouldSyncHistoryMessage: () => false,
    syncFullHistory: false,
    version,
  });
  socket = nextSocket;
  currentAccountJid = normalizeJid(nextSocket.user?.id || state.creds?.me?.id) || currentAccountJid;

  nextSocket.ev.on("creds.update", saveCreds);
  nextSocket.ev.on("connection.update", async (update) => {
    if (socket !== nextSocket) {
      return;
    }
    const statusCode = update?.lastDisconnect?.error?.output?.statusCode ?? null;
    currentAccountJid = normalizeJid(nextSocket.user?.id || nextSocket.authState?.creds?.me?.id) || currentAccountJid;
    if (update?.qr) {
      renderQr(update.qr);
      const qrSvg = await buildQrSvg(update.qr);
      emit({
        type: "status",
        connection: update.connection || "qr",
        qr_available: true,
        qr_svg: qrSvg,
        account_jid: currentAccountJid,
        at: nowIso(),
      });
    }
    if (update?.connection) {
      emit({
        type: "status",
        connection: update.connection,
        detail: statusCode == null ? null : describeDisconnect(statusCode),
        status_code: statusCode,
        account_jid: currentAccountJid,
        at: nowIso(),
      });
    }
    if (update?.connection === "open") {
      reconnectAttempt = 0;
      emit({
        type: "status",
        connection: "open",
        account_jid: currentAccountJid,
        at: nowIso(),
      });
      return;
    }
    if (update?.connection !== "close" || stopRequested) {
      return;
    }

    const reason = describeDisconnect(statusCode);
    if (statusCode === DisconnectReason.loggedOut || statusCode === DisconnectReason.badSession) {
      emit({
        type: "fatal",
        connection: "close",
        detail: reason,
        status_code: statusCode,
        account_jid: currentAccountJid,
        fatal: true,
        at: nowIso(),
      });
      await sleep(10);
      process.exit(1);
      return;
    }
    scheduleReconnect(reason, statusCode);
  });

  nextSocket.ev.on("messages.upsert", (upsert) => {
    if (socket !== nextSocket || upsert?.type !== "notify") {
      return;
    }
    for (const message of upsert.messages || []) {
      const text = extractText(message).trim();
      const messageId = String(message?.key?.id || "").trim();
      const conversationId = normalizeJid(message?.key?.remoteJid);
      const senderId = normalizeJid(message?.key?.participant || message?.participant || conversationId);
      if (!text || !messageId || !conversationId || conversationId === "status@broadcast") {
        continue;
      }
      emit({
        type: "incoming_message",
        message_id: messageId,
        conversation_id: conversationId,
        sender_id: senderId,
        sender_display_name: message?.pushName || null,
        text,
        received_at: nowIso(),
        is_group: conversationId.endsWith("@g.us"),
        is_from_self: Boolean(message?.key?.fromMe),
        account_jid: currentAccountJid,
      });
    }
  });
}

async function handleCommand(rawLine) {
  const line = String(rawLine || "").trim();
  if (!line) {
    return;
  }
  let command;
  try {
    command = JSON.parse(line);
  } catch (error) {
    log(`[twinr-whatsapp] Ignoring malformed command JSON: ${error}`);
    return;
  }

  if (command.type === "shutdown") {
    stopRequested = true;
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    try {
      socket?.end(undefined);
    } catch {
      // Best effort only.
    }
    emit({
      type: "status",
      connection: "closed",
      detail: "shutdown",
      account_jid: currentAccountJid,
      at: nowIso(),
    });
    await sleep(10);
    process.exit(0);
    return;
  }

  if (command.type !== "send_text") {
    log(`[twinr-whatsapp] Ignoring unsupported command type: ${command.type}`);
    return;
  }

  const requestId = String(command.request_id || "").trim();
  const chatJid = normalizeJid(command.chat_jid);
  const text = String(command.text || "").trim();
  if (!requestId) {
    log("[twinr-whatsapp] send_text command missing request_id");
    return;
  }
  if (!socket) {
    emit({
      type: "send_result",
      request_id: requestId,
      ok: false,
      error: "socket_not_ready",
    });
    return;
  }
  if (!chatJid || !text) {
    emit({
      type: "send_result",
      request_id: requestId,
      ok: false,
      error: "chat_jid_and_text_required",
    });
    return;
  }

  try {
    const sent = await socket.sendMessage(chatJid, { text });
    emit({
      type: "send_result",
      request_id: requestId,
      ok: true,
      message_id: sent?.key?.id || null,
    });
  } catch (error) {
    emit({
      type: "send_result",
      request_id: requestId,
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    });
  }
}

process.on("SIGINT", () => {
  stopRequested = true;
});
process.on("SIGTERM", () => {
  stopRequested = true;
});

emit({
  type: "worker_ready",
  connection: "booting",
  detail: "worker_ready",
  at: nowIso(),
});

const lineReader = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
});
lineReader.on("line", (line) => {
  void handleCommand(line);
});
lineReader.on("close", () => {
  stopRequested = true;
});

void startSocket().catch((error) => {
  emit({
    type: "fatal",
    connection: "boot_failed",
    detail: error instanceof Error ? error.message : String(error),
    fatal: true,
    account_jid: currentAccountJid,
    at: nowIso(),
  });
  process.exit(1);
});
