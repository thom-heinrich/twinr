function normalizeForComparison(value, normalizeJid) {
  const normalized = normalizeJid(value);
  return String(normalized || "").trim();
}

function canonicalizeSelfChatJid(value, { normalizeJid, accountJid, accountLid }) {
  const normalizedValue = normalizeForComparison(value, normalizeJid);
  const normalizedAccountJid = normalizeForComparison(accountJid, normalizeJid);
  const normalizedAccountLid = normalizeForComparison(accountLid, normalizeJid);
  if (normalizedValue && normalizedAccountLid && normalizedValue === normalizedAccountLid && normalizedAccountJid) {
    return normalizedAccountJid;
  }
  return normalizedValue;
}

function toNumber(value) {
  if (value == null) {
    return null;
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "bigint") {
    return Number(value);
  }
  if (typeof value === "object" && typeof value.toNumber === "function") {
    try {
      return Number(value.toNumber());
    } catch {
      return null;
    }
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function normalizeTimestampMs(message) {
  const rawSeconds = toNumber(message?.messageTimestamp);
  if (rawSeconds == null) {
    return null;
  }
  if (rawSeconds > 1000_000_000_000) {
    return Math.trunc(rawSeconds);
  }
  return Math.trunc(rawSeconds * 1000);
}

function buildLabelMaps(chats, contacts, normalizeJid) {
  const labels = new Map();
  for (const item of Array.isArray(contacts) ? contacts : []) {
    const id = normalizeForComparison(item?.id, normalizeJid);
    const label = String(item?.name || item?.notify || item?.verifiedName || "").trim();
    if (id && label && !labels.has(id)) {
      labels.set(id, label);
    }
  }
  for (const item of Array.isArray(chats) ? chats : []) {
    const id = normalizeForComparison(item?.id, normalizeJid);
    const label = String(item?.name || item?.conversationTimestamp || "").trim();
    if (id && label && !labels.has(id)) {
      labels.set(id, label);
    }
  }
  return labels;
}

export function buildHistoryBatchPayload(
  historySet,
  {
    extractText = () => "",
    normalizeJid = (value) => String(value || "").trim(),
    accountJid = null,
    accountLid = null,
    cutoffTimestampMs = null,
  } = {},
) {
  const messages = Array.isArray(historySet?.messages) ? historySet.messages : [];
  const labels = buildLabelMaps(historySet?.chats, historySet?.contacts, normalizeJid);
  const normalizedMessages = [];
  for (const message of messages) {
    const text = String(extractText(message) || "").trim();
    const messageId = String(message?.key?.id || "").trim();
    const conversationId = canonicalizeSelfChatJid(message?.key?.remoteJid, {
      normalizeJid,
      accountJid,
      accountLid,
    });
    const senderId = canonicalizeSelfChatJid(message?.key?.participant || message?.participant || conversationId, {
      normalizeJid,
      accountJid,
      accountLid,
    });
    const timestampMs = normalizeTimestampMs(message);
    if (!text || !messageId || !conversationId || conversationId === "status@broadcast" || timestampMs == null) {
      continue;
    }
    if (cutoffTimestampMs != null && timestampMs < cutoffTimestampMs) {
      continue;
    }
    normalizedMessages.push({
      message_id: messageId,
      conversation_id: conversationId,
      sender_id: senderId,
      text,
      timestamp_ms: timestampMs,
      is_group: conversationId.endsWith("@g.us"),
      is_from_self: Boolean(message?.key?.fromMe),
      chat_label: labels.get(conversationId) || null,
      sender_label: labels.get(senderId) || message?.pushName || null,
    });
  }

  return {
    type: "history_batch",
    sync_type: historySet?.syncType ?? null,
    progress: historySet?.progress ?? null,
    is_latest: historySet?.isLatest ?? null,
    peer_data_request_session_id: historySet?.peerDataRequestSessionId ?? null,
    message_count: normalizedMessages.length,
    messages: normalizedMessages,
  };
}
