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

export function shouldProcessIncomingUpsert(
  upsertType,
  message,
  {
    normalizeJid = (value) => String(value || "").trim(),
    accountJid = null,
    accountLid = null,
  } = {},
) {
  const normalizedType = String(upsertType || "").trim();
  if (normalizedType === "notify") {
    return true;
  }
  if (normalizedType === "append") {
    if (Boolean(message?.key?.fromMe)) {
      return true;
    }
    const normalizedAccountJid = normalizeForComparison(accountJid, normalizeJid);
    const normalizedAccountLid = normalizeForComparison(accountLid, normalizeJid);
    if (!normalizedAccountJid && !normalizedAccountLid) {
      return false;
    }
    const normalizedConversationId = normalizeForComparison(message?.key?.remoteJid, normalizeJid);
    return normalizedConversationId === normalizedAccountJid || normalizedConversationId === normalizedAccountLid;
  }
  return false;
}

export function buildIncomingMessagePayload(
  message,
  {
    extractText = () => "",
    normalizeJid = (value) => String(value || "").trim(),
    accountJid = null,
    accountLid = null,
    receivedAt = null,
  } = {},
) {
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
  if (!text || !messageId || !conversationId || conversationId === "status@broadcast") {
    return null;
  }
  return {
    type: "incoming_message",
    message_id: messageId,
    conversation_id: conversationId,
    sender_id: senderId,
    sender_display_name: message?.pushName || null,
    text,
    received_at: receivedAt,
    is_group: conversationId.endsWith("@g.us"),
    is_from_self: Boolean(message?.key?.fromMe),
    account_jid: accountJid,
  };
}

export function collectIncomingMessagePayloads(upsert, options = {}) {
  const messages = Array.isArray(upsert?.messages) ? upsert.messages : [];
  const upsertType = String(upsert?.type || "").trim();
  const payloads = [];
  for (const message of messages) {
    if (!shouldProcessIncomingUpsert(upsertType, message, options)) {
      continue;
    }
    const payload = buildIncomingMessagePayload(message, options);
    if (payload) {
      payloads.push(payload);
    }
  }
  return payloads;
}
