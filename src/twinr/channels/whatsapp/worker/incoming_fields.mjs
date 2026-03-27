function toTrimmedString(value) {
  const text = String(value || "").trim();
  return text || null;
}

function toTimestampString(value) {
  if (value == null) {
    return null;
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return String(Math.trunc(value));
  }
  if (typeof value === "bigint") {
    return String(value);
  }
  if (typeof value === "string") {
    const normalized = value.trim();
    return normalized || null;
  }
  if (typeof value === "object" && typeof value?.toNumber === "function") {
    try {
      return String(Math.trunc(value.toNumber()));
    } catch {
      return null;
    }
  }
  return null;
}

function firstContextInfo(message) {
  const content = message?.message;
  if (!content || typeof content !== "object") {
    return null;
  }
  for (const value of Object.values(content)) {
    if (value && typeof value === "object" && value.contextInfo && typeof value.contextInfo === "object") {
      return value.contextInfo;
    }
  }
  return null;
}

export function extractIncomingWorkerFields(message, { upsertType = null, requestId = null } = {}) {
  const contextInfo = firstContextInfo(message);
  return {
    upsert_type: toTrimmedString(upsertType),
    worker_request_id: toTrimmedString(requestId),
    raw_remote_jid: toTrimmedString(message?.key?.remoteJid),
    raw_remote_jid_alt: toTrimmedString(message?.key?.remoteJidAlt),
    raw_participant: toTrimmedString(message?.key?.participant || message?.participant),
    raw_participant_alt: toTrimmedString(message?.key?.participantAlt),
    message_timestamp: toTimestampString(message?.messageTimestamp),
    context_stanza_id: toTrimmedString(contextInfo?.stanzaId),
  };
}
