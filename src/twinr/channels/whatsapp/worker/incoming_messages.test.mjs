import assert from "node:assert/strict";
import test from "node:test";

import {
  buildIncomingMessagePayload,
  collectIncomingMessagePayloads,
  shouldProcessIncomingUpsert,
} from "./incoming_messages.mjs";

const TEST_ACCOUNT_JID = "15555554567@s.whatsapp.net";
const TEST_ACCOUNT_PHONE_JID = "15555554567:18@s.whatsapp.net";
const TEST_OTHER_JID = "15555550222@s.whatsapp.net";
const TEST_ACCOUNT_LID = "test-account:28@lid";

function extractText(message) {
  return message?.text || "";
}

function normalizeJid(value) {
  const text = String(value || "").trim();
  if (!text) {
    return "";
  }
  const [head, tail = ""] = text.split("@", 2);
  const [user] = head.split(":", 1);
  return tail ? `${user}@${tail}` : user;
}

test("notify upserts are processed", () => {
  assert.equal(
    shouldProcessIncomingUpsert("notify", { key: { fromMe: false } }),
    true,
  );
});

test("append upserts are processed for self-authored messages", () => {
  assert.equal(
    shouldProcessIncomingUpsert("append", { key: { fromMe: true } }),
    true,
  );
});

test("append upserts are processed for self-chat account messages even when fromMe is false", () => {
  assert.equal(
    shouldProcessIncomingUpsert(
      "append",
      {
        key: {
          fromMe: false,
          remoteJid: TEST_ACCOUNT_PHONE_JID,
        },
      },
      {
        normalizeJid,
        accountJid: TEST_ACCOUNT_JID,
      },
    ),
    true,
  );
});

test("append upserts are ignored when they are neither fromMe nor self-chat account messages", () => {
  assert.equal(
    shouldProcessIncomingUpsert(
      "append",
      {
        key: {
          fromMe: false,
          remoteJid: TEST_OTHER_JID,
        },
      },
      {
        normalizeJid,
        accountJid: TEST_ACCOUNT_JID,
      },
    ),
    false,
  );
});

test("buildIncomingMessagePayload normalizes a self-chat message", () => {
  const payload = buildIncomingMessagePayload(
    {
      key: {
        id: "msg-1",
        remoteJid: TEST_ACCOUNT_JID,
        fromMe: true,
      },
      pushName: "Twinr Test",
      text: "Hallo Twinr",
    },
    {
      extractText,
      normalizeJid,
      accountJid: TEST_ACCOUNT_JID,
      receivedAt: "2026-03-25T19:00:00Z",
    },
  );

  assert.deepEqual(payload, {
    type: "incoming_message",
    message_id: "msg-1",
    conversation_id: TEST_ACCOUNT_JID,
    sender_id: TEST_ACCOUNT_JID,
    sender_display_name: "Twinr Test",
    text: "Hallo Twinr",
    received_at: "2026-03-25T19:00:00Z",
    is_group: false,
    is_from_self: true,
    account_jid: TEST_ACCOUNT_JID,
    upsert_type: null,
    worker_request_id: null,
    raw_remote_jid: TEST_ACCOUNT_JID,
    raw_remote_jid_alt: null,
    raw_participant: null,
    raw_participant_alt: null,
    message_timestamp: null,
    context_stanza_id: null,
  });
});

test("buildIncomingMessagePayload canonicalizes own account lid traffic to the stable account jid", () => {
  const payload = buildIncomingMessagePayload(
    {
      key: {
        id: "msg-lid",
        remoteJid: TEST_ACCOUNT_LID,
        fromMe: true,
      },
      pushName: "Twinr Test",
      text: "Hallo vom Handy",
    },
    {
      extractText,
      normalizeJid,
      accountJid: TEST_ACCOUNT_JID,
      accountLid: TEST_ACCOUNT_LID,
      receivedAt: "2026-03-25T19:00:00Z",
    },
  );

  assert.deepEqual(payload, {
    type: "incoming_message",
    message_id: "msg-lid",
    conversation_id: TEST_ACCOUNT_JID,
    sender_id: TEST_ACCOUNT_JID,
    sender_display_name: "Twinr Test",
    text: "Hallo vom Handy",
    received_at: "2026-03-25T19:00:00Z",
    is_group: false,
    is_from_self: true,
    account_jid: TEST_ACCOUNT_JID,
    upsert_type: null,
    worker_request_id: null,
    raw_remote_jid: TEST_ACCOUNT_LID,
    raw_remote_jid_alt: null,
    raw_participant: null,
    raw_participant_alt: null,
    message_timestamp: null,
    context_stanza_id: null,
  });
});

test("buildIncomingMessagePayload carries worker provenance fields", () => {
  const payload = buildIncomingMessagePayload(
    {
      key: {
        id: "msg-debug",
        remoteJid: TEST_ACCOUNT_PHONE_JID,
        remoteJidAlt: TEST_ACCOUNT_LID,
        participant: TEST_ACCOUNT_PHONE_JID,
        participantAlt: TEST_ACCOUNT_LID,
        fromMe: false,
      },
      text: "Hallo Twinr",
      messageTimestamp: 1774539523,
      message: {
        extendedTextMessage: {
          text: "Hallo Twinr",
          contextInfo: {
            stanzaId: "stanza-123",
          },
        },
      },
    },
    {
      extractText,
      normalizeJid,
      accountJid: TEST_ACCOUNT_JID,
      accountLid: TEST_ACCOUNT_LID,
      receivedAt: "2026-03-25T19:00:00Z",
      upsertType: "append",
      workerRequestId: "req-123",
    },
  );

  assert.equal(payload?.upsert_type, "append");
  assert.equal(payload?.worker_request_id, "req-123");
  assert.equal(payload?.raw_remote_jid, TEST_ACCOUNT_PHONE_JID);
  assert.equal(payload?.raw_remote_jid_alt, TEST_ACCOUNT_LID);
  assert.equal(payload?.raw_participant, TEST_ACCOUNT_PHONE_JID);
  assert.equal(payload?.raw_participant_alt, TEST_ACCOUNT_LID);
  assert.equal(payload?.message_timestamp, "1774539523");
  assert.equal(payload?.context_stanza_id, "stanza-123");
});

test("collectIncomingMessagePayloads forwards append self-chat turns", () => {
  const payloads = collectIncomingMessagePayloads(
    {
      type: "append",
      messages: [
        {
          key: {
            id: "msg-append",
            remoteJid: TEST_ACCOUNT_JID,
            fromMe: true,
          },
          text: "Self chat message",
        },
      ],
    },
    {
      extractText,
      normalizeJid,
      accountJid: TEST_ACCOUNT_JID,
      receivedAt: "2026-03-25T19:00:00Z",
    },
  );

  assert.equal(payloads.length, 1);
  assert.equal(payloads[0].message_id, "msg-append");
  assert.equal(payloads[0].is_from_self, true);
});

test("collectIncomingMessagePayloads forwards append self-chat turns from the primary phone", () => {
  const payloads = collectIncomingMessagePayloads(
    {
      type: "append",
      messages: [
        {
          key: {
            id: "msg-phone",
            remoteJid: TEST_ACCOUNT_PHONE_JID,
            fromMe: false,
          },
          text: "Phone self chat message",
        },
      ],
    },
    {
      extractText,
      normalizeJid,
      accountJid: TEST_ACCOUNT_JID,
      receivedAt: "2026-03-25T19:00:00Z",
    },
  );

  assert.equal(payloads.length, 1);
  assert.equal(payloads[0].message_id, "msg-phone");
  assert.equal(payloads[0].conversation_id, TEST_ACCOUNT_JID);
  assert.equal(payloads[0].is_from_self, false);
});

test("collectIncomingMessagePayloads forwards notify self-chat turns that arrive on the account lid", () => {
  const payloads = collectIncomingMessagePayloads(
    {
      type: "notify",
      messages: [
        {
          key: {
            id: "msg-notify-lid",
            remoteJid: TEST_ACCOUNT_LID,
            fromMe: true,
          },
          text: "Hallo!",
        },
      ],
    },
    {
      extractText,
      normalizeJid,
      accountJid: TEST_ACCOUNT_JID,
      accountLid: TEST_ACCOUNT_LID,
      receivedAt: "2026-03-25T19:00:00Z",
    },
  );

  assert.equal(payloads.length, 1);
  assert.equal(payloads[0].conversation_id, TEST_ACCOUNT_JID);
  assert.equal(payloads[0].sender_id, TEST_ACCOUNT_JID);
  assert.equal(payloads[0].is_from_self, true);
});
