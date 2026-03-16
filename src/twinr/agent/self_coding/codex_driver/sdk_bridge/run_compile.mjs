#!/usr/bin/env node
/**
 * Stream one Twinr self_coding compile turn through the pinned Codex SDK.
 *
 * The Python runtime owns workspace construction, validation, status
 * persistence, and final compile-result parsing. This bridge is intentionally
 * thin: read one JSON request from stdin, run a single streamed SDK turn in the
 * provided workspace, and forward structured SDK events as JSONL on stdout.
 */

import { Codex } from "@openai/codex-sdk";
import { spawnSync } from "child_process";

/**
 * Read a single JSON request object from stdin.
 *
 * Returns:
 *   A parsed request payload with workspaceRoot, prompt, and outputSchema.
 */
async function readRequestFromStdin() {
  const chunks = [];
  for await (const chunk of process.stdin) {
    chunks.push(typeof chunk === "string" ? chunk : chunk.toString("utf8"));
  }
  const raw = chunks.join("");
  if (!raw.trim()) {
    throw new Error("stdin did not contain a codex-sdk bridge request payload");
  }
  let payload;
  try {
    payload = JSON.parse(raw);
  } catch (error) {
    throw new Error(`stdin did not contain valid JSON: ${formatError(error)}`);
  }
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new Error("bridge request payload must be a JSON object");
  }
  return {
    workspaceRoot: requireString(payload.workspaceRoot, "workspaceRoot"),
    prompt: requireString(payload.prompt, "prompt"),
    outputSchema: requireObject(payload.outputSchema, "outputSchema"),
  };
}

/**
 * Require a non-empty string field in the request payload.
 *
 * Args:
 *   value: Candidate field value.
 *   fieldName: Human-readable field name for errors.
 *
 * Returns:
 *   The normalized string.
 */
function requireString(value, fieldName) {
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`${fieldName} must be a non-empty string`);
  }
  return value;
}

/**
 * Require a plain object field in the request payload.
 *
 * Args:
 *   value: Candidate field value.
 *   fieldName: Human-readable field name for errors.
 *
 * Returns:
 *   The original object value.
 */
function requireObject(value, fieldName) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${fieldName} must be a JSON object`);
  }
  return value;
}

/**
 * Write one JSONL event to stdout.
 *
 * Args:
 *   event: Serializable SDK event payload.
 */
function emitEvent(event) {
  process.stdout.write(`${JSON.stringify(event)}\n`);
}

/**
 * Extract a readable error message from arbitrary thrown values.
 *
 * Args:
 *   error: Unknown thrown value.
 *
 * Returns:
 *   A human-readable message string.
 */
function formatError(error) {
  if (error instanceof Error) {
    return error.message || error.name || "unknown error";
  }
  if (typeof error === "string") {
    return error;
  }
  try {
    return JSON.stringify(error);
  } catch {
    return String(error);
  }
}

async function main() {
  if (process.argv.includes("--self-test")) {
    const codex = new Codex();
    const executablePath = codex.exec?.executablePath;
    if (typeof executablePath !== "string" || !executablePath.trim()) {
      throw new Error("Codex SDK did not expose a usable CLI executable path");
    }
    const versionResult = spawnSync(executablePath, ["--version"], {
      encoding: "utf8",
    });
    if (versionResult.error) {
      throw new Error(`Codex CLI self-test failed to start: ${formatError(versionResult.error)}`);
    }
    if (versionResult.status !== 0) {
      throw new Error(
        `Codex CLI self-test exited with code ${versionResult.status ?? 1}: ${formatError(versionResult.stderr || versionResult.stdout)}`
      );
    }
    emitEvent({
      ok: true,
      nodeVersion: process.versions.node,
      codexPath: executablePath,
      codexVersion: String(versionResult.stdout || "").trim(),
    });
    return;
  }

  const request = await readRequestFromStdin();
  const codex = new Codex();
  const thread = codex.startThread({
    workingDirectory: request.workspaceRoot,
    sandboxMode: "workspace-write",
    approvalPolicy: "never",
    skipGitRepoCheck: true,
    networkAccessEnabled: false,
    webSearchMode: "disabled",
  });
  const { events } = await thread.runStreamed(request.prompt, {
    outputSchema: request.outputSchema,
  });
  for await (const event of events) {
    emitEvent(event);
  }
}

main().catch((error) => {
  emitEvent({
    type: "error",
    message: formatError(error),
  });
  process.exitCode = 1;
});
