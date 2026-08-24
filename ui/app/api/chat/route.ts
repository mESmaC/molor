import { LocalRuntimeError, proxyLocalRuntime } from "./local-runtime";
import { RuntimeConfigurationError, loadRuntimeConfig } from "./runtime-config";
import { ChatRequest, createSimulatorResponse } from "./simulator";

const MAX_REQUEST_BYTES = 64 * 1024;
const IDENTIFIER = /^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$/;

class PayloadTooLargeError extends Error {}

function isLoopback(hostname: string) {
  return hostname === "localhost" || hostname === "127.0.0.1" || hostname === "[::1]";
}

function hasExactKeys(value: object, keys: string[]) {
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  return actual.length === expected.length && actual.every((key, index) => key === expected[index]);
}

async function readJsonBody(request: Request): Promise<unknown> {
  const declaredLength = Number(request.headers.get("content-length"));
  if (Number.isFinite(declaredLength) && declaredLength > MAX_REQUEST_BYTES) {
    throw new PayloadTooLargeError();
  }
  if (!request.body) throw new SyntaxError("Missing request body");

  const reader = request.body.getReader();
  const decoder = new TextDecoder("utf-8", { fatal: true });
  let totalBytes = 0;
  let text = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    totalBytes += value.byteLength;
    if (totalBytes > MAX_REQUEST_BYTES) {
      await reader.cancel();
      throw new PayloadTooLargeError();
    }
    text += decoder.decode(value, { stream: true });
  }
  text += decoder.decode();
  return JSON.parse(text);
}

function isChatRequest(value: unknown): value is ChatRequest {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  if (!hasExactKeys(value, ["conversation_id", "memory", "messages"])) return false;
  const candidate = value as Partial<ChatRequest>;
  if (candidate.memory !== "auto" && candidate.memory !== "off") return false;
  if (
    typeof candidate.conversation_id !== "string" ||
    !IDENTIFIER.test(candidate.conversation_id)
  ) {
    return false;
  }
  if (!Array.isArray(candidate.messages) || candidate.messages.length === 0) return false;
  if (candidate.messages.length > 64) return false;
  if (candidate.messages.at(-1)?.role !== "user") return false;
  return candidate.messages.every(
    (message) =>
      message &&
      typeof message === "object" &&
      !Array.isArray(message) &&
      hasExactKeys(message, ["content", "role"]) &&
      (message.role === "user" || message.role === "assistant") &&
      typeof message.content === "string" &&
      message.content.trim().length > 0 &&
      message.content.length <= 12_000,
  );
}

export async function POST(request: Request) {
  const requestUrl = new URL(request.url);
  const authenticatedUserId = request.headers.get("oai-authenticated-user-id");
  if (!isLoopback(requestUrl.hostname) && !authenticatedUserId) {
    return Response.json({ error: "Authentication required." }, { status: 401 });
  }
  const origin = request.headers.get("origin");
  if (origin) {
    try {
      if (new URL(origin).origin !== requestUrl.origin) {
        return Response.json(
          { error: "Cross-origin chat requests are not allowed." },
          { status: 403 },
        );
      }
    } catch {
      return Response.json(
        { error: "Cross-origin chat requests are not allowed." },
        { status: 403 },
      );
    }
  }
  if (request.headers.get("content-type")?.split(";", 1)[0].trim() !== "application/json") {
    return Response.json({ error: "Content-Type must be application/json." }, { status: 415 });
  }

  let body: unknown;
  try {
    body = await readJsonBody(request);
  } catch (error) {
    if (error instanceof PayloadTooLargeError) {
      return Response.json({ error: "Chat request exceeds 64 KiB." }, { status: 413 });
    }
    return Response.json({ error: "Request body must be JSON." }, { status: 400 });
  }
  if (!isChatRequest(body)) {
    return Response.json({ error: "Invalid chat request." }, { status: 400 });
  }
  const totalCharacters = body.messages.reduce((sum, message) => sum + message.content.length, 0);
  if (totalCharacters > 48_000) {
    return Response.json({ error: "Conversation payload exceeds the preview limit." }, { status: 413 });
  }

  // conversation_id is correlation only. The local runtime token selects a server-side
  // memory scope; no browser identity, cookie, or arbitrary header is forwarded upstream.
  let runtime;
  try {
    runtime = loadRuntimeConfig();
  } catch (error) {
    if (error instanceof RuntimeConfigurationError) {
      return Response.json(
        { error: "Local runtime configuration is invalid." },
        { status: 500 },
      );
    }
    throw error;
  }
  if (runtime.mode === "simulator") return createSimulatorResponse(body);

  try {
    return await proxyLocalRuntime(body, runtime, request.signal);
  } catch (error) {
    if (error instanceof LocalRuntimeError) {
      return Response.json({ error: error.message }, { status: error.status });
    }
    return Response.json({ error: "The local runtime bridge failed." }, { status: 502 });
  }
}
