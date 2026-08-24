import {
  ChatStreamEvent,
  ChatStreamRecord,
  NdjsonStreamDecoder,
  encodeNdjson,
  isNdjsonContentType,
} from "@/lib/chat-stream";

import type { LocalRuntimeConfig } from "./runtime-config";

type RuntimeChatRequest = {
  messages: { role: "user" | "assistant"; content: string }[];
  memory: "auto" | "off";
  conversation_id: string;
};

type AbortKind = "client" | "connect" | "first-event" | "idle" | "total" | null;

const SAFE_UPSTREAM_ERROR = "The local runtime reported a generation error.";

export class LocalRuntimeError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "LocalRuntimeError";
  }
}

function publicEvents(records: ChatStreamRecord[]): ChatStreamEvent[] {
  const events: ChatStreamEvent[] = [];
  for (const record of records) {
    if (record.type === "heartbeat") continue;
    if (record.type === "generation.started" && record.runtime_source !== "local") {
      throw new Error("The local runtime declared an invalid runtime source.");
    }
    if (record.type === "error") {
      events.push({ ...record, message: SAFE_UPSTREAM_ERROR });
    } else {
      events.push(record);
    }
  }
  return events;
}

function failureForAbort(kind: AbortKind) {
  if (kind === "connect") {
    return new LocalRuntimeError(504, "The local runtime did not accept the request in time.");
  }
  if (kind === "first-event") {
    return new LocalRuntimeError(504, "The local runtime did not acknowledge generation in time.");
  }
  if (kind === "idle") {
    return new LocalRuntimeError(504, "The local runtime stopped sending progress.");
  }
  if (kind === "total") {
    return new LocalRuntimeError(504, "The local generation exceeded its time limit.");
  }
  if (kind === "client") {
    return new LocalRuntimeError(499, "The chat request was cancelled.");
  }
  return new LocalRuntimeError(502, "The local runtime connection failed.");
}

export async function proxyLocalRuntime(
  body: RuntimeChatRequest,
  config: LocalRuntimeConfig,
  downstreamSignal: AbortSignal,
) {
  const requestId = `runtime-${crypto.randomUUID()}`;
  const upstreamController = new AbortController();
  let abortKind: AbortKind = null;
  let connectTimer: ReturnType<typeof setTimeout> | null = null;
  let firstEventTimer: ReturnType<typeof setTimeout> | null = null;
  let idleTimer: ReturnType<typeof setTimeout> | null = null;
  const totalTimer = setTimeout(() => abort("total"), config.totalTimeoutMs);

  const abort = (kind: Exclude<AbortKind, null>) => {
    if (upstreamController.signal.aborted) return;
    abortKind = kind;
    upstreamController.abort();
  };
  const handleDownstreamAbort = () => abort("client");
  downstreamSignal.addEventListener("abort", handleDownstreamAbort, { once: true });
  if (downstreamSignal.aborted) abort("client");

  const clearTimer = (timer: ReturnType<typeof setTimeout> | null) => {
    if (timer !== null) clearTimeout(timer);
  };
  const cleanup = () => {
    clearTimer(connectTimer);
    clearTimer(firstEventTimer);
    clearTimer(idleTimer);
    clearTimeout(totalTimer);
    downstreamSignal.removeEventListener("abort", handleDownstreamAbort);
  };
  const resetIdleTimer = () => {
    clearTimer(idleTimer);
    idleTimer = setTimeout(() => abort("idle"), config.idleTimeoutMs);
  };

  connectTimer = setTimeout(() => abort("connect"), config.connectTimeoutMs);
  let upstream: Response;
  try {
    upstream = await fetch(config.endpoint, {
      method: "POST",
      redirect: "error",
      headers: {
        accept: "application/x-ndjson",
        authorization: `Bearer ${config.token}`,
        "content-type": "application/json",
        "x-molor-protocol": "molor.chat-stream.v1",
      },
      body: JSON.stringify({
        schema_version: "molor.chat-request.v1",
        request_id: requestId,
        conversation_id: body.conversation_id,
        messages: body.messages,
        memory: body.memory,
      }),
      signal: upstreamController.signal,
    });
  } catch {
    cleanup();
    throw failureForAbort(abortKind);
  }
  clearTimer(connectTimer);
  connectTimer = null;

  if (upstream.status !== 200) {
    upstreamController.abort();
    cleanup();
    if (upstream.status === 503) {
      throw new LocalRuntimeError(503, "The local runtime is not ready.");
    }
    if (upstream.status === 401 || upstream.status === 403) {
      throw new LocalRuntimeError(502, "The local runtime rejected bridge authentication.");
    }
    throw new LocalRuntimeError(502, "The local runtime rejected the generation request.");
  }
  if (
    !isNdjsonContentType(upstream.headers.get("content-type")) ||
    upstream.headers.get("x-molor-protocol") !== "molor.chat-stream.v1"
  ) {
    upstreamController.abort();
    cleanup();
    throw new LocalRuntimeError(502, "The local runtime returned an invalid stream protocol.");
  }
  if (!upstream.body) {
    cleanup();
    throw new LocalRuntimeError(502, "The local runtime returned an empty response.");
  }

  const reader = upstream.body.getReader();
  const decoder = new NdjsonStreamDecoder({
    expectedRequestId: requestId,
    allowHeartbeat: true,
  });
  let pending: ChatStreamEvent[] = [];
  firstEventTimer = setTimeout(() => abort("first-event"), config.firstEventTimeoutMs);
  try {
    while (pending.length === 0) {
      const { done, value } = await reader.read();
      if (done) {
        pending = publicEvents(decoder.finish());
        break;
      }
      pending = publicEvents(decoder.push(value));
    }
  } catch {
    upstreamController.abort();
    await reader.cancel().catch(() => undefined);
    cleanup();
    throw failureForAbort(abortKind);
  }
  clearTimer(firstEventTimer);
  firstEventTimer = null;
  if (pending.length === 0) {
    await reader.cancel().catch(() => undefined);
    cleanup();
    throw new LocalRuntimeError(502, "The local runtime ended before acknowledging generation.");
  }
  resetIdleTimer();

  let downstreamCancelled = false;
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      let downstreamClosed = false;
      const emit = (record: ChatStreamEvent) => {
        if (!downstreamCancelled && !downstreamClosed) {
          controller.enqueue(encodeNdjson(record));
        }
      };
      const close = () => {
        if (!downstreamCancelled && !downstreamClosed) {
          downstreamClosed = true;
          controller.close();
        }
      };
      const emitRecords = (records: ChatStreamEvent[]) => {
        for (const record of records) emit(record);
        return decoder.terminalReceived;
      };

      try {
        if (emitRecords(pending)) {
          close();
          return;
        }
        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            emitRecords(publicEvents(decoder.finish()));
            close();
            return;
          }
          resetIdleTimer();
          if (emitRecords(publicEvents(decoder.push(value)))) {
            close();
            return;
          }
        }
      } catch {
        if (!downstreamCancelled && !downstreamClosed) {
          if (decoder.terminalReceived) {
            close();
          } else {
            emit({
              type: "error",
              request_id: requestId,
              message:
                abortKind === "idle"
                  ? "The local runtime stopped sending progress."
                  : abortKind === "total"
                    ? "The local generation exceeded its time limit."
                    : "The local runtime stream failed validation.",
            });
            close();
          }
        }
      } finally {
        upstreamController.abort();
        await reader.cancel().catch(() => undefined);
        cleanup();
      }
    },
    async cancel() {
      downstreamCancelled = true;
      abort("client");
      await reader.cancel().catch(() => undefined);
      cleanup();
    },
  });

  return new Response(stream, {
    headers: {
      "cache-control": "no-store",
      "content-type": "application/x-ndjson; charset=utf-8",
      "x-content-type-options": "nosniff",
      "x-molor-protocol": "molor.chat-stream.v1",
    },
  });
}
