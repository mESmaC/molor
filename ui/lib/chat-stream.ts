export type RuntimeSource = "simulator" | "local";

export type ChatStreamEvent =
  | {
      type: "generation.started";
      request_id: string;
      runtime_source: RuntimeSource;
      model_id: string;
      backend_mode: string;
      quantization: string;
      device: string;
      stack_fingerprint: string;
      context_used: number;
      context_limit: number;
      input_tokens: number;
      metrics_estimated: boolean;
    }
  | {
      type: "route";
      request_id: string;
      capability: { id: string; weight: number }[];
      memory: {
        mode: "applied" | "abstained" | "unavailable" | "off";
        version: string | null;
      };
    }
  | {
      type: "retrieval";
      request_id: string;
      count: number;
      latency_ms: number;
      consolidation_state: string;
    }
  | {
      type: "token";
      request_id: string;
      index: number;
      text: string;
      ttft_ms?: number;
    }
  | {
      type: "metrics";
      request_id: string;
      completion_tokens: number;
      context_used: number;
      elapsed_ms: number;
      tokens_per_second: number;
    }
  | {
      type: "generation.completed";
      request_id: string;
      completion_tokens: number;
      context_used: number;
      elapsed_ms: number;
      tokens_per_second: number;
      finish_reason: string;
    }
  | { type: "error"; request_id: string; message: string };

export type BridgeHeartbeat = {
  type: "heartbeat";
  request_id: string;
  phase: "loading" | "prefill" | "decoding";
};

export type ChatStreamRecord = ChatStreamEvent | BridgeHeartbeat;

const IDENTIFIER = /^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$/;
const MAX_EVENT_BYTES = 64 * 1024;
const MAX_STREAM_BYTES = 32 * 1024 * 1024;
const MAX_EVENTS = 200_000;

export class StreamProtocolError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "StreamProtocolError";
  }
}

export function isNdjsonContentType(value: string | null) {
  if (value === null) return false;
  return value.split(";", 1)[0].trim().toLowerCase() === "application/x-ndjson";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function exactKeys(value: Record<string, unknown>, keys: string[], label: string) {
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  if (
    actual.length !== expected.length ||
    actual.some((key, index) => key !== expected[index])
  ) {
    throw new StreamProtocolError(`${label} has an unexpected field set.`);
  }
}

function boundedString(value: unknown, label: string, maximum = 512) {
  if (typeof value !== "string" || !value || value.length > maximum) {
    throw new StreamProtocolError(`${label} must be a bounded non-empty string.`);
  }
  return value;
}

function identifier(value: unknown, label: string) {
  const normalized = boundedString(value, label, 256);
  if (!IDENTIFIER.test(normalized)) {
    throw new StreamProtocolError(`${label} is not a valid identifier.`);
  }
  return normalized;
}

function finiteNumber(value: unknown, label: string, maximum = Number.MAX_SAFE_INTEGER) {
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    value < 0 ||
    value > maximum
  ) {
    throw new StreamProtocolError(`${label} must be a finite non-negative number.`);
  }
  return value;
}

function nonnegativeInteger(value: unknown, label: string, maximum = Number.MAX_SAFE_INTEGER) {
  const normalized = finiteNumber(value, label, maximum);
  if (!Number.isInteger(normalized)) {
    throw new StreamProtocolError(`${label} must be an integer.`);
  }
  return normalized;
}

function requestId(value: Record<string, unknown>) {
  return identifier(value.request_id, "request_id");
}

export function validateChatStreamRecord(
  value: unknown,
  options: { allowHeartbeat?: boolean } = {},
): ChatStreamRecord {
  if (!isRecord(value) || typeof value.type !== "string") {
    throw new StreamProtocolError("Runtime event must be an object with a type.");
  }

  switch (value.type) {
    case "generation.started": {
      exactKeys(
        value,
        [
          "type",
          "request_id",
          "runtime_source",
          "model_id",
          "backend_mode",
          "quantization",
          "device",
          "stack_fingerprint",
          "context_used",
          "context_limit",
          "input_tokens",
          "metrics_estimated",
        ],
        value.type,
      );
      if (value.runtime_source !== "simulator" && value.runtime_source !== "local") {
        throw new StreamProtocolError("runtime_source is invalid.");
      }
      if (typeof value.metrics_estimated !== "boolean") {
        throw new StreamProtocolError("metrics_estimated must be boolean.");
      }
      const contextUsed = nonnegativeInteger(value.context_used, "context_used");
      const contextLimit = nonnegativeInteger(value.context_limit, "context_limit");
      if (contextLimit < 1 || contextUsed > contextLimit) {
        throw new StreamProtocolError("context usage exceeds the declared context limit.");
      }
      return {
        type: value.type,
        request_id: requestId(value),
        runtime_source: value.runtime_source,
        model_id: boundedString(value.model_id, "model_id"),
        backend_mode: boundedString(value.backend_mode, "backend_mode"),
        quantization: boundedString(value.quantization, "quantization"),
        device: boundedString(value.device, "device"),
        stack_fingerprint: boundedString(
          value.stack_fingerprint,
          "stack_fingerprint",
        ),
        context_used: contextUsed,
        context_limit: contextLimit,
        input_tokens: nonnegativeInteger(value.input_tokens, "input_tokens"),
        metrics_estimated: value.metrics_estimated,
      };
    }
    case "route": {
      exactKeys(value, ["type", "request_id", "capability", "memory"], value.type);
      if (!Array.isArray(value.capability) || !value.capability.length || value.capability.length > 32) {
        throw new StreamProtocolError("capability must contain 1-32 routes.");
      }
      const capability = value.capability.map((route, index) => {
        if (!isRecord(route)) throw new StreamProtocolError("capability route is invalid.");
        exactKeys(route, ["id", "weight"], `capability[${index}]`);
        const weight = finiteNumber(route.weight, `capability[${index}].weight`, 1);
        return { id: identifier(route.id, `capability[${index}].id`), weight };
      });
      if (new Set(capability.map((route) => route.id)).size !== capability.length) {
        throw new StreamProtocolError("capability route IDs must be unique.");
      }
      if (!isRecord(value.memory)) throw new StreamProtocolError("memory route is invalid.");
      exactKeys(value.memory, ["mode", "version"], "memory");
      if (
        value.memory.mode !== "applied" &&
        value.memory.mode !== "abstained" &&
        value.memory.mode !== "unavailable" &&
        value.memory.mode !== "off"
      ) {
        throw new StreamProtocolError("memory mode is invalid.");
      }
      if (value.memory.version !== null && typeof value.memory.version !== "string") {
        throw new StreamProtocolError("memory version must be a string or null.");
      }
      if (typeof value.memory.version === "string") {
        boundedString(value.memory.version, "memory.version", 256);
      }
      const hasVersion = typeof value.memory.version === "string";
      if (
        ((value.memory.mode === "applied" || value.memory.mode === "abstained") &&
          !hasVersion) ||
        ((value.memory.mode === "off" || value.memory.mode === "unavailable") &&
          hasVersion)
      ) {
        throw new StreamProtocolError(
          "memory mode and version availability are inconsistent.",
        );
      }
      return {
        type: value.type,
        request_id: requestId(value),
        capability,
        memory: {
          mode: value.memory.mode,
          version: value.memory.version,
        },
      };
    }
    case "retrieval": {
      exactKeys(
        value,
        ["type", "request_id", "count", "latency_ms", "consolidation_state"],
        value.type,
      );
      return {
        type: value.type,
        request_id: requestId(value),
        count: nonnegativeInteger(value.count, "count", 1_024),
        latency_ms: finiteNumber(value.latency_ms, "latency_ms"),
        consolidation_state: boundedString(
          value.consolidation_state,
          "consolidation_state",
          256,
        ),
      };
    }
    case "token": {
      exactKeys(
        value,
        value.ttft_ms === undefined
          ? ["type", "request_id", "index", "text"]
          : ["type", "request_id", "index", "text", "ttft_ms"],
        value.type,
      );
      const text = boundedString(value.text, "token text", 16_384);
      return {
        type: value.type,
        request_id: requestId(value),
        index: nonnegativeInteger(value.index, "token index"),
        text,
        ...(value.ttft_ms === undefined
          ? {}
          : { ttft_ms: finiteNumber(value.ttft_ms, "ttft_ms") }),
      };
    }
    case "metrics": {
      exactKeys(
        value,
        [
          "type",
          "request_id",
          "completion_tokens",
          "context_used",
          "elapsed_ms",
          "tokens_per_second",
        ],
        value.type,
      );
      return {
        type: value.type,
        request_id: requestId(value),
        completion_tokens: nonnegativeInteger(
          value.completion_tokens,
          "completion_tokens",
        ),
        context_used: nonnegativeInteger(value.context_used, "context_used"),
        elapsed_ms: finiteNumber(value.elapsed_ms, "elapsed_ms"),
        tokens_per_second: finiteNumber(
          value.tokens_per_second,
          "tokens_per_second",
        ),
      };
    }
    case "generation.completed": {
      exactKeys(
        value,
        [
          "type",
          "request_id",
          "completion_tokens",
          "context_used",
          "elapsed_ms",
          "tokens_per_second",
          "finish_reason",
        ],
        value.type,
      );
      return {
        type: value.type,
        request_id: requestId(value),
        completion_tokens: nonnegativeInteger(
          value.completion_tokens,
          "completion_tokens",
        ),
        context_used: nonnegativeInteger(value.context_used, "context_used"),
        elapsed_ms: finiteNumber(value.elapsed_ms, "elapsed_ms"),
        tokens_per_second: finiteNumber(
          value.tokens_per_second,
          "tokens_per_second",
        ),
        finish_reason: boundedString(value.finish_reason, "finish_reason", 128),
      };
    }
    case "error": {
      exactKeys(value, ["type", "request_id", "message"], value.type);
      return {
        type: value.type,
        request_id: requestId(value),
        message: boundedString(value.message, "error message", 1_024),
      };
    }
    case "heartbeat": {
      if (!options.allowHeartbeat) {
        throw new StreamProtocolError("Heartbeat records are bridge-internal only.");
      }
      exactKeys(value, ["type", "request_id", "phase"], value.type);
      if (
        value.phase !== "loading" &&
        value.phase !== "prefill" &&
        value.phase !== "decoding"
      ) {
        throw new StreamProtocolError("heartbeat phase is invalid.");
      }
      return {
        type: value.type,
        request_id: requestId(value),
        phase: value.phase,
      };
    }
    default:
      throw new StreamProtocolError("Unknown runtime event type.");
  }
}

export class StreamSequenceValidator {
  private requestId: string | null;
  private started = false;
  private routeReceived = false;
  private retrievalReceived = false;
  private nextTokenIndex = 0;
  private lastCompletionTokens = 0;
  private lastContextUsed = 0;
  private terminal = false;

  constructor(expectedRequestId?: string) {
    this.requestId = expectedRequestId ?? null;
  }

  accept(record: ChatStreamRecord) {
    if (this.terminal) {
      throw new StreamProtocolError("Runtime emitted data after a terminal event.");
    }
    if (this.requestId === null) this.requestId = record.request_id;
    if (record.request_id !== this.requestId) {
      throw new StreamProtocolError("Runtime switched request identity mid-stream.");
    }
    if (record.type === "heartbeat") {
      if (!this.started) {
        throw new StreamProtocolError("Runtime heartbeat arrived before generation.started.");
      }
      return;
    }
    if (!this.started) {
      if (record.type === "error") {
        this.terminal = true;
        return;
      }
      if (record.type !== "generation.started") {
        throw new StreamProtocolError("First runtime event must be generation.started.");
      }
      this.started = true;
      this.lastContextUsed = record.context_used;
      return;
    }

    switch (record.type) {
      case "generation.started":
        throw new StreamProtocolError("generation.started may only appear once.");
      case "route":
        if (this.routeReceived || this.retrievalReceived || this.nextTokenIndex > 0) {
          throw new StreamProtocolError("Capability route arrived out of order.");
        }
        this.routeReceived = true;
        return;
      case "retrieval":
        if (!this.routeReceived || this.retrievalReceived || this.nextTokenIndex > 0) {
          throw new StreamProtocolError("Retrieval telemetry arrived out of order.");
        }
        this.retrievalReceived = true;
        return;
      case "token":
        this.requireReadyForGeneration();
        if (record.index !== this.nextTokenIndex) {
          throw new StreamProtocolError("Token indexes must be contiguous.");
        }
        this.nextTokenIndex += 1;
        return;
      case "metrics":
        this.requireReadyForGeneration();
        this.validateMonotonicMetrics(record.completion_tokens, record.context_used);
        return;
      case "generation.completed":
        this.requireReadyForGeneration();
        this.validateMonotonicMetrics(record.completion_tokens, record.context_used);
        this.terminal = true;
        return;
      case "error":
        this.terminal = true;
        return;
    }
  }

  finish() {
    if (!this.terminal) {
      throw new StreamProtocolError("Runtime stream ended without a terminal event.");
    }
  }

  get isTerminal() {
    return this.terminal;
  }

  private requireReadyForGeneration() {
    if (!this.routeReceived || !this.retrievalReceived) {
      throw new StreamProtocolError(
        "Generation output arrived before route and retrieval telemetry.",
      );
    }
  }

  private validateMonotonicMetrics(completionTokens: number, contextUsed: number) {
    if (
      completionTokens < this.lastCompletionTokens ||
      contextUsed < this.lastContextUsed
    ) {
      throw new StreamProtocolError("Runtime metrics moved backwards.");
    }
    this.lastCompletionTokens = completionTokens;
    this.lastContextUsed = contextUsed;
  }
}

export class NdjsonStreamDecoder {
  private readonly decoder = new TextDecoder("utf-8", { fatal: true });
  private readonly sequence: StreamSequenceValidator;
  private readonly allowHeartbeat: boolean;
  private buffer = "";
  private lineBytes = 0;
  private totalBytes = 0;
  private eventCount = 0;

  constructor(options: { expectedRequestId?: string; allowHeartbeat?: boolean } = {}) {
    this.sequence = new StreamSequenceValidator(options.expectedRequestId);
    this.allowHeartbeat = options.allowHeartbeat ?? false;
  }

  push(chunk: Uint8Array): ChatStreamRecord[] {
    this.totalBytes += chunk.byteLength;
    if (this.totalBytes > MAX_STREAM_BYTES) {
      throw new StreamProtocolError("Runtime stream exceeded the total byte limit.");
    }
    for (const byte of chunk) {
      if (byte === 10) {
        this.lineBytes = 0;
      } else {
        this.lineBytes += 1;
        if (this.lineBytes > MAX_EVENT_BYTES) {
          throw new StreamProtocolError("Runtime event exceeded 64 KiB.");
        }
      }
    }
    this.buffer += this.decoder.decode(chunk, { stream: true });
    const lines = this.buffer.split("\n");
    this.buffer = lines.pop() ?? "";
    return lines.flatMap((line) => this.parseLine(line));
  }

  finish(): ChatStreamRecord[] {
    this.buffer += this.decoder.decode();
    const records = this.buffer.trim() ? this.parseLine(this.buffer) : [];
    this.buffer = "";
    this.sequence.finish();
    return records;
  }

  get terminalReceived() {
    return this.sequence.isTerminal;
  }

  private parseLine(rawLine: string): ChatStreamRecord[] {
    const line = rawLine.endsWith("\r") ? rawLine.slice(0, -1) : rawLine;
    if (!line.trim()) return [];
    this.eventCount += 1;
    if (this.eventCount > MAX_EVENTS) {
      throw new StreamProtocolError("Runtime stream exceeded the event-count limit.");
    }
    let value: unknown;
    try {
      value = JSON.parse(line);
    } catch {
      throw new StreamProtocolError("Runtime emitted malformed NDJSON.");
    }
    const record = validateChatStreamRecord(value, {
      allowHeartbeat: this.allowHeartbeat,
    });
    this.sequence.accept(record);
    return [record];
  }
}

export function encodeNdjson(record: ChatStreamRecord) {
  return new TextEncoder().encode(`${JSON.stringify(record)}\n`);
}
