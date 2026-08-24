export type SimulatorRuntimeConfig = {
  mode: "simulator";
};

export type LocalRuntimeConfig = {
  mode: "local";
  endpoint: string;
  token: string;
  connectTimeoutMs: number;
  firstEventTimeoutMs: number;
  idleTimeoutMs: number;
  totalTimeoutMs: number;
};

export type RuntimeConfig = SimulatorRuntimeConfig | LocalRuntimeConfig;

export class RuntimeConfigurationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "RuntimeConfigurationError";
  }
}

function timeout(
  value: string | undefined,
  fallback: number,
  label: string,
  minimum: number,
  maximum: number,
) {
  if (value === undefined || value === "") return fallback;
  if (!/^\d+$/.test(value)) {
    throw new RuntimeConfigurationError(`${label} must be an integer.`);
  }
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < minimum || parsed > maximum) {
    throw new RuntimeConfigurationError(`${label} is outside its allowed range.`);
  }
  return parsed;
}

function runtimeEndpoint(value: string | undefined) {
  if (!value) {
    throw new RuntimeConfigurationError("MOLOR_RUNTIME_URL is required in local mode.");
  }
  let endpoint: URL;
  try {
    endpoint = new URL(value);
  } catch {
    throw new RuntimeConfigurationError("MOLOR_RUNTIME_URL is invalid.");
  }
  if (endpoint.protocol !== "http:") {
    throw new RuntimeConfigurationError("The local runtime URL must use HTTP.");
  }
  if (
    endpoint.hostname !== "127.0.0.1" &&
    endpoint.hostname !== "[::1]"
  ) {
    throw new RuntimeConfigurationError(
      "The local runtime URL must use a literal loopback address.",
    );
  }
  if (!endpoint.port) {
    throw new RuntimeConfigurationError("The local runtime URL must declare an explicit port.");
  }
  if (endpoint.username || endpoint.password || endpoint.search || endpoint.hash) {
    throw new RuntimeConfigurationError(
      "The local runtime URL cannot contain credentials, a query, or a fragment.",
    );
  }
  if (endpoint.pathname !== "/v1/chat/stream") {
    throw new RuntimeConfigurationError(
      "The local runtime URL path must be /v1/chat/stream.",
    );
  }
  return endpoint.toString();
}

function runtimeToken(value: string | undefined) {
  if (!value || !/^[A-Za-z0-9._~-]{32,512}$/.test(value)) {
    throw new RuntimeConfigurationError(
      "MOLOR_RUNTIME_TOKEN must be a 32-512 character random token.",
    );
  }
  return value;
}

export function loadRuntimeConfig(
  environment: Record<string, string | undefined> = process.env,
): RuntimeConfig {
  const mode = environment.MOLOR_CHAT_BACKEND?.trim() || "simulator";
  if (mode === "simulator") return { mode };
  if (mode !== "local") {
    throw new RuntimeConfigurationError(
      "MOLOR_CHAT_BACKEND must be either simulator or local.",
    );
  }

  const connectTimeoutMs = timeout(
    environment.MOLOR_RUNTIME_CONNECT_TIMEOUT_MS,
    3_000,
    "MOLOR_RUNTIME_CONNECT_TIMEOUT_MS",
    100,
    60_000,
  );
  const firstEventTimeoutMs = timeout(
    environment.MOLOR_RUNTIME_FIRST_EVENT_TIMEOUT_MS,
    10_000,
    "MOLOR_RUNTIME_FIRST_EVENT_TIMEOUT_MS",
    100,
    120_000,
  );
  const idleTimeoutMs = timeout(
    environment.MOLOR_RUNTIME_IDLE_TIMEOUT_MS,
    120_000,
    "MOLOR_RUNTIME_IDLE_TIMEOUT_MS",
    1_000,
    30 * 60_000,
  );
  const totalTimeoutMs = timeout(
    environment.MOLOR_RUNTIME_TOTAL_TIMEOUT_MS,
    6 * 60 * 60_000,
    "MOLOR_RUNTIME_TOTAL_TIMEOUT_MS",
    10_000,
    24 * 60 * 60_000,
  );
  if (totalTimeoutMs <= firstEventTimeoutMs || totalTimeoutMs <= idleTimeoutMs) {
    throw new RuntimeConfigurationError(
      "The total runtime timeout must exceed first-event and idle timeouts.",
    );
  }

  return {
    mode,
    endpoint: runtimeEndpoint(environment.MOLOR_RUNTIME_URL),
    token: runtimeToken(environment.MOLOR_RUNTIME_TOKEN),
    connectTimeoutMs,
    firstEventTimeoutMs,
    idleTimeoutMs,
    totalTimeoutMs,
  };
}
