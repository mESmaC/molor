import assert from "node:assert/strict";
import { createServer } from "node:http";
import test from "node:test";

const workerUrl = new URL("../dist/server/index.js", import.meta.url);
workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
const { default: worker } = await import(workerUrl.href);

function dispatch(url, init = undefined) {
  return worker.fetch(
    new Request(url, init),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

function request(path = "/", init = undefined) {
  return dispatch(`http://localhost${path}`, init);
}

const LOCAL_RUNTIME_TOKEN = "local-runtime-token-0123456789abcdef";
const LOCAL_RUNTIME_ENVIRONMENT = [
  "MOLOR_CHAT_BACKEND",
  "MOLOR_RUNTIME_URL",
  "MOLOR_RUNTIME_TOKEN",
  "MOLOR_RUNTIME_CONNECT_TIMEOUT_MS",
  "MOLOR_RUNTIME_FIRST_EVENT_TIMEOUT_MS",
  "MOLOR_RUNTIME_IDLE_TIMEOUT_MS",
  "MOLOR_RUNTIME_TOTAL_TIMEOUT_MS",
];

function configureLocalRuntime(endpoint) {
  process.env.MOLOR_CHAT_BACKEND = "local";
  process.env.MOLOR_RUNTIME_URL = endpoint;
  process.env.MOLOR_RUNTIME_TOKEN = LOCAL_RUNTIME_TOKEN;
  process.env.MOLOR_RUNTIME_CONNECT_TIMEOUT_MS = "500";
  process.env.MOLOR_RUNTIME_FIRST_EVENT_TIMEOUT_MS = "500";
  process.env.MOLOR_RUNTIME_IDLE_TIMEOUT_MS = "1000";
  process.env.MOLOR_RUNTIME_TOTAL_TIMEOUT_MS = "10000";
}

function clearLocalRuntimeEnvironment() {
  for (const key of LOCAL_RUNTIME_ENVIRONMENT) delete process.env[key];
}

async function startRuntime(listener) {
  const runtime = createServer(listener);
  await new Promise((resolve) => runtime.listen(0, "127.0.0.1", resolve));
  const address = runtime.address();
  assert.ok(address && typeof address === "object");
  return { runtime, port: address.port };
}

async function stopRuntime(runtime) {
  if (!runtime.listening) return;
  const closed = new Promise((resolve, reject) =>
    runtime.close((error) => (error ? reject(error) : resolve())),
  );
  runtime.closeAllConnections();
  await closed;
}

async function readRuntimePayload(incoming) {
  const chunks = [];
  for await (const chunk of incoming) chunks.push(chunk);
  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

function completeRuntimeEvents(requestId, runtimeSource = "local") {
  return [
    {
      type: "generation.started",
      request_id: requestId,
      runtime_source: runtimeSource,
      model_id: "molor-test-runtime",
      backend_mode: "Prepared generation test",
      quantization: "4-bit NF4",
      device: "cuda:0",
      stack_fingerprint: "a".repeat(64),
      context_used: 10,
      context_limit: 1024,
      input_tokens: 10,
      metrics_estimated: false,
    },
    {
      type: "route",
      request_id: requestId,
      capability: [{ id: "base/null", weight: 1 }],
      memory: { mode: "off", version: null },
    },
    {
      type: "retrieval",
      request_id: requestId,
      count: 0,
      latency_ms: 0,
      consolidation_state: "Idle",
    },
    {
      type: "token",
      request_id: requestId,
      index: 0,
      text: "Local response.",
      ttft_ms: 4,
    },
    {
      type: "metrics",
      request_id: requestId,
      completion_tokens: 2,
      context_used: 12,
      elapsed_ms: 8,
      tokens_per_second: 250,
    },
    {
      type: "generation.completed",
      request_id: requestId,
      completion_tokens: 2,
      context_used: 12,
      elapsed_ms: 8,
      tokens_per_second: 250,
      finish_reason: "stop",
    },
  ];
}

function postLocalChat() {
  return request("/api/chat", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      messages: [{ role: "user", content: "hello locally" }],
      memory: "off",
      conversation_id: "local-test-session",
    }),
  });
}

async function within(promise, milliseconds = 3000) {
  let timer;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error("Timed out waiting for bridge cleanup.")), milliseconds);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

test("server-renders the molor chat surface", async () => {
  const response = await request();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>molor · Local intelligence<\/title>/i);
  assert.match(html, /Ask molor anything/);
  assert.match(html, /Runtime/);
  assert.match(html, /Persistent memory/);
  assert.match(html, /SIMULATOR/);
  assert.doesNotMatch(html, /codex-preview|SkeletonPreview|Your site is taking shape/i);
});

test("chat endpoint emits an ordered NDJSON generation stream", async () => {
  const response = await request("/api/chat", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      messages: [{ role: "user", content: "Explain the memory state" }],
      memory: "auto",
      conversation_id: "test-session",
    }),
  });

  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^application\/x-ndjson\b/i);
  const events = (await response.text())
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line));

  assert.equal(events[0].type, "generation.started");
  assert.equal(events[1].type, "route");
  assert.deepEqual(events[1].memory, { mode: "unavailable", version: null });
  assert.equal(events[2].type, "retrieval");
  assert.ok(events.some((event) => event.type === "token"));
  assert.equal(events.at(-1).type, "generation.completed");
  assert.equal(events.at(-1).finish_reason, "stop");
  assert.ok(events.at(-1).context_used > events[0].context_used);
});

test("chat endpoint rejects malformed payloads", async () => {
  const response = await request("/api/chat", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ messages: [] }),
  });
  assert.equal(response.status, 400);

  const whitespaceOnly = await request("/api/chat", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      messages: [{ role: "user", content: "   \t\r\n" }],
      memory: "off",
      conversation_id: "test-session",
    }),
  });
  assert.equal(whitespaceOnly.status, 400);
});

test("chat endpoint rejects extra fields and invalid conversation identities", async () => {
  const extraField = await request("/api/chat", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      messages: [{ role: "user", content: "hello" }],
      memory: "off",
      conversation_id: "test-session",
      ignored: "not accepted",
    }),
  });
  assert.equal(extraField.status, 400);

  const emptyIdentity = await request("/api/chat", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      messages: [{ role: "user", content: "hello" }],
      memory: "off",
      conversation_id: "",
    }),
  });
  assert.equal(emptyIdentity.status, 400);
});

test("chat endpoint rejects the raw body before parsing oversized extra data", async () => {
  const response = await request("/api/chat", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      messages: [{ role: "user", content: "hello" }],
      memory: "off",
      conversation_id: "test-session",
      ignored: "x".repeat(70 * 1024),
    }),
  });
  assert.equal(response.status, 413);
});

test("non-loopback chat requests require the trusted hosting identity", async () => {
  const response = await dispatch("https://molor.example/api/chat", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      messages: [{ role: "user", content: "hello" }],
      memory: "off",
      conversation_id: "test-session",
    }),
  });
  assert.equal(response.status, 401);
});

test("explicit local mode proxies only the bounded chat contract", async () => {
  let captured = null;
  const runtime = createServer(async (incoming, outgoing) => {
    const chunks = [];
    for await (const chunk of incoming) chunks.push(chunk);
    const payload = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    captured = { headers: incoming.headers, payload };
    const requestId = payload.request_id;
    const events = [
      {
        type: "generation.started",
        request_id: requestId,
        runtime_source: "local",
        model_id: "molor-test-runtime",
        backend_mode: "Prepared generation test",
        quantization: "4-bit NF4",
        device: "cuda:0",
        stack_fingerprint: "a".repeat(64),
        context_used: 10,
        context_limit: 1024,
        input_tokens: 10,
        metrics_estimated: false,
      },
      {
        type: "route",
        request_id: requestId,
        capability: [{ id: "base/null", weight: 1 }],
        memory: { mode: "off", version: null },
      },
      {
        type: "retrieval",
        request_id: requestId,
        count: 0,
        latency_ms: 0,
        consolidation_state: "Idle",
      },
      {
        type: "token",
        request_id: requestId,
        index: 0,
        text: "Local response.",
        ttft_ms: 4,
      },
      {
        type: "metrics",
        request_id: requestId,
        completion_tokens: 2,
        context_used: 12,
        elapsed_ms: 8,
        tokens_per_second: 250,
      },
      {
        type: "generation.completed",
        request_id: requestId,
        completion_tokens: 2,
        context_used: 12,
        elapsed_ms: 8,
        tokens_per_second: 250,
        finish_reason: "stop",
      },
    ];
    outgoing.writeHead(200, {
      "content-type": "application/x-ndjson; charset=utf-8",
      "x-molor-protocol": "molor.chat-stream.v1",
    });
    const encoded = `${events.map((event) => JSON.stringify(event)).join("\n")}\n`;
    outgoing.write(encoded.slice(0, 37));
    outgoing.end(encoded.slice(37));
  });
  await new Promise((resolve) => runtime.listen(0, "127.0.0.1", resolve));
  const address = runtime.address();
  assert.ok(address && typeof address === "object");
  const secret = "local-runtime-token-0123456789abcdef";
  process.env.MOLOR_CHAT_BACKEND = "local";
  process.env.MOLOR_RUNTIME_URL = `http://127.0.0.1:${address.port}/v1/chat/stream`;
  process.env.MOLOR_RUNTIME_TOKEN = secret;

  try {
    const response = await request("/api/chat", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        cookie: "browser-secret=must-not-forward",
        "x-untrusted-browser-header": "must-not-forward",
      },
      body: JSON.stringify({
        messages: [{ role: "user", content: "hello locally" }],
        memory: "off",
        conversation_id: "local-test-session",
      }),
    });
    assert.equal(response.status, 200);
    assert.equal(response.headers.get("x-molor-protocol"), "molor.chat-stream.v1");
    const text = await response.text();
    const events = text.trim().split("\n").map((line) => JSON.parse(line));
    assert.equal(events[0].runtime_source, "local");
    assert.equal(events.at(-1).type, "generation.completed");
    assert.doesNotMatch(text, new RegExp(secret));
  } finally {
    delete process.env.MOLOR_CHAT_BACKEND;
    delete process.env.MOLOR_RUNTIME_URL;
    delete process.env.MOLOR_RUNTIME_TOKEN;
    await new Promise((resolve, reject) =>
      runtime.close((error) => (error ? reject(error) : resolve())),
    );
  }

  assert.ok(captured);
  assert.equal(captured.headers.authorization, `Bearer ${secret}`);
  assert.equal(captured.headers["x-molor-protocol"], "molor.chat-stream.v1");
  assert.equal(captured.headers.cookie, undefined);
  assert.equal(captured.headers["x-untrusted-browser-header"], undefined);
  assert.deepEqual(Object.keys(captured.payload).sort(), [
    "conversation_id",
    "memory",
    "messages",
    "request_id",
    "schema_version",
  ]);
  assert.equal(captured.payload.schema_version, "molor.chat-request.v1");
  assert.match(captured.payload.request_id, /^runtime-[A-Za-z0-9-]+$/);
});

test("local mode rejects a non-loopback runtime before dispatch", async () => {
  process.env.MOLOR_CHAT_BACKEND = "local";
  process.env.MOLOR_RUNTIME_URL = "http://example.com:8765/v1/chat/stream";
  process.env.MOLOR_RUNTIME_TOKEN = "local-runtime-token-0123456789abcdef";
  try {
    const response = await request("/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        messages: [{ role: "user", content: "hello" }],
        memory: "off",
        conversation_id: "local-test-session",
      }),
    });
    assert.equal(response.status, 500);
    assert.deepEqual(await response.json(), {
      error: "Local runtime configuration is invalid.",
    });
  } finally {
    delete process.env.MOLOR_CHAT_BACKEND;
    delete process.env.MOLOR_RUNTIME_URL;
    delete process.env.MOLOR_RUNTIME_TOKEN;
  }
});

test("local mode rejects hostname resolution in place of a literal loopback address", async () => {
  configureLocalRuntime("http://localhost:8765/v1/chat/stream");
  try {
    const response = await postLocalChat();
    assert.equal(response.status, 500);
    assert.deepEqual(await response.json(), {
      error: "Local runtime configuration is invalid.",
    });
  } finally {
    clearLocalRuntimeEnvironment();
  }
});

test("local bridge accepts only the exact NDJSON media type", async () => {
  const { runtime, port } = await startRuntime(async (incoming, outgoing) => {
    const payload = await readRuntimePayload(incoming);
    outgoing.writeHead(200, {
      "content-type": "application/x-ndjson-evil; charset=utf-8",
      "x-molor-protocol": "molor.chat-stream.v1",
    });
    outgoing.end(
      `${completeRuntimeEvents(payload.request_id)
        .map((event) => JSON.stringify(event))
        .join("\n")}\n`,
    );
  });
  configureLocalRuntime(`http://127.0.0.1:${port}/v1/chat/stream`);

  try {
    const response = await postLocalChat();
    assert.equal(response.status, 502);
    assert.deepEqual(await response.json(), {
      error: "The local runtime returned an invalid stream protocol.",
    });
  } finally {
    clearLocalRuntimeEnvironment();
    await stopRuntime(runtime);
  }
});

test("local bridge rejects a simulator source from the configured local runtime", async () => {
  const { runtime, port } = await startRuntime(async (incoming, outgoing) => {
    const payload = await readRuntimePayload(incoming);
    const started = completeRuntimeEvents(payload.request_id, "simulator")[0];
    outgoing.writeHead(200, {
      "content-type": "application/x-ndjson; charset=utf-8",
      "x-molor-protocol": "molor.chat-stream.v1",
    });
    outgoing.end(`${JSON.stringify(started)}\n`);
  });
  configureLocalRuntime(`http://127.0.0.1:${port}/v1/chat/stream`);

  try {
    const response = await postLocalChat();
    assert.equal(response.status, 502);
    assert.deepEqual(await response.json(), {
      error: "The local runtime connection failed.",
    });
  } finally {
    clearLocalRuntimeEnvironment();
    await stopRuntime(runtime);
  }
});

test("local bridge sanitizes runtime-provided terminal errors", async () => {
  const sensitiveMessage =
    "Internal failure at C:\\private\\models\\weights with runtime-secret-value";
  const { runtime, port } = await startRuntime(async (incoming, outgoing) => {
    const payload = await readRuntimePayload(incoming);
    outgoing.writeHead(200, {
      "content-type": "application/x-ndjson; charset=utf-8",
      "x-molor-protocol": "molor.chat-stream.v1",
    });
    outgoing.end(
      `${JSON.stringify({
        type: "error",
        request_id: payload.request_id,
        message: sensitiveMessage,
      })}\n`,
    );
  });
  configureLocalRuntime(`http://127.0.0.1:${port}/v1/chat/stream`);

  try {
    const response = await postLocalChat();
    assert.equal(response.status, 200);
    const text = await response.text();
    assert.doesNotMatch(text, /private|runtime-secret-value/i);
    assert.deepEqual(
      text
        .trim()
        .split("\n")
        .map((line) => JSON.parse(line)),
      [
        {
          type: "error",
          request_id: JSON.parse(text).request_id,
          message: "The local runtime reported a generation error.",
        },
      ],
    );
  } finally {
    clearLocalRuntimeEnvironment();
    await stopRuntime(runtime);
  }
});

test("local bridge closes and cancels upstream as soon as a terminal event arrives", async () => {
  let resolveUpstreamClosed;
  const upstreamClosed = new Promise((resolve) => {
    resolveUpstreamClosed = resolve;
  });
  const { runtime, port } = await startRuntime(async (incoming, outgoing) => {
    const payload = await readRuntimePayload(incoming);
    outgoing.on("close", resolveUpstreamClosed);
    outgoing.writeHead(200, {
      "content-type": "application/x-ndjson; charset=utf-8",
      "x-molor-protocol": "molor.chat-stream.v1",
    });
    outgoing.write(
      `${completeRuntimeEvents(payload.request_id)
        .map((event) => JSON.stringify(event))
        .join("\n")}\n`,
    );
  });
  configureLocalRuntime(`http://127.0.0.1:${port}/v1/chat/stream`);

  try {
    const response = await postLocalChat();
    assert.equal(response.status, 200);
    const text = await within(response.text());
    assert.equal(JSON.parse(text.trim().split("\n").at(-1)).type, "generation.completed");
    await within(upstreamClosed);
  } finally {
    clearLocalRuntimeEnvironment();
    await stopRuntime(runtime);
  }
});
