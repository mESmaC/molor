import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createServer } from "node:http";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { startProdServer } from "vinext/server/prod-server";

const UI_ROOT = new URL("../", import.meta.url);
const OUT_DIR = fileURLToPath(new URL("../dist/", import.meta.url));
const RUNTIME_TOKEN = "local-runtime-token-0123456789abcdef";
const RUNTIME_ENVIRONMENT = [
  "MOLOR_CHAT_BACKEND",
  "MOLOR_RUNTIME_URL",
  "MOLOR_RUNTIME_TOKEN",
  "MOLOR_RUNTIME_CONNECT_TIMEOUT_MS",
  "MOLOR_RUNTIME_FIRST_EVENT_TIMEOUT_MS",
  "MOLOR_RUNTIME_IDLE_TIMEOUT_MS",
  "MOLOR_RUNTIME_TOTAL_TIMEOUT_MS",
];

function preserveRuntimeEnvironment() {
  const original = new Map(RUNTIME_ENVIRONMENT.map((name) => [name, process.env[name]]));
  return () => {
    for (const [name, value] of original) {
      if (value === undefined) delete process.env[name];
      else process.env[name] = value;
    }
  };
}

function configureLocalRuntime(port) {
  process.env.MOLOR_CHAT_BACKEND = "local";
  process.env.MOLOR_RUNTIME_URL = `http://127.0.0.1:${port}/v1/chat/stream`;
  process.env.MOLOR_RUNTIME_TOKEN = RUNTIME_TOKEN;
  process.env.MOLOR_RUNTIME_CONNECT_TIMEOUT_MS = "500";
  process.env.MOLOR_RUNTIME_FIRST_EVENT_TIMEOUT_MS = "1000";
  process.env.MOLOR_RUNTIME_IDLE_TIMEOUT_MS = "2000";
  process.env.MOLOR_RUNTIME_TOTAL_TIMEOUT_MS = "10000";
}

async function listen(server) {
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      server.off("error", reject);
      resolve();
    });
  });
  const address = server.address();
  assert.ok(address && typeof address === "object");
  assert.equal(address.address, "127.0.0.1");
  return address.port;
}

async function closeServer(server, force = false) {
  if (!server.listening) return;
  const closed = new Promise((resolve, reject) => {
    server.close((error) => (error ? reject(error) : resolve()));
  });
  if (force) server.closeAllConnections();
  await within(closed);
}

async function within(promise, milliseconds = 3000) {
  let timer;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error("Timed out waiting for cleanup.")), milliseconds);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

async function readJsonRequest(incoming) {
  const chunks = [];
  for await (const chunk of incoming) chunks.push(chunk);
  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

function runtimeEvents(requestId) {
  return [
    {
      type: "generation.started",
      request_id: requestId,
      runtime_source: "local",
      model_id: "molor-node-e2e",
      backend_mode: "Deterministic production-server test",
      quantization: "none",
      device: "cpu",
      stack_fingerprint: "a".repeat(64),
      context_used: 4,
      context_limit: 1024,
      input_tokens: 4,
      metrics_estimated: false,
    },
    { type: "heartbeat", request_id: requestId, phase: "decoding" },
    {
      type: "route",
      request_id: requestId,
      capability: [{ id: "base/null", weight: 1 }],
      memory: { mode: "unavailable", version: null },
    },
    {
      type: "retrieval",
      request_id: requestId,
      count: 0,
      latency_ms: 0,
      consolidation_state: "Idle",
    },
    { type: "token", request_id: requestId, index: 0, text: "Node-local response.", ttft_ms: 3 },
    {
      type: "metrics",
      request_id: requestId,
      completion_tokens: 3,
      context_used: 7,
      elapsed_ms: 6,
      tokens_per_second: 500,
    },
    {
      type: "generation.completed",
      request_id: requestId,
      completion_tokens: 3,
      context_used: 7,
      elapsed_ms: 6,
      tokens_per_second: 500,
      finish_reason: "stop",
    },
  ];
}

function chatRequest(baseUrl, content = "hello through the Node production server") {
  return fetch(`${baseUrl}/api/chat`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      cookie: "browser-secret=must-not-forward",
      "x-untrusted-browser-header": "must-not-forward",
    },
    body: JSON.stringify({
      messages: [{ role: "user", content }],
      memory: "auto",
      conversation_id: "node-local-e2e",
    }),
  });
}

test("local scripts use the supported vinext server and always bind loopback", async () => {
  const packageJson = JSON.parse(await readFile(new URL("package.json", UI_ROOT), "utf8"));
  assert.equal(packageJson.scripts["local:serve"], "vinext start --hostname 127.0.0.1");
  assert.equal(packageJson.scripts.local, "npm run build && npm run local:serve");
  assert.doesNotMatch(
    `${packageJson.scripts["local:serve"]} ${packageJson.scripts.local}`,
    /MOLOR_RUNTIME_TOKEN/,
  );
});

test("vinext production server reaches loopback runtime and streams the bounded contract", async () => {
  const restoreEnvironment = preserveRuntimeEnvironment();
  let captured;
  const runtime = createServer(async (incoming, outgoing) => {
    const payload = await readJsonRequest(incoming);
    captured = { headers: incoming.headers, method: incoming.method, payload, url: incoming.url };
    const encoded = `${runtimeEvents(payload.request_id)
      .map((event) => JSON.stringify(event))
      .join("\n")}\n`;
    outgoing.writeHead(200, {
      "content-type": "application/x-ndjson; charset=utf-8",
      "x-molor-protocol": "molor.chat-stream.v1",
    });
    outgoing.write(encoded.slice(0, 73));
    setImmediate(() => outgoing.end(encoded.slice(73)));
  });
  let appServer;

  try {
    const runtimePort = await listen(runtime);
    configureLocalRuntime(runtimePort);
    const started = await startProdServer({
      host: "127.0.0.1",
      port: 0,
      outDir: OUT_DIR,
      noCompression: true,
      silent: true,
    });
    appServer = started.server;
    const address = appServer.address();
    assert.ok(address && typeof address === "object");
    assert.equal(address.address, "127.0.0.1");
    const baseUrl = `http://127.0.0.1:${started.port}`;

    const pageResponse = await fetch(baseUrl);
    assert.equal(pageResponse.status, 200);
    const html = await pageResponse.text();
    assert.doesNotMatch(html, new RegExp(RUNTIME_TOKEN));
    const assetPath = html.match(/(?:src|href)="([^"]*\/_next\/static\/[^"]+)"/)?.[1];
    assert.ok(assetPath, "built HTML should reference a static asset");
    assert.equal((await fetch(new URL(assetPath, baseUrl))).status, 200);

    const response = await chatRequest(baseUrl);
    assert.equal(response.status, 200);
    assert.equal(response.headers.get("x-molor-protocol"), "molor.chat-stream.v1");
    const responseText = await response.text();
    const events = responseText
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line));
    assert.equal(events[0].runtime_source, "local");
    assert.equal(events.at(-1).type, "generation.completed");
    assert.ok(events.some((event) => event.type === "token"));
    assert.ok(events.every((event) => event.type !== "heartbeat"));
    assert.doesNotMatch(responseText, new RegExp(RUNTIME_TOKEN));

    assert.ok(captured);
    assert.equal(captured.method, "POST");
    assert.equal(captured.url, "/v1/chat/stream");
    assert.equal(captured.headers.authorization, `Bearer ${RUNTIME_TOKEN}`);
    assert.equal(captured.headers.cookie, undefined);
    assert.equal(captured.headers["x-untrusted-browser-header"], undefined);
    assert.equal(captured.payload.schema_version, "molor.chat-request.v1");

    const oversized = await chatRequest(baseUrl, "x".repeat(70 * 1024));
    assert.equal(oversized.status, 413);
    assert.deepEqual(await oversized.json(), { error: "Chat request exceeds 64 KiB." });

    const crossOrigin = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "content-type": "application/json", origin: "https://attacker.example" },
      body: JSON.stringify({
        messages: [{ role: "user", content: "hello" }],
        memory: "off",
        conversation_id: "node-local-e2e",
      }),
    });
    assert.equal(crossOrigin.status, 403);
  } finally {
    restoreEnvironment();
    if (appServer) await closeServer(appServer);
    await closeServer(runtime, true);
  }
});

test("disconnecting from vinext production streaming cancels the loopback runtime", async () => {
  const restoreEnvironment = preserveRuntimeEnvironment();
  let resolveUpstreamClosed;
  const upstreamClosed = new Promise((resolve) => {
    resolveUpstreamClosed = resolve;
  });
  let heartbeat;
  const runtime = createServer(async (incoming, outgoing) => {
    const payload = await readJsonRequest(incoming);
    outgoing.on("close", () => {
      clearInterval(heartbeat);
      resolveUpstreamClosed();
    });
    const [started] = runtimeEvents(payload.request_id);
    outgoing.writeHead(200, {
      "content-type": "application/x-ndjson; charset=utf-8",
      "x-molor-protocol": "molor.chat-stream.v1",
    });
    outgoing.write(`${JSON.stringify(started)}\n`);
    heartbeat = setInterval(() => {
      outgoing.write(
        `${JSON.stringify({
          type: "heartbeat",
          request_id: payload.request_id,
          phase: "decoding",
        })}\n`,
      );
    }, 25);
  });
  let appServer;

  try {
    const runtimePort = await listen(runtime);
    configureLocalRuntime(runtimePort);
    const started = await startProdServer({
      host: "127.0.0.1",
      port: 0,
      outDir: OUT_DIR,
      noCompression: true,
      silent: true,
    });
    appServer = started.server;
    const controller = new AbortController();
    const response = await fetch(`http://127.0.0.1:${started.port}/api/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        messages: [{ role: "user", content: "keep generating" }],
        memory: "off",
        conversation_id: "node-local-cancel",
      }),
      signal: controller.signal,
    });
    assert.equal(response.status, 200);
    const reader = response.body.getReader();
    const first = await within(reader.read());
    assert.equal(first.done, false);
    controller.abort();
    await reader.cancel().catch(() => undefined);
    await within(upstreamClosed);
  } finally {
    clearInterval(heartbeat);
    restoreEnvironment();
    if (appServer) await closeServer(appServer, true);
    await closeServer(runtime, true);
  }
});
