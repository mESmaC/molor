export type IncomingMessage = {
  role: "user" | "assistant";
  content: string;
};

export type ChatRequest = {
  messages: IncomingMessage[];
  memory: "auto" | "off";
  conversation_id: string;
};

const CONTEXT_LIMIT = 131_072;
const BASELINE_CONTEXT = 1_842;

function sleep(milliseconds: number) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function estimateTokens(value: string) {
  return Math.max(1, Math.ceil(value.length / 4));
}

function previewResponse(prompt: string) {
  const normalized = prompt.toLowerCase();
  if (normalized.includes("memory")) {
    return [
      "The memory plane is currently abstaining because no trained memory adapter is loaded.",
      "The runtime contract now pins one immutable memory view per generation, retrieves scoped ledger events independently, and keeps the memory adapter on a separate hard-gated axis.",
      "The event ledger remains authoritative; a memory LoRA is a rebuildable, lossy compilation of stable history.",
    ].join("\n\n");
  }
  if (normalized.includes("route") || normalized.includes("expert")) {
    return [
      "This preview is taking the explicit base/null route with weight 1.00.",
      "The prepared-generation coordinator now binds the exact router checkpoint, ordered capability bank, retrieval cut, and memory decision before an executor starts.",
      "Once model weights are connected, the runtime inspector will display those measured route IDs and weights through this same stream.",
    ].join("\n\n");
  }
  if (normalized.includes("evaluation") || normalized.includes("eval")) {
    return [
      "A useful first evaluation prompt should separate expert quality from router quality.",
      "Run the prompt once against every forced capability route, record correctness and cost, then compare the learned route with the oracle winner and frozen-base null route.",
      "For memory behavior, hold the capability route fixed so retrieval and adapter-gating rewards cannot hide routing regressions.",
    ].join("\n\n");
  }
  const excerpt = prompt.length > 110 ? `${prompt.slice(0, 107)}…` : prompt;
  return [
    `I received: “${excerpt}”`,
    "Token streaming, cancellation, context accounting, and the runtime panel are working through the applet's validated event boundary.",
    "The current response comes from a transparent simulator. The local bridge now fails closed when explicitly enabled, ready for a verified model executor.",
  ].join("\n\n");
}

export function createSimulatorResponse(body: ChatRequest) {
  const requestId = `simulator-${crypto.randomUUID()}`;
  const prompt = body.messages.at(-1)?.content ?? "";
  const responseText = previewResponse(prompt);
  const pieces = responseText.match(/\S+\s*/g) ?? [responseText];
  const inputTokens = body.messages.reduce(
    (sum, message) => sum + estimateTokens(message.content),
    0,
  );
  const contextUsed = BASELINE_CONTEXT + inputTokens;
  const encoder = new TextEncoder();
  let cancelled = false;
  const emitLine = (value: Record<string, unknown>) =>
    encoder.encode(`${JSON.stringify(value)}\n`);

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const startedAt = Date.now();
      const emit = (event: Record<string, unknown>) => {
        if (!cancelled) controller.enqueue(emitLine(event));
      };

      emit({
        type: "generation.started",
        request_id: requestId,
        runtime_source: "simulator",
        model_id: "molor-ui-simulator",
        backend_mode: "Deterministic preview",
        quantization: "Not loaded",
        device: "Worker preview",
        stack_fingerprint: "awaiting-runtime-manifest",
        context_used: contextUsed,
        context_limit: CONTEXT_LIMIT,
        input_tokens: inputTokens,
        metrics_estimated: true,
      });
      emit({
        type: "route",
        request_id: requestId,
        capability: [{ id: "base/null", weight: 1 }],
        memory: {
          mode: body.memory === "off" ? "off" : "unavailable",
          version: null,
        },
      });
      emit({
        type: "retrieval",
        request_id: requestId,
        count: 0,
        latency_ms: 0,
        consolidation_state: "Coordinator ready · worker not connected",
      });

      await sleep(180);
      const ttftMs = Date.now() - startedAt;
      let emittedText = "";
      for (let index = 0; index < pieces.length; index += 1) {
        if (cancelled) return;
        emittedText += pieces[index];
        emit({
          type: "token",
          request_id: requestId,
          index,
          text: pieces[index],
          ...(index === 0 ? { ttft_ms: ttftMs } : {}),
        });
        const elapsedMs = Math.max(1, Date.now() - startedAt);
        if ((index + 1) % 4 === 0 || index === pieces.length - 1) {
          emit({
            type: "metrics",
            request_id: requestId,
            completion_tokens: index + 1,
            context_used: contextUsed + estimateTokens(emittedText),
            elapsed_ms: elapsedMs,
            tokens_per_second: ((index + 1) * 1_000) / elapsedMs,
          });
        }
        await sleep(34 + (index % 3) * 7);
      }

      if (cancelled) return;
      const elapsedMs = Math.max(1, Date.now() - startedAt);
      emit({
        type: "generation.completed",
        request_id: requestId,
        completion_tokens: pieces.length,
        context_used: contextUsed + estimateTokens(emittedText),
        elapsed_ms: elapsedMs,
        tokens_per_second: (pieces.length * 1_000) / elapsedMs,
        finish_reason: "stop",
      });
      controller.close();
    },
    cancel() {
      cancelled = true;
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
