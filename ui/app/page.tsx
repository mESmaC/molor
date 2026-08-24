"use client";

import {
  FormEvent,
  KeyboardEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  ChatStreamEvent,
  NdjsonStreamDecoder,
  isNdjsonContentType,
} from "@/lib/chat-stream";

type MessageStatus = "complete" | "streaming" | "stopped" | "error";

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  status: MessageStatus;
};

type RuntimeStats = {
  status: "ready" | "streaming" | "stopped" | "error";
  runtimeSource: "simulator" | "local";
  modelId: string;
  backendMode: string;
  quantization: string;
  device: string;
  stackFingerprint: string;
  contextUsed: number;
  contextLimit: number;
  inputTokens: number;
  outputTokens: number;
  ttftMs: number | null;
  elapsedMs: number;
  tokensPerSecond: number;
  finishReason: string;
  routeId: string;
  routeWeight: number;
  memoryMode: "Applied" | "Abstained" | "Unavailable" | "Off";
  memoryVersion: string;
  retrievalCount: number;
  retrievalLatencyMs: number;
  consolidationState: string;
  metricsEstimated: boolean;
};

const INITIAL_MESSAGE: ChatMessage = {
  id: "welcome",
  role: "assistant",
  status: "complete",
  content:
    "The chat surface is ready. This preview uses the built-in stream simulator until the frozen base, router, and adapter loader are connected.",
};

const INITIAL_STATS: RuntimeStats = {
  status: "ready",
  runtimeSource: "simulator",
  modelId: "molor-ui-simulator",
  backendMode: "Deterministic preview",
  quantization: "Not loaded",
  device: "Worker preview",
  stackFingerprint: "Awaiting runtime",
  contextUsed: 1_842,
  contextLimit: 131_072,
  inputTokens: 0,
  outputTokens: 0,
  ttftMs: null,
  elapsedMs: 0,
  tokensPerSecond: 0,
  finishReason: "—",
  routeId: "base/null",
  routeWeight: 1,
  memoryMode: "Unavailable",
  memoryVersion: "No version pinned",
  retrievalCount: 0,
  retrievalLatencyMs: 0,
  consolidationState: "Idle",
  metricsEstimated: true,
};

const SUGGESTIONS = [
  "Explain the capability route",
  "Summarize the memory state",
  "Draft an evaluation prompt",
];

function compactNumber(value: number) {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(value);
}

function formatDuration(value: number | null) {
  if (value === null) return "—";
  if (value < 1_000) return `${Math.round(value)} ms`;
  return `${(value / 1_000).toFixed(2)} s`;
}

function shortFingerprint(value: string) {
  if (value.length < 18) return value;
  return `${value.slice(0, 8)}…${value.slice(-6)}`;
}

function makeId(prefix: string) {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return `${prefix}-${crypto.randomUUID()}`;
  }
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

async function boundedErrorMessage(response: Response) {
  if (!response.body) return `Chat request failed (${response.status})`;
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8", { fatal: true });
  let bytes = 0;
  let text = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    bytes += value.byteLength;
    if (bytes > 8 * 1024) {
      await reader.cancel();
      return `Chat request failed (${response.status})`;
    }
    text += decoder.decode(value, { stream: true });
  }
  text += decoder.decode();
  try {
    const payload = JSON.parse(text) as unknown;
    if (
      payload &&
      typeof payload === "object" &&
      !Array.isArray(payload) &&
      typeof (payload as { error?: unknown }).error === "string"
    ) {
      return (payload as { error: string }).error.slice(0, 1_024);
    }
  } catch {
    // Error responses are displayed only when they match the bounded JSON shape above.
  }
  return `Chat request failed (${response.status})`;
}

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([INITIAL_MESSAGE]);
  const [composer, setComposer] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [memoryEnabled, setMemoryEnabled] = useState(true);
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [showJump, setShowJump] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState("");
  const [compactLayout, setCompactLayout] = useState(false);
  const [liveAnnouncement, setLiveAnnouncement] = useState("");
  const [stats, setStats] = useState<RuntimeStats>(INITIAL_STATS);
  const [error, setError] = useState<string | null>(null);

  const transcriptRef = useRef<HTMLDivElement>(null);
  const composerRef = useRef<HTMLTextAreaElement>(null);
  const runtimeButtonRef = useRef<HTMLButtonElement>(null);
  const inspectorRef = useRef<HTMLElement>(null);
  const inspectorCloseRef = useRef<HTMLButtonElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const activeRequestRef = useRef<string | null>(null);
  const stickToBottomRef = useRef(true);
  const announcementBufferRef = useRef("");

  const contextPercent = useMemo(
    () => Math.min(100, (stats.contextUsed / stats.contextLimit) * 100),
    [stats.contextLimit, stats.contextUsed],
  );
  const contextTone =
    contextPercent >= 90 ? "danger" : contextPercent >= 75 ? "warning" : "normal";
  const statusLabel = {
    ready: stats.runtimeSource === "local" ? "Local runtime ready" : "Preview runtime ready",
    streaming: "Generating",
    stopped: "Generation stopped",
    error: "Runtime error",
  }[stats.status];
  const runtimeBadge = stats.runtimeSource === "local" ? "LOCAL" : "SIMULATOR";
  const throughputUnit = stats.metricsEstimated ? "chunk/s" : "tok/s";

  const scrollToLatest = useCallback((smooth = true) => {
    const transcript = transcriptRef.current;
    if (!transcript) return;
    transcript.scrollTo({
      top: transcript.scrollHeight,
      behavior: smooth ? "smooth" : "auto",
    });
    stickToBottomRef.current = true;
    setShowJump(false);
  }, []);

  useEffect(() => {
    if (stickToBottomRef.current) {
      requestAnimationFrame(() => scrollToLatest(false));
    }
  }, [messages, scrollToLatest]);

  useEffect(() => {
    const media = window.matchMedia("(max-width: 1100px)");
    const updateLayout = () => setCompactLayout(media.matches);
    updateLayout();
    media.addEventListener("change", updateLayout);
    return () => media.removeEventListener("change", updateLayout);
  }, []);

  useEffect(() => () => abortRef.current?.abort(), []);

  const stopGeneration = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    activeRequestRef.current = null;
    setIsStreaming(false);
    setMessages((current) =>
      current.map((message) =>
        message.status === "streaming" ? { ...message, status: "stopped" } : message,
      ),
    );
    setStats((current) => ({ ...current, status: "stopped", finishReason: "stopped" }));
    requestAnimationFrame(() => composerRef.current?.focus());
  }, []);

  const closeInspector = useCallback(() => {
    setInspectorOpen(false);
  }, []);

  useEffect(() => {
    if (!inspectorOpen || !compactLayout) return;
    const previousFocus = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : runtimeButtonRef.current;
    const restoreTarget = previousFocus ?? runtimeButtonRef.current;
    requestAnimationFrame(() => inspectorCloseRef.current?.focus());

    const trapFocus = (event: globalThis.KeyboardEvent) => {
      if (event.key !== "Tab" || !inspectorRef.current) return;
      const focusable = Array.from(
        inspectorRef.current.querySelectorAll<HTMLElement>(
          'button:not([disabled]), [href], input:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
        ),
      );
      if (focusable.length === 0) {
        event.preventDefault();
        inspectorRef.current.focus();
        return;
      }
      const first = focusable[0];
      const last = focusable.at(-1) ?? first;
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", trapFocus);
    return () => {
      document.removeEventListener("keydown", trapFocus);
      requestAnimationFrame(() => restoreTarget?.focus());
    };
  }, [compactLayout, inspectorOpen]);

  useEffect(() => {
    const handleEscape = (event: globalThis.KeyboardEvent) => {
      if (event.key !== "Escape") return;
      if (isStreaming) {
        stopGeneration();
      } else if (inspectorOpen) {
        closeInspector();
      }
    };
    window.addEventListener("keydown", handleEscape);
    return () => window.removeEventListener("keydown", handleEscape);
  }, [closeInspector, inspectorOpen, isStreaming, stopGeneration]);

  const handleStreamEvent = useCallback((event: ChatStreamEvent, assistantId: string) => {
    if (event.type === "error") {
      if (
        activeRequestRef.current === null ||
        activeRequestRef.current === event.request_id
      ) {
        throw new Error(event.message);
      }
      return;
    }
    if (event.type === "generation.started") {
      activeRequestRef.current = event.request_id;
      announcementBufferRef.current = "";
      setStats((current) => ({
        ...current,
        status: "streaming",
        runtimeSource: event.runtime_source,
        modelId: event.model_id,
        backendMode: event.backend_mode,
        quantization: event.quantization,
        device: event.device,
        stackFingerprint: event.stack_fingerprint,
        contextUsed: event.context_used,
        contextLimit: event.context_limit,
        inputTokens: event.input_tokens,
        outputTokens: 0,
        ttftMs: null,
        elapsedMs: 0,
        tokensPerSecond: 0,
        finishReason: "generating",
        routeId: "pending",
        routeWeight: 0,
        memoryMode: "Unavailable",
        memoryVersion: "No version pinned",
        retrievalCount: 0,
        retrievalLatencyMs: 0,
        consolidationState: "Idle",
        metricsEstimated: event.metrics_estimated,
      }));
      return;
    }
    if (activeRequestRef.current !== event.request_id) return;

    if (event.type === "route") {
      const route = event.capability[0] ?? { id: "base/null", weight: 1 };
      const memoryMode =
        event.memory.mode === "applied"
          ? "Applied"
          : event.memory.mode === "unavailable"
            ? "Unavailable"
          : event.memory.mode === "off"
            ? "Off"
            : "Abstained";
      setStats((current) => ({
        ...current,
        routeId: route.id,
        routeWeight: route.weight,
        memoryMode,
        memoryVersion: event.memory.version ?? "No version pinned",
      }));
      return;
    }
    if (event.type === "retrieval") {
      setStats((current) => ({
        ...current,
        retrievalCount: event.count,
        retrievalLatencyMs: event.latency_ms,
        consolidationState: event.consolidation_state,
      }));
      return;
    }
    if (event.type === "token") {
      announcementBufferRef.current += event.text;
      const sentences = announcementBufferRef.current.match(/[^.!?]+[.!?](?:\s|$)/g);
      if (sentences?.length) {
        setLiveAnnouncement(sentences.at(-1)?.trim() ?? "");
        const lastSentence = sentences.at(-1) ?? "";
        const lastIndex = announcementBufferRef.current.lastIndexOf(lastSentence);
        announcementBufferRef.current = announcementBufferRef.current.slice(
          lastIndex + lastSentence.length,
        );
      }
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId
            ? { ...message, content: `${message.content}${event.text}` }
            : message,
        ),
      );
      if (event.ttft_ms !== undefined) {
        setStats((current) => ({ ...current, ttftMs: event.ttft_ms ?? current.ttftMs }));
      }
      return;
    }
    if (event.type === "metrics") {
      setStats((current) => ({
        ...current,
        outputTokens: event.completion_tokens,
        contextUsed: event.context_used,
        elapsedMs: event.elapsed_ms,
        tokensPerSecond: event.tokens_per_second,
      }));
      return;
    }
    if (event.type === "generation.completed") {
      if (announcementBufferRef.current.trim()) {
        setLiveAnnouncement(announcementBufferRef.current.trim());
        announcementBufferRef.current = "";
      }
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId ? { ...message, status: "complete" } : message,
        ),
      );
      setStats((current) => ({
        ...current,
        status: "ready",
        outputTokens: event.completion_tokens,
        contextUsed: event.context_used,
        elapsedMs: event.elapsed_ms,
        tokensPerSecond: event.tokens_per_second,
        finishReason: event.finish_reason,
      }));
      return;
    }
  }, []);

  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || isStreaming) return;
      const conversationId = sessionId || makeId("conversation");
      if (!sessionId) setSessionId(conversationId);

      const userMessage: ChatMessage = {
        id: makeId("user"),
        role: "user",
        content: trimmed,
        status: "complete",
      };
      const assistantMessage: ChatMessage = {
        id: makeId("assistant"),
        role: "assistant",
        content: "",
        status: "streaming",
      };
      const requestMessages = [...messages, userMessage]
        .filter((message) => message.id !== "welcome")
        .map(({ role, content }) => ({ role, content }));

      setMessages((current) => [...current, userMessage, assistantMessage]);
      setComposer("");
      setError(null);
      setIsStreaming(true);
      setStats((current) => ({
        ...current,
        status: "streaming",
        inputTokens: 0,
        outputTokens: 0,
        ttftMs: null,
        elapsedMs: 0,
        tokensPerSecond: 0,
        finishReason: "starting",
        routeId: "pending",
        routeWeight: 0,
        memoryMode: memoryEnabled ? "Unavailable" : "Off",
        memoryVersion: "No version pinned",
        retrievalCount: 0,
        retrievalLatencyMs: 0,
        consolidationState: "Idle",
      }));
      stickToBottomRef.current = true;

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            messages: requestMessages,
            memory: memoryEnabled ? "auto" : "off",
            conversation_id: conversationId,
          }),
          signal: controller.signal,
        });
        if (!response.ok || !response.body) {
          throw new Error(await boundedErrorMessage(response));
        }
        if (
          !isNdjsonContentType(response.headers.get("content-type")) ||
          response.headers.get("x-molor-protocol") !== "molor.chat-stream.v1"
        ) {
          throw new Error("The runtime returned an invalid stream protocol.");
        }

        const reader = response.body.getReader();
        const decoder = new NdjsonStreamDecoder();
        const processRecords = (records: ReturnType<typeof decoder.push>) => {
          for (const event of records) {
            if (event.type === "heartbeat") {
              throw new Error("The browser received a bridge-internal heartbeat.");
            }
            handleStreamEvent(event, assistantMessage.id);
          }
        };
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          processRecords(decoder.push(value));
          if (decoder.terminalReceived) {
            await reader.cancel().catch(() => undefined);
            break;
          }
        }
        processRecords(decoder.finish());
      } catch (streamError) {
        if (streamError instanceof DOMException && streamError.name === "AbortError") return;
        controller.abort();
        const message =
          streamError instanceof Error ? streamError.message : "The generation stream failed.";
        setError(message);
        setStats((current) => ({ ...current, status: "error", finishReason: "error" }));
        setMessages((current) =>
          current.map((item) =>
            item.id === assistantMessage.id
              ? {
                  ...item,
                  content: item.content || "The runtime stream disconnected before responding.",
                  status: "error",
                }
              : item,
          ),
        );
      } finally {
        if (abortRef.current === controller) {
          abortRef.current = null;
          activeRequestRef.current = null;
          setIsStreaming(false);
          requestAnimationFrame(() => composerRef.current?.focus());
        }
      }
    },
    [handleStreamEvent, isStreaming, memoryEnabled, messages, sessionId],
  );

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    void sendMessage(composer);
  };

  const handleComposerKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendMessage(composer);
    }
  };

  const handleTranscriptScroll = () => {
    const transcript = transcriptRef.current;
    if (!transcript) return;
    const distance = transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight;
    stickToBottomRef.current = distance < 80;
    setShowJump(distance >= 80);
  };

  const handleComposerInput = (value: string) => {
    setComposer(value);
    const textarea = composerRef.current;
    if (!textarea) return;
    textarea.style.height = "0px";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 176)}px`;
  };

  const startNewSession = () => {
    if (isStreaming) stopGeneration();
    setMessages([INITIAL_MESSAGE]);
    setSessionId(makeId("conversation"));
    setStats(INITIAL_STATS);
    setError(null);
    setComposer("");
    requestAnimationFrame(() => composerRef.current?.focus());
  };

  const copyMessage = async (message: ChatMessage) => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopiedId(message.id);
      window.setTimeout(() => setCopiedId(null), 1_500);
    } catch {
      setError("Could not copy this response to the clipboard.");
    }
  };

  return (
    <div className="app-shell">
      <div className="sr-only" aria-live="polite" aria-atomic="true">
        {liveAnnouncement}
      </div>
      <header className="topbar" inert={compactLayout && inspectorOpen}>
        <div className="brand-lockup" aria-label="molor chat">
          <span className="brand-mark" aria-hidden="true">M</span>
          <span className="brand-copy">
            <strong>molor</strong>
            <span>local intelligence</span>
          </span>
        </div>

        <div className="topbar-status" aria-live="polite">
          <span className={`status-dot ${stats.status}`} aria-hidden="true" />
          <span>{statusLabel}</span>
          <span className={`preview-pill ${stats.runtimeSource}`}>{runtimeBadge}</span>
        </div>

        <div className="topbar-actions">
          <div className="compact-stat" title="Active context usage">
            <span>Context</span>
            <strong>{contextPercent.toFixed(1)}%</strong>
          </div>
          <div className="compact-stat speed-stat" title="Generation throughput">
            <span>Speed</span>
            <strong>
              {stats.tokensPerSecond
                ? `${stats.tokensPerSecond.toFixed(1)} ${stats.metricsEstimated ? "c/s" : "t/s"}`
                : "—"}
            </strong>
          </div>
          <button className="quiet-button" type="button" onClick={startNewSession}>
            <span aria-hidden="true">＋</span> New chat
          </button>
          <button
            ref={runtimeButtonRef}
            className={`quiet-button ${inspectorOpen ? "active" : ""}`}
            type="button"
            aria-expanded={inspectorOpen}
            aria-controls="runtime-inspector"
            onClick={() => (inspectorOpen ? closeInspector() : setInspectorOpen(true))}
          >
            Runtime
          </button>
        </div>
      </header>

      <div className={`workspace ${inspectorOpen ? "inspector-open" : ""}`}>
        <main className="chat-column" inert={compactLayout && inspectorOpen}>
          <div
            className="transcript"
            ref={transcriptRef}
            onScroll={handleTranscriptScroll}
            aria-label="Conversation"
          >
            <div className="transcript-inner">
              <div className="session-rule"><span>New local session</span></div>

              {messages.map((message) => (
                <article className={`message ${message.role} ${message.status}`} key={message.id}>
                  <div className="message-identity" aria-hidden="true">
                    {message.role === "assistant" ? "M" : "YOU"}
                  </div>
                  <div className="message-body">
                    <div className="message-meta">
                      <strong>{message.role === "assistant" ? "molor" : "You"}</strong>
                      {message.status === "stopped" && <span>Stopped</span>}
                      {message.status === "error" && <span className="error-label">Error</span>}
                    </div>
                    <div className="message-content">
                      {message.content}
                      {message.status === "streaming" && (
                        <span className="stream-cursor" aria-label="Generating" />
                      )}
                    </div>
                    {message.status !== "streaming" && message.content && (
                      <button
                        className="message-action"
                        type="button"
                        onClick={() => void copyMessage(message)}
                        aria-label={`Copy ${message.role} message`}
                      >
                        {copiedId === message.id ? "Copied" : "Copy"}
                      </button>
                    )}
                  </div>
                </article>
              ))}

              {messages.length === 1 && (
                <div className="suggestion-row" aria-label="Example prompts">
                  {SUGGESTIONS.map((suggestion) => (
                    <button
                      type="button"
                      key={suggestion}
                      onClick={() => {
                        setComposer(suggestion);
                        requestAnimationFrame(() => composerRef.current?.focus());
                      }}
                    >
                      <span>{suggestion}</span><span aria-hidden="true">↗</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          {showJump && (
            <button className="jump-button" type="button" onClick={() => scrollToLatest()}>
              Jump to latest <span aria-hidden="true">↓</span>
            </button>
          )}

          <div className="composer-zone">
            {error && (
              <div className="error-banner" role="alert">
                <span>{error}</span>
                <button type="button" onClick={() => setError(null)} aria-label="Dismiss error">×</button>
              </div>
            )}
            <form className="composer" onSubmit={handleSubmit}>
              <label className="sr-only" htmlFor="chat-composer">Message molor</label>
              <textarea
                id="chat-composer"
                ref={composerRef}
                value={composer}
                rows={1}
                maxLength={12_000}
                placeholder="Ask molor anything…"
                onChange={(event) => handleComposerInput(event.target.value)}
                onKeyDown={handleComposerKeyDown}
                disabled={isStreaming}
              />
              <div className="composer-footer">
                <div className="composer-options">
                  <button
                    type="button"
                    className={`memory-toggle ${memoryEnabled ? "enabled" : ""}`}
                    aria-pressed={memoryEnabled}
                    onClick={() => setMemoryEnabled((enabled) => !enabled)}
                    disabled={isStreaming}
                  >
                    <span className="memory-glyph" aria-hidden="true">◎</span>
                    Memory {memoryEnabled ? "auto" : "off"}
                  </button>
                  <span className="keyboard-hint">Shift + Enter for new line</span>
                </div>
                {isStreaming ? (
                  <button className="send-button stop" type="button" onClick={stopGeneration}>
                    <span className="stop-square" aria-hidden="true" /> Stop
                  </button>
                ) : (
                  <button
                    className="send-button"
                    type="submit"
                    disabled={!composer.trim()}
                  >
                    Send <span aria-hidden="true">↑</span>
                  </button>
                )}
              </div>
            </form>
            <div className="context-strip">
              <div className={`context-track ${contextTone}`}>
                <span style={{ width: `${contextPercent}%` }} />
              </div>
              <span>
                {compactNumber(stats.contextUsed)} / {compactNumber(stats.contextLimit)}
                {stats.metricsEstimated ? " estimated tokens" : " tokens"}
              </span>
              <span>{stats.consolidationState}</span>
            </div>
          </div>
        </main>

        {inspectorOpen && (
          <button
            className="inspector-scrim"
            type="button"
            aria-label="Close runtime inspector"
            onClick={closeInspector}
          />
        )}

        <aside
          ref={inspectorRef}
          className={`runtime-inspector ${inspectorOpen ? "open" : ""}`}
          id="runtime-inspector"
          aria-label="Runtime statistics"
          aria-hidden={!inspectorOpen}
          aria-modal={compactLayout && inspectorOpen ? true : undefined}
          role={compactLayout ? "dialog" : undefined}
          inert={!inspectorOpen}
          tabIndex={-1}
        >
          <div className="inspector-header">
            <div><span className="eyebrow">Live telemetry</span><h2>Runtime</h2></div>
            <button
              ref={inspectorCloseRef}
              type="button"
              onClick={closeInspector}
              aria-label="Close runtime"
            >
              ×
            </button>
          </div>

          <section className="inspector-section">
            <div className="section-heading">
              <h3>Model stack</h3>
              <span className={`live-badge ${stats.runtimeSource}`}>
                {stats.runtimeSource === "local" ? "local" : "preview"}
              </span>
            </div>
            <dl className="data-list">
              <div><dt>Checkpoint</dt><dd>{stats.modelId}</dd></div>
              <div><dt>Quantization</dt><dd>{stats.quantization}</dd></div>
              <div><dt>Device</dt><dd>{stats.device}</dd></div>
              <div><dt>Stack</dt><dd className="mono" title={stats.stackFingerprint}>{shortFingerprint(stats.stackFingerprint)}</dd></div>
            </dl>
          </section>

          <section className="inspector-section context-section">
            <div className="section-heading">
              <h3>Context</h3>
              <strong className={`context-percent ${contextTone}`}>{contextPercent.toFixed(1)}%</strong>
            </div>
            <div className={`large-context-track ${contextTone}`}><span style={{ width: `${contextPercent}%` }} /></div>
            <div className="token-summary">
              <div>
                <span>{stats.metricsEstimated ? "Active (est.)" : "Active"}</span>
                <strong>{compactNumber(stats.contextUsed)}</strong>
              </div>
              <div>
                <span>{stats.metricsEstimated ? "Limit (sim.)" : "Limit"}</span>
                <strong>{compactNumber(stats.contextLimit)}</strong>
              </div>
              <div>
                <span>{stats.metricsEstimated ? "Chunks" : "Generated"}</span>
                <strong>{compactNumber(stats.outputTokens)}</strong>
              </div>
            </div>
          </section>

          <section className="inspector-section">
            <div className="section-heading"><h3>Generation</h3><span className={`generation-state ${stats.status}`}>{stats.status}</span></div>
            <div className="metric-grid">
              <div>
                <span>Speed</span>
                <strong>
                  {stats.tokensPerSecond ? stats.tokensPerSecond.toFixed(1) : "—"}
                  <small> {throughputUnit}</small>
                </strong>
              </div>
              <div>
                <span>{stats.metricsEstimated ? "First chunk" : "First token"}</span>
                <strong>{formatDuration(stats.ttftMs)}</strong>
              </div>
              <div><span>Elapsed</span><strong>{formatDuration(stats.elapsedMs)}</strong></div>
              <div><span>Finish</span><strong>{stats.finishReason}</strong></div>
            </div>
          </section>

          <section className="inspector-section">
            <div className="section-heading"><h3>Capability route</h3><span className="route-weight">{stats.routeWeight.toFixed(2)}</span></div>
            <div className="route-card">
              <span className="route-orb" aria-hidden="true" />
              <div><strong>{stats.routeId}</strong><span>Top route selected for this prompt</span></div>
            </div>
          </section>

          <section className="inspector-section memory-section">
            <div className="section-heading">
              <h3>Persistent memory</h3>
              <span className={`memory-state ${stats.memoryMode.toLowerCase()}`}>{stats.memoryMode}</span>
            </div>
            <dl className="data-list">
              <div><dt>Version</dt><dd>{stats.memoryVersion}</dd></div>
              <div><dt>Retrieval</dt><dd>{stats.retrievalCount} items · {stats.retrievalLatencyMs} ms</dd></div>
              <div><dt>Consolidation</dt><dd>{stats.consolidationState}</dd></div>
            </dl>
          </section>

          <footer className="inspector-footer">
            <span>Session</span>
            <code title={sessionId}>{sessionId ? shortFingerprint(sessionId) : "Initializing…"}</code>
            <p>{stats.backendMode}. Values update from the generation event stream.</p>
          </footer>
        </aside>
      </div>
    </div>
  );
}
