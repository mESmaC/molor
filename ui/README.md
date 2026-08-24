# molor chat UI

A dark, single-purpose chat applet for the molor runtime. The default `/api/chat` route remains an
explicitly labelled in-process simulator. It exercises the UI boundary: validated NDJSON token
events, cancellation, context accounting, route selection, memory state, and generation metrics.

The Python package now contains verified frozen-base loading, a planning coordinator that freezes
the capability bank/router/memory identities, tokenizer-bound execution, an authenticated loopback
server, and a deterministic no-model backend. The applet's fail-closed bridge can run that complete
local control path now; loading LoRA tensors and generating with a real checkpoint remain backend
work.

## Local development

```powershell
npm install
npm run dev
```

`npm run dev` is the fast development path for the default simulator. The Sites/vinext development
worker is isolated from workstation loopback on some hosts, so it is not the supported path for an
attached Python runtime.

Copy `.env.example` to `.env.local` and select `MOLOR_CHAT_BACKEND=local` only when an authenticated
runtime is listening on the exact loopback `/v1/chat/stream` endpoint. Local mode requires a random
`MOLOR_RUNTIME_TOKEN` of at least 32 URL-safe characters. The bridge never forwards browser cookies,
ChatGPT identity headers, or arbitrary headers. Runtime URLs must use the literal `127.0.0.1` or
`[::1]` address; hostnames and remote addresses are rejected so the bearer secret is never sent
through name resolution.
If an explicitly selected local runtime is unavailable or violates the protocol, the request fails;
it is never silently replayed through the simulator.

With the runtime running, start the local UI through vinext's Node production server:

```powershell
npm run local
```

This builds the app, loads `.env.local` server-side, and serves it at `http://127.0.0.1:3000` by
default. Set `PORT` in the process environment or `.env.local` to choose another port. The script
always supplies the explicit loopback hostname because `vinext start` otherwise binds to all
interfaces. Keep `MOLOR_RUNTIME_TOKEN` unprefixed: never use `NEXT_PUBLIC_` or `VITE_`, and never put
the token in the command line. Stop the UI with Ctrl-C.

The private hosted preview intentionally stays in simulator mode. A hosted worker cannot directly
reach the model process on the workstation's loopback interface; that requires a separately
authorized private tunnel in a later deployment step.

## Generation event boundary

The implemented loopback endpoint and any later real-model backend use these ordered event types:

- `generation.started`: request identity, `runtime_source`, model/runtime identity, and input context;
- `route`: capability route weights plus independently gated memory state;
- `retrieval`: retrieval count, latency, and consolidation state;
- `token`: one real tokenizer/generator chunk and its monotonically increasing index;
- `metrics`: throttled completion count, elapsed time, and measured throughput;
- `generation.completed` or `error`.

The local runtime must echo the bridge-issued request ID, return
`x-molor-protocol: molor.chat-stream.v1`, and emit the ordered event sequence above. The bridge
enforces exact schemas, bounded event and stream sizes, contiguous token indexes, monotonic metrics,
one terminal event, fatal UTF-8 decoding, cancellation, and connect/first-event/idle/total timeouts.
Bridge-only heartbeat events may keep long model-load or prefill work alive but are stripped before
the browser receives them.

The browser cancels an active generation by aborting the streaming request. A production local
server should bind to `127.0.0.1`, keep the user's memory scope server-side, cap request sizes, and
stop generation when its client disconnects. The browser's `conversation_id` is correlation data,
not authorization: the serving process must derive the durable user scope itself and bind each
request to that scope before retrieval or memory pinning. Serving pins are request-scoped so later
turns can observe a newer retrieval cut without hot-swapping an in-flight generation.
