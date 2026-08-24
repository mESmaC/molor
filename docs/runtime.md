# Local serving boundary v0

Status: immutable request preparation, verified base-artifact staging, an offline-safe Transformers
base loader, tokenizer-aware execution planning, a deterministic executor, and the authenticated UI
bridge are implemented. Loading capability/memory adapter tensors and generating with a real
Transformers checkpoint are the next backend milestones.

## Purpose

The serving boundary keeps the frozen base, capability router, LoRA expert bank, retrieval view,
and optional memory LoRA from drifting independently during a response. Every request is prepared
before model execution and the resulting identities remain fixed through prefill and decode.

## Preparation order

1. Authenticate the local runtime token and derive `trusted_scope_id` on the server. Scope is never
   accepted from browser JSON.
2. Parse the bounded chat transcript and create a server-issued `GenerationRequest.request_id`.
   `conversation_id` remains correlation and durable turn-order data only.
3. Validate the exact `RouterBindingManifest` against the ordered capability bank and select a
   capability route.
4. For memory `auto`, create a request-scoped `SessionMemoryPin`, retrieve through that fixed cut,
   and render whole source-labelled records under byte and exact tokenizer-counted token caps. Raw
   store hits do not enter the execution handoff or either routing policy.
5. Resolve the pinned memory version through a typed registry boundary and revalidate its scope,
   base fingerprint, and content fingerprint. Expose a valid version as a separate
   one-adapter-plus-null bank and run the hard memory gate. The current control gate always
   abstains. A missing, substituted, or otherwise unverifiable version leaves the effective stack
   on the frozen base while preserving the attempted version and retrieval-cut identities.
6. Return a fingerprinted `PreparedGeneration`. The execution planner verifies the tokenizer and
   model-stack identities, reduces retrieval and history only at whole-record/whole-turn
   boundaries, reserves output space, and creates one immutable `ExecutionPlan` without rerouting
   or repinning during generation.

Memory `off` performs no store pin or retrieval. Retrieval continues when no memory adapter exists
or the pinned version cannot be validated; the memory gate then reports `unavailable`. Retrieved
content remains source-labelled, bounded, untrusted data and must not be inserted into system or
router instructions. The gate receives the original request and trusted version metadata, never
retrieved content.

## Local HTTP bridge

The applet defaults to the deterministic simulator. Explicit local mode accepts exactly one
operator-configured loopback endpoint:

```text
POST http://127.0.0.1:<port>/v1/chat/stream
Authorization: Bearer <runtime token>
Content-Type: application/json
Accept: application/x-ndjson
X-Molor-Protocol: molor.chat-stream.v1
```

Request body:

```json
{
  "schema_version": "molor.chat-request.v1",
  "request_id": "runtime-<server UUID>",
  "conversation_id": "conversation-...",
  "messages": [{"role": "user", "content": "..."}],
  "memory": "auto"
}
```

The bridge forwards no browser cookies, identity headers, arbitrary headers, or scope fields. The
runtime must echo the bridge-issued request ID and return
`X-Molor-Protocol: molor.chat-stream.v1`. The bridge validates and re-encodes every event; it never
raw-pipes a runtime response.

The public event order is:

```text
generation.started
route
retrieval
(token | metrics)*
generation.completed | error
```

Token indexes are contiguous, metrics are monotonic, and exactly one terminal event is required.
`generation.started` includes `runtime_source: "local" | "simulator"`. A local runtime should send
it immediately after accepting a request. During long loading or prefill it may send validated
bridge-only `heartbeat` records, which reset the idle timeout and are removed before browser output.

Explicit local mode is fail-closed. A connection, authentication, timeout, or protocol failure is
reported as an error and is never replayed through the simulator after the runtime may have accepted
the prompt. Browser cancellation propagates to the upstream request.

## Running the deterministic local stack

The smoke runtime executes the real coordinator, prompt renderer, execution planner, orchestrator,
cancellation path, and HTTP protocol. It intentionally loads no model or adapter weights, keeps no
durable memory, and returns fixed test data. Generate a fresh bearer secret and start it from the
workspace root:

```powershell
$env:MOLOR_RUNTIME_TOKEN = .\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
.\.venv\Scripts\python.exe -m molor.runtime.smoke
```

The equivalent installed console command is `molor-smoke-runtime`. Configuration is accepted only
through the environment: `MOLOR_RUNTIME_HOST` defaults to `127.0.0.1`, `MOLOR_RUNTIME_PORT` to
`8765`, and `MOLOR_RUNTIME_SCOPE_ID` to `user/local-smoke`. The bearer token is never accepted as a
command-line argument, where it could be exposed through process inspection.

In `ui/.env.local`, select the bridge and use the exact same secret:

```dotenv
MOLOR_CHAT_BACKEND=local
MOLOR_RUNTIME_URL=http://127.0.0.1:8765/v1/chat/stream
MOLOR_RUNTIME_TOKEN=<same generated token>
```

Then run `npm run local` inside `ui/`. That command builds the app and starts vinext's supported
Node production server explicitly on `127.0.0.1`, allowing its server-side bridge to reach the
Python loopback endpoint without exposing the bearer token to the browser. `npm run dev` remains the
fast simulator path because its worker runtime can be isolated from workstation loopback. A
successful response is prominently labelled `DETERMINISTIC SMOKE - NO MODEL WEIGHTS`; it proves
the local protocol path, not model inference or persistent-memory behavior.

## Verified base-artifact boundary

`BaseModelManifest` v2 declares `config.json`, tokenizer/special-token files, safetensors weights,
and an explicit safetensors index whenever more than one weight shard is present. Persisted
manifests are parsed with exact schemas and types: unknown or duplicate keys, invalid UTF-8,
non-finite numbers, oversized documents, and unknown schema versions are rejected.

`LocalArtifactMaterializer` streams only declared files into a fresh private staging directory and
checks every byte count and SHA-256 digest. The staged tree does not inherit undeclared files from
the source directory. The Transformers loader then uses only local files, safetensors, and
`trust_remote_code=False`; it checks architecture, tokenizer/chat-template compatibility, context
limits, embedding rows, loader diagnostics, and that every base parameter is frozen.

Loader v0 deliberately accepts only `quantization="none"` with `float16`, `bfloat16`, or
`float32`. A 4-bit/8-bit or NVMe-offloaded 37B deployment needs an explicit quantization and device
placement contract plus a generation backend whose LoRA application path is compatible with the
loaded linear-module type. It must not be smuggled through this unquantized boundary.

## Execution invariants

The implemented planner/orchestrator:

- uses the runtime tokenizer for prompt and exact output accounting;
- supplies that tokenizer's immutable identity and exact counter to the bounded retrieval renderer;
- binds the prepared capability and memory routes as independent axes for the full decode;
- keeps one prepared serving pin for the complete request;
- serializes model execution so mutable adapter state cannot cross requests;
- stops decoding when the downstream client disconnects;
- emits exact token indexes, context use, time-to-first-token, throughput, and one terminal event;
- sanitizes backend failures at the public boundary; and
- rejects a backend, tokenizer, context window, model, quantization, or model-stack identity that
  differs from the execution plan.

The remaining real-model backend must:

- verify router, capability-adapter, and memory-adapter artifacts before loading their tensors;
- enforce the bounded routing prefix and reuse the selected route throughout decode;
- apply capability and memory LoRA deltas on their independent axes;
- append admitted turns using `SQLiteMemoryStore.append_observation_auto` rather than racing on a
  caller-computed sequence, after a separate trusted admission/provenance decision;
- compact context only after a rollover snapshot and consolidation job are durably committed; and
- leave memory-LoRA training and promotion disabled until scope-global sequencing and invalidation
  epochs fence corrections and deletions.

SQLite connections currently retain their creating thread's affinity. A threaded serving process
must create one `SQLiteMemoryStore` per worker/request or serialize access through one owner thread.
Request IDs must be server-issued, globally unique, and never reused for different request content;
the current pin store does not bind a reused ID to the original request fingerprint.
