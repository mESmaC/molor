# Exact CPU/GPU context lab v1

This lab compares two exact Hugging Face `DynamicCache` paths with the same base model, synthetic
token IDs, and manual greedy decode loop:

- `dynamic` keeps the growing KV cache GPU-resident;
- `offloaded` uses Transformers' supported CPU-offloaded cache, moving layer KV to the GPU when
  that layer executes.

Using both answers a systems question that CPU-only or GPU-only measurements cannot: how much VRAM
does exact CPU offload save, what ordered latency tradeoff is observed, and does it preserve the
exact generated token sequence? The recorded delta is not a causal transfer-only measurement: the
runner performs no excluded warm-up and executes modes in declared CLI order. This is not a new
cache representation. The model still computes on the GPU and the offloaded KV remains exact,
uncompressed state in CPU memory.

"Exact cache" describes the KV representation and placement: both modes retain the same
model-produced, uncompressed K/V values, and CPU offload does not introduce an approximate cache.
It does **not** require deterministic floating-point kernel arithmetic. cuDNN may use
nondeterministic reduction/order choices even though decoding is greedy. The lab therefore keeps
exact completion-token parity across modes and repeats as a mandatory observed correctness check,
while recording the arithmetic policy separately.

The runner uses explicit `DynamicCache(config=..., offloading=False/True)` instances rather than
`generate()`. It performs one exact-length prefill, selects tokens with `argmax`, and then feeds one
token at a time with explicit attention masks and cache positions. Only the last-token logits are
materialized, avoiding a prompt-length-by-vocabulary logits allocation. EOS does not end a run
early, so every successful record contains exactly `--max-new-tokens` completion tokens.

## Safety and evidence contract

Every requested prompt length plus completion length must fit the loaded config's
`max_position_embeddings`; the runner has no override. A successful result therefore measures only
the checkpoint's existing position contract and does not establish useful longer-context quality.

Each run records full local checkpoint hashes before and after execution (or a pinned remote
commit), config and tokenizer identities, library/platform/device versions, prompt and completion
token identities, model-step-scoped prefill/TTFT and per-token timing, process RSS, CUDA allocator
peaks, and device-global free-memory observations.
When both modes complete, token IDs must match exactly across modes and repeats. A mismatch or any
runtime error marks the report failed and exits nonzero after writing the available evidence.

Reports are content-sealed JSON files created with exclusive, non-overwriting file semantics. The
lab writes no KV snapshot and makes no NVMe, disk-backed decoding, sparse-cache, compression, or
context-extension claim. CUDA device-global observations include other processes and are labelled
separately from this process's PyTorch allocator statistics.

## Run deliberately

No benchmark result is checked into this directory. Before running, confirm the selected GPU has
room alongside other active processes. The defaults refuse to load the model below 5 GiB global
free memory and retain an additional 1 GiB beyond the nominal dynamic KV payload.

Local diagnostics found that strict deterministic algorithms made automatic SDPA fall back to the
math backend; the 4,096-token offloaded case then produced non-finite logits in both BF16 and FP16.
Eager attention passed that small diagnostic but is not a scalable long-context path. Forced cuDNN
SDPA with deterministic algorithms disabled passed three offloaded repeats and three matched
offloaded-to-dynamic greedy-token comparisons. That establishes the safe local systems-test policy,
not a longer-context or universal numerical-stability claim.

```powershell
.\.venv\Scripts\molor-exact-context-lab.exe `
  --modes offloaded dynamic `
  --attn-implementation sdpa `
  --sdpa-backend cudnn `
  --no-deterministic-algorithms `
  --lengths 512 2048 4096 `
  --max-new-tokens 16 `
  --repeats 3
```

The default policy matches that cuDNN/nondeterministic selection. `--sdpa-backend` also accepts
`auto`, `math`, `efficient`, and `flash`. Forced backends are applied with
`torch.nn.attention.sdpa_kernel` around every model forward. The report records the selected
PyTorch enum, backend enablement/build information available without allocating model inputs, and
notes that input-shape eligibility is proven only by a successful forced forward. The runner
rejects eager attention with a forced SDPA backend and rejects strict deterministic algorithms with
forced cuDNN before loading model weights.

To benchmark a single cache path, pass only that mode. `offloaded` requires a CUDA model device.
Remote checkpoints require both `--allow-download` and a full 40-hex `--revision` commit identity.
