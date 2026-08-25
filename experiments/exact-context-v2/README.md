# Exact-context V2 development scaffold

This directory describes a development-only follow-up to the sealed and rejected
exact-context V1 experiment. V1 remains final and is neither edited nor reinterpreted.
V2 has no frozen contract, evaluator, formal report, or acceptance claim yet.

The worker in `molor.experimental.exact_context_v2_lab` changes the cache under test to
Transformers' exact `StaticCache` with:

- `max_cache_len=40_960`;
- `offloading=True`;
- `offload_only_non_sliding=False`;
- lazy initialization, so the full backing is never eagerly allocated on the GPU.

The model still executes on CUDA. Static K/V is preallocated and unquantized; this is not
compression, sparse attention, ATXY, or position extension. Generic Transformers offload
still transfers each fixed-size layer backing between CPU and GPU, so StaticCache may reduce
size-changing fragmentation without necessarily satisfying the 6 GiB V1 RSS threshold.

## Isolated worker contract

One worker process may execute exactly one length, placement, and repeat. Three trials are
therefore three sequential OS processes with three exclusive reports. Print the exact
development commands without loading a model:

```powershell
.\.venv\Scripts\python.exe -m molor.experimental.exact_context_v2_lab `
  --emit-command-contract
```

Do not combine those trials into one Python process: process isolation is the diagnostic that
prevents one trial's CPU allocator high-water mark from becoming the next trial's baseline.
The emitted command set is deliberately marked `development_unfrozen`.

## Added telemetry

Every synchronized memory observation records unique live `keys`/`values` backing storage,
deduplicated by device, data pointer, and storage size. CPU, CUDA, other-device, and total bytes
are separate because the offload prefetch stream can leave one cache layer resident on CUDA.

On Windows the worker uses `PROCESS_MEMORY_COUNTERS_EX` and records both working-set RSS and
`PrivateUsage` (private committed bytes). After decoding, the worker drops the only strong cache
reference, runs Python collection without trimming the working set, clears only this process's
CUDA allocator cache, and records `after_cache_release` while the model remains loaded. The cache
weak reference remains bound through that observation: the report explicitly records whether the
object was collected, and a surviving reference is a release failure with its live storage still
visible. Failed decode tracebacks are handled conservatively rather than being reported as a
successful release.

Existing output paths are rejected before model loading. Local model bytes are rechecked during
fail-safe finalization even when loading or decoding fails; a postflight mismatch is always failure
evidence and cannot be hidden by the earlier error.

## Compatibility gates required before preregistration

StaticCache returns its full preallocated K/V length, including unused future slots. Qwen's mask
construction is expected to hide those slots, but forced cuDNN SDPA eligibility and exact greedy
token parity against DynamicCache must be demonstrated in a small diagnostic before freezing V2.
Do not call `StaticCache.early_initialization`: it would allocate the complete 4.375 GiB backing
on CUDA before offload. If isolated StaticCache still exceeds the RSS requirement, the next fix is
a CPU-resident preallocated backing with reusable GPU staging buffers, not allocator trimming or a
larger acceptance threshold.
