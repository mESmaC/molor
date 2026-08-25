# Exact-context v3 shared-staging compatibility lab

This is a development-only harness for the exact `SharedStagingCache` prototype. Its first frozen
real-checkpoint cell completed on 2026-08-25. The cache passed exact 4K compatibility, but the result
does **not** authorize a 40K follow-up or a new context-length claim because process RSS still
exceeded the inherited capacity gate.

The lab asks one narrow question: does an exact cache with permanently CPU-owned K/V backing and
one reusable CUDA staging pair produce the same eight greedy token IDs as a GPU-resident
Transformers `DynamicCache` on the inherited 4,096-token synthetic prompt?

## First frozen result and stop decision

The one planned run completed on its first attempt from committed harness `60fb32a`, with canonical
implementation bundle `9b71fdaa07f7bb3dff1d8e6efe2785da593def904e3d8df6dfa5c23cda2d05e8`.
Both workers and the aggregate completed with empty error lists; all three integrity seals verify;
the frozen prompt, model, config, tokenizer, request, eight completion token IDs, and independently
recomputed completion hashes match exactly.

| Measured 4K + 8 result | DynamicCache | Shared staging | Difference |
| --- | ---: | ---: | ---: |
| TTFT | 0.795308 s | 2.949804 s | 3.709x |
| Decode rate | 3.75906 tok/s | 3.40827 tok/s | -9.33% |
| Maximum run-phase CUDA allocated | 3,738.996 MiB | 3,450.230 MiB | -288.766 MiB |
| Post-reset run-phase CUDA peak | 3,948 MiB | 3,500 MiB | -448 MiB |
| Incremental process RSS after model load | 0.777374 GiB | 7.795147 GiB | +7.017773 GiB |
| Incremental private commit after model load | 0.820324 GiB | 7.830288 GiB | +7.009964 GiB |

The shared cache owned exactly 4.375 GiB of CPU backing and one 160-MiB CUDA staging pair. It was
healthy and unpoisoned, every layer ended at 4,103 cached tokens, and the cache object and owned
storage were released successfully. Seven decode steps transferred 3,291,545,600 bytes of history
host-to-device, averaging 448.438 MiB per step; including the tiny decode append, mean total traffic
was 448.547 MiB per step. At native capacity the exact history transfer approaches 4.375 GiB per
generated token.

The inherited v1 capacity policy allows at most 6 GiB of incremental process RSS. Shared staging
used 7.795147 GiB, exceeding that ceiling by 1.795147 GiB. That is about a 49.4% reduction from the
old 15.403934-GiB failure, but it is still a gate failure. The 4K decode rate is numerically above
2 tok/s, but that threshold applies at the frozen 40,952-token cell and cannot be claimed here.
Therefore the 40K run is deliberately withheld while pinned-backing overhead is redesigned.

Sealed evidence:

- `runs/exact-context-v3/compatibility-4096-plus-8.dynamic.worker.json` — file SHA-256
  `deb58b0ff380096b13c1d78569297305ef80fe3a0c5653915a68913c65b4c6d9`;
- `runs/exact-context-v3/compatibility-4096-plus-8.shared_staging.worker.json` — file SHA-256
  `9b935388f2cb309fb1e2f9cd4c354994ed92c851f48314e22898eea32c903e3b`;
- `runs/exact-context-v3/compatibility-4096-plus-8.json` — file SHA-256
  `b6e636afe8e8375167ef6288685ea0e58cd184a7af5ed7cd695be3b1f530e470`.

## Isolation and evidence

`lab.py` is a supervisor. It launches the baseline and staged paths in separate, sequential OS
processes. The staged worker starts only after the baseline succeeds. Each child loads the frozen
local Qwen3-1.7B checkpoint independently, runs one cache mode, atomically publishes one complete
content-sealed JSON file without replacing any existing path, and exits. The supervisor then
verifies both seals, independently recomputes completion hashes, compares the complete worker
requests and prompt/model/config/tokenizer identities, and atomically publishes a sealed aggregate
report with the same no-replacement rule.

Before starting either worker, the supervisor hashes this lab, `shared_staging_cache.py`, the v1 and
v2 helper modules imported by the lab, and the inherited v1 contract into one canonical provenance
bundle. The exact snapshot and its bundle hash are passed in each child argv. Each worker checks the
snapshot before inference and again after model/cache cleanup. The supervisor requires both worker
snapshots to equal its pin and re-hashes its own five files immediately before sealing the aggregate.
Any one-time or mid-run source drift fails closed.

The workers are fixed to:

- local files only, with `trust_remote_code=False` and no download or Hub-ID option;
- batch 1, manual greedy decode, exactly eight completion tokens by default;
- SDPA with the cuDNN backend forced around every forward;
- exactly Transformers 5.15.1, the cache API version targeted by this prototype;
- nondeterministic PyTorch algorithms, matching the numerically stable v1/v2 local policy;
- the checkpoint's native 40,960-position limit, without RoPE or config overrides;
- one explicit CUDA device and no shell-based child launch.

The v1 model, config, tokenizer, and default 4,096-token prompt fingerprints are asserted before
inference. Existing aggregate or derived worker paths are refused before any child starts. A
timeout, if used, is scoped to the child created by this supervisor; the lab contains no code to
find, signal, or stop unrelated processes.

Every worker records TTFT, single-token decode timing and decode tokens/second; Windows RSS and
private committed bytes; PyTorch CUDA allocated/reserved/peak bytes; CUDA device-global free memory;
best-effort independent `nvidia-smi` global memory; and the post-release memory observation. These
scopes remain separate in the report because process memory, allocator accounting, and global GPU
free memory cannot be added meaningfully.

The staged worker additionally records the cache's exact owned-storage and bounded transfer ledger.
It fails unless the cache is unpoisoned, its update cycle is complete, all 28 layers have the exact
prefill/decode counts and final length, CPU backing is exactly one full native cache, device staging
is exactly one layer K/V pair, and the staging pair was allocated once. The ledger capacity is fixed
at 256 events: total copies/bytes, retained/dropped counts, the contiguous retained ordinal tail,
and phase/direction summaries are all reconstructed independently from the frozen geometry.

Parity also fails closed on malformed evidence. Each completion must contain exactly eight plain
integer token IDs by default; both reported hashes are recomputed, prompt hashes must agree between
the prompt and run records, and final cache length must be `prompt + completion - 1`. Matching but
fabricated hash strings therefore cannot pass.

## Projected native-capacity memory

For 28 layers, 8 KV heads, head dimension 128, BF16, and 40,960 positions:

| Exact state | Bytes | Binary units |
| --- | ---: | ---: |
| CPU-owned K/V backing | 4,697,620,480 | 4.375 GiB |
| One shared K/V staging pair | 167,772,160 | 160 MiB |

This bounds live CUDA staging allocation, but it does **not** remove the main bandwidth risk. Near
40,960 positions, a single generated token still requires about 4.375 GiB of host-to-device
history traffic across the 28 layers. The 4K compatibility cell therefore tests integration,
parity, and allocator behavior; it cannot establish acceptable full-context decode speed.

Pinned CPU backing is enabled so non-blocking copies can be ordered on the current CUDA stream. The
shared worker checks host capacity before model load and checks it again after model load,
immediately before the first forward can allocate backing. Both checks require detected available
host memory to cover the 4.375-GiB backing plus the configured 2-GiB reserve. Pinned allocation can
still fail on a specific OS/driver combination, and such a failure is evidence rather than
permission to relax the memory contract. The loaded model's resolved attention implementation is
also asserted to be `sdpa` before that forward.

## Inspect without running a model

```powershell
.\.venv\Scripts\python.exe experiments\exact-context-v3\lab.py --help

.\.venv\Scripts\python.exe experiments\exact-context-v3\lab.py `
  --emit-command-contract
```

The command contract prints both child argv arrays and performs no model load or CUDA operation.

## Deliberate compatibility run

First confirm the selected GPU and host have sufficient room alongside other active workloads.
Then choose a new output path; the aggregate and both derived worker paths must all be absent:

```powershell
.\.venv\Scripts\python.exe experiments\exact-context-v3\lab.py `
  --model models\Qwen3-1.7B `
  --length 4096 `
  --max-new-tokens 8 `
  --max-cache-len 40960 `
  --device cuda:0 `
  --dtype bfloat16 `
  --output runs\exact-context-v3\compatibility-4096-plus-8.json
```

That command may create:

- `compatibility-4096-plus-8.dynamic.worker.json`;
- `compatibility-4096-plus-8.shared_staging.worker.json`, only after baseline success;
- `compatibility-4096-plus-8.json`, the aggregate result even when a child fails.

A passing compatibility report proves exact greedy-token parity for that one synthetic cell and
documents its systems costs. It does not prove semantic recall, useful 40K quality, positions above
40,960, compression, sparsity, ATXY selection, production safety, or authorization to expand the
advertised model context.
