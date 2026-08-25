# Qwen3-0.6B development lane

This folder is the daily iteration lane. Its recipes keep the promoted Qwen3-1.7B experiments and
checkpoints unchanged, read the same prepared training shards, and write only to
`checkpoints/qwen3-0.6b-corpus-experts/`.

The 0.6B and 1.7B bases have different weights and dimensions, so their adapters and routers do
not transfer between lanes. Four rank-16 0.6B experts and the five-route v4 MLP router are now
trained under their isolated checkpoint namespaces. Successful ideas still need to be retrained
and checked on the preserved 1.7B promotion lane; no weights are copied between bases.

The recipes intentionally retain each promoted expert's initial hyperparameters as a comparable
starting point. Run one from the repository root, for example:

```powershell
.\.venv\Scripts\molor-train-experts.exe `
  .\experiments\qwen3-0.6b-development\software-architecture-v1-recipe.json `
  --local-files-only
```

`smoke-report.json` records the pinned official snapshot identity and the first local-only,
base-only CPU BF16 generation check. It is a load/forward receipt, not a quality benchmark or
context-expansion result.

## First trained baseline

The 2026-08-25 run completed 4,135 optimizer steps across the four experts. Structural validation
confirmed 196 target projections, 392 finite FP32 LoRA tensors, and 10,092,544 trained parameters
per expert against the exact pinned Qwen3-0.6B base. The router reached 99.69% on both its balanced
650-row validation and held-out test splits.

The recordable seven-policy full matrix completed all 448 scored generations with zero errors. All
policies were comparable and fully content-addressed:

| Policy | Quality index | Delta vs 0.6B base |
|---|---:|---:|
| Base | 519.25 / 1000 | baseline |
| `embedded-knowledge-v1` forced | 538.17 / 1000 | +18.92 |
| `ocb-can-verified-v1` forced | 450.33 / 1000 | -68.92 |
| `software-architecture-v1` forced | 489.25 / 1000 | -30.00 |
| `verified-embedded-code-v1` forced | 469.92 / 1000 | -49.33 |
| Auto top-1 | 503.25 / 1000 | -16.00 |
| Auto top-2 | **549.25 / 1000** | **+30.00; 2 wins / 62 ties / 0 losses** |

Auto top-2 is the benchmark-leading development candidate, but the desktop still initializes Auto
with top-k 1. Set **Top-k** to 2 explicitly to reproduce this result; no default-policy promotion
was made. Its median throughput was 24.92 tok/s versus 36.93 tok/s for base, so reducing the mixture
cost is part of the next iteration. The low OCB/CAN and verified-code training losses did not
translate into broad forced-route quality and must not be treated as generic model improvements.

For CLI reproduction, pass both `--route-top-k 2` and `--confidence-threshold 0.3`; the benchmark
CLI otherwise defaults to top-k 1 and threshold 0.0.

Summed trainer-internal elapsed time was 4,477 seconds here versus 4,644 seconds in the preserved
1.7B receipts; the preserved 1.7B sum was 1.037× the 0.6B sum. These timers start after base-model
loading and the runs were not a controlled end-to-end A/B. Smaller diagnostic subsets and a
dedicated speed benchmark are needed before making iteration-speed claims.

See the [training and benchmark receipt](training-benchmark-report-2026-08-25.json) for base,
adapter, router, dataset, and benchmark hashes; aggregate confidence intervals and throughput; the
cross-scale reference; and the explicit development decision. The benchmark is not
context-expansion evidence.
