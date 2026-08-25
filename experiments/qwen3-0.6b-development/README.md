# Qwen3-0.6B development lane

This folder is the daily iteration lane. Its recipes keep the promoted Qwen3-1.7B experiments and
checkpoints unchanged, read the same prepared training shards, and write only to
`checkpoints/qwen3-0.6b-corpus-experts/`.

The 0.6B and 1.7B bases have different weights and dimensions, so their adapters and routers do
not transfer between lanes. Five rank-16 0.6B experts are trained under their isolated checkpoint
namespace. The v4 MLP router remains frozen on the original four experts plus the base/null route;
the fifth, `reasoning-trace-v1`, is manual-only. Successful ideas still need to be retrained and
checked on the preserved 1.7B promotion lane; no weights are copied between bases.

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

The earlier 4,477-versus-4,644-second cross-scale comparison was invalid: the preserved 1.7B
receipts omit the first 750 OCB continuation steps. A controlled same-work 0.6B short-input profile
instead measured a 1.565× single-run speedup from disabling gradient checkpointing, BF16 adapter
autocast, one-time pretokenization, and stable length sorting within unchanged optimizer groups.
Disabling checkpointing alone accounted for 1.481× of that result. The actual 1,024-token reasoning
shape required checkpointing at micro-batch three on this 12 GiB GPU, so 1.565× is not a measured
speedup for the reasoning run. See `training-speed-report-2026-08-25.json` for the measurements and
limitations.

See the [training and benchmark receipt](training-benchmark-report-2026-08-25.json) for base,
adapter, router, dataset, and benchmark hashes; aggregate confidence intervals and throughput; the
cross-scale reference; and the explicit development decision. The benchmark is not
context-expansion evidence.

## Reasoning trace expert

`reasoning-trace-v1` trained for one epoch over 12,000 source-disjoint, explicit-thinking examples:
1,334 optimizer steps in 2,060 seconds, ending at training loss 1.1623. Structural validation
passed for all 392 finite FP32 LoRA tensors and 10,092,544 parameters. The weight-updating rows came
only from two sources labelled MIT; the Apache-2.0-labelled rows were reserved for validation and
test. We treat those labels as sufficient provenance for this private experiment, not as public
release clearance, and keep all rows and weights outside Git.

On the untouched 512-example test split, the expert reduced assistant-token NLL from 2.9424 to
2.7751, a 5.69% improvement. That imitation gain did not transfer to the frozen broad benchmark:
the forced expert scored 404.08/1000 versus 519.25 for base, including 50.0% versus 66.7% on general
reasoning and 0.0% versus 16.7% on quantitative. It therefore failed the preregistered target gate,
was not added to auto routing, and remains available only for manual experiments. The benchmark
used its comparable `enable_thinking=false` default; that train/eval format mismatch is recorded,
not post-result tuned.

See `reasoning-trace-v1-training-benchmark-report-2026-08-25.json` for corpus hashes, license
provenance, training and structural receipts, aggregate held-out metrics, frozen benchmark
fingerprints, and the non-promotion decision.
