# ATXY local-latent KV ceiling v1

This fixed development experiment asks the question that the earlier Tucker and raw-vector RVQ
screens left open: can a simple latent preserve dense attention *before* ATXY codebooks are added?
It fits an independent linear basis for every selected Qwen layer and KV head. The K basis balances
uncentered post-RoPE K and grouped post-RoPE Q second moments; the V basis is centered PCA. One
rank-128 fit supplies every nested rank prefix.

The file-disjoint 4,096-token calibration and held-out sources, layers 0/13/27, all 16 query heads,
the final eight held-out query positions, rank matrix, BF16 latent representation, FP32 basis
overhead, 40,960-token storage projection, and gates are frozen in `contract.json`. The gates reuse
the KV design's KL/top-1 fidelity targets and add top-32 and attention-output requirements. They
apply both overall and separately to every selected layer, so an aggregate cannot conceal a weak
layer.

The frozen dimensional accounting for all 28 layers at 40,960 tokens is:

| K/V rank | Total KV bytes | Net ratio | Storage gate |
|---|---:|---:|---|
| 64/96 | 2.752 GiB | 1.590x | eligible |
| 64/128 | 3.302 GiB | 1.325x | eligible |
| 80/112 | 3.302 GiB | 1.325x | eligible |
| 96/96 | 3.302 GiB | 1.325x | eligible |
| 96/112 | 3.577 GiB | 1.223x | fail |
| 112/112 | 3.852 GiB | 1.136x | fail |
| 128/128 | 4.402 GiB | 0.994x | fail |

Native BF16 KV is 4.375 GiB. These figures include BF16 K/V latents, both FP32 basis prefixes, and
the FP32 V centers; they are arithmetic, not measured allocation or speedup.

```powershell
.\.venv\Scripts\molor-atxy-local-latent-lab.exe `
  --contract experiments\atxy-local-latent-v1\contract.json
```

The lab writes one non-overwriting report under `runs/atxy-local-latent-v1/`. A report is tracked
only after its declared identities and recomputed summaries pass an independent artifact audit.

## Result and stop decision

The preregistered run produced 7,680 attention comparisons. An independent audit recomputed every
record, all 80 overall/per-layer summaries, every storage projection, and all fixed gates with zero
mismatches. No joint arm passed.

- The 1.590x K64/V96 arm reached only `0.7604` top-1 agreement, `0.1530` mean KL, and `0.7240`
  global output relative L2.
- The 1.325x K96/V96 arm reached only `0.8516` top-1 agreement, `0.04462` mean KL, and `0.6538`
  global output relative L2.
- Isolating rank-120 K still left `0.9609` top-1 agreement and `0.1476` global output relative L2.
  Isolating rank-120 V left `0.4022` global output relative L2, identifying V as the harder wall.
- Only K128/V128 was faithful, but its `0.9938x` ratio is larger than native BF16 KV and its worst
  selected-layer top-1 agreement still missed the fixed gate.

The local linear rank-prefix representation is therefore rejected for active KV. It does not
advance to ATXY codebooks and does not authorize context expansion. Exact ATXY page addressing
survives as physical identity; a future payload representation needs a qualitatively different
residual or multi-subspace design and a new preregistration. The machine-readable decision and
evidence hashes are in `stop-decision.json`.

This is deliberately not a context-extension run. A pass says only that a local linear latent is
credible enough to receive a separately versioned ATXY codebook experiment. It does not establish
all-layer quality, 40K quality, long-context positional behavior, a GPU kernel, a live cache,
semantic routing, or an end-to-end speedup. Exact `A.TXY` addresses remain physical page identity;
they are neither similarity hashes nor lossy payload codes.
