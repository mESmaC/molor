# ATXY KV shadow experiment v1

This experiment tests a new continuous tensor codec inspired by ATXY's four-axis organization.
It is separate from CDAS's exact `A.TXY` addresses and from MoLoR's approximate hashed-KV router.
It does not modify live inference.

The codec treats either K or V as `[layer, token, KV head, head channel]`, preserves the appendable
token axis, and fits fixed Tucker bases over the other three axes using calibration-only captures.
K and V use independent codecs. Held-out evaluation measures reconstruction, attention KL/top-k,
and attention-output error for compressed-K only, compressed-V only, and both compressed.

The checked-in development manifest uses file-disjoint software-architecture train and validation
sources at 512 tokens for a fast falsification run. It is not a sealed or representative corpus.

```powershell
.\.venv\Scripts\molor-atxy-kv-lab.exe `
  --dataset-manifest experiments\atxy-kv-v1\dataset-manifest.json `
  --rank 28x8x64 `
  --rank 14x8x64
```

Reports are explicitly non-promotional and are written under `runs/atxy-kv-v1/` and
`runs/atxy-vq-v1/`. The eight reports named by the hashes below are intentionally tracked as
immutable evidence despite the general `runs/` ignore rule. A codec must pass dense
reconstructed-attention tests before it can be combined with page routing or a live cache.

## Tucker stop decision

The first held-out matrix captured every Qwen layer for one file-disjoint 512-token calibration
context and one 512-token validation context. Bases used 64 evenly sampled calibration positions;
each arm was evaluated at the final two positions for four query heads in every layer (224 dense
attention comparisons per K-only, V-only, and K+V variant).

| Representative arm | Net ratio | Mean attention KL | Attention top-1 | Mean output relative L2 | Decision |
|---|---:|---:|---:|---:|---|
| channel-only `28x8x64` | 1.995x | 0.4045 | 85.7% | 0.7968 | reject |
| mixed K `28x8x120`, V `26x8x128` | 1.069x | 0.02041 | 95.1% | 0.1875 | reject |
| near-full K `28x8x128`, V `27x8x128` | 1.016x | 0.0000123 | 100% | 0.04038 | reject |

The final arm shows that even dropping only one cross-layer V component creates unacceptable output
error and a severe p99 tail for negligible savings. Cross-layer/head/channel Tucker factorization is
therefore stopped for active KV. This is a negative result for this specific continuous codec, not
for exact ATXY page addressing or literal four-codebook vector quantization.

The four source report SHA-256 values are:

- `77687a8fe216ca9da58a32d448997d6a4e0cffb3c95ac97a8260ee228f114a17`
- `db0f95b3aa3afde73c8f959ea404f0b0e8826269821f9ca1bab51ecd91588cfa`
- `e583aa117a945035b8bad18d4473852edbe9f5946e775a4047df8c3f362fb9af`
- `cd2f38f169af0e647ee6998e91c94ec8855d9c235dfb72ba21d878312f885e7e`

## Literal residual-VQ stop decision

The next arm implemented the literal codebook interpretation: each complete ATXY tuple indexes four
ordered 36-entry additive residual codebooks. Global and per-layer 512-token screens failed, so the
strongest follow-up used separate K/V codecs for every KV head in layers 0, 13, and 27, trained on
4,096 calibration tokens and evaluated on a file-disjoint 4,096-token context. All 16 query heads
and the final eight positions produced 384 attention comparisons per variant.

| Complete ATXY tuples/vector | Stored bits/vector | Net ratio | K relative L2 | V relative L2 | Mean KL | Attention top-1 | Output relative L2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 19.69x | 0.1273 | 0.7833 | 3.165 | 13.8% | 1.486 |
| 2 | 64 | 9.85x | 0.1126 | 0.7389 | 2.636 | 13.0% | 1.455 |
| 3 | 96 | 6.56x | 0.1052 | 0.7046 | 2.234 | 19.3% | 1.275 |
| 4 | 128 | 4.92x | 0.1000 | 0.6730 | 1.930 | 24.2% | 1.223 |

More tuples improve reconstruction monotonically, but raw V remains far outside the pilot gate.
Direct raw-vector `4x36` RVQ is therefore stopped for active KV. A future ATXY payload experiment
must first establish a per-layer/head latent representation whose own dense-attention ceiling passes;
codebook quantization cannot repair a lossy latent.

The four VQ report SHA-256 values are:

- `0c13406fa3ef37c0d44c0245de00590622423254fd9ac66cf67e949740963510`
- `193a6ce34b95c4906dfa9dd6b1da84081d477d053df6c6167874e54addc0237c`
- `a84e5c251a12ecd04ef996db6af848e4968603c7442cbdcb2c9b7f7ae8391c46`
- `2945ef2ed4c47649cbbee89ea1b6acd35747dfc51f6fc23270e2980e7a95e74c`
