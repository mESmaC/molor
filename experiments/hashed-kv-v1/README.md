# Hashed KV shadow experiment v1

This experiment measures whether compact Hamming-space routing codes can find the KV pages that
matter to real Qwen3-1.7B attention. It is deliberately **shadow-only**: selected-layer BF16 Q/K/V
is copied from an unchanged dense forward, no live cache is instantiated or altered, and every
approximate decision is graded against reconstructed exact attention.

The current model stores 28 layers x 8 KV heads x 128 dimensions x K/V x BF16, or 112 KiB per
token. That is 448 MiB at 4,096 tokens, about 106.8 GiB at one million tokens, and about 1.04 TiB at
ten million tokens. Page routing can reduce how much of that state must be read for each decode
step, but it does not by itself compress or eliminate the cold KV representation.

## Run

The default is a small local measurement. The output is written beneath `runs/hashed-kv-v1/`,
which is excluded from Git.

```powershell
.\.venv\Scripts\molor-hashed-kv-lab.exe --target-tokens 1024
```

Use a real heterogeneous text file for meaningful measurements:

```powershell
.\.venv\Scripts\molor-hashed-kv-lab.exe `
  --context-file .\path\to\context.txt `
  --target-tokens 4096 `
  --page-size 64 128 256 `
  --hash-bits 32 64 `
  --hash-tables 1 4 `
  --candidate-fraction 0.0625 0.125 0.25
```

The older cache-lab snapshot repeats a short sentence to reach an exact token count. It remains
valid for I/O and round-trip tests, but it must not be used to support semantic-retrieval or
deduplication claims.

## Decision rule

The current `manifest.json` describes a development sweep, not an enforced promotion evaluator.
Promotion requires the sealed-run gates in `docs/kv-cache.md`; the current smoke report lacks the
full identity, all-layer/head coverage, distributional metrics, and physical-fetch union required by
that contract. A failed training-free SimHash result leads to a stronger page baseline or an
asymmetric learned Q/K hash experiment, not directly to a live sparse cache.

Approximate routing hashes are not content identities. Hash collisions only add candidates. Any
future content-addressed deduplication must use a cryptographic digest of deterministic encoded
bytes followed by exact byte/code equality verification.

## Corrected 4K development result

The corrected built-in 4,096-token case places its recovery code at tokens 2,048–2,055 in one cold
page, outside the mandatory sink and recent pages. It samples the final two query positions, four query
heads, layers 0/13/27, one 32-bit table, and a 25% candidate-token budget. Each page size has 24
shadow evaluations (48 total). These are development measurements, not a sealed result:

| Page size / selector | Top-32 recall | Retained attention mass | Relative output L2 |
|---|---:|---:|---:|
| 32 / mandatory only | .483 | .687 | .327 |
| 32 / SimHash | .621 | .756 | .224 |
| 32 / min-max bound | .935 | .930 | .048 |
| 32 / forced-set oracle | .997 | .950 | .034 |
| 64 / mandatory only | .547 | .710 | .276 |
| 64 / SimHash | .671 | .782 | .184 |
| 64 / min-max bound | .939 | .931 | .048 |
| 64 / forced-set oracle | .991 | .946 | .038 |

The attribution view makes the routing result clearer:

| Page size / selector | Mass added beyond forced pages | Cold-mass recall | Evidence-page inclusion |
|---|---:|---:|---:|
| 32 / SimHash | .069 | .242 | .333 |
| 32 / min-max bound | .243 | .752 | .375 |
| 32 / forced-set oracle | .263 | .810 | .375 |
| 64 / SimHash | .071 | .253 | .333 |
| 64 / min-max bound | .221 | .713 | .375 |
| 64 / forced-set oracle | .236 | .764 | .375 |

Evidence-page inclusion is the fraction of sampled layer/head candidate sets containing the one page
that holds the recovery-code tokens. It is not answer accuracy, and even the attention-mass oracle
does not select that page for every head. The result rejects shared untrained SimHash as the next
live-cache selector: it recovers only about 24–25% of cold attention mass. The Quest-style min/max
bound tracks the forced-set oracle much more closely, but even the oracle's ~.95 total mass and
.034–.038 output error miss the promotion contract.

That quality comes with more metadata and arithmetic. The current float32 min/max descriptor is
1,024 bytes per page per indexed head: 32 bytes/token for 32-token pages and 16 bytes/token for
64-token pages, versus 4 bytes/token for one 32-bit SimHash table. It also uses floating-point page
scoring rather than XOR/popcount. The next work must therefore compare end-to-end cost as well as
quality on a real heterogeneous 4K–32K corpus, with all-layer/head capture and candidate-budget
curves, before trying an asymmetric learned Q/K or product-quantized router.
