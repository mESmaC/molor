# ATXY staged KV phase 1

This phase implements the correctness-first CPU reference primitives needed before a live
CPU/GPU KV cache can be trusted. It follows the exact-payload page-oracle stop decision: fixed
25–35% page budgets are rejected, so selection must refine progressively and fall back to all
pages whenever it cannot certify enough retained attention mass.

The implementation in `src/molor/experimental/atxy_staged_kv.py` provides:

- immutable `A.TXY` physical page identities and exact partial-page byte accounting;
- per-page K/V SHA-256 fingerprints over address, span, dtype, shape, and logical bytes;
- conservative per-query-head QK upper bounds from exact page key extrema;
- one layer-shared progressive page set for every GQA head;
- outward-rounded lower bounds on retained attention mass;
- explicit page-cap fallback with separate probed and final selected-mass certificates; and
- numerically stable blockwise GQA attention over exact pages in arbitrary order.

The selector accepts fingerprint-verified page keys and computes exact page logits internally.
It does not accept caller-supplied logits. Observation batches are atomic: invalid shape, dtype,
fingerprint, bounds, or page membership leaves selector state unchanged.

## Correctness scope

A non-fallback decision with `target_mass < 1` is a certified retained-mass approximation, not an
exact-output claim. Exact K/V payloads are retained, but omitted attention mass can still affect
the value output. Observing every page, or taking dense fallback and accumulating every selected
page, reproduces dense single-token attention in the CPU reference.

The phase-one accumulator is a numerical kernel. Live integration must verify both the K and V
page fingerprints before using staged payloads. Prefill, multi-token/speculative decode, attention
bias, GPU transfers, model integration, latency, and context extension are deliberately outside
this phase. No speed or context-length claim is made.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_atxy_staged_kv.py `
  tests/test_atxy_page_oracle.py tests/test_hashed_kv.py -q
.\.venv\Scripts\python.exe -m ruff check `
  src/molor/experimental/atxy_staged_kv.py tests/test_atxy_staged_kv.py
```

The final independent re-audit passed 1,000 randomized min/max-bound cases, 1,962 progressive
certificate rounds comprising 8,428 high-precision head checks, and 1,000 shuffled page-order
dense-attention equivalence cases. It included 671 explicit dense fallbacks, 88 certified subset
approximations, and 241 all-page exact selections.

The next phase is an exact CPU-owned cache with reusable bounded GPU staging. Only after byte-exact
round trips, storage bounds, stream ownership, token parity, and memory telemetry pass should these
page primitives be connected to a Qwen attention path.
