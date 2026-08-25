# Hashed KV cache: `hashed-kv-shadow-v1`

Status: the first offline development smoke is implemented; the full sealed observer and matrix are
pending. This document is a design and promotion contract, not a claim that MoLoR has a live sparse,
paged, tiered, or disk-backed KV cache.

## Decision

The first milestone is a **page-level approximate-routing observer** over the exact KV cache. It
will answer one question before changing inference: can compact binary query/key descriptors
identify the historical pages that receive dense-attention mass, at a useful candidate budget,
across the actual Qwen3-1.7B stack?

`hashed-kv-shadow-v1` therefore:

- retains captured BF16 Q/K/V as the canonical evaluation truth;
- partitions the token axis into logical pages and builds rebuildable routing metadata from keys;
- selects hypothetical candidates from mandatory recent/hot pages plus hash-matched cold pages;
- runs the real dense attention path unchanged;
- compares the hypothetical candidates with a reconstructed float32 attention reference from the
  captured Q/K; and
- records the dense truth needed for later heat/TTL policy experiments without changing state.

It does **not** omit tokens from attention, move live pages between devices, serve KV directly from
NVMe, quantize or factorize KV, share storage between equal-looking entries, or claim a latency or
memory win. Those require later milestones and their own gates.

An independent `molor.atxy-kv-shadow-report.v1` lab now tests payload codecs without routing. Its
first Tucker formulation factors the layer, KV-head, and channel axes while retaining time. On the
initial file-disjoint 512-token development split, a 1.995x arm had mean attention KL `0.4045` and
mean output relative L2 `0.7968`; even a 1.016x near-full-rank arm retained `0.04038` mean output
error with a severe p99 tail. That formulation is rejected for active KV. These results do not
invalidate exact ATXY page identity or a later literal A/T/X/Y residual-codebook experiment. See
`experiments/atxy-kv-v1/README.md` for the bounded protocol and report hashes.

That literal experiment is now also bounded. One through four complete ATXY residual tuples were
tested with separate per-layer/head K and V codebooks on file-disjoint 4K captures. At four tuples
the net ratio was `4.92x`, but V relative L2 remained `0.673`, mean attention KL was `1.930`, and
mean output relative L2 was `1.223`. Raw-vector ATXY RVQ is rejected for active KV. Any successor
must first pass a dense-attention ceiling using a separately validated per-layer/head latent; it
must not combine a failing latent with routing and attribute the resulting error to retrieval.

The initial retrieval unit is a logical token page, not an individual token. The CLI defaults to
64 tokens so early measurements do not hide pivotal tokens in a 28 MiB whole-model page; 128, 256,
and 512-token alternatives are measured rather than assumed. A logical
page spans the same positions across all 28 layers; each layer is a separately addressable shard so
a future runtime does not have to read all layers at once.

## Two meanings of “hashed KV”

These mechanisms use hashes for different purposes and must never share an API that allows one to
be mistaken for the other.

### Approximate routing hashes

An approximate routing hash is auxiliary search metadata:

```text
layer query q -> query projector -> b-bit routing code
page keys K   -> page prototypes -> key projector -> b-bit routing codes
                                      |
                                      +-> Hamming candidate search
```

Nearby codes are a hypothesis that a page may deserve attention. Equal codes do not imply equal
keys, and unequal codes do not prove irrelevance. Collisions normally add candidates; they must not
cause payload aliasing or data loss. A selected page still uses the canonical keys and values for
the precise attention operation.

The query and key projectors may be seeded random hyperplanes or separately learned maps. Their
version, weights, training split, bit width, and seeds are part of every run identity. The routing
code is always called `routing_code`, never `content_hash`.

### Exact content-addressed deduplication

Content-addressed deduplication is a storage identity mechanism:

```text
canonical serialized payload -> cryptographic digest -> candidate existing object
                                                   -> byte-for-byte verification
                                                   -> immutable shared object + references
```

Even a cryptographic digest is only an index lookup: the implementation must verify equality before
aliasing storage. The digest namespace must include the cache schema, model and adapter identity,
dtype, layout, positional convention, layer, and any quantizer/codebook version. An approximate
routing code must never be used as a content address.

Exact deduplication is deferred because:

1. Contextual K/V tensors are unlikely to be byte-identical. Keys also contain positional effects,
   so repeated text at different positions is not an obvious duplicate.
2. MoLoR does not yet have a validated discrete KV representation. Repeated codebook tuples may
   become common after quantization or factorization, but sharing two approximately equal payloads
   would itself be a new lossy algorithm, not exact deduplication.
3. Dedup requires immutable objects, reference counts, tenant/session isolation, crash-consistent
   reclamation, and protection against an ABA-style stale reference. None helps establish whether
   hash routing has adequate recall.
4. A digest table consumes memory and CPU even when the duplicate rate is zero. The duplicate rate
   should be measured before building a live ownership system.

A later experiment may perform a read-only duplicate census over a stable quantized representation.
It must report gross and net savings after digest tables, references, padding, and verification.
That experiment is not part of `hashed-kv-shadow-v1`.

### Exact prefix-cache keys are a third mechanism

An exact prefix cache hashes the token IDs and complete model/route contract for an ordered prompt
prefix, then reuses previously computed K/V only when that prefix contract matches exactly. It can
avoid repeated prefill for shared prefixes, but it does not find semantically related history and it
does not select sparse pages within one long sequence. Prefix identity uses an exact digest and an
exact token/contract check; it never uses Hamming-near semantic routing codes. Prefix caching,
content-addressed payload deduplication, and approximate attention retrieval remain three separately
versioned features.

### Prior-art constraints carried into this design

Prior results motivate an experiment, not a systems claim. [HashAttention](https://arxiv.org/abs/2412.14468)'s
compact auxiliary state is 32 bits **per token per attention head**, not simply 32 bits per token for
the whole model, and its highest reported sparsity required task-specific tuning. Its evaluated path
kept the full KV resident; it did not establish a live CPU/NVMe tier.
[MagicPIG](https://arxiv.org/abs/2410.16179)-style LSH retrieval demonstrates a hot-GPU/cold-CPU
direction, but configurations using hundreds of LSH tables can consume substantial CPU index
memory. MoLoR must therefore report total index bytes, table count, tuning provenance, and payload
residency alongside recall. Neither paper's sparsity factor is inherited as a target or expected
result here.

[RetrievalAttention](https://arxiv.org/abs/2409.10516) is the warning against assuming a symmetric,
off-the-shelf ANN index: attention queries and cached keys have different distributions, and GQA
query heads may need separate calibration. [PQCache](https://arxiv.org/abs/2407.12820) motivates a
later product-quantized router in which codebook indices support asymmetric query-to-centroid lookup.
Neither result licenses payload aliasing: routing codes and stored KV identity remain separate.

## Measured baseline: local Qwen3-1.7B

The local `models/Qwen3-1.7B/config.json` and the existing real snapshots establish this baseline:

| Property | Current value |
|---|---:|
| Transformer layers | 28 |
| Hidden width | 2,048 |
| Query heads | 16 |
| KV heads | 8 |
| Query heads per KV head | 2 |
| Head dimension | 128 |
| Cache dtype | BF16, 2 bytes/element |
| Configured maximum positions | 40,960 |
| Sliding-window attention | disabled |

For batch size one, one token of dense KV is:

```text
28 layers * 2 tensors (K and V) * 8 KV heads * 128 values * 2 bytes
= 114,688 bytes
= 112 KiB/token
```

That makes a 256-token logical page 28 MiB across the model, or 1 MiB per layer shard. It also gives
the following uncompressed payload sizes, excluding allocator and index overhead:

| Tokens | 256-token logical pages | Exact BF16 KV payload |
|---:|---:|---:|
| 4,096 | 16 | 448 MiB |
| 40,960 | 160 | 4.375 GiB |
| 1,000,000 | 3,907, including the final partial page | 106.81 GiB |
| 10,000,000 | 39,063, including the final partial page | 1.044 TiB |

The million-token rows are capacity projections, not supported context lengths. This checkpoint is
configured for at most 40,960 positions; longer tests require a separately validated model/position
contract and cannot be reported as Qwen3-1.7B baseline results.

The existing `molor.kv-snapshot.v1` 4,096-token run measured 469,762,048 tensor bytes (exactly
448 MiB) and a 469,767,400-byte safetensors file. It recorded 1.25245 seconds for prefill, 0.14393
seconds to stage GPU tensors to CPU, 0.25442 seconds to write and `fsync`, and 0.14609 seconds for an
immediate read. Reload produced the same next token with maximum logit difference `0.0`.

Those measurements prove snapshot round-tripping on that run. The immediate read was explicitly
OS-page-cache affected. It is not a cold-NVMe latency measurement and does not show that disk can
feed live decoding.

## Cache and page model

A logical page owns a causal token interval `[start, end)` for one sequence. Its 28 layer shards
contain exact K and V tensors in the current milestone. Retrieval metadata is derived from K only,
because attention eligibility is determined by query/key scores; V is payload consumed after a page
is selected.

Conceptual metadata is:

```text
KVPage {
    page_id
    sequence_id
    token_start
    token_end
    valid_token_count

    model_contract_digest
    route_contract_digest
    cache_layout_version
    source_kv_digest          # integrity only, not dedup in v1

    layer_shards[28]          # exact BF16 K/V remains canonical in v1
    routing_descriptors[]     # layer, KV head, prototype, table, routing code

    logical_tier              # shadow label only in v1
    heat
    ttl
    last_observed_query
    state_epoch
}
```

`route_contract_digest` includes every adapter and weight that affected K/V, its coefficient, and
the request-pinned router decision. A cache created under the base route is not silently compatible
with a cache created under an expert route. A route cannot change midway through a cached sequence.

The intended eventual hierarchy is:

| Tier | Eventual meaning | `hashed-kv-shadow-v1` meaning |
|---|---|---|
| Hot | GPU-resident high-fidelity pages, always candidates | Logical forced-candidate label |
| Warm | CPU-pinned or constrained GPU pages, asynchronously fetchable | Logical label only |
| Cold | NVMe-resident compressed or exact page shards, hash indexed | Logical hash-eligible label only |
| Archive | Point-in-time durable snapshot, not latency-bound serving state | Existing snapshot lab only |

No page physically moves in the shadow milestone. The current offline smoke uses an ordinary dense
`use_cache=False` forward and retains selected-layer Q/K/V copies; it does not instantiate or modify
a `DynamicCache`. A later live observer must leave the production cache responsible for real
execution and placement.

## Page descriptors and hash search

### Layer and head scope

Queries change at every layer, so a single prompt-level hash is insufficient. Descriptors are built
per layer and KV head. Under grouped-query attention, query heads `2g` and `2g + 1` map to KV head
`g`.

The first implementation assigns one packed routing code to every token for each measured
layer/head, then scores a page by the minimum query-to-token Hamming distance within it. This is the
high-recall metadata baseline; it costs `hash_bits * hash_tables` bits per token per indexed head.
Page prototypes are a later metadata-compression experiment. A single mean is a required baseline
but can hide one pivotal token inside an otherwise unrelated page; one, two, four, and eight
deterministic prototypes are therefore measured as ablations. Prototype construction and
normalization are versioned.

For table `r` and bit width `b`:

```text
c_key[l,g,t,r]   = sign(f_key[l,g,r](key[l,g,t]))
c_query[l,h,r]   = sign(f_query[l,h,r](q[l,h]))
distance(l,h,p)  = min over tokens t in page p and tables r Hamming(c_query, c_key)
```

Random SimHash uses a recorded seed and fixed hyperplanes. A learned dual projector may use
different `f_query` and `f_key`, but it trains only on dense-attention labels from the training
split. Projector training may not see validation/test documents, sessions, query positions, or
answer-support annotations.

The first page index uses a flat packed-code scan with XOR plus population count. At 40,960 tokens
there are only 160 default-size pages; at one million tokens there would be about 3,907. A flat scan
provides a simple, deterministic baseline without the recall surprises of bucket enumeration.
Multi-index buckets and explicit Hamming-radius probes are later systems optimizations and must
match the flat scan's candidate ordering on the same codes.

### Candidate construction

For the current query and each `(layer, KV head)` pair:

```text
forced = sink pages union causal recent pages
cold_ranked = cold pages ordered by (Hamming distance, page_id)
candidate_pages = forced union first B cold_ranked pages
```

Ties use stable page IDs so replay is deterministic. The development smoke evaluates selected query
heads independently. The sealed observer's primary fidelity record must keep a candidate set per KV
head, then take the layer-wide union across all eight KV heads. That union is the physical-fetch
estimate and exposes I/O amplification hidden by per-head sparsity.

The current incomplete page and a configurable recent window are always forced. The budget report
must count forced pages, hash-selected pages, and their union separately. A result does not qualify
as sparse if an oversized recent/hot set quietly supplies most of the context.

### Heat, TTL, and causality

Heat, TTL, promotion, and demotion are deliberately not implemented in the first observer. They
belong to Stage C after a retrieval selector passes the static shadow gates. The following is the
policy contract for that later milestone.

Tier state used for query `t` is frozen before that query. After dense attention has completed, a
page utility is calculated from the attention mass assigned to that page across layers and heads:

```text
heat_p(t) = alpha * heat_p(t-1) + (1 - alpha) * utility_p(t)
```

The default utility is the maximum per-layer/per-head page mass, chosen conservatively so a page
that is pivotal to one head is not erased by averaging. Mean and percentile aggregations are
required ablations. A page above `refresh_threshold` resets TTL. Promotion requires repeated hits;
demotion requires both TTL expiry and heat below a lower threshold. Separate promote/demote
thresholds provide hysteresis. Every threshold, capacity, and update order is versioned.

Dense attention from query `t` may update state for query `t + 1`; it may not alter candidates for
query `t`. This prevents a shadow-only oracle leak. In a future sparse runtime, attention is observed
only for selected pages, so the policy also needs periodic dense audits, a small exploration budget,
and dense fallback on low confidence. Otherwise a false-negative cold page can never become hot.

TTL expiry means “eligible for demotion,” not deletion. Cold pages remain indexed until the sequence
is explicitly released.

## Shadow execution

The observer proceeds in this order:

1. Validate and record the immutable model, tokenizer, route, cache, and environment contracts.
2. Capture the exact post-positional-transform keys used by attention and the corresponding layer
   queries without replacing the model output.
3. Partition only past causal positions into pages and build descriptors/codes.
4. Snapshot pre-query tier state and compute hypothetical candidates.
5. Execute ordinary dense attention and return its untouched result.
6. Reconstruct or capture dense attention weights for evaluation and calculate retrieval metrics.
7. In Stage C only, simulate heat, TTL, and logical tiers for the next query; the initial observer
   skips this step.
8. Append an immutable query record. Never rewrite an earlier decision after observing later data.

If the production attention backend does not expose weights, the lab may recompute QK/softmax for
sampled query positions using a clearly labelled reference path. That reconstruction must first pass
parity tests against an eager dense implementation, including causal masks, RoPE state, BF16
rounding, GQA head mapping, and partial final pages. Switching the generation backend merely to
obtain weights is a different numerical baseline and must be reported as such.

## Workloads and split discipline

The corpus must expose both average behavior and catastrophic misses:

- natural long documents, long conversations, source trees, and mixed-topic concatenations;
- exact and paraphrased needle retrieval at early, middle, and late positions;
- multiple relevant passages whose evidence must be combined;
- repetitive text and duplicated templates that stress broad hash buckets;
- semantically close distractors that stress false positives;
- one-token pivots hidden inside otherwise unrelated pages, stressing page prototypes; and
- query/key distribution shifts caused by each supported adapter route.

Use document/session-disjoint train, validation, and sealed test partitions. Synthetic templates,
source documents, insertion positions, paraphrases, and random seeds are split before examples are
rendered. A learned hasher is selected on validation once and evaluated on the sealed split once.

Run lengths of 4,096, 8,192, 16,384, 32,768, and a safe near-limit length no greater than 40,959
tokens. Promotion statistics use 16,384 tokens and above; shorter contexts are retained for overhead
and regression analysis. Queries are sampled throughout each sequence, not only at the final token.

Start with the null/base route, then stratify the locked configuration over the currently promoted
resident experts (`embedded-knowledge-v1`, `ocb-can-verified-v1`, `software-architecture-v1`, and
`verified-embedded-code-v1`) and the pinned auto-router decision. Results must be reported per route;
an aggregate may not conceal one incompatible expert distribution.

## Experiment matrix

The matrix is staged to avoid choosing a learned hasher while simultaneously tuning every systems
parameter.

### Stage A: truth and aggregation

Hold hash search out of the loop and measure:

| Factor | Values |
|---|---|
| Page size | 64, 128, 256, 512 tokens |
| Prototypes per page/head | 1 mean, 2, 4, 8 |
| Page utility aggregation | mean, p95, maximum across layer/head |
| Logical candidate form | per-KV-head; layer-wide physical union |

For each budget, record the dense oracle: pages ranked by actual attention mass. This is an upper
bound for page granularity. If the oracle itself cannot satisfy the quality gate at an acceptable
budget, hashing is not the immediate bottleneck; page size or candidate granularity must change.

### Stage B: retrieval baselines

At identical total candidate budgets compare:

| ID | Selector | Purpose |
|---|---|---|
| B0 | All causal pages | Dense fidelity reference |
| B1 | Recent pages only | Required locality baseline |
| B2 | Recent plus seeded random cold pages | Budget-matched chance baseline |
| B3 | Float page-prototype QK score | Descriptor-quality ceiling; not a hash result |
| B4 | Quest-style per-dimension page min/max bound | Training-free page-retrieval baseline |
| B5 | One-table seeded random SimHash | Reproducible untrained hash baseline |
| B6 | Multi-table/multi-probe SimHash | Collision/recall trade-off |
| B7 | Learned dual Q/K binary projector | Learned semantic-routing candidate |

The current CLI implements packed 32- and 64-bit codes. The sealed Stage B implementation must add
128-bit multiword codes before sweeping bit widths 32, 64, and 128; table counts 1, 2, and 4; recent
windows 1, 2, and 4 pages; logical hot capacities 0, 2, 4, and 8 pages; and total target candidate
fractions 6.25%, 12.5%, 25%, and 50%. Use a coarse sweep on training/validation, retain Pareto-front
configurations, and run only those on the sealed split. Do not report the full cross-product's best
sealed result.

The [Quest](https://arxiv.org/abs/2406.10774)-style baseline stores the componentwise minimum and
maximum key values for each page and uses the query signs to calculate a cheap page score bound. It
is deliberately training-free and non-hashed; it helps distinguish “binary projection is useful”
from the more basic result that page metadata can prune attention. Its descriptor bytes and scoring
cost count against it just as hash index overhead counts against hashed selectors.
The development report records the float32 descriptor footprint: at head dimension 128, the two
bound vectors cost 1,024 bytes per page per indexed head, before indexing or alignment overhead.

### Stage C: state policy

With hash configuration locked, sweep heat decay, refresh threshold, TTL, promotion hit count,
hysteresis, exploration budget, and audit interval. Compare against a fixed hot set and pure LRU.
Report warm-up separately from steady state and include abrupt topic changes.

### Stage D: counterfactual sparse replay

Only after the shadow routing gate passes, rerun recorded queries with non-candidate attention logits
masked. This is still a lab replay, not a fast kernel and not live paging. Compare logits,
perplexity, task outcomes, and multi-token generation with the dense reference. Include the actual
dense-fallback policy in candidate and quality statistics.

Codebook-derived routing keys, KV factorization, ternary/additive representations, residual dropping,
and exact deduplication are excluded from this matrix. They become separate experiments after a
plain exact-KV hash baseline exists.

## Metrics

### Retrieval fidelity

For dense attention probability `a[l,h,t]` and candidate tokens `C[l,h]`, attention-mass recall is:

```text
AMR[l,h] = sum(a[l,h,t] for t in C[l,h]) / sum(a[l,h,t] for all causal t)
```

Report macro means and p01/p05/p50/p95 over queries, plus worst layer, head, context band, workload,
and adapter route. Also report:

- cold-only AMR after excluding forced recent/hot pages from numerator and denominator;
- recall of the maximum-attention token and top-1/top-4 attention-mass pages;
- answer-support/needle page recall where ground truth exists;
- candidate token fraction, page fraction, and layer-union physical-fetch fraction;
- forced-versus-hash contribution and duplicate candidate rate;
- Hamming-distance and margin distributions;
- bucket/neighbor load p50/p95/p99 for bucketed implementations; and
- oracle page recall at the same budget.

Attention mass is the primary proxy, not candidate precision. Missing one pivotal page can change a
generation while retrieving several irrelevant pages usually costs only work.

### Model fidelity

Shadow output must remain unchanged. Counterfactual replay additionally records:

- maximum and RMS logit error;
- next-token top-1 and top-5 agreement;
- KL divergence from dense next-token distributions;
- teacher-forced negative-log-likelihood/perplexity delta;
- exact task and needle accuracy; and
- free-generation divergence, benchmark score, and fallback rate.

Quality comparisons use paired examples and bootstrap confidence intervals. A mean may not hide a
sealed needle failure or a route-specific regression.

### Systems and storage

Record index bytes per token and page, descriptor build time, query-hash time, packed-code scan time,
candidate materialization time, CPU/GPU synchronization, and p50/p95/p99 latency. For later tiered
runs also record bytes requested versus read, layer-union read amplification, queue depth, prefetch
hit rate, page faults, promotions/demotions, tier residency, and write amplification.

Shadow candidate fraction is not a speedup. Counterfactual masking with a dense kernel is not a
speedup. Only end-to-end measurements from an implementation that avoids the work and data movement
may claim one.

### Deduplication metrics, when separately authorized

An eventual dedup census reports byte-exact duplicate rate, unique payload count, digest and lookup
cost, verification cost, reference-table bytes, fragmentation, and net retained bytes. Similar
routing codes and repeated quantized codes are not counted as exact duplicates unless the declared
canonical payloads are byte-identical.

## Promotion gates

All thresholds apply to a preregistered configuration on the sealed, document-disjoint test set.
Changing a threshold or selector after sealed evaluation creates a new version and new sealed run.

### Gate 0: observer integrity

Before retrieval results are considered:

- every sampled query has complete records for 28 layers, 16 query heads, and the correct mapping to
  8 KV heads;
- tensor shapes and byte counts reproduce 112 KiB/token for the current BF16 model;
- causal positions, partial pages, masks, and position IDs pass boundary fixtures;
- shadow-on and shadow-off return bit-identical logits under the same deterministic backend; if a
  backend prevents bit identity, a preregistered numerical tolerance and the reason are reported,
  and the run is not described as zero-effect;
- reconstructed dense attention passes eager-reference parity;
- current-query candidates never depend on current/future dense attention;
- identical run contracts reproduce identical routing decisions; and
- corrupt, stale, or mismatched metadata fails closed to ordinary dense attention.

### Gate 1: useful hash routing

On contexts of at least 16,384 tokens:

- the 95% bootstrap lower confidence bound of mean total AMR is at least `0.995`;
- p01 total AMR is at least `0.98`;
- maximum-attention-token page recall is at least `0.995`;
- every preregistered critical needle/support page is retrieved;
- mean cold-only AMR improves by at least 0.05 absolute over recent-plus-random at the same budget;
- median total candidate token fraction is at most `0.25`, p95 is at most `0.35`, and the
  layer-union physical-fetch fraction is reported; and
- no workload, length band, layer quartile, or adapter route misses the total-AMR target by more
  than 0.01.

The float-prototype ceiling must also pass. If it fails, do not compensate with more hash training;
change page/prototype granularity.

### Gate 2: counterfactual model fidelity

With the routing configuration frozen:

- next-token top-1 agreement with dense is at least `0.995`;
- mean KL divergence is at most `0.01` nat and p99 is at most `0.10` nat;
- relative perplexity increase is at most `1%`;
- aggregate executable/task score drops by no more than one percentage point and no reported
  category drops by more than two points;
- the critical needle suite has zero additional failures;
- dense fallback is at most `5%` of eligible queries; and
- effective candidate fraction, counting each fallback as 100%, remains at most `0.30` median.

Failure of any fidelity gate keeps the feature in shadow. Candidate fraction may not be relaxed
after seeing sealed quality without assigning a new experiment version.

### Gate 3: live sparse GPU attention

A later GPU-resident implementation must reproduce the accepted counterfactual mask and quality,
pass race and mixed-batch tests, and on the pinned target hardware at 32K+ context demonstrate:

- at least `1.5x` improvement in the isolated attention component;
- at least `1.2x` end-to-end decode throughput or latency improvement;
- no p95 latency regression at shorter supported contexts; and
- index/search/materialization overhead below `10%` of decode time.

These are promotion targets, not current results. Report TTFT and decode separately.

### Gate 4: live tiering and NVMe

CPU and then NVMe tiering require all prior quality gates plus controlled cold-read measurements,
bounded p99 stalls, crash/cancellation safety, checksum validation, backpressure, and an exact dense
fallback. OS-page-cache-warm reads must be separated from controlled cold and steady-state reads.
The NVMe configuration must improve an explicitly memory-constrained workload end to end; raw
sequential bandwidth alone is insufficient.

## Versioned artifacts

The current development smoke emits one compact JSON report under `runs/hashed-kv-v1/`, grouped by
comparable configuration and explicitly marked ineligible for promotion. A sealed matrix run is
immutable under the following stricter artifact contract:

```text
runs/kv-cache/hashed-kv-shadow-v1/<timestamp>-<config-digest>/
```

The required files are:

| Artifact | Contract |
|---|---|
| `manifest.json` | `molor.hashed-kv-shadow-run.v1`; run identity and artifact digests |
| `config.json` | Canonical page, hash, candidate, heat, TTL, fallback, and seed settings |
| `environment.json` | OS, CPU, GPU, driver, CUDA, PyTorch, Transformers, attention backend |
| `dataset-manifest.json` | Source/split/template hashes and query-position derivation |
| `page-catalog.parquet` | Immutable page identities, spans, descriptor references, source digests |
| `routing-records.parquet` | Pre-query state, candidates, distances, dense truth, post-query update |
| `projector.safetensors` | Learned projector weights, when used |
| `projector-train.json` | Training inputs/splits, optimizer, seed, metrics, parent digest |
| `summary.json` | Machine-readable metrics and gate decisions |
| `report.md` | Human-readable results with limitations and failed strata |
| `checksums.sha256` | Digest of every emitted artifact |

The manifest pins model weight/config/tokenizer/chat-template digests, cache dtype/layout, RoPE and
position settings, adapter set and router decision, source commit, dirty-tree state, workload IDs,
all random seeds, and these independent schema IDs:

- `page_layout_version`;
- `prototype_version`;
- `routing_hash_version`;
- `index_version`;
- `tier_policy_version`;
- `dense_truth_version`; and
- `metric_version`.

Changing any one produces a new run identity. Reports compare only compatible dense truth and
metric versions. Large exact KV/attention references may be stored once by integrity digest, but
the manifest must state their location and retention; that storage reuse is not live KV dedup.

## Invariants

1. Captured BF16 Q/K/V is authoritative evaluation truth throughout `hashed-kv-shadow-v1`.
2. Routing codes are rebuildable metadata and never payload identity.
3. Hash collisions can add candidates but cannot alias, overwrite, or discard KV.
4. Every candidate is causal, belongs to the same sequence, and matches the pinned model/route
   contract.
5. The current page, mandatory recent window, and declared hot pages are always candidates.
6. Current-query dense attention updates only future state.
7. K determines routing; selected pages retain their matching K and V together.
8. Page IDs and tie-breaking are deterministic. Partial pages carry an explicit valid length.
9. An invalid/missing index, stale epoch, checksum error, unsupported backend, or exhausted fetch
   deadline falls back to exact dense behavior rather than silently shrinking context.
10. Index and logical tier deletion never delete canonical sequence state.
11. Session/tenant boundaries are not crossed. Telemetry excludes raw KV and recoverable prompt
    content unless an explicitly secured research artifact requires them.
12. A shadow result cannot be described as compression, acceleration, or live tiering.

## Failure modes and required responses

| Failure mode | Detection | Response |
|---|---|---|
| Coarse codes create overloaded neighborhoods | Bucket/load and candidate-fraction tails | More bits/tables, better projector, or reject configuration |
| Fine codes miss relevant pages | AMR, pivotal-token, needle recall | Multi-probe/tables, larger budget, dense fallback, or reject |
| Page mean hides one important token | Oracle/prototype ablation and planted pivots | More prototypes or smaller pages |
| Recent window appears to make hashing succeed | Cold-only AMR and budget-matched baselines | Require hash lift gate |
| One head/layer is masked by the mean | Per-head/layer p01 and worst-stratum reports | Head-aware budget or reject |
| Per-head sparsity causes broad physical union | Layer-union fetch fraction | Co-design selector/page layout before paging |
| Heat/TTL oscillates | Promotion/demotion and residency churn | Hysteresis, minimum residency, rate limits |
| Missed cold pages never regain heat | Audit miss rate and exploration hits | Periodic dense audit, exploration, confidence fallback |
| Adapter or router changes Q/K distribution | Per-route metrics and contract mismatch | Route-specific calibration or dense fallback |
| Train/test leakage inflates learned hashing | Dataset and projector lineage audit | Invalidate run and rebuild splits |
| RoPE/mask/partial-page bug corrupts truth | Eager parity and boundary fixtures | Fail Gate 0 |
| Index work exceeds saved attention | End-to-end profiles | Keep shadow or disable below break-even length |
| CPU/GPU synchronization serializes decode | Timeline traces and p95/p99 latency | Async queues, batching, resident metadata, or reject |
| NVMe read amplification dominates | Requested/read bytes and layer-union ratio | Layer shards, larger batches, better prefetch |
| OS page cache is reported as cold NVMe | Controlled warm/cold protocol | Relabel or invalidate claim |
| Page corruption or partial write | Version, length, checksum, atomic commit | Reject page and use exact fallback/recompute |
| Routing hash is treated as a content hash | Type/schema assertion and code review | Hard error; APIs remain distinct |
| Cross-session reuse leaks content | Namespace and authorization audit | No cross-session reuse; revoke artifacts |
| Dense fallback hides poor sparsity | Effective candidate fraction counts fallback as 100% | Fail budget gate |
| Longer-than-config tests are misreported | Position-contract validation | Reject run or identify a new validated model contract |

## Staged roadmap to a live tiered cache

### 0. Preserve the exact baseline

Keep the current dynamic/offloaded/disabled cache modes and `molor.kv-snapshot.v1` archive lab.
Strengthen model, route, environment, and checksum manifests. The archive remains explicitly
different from live disk decode.

### 1. Offline `hashed-kv-shadow-v1`

Capture exact Q/K and dense truth from supported contexts, implement logical pages and flat Hamming
search, run the baseline matrix, and emit the versioned artifacts above. No inference changes.

### 2. Live observer, still dense

Attach the accepted observer to ordinary generation. Build/query the index asynchronously while the
model continues dense attention. Validate state ordering, batching, cancellation, adapter routes,
and sustained overhead. Failure simply disables observation.

### 3. Counterfactual sparse replay

Mask recorded non-candidates in a correctness lab. Add periodic audits, exploration, and confidence
fallback. Pass Gate 2 before designing conclusions around speed.

### 4. GPU-resident sparse attention

Keep all canonical KV on GPU but execute QK/softmax/value aggregation only over selected pages with
a kernel that actually avoids dense work. This isolates sparse-compute benefit from paging. Pass
Gate 3 and identify the context-length break-even point.

### 5. GPU hot plus CPU warm

Make page layer-shards immutable while referenced, pin warm CPU buffers, prefetch asynchronously,
and use generation epochs to prevent stale completions. Start with exact BF16 payloads. Bound memory,
queue depth, cancellation time, and fallback stalls before introducing lossy compression.

### 6. Live NVMe cold tier

Store cold layer-shards in an append/commit format with checksums, aligned reads, an in-memory hash
index, bounded async queues, double buffering, and explicit backpressure. Fetch hash-matched cold
pages before their layer deadline; low-confidence or late fetches use the declared fallback. Measure
controlled cold, warm, and steady-state workloads and pass Gate 4. Archive snapshots remain a
separate durability feature.

### 7. Factorized/quantized KV and codebook-derived routing

Train and validate low-rank, ternary, additive-codebook, and residual representations independently.
Compare hashes derived directly from code tuples with the accepted query/key hasher. Retain an exact
or higher-fidelity hot path and measure error by layer, head, route, and age.

Only after the discrete payload contract is stable should exact content-addressed deduplication be
revisited. Approximate sharing requires its own quality gates and must never be labelled exact dedup.

### 8. Longer-context model contract

Move beyond 40,960 tokens only with a model whose positional behavior and long-context quality are
separately validated. Repeat retrieval and systems gates at each length. Capacity arithmetic for
one or ten million tokens is motivation, not evidence that the current model can use that context.

## Milestone completion definition

`hashed-kv-shadow-v1` is complete when one immutable sealed run either passes Gate 0 and Gate 1 or
records a clear negative result, with the full matrix lineage and no effect on generated outputs.
A negative result is useful: it identifies whether page granularity, descriptor quality, hash
recall, route shift, or mandatory-set cost prevents promotion.

Passing the milestone authorizes counterfactual sparse replay. It does not authorize live sparse
attention, CPU/NVMe paging, lossy KV, deduplication, or a million-token claim.
