# Persistent memory v0

Status: accepted architecture with a framework-level control-plane prototype. The immutable event,
snapshot, job, version, head, pin, and model-stack identities are implemented. The SQLite/WAL v0
store implements scoped events, logical correction/tombstone resolution, lexical retrieval,
atomic snapshot registration plus consolidation-job enqueue, leased retries, candidate registration,
CAS forward promotion, session-pinned as-of lexical reads, and transactionally allocated
per-session observation sequences. A model-independent runtime coordinator now produces immutable
prepared-generation plans that bind the capability bank/router checkpoint to a request-scoped
memory head and retrieval cut. A tokenizer-bound executor now consumes those plans and a
deterministic backend exercises the complete local stream; the verified loader can load a frozen
unquantized base, but capability and memory adapter tensors are not yet materialized into that
model.
Retrieved records are now handed off as source-labelled untrusted JSON under whole-record byte and
injected-tokenizer token caps; their raw payloads do not enter the capability router or memory gate.
The semantic retrieval projector, actual LoRA training worker, adapter artifact loader, learned
memory gate, invalidation epochs, and real model-generation backend remain future milestones.

The current lexical retriever exists to exercise persistence and isolation semantics; it is not the
intended quality baseline for the experiment. Likewise, the factorized PyTorch layer proves the
two-axis addition rule but does not yet constitute a verified Transformers integration.

### Implemented subset versus target contract

| Concern | Implemented control-plane subset | Target serving contract |
| --- | --- | --- |
| Event order | Caller-supplied sequence, contiguous within one session | Ledger-assigned scope-global sequence safe under concurrent sessions |
| Event schema | Observation, correction, tombstone, JSON payload, exact target identity | Provenance, trust, consent, sensitivity, validity time, retention, ACL, and encrypted payload references |
| Retrieval | Scope-filtered lexical search over logically active events | Versioned semantic projection with ACL, bitemporal validity, lineage, and a durable checkpoint |
| Consolidation cutoff | Caller-declared scalar used to exercise queue/version invariants | Scope-global source cut plus invalidation epoch and exact included/excluded event identities |
| Memory version | Exact job, base, parent, cutoff, adapter, corpus/trainer/evaluation/policy fingerprints | Complete training, replay, transformation, exclusion, reward, and environment provenance |
| Serving pin | Capability bank plus optional memory version, memory-head generation, and a store-issued database commit cutoff enforced by lexical reads | Router/gate/index/generation/seed identities and one coordinator-enforced request epoch |
| Memory execution | Separate one-adapter/null bank summed independently from capability deltas | Verified version loader and a guard-enforced binary apply/null gate |
| Deletion | Logical exclusion from active-event reads; source payload remains in SQLite | Projection/payload erasure, immediate version invalidation, rebuild, and non-recall evaluation |
| Publication | Parent/generation-checked forward CAS promotion | Invalidation/checkpoint fencing, promotion ledger, crash recovery, rollback, and rollback-to-null |

The remainder of this document is the target serving contract unless a paragraph explicitly says it
describes the implemented control-plane subset. In particular, the current consolidation cutoff
must not be interpreted as a safe scope-wide event frontier when several sessions can reuse the
same sequence; the lexical retrieval pin instead uses the database's monotonic commit sequence.

## Purpose and scope

Memory v0 gives a frozen-base `molor` deployment durable cross-session memory without turning live
serving into an uncontrolled continual-training loop. It separates three mechanisms that have
different safety and lifecycle requirements:

1. capability LoRAs provide reusable skills and are selected by the capability router;
2. one separately gated memory LoRA provides slow parametric memory for the authorized scope; and
3. retrieval supplies reversible facts and episodes directly from a durable store.

The first version supports one memory scope and at most one memory LoRA per request. A scope may be
a user, workspace, or agent identity, but scopes must not be combined implicitly. Hierarchical or
multi-party scopes require an explicit later design for precedence, consent, and information flow.

The frozen base never changes in this protocol. Active adapters are never updated in place. Live
requests never train. All parameter writes happen in an asynchronous worker and become visible only
through an atomic version rollover.

## Normative invariants

The words MUST, MUST NOT, SHOULD, and MAY are normative.

- The append-only event ledger is the source of truth. Retrieval records, summaries, training
  examples, embeddings, and LoRA weights are derived projections.
- A correction or deletion is a new ledger event. It never rewrites history silently.
- Retrieval and memory-LoRA activation are independent. Retrieval MUST work when the memory gate is
  closed, no memory version exists, or a memory artifact cannot be loaded.
- Capability route IDs and indices MUST NOT include memory versions. A memory rollover therefore
  cannot change the meaning or width of the capability router's action space.
- The target serving coordinator MUST pin each request's capability bank, capability-router
  checkpoint, memory version, memory-gate policy, retrieval cut, generation settings, and seed
  before generation starts.
- The pinned bindings MUST remain unchanged through prefill and decoding. A rollover affects only
  later requests.
- The v0 memory gate is hard and prompt-level: the selected memory delta is applied at unit scale or
  not at all. A learned fractional scale is a later experiment.
- A memory version MUST bind to the exact base, tokenizer, chat template, target-module contract,
  training inputs, selector and actuator policies, and evaluation artifacts that produced it.
- A stale, unauthorized, unverified, or epoch-mismatched memory version MUST fail closed to the null
  memory route. It MUST NOT prevent retrieval-only service.
- Model-generated summaries, reflections, labels, and training examples are proposals. Their
  provenance MUST point to source events, and they never become authoritative facts by themselves.
- Personally sensitive or externally supplied content MUST NOT enter parameter memory merely
  because it is novel or frequently retrieved.

## The two adaptation axes

For an adapted linear layer `l`, the v0 computation is:

```text
y_l = W_l x_l
      + sum[p_e * DeltaW_capability(l, e) x_l for e in selected_capability_routes]
      + g_memory * DeltaW_memory(l, memory_version) x_l
```

`W_l` is the immutable base projection. The capability router supplies `p_e` for routes in one exact
`AdapterBankManifest`. The memory gate supplies `g_memory` in `{0, 1}` for one separately loaded and
pinned `MemoryVersionManifest`. Each term is a complete LoRA delta; factors from different adapters
are never averaged separately.

The axes have separate meanings and training schedules:

| Axis | Decision | Identity | v0 update policy |
| --- | --- | --- | --- |
| Capability | Which skill expert or null route should solve this prompt? | Ordered capability-bank fingerprint and stable route IDs | Capability router and experts follow `architecture.md` and `rl-curriculum.md` |
| Memory | May this request use the active memory LoRA for its authorized scope? | Scope, memory-version fingerprint, and gate-policy fingerprint | Hard apply/null gate; memory adapter is produced only by asynchronous consolidation |

Retrieval is not a third adapter route. It supplies source-tagged generator context independently of
the two adapter decisions. In v0, the capability router and memory gate consume the original bounded
routing prefix plus trusted request metadata, not retrieved text. A policy that conditions routing
on retrieval requires a new manifest and evaluation experiment. A retrieved correction creates an
explicit source-backed override path, but evaluation must prove that it dominates any stale
parametric tendency. Safe service requires the future coordinator to disable the stale memory LoRA
while that evidence is unresolved.

The capability router MUST run against the exact capability bank it was trained for. The memory
overlay MUST be implemented as a separate runtime binding, not appended to that bank. A deployment
binding records the exact base, capability bank, capability router, memory version, memory gate, and
retrieval policy evaluated together. Changing any one of them requires a new binding and memory
compatibility evaluation.

The target memory gate evaluates in two stages. Non-learned guards first require matching scope
authorization, verified artifacts, exact base/format compatibility, a current invalidation epoch,
and an enabled deployment binding. If any guard fails, `g_memory = 0`. Only then may the versioned
relevance policy choose apply or null from the bounded routing view and trusted metadata. The gate
manifest fixes its feature extractor, checkpoint or rule set, threshold, calibration data, and
abstention semantics. Its report records every guard result, relevance score, threshold, final
decision, and version fingerprint. It never emits a continuous LoRA scale in v0.

The implemented `FactorizedMixtureLoRALinear` does not enforce that gate protocol: it accepts route
weight tensors so its composition math can be tested. Today it enforces only separate axes and a
one-adapter-plus-null memory topology. The verified serving coordinator must construct the binary
memory route and bind it to the pinned `MemoryVersionManifest`.

## Memory hierarchy

The tier number expresses authority and write lifecycle, not physical latency.

| Tier | Contents | Write timing | Persistence | Authority |
| --- | --- | --- | --- | --- |
| T0: event ledger | User assertions, observed outcomes, preferences, procedures, corrections, consent changes, and tombstones | Transactional append | Durable | Sole source of truth |
| T1: session state | Current conversation, tool results, route state, and optional fast neural state | During one request/session | Volatile | Never authoritative after reset |
| T2: retrieval projection | Canonical facts, episodes, reflections, embeddings, validity intervals, and source lineage | Asynchronous but near-real-time projection from T0 | Durable and rebuildable | Read cache governed by T0 |
| T3: memory LoRA version | Stable behavioral tendencies distilled from an authorized T0 cut plus replay | Offline/asynchronous | Immutable artifact | Derived accelerator, never source of truth |
| T4: packed memory generation | Optional consolidation of several proven T3 shards into a replacement memory LoRA | Rare offline maintenance | Immutable artifact | Derived; subject to the same gates as T3 |

T1 fast weights, if later added, MUST be reset at scope boundaries and MUST NOT be checkpointed as
durable memory. Test-time neural-memory work demonstrates useful sequence state, not safe
tenant-scoped persistence. T4 MUST NOT merge anything into the frozen base in v0.

Volatile facts, dates, credentials, secrets, one-off anecdotes, and unverified imported content
belong in T2. Repeated and stable preferences, procedures, strategies, and goal-conditioned
tendencies are candidates for T3. A fact MAY become training material only when it is stable,
authorized, independently verifiable, and accompanied by correction and deletion tests.

## Durable records and content identity

All persisted schemas require a `schema_version`. Canonical JSON and SHA-256 fingerprinting SHOULD
reuse the rules in `molor.identity`. A fingerprint establishes content integrity after trusted code
hashes the materialized bytes; it does not establish publisher authenticity or anonymize
low-entropy content. Scope IDs MUST be opaque rather than names or email addresses, and manifests,
indexes, training sets, and adapters inherit the scope's access controls and encryption policy.

### Target v2 MemoryEvent

The implemented v1 `MemoryEvent` is the smaller subset shown in the table above. The target v2 event
is immutable and contains at least:

- `event_id`, `scope_id`, producer identity, and a producer-scoped idempotency key;
- a monotonically increasing `scope_seq` assigned by the ledger;
- event kind: `assertion`, `observation`, `preference`, `procedure`, `outcome`, `correction`,
  `tombstone`, `consent`, or `policy`;
- typed payload or an encrypted/content-addressed payload reference;
- origin kind and source reference, such as a user turn, tool result, imported artifact digest, or
  model-derived proposal;
- observation time, effective-validity interval, and ledger commit time;
- confidence, trust class, sensitivity class, retention policy, and access-control scope;
- `supersedes_event_ids` and `tombstones_event_ids` when applicable; and
- a fingerprint covering every semantic field, including scope and provenance.

The ledger rejects duplicate idempotency keys with different content. Corrections, consent changes,
scope changes, and tombstones increment a per-scope `invalidation_epoch` and set
`last_invalidation_seq` to their assigned sequence. Ordinary additive events advance `scope_seq` but
do not invalidate an already active memory version. An assertion that conflicts with an active fact
MUST be normalized to a correction or quarantined; the producer cannot bypass invalidation by
labelling a replacement as an ordinary assertion.

An immutable ledger does not justify retaining erased personal content forever. Sensitive payloads
SHOULD be stored as separately encrypted blobs. An erasure transaction destroys the payload and its
searchable derivatives while retaining only the minimum non-reversible tombstone and audit metadata
permitted by policy.

### Target RetrievalRecord

`RetrievalRecord` is a rebuildable projection containing:

- a stable record ID and fingerprints of every source event;
- canonical text or structured data, event and validity times, and active/superseded/deleted state;
- scope and access-control labels;
- the exact embedding model, tokenizer, normalization, and index-configuration fingerprints;
- the highest applied `scope_seq`; and
- derivation lineage for summaries and reflections.

The retrieval API returns record IDs, source-event fingerprints, scores, validity times, and the
retrieval-policy fingerprint with the text supplied to the model. It filters access controls,
tombstones, expired facts, and superseded versions before ranking. A derived reflection is removed
or recomputed when any supporting event is corrected or deleted. Promotion MAY add an advisory
`consolidated_into` version fingerprint, but MUST NOT delete or silently suppress the retrieval
record. Any down-ranking of already consolidated material belongs to a separately versioned and
evaluated retrieval policy.

### Target v2 MemoryVersionManifest

The implemented v1 manifest uses content fingerprints as extension points. Every trained production
memory artifact requires a target v2 manifest binding at least:

- stable logical `memory_id`, `scope_id`, and optional parent fingerprint;
- exact `BaseModelManifest` and `LoRAFormatManifest` fingerprints;
- the complete `AdapterManifest` and verified safetensors descriptors;
- source cut: `through_scope_seq`, `invalidation_epoch`, included event fingerprints, and their
  canonical set digest;
- excluded, deferred, and quarantined candidate IDs with reason codes;
- dataset, transformation, replay-set, initialization, optimizer, seed, and training-run manifests;
- selection-policy and actuation-policy manifests and fingerprints;
- promotion-policy, verifier, reward-engine, generation, cost-normalizer, and evaluation-set
  fingerprints; and
- creation time and software/runtime versions for audit, not as substitutes for content identity.

`version_fingerprint` is computed from the canonical manifest and is never an input field in that
manifest. A trusted catalog or signed promotion record authorizes the resulting fingerprint rather
than participating in its self-hash. Retraining the same logical memory creates a new version
fingerprint. Runtime adapter names are not identity. The parent is lineage only: a candidate built
from a parent is still a complete new artifact and never mutates the parent.

### Target ActiveMemoryHead and ServingMemoryBinding

The implemented `MemoryHead` supports monotonic forward promotion only. The target
`ActiveMemoryHead` is the only mutable publication pointer. It contains `scope_id`, a monotonically
increasing generation, the active memory-version fingerprint or null, and the promotion-event ID.

`ServingMemoryBinding` pins an exact capability bank, capability-router checkpoint, memory version
or null, memory-gate policy, retrieval policy/index generation, base, and generation configuration.
Its fingerprint is logged on every rollout. A memory version can be immutable and valid while no
longer being active.

## Target read path

Before generating a response, the serving coordinator performs this order:

1. Authenticate the caller and resolve exactly one authorized `scope_id`.
2. Read one consistent snapshot containing the scope's `ActiveMemoryHead`, current
   `invalidation_epoch`, `last_invalidation_seq`, and retrieval projection checkpoint.
3. Pin the exact base, capability bank/router, memory gate, memory version, retrieval policy,
   generation manifest, and seed in a `ServingMemoryBinding`.
4. Close the memory gate if the version is absent, its artifacts are unverified, its base or format
   mismatches, its source epoch differs from the current invalidation epoch, or scope authorization
   fails.
5. Require the retrieval checkpoint to cover `last_invalidation_seq`, then retrieve active records
   at the pinned index checkpoint. Apply access control, effective-time, correction, tombstone, and
   TTL rules before semantic ranking.
6. Make one prompt-level capability decision and one independent prompt-level memory-gate decision
   from the bounded routing view, excluding retrieved record text.
7. Reuse both decisions and the pinned artifacts for the whole generation.
8. Log the binding fingerprint, retrieved record/source IDs, route trace, gate decision and reason,
   normalized costs, reward-report fingerprint when available, and output fingerprint.

If the memory artifact cannot be loaded, serving falls back to `g_memory = 0` and retrieval. It MUST
NOT silently substitute another memory version. If retrieval is behind `last_invalidation_seq`, the
request MUST wait for that projection cut or omit the affected memory; stale retrieval is not an
acceptable fallback.

## Target ingestion, queueing, and consolidation

### Transactional ingestion

The event ledger and its outbox MUST share one database transaction:

1. validate authorization, schema, idempotency, retention, payload limits, and conflicts with active
   records;
2. append the `MemoryEvent` and allocate `scope_seq`;
3. increment `invalidation_epoch` for corrections, tombstones, consent changes, or scope changes;
4. append an outbox item for retrieval projection and consolidation consideration; and
5. commit.

The retrieval projector consumes the outbox at least once and applies each event idempotently. It
updates its checkpoint only after the record, embedding, lineage, and tombstone changes commit.
Serving may use the new retrieval state as soon as that commit is visible; it does not wait for
parameter consolidation.

### Secondary selection/actuation factorization

Inside the memory pipeline, learning is separated into two policies:

- The **selection policy** chooses `retrieval_only`, `defer`, `quarantine`, or `consolidate` for an
  event or coherent event set. Its inputs include provenance, recurrence, stability across time,
  verified outcomes, future utility, sensitivity, contradiction state, and estimated interference.
- The **actuation policy** receives only an approved set and chooses `no_write` or one bounded write
  recipe: initialization, target layers, rank, learning rate, steps, replay mix, and regularization.

Both policies have immutable manifests and are logged independently. v0 uses deterministic selection
rules and a small allowlisted actuator grid. Learned contextual-bandit or RL policies run in shadow
until they beat those controls. Selection and actuation MUST NOT be updated in the same experiment
initially. A matched-random selection control at the same write count distinguishes good selection
from mere sparsity; a fixed-actuator control distinguishes selection quality from write strength.

The delayed reward is measured on the resulting candidate, not on how plausible a selected memory
sounds:

```text
R_memory = target_utility
           + prior_memory_retention
           + capability_and_base_retention
           + correction_and_deletion_success
           - privacy_and_security_failures
           - training_cost
           - measured_serving_cost
```

Masked verifier or infrastructure failures are retried or quarantined and never converted to zero.
Selection rewards and actuation rewards use separate rollout groups and fingerprints. Cost terms are
introduced only after minimum quality and safety gates are met.

### ConsolidationJob and queue semantics

The implemented queue is at-least-once. Its content-addressed `ConsolidationJobSpec` covers scope,
exact base, expected parent version, ordered rollover snapshots, the caller-declared cutoff,
dataset builder, trainer, evaluation suite, target LoRA format, and consolidation policy. The store
captures expected head generation as operational queue state; it is not currently part of the job
key. There is no implemented invalidation epoch or projection checkpoint.

The target `ConsolidationJob` additionally contains:

- job ID and deterministic deduplication key;
- scope, `through_scope_seq`, `invalidation_epoch`, expected active-head generation, and expected
  parent fingerprint;
- selection, actuation, training, replay, evaluation, and promotion-policy fingerprints;
- state: `queued`, `leased`, `training`, `evaluating`, `candidate`, `promoted`, `superseded`,
  `retryable_failure`, `terminal_failure`, or `cancelled`;
- lease owner/expiry, attempt count, candidate fingerprint, and bounded failure diagnostics.

The target deduplication key covers every semantic input above, including the invalidation epoch. A
worker claims a renewable lease; no correctness property depends on a lease being unique. Duplicate
workers may produce candidates, but only compare-and-swap publication can activate one. Queue
acknowledgement occurs after publication, not before.

A candidate is normally trained fresh from the exact frozen base using the complete currently
active, selected memory set through its source cut plus general and capability replay. This is the
conservative v0 rule because the project has no validated inverse-update contract for removing a
contribution from a chain of incremental weight updates. The cited interference literature
motivates testing this rule; it does not prove fresh rebuilding is universally optimal. Warm-starting
a parent is allowed only for additive cuts, must be declared in the training manifest, and is never
allowed for a rebuild triggered by correction, deletion, consent, or scope change.

### Target rollover transaction order

Rollover MUST use the following prepare-then-CAS order:

1. **Cut.** Read the expected active head, current invalidation epoch, and a high-water
   `through_scope_seq`. Materialize the active event set and replay/evaluation sets for that cut.
2. **Select.** Record a decision and reason for every eligible event; quarantine unresolved
   contradictions or untrusted sources.
3. **Train.** Produce a new adapter without modifying the active adapter or base.
4. **Evaluate.** Compare the candidate with the current memory version, retrieval-only, capability
   experts without memory, and frozen-base controls under the exact proposed serving binding.
5. **Prepare.** Store adapter bytes, manifests, evaluation reports, and the candidate version in
   immutable content-addressed storage. Mark the queue job `candidate` only after every artifact is
   materialized and verified.
6. **Catch up retrieval.** Require the retrieval projector checkpoint to cover at least
   `through_scope_seq`. This guarantees a safe retrieval fallback at commit time.
7. **Revalidate.** Confirm that the scope's current invalidation epoch, active-head generation, and
   parent fingerprint still equal the job's expected values. Later ordinary additive events may
   remain in retrieval and a future job; any intervening correction, deletion, consent, or scope
   change invalidates this candidate.
8. **CAS publish.** In one database transaction, conditionally update `ActiveMemoryHead` from the
   expected generation/parent to the candidate and next generation, then append a promotion event.
9. **Finish.** After commit, mark the job `promoted` and acknowledge the queue item. Warm the new
   artifact in serving caches without evicting versions pinned by in-flight requests.

The publication transaction is equivalent to:

```text
BEGIN
  assert job.state == JobState.CANDIDATE
  assert active_head.generation == expected_generation
  assert active_head.version == expected_parent
  assert scope.invalidation_epoch == candidate.source_cut.invalidation_epoch
  assert retrieval_checkpoint >= candidate.source_cut.through_scope_seq
  update active_head to (generation + 1, candidate.version_fingerprint)
    where generation == expected_generation and version == expected_parent
  assert exactly one row changed
  append promotion_event(candidate, old_head, new_head, policy_fingerprint)
COMMIT
```

If CAS changes zero rows, the worker marks the candidate `superseded` and requeues from the new
head; it never overwrites the winner. A crash before CAS leaves an inactive immutable candidate. A
crash after CAS but before acknowledgement is resolved idempotently by the promotion event and head
fingerprint. Orphan candidates may be garbage-collected only after a retention window and reference
scan.

## Target correction, expiration, and deletion

The implemented store validates a single correction/tombstone lineage and excludes superseded or
tombstoned events from `active_events` and lexical search. It does not erase source payloads,
invalidate a loaded adapter, rebuild LoRA weights, or perform non-recall evaluation. Everything
else in this section is the target operational protocol.

### Correction

A correction appends a new event that names the superseded event, increments
`invalidation_epoch`, and updates retrieval first. From that commit onward, every memory version
from an older epoch is ineligible for activation. Retrieval serves the corrected value while a
mandatory fresh rebuild is queued. A candidate trained concurrently against the old epoch fails CAS.

Temporal facts retain both observation time and effective-validity time. Retrieval returns the fact
valid for the question's requested time rather than treating the newest row as universally true.

### Expiration

TTL expiry is represented by a deterministic projection transition or an explicit expiration event.
If expired content contributed to parameter memory, expiry increments `invalidation_epoch` and
requires a fresh rebuild. Secrets and short-lived operational facts SHOULD be marked
`retrieval_only` so expiry never requires neural unlearning.

### Deletion

Deletion follows this order:

1. authorize the request and append a tombstone naming all affected source and derived records;
2. increment `invalidation_epoch`, closing the stale memory gate immediately;
3. remove the records, embeddings, caches, and derived reflections from the retrieval projection;
4. destroy separately stored encrypted payloads when erasure policy requires it;
5. enqueue a mandatory fresh-from-base memory rebuild that excludes the tombstoned lineage;
6. run deletion-specific non-recall and extraction tests; and
7. promote only after the normal gates pass.

Until the rebuild is promoted, service remains retrieval-only for that scope. Rollback MUST NOT
reactivate any old version from a prior invalidation epoch. If no safe version exists, the correct
rollback target is null memory.

The target deletion guarantee is operational: molor removes derived records and the memory-adapter
contribution from service and verifies non-recall under its test suite. Even after implementation,
this will not prove that the frozen base never learned similar public information or that black-box
extraction is impossible. The current prototype guarantees only logical exclusion from its active
event reads; the original payload remains in the SQLite ledger.

## Evaluation matrix and provenance

Memory promotion requires a declared dense counterfactual grid rather than a list of favorable
candidate rollouts:

```text
prompt
  x forced capability route (every expert plus base/null)
  x forced memory arm (null, current when present, candidate)
  x retrieval arm (off, on at one exact projection cut)
```

The retrieval-off arm is an evaluation control for parametric memory depth, not a production mode
for corrections or deletions. The grid declares all prompts and arms before collection, uses the
same seed across arms for each prompt/replicate, and binds the capability bank, both memory-version
manifests, retrieval records/cut, generation manifest, cost normalizer, reward engine, and forced
execution trace. Missing or masked cells remain missing; they are never zero-filled. Target,
retention, locality, correction/deletion, security, and compound capability-memory prompt families
each receive source-held-out partitions.

This requires a future `MemoryEvaluationTable` because the implemented `RewardTable` has only a
prompt-by-capability-route grid. Its canonical export follows the same header/cells/footer and
content-fingerprint rules. Promotion reports cite the complete table fingerprint rather than
copying aggregate scores without their counterfactual evidence.

## Promotion gates

Thresholds live in an immutable `MemoryPromotionPolicyManifest`. Every gate below is required; a
masked or missing result blocks promotion.

### Identity and data gates

- Every artifact digest, base/format binding, source event, transform, selector, actuator, seed,
  verifier, reward contract, generation spec, and evaluation set is present and verified.
- The candidate's scope, source cut, parent, expected head, and invalidation epoch match.
- Every selected event is active, authorized, within retention policy, and free of unresolved
  correction, deletion, or provenance conflicts.
- No training, replay, or evaluation split crosses a source/problem-family boundary accidentally.
- Model-derived data retain source lineage and pass deterministic checks where available.

### Behavioral gates

- Target memories improve over the current version and retrieval-only control on held-out direct,
  paraphrased, and compositional prompts.
- All previously active memories meet a retention floor on a fixed replay suite.
- Unrelated prompts preserve frozen-base and standalone capability-expert behavior.
- Capability route decisions, oracle regret, and compound-task performance do not regress under the
  exact combined serving binding.
- Corrections prefer the new valid fact, temporal queries respect validity time, and tombstoned
  content fails direct, paraphrased, and extraction-oriented recall tests.
- The memory gate meets false-positive, false-negative, calibration, and null-route retention
  thresholds. Ambiguous or out-of-scope prompts abstain.
- Multi-hop tests verify consequences of a memory update rather than only literal recall.

### Interference and operational gates

- Delta, activation, and edited-layer norms remain within policy bounds; long-sequence replay shows
  no accelerating degradation or loss of plasticity.
- The candidate is tested alone and with every capability expert, plus deliberately conflicting
  capability/memory combinations.
- Results hold across the required seeds and untouched source-held-out families.
- Measured adapter bytes, load time, cache-hit rate, time to first token, p50/p95 latency, and
  throughput satisfy the serving budget.
- Artifact-load failure, stale epoch, authorization failure, and CAS races all exercise the
  retrieval-only fallback.

### Security and privacy gates

- Imported documents, tool output, and model-authored text cannot self-authorize consolidation.
- Poisoning and indirect-injection suites test both memory admission and later activation.
- Cross-scope prompts cannot retrieve, route to, infer, or extract another scope's memory.
- Sensitive strings and canaries do not appear outside their authorized retrieval path.
- Correction, deletion, gate-disable, rollback-to-null, and audit reconstruction are exercised
  end-to-end.

## Target rollback and retention

Rollback will be another CAS update of `ActiveMemoryHead` plus an immutable rollback event. The target
version must match the current invalidation epoch and pass artifact verification. Otherwise rollback
sets the head to null. Existing requests finish on their pinned generation; new requests see the new
head.

The implemented `MemoryHead` cannot yet express null at a positive generation and exposes only
forward promotion, so it does not implement rollback or rollback-to-null.

Old manifests, promotion reports, and minimal audit lineage are retained according to policy.
Adapter bytes and training payloads may be deleted only after proving that no active head,
in-flight binding, rollback window, or legal hold references them. Garbage collection never infers
liveness from queue state alone.

## Remaining implementation sequence

1. Extend the implemented `MemoryEvent` ledger with a scope-global sequence, invalidation epoch,
   transactional projection outbox, richer provenance/retention fields, and encrypted payload
   erasure support.
2. Replace the lexical control with a T2 projector and retrieval API supporting bitemporal validity,
   tombstones, ACLs, checkpoints, and lineage tests.
3. Add a trusted artifact loader around the implemented `MemoryVersionManifest`, model-stack
   identity, session pin, and factorized LoRA layer; then add the hard apply/null memory gate.
4. Bind the implemented serving pin into generation traces and a strict
   capability-by-memory-by-retrieval evaluation table before adding training.
5. Extend the implemented leased queue and CAS head rollover with a projection checkpoint,
   invalidation-epoch comparison, promotion ledger, fencing tokens, and crash-injection tests.
6. Build deterministic selection rules, fixed actuator recipes, fresh-from-base training, replay,
   and complete promotion reports using the existing verifier and reward semantics.
7. Add correction/deletion invalidation and mandatory rebuild tests.
8. Only then run shadow RL experiments for selection, actuation, and the memory gate.

## Primary research basis

The architecture follows evidence that direct parameter editing, retrieval, and routing have
different strengths rather than treating them as interchangeable. [WISE](https://arxiv.org/abs/2405.14768)
uses immutable main memory, editable side memory, routing, and parameter sharding, while
[MELO](https://arxiv.org/abs/2312.11795) indexes dynamically activated LoRA blocks.
[SERAC](https://proceedings.mlr.press/v162/mitchell22a.html) and
[GRACE](https://arxiv.org/abs/2211.11031) support explicit local edit memories that leave the base
unchanged.

The tiered external-memory design is informed by [MemGPT](https://arxiv.org/abs/2310.08560),
[Generative Agents](https://arxiv.org/abs/2304.03442), and the frozen-encoder/adaptive-reader split
in [LongMem](https://arxiv.org/abs/2306.07174). [Larimar](https://arxiv.org/abs/2403.11901)
demonstrates one-shot episodic updates and selective forgetting. [Titans](https://arxiv.org/abs/2501.00663),
[test-time training layers](https://arxiv.org/abs/2407.04620), and
[Nested Learning](https://arxiv.org/abs/2512.24695) motivate distinct update timescales, but do not
by themselves establish safe durable cross-user weight mutation.

The secondary selection/actuation split is directly tested within EVAF's loop-drift protocol and
evaluated model set. It reports separable selection quality and model-dependent write strength and
identifies stale-memory invalidation as unresolved; this is a scoped design hypothesis, not mature
deployment evidence:
[EVAF](https://arxiv.org/abs/2606.26806). [SEAL](https://arxiv.org/abs/2506.10943) shows that a model
can propose persistent self-edits, which motivates the proposal/verifier boundary rather than
authorizing self-written truth.

Interference controls draw on language-model continual low-rank subspaces in
[O-LoRA](https://arxiv.org/abs/2310.14152) and cross-domain computer-vision evidence from
[InfLoRA](https://arxiv.org/abs/2404.00228). Norm monitoring is a transferred diagnostic from
locate-then-edit experiments in [ENCORE](https://arxiv.org/abs/2502.01636), not a validated control
for memory LoRAs. Merge-risk evidence comes from
[TIES-Merging](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1644c9af28ab7916874f6fd6228a9bcf-Abstract-Conference.html)
and LoRA-specific [OSRM](https://aclanthology.org/2025.acl-long.1284/). We infer from these failure
modes that v0 should keep versions separate and prefer reproducible rebuilds over untested
arithmetic merging; the sources do not prove this policy is universally optimal.

Promotion includes multi-hop and security tests because editors can pass direct recall while
failing related reasoning in [MQuAKE](https://arxiv.org/abs/2305.14795) and
[ReCoE](https://arxiv.org/abs/2401.17585), and because long-term memories and knowledge bases are
demonstrated poisoning surfaces in
[AgentPoison](https://proceedings.neurips.cc/paper_files/paper/2024/hash/eb113910e9c3f6242541c1652e30dfd6-Abstract-Conference.html)
and [PoisonedRAG](https://www.usenix.org/conference/usenixsecurity25/presentation/zou-poisonedrag).
