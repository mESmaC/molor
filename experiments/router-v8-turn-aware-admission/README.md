# Router v8: turn-aware Taskmaster admission (quarantined preregistration)

Status: **frozen representation scaffold; training, blind evaluation, registration, and live
backend integration are not authorized.**

This experiment tests one narrow hypothesis: v7's current-user-only masked mean loses the
dialogue state needed to admit the Taskmaster coffee expert. Router v8 changes only the binary
admission representation. A gate-negative request must return the exact hash-pinned v4
`AutoRouteDecision` object unchanged. A gate-positive request selects only
`taskmaster-coffee-api-v2-full-dialog-r16` and must not consult v4.

This directory contains no dataset, checkpoint, prediction, blind-probe descriptor, or evaluation
result. The checked-in JSON contract binds this README, the representation module, and its focused
tests by SHA-256. A separate artifact-complete policy would be required before any training or
one-shot evaluation.

## Frozen comparison

Both arms use the same frozen Qwen input-embedding table, binary route axis, threshold `0.5`,
training rows, optimizer schedule, and candidate seeds `83`, `8419`, and `27183`. Every seed must
freshly pass all preblind gates. Among passing candidates, selection is the lowest fresh original
validation cross-entropy, then the lowest numeric seed. No blind result may influence selection.

Arm A is the current control: tokenize only current user content to 256 tokens with right
truncation and special tokens; masked-mean pool 2048-wide input embeddings; then apply
`LayerNorm(2048) -> Linear(2048, 2)`.

Arm B accepts structured `system`, `user`, `assistant`, and `tool` turns. It retains at most the
newest 12 turns and at most 512 encoded tokens. The final turn must be the current user. The
current user is allocated first and must fit in full; an over-budget current user is rejected.
Earlier turns are allocated newest-first. The first partially fitting earlier turn contributes its
token suffix, after which allocation stops. Retained turns are emitted in chronological order with
explicit role IDs and integer recency IDs (`0` is current).

Each Arm-B turn is masked-mean pooled from the same frozen 2048-wide input embeddings. The trainable
representation is exactly:

1. content projection `Linear(2048, 224)`;
2. a 16-wide role embedding and 16-wide recency embedding;
3. concatenated 256-wide per-turn features;
4. one bidirectional GRU layer with 128 hidden units per direction;
5. `LayerNorm(256) -> Dropout(0.1) -> Linear(256, 1)` and sigmoid Taskmaster score.

The implementation in
[`src/molor/experimental/router_v8_turn_aware.py`](../../src/molor/experimental/router_v8_turn_aware.py)
is representation-only. It does not load the base model, adapters, datasets, checkpoints, or any
evaluation artifact, and it is not exported into a runtime/backend path.

## Preblind gates

All three seeds must independently satisfy the following before candidate selection:

- Frozen original validation: accuracy at least `0.995`, Taskmaster precision and recall each at
  least `0.99`, and zero Taskmaster false positives among 650 negatives.
- A separately built, non-blind 240-row turn-aware development challenge: 60 Taskmaster positives
  and 180 legacy negatives; at least 50/60 Taskmaster recall, no more than two false Taskmaster
  admissions, and at least `0.95` overall accuracy.
- Paired conversation consistency at least `0.95`.
- Arm B improves Taskmaster recall by at least 30 percentage points over Arm A while adding no more
  than one false Taskmaster admission.
- Every gate-negative result has exact Python object identity with the pinned v4 delegate result.
- Serialized candidate is no more than 8 MiB; CPU p95 admission latency is no more than 25 ms and
  no more than 2x Arm A under a later frozen benchmark harness.

The development challenge is intentionally absent from this scaffold. Its builder, source
identities, split procedure, duplicate checks, and row-level provenance must be frozen and audited
before training can be authorized. It must not contain or derive from Taskmaster test rows, sealed
probe content, predictions, per-example evaluation output, or a one-shot report.

## Future one-shot promotion guards

Only an independently authored and sealed future policy may bind an 84-row blind probe and enable a
single evaluation. The already frozen promotion thresholds are:

- at least `0.75` overall exact-or-acceptable and at least eight more exact-or-acceptable outcomes
  than v4;
- at least 10/12 Taskmaster outcomes exact-or-acceptable;
- no more than one false Taskmaster admission among 72 negatives;
- no more than one exact and one acceptable regression in each legacy and mixed slice;
- one attempt, no post-probe tuning or retry, no copy/register/alias/default/backend side effect.

These numbers preregister the decision rule; they do not authorize creating or reading a probe.

## Identity and fail-closed rules

Canonical metadata uses UTF-8 JSON sorted by key with compact separators and no NaN, then SHA-256.
Rendered conversations include exact token IDs, roles, recency, source indices, truncation flags,
and the rendering-policy fingerprint in their identity. The contract loader rejects duplicate JSON
keys, semantic drift, authorization changes, path escape, symlinks, byte/hash/size drift, and files
that change during hashing.

Synthetic representation tests may run without model inference:

```powershell
python -m pytest -q tests/test_router_v8_turn_aware.py
python -m ruff check src/molor/experimental/router_v8_turn_aware.py tests/test_router_v8_turn_aware.py
```

Do not use this scaffold to train or evaluate until a separate, independently audited policy pins
the structured dataset builder and train/validation artifacts, frozen base/tokenizer sources, v4
delegate, Taskmaster adapter, candidate destinations, launcher, evaluator, and resource benchmark.
