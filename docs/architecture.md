# Mixture of LoRA adapter experts: architecture v0

Status: accepted as the first experiment; exact backbone and expert datasets remain configurable.

## Goal

Build a frozen language model that can retain a growing library of independently trained LoRA
specialists and select or compose a small number of them for each prompt. The experiment succeeds
only if routing recovers most of an oracle selector's gain, preserves the unadapted model on general
prompts, and improves genuinely compound tasks over the best single adapter.

## First decision: use an existing post-trained base

The first run should use a strong instruction/post-trained checkpoint. Training a new base would
combine two research questions and make a routing failure hard to diagnose. Starting from a raw
pretraining checkpoint also makes every expert spend capacity relearning instruction following.

For a local smoke test, `Qwen/Qwen3-1.7B` is a practical starting point. The same experiment should
then move to a stronger checkpoint once the mechanics are correct. If we later create our own base,
the order should be pretraining or continued pretraining, shared post-training/alignment, a frozen
versioned checkpoint, and only then specialist adapter training.

An adapter is valid only for its exact base contract:

- model repository and immutable revision/hash;
- tokenizer, special tokens, and chat template;
- target-module names and tensor shapes;
- LoRA rank, scaling convention, and adapter dtype;
- any quantization used while training the frozen base.

Changing the base after specialists are trained creates a new adapter ecosystem. We should not
silently carry adapters across that boundary.

## Computation

For adapted linear layer `l`, base projection `W_l`, and selected whole-adapter IDs `S`:

```text
y_l = W_l x_l + sum[p_e * (alpha / rank) * B_(l,e) A_(l,e) x_l for e in S]
```

The base matrix is frozen. Each expert owns an `A` and `B` pair at every targeted layer. Expert IDs
are stable across layers, and the v0 router supplies one weight vector for the whole prompt.

We sum complete deltas `B_e A_e x`. We must not average `A` factors and `B` factors separately,
because multiplying those averages creates cross-expert terms that were never trained.

There is also a null/base route whose delta is exactly zero. It lets general prompts retain the base
behavior instead of forcing an irrelevant specialist.

## Persistent memory plane

Persistent memory is a second adaptation axis, not another capability expert in the same router:

```text
y_l = W_l x_l
      + sum[p_e * DeltaW_capability(l, e) x_l]
      + g_memory * DeltaW_memory(l, pinned_version) x_l
```

The capability route and memory gate are selected, normalized, logged, and trained independently.
Memory v0 permits one scope-pinned memory adapter plus an explicit memory-null route; retrieval is a
separate source-backed input channel and remains available when that adapter is absent or disabled.
The base is still frozen, and memory versions are complete immutable adapters rather than an
ever-growing chain of deltas.

This gives effectively unbounded *retained history*, not an unbounded attention operation. Each
forward pass still has a finite context: older turns are durably sealed before the context
high-water mark, made retrieval-visible, and queued for asynchronous LoRA consolidation. The raw
event ledger remains authoritative because adapter weights are lossy and cannot provide exact
correction or deletion semantics.

Requests pin one memory version for their entire prefill and decode. A worker may train and evaluate
a candidate concurrently, but it becomes live only through compare-and-swap publication of a new
memory head; an in-flight request never hot-swaps weights. See the full
[persistent-memory specification](memory.md).

## Router v0

The first integration/control router is deliberately prompt-level:

```text
prompt token embeddings
        -> masked mean pool over a bounded prefix
        -> LayerNorm -> small MLP -> expert logits
        -> top-1 or top-2 whole-adapter selection
```

The route is cached for prefill and generation. This is cheaper to evaluate, keeps batches groupable
by selected adapters, and permits selected adapter weights to be prefetched before the transformer
runs. It also gives a clean baseline against later token-wise routing.

The current `TopKRouter` primitive mean-pools every hidden state supplied to it. The Transformers
integration must enforce the bounded prefix and cache/reuse the resulting `RouterOutput` during
generation; those serving behaviors are design requirements, not yet implemented by the primitive.

Top-1 routing is trained as classification/distillation and used as a hard choice at inference.
Top-2 routing compares normalized learned weights with fixed equal weights. The fixed-weight variant
treats expert selection as the RL action instead of letting one learned coefficient quietly dominate
the other. We begin router training with dense probabilities or supervised targets because a
normalized hard top-1 gate has no useful task-loss gradient through its selected weight. A
straight-through hard top-1 path is also a required comparison: it applies the selected frozen LoRA
at its original unit scale in the forward pass while using soft router probabilities for the
backward pass.

The PyTorch primitive also supports token-level routing so the actual MoE experiment does not require
a rewrite. After the control works, the first token-level trial will use one gate at a transformer
block boundary and share the selected whole-adapter IDs across every adapted projection in that
block. That preserves a coherent adapter identity while still allowing routes to change through the
network and over a generated sequence. Separate gates at every linear projection remain an
ablation. A cheaper intermediate experiment can restrict dynamic routing to upper transformer
layers and initialize it from the prompt router.

Prompt and token routing imply different storage behavior. Prompt routing can select and prefetch a
small adapter set before prefill. Token routing cannot put an NVMe read on every routing decision;
its active expert bank must be resident in the serving hot tier and its selected tokens grouped into
batched matrix multiplications.

## Initial specialists

Start with four capabilities whose outcomes can be checked automatically:

1. code generation and debugging;
2. mathematical problem solving;
3. structured output, tool calling, and executable SQL;
4. critique, verification, and repair.

The critic is intentionally cross-cutting. It creates meaningful code-plus-critic and
math-plus-critic composition tests. These names are a proposed first taxonomy, not a permanent API.

All experts in one bank initially use identical target modules, rank, alpha, dropout, chat format,
and training budget. A reasonable first recipe is rank 16, alpha 32, dropout 0.05, targeting the
attention projections and MLP projections. Equal structure makes exact vectorized composition and
fair baselines straightforward.

Each specialist dataset should include a small amount of shared, high-quality general-instruction
replay. Splits must be by source or problem family rather than random rows. Unit tests, solvers,
schema validators, and execution should produce labels whenever possible.

## Training curriculum

### Stage 0: establish baselines

Evaluate the frozen base and one conventional multitask LoRA before training a router. Freeze the
task taxonomy, source-held-out splits, and automatic validators.

### Stage 1: train experts independently

For each expert, freeze the base and all other adapters. Optimize only that expert's parameters.
Evaluate every expert on every domain, not just its own. The resulting expert-by-domain score matrix
must show real specialization rather than one universally dominant adapter.

Each expert first gets a reproducible supervised checkpoint and can then receive an isolated RL
phase using domain-verifiable rewards. The base and every other expert remain frozen. The detailed
parameter, reward, and rollback contracts are in the [RL curriculum](rl-curriculum.md).

### Stage 2: construct routing targets

Run every frozen expert plus the null route on each router-training example. Derive soft targets from
actual held-out loss or task reward. Dataset names can warm-start the router, but they are not ground
truth for which expert works best.

The implemented RewardTable declares the prompt axis and ordered bank routes before collection, then
requires their exact Cartesian product. Its header embeds the immutable bank, reward-engine,
generation, and cost-normalizer manifests. Every cell carries a generation-time forced-route record,
matched per-row seed, cost-normalization record, and strictly validated reward report. Oracle labels,
soft targets, and regret are emitted only for complete scorable rows. See
[the provenance guide](provenance.md).

### Stage 3: train the router

Freeze the base and adapters. Train the router to match the measured expert distribution. Include
general prompts that should choose null, hard negatives, paraphrases that remove dataset-specific
cues, and compound prompts.

Track calibration and utilization. A small load-balancing term can prevent accidental starvation,
but uniform utilization is not itself a goal when the natural task mix is skewed.

After supervised routing is stable, treat prompt routing as a contextual-bandit policy: prompt state
in, expert choice out, verified task reward minus explicit serving cost. Initially score all routes
offline so the router receives full-information targets; use sampled on-policy updates only after
that control is strong.

### Stage 4: calibrate composition

Train and evaluate top-2 selection on compound tasks. Keep the top-1 path as a strong fallback
because independently trained deltas can interfere even when both experts are relevant. Compare
normalized soft top-2, fixed-equal top-2, and hard unit-scale top-1 rather than assuming mixture is
always beneficial. RL rewards compound-task success and charges explicitly for activating or
cold-loading a second adapter.

### Stage 5: optional joint refinement

Only after stable routing, optionally alternate router-only RL steps with selected-adapter RL steps
at a much lower learning rate. Do not initially update router and experts in the same optimizer step:
that creates a moving policy and moving action meanings simultaneously. Independent checkpoints
remain the rollback point. Joint training is not part of the first causal test because it can
collapse expert identities.

## Required comparisons

- frozen base;
- every individual specialist and an oracle per-example specialist;
- learned top-1 and learned top-2 router;
- random router and naive all-expert blend;
- one multitask LoRA with the same active rank/compute;
- one multitask LoRA with approximately the same total adapter parameters.

The central routing metric is oracle regret: the gap between the best available expert for an
example and the routed result. Also record task success, top-1 accuracy, top-2 oracle recall,
calibration, expert utilization, null-route retention, compound-task gain, tokens per second, peak
memory, adapter bytes loaded, and adapter cache hit rate.

## Promotion gates

Do not move to token-level routing until all of these are true:

- each specialist materially beats the frozen base in its intended domain;
- specialization survives source-held-out and paraphrased evaluation;
- the learned router captures most of the oracle selector's improvement;
- top-2 beats best top-1 on a deliberately compound test set;
- the null route prevents measurable general-capability regression;
- routing and adapter loading overhead are recorded, not hidden.

## Known failure modes

- Router learns dataset keywords instead of capabilities: use source-held-out data, paraphrases, and
  adversarial hard negatives.
- One expert wins everywhere: fix taxonomy/data before adding routing complexity.
- Top-2 deltas destructively interfere: preserve top-1, log delta norms, and learn calibration.
- Adapter recipes differ: reject mixed manifests at bank construction and reject mismatched bound
  checkpoints/runtime routes. Base artifacts now have verified private staging; capability and
  memory adapter artifacts still require an equivalent typed loader before real serving.
- General prompts are forced through a specialist: retain the explicit null route.
- Joint training erases specialist identity: stage training and keep immutable expert checkpoints.
- Reward improves while actual behavior regresses: version rewards, retain untouched validators,
  inspect rollouts, and gate promotion on both proxy reward and task outcomes.
- Router changes between rollout and policy update: record/replay route traces and monitor
  router-shift statistics rather than treating the model like a dense policy.
- Prompt routing misses mid-generation task changes: test token/layer routing only after v0 works.
- Dynamic selection fragments batches: group requests by route and cache/prefetch popular adapters.
- An adapter is added, replaced, or retrained: re-score it and recalibrate/version the router because
  the changed deltas also change downstream hidden states seen by later gates.

## Research lineage

This design starts from the original [LoRA formulation](https://arxiv.org/abs/2106.09685). The
closest direct precedent is [MeteoRA](https://arxiv.org/abs/2405.13053), which freezes separately
trained adapters and learns token-level gates; its composite-task results make top-2 a particularly
important comparison. Related composition systems include
[X-LoRA](https://arxiv.org/abs/2402.07148) and
[Mixture of LoRA Experts](https://arxiv.org/abs/2404.13628).
[MiLoRA](https://arxiv.org/abs/2410.18035) motivates the lower-latency prompt-routing control, while
[MixLoRA](https://arxiv.org/abs/2404.15159) is a useful comparison for jointly trained token-level
LoRA experts. Architecture v0 uses whole-adapter prompt routing as the control so that specialization,
composition, and systems behavior can be measured separately before the token-by-block experiment.
The RL path is especially close to
[Hard-Routed MoR-LoRA](https://arxiv.org/abs/2606.31413), which independently trains reasoning LoRAs
with verifiable rewards and preserves them with unit-scale hard routing. For later joint MoE policy
updates, [router-shift analysis](https://aclanthology.org/2026.acl-long.1165/) motivates explicit
routing-aware off-policy diagnostics.
[ReMix](https://arxiv.org/abs/2603.10160) provides the direct fixed-equal top-k and RLOO routing
comparison, motivated by collapse of learned mixture coefficients.
