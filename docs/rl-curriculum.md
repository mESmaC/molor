# Reinforcement-learning curriculum

Status: part of the planned experiment, gated behind reproducible supervised experts and validators.

## Why RL belongs here

There are two distinct policies in this system:

1. each LoRA expert changes the language model's behavior; and
2. the router chooses which behavior is active.

RL can optimize both from outcomes rather than imitation alone. They should not be optimized at the
same time initially. If expert behavior and routing both move during one update, the router's action
meanings become non-stationary and failures are difficult to attribute.

Persistent memory adds separate selection, actuation, retrieval/gating, and consolidation decisions.
Those policies are trained only after the deterministic memory pipeline works, and never share an
undifferentiated update with the capability router. Their exact rollout identities, rewards, and
promotion gates are specified in the [persistent-memory design](memory.md).

## Update sequence

| Phase | Trainable parameters | State | Action | Primary reward |
| --- | --- | --- | --- | --- |
| A: specialist RL | One LoRA expert | Prompt and generated prefix | Output tokens | Domain verifier |
| B: router full-information warmup | Prompt router | Frozen prompt representation | Expert or null ID | Precomputed reward for every route |
| C: prompt bandit | Prompt router | Frozen prompt representation | Top-1 or top-2 route | End-to-end result minus serving cost |
| D: token/block routing | Token/block router | Current hidden state and route history | Next expert ID(s) | Episode reward with routing credit |
| E: joint refinement | Alternating router and selected LoRA | Prompt, generation, route trace | Route and output tokens | Task, behavior, retention, and cost |

The frozen base remains frozen in every phase.

## Phase A: shape each specialist

Begin from that expert's supervised checkpoint. Generate multiple responses per prompt, score them
with deterministic validators where possible, and apply a group-relative or policy-gradient update
only to that LoRA. Keep an immutable supervised reference for KL anchoring and rollback.
The implemented status, masking, composition, and sandbox protocol are specified in
[Verifier and reward contracts](verifiers.md).

Initial reward contracts:

| Expert | Positive outcome reward | Anti-gaming checks |
| --- | --- | --- |
| Code/debug | Held-out tests compile and pass; requested API is obeyed | Hidden tests, mutation tests, timeout and sandbox checks |
| Math | Extracted final answer verifies; intermediate claims are consistent when checkable | Perturbed numbers, alternate wording, answer-format traps |
| Structured/tools/SQL | Schema-valid call and correct execution result | Invalid-tool decoys, SQL sandbox, argument-level validation |
| Critic/repair | Fault is localized and the repair passes the original failure | Clean inputs where the correct action is no change |

Use task-family reward normalization before mixing domains. Preserve general-instruction replay and
measure behavior against both the supervised expert and frozen base. A reward increase alone is not
a promotion criterion.

## Phase B: exploit the full expert reward table

Prompt routing is a one-step contextual decision. With a small bank, run the null route and every
expert on every router-training prompt. This yields a counterfactual reward vector instead of only
the reward for one sampled action.

Use that full table to train a calibrated policy before on-policy RL. It is lower variance and more
data-efficient than pretending the unobserved expert rewards are unavailable. Save the table with
the exact base, adapter, generation-config, verifier, reward, cost-normalizer, and dataset versions
that produced it. The implemented table embeds every one of those specs except dataset provenance;
a future DatasetManifest must add repository/revision/config/split/transform identity.

For independently trained frozen experts, hard top-1 has a preservation advantage: the selected
LoRA is applied with the unit scale it saw during training. The straight-through router uses a hard
one-hot forward decision and soft probabilities for gradients. For compound prompts, compare
normalized soft top-2 with fixed equal top-2. The equal variant prevents learned mixture weights from
collapsing to an effective top-1 and trains selection through a policy-gradient loss.

## Phase C: prompt-level policy optimization

Once the full-information router is reliable, sample routes on new prompts and score the resulting
generation. A starting objective is:

```text
R = normalized_task_success
    + behavior_reward
    + retention_reward
    - adapter_activation_cost
    - measured_cold_load_cost
    - latency_cost
```

Introduce cost terms only after a quality threshold is met; otherwise the cheapest policy can win by
overusing null or top-1. Use actual measured latency and bytes loaded rather than theoretical FLOPs.
Keep a KL or trust-region constraint to the supervised router, an entropy floor during exploration,
and per-domain utilization reports. Uniform routing is not the objective.

When a hard serving budget matters, prefer a target constraint over guessing one permanent scalar
cost. Update a dual variable to meet the desired mean active-adapter count or measured latency, and
optionally expose the requested budget as router input. Calibrate any cheap training proxy against
p50/p95 measurements from the actual kernel, cache, and hardware.

## Phase D: token-by-block policy optimization

Here the state includes the current hidden representation and recent routing history, and each block
can select a different whole-adapter ID. The terminal task reward must be credited across many route
decisions. Start with experts frozen and top-1 unit-scale selection; add top-2 only as an ablation.

Record the exact routing trace during rollout. When scoring a policy update, either replay that trace
or use routing-aware importance corrections. A changed router can otherwise send the same token
through different parameters during generation and learning, creating an MoE-specific off-policy
shift. Monitor route-change rate, importance ratios, clipping rate, expert starvation, and reward
collapse together.

## Phase E: controlled joint behavioral tuning

Only after router-only RL works:

1. freeze experts and take a router update;
2. freeze the router and update only the selected expert on its domain replay;
3. evaluate the complete expert-by-domain matrix and general retention;
4. roll back if specialization or oracle quality falls.

This alternating schedule keeps action meanings more stable than simultaneous updates. Anchor each
expert to its independent checkpoint with KL on generated behavior and, if useful, a direct parameter
distance penalty. Never discard the independent expert artifacts.

## Behavioral expert factorization

A later experiment can separate capability from behavior instead of training every Cartesian
combination:

```text
layer output = frozen base
             + routed capability-adapter delta
             + routed behavior-adapter delta
```

Capability experts might cover math, code, and tools. RL-shaped behavior experts might explore,
verify, answer concisely, abstain when evidence is insufficient, or prefer tool use. Selecting one
from each bank would allow combinations such as `math + verifier` and `code + concise` without
training a distinct adapter for every pairing. This is deliberately after the single-bank baseline.

## Memory-policy RL

Memory experiments hold the capability route fixed while varying one memory decision at a time:

1. selection chooses `retrieval_only`, `defer`, `quarantine`, or `consolidate` for authoritative
   events;
2. actuation chooses a bounded LoRA write recipe for an approved set;
3. the prompt-level memory gate chooses apply or null for one pinned memory version; and
4. a later scheduler may choose batching priority and training budget, never memory truth.

Start with deterministic selection and a fixed actuator grid. Build a counterfactual
`prompt x memory-plan` table with the capability route pinned, just as capability-router training
uses `prompt x capability-route` with memory pinned. Reward cross-session utility, prior-memory and
base retention, correction/deletion compliance, and capability composition; treat privacy,
cross-scope leakage, and failed deletion checks as hard gates rather than tradeable penalties.

Do not update a candidate memory LoRA, its memory gate, and the capability router in one optimizer
step. First establish each main effect, then evaluate a small factorial interaction set. Live memory
versions remain immutable; RL can only propose a candidate for the normal evaluation and CAS
promotion path.

## Promotion and rollback gates

- Evaluate proxy reward and real task success separately on untouched data.
- Require gains across multiple seeds and source-held-out prompt families.
- Track KL from the supervised expert, response length, formatting, refusal, and exploit signatures.
- Keep clean negative cases so critic, tool, and abstention behaviors do not fire indiscriminately.
- Compare hard top-1, soft top-2, oracle, random, and null routing at every router milestone.
- Reject a checkpoint if any expert loses its standalone advantage or if router oracle regret grows.
- Version reward code, validators, sandboxes, generation settings, route traces, and all model hashes.

## Relevant evidence

- [Hard-Routed MoR-LoRA](https://arxiv.org/abs/2606.31413) independently trains reasoning LoRAs with
  reinforcement learning from verifiable feedback, then freezes them and uses unit-scale hard top-1
  routing with a straight-through estimator.
- [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) reports policy-gradient experiments
  in which low-rank adaptation matched full fine-tuning within their evaluated RL settings, supporting
  LoRA as the behavior-update mechanism rather than unfreezing the base.
- [Towards Stable and Effective Reinforcement Learning for Mixture-of-Experts](https://aclanthology.org/2026.acl-long.1165/)
  identifies router shift as an MoE-specific source of off-policy instability and motivates explicit
  routing-aware diagnostics or corrections.
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) introduced GRPO for mathematical reasoning and is a
  useful starting reference for grouped, verifier-driven specialist updates.
- [ReMix](https://arxiv.org/abs/2603.10160) treats the LoRA router as an RL policy, uses fixed equal
  weights for active experts, and derives an RLOO routing estimator after observing learned-weight
  collapse.
