# Verifier and reward contracts

Status: implemented core plus provenance-checked prompt-by-route tables; executable code/SQL
sandboxes and append-only rollout logging remain separate integration milestones.

## Purpose

The reward layer converts one prompt/completion pair into a replayable report. It deliberately has
no dependency on Transformers, a model, or router tensors. Its responsibilities are:

- run deterministic, versioned checks;
- retain every raw component instead of only a weighted scalar;
- distinguish model failure from invalid gold data and verifier/infrastructure failure;
- apply explicit quality weights, hard gates, and measured cost penalties;
- fingerprint the sample, reward contract, and verifier configuration;
- identify rollout groups that have no relative-policy learning signal.

## Minimal example

```python
from molor.rewards import (
    CostTerm,
    EvaluationSample,
    NumericAnswerVerifier,
    RegexVerifier,
    RewardContract,
    RewardEngine,
    RewardTerm,
)

contract = RewardContract(
    name="math-v1",
    version="1",
    terms=(
        RewardTerm("answer", weight=0.95),
        RewardTerm("format", weight=0.05, gate=True),
    ),
    costs=(CostTerm("normalized_latency", weight=0.02),),
)

engine = RewardEngine(
    contract,
    [
        NumericAnswerVerifier(name="answer", extraction="final"),
        RegexVerifier(
            r"(?s).*<answer>.*</answer>\s*$",
            name="format",
        ),
    ],
)

sample = EvaluationSample(
    sample_id="math/test/0001",
    prompt="What is 6 * 7?",
    completion="<answer>42</answer>",
    reference=42,
)

report = engine.evaluate(sample, normalized_costs={"normalized_latency": 0.1})
```

`report.to_dict()` is JSON-serializable and contains the immutable engine manifest, raw verifier
results, typed statuses, timings, component reward, cost penalty, gate failures, and content
fingerprints. Public report construction replays the embedded contract and rejects contradictory
mask, gate, cost, quality, or final-reward fields.

Verifier manifests include a SHA-256 digest of their defining Python module in addition to the
declared semantic version and configuration. Editing verifier code therefore invalidates the reward
contract fingerprint even if somebody forgets to bump a version string. External workers add an
operator-supplied `runner_fingerprint`; the operator remains responsible for resolving and verifying
that identity against the actual runner artifact.

## Reward semantics

Quality-term weights must sum to one. Every contract-referenced verifier runs exactly once, and its
score is kept in the report. Extra verifier objects are rejected. A scorable result has a finite
score in `[0, 1]`.

The statuses `verifier_error`, `infra_error`, and `invalid_reference` are deliberately unscorable.
If any configured quality result is unscorable, the report is masked and both `reward` and
`quality_reward` are `None`. They must be retried, repaired, or quarantined—not inserted into an RL
batch as zero reward.

For scorable reports:

```text
quality_reward = sum(term_weight * verifier_score)
reward = clip(quality_reward - measured_cost_penalty)
```

If a scorable hard gate fails, the fixed contract `failure_reward` takes precedence. Required cost
measurements cannot silently default to zero; a missing required measurement masks the report. v0
costs are normalized values in `[0, 1]`. The router bridge binds raw measurements and the normalized
values used for scoring in one `CostNormalizationRecord`.

## Built-in verifiers

- `ExactMatchVerifier` is strict by default. Stripping, case folding, whitespace collapse, and
  Unicode normalization are opt-in and fingerprinted.
- `NumericAnswerVerifier` defaults to a whole-output numeric value. `extraction="final"` first looks
  for one boxed, `<answer>`, or `Answer:` value; when no marker exists, it accepts exactly one numeric
  literal anywhere in the completion. Multiple candidates are ambiguous. It never evaluates
  symbolic expressions.
- `JsonSubsetVerifier` rejects duplicate keys, `NaN`/`Infinity`, oversized/deep payloads, and strict
  scalar type mismatches such as `true` versus `1`. Markdown fences are opt-in.
- `RegexVerifier` provides a separate, low-weight format signal from an operator-defined pattern.

Complex extraction should become its own versioned component. Full JSON Schema validation should
use a real validator library with remote reference resolution disabled rather than expanding the
subset verifier into an incomplete schema engine.

## External sandbox protocol

`ExternalCommandVerifier` sends a versioned JSON request on stdin to a fixed operator-configured
command. Model output is data in the request; it is never interpolated into a command or shell.

```json
{
  "protocol": "molor.verifier.v1",
  "request_id": "case-id",
  "sample": {
    "sample_id": "case-id",
    "prompt": "...",
    "completion": "...",
    "reference": {},
    "metadata": {}
  }
}
```

The worker returns the same protocol and request ID plus `status`, `score`, `passed`, bounded
`details`, and optional `error`. An operator-supplied runner identity becomes part of the
reward-contract fingerprint; the current command adapter does not verify an image digest itself.

The command adapter is not a sandbox. A code/SQL worker must independently enforce an ephemeral
container or microVM, no credentials, no network, a non-root identity, read-only test assets,
disposable writable storage, CPU/RAM/PID/file/output/wall-time limits, and whole-process-tree cleanup.
A subprocess timeout alone is not a security boundary.

## Relative-policy groups

`summarize_reward_group()` reports scorable/masked counts, mean, population standard deviation, and
whether a completion or route group has usable relative-policy signal. Equal-reward groups are
marked `zero_variance` so downstream training can exclude them from an RLOO/GRPO denominator;
masked results are never converted to zero. Easy and hard zero-variance prompts remain useful for
evaluation and curricula.

## Identity rules for the router bridge

The implemented reward-table bridge persists stable route IDs as well as bank-local route indices.
Its header embeds the bank, reward-engine, generation, and cost-normalizer manifests and their
fingerprints. Each completion carries a forced-route record and cost-normalization record.
Positional indices alone are unsafe because inserting, reordering, or retraining an adapter could
silently relabel historical rewards. The null route has the stable ID base/null by default and
remains structurally last in the v1 bank.

The sample prompt fingerprint excludes completion and rollout ID, while the semantic report
fingerprint covers all reward-affecting results and normalized costs. See
[the provenance guide](provenance.md) for the manifest hierarchy, runtime binding, table density,
and oracle rules.

## Next implementation boundaries

1. append-only rollout/RLOO event logs;
2. an isolated executable verifier service;
3. the Transformers training bridge;
4. router training datasets exported from complete reward tables.
