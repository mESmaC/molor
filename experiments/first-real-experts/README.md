# First real experts

This experiment keeps `~/Documents/datasets` read-only. Builders write normalized
chat shards, provenance sidecars, hashes, and audit reports under `data/`. The current rank-16
corpus adapters and their summaries live under `checkpoints/qwen3-1.7b-corpus-experts/`; the
earlier rank-8 experiments remain under `checkpoints/qwen3-1.7b-real-experts/`.

## Current rank-16 corpus experts

All four use the same frozen local Qwen3-1.7B base and all seven attention/MLP projections.

| Adapter | Data split | Held-out result | Honest status |
|---|---:|---|---|
| `embedded-knowledge-v1` | 6,000 train / 512 file-held-out | NLL 3.6013 -> 1.4356; on 64 generations, character similarity was .1003 base / .1175 earlier rank-8 adapter / .3070 new expert | Clear corpus-use improvement, but generated code and technical claims are not mechanically verified. A separate 35-row seen-recall probe exists and is explicitly not held-out. |
| `ocb-can-verified-v1` | 12,320 train / 1,024 message-group-held-out | NLL 4.0179 -> .07419; on 128 generations, structured-field accuracy rose .20295 -> .78771 and all-fields exact rose 0 -> 36 | Raw-value accuracy .359 and physical-value accuracy .344 are still weak, so deterministic decoding remains necessary. The completed run retains its warm-restart provenance. |
| `software-architecture-v1` | 12,000 train / 1,000 validation; zero normalized-prompt overlap | NLL 2.4796 -> 1.2808; on 64 generations, term F1 rose .2661 -> .3657 and quality-signal recall .4243 -> .5776 | Concept recall was effectively flat (.4177 -> .4142). The generation slice spans 14/15 topic families; these lexical/reference metrics do not prove semantic correctness. A blind review pack is available. |
| `verified-embedded-code-v1` | 920 compiler-verified training rows | Fresh executable sibling score 66.8 base -> 152.2; sealed executable score 303.4 prior Auto -> 502.6 forced | Strongest current executable-code adapter, but hidden-test failures remain; compiler/verifier feedback is still required. |

The architecture blind pack is `runs/evaluation/software-architecture-v1-blind-28.jsonl`, with its
separate key beside it. It is intended for blinded human/teacher comparison rather than being
counted as another automatic win.

## Earlier rank-8 experiments

| Adapter | Data | Held-out result | Honest status |
|---|---:|---|---|
| `financial-analysis-verified` | 2,026 rows | NLL 5.225 -> 0.306; 42.2% numeric+unit accuracy on 128 generated calculations | Clean, group-separated finance-format expert; still needs the included verifier or a calculator tool |
| `local-embedded-code` | 2,200 rows / 1,568 files | NLL 1.865 -> 1.001; character similarity 0.256 -> 0.314 on 50 file-held-out continuations | Useful local C/C++/asm/linker/build completion and style adapter; not a verified debugger |
| `terminal-cli` | 540 traces | NLL 1.928 -> 1.002 on 60 task-held-out traces | Learns inspect/run/test behavior; requires a real command-result loop before autonomous use |
| `grounded-can` | 226 rows | 21/26 definition audits fully correct, but only 14/26 payload raw values correct | Experimental definition explainer; deterministic DBC decoding must remain a tool |

The quarantined rank-16 `financial-analysis-verified-r16-v1` follow-up is also a negative result.
It improved held-out NLL to .2971, but on 149 eligible calculation rows reached only 38.3%
numeric+unit accuracy, 38.9% verifier pass, and 61.1% calculation consistency, below the stronger
rank-8 finance comparator. It is not part of the resident rack or router.

The rank-16 `taskmaster-coffee-api-v1` candidate is likewise quarantined. It was trained for one
epoch on 1,176 group-separated positive API calls from Taskmaster-4, with 181 validation and 145
test rows. The held-out candidate produced parseable, schema-valid JSON on all 145 cases, versus
77.9% parseable and 20.7% schema-valid for base. It reached 77.9% API-name exactness, 47.6%
argument exactness, and 45.5% full-object exactness, missing the run-declared 85% / 70% / 60%
promotion floors. The checkpoint is not in the resident rack or router, and explicit quarantine
directories are excluded from normal desktop adapter discovery. Exact-path evaluation remains
possible. The exposed v1 test is closed to further candidate selection; a successor needs a fresh
group-disjoint holdout. Taskmaster-4 is licensed CC-BY-4.0 and attributed to Bill Byrne and
Saravanan Ganesh of Google LLC; the pinned source manifest and policy are stored beside the recipe.

The `taskmaster-coffee-api-v2-full-dialog-r16` successor completed its single sealed rank-16 run on
11,626 full-dialog training examples. Its builder emitted 14,436 conflict-filtered,
component-disjoint examples: 11,626 train, 1,380 validation, and 1,430 test. It consumes zero reward
rows, has zero overlap with all 1,623 accepted v1 source conversations, quarantines all 120 dialogs
containing malformed embedded JSON, and derives the prompt tool catalog from train only.

On the fixed 1,430-example comparison, the candidate reached 100% parseable/schema-valid output,
99.44% API-name exactness, and 87.06% argument/semantic exactness. Frozen base reached 23.22% /
14.06% / 13.43% on API name / arguments / semantics; quarantined v1 reached 29.79% / 34.13% /
15.31%. Every declared minimum-rate, base-improvement, and v1-non-regression gate passed. The
result does not authorize router or resident-rack registration, so the adapter remains an explicit
candidate until a separate routing decision is evaluated.

That routing decision is now complete and negative. Router v6 preserved all 4,680/650/650 promoted
v4 rows and added 936/130/130 component-disjoint Taskmaster positives. Three frozen CPU seeds all
passed fresh preblind validation; seed 8419 was selected by the lowest recomputed validation loss
(`0.0617127`, `99.7436%` accuracy). On the preregistered 84-prompt one-shot probe, however, it
reached only 47/84 exact and 51/84 acceptable routes, improvements of 2 and 3 over v4, and routed
0/12 Taskmaster prompts correctly. It produced zero Taskmaster false positives and passed every
legacy-category protection gate, but failed both overall-accuracy floors, both improvement floors,
and the Taskmaster-recall floor. Registration was not changed; the v4 router remains live, and the
Taskmaster expert remains quarantined. The preblind receipt and one-shot report are preserved under
`runs/router-v6/` and `runs/evaluation/` respectively. Their prompts and per-example outputs must
not be reused for training, tuning, threshold selection, or seed selection.

This result is recovery-qualified rather than a pristine one-shot. The first evaluator wrote its
pre-access receipt and exposed the test bytes, then failed before model loading or generation
because its context validator rejected builder-defined prior API call/response turns. The defect
was reproduced and repaired using train/validation only; all 11,626 train and 1,380 validation
contexts then passed, and every model, adapter, generation, metric, threshold, and artifact identity
was held fixed for one documented recovery. The original receipt, incident, recovery receipt,
4,290 content-addressed predictions, and final report are preserved under `runs/evaluation/`. The
exposed v2 test must never be used for tuning, checkpoint selection, threshold adjustment, or
training.

```powershell
# Safe identity verification; does not load the model or open the v2 test.
.\.venv\Scripts\python.exe experiments/first-real-experts/run_taskmaster_coffee_api_v2_training.py --preflight-only

# The single fixed quarantined run. Do not rerun after an emitted candidate or v2 recovery checkpoint.
.\.venv\Scripts\python.exe experiments/first-real-experts/run_taskmaster_coffee_api_v2_training.py

# Historical first-access command. The v2 test is now exposed and this must not be rerun.
.\.venv\Scripts\python.exe experiments/first-real-experts/evaluate_taskmaster_coffee_api_v2.py --acknowledge-one-shot
```

`grounded-can-curriculum` is preserved as a negative result. Expanding training to 484 legal
payload variants improved template/mechanics fields but reduced raw decode accuracy to 13/26. The
older `trusted-can` v2 is an invalid lookup/memorization experiment and is marked do-not-use beside
its weights. `financial-analysis` v1 is retained for comparison; it slightly beat the filtered
retrain on one 128-example generation slice, despite containing labels rejected by the verifier.

Every adapter directory contains `training_summary.json` and `evaluation_summary.json`. Full
generation reports are in `runs/evaluation/`.

## Dataset review decisions

| Prepared source | Decision | Reason |
|---|---|---|
| local `code-corpus` | trained, then expanded | 3,841 deduplicated local files first supported the narrow rank-8 adapter; the current file-separated embedded corpus expert uses the larger 6,000-row build |
| Complete FABLE traces | filtered to terminal | 540 permissively licensed, task-separated terminal traces fit the current SFT objective |
| OCB CAN database | rebuilt and trained as `ocb-can-verified-v1` | Original QA rows contained invalid IDs, untrusted records, leakage, and impossible held-out lookups; the replacement uses message-group holdout and generated-answer verification |
| `software-architecture` | repaired and trained as `software-architecture-v1` | Prompt-group normalization, best-response selection, corruption filtering, topic balancing, and a new split removed normalized-prompt leakage; the source remains synthetic |
| Taskmaster-4 coffee API calls | trained once, then rejected to quarantine | 1,502 conflict-filtered examples across 749 disjoint groups; JSON format generalized, but API-name, argument, and full-object exactness missed promotion floors |
| Taskmaster-4 full-dialog coffee API v2 | trained once; all expert gates passed; registration held | 14,436 component-disjoint examples; candidate reached 99.44% API-name and 87.06% argument/semantic exactness on the exposed, recovery-qualified 1,430-example test |
| MultiDoGO customer shard | rejected | support transcripts with PII/duplication, not go-to-market expertise |
| `repos-code-corpus` | held | unmanifested/incomplete and currently sourced from CDAS only; it needs a proper code/DAPT pipeline |

There is still no downloaded broad business, marketing, investing, funding, or operations corpus in
the prepared directory. The finance adapter is narrow financial-statement QA, not the large business
expert planned in `experiments/business-expert/source_manifest.json`.

### Verified OCB corpus expert

`ocb-can-verified-v1-recipe.json` produced the completed rank-16 adapter from the GLM-prepared OCB
shard, not from the older CAN adapters. Its 12,320 training rows and 1,024
message-group-held-out rows are regenerated only from the 417 records labeled
`trusted`/`gold`/`gold`; prepared QA answers are not used, including 28,017 unsafe encode-100
prompts. Every synthetic decode payload is round-trip checked, and line-aligned source records and
hashes are in `data/ocb-can-verified.*.provenance.jsonl` and
`data/ocb-can-verified.report.json`. The training artifacts also preserve the warm-restart
provenance. These trust labels are corpus metadata, not independent external verification of the
underlying DBC facts, and the weak generated raw/physical accuracies still require a deterministic
decoder.

## Run and retrain

The desktop selects all four promoted rank-16 corpus experts into one resident rack and loads the
`qwen3-1.7b-corpus-router-v4-mlp-controlled` prompt router by default. Auto routing is primary;
base/null and each expert remain force-selectable for ablations:

```powershell
.\.venv\Scripts\molor-desktop.exe
```

Example retraining and evaluation commands:

```powershell
.\.venv\Scripts\python.exe `
  .\experiments\first-real-experts\build_router_v4_hard_negative_dataset.py `
  --output-prefix .\experiments\first-real-experts\data\router-bootstrap-v4-hard-negative-s307

.\.venv\Scripts\molor-train-router.exe `
  --train .\experiments\first-real-experts\data\router-bootstrap-v4-hard-negative-s307.train.jsonl `
  --validation .\experiments\first-real-experts\data\router-bootstrap-v4-hard-negative-s307.validation.jsonl `
  --test .\experiments\first-real-experts\data\router-bootstrap-v4-hard-negative-s307.test.jsonl `
  --adapter verified-embedded-code-v1=.\checkpoints\qwen3-1.7b-corpus-experts\verified-embedded-code-v1 `
  --output-dir .\checkpoints\qwen3-1.7b-corpus-router-v4-mlp-controlled `
  --cache-dir .\runs\router-features\bootstrap-v4-hard-negative-s307 `
  --device cpu --router-hidden-size 256 --seed 83

.\.venv\Scripts\molor-train-experts.exe `
  .\experiments\first-real-experts\software-architecture-v1-recipe.json --local-files-only

.\.venv\Scripts\python.exe .\experiments\first-real-experts\evaluate_adapter.py `
  --model .\models\Qwen3-1.7B `
  --adapter .\checkpoints\qwen3-1.7b-corpus-experts\software-architecture-v1 `
  --validation .\experiments\first-real-experts\data\software-architecture-v1.validation.jsonl `
  --limit 64 --batch-size 2 --max-length 1024
```

The promoted curriculum contains 4,680 train, 650 validation, and 650 test prompts balanced over
the four experts plus base/null. It excludes system/assistant text and has zero normalized-prompt
and semantic-family overlap. Seed 83 reached 99.69% validation and 99.23% file-held-out test
route-label accuracy. On a separately authored 60-prompt blind probe it improved acceptable
routing over v3 from 66.7% to 73.3% and exact single-label accuracy from 64.0% to 72.0%. CAN rose
from 5/10 to 8/10, embedded knowledge from 6/10 to 8/10, base/null stayed 9/10, and the executable
suite tied v3 at 502.5604 with no case-level regression. Verified-code intent is still a weak lane:
it moved from 2/10 to 1/10 on the blind probe and needs a new curriculum iteration.

V4 adds split-owned CAN, terse embedded, and base/null hard-negative families without copying
exact probe rows. Aggregate failures on the older probes informed those family choices, so the
older probe scores are development diagnostics rather than blind evidence. An initial v4
ablation that resampled unrelated v3 rows was rejected. The promoted controlled build keeps the
original v3 selection seed 307, isolating the new rows. This is still a routing-taxonomy bootstrap,
not proof that the selected expert beats every forced route.

V5 then replaced 144 split-owned generated-code rows with complete-project and driver-intent
contrastives while preserving the 4,680 / 650 / 650 balance and all 5,836 unaffected rows at their
original positions. Seed 83 retained 99.23% file-held-out route-label accuracy. On a new independent
72-prompt blind probe, overall acceptable routing rose from v4's 66.7% to 69.4% and verified-code
rose from 0/12 to 4/12, but the latter missed its required 8/12 floor; embedded knowledge regressed
from 6/12 to 4/12. V5 was rejected before executable evaluation, with no threshold tuning or seed
shopping.

A fresh complete five-route reward matrix was also collected: 30 new prompts, 150/150 successful
forced-route rollouts, 150 route-blind grades, and 19 same-prompt preferences. The manifest separates
20 train from 10 held-out prompts by semantic family and binds the exact base, v4 router, and four
adapter artifacts. Across beta 0.25/0.5/1.0 and seeds 8419/27183, all six conservative RL trials
selected epoch zero. Held-out reward gains and route changes were zero, and every saved checkpoint
was byte-identical to v4, so no RL router was promoted.

The four current corpus experts use rank 16 LoRA over Qwen's seven attention/MLP projections. The
adapters under `qwen3-1.7b-real-experts` are the older rank-8 experiments. Training uses BF16 and
gradient checkpointing; recipe batch details vary by corpus. The older 4,096-token terminal recipe
nearly saturates the 12 GB RTX 5070, so longer-context adapter training needs quantization, more
VRAM, or smaller batches.

The local embedded corpus contains mixed-license and internal material. Keep its data and adapter
local unless each source repository is separately cleared for redistribution.
