# First real experts

This experiment keeps `~/Documents/datasets` read-only. Builders write normalized
chat shards, provenance sidecars, hashes, and audit reports under `data/`. The current rank-16
corpus adapters and their summaries live under `checkpoints/qwen3-1.7b-corpus-experts/`; the
earlier rank-8 experiments remain under `checkpoints/qwen3-1.7b-real-experts/`.

## Current rank-16 corpus experts

All three use the same frozen local Qwen3-1.7B base and all seven attention/MLP projections.

| Adapter | Data split | Held-out result | Honest status |
|---|---:|---|---|
| `embedded-knowledge-v1` | 6,000 train / 512 file-held-out | NLL 3.6013 -> 1.4356; on 64 generations, character similarity was .1003 base / .1175 earlier rank-8 adapter / .3070 new expert | Clear corpus-use improvement, but generated code and technical claims are not mechanically verified. A separate 35-row seen-recall probe exists and is explicitly not held-out. |
| `ocb-can-verified-v1` | 12,320 train / 1,024 message-group-held-out | NLL 4.0179 -> .07419; on 128 generations, structured-field accuracy rose .20295 -> .78771 and all-fields exact rose 0 -> 36 | Raw-value accuracy .359 and physical-value accuracy .344 are still weak, so deterministic decoding remains necessary. The completed run retains its warm-restart provenance. |
| `software-architecture-v1` | 12,000 train / 1,000 validation; zero normalized-prompt overlap | NLL 2.4796 -> 1.2808; on 64 generations, term F1 rose .2661 -> .3657 and quality-signal recall .4243 -> .5776 | Concept recall was effectively flat (.4177 -> .4142). The generation slice spans 14/15 topic families; these lexical/reference metrics do not prove semantic correctness. A blind review pack is available. |

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

The desktop selects all three rank-16 corpus experts into one resident rack and loads the
`qwen3-1.7b-corpus-router-v1` prompt router by default. Auto routing is primary; base/null and each
expert remain force-selectable for ablations:

```powershell
.\.venv\Scripts\molor-desktop.exe
```

Example retraining and evaluation commands:

```powershell
.\.venv\Scripts\python.exe `
  .\experiments\first-real-experts\build_router_dataset.py --per-route 3000

.\.venv\Scripts\molor-train-router.exe --device cpu

.\.venv\Scripts\molor-train-experts.exe `
  .\experiments\first-real-experts\software-architecture-v1-recipe.json --local-files-only

.\.venv\Scripts\python.exe .\experiments\first-real-experts\evaluate_adapter.py `
  --model .\models\Qwen3-1.7B `
  --adapter .\checkpoints\qwen3-1.7b-corpus-experts\software-architecture-v1 `
  --validation .\experiments\first-real-experts\data\software-architecture-v1.validation.jsonl `
  --limit 64 --batch-size 2 --max-length 1024
```

The router curriculum contains 9,600 train, 1,200 validation, and 1,200 test prompts balanced over
the three experts plus base/null. It uses training corpora only, excludes system/assistant text,
has zero normalized-prompt and source-family overlap, and reserves 336 manually labelled intent
prompts. The current linear embedding-mean head reached 99.3% validation and 99.0% test route-label
accuracy. This is a routing-taxonomy bootstrap, not an end-to-end claim that the selected expert
beats every forced route; reward/NLL distillation is still needed.

`evaluate_router.py` runs a separate 40-prompt intent probe that is never read by the dataset
builder. With the desktop defaults it currently reaches 80.0% acceptable-route accuracy and 75.8%
exact accuracy on the 33 single-label rows; the retained misses are the useful part of this probe.

The staged router-v2 tooling adds `verified-embedded-code-v1` without changing the current router
checkpoint or service. Build its five-route, semantic-group-disjoint corpus and, when ready, train
to new output/cache directories:

```powershell
.\.venv\Scripts\python.exe `
  .\experiments\first-real-experts\build_router_v2_dataset.py

.\.venv\Scripts\molor-train-router.exe `
  --train .\experiments\first-real-experts\data\router-bootstrap-v2.train.jsonl `
  --validation .\experiments\first-real-experts\data\router-bootstrap-v2.validation.jsonl `
  --test .\experiments\first-real-experts\data\router-bootstrap-v2.test.jsonl `
  --adapter verified-embedded-code-v1=.\checkpoints\qwen3-1.7b-corpus-experts\verified-embedded-code-v1 `
  --output-dir .\checkpoints\qwen3-1.7b-corpus-router-v2 `
  --cache-dir .\runs\router-features\bootstrap-v2 --device cpu
```

After training, use `data/router-intent-probe-v2.jsonl` with `evaluate_router.py` to distinguish
code implementation/compile requests from embedded reference questions and ordinary base traffic.

The three current corpus experts use rank 16 LoRA over Qwen's seven attention/MLP projections. The
adapters under `qwen3-1.7b-real-experts` are the older rank-8 experiments. Training uses BF16 and
gradient checkpointing; recipe batch details vary by corpus. The older 4,096-token terminal recipe
nearly saturates the 12 GB RTX 5070, so longer-context adapter training needs quantization, more
VRAM, or smaller batches.

The local embedded corpus contains mixed-license and internal material. Keep its data and adapter
local unless each source repository is separately cleared for redistribution.
