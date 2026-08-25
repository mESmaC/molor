# MoLoRA Lab

A local playground for routing and mixing complete LoRA adapters on one frozen language model.
The current experiment uses Qwen3-1.7B, can keep an arbitrary rack of adapters resident, and
applies the selected LoRA deltas during ordinary Hugging Face generation.

This is model tooling for an experiment, not a product. The old `ui/` prototype is not used by the
current runtime.

## What works now

- a frozen local Qwen3-1.7B checkpoint on CUDA;
- any number of simultaneously resident LoRA experts;
- a trained prompt-level auto-router over the current four corpus experts plus base/null;
- blind forced-route evaluation and conservative teacher-reward optimization for that router;
- a versioned 64-case model benchmark with a 0–1000 quality index, category scores, paired
  checkpoint deltas, exact stack hashes, and separate TTFT/throughput measurements;
- a harder six-case executable lane for C/C++, MISRA-oriented embedded code, PIC32 PPS, CRC, and
  generated logic puzzles, graded in a networkless WSL/Bubblewrap sandbox;
- request-time top-k, router temperature, confidence floor, null bias, token budget, and expert
  masking, with exact manual forcing retained for ablations;
- streamed generation with token count, context use, TTFT, and tokens/second;
- exact context budgets with oldest-whole-turn eviction;
- GPU, CPU-offloaded, or disabled KV-cache modes;
- independent sequential adapter training from small chat JSONL files;
- local scalar rewards, corrected answers, and pairwise preferences;
- independent multi-model teacher grades with confidence-weighted consensus;
- SFT and offline DPO-style LoRA training from those annotations;
- an NVMe KV snapshot/reload and cache-mode benchmark lab;
- a shadow-only hashed-KV page-routing lab graded against exact Qwen attention;
- a native dark-mode PySide6 chat with streamed Markdown and collapsed Qwen reasoning traces,
  plus annotation, teacher-RL, benchmark, training, and context-lab tools;
- a terminal command that compares base, each expert, and a 50/50 blend on the same prompt.

The promoted router is still the controlled five-way v4 MLP checkpoint. Its hard-negative
curriculum improved acceptable routing from 66.7% to 73.3% on a 60-prompt blind probe while exactly
tying v3 on the six-case executable lane. A verified-code-focused v5 candidate and a fresh
150-rollout five-route RL sweep both failed their held-out promotion gates, so neither changed the
live router. A Taskmaster coffee-API adapter produced valid JSON consistently but missed its API,
argument, and full-object exactness floors and remains quarantined. Persistent cross-session memory
and Memory-LoRA remain the next model-side pieces. The existing `docs/` directory contains design
notes for those experiments.

## Run the desktop app

The local model, four corpus experts, and router checkpoint are auto-detected from `models/` and
`checkpoints/`:

```powershell
.\.venv\Scripts\molor-desktop.exe
```

Click **Load model + selected experts**. Auto is the default route mode; the sidebar shows the
complete probability distribution and applied route before token generation begins. The resident
rack is a checkable list, while a separate eligibility list masks experts per request without
unloading them. Manual mode force-selects base/null or one loaded expert for comparisons. These
rank-16 adapters live under `checkpoints/qwen3-1.7b-corpus-experts/`; the earlier rank-8 finance,
local-code, terminal, and CAN experiments remain available under
`checkpoints/qwen3-1.7b-real-experts/`.

The current adapters are experiments, not generic upgrades:

| Adapter | Corpus and held-out result | Important limit |
|---|---|---|
| `embedded-knowledge-v1` | 6,000 train / 512 file-held-out; NLL 3.6013 -> 1.4356; 64-generation character similarity .1003 base / .1175 old rank-8 / .3070 new | A broader embedded reference/completion expert, not a verified compiler or debugger. The separate 35-row seen-recall probe is explicitly not held-out. |
| `ocb-can-verified-v1` | 12,320 train / 1,024 message-group-held-out; NLL 4.0179 -> .07419; 128-generation structured-field accuracy .20295 -> .78771 and all-fields exact 0 -> 36 | Raw-value accuracy .359 and physical-value accuracy .344 remain weak; keep deterministic decoding in the loop. Warm-restart provenance is retained with the run. |
| `software-architecture-v1` | 12,000 train / 1,000 validation with zero normalized-prompt overlap; NLL 2.4796 -> 1.2808; 64-generation term F1 .2661 -> .3657 and quality-signal recall .4243 -> .5776 | Concept recall was flat (.4177 -> .4142). The slice covers 14/15 families, and lexical overlap does not establish architectural correctness; a blind review pack is included. |
| `verified-embedded-code-v1` | 920 compiler-verified training rows; fresh executable sibling score 66.8 base -> 152.2; sealed executable score 303.4 prior Auto -> 502.6 forced | Strongest current executable-code adapter, especially CRC and functionally correct MISRA-oriented C. It still fails PIC32 PPS and several hidden tests, so compiler/verifier feedback remains essential. |

There is no downloaded broad business, marketing, investing, funding, or operations corpus yet.
The business source manifest is a collection plan, not a trained business expert.

See `experiments/first-real-experts/README.md` for data audits, recipes, hashes, and full held-out
results.

The other tabs are live experiment tools:

- **Annotation / Reward** records -2..+2 scores, dimension scores, notes, corrected answers, and
  chosen/rejected pairs. **Regenerate last prompt** creates a comparable candidate without changing
  the prompt history.
- **Teacher RL** launches a loopback-only JSON tool API. Codex, Claude, GLM, Kimi, or another agent
  can prompt the student, generate comparable candidates, submit independent rubric grades and
  corrections, create pairwise preferences, and inspect consensus/disagreement.
- **Benchmarks** runs smoke, quick, or full deterministic suites against the teacher API or another
  OpenAI-compatible local server. It can force base/an expert, exercise the auto-router or top-k
  mixture, compare the result with a prior run, or run the entire resident route rack as one
  base/expert/top-k matrix.
- **Expert Training** can build and launch a small adapter recipe, train from positive annotations,
  run an existing recipe, or train a DPO-style adapter from pairwise preferences.
- **Context Lab** controls the chat context budget and KV-cache mode, then benchmarks exact prompt
  lengths and snapshots a real cache to the selected NVMe directory.

Reasoning presets use Qwen's native thinking switch and its recommended sampling regimes. The
system instruction remains editable, so the presets are starting points rather than hidden magic.
Leading `<think>...</think>` output is separated from the final answer and shown in a collapsed
**Reasoning** disclosure. The final answer alone enters chat history and annotation training data;
for completed recorded generations, the trace is retained only as local telemetry.

## Compare routes in the terminal

```powershell
.\.venv\Scripts\molor-compare-experts.exe "Explain why CPU cache matters" --max-new-tokens 128
```

The base model is loaded once. The command then streams the same seeded prompt through base-only,
Expert-A-only, Expert-B-only, and a 50/50 expert mixture. Its defaults match the desktop and can be
overridden with `--expert-a` and `--expert-b`.

## Benchmark models and checkpoints

Run the 30-case quick suite against the currently loaded teacher API:

```powershell
.\.venv\Scripts\molor-benchmark.exe run --label my-checkpoint --profile quick --route auto
```

Use `--profile full` for the 64-case result worth recording. The report includes a 0–1000 quality
index; category scores for reasoning, math, C/C++/assembly, embedded/CAN, terminal use,
architecture, business, finance, instruction following, and context retrieval; route accuracy;
TTFT; tokens/sec; latency; and a stratified bootstrap interval. `--baseline <prior-run>` adds
paired category deltas and wins/ties/losses.

Every run snapshots the suite and fingerprints the exact base weights/tokenizer, adapters, router,
route-policy knobs, evaluator semantics, and workload cells. A comparison is refused unless the
profile, cases, repetitions, decoding settings, evaluator, and every paired raw result match.
Incomplete runs are not written to score history, and benchmark outputs do not enter the feedback
database. The current full route matrix is:

| Route policy | Quality index | Delta vs base |
|---|---:|---:|
| Frozen base | 617.9 / 1000 | baseline |
| `embedded-knowledge-v1` forced | 641.9 / 1000 | +24.0 |
| `ocb-can-verified-v1` forced | 583.0 / 1000 | -34.9 |
| `software-architecture-v1` forced | 537.2 / 1000 | -80.7 |
| RL autorouter top-1 | 617.9 / 1000 | +0.0; 1 win / 62 ties / 1 loss |
| RL autorouter top-2 | 591.2 / 1000 | -26.7 |

The per-case best-forced-route oracle reaches 794.2, leaving 176.3 points of top-1 router regret.
That oracle is an upper bound chosen after seeing correctness, not a deployable score, but it shows
there is useful route-selection headroom. See `benchmarks/README.md` for the recorded artifacts,
LM Studio/OpenAI-compatible comparisons, the pinned `lm-evaluation-harness` lane (ARC, GSM8K,
IFEval, HellaSwag, and MMLU), score interpretation, and artifact layout.

On the harder six-case executable lane, the promoted five-way MLP router at a 0.30 confidence
floor scores **502.6 / 1000**, exactly matching forced `verified-embedded-code-v1` and improving
the previous base-only Auto behavior from **303.4**. The narrow v2 and reasoning-rich v3 adapter
candidates were retained as rejected experiments after scoring 317.6 and 411.9 respectively on
the sealed lane.

## Train the current experts

Training keeps the base frozen and optimizes one adapter at a time. For example:

```powershell
.\.venv\Scripts\molor-train-experts.exe experiments/first-real-experts/software-architecture-v1-recipe.json --local-files-only
```

The completed corpus recipes use rank 16 over all seven Qwen attention/MLP projections and write
to `checkpoints/qwen3-1.7b-corpus-experts/`. Recipes and checkpoints under
`qwen3-1.7b-real-experts` are the earlier rank-8 experiments.

The tiny two-style recipe remains useful as a trainer smoke run or to select a single adapter:

```powershell
.\.venv\Scripts\molor-train-experts.exe --local-files-only --expert creative experiments/qwen3-1.7b-two-styles/recipe.json
```

Each JSONL row has a chat ending in the assistant answer to train. Prompt tokens are masked, so the
loss is applied only to that final assistant response.

## Train the auto-router

The router sees only the latest user prompt. It masked-mean pools at most 256 frozen Qwen
input-token embeddings and trains a small five-route MLP; it never loads or runs the causal
transformer. Rebuild the controlled hard-negative curriculum and retrain it on CPU with:

```powershell
.\.venv\Scripts\python.exe experiments/first-real-experts/build_router_v4_hard_negative_dataset.py --output-prefix experiments/first-real-experts/data/router-bootstrap-v4-hard-negative-s307
.\.venv\Scripts\molor-train-router.exe --train experiments/first-real-experts/data/router-bootstrap-v4-hard-negative-s307.train.jsonl --validation experiments/first-real-experts/data/router-bootstrap-v4-hard-negative-s307.validation.jsonl --test experiments/first-real-experts/data/router-bootstrap-v4-hard-negative-s307.test.jsonl --adapter verified-embedded-code-v1=checkpoints/qwen3-1.7b-corpus-experts/verified-embedded-code-v1 --output-dir checkpoints/qwen3-1.7b-corpus-router-v4-mlp-controlled --cache-dir runs/router-features/bootstrap-v4-hard-negative-s307 --device cpu --router-hidden-size 256 --seed 83
```

The emitted `checkpoints/qwen3-1.7b-corpus-router-v4-mlp-controlled/` contains the router head and
exact identity/metrics sidecar. Its 4,680 / 650 / 650 split is balanced across four experts plus
base/null and has no normalized-prompt or semantic-family overlap. Seed 83 reached 99.69%
validation and 99.23% file-held-out test accuracy. A separately authored 60-prompt blind probe
raised acceptable-route accuracy from v3's 66.7% to 73.3% and exact single-label accuracy from
64.0% to 72.0%; CAN rose from 50% to 80%, embedded knowledge from 60% to 80%, and base/null stayed
at 90%. Verified-code intent remains weak and slipped from 2/10 to 1/10. The verifier-backed
executable lane nevertheless tied v3 exactly at 502.5604, with zero per-case regressions.

The hard-negative family choices were informed by aggregate failures on the older intent probes,
so those older scores are development diagnostics, not blind promotion evidence. A first v4
ablation also resampled unrelated v3 rows and was rejected; the promoted build holds the original
v3 selection seed fixed at 307 so the comparison isolates the new hard negatives.

The follow-up v5 curriculum replaced 144 split-owned generated-code rows with complete-project and
driver intent contrastives while preserving route balance and all unaffected row positions. Its
selected seed retained 99.23% file-held-out accuracy, but the new 72-prompt blind gate only moved
overall acceptable routing from 66.7% to 69.4% and verified-code routing from 0/12 to 4/12, short of
the required 8/12. Embedded routing also fell from 6/12 to 4/12, beyond the one-case regression
allowance. V5 was rejected before executable evaluation; v4 remains the default.

The teacher-RL pilot evaluates 24 immutable prompts through base, all three experts, and auto, then
trains only on a separate 32-prompt/128-rollout sibling set. Reproduce the collection and the
conservative full-information contextual-bandit update with:

```powershell
.\.venv\Scripts\python.exe experiments/teacher-rl-v1/run_teacher_eval.py collect --continue-on-error
.\.venv\Scripts\python.exe experiments/teacher-rl-v1/run_teacher_eval.py --manifest experiments/teacher-rl-v1/train_manifest.json collect --continue-on-error
.\.venv\Scripts\python.exe experiments/teacher-rl-v1/optimize_reward_router.py train --manifest experiments/teacher-rl-v1/train_manifest.json --feedback-db runs/feedback/feedback.sqlite3 --output checkpoints/qwen3-1.7b-corpus-router-teacher-rl-v1 --beta 0.5 --seed 8419 --device cpu
```

The candidate is deliberately not the desktop default. On the 24-prompt replay it improved one
answer, tied 23, and regressed none, but remained behind always-base; acceptable-route alignment
also fell from 13/24 to 12/24 as confidence rose. The real `top_k=2` blend was worse than top-1
(2 better / 17 tied / 5 worse), so top-1 remains the default. Full raw rollouts, blind judgments,
exports, and the evaluation summary live under `runs/teacher-rl-v1/`.

The optimizer now accepts an arbitrary expert bank while strictly binding the parent action axis,
runtime aliases, adapter order, route mapping, and checkpoint bytes. New collections fingerprint
the complete local base, router, and adapter artifacts and reject mixed or drifted rollout sets.
Historical collections are retained only with an explicit `legacy_path_identity_only` status after
strict runtime path/name checks. A follow-up beta/seed sweep over the complete legacy 32-prompt,
four-route reward matrix selected epoch zero in all six trials: no split improved hard top-1 reward,
and every saved candidate stayed byte-identical to its parent.

The replacement five-route collection is now complete: 30 new prompts were each run through all
four experts plus base/null for 150/150 successful, content-addressed rollouts. Three route-blind
review shards produced 150 grades and 19 same-prompt preferences. The same six beta/seed trials
again selected epoch zero: held-out hard and expected reward gains, route changes, and KL were all
zero, and every saved weight file was byte-identical to v4. This is a valid negative RL result, not
a promoted policy.

The quarantined `financial-analysis-verified-r16-v1` experiment trained 2,026 clean rows for one
rank-16 epoch. Held-out NLL improved from 5.2249 to 0.2971, but the 149 eligible calculation rows
scored only 38.3% numeric+unit accuracy, 38.9% verifier pass, and 61.1% calculation consistency,
all below the stronger rank-8 comparator. It was not added to the rack. Exact hashes, seed results,
gates, and decisions are recorded in `experiments/training-update-2026-08-24.json`.

The quarantined `taskmaster-coffee-api-v1` experiment trained 1,176 group-separated Taskmaster-4
examples under CC-BY-4.0. On 145 held-out cases it reached 100% JSON/schema validity, 77.9% API-name
exactness, 47.6% argument exactness, and 45.5% full-object exactness. Those last three missed the
85% / 70% / 60% floors, so the adapter is excluded from defaults, router training, and normal
desktop discovery. Its exposed v1 test must not be used to tune a successor; a v2 attempt needs a
new group-disjoint holdout.

Run the blind intent probe with the desktop's default routing policy using:

```powershell
.\.venv\Scripts\python.exe experiments/first-real-experts/evaluate_router.py --checkpoint checkpoints/qwen3-1.7b-corpus-router-v4-mlp-controlled --probe experiments/first-real-experts/router-blind-probe-v3.jsonl --confidence-threshold 0.30
```

The promoted 0.30 policy gets 73.3% acceptable-route accuracy and 72.0% exact accuracy on the new
60-prompt blind probe. The older 40- and 18-prompt probes remain useful development diagnostics,
but their aggregate failure patterns informed v4's curriculum and they are not promotion evidence.
Misses and full probability vectors are retained under `runs/evaluation/`.

## Manual rewards and preference training

Feedback stays local in `runs/feedback/feedback.sqlite3`. Export all three training views from the
desktop Annotation tab or from the terminal:

```powershell
.\.venv\Scripts\molor-feedback.exe
```

After saving at least one chosen/rejected pair, train a fresh preference adapter:

```powershell
.\.venv\Scripts\molor-train-preference.exe --local-files-only experiments/qwen3-1.7b-two-styles/preference_recipe.json
```

This is offline DPO-style preference optimization: the frozen base/null route supplies reference
log-probabilities and only the new LoRA expert is updated.

## Model-teacher interface

Start it from the desktop **Teacher RL** tab, or directly with the current expert rack and router:

```powershell
.\.venv\Scripts\molor-teacher.exe serve --model .\models\Qwen3-1.7B --expert-a .\checkpoints\qwen3-1.7b-corpus-experts\embedded-knowledge-v1 --expert-b .\checkpoints\qwen3-1.7b-corpus-experts\ocb-can-verified-v1 --expert .\checkpoints\qwen3-1.7b-corpus-experts\software-architecture-v1 --expert .\checkpoints\qwen3-1.7b-corpus-experts\verified-embedded-code-v1 --router .\checkpoints\qwen3-1.7b-corpus-router-v4-mlp-controlled --port 8765
```

The model is lazy-loaded on the first prompt. The API binds only to `127.0.0.1`; its rubric and
JSON tool schemas are available at `http://127.0.0.1:8765/v1/capabilities` and its OpenAPI summary
at `http://127.0.0.1:8765/openapi.json`.

PowerShell example for an agent or orchestration script:

```powershell
$request = @{
  prompt = "Explain volatile in embedded C"
  candidates = 2
  settings = @{ max_new_tokens = 128; enable_thinking = $false }
} | ConvertTo-Json -Depth 4

$request | .\.venv\Scripts\molor-teacher.exe invoke ask_student
```

Every teacher judgment is retained separately with evaluator/model identity, confidence, detailed
dimension scores, critique, and an optional corrected answer. The Annotation tab shows weighted
consensus and disagreement. Training SFT exports combine human positives with confidence-filtered
teacher corrections; preference exports feed the existing DPO trainer.

## Context and NVMe cache lab

```powershell
.\.venv\Scripts\molor-cache-lab.exe --benchmark-prompt-tokens 512 2048 4096 --prefill-tokens 4096
```

Transformers' supported live cache offload moves layer KV state to CPU, not directly to disk. The
NVMe portion of this lab is deliberately labelled **snapshot/archive**: it writes a real
`DynamicCache`, reloads it, and verifies continuation logits. It gives measured I/O data for a
future custom paged cache without presenting disk as VRAM.

The first hashed/paged-cache milestone also runs in shadow mode. It captures real post-RoPE Q/K and
matching V from selected Qwen layers, then compares deterministic 32/64-bit SimHash, a Quest-style
min/max page bound, mandatory-only locality, and a forced-set exact-attention oracle at equal
budgets. Reports group compatible configurations and separate forced from routed attention mass. It
does not alter generation or evict cache entries:

```powershell
.\.venv\Scripts\molor-hashed-kv-lab.exe --target-tokens 1024
```

See `docs/kv-cache.md` and `experiments/hashed-kv-v1/` for the architecture, sweep, promotion
gates, and the required separation between approximate routing hashes and exact content identity.

## Fresh environment

Install the CUDA build of PyTorch appropriate for the machine, then the local extras:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[hf,desktop]"
```

The default model path is `models/Qwen3-1.7B`. A different local checkpoint or Hugging Face model
ID can be entered directly in the desktop app.

## Useful code

- `src/molor/experimental/cache_lab.py` - cache-mode and NVMe snapshot measurements;
- `src/molor/feedback/` - local reward/preference storage and training exports;
- `src/molor/teacher/` - loopback teacher API, CLI transport, rubric, and consensus logic;
- `src/molor/benchmark/` and `benchmarks/` - content-locked benchmark runner, deterministic
  scorers, reports, checkpoint comparisons, standard-task launcher, and immutable suite;
- `experiments/teacher-rl-v1/` - immutable evaluation, sibling reward collection, blind review,
  resumable submission, and conservative router-RL tooling;
- `src/molor/training/preference.py` - assistant-token DPO-style preference training;
- `experiments/embedded-expert/source_manifest.json` - ranked embedded/C/C++/assembly/linker,
  MISRA-oriented, Microchip, and C167 sources with acquisition and licensing posture;
- `experiments/terminal-expert/source_manifest.json` - ranked Bash, PowerShell, cmd, build,
  flash/debug, trajectory, verifier, and terminal-RL sources with dataset holdouts;
- `experiments/business-expert/source_manifest.json` - large-scale business, marketing,
  investing, funding, finance, operations, case-generation, and business-RL sources;
- `experiments/first-real-experts/README.md` - trained adapters, dataset decisions, and evaluations;
- `src/molor/experimental/mixture.py` - real Transformer instrumentation, routing, streaming, and
  adapter serialization;
- `src/molor/training/` - assistant-only collation and the small sequential SFT loop;
- `src/molor/desktop/` - native desktop application and model bridge;
- `experiments/qwen3-1.7b-two-styles/` - the editable recipe and two intentionally tiny datasets.
