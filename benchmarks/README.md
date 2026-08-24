# MoLoRA benchmarks

This directory contains immutable evaluation inputs. Never add these cases to an SFT, DPO, RL,
router, retrieval, or memory-training corpus. Changing a prompt, answer, scorer, category weight, or
generation setting changes the suite fingerprint and starts a new score series. Scorer semantics
have their own version; the suite and scorer together form the evaluation fingerprint.

## Local routed-model benchmark

The bundled `molor-core-v1` suite has 64 fresh deterministic cases and three profiles:

- `smoke`: 10 cases, one from every category, for checking the plumbing;
- `quick`: 30 cases, three from every category, for frequent checkpoint iteration;
- `full`: all 64 cases, for a score worth recording.

It covers general reasoning, quantitative work, C/C++/assembly, embedded/CAN, terminal use,
software architecture, business/GTM, finance/investing, instruction following, and context
retrieval. It deliberately does not execute generated code or shell commands. Treat this as a
project-specific regression gauge, not a public leaderboard; use the standard lane below for
recognizable scores against other checkpoints.

With the Teacher API running:

```powershell
.\.venv\Scripts\molor-benchmark.exe validate

.\.venv\Scripts\molor-benchmark.exe run `
  --label qwen3-router-rl-v1 `
  --profile quick `
  --endpoint http://127.0.0.1:8765 `
  --route auto
```

Run a full, forced-base baseline and then compare the automatic router against it:

```powershell
.\.venv\Scripts\molor-benchmark.exe run `
  --label qwen3-base `
  --profile full `
  --route base/null

.\.venv\Scripts\molor-benchmark.exe run `
  --label qwen3-auto `
  --profile full `
  --route auto `
  --baseline .\runs\benchmarks\<base-run>\summary.json
```

Force any resident expert with `--route <runtime-expert-name>`. Test a real probability-weighted
mixture with, for example, `--route-top-k 2`. The score report always records the actual routing
decision and applied weights returned by the server.

Run the complete resident rack in one command:

```powershell
.\.venv\Scripts\molor-benchmark.exe matrix `
  --label qwen3-corpus-rack `
  --profile full
```

That runs forced base, every loaded expert, auto top-1, and auto top-2. Its report includes every
policy score, category vectors, a best-forced-route oracle, and the router's regret to that oracle.
The same mode is available as a checkbox in the native **Benchmarks** tab. Use
`--route-temperature`, `--confidence-threshold`, and `--null-bias` to create separate, fully
identified router-policy sweeps.

The runner writes a self-contained directory under `runs/benchmarks/` containing:

- `suite.snapshot.json` and its SHA-256 fingerprint;
- `stack.lock.json`, with hashes of base weights/tokenizer, every adapter, and the router;
- `run.json` with backend, seeds, environment, hardware, generation settings, and an exact workload
  contract for every case/repetition;
- append-only `results.jsonl`, plus `summary.json`, `scores.csv`, and `report.md`;
- `comparison.json` when a baseline was supplied.

The headline **MoLoRA Quality Index** is 0–1000. It is a weighted category macro-score, so adding
more questions to one category cannot silently make that category dominate. Tokens/sec, TTFT,
latency, and GPU information are separate from quality. A paired comparison reports index delta,
per-category deltas, wins/ties/losses, a category-weighted bootstrap interval, and regressions of at
least five percentage points. The CLI refuses comparison unless suite, evaluator, selected cases,
repetitions, decoding settings, and every verified raw result cell match.

Benchmark generations set `persist=false`, so they do not flood the annotation/feedback database.

## Executable engineering suite

`molor-executable-v1` is the harder local lane. Its first six cases compile and execute model-written
C/C++ for an overflow-safe sliding-window algorithm, two embedded/MISRA-oriented tasks, PIC32 UART
PPS setup, CRC-16/CCITT-FALSE, and generated duplicate-aware Mastermind puzzles. Hidden harnesses run
under WSL2 + Bubblewrap with no network, a private PID/user namespace, no capabilities, fixed CPU and
memory limits, and no view of the workspace. Harness result records use a per-run nonce so candidate
code cannot claim an invented test count.

Run the smoke profile (PPS, CRC, and logic) or the complete six-case lane against the locally served
model:

```powershell
.\.venv\Scripts\molor-benchmark.exe run `
  --suite .\benchmarks\suites\molor-executable-v1.json `
  --profile smoke `
  --label my-model-executable-smoke `
  --verifier-config .\benchmarks\verifiers\wsl-local-v1.json

.\.venv\Scripts\molor-benchmark.exe run `
  --suite .\benchmarks\suites\molor-executable-v1.json `
  --profile full `
  --label my-model-executable-full `
  --verifier-config .\benchmarks\verifiers\wsl-local-v1.json
```

The current verifier expects the `Ubuntu` WSL distribution with GCC, Bubblewrap, Python 3, and
Cppcheck plus its MISRA addon installed. The two MISRA cases report a **Cppcheck MISRA-addon-oriented
gate**, not certification or complete MISRA C coverage. Rule 8.7 has one declared harness deviation:
the public function must have external linkage so the independently compiled hidden harness can
call it. Sanitizer, compiler, analyzer, fixture, and partial-test evidence is retained in each JSONL
result.

The first recorded calibration run, GLM 5.3 with its `max` reasoning variant, scored **970.0/1000**
on all six cases. It passed every functional/property check; both MISRA-oriented answers lost only
the 10% analyzer component (reported rule IDs 8.4/15.5 and 8.4/11.4/15.5 respectively). The complete
evidence is in `runs/benchmarks/glm-5.3-max-thinking-executable-v1-full-recorded/`.

## Comparing another local model

Any OpenAI-compatible local server, including LM Studio, Ollama, or vLLM, can run the exact same
interest-specific suite:

```powershell
.\.venv\Scripts\molor-benchmark.exe run `
  --backend openai `
  --endpoint http://127.0.0.1:1234/v1 `
  --model other-model-id `
  --label other-model-full `
  --profile full
```

Remote/API model identity cannot be content-hashed, so its report is marked as not fully
content-addressed. Record an exact provider model/version when using that lane. The report also
records which generation controls the compatible endpoint can receive; generic OpenAI chat
endpoints do not standardize Qwen thinking, KV-cache mode, context budget, or `top_k` controls.

## Current recorded route matrix

The full v1.2 evaluator run on the currently loaded Qwen3-1.7B stack produced:

| Policy | Quality index | Delta vs base |
|---|---:|---:|
| base | 617.9 | +0.0 |
| `embedded-knowledge-v1` | 641.9 | +24.0 |
| `ocb-can-verified-v1` | 583.0 | -34.9 |
| `software-architecture-v1` | 537.2 | -80.7 |
| auto top-1 | 617.9 | +0.0 |
| auto top-2 | 591.2 | -26.7 |

The best-forced-route oracle is 794.2, and auto top-1 regret is 176.3 points. The exact artifacts
are in `runs/benchmarks/matrices/qwen3-corpus-rack-v2-recorded/`. Do not compare these numbers with
older v1.1 records; scorer semantics and workload provenance changed.

## Standard benchmark lane

The standard lane is pinned to
[`lm-evaluation-harness` 0.4.12](https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.12)
and leaves `trust_remote_code` disabled. Use a separate environment so its dependency pins cannot
disturb the model-training environment:

```powershell
py -3.12 -m venv .venv-eval
.\.venv-eval\Scripts\python.exe -m pip install --upgrade pip
.\.venv-eval\Scripts\python.exe -m pip install "lm_eval[hf,ifeval]==0.4.12"
.\.venv-eval\Scripts\python.exe -m pip install -e . --no-deps
```

Run the fast checkpoint vector (ARC Challenge, GSM8K, IFEval, and MMLU electrical engineering,
computer science, management, and marketing):

```powershell
.\.venv-eval\Scripts\molor-benchmark.exe standard `
  --model .\models\Qwen3-1.7B `
  --preset fast `
  --label qwen3-base-standard
```

Use `--preset nightly` for ARC Challenge, HellaSwag, full MMLU, GSM8K, and IFEval. The first run
downloads their datasets; `--offline` uses warmed Hugging Face caches. The main lane applies the
Qwen chat template with thinking disabled and greedy decoding. Keep thinking-on results in a
separate run.

`--limit` exists only to test installation and is visibly marked non-comparable. Never write down a
limited run as a model score. The task vector is the canonical result; its simple macro index is a
convenient local summary, not a claim of universal intelligence or an official leaderboard number.
The launcher locks local checkpoint/tokenizer bytes, records harness/dependency versions, task
metadata and fixed seeds. Group benchmarks such as full MMLU contribute one aggregate component to
the convenience macro instead of being counted once per subject. Hugging Face dataset revisions are
not content-locked by the harness, so standard-lane summaries distinguish a conventional public
benchmark comparison from the byte-exact regression claims supported by the local core suite.

The standard lane currently targets a plain Hugging Face checkpoint. The routed multi-LoRA system
uses the local core suite; a future custom lm-eval `LM` adapter must route once on the shared MC
question context and reuse that route for every continuation to keep multiple-choice likelihoods
valid.
