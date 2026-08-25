# ATXY exact-payload page-oracle development lab

This lab measures the best retained dense-attention mass available to a page selector at fixed
page budgets. It captures every Qwen3-1.7B layer and query head, keeps BF16 K/V exact, and ranks
non-mandatory pages using the current query's exact dense attention mass. That current-query leak
makes it an offline ceiling, not a deployable selector.

The experiment is development-only. It does not preregister a sealed corpus, install or modify a
live cache, compress KV, establish sparse-attention speed, authorize context extension, or make a
promotion claim. `A.TXY` is used only as exact physical `(layer, page)` identity. Its base-36 digits
are not semantic routing codes.

## Why this experiment comes next

The corrected 4K hashed-KV smoke showed that, at a 25% token budget, the exact page-mass oracle
retained only about `.95` mean attention mass for 32- and 64-token pages. The tails were worse than
the means. Meanwhile, every tested ATXY payload representation failed its independent dense
attention ceiling:

- cross-axis Tucker compression produced large output error even near full rank;
- four-tuples-per-vector raw residual VQ retained `1.223` mean output relative L2; and
- the strongest storage-eligible local linear latent retained `0.6538` global output relative L2.

This lab therefore holds payload exact and asks whether page geometry and candidate budget are
viable across the full stack before another selector or codec is trained.

## Development corpus shape

The intended smallest falsification matrix is four independently rendered heterogeneous 32K
prompts. Each prompt should mix non-Taskmaster software, embedded/CAN, financial, and
customer/terminal-style material in a different deterministic order. Place a unique support span
and close distractor before causal query checkpoints at 4K, 8K, 16K, and 32K. A 32,767-token
checkpoint is useful for exercising a partial final page.

Every checkpoint is captured with its own independent causal-prefix forward. A 4K checkpoint sees
exactly the first 4K tokens, an 8K checkpoint sees exactly the first 8K, and so on. The lab has no
single-full-forward shortcut or fallback protocol. This costs more prefill work, but avoids assuming
that a backend produces prefix-identical Q/K/V when its kernel shape includes later masked tokens.
Only one checkpoint capture is retained and scored at a time.

The checked-in development dataset is rendered entirely from deterministic synthetic templates.
It contains no Taskmaster examples or router evaluation inputs. Every 32K case includes software,
embedded/CAN, financial, and simulated customer-terminal records in a different segment order.
Each query has one unique `ACTIVE` support record plus two nearby `ARCHIVED`/near-anchor
distractors. The lab consumes these exact-token-count text files through a checksummed manifest:

```json
{
  "schema_version": "molor.atxy-page-oracle-development-dataset.v1",
  "status": "development-only",
  "cases": [
    {
      "id": "mixed-development-01",
      "path": "prompts/mixed-development-01.txt",
      "sha256": "<64 lowercase hex characters>",
      "token_count": 32768,
      "queries": [
        {"token_count": 4096, "support_token_spans": [[2048, 2056]]},
        {"token_count": 8192, "support_token_spans": [[6144, 6152]]},
        {"token_count": 16384, "support_token_spans": [[12288, 12296]]},
        {"token_count": 32767, "support_token_spans": [[24576, 24584]]},
        {"token_count": 32768, "support_token_spans": [[24576, 24584]]}
      ]
    }
  ]
}
```

Spans are half-open token intervals and must be causal for their query. The tokenizer output must
equal `token_count` exactly; the lab never silently truncates or pads a prompt. The manifest and
every source are read into immutable byte snapshots once. Tokenization uses the snapshotted UTF-8
text rather than reopening a source, and exact manifest/source byte equality is verified again
before any report can be written.

## Deterministic dataset builder

The tokenizer-only builder loads `models/Qwen3-1.7B` locally, renders four heterogeneous cases of
exactly 32,768 tokens, re-tokenizes the completed UTF-8 sources, and rejects any support marker
whose token subsequence is absent, repeated, non-causal, or insufficiently distinct from its close
distractors. It also emits a one-case pilot manifest that reuses case 01 at checkpoints 16,384,
32,767, and 32,768, plus a 4,096-token smoke source that is byte-for-byte the first validated causal
checkpoint prefix of case 01.

```powershell
.\.venv\Scripts\python.exe .\experiments\atxy-page-oracle-v1\build_dataset.py
```

Generated assets live under `dataset/`:

- `development-manifest.json` and four sources under `prompts/`;
- `pilot-manifest.json`, which reuses `prompts/mixed-development-01.txt` without duplicating it;
- `smoke-manifest.json` and one source under `smoke-prompts/`; and
- `SHA256SUMS`, which covers all three manifests and all five sources.

Writes are exclusive and idempotent. A rerun accepts existing byte-identical outputs and prints
`verified`; if any generated target differs, it fails without replacing or deleting anything.
Moving the differing file out of the dataset directory is therefore an explicit prerequisite for
regeneration.

## Run

Run the cheap one-case smoke path first:

```powershell
.\.venv\Scripts\python.exe -m molor.experimental.atxy_page_oracle_lab `
  --dataset-manifest .\experiments\atxy-page-oracle-v1\dataset\smoke-manifest.json `
  --model .\models\Qwen3-1.7B `
  --page-size 64 256 `
  --candidate-fraction 0.125 0.25 0.35 0.5
```

### 4K smoke result

The one-case smoke completed its full all-layer/head matrix and passed an independent integrity
audit. It is valid development evidence for the capture and scoring path, but it is not a selector
or context-extension result. At the 25% logical page budget:

| Page size | Mean retained mass | p01 retained mass | Global output relative L2 | Mean physical fetch |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 0.9639 | 0.6225 | 0.04035 | 71.7% |
| 256 | 0.9575 | 0.5703 | 0.04746 | 61.6% |

The output diagnostics pass the development L2/cosine screens, but attention-mass tails fail
badly and independent head choices union into mostly dense physical reads. Page-256 support recall
is structurally guaranteed in this smoke because the support lies in the mandatory recent page.
The result therefore authorizes only the focused longer-length pilot below; it does not justify
training a fixed-budget selector.

The report contains 3,584 head-budget records, 2,688 minimum-mass records, and 224 physical-union
records. File SHA-256:
`d164e11679d3469f50e20a8c0f7555ab2149d60bb8949e36c0120335b06e26d0`.
Canonical in-report integrity SHA-256:
`a865ea414a436d3a84d864ca1037cdba5c0f82e0ce22a3b2a7e72277d4405479`.

After that capture/scoring/report path passes, run the focused one-case pilot:

```powershell
.\.venv\Scripts\python.exe -m molor.experimental.atxy_page_oracle_lab `
  --dataset-manifest .\experiments\atxy-page-oracle-v1\dataset\pilot-manifest.json `
  --model .\models\Qwen3-1.7B `
  --page-size 64 256 `
  --candidate-fraction 0.25 0.35
```

The pilot performs three independent causal prefix captures. The 16,384 checkpoint supplies the
first decisive length, while 32,767 exercises a partial final page and adjacent 32,768 is its
full-page control. If oracle attention retention or the layer-wide physical union is already
unacceptable here, stop before selector work and before spending on the four-case matrix. Passing
the pilot authorizes only that larger development matrix; it does not establish context extension
or promotion readiness.

After the pilot passes, invoke the four-case matrix:

```powershell
.\.venv\Scripts\python.exe -m molor.experimental.atxy_page_oracle_lab `
  --dataset-manifest .\experiments\atxy-page-oracle-v1\dataset\development-manifest.json `
  --model .\models\Qwen3-1.7B `
  --page-size 64 256 `
  --candidate-fraction 0.125 0.25 0.35 0.5
```

Reports are exclusively created beneath `runs/atxy-page-oracle-v1/` by default and are explicitly
ineligible for promotion. The checked-in manifests are deterministic development inputs, not a
preregistered corpus, and no formal report is checked in.

The model must be a local Hugging Face directory. Before configuration, tokenizer, or weights are
loaded, the lab fully hashes the behavior-bearing local stack: config and generation metadata,
tokenizer payload/config/chat-template files, weight indexes, and every referenced/model weight
file. The same full byte lock is recomputed after capture and scoring; any difference rejects the
run. Implementation bytes receive the same pre/post check.

## Exact oracle calculation

For each independently captured causal checkpoint, layer, and all 16 query heads, query head `h`
maps to KV head `h // 2`. Dense float32 CPU QK/softmax is calculated once and the validated weight
vector is reused across every page size. Each page is then reduced to:

```text
M[p] = sum of exact attention weights in page p
U[p] = sum of exact attention weight times exact V in page p
```

Sink and recent pages are mandatory. Remaining pages are ordered by `(-M[p], page_id)`. Cumulative
`M` gives retained attention mass for every budget, while cumulative `U / M` gives the exact-payload
candidate output without repeating QK, softmax, or token-level V aggregation for each fraction.

The report records:

- retained total and cold-only attention mass;
- top-token, top-32-token, and top-attention-page recall;
- support-page recall;
- exact-payload output relative L2 and cosine similarity;
- minimum page/token fractions needed for `.98`, `.995`, and `.999` mass;
- the union of the two query-head selections sharing each KV head; and
- the layer-wide physical page union across all eight KV heads, including exact BF16 bytes.

The last metric matters because independently sparse head selections can union into a nearly dense
physical read. A logical 25% candidate fraction is not a 4x I/O reduction unless the layer-wide
union remains close to 25%.

## Development interpretation

The 16K and 32K rows should be screened against the existing retrieval targets:

- mean total attention-mass recall at least `.995` at a 25% budget;
- p01 total attention-mass recall at least `.98`;
- maximum-attention-token page recall at least `.995`;
- no missed declared support page in a layer-wide physical union;
- median minimum fraction for `.995` mass at most `.25`, p95 at most `.35`;
- global exact-payload output relative L2 at most `.05`; and
- mean output cosine similarity at least `.995`.

These are development screens, not automatic promotion gates. A pass authorizes only a larger,
separately preregistered, document-disjoint selector study. A failure of the attention-mass oracle
is definitive for that page size and budget: no other selector can retain more exact attention mass
with the same mandatory pages and page count. Output error is diagnostic rather than a mathematical
upper bound because a V-aware subset could exploit value cancellation despite retaining less mass.

Page size 256 preserves the current `A.TXY` capacity of 11,943,936 tokens per layer. Page size 64
is a fine-granularity diagnostic but the same three-digit page namespace covers only 2,985,984
tokens. If 64 passes and 256 fails, longer-term addressing needs an extension or superpage namespace.

## Selector and payload attribution

This exact-payload oracle isolates page geometry. A later selector is measured with the same exact
K/V and page count; its loss relative to this oracle is selector loss. A future payload codec must
first pass dense all-page evaluation, then reuse frozen candidate pages with K-only, V-only, and
K+V ablations. Allowing compressed K to change the candidates while grading compression would
confound payload and routing error.

At 32K, exact BF16 captured KV is 3.5 GiB for this model. The implementation holds at most one
checkpoint capture, scores and releases its captured layers sequentially on CPU, then performs a
new causal forward for the next checkpoint. It does not retain multiple checkpoints or float32
copies of the whole stack. Model inference should be scheduled only when the GPU has enough
headroom for the unchanged dense 32K forward.
