# Business reference mixtures v1

These recipes select grounded business documents without converting them into
synthetic conversations. Input and output rows remain exactly:

```json
{"document_id":"...","title":"...","text":"..."}
```

Each recipe pins every source shard's absolute path, SHA-256, record count, leaf,
leaf revision, and split. Train, validation, and test each use every other split as
a leakage guard for both `document_id` and the exact `{title,text}` payload. Every
input row terminates as `emitted`, `duplicate`, `quarantined`, `rejected`, or
`capped` in `provenance.jsonl` and the aggregate report.

## Canonical recipes

- `train.recipe.json`: 7,132 references, training use
- `validation.recipe.json`: 435 references, evaluation use
- `test.recipe.json`: 364 references, evaluation use

The full recipes use equal shard weights and no caps. Their content identities and
the source-manifest identity are pinned in `recipes-receipt.json`. The completed
real-build identities are in `actual-mixtures-receipt-v1.json`.

## Rebuild the recipes

From the repository root:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe experiments/business-expert/build_reference_mixture_recipes.py `
  --manifest C:\Users\engin\Documents\datasets\processed\molor-business-reference-v1\manifest.json `
  --output <new-empty-recipe-directory>
```

The destination must not already exist. Generation is staged and renamed into
place only after every recipe and checksum has been written.

## Build or inspect a mixture

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe experiments/business-expert/build_reference_mixture.py `
  --recipe experiments/expert-rack-v5/business-reference-mixtures-v1/train.recipe.json `
  --output <new-empty-output-directory> `
  --dry-run
```

`--dry-run` validates every pin and guard, performs the exact selection, and
returns artifact hashes without creating the output directory.

Weights and caps are repeatable runtime overrides. Overrides create a new resolved
recipe identity while leaving the canonical recipe file unchanged:

```powershell
.\.venv\Scripts\python.exe experiments/business-expert/build_reference_mixture.py `
  --recipe experiments/expert-rack-v5/business-reference-mixtures-v1/train.recipe.json `
  --output <new-empty-output-directory> `
  --weight leaf:growth-gtm=2.5 `
  --weight shard:business-reference-business-operations-train-v1=4 `
  --cap leaf:investing-markets=32
```

Selectors are `leaf:ID` or `shard:ID`. A leaf cap applies to each matching source
shard in the selected output split (the canonical corpus has one shard per leaf and
split). `--cap ...=none` removes a canonical cap. Selection is deterministic,
weighted, and without replacement.

The resulting directory contains `documents.jsonl`, `provenance.jsonl`,
`resolved_recipe.json`, `aggregate_report.json`, and `sha256sums.txt`. Text appears
only in `documents.jsonl`; provenance and reports contain identifiers and hashes.
