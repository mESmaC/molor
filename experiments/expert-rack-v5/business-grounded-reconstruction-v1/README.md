# Business grounded reconstruction v1

This corpus is an honest domain-adaptation SFT view of the canonical business
reference documents. It is explicitly **grounded reference reconstruction**, not
synthetic question answering and not evidence of answer quality.

Each conversation has three messages:

1. A fixed system instruction labeling the reconstruction contract.
2. A deterministic request grounded only in the canonical title and expert-leaf
   topic label.
3. The canonical source passage, verbatim, as the assistant target.

Training shards contain only `messages`. Exact `source_id`, `document_id`, source
artifact identity, source family, canonical split, rendered token count, and tier
are retained in matching provenance sidecars.

The build preserves all six business leaves and all three canonical splits. It
uses held-out-first deduplication precedence (`test`, `validation`, `train`) and
guards document IDs, exact title/text payloads, and source-family IDs across
splits. Qwen3-0.6B chat-template rendering is performed with thinking disabled,
truncation disabled, and a hard failure if any row exceeds 32,768 tokens.

## Build

From the repository root:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe experiments/business-expert/build_grounded_reconstruction.py `
  --reference-manifest C:\Users\engin\Documents\datasets\processed\molor-business-reference-v1\manifest.json `
  --reference-manifest-sha256 294a4306c1bacddba7cf24556396e89d55aa5592129cf7d01d1d86106e1c16bb `
  --tokenizer models/Qwen3-0.6B `
  --tokenizer-aggregate-sha256 4c6d93851307c525ca8fa64bb5f0dcaf61b4b854a4d4123a1b83aedde94d973c `
  --chat-template-sha256 a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8 `
  --output <new-empty-output-directory>
```

The output directory is atomically published and contains:

- `messages/<leaf>/<split>/<tier>.jsonl`
- `provenance/<leaf>/<split>/<tier>.jsonl`
- `accounting.jsonl`
- `manifest.json`
- `sha256sums.txt`

All four non-deferred tiers are materialized for every leaf/split pair, including
empty tier files. The real build and independent byte-for-byte verification are
pinned in `actual-build-receipt-v1.json`.
