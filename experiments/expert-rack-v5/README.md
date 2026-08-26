# Expert-rack v5 corpus mixtures

`build_corpus_mixture.py` materializes one deterministic messages-only training or
evaluation mixture. A recipe pins every source by versioned shard and leaf IDs, split,
template mode, path, SHA-256, and physical JSONL record count. Evaluation recipes also
pin one or more train shards used as an exact normalized-message leakage guard.

The recipe's top-level fields are:

```json
{
  "schema_version": "molor.corpus-mixture-recipe.v1",
  "recipe_id": "business-eval-v1",
  "revision": "1",
  "usage": "evaluation",
  "output_split": "validation",
  "template_mode": "direct-v1",
  "seed": "business-eval-seed-v1",
  "target_records": 10000,
  "shortfall_policy": "error",
  "shards": [],
  "leakage_guard_shards": [],
  "content_sha256": "..."
}
```

Use the Python API's `corpus_mixture_recipe_json_bytes()` to create the self-hashing
artifact. Each mixture shard adds a positive `weight` and nullable positive `cap` to its
descriptor. Guard descriptors omit those two fields.

Run a validation-only pass:

```powershell
.\.venv\Scripts\python.exe experiments\expert-rack-v5\build_corpus_mixture.py `
  --recipe experiments\expert-rack-v5\recipes\business-eval-v1.json `
  --output datasets\mixtures\business-eval-v1 `
  --dry-run
```

Weights can be changed without modifying the canonical recipe. Leaf overrides preserve
the relative weights of shards already assigned to that leaf; shard overrides act last.

```powershell
.\.venv\Scripts\python.exe experiments\expert-rack-v5\build_corpus_mixture.py `
  --recipe experiments\expert-rack-v5\recipes\business-eval-v1.json `
  --output datasets\mixtures\business-eval-ops-heavy-v1 `
  --weight leaf:business-operations-v1=3 `
  --weight shard:startup-library-validation-v1=2
```

Materialized directories contain `messages.jsonl` (the only training/evaluation
payload), `provenance.jsonl` (one content-free accounting row per input record),
`resolved_recipe.json`, `aggregate_report.json`, and `sha256sums.txt`.
