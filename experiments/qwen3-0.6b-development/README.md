# Qwen3-0.6B development lane

This folder is the fast daily iteration lane. Its recipes keep the promoted Qwen3-1.7B
experiments and checkpoints unchanged, read the same prepared training shards, and write only to
`checkpoints/qwen3-0.6b-corpus-experts/`.

The 0.6B and 1.7B bases have different weights and dimensions, so their adapters and routers do
not transfer between lanes. Until 0.6B experts and a 0.6B router have been trained and evaluated,
the 0.6B lane should run base-only. Successful ideas can later be retrained and checked on the
preserved 1.7B promotion lane.

The recipes intentionally retain each promoted expert's initial hyperparameters as a comparable
starting point. Run one from the repository root, for example:

```powershell
.\.venv\Scripts\molor-train-experts.exe `
  .\experiments\qwen3-0.6b-development\software-architecture-v1-recipe.json `
  --local-files-only
```

`smoke-report.json` records the pinned official snapshot identity and the first local-only,
base-only CPU BF16 generation check. It is a load/forward receipt, not a quality benchmark or
context-expansion result.
