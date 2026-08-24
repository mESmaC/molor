# Two-style adapter toy

This is deliberately a toy, not a benchmark. It trains two rank-8 adapters against one frozen
local `Qwen3-1.7B` load. The datasets use the same prompts with sharply different answers so manual
routes and weighted mixtures should become visible quickly.

From the workspace root, run:

```powershell
$env:PYTHONPATH = (Resolve-Path src)
.\.venv\Scripts\python.exe -m molor.training.cli --local-files-only experiments/qwen3-1.7b-two-styles/recipe.json
```

To retrain only one checkpoint, add `--expert precision` or `--expert creative`.

Adapters are written to `checkpoints/qwen3-1.7b-two-styles/precision` and `creative`. On a 12 GB GPU the
recipe uses bf16, micro-batch size one, four-step gradient accumulation, short sequences, and
gradient checkpointing. Rank 8 and four short epochs keep the tiny dataset from overpowering the
base model. The creative expert uses its own seven-epoch override because its answers are longer.

Each JSONL row has one field:

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

The final assistant message supplies the loss. System/user text and all earlier turns are masked.

## Manual rewards and preferences

Desktop annotations live in `runs/feedback/feedback.sqlite3`. Export all local training views with:

```powershell
$env:PYTHONPATH = (Resolve-Path src)
.\.venv\Scripts\python.exe -m molor.feedback.cli
```

After recording at least one pairwise choice, train a fresh DPO-style adapter with:

```powershell
.\.venv\Scripts\python.exe -m molor.training.preference_cli --local-files-only experiments/qwen3-1.7b-two-styles/preference_recipe.json
```

The frozen base/null route supplies cached reference log-probabilities. Only the new
`manual_preference` adapter is optimized, using assistant tokens from each chosen/rejected pair.
