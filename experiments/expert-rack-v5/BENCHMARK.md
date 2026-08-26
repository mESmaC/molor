# Rack-v5 fast-r8 benchmark launch

This campaign uses the unchanged `molor-core-v1` v1.0.0 suite and deterministic scorer v1.2.
Its config pins the suite fingerprint, taxonomy fingerprint, Qwen3-0.6B base, 25 canonical
rank-8 adapter directories, hierarchical router directory, policy values, isolated output root,
and the 64-case canonical v5 routing-evaluation contract. The frozen suite continues to define
prompts, generation, scoring, and quality fingerprints; the separate route contract only defines
which canonical v5 leaves are acceptable for routing telemetry. The benchmark runner will still
write its normal content-addressed `stack.lock.json` for every policy run.

No command in this launcher reads training corpora. `plan` only validates the checked-in suite,
taxonomy, and config; `preflight` reads the local model/adapter/router artifact metadata without
loading weights or allocating a GPU.

From the repository root, inspect the exact expanded commands first:

```powershell
.\.venv\Scripts\python.exe .\experiments\expert-rack-v5\benchmark_rack_v5.py plan
.\.venv\Scripts\python.exe .\experiments\expert-rack-v5\benchmark_rack_v5.py preflight
```

Then use two terminals. Keep the service in terminal 1:

```powershell
.\.venv\Scripts\python.exe .\experiments\expert-rack-v5\benchmark_rack_v5.py serve
```

After eager loading completes, run terminal 2 in this order:

```powershell
.\.venv\Scripts\python.exe .\experiments\expert-rack-v5\benchmark_rack_v5.py verify-server
.\.venv\Scripts\python.exe .\experiments\expert-rack-v5\benchmark_rack_v5.py smoke
.\.venv\Scripts\python.exe .\experiments\expert-rack-v5\benchmark_rack_v5.py full --thinking-mode off
.\.venv\Scripts\python.exe .\experiments\expert-rack-v5\benchmark_rack_v5.py full --thinking-mode on
```

The smoke command runs 28 policies over the 10-case profile: forced base, all 25 forced leaves,
automatic top-1, and automatic top-2, with thinking disabled. Each full command runs only forced
base, automatic top-1, and automatic top-2 over all 64 cases. Thinking-off and thinking-on use
separate fixed, resumable directories and compare automatic policies only with the matching base.

`serve`, `smoke`, and `full` fail closed if artifacts or the live service differ from the pinned
base, exact ordered adapter rack, router, taxonomy, or route contract. An existing campaign
directory is resumed by the benchmark runner; changing a suite, workload, decoding setting,
artifact identity, route evaluation, or fixed matrix contract is rejected before existing
provenance is overwritten.
