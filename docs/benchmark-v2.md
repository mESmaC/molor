# MoLoRA benchmark v2 design

Status: implementation design. This document does not change the recorded `molor-core-v1`
score series.

## Goal

`molor-core-v2` should measure whether a model can produce working engineering artifacts, not
whether it can recognize a familiar multiple-choice answer. It should compile patches, execute
generated programs, run a pinned MISRA-oriented static-analysis profile, simulate PIC32 register
writes, verify public mathematical derivations, and grade the final state of an interactive
terminal task.

The benchmark has two deliberately separate products:

1. **Engineering Index**: deterministic or executable cases with an objective 0-1000 score.
2. **Frontier report**: proof audits, genuine open-problem work, and designs with no single optimum.
   This report has component metrics and judge disagreement, but it never changes the Engineering
   Index.

The separation matters. Agreement among teacher models cannot establish a new Millennium Prize
proof, and an eloquent research memo must not outscore code that actually passes its tests.

## Score and tier structure

The Engineering Index contains 26 case templates. Every released run renders three held-out
variants of each template from recorded seeds, giving 78 scored cells. Comparisons require the same
rendered prompts, fixtures, hidden graders, toolchain image, seeds, and inference settings.

| Engineering category | Weight | Templates |
|---|---:|---:|
| Compiler and build diagnosis | 15% | 4 |
| Executable algorithms | 20% | 5 |
| MISRA-oriented C | 20% | 4 |
| PIC32 and PPS | 15% | 4 |
| Checkable mathematics | 15% | 5 |
| Terminal agent | 15% | 4 |

Each category is a macro-average, followed by the fixed category weighting above. Adding more
hidden tests to one category therefore cannot silently make that category dominate. Report the
mean and worst rendered-variant score for every template. Runtime, token rate, action count, and
reasoning-token use remain separate performance statistics.

Difficulty labels are descriptive breakdowns, not score multipliers:

- **T1 — bounded expert diagnosis:** a difficult but closed patch, mapping, or calculation.
- **T2 — executable expert work:** implementation evaluated by compilers, analyzers, tests, or a
  formal verifier.
- **T3 — agentic systems work:** multiple operations against a stateful simulator or terminal.
- **FX — frontier/research:** an open-ended or research case outside the Engineering Index.

## Why this requires a v2 runner

The current `molor.benchmark-suite.v1` model accepts a prompt and a text scorer. The current runner
generates one response and `score_response()` grades that string using choice, numeric, exact,
contains, regex, command, JSON, or composite matching. That is appropriate for `molor-core-v1`,
but it cannot safely express executable artifacts or interactive state.

Create a new `molor.benchmark-suite.v2` schema rather than changing v1 semantics. A v2 case needs
at least:

```json
{
  "id": "A01",
  "category": "algorithms",
  "tier": "T2",
  "task_type": "artifact",
  "prompt_template": "cases/A01/prompt.txt",
  "fixture_generator": "cases/A01/generate.py",
  "artifact_contract": {
    "allowed_files": ["solution.cpp"],
    "max_total_bytes": 65536
  },
  "grader": {
    "type": "sandbox",
    "manifest": "cases/A01/grader.json"
  },
  "variant_count": 3
}
```

Required task types are:

- `patch`: extract and apply one unified diff to an allowlisted fixture tree;
- `artifact`: extract one or more allowlisted files;
- `structured_math`: parse a declared JSON proof/derivation schema and invoke a symbolic verifier;
- `hardware_trace`: compile against mocked SFRs and grade the ordered register-write trace;
- `terminal_agent`: repeatedly generate an action, execute it in an isolated task environment, and
  grade final state;
- `frontier`: collect artifacts and rubric evaluations without contributing to the Engineering
  Index.

Do not make the existing pure text scorer launch compilers. Suite loading currently validates a
scorer by calling it with an empty response, and run summarization rescans raw strings. Introduce an
evaluator registry with separate specification validation and execution. Store an immutable
evaluation record containing response/artifact hashes, component scores, grader version, fixture
hash, hidden-test hash, and environment digest. Summary validation can then verify those hashes and
optionally rerun a grader explicitly.

## Model output protocols

Patch response:

```text
<<<PATCH>>>
diff --git a/include/example.h b/include/example.h
...
<<<END PATCH>>>
```

File response:

```text
<<<FILE:solution.c>>>
...
<<<END FILE>>>
```

Reject absolute paths, `..`, alternate data streams, symlinks, hard links, files outside the
allowlist, duplicate file markers, binary payloads unless explicitly enabled, and output beyond the
case byte limit. The model never chooses the grader command.

For checkable mathematics, require a public derivation artifact with a case-specific JSON schema.
Do not grade hidden `<think>` text. Private reasoning capture is provider-specific, may not be
available, and is not a substitute for a checkable solution.

Terminal actions use exactly one JSON object per turn:

```json
{"cmd":"rg -n \"failure\" logs"}
```

The runner returns stdout, stderr, exit code, and truncation metadata. The model ends with:

```json
{"final":"completed"}
```

The default limit is 20 actions, 60 seconds of task CPU, 1 GiB RAM, 256 processes, bounded output,
an unprivileged user, no network, a read-only base image, and a writable copy of only that task's
fixture. Hidden graders and answer data must never be mounted inside the task environment.

## Engineering case matrix

### Compiler and build diagnosis

#### C01 — const-safe callback (`T1`)

Prompt concept: a C11 repository fails under `-Werror=incompatible-pointer-types` because a bus API
declares a mutable frame callback while the supplied callback accepts `const uint8_t *`. The bus
owns the frame and the contract requires read-only callback access. Ask for a production-only patch;
forbid function-pointer casts, removing `const`, warning suppression, test edits, and build-flag
changes.

Objective grader:

- patch parses and applies: 5%;
- strict C11 build: 25%;
- visible regressions: 20%;
- hidden C and API-contract tests: 35%;
- AST/command anti-cheat checks: 15%.

The expected semantic repair is to make the callback typedef and all declarations consistently
read-only, but the grader accepts any genuinely equivalent safe patch.

#### C02 — incomplete `std::unique_ptr` PIMPL (`T1`)

Prompt concept: a C++20 consumer including `engine.hpp` fails in `std::default_delete` with
`invalid application of sizeof to incomplete type Engine::Impl`. Preserve PIMPL, automatic
ownership, and public signatures. Forbid raw owning pointers and no-op deleters.

Objective grader: patch application, build of multiple independent consumer translation units,
construction/destruction tests, AddressSanitizer or equivalent leak checks, and API/ABI assertions.
The normal solution declares the destructor in the header and defines it where `Impl` is complete.

#### C03 — C/C++ linkage boundary (`T1`)

Prompt concept: `sensor.c` is compiled as C17 into a static library, while `app.cpp` includes the
same header and fails to link `sensor_read(int)`. The public header must remain valid in both C and
C++. Forbid symbol renaming or link/build flag changes.

Objective grader: compile the header and a caller as C; compile it and a caller as C++; inspect the
exported symbol; link and execute hidden calls. This tests correct conditional `extern "C"` guards,
not recognition of the diagnostic text.

#### C04 — linker-script LMA overlap (`T2`)

Prompt concept: GNU ld reports that the `.data` load range overlaps `.text`. Require `.text` and
`.rodata` to remain in FLASH, `.data` to execute in RAM while loading immediately after the final
FLASH section, `.bss` to consume no image bytes, and `__data_load__`, `__data_start__`, and
`__data_end__` to be defined. Forbid changing `MEMORY`, discarding inputs, or editing sources.

Objective grader: link the fixture, parse its map and ELF metadata, and assert section bounds,
VMA/LMA relationships, symbol values, non-overlap, and image size.

### Executable algorithms

#### A01 — extreme-safe monotonic window (`T2`)

Prompt concept: implement
`size_t longest_span_window(std::span<const int32_t> values, uint64_t limit)`. Return the longest
contiguous window whose mathematical maximum minus minimum is at most `limit`. Handle
`INT32_MIN`/`INT32_MAX` without overflow and require O(n) time.

Objective grader: strict C++20 compile, visible cases, hidden edge cases, randomized differential
testing against brute force, sanitizers, and a 200,000-element scale gate that rejects quadratic
solutions.

#### A02 — one-discount shortest path (`T2`)

Prompt concept: implement a directed shortest-path API over as many as 200,000 vertices and 500,000
edges. At most one traversed edge may cost `floor(weight/2)`. Return `UINT64_MAX` for invalid or
unreachable inputs and avoid accumulation overflow.

Objective grader: compare with a locked two-state Dijkstra implementation on generated graphs,
including cycles, parallel edges, no-discount optima, unreachable targets, and weights near integer
limits; enforce an O((V+E) log V)-appropriate scale gate.

#### A03 — chunk-independent framed-stream parser (`T2`)

Prompt concept: implement a caller-owned C parser for frames
`[0xA5][length 0..32][payload][CRC-8]`; CRC polynomial is `0x07`, initial value zero, calculated over
length and payload. Input can be split at any byte, includes noise, and corrupt candidates emit
nothing before scanning for the next sync. Dynamic allocation is forbidden.

Objective grader: feed the same generated byte stream under every chunk partition for small cases,
fuzz longer noise/corruption cases, and compare emitted frames to an independent reference parser.

#### A04 — dynamic maximum-prefix queries (`T2`)

Prompt concept: implement a C++20 `PrefixLedger` supporting point replacement and maximum nonempty
prefix-sum queries over an inclusive range. It must handle 200,000 values and operations in O(log n)
per call and throw on invalid ranges.

Objective grader: reference segment tree, randomized update/query sequences, overflow-safe fixture
generation, exception tests, and a scale test.

#### A05 — stable streaming K-way merge (`T2`)

Prompt concept: merge sorted callback-driven `int64_t` streams with O(k) memory and O(N log k)
time. Equal values order by source index and then source order. A source error stops emission
immediately.

Objective grader: lazy sources, empty sources, large duplicate runs, integer extremes, injected
errors, callback-consumption accounting, memory-growth checks, and exact reference output.

### MISRA-oriented C

These cases use strict C11 compilation, undefined-behavior instrumentation where supported,
Cppcheck 2.20.0, and a versioned subset of the Cppcheck MISRA addon. The profile, checked rule IDs,
preprocessor definitions, and platform widths are part of the evaluator fingerprint. Inline
suppression comments, `#pragma`, dynamic allocation, recursion, and union type-punning are banned.

This is a reproducible **MISRA-oriented benchmark gate**, not a claim of full MISRA certification.
Cppcheck documents that its open-source MISRA coverage is partial and that rule texts must be
supplied separately. Never copy proprietary rule text into the repository; store only permitted
rule identifiers/profile configuration.

Default component weights for M01-M04 are functional tests 50%, strict compile and UB checks 15%,
base Cppcheck checks 10%, pinned MISRA findings 20%, and forbidden-construct checks 5%. A compiler
error yields zero functional points. Suppression or grader tampering zeros the affected analysis
components.

#### M01 — safe signed CAN extraction (`T2`)

Prompt concept: implement `decode_torque`. Bytes 1 and the low nibble of byte 2 form a 12-bit
little-endian two's-complement signal; compute `raw*5-200`. Null input or output returns false and
must not write output. The implementation must avoid signed-shift and promotion traps.

Objective grader: exhaustive testing of all 4096 raw values, null behavior, sentinel protection,
strict compilation, UB instrumentation, and the pinned analysis profile.

#### M02 — fixed-capacity ring buffer (`T2`)

Prompt concept: implement initialize/push/pop for an exactly 17-byte caller-owned ring. Failed
operations cannot mutate state or output, and valid long-running sequences cannot overflow an index
or counter.

Objective grader: exhaustive short operation sequences against a mathematical queue, long
wraparound traces, state snapshots on failure, strict compilation, and static analysis.

#### M03 — defined Q15 multiply-accumulate (`T2`)

Prompt concept: form the mathematical integer
`S = a*b + accumulator*32768`, divide by 32768 with round-to-nearest and exact ties away from zero,
then saturate to signed 16-bit range. Forbid reliance on signed overflow, right shift of negative
values, or left shift of negative values.

Objective grader: arbitrary-precision reference arithmetic, dense tie/saturation boundary tests,
randomized inputs, UB instrumentation, and static analysis.

#### M04 — debounce state machine (`T2`)

Prompt concept: active-low input, initially released, transition only after seven consecutive samples
of the candidate state, and emit exactly one `PRESSED` or `RELEASED` event per stable transition.
All state must be caller-owned.

Objective grader: exhaustive Boolean sequences through a bounded length, long noisy property tests,
state-reinitialization checks, strict compilation, and static analysis.

### PIC32 and peripheral pin select

Every prompt pins an exact device, package, Microchip Device Family Pack revision, and data-sheet
revision. A question that says only “PIC32” is invalid because PPS registers, selector values,
unlock mechanisms, and available pins vary by family.

The first release targets PIC32MX270F256D. Its rendered prompts include only the relevant official
input/output table rows, so the task measures accurate table use as well as retained knowledge. A
generated case can omit the table only if it is explicitly labeled closed-book and kept in a
separate score lane.

#### P01 — UART1 PPS initialization (`T2`)

Prompt concept: with PPS already unlocked, map U1TX to RPB7 and U1RX from RPB13, configure both pins
as digital, and set correct directions without touching unrelated SFRs.

Objective grader: compile against traceable mocked SFR definitions and require the equivalent final
state of `RPB7R=1`, `U1RXR=3`, RB7 output, RB13 input, and analog disabled. Grade the trace rather than
source regex so safe `SET`/`CLR` equivalents pass.

#### P02 — SPI1 PPS initialization (`T2`)

Prompt concept: map SDO1 to RPA1 and SDI1 from RPB5 with correct analog/digital and direction setup,
assuming PPS is already unlocked.

Objective grader: trace final state including `RPA1R=3`, `SDI1R=1`, RA1 output, RB5 input, no
reserved selector values, and no unrelated writes.

#### P03 — constrained pin assignment (`T1`)

Prompt concept: select UART1 TX/RX pins from a supplied table subject to generated forbidden pins,
package availability, board conflicts, and integer pin costs. Minimize total cost and break ties
lexicographically. Return exact-schema JSON with each pin, register, selector, and total cost.

Objective grader: feed the returned mapping to an independent constraint solver. Do not hard-code a
single expected JSON string. Generate costs and conflicts only after selecting the held-out seed.

#### P04 — PPS unlock/order repair (`T3`)

Prompt concept: repair code that writes mappings while IOLOCK is set, reverses an input/output
register, and restores interrupts incorrectly. Require preservation of prior interrupt state,
adjacent SYSKEY values, unlock, complete mapping/direction/analog setup, PPS relock, SYSKEY relock,
and no globally unlocked or interrupt-disabled exit.

Objective grader: compile against a state-machine SFR simulator and validate every ordered write,
critical-section invariant, final register state, and all early-exit paths.

### Checkable show-your-work mathematics

Structured derivations are public solution artifacts. A final number alone earns only the answer
component. A verbose uncheckable paragraph does not earn derivation credit.

#### Q01 — complete Diophantine solution (`T1`)

Prompt concept: find every integer `n` in a large interval for which
`n^2 + 4n + 13` is a square, taking `m >= 0`. Require JSON containing sorted solutions, the
completed-square identity, the integer factorization with product 9, and every admissible ordered
factor pair.

Objective grader: expected solutions are `[-6,-2,2]`; a symbolic verifier checks each identity,
factor-pair validity, parity, reconstruction, interval membership, and completeness.

#### Q02 — exact Bayesian posterior (`T1`)

Prompt concept: select line A with probability 7/10 or B with 3/10; their defect probabilities are
1/50 and 1/20. Observe D,D,G from the one selected line. Require exact JSON priors, likelihoods,
unnormalized weights, normalization, and reduced posterior.

Objective grader: final answer `2744/9869`; independently recompute and reduce every rational field.

#### Q03 — biased absorbing random walk (`T2`)

Prompt concept: states 0 through 4 with endpoints absorbing; move up with probability 2/3 and down
with 1/3 from internal states; start at 2. Require all three expectation recurrences, their exact
rational solution, and the requested expectation.

Objective grader: expected `E2=18/5`; parse and symbolically substitute every recurrence, solve the
system independently, and check uniqueness.

#### Q04 — integer optimization with certificate (`T2`)

Prompt concept: maximize `34x+37y` for nonnegative integers under
`4x+7y <= 143` and `6x+3y <= 117`. Require the optimizer, objective, dual multipliers, dual
inequalities, and equality certificate.

Objective grader: expected optimizer `(13,13)` and objective `923`; verify primal feasibility,
dual feasibility, upper bound, equality, and uniqueness. Coefficients should be procedurally varied
in later releases by generating the certificate first.

#### Q05 — machine-checked Lean proof (`T3`)

Prompt concept: complete only the proof after `:= by` for a generated modular-arithmetic,
finite-combinatorics, or invariant theorem. Forbid `sorry`, `admit`, unsafe axioms, new declarations,
and source imports beyond the allowlist.

Objective grader: lexical/AST ban check followed by `lake env lean Solution.lean` in a pinned image.
The theorem generator must first validate its reference proof and then remove only the proof body.

### Cryptography, CRC reverse-engineering, and generated logic expansion

This is an objective expansion lane, not an implicit change to the 26-template Engineering Index.
Publish it as a separate sub-index until its weights and reference distributions are frozen; adding
it to the main index requires a new baseline and suite fingerprint. All numeric fields use an
explicit byte order and CRC polynomials use one declared canonical representation.

`XCRC01` and `XLOG01` are already present in the executable first slice (`molor-executable-v1`) as
fixtures `crypto/crc16-ccitt-false-v1` and `logic/mastermind-unique-v1`. They are documented below,
not proposed duplicates.

#### XCRC01 — CRC-16/CCITT-FALSE implementation (`T2`, implemented)

Artifact: one strict-C11 implementation of `crc16_ccitt_false`, including defined zero-length/null
behavior. The `wsl_code_v1` verifier compiles it in the isolated WSL sandbox and compares it with an
independent bitwise oracle over the standard check vector, generated payloads, and every prefix of
a held-out stream.

Exact score: artifact extraction 5%, strict compilation 20%, hidden/generated vectors 55%, warning
cleanliness 10%, and forbidden-construct/policy checks 10%. Compile failure receives no downstream
test credit.

#### XLOG01 — generated Mastermind constraint solver (`T2`, implemented)

Artifact: one C++20 solver over four symbols drawn from digits 0–5 with repetition. It must use
duplicate-aware exact/misplaced accounting, write an answer only for a unique solution, and reject
invalid, contradictory, or ambiguous clue sets without mutating the caller's output.

The `wsl_code_v1` verifier generates 48 held-out uniquely solvable puzzles from independently
enumerated candidate sets, then checks malformed, contradictory, and ambiguous cases. Exact score:
artifact extraction 5%, strict compilation 20%, all generated/negative cases 55%, warning
cleanliness 10%, and policy checks 10%.

#### CR02 — catalog CRC identification (`T1`)

Prompt: given generated message/check pairs and a rendered catalog of 32–128 CRC definitions,
return exact-schema JSON naming the only matching definition and predict checksums for two new
messages. The catalog varies width, polynomial, initialization, reflection, xorout, and byte order;
the generator retains only instances with one semantically unique candidate.

Verifier `crc_catalog_v1` uses a separately implemented bit-at-a-time oracle. Exact score: JSON
schema 5%, consistency with all observed pairs 25%, correct semantic parameter set 40%, and the two
predictions 15% each. Equivalent aliases are accepted only when they induce the same CRC function
over the declared message domain.

#### CR03 — unknown-parameter CRC solver (`T2`)

Artifact: implement a supplied `recover_crc(captures, width, refin)` API. Width is 8, 12, or 16;
the polynomial, initialization, and xorout are unknown within declared bounded domains. The API
must return one canonical parameter tuple, `AMBIGUOUS`, or `NO_SOLUTION`. Hidden tests include
unique, deliberately ambiguous, inconsistent, and scale-gate capture sets.

Verifier `crc_recovery_v1` checks candidates with an independent GF(2)/SAT model and then tests the
returned function on unrevealed messages. Exact score: artifact/import 5%, visible examples 15%,
unique hidden recoveries 40%, correct ambiguous/no-solution classification 20%, held-out checksum
predictions 15%, and scale gate 5%. A different tuple receives full semantic credit if the verifier
proves it equivalent on the complete bounded domain.

#### CR04 — CRC syndrome frame repair (`T2`)

Artifact: implement a supplied API that locates zero, one, or two flipped payload bits from a frame
and its received CRC, returning the unique minimum repair or `AMBIGUOUS`. CRC parameters and maximum
frame length are rendered. Cases include clean frames, adjacent burst errors, two nonadjacent flips,
unrepairable frames, and collisions with more than one minimum repair.

Verifier `crc_repair_v1` exhaustively enumerates the permitted error space with an independent CRC
oracle. Exact score: extraction/build 10%, correct repaired payloads and bit indices 55%, clean and
unrepairable classification 15%, ambiguity handling 15%, and sanitizer/policy checks 5%.

#### CY01 — RSA-CRT fault recovery witness (`T2`)

Prompt: a freshly generated toy RSA public key is accompanied by one correct signature and one CRT
signature with a planted single-branch fault. Return JSON containing a nontrivial factorization,
the recovered private exponent, and a corrected signature. Key sizes are deliberately educational,
not security claims, and every run uses new primes.

Verifier `rsa_fault_v1` performs arbitrary-precision arithmetic without consulting a stored answer:
it checks nontrivial factors and `p*q=n`, the modular inverse defining `d`, and both signature
equations. Exact score: schema 5%, valid factorization 45%, valid private exponent 20%, corrected
signature 20%, and a machine-checkable GCD/congruence witness 10%.

#### L01 — generated logic-grid deduction (`T1`)

Prompt: solve a freshly generated 4x4 or 5x5 logic grid and return assignment JSON plus a list that
maps every clue ID to the constraint it asserts. Names, surface wording, clue order, and relation
types vary. The generator starts from a sampled solution, emits clues, and removes redundant clues
while preserving a unique solution.

Verifier `logic_csp_v1` parses the returned assignment into an independent finite-domain solver.
Exact score: schema/domain validity 10%, satisfaction of all clues 25%, exact unique assignment
45%, and correct clue-to-constraint mappings 20%. Mere prose receives zero.

#### L02 — contradiction and minimal-clue repair (`T2`)

Prompt: one clue in a generated logic puzzle was corrupted. Return every member of the unique
minimum unsatisfiable clue core, identify the corrupted clue, select its corrected form from a
rendered set, and give the repaired puzzle's unique solution. Seeds with tied minimum cores or
multiple repaired solutions are rejected before release.

Verifier `logic_unsat_v1` independently enumerates subsets and solutions. Exact score: schema 5%,
valid unsatisfiable core 25%, proof that the core is minimum 20%, corrupted-clue identification 15%,
valid correction 15%, and repaired unique solution 20%.

#### L03 — interactive constraint codebreaker (`T3`)

The stateful runner samples a code from a rendered alphabet with generated distinctness and parity
constraints. Each JSON guess receives exact-position and present-wrong-position counts; the model
must return a final code within 12 actions. Invalid guesses consume an action, and the hidden code
and solver are never mounted in the task environment.

Verifier `logic_oracle_agent_v1` replays the transcript against the committed seed and checks every
response. Exact score: correct final code 60%, valid protocol/actions 10%, transcript consistency
10%, and query efficiency 20% (`12` actions = 0 efficiency credit, `6` or fewer = full credit,
linear between). A wrong final code receives zero correctness and efficiency credit.

Expansion-specific contamination controls:

1. Generate the solution, CRC parameters, fault, or constraint model first; independently verify
   it, then render identifiers and prose. Reject non-unique cases unless ambiguity is the target.
2. Keep generator seeds, hidden payloads, oracle transcripts, SAT models, primes, and correction
   candidates sealed. Only public examples may appear in prompts.
3. Use a second implementation style for grading: bitwise versus table-driven CRC, GF(2)/SAT versus
   candidate code, and finite-domain solver versus the puzzle generator.
4. Randomize semantic structure, not merely names: polynomial families, widths, reflection,
   capture lengths, error patterns, clue graphs, relation types, and oracle constraints all vary.
5. Run exact and MinHash overlap checks against SFT, RL, memory, and retrieval corpora. Train only
   on independently generated sibling seeds and never persist benchmark interactions to memory.

### Terminal agent

Terminal correctness is final-state correctness. Do not use a command allowlist as the primary
scorer; many valid command sequences exist. A small deny policy still blocks network access,
privilege escalation, host paths, mounts, raw devices, and attempts to inspect the hidden grader.

#### T01 — repair a failing build (`T3`)

Prompt concept: inspect and repair a CMake/Ninja project so clean configure, strict build, and CTest
all pass. Forbid test edits, deleting targets, weakening warnings, and replacing behavior with
constants.

Objective grader: discard the model's build directory, rebuild clean, run visible and hidden tests,
and inspect the patch for forbidden scope. Randomize the concrete dependency/configuration defect
across variants.

#### T02 — safe Git recovery (`T3`)

Prompt concept: recover an unreachable commit identified by subject into branch `recovered` while
preserving an uncommitted operator file byte-for-byte. Forbid `reset --hard`, `clean`, and reflog
destruction.

Objective grader: inspect the commit graph, branch target, parent ancestry, blob hashes, worktree and
index state, preservation of the operator file, and command-policy log.

#### T03 — service and port diagnosis (`T3`)

Prompt concept: repair a telemetry service returning HTTP 503. A generated file specifies its
required port; another process is protected and cannot be killed or modified. Require persistent
configuration, a safe restart, and exact health JSON.

Objective grader: health probe before and after a second restart, expected bound port/process
identity, protected-process survival, configuration state, log assertions, and prohibited-action
audit. Randomize ports, PIDs, filenames, and stale decoys.

#### T04 — robust PowerShell data repair (`T3`)

Prompt concept: create `solve.ps1` that reads a literal directory whose name contains brackets and
wildcards, joins event/device CSV data, records malformed input line numbers, and writes canonical
summary JSON. Paths include spaces, apostrophes, brackets, and wildcard characters.

Objective grader: rerun the produced script against hidden paths and rows, validate exact JSON data
and malformed-line reporting, and ensure source inputs remain unchanged. This defeats a manually
written answer that only matches the visible fixture.

## Frontier case matrix

Frontier cases produce a separate report with their raw artifacts, objective subchecks, rubric
scores, judge identities/versions, per-judge results, median, dispersion, and human annotations.
They never contribute to the 0-1000 Engineering Index.

#### F01 — Millennium-proof audit (`FX`, mostly objective)

Prompt concept: audit a numbered manuscript claiming the Riemann hypothesis. Identify the earliest
invalid inference, its missing condition, and an explicit counterexample to that inference. Forbid
skipping to later errors.

The manuscript is freshly constructed around a planted but mathematically real error such as an
invalid limit exchange or unsupported uniform bound. Objective components grade the first invalid
step, a machine-checked counterexample, and whether the named additional condition would repair that
specific inference. A small judged component may evaluate explanation clarity.

#### F02 — genuine open-problem research memo (`FX`, judged)

Prompt concept: attempt progress on a precisely scoped P-versus-NP subproblem in standard ZFC.
Require assumptions, one formal proposed lemma, dependency graph, smallest unresolved obligation, a
falsifiable computational experiment, and the strongest conclusion actually justified. A claim of
resolution requires a separately checkable formal artifact.

Report logical rigor, calibration, falsifiability, novelty, reproducibility, and judge disagreement.
Use anonymous outputs, randomized pair order, at least three independent teacher judgments, and a
human annotation slot. Never label the result a correctness score.

#### F03 — no-single-optimum systems design (`FX`, hybrid)

Prompt concept: design a local one-trillion-parameter inference and NVMe KV-spill system under
rendered PCIe-lane, VRAM, RAM, endurance, power, budget, latency, workload, and single-drive-failure
constraints. Require machine-readable topology and calculations, degradation/recovery behavior,
assumptions, and rejected alternatives.

An objective simulator reports feasibility, steady-state bandwidth, tail latency, capacity,
endurance, recovery, and cost. Feasible results form a Pareto set rather than one gold answer.
Independent judges separately assess assumption quality, failure analysis, and practical coherence.

## Objective versus judged boundary

The following may enter the Engineering Index:

- compiler/linker exit status and diagnostics under a pinned command;
- program output and state under hidden deterministic and generated tests;
- sanitizer and static-analyzer findings under a pinned profile;
- asymptotic scale gates with a fixed environment and generous threshold;
- constraint-solver feasibility/optimality;
- symbolic equivalence, rational arithmetic, proof-assistant acceptance;
- simulated SFR state and ordered hardware invariants;
- terminal filesystem, Git, process, service, and command-policy state.

The following must remain separate judged dimensions:

- prose elegance and explanation clarity;
- novelty of research ideas;
- whether a proposed architecture is preferable on unstated human priorities;
- correctness of an alleged proof for a genuinely unresolved problem when no accepted formal
  verifier establishes it;
- teacher preference, even when several teachers agree.

Judge controls:

- hide model/provider identity and route;
- randomize A/B order and repeat with reversed order;
- pin judge prompt, model version, temperature, and rubric;
- use multiple independent judges rather than a conversational consensus;
- expose disagreement and confidence intervals instead of only an average;
- allow manual annotation and adjudication in the desktop application's RL/annotation panel;
- keep all frontier judgments out of automated training unless the user deliberately exports them.

## Contamination resistance

1. **Render after certificate generation.** Math and optimization generators create a valid witness,
   proof, or dual certificate first, validate it, and only then render the problem.
2. **Multiple variants.** Use three fixed held-out variants per template for recorded comparisons and
   report worst-variant performance.
3. **Semantic variation.** Vary constraints, data widths, tie-breaking, error behavior, and API shape;
   renaming a famous LeetCode problem alone is not meaningful resistance.
4. **Independent property graders.** Algorithm fixtures use a separately written reference plus
   metamorphic properties and randomized differential testing.
5. **Versioned hardware data.** Extract PIC mappings from a locked, reviewed Microchip table dataset;
   generate legal and illegal board assignments from it rather than copying one recurring mapping.
6. **Stateful randomization.** Terminal fixtures randomize ports, paths, PIDs, Git object IDs, logs,
   decoys, and data while retaining a deterministic seed.
7. **Training overlap scan.** Before release, run MinHash and exact/n-gram overlap scans against all
   SFT, DPO, RL, memory, and retrieval corpora. Reject or regenerate suspicious cases.
8. **No persistence.** Set benchmark generations to `persist=false`. Do not put prompts, fixtures,
   hidden tests, outputs, gold patches, or judge feedback into cross-session memory or training data.
9. **Isolation from agent tools.** Non-terminal cases run with tools denied and an isolated working
   directory. Terminal cases mount only their fixture, never the workspace or hidden grader.
10. **Canaries and rotation.** Embed non-semantic canary identifiers in rendered holdouts, audit
    corpora for them, and publish a new suite fingerprint if leakage is detected.
11. **Immutable provenance.** Hash rendered prompt, fixture tree, generator source, seeds, grader,
    toolchain/container, reference implementation, and hidden tests. A change starts a new score
    series.

Benchmark feedback is especially dangerous for RL: grading a holdout and then training directly on
that prompt destroys its value. Train on newly generated sibling instances and retain sealed seeds
for evaluation.

## Grader component conventions

Each evaluator returns component scores from 0 to 1, a weighted total, pass/fail gates, diagnostics,
and artifact hashes. Recommended conventions:

- failure to parse the declared artifact: zero;
- patch outside the allowlist or grader tampering: zero;
- compile failure: no functional/test credit, but a small patch/application component may remain;
- timeout/resource breach: failed scale component and terminated sandbox;
- sanitizer fault: failed safety component and a configurable total-score cap;
- hidden test failure: only the affected test/property component fails;
- forbidden suppression in a MISRA case: zero analysis credit;
- terminal prohibited action: terminate the case and mark its safety gate failed;
- frontier judge failure: retain objective subresults and mark that judge missing, never substitute a
  perfect or zero research score silently.

Use operation counts or broad scale separation where possible. Extremely tight wall-clock cutoffs
are sensitive to host load and should not become a hidden speed benchmark.

## Current local toolchain feasibility

Read-only inspection on 2026-08-23 found:

| Tool | Local state |
|---|---|
| GCC | 16.1.0, available |
| Cppcheck | native 2.20.0; WSL sandbox 2.19.0 with MISRA addon |
| Bubblewrap | WSL 0.11.1, available |
| CMake | 4.3.2, available |
| Ninja | 1.13.2, available |
| Docker client | 29.6.1, available |
| Docker engine | not running |
| Clang | not found |
| Lean | not found |
| PowerShell 7 (`pwsh`) | not found |

The first executable slice now uses WSL2 + Bubblewrap for C/C++ compilation, UBSan, hidden tests,
and Cppcheck MISRA-addon analysis. It is locally isolated but not yet a content-addressed evaluator
image: WSL package upgrades and host load must not silently be compared across score series.

Recommended path:

1. Start Docker Desktop and build one content-addressed Linux evaluator image containing GCC,
   Clang, CMake, Ninja, Cppcheck, Python, and required sanitizers.
2. Build a separate Lean image so upgrading Lean/mathlib does not invalidate all engineering
   graders.
3. Use a Windows/PowerShell-specific image or isolated Windows job for T04; record its exact OS and
   PowerShell version as part of that case fingerprint.
4. Add XC32 and the exact Microchip DFP only if licensing/distribution permits it. Until then,
   compile PIC32 solutions against reviewed mocked SFR headers and grade the hardware trace.
5. Never describe Cppcheck-only results as full MISRA compliance.

## Implementation sequence

### Phase 1 — artifact evaluator

- add the v2 suite model without changing v1;
- implement marker extraction and path/size validation;
- implement isolated patch/file workspaces;
- add deterministic grader manifests and evaluation records;
- implement C01-C03, A01, M01, M02, P01, Q02 as a vertical slice;
- record all grader and fixture hashes in summaries.

Acceptance criterion: two different valid implementations can receive full credit, a hard-coded
answer cannot, and a run can be reproduced from its recorded artifacts.

### Phase 2 — complete executable engineering lane

- add linker/ELF inspection, property fuzzing, scale gates, remaining algorithm cases, MISRA cases,
  PIC constraint/state simulation, and symbolic math;
- build all three held-out variants per template;
- run reference solutions and intentionally flawed mutants to calibrate each component;
- confirm that every mutant fails the component designed to catch it.

### Phase 3 — terminal agent runner

- add action-loop backend support independent of ordinary chat generation;
- isolate task files and processes from the workspace and hidden grader;
- implement state snapshots, action transcripts, command-policy auditing, and deterministic resets;
- ship T01-T04 only after each task survives adversarial escape and shortcut testing.

### Phase 4 — frontier report

- add blinded multi-judge orchestration and manual annotation;
- keep objective frontier subchecks visible but outside the Engineering Index;
- report judge dispersion and pair-order sensitivity;
- never automatically export frontier cases or judgments into RL data.

## Release checklist

- Reference solutions score 100% on all three rendered variants.
- At least three plausible flawed mutants exist per template and fail for the intended reason.
- No production benchmark prompt, fixture, or gold artifact overlaps the training corpora above the
  declared threshold.
- Network and host workspace access are unavailable to graders and terminal tasks.
- All case resources and toolchains are content-hashed.
- Direct/no-thinking and deliberate/thinking runs use separate workload fingerprints.
- Report names the exact model, route/expert policy, inference settings, evaluator version, and
  rendered variant seeds.
- Engineering, performance, routing, and frontier scores are presented separately.
- Any change to a prompt template, generator, fixture, test, toolchain, scoring weight, or judge
  rubric starts a new versioned score series.
