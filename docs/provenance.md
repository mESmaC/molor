# Adapter identity and counterfactual reward tables

Status: implemented manifest identities, guarded runtime bindings, forced-route provenance, dense
reward-table semantics, strict persisted `BaseModelManifest` parsing, and verified frozen-base
loading. A PEFT capability/memory-adapter loader and strict parsers for the remaining manifest types
remain future integration work.

## Logical identity versus content identity

Every route has two kinds of identity:

- route_id is the stable capability label, such as math or base/null;
- route_index is its position inside one exact ordered bank;
- adapter_fingerprint identifies the complete adapter manifest;
- bank_fingerprint identifies the base, adapter manifests, and route order.

Retraining math retains its logical route ID but changes its adapter and bank fingerprints.
Reordering experts changes their indices and the bank fingerprint. Old router checkpoints and reward
tables therefore cannot silently acquire new action meanings. Adapter fingerprints cover provenance
fields as well as behavior-bearing files, so changing a revision, tag, training-run digest, or
library version also creates a new bank identity.

PEFT runtime adapter names are not content identity: adapter names may be assigned at load time and
are not retained as weight-key namespaces in saved checkpoints. The manifest remains independent of
the loader's in-memory name. See the
[PEFT checkpoint format](https://huggingface.co/docs/peft/developer_guides/checkpoint#adapter-model).

## Manifest hierarchy

ArtifactDescriptor records a normalized logical name, media type, byte count, and SHA-256 digest.
The name is a label only; manifest code neither resolves it as a path nor downloads it. The base
materializer now consumes those descriptors to copy and verify declared local bytes inside a fresh
private staging tree; adapter descriptors do not yet have an equivalent materializer.
This follows the consumer-verification shape of an
[OCI content descriptor](https://github.com/opencontainers/image-spec/blob/main/descriptor.md).

BaseModelManifest covers:

- requested revision and an immutable 40- or 64-hex resolved revision;
- architecture, dtype, and quantization identity;
- JSON configuration and safetensors weight descriptors;
- tokenizer artifacts, special tokens, and chat template;
- pinned custom code descriptors, if any.

The integration that creates a base manifest must resolve a branch or tag to a commit and verify
every file's size and digest. A missing PEFT revision can otherwise fall back to a mutable default;
see [PEFT model loading](https://huggingface.co/docs/peft/package_reference/peft_model) and
[Hub revision resolution](https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download).
The exact chat template is part of model behavior and is pinned separately, following
[Transformers chat-template storage](https://huggingface.co/docs/transformers/en/chat_templating_writing#storing-and-loading-chat-templates).

LoRAFormatManifest covers target-module selectors, exact per-projection shapes, a tensor-schema
digest, rank, alpha, scaling convention, dropout, dtype, bias, DoRA/RS-LoRA flags,
modules-to-save, and format version. AdapterManifest binds that recipe and exact base to
safetensors adapter files, PEFT configuration, a training-run digest, and library versions.

AdapterBankManifest accepts only adapters with identical base and LoRA-format fingerprints. Routes
are generated in adapter order followed by exactly one optional null route. The current vectorized
runtime rejects heterogeneous recipes and unsupported bias, DoRA, RS-LoRA, fan-in/fan-out, and
modules-to-save settings instead of silently dropping parameters.

SHA-256 establishes content integrity only after trusted code hashes the obtained bytes. It does not
establish publisher authenticity; untrusted manifests still require a trusted catalog or signature
policy.

## Runtime binding and loader handoff

TopKRouter and LoRAExpertBank accept an expected bank manifest. Binding checks route count and null
layout. A projection bank additionally checks rank, alpha, dropout, dtype, exact module name, input
and output shape, and unsupported LoRA features.

RouterOutput carries the bank fingerprint and ordered route IDs. A bound expert bank refuses output
from another bank even when its width is identical. Raw route tensors require the expected bank
fingerprint explicitly.

Checkpoint extra state uses molor.router-binding.v1 and molor.lora-bank-binding.v1. It persists the
bank fingerprint, ordered route IDs, and a runtime-contract fingerprint. Loading rejects different
route order, alpha/dropout/rank/dtype/module shapes, or router mode/top-k/temperature/gate semantics,
including when the destination began unbound.

Binding declares expected identity; it does not prove freshly initialized tensors came from an
adapter artifact. A bound projection starts with parameters_verified false and refuses execution.
After a safetensors loader verifies every adapter file and computes the loaded tensor-schema digest,
it calls confirm_loaded_artifacts as the explicit trust handoff. Verification state is not trusted
from a state dictionary and must be re-established after loading. The implemented Transformers base
bridge safely materializes only declared base artifacts and never loads pickle, enables remote code,
or resolves manifest URLs. A future adapter bridge must establish the corresponding tensor-schema
evidence before enabling each LoRA bank.

One AdapterManifest describes a whole-model adapter across every targeted projection. One
LoRAExpertBank contains the corresponding A/B tensors for a single projection, so every
instrumented layer binds to the same bank but validates its own exact module specification.

Unbound router and bank primitives remain available for low-level mathematics tests and adapter
training. Production integration must use the bound and artifact-verified path.

## Prompt-by-route reward table

RewardTable declares its prompt and route axes before collection and embeds the immutable specs that
determine scoring:

    prompt = EvaluationSample(
        sample_id="definition-only",
        prompt="What is 6 * 7?",
        completion="",
        reference="42",
    )

    table = RewardTable(
        bank_manifest,
        prompts=(PromptRef.from_sample("math/0001", prompt),),
        engine_manifest=reward_engine.manifest,
        generation_manifest={"temperature": 0, "max_tokens": 128},
        cost_normalizer_manifest={"name": "latency-v1", "version": "1"},
    )

PromptRef fingerprints prompt, reference, and metadata while excluding completion and rollout ID.
Every result in a row must carry that same prompt fingerprint.

A caller cannot add a free route label beside an already scored completion. ForcedRouteRecord is
created where generation has the completed EvaluationSample and binds its sample fingerprint,
bank, route ID/index, adapter fingerprint, generation spec, and seed. RewardTable checks that record
against the RewardReport. Swapping another completion or route trace fails.

CostNormalizationRecord binds raw measurements, normalized values, and the normalizer fingerprint.
Its normalized values must exactly match those used by RewardEngine. This makes a cost-aware oracle
auditable even though the normalizer implementation remains an integration component.

Each table permits exactly one cell per prompt_id and route_id and requires globally unique rollout
IDs. All routes in a prompt row use the same generation seed; stochastic replicates belong in
separate tables until an explicit replicate axis is implemented.

The expected grid is the exact declared prompt × bank-route Cartesian product. Missing whole rows or
columns remain visible. Duplicate cells/rollouts, changed prompt content, reordered banks, and mixed
reward, generation, or normalization specs are rejected.

structurally_complete means every declared cell exists. oracle_ready additionally means no cell is
masked. reward_matrix returns values plus a validity mask; masked verifier/infrastructure/reference
failures remain null and never become zero. The record-stream export includes a header with axes and
manifests, cell records, and a footer with completeness and the table fingerprint.

The exported dictionaries are canonical JSON-ready write formats, not yet a public parser contract.
A future strict loader must reject unknown fields/schema versions and revalidate the final digest
before these files are accepted from an untrusted source.

## Report and oracle semantics

evaluation_fingerprint identifies the exact EvaluationSample plus RewardEngine spec. Because the
sample fingerprint includes sample ID, prompt, completion, reference, and metadata, it is a rollout
identity—not a prompt identity. It excludes verifier outcomes, normalized costs, and timing.

RewardReport embeds its engine manifest and reconstructs the declared contract during validation.
It recomputes verifier membership, mask/gate state, weighted quality, cost penalty, and final clipped
reward. RewardReport.fingerprint covers serialized report semantics, including verifier details and
errors, but removes duration_ms. Only reports whose other diagnostics and reward semantics match
therefore share that fingerprint.

Oracle labels, soft targets, and regret require the requested row to be complete and scorable.
Route means and oracle mean require the entire table to be complete and scorable. oracle returns all
tied best routes. margin is best reward minus the second-highest route reward, so it is zero for a
tie.
