"""Compare a saved expert with its frozen base on assistant-only holdout loss."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from molor.experimental.mixture import BASE_ROUTE, MixtureEngine
from molor.training.data import (
    AssistantOnlyCollator,
    ChatExample,
    EncodedChatExample,
    load_jsonl,
)


def adapter_targets(directory: Path) -> tuple[str, ...]:
    payload = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    return tuple(payload["target_modules"])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_examples(
    engine: MixtureEngine,
    examples: list[ChatExample],
    *,
    max_length: int,
    enable_thinking: bool,
) -> tuple[AssistantOnlyCollator, list[EncodedChatExample]]:
    collator = AssistantOnlyCollator(
        engine.tokenizer,
        max_length=max_length,
        chat_template_kwargs={"enable_thinking": enable_thinking},
    )
    encoded = [collator.encode(example) for example in examples]
    # Evaluation is order-independent. Stable length sorting avoids padding work while
    # preserving the exact example set and assistant-token objective.
    encoded.sort(key=lambda example: len(example.input_ids))
    return collator, encoded


def evaluate(
    engine: MixtureEngine,
    collator: AssistantOnlyCollator,
    examples: list[EncodedChatExample],
    *,
    route: dict[str, float],
    batch_size: int,
) -> tuple[float, int]:
    device = engine.model.get_input_embeddings().weight.device
    loss_sum = 0.0
    supervised_tokens = 0
    engine.model.eval()
    with engine.use_route(route, normalize=False), torch.inference_mode():
        for offset in range(0, len(examples), batch_size):
            batch = {
                key: value.to(device)
                for key, value in collator.collate_encoded(
                    examples[offset : offset + batch_size]
                ).items()
            }
            output = engine.model(**batch, use_cache=False)
            tokens = int((batch["labels"][:, 1:] != -100).sum().item())
            loss_sum += float(output.loss) * tokens
            supervised_tokens += tokens
    return loss_sum / supervised_tokens, supervised_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.limit < 1 or args.batch_size < 1 or args.max_length < 8:
        parser.error("limit and batch-size must be positive; max-length must be at least 8")

    model = args.model.resolve()
    adapter = args.adapter.resolve()
    validation = args.validation.resolve()
    examples = load_jsonl(validation)[: args.limit]
    print(f"Loading {model} and {adapter}", flush=True)
    engine = MixtureEngine.from_pretrained(
        str(model),
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
        target_modules=adapter_targets(adapter),
        local_files_only=True,
    )
    expert_name = engine.load_expert(adapter)
    collator, encoded = encode_examples(
        engine,
        examples,
        max_length=args.max_length,
        enable_thinking=args.enable_thinking,
    )
    started = time.perf_counter()
    base_nll, base_tokens = evaluate(
        engine,
        collator,
        encoded,
        route={BASE_ROUTE: 1.0},
        batch_size=args.batch_size,
    )
    expert_nll, expert_tokens = evaluate(
        engine,
        collator,
        encoded,
        route={expert_name: 1.0},
        batch_size=args.batch_size,
    )
    result = {
        "schema_version": "molor.adapter-heldout-nll.v1",
        "model": str(model),
        "adapter": str(adapter),
        "adapter_sha256": file_sha256(adapter / "adapter_model.safetensors"),
        "adapter_config_sha256": file_sha256(adapter / "adapter_config.json"),
        "dataset": str(validation),
        "dataset_sha256": file_sha256(validation),
        "examples": len(examples),
        "supervised_tokens": min(base_tokens, expert_tokens),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "enable_thinking": args.enable_thinking,
        "length_sorted": True,
        "base_nll": base_nll,
        "base_perplexity": math.exp(min(base_nll, 20.0)),
        "expert_nll": expert_nll,
        "expert_perplexity": math.exp(min(expert_nll, 20.0)),
        "nll_improvement_percent": 100.0 * (base_nll - expert_nll) / base_nll,
        "elapsed_seconds": time.perf_counter() - started,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        destination = args.output.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
