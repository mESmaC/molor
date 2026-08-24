"""Compare a saved expert with its frozen base on assistant-only holdout loss."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from molor.experimental.mixture import BASE_ROUTE, MixtureEngine
from molor.training.data import AssistantOnlyCollator, ChatExample, load_jsonl


def adapter_targets(directory: Path) -> tuple[str, ...]:
    payload = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    return tuple(payload["target_modules"])


def evaluate(
    engine: MixtureEngine,
    examples: list[ChatExample],
    *,
    route: dict[str, float],
    batch_size: int,
    max_length: int,
) -> tuple[float, int]:
    collator = AssistantOnlyCollator(
        engine.tokenizer,
        max_length=max_length,
        chat_template_kwargs={"enable_thinking": False},
    )
    device = engine.model.get_input_embeddings().weight.device
    loss_sum = 0.0
    supervised_tokens = 0
    engine.model.eval()
    with engine.use_route(route, normalize=False), torch.inference_mode():
        for offset in range(0, len(examples), batch_size):
            batch = {
                key: value.to(device)
                for key, value in collator(examples[offset : offset + batch_size]).items()
            }
            output = engine.model(**batch)
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
    args = parser.parse_args()
    if args.limit < 1 or args.batch_size < 1 or args.max_length < 8:
        parser.error("limit and batch-size must be positive; max-length must be at least 8")

    model = args.model.resolve()
    adapter = args.adapter.resolve()
    examples = load_jsonl(args.validation.resolve())[: args.limit]
    print(f"Loading {model} and {adapter}", flush=True)
    engine = MixtureEngine.from_pretrained(
        str(model),
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
        target_modules=adapter_targets(adapter),
        local_files_only=True,
    )
    expert_name = engine.load_expert(adapter)
    base_nll, base_tokens = evaluate(
        engine,
        examples,
        route={BASE_ROUTE: 1.0},
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    expert_nll, expert_tokens = evaluate(
        engine,
        examples,
        route={expert_name: 1.0},
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    result = {
        "examples": len(examples),
        "supervised_tokens": min(base_tokens, expert_tokens),
        "base_nll": base_nll,
        "base_perplexity": math.exp(min(base_nll, 20.0)),
        "expert_nll": expert_nll,
        "expert_perplexity": math.exp(min(expert_nll, 20.0)),
        "nll_improvement_percent": 100.0 * (base_nll - expert_nll) / base_nll,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
