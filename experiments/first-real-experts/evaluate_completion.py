"""Greedy held-out source-continuation evaluation for one saved LoRA expert.

Exact match is deliberately strict: no whitespace, fence, or commentary normalization is
performed. Prefix and position metrics show partial progress without presenting it as exact
source recovery. Markdown code fences are valid presentation for the conversational knowledge
datasets and are already penalized naturally when they differ from the reference; they are not
classified as a forbidden preamble.
"""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import torch

from molor.experimental.mixture import BASE_ROUTE, MixtureEngine
from molor.training.data import ChatExample, load_jsonl


FORBIDDEN_PREAMBLES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("thinking_tag", re.compile(r"^\s*<(?:think|analysis)>", re.IGNORECASE)),
    (
        "courtesy_preface",
        re.compile(r"^\s*(?:sure|certainly|of course)(?:\b|[!,:])", re.IGNORECASE),
    ),
    (
        "here_is_preface",
        re.compile(r"^\s*(?:here(?:'s| is)|below is)\b", re.IGNORECASE),
    ),
    (
        "continuation_label",
        re.compile(
            r"^\s*(?:the\s+)?(?:next|continuation)(?:\s+(?:source|code|text))?"
            r"(?:\s+is\b|\s*:)",
            re.IGNORECASE,
        ),
    ),
    (
        "intent_preface",
        re.compile(r"^\s*(?:i(?:'ll| will)|let me)\b", re.IGNORECASE),
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def adapter_targets(directory: Path) -> tuple[str, ...]:
    payload = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    return tuple(payload["target_modules"])


def adapter_hashes(directory: Path) -> dict[str, str]:
    """Hash the small set of files that determines a PEFT adapter."""

    names = ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")
    return {name: sha256_file(directory / name) for name in names if (directory / name).is_file()}


def forbidden_preambles(text: str) -> list[str]:
    return [name for name, pattern in FORBIDDEN_PREAMBLES if pattern.search(text)]


def generate(
    engine: MixtureEngine,
    examples: list[ChatExample],
    *,
    route: dict[str, float],
    batch_size: int,
    max_prompt_tokens: int,
    max_new_tokens: int,
) -> list[str]:
    tokenizer = engine.tokenizer
    previous_padding_side = tokenizer.padding_side
    previous_truncation_side = tokenizer.truncation_side
    tokenizer.padding_side = "left"
    # If a caller lowers the prompt budget, preserve the source immediately before the cut.
    tokenizer.truncation_side = "left"
    device = engine.model.get_input_embeddings().weight.device
    generated: list[str] = []
    engine.model.eval()
    try:
        with engine.use_route(route, normalize=False), torch.inference_mode():
            for offset in range(0, len(examples), batch_size):
                batch_examples = examples[offset : offset + batch_size]
                prompts = [
                    tokenizer.apply_chat_template(
                        [message.to_dict() for message in example.messages[:-1]],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    for example in batch_examples
                ]
                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_prompt_tokens,
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                prompt_width = int(inputs["input_ids"].shape[1])
                outputs = engine.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
                generated.extend(
                    tokenizer.batch_decode(
                        outputs[:, prompt_width:],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                )
    finally:
        tokenizer.padding_side = previous_padding_side
        tokenizer.truncation_side = previous_truncation_side
    return generated


def token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    ids = encoded["input_ids"]
    if not isinstance(ids, list) or any(type(token_id) is not int for token_id in ids):
        raise TypeError("tokenizer returned invalid input_ids")
    return ids


def score_one(tokenizer: Any, reference: str, generated: str) -> dict[str, Any]:
    reference_ids = token_ids(tokenizer, reference)
    generated_ids = token_ids(tokenizer, generated)
    if not reference_ids:
        raise ValueError("reference continuation tokenized to zero tokens")

    common_prefix = 0
    for reference_id, generated_id in zip(reference_ids, generated_ids):
        if reference_id != generated_id:
            break
        common_prefix += 1
    position_matches = sum(
        reference_id == generated_id
        for reference_id, generated_id in zip(reference_ids, generated_ids)
    )
    preambles = forbidden_preambles(generated)
    return {
        "exact_match": generated == reference,
        "reference_tokens": len(reference_ids),
        "generated_tokens": len(generated_ids),
        "generated_minus_reference_tokens": len(generated_ids) - len(reference_ids),
        "longest_common_token_prefix": common_prefix,
        "longest_common_token_prefix_ratio": common_prefix / len(reference_ids),
        "token_position_matches": position_matches,
        "token_position_accuracy_over_reference": position_matches / len(reference_ids),
        "character_similarity": SequenceMatcher(None, reference, generated, autojunk=False).ratio(),
        "forbidden_preamble": bool(preambles),
        "forbidden_preamble_types": preambles,
    }


def score_route(
    tokenizer: Any,
    examples: list[ChatExample],
    outputs: list[str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, (example, generated) in enumerate(zip(examples, outputs, strict=True)):
        reference = example.messages[-1].content
        metrics = score_one(tokenizer, reference, generated)
        rows.append(
            {
                "index": index,
                "prompt_messages": [message.to_dict() for message in example.messages[:-1]],
                "reference": reference,
                "generated": generated,
                "reference_sha256": sha256_text(reference),
                "generated_sha256": sha256_text(generated),
                **metrics,
            }
        )

    count = len(rows)
    exact = sum(bool(row["exact_match"]) for row in rows)
    preambles = sum(bool(row["forbidden_preamble"]) for row in rows)
    failures = [
        {
            "index": row["index"],
            "reasons": [
                *([] if row["exact_match"] else ["not_exact_match"]),
                *(
                    [f"forbidden_preamble:{name}" for name in row["forbidden_preamble_types"]]
                    if row["forbidden_preamble"]
                    else []
                ),
            ],
            "longest_common_token_prefix_ratio": row["longest_common_token_prefix_ratio"],
            "token_position_accuracy_over_reference": row[
                "token_position_accuracy_over_reference"
            ],
            "character_similarity": row["character_similarity"],
        }
        for row in rows
        if not row["exact_match"] or row["forbidden_preamble"]
    ]
    return {
        "summary": {
            "examples": count,
            "exact_matches": exact,
            "exact_match_rate": exact / count,
            "mean_longest_common_token_prefix_ratio": sum(
                float(row["longest_common_token_prefix_ratio"]) for row in rows
            )
            / count,
            "mean_token_position_accuracy_over_reference": sum(
                float(row["token_position_accuracy_over_reference"]) for row in rows
            )
            / count,
            "mean_character_similarity": sum(float(row["character_similarity"]) for row in rows)
            / count,
            "forbidden_preamble_count": preambles,
            "forbidden_preamble_rate": preambles / count,
            "empty_generation_rate": sum(not str(row["generated"]) for row in rows) / count,
            "failure_count": len(failures),
        },
        "generation_sha256": sha256_text(
            json.dumps(outputs, ensure_ascii=False, separators=(",", ":"))
        ),
        "failures": failures,
        "examples": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare base and LoRA expert on exact held-out source continuation."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/evaluation/local-embedded-code-completion.json"),
    )
    args = parser.parse_args()
    if args.limit < 1 or args.batch_size < 1:
        parser.error("limit and batch-size must be positive")
    if args.max_prompt_tokens < 8 or args.max_new_tokens < 1:
        parser.error("max-prompt-tokens must be at least 8 and max-new-tokens must be positive")

    model = args.model.resolve()
    adapter = args.adapter.resolve()
    validation = args.validation.resolve()
    all_examples = load_jsonl(validation)
    examples = all_examples[: args.limit]
    dtype = getattr(torch, args.dtype)

    print(f"Loading {model} and {adapter}", flush=True)
    engine = MixtureEngine.from_pretrained(
        str(model),
        dtype=dtype,
        device_map={"": args.device},
        target_modules=adapter_targets(adapter),
        local_files_only=True,
    )
    expert_name = engine.load_expert(adapter)
    routes = {"expert": {expert_name: 1.0}}
    if not args.skip_base:
        routes = {"base": {BASE_ROUTE: 1.0}, **routes}

    report: dict[str, Any] = {
        "model": str(model),
        "adapter": str(adapter),
        "adapter_sha256": adapter_hashes(adapter),
        "validation": str(validation),
        "validation_sha256": sha256_file(validation),
        "available_validation_examples": len(all_examples),
        "evaluated_examples": len(examples),
        "generation": {
            "greedy": True,
            "thinking": False,
            "batch_size": args.batch_size,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_new_tokens": args.max_new_tokens,
            "dtype": args.dtype,
            "device": args.device,
        },
        "metric_notes": {
            "exact_match": "Raw decoded continuation equals the reference; no normalization.",
            "longest_common_token_prefix_ratio": (
                "Matching leading tokenizer IDs divided by reference token count."
            ),
            "token_position_accuracy_over_reference": (
                "Equal tokenizer IDs at each aligned position divided by reference token count; "
                "missing generated positions count as incorrect."
            ),
            "character_similarity": "difflib SequenceMatcher ratio on raw text.",
        },
        "routes": {},
    }
    for label, route in routes.items():
        print(f"Generating {label} continuations...", flush=True)
        outputs = generate(
            engine,
            examples,
            route=route,
            batch_size=args.batch_size,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_new_tokens,
        )
        report["routes"][label] = score_route(engine.tokenizer, examples, outputs)

    # ASCII escaping is still lossless JSON/UTF-8 and avoids Windows console/editor codec traps.
    rendered = json.dumps(report, indent=2, ensure_ascii=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(f"Saved full UTF-8 report to {args.output.resolve()}", flush=True)
    print(
        json.dumps(
            {label: result["summary"] for label, result in report["routes"].items()},
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
