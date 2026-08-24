"""Measure direct numeric-answer accuracy for a base model and one saved expert."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import unicodedata

import torch

from molor.experimental.mixture import BASE_ROUTE, MixtureEngine
from molor.rewards import EvaluationSample, FinanceCalculationVerifier
from molor.training.data import ChatExample, load_jsonl


ANSWER_PATTERN = re.compile(
    r"(?im)^\s*(?:\*{1,2})?answer(?:\*{1,2})?\s*:\s*(?:\*{1,2})?(.+?)(?:\*{1,2})?\s*$"
)
NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)")


def answer_text(text: str) -> tuple[str, bool]:
    matches = ANSWER_PATTERN.findall(text)
    return (matches[-1].strip(), True) if matches else (text.strip(), False)


def numeric_value(text: str) -> float | None:
    normalized = unicodedata.normalize("NFKC", text).replace(",", "")
    matches = NUMBER_PATTERN.findall(normalized)
    return float(matches[-1]) if matches else None


def unit_class(text: str) -> str:
    lowered = text.lower()
    if "%" in lowered or "percent" in lowered:
        return "percent"
    for unit in ("trillion", "billion", "million", "thousand"):
        if unit in lowered:
            return unit
    return "unspecified"


def adapter_targets(directory: Path) -> tuple[str, ...]:
    payload = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    return tuple(payload["target_modules"])


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
    tokenizer.padding_side = "left"
    device = engine.model.get_input_embeddings().weight.device
    generated: list[str] = []
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
    return generated


def score(examples: list[ChatExample], outputs: list[str]) -> dict[str, object]:
    verifier = FinanceCalculationVerifier()
    correct = 0
    labels = 0
    unit_matches = 0
    verifier_passes = 0
    verifier_score_sum = 0.0
    calculation_matches = 0
    reference_matches = 0
    verifier_statuses: dict[str, int] = {}
    failures: list[dict[str, object]] = []
    for index, (example, output) in enumerate(zip(examples, outputs, strict=True)):
        expected_text, _ = answer_text(example.messages[-1].content)
        generated_text, has_label = answer_text(output)
        expected = numeric_value(expected_text)
        generated = numeric_value(generated_text) if has_label else None
        expected_unit = unit_class(expected_text)
        generated_unit = unit_class(generated_text)
        verification = verifier.verify(
            EvaluationSample(
                sample_id=str(index),
                prompt=example.messages[-2].content,
                completion=output,
                reference=expected_text,
            )
        )
        status = verification.status.value
        verifier_statuses[status] = verifier_statuses.get(status, 0) + 1
        verifier_passes += int(verification.passed)
        verifier_score_sum += float(verification.score or 0.0)
        calculation_matches += int(
            bool(verification.details.get("calculation_matches_answer", False))
        )
        reference_matches += int(
            bool(verification.details.get("answer_matches_reference", False))
        )
        labels += int(has_label)
        unit_matches += int(
            has_label
            and (
                expected_unit == "unspecified"
                or expected_unit == generated_unit
            )
        )
        tolerance = None if expected is None else max(0.01, abs(expected) * 0.005)
        passed = (
            expected is not None
            and generated is not None
            and math.isclose(expected, generated, abs_tol=tolerance, rel_tol=0.005)
            and (expected_unit == "unspecified" or expected_unit == generated_unit)
        )
        correct += int(passed)
        if not passed and len(failures) < 12:
            question = example.messages[-2].content.rsplit("Question:", maxsplit=1)[-1].strip()
            failures.append(
                {
                    "index": index,
                    "question": question,
                    "expected": expected_text,
                    "generated": output.strip(),
                }
            )
    count = len(examples)
    return {
        "examples": count,
        "answer_label_rate": labels / count,
        "numeric_and_unit_accuracy": correct / count,
        "unit_match_rate": unit_matches / count,
        "finance_verifier_pass_rate": verifier_passes / count,
        "finance_verifier_mean_reward": verifier_score_sum / count,
        "calculation_answer_consistency_rate": calculation_matches / count,
        "answer_reference_match_rate": reference_matches / count,
        "finance_verifier_statuses": dict(sorted(verifier_statuses.items())),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    all_examples = load_jsonl(args.validation.resolve())
    numeric_examples = [
        example
        for example in all_examples
        if "Calculation:" in example.messages[-1].content
        and numeric_value(answer_text(example.messages[-1].content)[0]) is not None
    ][: args.limit]
    if not numeric_examples:
        raise SystemExit("validation set has no numeric calculation examples")
    model = args.model.resolve()
    adapter = args.adapter.resolve()
    engine = MixtureEngine.from_pretrained(
        str(model),
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
        target_modules=adapter_targets(adapter),
        local_files_only=True,
    )
    expert_name = engine.load_expert(adapter)
    routes = {"expert": {expert_name: 1.0}}
    if not args.skip_base:
        routes = {"base": {BASE_ROUTE: 1.0}, **routes}
    report: dict[str, object] = {
        "model": str(model),
        "adapter": str(adapter),
        "validation": str(args.validation.resolve()),
    }
    for label, route in routes.items():
        print(f"Generating {label} answers...", flush=True)
        outputs = generate(
            engine,
            numeric_examples,
            route=route,
            batch_size=args.batch_size,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_new_tokens,
        )
        report[label] = score(numeric_examples, outputs)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    # Windows terminals commonly default to cp1252. Keep the persisted report
    # readable UTF-8, but escape unsupported console glyphs such as ≈.
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
