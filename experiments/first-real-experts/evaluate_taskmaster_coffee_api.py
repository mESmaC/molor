"""Deterministically generate or score held-out Taskmaster-4 API calls.

The primary metric is semantic equality after strict JSON parsing.  Canonical exact
match is deliberately stricter: the raw decoded text must equal the sorted,
minified reference bytes.  Schema, API-name, and argument metrics keep failures
diagnosable without relaxing the promotion signal.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from build_taskmaster_coffee_api import (
    EVALUATION_SCHEMA,
    SYSTEM_PROMPT,
    SourceRowError,
    canonical_json,
    sha256_file,
    sha256_text,
    strict_json_loads,
    validate_api_generation,
)


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    example_id: str
    group_id: str
    messages: tuple[dict[str, str], ...]
    expected: dict[str, Any]
    expected_canonical: str
    expected_sha256: str


def load_evaluation(path: Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = strict_json_loads(line)
                required = {
                    "schema_version",
                    "example_id",
                    "expected",
                    "expected_canonical",
                    "expected_sha256",
                    "group_id",
                    "messages",
                }
                if not isinstance(raw, dict) or set(raw) != required:
                    raise ValueError("evaluation row has unexpected fields")
                if raw["schema_version"] != EVALUATION_SCHEMA:
                    raise ValueError("unsupported evaluation schema")
                example_id = raw["example_id"]
                group = raw["group_id"]
                if not isinstance(example_id, str) or not example_id.startswith("tm4-coffee-api:"):
                    raise ValueError("invalid example_id")
                if example_id in seen:
                    raise ValueError("duplicate example_id")
                if not isinstance(group, str) or not group.startswith("tm4-coffee-trajectory:"):
                    raise ValueError("invalid group_id")
                messages = raw["messages"]
                if (
                    not isinstance(messages, list)
                    or len(messages) != 2
                    or messages[0] != {"role": "system", "content": SYSTEM_PROMPT}
                    or not isinstance(messages[1], dict)
                    or set(messages[1]) != {"role", "content"}
                    or messages[1].get("role") != "user"
                    or not isinstance(messages[1].get("content"), str)
                    or not messages[1]["content"].strip()
                ):
                    raise ValueError("invalid prompt messages")
                expected = validate_api_generation(raw["expected"])
                expected_canonical = canonical_json(expected)
                if raw["expected_canonical"] != expected_canonical:
                    raise ValueError("expected_canonical does not match expected")
                expected_hash = sha256_text(expected_canonical)
                if raw["expected_sha256"] != expected_hash:
                    raise ValueError("expected_sha256 does not match expected")
                full_messages = [
                    *messages,
                    {"role": "assistant", "content": expected_canonical},
                ]
                if example_id != f"tm4-coffee-api:{sha256_text(canonical_json(full_messages))}":
                    raise ValueError("example_id does not match prompt and expected content")
            except (KeyError, SourceRowError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            seen.add(example_id)
            cases.append(
                EvaluationCase(
                    example_id=example_id,
                    group_id=group,
                    messages=tuple(messages),
                    expected=expected,
                    expected_canonical=expected_canonical,
                    expected_sha256=expected_hash,
                )
            )
    if not cases:
        raise ValueError(f"{path} contains no evaluation cases")
    if [case.example_id for case in cases] != sorted(case.example_id for case in cases):
        raise ValueError("evaluation cases must be sorted by example_id")
    return cases


def load_predictions(path: Path, cases: list[EvaluationCase]) -> list[str]:
    expected_ids = {case.example_id for case in cases}
    predictions: dict[str, str] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = strict_json_loads(line)
                if not isinstance(raw, dict) or set(raw) != {"example_id", "output"}:
                    raise ValueError("prediction row needs exactly example_id and output")
                example_id = raw["example_id"]
                output = raw["output"]
                if not isinstance(example_id, str) or not isinstance(output, str):
                    raise ValueError("prediction example_id and output must be strings")
                if example_id in predictions:
                    raise ValueError("duplicate prediction example_id")
                predictions[example_id] = output
            except (SourceRowError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    missing = expected_ids.difference(predictions)
    extra = set(predictions).difference(expected_ids)
    if missing or extra:
        raise ValueError(
            f"prediction IDs do not match evaluation: {len(missing)} missing, {len(extra)} extra"
        )
    return [predictions[case.example_id] for case in cases]


def _api_parts(value: Any) -> tuple[str, dict[str, Any]] | None:
    try:
        generation = validate_api_generation(value)
    except SourceRowError:
        return None
    name, arguments = next(iter(generation["api"].items()))
    return name, arguments


def score_one(case: EvaluationCase, output: str) -> dict[str, Any]:
    parsed: Any = None
    parse_error: str | None = None
    try:
        parsed = strict_json_loads(output)
    except SourceRowError as exc:
        parse_error = str(exc)
    expected_parts = _api_parts(case.expected)
    if expected_parts is None:
        raise RuntimeError("validated evaluation reference lost its API schema")
    actual_parts = _api_parts(parsed)
    expected_name, expected_arguments = expected_parts
    actual_name, actual_arguments = actual_parts or (None, None)
    return {
        "example_id": case.example_id,
        "group_id": case.group_id,
        "expected_sha256": case.expected_sha256,
        "output_sha256": sha256_text(output),
        "output": output,
        "json_parseable": parse_error is None,
        "parse_error": parse_error,
        "schema_valid": actual_parts is not None,
        "semantic_exact": parsed == case.expected,
        "canonical_exact": output == case.expected_canonical,
        "api_name_exact": actual_name == expected_name,
        "arguments_exact": actual_arguments == expected_arguments,
        "expected_api_name": expected_name,
        "actual_api_name": actual_name,
    }


def score(cases: list[EvaluationCase], outputs: list[str]) -> dict[str, Any]:
    if len(cases) != len(outputs):
        raise ValueError("output count does not match evaluation case count")
    rows = [score_one(case, output) for case, output in zip(cases, outputs, strict=True)]
    count = len(rows)
    metric_names = (
        "json_parseable",
        "schema_valid",
        "semantic_exact",
        "canonical_exact",
        "api_name_exact",
        "arguments_exact",
    )
    summary: dict[str, Any] = {"examples": count}
    for name in metric_names:
        matches = sum(bool(row[name]) for row in rows)
        summary[f"{name}_count"] = matches
        summary[f"{name}_rate"] = matches / count
    summary["expected_api_counts"] = dict(
        sorted(Counter(str(row["expected_api_name"]) for row in rows).items())
    )
    summary["semantic_failures"] = count - int(summary["semantic_exact_count"])
    return {
        "summary": summary,
        "outputs_sha256": sha256_text(canonical_json(outputs)),
        "examples": rows,
    }


def adapter_targets(directory: Path) -> tuple[str, ...]:
    payload = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    targets = payload.get("target_modules")
    if (
        not isinstance(targets, list)
        or not targets
        or not all(isinstance(item, str) and item for item in targets)
    ):
        raise ValueError("adapter_config.json has invalid target_modules")
    return tuple(targets)


def generate(
    cases: list[EvaluationCase],
    *,
    model: Path,
    adapter: Path | None,
    device: str,
    dtype_name: str,
    batch_size: int,
    max_prompt_tokens: int,
    max_new_tokens: int,
) -> tuple[list[str], dict[str, Any]]:
    import torch

    from molor.experimental.mixture import BASE_ROUTE, MixtureEngine

    dtype = getattr(torch, dtype_name)
    engine_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_map": {"": device},
        "local_files_only": True,
    }
    if adapter is not None:
        engine_kwargs["target_modules"] = adapter_targets(adapter)
    engine = MixtureEngine.from_pretrained(str(model), **engine_kwargs)
    expert_name = engine.load_expert(adapter) if adapter is not None else None
    route = {expert_name: 1.0} if expert_name is not None else {BASE_ROUTE: 1.0}
    tokenizer = engine.tokenizer
    previous_padding_side = tokenizer.padding_side
    previous_truncation_side = tokenizer.truncation_side
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    model_device = engine.model.get_input_embeddings().weight.device
    outputs: list[str] = []
    engine.model.eval()
    try:
        with engine.use_route(route, normalize=False), torch.inference_mode():
            for offset in range(0, len(cases), batch_size):
                batch = cases[offset : offset + batch_size]
                prompts = [
                    tokenizer.apply_chat_template(
                        list(case.messages),
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    for case in batch
                ]
                encoded = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_prompt_tokens,
                )
                encoded = {key: value.to(model_device) for key, value in encoded.items()}
                prompt_width = int(encoded["input_ids"].shape[1])
                generated = engine.model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
                outputs.extend(
                    tokenizer.batch_decode(
                        generated[:, prompt_width:],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                )
    finally:
        tokenizer.padding_side = previous_padding_side
        tokenizer.truncation_side = previous_truncation_side
    metadata: dict[str, Any] = {
        "batch_size": batch_size,
        "device": device,
        "dtype": dtype_name,
        "expert_name": expert_name,
        "greedy": True,
        "max_new_tokens": max_new_tokens,
        "max_prompt_tokens": max_prompt_tokens,
        "model": str(model),
        "route": "expert" if adapter is not None else "base/null",
        "thinking": False,
    }
    if adapter is not None:
        metadata.update(
            {
                "adapter": str(adapter),
                "adapter_config_sha256": sha256_file(adapter / "adapter_config.json"),
                "adapter_weights_sha256": {
                    name: sha256_file(adapter / name)
                    for name in ("adapter_model.safetensors", "adapter_model.bin")
                    if (adapter / name).is_file()
                },
            }
        )
    return outputs, metadata


def write_predictions(path: Path, cases: list[EvaluationCase], outputs: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for case, output in zip(cases, outputs, strict=True):
            stream.write(canonical_json({"example_id": case.example_id, "output": output}))
            stream.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or score strict JSON outputs for the Taskmaster API holdout."
    )
    parser.add_argument("--evaluation", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--predictions", type=Path)
    mode.add_argument("--adapter", type=Path)
    mode.add_argument(
        "--base",
        action="store_true",
        help="generate with the frozen base/null route; requires --model",
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("runs/evaluation/taskmaster-coffee-api-v1.predictions.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/evaluation/taskmaster-coffee-api-v1.json"),
    )
    return parser


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(arguments)
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if args.batch_size < 1 or args.max_prompt_tokens < 8 or args.max_new_tokens < 1:
        parser.error("invalid batch or token limit")
    if (args.adapter is not None or args.base) and args.model is None:
        parser.error("--model is required with --adapter or --base")
    return args


def main() -> None:
    args = parse_args()

    evaluation = args.evaluation.resolve()
    all_cases = load_evaluation(evaluation)
    cases = all_cases if args.limit is None else all_cases[: args.limit]
    generation: dict[str, Any] | None = None
    predictions_path: Path
    if args.predictions is not None:
        predictions_path = args.predictions.resolve()
        outputs = load_predictions(predictions_path, cases)
    else:
        assert args.model is not None
        outputs, generation = generate(
            cases,
            model=args.model.resolve(),
            adapter=args.adapter.resolve() if args.adapter is not None else None,
            device=args.device,
            dtype_name=args.dtype,
            batch_size=args.batch_size,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_new_tokens,
        )
        predictions_path = args.predictions_output.resolve()
        write_predictions(predictions_path, cases, outputs)

    report = {
        "schema_version": "molor.taskmaster-coffee-api-evaluation-report.v1",
        "evaluation": str(evaluation),
        "evaluation_sha256": sha256_file(evaluation),
        "available_examples": len(all_cases),
        "evaluated_examples": len(cases),
        "predictions": str(predictions_path),
        "predictions_sha256": sha256_file(predictions_path),
        "generation": generation,
        **score(cases, outputs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report["summary"], indent=2, ensure_ascii=True))
    print(f"Saved report to {args.output.resolve()}")


if __name__ == "__main__":
    main()
