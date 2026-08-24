"""Benchmark HF cache modes and archive one real KV-cache snapshot."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import torch
from torch import Tensor

from molor.experimental.mixture import MixtureEngine

DEFAULT_MODEL = Path("models/Qwen3-1.7B")
DEFAULT_SNAPSHOT_DIRECTORY = Path("runs/kv-cache")
DEFAULT_PROMPT = "Explain in two sentences why ocean tides occur."
MIB = 1024 * 1024


def _local_or_hub_model(value: str) -> str:
    path = Path(value).expanduser()
    return str(path.resolve()) if path.exists() else value


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _mib_per_second(byte_count: int, seconds: float) -> float:
    return byte_count / MIB / max(seconds, 1e-9)


def _benchmark_modes(
    engine: MixtureEngine,
    *,
    prompt: str,
    modes: list[str],
    max_new_tokens: int,
    context_token_budget: int,
    benchmark_prompt_tokens: list[int],
) -> None:
    print("\nCache-mode decode benchmark", flush=True)
    print(
        "(Synthetic exact-length token prompts; HF offloaded uses CPU memory, while disabled "
        "recomputes without a KV cache.)",
        flush=True,
    )
    model = engine.model
    tokenizer = engine.tokenizer
    device = model.get_input_embeddings().weight.device
    context_limit = engine.context_limit
    largest_allowed = (
        context_token_budget if context_limit is None else min(context_token_budget, context_limit)
    )
    for prompt_tokens in benchmark_prompt_tokens:
        if prompt_tokens < 1 or prompt_tokens + max_new_tokens > largest_allowed:
            raise SystemExit(
                f"benchmark prompt {prompt_tokens:,} + max_new_tokens {max_new_tokens:,} "
                f"exceeds the effective context budget {largest_allowed:,}"
            )

    warmup_inputs = _prefill_inputs(tokenizer, prompt, 8, device)
    with torch.inference_mode():
        model.generate(
            **warmup_inputs,
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
            cache_implementation="dynamic",
            pad_token_id=tokenizer.pad_token_id,
        )

    for prompt_tokens in benchmark_prompt_tokens:
        print(f"\n  exact prompt: {prompt_tokens:,} tokens", flush=True)
        inputs = _prefill_inputs(tokenizer, prompt, prompt_tokens, device)
        for mode in modes:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(0)
            cache_kwargs: dict[str, object]
            if mode == "disabled":
                cache_kwargs = {"use_cache": False}
            else:
                cache_kwargs = {"use_cache": True, "cache_implementation": mode}
            _sync_cuda()
            wall_started = time.perf_counter()
            try:
                with torch.inference_mode():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        top_k=None,
                        pad_token_id=tokenizer.pad_token_id,
                        **cache_kwargs,
                    )
                _sync_cuda()
            except Exception as error:
                print(f"    {mode:9} ERROR: {error}", flush=True)
                continue
            wall_seconds = time.perf_counter() - wall_started
            completion_ids = output_ids[0, prompt_tokens:]
            completion_tokens = int(completion_ids.numel())
            peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            print(
                f"    {mode:9} {completion_tokens:4d} output | "
                f"{completion_tokens / max(wall_seconds, 1e-9):6.2f} tok/s | "
                f"wall {wall_seconds:6.2f}s | peak CUDA {peak / MIB:,.0f} MiB",
                flush=True,
            )
            print(
                f"               {tokenizer.decode(completion_ids, skip_special_tokens=True).strip()}",
                flush=True,
            )


def _prefill_inputs(
    tokenizer: Any, prompt: str, token_count: int, device: torch.device
) -> dict[str, Tensor]:
    encoded = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    if encoded.numel() == 0:
        encoded = torch.tensor([tokenizer.eos_token_id], dtype=torch.long)
    repeats = math.ceil(token_count / encoded.numel())
    input_ids = encoded.repeat(repeats)[:token_count].unsqueeze(0).to(device)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }


def _snapshot_cache(
    engine: MixtureEngine,
    *,
    prompt: str,
    prefill_tokens: int,
    directory: Path,
) -> None:
    try:
        from safetensors.torch import load_file, save_file
        from transformers import DynamicCache
    except ImportError as error:  # pragma: no cover - optional install
        raise SystemExit("Install molor[hf] for the cache lab") from error

    model = engine.model
    tokenizer = engine.tokenizer
    device = model.get_input_embeddings().weight.device
    context_limit = engine.context_limit
    if context_limit is not None and prefill_tokens + 1 > context_limit:
        raise SystemExit(
            f"prefill_tokens must be below the model context limit ({context_limit:,})"
        )
    inputs = _prefill_inputs(tokenizer, prompt, prefill_tokens, device)

    print("\nKV snapshot/archive experiment", flush=True)
    print(
        "This is a point-in-time archive tier, not live disk-backed decoding. "
        "Transformers' live offloaded cache uses CPU memory.",
        flush=True,
    )
    cache = DynamicCache(config=model.config)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _sync_cuda()
    prefill_started = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**inputs, past_key_values=cache, use_cache=True)
    _sync_cuda()
    prefill_seconds = time.perf_counter() - prefill_started
    cache = outputs.past_key_values

    tensors: dict[str, Tensor] = {}
    _sync_cuda()
    stage_started = time.perf_counter()
    for index, layer in enumerate(cache.layers):
        if layer.keys is None or layer.values is None:
            raise RuntimeError(f"cache layer {index} was not initialized")
        tensors[f"layer.{index}.key"] = layer.keys.detach().to("cpu").contiguous()
        tensors[f"layer.{index}.value"] = layer.values.detach().to("cpu").contiguous()
    _sync_cuda()
    stage_seconds = time.perf_counter() - stage_started
    tensor_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())

    directory = directory.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    snapshot_path = directory / f"kv-{prefill_tokens}t-{timestamp}.safetensors"
    write_started = time.perf_counter()
    save_file(
        tensors,
        str(snapshot_path),
        metadata={
            "format": "molor.kv-snapshot.v1",
            "model_id": engine.base_model_id,
            "sequence_length": str(prefill_tokens),
            "tier": "snapshot/archive; not live decode",
        },
    )
    with snapshot_path.open("r+b", buffering=0) as handle:
        os.fsync(handle.fileno())
    write_seconds = time.perf_counter() - write_started
    file_bytes = snapshot_path.stat().st_size

    read_started = time.perf_counter()
    bytes_read = 0
    with snapshot_path.open("rb", buffering=0) as handle:
        while chunk := handle.read(8 * MIB):
            bytes_read += len(chunk)
    read_seconds = time.perf_counter() - read_started

    load_started = time.perf_counter()
    loaded = load_file(str(snapshot_path), device="cpu")
    load_seconds = time.perf_counter() - load_started
    restored_pairs = [
        (
            loaded[f"layer.{index}.key"].to(device),
            loaded[f"layer.{index}.value"].to(device),
        )
        for index in range(len(cache.layers))
    ]
    restored = DynamicCache(restored_pairs, config=model.config)

    next_input = inputs["input_ids"][:, -1:]
    next_attention = torch.ones(
        (1, prefill_tokens + 1), device=device, dtype=inputs["attention_mask"].dtype
    )
    with torch.inference_mode():
        reference_logits = model(
            input_ids=next_input,
            attention_mask=next_attention,
            past_key_values=cache,
            use_cache=True,
        ).logits[:, -1]
        restored_logits = model(
            input_ids=next_input,
            attention_mask=next_attention,
            past_key_values=restored,
            use_cache=True,
        ).logits[:, -1]
    _sync_cuda()
    max_logit_difference = float((reference_logits - restored_logits).abs().max().item())
    same_next_token = bool(reference_logits.argmax().item() == restored_logits.argmax().item())
    restoration_viable = restored.get_seq_length() == prefill_tokens + 1 and same_next_token

    manifest = {
        "format": "molor.kv-snapshot.v1",
        "model_id": engine.base_model_id,
        "snapshot": snapshot_path.name,
        "tier": "snapshot/archive; not live disk decode",
        "prefill_tokens": prefill_tokens,
        "layers": len(cache.layers),
        "tensor_bytes": tensor_bytes,
        "file_bytes": file_bytes,
        "prefill_seconds": prefill_seconds,
        "gpu_to_cpu_stage_seconds": stage_seconds,
        "write_and_fsync_seconds": write_seconds,
        "immediate_read_seconds": read_seconds,
        "safetensors_load_seconds": load_seconds,
        "restoration_viable": restoration_viable,
        "same_next_token": same_next_token,
        "max_logit_difference": max_logit_difference,
    }
    manifest_path = snapshot_path.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"  path:              {snapshot_path}", flush=True)
    print(
        f"  prefill:           {prefill_tokens:,} tokens in {prefill_seconds:.2f}s",
        flush=True,
    )
    print(
        f"  KV tensors:        {tensor_bytes / MIB:,.1f} MiB "
        f"({tensor_bytes / prefill_tokens / 1024:,.1f} KiB/token)",
        flush=True,
    )
    print(f"  GPU -> CPU stage:  {stage_seconds:.2f}s", flush=True)
    print(
        f"  write + fsync:     {_mib_per_second(file_bytes, write_seconds):,.1f} MiB/s "
        f"({write_seconds:.2f}s)",
        flush=True,
    )
    print(
        f"  immediate read:    {_mib_per_second(bytes_read, read_seconds):,.1f} MiB/s "
        f"({read_seconds:.2f}s; OS page-cache affected, not a cold-read claim)",
        flush=True,
    )
    print(
        f"  restoration:       {'viable' if restoration_viable else 'FAILED'}; "
        f"same next token={same_next_token}, max logit diff={max_logit_difference:g}",
        flush=True,
    )
    print(f"  manifest:          {manifest_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark HF cache modes and a disk KV snapshot/archive tier."
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--context-token-budget", type=int, default=4096)
    parser.add_argument(
        "--benchmark-prompt-tokens",
        type=int,
        nargs="+",
        default=[512],
        help="one or more exact synthetic prompt lengths, e.g. 2048 4096 8192",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("dynamic", "offloaded", "disabled"),
        default=["dynamic", "offloaded", "disabled"],
    )
    parser.add_argument("--prefill-tokens", type=int, default=512)
    parser.add_argument(
        "--snapshot-directory",
        type=Path,
        default=DEFAULT_SNAPSHOT_DIRECTORY,
        help="snapshot/archive destination; point this at the NVMe volume to measure",
    )
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if args.prefill_tokens < 1:
        parser.error("--prefill-tokens must be positive")

    model_id = _local_or_hub_model(args.model)
    print(f"Loading {model_id} in bfloat16", flush=True)
    engine = MixtureEngine.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        target_modules=("q_proj",),
        local_files_only=not args.allow_download,
    )
    _benchmark_modes(
        engine,
        prompt=args.prompt,
        modes=args.modes,
        max_new_tokens=args.max_new_tokens,
        context_token_budget=args.context_token_budget,
        benchmark_prompt_tokens=args.benchmark_prompt_tokens,
    )
    _snapshot_cache(
        engine,
        prompt=args.prompt,
        prefill_tokens=args.prefill_tokens,
        directory=args.snapshot_directory,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
